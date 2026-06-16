#!/usr/bin/env python3
"""CloudCropper BUFFER-X worker (persistent process).

Spawned once by the C++ bufferx backend (PythonWorker) and kept alive so the
multi-second torch import and the model weight-load cost is paid a single time.
Speaks the SAME JSON-lines protocol as gsdf_worker.py: one UTF-8 JSON object
per line on stdin (request) and stdout (response), strictly one request in
flight. Point clouds are handed over as NPZ files (CloudCropper exports: key
"xyz" (N,3) f4, optional "normal"; BUFFER-X only needs "xyz").

Handshake (before any request): the worker emits
    {"event":"loading","pid":<pid>}            immediately, then after imports
    {"event":"ready","pid":...,"device":"cuda"|"cpu","torch":"<version>",
     "bufferx":0|1}
or, if the core deps (numpy/torch) fail,
    {"event":"fatal","error":{"type","message","traceback"}}  and exits 1.

Ops:
    {"id":N,"op":"ping"}      -> {"id":N,"ok":true,"result":{device/torch/cuda/
                                  bufferx}}
    {"id":N,"op":"register","source":...,"target":...,"target_key":...,
     "device":"cuda","voxel_size":0.0,"refine":false, ...}
                              -> {"id":N,"ok":true,"result":{"transform":[16],
                                  "converged":0|1,"num_inliers":...,
                                  "num_mutual_inliers":...,"scales_used":...,
                                  "device":...,"seconds":...,"cache_hit":0|1}}
    {"id":N,"op":"shutdown"}  -> {"id":N,"ok":true,"result":{}} then exit 0

Every other request key is matched against the typed table below (_BX_KW) so a
knob in config/bufferx.yaml reaches the algorithm; unknown keys are ignored with
a stderr warning. The GICP refine is done on the C++ side, so `refine` from the
bridge is always false here.

The vendored upstream BUFFER-X core lives in `./bufferx_upstream/` (see
VENDORED.md) and the pretrained weights in `./weights/snapshot/<source>/<stage>/
best.pth` (see download_weights.sh). When the core's CUDA extensions
(pointnet2_ops / knn_cuda / torch_batch_svd) or the weights are missing, the
worker still comes up `ready` and the `register` op returns an IDENTITY
transform with `converged:false` and an explanatory `note` — it does NOT
fabricate inlier numbers as if a real alignment happened.

Op failures answer {"id":N,"ok":false,"error":{...}} and keep the loop alive;
stdin EOF (parent died) exits cleanly. Diagnostics go to stderr, which the C++
side redirects to <tmpdir>/worker.log.

Debug use without the C++ side:
    python3 bufferx_worker.py --oneshot source.npz target.npz
"""
# NOTE: only stdlib imports at module level — the protocol bootstrap (loading/
# fatal events) must work even when torch is not installed.
import json
import os
import sys
import time
import traceback

# The vendored upstream core sits in ./bufferx_upstream next to this script; it
# uses absolute imports (`models.*`, `config`, `utils.*`), so that directory must
# be on sys.path. Running the script directly already puts _HERE at sys.path[0].
_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM = os.path.join(_HERE, "bufferx_upstream")
for _p in (_HERE, _UPSTREAM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Which pretrained checkpoint to load at startup (we ship threedmatch — the
# indoor/general zero-shot generalist; kitti is the outdoor model).
_EXPERIMENT_ID = os.environ.get("CLOUDCROPPER_BUFFERX_EXPERIMENT", "threedmatch")


def _error_obj(exc):
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=8),
    }


def _to_bool(v):
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


# Request keys consumed explicitly by the worker itself (never forwarded).
_RESERVED = {"id", "op", "source", "target", "target_key", "device",
             "voxel_size", "refine", "weights_dir"}

# BUFFER-X inference kwargs that are SAFE to override per call: name -> caster.
# (num_scales is intentionally NOT here: it must stay in lock-step with the
# config's search_radius_thresholds, see BUFFERX.py forward assert.)
_BX_KW = {
    "num_fps": int,            # farthest-point-sampled keypoints per scale
    "pose_estimator": str,     # "ransac" (default) | "kiss_matcher"
}

# Identity 4x4 (row-major, target <- source) used by the no-core/no-weights path.
_IDENTITY16 = [1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 1.0]


def _run_bufferx(worker, src_pts, tgt_pts, *, device, voxel_size, **kwargs):
    """Real BUFFER-X inference (target<-source 4x4 + info dict).

    Replicates the upstream test-time preprocessing (sphericity-based voxel-size
    estimation + voxel downsample → first-level downsampled clouds) and calls
    `model(data_source)`. The model returns
    (pose, times, num_inliers, num_mutual_inliers, num_inlier_ind, scales_used);
    `pose` maps source -> target (= CloudCropper's target<-source convention).

    When the core/weights are unavailable (worker.model is None), returns an
    IDENTITY transform with converged False and a note — it MUST NOT invent
    inlier numbers, so those keys are simply absent in fallback mode.
    """
    if worker.model is None:
        return list(_IDENTITY16), {
            "converged": False,
            "note": worker.model_error
            or "BUFFER-X core/weights not available; returning identity",
        }

    np, torch, o3d = worker.np, worker.torch, worker.o3d
    cfg = worker.cfg

    # Optional per-call overrides (safe knobs only).
    if "pose_estimator" in kwargs:
        cfg.match.pose_estimator = kwargs["pose_estimator"]
        worker.model.pose_estimator.pose_estimator = kwargs["pose_estimator"]
    if "num_fps" in kwargs:
        cfg.patch.num_fps = int(kwargs["num_fps"])

    # --- preprocessing: build the first-level downsampled clouds ----------------
    src_o3d = o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(np.ascontiguousarray(src_pts))
    tgt_o3d = o3d.geometry.PointCloud()
    tgt_o3d.points = o3d.utility.Vector3dVector(np.ascontiguousarray(tgt_pts))

    if voxel_size and float(voxel_size) > 0:
        ds = float(voxel_size)
    else:
        # Sphericity-based automatic voxel-size estimation (the zero-shot core).
        ds, _sphericity, _ = worker.sphericity_fn(src_o3d, tgt_o3d)
    cfg.data.downsample = ds

    src_ds = o3d.geometry.PointCloud.voxel_down_sample(src_o3d, voxel_size=ds)
    tgt_ds = o3d.geometry.PointCloud.voxel_down_sample(tgt_o3d, voxel_size=ds)
    src_fds = np.asarray(src_ds.points, dtype=np.float32)
    tgt_fds = np.asarray(tgt_ds.points, dtype=np.float32)
    np.random.shuffle(src_fds)
    np.random.shuffle(tgt_fds)
    print(f"worker: voxel {ds:.4f} -> fds src {len(src_fds)} / tgt {len(tgt_fds)} pts",
          file=sys.stderr)
    if len(src_fds) < cfg.patch.num_fps or len(tgt_fds) < cfg.patch.num_fps:
        print(f"worker: WARNING fds points < num_fps ({cfg.patch.num_fps}); "
              f"FPS will sample with replacement", file=sys.stderr)

    dev = worker.device
    data_source = {
        "src_fds_pcd": torch.tensor(src_fds, dtype=torch.float32, device=dev),
        "tgt_fds_pcd": torch.tensor(tgt_fds, dtype=torch.float32, device=dev),
        "is_aligned_to_global_z": bool(cfg.patch.is_aligned_to_global_z),
        "src_id": "source",
        "tgt_id": "target",
    }

    with torch.no_grad():
        pose, _times, num_inliers, num_mutual, _inlier_ind, scales_used = \
            worker.model(data_source)

    if pose is None:
        return list(_IDENTITY16), {
            "converged": False,
            "num_inliers": 0,
            "note": "BUFFER-X returned no pose (too few correspondences)",
        }
    T = [float(v) for v in np.asarray(pose, dtype=np.float64).reshape(-1)]
    n_inl = int(num_inliers)
    return T, {
        "converged": n_inl > 0,
        "num_inliers": n_inl,
        "num_mutual_inliers": int(num_mutual),
        "scales_used": int(scales_used),
    }


class Worker:
    """Holds the heavy modules and the loaded BUFFER-X model (if available)."""

    def __init__(self, np, torch, o3d, model, cfg, sphericity_fn, model_error):
        self.np = np
        self.torch = torch
        self.o3d = o3d
        self.model = model              # None when the core/weights are missing
        self.cfg = cfg                  # easydict config (None when model is None)
        self.sphericity_fn = sphericity_fn  # upstream auto voxel-size estimator
        self.model_error = model_error  # why model is None (for the fallback note)
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))

    # ------------------------------------------------------------------ ops
    def ping(self, req):
        return {
            "device": "cuda" if self.torch.cuda.is_available() else "cpu",
            "torch": self.torch.__version__,
            "cuda": 1 if self.torch.cuda.is_available() else 0,
            "bufferx": 1 if self.model is not None else 0,
        }

    def register(self, req):
        np = self.np
        t0 = time.time()
        src = np.load(req["source"])
        tgt = np.load(req["target"])
        src_pts = np.asarray(src["xyz"], dtype=np.float64)
        tgt_pts = np.asarray(tgt["xyz"], dtype=np.float64)
        print(f"worker: source {len(src_pts)} pts, target {len(tgt_pts)} pts",
              file=sys.stderr)

        voxel_size = float(req.get("voxel_size", 0.0))

        # --- inference kwargs: generic typed table (refine is C++-side) -------
        kwargs = {}
        for k, v in req.items():
            if k in _RESERVED or v is None:
                continue
            if k in _BX_KW:
                kwargs[k] = _BX_KW[k](v)
            else:
                print(f"worker: WARNING unknown param {k!r} ignored",
                      file=sys.stderr)

        try:
            transform, info = _run_bufferx(self, src_pts, tgt_pts,
                                           device=self.device,
                                           voxel_size=voxel_size, **kwargs)
        finally:
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()

        seconds = time.time() - t0
        note = info.get("note")
        if note:
            print(f"worker: {note}", file=sys.stderr)
        print(f"worker: done on {self.device.type} in {seconds:.1f}s "
              f"(converged {bool(info.get('converged'))})", file=sys.stderr)

        result = {
            "transform": [float(v) for v in np.asarray(transform).ravel()],
            "converged": 1 if info.get("converged") else 0,
            "device": self.device.type,
            "seconds": round(seconds, 3),
            "cache_hit": 0,
        }
        # Only surface real quality numbers; never fabricate them in fallback.
        for k in ("num_inliers", "num_mutual_inliers", "scales_used"):
            if k in info:
                result[k] = int(info[k])
        if note:
            result["note"] = note
        return result


def _send(proto, obj):
    proto.write(json.dumps(obj, separators=(",", ":")) + "\n")
    proto.flush()


def _import_heavy(weights_dir=None):
    """Returns a Worker; raises on missing/broken core deps (numpy/torch).

    The BUFFER-X core (vendored upstream package + its CUDA extensions) and the
    weights are loaded best-effort: when they are not available, the worker still
    comes up `ready` (so ping works and register answers with the identity
    fallback) and remembers why.
    """
    import numpy as np
    import torch
    import open3d as o3d

    model, cfg, sphericity_fn, model_error = None, None, None, None
    try:
        # The upstream knn_cuda package (custom CUDA k-NN) had its GitHub repo
        # removed, so when it is not installed we fall back to the pure-torch
        # equivalent in ./_shims (appended LAST, so a real install still wins).
        try:
            import knn_cuda  # noqa: F401
        except Exception:
            _shims = os.path.join(_HERE, "_shims")
            if _shims not in sys.path:
                sys.path.append(_shims)

        # Importing BufferX pulls in pointnet2_ops / knn_cuda / torch_batch_svd
        # (CUDA-compiled); a missing extension raises here and we fall back.
        from config import make_cfg
        from models.BUFFERX import BufferX
        from utils.tools import sphericity_based_voxel_analysis

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = make_cfg("3DMatch", _UPSTREAM)
        cfg[cfg.data.dataset] = cfg.copy()  # upstream test.py does this
        cfg.stage = "test"

        model = BufferX(cfg)
        wd = weights_dir or os.path.join(_HERE, "weights")
        for stage in cfg.train.all_stage:  # ["Desc", "Pose"]
            ckpt = os.path.join(wd, "snapshot", _EXPERIMENT_ID, stage, "best.pth")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"missing checkpoint: {ckpt}")
            state = torch.load(ckpt, map_location=device)
            stage_dict = {k: v for k, v in state.items() if stage in k}
            md = model.state_dict()
            md.update(stage_dict)
            model.load_state_dict(md)
        model = model.to(device)
        model.eval()
        sphericity_fn = sphericity_based_voxel_analysis
        print(f"worker: BUFFER-X model loaded ({_EXPERIMENT_ID}) on {device}",
              file=sys.stderr)
    except BaseException as exc:  # noqa: BLE001 — tolerate; report in fallback
        model, cfg, sphericity_fn = None, None, None
        model_error = f"{type(exc).__name__}: {exc}"
        print(f"worker: BUFFER-X core/weights unavailable ({model_error}); "
              f"register will return identity (converged=false)", file=sys.stderr)
    return Worker(np, torch, o3d, model, cfg, sphericity_fn, model_error)


def serve():
    # stdout hijack — MUST be first: anything torch/CUDA extensions print to fd 1
    # would corrupt the protocol stream. The protocol keeps a private dup of the
    # original stdout; fd 1 is then pointed at stderr.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    _send(proto, {"event": "loading", "pid": os.getpid()})
    try:
        worker = _import_heavy(os.environ.get("CLOUDCROPPER_BUFFERX_WEIGHTS"))
    except BaseException as exc:  # noqa: BLE001 — report anything, then die
        _send(proto, {"event": "fatal", "error": _error_obj(exc)})
        return 1
    _send(proto, {
        "event": "ready",
        "pid": os.getpid(),
        "device": "cuda" if worker.torch.cuda.is_available() else "cpu",
        "torch": worker.torch.__version__,
        "bufferx": 1 if worker.model is not None else 0,
    })

    ops = {"ping": worker.ping, "register": worker.register}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            op = req.get("op")
            if op == "shutdown":
                _send(proto, {"id": rid, "ok": True, "result": {}})
                return 0
            if op not in ops:
                raise ValueError(f"unknown op {op!r}")
            _send(proto, {"id": rid, "ok": True, "result": ops[op](req)})
        except Exception as exc:  # noqa: BLE001 — the loop must survive any op
            _send(proto, {"id": rid, "ok": False, "error": _error_obj(exc)})
    return 0  # stdin EOF: parent is gone, don't linger as an orphan


def oneshot(source, target):
    """Debug mode: one registration, result printed as JSON to stdout."""
    worker = _import_heavy(os.environ.get("CLOUDCROPPER_BUFFERX_WEIGHTS"))
    res = worker.register({"source": source, "target": target,
                           "target_key": "", "device": "cuda",
                           "voxel_size": 0.0, "refine": False})
    json.dump(res, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if res["converged"] else 1


def main(argv):
    if len(argv) >= 2 and argv[1] == "--oneshot":
        if len(argv) != 4:
            print("usage: bufferx_worker.py --oneshot <source.npz> <target.npz>",
                  file=sys.stderr)
            return 2
        return oneshot(argv[2], argv[3])
    return serve()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
