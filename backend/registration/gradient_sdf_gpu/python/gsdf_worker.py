#!/usr/bin/env python3
"""CloudCropper gradient-SDF GPU worker (persistent process).

Spawned once by the C++ gsdf-gpu backend (PythonWorker) and kept alive so the
multi-second torch/open3d import cost is paid a single time. Speaks a JSON-lines
protocol: one UTF-8 JSON object per line on stdin (request) and stdout
(response), strictly one request in flight. Point clouds are handed over as
NPZ files (CloudCropper exports: key "xyz" (N,3) f4, optional "normal").

Handshake (before any request): the worker emits
    {"event":"loading","pid":<pid>}            immediately, then after imports
    {"event":"ready","pid":...,"device":"cuda"|"cpu","torch":"<version>"}
or, if the heavy imports fail,
    {"event":"fatal","error":{"type","message","traceback"}}  and exits 1.

Ops:
    {"id":N,"op":"ping"}      -> {"id":N,"ok":true,"result":{device/torch/cuda}}
    {"id":N,"op":"register","source":...,"target":...,"target_key":...,
     "device":"cuda","resolution":100,"poisson_depth":9,"voxel_size":0.0,
     "refine":true,"uncertainty":true,"trunc_mul":4.0, ...}
                              -> {"id":N,"ok":true,"result":{"transform":[16],
                                  "converged":0|1,"loss":...,"iou":...,
                                  "confidence":...,"norm_residual":...,
                                  "device":...,"seconds":...,"cache_hit":0|1}}
    {"id":N,"op":"shutdown"}  -> {"id":N,"ok":true,"result":{}} then exit 0

Every other request key is matched against the typed tables below (_REG_KW =
PCARegistration.register() kwargs, _ENGINE_KW = its constructor) so any knob in
config/gradient-sdf-gpu.yaml reaches the algorithm; unknown keys are ignored
with a stderr warning. init_mode defaults to "fft" (exhaustive grid).

"uncertainty": true builds the GPIS variance channel on the SDF field
(heteroscedastic Cauchy weighting) and reports a trust score on the final
pose: confidence = fraction of chi-square-normalized residuals u = sdf²/var
below 4 inside the field's confident zone (var < 0.5·trunc²), norm_residual =
mean u there. Both are -1 when uncertainty is off.

Op failures answer {"id":N,"ok":false,"error":{...}} and keep the loop alive;
stdin EOF (parent died) exits cleanly. Diagnostics go to stderr, which the C++
side redirects to <tmpdir>/worker.log.

Debug use without the C++ side:
    python3 gsdf_worker.py --oneshot source.npz target.npz
"""
# NOTE: only stdlib imports at module level — the protocol bootstrap (loading/
# fatal events) must work even when torch/open3d are not installed.
import json
import os
import sys
import time
import traceback

# The vendored package sits next to this script; running the script directly
# puts that directory at sys.path[0], so `import gradient_sdf_registration`
# needs no install. (Explicit insert keeps --oneshot working from anywhere.)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


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
             "resolution", "poisson_depth", "refine", "uncertainty",
             "trunc_mul", "voxel_size"}

# PCARegistration.register() kwargs: name -> caster. Everything here is
# yaml-settable; the C++ side forwards yaml keys verbatim.
_REG_KW = {
    "n_steps": int, "early_stop_patience": int, "min_steps": int,
    "min_improvement": float, "loss_threshold": float,
    "gicp_voxel_size": float, "gicp_num_threads": int,
    "compute_iou": _to_bool, "iou_voxel_size": float, "iou_max_points": int,
    "distance_threshold": float, "normal_threshold": float,
    "normal_weight": float, "normal_radius": float,
    "normal_loss_weight": float, "normal_loss_radius": float,
    "normal_loss_max_nn": int, "use_normal_score": _to_bool,
    "init_mode": str, "fft_voxel_size": float, "fft_rotation_choice": str,
    "fft_topk": int, "fft_target_samples": int, "fft_peaks_per_rotation": int,
    "fft_min_peak_separation_m": float, "fft_refine_steps": int,
    "fft_expand_translation_frac": float, "fft_expand_yaw_deg": float,
    "fft_expand_tilt_deg": float, "fft_max_candidates": int,
    "yaw_prior_deg": float, "yaw_prior_tolerance_deg": float,
}

# PCARegistration constructor kwargs.
_ENGINE_KW = {
    "n_candidates": int, "cauchy_c": float, "learning_rate": float,
    "use_gradient_weighting": _to_bool, "use_amp": _to_bool,
    "pin_memory": _to_bool,
}


class Worker:
    """Holds the heavy modules and the single-entry SDF field cache."""

    def __init__(self, np, o3d, torch, gsdf_field_cls, pca_reg_cls):
        self.np = np
        self.o3d = o3d
        self.torch = torch
        self.GradientSDFField = gsdf_field_cls
        self.PCARegistration = pca_reg_cls
        self._cache_key = None    # target_key of the cached field
        self._cache_field = None  # GradientSDFField

    # ------------------------------------------------------------------ ops
    def ping(self, req):
        return {
            "device": "cuda" if self.torch.cuda.is_available() else "cpu",
            "torch": self.torch.__version__,
            "cuda": 1 if self.torch.cuda.is_available() else 0,
        }

    def register(self, req):
        np, o3d, torch = self.np, self.o3d, self.torch
        t0 = time.time()
        src = np.load(req["source"])
        tgt = np.load(req["target"])
        src_pts = np.asarray(src["xyz"], dtype=np.float64)
        tgt_pts = np.asarray(tgt["xyz"], dtype=np.float64)
        print(f"worker: source {len(src_pts)} pts, target {len(tgt_pts)} pts",
              file=sys.stderr)

        want = req.get("device", "cuda")
        device = torch.device(want if (want != "cuda" or torch.cuda.is_available())
                              else "cpu")
        if want == "cuda" and device.type != "cuda":
            print("worker: WARNING cuda requested but unavailable -> cpu",
                  file=sys.stderr)

        resolution = int(req.get("resolution", 100))
        poisson_depth = int(req.get("poisson_depth", 9))
        uncertainty = _to_bool(req.get("uncertainty", True))
        trunc_mul = float(req.get("trunc_mul", 4.0))
        target_key = req.get("target_key", "")

        # --- target -> Poisson mesh -> GradientSDFField (single-entry cache) ---
        cache_hit = bool(target_key) and target_key == self._cache_key
        if cache_hit:
            field = self._cache_field
            print(f"worker: SDF field cache hit ({target_key})", file=sys.stderr)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(tgt_pts)
            if "normal" in tgt.files:
                pcd.normals = o3d.utility.Vector3dVector(
                    np.asarray(tgt["normal"], dtype=np.float64))
            else:
                pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
                pcd.orient_normals_consistent_tangent_plane(30)
            mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth)
            dens = np.asarray(dens)
            mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.03))  # trim bubbles
            print(f"worker: poisson mesh {len(mesh.vertices)} v / "
                  f"{len(mesh.triangles)} f ({time.time()-t0:.1f}s)", file=sys.stderr)
            field = self.GradientSDFField(mesh, resolution=resolution, device=device)
            if uncertainty:
                spacing, trunc = self._spacing_and_trunc(tgt_pts, trunc_mul)
                field.add_uncertainty_channel(
                    tgt_pts, np.asarray(pcd.normals), trunc=trunc, spacing=spacing)
                print(f"worker: uncertainty channel (spacing {spacing:.4g}, "
                      f"trunc {trunc:.4g}, median var {field.median_variance:.4g})",
                      file=sys.stderr)
            self._cache_key = target_key or None
            self._cache_field = field if target_key else None

        # --- registration kwargs: explicit specials + generic typed tables ---
        engine_kwargs = {}
        kwargs = dict(
            use_gicp_refinement=_to_bool(req.get("refine", True)),
            use_uncertainty=uncertainty,
        )
        for k, v in req.items():
            if k in _RESERVED or v is None:
                continue
            if k in _REG_KW:
                kwargs[k] = _REG_KW[k](v)
            elif k in _ENGINE_KW:
                engine_kwargs[k] = _ENGINE_KW[k](v)
            else:
                print(f"worker: WARNING unknown param {k!r} ignored",
                      file=sys.stderr)
        kwargs.setdefault("init_mode", "fft")
        kwargs.setdefault("compute_iou", True)
        # voxel_size 0 means "use the reference default" — passing None would
        # turn the auto-downsample OFF and large sources then OOM the GPU.
        if float(req.get("voxel_size", 0.0)) > 0:
            kwargs["voxel_size"] = float(req["voxel_size"])

        reg = self.PCARegistration(device=device, **engine_kwargs)
        try:
            transform, info = reg.register(src_pts, field, target_points=tgt_pts,
                                           **kwargs)
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

        confidence, norm_residual = self._trust(field, np.asarray(transform),
                                                src_pts)

        # The reference's `converged` flag only reports the early-stop path; a
        # run that used every step but reached a tiny loss is still a success.
        ok = bool(info.get("converged", False)) or \
            float(info.get("final_loss", 1e9)) <= 1.0e-2
        seconds = time.time() - t0
        print(f"worker: done on {device.type} in {seconds:.1f}s "
              f"(loss {info.get('final_loss', 0):.4g}, iou {info.get('iou', -1):.3f}"
              f", conf {confidence:.3f}"
              f"{', cached field' if cache_hit else ''})", file=sys.stderr)
        return {
            "transform": [float(v) for v in self.np.asarray(transform).ravel()],
            "converged": 1 if ok else 0,
            "loss": float(info.get("final_loss", 0.0)),
            "iou": float(info.get("iou", -1.0)),
            "confidence": confidence,
            "norm_residual": norm_residual,
            "device": device.type,
            "seconds": round(seconds, 3),
            "cache_hit": 1 if cache_hit else 0,
        }

    # ------------------------------------------------------- uncertainty utils
    def _spacing_and_trunc(self, pts, trunc_mul):
        """Median 1-NN spacing (sampled) and the truncation band, exactly the
        native heuristics: trunc = max(trunc_mul * spacing, 1% bbox diag)."""
        np = self.np
        from scipy.spatial import cKDTree
        sample = pts if len(pts) <= 100_000 else pts[:: len(pts) // 100_000 + 1]
        d, _ = cKDTree(sample).query(sample, k=2, workers=-1)
        spacing = max(float(np.median(d[:, 1])), 1e-9)
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        return spacing, max(float(trunc_mul) * spacing, 0.01 * diag)

    def _trust(self, field, T, src_pts):
        """Trust score at the final pose (port of the native trustAt):
        chi-square-normalized residuals u = sdf^2/var, evaluated ONLY where the
        field claims to know the surface (var < 0.5*trunc^2). Decouples pose
        trust from overlap; denominator floored at 5% of the source so a
        handful of accidental matches cannot fake a high score."""
        if not getattr(field, "has_uncertainty", False):
            return -1.0, -1.0
        np, torch = self.np, self.torch
        sp = src_pts[:: max(1, len(src_pts) // 4000)]  # <=4k stride, like native
        if not len(sp):
            return -1.0, -1.0
        q = sp @ T[:3, :3].T + T[:3, 3]  # package convention: target <- source
        with torch.no_grad():
            qt = torch.as_tensor(q, dtype=torch.float32, device=field.device)
            sdf, _, var = field.query_sdf_and_gradient(qt, return_variance=True)
            sdf, var = sdf.float(), var.float()  # AMP may yield fp16
            zone = 0.5 * field.uncertainty_trunc ** 2  # confident zone
            mask = var < zone
            cnt = int(mask.sum())
            u = (sdf * sdf) / var.clamp_min(1e-12)
            ok = int(((u < 4.0) & mask).sum())
            conf = ok / max(float(cnt), 0.05 * len(sp))
            nres = float(u[mask].mean()) if cnt else -1.0
        return float(conf), nres


def _send(proto, obj):
    proto.write(json.dumps(obj, separators=(",", ":")) + "\n")
    proto.flush()


def _import_heavy():
    """Returns a Worker; raises on missing/broken dependencies."""
    import numpy as np
    import open3d as o3d
    import torch
    from gradient_sdf_registration import GradientSDFField, PCARegistration
    return Worker(np, o3d, torch, GradientSDFField, PCARegistration)


def serve():
    # stdout hijack — MUST be first: anything open3d/torch print to fd 1 would
    # corrupt the protocol stream. The protocol keeps a private dup of the
    # original stdout; fd 1 is then pointed at stderr.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    _send(proto, {"event": "loading", "pid": os.getpid()})
    try:
        worker = _import_heavy()
    except BaseException as exc:  # noqa: BLE001 — report anything, then die
        _send(proto, {"event": "fatal", "error": _error_obj(exc)})
        return 1
    _send(proto, {
        "event": "ready",
        "pid": os.getpid(),
        "device": "cuda" if worker.torch.cuda.is_available() else "cpu",
        "torch": worker.torch.__version__,
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
    worker = _import_heavy()
    res = worker.register({"source": source, "target": target,
                           "target_key": "", "device": "cuda",
                           "uncertainty": True, "trunc_mul": 4.0})
    json.dump(res, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if res["converged"] else 1


def main(argv):
    if len(argv) >= 2 and argv[1] == "--oneshot":
        if len(argv) != 4:
            print("usage: gsdf_worker.py --oneshot <source.npz> <target.npz>",
                  file=sys.stderr)
            return 2
        return oneshot(argv[2], argv[3])
    return serve()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
