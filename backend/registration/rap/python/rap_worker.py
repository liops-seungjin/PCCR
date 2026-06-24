#!/usr/bin/env python3
"""CloudCropper RAP worker (persistent process).

Spawned once by the C++ rap backend (PythonWorker) and kept alive so the
multi-second torch import and the ~340 MB checkpoint load is paid a single time.
Speaks the SAME JSON-lines protocol as bufferx_worker.py: one UTF-8 JSON object
per line on stdin (request) and stdout (response), strictly one request in
flight. Point clouds are handed over as NPZ files (CloudCropper exports: key
"xyz" (N,3) f4, optional "normal"; RAP only needs "xyz").

RAP = "Register Any Point: Scaling 3D Point Cloud Registration by Flow Matching"
(arXiv:2512.01850, PRBonn, MIT), built on Rectified Point Flow (RPF). Pairwise
use feeds [target, source] as two "parts"; the model generates the assembled
cloud over N flow steps, per-part 4x4 WORLD poses are recovered and written to
`*_<part>_transform.txt`. The pairwise result CloudCropper wants is
    target<-source = inv(T_target) @ T_source         (row-major, p_target = T*p_source)
matching CloudCropper's RegResult convention.

REALIZED DESIGN (see docs/design/08-rap-backend.md):
  * conda env `rap` (torch 2.7.0+cu128) so the RTX 5060 / Blackwell (sm_120) is
    usable; flash-attn and pytorch3d are intentionally NOT installed.
  * A pure-torch `flash_attn` shim (SDPA drop-in, forces FLASH/EFFICIENT kernels)
    and a pure-torch `pytorch3d` shim live in `./_shims/`; the worker inserts
    that dir on sys.path so the upstream `import flash_attn` / `import pytorch3d`
    resolve to the shims.
  * Default model is the points-only variant `rap_12_po` (skips mini-SpinNet
    feature extraction), checkpoint `rap_model_12.ckpt`.
  * The RAP model + a Lightning Trainer are loaded ONCE at startup (hydra-compose
    of `RAP_inference` + `setup()` from sample.py). Each `register` writes the two
    clouds as PLY into a per-call sample dir, runs the upstream preprocessing
    (`process_point_clouds`, points-only) and `trainer.test(model, datamodule)`
    with a freshly-built datamodule -- the model weights stay resident, only the
    dataloader is rebuilt.

Handshake (before any request): the worker emits
    {"event":"loading","pid":<pid>}            immediately, then after imports
    {"event":"ready","pid":...,"device":"cuda"|"cpu","torch":"<version>",
     "rap":0|1}
or, if the core deps (numpy/torch) fail,
    {"event":"fatal","error":{"type","message","traceback"}}  and exits 1.

Ops:
    {"id":N,"op":"ping"}      -> {"id":N,"ok":true,"result":{device/torch/cuda/
                                  rap}}
    {"id":N,"op":"register","source":...,"target":...,"target_key":...,
     "device":"cuda","voxel_size":0.0,"refine":false, ...}
                              -> {"id":N,"ok":true,"result":{"transform":[16],
                                  "converged":0|1,"device":...,"seconds":...,
                                  "cache_hit":0|1, + real quality keys only when
                                  available}}
    {"id":N,"op":"shutdown"}  -> {"id":N,"ok":true,"result":{}} then exit 0

Every other request key is matched against the typed table below (_RAP_KW) so a
knob in config/rap.yaml reaches the algorithm; unknown keys are ignored with a
stderr warning. The GICP refine is done on the C++ side, so `refine` from the
bridge is always false here.

When the core / weights / shims are missing, the worker still comes up `ready`
and the `register` op returns an IDENTITY transform with `converged:false` and an
explanatory `note` -- it does NOT fabricate inlier numbers as if a real
alignment happened.

Op failures answer {"id":N,"ok":false,"error":{...}} and keep the loop alive;
stdin EOF (parent died) exits cleanly. Diagnostics go to stderr, which the C++
side redirects to <tmpdir>/worker.log.

Debug use without the C++ side:
    python3 rap_worker.py --oneshot source.npz target.npz
"""
# NOTE: only stdlib imports at module level -- the protocol bootstrap (loading/
# fatal events) must work even when torch is not installed.
import json
import os
import sys
import time
import traceback

# Self-insert the vendored upstream core, the dataset_process helpers, and the
# pure-torch shims (flash_attn / pytorch3d) onto sys.path -- like
# bufferx_worker.py does -- so no PYTHONPATH env is required from the C++ side.
# Order matters: _shims FIRST so the upstream `import flash_attn` / `import
# pytorch3d` resolve to the shims rather than any real (arch-mismatched) wheel.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SHIMS = os.path.join(_HERE, "_shims")
_UPSTREAM = os.path.join(_HERE, "rap_upstream")
_DATASET_PROCESS = os.path.join(_UPSTREAM, "dataset_process")
for _p in (_DATASET_PROCESS, _UPSTREAM, _HERE, _SHIMS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# expandable_segments keeps the 8 GB card from fragmenting under the global
# attention; must be set before torch's CUDA caching allocator initializes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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

# RAP inference kwargs that are SAFE to override per call: name -> caster.
# (`model` selects the checkpoint VARIANT; the checkpoint is loaded once at
# startup, so switching `model` to a variant whose weights were not loaded falls
# back to identity with a note -- keep it in lock-step with the `model` the
# worker started with for real inference.)
_RAP_KW = {
    "n_generations":            int,    # multi-hypothesis generations
    "inference_sampling_steps": int,    # flow-matching steps
    "des_r":                    float,  # mini-SpinNet feature radius (meters)
    "voxel_ratio":              float,  # adaptive per-part point budget ratio
    "allocation_method":        str,    # voxel_adaptive | point_count | spatial_coverage
    "min_points_per_part":      int,
    "max_points_per_part":      int,
    "target_points_per_scan":   int,    # random-downsample budget per cloud
    "model":                    str,    # rap_10 | rap_12 | rap_12_po (points-only)
}

# Identity 4x4 (row-major, target <- source) used by the no-core/no-weights path.
_IDENTITY16 = [1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 1.0]


def _run_rap(worker, src_pts, tgt_pts, *, device, voxel_size, **kwargs):
    """Real RAP inference (target<-source 4x4 + info dict).

    Replicates the proven demo.py pipeline WITHOUT reloading the model:
      1. write [target, source] as PLY into a per-call sample input dir,
      2. upstream `process_point_clouds` (points-only: feature_extraction_on=
         False, random downsample to target_points_per_scan) builds the sample
         folder + data_split under a fresh data_root,
      3. run the RESIDENT model via trainer.test on a freshly-built datamodule
         pointed at that data_root (weights stay resident; only the dataloader
         is rebuilt),
      4. read the two `*_<part>_transform.txt` WORLD poses and return
         T = inv(T_target) @ T_source  (row-major target<-source).

    `parts = [target, source]`, so the target is part index 0 / reference part.

    When the core/weights/shims are unavailable (worker.model is None), returns
    an IDENTITY transform with converged False and a note -- it MUST NOT invent
    inlier numbers, so those keys are simply absent in fallback mode.
    """
    if worker.model is None:
        return list(_IDENTITY16), {
            "converged": False,
            "note": worker.model_error
            or "RAP core/weights/shims not available; returning identity",
        }

    np = worker.np

    # Optional per-call overrides (safe knobs only).
    over = {}
    for k in ("n_generations", "inference_sampling_steps", "des_r", "voxel_ratio",
              "allocation_method", "min_points_per_part", "max_points_per_part",
              "target_points_per_scan", "model"):
        if k in kwargs:
            over[k] = kwargs[k]

    transform = worker.infer_pair(tgt_pts, src_pts, voxel_size=voxel_size,
                                  overrides=over)
    T = [float(v) for v in np.asarray(transform, dtype=np.float64).reshape(-1)]
    return T, {"converged": True}


class Worker:
    """Holds the heavy modules and the loaded RAP model (if available)."""

    def __init__(self, np, torch, model, cfg, deps, model_error):
        self.np = np
        self.torch = torch
        self.model = model              # None when the core/weights are missing
        self.cfg = cfg                  # base hydra cfg used to build per-call datamodules/trainers
        self.deps = deps                # dict of upstream callables/modules (None if no model)
        self.model_error = model_error  # why model is None (for the fallback note)
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))
        # Per-call inputs live under a persistent dir next to the script (NOT
        # /tmp, which this box wipes on power loss).
        self._work_root = os.path.join(_HERE, "rap_worker_runs")
        os.makedirs(self._work_root, exist_ok=True)
        self._call = 0
        # default per-cloud point budget for the 8 GB card's global attention
        self._default_tpps = int(os.environ.get("CLOUDCROPPER_RAP_TPPS", "8000"))

    # ------------------------------------------------------------------ ops
    def ping(self, req):
        return {
            "device": "cuda" if self.torch.cuda.is_available() else "cpu",
            "torch": self.torch.__version__,
            "cuda": 1 if self.torch.cuda.is_available() else 0,
            "rap": 1 if self.model is not None else 0,
        }

    # ------------------------------------------------------- real inference
    def infer_pair(self, tgt_pts, src_pts, *, voxel_size, overrides):
        """Run the resident model on one [target, source] pair; return the
        row-major 4x4 target<-source transform as a (16,) list-ish array."""
        np = self.np
        d = self.deps
        process_point_clouds = d["process_point_clouds"]
        hydra = d["hydra"]

        cleanup = []
        prev_cwd = os.getcwd()
        try:
            self._call += 1
            run_dir = os.path.join(self._work_root, f"call_{self._call:06d}")
            if os.path.isdir(run_dir):
                d["shutil"].rmtree(run_dir, ignore_errors=True)
            os.makedirs(run_dir, exist_ok=True)
            cleanup.append(run_dir)

            dataset_name = "pair"
            data_root = os.path.join(run_dir, "data")
            dataset_folder = os.path.join(data_root, dataset_name)
            log_dir = os.path.join(run_dir, "logs")
            os.makedirs(dataset_folder, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            # parts = [target, source]: part index 0 is the reference (target).
            # Stable part names so we can pick out the right transform files.
            loaded = [
                ("part0_target", np.asarray(tgt_pts, dtype=np.float64), None),
                ("part1_source", np.asarray(src_pts, dtype=np.float64), None),
            ]

            cfg = self.cfg
            steps = int(overrides.get("inference_sampling_steps",
                                      cfg.model.inference_sampling_steps))
            ngen = int(overrides.get("n_generations", cfg.model.n_generations))
            vs = float(voxel_size) if voxel_size and float(voxel_size) > 0 else 0.25
            tpps = int(overrides.get("target_points_per_scan",
                                     self._default_tpps))
            alloc = overrides.get("allocation_method", "voxel_adaptive")
            vratio = float(overrides.get("voxel_ratio", 0.05))
            minp = int(overrides.get("min_points_per_part", 200))
            maxp = int(overrides.get("max_points_per_part", 20000))

            # --- Step 1+2: preprocess (points-only) into the sample folder ----
            sample_output_dir = process_point_clouds(
                loaded_point_clouds=loaded,
                output_folder=dataset_folder,
                voxel_size=vs,
                des_r=float(overrides.get("des_r", 5.0)),
                remove_outliers=True,
                allocation_method=alloc,
                voxel_ratio=vratio,
                min_points_per_part=minp,
                max_points_per_part=maxp,
                global_seed=42,
                feature_extraction_on=False,        # points-only (rap_12_po)
                use_random_downsample=True,
                target_points_per_scan=tpps,
            )
            sample_name = os.path.basename(sample_output_dir)

            # data_split/val.txt: the datamodule reads split="val" even for test.
            split_dir = os.path.join(dataset_folder, "data_split")
            os.makedirs(split_dir, exist_ok=True)
            with open(os.path.join(split_dir, "val.txt"), "w") as f:
                f.write(sample_name + "\n")

            # --- Step 3: run the RESIDENT model on a fresh datamodule ---------
            # Rebuild ONLY the dataloader: point cfg.data at this data_root and
            # re-instantiate the datamodule. The model + trainer are resident.
            OmegaConf = d["OmegaConf"]
            data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=False))
            data_cfg.data_root = data_root
            data_cfg.dataset_names = [dataset_name]
            # honor per-call sampling steps / generations on the resident model
            self.model.inference_sampling_steps = steps
            self.model.n_generations = ngen

            datamodule = hydra.utils.instantiate(data_cfg)

            # The evaluator writes transforms under `trainer.log_dir` (which, with
            # no Lightning logger, == default_root_dir). That dir is baked into
            # the Trainer at construction, so build a FRESH trainer per call
            # pointed at this call's log dir -- cheap; the MODEL (the expensive
            # part) stays resident. Mirror sample.py:setup()'s trainer/callbacks.
            callbacks = []
            vis_cfg = cfg.get("visualizer", {})
            if vis_cfg:
                # save_dir defaults to ${log_dir} (the startup value); point it
                # at this call's log dir. max_samples_per_batch=0 keeps it from
                # rendering (we only need the evaluator's transform.txt files).
                callbacks.append(hydra.utils.instantiate(vis_cfg, save_dir=log_dir))
            trainer = hydra.utils.instantiate(
                cfg.trainer,
                callbacks=callbacks,
                default_root_dir=log_dir,
                enable_checkpointing=False,
                logger=False,
            )

            with self.torch.no_grad():
                trainer.test(model=self.model, datamodule=datamodule,
                             verbose=False)

            # --- Step 4: read per-part WORLD poses and make relative ----------
            results_dir = os.path.join(log_dir, "results", dataset_name)
            sample_results = os.path.join(results_dir, sample_name)
            if os.path.isdir(sample_results):
                results_dir = sample_results

            T_tgt = self._read_transform(results_dir, "part0_target", 0)
            T_src = self._read_transform(results_dir, "part1_source", 1)
            rel = np.linalg.inv(T_tgt) @ T_src   # row-major target<-source
            return rel
        finally:
            os.chdir(prev_cwd)
            for p in cleanup:
                d["shutil"].rmtree(p, ignore_errors=True)

    def _read_transform(self, results_dir, part_name, part_idx):
        # The evaluator names the file either with the input part filename
        # (e.g. ..._part0_target_transform.txt) or, in some builds, a positional
        # ..._part{idx:02d}_transform.txt -- try both.
        np = self.np
        glob = self.deps["glob"]
        pats = [
            os.path.join(results_dir, f"*generation00_{part_name}_transform.txt"),
            os.path.join(results_dir, f"*_{part_name}_transform.txt"),
            os.path.join(results_dir, f"*{part_name}*transform.txt"),
            os.path.join(results_dir, f"*generation00_part{part_idx:02d}_transform.txt"),
            os.path.join(results_dir, f"*_part{part_idx:02d}_transform.txt"),
        ]
        for pat in pats:
            matches = sorted(glob.glob(pat))
            if matches:
                return np.loadtxt(matches[0]).reshape(4, 4)
        raise FileNotFoundError(
            f"RAP produced no transform for {part_name!r} under {results_dir}")

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
            if k in _RAP_KW:
                kwargs[k] = _RAP_KW[k](v)
            else:
                print(f"worker: WARNING unknown param {k!r} ignored",
                      file=sys.stderr)

        try:
            transform, info = _run_rap(self, src_pts, tgt_pts,
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
        for k in ("num_inliers", "fitness"):
            if k in info:
                result[k] = info[k]
        if note:
            result["note"] = note
        return result


def _send(proto, obj):
    proto.write(json.dumps(obj, separators=(",", ":")) + "\n")
    proto.flush()


def _abs_weights_dir(weights_dir):
    return weights_dir or os.path.join(_UPSTREAM, "weights")


def _load_rap_model(np, torch, weights_dir):
    """Compose the RAP config and load the model + checkpoint ONCE, return
    (model, cfg, deps). Raises on any failure (caller degrades). A fresh Trainer
    + datamodule are built per call in Worker.infer_pair; only the model (the
    expensive ~340 MB checkpoint load) stays resident."""
    import glob
    import shutil
    import open3d as o3d  # noqa: F401  (used per call; fail fast if missing)
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    # The shimmed flash_attn / pytorch3d must import before the flow model.
    import flash_attn  # noqa: F401
    import pytorch3d   # noqa: F401

    model_variant = os.environ.get("CLOUDCROPPER_RAP_MODEL", "rap_12_po")
    ckpt_name = {
        "rap_12_po": "rap_model_12.ckpt",
        "rap_12":    "rap_model_12.ckpt",
        "rap_10":    "rap_model_10.ckpt",
    }.get(model_variant, "rap_model_12.ckpt")
    wd = _abs_weights_dir(weights_dir)
    ckpt_path = os.path.abspath(os.path.join(wd, ckpt_name))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"RAP checkpoint not found: {ckpt_path}")

    # demo/sample resolve config_path "./config" relative to rap_upstream; use
    # the absolute config dir so we are cwd-independent. Hydra may only be
    # initialized once per process -> clear any prior global state first.
    config_dir = os.path.join(_UPSTREAM, "config")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    overrides = [
        f"model={model_variant}",
        f"ckpt_path={ckpt_path}",
        "model.n_generations=1",
        "model.inference_sampling_steps=10",
        # keep the demo's batch/limits; 1 sample, 1 worker for the persistent box
        "data.batch_size=1",
        "data.num_workers=1",
        "visualizer.max_samples_per_batch=0",
    ]

    # sample.py:setup() and process_point_clouds use cwd-relative paths in spots;
    # run the compose + setup with cwd = rap_upstream to match the proven command.
    prev_cwd = os.getcwd()
    os.chdir(_UPSTREAM)
    try:
        with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = hydra.compose(config_name="RAP_inference", overrides=overrides)
        from sample import setup
        model, _datamodule, _trainer = setup(cfg)
        model.eval()
    finally:
        os.chdir(prev_cwd)

    from demo import process_point_clouds
    deps = {
        "o3d": o3d, "hydra": hydra, "OmegaConf": OmegaConf,
        "process_point_clouds": process_point_clouds,
        "glob": glob, "shutil": shutil,
    }
    return model, cfg, deps


def _import_heavy(weights_dir=None):
    """Returns a Worker; raises on missing/broken core deps (numpy/torch).

    The RAP core (vendored upstream package + shims) and the weights are loaded
    best-effort: when they are not available, the worker still comes up `ready`
    (so ping works and register answers with the identity fallback) and
    remembers why.
    """
    import numpy as np
    import torch

    model, cfg, deps, model_error = None, None, None, None
    try:
        model, cfg, deps = _load_rap_model(np, torch, weights_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        variant = os.environ.get("CLOUDCROPPER_RAP_MODEL", "rap_12_po")
        print(f"worker: RAP model loaded ({variant}) on {device}",
              file=sys.stderr)
    except BaseException as exc:  # noqa: BLE001 -- tolerate; report in fallback
        model, cfg, deps = None, None, None
        model_error = f"{type(exc).__name__}: {exc}"
        print(f"worker: RAP core/weights/shims unavailable ({model_error}); "
              f"register will return identity (converged=false)", file=sys.stderr)
    return Worker(np, torch, model, cfg, deps, model_error)


def serve():
    # stdout hijack -- MUST be first: anything torch/CUDA extensions print to fd
    # 1 would corrupt the protocol stream. The protocol keeps a private dup of
    # the original stdout; fd 1 is then pointed at stderr.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    _send(proto, {"event": "loading", "pid": os.getpid()})
    try:
        worker = _import_heavy(os.environ.get("CLOUDCROPPER_RAP_WEIGHTS"))
    except BaseException as exc:  # noqa: BLE001 -- report anything, then die
        _send(proto, {"event": "fatal", "error": _error_obj(exc)})
        return 1
    _send(proto, {
        "event": "ready",
        "pid": os.getpid(),
        "device": "cuda" if worker.torch.cuda.is_available() else "cpu",
        "torch": worker.torch.__version__,
        "rap": 1 if worker.model is not None else 0,
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
        except Exception as exc:  # noqa: BLE001 -- the loop must survive any op
            _send(proto, {"id": rid, "ok": False, "error": _error_obj(exc)})
    return 0  # stdin EOF: parent is gone, don't linger as an orphan


def oneshot(source, target):
    """Debug mode: one registration, result printed as JSON to stdout."""
    worker = _import_heavy(os.environ.get("CLOUDCROPPER_RAP_WEIGHTS"))
    res = worker.register({"source": source, "target": target,
                           "target_key": "", "device": "cuda",
                           "voxel_size": 0.0, "refine": False})
    json.dump(res, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if res["converged"] else 1


def main(argv):
    if len(argv) >= 2 and argv[1] == "--oneshot":
        if len(argv) != 4:
            print("usage: rap_worker.py --oneshot <source.npz> <target.npz>",
                  file=sys.stderr)
            return 2
        return oneshot(argv[2], argv[3])
    return serve()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
