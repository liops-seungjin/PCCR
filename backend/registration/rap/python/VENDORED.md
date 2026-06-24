# Vendored: RAP (Register Any Point — flow-matching point cloud registration)

| | |
|---|---|
| Upstream | <https://github.com/PRBonn/RAP> (arXiv:2512.01850, 2025, PRBonn) — built on [Rectified Point Flow](https://github.com/GradientSpaces/Rectified-Point-Flow) (RPF, NeurIPS 2025 Spotlight) |
| License | MIT (PRBonn/RAP, HF `YuePanEdward/RAP`, and upstream RPF) |
| Vendored | **YES** — the full PRBonn/RAP repo lives in `./rap_upstream/`; the worker self-inserts `_shims`, `rap_upstream`, and `rap_upstream/dataset_process` onto `sys.path` |
| Weights | `./rap_upstream/weights/` (`rap_model_12.ckpt` L (default), `rap_model_10.ckpt` M, `mini_spinnet_t.pth`); `.gitignored`, never committed. `download_weights.sh` fetches from HF `YuePanEdward/RAP` or ipb.uni-bonn.de `weights.zip` |
| Demo data | `./rap_upstream/demo_example_data/` (Livox source/target PLY pair used for the smoke test) |

## Status — REAL inference wired (2026-06-17)

- ✅ C++ bridge (`../rap_backend.cpp`) + JSON-lines/NPZ protocol complete.
- ✅ `rap_worker.py` loads the RAP model + checkpoint **once** at startup and runs
  the proven `demo.py` pipeline per `register` call (model resident; only the
  datamodule/trainer are rebuilt). See `docs/design/08-rap-backend.md` §0.2.
- ✅ Graceful degradation preserved: if the shims/core/weights are missing the
  worker still comes up `ready` (`rap:0`) and `register` returns an honest
  identity result with `converged:false` and a `note`. It NEVER fabricates
  inlier/quality numbers.

## The flash-attn / Blackwell SOLUTION (pure-torch shims)

The RAP flow model does a **bare `import flash_attn`** (no SDPA fallback) and
uses `pytorch3d`. The pinned flash-attn 2.7.4.post1 wheel does **not** support
consumer Blackwell (sm_120, RTX 50xx). Instead of installing those wheels we run
under a newer torch and shim the two imports:

- **conda env `rap`** = `/home/sjjung/miniconda3/envs/rap/bin/python`, **torch
  2.7.0+cu128** → runs on the RTX 5060 / Blackwell (sm_120). Also has
  diffusers / lightning / hydra / omegaconf / open3d / h5py / scipy / trimesh /
  sklearn / einops. **flash-attn and pytorch3d are intentionally NOT installed.**
- **`./_shims/`** (placed FIRST on `sys.path` by the worker):
  - `flash_attn.py` — an **SDPA drop-in** for
    `flash_attn.flash_attn_varlen_qkvpacked_func`. Forces the FLASH / EFFICIENT
    `sdpa_kernel` backends (the default MATH backend OOMs the global attention on
    the 8 GB card).
  - `pytorch3d/` — **real** `ops.ball_query`, `ops.sample_farthest_points`,
    `loss/chamfer`; **stubs** for `structures`, `renderer`,
    `ops.iterative_closest_point` (viz / eval only, never on the inference path).

So the upstream `import flash_attn` / `import pytorch3d` resolve to the shims and
no arch-mismatched wheels are needed.

## Default model: `rap_12_po` (points-only)

The default variant is **`rap_12_po`** (a points-only config copied from
`rap_12.yaml`): it skips mini-SpinNet feature extraction, so the worker runs with
`feature_extraction_on=False`. The full mini-SpinNet (feature) path is **not yet
validated** in this integration.

## One-time setup

```bash
bash backend/registration/rap/python/download_weights.sh   # -> rap_upstream/weights/
# conda env `rap` (torch 2.7+cu128) is the interpreter; set config/rap.yaml `python:` to it.
```

See `docs/design/08-rap-backend.md` (§0.2 realized design, §0.3 validation
commands) and `docs/design/_rap-upstream-notes.md` for the full recon.
