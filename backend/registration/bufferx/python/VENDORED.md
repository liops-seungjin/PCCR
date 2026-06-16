# Vendored: BUFFER-X (zero-shot point cloud registration)

| | |
|---|---|
| Upstream | <https://github.com/MIT-SPARK/BUFFER-X> (ICCV 2025 Highlight, MIT-SPARK) |
| License | MIT |
| Commit | depth-1 clone of `main` on 2026-06-16 (re-pin if updated) |
| Vendored | **YES** — `./bufferx_upstream/` (config / models / utils / dataset / loss + support); the worker adds it to `sys.path` |
| Excluded | `.git`, `fig/`, `.github/`, `__pycache__/`, datasets/snapshots, the weights themselves (see `download_weights.sh`; `weights/` is `.gitignored`) |
| Weights | **present**: `./weights/snapshot/threedmatch/{Desc,Pose}/best.pth` (~3.67 MB each), downloaded; `.gitignored`, never committed |

## Status — wiring DONE; real inference gated on CUDA extensions

- ✅ Core vendored under `./bufferx_upstream/`; weights downloaded under `./weights/`.
- ✅ `bufferx_worker.py` `_run_bufferx()` is wired to the real upstream path:
  sphericity-based voxel estimation + voxel downsample → `data_source` →
  `model(data_source)` (loads `BufferX(make_cfg("3DMatch"))` + per-stage weights
  once at startup). Returns the 4×4 (source→target) + `num_inliers` /
  `num_mutual_inliers` / `scales_used`.
- ✅ Pure-python deps installed: `numpy`, `open3d`, `einops`, `kornia`, `easydict`.
- ⏳ **Remaining: the CUDA-compiled extensions** `pointnet2_ops`, `knn_cuda`,
  `torch_batch_svd` (see `requirements.txt` for the exact external sources).
  These compile third-party code from external GitHub repos, so they are NOT
  auto-installed. **Until installed, the worker comes up `ready` and `register`
  returns an honest identity result with `converged:false` and a `note` — it
  never fabricates inlier numbers.**

To finish: install the three CUDA extensions (see `requirements.txt`; for an
RTX 50-series set `TORCH_CUDA_ARCH_LIST="12.0"`), then run a real
`--oneshot src.npz tgt.npz` check. No code changes are required after that — the
worker's `model is None` guard flips to the real path automatically once the
imports succeed.

See `docs/design/_bufferx-upstream-notes.md` for the full sourced recon.

## Key upstream facts (recon 2026-06-16)

- **No sparse-conv backbone** (no MinkowskiEngine/spconv/torchsparse). Descriptor
  = SpinNet-style `MiniSpinNet`; pose stage uses `pointnet2_ops` / `KNN_CUDA` /
  `torch-batch-svd` / `cpp_wrappers` CUDA extensions.
- **Input = XYZ only** (no normals/colors). Scale handled automatically by
  density-aware radius estimation + per-patch normalization (the zero-shot core).
- **No single-pair `register(src, tgt)` API.** Inference is `model(data_source)`,
  where `data_source` mimics the dataset-loader output; it returns
  `(trans_est, times, num_inliers, num_mutual_inliers, num_inlier_ind, scales)` —
  **no fitness scalar** (derive one from the inlier ratio if needed).
- **Two source checkpoints** (`threedmatch`, `kitti`), each used zero-shot;
  3DMatch = indoor/general generalist (the default for CloudCropper).

## To finish the integration

Only one step remains — installing the three CUDA extensions (the vendoring,
weight download, and `_run_bufferx()` wiring are already done):

1. Install `pointnet2_ops`, `knn_cuda`, `torch_batch_svd` from the external
   sources listed in `requirements.txt` (for an RTX 50-series set
   `TORCH_CUDA_ARCH_LIST="12.0"`). These compile third-party CUDA code, so do it
   deliberately, not via an automated agent.
2. Re-run a real `--oneshot src.npz tgt.npz` check (and the C++ bufferx tests).

No code changes are required after that: the worker's `model is None` guard in
`_run_bufferx()` flips to the real `model(data_source)` path automatically once
the imports succeed. The C++ bridge (`../bufferx_backend.cpp`) and the JSON-lines
protocol are complete and do not change.
