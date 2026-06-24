# Vendored: BUFFER-X (zero-shot point cloud registration)

| | |
|---|---|
| Upstream | <https://github.com/MIT-SPARK/BUFFER-X> (ICCV 2025 Highlight, MIT-SPARK) |
| License | MIT |
| Commit | depth-1 clone of `main` on 2026-06-16 (re-pin if updated) |
| Vendored | **YES** — `./bufferx_upstream/` (config / models / utils / dataset / loss + support); the worker adds it to `sys.path` |
| Excluded | `.git`, `fig/`, `.github/`, `__pycache__/`, datasets/snapshots, the weights themselves (see `download_weights.sh`; `weights/` is `.gitignored`) |
| Weights | **present**: `./weights/snapshot/threedmatch/{Desc,Pose}/best.pth` (~3.67 MB each), downloaded; `.gitignored`, never committed |

## Status — real inference LIVE via pure-torch shims (2026-06-22)

- ✅ Core vendored under `./bufferx_upstream/`; weights downloaded under `./weights/`.
- ✅ `bufferx_worker.py` `_run_bufferx()` is wired to the real upstream path:
  sphericity-based voxel estimation + voxel downsample → `data_source` →
  `model(data_source)` (loads `BufferX(make_cfg("3DMatch"))` + per-stage weights
  once at startup). Returns the 4×4 (source→target) + `num_inliers` /
  `num_mutual_inliers` / `scales_used`.
- ✅ Runs on the `rap` conda env (`/home/sjjung/miniconda3/envs/rap/bin/python`,
  torch 2.12+cu128 with **sm_120/Blackwell** support) — set in `config/bufferx.yaml`.
- ✅ **The three CUDA extensions are replaced by pure-torch shims in `./_shims/`**
  (no Blackwell wheels exist for them):
  - `knn_cuda` → `cdist + topk` (pre-existing).
  - `pointnet2_ops` → `furthest_point_sample` / `gather_operation` /
    `ball_query` (chunked, exact CUDA fill semantics) / `grouping_operation`.
  - `torch_batch_svd` → `torch.linalg.svd` (returns the `(U,S,V)` triple).
  The worker appends `./_shims` to `sys.path` **last**, so a genuine install
  still wins; the `model is None` guard stays as the honest fallback when weights
  or deps are missing.
- ✅ Verified end-to-end on RTX 5060 (`--oneshot` + C++ `register`). On the
  `tests/data/reg_pairs/` sanity-check pairs, BUFFER-X is the strongest or
  tied-strongest candidate across object / indoor-fragment / Livox, especially
  versus RAP and plain GICP. The saved `_compare.tsv` does not include BUFFER-X
  rows; see `docs/research/bufferx-integration-oversight.md` §5 for the
  re-run table and caveats.

To re-verify: `bufferx_worker.py --oneshot src.npz tgt.npz` (expects
`ready` with `bufferx:1`, a non-identity transform, `converged:true`), then the
C++ `bufferx`/`bufferx-gicp` tests.

See `docs/design/_bufferx-upstream-notes.md` for the full sourced recon and
`./_shims/*.py` for the per-op equivalence notes.

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

## Integration complete

The vendoring, weight download, `_run_bufferx()` wiring, and the pure-torch
shims for the three CUDA extensions are all done — real inference runs on the
`rap` conda env (sm_120). No further setup is required on this box.

If you ever move to hardware with prebuilt wheels for `pointnet2_ops`,
`knn_cuda`, `torch_batch_svd` (external sources in `requirements.txt`), a genuine
install transparently takes precedence over `./_shims` (appended last). The C++
bridge (`../bufferx_backend.cpp`) and the JSON-lines protocol do not change.
