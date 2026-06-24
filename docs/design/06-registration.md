# 06 — Registration backends & panel

## Goal

Confirm that a cropped/saved cloud (the **target**, e.g. the part template just
exported) and another scan (the **source**, the cloud open in the viewer) align:
estimate the rigid transform `T` with `p_target = T · p_source`, visualize the
overlay, and bake/save the aligned source.

## Layout — one package per algorithm

```
backend/registration/
├── include/cloudcropper/registration/registration.hpp   # public API (no Eigen)
├── common/            # dispatcher, Vec3<->Eigen bridges, shared metric, PythonWorker
├── gicp/              # small_gicp wrapper: ICP / Plane-ICP / GICP / VGICP
├── kiss_matcher/      # KISS-Matcher (FetchContent from GitHub) + fetch .cmake
└── gradient_sdf_gpu/  # gradient-SDF: persistent Python worker + vendored package
```

One static lib `cloudcropper::registration`, built when the vcpkg `registration`
feature provides Eigen + small_gicp (presets `vcpkg`/`gui`). KISS-Matcher is
fetched from GitHub at configure time (`CLOUDCROPPER_WITH_KISS_MATCHER=ON`,
pinned tag) and adds `CLOUDCROPPER_HAS_KISS_MATCHER`; everything else defines
`CLOUDCROPPER_HAS_REGISTRATION`. The `dev` preset compiles it all out.

## Public API (dependency-free header)

```cpp
enum class RegAlgo { Icp, PlaneIcp, Gicp, VGicp, KissMatcher, KissGicp, GradientSdfGpu };
Result<RegResult> registerClouds(const PointCloud& source, const PointCloud& target,
                                 const RegOptions& opt = {});
void applyTransform(PointCloud&, const std::array<double,16>& T);  // row-major
```

`RegResult.transform` is row-major 4×4, target ← source. After any backend runs,
the dispatcher recomputes a **backend-independent metric** so algorithms are
comparable: RMS source→target nearest-neighbour distance after alignment
(sampled ≤50k) and the inlier count (< 3× target spacing), using the core
GridIndex.

`applyTransform()` bakes the same rigid transform into cloud-associated pose
metadata. Point-like metadata (`object_pose_origin_local`, `tailstock_tip_local`,
`bar_axis_point_local`, `canonical_center`) is transformed with `R·p+t`;
direction-like metadata (`object_pose_dir_local`, `bar_axis_dir_local`,
`canonical_axis`) is rotated with `R` and normalized. If bbox metadata is present,
it is recomputed from the transformed points.

Auto parameters derive from `estimatePointSpacing(target)`: GICP downsample
≈ 2× spacing, max-corr ≈ 20× spacing; KISS resolution ≈ 1.5× spacing with a
0.75× retry when the final-inlier count is low (its 0.3 m default assumes
outdoor LiDAR and degenerates on dense/small clouds).

## gradient-SDF (gradient_sdf_gpu/)

The algorithm lives in the vendored Python package
(`gradient_sdf_gpu/python/gradient_sdf_registration/`, PyTorch, FFT
exhaustive-grid initialization) and runs in a persistent worker process
(`gsdf_worker.py`) spoken to over JSON-lines + NPZ temp files. Every key in
`config/gradient-sdf-gpu.yaml` is forwarded to the worker, so all algorithm
knobs are yaml-settable without C++ changes. There is no native fallback — a
broken Python environment surfaces as the registration error.

### Uncertainty channel (ported into the vendored package)

The (since removed) native C++ backend's GPIS-style variance channel was
ported into the Python package, marked with `# CloudCropper:` comments (see
`python/VENDORED.md`). With `uncertainty: true` (yaml) / `--reg-uncertainty`:

- **Field**: `GradientSDFField.add_uncertainty_channel(points, normals)` adds a
  5th grid channel, per voxel `var = σ_f²·(1−ρ₃ᐟ₂(r)) + s² + (0.25h)²` —
  Matern-3/2 data proximity (`ρ₃ᐟ₂(r) = (1+√3r/ℓ)e^(−√3r/ℓ)`, `ℓ = 3·spacing`,
  `σ_f = trunc = max(trunc_mul·spacing, 1% bbox diag)`), the local plane
  residual `s²` of the nearest target point's 8-NN, and the voxel quantization
  floor.
- **Loss**: `RobustSDFLoss(..., variances=u)` switches to the heteroscedastic
  Cauchy `ρ = (c²/2)·log1p(r²/(c²u))` with `u = var/median_var` — confident
  voxels count more, occluded/sparse ones are down-weighted. `u` is **detached**
  before entering the loss: ρ is decreasing in u, so an attached u would reward
  pushing points into unobserved space.
- **Trust score** (computed in the worker on the FINAL, post-GICP pose):
  normalized residuals `u_i = sdf²/var` evaluated only inside the confident
  zone (`var < 0.5·trunc²`); `confidence` = fraction with `u_i < 4` (floored
  denominator at 5% of the source), `norm_residual` = mean `u_i`. A
  plausible-but-wrong pose drops points into confident zones the field cannot
  explain → low confidence even when rmse looks fine.

## Viewer panel (right side) & CLI

`Registration` window: TARGET (path / *Use last export* — wired from the
Crop+Export handler), ALGORITHM (combo + per-algo params), accented *Register*
(disabled without source+target), RESULT (4×4, RMSE, inliers, *Apply to source*,
*Save aligned source*, *Reset*). The solve runs on a `std::thread` over **copies**
of both clouds (UI stays interactive; results hand off under a mutex + atomic
state). The scene draws the source through the result transform (orange tint)
over the cyan-tinted target via the PointRenderer `model`/`tint` uniforms.

CLI: `cloudcropper register <source> <target> [--reg-algo …] [-o aligned.npz]` —
used by the synthetic ctest cases (known-transform recovery: GICP 12°, KISS &
gradient-SDF 90° + offset, gradient-SDF 60%-overlap partial). When the aligned
output is NPZ, known per-cloud metadata such as object pose is preserved.
