# `.npz` Template Runtime Schema (reference)

> Reference snippet for [`../05-sdf-pipeline.md`](../05-sdf-pipeline.md). Units are
> **meters** everywhere. Frame is the **template/canonical frame** (see doc 05 §6).
> All arrays little-endian. `N` = point count.

## Per-point arrays (one row per point, `shape[0] == N`)

| key               | dtype     | shape   | meaning                                  | CloudCropper column |
|-------------------|-----------|---------|------------------------------------------|---------------------|
| `surface_points`  | `<f4`     | `(N,3)` | point positions, meters                  | `positions()`       |
| `surface_normals` | `<f4`     | `(N,3)` | unit normals (optional, recommended)     | `"normal"` column   |

## Per-cloud arrays (NOT per-point — `shape[0] != N`)

| key                    | dtype  | shape   | meaning                                   |
|------------------------|--------|---------|-------------------------------------------|
| `bbox_min`             | `<f4`  | `(3,)`  | AABB min corner, meters                   |
| `bbox_max`             | `<f4`  | `(3,)`  | AABB max corner, meters                   |
| `bbox_center`          | `<f4`  | `(3,)`  | `(bbox_min+bbox_max)/2`, meters           |
| `canonical_center`     | `<f4`  | `(1,3)` | origin of the canonical frame, meters     |
| `canonical_axis`       | `<f4`  | `(1,3)` | unit axis of the canonical frame          |
| `tailstock_tip_local`  | `<f4`  | `(3,)`  | tip point, template frame, meters         |
| `bar_axis_point_local` | `<f4`  | `(3,)`  | a point on the bar axis, template frame   |
| `bar_axis_dir_local`   | `<f4`  | `(3,)`  | unit direction of the bar axis            |
| `object_pose_origin_local` | `<f4` | `(3,)` | generic object pose origin, template frame, meters |
| `object_pose_dir_local` | `<f4` | `(3,)` | generic object pose direction, unit vector |
| `point_spacing_m`      | `<f4`  | `()`    | nominal point spacing, meters (scalar)    |

## Optional provenance array (recommended)

| key       | dtype  | shape | meaning                                                    |
|-----------|--------|-------|------------------------------------------------------------|
| `__meta__`| `|S1`  | `(L,)`| UTF-8 JSON bytes: `{"units":"m","frame":"template","src":…}`|

## numpy producer reference

```python
import numpy as np
np.savez(
    "tailstock_seating_template.npz",
    surface_points        = pts.astype(np.float32),        # (N,3)
    surface_normals       = nrm.astype(np.float32),        # (N,3)
    bbox_min              = pts.min(0).astype(np.float32),  # (3,)
    bbox_max              = pts.max(0).astype(np.float32),  # (3,)
    bbox_center           = ((pts.min(0)+pts.max(0))/2).astype(np.float32),
    canonical_center      = c_center.reshape(1,3).astype(np.float32),
    canonical_axis        = c_axis.reshape(1,3).astype(np.float32),
    tailstock_tip_local   = tip.astype(np.float32),         # (3,)
    bar_axis_point_local  = bar_p.astype(np.float32),       # (3,)
    bar_axis_dir_local    = bar_d.astype(np.float32),       # (3,)
    object_pose_origin_local = pose_o.astype(np.float32),    # (3,)
    object_pose_dir_local    = pose_d.astype(np.float32),    # (3,), unit
    point_spacing_m       = np.float32(spacing),            # scalar ()
)
```

## numpy consumer reference

```python
z = np.load("tailstock_seating_template.npz")
pts  = z["surface_points"]     # (N,3) float32, meters
nrm  = z["surface_normals"]    # (N,3) float32, unit
spm  = float(z["point_spacing_m"])
pose_o = z["object_pose_origin_local"]  # (3,) float32, meters
pose_d = z["object_pose_dir_local"]     # (3,) float32, unit
```
