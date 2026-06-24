# 05 — SDF Pipeline: Normals, Mesh Reconstruction & Template Export

**Owner:** SDF-pipeline architect
**Scope:** how a CloudCropper-cropped point cloud becomes a high-precision
oriented-normal surface template, gets exported as `.ply` (inspection) + `.npz`
(runtime), and feeds a downstream Open3D-mesh → gradient-SDF stage.
**Coordinates with:** [`01-format-io.md`](01-format-io.md) (codecs),
[`03-core-architecture.md`](03-core-architecture.md) (core model, `SpatialIndex`,
nanoflann), [`00-architecture-overview.md`](00-architecture-overview.md).
**Status:** design + analysis (no production code in this doc).
**Units:** **meters everywhere.** This is load-bearing — see §4.4.

> **TL;DR.** CloudCropper owns *points → oriented-normal template → `.ply`/`.npz`*.
> It does **not** own mesh reconstruction or SDF generation — those stay in the
> downstream Open3D/Python stack that already hosts `GradientSDFField`. The single
> highest-value addition is **native PCA normal estimation in `core` over the
> already-linked nanoflann KD-tree**, exposed via `--estimate-normals`, plus an
> **NPZ "template" writer mode** that emits the meters schema and carries the small
> per-cloud metadata arrays.

---

## 0. The user's actual problem

They build high-precision point-cloud *buffers* of physical parts that have **no
CAD**, and use the point cloud **as** the CAD. The downstream consumer is a
`GradientSDFField`. Gradient-SDF is **not** fed raw points: SDF generation needs an
Open3D `TriangleMesh` (a *surface*), so a **mesh-reconstruction / SDF-generation
stage sits between the point cloud and the SDF.** Two concrete artifacts must come
out of CloudCropper:

- **(A) Inspection `.ply`** — meters, `x y z` (ideally `+ nx ny nz`), openable in
  RViz / Open3D / CloudCompare. Point-only PLY *cannot* go straight into
  `GradientSDFField`; a mesh/SDF step is required.
- **(B) Runtime `.npz`** — meters, a fixed schema (§4.2) carrying the surface
  points + normals **and** several small per-cloud metadata arrays
  (bbox/canonical/axis/spacing).

The crux question is **normal estimation**: when normals are absent, how do we get
them, and how do we make them *high-precision*?

---

## 1. Normal estimation analysis (the core)

### 1.1 What "a normal" is here, and why orientation is the hard half

A surface normal at a point is recovered by **local plane fitting**: take the `k`
nearest neighbours, form the `3×3` covariance matrix, and take the eigenvector of
the **smallest** eigenvalue. That gives a *line*, i.e. **two opposite candidate
directions** — Open3D's own docs note "covariance analysis produces two opposite
directions as normal candidates." Computing the direction is cheap and well-posed;
**globally orienting all normals consistently (and outward) is the hard, failure-
prone half.** Poisson reconstruction in particular *requires consistently oriented*
normals or it produces inside-out / bubbled surfaces.

### 1.2 Open3D path (current 2025-2026 API)

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("template.ply")  # meters

# (optional but recommended) denoise FIRST — see §1.4
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# 1. estimate direction (covariance / local plane fit)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=R, max_nn=K))

# 2a. global orientation toward a known sensor origin (BEST when available)
pcd.orient_normals_towards_camera_location(camera_location=sensor_origin_m)
#  or fixed direction:
# pcd.orient_normals_to_align_with_direction([0,0,1])

# 2b. otherwise propagate consistency via MST over the Riemannian graph
pcd.orient_normals_consistent_tangent_plane(k=30)
```

**`estimate_normals(search_param=...)`** — three search params:

| search param                          | knobs              | behaviour |
|---------------------------------------|--------------------|-----------|
| `KDTreeSearchParamKNN(knn)`           | `knn` (default 30) | fixed neighbour count; **scale-adaptive** to density, but at sharp edges/thin walls it can reach across a gap. |
| `KDTreeSearchParamRadius(radius)`     | `radius`           | fixed metric radius; respects geometry scale but **starves** in sparse regions (too few neighbours ⇒ unstable fit). |
| `KDTreeSearchParamHybrid(radius,max_nn)` | both            | **recommended.** Radius bounds the support to true-local geometry; `max_nn` caps cost and noise averaging. |

**Hybrid is the default recommendation.** Tie `radius` to point spacing — a good
starting point is `radius ≈ 3·point_spacing_m` (so `R≈3·` the §4.2
`point_spacing_m`), `max_nn ≈ 30–60`. Because we export `point_spacing_m`, the
downstream call can be **auto-tuned** from it rather than hand-set.

**Orientation functions:**
- `orient_normals_towards_camera_location(camera_location)` — flips each normal to
  face the sensor. **Use this whenever an acquisition viewpoint is known** — it is
  the most reliable global orientation for a single-view scan.
- `orient_normals_to_align_with_direction(direction)` — flips toward a fixed world
  direction; only valid if the part is roughly a height-field along that axis.
- `orient_normals_consistent_tangent_plane(k)` — builds a Riemannian graph over `k`
  neighbours and propagates orientation along a **minimum spanning tree** (Hoppe et
  al. 1992). The fallback when no viewpoint is known. Caveats: known to be **slow /
  occasionally non-terminating on pathological inputs** (Open3D issue #4983) and it
  can flip whole patches across thin walls or disconnected components.

### 1.3 Effect of density & noise; hybrid vs knn vs radius

- **Density:** kNN auto-adapts to density (always `k` points) but its *metric* scale
  drifts with density; radius keeps metric scale fixed but its *count* drifts. For a
  machined part scanned at roughly uniform spacing, **hybrid** gives the best of both.
- **Noise:** the smallest-eigenvalue eigenvector is a *least-squares plane* — larger
  `k`/radius averages out sensor noise (lower variance) but **bevels sharp edges and
  rounds corners** (bias). This bias/variance trade-off is the central tuning tension
  for a machined part with crisp features (the tailstock tip, bar edges). **Denoise
  first, then keep the support modest** rather than smoothing with a huge `k`.

### 1.4 Higher-precision options (opinionated ladder)

1. **Denoise before estimating.** `remove_statistical_outlier(nb_neighbors=20,
   std_ratio=2.0)` and/or `remove_radius_outlier(nb_points, radius)` remove fliers
   that otherwise tilt the local plane. **Always do this first.**
2. **Hybrid search tuned to `point_spacing_m`** (see §1.2) instead of a blind `knn=30`.
3. **Known acquisition viewpoint(s) for global orientation.** A single sensor origin
   per scan + `orient_normals_towards_camera_location` beats MST every time. For
   multi-view captures, orient *per view in its own frame* before merging.
4. **Multi-scale estimation.** Estimate at 2–3 radii and pick per point by surface
   variation (small radius near edges to preserve features, large radius on flats to
   suppress noise). Higher precision at higher cost; warranted for crisp machined
   features.
5. **Robust / weighted PCA.** Distance-weight neighbours (Gaussian on range) and/or
   iteratively reweight to reject the local-plane outliers — reduces edge bevel vs
   plain PCA. (Not in stock Open3D; a small native addition — see §1.5.)
6. **Re-estimate on the reconstructed mesh.** After Poisson, `mesh.compute_vertex_
   normals()` gives globally consistent, watertight-surface normals; if you need a
   clean oriented cloud you can sample the mesh back to points. Effectively
   *"orientation by reconstruction."*

### 1.5 Native C++ in `core` vs delegate to Open3D — **recommendation**

The estimation *kernel* (kNN → covariance → smallest-eigenvector) is small and the
**infrastructure already exists in `core`**: doc 03 §3.3 specifies a `SpatialIndex`
with a **nanoflann KD-tree** ("zero data copy — it indexes our `xyz_` span
directly"), and **nanoflann is already linked** (confirmed in `vcpkg.json`,
`CMakeLists.txt`). A `3×3` symmetric eigensolve is a closed-form / tiny routine — no
Eigen dependency needed (and doc 01 explicitly keeps Eigen out of core).

**Recommendation — a split, by responsibility:**

| sub-task | where | why |
|---|---|---|
| **Normal *direction*** (kNN-PCA) | **native `core`** | tiny, hot, reuses the nanoflann index already planned/linked; runs in the same headless C++ pass that crops + exports, so `--estimate-normals` produces a normal-carrying `.ply`/`.npz` with **zero Python on the export path**. |
| **Global *orientation*** | **viewpoint-flip native; MST delegated** | viewpoint flip is trivial (one dot-product sign test) and worth doing in core when a sensor origin is known. The full Hoppe MST-tangent-plane propagation is fiddly and already battle-tested in Open3D — **do not reimplement it**; if no viewpoint is known, leave the cloud's normals unoriented-but-consistent-locally and let the downstream Open3D step run `orient_normals_consistent_tangent_plane`. |
| **Mesh reconstruction + SDF** | **delegate to Open3D/Python** | see §1.7 / §2. |

**Trade-offs:**
- *Precision:* identical math (both do PCA); native lets us add weighted/robust PCA
  (§1.4-5) later without a Python hop.
- *Dependencies:* native adds **nothing** (nanoflann already in); delegating to Open3D
  pulls Open3D into the runtime path, which the project deliberately rejected for the
  viewer (doc 00 §4) — but Open3D is **already** required downstream for meshing/SDF,
  so it is not a *new* dependency *there*, only on the export hot path.
- *Where it runs:* native runs inside the existing headless CLI/pipeline; delegation
  runs in their Python preprocessing. The recommendation keeps the **fast, no-Python
  direction estimate in CloudCropper** and the **hard global-orientation/meshing in
  Open3D**, matching where each is strong.

### 1.6 Quality checks (validate orientation & accuracy)

- **Orientation consistency:** for each point, dot its normal with each neighbour's
  normal; the fraction of **negative** dots is a flip-rate. A consistently oriented
  surface has near-zero flip-rate except at true creases. Compute it over the
  nanoflann index as a one-pass diagnostic.
- **Outward check (viewpoint known):** `dot(normal, sensor_origin − point)` should be
  **> 0** for (almost) all points; report the violating fraction.
- **Accuracy vs local plane:** residual = mean distance of the `k` neighbours to the
  fitted plane; large residual ⇒ wrong scale or noise (flag for denoise/retune).
- **Curvature proxy:** `λ0 / (λ0+λ1+λ2)` (surface variation) — spikes localise edges;
  sanity-checks that crisp features survived.
- **Reconstruction round-trip:** Poisson + `compute_vertex_normals`, then compare to
  the estimated cloud normals; large disagreement reveals orientation errors.

### 1.7 Mesh-reconstruction & gradient-SDF background (for §2)

- **Poisson** `create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1,
  linear_fit=False, n_threads=-1) -> (mesh, densities)` — **requires oriented
  normals**; solves a regularized global problem ⇒ smooth, **watertight** surface;
  returns per-vertex `densities` for trimming low-support regions. **Best default for
  a CAD-less machined part you want as a closed SDF surface.**
- **Ball-Pivoting** `create_from_point_cloud_ball_pivoting(pcd, radii)` — **requires
  normals**; rolls a ball of each radius, interpolates nothing (vertices = original
  points) ⇒ preserves crisp features but leaves holes where spacing > ball. Good when
  fidelity to the exact measured points matters more than watertightness; `radii`
  tied to `point_spacing_m`.
- **Alpha shapes** `create_from_point_cloud_alpha_shape(pcd, alpha)` — **no normals
  needed**; generalises the convex hull. Quick blocky envelope; useful for a coarse
  proxy or when normals are untrustworthy.
- **Gradient-SDF:** a semi-implicit field storing per-voxel SDF **and its gradient**
  (≈ surface normal), midway between a voxel SDF and explicit surfels (Sommer et al.,
  *Gradient-SDF*, arXiv 2111.13652). Generated from a `TriangleMesh` by sampling
  signed distances + gradients on/around a grid (mesh→SDF kernels such as Bridson-
  style `sdf-gen`, `TriangleMeshDistance`, `sxyv/sdf`). This is **why normals matter
  twice**: once to *make* the mesh (Poisson/BPA) and once because the SDF's gradient
  field *is* a normal field — bad input normals corrupt both.

---

## 2. End-to-end pipeline

```
            ┌──────────────────────── CloudCropper (C++, this repo) ───────────────────────┐
            │                                                                              │
 capture /  │   load        crop / box       [denoise]      normal est.+orient   export    │
 accumulate │  (io: PLY/    (core CropEngine  (statistical/  (core PCA over       (io)      │
 hi-prec    │   PCD/NPZ →    OBB/AABB →        radius outlier  nanoflann KD-tree;  PLY +     │
 points ───►│   PointCloud)  index mask →      removal)        viewpoint flip)     NPZ       │
            │                gather())                                              writers) │
            │                                                                        │       │
            └────────────────────────────────────────────────────────────────────── │ ──────┘
                                                                                     ▼
                          ARTIFACTS:  template.ply  (inspection, meters, x y z [+nx ny nz])
                                      template.npz  (runtime, meters, schema §4.2)
                                                                                     │
            ┌──────────────── Downstream (Open3D / Python, their stack) ──────────── ▼ ──────┐
            │                                                                                 │
            │  load npz/ply ─► (orient_normals_consistent_tangent_plane if not yet oriented)  │
            │      │                                                                          │
            │      ▼                                                                          │
            │  mesh reconstruction ──►  TriangleMesh  ──►  mesh→SDF sampling  ──►  GradientSDFField
            │   Poisson | BallPivot | Alpha            (signed dist + gradient grid)          │
            │                                                                                 │
            └─────────────────────────────────────────────────────────────────────────────────┘
```

**Where CloudCropper sits (the boundary):** it produces the **cropped, attribute-
carrying template point cloud** and writes `.ply` + `.npz` (meters, with normals when
`--estimate-normals` is on). Everything from `TriangleMesh` onward — reconstruction
*and* SDF/gradient-SDF generation — is **downstream**, in the Open3D/Python stack that
already hosts `GradientSDFField`. CloudCropper deliberately does **not** link Open3D
(doc 00 §4); the handoff is the **two files**, not an in-process API.

**Which reconstruction for a CAD-less machined part:**

| situation | pick | why |
|---|---|---|
| want a **closed, watertight** surface for a clean SDF | **Poisson** (`depth 8–10`) | global smooth solve, fills small gaps, returns `densities` to trim hallucinated regions; needs good oriented normals (we provide them). |
| must **honor exact measured points / crisp edges**, gaps acceptable | **Ball-Pivoting** (`radii ≈ [1,2,4]·point_spacing_m`) | vertices stay on measured points; no smoothing bias at the tip/edges. |
| normals untrustworthy or only need a **coarse envelope** | **Alpha shape** (`alpha` ~ few·spacing) | needs no normals; quick proxy. |

Pragmatic default for these parts: **Poisson** with our exported oriented normals,
trimmed by `densities`, with **Ball-Pivoting as the feature-preserving alternative**
when the tip/edge geometry must not be rounded.

---

## 3. The two formats

### 3.1 `.ply` inspection format

Field layout (meters): `x y z` minimum; `+ nx ny nz` recommended. Full snippet:
[`schemas/ply-inspection.schema.md`](schemas/ply-inspection.schema.md).

**Confirmed against `src/io/ply.cpp`:** the writer emits `x y z` first
(`ply.cpp` ~L302), and **emits `nx ny nz` from a `"normal"` (F32, arity 3) column**
when that column is present and selected (`ply.cpp` ~L313-316). So as soon as
`--estimate-normals` populates a `"normal"` column, the existing PLY writer produces
the recommended inspection file with **no writer change**.

**Limitation (confirmed):** **vertex-first, no list properties** — the reader
rejects a non-`vertex` first element and list props on `vertex` (`ply.cpp` ~L177,
L180); the writer only ever emits a `vertex` element. ⇒ this PLY is **point-only,
no faces**, which is exactly why it **cannot** feed `GradientSDFField` directly and a
mesh stage is required (§2). `binary_little_endian` (default) or ASCII only.

### 3.2 `.npz` runtime schema

Formalized in [`schemas/template-npz.schema.md`](schemas/template-npz.schema.md).
Summary (meters, little-endian, `N`=points):

| key | dtype | shape | per-point? |
|---|---|---|---|
| `surface_points`  | `<f4` | `(N,3)` | yes |
| `surface_normals` | `<f4` | `(N,3)` | yes (optional, recommended) |
| `bbox_min` / `bbox_max` / `bbox_center` | `<f4` | `(3,)` | no |
| `canonical_center` / `canonical_axis` | `<f4` | `(1,3)` | no |
| `tailstock_tip_local` / `bar_axis_point_local` / `bar_axis_dir_local` | `<f4` | `(3,)` | no |
| `object_pose_origin_local` / `object_pose_dir_local` | `<f4` | `(3,)` | no |
| `point_spacing_m` | `<f4` | `()` scalar | no |

#### 3.2.1 Mapping each entry onto `NpzWriter` / `NpzReader` (`src/io/npz.cpp`)

**`surface_points` ↔ positions — there is a real gap.** The writer **hard-codes the
positions key as `"xyz"`** (`npz.cpp` ~L274: `addEntry("xyz", …)`). The runtime
schema wants the key **`surface_points`**. The reader only accepts `xyz`/`points` for
positions (`npz.cpp` ~L191). So today's NPZ output is **not** schema-conformant.
Two ways to close it:

- **(a) Extend the writer (recommended): a "template export" mode / configurable
  positions key.** Add a `WriteOptions`-level option (e.g. `positions_key =
  "surface_points"`, or a `template=true` flag) so the positions entry is written
  under the schema name. Minimal, keeps everything in one writer pass. Symmetrically
  teach the reader to also accept `surface_points` for positions (one extra name in
  the `npz.cpp` ~L191 check) so the file round-trips.
- **(b) Thin post-process step** that renames `xyz.npy → surface_points.npy` inside
  the zip. Zero core change but adds an out-of-band tool and a second artifact pass.
  **Rejected** as the primary — it splits the export across two programs.

**`surface_normals` ↔ the `"normal"` column.** The writer writes any attribute
column under **its own column name** (`npz.cpp` ~L278-283), and the column is named
`"normal"` (`attr::kNormal`). So today it would emit `normal.npy`, not
`surface_normals.npy`. Same fix as positions: in template mode, map the canonical
`"normal"` column to the key `surface_normals` (and have the reader's normal-name set
— currently `normal`/`normals` at `npz.cpp` ~L222 — also accept `surface_normals`).

**The small per-cloud arrays are the structural mismatch.** `bbox_*`,
`canonical_*`, `*_local`, `object_pose_*`, `point_spacing_m` are **NOT per-point**. The core model
(doc 03 §2.2 / `core/point_cloud.hpp`) holds **only** per-point columns plus a
`std::map<std::string,std::string> metadata()` — there is **no place for a small
typed `(3,)`/`(1,3)`/scalar array**. Worse, the **reader actively drops them**: it
skips any array whose `shape[0] != n` (`npz.cpp` ~L205, the
`if (a.shape.empty() || a.shape[0] != n) … continue;` guard). So a metadata array
written into the zip would be **silently discarded on read**. Three options:

| option | how | verdict |
|---|---|---|
| **metadata-string convention** | serialize each small array as a string in `metadata()` (e.g. `metadata()["bbox_min"]="x y z"`), and have the template writer emit those as separate non-`(N,*)` npy entries; reader parses them back into `metadata()`. | **Recommended.** No new core type; uses the existing `metadata()` map; the writer/reader gain a small "extra arrays" path. |
| **extra non-`(N,*)` npz entries** (no model home) | template writer takes the metadata values as a side input (struct) and writes the entries directly, bypassing the per-point path; reader exposes them via a side struct. | Good if we don't want them in `metadata()`; needs a small `TemplateMeta` struct in `io`. |
| **JSON sidecar / `__meta__` entry** | one `__meta__.npy` of UTF-8 JSON bytes (doc 01 §4.1 already lists `__meta__` for NPZ metadata). | Fine fallback; loses the *typed array* shape the schema specifies, so **secondary**. |

**Recommendation:** combine **option (b)+(a)** — add a small **`TemplateMeta`** input
struct in `io` carrying the metadata values, and a **template writer mode** that
(1) writes `surface_points` / `surface_normals` under the schema keys and (2) writes
each metadata array as its own correctly-shaped npy entry (`bbox_min (3,)`,
`canonical_center (1,3)`, `point_spacing_m ()`, …). On read, **lift the `shape[0]!=n`
guard for a known metadata allowlist** so these survive into a `TemplateMeta` /
`metadata()`. This keeps the per-point fast path untouched and confines the schema
knowledge to the template mode. The current implementation follows this path via
`TemplateMeta`, `writeTemplateNpz()`, and the NPZ reader/writer metadata allowlist.

> Note on the scalar `()` shape: `serializeNpy` (`npz.cpp` ~L118) builds the shape
> tuple from a `vector<size_t>`; an **empty** shape vector must emit `()` (numpy
> 0-d). The current code special-cases 1-element tuples (`shape.size()==1` ⇒ trailing
> comma) but does not exercise the empty case — the template writer must pass the
> right shape so `point_spacing_m` reads back as a numpy scalar.

---

## 4. Recommendations & minimal-change plan

Prioritized, tied to the existing module layout (`core`, `io`, `pipeline`, `app/cli`).

### 4.1 P0 — Native PCA normal estimation in `core` (over nanoflann)

- New `core` routine, e.g. `core/normals.hpp` →
  `estimateNormals(PointCloud&, const NormalParams&)`, that builds/reuses the
  nanoflann KD-tree `SpatialIndex` (doc 03 §3.3, already linked), runs kNN-PCA per
  point (closed-form `3×3` symmetric eigensolve, **no Eigen**), and writes a
  `"normal"` (F32, arity 3) column via `PointCloud::add`.
- `NormalParams{ int k; float radius; enum Search{Knn,Radius,Hybrid}; std::optional<Vec3> viewpoint; }`.
  When `viewpoint` is set, flip each normal so `dot(n, viewpoint−p) > 0` (native).
- Parallelize over the `common::ThreadPool` (doc 03 §7); it's embarrassingly parallel.
- **Lives in `core`** (depends only on `common`), so GUI/CLI/worker all share it.

### 4.2 P0 — `--estimate-normals` CLI flag + emit normals on export

- `app/cli` flag `--estimate-normals[=hybrid|knn|radius]`, `--normal-k`,
  `--normal-radius`, `--viewpoint x,y,z`. Runs §4.1 after crop, before write.
- Because the PLY writer already emits `nx ny nz` from a `"normal"` column
  (`ply.cpp` ~L313) and the NPZ writer already emits attribute columns, this flag
  alone gets normals into both artifacts (modulo the NPZ key rename in §4.3).

### 4.3 P1 — NPZ "template" writer mode (meters schema + per-cloud arrays)

- Add a template path to `NpzWriter` (`src/io/npz.cpp`): positions key
  `surface_points`, normal column → `surface_normals`, plus the per-cloud metadata
  arrays from a `TemplateMeta` struct (§3.2.1). Teach `NpzReader` to accept
  `surface_points`/`surface_normals` as position/normal aliases and to **keep**
  allowlisted non-`(N,*)` metadata arrays (lift the `shape[0]!=n` drop for those).
- Surface via `WriteOptions` (e.g. `template=true` + a `TemplateMeta`) and a CLI
  `--export-template`. Keep the generic NPZ path byte-for-byte unchanged.

### 4.4 P1 — Units / frame metadata convention (avoid the jaw-CAD unit bug)

- Standardize `metadata()["units"]="m"` and `metadata()["frame"]="template"` (doc 01
  §6 already flags `units`/`frame` as the agreed keys). PLY: emit as
  `comment units=m` / `comment frame=template` (the writer currently emits only a
  fixed comment — add these). NPZ: include in the `__meta__`/`TemplateMeta`. Validate
  on read; **warn loudly** if units are absent or not meters.

### 4.5 P2 — Normal-quality diagnostics & denoise hooks

- A `core` diagnostic (flip-rate, outward-violation fraction, mean plane residual —
  §1.6) printed by the CLI after estimation.
- Optional native statistical/radius outlier removal in `core` (reuses the same
  nanoflann index) so denoise-before-estimate (§1.4) can run without Python. Lower
  priority — Open3D can do this downstream.

### 4.6 Explicitly NOT in CloudCropper

Mesh reconstruction (Poisson/BPA/Alpha) and SDF/gradient-SDF generation stay
**downstream in Open3D/Python** (§2). CloudCropper does not link Open3D. The full
Hoppe MST orientation also stays downstream (§1.5) — native does viewpoint-flip only.

---

## 5. Priority summary

| P | change | module | unlocks |
|---|---|---|---|
| P0 | native PCA normal estimation over nanoflann | `core` | high-precision normals, no Python |
| P0 | `--estimate-normals` + emit on export | `app/cli` (+existing writers) | `.ply`/`.npz` carry `nx ny nz` |
| P1 | NPZ template mode (schema keys + per-cloud arrays) | `io` | schema-(B)-conformant `.npz` |
| P1 | units/frame metadata convention | `io`/`app` | kills the meters/mm ambiguity |
| P2 | normal diagnostics + native denoise | `core` | validation, denoise-before-estimate |

---

## 6. Open questions for the user

1. **Who owns mesh reconstruction + SDF — CloudCropper (C++) or the Python stack?**
   This doc assumes **downstream Python/Open3D** owns `TriangleMesh`→`GradientSDFField`
   and CloudCropper stops at the two files. Confirm. (If you want CloudCropper to also
   emit a mesh, that's a much larger scope: Open3D-in-C++ or a native Poisson/BPA.)
2. **Canonical frame & axis — derived or provided?** Are `canonical_center` /
   `canonical_axis` / the `*_local` tip & bar-axis points **computed** by CloudCropper
   (e.g. PCA/axis fit on the crop) or supplied externally and merely carried through?
   This decides whether §4.3 needs a *fitting* step or just a *pass-through* struct.
3. **Acquisition viewpoint(s)?** Is a sensor origin available per scan (enables the
   superior `orient_normals_towards_camera_location` / native viewpoint-flip), or must
   we always fall back to MST orientation downstream?
4. **Expected point counts & spacing?** Drives the in-memory budget (doc 03 §6.1),
   nanoflann tuning, and default normal `radius`/`k`. Roughly how many points per
   template and what `point_spacing_m`?
5. **Normals required or optional in `.npz`?** Schema marks `surface_normals` optional
   but recommended. If a consumer always needs them, we make `--estimate-normals`
   implied by `--export-template`.
6. **Poisson vs Ball-Pivoting as the downstream default** for these machined parts —
   watertight-but-smoothed vs exact-points-but-holey (§2)? Affects how aggressively we
   should preserve crisp edges in normal estimation (§1.3).

---

## Sources

- Open3D — Point cloud tutorial (estimate_normals, KDTreeSearchParamHybrid/KNN/Radius,
  orient_normals_*): https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
- Open3D — `open3d.geometry.PointCloud` API (normals, orientation):
  https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
- Open3D — Surface reconstruction tutorial (Alpha shapes, Ball pivoting, Poisson):
  https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html
- Open3D — `open3d.geometry.TriangleMesh` API (exact Poisson/BPA/alpha signatures &
  defaults): https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html
- Open3D issue #4983 — `orient_normals_consistent_tangent_plane()` non-termination:
  https://github.com/isl-org/Open3D/issues/4983
- Hoppe et al., *Surface Reconstruction from Unorganized Points*, 1992 (MST tangent-
  plane orientation) — basis of `orient_normals_consistent_tangent_plane`.
- Sommer et al., *Gradient-SDF: A Semi-Implicit Surface Representation for 3D
  Reconstruction*, arXiv:2111.13652: https://arxiv.org/pdf/2111.13652
- mesh→SDF kernels: `InteractiveComputerGraphics/TriangleMeshDistance`
  (https://github.com/InteractiveComputerGraphics/TriangleMeshDistance),
  `sxyu/sdf` (https://github.com/sxyu/sdf),
  `hamzamerzic/sdf-gen` (Bridson-style level set) (https://github.com/hamzamerzic/sdf-gen).
- nanoflann (already linked; header-only C++11 KD-tree):
  https://github.com/jlblancoc/nanoflann
- CloudCropper internal: `src/io/ply.cpp`, `src/io/npz.cpp`, `src/io/pcd.cpp`,
  `include/cloudcropper/core/point_cloud.hpp`, `…/core/attribute.hpp`,
  [`03-core-architecture.md`](03-core-architecture.md) §3.3 (SpatialIndex/nanoflann),
  [`01-format-io.md`](01-format-io.md) §3.3/§4.1 (NPZ mapping, metadata).
```
