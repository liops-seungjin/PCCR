# CloudCropper

Interactive **C++20** tool to load point clouds, crop them with bounding boxes,
optionally denoise and estimate normals, and export the result to multiple
formats — from an interactive viewer or a scriptable headless CLI.

## Overview

CloudCropper is a focused point-cloud pipeline:

```
load ─► crop ─► denoise ─► estimate normals ─► export
 PLY      AABB /     statistical    kNN-PCA          PLY / PCD / NPZ
 PCD      OBB boxes  outlier        (only if          (+ optional .gz,
 NPZ      (incl/     removal         missing)          field selection,
 [.gz]    excl)                                        encoding)
```

The same pipeline runs two ways:

- **Interactive** — `cloudcropper view <file>` opens a 3D viewer; draw and edit
  bounding boxes with a transform gizmo, preview which points are kept, then
  crop + export.
- **Headless** — `cloudcropper <in> -o <out> …` does the same crop/denoise/normal/
  export steps from flags, for scripting and CI.

It deliberately stops at an **oriented point cloud**. Mesh reconstruction and
SDF / gradient-SDF generation are out of scope and stay in the downstream
consumer — see [docs/design/05-sdf-pipeline.md](docs/design/05-sdf-pipeline.md).

## Features

- **Formats** — PLY, PCD, NPZ (read + write), pluggable and optional at build time.
- **Crop** — axis-aligned and oriented bounding boxes; multiple boxes combined
  with include / exclude and Union / Intersection. The cropped output is
  **recentered to the box centre** (box centre → origin); the applied translation
  is recorded in `crop_offset` metadata (original = stored + offset).
- **Denoise** — statistical outlier removal (`--denoise`) so a stray point can't
  skew the bounds or normals (matches Open3D's filter count).
- **Normals** — self-contained kNN-PCA normal estimation (`--estimate-normals`)
  when the input has none; oriented toward a viewpoint or outward.
- **gzip** — any input/output path ending in `.gz` is transparently
  (de)compressed; standard RFC-1952 gzip, readable by `gzip`/`zcat`/Python.
- **ROS2 bags** — read `sensor_msgs/PointCloud2` straight out of a rosbag2 bag,
  both SQLite (`.db3`) and **`.mcap`** (lz4/zstd chunks); self-contained CDR
  decoder, no ROS install needed. Fields (xyz, rgb, normals, intensity, …) come
  through; if normals are absent they're filled in by `--estimate-normals` like
  any other input.
- **Registration** — align the open cloud (source) onto a saved/loaded cloud
  (target) with **GICP / VGICP / ICP / point-to-plane ICP** (small_gicp),
  **KISS-Matcher** global registration (+ GICP refine), **gradient-SDF**
  (PyTorch/CUDA, FFT exhaustive-grid initialization, optional GPIS uncertainty
  channel with a confidence score), or **BUFFER-X** (ICCV'25 zero-shot learned
  global registration; vendored core + persistent Python worker — `bufferx` /
  `bufferx-gicp`). The Python-worker backends carry **zero build-time deps**.
  Backends live in `backend/registration/<package>/`; the viewer gets a
  right-side panel, the CLI a `register` command. SOTA surveys and the
  cross-validated mm-grade results are in [`docs/research/`](docs/research/).
- **Field selection** — export only the attributes you choose (`--fields …`) in
  ascii / binary / binary_compressed encodings.

## Build

CMake ≥ 3.24 + Ninja, a C++20 compiler. Three presets:

```bash
# dev — zero external dependencies (pure C++20 + std). Builds & tests anywhere.
cmake --preset dev   && cmake --build --preset dev   && ctest --preset dev

# vcpkg — manifest mode; installs real deps (vcpkg is vendored in third_party/).
cmake --preset vcpkg && cmake --build --preset vcpkg && ctest --preset vcpkg

# gui — adds the OpenGL/ImGui viewer (vcpkg 'gui' feature; needs a display).
cmake --preset gui   && cmake --build --preset gui   && ctest --preset gui
```

The binary lands at `build/<preset>/src/app/cloudcropper`. What each preset adds:

| Capability        | dev | vcpkg | gui | Provided by |
|-------------------|:---:|:-----:|:---:|-------------|
| PLY ascii/binary  |  ✓  |   ✓   |  ✓  | self-contained codec |
| PCD ascii/binary  |  ✓  |   ✓   |  ✓  | self-contained codec |
| PCD binary_compressed | — | ✓ |  ✓  | liblzf (LZF) |
| NPZ (`numpy.savez`) | — |  ✓   |  ✓  | own NPY parser + miniz (ZIP) |
| ROS2 bag (`.db3`/`.mcap`) read | — | ✓ |  ✓  | sqlite3 / mcap + own CDR decoder |
| Registration (GICP/KISS/gradient-SDF/BUFFER-X) | — | ✓ |  ✓  | Eigen + small_gicp + KISS-Matcher (FetchContent); Python worker for gradient-SDF / BUFFER-X |
| gzip `.gz` wrapper |  ✓  |   ✓   |  ✓  | zlib (system in dev, vcpkg otherwise) |
| Denoise / normals |  ✓  |   ✓   |  ✓  | core (no external dep) |
| Interactive viewer | — | —    |  ✓  | OpenGL + GLFW + ImGui + ImGuizmo |

> `cc::Result<T>` is dual-mode: a std-only stand-in under `dev`, automatically
> aliased to `tl::expected` when vcpkg provides it. Call sites are identical.

## Usage — CLI

```
cloudcropper <input> -o <output> [options]
cloudcropper view <input> [--screenshot file.png] [--frames N] [--point-size S]
cloudcropper --bag-info <bag.db3|bag.mcap|bagdir>
cloudcropper register <source> <target> [--reg-algo …] [-o aligned.ply]
cloudcropper --list-formats
```

The `<input>` is a point-cloud file (`.ply` / `.pcd` / `.npz`, optionally `.gz`)
or a **ROS2 bag** (`.db3`/`.mcap` file or a bag directory). Steps run in order:
**load → crop → denoise → normals → write**; each is optional.

| Option | Meaning |
|---|---|
| `-o <path>` | output file (required for convert/crop) |
| `--aabb x0 y0 z0 x1 y1 z1` | crop to this axis-aligned box (min then max) |
| `--denoise` | statistical outlier removal before normals |
| `--denoise-k N` | neighbours for the denoise statistic (default 16) |
| `--denoise-std R` | keep points within `mean + R·stddev` (default 2.0) |
| `--estimate-normals` | add a `normal` column if the cloud has none |
| `--normal-k N` | neighbours for normal PCA (default 30) |
| `--normal-search knn\|radius\|hybrid` | neighbour search for normals (radius ≈ 3·spacing) |
| `--normal-radius R` | support radius for radius/hybrid search |
| `--viewpoint x,y,z` | orient normals toward this point (auto from PCD VIEWPOINT; else outward) |
| `--fields a,b,c` | export only these attributes (default: all) |
| `--ascii` | ascii encoding (PLY/PCD) |
| `--compressed` | `binary_compressed` encoding (PCD, needs liblzf) |
| `--bag-topic T` | which PointCloud2 topic to read (default: the only one) |
| `--bag-merge` | concatenate all messages on the topic (default: first frame) |
| `--bag-frame N` | which message index to read when not merging (default 0) |
| `--bag-info <bag>` | list a bag's topics, types, and message counts |
| `--export-template` | write a meters `.npz` template (surface_points/normals + bbox/canonical/spacing meta); estimates normals if absent |
| `--units m` / `--tip x,y,z` / `--bar-point` / `--bar-dir` | template metadata (units + optional tip/bar-axis points) |
| `--list-formats` | print registered readers/writers |

`register` sub-command (`cloudcropper register <source> <target> ...`) — estimates
the rigid transform mapping the source onto the target, prints the 4x4 + RMSE /
inliers / convergence, and optionally writes the aligned source with `-o`:

| Option | Meaning |
|---|---|
| `--reg-algo icp\|icp-plane\|gicp\|vgicp\|kiss\|kiss-gicp\|gsdf\|gsdf-gpu\|bufferx\|bufferx-gicp` | algorithm (default `gicp`; `kiss*` = KISS-Matcher global; `gsdf` / `gsdf-gpu` = gradient-SDF on the Python worker; `bufferx` / `bufferx-gicp` = BUFFER-X zero-shot global, optionally chained into GICP) |
| `--reg-downsample S` | preprocess leaf size in meters (0 = auto from target spacing) |
| `--reg-max-corr D` | max correspondence distance (0 = auto) |
| `--reg-threads N` | worker threads (default 4) |
| `--reg-kiss-res R` | KISS-Matcher working resolution (0 = auto ~1.5x spacing, finer retry) |
| `--reg-no-refine` | skip the GICP refinement after kiss/gsdf/bufferx |
| `--reg-bufferx-voxel V` | BUFFER-X input downsample voxel in meters (0 = auto sphericity-based) |
| `--reg-uncertainty` / `--reg-no-uncertainty` | gradient-SDF: toggle the GPIS variance channel + confidence score |

Per-package defaults live in **`config/`** (`gicp.yaml`, `kiss-matcher.yaml`,
`gradient-sdf-gpu.yaml`) — flat `key: value` YAML, applied by both the CLI and
the viewer panel before explicit flags / UI edits override them. Search order:
`$CLOUDCROPPER_CONFIG_DIR`, `./config/`, then a `config/` directory found next
to (or above) the executable.

`gsdf-gpu` runs the PyTorch implementation on CUDA (cpu fallback). The
`gradient_sdf_registration` package is vendored under
`backend/registration/gradient_sdf_gpu/python/` and driven by a **persistent
worker process** (`gsdf_worker.py`): spawned lazily on the first solve, it
keeps torch/open3d imported and caches the target's SDF field, so repeated
registrations against the same target skip the Poisson meshing + field build.
Initialization uses the FFT exhaustive grid search (`init_mode="fft"`).
**Every algorithm knob is settable in `config/gradient-sdf-gpu.yaml`** — the
C++ side forwards the whole file to the worker — including
`uncertainty: true|false`: a per-voxel GPIS variance channel on the SDF field
that weights the robust loss heteroscedastically and reports a
`confidence` / `norm-residual` trust score on the result (a plausible-but-wrong
pose scores low even when rmse looks fine). Runtime-only requirement (never
build-time):

```bash
pip install -r backend/registration/gradient_sdf_gpu/python/requirements.txt
```

There is no native fallback: if the worker cannot run (no python/torch, CUDA
OOM, crash, timeout) the registration fails with the worker's error and the
tail of `worker.log`.

### Examples

```bash
# Crop a region and write a binary PLY
cloudcropper scan.ply -o region.ply --aabb 0 0 0 1 1 1

# Convert PCD → NPZ, keeping only xyz + rgb
cloudcropper cloud.pcd -o cloud.npz --fields rgb

# Clean up + add normals, write a gzip-compressed PLY (meters in, meters out)
cloudcropper raw.pcd -o template.ply.gz --denoise --estimate-normals

# Estimate normals oriented toward a sensor origin
cloudcropper scan.npz -o oriented.ply --estimate-normals --viewpoint 0,0,1.5

# Read a gzipped cloud, crop, write a gzipped PCD
cloudcropper cloud.ply.gz -o crop.pcd.gz --aabb -1 -1 -1 1 1 1

# ROS2 bag: inspect, then read a PointCloud2 topic (merging all frames) and
# ensure normals are present, exporting an .npz
cloudcropper --bag-info my_bag/
cloudcropper my_bag/ -o cloud.npz --bag-topic /lidar/points --bag-merge --estimate-normals
```

## Usage — Viewer

```bash
./scripts/cloudcropper-viewer.sh     # launch (builds the gui preset on first run)
cloudcropper view                    # open an empty window, then load a file
cloudcropper view scan.ply           # open with a file already loaded
cloudcropper view scan.ply --frames 30 --screenshot shot.png   # headless render
```

- **Loading** — launch with **no file** to get an empty window, then load a cloud
  by **dragging** a `.ply` / `.pcd` / `.npz` / `.db3` (`.gz` ok) onto it, the
  **Open...** button (native file picker), or typing a path and pressing **Load**.
  Loading again swaps the cloud without relaunching. (Drag-drop and Open... need a
  desktop session; the path box always works.)
- **ROS2 bag playback** — drop a `.db3` (or bag directory) to enter bag mode: a
  **topic dropdown** (pick among the bag's PointCloud2 topics) and a frame
  navigator — `|<` / `prev` / `next`, a frame **slider**, and **play** at an
  adjustable fps to step through frames one at a time. The camera and crop box
  stay put while stepping, so you can watch a region across frames.
- **Camera** — left-drag to orbit (horizontal = world **yaw**, vertical = pitch),
  **Alt**+left-drag to **roll**; right/middle-drag pan, scroll to zoom, `F` to fit.
  Orientation is yaw/pitch/roll angles recomposed each frame and eased toward the
  target, so horizontal drag is pure yaw with no roll wobble and no gimbal lock.
  The **up axis** (X/Y/Z, next to *Fit camera*) sets what the camera treats as
  vertical — files default to **Y-up**, ROS bags to **Z-up** (REP-103).
- **Box editing** — points **inside the selected box turn green** (so you see how
  much is selected) while the rest dim. Move the selected box **along its own
  axes** with **WASD** (box X = A/D, box Y = W/S) and **Q/E** (box Z); movement
  follows the box's rotation, not the camera. Or use the transform gizmo. The
  `kept N / total` count updates live.
- **Panel** — a modern rounded dark theme (proportional system font, soft palette,
  one indigo accent) organized by workflow into **SOURCE → VIEW → BOXES →
  CROP & EXPORT** sections with an aligned label column, a status footer, and
  downstream sections greyed out until a cloud is loaded. Holds point size, LOD budget, colour mode
  (flat / rgb / scalar / height), up-axis + Fit, the box list (a table: enable /
  select / include-vs-exclude / delete, plus Union / Intersection), transform gizmo
  (T/R/S), numeric centre & half-size, *Snap to AABB*, and the accented
  **Crop + Export** (writes via the same codecs, binary or ascii).
- **Registration panel** (right side) — load a target cloud (or *Use last
  export* after a Crop + Export), pick GICP / VGICP / ICP / Plane-ICP /
  KISS-Matcher / KISS+GICP / gradient-SDF (GPU), and *Register*: the solve runs
  on a worker thread, the aligned source overlays in orange against the cyan
  target, and the result (4x4, RMSE, inliers, confidence when the uncertainty
  channel is on) can be applied to the source or saved.
- **Crop preview** — *Crop + Export* doesn't crop immediately: it opens a preview
  of just the cropped cloud, **auto-rotating in yaw**. Drag to inspect from any
  angle; release and it eases back to the start and resumes spinning. **OK**
  commits the crop+export (recentred to the box centre), **Cancel** returns.
- **Headless** — `--frames N --screenshot file.png` renders N frames, writes a
  PNG, and exits (used for CI smoke tests).

The launcher `scripts/cloudcropper-viewer.sh` opens the empty viewer (building the
`gui` preset the first time); pass a file to it to open with that cloud loaded.

## Project layout

```
cloudcropper/
├── include/cloudcropper/{common,core,io,transport,viewer}/   # public headers
├── src/
│   ├── common/        # header-only: cc::Result, geometry (Vec3/Quat/AABB/OBB)
│   ├── core/          # PointCloud (SoA), crop engine, normals, denoise, grid index
│   ├── io/            # byte streams, format registry, PLY/PCD/NPZ codecs, ROS2 bag (CDR)
│   ├── transport/     # gzip (zlib)
│   ├── viewer/        # OpenGL + ImGui + ImGuizmo viewer (gui builds only)
│   └── app/           # CLI entry point
├── tests/             # self-contained test runner (GoogleTest planned)
├── cmake/             # CompilerWarnings, etc.
├── third_party/vcpkg/ # vendored package manager (gitignored)
└── docs/design/       # architecture + design docs (read 00 first)
```

## Tech stack

| Concern | Current | Planned |
|---|---|---|
| Language / build | C++20, CMake ≥3.24, vcpkg manifest | — |
| Error handling | `cc::Result<T>` = tl::expected (dev: std-only shim) | std::expected on C++23 |
| Math | self-contained `Vec3/Quat/AABB/OBB` (glm in viewer) | unify on glm |
| PLY / PCD / NPZ | self-contained codecs (+ liblzf, miniz) | happly fast-path |
| ROS2 bag | sqlite3 + mcap + own CDR decoder (`.db3`, `.mcap`) | — |
| Compression | zlib (gzip transport) | — |
| Normals / neighbours | kNN-PCA over a uniform grid index | nanoflann KD-tree |
| Crop index | octree over the include-box AABB (size-thresholded); brute-force otherwise | calibrate threshold (`tests/bench`) |
| Viewer | OpenGL 3.3 + GLFW + ImGui + ImGuizmo + glad | LOD, screen-rect box |
| Tests | self-contained runner + CTest | GoogleTest |

## Design docs

- **[00 — architecture overview](docs/design/00-architecture-overview.md)** — read first
- [01 — formats & data model](docs/design/01-format-io.md)
- [02 — viewer & box editing](docs/design/02-viewer-interaction.md)
- [03 — core, crop engine, transport](docs/design/03-core-architecture.md)
- [04 — build, deps, CI](docs/design/04-build-tooling.md)
- [05 — point-cloud → SDF pipeline & normals](docs/design/05-sdf-pipeline.md)

## Status & roadmap

**Done:** PLY/PCD/NPZ read+write · ROS2 bag (`.db3`) read · AABB/OBB crop (CLI +
viewer, recentered to box centre) · statistical denoise · kNN-PCA normals
(knn/radius/hybrid search, viewpoint auto, diagnostics) · **meters template
export** (`--export-template`: surface_points/normals + bbox/canonical/spacing
metadata) · units metadata · **octree crop index** · transparent gzip ·
interactive viewer. All three presets build and test green; codecs, normals,
the template `.npz` and the octree crop are cross-checked against NumPy / Open3D /
a ROS Humble bag.

Performance: crop / normals / denoise run **multi-threaded**, the viewer
**decimates** huge clouds (LOD point budget), and oversized loads **fail loud**
(`--max-points`, default ~RAM-derived) instead of OOM.

**Next:** `tests/bench` to calibrate the octree size threshold · LAS/LAZ ·
adopt GoogleTest/CLI11 · swap core geometry to glm.
