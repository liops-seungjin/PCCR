# CloudCropper — Architecture Overview

> Synthesis of the design-team docs (01–04). Read this first; the per-area docs
> hold the detail.

## 1. What we're building

A **C++** application that:

1. **Loads** point clouds from **PLY / PCD / NPZ** — formats are optional and
   pluggable (build-time and run-time).
2. **Displays** them in an interactive 3D **viewer**.
3. Lets the user draw and manipulate **bounding boxes** (AABB/OBB) to select a
   sub-region.
4. **Crops** to the selected points and **exports** to multiple optional output
   formats, writing only the **optional attribute fields** the user chooses and
   the target format can carry.

C++ was chosen because Python workers were too slow on the hot path. Bulk data
transfer (cache, IPC, persisted exports) is **gzip**-compressed. Target data
scale is unknown, so everything is built on an **in-memory** implementation
today behind a **streaming-ready** abstraction for later.

## 2. Module map

```
                 +------------------+
                 |   app / cli      |   interactive GUI  +  headless batch CLI
                 +---------+--------+
                           |
                 +---------v--------+
                 |     pipeline     |   stage driver: load→index→view→crop→assemble→write
                 +--+----+-----+----+
                    |    |     |
        +-----------+    |     +-----------+
        |                |                 |
  +-----v-----+    +-----v-----+     +-----v-----+
  |    io     |    |  viewer   |     | transport |
  | PLY/PCD/  |    | GL+ImGui+ |     | gzip /    |
  | NPZ R/W   |    | ImGuizmo  |     | CCZ wire  |
  +-----+-----+    +-----+-----+     +-----+-----+
        |                |                 |
        +--------+-------+--------+--------+
                 |                |
            +----v----+      +----v----+
            |  core   |      | common  |
            | model + |----->| geom,   |
            | crop +  |      | Result, |
            | sources |      | pool,   |
            +---------+      | progress|
            +---------+      +---------+
```

Dependency rule: **acyclic**, pointing down. `core` depends only on `common`, so
the crop engine is byte-for-byte identical across GUI, CLI, and a future worker
process. `pipeline` does **not** depend on `viewer` → headless builds link no GL.

## 3. The data model (the spine)

A **columnar / struct-of-arrays (SoA)** point cloud:

- `xyz` is the only required column.
- Optional columns exist **iff the source had them**: `rgb`, `intensity`,
  `normal`, `label`, `timestamp`, plus **arbitrary named** scalar/vector fields.
- Every column is a typed buffer exposed only through `std::span` / typed views,
  so a future chunked/streaming store drops in **without changing any API**.
- Cropping produces an **index list**; `assemble.gather(indices)` permutes `xyz`
  **and every selected attribute by the same indices** — attributes can never
  drift out of alignment with points.

This single model is shared by io (read fills it / write reads it), core (crop
operates on it), and viewer (renders a decimated view of it).

## 4. Key decisions (resolved)

| Area | Decision |
|---|---|
| Language std | **C++20**, no extensions. GCC ≥11 / Clang ≥14 / MSVC VS2022. |
| Error handling | **`cc::Result<T> = tl::expected<T, Error>`** (single-header, CC0) at all I/O/parse boundaries — recoverable errors are returned, not thrown; exceptions/asserts reserved for invariant violations. The alias keeps a one-line migration to `std::expected` once the toolchain is C++23. |
| Renderer | **Custom OpenGL 3.3 core + Dear ImGui (docking) + ImGuizmo.** Not VTK/PCL/Open3D — they own the render loop and fight the custom box gizmo. GL isolated behind `IRenderDevice`. |
| Box geometry | **OBB-native** (center + quaternion + half-extents); AABB = identity rotation. Membership = transform point into box-local frame, then slab/AABB test — O(1) per point, SIMD/thread/GPU-friendly. A shared `obb_math.hpp` guarantees **viewer preview === exported crop**. |
| Box creation | 2D screen-rect → frustum **slab** (fast path) + direct 3D placement; edit via ImGuizmo + bounds handles. Multi-box boolean: Union/Intersection over *include* boxes, *exclude* boxes always subtract. |
| Viewer ↔ core | **In-process, single binary, zero-copy** (`std::span` borrow). No gzip on the live hot path. Seam kept clean (`ICloudView` / `ICropController`) so an out-of-process thin client is additive later. |
| PLY | **happly** (MIT, header-only) for symmetric ascii+binary R/W; tinyply optional read fast-path behind the same interface. |
| PCD | **Own codec** + vendored **liblzf** (BSD-2) for `binary_compressed`. PCL rejected as a dependency for a simple format. |
| NPZ/NPY | **Own ~1-file NPY parser/serializer** for full dtype↔attribute control + **miniz** (MIT) as the **ZIP-container handler** (peer of happly-for-PLY), not a second codec. |
| Spatial index | **Octree** primary for box range queries; **nanoflann KD-tree** available via the same `SpatialIndex` for NN/picking. |
| Crop strategy | **`IndexPolicy{ mode=Auto; auto_point_threshold=200k (PROVISIONAL) }`** with `mode ∈ {Never, Always, Auto}`. Auto builds an octree only when **N > threshold AND (interactive ‖ expected-queries > 1)**; one-shot CLI crop stays brute-force. Threshold calibrated by a `tests/bench` sweep; exposed as `--index=auto\|never\|always`. |
| Compression codec | **One codec** behind the `Codec` seam: **zlib-ng** (or plain zlib), gzip framing via `deflateInit2(windowBits\|16)`. miniz is *not* a second codec — it only handles the NPZ ZIP container. |
| gzip transport | gzip used in exactly two places: on-disk `.ccz` cache/output and (multi-process only) IPC framing. **Never on live memory.** |
| Build / deps | **CMake ≥3.24** modern targets + **vcpkg manifest mode** (written `find_package`-first so system/Conan still resolve). FetchContent fallback for tiny header-only libs. |
| Feature flags | `CLOUDCROPPER_BUILD_GUI`, `_WITH_PLY/_PCD/_NPZ/_PCL`, `_TRANSPORT_GZIP`, `_BUILD_TESTS`, `_ENABLE_SANITIZERS` → true headless build possible. |
| Tests | **GoogleTest** via CTest: IO round-trips, crop idempotence, KD-tree-vs-brute-force parity, gzip round-trip. |

## 5. Resolved decisions (formerly open)

These four were open in the first synthesis; all are now settled (rationale folded
into §4 and the per-area docs).

1. **Error type → `cc::Result<T> = tl::expected<T, Error>`.** io had proposed
   `std::expected` (C++23) but the build target is **C++20**. Resolution: a
   `cc::Result<T>` alias over **`tl::expected`** (single-header, CC0) at every
   I/O/parse boundary — recoverable errors returned, not thrown; exceptions/asserts
   only for invariant violations. The alias swaps to `std::expected` in one line
   when the toolchain moves to C++23.
2. **Compression codec → one codec; miniz is the NPZ ZIP-container handler.** io
   leaned on miniz; core found miniz lacks full **gzip (RFC 1952) framing**.
   Resolution: a single `Codec` = **zlib-ng** (or plain zlib) for all gzip
   transport (`.ccz`/IPC), with **miniz reframed as the NPZ ZIP-container handler**
   — a format dependency on the same footing as happly-for-PLY, *not* a second
   compression codec. This dissolves the "two compression libs" smell: behind the
   seam there is exactly one codec.
3. **Index threshold → policy object + two-factor heuristic + benchmark.** The
   bare ~200k constant is replaced by `IndexPolicy{ mode=Auto;
   auto_point_threshold=200'000 /*PROVISIONAL*/ }`, `mode ∈ {Never, Always, Auto}`.
   Auto builds an octree only when **N > threshold AND (interactive ‖ expected
   queries > 1)**. A `tests/bench` sweep (N 10k→50M × box selectivity) calibrates
   the threshold on target hardware; `--index=auto|never|always` overrides.
4. **Streaming milestone → seam in v1, `TiledPointSource` triggered by data.** v1
   ships the `PointSource`/`ChunkIterator` seam + `InMemoryPointCloud` only.
   `TiledPointSource` is triggered by data size (~RAM 50–70% or ~50–100M points),
   not a date; until then an over-limit load **fails loud** with an explicit error
   (no OOM). The out-of-core octree tile format reuses the `.ccz` chunked wire
   schema (doc 03), so v2 is integration, not redesign.

## 6. Pipeline (end to end)

```
load(file)                      io::Reader → InMemoryPointCloud (SoA, optional cols)
  → [optional] build index      core::SpatialIndex (octree) if large & selective
  → view                        viewer renders decimated display set (full-res kept in core)
  → select boxes                user draws OBB(s); include/exclude/boolean combine
  → crop                        core::CropEngine → index mask
  → assemble(fields)            gather() selected points + user-chosen attributes
  → write(format, fields, gzip) io::Writer (format-aware field checklist) [+ gzip]
```

Both an interactive GUI session and a scriptable headless CLI drive the **same**
`pipeline` stages; the GUI just adds the viewer + box-editing front end.

## 7. Concurrency

UI thread never blocks. Load / index / crop / write run on a `common::ThreadPool`
with per-chunk `Progress` reporting and `stop_token` cancellation. Results drain
via a queue (GUI) or are awaited synchronously (CLI) — same machinery, both modes.

## 8. Where to read more

- [`01-format-io.md`](01-format-io.md) — readers/writers, data model, NPY parsing, field-aware export.
- [`02-viewer-interaction.md`](02-viewer-interaction.md) — renderer, box gizmos, picking, LOD.
- [`03-core-architecture.md`](03-core-architecture.md) — modules, crop engine, CCZ wire protocol, streaming seam.
- [`04-build-tooling.md`](04-build-tooling.md) — directory tree, CMake/vcpkg skeletons, CI, tests.
