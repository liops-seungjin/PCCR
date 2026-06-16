# 03 — Core Architecture, Data Flow, Crop Engine & Transport

**Owner:** Core/Pipeline architect
**Scope:** module decomposition, end-to-end pipeline, crop engine, transport (gzip), streaming abstraction, concurrency.
**Coordinates with:** `format-io` (concrete PLY/PCD/NPZ readers/writers), `viewer-interaction` (rendering, box gizmos).
**Status:** design baseline (v1).

---

## 0. Design principles

1. **In-memory now, streaming-ready later.** Everything today is an `InMemoryPointCloud`, but the crop engine and viewer only ever see the `PointSource` interface, so a `TiledPointSource` can drop in later with zero API churn.
2. **Single binary, in-process by default.** The viewer and core share an address space. Multi-process is an *optional* deployment for an out-of-process headless worker, not the default. (Justification in §5.)
3. **One canonical data model.** `io`, `core`, `viewer`, `transport` all agree on `PointCloud` + a structure-of-arrays attribute layout. No format-specific types leak past `io`.
4. **The same pipeline runs interactive and headless.** GUI and CLI are two thin frontends over the identical `Pipeline` API; boxes come from mouse gizmos or from a JSON/CLI spec — the engine cannot tell the difference.
5. **gzip is a transport/cache concern only.** The in-memory data model is never compressed; gzip wraps the *serialized wire/cache form* and nothing else.

---

## 1. Module decomposition & dependency graph

```
                         +------------------+
                         |      app/cli     |   frontends (thin)
                         |  - gui_app       |   - argument/JSON parsing
                         |  - cli_app       |   - wires Pipeline to a frontend
                         +---------+--------+
                                   |
                  +----------------+----------------+
                  |                                 |
                  v                                 v
          +---------------+                 +----------------+
          |    viewer     |                 |   pipeline     |  orchestration
          | (rendering,   |  PointSource    | (stage driver, |
          |  box gizmos)  |<--------------- |  job/threads)  |
          +-------+-------+   read-only view +----+------+----+
                  |                               |      |
                  | uses                          |uses  | uses
                  v                               v      v
          +---------------+   +----------------+   +----------------+
          |  transport    |   |     core       |   |      io        |
          | (serialize +  |   | - data model   |   | - PLY/PCD/NPZ  |
          |  gzip codec)  |   | - PointSource  |   |   readers      |
          |               |   | - crop engine  |   | - writers      |
          |               |   | - spatial index|   | - field schema |
          +-------+-------+   +-------+--------+   +-------+--------+
                  |                   |                    |
                  +-------------------+--------------------+
                                      v
                              +---------------+
                              |   common      |  errors, Result<T>,
                              | (foundation)  |  logging, Progress,
                              +---------------+  geometry (AABB/OBB), ThreadPool
```

### Dependency rules (acyclic, lower may not depend on higher)

| Module      | Depends on                | Must NOT depend on              |
|-------------|---------------------------|---------------------------------|
| `common`    | (std only)                | everything else                 |
| `core`      | `common`                  | `io`, `viewer`, `transport`, `app` |
| `io`        | `common`, `core`          | `viewer`, `transport`, `app`    |
| `transport` | `common`, `core`          | `io`, `viewer`, `app`           |
| `viewer`    | `common`, `core`          | `io`, `transport`, `app`        |
| `pipeline`  | `common`, `core`, `io`, `transport` | `viewer`, `app`       |
| `app/cli`   | all                       | —                               |

**Key invariant:** `core` (data model + crop engine) depends on nothing but `common`. Both `io` and `transport` adapt *into* the `core` model. `viewer` consumes `core` read-only. This is what lets the engine be reused identically in GUI, CLI, and an out-of-process worker.

> Note: `pipeline` deliberately does **not** depend on `viewer`. The viewer subscribes to pipeline events/`PointSource` handles; the pipeline never calls into rendering. This keeps headless mode free of any GL/window dependency.

> Note: `common`'s `Result<T>` is `cc::Result<T> = tl::expected<T, Error>` (resolved; C++20 target, swaps to `std::expected` on C++23 — see doc 00 §4). Every stage signature `Stage(In, Progress&) -> Result<Out>` uses it.

---

## 2. End-to-end pipeline (stages + data structures)

```
 load ──► [index build?] ──► view ──► select boxes ──► crop ──► assemble ──► write
  │            │               │           │             │          │          │
LoadSpec   PointSource    (viewer reads  CropSpec   CropResult  OutputCloud OutputSpec
  └──────► PointCloud ────► PointSource)    │         (mask +      (SoA       (formats,
            (+ SpatialIndex, lazy)          │          counts)     selected    fields,
                                            │                      fields)     gzip?)
                                       BoxSet (AABB/OBB,
                                       include/exclude, bool op)
```

Each stage is a pure-ish function `Stage(In, Progress&) -> Result<Out>` so stages are independently testable and reorderable.

### 2.1 Stage I/O contracts

```cpp
// ---- load ----
struct LoadSpec {
  std::filesystem::path path;
  Format               format = Format::Auto;   // resolved by io from extension/magic
  FieldSelection       want   = FieldSelection::All; // which attrs to materialize
  bool                 buildIndex = false;       // hint; pipeline may override (§3.5)
};
// io produces a PointSource; the default impl is InMemoryPointCloud.
using LoadOut = std::shared_ptr<PointSource>;

// ---- index build (optional) ----
struct IndexSpec { IndexKind kind = IndexKind::Auto; };  // Auto | None | KdTree | Octree
// Attaches a SpatialIndex to the source (or no-op for small clouds).

// ---- select boxes (interactive gizmo OR cli/json) ----
struct Box {
  BoxKind         kind;        // AABB | OBB
  Vec3            center;
  Vec3            halfExtents; // local axes
  Quat            rotation;    // identity for AABB
  BoxRole         role;        // Include | Exclude
};
struct CropSpec {
  std::vector<Box> boxes;
  BoolOp           combine = BoolOp::Union;  // Union | Intersection (of Include boxes)
                                             // Exclude boxes always subtract last
  CoordFrame       frame   = CoordFrame::World;
};

// ---- crop ----
struct CropResult {
  std::vector<uint8_t> mask;   // 1 byte/point (cache-friendly); or roaring bitset for huge N
  std::uint64_t        inCount;
  std::uint64_t        totalCount;
  // chunk-local masks when streaming (see §6).
};

// ---- assemble ----
struct AssembleSpec {
  FieldSelection fields;       // which attributes to carry into output
  bool           compact = true;  // gather to dense arrays (vs. keep mask view)
};
using OutputCloud = PointCloud;  // dense SoA of selected fields only

// ---- write ----
struct OutputSpec {
  std::vector<WriteTarget> targets;  // one per requested format
};
struct WriteTarget {
  std::filesystem::path path;
  Format                format;       // Ply | Pcd | Npz
  FieldSelection        fields;       // may differ per target
  std::optional<Metadata> meta;       // optional crop provenance, units, transform
  bool                  gzip = false; // on-disk gzip for this artifact
};
```

### 2.2 Canonical data model (`core`)

Structure-of-Arrays. Positions are mandatory; everything else is an optional named attribute. This keeps attributes trivially aligned with points (parallel arrays, one shared index space) and makes the crop "gather" a single index permutation applied to every column.

```cpp
enum class AttrType : uint8_t { F32, F64, U8, U16, U32, I32, U64 };

struct AttributeColumn {
  std::string           name;     // "intensity", "rgb", "label", "normal_x"...
  AttrType              type;
  uint8_t               arity;    // 1 = scalar, 3 = vec3, etc.
  std::vector<std::byte> data;    // arity * typeSize * N bytes, tightly packed
};

class PointCloud {                // the in-memory realization
  std::uint64_t              n_ = 0;
  std::vector<Vec3>          xyz_;        // mandatory positions (f32 by default)
  std::vector<AttributeColumn> attrs_;    // optional, name-keyed
  std::optional<Aabb>        bounds_;     // cached
 public:
  std::uint64_t size() const;
  std::span<const Vec3> positions() const;
  const AttributeColumn* attr(std::string_view name) const;
  // gather(indices) -> new PointCloud with positions+attrs permuted (assemble stage)
};
```

`FieldSelection` is just a name set (`All`, `None`, or explicit list); `io` and `assemble` honor it so users only pay for the attributes they keep.

---

## 3. Crop engine

Lives entirely in `core`, operates on `PointSource` + `CropSpec`, returns `CropResult`. No knowledge of file formats, GL, or transport.

### 3.1 Membership primitives

```cpp
// AABB: trivial component compare in box-local frame.
bool insideAabb(const Vec3& p, const Aabb& b);
// OBB: transform point into box frame, then AABB test against halfExtents.
//   pl = rotation^{-1} * (p - center);  inside iff |pl| <= halfExtents componentwise.
bool insideObb(const Vec3& p, const Obb& b);
```

OBB membership reduces to AABB after a rigid transform, so we keep a single hot inner loop and just pre-transform per box.

### 3.2 Boolean combination (multi-box)

For each point compute per-box predicates, then fold:

```
include_hit = combine(Union|Intersection) over all Include boxes
exclude_hit = OR over all Exclude boxes
member      = include_hit AND NOT exclude_hit
```

- **Union** (default): point kept if inside *any* include box — natural for "grab these regions".
- **Intersection**: kept only if inside *all* include boxes — for narrowing.
- Exclude boxes always subtract, evaluated last, so "carve a hole" composes with either mode.

The result is the `mask` (1 byte/point). Counts are accumulated during the same pass.

### 3.3 Brute-force vs spatial index

```
                 brute force (linear scan)        spatial index (KD-tree / octree)
 build cost      none                             O(N log N), real memory
 query cost      O(N · B)  (B = #boxes)           O(log N + k) per box, k = hits
 best when       small N, many points inside box, the box is large relative to cloud
 wins when       N large AND boxes select a small fraction of the cloud
```

**Index choice:** AABB/OBB cropping is a *range* query, not nearest-neighbor.
- **Octree** is the primary index: an axis-aligned box range query prunes whole octants cheaply, and OBB queries use the OBB's enclosing AABB to gather candidates, then exact-test the survivors. Octrees also map cleanly onto the future tiled streaming source (a tile ≈ a top-level octant).
- **KD-tree (nanoflann)** is available as an alternative/secondary. nanoflann is C++11 header-only, adaptor-based (zero data copy — it indexes our `xyz_` span directly), and fast; we use it where we also need radius/NN queries (e.g. viewer picking, future denoising). For pure box range queries the octree is the better fit, so the engine defaults to octree for the crop path and exposes nanoflann via the same `SpatialIndex` interface.

```cpp
class SpatialIndex {
 public:
  virtual ~SpatialIndex() = default;
  // candidate point indices whose region overlaps the AABB (superset; engine exact-tests)
  virtual void queryAabb(const Aabb& box,
                         std::vector<uint32_t>& outCandidates) const = 0;
};
```

### 3.4 Engine algorithm

```cpp
CropResult crop(const PointSource& src, const CropSpec& spec, Progress& p) {
  const auto box = unionAabbOf(spec.boxes);          // overall region of interest
  if (useIndex(src)) {                                // §3.5 decision
    auto idx = src.index();                           // built lazily / cached
    std::vector<uint32_t> cand;
    idx->queryAabb(box, cand);                        // prune
    return testCandidates(src, cand, spec, p);        // exact per-box boolean
  }
  return scanAll(src, spec, p);                       // brute force, chunked, parallel
}
```

Both paths funnel through the same per-point predicate, so include/exclude/boolean semantics are identical regardless of index. Index only changes *which* points are visited.

### 3.5 When to build an index — `IndexPolicy` (resolved)

The decision is **not** a hard-coded constant. It is an explicit policy object,
overridable per-load (`LoadSpec.buildIndex` / `IndexSpec`) and via the CLI
(`--index=auto|never|always`):

```cpp
struct IndexPolicy {
  enum class Mode { Never, Always, Auto } mode = Mode::Auto;
  std::uint64_t auto_point_threshold = 200'000;  // PROVISIONAL — calibrated by tests/bench
};

// Auto rule (two-factor): build an octree iff
//   N > auto_point_threshold  AND  (interactive || expected_queries > 1)
bool shouldBuildIndex(const IndexPolicy& p, std::uint64_t N,
                      bool interactive, unsigned expectedQueries);
```

| Mode / situation                                         | Decision |
|----------------------------------------------------------|----------|
| `Never`                                                  | always brute-force scan |
| `Always`                                                 | always build octree |
| `Auto`, `N <= threshold`                                 | **skip** — build cost not recovered |
| `Auto`, `N > threshold`, interactive (repeated crops)    | **build octree** (amortized over many queries) |
| `Auto`, `N > threshold`, one-shot CLI, single crop       | **skip** — a single scan beats build + one query |
| `Auto`, `N > threshold`, one-shot CLI, expected queries >1| **build** |

The `200k` default is **PROVISIONAL**: a `tests/bench` sweep (N 10k→50M × box
selectivity, on target hardware) calibrates it; it is a tunable, never a magic
number baked into the engine. Index build is itself a worker-thread job with
progress; the first crop blocks on it only if needed.

### 3.6 Attribute alignment

The crop produces *indices*; `assemble.gather(indices)` then permutes positions **and every selected `AttributeColumn` by the same index list**, so attributes can never drift out of sync with points. Attributes not in `FieldSelection` are simply never gathered (zero cost).

---

## 4. Transport (gzip)

### 4.1 Where gzip is used (exactly two places)

1. **On-disk cache / intermediate artifacts** — a serialized `PointCloud` (or a cropped result) written as a single gzip-wrapped `.ccz` blob for fast reload and for the optional gzip output targets (`WriteTarget.gzip`).
2. **IPC framing** *(only in the optional multi-process deployment, §5)* — the same serialized blob streamed gzip-compressed over a pipe/socket between the headless core worker and a frontend.

gzip is **never** applied to the live in-memory model and never to GL buffers.

### 4.2 Codec choice: zlib-ng (single codec, RFC-1952 gzip)

Research finding: **miniz** is single-file and trivial to vendor and is a drop-in for zlib's *deflate/zlib-format* APIs — but its **gzip-format framing (RFC 1952) support is limited**. zlib/**zlib-ng** has first-class gzip (`deflateInit2(..., 15|16, ...)` / `gzopen`).

**Decision (resolved):** the product spec says "data transfer uses gzip," so we standardize on the **gzip container (RFC 1952)** for both cache files and IPC, with **zlib-ng** (a faster, zlib-API-compatible drop-in; plain zlib is an acceptable substitute) as the **single** compression codec behind `transport::Codec`. There is **no second codec**: miniz is *not* used here — it is scoped to the NPZ ZIP container in `io` (doc 01 §1.3/§5), a per-format container handler on the same footing as happly/liblzf. This keeps exactly one general-purpose compression library behind the seam, sidestepping miniz's gzip-framing gap.

```cpp
namespace transport {
struct Codec {  // streaming, chunk-at-a-time so it composes with PointSource
  virtual void   reset(int level) = 0;
  virtual size_t compress  (std::span<const std::byte> in, std::span<std::byte> out, bool finish) = 0;
  virtual size_t decompress(std::span<const std::byte> in, std::span<std::byte> out) = 0;
};
std::unique_ptr<Codec> makeGzipCodec();  // zlib-ng-backed; RFC-1952 framing
}
```

### 4.3 Wire / cache serialization format (length-prefixed binary)

Little-endian. The *uncompressed* logical stream is defined below; gzip wraps the whole stream (whole-file for cache; per-message for IPC).

```
File/Message = Gzip( Payload )

Payload layout:
+----------------------------------------------------------------+
| Header                                                         |
|   magic        : u32   = 0x43435A31  ("CCZ1")                  |
|   version      : u16   = 1                                     |
|   flags        : u16   (bit0 = has_meta, bit1 = streamed/chunked)|
|   point_count  : u64                                           |
|   attr_count   : u32                                           |
|   bbox         : 6 x f32  (minx,miny,minz,maxx,maxy,maxz)      |
+----------------------------------------------------------------+
| Positions block                                               |
|   pos_type     : u8    (F32|F64)                              |
|   pos_bytes    : u64                                          |
|   data         : pos_bytes  (xyz interleaved, point_count*3)  |
+----------------------------------------------------------------+
| Attribute table  (repeated attr_count times)                  |
|   name_len     : u16                                         |
|   name         : name_len bytes (utf-8)                       |
|   type         : u8    (AttrType)                            |
|   arity        : u8                                          |
|   data_bytes   : u64                                         |
|   data         : data_bytes  (SoA column, point_count*arity) |
+----------------------------------------------------------------+
| [optional] Metadata block   (present iff flags.has_meta)      |
|   meta_len     : u32                                         |
|   meta         : meta_len bytes (UTF-8 JSON: units, transform,|
|                  source path, crop boxes for provenance)      |
+----------------------------------------------------------------+
| [chunked mode] repeat Positions+Attribute blocks per chunk,   |
|   each prefixed with chunk_point_count:u64, until count=0.    |
+----------------------------------------------------------------+
```

Every variable-length field is length-prefixed (no delimiters, no escaping). Decoders can skip unknown attributes by `data_bytes`. The chunked variant carries one block group per `PointSource` chunk, enabling streaming compress/decompress without buffering the whole cloud.

### 4.4 IPC message protocol (multi-process only)

Each direction is a stream of frames over a duplex pipe/socket:

```
Frame = | type:u8 | corr_id:u32 | gzip_len:u32 | GzipPayload[gzip_len] |

type:  0x01 LOAD_REQ    (payload = LoadSpec json)
       0x02 CLOUD_PUSH  (payload = §4.3 Payload, possibly chunked)
       0x03 CROP_REQ    (payload = CropSpec json)
       0x04 CROP_RESULT (payload = mask + counts, or assembled OutputCloud)
       0x05 PROGRESS    (payload = {stage, fraction, msg})
       0x06 WRITE_REQ   (payload = OutputSpec json)
       0x07 ERROR       (payload = {code, message})
```

`corr_id` ties async replies/progress to a request. Progress frames interleave freely with data frames.

---

## 5. In-process vs multi-process — decision

**Decision: in-process (single binary) by default; multi-process as an optional headless-worker deployment.**

Rationale:
- The original move to C++ was for *speed*; the crop engine and viewer share large point buffers. In-process means the viewer renders **the same memory** the engine cropped — zero copy, zero serialization, no gzip on the hot path. Crossing a process boundary would force us to serialize+gzip the whole cloud just to display it, which is exactly the cost we are trying to avoid.
- A single binary is simpler to ship, debug, and reason about; the dependency graph (§1) already enforces clean module seams, so we get most of the isolation benefits without an IPC tax.

When multi-process *is* worth it (all optional, behind a flag):
- **Headless batch farm:** a long-running core worker process serving many crop requests, driven by a thin client — here data already lives on disk and gzip-over-pipe is acceptable.
- **Crash isolation for untrusted/huge files:** parse in a sandboxed child; stream results back via §4.4.
- **Future out-of-core node:** a `TiledPointSource` backed by a separate streaming process.

Because the engine only sees `PointSource` and the wire format is already defined (§4.3), flipping to multi-process is an integration change, not a redesign: the child speaks `CLOUD_PUSH`/`CROP_RESULT`, the parent wraps the received bytes in a `PointSource`.

---

## 6. Streaming / large-scale abstraction

The whole point of the `PointSource` seam: the crop engine and viewer are written against it today, so a tiled/out-of-core source slots in later untouched.

```cpp
struct Chunk {
  std::uint64_t      baseIndex;   // global index of first point in this chunk
  std::span<const Vec3> positions;
  // attribute columns for this chunk, name-keyed; lifetime = until next() call
  const AttributeColumnView* attr(std::string_view name) const;
  std::uint64_t      size() const;
};

class ChunkIterator {              // forward, single-pass
 public:
  virtual ~ChunkIterator() = default;
  virtual bool next(Chunk& out) = 0;   // false at end
};

class PointSource {                // the universal handle (core)
 public:
  virtual ~PointSource() = default;
  virtual std::uint64_t size() const = 0;
  virtual Aabb bounds() const = 0;
  virtual FieldSchema schema() const = 0;          // available attributes + types
  virtual std::unique_ptr<ChunkIterator> chunks(   // streaming access
            const FieldSelection& want,
            std::optional<Aabb> roi = std::nullopt  // source may pre-prune by region
          ) const = 0;
  virtual const SpatialIndex* index() const = 0;    // null if none/not built
  virtual bool randomAccess() const = 0;            // true for in-memory
};
```

Two implementations:

```cpp
class InMemoryPointCloud : public PointSource {     // today
  PointCloud cloud_;
  // chunks() yields one (or a few) big chunks over the resident arrays.
  // randomAccess()==true, supports gather() directly.
};

class TiledPointSource : public PointSource {       // tomorrow (no engine changes)
  // backed by on-disk tiles (each a §4.3 gzip blob or octree leaf);
  // chunks() streams tiles overlapping `roi`, decompressing on the fly;
  // index() returns the on-disk octree; randomAccess()==false.
};
```

**Why the crop engine doesn't change:** §3.4 already iterates via `chunks()` and accumulates a `mask`. For a tiled source it simply processes chunk masks (chunk-local, offset by `baseIndex`) and the ROI lets the source skip non-overlapping tiles entirely. The viewer likewise pulls `chunks()` (with an ROI = frustum/LOD region) for rendering, so neither the engine nor the viewer API mentions "in-memory" anywhere.

### 6.1 v1 scope & the over-limit guard (resolved)

**v1 ships the seam, not the tiling.** Only `InMemoryPointCloud` is implemented;
the `PointSource`/`ChunkIterator` interface exists so `TiledPointSource` slots in
later untouched. `TiledPointSource` is funded by a **data trigger, not a date** —
when real inputs exceed a comfortable RAM fraction (≈50–70% of target RAM, or
≈50–100M points). Until then, `load()` enforces an explicit budget and **fails
loud** rather than thrashing/OOM:

```cpp
// in the loader, before materializing an InMemoryPointCloud:
if (estimatedBytes > policy.in_memory_limit_bytes)
  return cc::err(ErrorCode::CloudTooLarge,
                 "cloud exceeds in-memory limit; tiled source not yet implemented");
```

The out-of-core octree tile format reuses the §4.3 `.ccz` chunked wire schema
(each tile = one chunked-payload blob / octree leaf), so the v2 tiled source is an
integration of already-specified pieces, not a redesign.

---

## 7. Concurrency / threading model

```
 UI thread (main)                Worker pool (common::ThreadPool)
 ----------------                --------------------------------
 - GL render loop                - load job        ─┐
 - input / box gizmos            - index build job  │ each posts
 - submits Jobs ───────────────► - crop job         │ Progress(frac,msg)
 - drains result/event queue ◄── - assemble job     │ back to a
   (lock-free SPSC/MPSC queue)   - write job        ─┘ thread-safe queue
```

- **Never block the UI thread** on load/index/crop/write. The frontend submits a `Job` to the pool and gets a `JobHandle`; results and progress arrive on a queue drained each frame (GUI) or awaited (CLI).
- **Progress:** every long stage takes a `Progress&` (from `common`) and reports a 0..1 fraction + message; the pipeline maps stage progress onto an overall bar. Jobs are cancelable via a `std::stop_token`; the engine checks it per chunk.
- **Parallel crop:** the brute-force/candidate scan is data-parallel over chunks (`parallel_for` on the pool), each thread writing a disjoint mask range — no contention, counts reduced at the end. Index build (octree) is parallelized over top-level octants.
- **Headless/CLI** uses the *same* `Job`/`Progress` machinery but a synchronous driver (submit → wait → next stage), so batch runs are deterministic and scriptable while reusing every engine/IO/transport component unchanged.

---

## 8. Open questions for the team

- **format-io:** does NPZ preserve arbitrary named arrays we should surface as attributes 1:1? Confirm dtype → `AttrType` mapping (esp. f16, bool, structured dtypes).
- **viewer-interaction:** preferred ROI handshake for LOD — does the viewer want `chunks(roi=frustum)` or a separate decimated `PointSource`?
- **Scale calibration:** the ~200k index threshold is now `IndexPolicy::auto_point_threshold` (§3.5), marked PROVISIONAL; a `tests/bench` sweep calibrates it once representative clouds exist. *(decision resolved; value pending benchmark)*
- **Cache invalidation:** key `.ccz` cache by (source path, mtime, FieldSelection, format version).

---

### Sources
- [nanoflann (header-only C++11 KD-tree)](https://github.com/jlblancoc/nanoflann)
- [PCL search structures (KdTree / nanoflann / octree)](https://deepwiki.com/PointCloudLibrary/pcl/5.3-search-structures)
- [miniz single-file zlib replacement](https://github.com/richgel999/miniz)
- [zlib FAQ (gzip format support)](https://www.zlib.net/zlib_faq.html)
- [zstr — C++ header-only zlib/gzip stream wrapper](https://github.com/mateidavid/zstr)
