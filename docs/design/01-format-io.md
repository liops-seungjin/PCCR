# CloudCropper — Format Input/Output Layer Design

**Component:** Format I/O (parsing + serialization)
**Author:** Format/IO architect
**Date:** 2026-06-04
**Status:** Draft for lead review

---

## 0. Scope & Goals

This document covers ONLY the format input/output layer of CloudCropper:

1. Input parsing for **PLY** (ascii + binary), **PCD** (ascii + binary + binary_compressed), and **NPZ/NPY**.
2. A **unified in-memory point-cloud data model** with required `xyz` and *optional* attributes.
3. A **pluggable Reader/Writer interface** (abstract base + per-format impls + registry) so formats are optional and additive.
4. **Output** to PLY/PCD/NPZ, with per-format field capability matrix and a user field-selection mechanism.
5. How **gzip** fits — transport vs persisted files.

Non-goals: the viewer, the crop/box geometry, the actual gzip transport channel between components (we only define where serialization hands off to it). Design starts **in-memory** but the interface is shaped so a streaming/chunked backend can be slotted in later without touching the core or callers.

---

## 1. Library Survey & Recommendations

All candidates were evaluated for: license (must be permissive — MIT/BSD/PD, **no GPL/LGPL**), header-only vs compiled, maintenance, and C++ standard. CloudCropper targets **C++20** (see doc 04); error handling at this layer uses `cc::Result<T>` (= `tl::expected<T, Error>`, see §4).

### 1.1 PLY

| Library | License | Form | Read | Write | Notes |
|---|---|---|---|---|---|
| **happly** (`nmwsharp/happly`) | MIT | single header `happly.h` | ascii + binary (LE/BE) | ascii + binary (LE/BE) | Generic element/property API; arbitrary named properties map cleanly to our optional-attribute model. Cannot write `inf`/`nan` in ascii; list counts forced to `uchar`. Little-endian assumed for host. |
| tinyply (`ddiakopoulos/tinyply`) | Public domain / BSD-2 | single header (C++17) | ascii + binary, **reads** big-endian | ascii + binary, **cannot write** big-endian | 3.0 has a fast-path LE-binary parser; very fast. API is request-by-property, slightly more code to wire arbitrary fields. |
| miniply (`vilya/miniply`) | MIT | header + cpp | binary fast | **read-only** | Fastest reader but no write — disqualifies it as our single PLY lib. |

**Recommendation: happly.** Rationale: single header (MIT), symmetric read+write for both ascii and binary, and its element/property model is the cleanest fit for "copy whatever named scalar fields the file has into our generic attribute table." Performance is adequate for in-memory scale; if profiling later shows PLY read throughput is a bottleneck we can add tinyply purely as a *read* fast-path behind the same `IReader` interface (the registry makes that swap invisible to callers). We explicitly reject miniply as the primary because we need write.

### 1.2 PCD

Two realistic paths:

- **PCL** (`PointCloudLibrary/pcl`) — full-featured but heavyweight: pulls Boost, Eigen, FLANN, VTK transitively; **BSD** licensed (fine), but it is a large build dependency for what is fundamentally a text-header + (LZF) binary-body format. Adopting PCL just for PCD I/O is disproportionate.
- **Self-written PCD reader/writer** — the PCD format is small and fully documented. Header is line-oriented ASCII (`VERSION FIELDS SIZE TYPE COUNT WIDTH HEIGHT VIEWPOINT POINTS DATA`). Bodies: `ascii`, `binary` (raw SoA-per-point memory image), and `binary_compressed` (two leading `uint32` = compressed size + uncompressed size, then **LZF** (Marc Lehmann's liblzf) over a **structure-of-arrays reordered** buffer).

**Recommendation: write our own PCD reader/writer** using a small vendored **LZF** implementation (liblzf, BSD/GPL-dual — we take the BSD terms; ~2 small files) for the `binary_compressed` path. Rationale: avoids dragging the entire PCL/Boost/VTK stack into CloudCropper, keeps the dependency surface tiny, and the format is simple and stable. Risk: we must correctly implement (a) the TYPE/SIZE → C++ type mapping, (b) the `COUNT > 1` multi-element fields, (c) the **SoA reorder** for `binary_compressed` (data is transposed to `XX..XX YY..YY ...` field-blocks before LZF). These are well-specified; we cover them in §3.2.

### 1.3 NPZ / NPY (the tricky one in C++)

An `.npz` is **just a ZIP archive** whose entries are `name.npy` files. `np.savez` stores entries **STORED** (uncompressed); `np.savez_compressed` stores them **DEFLATE**-compressed. So NPZ parsing decomposes into two independent problems: **(a) a ZIP container reader/writer** and **(b) an NPY blob parser**.

**NPY format** (must parse by hand — there is no canonical lightweight "just NPY" std lib):
- Bytes `0..5`: magic `\x93NUMPY`.
- Byte `6`: major version, byte `7`: minor.
- Header length field: **v1.0** → 2-byte LE `uint16`; **v2.0/3.0** → 4-byte LE `uint32`. (v3.0 = UTF-8 header.)
- Header is an ASCII/UTF-8 Python-dict literal with keys `descr` (dtype string e.g. `<f4`, `<f8`, `|u1`, `<i4`), `fortran_order` (bool), `shape` (tuple). Padded with spaces, `\n`-terminated, total prefix length a multiple of 64.
- Then raw contiguous data: `prod(shape) * itemsize` bytes, C- or Fortran-order per `fortran_order`. (Object arrays = pickled → **we reject** these; not relevant to point data.)

Candidate libraries:

| Library | License | Form | NPY | NPZ | zlib dep | Notes |
|---|---|---|---|---|---|---|
| **cnpy** (`rogersce/cnpy`) | MIT | compiled lib | r/w | r/w | requires zlib (`-lz`) | Widely used; less actively maintained; not header-only. |
| **TinyNPY** (`cdcseacave/TinyNPY`) | MIT | `.h`+`.cpp` | r/w | r (read-focused) | zlib only for *compressed* npz | No deps for npy + uncompressed npz; minimal. |
| cnpy++ (`mreininghaus/cnpy++`) | (check) BSD-ish | C++17 lib | r/w | r/w | zlib | Modern C++17 rewrite, iterator API; heavier. |

**Recommendation: implement our own thin NPY parser/serializer (≈1 file) + use `miniz` purely as the ZIP-container handler.** Reasoning:
- The NPY header parse/emit is ~100 lines and we want exact control over the **dtype ↔ attribute** mapping (endianness, `<f4` vs `<f8`, `|u1` for rgb/label) and over **shape conventions** (we standardize on `(N,)` for scalars and `(N,3)` for xyz/rgb/normals — see §3.3). Pulling in a general npy lib gives us less control over precisely this mapping and still doesn't solve the ZIP side.
- **miniz** (`richgel999/miniz`, **MIT**, single C source `miniz.c`/`miniz.h`, v3.x) gives us a full ZIP archive read/write API (`mz_zip_reader_*` / `mz_zip_writer_*`) in one drop-in file — hand-rolling the ZIP central directory / local headers / CRC32 is the bug-prone part, and miniz owns exactly that niche. **Decision (resolved):** miniz is scoped to NPZ's ZIP container *only* — it is a format-handler peer of happly-for-PLY, **not** the project's general compression codec. The gzip transport/`.gz` path (§5) does **not** go through miniz; it uses **zlib-ng** (the single `Codec`, see doc 03 §4.2), which has first-class RFC-1952 gzip framing that miniz lacks.

If we wanted to avoid hand-writing the NPY parser, **TinyNPY** is the fallback (MIT, minimal, no dep for the common uncompressed case), but it is read-leaning and we need symmetric write for the NPZ exporter. Net: **own NPY codec + miniz** is the opinionated choice; TinyNPY is the documented escape hatch.

### 1.4 Dependency summary (final)

| Concern | Choice | License | Form |
|---|---|---|---|
| PLY r/w | **happly** | MIT | header-only |
| PCD r/w | **own codec** + vendored **liblzf** | (ours) + BSD | source |
| NPY codec | **own codec** | (ours) | source |
| ZIP container (npz) | **miniz** (NPZ-only; not the general codec) | MIT | single source |
| gzip transport / `.gz` codec | **zlib-ng** (or plain zlib) | zlib license | source (see doc 03 §4.2) |
| (optional PLY read fast-path) | tinyply | PD/BSD-2 | header-only |

Every dependency is permissive (MIT / BSD / public-domain). **No (L)GPL, no PCL/Boost/VTK, no Eigen** is forced on the core by this layer.

---

## 2. Unified Internal Point-Cloud Data Model

Design constraints:
- `xyz` is **required**; everything else is **present only if the source had it**.
- Must hold standard semantic attributes (rgb, intensity, normals, label/semantic id, timestamp) **and** arbitrary named scalar/vector fields.
- Must be backend-agnostic so a future streaming/chunked store replaces the in-memory store without changing the model's public shape.

### 2.1 Attribute-table model

We use a **typed, columnar (SoA), name-keyed attribute table**, not a fat fixed struct. This matches how all three formats actually store data (PLY properties, PCD FIELDS, NPZ named arrays are all column-oriented) and makes "optional" natural — a field exists iff there is a column for it.

```cpp
namespace cc::io {

enum class Scalar { U8, I8, U16, I16, U32, I32, F32, F64 };

// One column = one named attribute, N rows, `components` wide (1 = scalar, 3 = vec3...).
class AttributeColumn {
public:
    Scalar     dtype()      const;
    std::size_t components() const;   // 1 for intensity/label, 3 for xyz/rgb/normal
    std::size_t size()      const;    // number of points (rows)

    // Raw contiguous bytes: row-major, components interleaved per row.
    std::span<const std::byte> bytes() const;
    std::span<std::byte>       bytes();

    // Typed views (assert dtype/components match)
    template <class T> std::span<const T> as() const;
};

// Canonical names for well-known attributes (stringly-typed under the hood so
// arbitrary fields use the same path). Keeps semantic fields discoverable.
namespace attr {
    inline constexpr std::string_view kXYZ       = "xyz";        // F32/F64, 3 comp (REQUIRED)
    inline constexpr std::string_view kRGB       = "rgb";        // U8,      3 comp
    inline constexpr std::string_view kIntensity = "intensity";  // F32/U16, 1 comp
    inline constexpr std::string_view kNormal    = "normal";     // F32,     3 comp
    inline constexpr std::string_view kLabel     = "label";      // U32/I32, 1 comp
    inline constexpr std::string_view kTimestamp = "timestamp";  // F64,     1 comp
}

class PointCloud {
public:
    std::size_t size() const;                       // # points; 0 if empty

    bool has(std::string_view name) const;
    const AttributeColumn* find(std::string_view name) const; // nullptr if absent
    AttributeColumn&       emplace(std::string_view name,
                                   Scalar dtype, std::size_t components);

    // Required accessor; throws/returns error if xyz missing.
    const AttributeColumn& xyz() const;

    std::vector<std::string> attribute_names() const; // for export field selection UI

    // Free-form provenance/metadata carried through (e.g. PCD VIEWPOINT,
    // source format, units). Opaque key->value; writers pick what they can emit.
    std::map<std::string, std::string>& metadata();

private:
    std::size_t npoints_ = 0;
    std::map<std::string, AttributeColumn, std::less<>> columns_; // heterogeneous
    std::map<std::string, std::string> metadata_;
};

} // namespace cc::io
```

Key points:
- **Optionality is structural**: `has("rgb")` / `find("normal")` — no sentinel values, no booleans-per-field. A reader only creates columns that existed in the source.
- **Arbitrary fields** (`curvature`, `ring`, `gps_time`, a 308-wide VFH vector, …) use the exact same `emplace(name, dtype, components)` path as the canonical ones. The canonical names in `attr::` are just well-known keys.
- **Columnar/SoA** means the crop operation is a row-gather across columns, and each writer pulls exactly the columns/dtypes it supports.
- **Backend seam**: `AttributeColumn` exposes data only via `std::span` + typed views, not via an exposed `std::vector`. Today it wraps an owned `std::vector<std::byte>`; later a chunked/mmap/streaming column can implement the same surface. Callers and writers never see the storage.
- `metadata()` carries non-point provenance (PCD `VIEWPOINT`, original units, source path, source format) so it can round-trip where a target format supports it.

---

## 3. Per-Format Mapping Notes

### 3.1 PLY ↔ model
- PLY `vertex` element properties map 1:1 to columns. `x,y,z` → `xyz`; `red,green,blue[,alpha]` (`uchar`) → `rgb`; `nx,ny,nz` → `normal`; `intensity`/`scalar_*` → scalar columns of matching dtype.
- Unknown vertex properties become arbitrary scalar columns keyed by their PLY name (lossless round-trip).
- We support point clouds (vertex element); face/edge elements are read into metadata-only or ignored (CloudCropper is point-centric) — documented limitation.

### 3.2 PCD ↔ model
- Parse header lines into per-field `(name, SIZE, TYPE, COUNT)`. `TYPE∈{I,U,F}` × `SIZE∈{1,2,4,8}` → our `Scalar`.
- `COUNT==1` → scalar column; `COUNT==k>1` (e.g. descriptors) → `components=k` vector column named after the field.
- **rgb packing**: PCD typically stores `rgb`/`rgba` as a single `F`/`U` 4-byte field whose bits pack `0x00RRGGBB` / `0xAARRGGBB`. The PCD reader unpacks this into a `U8 x3` (or x4) `rgb` column; the writer re-packs on export. This bit-cast convention is handled in the PCD codec, not leaked into the model.
- `binary`: raw per-point SoA image → direct column fill.
- `binary_compressed`: read `uint32 comp_size`, `uint32 uncomp_size`; LZF-decompress; the decompressed buffer is **field-major (SoA)** — i.e. all `x`, then all `y`, … — so we de-transpose into our columns. Writer reverses: build field-major buffer, LZF-compress, prepend the two sizes. (liblzf vendored for this.)
- `VIEWPOINT` → `metadata()["viewpoint"]`.

### 3.3 NPY/NPZ ↔ model
- Each named array in the `.npz` is one logical attribute. Convention:
  - `xyz` → `(N,3)` `<f4`/`<f8`; `rgb` → `(N,3)` `|u1`; `normal` → `(N,3)` `<f4`; scalar fields (`intensity`, `label`, `timestamp`, custom) → `(N,)`.
  - We also accept a single packed `(N,M)` array plus a sidecar `fields.json`/`__fields__` naming convention as an alternative ingest path (documented).
- `descr` dtype string → `Scalar` (`<f4`→F32, `<f8`→F64, `|u1`→U8, `<i4`→I32, `<u4`→U32, `<u2`→U16, …). We **require little-endian** and reject big-endian (`>`) and object (`O`) dtypes with a clear error. `fortran_order` must be handled: for `(N,k)` we want C-order; if `fortran_order==True` we transpose on read.
- On write we emit v1.0 headers (or v2.0 automatically if header > 65535 bytes), 64-byte aligned, always C-order LE.
- ZIP side via miniz: read → enumerate entries, inflate each, NPY-parse; write → NPY-serialize each column, add as a ZIP entry. We default exports to **STORED** entries but offer **DEFLATE** (= `savez_compressed`) as an option.

---

## 4. Pluggable Reader/Writer Interface + Registry

Three pieces: an abstract `IReader`/`IWriter`, per-format implementations, and a `FormatRegistry`. Adding a format = implement two interfaces + one registration line; **the core, the viewer, and the crop logic never change**.

```cpp
namespace cc::io {

// Project-wide Result/Error (defined in `common`) keeps the interface
// exception-policy-neutral. Resolved: C++20 target => tl::expected backing.
struct Error { std::string message; int code = 0; };
template <class T> using Result = tl::expected<T, Error>;   // cc::Result; swaps to std::expected on C++23

// ---- abstract source/sink (the streaming seam) ----------------------------
// Today: file-backed or memory-backed. Later: a gzip-transport stream can
// implement these without touching readers/writers.
struct IByteSource {
    virtual ~IByteSource() = default;
    virtual std::size_t read(std::span<std::byte> dst) = 0;   // returns bytes read
    virtual bool        eof()  const = 0;
};
struct IByteSink {
    virtual ~IByteSink() = default;
    virtual void write(std::span<const std::byte> src) = 0;
    virtual void flush() = 0;
};

// ---- per-format capability descriptor -------------------------------------
struct FormatInfo {
    std::string id;                          // "ply", "pcd", "npz"
    std::vector<std::string> extensions;     // {".ply"}, {".pcd"}, {".npz",".npy"}
    bool can_read  = false;
    bool can_write = false;
};

// ---- options ---------------------------------------------------------------
struct ReadOptions {
    // future: lazy/streaming, subset-of-attributes, max points, etc.
};
struct WriteOptions {
    // User-driven field selection for export (see §4.2). Empty => "all writable".
    std::vector<std::string> fields;         // attribute names to emit, in order
    enum class Encoding { Auto, Ascii, Binary, BinaryCompressed } encoding = Encoding::Auto;
    bool gzip = false;                       // wrap persisted output in gzip (§5)
};

// ---- the interfaces --------------------------------------------------------
class IReader {
public:
    virtual ~IReader() = default;
    virtual FormatInfo info() const = 0;
    // Sniff first bytes / extension to decide if this reader handles the input.
    virtual bool can_handle(std::string_view ext,
                            std::span<const std::byte> magic) const = 0;
    virtual Result<PointCloud> read(IByteSource& src,
                                    const ReadOptions& opt) const = 0;
};

class IWriter {
public:
    virtual ~IWriter() = default;
    virtual FormatInfo info() const = 0;
    // Which attribute names THIS format can actually emit for `pc`
    // (drives the export field-selection UI). See §4.1 capability matrix.
    virtual std::vector<std::string> writable_fields(const PointCloud& pc) const = 0;
    virtual Result<void> write(const PointCloud& pc,
                               IByteSink& sink,
                               const WriteOptions& opt) const = 0;
};

// ---- registry --------------------------------------------------------------
class FormatRegistry {
public:
    void register_reader(std::shared_ptr<IReader> r);
    void register_writer(std::shared_ptr<IWriter> w);

    // Selection by explicit id, by extension, or by content sniffing.
    std::shared_ptr<IReader> reader_for(std::string_view ext,
                                        std::span<const std::byte> magic) const;
    std::shared_ptr<IWriter> writer_for_id(std::string_view id) const;

    std::vector<FormatInfo> available() const;   // for menus / CLI help

    static FormatRegistry& instance();           // process-wide default registry
};

// Each format self-registers (optional formats => optional TUs / link units):
//   static const bool ply_reg = (FormatRegistry::instance().register_reader(
//                                    std::make_shared<PlyReader>()),
//                                FormatRegistry::instance().register_writer(
//                                    std::make_shared<PlyWriter>()), true);

} // namespace cc::io
```

Design notes:
- **Optional formats**: each format lives in its own translation unit / static lib (`cc_io_ply`, `cc_io_pcd`, `cc_io_npz`). A build that omits `cc_io_npz` simply never registers NPZ; nothing else changes. This is how "formats are optional/pluggable" is realized concretely.
- **Selection** is by content magic first (`\x93NUMPY`, `ply\n`, `# .PCD`), extension second — robust to mis-named files.
- **Streaming seam**: readers/writers consume `IByteSource`/`IByteSink`, *not* file paths. The in-memory `PointCloud` is still materialized fully today, but the *transport* is already abstracted, and `ReadOptions` has room for a future chunked/lazy mode without an interface break.
- Error handling uses `cc::Result<T>` (= `tl::expected<T, Error>`) on the C++20 target, so error handling is uniform and exception policy stays the caller's choice; the alias migrates to `std::expected` in one line on C++23.

### 4.1 Field capability matrix (output)

`IWriter::writable_fields()` is the authority; conceptually:

| Attribute | PLY | PCD | NPZ |
|---|---|---|---|
| xyz (required) | ✓ `x y z` | ✓ `x y z` | ✓ `xyz (N,3)` |
| rgb | ✓ `red green blue [alpha]` | ✓ packed `rgb`/`rgba` | ✓ `rgb (N,3) u1` |
| intensity | ✓ scalar prop | ✓ field | ✓ `(N,)` |
| normals | ✓ `nx ny nz` | ✓ `normal_x/y/z` | ✓ `normal (N,3)` |
| label / semantic id | ✓ scalar prop | ✓ field | ✓ `(N,)` |
| timestamp | ✓ scalar prop | ✓ field (`F8`) | ✓ `(N,)` |
| arbitrary scalar | ✓ named prop | ✓ named field | ✓ named `(N,)` |
| arbitrary vector (k>1) | ⚠ as k scalar props | ✓ `COUNT k` field | ✓ named `(N,k)` |
| metadata (viewpoint, units) | ⚠ comments only | ✓ `VIEWPOINT` etc. | ⚠ sidecar `__meta__` |

NPZ is the most faithful (named typed arrays, lossless). PCD handles vector fields natively via `COUNT`. PLY round-trips named scalars well but must split vectors into components and can only stash metadata in comments.

### 4.2 How the user selects fields per export

1. UI/CLI calls `registry.writer_for_id(targetFormat)`.
2. UI calls `writer->writable_fields(pc)` → the intersection of "attributes present in `pc`" and "attributes this format can emit" → presented as a checklist.
3. User's selection (subset, ordered) is passed as `WriteOptions::fields`. `xyz` is always forced/validated present. Empty list ⇒ "all writable fields".
4. `WriteOptions::encoding` chooses ascii/binary/binary_compressed where the format supports it (`Auto` picks the format's sensible default: PLY→binary LE, PCD→binary_compressed, NPZ→STORED).

This keeps field selection format-aware (you can't tick a field PCD can't carry) and entirely data-driven.

---

## 5. Where gzip Fits

Two distinct uses, deliberately separated:

1. **In-process transport** (per the product brief, components exchange gzip-compressed data). This is *not* a file format concern: the serializer writes a format byte-stream into an `IByteSink`, and the transport layer wraps that sink in a gzip stream. Because our writers target `IByteSink` (not paths), a `GzipSink` decorator (backed by the **zlib-ng** `Codec` from doc 03 §4.2, RFC-1952 gzip framing) composes transparently — the PLY/PCD/NPZ writer is unaware. Same for reads via a `GzipSource` decorator. **Recommended for transport**: serialize to the most compact native form first (PCD `binary_compressed` or NPZ-deflate), then gzip — avoid double-paying compression by preferring gzip over an *uncompressed* binary serialization for transport.

2. **Persisted `.gz` files** (e.g. `cloud.ply.gz`). Implemented by the same `GzipSink`/`GzipSource` decorators driven by `WriteOptions::gzip` / detected on the read side by the `.gz` extension (peel it, then dispatch on the inner extension/magic). This is generic and works for every registered format for free.

Important interaction with NPZ: an `.npz` is **already a ZIP (DEFLATE) container**. Gzipping an `.npz` is redundant and wasteful — the registry/writer should **refuse or no-op `gzip=true` for NPZ** (and the NPZ writer instead exposes STORED-vs-DEFLATE-entry as its compression knob). Likewise PCD `binary_compressed` already LZF-compresses the body; gzip-on-top yields little. The rule of thumb encoded in defaults: **let the format's native compression do the work; reserve gzip for transport and for formats with no built-in compression (ascii/uncompressed binary PLY/PCD).**

Compression responsibilities are split by role (resolved decision): **miniz** owns the **NPZ ZIP container** (DEFLATE for NPZ entries — its native niche), while the **zlib-ng `Codec`** owns the **gzip transport/`.gz`** path (RFC-1952 framing miniz lacks). Behind the `transport::Codec` seam there is exactly **one** general compression codec (zlib-ng); miniz is a format-handler peer of happly/liblzf, not a competing codec — so "two compression libs" is not a smell, it's one codec plus per-format container handlers.

---

## 6. Open Questions / Hand-offs

- **Error/exception policy** — *resolved*: `cc::Result<T>` (= `tl::expected<T, Error>`) at the I/O boundary on the C++20 target; deeper exceptions reserved for invariant violations. See doc 00 §4–§5.
- **Compression role split** — *resolved*: miniz = NPZ ZIP container only; zlib-ng = gzip transport/`.gz` codec. See doc 03 §4.2.
- **Units & coordinate frame** are carried as opaque `metadata` strings; the geometry/crop component should agree on canonical keys (`units`, `frame`).
- **liblzf vendoring**: confirm we take its BSD terms (it is BSD-2-or-GPL dual; we choose BSD-2).
- **Streaming**: `ReadOptions`/`IByteSource` reserve room for chunked reads; the actual chunked store is future work (v2 `TiledPointSource`, see doc 03 §6) and out of this layer's first cut.

---

## Sources

- happly — https://github.com/nmwsharp/happly (MIT, header-only)
- tinyply — https://github.com/ddiakopoulos/tinyply (PD / BSD-2)
- miniply — https://github.com/vilya/miniply (MIT, read-only)
- cnpy — https://github.com/rogersce/cnpy (MIT, needs zlib)
- TinyNPY — https://github.com/cdcseacave/TinyNPY (MIT)
- cnpy++ — https://www.sciencedirect.com/science/article/pii/S2352711023000201
- NumPy NPY format spec — https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
- PCD file format — https://pointclouds.org/documentation/tutorials/pcd_file_format.html
- miniz — https://github.com/richgel999/miniz (MIT, single source; ZIP + DEFLATE)
