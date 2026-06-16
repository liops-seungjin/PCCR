#include "cloudcropper/io/npz.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <miniz.h>

namespace cc::io {
namespace {

// ---- dtype mapping --------------------------------------------------------
const char* attrToDescr(AttrType t) {
    switch (t) {
        case AttrType::F32: return "<f4";
        case AttrType::F64: return "<f8";
        case AttrType::U8:  return "|u1";
        case AttrType::I8:  return "|i1";
        case AttrType::U16: return "<u2";
        case AttrType::I16: return "<i2";
        case AttrType::U32: return "<u4";
        case AttrType::I32: return "<i4";
        case AttrType::U64: return "<u8";
        case AttrType::I64: return "<i8";
    }
    return "<f4";
}

std::optional<AttrType> descrToAttr(const std::string& d) {
    if (d.empty()) return std::nullopt;
    std::size_t k = (d[0] == '<' || d[0] == '>' || d[0] == '=' || d[0] == '|') ? 1 : 0;
    if (d[0] == '>' && d.size() > k + 1 && d[k + 1] != '1')
        return std::nullopt;  // reject big-endian multibyte
    if (k >= d.size()) return std::nullopt;
    const char kind = d[k];
    const int  sz   = std::stoi(d.substr(k + 1));
    if (kind == 'f') return sz == 8 ? AttrType::F64 : AttrType::F32;
    if (kind == 'u' || kind == 'b')
        return sz == 1 ? AttrType::U8 : sz == 2 ? AttrType::U16 : sz == 8 ? AttrType::U64 : AttrType::U32;
    if (kind == 'i')
        return sz == 1 ? AttrType::I8 : sz == 2 ? AttrType::I16 : sz == 8 ? AttrType::I64 : AttrType::I32;
    return std::nullopt;
}

double loadDouble(const std::byte* p, AttrType t) {
    switch (t) {
        case AttrType::I8:  { std::int8_t v;   std::memcpy(&v, p, 1); return v; }
        case AttrType::U8:  { std::uint8_t v;  std::memcpy(&v, p, 1); return v; }
        case AttrType::I16: { std::int16_t v;  std::memcpy(&v, p, 2); return v; }
        case AttrType::U16: { std::uint16_t v; std::memcpy(&v, p, 2); return v; }
        case AttrType::I32: { std::int32_t v;  std::memcpy(&v, p, 4); return v; }
        case AttrType::U32: { std::uint32_t v; std::memcpy(&v, p, 4); return v; }
        case AttrType::I64: { std::int64_t v;  std::memcpy(&v, p, 8); return static_cast<double>(v); }
        case AttrType::U64: { std::uint64_t v; std::memcpy(&v, p, 8); return static_cast<double>(v); }
        case AttrType::F32: { float v;         std::memcpy(&v, p, 4); return v; }
        case AttrType::F64: { double v;        std::memcpy(&v, p, 8); return v; }
    }
    return 0.0;
}

// ---- NPY ------------------------------------------------------------------
struct NpyArray {
    AttrType                 type = AttrType::F32;
    std::vector<std::size_t> shape;
    bool                     fortran = false;
    std::vector<std::byte>   data;
};

std::optional<NpyArray> parseNpy(const std::byte* buf, std::size_t size) {
    if (size < 10 || std::memcmp(buf, "\x93NUMPY", 6) != 0) return std::nullopt;
    const std::uint8_t major = static_cast<std::uint8_t>(buf[6]);
    std::size_t        hstart, hlen;
    if (major == 1) {
        std::uint16_t v;
        std::memcpy(&v, buf + 8, 2);
        hlen   = v;
        hstart = 10;
    } else {
        std::uint32_t v;
        std::memcpy(&v, buf + 8, 4);
        hlen   = v;
        hstart = 12;
    }
    if (hstart + hlen > size) return std::nullopt;
    const std::string dict(reinterpret_cast<const char*>(buf + hstart), hlen);

    NpyArray a;
    // descr
    auto dp = dict.find("'descr'");
    if (dp == std::string::npos) return std::nullopt;
    auto q1 = dict.find('\'', dict.find(':', dp) + 1);
    auto q2 = dict.find('\'', q1 + 1);
    auto at = descrToAttr(dict.substr(q1 + 1, q2 - q1 - 1));
    if (!at) return std::nullopt;
    a.type = *at;
    // fortran_order
    auto fp  = dict.find("'fortran_order'");
    a.fortran = fp != std::string::npos && dict.find("True", fp) < dict.find("'shape'", fp);
    // shape
    auto sp = dict.find("'shape'");
    auto o  = dict.find('(', sp);
    auto cl = dict.find(')', o);
    std::string inside = dict.substr(o + 1, cl - o - 1);
    std::string num;
    for (char ch : inside) {
        if (ch >= '0' && ch <= '9') num += ch;
        else if (!num.empty()) { a.shape.push_back(std::stoull(num)); num.clear(); }
    }
    if (!num.empty()) a.shape.push_back(std::stoull(num));

    const std::size_t dataStart = hstart + hlen;
    a.data.assign(buf + dataStart, buf + size);
    return a;
}

std::vector<std::byte> serializeNpy(const char* descr, const std::vector<std::size_t>& shape,
                                    const std::byte* data, std::size_t dataLen) {
    std::string shapeStr = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        shapeStr += std::to_string(shape[i]);
        if (i + 1 < shape.size() || shape.size() == 1) shapeStr += ",";
        if (i + 1 < shape.size()) shapeStr += " ";
    }
    shapeStr += ")";

    std::string dict = "{'descr': '" + std::string(descr) +
                       "', 'fortran_order': False, 'shape': " + shapeStr + ", }";
    const std::size_t base = 10 + dict.size() + 1;
    const std::size_t pad  = (64 - (base % 64)) % 64;
    dict.append(pad, ' ');
    dict += '\n';

    std::vector<std::byte> out;
    const char             magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
    out.insert(out.end(), reinterpret_cast<const std::byte*>(magic),
               reinterpret_cast<const std::byte*>(magic) + 8);
    std::uint16_t hlen = static_cast<std::uint16_t>(dict.size());
    out.insert(out.end(), reinterpret_cast<std::byte*>(&hlen), reinterpret_cast<std::byte*>(&hlen) + 2);
    out.insert(out.end(), reinterpret_cast<const std::byte*>(dict.data()),
               reinterpret_cast<const std::byte*>(dict.data()) + dict.size());
    out.insert(out.end(), data, data + dataLen);
    return out;
}

}  // namespace

// ===========================================================================
// Reader
// ===========================================================================
FormatInfo NpzReader::info() const { return {"npz", {".npz"}, true, false}; }

bool NpzReader::can_handle(std::string_view ext, std::span<const std::byte> magic) const {
    // ZIP local-file-header magic "PK\x03\x04"; but .npy (PK absent) also lands
    // here by extension. Dispatch by extension to avoid grabbing arbitrary zips.
    (void)magic;
    return ext == ".npz";
}

Result<PointCloud> NpzReader::read(IByteSource& src, const ReadOptions& opt) const {
    const std::vector<std::byte> buf = readAll(src);

    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof(zip));
    if (!mz_zip_reader_init_mem(&zip, buf.data(), buf.size(), 0))
        return makeError(ErrorCode::ParseError, "npz: not a valid zip archive");

    std::vector<std::pair<std::string, NpyArray>> arrays;
    const mz_uint                                 count = mz_zip_reader_get_num_files(&zip);
    for (mz_uint i = 0; i < count; ++i) {
        mz_zip_archive_file_stat st;
        if (!mz_zip_reader_file_stat(&zip, i, &st)) continue;
        std::string name = st.m_filename;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".npy")
            name = name.substr(0, name.size() - 4);

        std::size_t osz = 0;
        void*       p   = mz_zip_reader_extract_to_heap(&zip, i, &osz, 0);
        if (!p) { mz_zip_reader_end(&zip); return makeError(ErrorCode::IoError, "npz: extract failed"); }
        auto arr = parseNpy(static_cast<const std::byte*>(p), osz);
        mz_free(p);
        if (!arr) { mz_zip_reader_end(&zip); return makeError(ErrorCode::Unsupported, "npz: bad npy " + name); }
        arrays.emplace_back(std::move(name), std::move(*arr));
    }
    mz_zip_reader_end(&zip);

    // Find positions (xyz / points / surface_points), establish N.
    const NpyArray* posArr = nullptr;
    for (auto& [name, a] : arrays)
        if (name == "xyz" || name == "points" || name == "surface_points") { posArr = &a; break; }
    if (!posArr || posArr->shape.empty() || (posArr->shape.size() == 2 && posArr->shape[1] != 3))
        return makeError(ErrorCode::Unsupported, "npz: need an 'xyz'/'surface_points' (N,3) array");

    const std::size_t n = posArr->shape[0];
    if (opt.maxPoints && n > opt.maxPoints)
        return makeError(ErrorCode::CloudTooLarge,
                         "npz: " + std::to_string(n) + " points exceeds the limit");
    PointCloud        pc;
    pc.positions().resize(n);

    // Capture per-cloud template metadata (non-(N,*) arrays + __meta__) into the
    // metadata map (template schema, docs/design/05) instead of dropping it.
    static const char* kMetaKeys[] = {"bbox_min",      "bbox_max",         "bbox_center",
                                      "canonical_center", "canonical_axis", "tailstock_tip_local",
                                      "bar_axis_point_local", "bar_axis_dir_local", "point_spacing_m"};
    for (auto& [name, a] : arrays) {
        if (name == "__meta__") {
            std::string js(reinterpret_cast<const char*>(a.data.data()), a.data.size());
            auto field = [&](const char* key) -> std::string {
                auto k = js.find(std::string("\"") + key + "\"");
                if (k == std::string::npos) return {};
                auto q1 = js.find('"', js.find(':', k) + 1);
                auto q2 = js.find('"', q1 + 1);
                return (q1 == std::string::npos || q2 == std::string::npos) ? std::string{}
                                                                            : js.substr(q1 + 1, q2 - q1 - 1);
            };
            if (auto u = field("units"); !u.empty()) pc.metadata()["units"] = u;
            if (auto fr = field("frame"); !fr.empty()) pc.metadata()["frame"] = fr;
            continue;
        }
        bool known = false;
        for (const char* k : kMetaKeys)
            if (name == k) { known = true; break; }
        if (!known) continue;
        std::size_t total = 1;
        for (std::size_t s : a.shape) total *= s;
        if (a.shape.empty()) total = a.data.size() / attrTypeSize(a.type);  // 0-d scalar
        std::string s;
        char        tmp[40];
        for (std::size_t e = 0; e < total; ++e) {
            std::snprintf(tmp, sizeof(tmp), "%.9g",
                          loadDouble(a.data.data() + e * attrTypeSize(a.type), a.type));
            if (e) s += ' ';
            s += tmp;
        }
        pc.metadata()[name] = s;
    }

    auto elemAt = [](const NpyArray& a, std::size_t i, std::size_t c, std::size_t comps) {
        const std::size_t idx = a.fortran ? c * a.shape[0] + i : i * comps + c;
        return loadDouble(a.data.data() + idx * attrTypeSize(a.type), a.type);
    };

    for (auto& [name, a] : arrays) {
        if (a.shape.empty() || a.shape[0] != n) {
            if (&a == posArr) {}  // positions already validated
            else continue;        // skip arrays that don't match the point count
        }
        const std::size_t comps = a.shape.size() > 1 ? a.shape[1] : 1;

        if (&a == posArr) {
            for (std::size_t i = 0; i < n; ++i)
                pc.positions()[i] = {static_cast<float>(elemAt(a, i, 0, 3)),
                                     static_cast<float>(elemAt(a, i, 1, 3)),
                                     static_cast<float>(elemAt(a, i, 2, 3))};
        } else if ((name == "rgb" || name == "colors") && comps >= 3) {
            AttributeColumn col(std::string(attr::kRGB), AttrType::U8, static_cast<std::uint8_t>(comps), n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t c = 0; c < comps; ++c)
                    writeScalar(col, i * comps + c, elemAt(a, i, c, comps));
            pc.add(std::move(col));
        } else if ((name == "normal" || name == "normals" || name == "surface_normals") && comps == 3) {
            AttributeColumn col(std::string(attr::kNormal), AttrType::F32, 3, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t c = 0; c < 3; ++c)
                    writeScalar(col, i * 3 + c, elemAt(a, i, c, 3));
            pc.add(std::move(col));
        } else {
            AttributeColumn col(name, a.type, static_cast<std::uint8_t>(comps), n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t c = 0; c < comps; ++c)
                    writeScalar(col, i * comps + c, elemAt(a, i, c, comps));
            pc.add(std::move(col));
        }
    }
    return pc;
}

// ===========================================================================
// Writer
// ===========================================================================
FormatInfo NpzWriter::info() const { return {"npz", {".npz"}, false, true}; }

std::vector<std::string> NpzWriter::writable_fields(const PointCloud& pc) const {
    std::vector<std::string> out;
    for (const auto& c : pc.attributes()) out.push_back(c.name());
    return out;
}

Result<void> NpzWriter::write(const PointCloud& pc, IByteSink& sink, const WriteOptions& opt) const {
    auto selected = [&](const std::string& name) {
        if (opt.fields.empty()) return true;
        for (const auto& f : opt.fields)
            if (f == name) return true;
        return false;
    };

    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof(zip));
    mz_zip_writer_init_heap(&zip, 0, 0);

    auto addEntry = [&](const std::string& key, const std::vector<std::byte>& npy) {
        const std::string fn = key + ".npy";
        mz_zip_writer_add_mem(&zip, fn.c_str(), npy.data(), npy.size(), MZ_NO_COMPRESSION);
    };

    const std::size_t n = pc.size();

    // positions -> xyz (N,3) f4
    {
        std::vector<float> xyz;
        xyz.reserve(n * 3);
        for (const Vec3& p : pc.positions()) { xyz.push_back(p.x); xyz.push_back(p.y); xyz.push_back(p.z); }
        addEntry("xyz", serializeNpy("<f4", {n, 3}, reinterpret_cast<const std::byte*>(xyz.data()),
                                     xyz.size() * sizeof(float)));
    }

    for (const AttributeColumn& c : pc.attributes()) {
        if (!selected(c.name())) continue;
        std::vector<std::size_t> shape = c.arity() == 1 ? std::vector<std::size_t>{n}
                                                        : std::vector<std::size_t>{n, c.arity()};
        addEntry(c.name(), serializeNpy(attrToDescr(c.type()), shape, c.bytes().data(),
                                        c.bytes().size()));
    }

    void*       p   = nullptr;
    std::size_t psz = 0;
    if (!mz_zip_writer_finalize_heap_archive(&zip, &p, &psz)) {
        mz_zip_writer_end(&zip);
        return makeError(ErrorCode::IoError, "npz: zip finalize failed");
    }
    sink.write({static_cast<const std::byte*>(p), psz});
    mz_free(p);
    mz_zip_writer_end(&zip);
    sink.flush();
    return {};
}

Result<void> writeTemplateNpz(const PointCloud& pc, const TemplateMeta& meta, IByteSink& sink) {
    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof(zip));
    mz_zip_writer_init_heap(&zip, 0, 0);

    auto addEntry = [&](const std::string& key, const std::vector<std::byte>& npy) {
        const std::string fn = key + ".npy";
        mz_zip_writer_add_mem(&zip, fn.c_str(), npy.data(), npy.size(), MZ_NO_COMPRESSION);
    };
    auto addVec = [&](const std::string& key, const Vec3& v, bool rowVec) {
        const float              f[3] = {v.x, v.y, v.z};
        std::vector<std::size_t> shape =
            rowVec ? std::vector<std::size_t>{1, 3} : std::vector<std::size_t>{3};
        addEntry(key, serializeNpy("<f4", shape, reinterpret_cast<const std::byte*>(f), sizeof(f)));
    };

    const std::size_t n = pc.size();

    // surface_points (N,3) f4
    {
        std::vector<float> xyz;
        xyz.reserve(n * 3);
        for (const Vec3& p : pc.positions()) { xyz.push_back(p.x); xyz.push_back(p.y); xyz.push_back(p.z); }
        addEntry("surface_points", serializeNpy("<f4", {n, 3},
                                                reinterpret_cast<const std::byte*>(xyz.data()),
                                                xyz.size() * sizeof(float)));
    }
    // surface_normals (N,3) f4 — only if the cloud carries normals
    if (const AttributeColumn* nc = pc.find(attr::kNormal);
        nc && nc->arity() == 3 && nc->type() == AttrType::F32) {
        addEntry("surface_normals", serializeNpy("<f4", {n, 3}, nc->bytes().data(), nc->bytes().size()));
    }

    // per-cloud metadata arrays
    addVec("bbox_min", meta.bbox_min, false);
    addVec("bbox_max", meta.bbox_max, false);
    addVec("bbox_center", meta.bbox_center, false);
    addVec("canonical_center", meta.canonical_center, true);
    addVec("canonical_axis", meta.canonical_axis, true);
    if (meta.tip_local) addVec("tailstock_tip_local", *meta.tip_local, false);
    if (meta.bar_point_local) addVec("bar_axis_point_local", *meta.bar_point_local, false);
    if (meta.bar_dir_local) addVec("bar_axis_dir_local", *meta.bar_dir_local, false);
    {
        const float sp = meta.point_spacing_m;  // scalar () shape
        addEntry("point_spacing_m",
                 serializeNpy("<f4", {}, reinterpret_cast<const std::byte*>(&sp), sizeof(sp)));
    }
    {
        const std::string js =
            "{\"units\": \"" + meta.units + "\", \"frame\": \"" + meta.frame + "\"}";
        addEntry("__meta__", serializeNpy("|u1", {js.size()},
                                          reinterpret_cast<const std::byte*>(js.data()), js.size()));
    }

    void*       p   = nullptr;
    std::size_t psz = 0;
    if (!mz_zip_writer_finalize_heap_archive(&zip, &p, &psz)) {
        mz_zip_writer_end(&zip);
        return makeError(ErrorCode::IoError, "npz: zip finalize failed");
    }
    sink.write({static_cast<const std::byte*>(p), psz});
    mz_free(p);
    mz_zip_writer_end(&zip);
    sink.flush();
    return {};
}

}  // namespace cc::io
