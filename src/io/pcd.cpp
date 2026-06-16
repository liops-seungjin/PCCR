#include "cloudcropper/io/pcd.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#if defined(CLOUDCROPPER_HAS_LZF)
#include <lzf.h>
#endif

namespace cc::io {
namespace {

struct PcdField {
    std::string name;
    char        type = 'F';  // I | U | F
    int         size = 4;    // 1,2,4,8
    int         count = 1;
};

std::optional<AttrType> attrFromPcd(char type, int size) {
    switch (type) {
        case 'F': return size == 8 ? AttrType::F64 : AttrType::F32;
        case 'U':
            return size == 1 ? AttrType::U8 : size == 2 ? AttrType::U16
                           : size == 8     ? AttrType::U64
                                           : AttrType::U32;
        case 'I':
            return size == 1 ? AttrType::I8 : size == 2 ? AttrType::I16
                           : size == 8     ? AttrType::I64
                                           : AttrType::I32;
    }
    return std::nullopt;
}

struct PcdType {
    char type;
    int  size;
};
PcdType pcdFromAttr(AttrType t) {
    switch (t) {
        case AttrType::F32: return {'F', 4};
        case AttrType::F64: return {'F', 8};
        case AttrType::U8:  return {'U', 1};
        case AttrType::U16: return {'U', 2};
        case AttrType::U32: return {'U', 4};
        case AttrType::U64: return {'U', 8};
        case AttrType::I8:  return {'I', 1};
        case AttrType::I16: return {'I', 2};
        case AttrType::I32: return {'I', 4};
        case AttrType::I64: return {'I', 8};
    }
    return {'F', 4};
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

void appendBytes(std::vector<std::byte>& out, AttrType t, double value) {
    std::byte   buf[8];
    std::size_t n = attrTypeSize(t);
    switch (t) {
        case AttrType::I8:  { auto v = static_cast<std::int8_t>(value);   std::memcpy(buf, &v, n); break; }
        case AttrType::U8:  { auto v = static_cast<std::uint8_t>(value);  std::memcpy(buf, &v, n); break; }
        case AttrType::I16: { auto v = static_cast<std::int16_t>(value);  std::memcpy(buf, &v, n); break; }
        case AttrType::U16: { auto v = static_cast<std::uint16_t>(value); std::memcpy(buf, &v, n); break; }
        case AttrType::I32: { auto v = static_cast<std::int32_t>(value);  std::memcpy(buf, &v, n); break; }
        case AttrType::U32: { auto v = static_cast<std::uint32_t>(value); std::memcpy(buf, &v, n); break; }
        case AttrType::F32: { auto v = static_cast<float>(value);         std::memcpy(buf, &v, n); break; }
        case AttrType::F64: { double v = value;                          std::memcpy(buf, &v, n); break; }
        case AttrType::I64: { auto v = static_cast<double>(value);        std::memcpy(buf, &v, n); break; }
        case AttrType::U64: { auto v = static_cast<double>(value);        std::memcpy(buf, &v, n); break; }
    }
    out.insert(out.end(), buf, buf + n);
}

std::vector<std::string> tokenize(const std::string& line) {
    std::istringstream       iss(line);
    std::vector<std::string> tok;
    std::string              t;
    while (iss >> t) tok.push_back(t);
    return tok;
}

}  // namespace

// ===========================================================================
// Reader
// ===========================================================================
FormatInfo PcdReader::info() const { return {"pcd", {".pcd"}, true, false}; }

bool PcdReader::can_handle(std::string_view, std::span<const std::byte> magic) const {
    // PCD files start with a comment line "# .PCD" or directly "VERSION".
    std::string head(reinterpret_cast<const char*>(magic.data()),
                     std::min<std::size_t>(magic.size(), 16));
    return head.rfind("# .PCD", 0) == 0 || head.rfind("VERSION", 0) == 0;
}

Result<PointCloud> PcdReader::read(IByteSource& src, const ReadOptions& opt) const {
    const std::vector<std::byte> buf = readAll(src);

    std::vector<std::string> fieldNames, sizeTok, typeTok, countTok;
    std::uint64_t            points = 0, width = 0, height = 1;
    std::string              dataMode;
    std::string              viewpointStr;  // VIEWPOINT translation (sensor origin)
    std::size_t              pos = 0, bodyStart = 0;
    bool                     gotData = false;

    while (pos < buf.size()) {
        std::size_t nl = pos;
        while (nl < buf.size() && static_cast<char>(buf[nl]) != '\n') ++nl;
        std::string line(reinterpret_cast<const char*>(buf.data() + pos), nl - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        pos = (nl < buf.size()) ? nl + 1 : nl;
        const auto tok = tokenize(line);
        if (tok.empty() || tok[0][0] == '#') continue;
        if (tok[0] == "FIELDS")        fieldNames.assign(tok.begin() + 1, tok.end());
        else if (tok[0] == "SIZE")     sizeTok.assign(tok.begin() + 1, tok.end());
        else if (tok[0] == "TYPE")     typeTok.assign(tok.begin() + 1, tok.end());
        else if (tok[0] == "COUNT")    countTok.assign(tok.begin() + 1, tok.end());
        else if (tok[0] == "WIDTH" && tok.size() > 1)  width = std::stoull(tok[1]);
        else if (tok[0] == "HEIGHT" && tok.size() > 1) height = std::stoull(tok[1]);
        else if (tok[0] == "POINTS" && tok.size() > 1) points = std::stoull(tok[1]);
        else if (tok[0] == "VIEWPOINT" && tok.size() >= 4)
            viewpointStr = tok[1] + " " + tok[2] + " " + tok[3];  // translation = sensor origin
        else if (tok[0] == "DATA") {
            dataMode  = tok.size() > 1 ? tok[1] : "ascii";
            bodyStart = pos;
            gotData   = true;
            break;
        }
    }
    if (!gotData) return makeError(ErrorCode::ParseError, "pcd: missing DATA");
    if (fieldNames.empty() || sizeTok.size() != fieldNames.size() ||
        typeTok.size() != fieldNames.size())
        return makeError(ErrorCode::ParseError, "pcd: FIELDS/SIZE/TYPE mismatch");
    if (points == 0) points = width * height;

    std::vector<PcdField> fields(fieldNames.size());
    for (std::size_t i = 0; i < fields.size(); ++i) {
        fields[i].name  = fieldNames[i];
        fields[i].type  = typeTok[i][0];
        fields[i].size  = std::stoi(sizeTok[i]);
        fields[i].count = countTok.size() == fields.size() ? std::stoi(countTok[i]) : 1;
    }

    const std::uint64_t n = points;
    if (opt.maxPoints && n > opt.maxPoints)
        return makeError(ErrorCode::CloudTooLarge,
                         "pcd: " + std::to_string(n) + " points exceeds the limit");
    PointCloud          pc;
    pc.positions().resize(n);
    if (!viewpointStr.empty()) pc.metadata()["viewpoint"] = viewpointStr;

    // Destination columns.
    bool hasRgb = false, hasAlpha = false, hasNormal = false;
    for (const auto& f : fields) {
        if (f.name == "rgb") hasRgb = true;
        if (f.name == "rgba") { hasRgb = true; hasAlpha = true; }
        if (f.name == "normal_x" || f.name == "normal_y" || f.name == "normal_z") hasNormal = true;
    }
    const std::uint8_t rgbArity = hasAlpha ? 4 : 3;
    AttributeColumn rgbCol = hasRgb ? AttributeColumn(std::string(attr::kRGB), AttrType::U8, rgbArity, n)
                                    : AttributeColumn();
    AttributeColumn nrmCol = hasNormal ? AttributeColumn(std::string(attr::kNormal), AttrType::F32, 3, n)
                                       : AttributeColumn();
    std::vector<AttributeColumn> generics;
    std::vector<int>             genericOf(fields.size(), -1);
    for (std::size_t fi = 0; fi < fields.size(); ++fi) {
        const auto& f = fields[fi];
        if (f.name == "x" || f.name == "y" || f.name == "z" || f.name == "rgb" || f.name == "rgba" ||
            f.name == "normal_x" || f.name == "normal_y" || f.name == "normal_z")
            continue;
        auto at = attrFromPcd(f.type, f.size);
        if (!at) return makeError(ErrorCode::Unsupported, "pcd: bad type for " + f.name);
        genericOf[fi] = static_cast<int>(generics.size());
        generics.emplace_back(f.name, *at, static_cast<std::uint8_t>(f.count), n);
    }

    auto routeValue = [&](std::uint64_t i, std::size_t fi, int comp, double v) {
        const auto& f = fields[fi];
        if (f.name == "x") pc.positions()[i].x = static_cast<float>(v);
        else if (f.name == "y") pc.positions()[i].y = static_cast<float>(v);
        else if (f.name == "z") pc.positions()[i].z = static_cast<float>(v);
        else if (f.name == "normal_x") writeScalar(nrmCol, i * 3 + 0, v);
        else if (f.name == "normal_y") writeScalar(nrmCol, i * 3 + 1, v);
        else if (f.name == "normal_z") writeScalar(nrmCol, i * 3 + 2, v);
        else if (genericOf[fi] >= 0)
            writeScalar(generics[genericOf[fi]], i * f.count + comp, v);
    };
    auto routeRgbBits = [&](std::uint64_t i, std::uint32_t bits) {
        auto b = rgbCol.as<std::uint8_t>();
        b[i * rgbArity + 0] = static_cast<std::uint8_t>((bits >> 16) & 0xff);
        b[i * rgbArity + 1] = static_cast<std::uint8_t>((bits >> 8) & 0xff);
        b[i * rgbArity + 2] = static_cast<std::uint8_t>(bits & 0xff);
        if (rgbArity == 4) b[i * rgbArity + 3] = static_cast<std::uint8_t>((bits >> 24) & 0xff);
    };

    if (dataMode == "ascii") {
        std::string body(reinterpret_cast<const char*>(buf.data() + bodyStart), buf.size() - bodyStart);
        std::istringstream iss(body);
        for (std::uint64_t i = 0; i < n; ++i) {
            for (std::size_t fi = 0; fi < fields.size(); ++fi) {
                const auto& f = fields[fi];
                for (int c = 0; c < f.count; ++c) {
                    double tokv = 0.0;
                    if (!(iss >> tokv)) return makeError(ErrorCode::ParseError, "pcd: short ascii body");
                    if (f.name == "rgb" || f.name == "rgba") {
                        std::uint32_t bits;
                        if (f.type == 'F') { float fv = static_cast<float>(tokv); std::memcpy(&bits, &fv, 4); }
                        else bits = static_cast<std::uint32_t>(tokv);
                        routeRgbBits(i, bits);
                    } else {
                        routeValue(i, fi, c, tokv);
                    }
                }
            }
        }
    } else {
        // Resolve a contiguous binary buffer (decompressing if needed).
        std::vector<std::byte> raw;
        const std::byte*       data = nullptr;
        std::size_t            dataLen = 0;
        bool                   soa = false;

        if (dataMode == "binary") {
            data    = buf.data() + bodyStart;
            dataLen = buf.size() - bodyStart;
        } else if (dataMode == "binary_compressed") {
#if defined(CLOUDCROPPER_HAS_LZF)
            if (bodyStart + 8 > buf.size())
                return makeError(ErrorCode::ParseError, "pcd: truncated compressed header");
            std::uint32_t compLen = 0, uncompLen = 0;
            std::memcpy(&compLen, buf.data() + bodyStart, 4);
            std::memcpy(&uncompLen, buf.data() + bodyStart + 4, 4);
            raw.resize(uncompLen);
            unsigned got = lzf_decompress(buf.data() + bodyStart + 8, compLen, raw.data(), uncompLen);
            if (got != uncompLen)
                return makeError(ErrorCode::ParseError, "pcd: LZF decompress failed");
            data    = raw.data();
            dataLen = raw.size();
            soa     = true;  // binary_compressed is field-major
#else
            return makeError(ErrorCode::Unsupported,
                             "pcd: binary_compressed needs liblzf (build with vcpkg)");
#endif
        } else {
            return makeError(ErrorCode::Unsupported, "pcd: unknown DATA mode " + dataMode);
        }

        // Layout offsets.
        std::vector<std::size_t> fieldBytes(fields.size());
        std::size_t              pointStride = 0;
        for (std::size_t fi = 0; fi < fields.size(); ++fi) {
            fieldBytes[fi] = static_cast<std::size_t>(fields[fi].size) * fields[fi].count;
            pointStride += fieldBytes[fi];
        }
        if (dataLen < pointStride * n)
            return makeError(ErrorCode::ParseError, "pcd: truncated binary body");
        std::vector<std::size_t> soaBlock(fields.size(), 0);
        for (std::size_t fi = 1; fi < fields.size(); ++fi)
            soaBlock[fi] = soaBlock[fi - 1] + fieldBytes[fi - 1] * n;

        auto fieldPtr = [&](std::uint64_t i, std::size_t fi) -> const std::byte* {
            return soa ? data + soaBlock[fi] + i * fieldBytes[fi]
                       : data + i * pointStride + [&] {
                             std::size_t off = 0;
                             for (std::size_t k = 0; k < fi; ++k) off += fieldBytes[k];
                             return off;
                         }();
        };

        for (std::uint64_t i = 0; i < n; ++i) {
            for (std::size_t fi = 0; fi < fields.size(); ++fi) {
                const auto&      f  = fields[fi];
                const std::byte* fp = fieldPtr(i, fi);
                if (f.name == "rgb" || f.name == "rgba") {
                    std::uint32_t bits;
                    std::memcpy(&bits, fp, 4);
                    routeRgbBits(i, bits);
                    continue;
                }
                auto at = attrFromPcd(f.type, f.size);
                if (!at) continue;
                for (int c = 0; c < f.count; ++c)
                    routeValue(i, fi, c, loadDouble(fp + static_cast<std::size_t>(c) * f.size, *at));
            }
        }
    }

    if (hasRgb) pc.add(std::move(rgbCol));
    if (hasNormal) pc.add(std::move(nrmCol));
    for (auto& g : generics) pc.add(std::move(g));
    return pc;
}

// ===========================================================================
// Writer
// ===========================================================================
FormatInfo PcdWriter::info() const { return {"pcd", {".pcd"}, false, true}; }

std::vector<std::string> PcdWriter::writable_fields(const PointCloud& pc) const {
    std::vector<std::string> out;
    for (const auto& c : pc.attributes()) out.push_back(c.name());
    return out;
}

namespace {
// One emitted PCD field. kind: 0 pos(comp x/y/z), 1 rgb-pack, 2 normal(comp),
// 3 generic(col, count).
struct OutField {
    std::string            name;
    char                   type;
    int                    size;
    int                    count;
    int                    kind;
    int                    comp;
    const AttributeColumn* col;
};
}  // namespace

Result<void> PcdWriter::write(const PointCloud& pc, IByteSink& sink, const WriteOptions& opt) const {
    auto selected = [&](const std::string& name) {
        if (opt.fields.empty()) return true;
        for (const auto& f : opt.fields)
            if (f == name) return true;
        return false;
    };

    std::vector<OutField> fields;
    fields.push_back({"x", 'F', 4, 1, 0, 0, nullptr});
    fields.push_back({"y", 'F', 4, 1, 0, 1, nullptr});
    fields.push_back({"z", 'F', 4, 1, 0, 2, nullptr});
    for (const AttributeColumn& c : pc.attributes()) {
        if (!selected(c.name())) continue;
        if (c.name() == attr::kRGB && c.type() == AttrType::U8 && (c.arity() == 3 || c.arity() == 4)) {
            fields.push_back({c.arity() == 4 ? "rgba" : "rgb", 'U', 4, 1, 1, 0, &c});
        } else if (c.name() == attr::kNormal && c.arity() == 3) {
            fields.push_back({"normal_x", 'F', 4, 1, 2, 0, &c});
            fields.push_back({"normal_y", 'F', 4, 1, 2, 1, &c});
            fields.push_back({"normal_z", 'F', 4, 1, 2, 2, &c});
        } else {
            PcdType pt = pcdFromAttr(c.type());
            fields.push_back({c.name(), pt.type, pt.size, c.arity(), 3, 0, &c});
        }
    }

    const std::uint64_t n = pc.size();

    // RGB packed value for point i.
    auto rgbBits = [&](const AttributeColumn& col, std::uint64_t i) -> std::uint32_t {
        auto          b = col.as<std::uint8_t>();
        std::uint32_t v = (static_cast<std::uint32_t>(b[i * col.arity() + 0]) << 16) |
                          (static_cast<std::uint32_t>(b[i * col.arity() + 1]) << 8) |
                          (static_cast<std::uint32_t>(b[i * col.arity() + 2]));
        if (col.arity() == 4) v |= static_cast<std::uint32_t>(b[i * col.arity() + 3]) << 24;
        return v;
    };

    // --- header ---
    std::string fline = "FIELDS", sline = "SIZE", tline = "TYPE", cline = "COUNT";
    for (const auto& f : fields) {
        fline += " " + f.name;
        sline += " " + std::to_string(f.size);
        tline += std::string(" ") + f.type;
        cline += " " + std::to_string(f.count);
    }

    const bool ascii      = (opt.encoding == Encoding::Ascii);
    const bool compressed = (opt.encoding == Encoding::BinaryCompressed);
    std::string dataMode  = ascii ? "ascii" : compressed ? "binary_compressed" : "binary";
#if !defined(CLOUDCROPPER_HAS_LZF)
    if (compressed) return makeError(ErrorCode::Unsupported,
                                     "pcd: binary_compressed needs liblzf (build with vcpkg)");
#endif

    std::string header = "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n";
    header += fline + "\n" + sline + "\n" + tline + "\n" + cline + "\n";
    header += "WIDTH " + std::to_string(n) + "\nHEIGHT 1\n";
    header += "VIEWPOINT 0 0 0 1 0 0 0\n";
    header += "POINTS " + std::to_string(n) + "\nDATA " + dataMode + "\n";
    sink.write({reinterpret_cast<const std::byte*>(header.data()), header.size()});

    // Append one field's value(s) for point i to a byte buffer.
    auto appendField = [&](std::vector<std::byte>& out, const OutField& f, std::uint64_t i) {
        if (f.kind == 0) {
            float v = f.comp == 0 ? pc.positions()[i].x : f.comp == 1 ? pc.positions()[i].y
                                                                      : pc.positions()[i].z;
            appendBytes(out, AttrType::F32, v);
        } else if (f.kind == 1) {
            std::uint32_t bits = rgbBits(*f.col, i);
            out.insert(out.end(), reinterpret_cast<std::byte*>(&bits),
                       reinterpret_cast<std::byte*>(&bits) + 4);
        } else if (f.kind == 2) {
            appendBytes(out, AttrType::F32, readScalar(*f.col, i * 3 + f.comp));
        } else {
            for (int c = 0; c < f.count; ++c)
                appendBytes(out, f.col->type(),
                            readScalar(*f.col, i * f.col->arity() + static_cast<std::size_t>(c)));
        }
    };

    if (ascii) {
        std::string body;
        char        tmp[64];
        for (std::uint64_t i = 0; i < n; ++i) {
            std::string line;
            for (std::size_t fi = 0; fi < fields.size(); ++fi) {
                const OutField& f = fields[fi];
                auto            emit = [&](double v, bool integer) {
                    if (integer) std::snprintf(tmp, sizeof(tmp), "%lld", static_cast<long long>(v));
                    else std::snprintf(tmp, sizeof(tmp), "%.9g", v);
                    if (!line.empty()) line += ' ';
                    line += tmp;
                };
                if (f.kind == 0) emit(f.comp == 0 ? pc.positions()[i].x : f.comp == 1 ? pc.positions()[i].y : pc.positions()[i].z, false);
                else if (f.kind == 1) emit(rgbBits(*f.col, i), true);
                else if (f.kind == 2) emit(readScalar(*f.col, i * 3 + f.comp), false);
                else
                    for (int c = 0; c < f.count; ++c)
                        emit(readScalar(*f.col, i * f.col->arity() + static_cast<std::size_t>(c)),
                             f.type != 'F');
            }
            body += line;
            body += '\n';
        }
        sink.write({reinterpret_cast<const std::byte*>(body.data()), body.size()});
        sink.flush();
        return {};
    }

    if (!compressed) {  // binary (AoS)
        std::vector<std::byte> body;
        for (std::uint64_t i = 0; i < n; ++i)
            for (const auto& f : fields) appendField(body, f, i);
        sink.write(body);
        sink.flush();
        return {};
    }

#if defined(CLOUDCROPPER_HAS_LZF)
    // binary_compressed: build field-major (SoA) buffer, then LZF-compress.
    std::vector<std::byte> soa;
    for (const auto& f : fields)
        for (std::uint64_t i = 0; i < n; ++i) appendField(soa, f, i);

    std::vector<std::byte> comp(soa.size() + soa.size() / 16 + 64);
    unsigned compLen = lzf_compress(soa.data(), static_cast<unsigned>(soa.size()), comp.data(),
                                    static_cast<unsigned>(comp.size()));
    if (compLen == 0) return makeError(ErrorCode::IoError, "pcd: LZF compress failed");
    std::uint32_t cl = compLen, ul = static_cast<std::uint32_t>(soa.size());
    std::vector<std::byte> out;
    out.insert(out.end(), reinterpret_cast<std::byte*>(&cl), reinterpret_cast<std::byte*>(&cl) + 4);
    out.insert(out.end(), reinterpret_cast<std::byte*>(&ul), reinterpret_cast<std::byte*>(&ul) + 4);
    out.insert(out.end(), comp.begin(), comp.begin() + compLen);
    sink.write(out);
    sink.flush();
    return {};
#else
    return makeError(ErrorCode::Unsupported, "pcd: binary_compressed unavailable");
#endif
}

}  // namespace cc::io
