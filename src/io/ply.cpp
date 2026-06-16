#include "cloudcropper/io/ply.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace cc::io {
namespace {

// ---- type mapping ---------------------------------------------------------
std::optional<AttrType> plyTypeFromString(const std::string& s) {
    if (s == "char" || s == "int8") return AttrType::I8;
    if (s == "uchar" || s == "uint8") return AttrType::U8;
    if (s == "short" || s == "int16") return AttrType::I16;
    if (s == "ushort" || s == "uint16") return AttrType::U16;
    if (s == "int" || s == "int32") return AttrType::I32;
    if (s == "uint" || s == "uint32") return AttrType::U32;
    if (s == "long" || s == "int64") return AttrType::I64;
    if (s == "ulong" || s == "uint64") return AttrType::U64;
    if (s == "float" || s == "float32") return AttrType::F32;
    if (s == "double" || s == "float64") return AttrType::F64;
    return std::nullopt;
}

const char* plyTypeToString(AttrType t) {
    switch (t) {
        case AttrType::I8:  return "char";
        case AttrType::U8:  return "uchar";
        case AttrType::I16: return "short";
        case AttrType::U16: return "ushort";
        case AttrType::I32: return "int";
        case AttrType::U32: return "uint";
        case AttrType::F32: return "float";
        case AttrType::F64: return "double";
        case AttrType::I64:
        case AttrType::U64: return "double";  // not std PLY; widen losslessly enough
    }
    return "float";
}

bool isIntegerType(AttrType t) {
    return t != AttrType::F32 && t != AttrType::F64;
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
    std::byte buf[8];
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

// ---- header model ---------------------------------------------------------
struct PlyProp {
    std::string name;
    AttrType    type = AttrType::F32;
    bool        list = false;
};
struct PlyElement {
    std::string          name;
    std::uint64_t        count = 0;
    std::vector<PlyProp> props;
};

std::vector<std::string> tokenize(const std::string& line) {
    std::istringstream       iss(line);
    std::vector<std::string> tok;
    std::string              t;
    while (iss >> t) tok.push_back(t);
    return tok;
}

// Routing of a vertex property to a destination in the PointCloud.
enum class Dst { PosX, PosY, PosZ, Rgb, Normal, Generic, Ignore };
struct Route {
    Dst         dst  = Dst::Ignore;
    int         comp = 0;
    std::size_t genericIndex = 0;  // index into the generic columns vector
};

}  // namespace

// ===========================================================================
// Reader
// ===========================================================================
FormatInfo PlyReader::info() const { return {"ply", {".ply"}, true, false}; }

bool PlyReader::can_handle(std::string_view, std::span<const std::byte> magic) const {
    return magic.size() >= 3 && static_cast<char>(magic[0]) == 'p' &&
           static_cast<char>(magic[1]) == 'l' && static_cast<char>(magic[2]) == 'y';
}

Result<PointCloud> PlyReader::read(IByteSource& src, const ReadOptions& opt) const {
    const std::vector<std::byte> buf = readAll(src);
    if (buf.size() < 4) return makeError(ErrorCode::ParseError, "ply: file too small");

    // --- parse header lines ---
    std::vector<std::string> header;
    std::size_t              pos       = 0;
    std::size_t              bodyStart = 0;
    bool                     gotEnd    = false;
    while (pos < buf.size()) {
        std::size_t nl = pos;
        while (nl < buf.size() && static_cast<char>(buf[nl]) != '\n') ++nl;
        std::string line(reinterpret_cast<const char*>(buf.data() + pos), nl - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        pos = (nl < buf.size()) ? nl + 1 : nl;
        if (line == "end_header") {
            bodyStart = pos;
            gotEnd    = true;
            break;
        }
        header.push_back(line);
    }
    if (!gotEnd) return makeError(ErrorCode::ParseError, "ply: missing end_header");
    if (header.empty() || header[0] != "ply")
        return makeError(ErrorCode::ParseError, "ply: missing magic");

    // --- format + elements ---
    enum class Fmt { Ascii, BinaryLE } fmt = Fmt::Ascii;
    std::vector<PlyElement> elems;
    for (std::size_t i = 1; i < header.size(); ++i) {
        const auto tok = tokenize(header[i]);
        if (tok.empty()) continue;
        if (tok[0] == "format") {
            if (tok.size() < 2) return makeError(ErrorCode::ParseError, "ply: bad format line");
            if (tok[1] == "ascii") fmt = Fmt::Ascii;
            else if (tok[1] == "binary_little_endian") fmt = Fmt::BinaryLE;
            else return makeError(ErrorCode::Unsupported, "ply: only ascii/binary_little_endian");
        } else if (tok[0] == "element") {
            if (tok.size() < 3) return makeError(ErrorCode::ParseError, "ply: bad element line");
            elems.push_back({tok[1], std::stoull(tok[2]), {}});
        } else if (tok[0] == "property") {
            if (elems.empty()) return makeError(ErrorCode::ParseError, "ply: property before element");
            if (tok.size() >= 2 && tok[1] == "list") {
                elems.back().props.push_back({tok.back(), AttrType::I32, true});
            } else {
                if (tok.size() < 3) return makeError(ErrorCode::ParseError, "ply: bad property");
                auto t = plyTypeFromString(tok[1]);
                if (!t) return makeError(ErrorCode::Unsupported, "ply: unknown type " + tok[1]);
                elems.back().props.push_back({tok[2], *t, false});
            }
        }
    }

    if (elems.empty() || elems[0].name != "vertex")
        return makeError(ErrorCode::Unsupported, "ply: first element must be 'vertex'");
    const PlyElement& vert = elems[0];
    for (const PlyProp& p : vert.props) {
        if (p.list) return makeError(ErrorCode::Unsupported, "ply: list properties on vertex");
    }

    // --- build routing + destination columns ---
    const std::size_t n = vert.count;
    if (opt.maxPoints && n > opt.maxPoints)
        return makeError(ErrorCode::CloudTooLarge,
                         "ply: " + std::to_string(n) + " points exceeds the limit");
    PointCloud        pc;
    pc.positions().resize(n);

    // Carry `comment key=value` header lines into metadata (units, frame, ...).
    for (const std::string& h : header) {
        if (h.rfind("comment ", 0) != 0) continue;
        const std::string rest = h.substr(8);
        const auto        eq   = rest.find('=');
        if (eq != std::string::npos && eq > 0)
            pc.metadata()[rest.substr(0, eq)] = rest.substr(eq + 1);
    }

    bool         hasRgb = false, hasAlpha = false, hasNormal = false;
    for (const PlyProp& p : vert.props) {
        if (p.name == "red" || p.name == "green" || p.name == "blue") hasRgb = true;
        if (p.name == "alpha") hasAlpha = true;
        if (p.name == "nx" || p.name == "ny" || p.name == "nz") hasNormal = true;
    }
    const std::uint8_t rgbArity = hasAlpha ? 4 : 3;

    AttributeColumn              rgbCol = hasRgb ? AttributeColumn(std::string(attr::kRGB),
                                                                  AttrType::U8, rgbArity, n)
                                                : AttributeColumn();
    AttributeColumn              nrmCol = hasNormal ? AttributeColumn(std::string(attr::kNormal),
                                                                     AttrType::F32, 3, n)
                                                    : AttributeColumn();
    std::vector<AttributeColumn> generics;

    std::vector<Route> routes;
    routes.reserve(vert.props.size());
    for (const PlyProp& p : vert.props) {
        Route r;
        if (p.name == "x") r = {Dst::PosX, 0, 0};
        else if (p.name == "y") r = {Dst::PosY, 0, 0};
        else if (p.name == "z") r = {Dst::PosZ, 0, 0};
        else if (p.name == "red") r = {Dst::Rgb, 0, 0};
        else if (p.name == "green") r = {Dst::Rgb, 1, 0};
        else if (p.name == "blue") r = {Dst::Rgb, 2, 0};
        else if (p.name == "alpha") r = {Dst::Rgb, 3, 0};
        else if (p.name == "nx") r = {Dst::Normal, 0, 0};
        else if (p.name == "ny") r = {Dst::Normal, 1, 0};
        else if (p.name == "nz") r = {Dst::Normal, 2, 0};
        else {
            r = {Dst::Generic, 0, generics.size()};
            generics.emplace_back(p.name, p.type, 1, n);
        }
        routes.push_back(r);
    }

    // --- read body ---
    auto store = [&](std::size_t i, std::size_t pi, double v) {
        const Route& r = routes[pi];
        switch (r.dst) {
            case Dst::PosX: pc.positions()[i].x = static_cast<float>(v); break;
            case Dst::PosY: pc.positions()[i].y = static_cast<float>(v); break;
            case Dst::PosZ: pc.positions()[i].z = static_cast<float>(v); break;
            case Dst::Rgb: writeScalar(rgbCol, i * rgbArity + r.comp, v); break;
            case Dst::Normal: writeScalar(nrmCol, i * 3 + r.comp, v); break;
            case Dst::Generic: writeScalar(generics[r.genericIndex], i, v); break;
            case Dst::Ignore: break;
        }
    };

    if (fmt == Fmt::Ascii) {
        std::string body(reinterpret_cast<const char*>(buf.data() + bodyStart), buf.size() - bodyStart);
        std::istringstream iss(body);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t pi = 0; pi < vert.props.size(); ++pi) {
                double v = 0.0;
                if (!(iss >> v)) return makeError(ErrorCode::ParseError, "ply: truncated ascii body");
                store(i, pi, v);
            }
        }
    } else {
        std::size_t off = bodyStart;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t pi = 0; pi < vert.props.size(); ++pi) {
                const AttrType    t = vert.props[pi].type;
                const std::size_t sz = attrTypeSize(t);
                if (off + sz > buf.size())
                    return makeError(ErrorCode::ParseError, "ply: truncated binary body");
                store(i, pi, loadDouble(buf.data() + off, t));
                off += sz;
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
FormatInfo PlyWriter::info() const { return {"ply", {".ply"}, false, true}; }

std::vector<std::string> PlyWriter::writable_fields(const PointCloud& pc) const {
    std::vector<std::string> out;
    for (const auto& c : pc.attributes()) out.push_back(c.name());
    return out;
}

namespace {
struct OutProp {
    std::string            name;
    AttrType               type;
    Dst                    dst;
    int                    comp;
    const AttributeColumn* col;  // for Rgb/Normal/Generic
};
}  // namespace

Result<void> PlyWriter::write(const PointCloud& pc, IByteSink& sink, const WriteOptions& opt) const {
    const bool ascii = (opt.encoding == Encoding::Ascii);

    auto selected = [&](const std::string& name) {
        if (opt.fields.empty()) return true;
        for (const auto& f : opt.fields)
            if (f == name) return true;
        return false;
    };

    // Build output property list: xyz first, then selected attributes.
    std::vector<OutProp> props;
    props.push_back({"x", AttrType::F32, Dst::PosX, 0, nullptr});
    props.push_back({"y", AttrType::F32, Dst::PosY, 0, nullptr});
    props.push_back({"z", AttrType::F32, Dst::PosZ, 0, nullptr});

    for (const AttributeColumn& c : pc.attributes()) {
        if (!selected(c.name())) continue;
        if (c.name() == attr::kRGB && c.type() == AttrType::U8 &&
            (c.arity() == 3 || c.arity() == 4)) {
            const char* names[4] = {"red", "green", "blue", "alpha"};
            for (int k = 0; k < c.arity(); ++k)
                props.push_back({names[k], AttrType::U8, Dst::Rgb, k, &c});
        } else if (c.name() == attr::kNormal && c.arity() == 3) {
            const char* names[3] = {"nx", "ny", "nz"};
            for (int k = 0; k < 3; ++k)
                props.push_back({names[k], AttrType::F32, Dst::Normal, k, &c});
        } else if (c.arity() == 1) {
            props.push_back({c.name(), c.type(), Dst::Generic, 0, &c});
        } else {
            for (int k = 0; k < c.arity(); ++k)
                props.push_back({c.name() + "_" + std::to_string(k), c.type(), Dst::Generic, k, &c});
        }
    }

    auto valueAt = [&](const OutProp& p, std::size_t row) -> double {
        switch (p.dst) {
            case Dst::PosX: return pc.positions()[row].x;
            case Dst::PosY: return pc.positions()[row].y;
            case Dst::PosZ: return pc.positions()[row].z;
            case Dst::Rgb:
            case Dst::Normal:
            case Dst::Generic:
                return readScalar(*p.col, row * p.col->arity() + static_cast<std::size_t>(p.comp));
            case Dst::Ignore: return 0.0;
        }
        return 0.0;
    };

    // --- header ---
    std::string header = "ply\n";
    header += ascii ? "format ascii 1.0\n" : "format binary_little_endian 1.0\n";
    header += "comment Generated by CloudCropper\n";
    for (const auto& [k, v] : pc.metadata())
        header += "comment " + k + "=" + v + "\n";  // units=m, frame=..., etc.
    header += "element vertex " + std::to_string(pc.size()) + "\n";
    for (const OutProp& p : props) {
        header += "property ";
        header += plyTypeToString(p.type);
        header += " ";
        header += p.name;
        header += "\n";
    }
    header += "end_header\n";
    sink.write({reinterpret_cast<const std::byte*>(header.data()), header.size()});

    // --- body ---
    if (ascii) {
        std::string body;
        char        tmp[64];
        for (std::size_t i = 0; i < pc.size(); ++i) {
            for (std::size_t pi = 0; pi < props.size(); ++pi) {
                const double v = valueAt(props[pi], i);
                if (isIntegerType(props[pi].type))
                    std::snprintf(tmp, sizeof(tmp), "%lld", static_cast<long long>(v));
                else if (props[pi].type == AttrType::F64)
                    std::snprintf(tmp, sizeof(tmp), "%.17g", v);
                else
                    std::snprintf(tmp, sizeof(tmp), "%.9g", v);
                body += tmp;
                body += (pi + 1 < props.size()) ? ' ' : '\n';
            }
        }
        sink.write({reinterpret_cast<const std::byte*>(body.data()), body.size()});
    } else {
        std::vector<std::byte> body;
        body.reserve(pc.size() * props.size() * 4);
        for (std::size_t i = 0; i < pc.size(); ++i)
            for (const OutProp& p : props) appendBytes(body, p.type, valueAt(p, i));
        sink.write(body);
    }
    sink.flush();
    return {};
}

}  // namespace cc::io
