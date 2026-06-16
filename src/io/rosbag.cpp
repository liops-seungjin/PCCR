#include "cloudcropper/io/rosbag.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <optional>
#include <utility>

#include <sqlite3.h>

#if defined(CLOUDCROPPER_HAS_MCAP)
// Header-only vendor header; not clean under our warning set, so isolate it.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <mcap/reader.hpp>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

namespace fs = std::filesystem;

namespace cc::io {
namespace {

// ---- CDR (little-endian XCDR1) reader; alignment is relative to body start ----
class Cdr {
public:
    Cdr(const std::uint8_t* p, std::size_t n) : p_(p), n_(n) {}
    [[nodiscard]] bool        ok() const { return ok_; }
    [[nodiscard]] std::size_t pos() const { return pos_; }

    void align(std::size_t a) {
        const std::size_t m = pos_ % a;
        if (m) pos_ += (a - m);
    }
    template <class T>
    T read() {
        align(sizeof(T));
        T v{};
        if (pos_ + sizeof(T) > n_) { ok_ = false; return v; }
        std::memcpy(&v, p_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::uint8_t  u8() { if (pos_ + 1 > n_) { ok_ = false; return 0; } return p_[pos_++]; }
    std::uint32_t u32() { return read<std::uint32_t>(); }
    std::int32_t  i32() { return read<std::int32_t>(); }
    bool          b() { return u8() != 0; }
    std::string   str() {
        const std::uint32_t len = u32();  // includes the trailing '\0'
        if (!ok_ || pos_ + len > n_) { ok_ = false; return {}; }
        std::string s(reinterpret_cast<const char*>(p_ + pos_), len ? len - 1 : 0);
        pos_ += len;
        return s;
    }

private:
    const std::uint8_t* p_;
    std::size_t         n_;
    std::size_t         pos_ = 0;
    bool                ok_  = true;
};

// sensor_msgs/PointField datatypes.
std::optional<AttrType> dtToAttr(std::uint8_t dt) {
    switch (dt) {
        case 1: return AttrType::I8;
        case 2: return AttrType::U8;
        case 3: return AttrType::I16;
        case 4: return AttrType::U16;
        case 5: return AttrType::I32;
        case 6: return AttrType::U32;
        case 7: return AttrType::F32;
        case 8: return AttrType::F64;
    }
    return std::nullopt;
}
std::size_t dtSize(std::uint8_t dt) {
    switch (dt) {
        case 1: case 2: return 1;
        case 3: case 4: return 2;
        case 5: case 6: case 7: return 4;
        case 8: return 8;
    }
    return 0;
}
double loadDouble(const std::uint8_t* p, std::uint8_t dt) {
    switch (dt) {
        case 1: { std::int8_t v;   std::memcpy(&v, p, 1); return v; }
        case 2: { std::uint8_t v;  std::memcpy(&v, p, 1); return v; }
        case 3: { std::int16_t v;  std::memcpy(&v, p, 2); return v; }
        case 4: { std::uint16_t v; std::memcpy(&v, p, 2); return v; }
        case 5: { std::int32_t v;  std::memcpy(&v, p, 4); return v; }
        case 6: { std::uint32_t v; std::memcpy(&v, p, 4); return v; }
        case 7: { float v;         std::memcpy(&v, p, 4); return v; }
        case 8: { double v;        std::memcpy(&v, p, 8); return v; }
    }
    return 0.0;
}

struct PField {
    std::string   name;
    std::uint32_t offset;
    std::uint8_t  dt;
    std::uint32_t count;
};

Result<PointCloud> parseMessage(const std::uint8_t* blob, std::size_t len) {
    if (len < 4 || blob[0] != 0x00 || (blob[1] & 0x01) == 0)
        return makeError(ErrorCode::Unsupported, "rosbag: expected little-endian CDR");
    Cdr c(blob + 4, len - 4);
    c.i32();  // header.stamp.sec
    c.u32();  // header.stamp.nanosec
    c.str();  // header.frame_id
    const std::uint32_t height = c.u32();
    const std::uint32_t width  = c.u32();
    const std::uint32_t nf     = c.u32();
    std::vector<PField> fields;
    fields.reserve(nf);
    for (std::uint32_t i = 0; i < nf && c.ok(); ++i) {
        PField f;
        f.name   = c.str();
        f.offset = c.u32();
        f.dt     = c.u8();
        f.count  = c.u32();
        fields.push_back(std::move(f));
    }
    const bool          bigendian  = c.b();
    const std::uint32_t point_step = c.u32();
    c.u32();  // row_step
    const std::uint32_t dlen = c.u32();
    if (!c.ok()) return makeError(ErrorCode::ParseError, "rosbag: truncated PointCloud2 header");
    if (bigendian) return makeError(ErrorCode::Unsupported, "rosbag: big-endian point data");

    const std::uint8_t* pdata = blob + 4 + c.pos();
    const std::uint64_t npts  = static_cast<std::uint64_t>(width) * height;
    if (4 + c.pos() + dlen > len || point_step == 0 ||
        static_cast<std::uint64_t>(point_step) * npts > dlen)
        return makeError(ErrorCode::ParseError, "rosbag: PointCloud2 data too small");

    PointCloud pc;
    pc.positions().resize(npts);
    bool hasRgb = false, hasAlpha = false, hasNormal = false;
    for (const auto& f : fields) {
        if (f.name == "rgb") hasRgb = true;
        if (f.name == "rgba") { hasRgb = true; hasAlpha = true; }
        if (f.name == "normal_x" || f.name == "normal_y" || f.name == "normal_z") hasNormal = true;
    }
    const std::uint8_t rgbArity = hasAlpha ? 4 : 3;
    AttributeColumn rgbCol = hasRgb ? AttributeColumn(std::string(attr::kRGB), AttrType::U8, rgbArity, npts)
                                    : AttributeColumn();
    AttributeColumn nrmCol = hasNormal ? AttributeColumn(std::string(attr::kNormal), AttrType::F32, 3, npts)
                                       : AttributeColumn();
    std::vector<AttributeColumn> generics;
    std::vector<int>             genIdx(fields.size(), -1);
    for (std::size_t fi = 0; fi < fields.size(); ++fi) {
        const auto& f = fields[fi];
        if (f.name == "x" || f.name == "y" || f.name == "z" || f.name == "rgb" || f.name == "rgba" ||
            f.name == "normal_x" || f.name == "normal_y" || f.name == "normal_z")
            continue;
        auto at = dtToAttr(f.dt);
        if (!at || f.count == 0) continue;
        genIdx[fi] = static_cast<int>(generics.size());
        generics.emplace_back(f.name, *at, static_cast<std::uint8_t>(f.count), npts);
    }

    for (std::uint64_t i = 0; i < npts; ++i) {
        const std::uint8_t* base = pdata + i * point_step;
        for (std::size_t fi = 0; fi < fields.size(); ++fi) {
            const PField&       f  = fields[fi];
            const std::uint8_t* fp = base + f.offset;
            if (f.name == "x") pc.positions()[i].x = static_cast<float>(loadDouble(fp, f.dt));
            else if (f.name == "y") pc.positions()[i].y = static_cast<float>(loadDouble(fp, f.dt));
            else if (f.name == "z") pc.positions()[i].z = static_cast<float>(loadDouble(fp, f.dt));
            else if (f.name == "rgb" || f.name == "rgba") {
                std::uint32_t bits;
                std::memcpy(&bits, fp, 4);
                auto col = rgbCol.as<std::uint8_t>();
                col[i * rgbArity + 0] = static_cast<std::uint8_t>((bits >> 16) & 0xff);
                col[i * rgbArity + 1] = static_cast<std::uint8_t>((bits >> 8) & 0xff);
                col[i * rgbArity + 2] = static_cast<std::uint8_t>(bits & 0xff);
                if (rgbArity == 4) col[i * rgbArity + 3] = static_cast<std::uint8_t>((bits >> 24) & 0xff);
            } else if (f.name == "normal_x") writeScalar(nrmCol, i * 3 + 0, loadDouble(fp, f.dt));
            else if (f.name == "normal_y") writeScalar(nrmCol, i * 3 + 1, loadDouble(fp, f.dt));
            else if (f.name == "normal_z") writeScalar(nrmCol, i * 3 + 2, loadDouble(fp, f.dt));
            else if (genIdx[fi] >= 0) {
                const std::size_t sz = dtSize(f.dt);
                for (std::uint32_t k = 0; k < f.count; ++k)
                    writeScalar(generics[genIdx[fi]], i * f.count + k, loadDouble(fp + k * sz, f.dt));
            }
        }
    }
    if (hasRgb) pc.add(std::move(rgbCol));
    if (hasNormal) pc.add(std::move(nrmCol));
    for (auto& g : generics) pc.add(std::move(g));
    return pc;
}

void appendCloud(PointCloud& dst, const PointCloud& src) {
    auto& dp = dst.positions();
    dp.insert(dp.end(), src.positions().begin(), src.positions().end());
    for (const AttributeColumn& sc : src.attributes()) {
        AttributeColumn* dc = dst.find(sc.name());
        if (dc && dc->type() == sc.type() && dc->arity() == sc.arity())
            dc->raw().insert(dc->raw().end(), sc.raw().begin(), sc.raw().end());
    }
}

// A bag is either a rosbag2 SQLite `.db3` or an `.mcap` file (possibly inside
// the bag directory). Resolve a user path to the concrete container file.
struct BagFile {
    std::string path;
    bool        mcap = false;
};

Result<BagFile> resolveBag(const std::string& path) {
    std::error_code ec;
    if (fs::is_directory(path, ec)) {
        std::vector<std::string> db3, mc;
        for (const auto& e : fs::directory_iterator(path, ec)) {
            if (e.path().extension() == ".db3") db3.push_back(e.path().string());
            if (e.path().extension() == ".mcap") mc.push_back(e.path().string());
        }
        if (!db3.empty()) {
            std::sort(db3.begin(), db3.end());
            return BagFile{db3.front(), false};
        }
        if (!mc.empty()) {
            std::sort(mc.begin(), mc.end());
            return BagFile{mc.front(), true};
        }
        return makeError(ErrorCode::NotFound, "rosbag: no .db3/.mcap in " + path);
    }
    const bool isMcap =
        path.size() >= 5 && path.compare(path.size() - 5, 5, ".mcap") == 0;
    return BagFile{path, isMcap};
}

#if defined(CLOUDCROPPER_HAS_MCAP)

constexpr const char* kPc2Type = "sensor_msgs/msg/PointCloud2";

Result<std::vector<BagTopic>> listMcapTopics(const std::string& path) {
    mcap::McapReader reader;
    auto             st = reader.open(path);
    if (!st.ok()) return makeError(ErrorCode::IoError, "mcap: " + st.message);
    (void)reader.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);

    std::vector<BagTopic> out;
    for (const auto& [cid, ch] : reader.channels()) {
        BagTopic bt;
        bt.name        = ch->topic;
        const auto sit = reader.schemas().find(ch->schemaId);
        if (sit != reader.schemas().end()) bt.type = sit->second->name;
        if (const auto& stats = reader.statistics()) {
            const auto it = stats->channelMessageCounts.find(cid);
            if (it != stats->channelMessageCounts.end()) bt.count = it->second;
        }
        bool merged = false;  // a topic can span several channels
        for (auto& existing : out)
            if (existing.name == bt.name) {
                existing.count += bt.count;
                merged = true;
                break;
            }
        if (!merged) out.push_back(std::move(bt));
    }
    if (!reader.statistics()) {  // no summary statistics: count by scanning
        const auto onProblem = [](const mcap::Status&) {};
        for (const auto& mv : reader.readMessages(onProblem, mcap::ReadMessageOptions{})) {
            for (auto& bt : out)
                if (bt.name == mv.channel->topic) {
                    ++bt.count;
                    break;
                }
        }
    }
    reader.close();
    return out;
}

Result<PointCloud> readMcap(const std::string& path, const BagReadOptions& opt) {
    mcap::McapReader reader;
    auto             st = reader.open(path);
    if (!st.ok()) return makeError(ErrorCode::IoError, "mcap: " + st.message);
    (void)reader.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);

    // candidate PointCloud2 topics
    std::vector<std::string> pcTopics;
    for (const auto& [cid, ch] : reader.channels()) {
        (void)cid;
        const auto sit = reader.schemas().find(ch->schemaId);
        if (sit == reader.schemas().end() || sit->second->name != kPc2Type) continue;
        if (std::find(pcTopics.begin(), pcTopics.end(), ch->topic) == pcTopics.end())
            pcTopics.push_back(ch->topic);
    }
    if (pcTopics.empty()) return makeError(ErrorCode::NotFound, "rosbag: no PointCloud2 topic");
    std::string topic;
    if (!opt.topic.empty()) {
        if (std::find(pcTopics.begin(), pcTopics.end(), opt.topic) == pcTopics.end())
            return makeError(ErrorCode::NotFound, "rosbag: topic not found: " + opt.topic);
        topic = opt.topic;
    } else if (pcTopics.size() == 1) {
        topic = pcTopics.front();
    } else {
        std::string names;
        for (const auto& n : pcTopics) names += " " + n;
        return makeError(ErrorCode::InvalidArgument,
                         "rosbag: multiple PointCloud2 topics; use --bag-topic (" + names + " )");
    }

    mcap::ReadMessageOptions ro;
    ro.topicFilter        = [&](std::string_view t) { return t == topic; };
    const auto onProblem  = [](const mcap::Status&) {};

    PointCloud out;
    bool       first = true;
    int        idx   = 0;
    for (const auto& mv : reader.readMessages(onProblem, ro)) {
        if (!opt.merge && idx++ < std::max(0, opt.frame)) continue;
        auto msg = parseMessage(reinterpret_cast<const std::uint8_t*>(mv.message.data),
                                mv.message.dataSize);
        if (!msg) return makeError(msg.error().code, msg.error().message);
        if (first) {
            out   = std::move(msg.value());
            first = false;
        } else {
            appendCloud(out, msg.value());
        }
        if (opt.maxPoints && out.size() > opt.maxPoints)
            return makeError(ErrorCode::CloudTooLarge,
                             "rosbag: " + std::to_string(out.size()) + " points exceeds the limit");
        if (!opt.merge) break;
    }
    reader.close();
    if (first) return makeError(ErrorCode::NotFound, "rosbag: no messages on the topic");
    return out;
}

#endif  // CLOUDCROPPER_HAS_MCAP

}  // namespace

bool isRosbagPath(const std::string& path) {
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".db3") return true;
#if defined(CLOUDCROPPER_HAS_MCAP)
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".mcap") return true;
#endif
    std::error_code ec;
    if (fs::is_directory(path, ec))
        for (const auto& e : fs::directory_iterator(path, ec)) {
            if (e.path().extension() == ".db3") return true;
#if defined(CLOUDCROPPER_HAS_MCAP)
            if (e.path().extension() == ".mcap") return true;
#endif
        }
    return false;
}

Result<std::vector<BagTopic>> listBagTopics(const std::string& path) {
    auto bag = resolveBag(path);
    if (!bag) return makeError(bag.error().code, bag.error().message);
#if defined(CLOUDCROPPER_HAS_MCAP)
    if (bag->mcap) return listMcapTopics(bag->path);
#else
    if (bag->mcap)
        return makeError(ErrorCode::Unsupported, "rosbag: .mcap support not built in");
#endif
    const std::string& db3path = bag->path;
    sqlite3*           db      = nullptr;
    if (sqlite3_open_v2(db3path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        std::string e = db ? sqlite3_errmsg(db) : "open failed";
        sqlite3_close(db);
        return makeError(ErrorCode::IoError, "rosbag: " + e);
    }
    std::vector<BagTopic> topics;
    sqlite3_stmt*         st = nullptr;
    if (sqlite3_prepare_v2(db,
                           "SELECT t.name, t.type, COUNT(m.id) FROM topics t "
                           "LEFT JOIN messages m ON m.topic_id=t.id GROUP BY t.id",
                           -1, &st, nullptr) == SQLITE_OK) {
        while (sqlite3_step(st) == SQLITE_ROW) {
            BagTopic bt;
            bt.name  = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
            bt.type  = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
            bt.count = static_cast<std::uint64_t>(sqlite3_column_int64(st, 2));
            topics.push_back(std::move(bt));
        }
    }
    sqlite3_finalize(st);
    sqlite3_close(db);
    return topics;
}

Result<PointCloud> readRosbag(const std::string& path, const BagReadOptions& opt) {
    auto bag = resolveBag(path);
    if (!bag) return makeError(bag.error().code, bag.error().message);
#if defined(CLOUDCROPPER_HAS_MCAP)
    if (bag->mcap) return readMcap(bag->path, opt);
#else
    if (bag->mcap)
        return makeError(ErrorCode::Unsupported, "rosbag: .mcap support not built in");
#endif
    const std::string& db3path = bag->path;
    sqlite3*           db      = nullptr;
    if (sqlite3_open_v2(db3path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        std::string e = db ? sqlite3_errmsg(db) : "open failed";
        sqlite3_close(db);
        return makeError(ErrorCode::IoError, "rosbag: " + e);
    }
    auto fail = [&](ErrorCode code, std::string msg) {
        sqlite3_close(db);
        return makeError(code, std::move(msg));
    };

    // pick the PointCloud2 topic
    std::vector<std::pair<long long, std::string>> pcTopics;
    {
        sqlite3_stmt* st = nullptr;
        sqlite3_prepare_v2(db, "SELECT id,name FROM topics WHERE type='sensor_msgs/msg/PointCloud2'",
                           -1, &st, nullptr);
        while (st && sqlite3_step(st) == SQLITE_ROW)
            pcTopics.emplace_back(sqlite3_column_int64(st, 0),
                                  reinterpret_cast<const char*>(sqlite3_column_text(st, 1)));
        sqlite3_finalize(st);
    }
    if (pcTopics.empty()) return fail(ErrorCode::NotFound, "rosbag: no PointCloud2 topic");
    long long topicId = -1;
    if (!opt.topic.empty()) {
        for (auto& [id, name] : pcTopics)
            if (name == opt.topic) topicId = id;
        if (topicId < 0) return fail(ErrorCode::NotFound, "rosbag: topic not found: " + opt.topic);
    } else if (pcTopics.size() == 1) {
        topicId = pcTopics.front().first;
    } else {
        std::string names;
        for (auto& [id, n] : pcTopics) { (void)id; names += " " + n; }
        return fail(ErrorCode::InvalidArgument, "rosbag: multiple PointCloud2 topics; use --bag-topic (" + names + " )");
    }

    const char* sql = opt.merge
                          ? "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp"
                          : "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1 OFFSET ?";
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
    sqlite3_bind_int64(st, 1, topicId);
    if (!opt.merge) sqlite3_bind_int(st, 2, std::max(0, opt.frame));

    PointCloud out;
    bool       first = true;
    while (st && sqlite3_step(st) == SQLITE_ROW) {
        const void* blob = sqlite3_column_blob(st, 0);
        const int   sz   = sqlite3_column_bytes(st, 0);
        auto msg = parseMessage(static_cast<const std::uint8_t*>(blob), static_cast<std::size_t>(sz));
        if (!msg) {
            const Error e = msg.error();
            sqlite3_finalize(st);
            return fail(e.code, e.message);
        }
        if (first) { out = std::move(msg.value()); first = false; }
        else appendCloud(out, msg.value());
        if (opt.maxPoints && out.size() > opt.maxPoints) {
            sqlite3_finalize(st);
            return fail(ErrorCode::CloudTooLarge,
                        "rosbag: " + std::to_string(out.size()) + " points exceeds the limit");
        }
        if (!opt.merge) break;
    }
    sqlite3_finalize(st);
    sqlite3_close(db);
    if (first) return makeError(ErrorCode::NotFound, "rosbag: no messages on the topic");
    return out;
}

}  // namespace cc::io
