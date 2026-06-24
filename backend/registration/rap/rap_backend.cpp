#include "rap_backend.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>

#include "../gicp/gicp_backend.hpp"
#include "cloudcropper/registration/config.hpp"
#include "cloudcropper/registration/python_worker.hpp"
#if defined(CLOUDCROPPER_HAS_NPZ)
#include "cloudcropper/io/byte_stream.hpp"
#include "cloudcropper/io/npz.hpp"
#endif

namespace fs = std::filesystem;

namespace cc::reg::rap {

namespace {

std::string cfgGet(const std::map<std::string, std::string>& cfg, const char* k,
                   const char* dflt) {
    const auto it = cfg.find(k);
    return it == cfg.end() ? std::string(dflt) : it->second;
}

}  // namespace

#if !defined(CLOUDCROPPER_HAS_NPZ)

Result<RegResult> run(const PointCloud&, const PointCloud&, const RegOptions&) {
    return makeError(ErrorCode::Unsupported,
                     "rap: needs the NPZ codec (vcpkg npz feature) for the handoff");
}

#else

namespace {

// The worker script lives in the repo; search like the config files do.
fs::path findScript() {
    std::error_code ec;
    if (const char* p = std::getenv("CLOUDCROPPER_RAP_SCRIPT")) {
        if (fs::exists(p, ec)) return p;
    }
    const fs::path rel = "backend/registration/rap/python/rap_worker.py";
    if (fs::exists(rel, ec)) return rel;
    fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        fs::path d = exe.parent_path();
        for (int up = 0; up < 6 && !d.empty(); ++up, d = d.parent_path())
            if (fs::exists(d / rel, ec)) return d / rel;
    }
    return {};
}

Result<void> writeNpz(const PointCloud& pc, const fs::path& path) {
    io::FileByteSink sink(path.string());
    if (!sink.ok()) return makeError(ErrorCode::IoError, "rap: cannot create " + path.string());
    io::NpzWriter w;
    return w.write(pc, sink, {});
}

// yaml values are untyped strings; render them as the matching JSON scalar so
// the worker's typed tables receive proper booleans/numbers.
std::string jsonScalar(const std::string& v) {
    if (v == "true" || v == "false") return v;
    char* end = nullptr;
    std::strtod(v.c_str(), &end);
    if (end != v.c_str() && end != nullptr && *end == '\0') return v;  // bare number
    return "\"" + jsonEscape(v) + "\"";
}

// FNV-1a-64 over the raw position bytes: the worker's target-feature cache key
// must change whenever the target geometry does.
std::uint64_t fnv1a64(const void* data, std::size_t n) {
    const auto*   p = static_cast<const unsigned char*>(data);
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

// Declared BEFORE the worker static below, so it is destroyed AFTER the
// worker: worker.log is closed by then and the directory can go away.
struct TempDirGuard {
    fs::path path;
    ~TempDirGuard() {
        std::error_code ec;
        if (!path.empty()) fs::remove_all(path, ec);
    }
};

}  // namespace

Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt) {
    const fs::path script = findScript();
    if (script.empty())
        return makeError(ErrorCode::NotFound, "rap: rap_worker.py not found "
                                              "(set CLOUDCROPPER_RAP_SCRIPT)");

    const auto cfg = configValues("rap.yaml");
    auto       get = [&](const char* k, const char* dflt) { return cfgGet(cfg, k, dflt); };

    // The worker loads its weights ONCE at startup from this env var (it is a
    // persistent process; a model loaded at import can't be retargeted per
    // call). Export it BEFORE the lazy spawn below so the child inherits it.
    if (const std::string wd = get("weights_dir", ""); !wd.empty())
        ::setenv("CLOUDCROPPER_RAP_WEIGHTS", wd.c_str(), /*overwrite=*/1);

    // Temp handoff dir + the persistent worker: both lazy, both live until app
    // exit. NOTE: `python:` config changes need an app restart — the worker is
    // configured once, on its first use.
    static TempDirGuard tmpGuard{
        fs::temp_directory_path() /
        ("cc_rap_" + std::to_string(static_cast<long>(::getpid())))};
    const fs::path& dir = tmpGuard.path;
    std::error_code ec;
    fs::create_directories(dir, ec);

    static PythonWorker worker([&] {
        PythonWorker::Options o;
        o.python  = get("python", "python3");
        o.script  = script.string();
        o.logFile = (dir / "worker.log").string();
        return o;
    }());

    // Handoff files: overwritten per call (calls are serialized by the worker).
    const fs::path srcNpz = dir / "source.npz", tgtNpz = dir / "target.npz";
    if (auto w = writeNpz(source, srcNpz); !w) return makeError(w.error().code, w.error().message);
    if (auto w = writeNpz(target, tgtNpz); !w) return makeError(w.error().code, w.error().message);

    const std::string device     = get("device", "cuda");
    const std::string voxelSize  = opt.rapVoxel > 0 ? std::to_string(opt.rapVoxel)
                                                    : get("voxel_size", "0.0");
    const std::string model      = get("model", "rap_12");
    const auto&       pos        = target.positions();
    // Target-feature cache key: geometry hash + downsample voxel + device + the
    // RAP model variant (a different checkpoint yields different features).
    std::ostringstream key;
    key << std::hex << fnv1a64(pos.data(), pos.size() * sizeof(pos[0])) << std::dec << ":"
        << voxelSize << ":" << device << ":" << model;

    // Explicit keys come from RegOptions (CLI/viewer overrides win over yaml);
    // every other yaml key is forwarded verbatim — the worker's typed tables
    // are the schema, so new knobs need no C++ change. `refine` is always false
    // to the worker: the optional GICP refine is chained in C++ below.
    std::ostringstream params;
    params << "\"source\":\"" << jsonEscape(srcNpz.string()) << "\""
           << ",\"target\":\"" << jsonEscape(tgtNpz.string()) << "\""
           << ",\"target_key\":\"" << key.str() << "\""
           << ",\"device\":\"" << jsonEscape(device) << "\""
           << ",\"voxel_size\":" << voxelSize
           << ",\"refine\":false";
    static const char* kReserved[] = {"python",  "timeout_sec", "weights_dir",
                                      "device",  "voxel_size",  "refine"};
    for (const auto& [k, v] : cfg) {
        bool reserved = false;
        for (const char* r : kReserved) reserved = reserved || k == r;
        if (reserved || v.empty()) continue;
        params << ",\"" << jsonEscape(k) << "\":" << jsonScalar(v);
    }

    // Flow models are heavy, so the default timeout is larger than bufferx's 600.
    int timeoutSec = 900;
    try {
        timeoutSec = std::stoi(get("timeout_sec", "900"));
    } catch (...) {}

    auto resp = worker.call("register", params.str(), timeoutSec);
    if (!resp) return makeError(resp.error().code, resp.error().message);

    const JsonValue* res = resp->find("result");
    const JsonValue* t   = res ? res->find("transform") : nullptr;
    if (!t || t->array.size() != 16)
        return makeError(ErrorCode::ParseError, "rap: worker returned no 4x4 transform");

    RegResult out;
    for (std::size_t i = 0; i < 16; ++i) out.transform[i] = t->array[i].asDouble();
    out.converged = res->find("converged") && res->find("converged")->asBool();
    // RAP does not produce the GPIS confidence / norm-residual signals, so those
    // RegResult fields keep their -1 (not-available) defaults.

    const std::string dev = res->find("device") ? res->find("device")->asString("?") : "?";
    std::ostringstream detail;
    detail << "RAP (worker, " << dev << "): ";
    if (const JsonValue* v = res->find("num_inliers"))
        detail << static_cast<long long>(v->asDouble(-1.0)) << " inliers";
    else detail << "? inliers";
    detail.precision(3);
    if (const JsonValue* v = res->find("fitness")) detail << ", fitness " << v->asDouble(-1.0);
    const bool cached = res->find("cache_hit") && res->find("cache_hit")->asBool();
    if (cached) detail << ", cached target";
    // Surface the worker's fallback note (e.g. core/weights/flash-attn missing)
    // so an identity result is never mistaken for a real alignment in the UI/CSV.
    if (const JsonValue* v = res->find("note"); v && !v->asString().empty())
        detail << " [" << v->asString() << "]";
    out.detail = detail.str();

    // RapGicp: refine the coarse global pose with GICP (small_gicp, already
    // linked in C++) seeded by the RAP 4x4 — keeps the worker inference-only.
    if (opt.algo == RegAlgo::RapGicp && opt.refine) {
        RegOptions ro = opt;
        ro.algo       = RegAlgo::Gicp;
        ro.init       = out.transform;
        auto refined  = gicp::run(source, target, ro);
        if (refined) {
            refined->detail = out.detail + "  ->  " + refined->detail;
            return refined;
        }
    }
    return out;
}

#endif  // CLOUDCROPPER_HAS_NPZ

}  // namespace cc::reg::rap
