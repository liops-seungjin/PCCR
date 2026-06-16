#include "gsdf_gpu.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>

#include "cloudcropper/registration/config.hpp"
#include "cloudcropper/registration/python_worker.hpp"
#if defined(CLOUDCROPPER_HAS_NPZ)
#include "cloudcropper/io/byte_stream.hpp"
#include "cloudcropper/io/npz.hpp"
#endif

namespace fs = std::filesystem;

namespace cc::reg::gsdf_gpu {

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
                     "gsdf-gpu: needs the NPZ codec (vcpkg npz feature) for the handoff");
}

#else

namespace {

// The worker script lives in the repo; search like the config files do.
fs::path findScript() {
    std::error_code ec;
    if (const char* p = std::getenv("CLOUDCROPPER_GSDF_GPU_SCRIPT")) {
        if (fs::exists(p, ec)) return p;
    }
    const fs::path rel = "backend/registration/gradient_sdf_gpu/python/gsdf_worker.py";
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
    if (!sink.ok()) return makeError(ErrorCode::IoError, "gsdf-gpu: cannot create " + path.string());
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

// FNV-1a-64 over the raw position bytes: the worker's SDF-field cache key must
// change whenever the target geometry does.
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
        return makeError(ErrorCode::NotFound, "gsdf-gpu: gsdf_worker.py not found "
                                              "(set CLOUDCROPPER_GSDF_GPU_SCRIPT)");

    const auto cfg = configValues("gradient-sdf-gpu.yaml");
    auto       get = [&](const char* k, const char* dflt) { return cfgGet(cfg, k, dflt); };

    // Temp handoff dir + the persistent worker: both lazy, both live until app
    // exit. NOTE: `python:` config changes need an app restart — the worker is
    // configured once, on its first use.
    static TempDirGuard tmpGuard{
        fs::temp_directory_path() /
        ("cc_gsdf_gpu_" + std::to_string(static_cast<long>(::getpid())))};
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

    const std::string resolution = opt.sdfResolution > 0 ? std::to_string(opt.sdfResolution)
                                                         : get("resolution", "100");
    const std::string poissonDepth = get("poisson_depth", "9");
    const std::string device       = get("device", "cuda");
    const std::string truncMul     = std::to_string(opt.sdfTruncMul);
    const auto&       pos          = target.positions();
    // Field-shaping inputs all participate in the cache key: geometry hash,
    // grid resolution, mesh depth, device, uncertainty toggle, truncation.
    std::ostringstream key;
    key << std::hex << fnv1a64(pos.data(), pos.size() * sizeof(pos[0])) << std::dec << ":"
        << resolution << ":" << poissonDepth << ":" << device
        << ":u" << (opt.sdfUncertainty ? 1 : 0) << ":tm" << truncMul;

    // Explicit keys come from RegOptions (CLI/viewer overrides win over yaml);
    // every other yaml key is forwarded verbatim — the worker's typed tables
    // are the schema, so new knobs need no C++ change.
    std::ostringstream params;
    params << "\"source\":\"" << jsonEscape(srcNpz.string()) << "\""
           << ",\"target\":\"" << jsonEscape(tgtNpz.string()) << "\""
           << ",\"target_key\":\"" << key.str() << "\""
           << ",\"device\":\"" << jsonEscape(device) << "\""
           << ",\"resolution\":" << resolution
           << ",\"poisson_depth\":" << poissonDepth
           << ",\"refine\":" << (opt.refine ? "true" : "false")
           << ",\"uncertainty\":" << (opt.sdfUncertainty ? "true" : "false")
           << ",\"trunc_mul\":" << truncMul;
    static const char* kReserved[] = {"python",     "timeout_sec", "device",
                                      "resolution", "poisson_depth", "refine",
                                      "uncertainty", "trunc_mul"};
    for (const auto& [k, v] : cfg) {
        bool reserved = false;
        for (const char* r : kReserved) reserved = reserved || k == r;
        if (reserved || v.empty()) continue;
        params << ",\"" << jsonEscape(k) << "\":" << jsonScalar(v);
    }

    int timeoutSec = 600;
    try {
        timeoutSec = std::stoi(get("timeout_sec", "600"));
    } catch (...) {}

    auto resp = worker.call("register", params.str(), timeoutSec);
    if (!resp) return makeError(resp.error().code, resp.error().message);

    const JsonValue* res = resp->find("result");
    const JsonValue* t   = res ? res->find("transform") : nullptr;
    if (!t || t->array.size() != 16)
        return makeError(ErrorCode::ParseError, "gsdf-gpu: worker returned no 4x4 transform");

    RegResult out;
    for (std::size_t i = 0; i < 16; ++i) out.transform[i] = t->array[i].asDouble();
    out.converged = res->find("converged") && res->find("converged")->asBool();
    if (const JsonValue* v = res->find("confidence")) out.confidence = v->asDouble(-1.0);
    if (const JsonValue* v = res->find("norm_residual")) out.normResidual = v->asDouble(-1.0);

    const std::string dev = res->find("device") ? res->find("device")->asString("?") : "?";
    std::ostringstream iou;
    iou.precision(3);
    if (const JsonValue* v = res->find("iou")) iou << v->asDouble(-1.0);
    else iou << "?";
    const bool cached = res->find("cache_hit") && res->find("cache_hit")->asBool();
    out.detail = "gradient-SDF (worker, " + dev + "): iou " + iou.str() +
                 (cached ? ", cached field" : "");
    return out;
}

#endif  // CLOUDCROPPER_HAS_NPZ

}  // namespace cc::reg::gsdf_gpu
