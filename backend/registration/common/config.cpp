#include "cloudcropper/registration/config.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

namespace fs = std::filesystem;

namespace cc::reg {
namespace {

// Minimal flat-YAML reader: `key: value` lines, '#' comments, whitespace
// trimmed. Intentionally tiny — the config files are flat by design, so a
// YAML library dependency isn't warranted.
std::map<std::string, std::string> readFlatYaml(const fs::path& file) {
    std::map<std::string, std::string> kv;
    std::ifstream                      in(file);
    std::string                        line;
    auto trim = [](std::string s) {
        const char* ws = " \t\r\n";
        const auto  b  = s.find_first_not_of(ws);
        if (b == std::string::npos) return std::string{};
        return s.substr(b, s.find_last_not_of(ws) - b + 1);
    };
    while (std::getline(in, line)) {
        if (const auto hash = line.find('#'); hash != std::string::npos) line.erase(hash);
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim(line.substr(0, colon));
        const std::string val = trim(line.substr(colon + 1));
        if (!key.empty() && !val.empty()) kv[key] = val;
    }
    return kv;
}

// Locate config/<name>: $CLOUDCROPPER_CONFIG_DIR, ./config/, then config/
// next to or above the executable (covers build/<preset>/src/app layouts).
fs::path findConfig(const std::string& name) {
    std::error_code ec;
    if (const char* dir = std::getenv("CLOUDCROPPER_CONFIG_DIR")) {
        const fs::path p = fs::path(dir) / name;
        if (fs::exists(p, ec)) return p;
    }
    if (const fs::path p = fs::path("config") / name; fs::exists(p, ec)) return p;
    fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        fs::path d = exe.parent_path();
        for (int up = 0; up < 6 && !d.empty(); ++up, d = d.parent_path()) {
            const fs::path p = d / "config" / name;
            if (fs::exists(p, ec)) return p;
        }
    }
    return {};
}

float getF(const std::map<std::string, std::string>& kv, const char* key, float dflt) {
    const auto it = kv.find(key);
    if (it == kv.end()) return dflt;
    try {
        return std::stof(it->second);
    } catch (...) {
        return dflt;
    }
}
int getI(const std::map<std::string, std::string>& kv, const char* key, int dflt) {
    const auto it = kv.find(key);
    if (it == kv.end()) return dflt;
    try {
        return std::stoi(it->second);
    } catch (...) {
        return dflt;
    }
}
bool getB(const std::map<std::string, std::string>& kv, const char* key, bool dflt) {
    const auto it = kv.find(key);
    if (it == kv.end()) return dflt;
    return it->second == "true" || it->second == "1" || it->second == "yes" ||
           it->second == "on";
}

}  // namespace

const char* configFileFor(RegAlgo a) {
    switch (a) {
        case RegAlgo::KissMatcher:
        case RegAlgo::KissGicp: return "kiss-matcher.yaml";
        case RegAlgo::GradientSdfGpu: return "gradient-sdf-gpu.yaml";
        case RegAlgo::BufferX:
        case RegAlgo::BufferXGicp: return "bufferx.yaml";
        case RegAlgo::G3Reg:
        case RegAlgo::G3RegGicp: return "g3reg.yaml";
        case RegAlgo::Rap:
        case RegAlgo::RapGicp: return "rap.yaml";
        default: return "gicp.yaml";
    }
}

std::map<std::string, std::string> configValues(const char* filename) {
    const fs::path file = findConfig(filename);
    if (file.empty()) return {};
    return readFlatYaml(file);
}

RegOptions defaultsFor(RegAlgo algo) {
    RegOptions opt;
    opt.algo = algo;

    const fs::path file = findConfig(configFileFor(algo));
    if (file.empty()) return opt;  // no config found: compiled defaults
    const auto kv = readFlatYaml(file);

    opt.downsample = getF(kv, "downsample", opt.downsample);
    opt.maxCorr    = getF(kv, "max_corr", opt.maxCorr);
    opt.threads    = getI(kv, "threads", opt.threads);
    opt.refine     = getB(kv, "refine", opt.refine);
    switch (algo) {
        case RegAlgo::KissMatcher:
        case RegAlgo::KissGicp:
            opt.kissResolution = getF(kv, "resolution", opt.kissResolution);
            break;
        case RegAlgo::GradientSdfGpu:
            opt.sdfResolution  = getI(kv, "resolution", opt.sdfResolution);
            opt.sdfTruncMul    = getF(kv, "trunc_mul", opt.sdfTruncMul);
            opt.sdfUncertainty = getB(kv, "uncertainty", opt.sdfUncertainty);
            break;
        case RegAlgo::BufferX:
        case RegAlgo::BufferXGicp:
            opt.bufferxVoxel = getF(kv, "voxel_size", opt.bufferxVoxel);
            break;
        case RegAlgo::G3Reg:
        case RegAlgo::G3RegGicp:
            break;  // G3Reg knobs live in the external yaml; refine loaded above
        case RegAlgo::Rap:
        case RegAlgo::RapGicp:
            opt.rapVoxel = getF(kv, "voxel_size", opt.rapVoxel);
            break;
        default: break;
    }
    return opt;
}

}  // namespace cc::reg
