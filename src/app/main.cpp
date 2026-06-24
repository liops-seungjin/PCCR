// CloudCropper CLI — headless slice of the pipeline:
//   load -> (optional AABB crop) -> write
//
// Usage:
//   cloudcropper <input> -o <output> [--aabb x0 y0 z0 x1 y1 z1] [--ascii] [--fields a,b,c]
//   cloudcropper --list-formats
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>  // sysconf for the RAM-derived point limit
#endif

#include "cloudcropper/core/analysis.hpp"
#include "cloudcropper/core/crop.hpp"
#include "cloudcropper/core/denoise.hpp"
#include "cloudcropper/core/normals.hpp"
#include "cloudcropper/io/byte_stream.hpp"
#include "cloudcropper/io/registry.hpp"
#include "cloudcropper/io/rosbag.hpp"  // declarations only; calls guarded by HAS_ROSBAG
#if defined(CLOUDCROPPER_HAS_NPZ)
#include "cloudcropper/io/npz.hpp"  // TemplateMeta + writeTemplateNpz
#endif
#if defined(CLOUDCROPPER_HAS_REGISTRATION)
#include "cloudcropper/registration/config.hpp"
#include "cloudcropper/registration/registration.hpp"
#endif

#if defined(CLOUDCROPPER_HAS_GUI)
#include "cloudcropper/viewer/viewer.hpp"
#endif
#if defined(CLOUDCROPPER_HAS_GZIP)
#include "cloudcropper/transport/gzip.hpp"
#endif

namespace {

std::string extOf(const std::string& path) {
    const auto dot = path.find_last_of('.');
    return dot == std::string::npos ? std::string{} : path.substr(dot);
}

bool endsWith(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}
bool wantsGz(const std::string& p) { return endsWith(p, ".gz"); }
// Format-dispatch extension, peeling a trailing .gz ("cloud.ply.gz" -> ".ply").
std::string fmtExt(const std::string& p) {
    return extOf(wantsGz(p) ? p.substr(0, p.size() - 3) : p);
}

std::vector<std::string> splitCsv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream        ss(s);
    std::string              item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

std::optional<cc::Vec3> parseVec3(const std::string& s) {
    const auto v = splitCsv(s);
    if (v.size() != 3) return std::nullopt;
    return cc::Vec3{std::stof(v[0]), std::stof(v[1]), std::stof(v[2])};
}

// Space-separated "x y z" (as stored in metadata).
std::optional<cc::Vec3> parseSpaceVec3(const std::string& s) {
    std::istringstream iss(s);
    float              x, y, z;
    if (iss >> x >> y >> z) return cc::Vec3{x, y, z};
    return std::nullopt;
}

bool normalizeVec3(cc::Vec3& v) {
    const float n2 = cc::dot(v, v);
    if (n2 <= 1e-12f) return false;
    v = v * (1.0f / std::sqrt(n2));
    return true;
}

// A generous default point cap (~60% of RAM at ~32 bytes/point) so absurd loads
// fail loud instead of OOM; 0 (unlimited) if the platform can't report RAM.
std::uint64_t defaultMaxPoints() {
#if defined(__unix__) || defined(__APPLE__)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long psz   = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && psz > 0) {
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(psz);
        return (bytes * 6 / 10) / 32;
    }
#endif
    return 0;
}

int usage() {
    std::cerr << "usage: cloudcropper <input> -o <output> [--aabb x0 y0 z0 x1 y1 z1]"
                 " [--ascii|--compressed] [--fields a,b,c]\n"
                 "                   [--denoise] [--denoise-k N] [--denoise-std R]\n"
                 "                   [--estimate-normals] [--normal-k N]"
                 " [--normal-search knn|radius|hybrid] [--normal-radius R] [--viewpoint x,y,z]\n"
                 "                   [--export-template] [--units m] [--tip x,y,z]"
                 " [--bar-point x,y,z] [--bar-dir x,y,z]"
                 " [--pose-origin x,y,z --pose-dir x,y,z]\n"
                 "       (input may be a ROS2 bag .db3/.mcap / bag dir: [--bag-topic T]"
                 " [--bag-merge] [--bag-frame N])\n"
                 "                   [--max-points N]\n"
                 "       (input/output may end in .gz for transparent gzip)\n"
                 "       cloudcropper --bag-info <bag.db3|bag.mcap|bagdir>\n"
                 "       cloudcropper register <source> <target> [--reg-algo gicp|...]"
                 " [-o aligned.ply]\n"
                 "       cloudcropper view [input] [--screenshot file.png] [--frames N]"
                 " [--point-size S]\n"
                 "         (no input opens an empty window; load via drag-drop / Open / path)\n"
                 "       cloudcropper --list-formats\n";
    return 2;
}

// Loads a point cloud from `input`: a ROS2 bag (.db3/.mcap) or a point-cloud
// file (magic then extension; peels a trailing .gz).
cc::Result<cc::PointCloud> loadCloud(cc::io::FormatRegistry& registry, const std::string& input,
                                     const cc::io::BagReadOptions& bagOpt = {}) {
#if defined(CLOUDCROPPER_HAS_ROSBAG)
    if (cc::io::isRosbagPath(input)) return cc::io::readRosbag(input, bagOpt);
#else
    (void)bagOpt;
    if (input.size() >= 4 && input.substr(input.size() - 4) == ".db3")
        return cc::makeError(cc::ErrorCode::Unsupported, ".db3 (ROS2 bag) needs the rosbag build");
#endif
    cc::io::FileByteSource src(input);
    if (!src.ok()) return cc::makeError(cc::ErrorCode::IoError, "cannot open " + input);

    std::vector<std::byte> all = cc::io::readAll(src);
    if (wantsGz(input)) {
#if defined(CLOUDCROPPER_HAS_GZIP)
        auto dec = cc::transport::gzipDecompress(all);
        if (!dec) return cc::makeError(cc::ErrorCode::ParseError, "gunzip failed: " + dec.error().message);
        all = std::move(dec.value());
#else
        return cc::makeError(cc::ErrorCode::Unsupported, ".gz input needs the gzip build");
#endif
    }
    std::span<const std::byte> magic(all.data(), std::min<std::size_t>(all.size(), 16));
    auto                       reader = registry.readerFor(fmtExt(input), magic);
    if (!reader) return cc::makeError(cc::ErrorCode::Unsupported, "no reader for " + input);

    cc::io::MemoryByteSource memsrc(all);
    cc::io::ReadOptions      ro;
    ro.maxPoints = bagOpt.maxPoints;  // share the limit between bag and file reads
    auto loaded  = reader->read(memsrc, ro);
    if (!loaded) return cc::makeError(cc::ErrorCode::ParseError, loaded.error().message);
    return std::move(loaded.value());
}

// `cloudcropper view [input] ...` — the interactive viewer subcommand. The file
// is optional: with none, the viewer opens empty and loads via drag-and-drop /
// Open… / the path box.
int runView(cc::io::FormatRegistry& registry, const std::vector<std::string>& args) {
#if defined(CLOUDCROPPER_HAS_GUI)
    std::string input;
    std::string screenshot;
    int         frames    = 0;
    float       pointSize = 2.0f;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--screenshot" && i + 1 < args.size())
            screenshot = args[++i];
        else if (a == "--frames" && i + 1 < args.size())
            frames = std::stoi(args[++i]);
        else if (a == "--point-size" && i + 1 < args.size())
            pointSize = std::stof(args[++i]);
        else if (!a.empty() && a[0] != '-')
            input = a;
        else {
            std::cerr << "view: unknown option " << a << "\n";
            return 2;
        }
    }

    cc::viewer::ViewerOptions opt;
    opt.title       = input.empty() ? "CloudCropper" : input;
    opt.frames      = frames;
    opt.screenshot  = screenshot;
    opt.pointSize   = pointSize;
    opt.initialPath = input;
    auto r = cc::viewer::runViewer(
        registry, opt,
        [&registry](const std::string& p) { return loadCloud(registry, p); });
    if (!r) {
        std::cerr << "viewer error: " << r.error().message << "\n";
        return 1;
    }
    return 0;
#else
    (void)registry;
    (void)args;
    std::cerr << "error: this build has no GUI. Reconfigure with the 'gui' preset:\n"
                 "  cmake --preset gui && cmake --build --preset gui\n";
    return 1;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    cc::io::FormatRegistry registry;
    cc::io::registerBuiltinFormats(registry);

    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) return usage();

    if (args[0] == "view") {
        return runView(registry, std::vector<std::string>(args.begin() + 1, args.end()));
    }

    if (args[0] == "--list-formats") {
        for (const auto& f : registry.available()) {
            std::cout << f.id << (f.can_read ? " r" : "") << (f.can_write ? "w" : "") << " [";
            for (const auto& e : f.extensions) std::cout << e << ' ';
            std::cout << "]\n";
        }
        return 0;
    }

    if (args[0] == "--bag-info") {
#if defined(CLOUDCROPPER_HAS_ROSBAG)
        if (args.size() < 2) {
            std::cerr << "usage: cloudcropper --bag-info <bag.db3|bagdir>\n";
            return 2;
        }
        auto topics = cc::io::listBagTopics(args[1]);
        if (!topics) {
            std::cerr << "error: " << topics.error().message << "\n";
            return 1;
        }
        for (const auto& t : topics.value())
            std::cout << t.name << "  [" << t.type << "]  " << t.count << " msgs\n";
        return 0;
#else
        std::cerr << "error: --bag-info needs the rosbag build (vcpkg/gui preset)\n";
        return 1;
#endif
    }

    if (args[0] == "register") {
#if defined(CLOUDCROPPER_HAS_REGISTRATION)
        if (args.size() < 3) {
            std::cerr << "usage: cloudcropper register <source> <target>"
                         " [--reg-algo icp|icp-plane|gicp|vgicp|kiss|kiss-gicp|gsdf|gsdf-gpu|bufferx|bufferx-gicp|g3reg|g3reg-gicp|rap|rap-gicp]\n"
                         "         [--reg-downsample S] [--reg-max-corr D] [--reg-threads N]"
                         " [--reg-kiss-res R] [--reg-bufferx-voxel V] [--reg-rap-voxel V] [--reg-no-refine]\n"
                         "         [--reg-uncertainty|--reg-no-uncertainty] [-o aligned.ply]\n"
                         "         (bag source/target: [--bag-topic T] [--bag-merge]"
                         " [--bag-frame N])\n";
            return 2;
        }
        const std::string srcPath = args[1], tgtPath = args[2];
        // First resolve the algorithm, then apply the package's config-file
        // defaults (config/<pkg>.yaml); the remaining flags override those.
        auto parseAlgo = [](const std::string& v, cc::reg::RegAlgo& out) {
            if (v == "icp") out = cc::reg::RegAlgo::Icp;
            else if (v == "icp-plane") out = cc::reg::RegAlgo::PlaneIcp;
            else if (v == "gicp") out = cc::reg::RegAlgo::Gicp;
            else if (v == "vgicp") out = cc::reg::RegAlgo::VGicp;
            else if (v == "kiss") out = cc::reg::RegAlgo::KissMatcher;
            else if (v == "kiss-gicp") out = cc::reg::RegAlgo::KissGicp;
            // "gsdf" is an alias: only the Python-worker gradient-SDF remains.
            else if (v == "gsdf" || v == "gsdf-gpu") out = cc::reg::RegAlgo::GradientSdfGpu;
            else if (v == "bufferx") out = cc::reg::RegAlgo::BufferX;
            else if (v == "bufferx-gicp") out = cc::reg::RegAlgo::BufferXGicp;
            else if (v == "g3reg") out = cc::reg::RegAlgo::G3Reg;
            else if (v == "g3reg-gicp") out = cc::reg::RegAlgo::G3RegGicp;
            else if (v == "rap") out = cc::reg::RegAlgo::Rap;
            else if (v == "rap-gicp") out = cc::reg::RegAlgo::RapGicp;
            else return false;
            return true;
        };
        cc::reg::RegAlgo algo = cc::reg::RegAlgo::Gicp;
        for (std::size_t i = 3; i + 1 < args.size(); ++i)
            if (args[i] == "--reg-algo" && !parseAlgo(args[i + 1], algo)) {
                std::cerr << "error: unknown --reg-algo " << args[i + 1] << "\n";
                return 2;
            }
        cc::reg::RegOptions    ro = cc::reg::defaultsFor(algo);
        std::string            outPath;
        cc::io::BagReadOptions bagOpt;
        for (std::size_t i = 3; i < args.size(); ++i) {
            const std::string& a = args[i];
            auto next = [&]() -> std::string {
                return (i + 1 < args.size()) ? args[++i] : std::string{};
            };
            if (a == "--reg-algo") {
                next();  // already applied in the pre-scan
            } else if (a == "--reg-downsample") ro.downsample = std::stof(next());
            else if (a == "--reg-max-corr") ro.maxCorr = std::stof(next());
            else if (a == "--reg-threads") ro.threads = std::stoi(next());
            else if (a == "--reg-kiss-res") ro.kissResolution = std::stof(next());
            else if (a == "--reg-bufferx-voxel") ro.bufferxVoxel = std::stof(next());
            else if (a == "--reg-rap-voxel") ro.rapVoxel = std::stof(next());
            else if (a == "--reg-no-refine") ro.refine = false;
            else if (a == "--reg-uncertainty") ro.sdfUncertainty = true;
            else if (a == "--reg-no-uncertainty") ro.sdfUncertainty = false;
            else if (a == "--bag-topic") bagOpt.topic = next();
            else if (a == "--bag-merge") bagOpt.merge = true;
            else if (a == "--bag-frame") bagOpt.frame = std::stoi(next());
            else if (a == "-o") outPath = next();
            else { std::cerr << "error: unknown register option " << a << "\n"; return 2; }
        }
        auto src = loadCloud(registry, srcPath, bagOpt);
        if (!src) { std::cerr << "error: source: " << src.error().message << "\n"; return 1; }
        auto tgt = loadCloud(registry, tgtPath, bagOpt);
        if (!tgt) { std::cerr << "error: target: " << tgt.error().message << "\n"; return 1; }
        std::cerr << "register: " << cc::reg::algoName(ro.algo) << "  source "
                  << src->size() << " pts -> target " << tgt->size() << " pts\n";

        auto rr = cc::reg::registerClouds(src.value(), tgt.value(), ro);
        if (!rr) { std::cerr << "error: " << rr.error().message << "\n"; return 1; }

        std::printf("transform (target <- source, row-major):\n");
        for (int r = 0; r < 4; ++r)
            std::printf("  [% .6f % .6f % .6f % .6f]\n", rr->transform[r * 4 + 0],
                        rr->transform[r * 4 + 1], rr->transform[r * 4 + 2],
                        rr->transform[r * 4 + 3]);
        std::printf("converged: %s   rmse: %.6g   inliers: %zu   time: %.2fs\n",
                    rr->converged ? "yes" : "no", rr->rmse, rr->inliers, rr->seconds);
        if (rr->confidence >= 0.0)
            std::printf("confidence: %.3f   norm-residual: %.3g\n", rr->confidence,
                        rr->normResidual);
        if (!rr->detail.empty()) std::printf("%s\n", rr->detail.c_str());

        if (!outPath.empty()) {
            cc::PointCloud aligned = src.value();
            cc::reg::applyTransform(aligned, rr->transform);
            auto w = registry.writerForExt(fmtExt(outPath));
            if (!w) { std::cerr << "error: no writer for " << outPath << "\n"; return 1; }
            cc::io::FileByteSink sink(outPath);
            if (!sink.ok()) { std::cerr << "error: cannot create " << outPath << "\n"; return 1; }
            auto wr = w->write(aligned, sink, {});
            if (!wr) { std::cerr << "error: write failed: " << wr.error().message << "\n"; return 1; }
            std::cerr << "wrote aligned source (" << aligned.size() << " pts) to " << outPath << "\n";
        }
        return rr->converged ? 0 : 1;
#else
        std::cerr << "error: register needs the registration build (vcpkg/gui preset)\n";
        return 1;
#endif
    }

    std::string              input, output;
    bool                     ascii = false;
    bool                     compressed = false;
    bool                     doCrop = false;
    bool                     estNormals = false;
    int                      normalK = 30;
    std::string              normalSearch = "knn";
    float                    normalRadius = 0.0f;
    std::optional<cc::Vec3>  viewpoint;
    bool                     exportTemplate = false;
    std::optional<cc::Vec3>  tip, barPoint, barDir, poseOrigin, poseDir;
    std::string              units = "m";
    std::uint64_t            maxPoints = defaultMaxPoints();
    bool                     denoise = false;
    int                      denoiseK = 16;
    float                    denoiseStd = 2.0f;
    cc::io::BagReadOptions   bagOpt;
    cc::Aabb                 box;
    std::vector<std::string> fields;

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "-o" && i + 1 < args.size()) {
            output = args[++i];
        } else if (a == "--ascii") {
            ascii = true;
        } else if (a == "--compressed") {
            compressed = true;
        } else if (a == "--denoise") {
            denoise = true;
        } else if (a == "--denoise-k" && i + 1 < args.size()) {
            denoiseK = std::stoi(args[++i]);
        } else if (a == "--denoise-std" && i + 1 < args.size()) {
            denoiseStd = std::stof(args[++i]);
        } else if (a == "--bag-topic" && i + 1 < args.size()) {
            bagOpt.topic = args[++i];
        } else if (a == "--bag-merge") {
            bagOpt.merge = true;
        } else if (a == "--bag-frame" && i + 1 < args.size()) {
            bagOpt.frame = std::stoi(args[++i]);
        } else if (a == "--estimate-normals") {
            estNormals = true;
        } else if (a == "--normal-k" && i + 1 < args.size()) {
            normalK = std::stoi(args[++i]);
        } else if (a == "--normal-search" && i + 1 < args.size()) {
            normalSearch = args[++i];
        } else if (a == "--normal-radius" && i + 1 < args.size()) {
            normalRadius = std::stof(args[++i]);
        } else if (a == "--viewpoint" && i + 1 < args.size()) {
            viewpoint = parseVec3(args[++i]);
        } else if (a == "--export-template") {
            exportTemplate = true;
        } else if (a == "--tip" && i + 1 < args.size()) {
            tip = parseVec3(args[++i]);
        } else if (a == "--bar-point" && i + 1 < args.size()) {
            barPoint = parseVec3(args[++i]);
        } else if (a == "--bar-dir" && i + 1 < args.size()) {
            barDir = parseVec3(args[++i]);
            if (barDir) normalizeVec3(*barDir);
        } else if (a == "--pose-origin" && i + 1 < args.size()) {
            poseOrigin = parseVec3(args[++i]);
            if (!poseOrigin) {
                std::cerr << "error: --pose-origin expects x,y,z\n";
                return 2;
            }
        } else if (a == "--pose-dir" && i + 1 < args.size()) {
            poseDir = parseVec3(args[++i]);
            if (!poseDir || !normalizeVec3(*poseDir)) {
                std::cerr << "error: --pose-dir expects a non-zero x,y,z vector\n";
                return 2;
            }
        } else if (a == "--units" && i + 1 < args.size()) {
            units = args[++i];
        } else if (a == "--max-points" && i + 1 < args.size()) {
            maxPoints = std::stoull(args[++i]);
        } else if (a == "--fields" && i + 1 < args.size()) {
            fields = splitCsv(args[++i]);
        } else if (a == "--aabb" && i + 6 < args.size()) {
            box.min = {std::stof(args[i + 1]), std::stof(args[i + 2]), std::stof(args[i + 3])};
            box.max = {std::stof(args[i + 4]), std::stof(args[i + 5]), std::stof(args[i + 6])};
            i += 6;
            doCrop = true;
        } else if (!a.empty() && a[0] != '-') {
            input = a;
        } else {
            return usage();
        }
    }

    if (input.empty() || output.empty()) return usage();
    if (static_cast<bool>(poseOrigin) != static_cast<bool>(poseDir)) {
        std::cerr << "error: --pose-origin and --pose-dir must be provided together\n";
        return 2;
    }

    // --- load ---
    bagOpt.maxPoints = maxPoints;  // applies to both file and bag readers
    auto loaded      = loadCloud(registry, input, bagOpt);
    if (!loaded) {
        std::cerr << "error: " << loaded.error().message << "\n";
        return 1;
    }
    cc::PointCloud cloud = std::move(loaded.value());
    std::cerr << "loaded " << cloud.size() << " points from " << input << "\n";
    if (auto u = cloud.metadata().find("units");
        u != cloud.metadata().end() && u->second != "m")
        std::cerr << "warning: input units = '" << u->second << "' (expected meters)\n";

    // --- optional crop ---
    if (doCrop) {
        cc::CropSpec spec;
        spec.boxes.push_back({cc::Obb::fromAabb(box), cc::BoxRole::Include});
        cloud = cc::cropToCloud(cloud, spec);
        std::cerr << "cropped to " << cloud.size() << " points\n";
    }

    // --- denoise: statistical outlier removal (before normals) ---
    if (denoise) {
        cc::DenoiseParams dp;
        dp.k             = denoiseK;
        dp.stdRatio      = denoiseStd;
        const std::size_t before = cloud.size();
        cloud                    = cc::removeStatisticalOutliers(cloud, dp);
        std::cerr << "denoised " << before << " -> " << cloud.size() << " points (k=" << dp.k
                  << ", std=" << dp.stdRatio << ")\n";
    }

    // Build NormalParams from the flags, auto-filling the viewpoint from metadata
    // (e.g. a PCD VIEWPOINT) when none was given on the command line.
    auto buildNP = [&]() {
        cc::NormalParams np;
        np.k      = normalK;
        np.radius = normalRadius;
        np.search = (normalSearch == "radius")   ? cc::NormalParams::Search::Radius
                    : (normalSearch == "hybrid") ? cc::NormalParams::Search::Hybrid
                                                 : cc::NormalParams::Search::Knn;
        np.viewpoint = viewpoint;
        if (!np.viewpoint) {
            auto it = cloud.metadata().find("viewpoint");
            if (it != cloud.metadata().end()) np.viewpoint = parseSpaceVec3(it->second);
        }
        return np;
    };
    auto runNormals = [&](const char* tag) {
        cc::NormalParams np = buildNP();
        cc::estimateNormals(cloud, np);
        const cc::NormalStats d = cc::normalDiagnostics(cloud, np.viewpoint);
        std::cerr << tag << " (" << normalSearch << ", k=" << np.k << ")"
                  << (np.viewpoint ? " viewpoint-oriented" : " outward-oriented")
                  << "  flip=" << d.flipRate << " residual=" << d.meanResidual;
        if (np.viewpoint) std::cerr << " outward-viol=" << d.outwardViolation;
        std::cerr << "\n";
    };

    // --- normals: estimate (only if absent) so the export carries normal_x/y/z ---
    if (estNormals) {
        if (cloud.has(cc::attr::kNormal))
            std::cerr << "normals already present; skipping estimation\n";
        else
            runNormals("estimated normals");
    }

    // Carry the unit convention so writers (template / future PLY comments) emit it.
    cloud.metadata()["units"] = units;

    // --- template export: meters .npz with surface_points/normals + metadata ---
    if (exportTemplate) {
#if defined(CLOUDCROPPER_HAS_NPZ)
        if (!cloud.has(cc::attr::kNormal)) runNormals("estimated normals for template");
        cc::io::TemplateMeta tm;
        const cc::Aabb       b = cloud.bounds();
        tm.bbox_min            = b.min;
        tm.bbox_max            = b.max;
        tm.bbox_center         = (b.min + b.max) * 0.5f;
        const cc::CanonicalFrame cf = cc::pcaFrame(cloud);
        tm.canonical_center         = cf.center;
        tm.canonical_axis           = cf.axis;
        tm.point_spacing_m          = cc::estimatePointSpacing(cloud);
        tm.tip_local                = tip;
        tm.bar_point_local          = barPoint;
        tm.bar_dir_local            = barDir;
        tm.object_pose_origin_local = poseOrigin;
        tm.object_pose_dir_local    = poseDir;
        tm.units                    = units;
        cc::io::FileByteSink sink(output);
        if (!sink.ok()) {
            std::cerr << "error: cannot create " << output << "\n";
            return 1;
        }
        auto wr = cc::io::writeTemplateNpz(cloud, tm, sink);
        if (!wr) {
            std::cerr << "error: template write failed: " << wr.error().message << "\n";
            return 1;
        }
        std::cerr << "wrote template " << cloud.size() << " pts (spacing "
                  << tm.point_spacing_m << " m) to " << output << "\n";
        return 0;
#else
        std::cerr << "error: --export-template needs the npz build (vcpkg/gui preset)\n";
        return 1;
#endif
    }

    // --- write (transparently gzip-wraps when the path ends in .gz) ---
    auto writer = registry.writerForExt(fmtExt(output));
    if (!writer) {
        std::cerr << "error: no writer for " << output << "\n";
        return 1;
    }
    cc::io::WriteOptions opt;
    opt.fields   = fields;
    opt.encoding = ascii ? cc::io::Encoding::Ascii
                 : compressed ? cc::io::Encoding::BinaryCompressed
                              : cc::io::Encoding::Binary;

    if (wantsGz(output)) {
#if defined(CLOUDCROPPER_HAS_GZIP)
        cc::io::MemoryByteSink msink;
        auto                   wr = writer->write(cloud, msink, opt);
        if (!wr) {
            std::cerr << "error: write failed: " << wr.error().message << "\n";
            return 1;
        }
        auto comp = cc::transport::gzipCompress(msink.data());
        if (!comp) {
            std::cerr << "error: gzip failed: " << comp.error().message << "\n";
            return 1;
        }
        cc::io::FileByteSink fsink(output);
        if (!fsink.ok()) {
            std::cerr << "error: cannot create " << output << "\n";
            return 1;
        }
        fsink.write(comp.value());
#else
        std::cerr << "error: .gz output needs the gzip build (vcpkg/gui preset)\n";
        return 1;
#endif
    } else {
        cc::io::FileByteSink sink(output);
        if (!sink.ok()) {
            std::cerr << "error: cannot create " << output << "\n";
            return 1;
        }
        auto wr = writer->write(cloud, sink, opt);
        if (!wr) {
            std::cerr << "error: write failed: " << wr.error().message << "\n";
            return 1;
        }
    }
    std::cerr << "wrote " << cloud.size() << " points to " << output << "\n";
    return 0;
}
