// Dispatcher for the public registration API: routes RegAlgo to the package
// that implements it and applies the shared post-alignment metric.
#include "cloudcropper/registration/registration.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>
#include <sstream>

#include "../bufferx/bufferx_backend.hpp"
#include "../g3reg/g3reg_backend.hpp"
#include "../gicp/gicp_backend.hpp"
#include "../gradient_sdf_gpu/gsdf_gpu.hpp"
#include "../rap/rap_backend.hpp"
#include "reg_common.hpp"
#if defined(CLOUDCROPPER_HAS_KISS_MATCHER)
#include "../kiss_matcher/kiss_backend.hpp"
#endif

namespace cc::reg {

namespace {

std::optional<Eigen::Vector3d> parseMetadataVec3(const std::string& s) {
    std::istringstream iss(s);
    double             x = 0.0, y = 0.0, z = 0.0;
    if (iss >> x >> y >> z) return Eigen::Vector3d{x, y, z};
    return std::nullopt;
}

std::string formatMetadataVec3(const Eigen::Vector3d& v) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%.9g %.9g %.9g", v.x(), v.y(), v.z());
    return buf;
}

void transformPointMetadata(PointCloud& pc, const char* key, const Eigen::Isometry3d& iso) {
    auto it = pc.metadata().find(key);
    if (it == pc.metadata().end()) return;
    if (auto v = parseMetadataVec3(it->second)) it->second = formatMetadataVec3(iso * *v);
}

void transformDirectionMetadata(PointCloud& pc, const char* key, const Eigen::Matrix3d& R) {
    auto it = pc.metadata().find(key);
    if (it == pc.metadata().end()) return;
    if (auto v = parseMetadataVec3(it->second)) {
        Eigen::Vector3d d = R * *v;
        const double    n = d.norm();
        if (n > 1e-12) d /= n;
        it->second = formatMetadataVec3(d);
    }
}

}  // namespace

const char* algoName(RegAlgo a) {
    switch (a) {
        case RegAlgo::Icp: return "ICP";
        case RegAlgo::PlaneIcp: return "Plane-ICP";
        case RegAlgo::Gicp: return "GICP";
        case RegAlgo::VGicp: return "VGICP";
        case RegAlgo::KissMatcher: return "KISS-Matcher";
        case RegAlgo::KissGicp: return "KISS-Matcher + GICP";
        case RegAlgo::GradientSdfGpu: return "gradient-SDF (GPU)";
        case RegAlgo::BufferX: return "BUFFER-X";
        case RegAlgo::BufferXGicp: return "BUFFER-X + GICP";
        case RegAlgo::G3Reg: return "G3Reg";
        case RegAlgo::G3RegGicp: return "G3Reg + GICP";
        case RegAlgo::Rap: return "RAP";
        case RegAlgo::RapGicp: return "RAP + GICP";
    }
    return "?";
}

Result<RegResult> registerClouds(const PointCloud& source, const PointCloud& target,
                                 const RegOptions& opt) {
    if (source.size() < 10 || target.size() < 10)
        return makeError(ErrorCode::InvalidArgument, "registration: clouds too small");

    const auto t0 = std::chrono::steady_clock::now();

    Result<RegResult> r = makeError(ErrorCode::Unsupported, "registration: backend not built");
    switch (opt.algo) {
        case RegAlgo::Icp:
        case RegAlgo::PlaneIcp:
        case RegAlgo::Gicp:
        case RegAlgo::VGicp:
            r = gicp::run(source, target, opt);
            break;
        case RegAlgo::KissMatcher:
        case RegAlgo::KissGicp:
#if defined(CLOUDCROPPER_HAS_KISS_MATCHER)
            r = kiss::run(source, target, opt);
#else
            r = makeError(ErrorCode::Unsupported,
                          "registration: KISS-Matcher backend not built");
#endif
            break;
        case RegAlgo::GradientSdfGpu:
            r = gsdf_gpu::run(source, target, opt);
            break;
        case RegAlgo::BufferX:
        case RegAlgo::BufferXGicp:
            r = bufferx::run(source, target, opt);
            break;
        case RegAlgo::G3Reg:
        case RegAlgo::G3RegGicp:
            r = g3reg::run(source, target, opt);
            break;
        case RegAlgo::Rap:
        case RegAlgo::RapGicp:
            r = rap::run(source, target, opt);
            break;
    }
    if (!r) return r;

    // Backend-independent quality metric so algorithms are comparable.
    const detail::AlignMetric m =
        detail::alignmentMetric(source, target, detail::toIsometry(r->transform));
    r->rmse    = m.rmse;
    r->inliers = m.inliers;
    r->seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

void applyTransform(PointCloud& pc, const std::array<double, 16>& T) {
    const Eigen::Isometry3d iso = detail::toIsometry(T);
    const Eigen::Matrix3d   R   = iso.rotation();
    for (Vec3& p : pc.positions()) {
        const Eigen::Vector3d q = iso * Eigen::Vector3d(p.x, p.y, p.z);
        p = {static_cast<float>(q.x()), static_cast<float>(q.y()),
             static_cast<float>(q.z())};
    }
    if (AttributeColumn* nrm = pc.find(attr::kNormal); nrm && nrm->arity() == 3) {
        auto v = nrm->as<float>();
        for (std::size_t i = 0; i + 2 < v.size(); i += 3) {
            const Eigen::Vector3d n = R * Eigen::Vector3d(v[i], v[i + 1], v[i + 2]);
            v[i]     = static_cast<float>(n.x());
            v[i + 1] = static_cast<float>(n.y());
            v[i + 2] = static_cast<float>(n.z());
        }
    }

    transformPointMetadata(pc, "object_pose_origin_local", iso);
    transformDirectionMetadata(pc, "object_pose_dir_local", R);
    transformPointMetadata(pc, "tailstock_tip_local", iso);
    transformPointMetadata(pc, "bar_axis_point_local", iso);
    transformDirectionMetadata(pc, "bar_axis_dir_local", R);
    transformPointMetadata(pc, "canonical_center", iso);
    transformDirectionMetadata(pc, "canonical_axis", R);

    if (pc.metadata().count("bbox_min") || pc.metadata().count("bbox_max") ||
        pc.metadata().count("bbox_center")) {
        const Aabb b = pc.bounds();
        if (b.valid()) {
            if (pc.metadata().count("bbox_min"))
                pc.metadata()["bbox_min"] =
                    formatMetadataVec3(Eigen::Vector3d{b.min.x, b.min.y, b.min.z});
            if (pc.metadata().count("bbox_max"))
                pc.metadata()["bbox_max"] =
                    formatMetadataVec3(Eigen::Vector3d{b.max.x, b.max.y, b.max.z});
            if (pc.metadata().count("bbox_center")) {
                const Vec3 c = (b.min + b.max) * 0.5f;
                pc.metadata()["bbox_center"] =
                    formatMetadataVec3(Eigen::Vector3d{c.x, c.y, c.z});
            }
        }
    }
}

}  // namespace cc::reg
