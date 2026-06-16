// Dispatcher for the public registration API: routes RegAlgo to the package
// that implements it and applies the shared post-alignment metric.
#include "cloudcropper/registration/registration.hpp"

#include <chrono>

#include "../bufferx/bufferx_backend.hpp"
#include "../gicp/gicp_backend.hpp"
#include "../gradient_sdf_gpu/gsdf_gpu.hpp"
#include "reg_common.hpp"
#if defined(CLOUDCROPPER_HAS_KISS_MATCHER)
#include "../kiss_matcher/kiss_backend.hpp"
#endif

namespace cc::reg {

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
}

}  // namespace cc::reg
