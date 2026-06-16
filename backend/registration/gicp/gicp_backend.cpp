#include "gicp_backend.hpp"

#include <algorithm>
#include <cstdio>

// small_gicp is not warning-clean under the project flags; isolate the include.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <small_gicp/registration/registration_helper.hpp>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "../common/reg_common.hpp"

namespace cc::reg::gicp {

Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt) {
    const float spacing = detail::safeSpacing(target);

    small_gicp::RegistrationSetting setting;
    switch (opt.algo) {
        case RegAlgo::Icp: setting.type = small_gicp::RegistrationSetting::ICP; break;
        case RegAlgo::PlaneIcp: setting.type = small_gicp::RegistrationSetting::PLANE_ICP; break;
        case RegAlgo::VGicp: setting.type = small_gicp::RegistrationSetting::VGICP; break;
        case RegAlgo::Gicp:
        default: setting.type = small_gicp::RegistrationSetting::GICP; break;
    }
    setting.downsampling_resolution =
        opt.downsample > 0.0f ? static_cast<double>(opt.downsample)
                              : static_cast<double>(2.0f * spacing);
    setting.max_correspondence_distance =
        opt.maxCorr > 0.0f ? static_cast<double>(opt.maxCorr)
                           : static_cast<double>(20.0f * spacing);
    setting.voxel_resolution = setting.downsampling_resolution * 4.0;  // VGICP map
    setting.num_threads      = std::max(1, opt.threads);

    // small_gicp::align(target, source, init) estimates T with
    // p_target = T * p_source — same convention as our public API.
    const auto tgt = detail::toEigenD(target.positions());
    const auto src = detail::toEigenD(source.positions());

    const small_gicp::RegistrationResult res =
        small_gicp::align(tgt, src, detail::toIsometry(opt.init), setting);

    RegResult out;
    out.transform = detail::fromIsometry(Eigen::Isometry3d(res.T_target_source));
    out.converged = res.converged;
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s: %ld iters, %zu inlier pairs, err %.4g",
                  algoName(opt.algo), static_cast<long>(res.iterations),
                  static_cast<std::size_t>(res.num_inliers), res.error);
    out.detail = buf;
    return out;
}

}  // namespace cc::reg::gicp
