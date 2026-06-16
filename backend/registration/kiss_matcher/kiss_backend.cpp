#include "kiss_backend.hpp"

#include <algorithm>
#include <cstdio>

// Vendor headers are not warning-clean under the project flags.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <kiss_matcher/KISSMatcher.hpp>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "../common/reg_common.hpp"
#include "../gicp/gicp_backend.hpp"

namespace cc::reg::kiss {

Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt) {
    // Auto resolution: ~1.5x the target point spacing, with a finer retry when
    // confidence is low. KISS-Matcher's own 0.3 m default assumes outdoor LiDAR;
    // dense/small clouds need it near the spacing (measured: 1-2x spacing
    // recovers a 90deg motion, >=3x degenerates to identity).
    const float spacing = detail::safeSpacing(target);

    const auto src = detail::toEigenF(source.positions());
    const auto tgt = detail::toEigenF(target.positions());

    auto solveAt = [&](float res) {
        kiss_matcher::KISSMatcherConfig cfg(res);
        kiss_matcher::KISSMatcher       matcher(cfg);
        const auto sol     = matcher.estimate(src, tgt);  // tgt ~= R*src + t
        const auto inliers = matcher.getNumFinalInliers();
        return std::pair(sol, inliers);
    };

    float res        = opt.kissResolution > 0.0f ? opt.kissResolution : 1.5f * spacing;
    auto [sol, fin]  = solveAt(res);
    bool  retried    = false;
    if (opt.kissResolution <= 0.0f && fin < 10) {  // auto mode: low confidence -> finer
        res            = 0.75f * spacing;
        std::tie(sol, fin) = solveAt(res);
        retried        = true;
    }

    Eigen::Isometry3d coarse = Eigen::Isometry3d::Identity();
    coarse.linear()          = sol.rotation;
    coarse.translation()     = sol.translation;

    char buf[160];
    std::snprintf(buf, sizeof(buf), "KISS-Matcher: res %.3g%s, %zu final inliers%s", res,
                  retried ? " (retried finer)" : "", fin,
                  sol.valid ? "" : " (low confidence)");

    if (opt.algo == RegAlgo::KissGicp && opt.refine) {
        RegOptions ro = opt;
        ro.algo       = RegAlgo::Gicp;
        ro.init       = detail::fromIsometry(coarse);
        auto refined  = gicp::run(source, target, ro);
        if (refined) {
            refined->detail = std::string(buf) + "  ->  " + refined->detail;
            return refined;
        }
    }

    RegResult out;
    out.transform = detail::fromIsometry(coarse);
    out.converged = sol.valid;
    out.detail    = buf;
    return out;
}

}  // namespace cc::reg::kiss
