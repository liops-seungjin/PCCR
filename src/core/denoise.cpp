#include "cloudcropper/core/denoise.hpp"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "cloudcropper/common/parallel.hpp"
#include "grid_index.hpp"

namespace cc {

PointCloud removeStatisticalOutliers(const PointCloud& pc, const DenoiseParams& params) {
    const std::vector<Vec3>& pts = pc.positions();
    const std::size_t        n   = pts.size();
    if (n < static_cast<std::size_t>(params.k) + 1) return pc;  // too small to judge

    const detail::GridIndex grid(pts);
    if (!grid.valid()) return pc;

    // Per-point mean distance to its k nearest neighbours (excluding self).
    // Parallel: each thread owns its query scratch and writes disjoint meanDist[i].
    std::vector<float> meanDist(n, 0.0f);
    parallelForRange(n, [&](std::size_t begin, std::size_t end) {
        std::vector<std::pair<float, std::uint32_t>> cand;
        for (std::size_t i = begin; i < end; ++i) {
            grid.kNearest(pts[i], params.k + 1, cand);
            double m   = 0.0;
            int    cnt = 0;
            for (std::size_t j = 1; j < cand.size(); ++j) {
                m += std::sqrt(static_cast<double>(cand[j].first));
                ++cnt;
            }
            meanDist[i] = cnt ? static_cast<float>(m / cnt) : 0.0f;
        }
    });
    double sum = 0.0, sum2 = 0.0;
    for (float md : meanDist) {
        sum += md;
        sum2 += static_cast<double>(md) * md;
    }

    const double mu     = sum / static_cast<double>(n);
    const double var    = std::max(0.0, sum2 / static_cast<double>(n) - mu * mu);
    const double sigma  = std::sqrt(var);
    const double thresh = mu + static_cast<double>(params.stdRatio) * sigma;

    std::vector<std::uint32_t> keep;
    keep.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        if (static_cast<double>(meanDist[i]) <= thresh) keep.push_back(i);
    }
    return pc.gather(keep);
}

}  // namespace cc
