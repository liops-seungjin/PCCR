#include "reg_common.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "cloudcropper/core/analysis.hpp"
#include "grid_index.hpp"  // core internal: NN queries for the shared metric

namespace cc::reg::detail {

std::vector<Eigen::Vector3d> toEigenD(const std::vector<Vec3>& pts) {
    std::vector<Eigen::Vector3d> out;
    out.reserve(pts.size());
    for (const Vec3& p : pts) out.emplace_back(p.x, p.y, p.z);
    return out;
}

std::vector<Eigen::Vector3f> toEigenF(const std::vector<Vec3>& pts) {
    std::vector<Eigen::Vector3f> out;
    out.reserve(pts.size());
    for (const Vec3& p : pts) out.emplace_back(p.x, p.y, p.z);
    return out;
}

std::array<double, 16> fromIsometry(const Eigen::Isometry3d& T) {
    std::array<double, 16> a{};
    const Eigen::Matrix4d  m = T.matrix();
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) a[static_cast<std::size_t>(r * 4 + c)] = m(r, c);
    return a;
}

Eigen::Isometry3d toIsometry(const std::array<double, 16>& T) {
    Eigen::Matrix4d m;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) m(r, c) = T[static_cast<std::size_t>(r * 4 + c)];
    Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
    iso.matrix()          = m;
    return iso;
}

float safeSpacing(const PointCloud& pc) {
    const float s = estimatePointSpacing(pc);
    return std::max(s, 1e-6f);
}

AlignMetric alignmentMetric(const PointCloud& source, const PointCloud& target,
                            const Eigen::Isometry3d& T) {
    AlignMetric m;
    if (source.size() == 0 || target.size() == 0) return m;
    const cc::detail::GridIndex idx(target.positions());
    if (!idx.valid()) return m;
    const float inlierR = 3.0f * safeSpacing(target);

    const std::size_t n      = source.size();
    const std::size_t stride = std::max<std::size_t>(1, n / 50'000);
    std::vector<std::pair<float, std::uint32_t>> nn;
    double      sum2 = 0.0;
    std::size_t cnt = 0, in = 0;
    for (std::size_t i = 0; i < n; i += stride) {
        const Vec3            p = source.positions()[i];
        const Eigen::Vector3d q = T * Eigen::Vector3d(p.x, p.y, p.z);
        idx.kNearest(Vec3{static_cast<float>(q.x()), static_cast<float>(q.y()),
                          static_cast<float>(q.z())},
                     1, nn);
        if (nn.empty()) continue;
        const double d2 = static_cast<double>(nn.front().first);  // squared distance
        sum2 += d2;
        ++cnt;
        if (d2 <= static_cast<double>(inlierR) * inlierR) ++in;
    }
    if (cnt) {
        m.rmse    = std::sqrt(sum2 / static_cast<double>(cnt));
        m.inliers = in * stride;  // scale the sample back up
    }
    return m;
}

}  // namespace cc::reg::detail
