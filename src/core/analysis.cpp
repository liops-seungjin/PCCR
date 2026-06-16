#include "cloudcropper/core/analysis.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "eigen3x3.hpp"
#include "grid_index.hpp"

namespace cc {

CanonicalFrame pcaFrame(const PointCloud& pc) {
    CanonicalFrame f;
    const std::vector<Vec3>& pts = pc.positions();
    const std::size_t        n   = pts.size();
    if (n == 0) {
        f.axis = {0, 0, 1};
        return f;
    }

    Vec3 c{};
    for (const Vec3& p : pts) c = c + p;
    c       = c * (1.0f / static_cast<float>(n));
    f.center = c;

    double cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (const Vec3& p : pts) {
        const Vec3 d = p - c;
        cov[0][0] += double(d.x) * d.x;
        cov[0][1] += double(d.x) * d.y;
        cov[0][2] += double(d.x) * d.z;
        cov[1][1] += double(d.y) * d.y;
        cov[1][2] += double(d.y) * d.z;
        cov[2][2] += double(d.z) * d.z;
    }
    const double inv = 1.0 / static_cast<double>(n);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) cov[i][j] *= inv;
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    double w[3], v[3][3];
    detail::jacobiEigen(cov, w, v);

    // largest eigenvalue index -> principal axis
    int lg = 0;
    if (w[1] > w[lg]) lg = 1;
    if (w[2] > w[lg]) lg = 2;
    Vec3        axis{static_cast<float>(v[0][lg]), static_cast<float>(v[1][lg]),
                 static_cast<float>(v[2][lg])};
    const float len = std::sqrt(dot(axis, axis));
    f.axis          = (len < 1e-12f) ? Vec3{0, 0, 1} : axis * (1.0f / len);
    f.extents       = {static_cast<float>(std::sqrt(std::max(0.0, w[0]))),
                 static_cast<float>(std::sqrt(std::max(0.0, w[1]))),
                 static_cast<float>(std::sqrt(std::max(0.0, w[2])))};
    return f;
}

float estimatePointSpacing(const PointCloud& pc) {
    const std::vector<Vec3>& pts = pc.positions();
    const std::size_t        n   = pts.size();
    if (n < 2) return 0.0f;

    const detail::GridIndex grid(pts);
    if (!grid.valid()) return 0.0f;

    // Median 1-NN distance over a bounded sample (stride so cost stays O(sample)).
    const std::size_t target = std::min<std::size_t>(n, 2000);
    const std::size_t stride = std::max<std::size_t>(1, n / target);
    std::vector<float>                           nn;
    std::vector<std::pair<float, std::uint32_t>> cand;
    for (std::size_t i = 0; i < n; i += stride) {
        grid.kNearest(pts[i], 2, cand);  // [0] is self (dist 0)
        if (cand.size() >= 2) nn.push_back(std::sqrt(cand[1].first));
    }
    if (nn.empty()) return grid.spacing();
    std::nth_element(nn.begin(), nn.begin() + nn.size() / 2, nn.end());
    return nn[nn.size() / 2];
}

}  // namespace cc
