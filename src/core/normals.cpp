#include "cloudcropper/core/normals.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "cloudcropper/common/parallel.hpp"
#include "eigen3x3.hpp"
#include "grid_index.hpp"

namespace cc {

void estimateNormals(PointCloud& pc, const NormalParams& params) {
    const std::vector<Vec3>& pts = pc.positions();
    const std::size_t        n   = pts.size();

    AttributeColumn normalCol(std::string(attr::kNormal), AttrType::F32, 3, n);
    auto            nout = normalCol.as<float>();
    auto fallback = [&](std::size_t i) {
        nout[i * 3 + 0] = 0.0f;
        nout[i * 3 + 1] = 0.0f;
        nout[i * 3 + 2] = 1.0f;
    };

    const detail::GridIndex grid(pts);
    if (n < 3 || !grid.valid()) {
        for (std::size_t i = 0; i < n; ++i) fallback(i);
        pc.add(std::move(normalCol));
        return;
    }

    Vec3 centroid{};
    for (const Vec3& p : pts) centroid = centroid + p;
    centroid = centroid * (1.0f / static_cast<float>(n));

    const int   k       = std::min<int>(params.k, static_cast<int>(n) - 1);
    const float radius  = params.radius > 0.0f ? params.radius : 3.0f * std::max(grid.spacing(), 1e-6f);

    // Parallel over points: each thread owns its neighbour scratch, and writes
    // only its own disjoint normal rows, so no locking is needed.
    parallelForRange(n, [&](std::size_t begin, std::size_t end) {
        std::vector<std::pair<float, std::uint32_t>> cand;
        std::vector<std::uint32_t>                   rad;
        std::vector<std::uint32_t>                   nbr;  // neighbour indices (excludes self)
        auto knnInto = [&](const Vec3& p) {
            grid.kNearest(p, k + 1, cand);
            nbr.clear();
            for (std::size_t j = 1; j < cand.size(); ++j) nbr.push_back(cand[j].second);
        };
        for (std::size_t i = begin; i < end; ++i) {
            const Vec3 p = pts[i];
            if (params.search == NormalParams::Search::Knn) {
                knnInto(p);
            } else {
                grid.radiusNeighbors(p, radius, rad);
                nbr.clear();
                for (std::uint32_t j : rad)
                    if (j != i) nbr.push_back(j);
                if (params.search == NormalParams::Search::Hybrid) {
                    if (static_cast<int>(nbr.size()) > k) {  // cap to the k nearest in the radius
                        std::partial_sort(nbr.begin(), nbr.begin() + k, nbr.end(),
                                          [&](std::uint32_t a, std::uint32_t b) {
                                              return dot(pts[a] - p, pts[a] - p) <
                                                     dot(pts[b] - p, pts[b] - p);
                                          });
                        nbr.resize(static_cast<std::size_t>(k));
                    }
                    if (nbr.size() < 4) knnInto(p);  // sparse region: fall back to kNN
                }
            }
            if (nbr.size() < 2) {
                fallback(i);
                continue;
            }

            Vec3 mean{};
            for (std::uint32_t j : nbr) mean = mean + pts[j];
            mean = mean * (1.0f / static_cast<float>(nbr.size()));

            double cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            for (std::uint32_t j : nbr) {
                const Vec3 d = pts[j] - mean;
                cov[0][0] += double(d.x) * d.x;
                cov[0][1] += double(d.x) * d.y;
                cov[0][2] += double(d.x) * d.z;
                cov[1][1] += double(d.y) * d.y;
                cov[1][2] += double(d.y) * d.z;
                cov[2][2] += double(d.z) * d.z;
            }
            cov[1][0] = cov[0][1];
            cov[2][0] = cov[0][2];
            cov[2][1] = cov[1][2];

            double w[3], v[3][3];
            detail::jacobiEigen(cov, w, v);
            int sm = 0;
            if (w[1] < w[sm]) sm = 1;
            if (w[2] < w[sm]) sm = 2;
            Vec3        nrm{static_cast<float>(v[0][sm]), static_cast<float>(v[1][sm]),
                     static_cast<float>(v[2][sm])};
            const float len = std::sqrt(dot(nrm, nrm));
            nrm             = (len < 1e-12f) ? Vec3{0, 0, 1} : nrm * (1.0f / len);

            Vec3 ref{};
            if (params.viewpoint)
                ref = *params.viewpoint - p;
            else if (params.orientOutward)
                ref = p - centroid;
            if (dot(nrm, ref) < 0.0f) nrm = nrm * -1.0f;

            nout[i * 3 + 0] = nrm.x;
            nout[i * 3 + 1] = nrm.y;
            nout[i * 3 + 2] = nrm.z;
        }
    });

    pc.add(std::move(normalCol));
}

NormalStats normalDiagnostics(const PointCloud& pc, const std::optional<Vec3>& viewpoint) {
    NormalStats              st;
    const std::vector<Vec3>& pts = pc.positions();
    const std::size_t        n   = pts.size();
    const AttributeColumn*   nc  = pc.find(attr::kNormal);
    if (n < 3 || !nc || nc->arity() != 3) return st;
    auto nrm = nc->as<float>();
    auto N   = [&](std::size_t i) { return Vec3{nrm[i * 3 + 0], nrm[i * 3 + 1], nrm[i * 3 + 2]}; };

    const detail::GridIndex grid(pts);
    if (!grid.valid()) return st;

    const std::size_t target = std::min<std::size_t>(n, 4000);
    const std::size_t stride = std::max<std::size_t>(1, n / target);
    std::vector<std::pair<float, std::uint32_t>> cand;
    std::size_t                                  samples = 0;
    double                                       flip = 0, flipDen = 0, resid = 0, viol = 0;
    for (std::size_t i = 0; i < n; i += stride) {
        const Vec3 p = pts[i], ni = N(i);
        grid.kNearest(p, 9, cand);  // self + up to 8
        double rsum = 0;
        int    rc   = 0;
        for (std::size_t j = 1; j < cand.size(); ++j) {
            const std::uint32_t idx = cand[j].second;
            if (dot(ni, N(idx)) < 0.0f) flip += 1.0;
            flipDen += 1.0;
            rsum += std::fabs(dot(ni, pts[idx] - p));
            ++rc;
        }
        if (rc) resid += rsum / rc;
        if (viewpoint && dot(ni, *viewpoint - p) < 0.0f) viol += 1.0;
        ++samples;
    }
    if (flipDen > 0) st.flipRate = static_cast<float>(flip / flipDen);
    if (samples > 0) {
        st.meanResidual = static_cast<float>(resid / static_cast<double>(samples));
        if (viewpoint) st.outwardViolation = static_cast<float>(viol / static_cast<double>(samples));
    }
    return st;
}

}  // namespace cc
