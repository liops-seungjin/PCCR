// Internal uniform-grid neighbour index shared by normal estimation and
// denoise. Self-contained (no nanoflann), dimensionality-aware cell sizing so a
// flat/linear cloud doesn't collapse the cell size. Not a public header.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cloudcropper/common/geometry.hpp"

namespace cc::detail {

struct CellKey {
    int  x, y, z;
    bool operator==(const CellKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct CellHash {
    std::size_t operator()(const CellKey& k) const {
        return static_cast<std::size_t>(k.x) * 73856093u ^
               static_cast<std::size_t>(k.y) * 19349663u ^
               static_cast<std::size_t>(k.z) * 83492791u;
    }
};

class GridIndex {
public:
    explicit GridIndex(const std::vector<Vec3>& pts) : pts_(pts) {
        Aabb b;
        for (const Vec3& p : pts) b.expand(p);
        origin_ = b.min;
        const Vec3   ext     = b.max - b.min;
        const float  exts[3] = {ext.x, ext.y, ext.z};
        const double diag    = std::sqrt(double(ext.x) * ext.x + double(ext.y) * ext.y +
                                      double(ext.z) * ext.z);
        if (pts.empty() || diag < 1e-12) {
            valid_ = false;
            return;
        }
        int    effDim = 0;
        double prod   = 1.0;
        for (int a = 0; a < 3; ++a) {
            if (exts[a] > diag * 1e-3) {
                ++effDim;
                prod *= exts[a];
            }
        }
        const double sp = effDim > 0 ? std::pow(prod / double(pts.size()), 1.0 / effDim) : diag * 1e-3;
        spacing_        = static_cast<float>(sp);
        cell_           = std::max(static_cast<float>(sp * 2.0), static_cast<float>(diag * 1e-3));
        cells_.reserve(pts.size());
        for (std::uint32_t i = 0; i < pts.size(); ++i) cells_[keyOf(pts[i])].push_back(i);
        valid_ = true;
    }

    [[nodiscard]] bool  valid() const { return valid_; }
    [[nodiscard]] float spacing() const { return spacing_; }

    // Up to `count` nearest stored points to p, sorted ascending by squared
    // distance. If p is itself a stored point it appears first (distance 0).
    void kNearest(Vec3 p, int count, std::vector<std::pair<float, std::uint32_t>>& out) const {
        out.clear();
        if (!valid_ || count <= 0) return;
        const CellKey kc = keyOf(p);
        int           ring = 0, enough = -1;
        while (true) {
            gatherRing(kc, ring, p, out);
            if (enough < 0 && static_cast<int>(out.size()) >= count) enough = ring;
            if (enough >= 0 && ring >= enough + 1) break;
            if (ring > 256) break;  // safety
            ++ring;
        }
        const int take = std::min<int>(count, static_cast<int>(out.size()));
        std::partial_sort(out.begin(), out.begin() + take, out.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
        out.resize(take);
    }

    // All stored points within Euclidean radius r of p.
    void radiusNeighbors(Vec3 p, float r, std::vector<std::uint32_t>& out) const {
        out.clear();
        if (!valid_) return;
        const float   r2    = r * r;
        const int     reach = std::max(1, static_cast<int>(std::ceil(r / cell_)));
        const CellKey kc    = keyOf(p);
        for (int dx = -reach; dx <= reach; ++dx)
            for (int dy = -reach; dy <= reach; ++dy)
                for (int dz = -reach; dz <= reach; ++dz) {
                    auto it = cells_.find({kc.x + dx, kc.y + dy, kc.z + dz});
                    if (it == cells_.end()) continue;
                    for (std::uint32_t j : it->second) {
                        const Vec3 d = pts_[j] - p;
                        if (dot(d, d) <= r2) out.push_back(j);
                    }
                }
    }

private:
    CellKey keyOf(Vec3 p) const {
        return {static_cast<int>(std::floor((p.x - origin_.x) / cell_)),
                static_cast<int>(std::floor((p.y - origin_.y) / cell_)),
                static_cast<int>(std::floor((p.z - origin_.z) / cell_))};
    }
    void gatherRing(CellKey kc, int ring, Vec3 p,
                    std::vector<std::pair<float, std::uint32_t>>& out) const {
        for (int dx = -ring; dx <= ring; ++dx)
            for (int dy = -ring; dy <= ring; ++dy)
                for (int dz = -ring; dz <= ring; ++dz) {
                    if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != ring) continue;
                    auto it = cells_.find({kc.x + dx, kc.y + dy, kc.z + dz});
                    if (it == cells_.end()) continue;
                    for (std::uint32_t j : it->second) {
                        const Vec3 d = pts_[j] - p;
                        out.emplace_back(dot(d, d), j);
                    }
                }
    }

    const std::vector<Vec3>&                                         pts_;
    Vec3                                                             origin_{};
    float                                                            cell_    = 1.0f;
    float                                                            spacing_ = 0.0f;
    bool                                                             valid_   = false;
    std::unordered_map<CellKey, std::vector<std::uint32_t>, CellHash> cells_;
};

}  // namespace cc::detail
