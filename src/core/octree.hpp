// Internal bounded octree over point positions for AABB range queries (the crop
// candidate-gathering path). Header-only, like grid_index.hpp. queryAabb returns
// a superset of indices whose octants overlap the box; the caller exact-tests.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "cloudcropper/common/geometry.hpp"

namespace cc::detail {

class Octree {
public:
    explicit Octree(const std::vector<Vec3>& pts, int leafCap = 32, int maxDepth = 16)
        : pts_(pts), leafCap_(leafCap), maxDepth_(maxDepth) {
        if (pts.empty()) return;
        Aabb b;
        for (const Vec3& p : pts) b.expand(p);
        std::vector<std::uint32_t> all(pts.size());
        for (std::uint32_t i = 0; i < pts.size(); ++i) all[i] = i;
        nodes_.reserve(pts.size() / static_cast<std::size_t>(std::max(1, leafCap_)) + 8);
        build(b, std::move(all), 0);
    }

    [[nodiscard]] bool valid() const { return !nodes_.empty(); }

    void queryAabb(const Aabb& q, std::vector<std::uint32_t>& out) const {
        out.clear();
        if (!nodes_.empty()) recurse(0, q, out);
    }

private:
    struct Node {
        Aabb                       box;
        std::array<int, 8>         child{-1, -1, -1, -1, -1, -1, -1, -1};
        std::vector<std::uint32_t> idx;
        bool                       leaf = false;
    };

    static bool overlap(const Aabb& a, const Aabb& b) {
        return a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y &&
               a.max.y >= b.min.y && a.min.z <= b.max.z && a.max.z >= b.min.z;
    }

    int build(const Aabb& box, std::vector<std::uint32_t> idx, int depth) {
        const int id = static_cast<int>(nodes_.size());
        nodes_.push_back(Node{});
        nodes_[id].box = box;
        if (static_cast<int>(idx.size()) <= leafCap_ || depth >= maxDepth_) {
            nodes_[id].leaf = true;
            nodes_[id].idx  = std::move(idx);
            return id;
        }
        const Vec3 c = (box.min + box.max) * 0.5f;
        std::array<std::vector<std::uint32_t>, 8> oct;
        for (std::uint32_t i : idx) {
            const Vec3 p = pts_[i];
            const int  o = (p.x > c.x ? 1 : 0) | (p.y > c.y ? 2 : 0) | (p.z > c.z ? 4 : 0);
            oct[static_cast<std::size_t>(o)].push_back(i);
        }
        std::array<int, 8> childIds{-1, -1, -1, -1, -1, -1, -1, -1};
        for (int k = 0; k < 8; ++k) {
            if (oct[static_cast<std::size_t>(k)].empty()) continue;
            Aabb cb;
            cb.min.x = (k & 1) ? c.x : box.min.x;
            cb.max.x = (k & 1) ? box.max.x : c.x;
            cb.min.y = (k & 2) ? c.y : box.min.y;
            cb.max.y = (k & 2) ? box.max.y : c.y;
            cb.min.z = (k & 4) ? c.z : box.min.z;
            cb.max.z = (k & 4) ? box.max.z : c.z;
            childIds[static_cast<std::size_t>(k)] =
                build(cb, std::move(oct[static_cast<std::size_t>(k)]), depth + 1);
        }
        nodes_[id].child = childIds;  // re-index after recursion (vector may have grown)
        nodes_[id].leaf  = false;
        return id;
    }

    void recurse(int id, const Aabb& q, std::vector<std::uint32_t>& out) const {
        const Node& nd = nodes_[static_cast<std::size_t>(id)];
        if (!overlap(nd.box, q)) return;
        if (nd.leaf) {
            out.insert(out.end(), nd.idx.begin(), nd.idx.end());
        } else {
            for (int k = 0; k < 8; ++k)
                if (nd.child[static_cast<std::size_t>(k)] >= 0)
                    recurse(nd.child[static_cast<std::size_t>(k)], q, out);
        }
    }

    const std::vector<Vec3>& pts_;
    std::vector<Node>        nodes_;
    int                      leafCap_  = 32;
    int                      maxDepth_ = 16;
};

}  // namespace cc::detail
