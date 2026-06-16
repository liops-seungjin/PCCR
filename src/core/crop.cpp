#include "cloudcropper/core/crop.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "cloudcropper/common/parallel.hpp"
#include "octree.hpp"

namespace cc {
namespace {

// World-space AABB enclosing an oriented box (its 8 rotated corners).
Aabb obbWorldAabb(const Obb& o) {
    Aabb b;
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2)
            for (int sz = -1; sz <= 1; sz += 2) {
                const Vec3 corner{o.halfExtents.x * static_cast<float>(sx),
                                  o.halfExtents.y * static_cast<float>(sy),
                                  o.halfExtents.z * static_cast<float>(sz)};
                b.expand(o.center + rotate(o.rotation, corner));
            }
    return b;
}

}  // namespace

CropResult crop(const PointCloud& pc, const CropSpec& spec, const IndexPolicy& policy) {
    CropResult result;
    result.totalCount = pc.size();
    result.mask.assign(pc.size(), 0);

    bool hasInclude = false;
    for (const CropBox& b : spec.boxes) {
        if (b.role == BoxRole::Include) {
            hasInclude = true;
            break;
        }
    }

    const std::vector<Vec3>& pts = pc.positions();

    // Per-point membership predicate (identical for the brute-force and octree paths).
    auto testPoint = [&](const Vec3& p) -> bool {
        bool includeHit = !hasInclude;  // no positive filter => everything passes
        if (hasInclude) {
            if (spec.combine == BoolOp::Intersection) {
                includeHit = true;
                for (const CropBox& b : spec.boxes)
                    if (b.role == BoxRole::Include && !b.box.contains(p)) { includeHit = false; break; }
            } else {  // Union
                includeHit = false;
                for (const CropBox& b : spec.boxes)
                    if (b.role == BoxRole::Include && b.box.contains(p)) { includeHit = true; break; }
            }
        }
        bool excludeHit = false;
        for (const CropBox& b : spec.boxes)
            if (b.role == BoxRole::Exclude && b.box.contains(p)) { excludeHit = true; break; }
        return includeHit && !excludeHit;
    };

    // Use the octree only when there's a positive (Include) region to bound the
    // candidates, and the policy's size threshold says it's worth building.
    const std::size_t n = pts.size();
    const bool        useIndex =
        hasInclude && (policy.mode == IndexPolicy::Mode::Always ||
                       (policy.mode == IndexPolicy::Mode::Auto && n > policy.auto_point_threshold));

    if (useIndex) {
        Aabb uni;  // union AABB of the Include boxes (superset of any include hit)
        for (const CropBox& b : spec.boxes)
            if (b.role == BoxRole::Include) {
                const Aabb wb = obbWorldAabb(b.box);
                uni.expand(wb.min);
                uni.expand(wb.max);
            }
        const detail::Octree       oct(pts);
        std::vector<std::uint32_t> cand;
        oct.queryAabb(uni, cand);
        // Disjoint candidate indices -> parallel mask fill without locks.
        parallelForRange(cand.size(), [&](std::size_t b, std::size_t e) {
            for (std::size_t j = b; j < e; ++j)
                if (testPoint(pts[cand[j]])) result.mask[cand[j]] = 1;
        });
    } else {
        parallelForRange(n, [&](std::size_t b, std::size_t e) {
            for (std::size_t i = b; i < e; ++i)
                if (testPoint(pts[i])) result.mask[i] = 1;
        });
    }
    result.inCount =
        static_cast<std::uint64_t>(std::count(result.mask.begin(), result.mask.end(), std::uint8_t{1}));
    return result;
}

std::vector<std::uint32_t> selectedIndices(const CropResult& result) {
    std::vector<std::uint32_t> idx;
    idx.reserve(result.inCount);
    for (std::size_t i = 0; i < result.mask.size(); ++i) {
        if (result.mask[i]) idx.push_back(static_cast<std::uint32_t>(i));
    }
    return idx;
}

PointCloud cropToCloud(const PointCloud& pc, const CropSpec& spec) {
    const CropResult r   = crop(pc, spec);
    PointCloud       out = pc.gather(selectedIndices(r));

    // Recenter: translate so the crop's center lands at the world origin.
    // The offset is the average of all Include box centers (for the common
    // single-include case this is exactly that box's center). With no Include
    // boxes (e.g. exclude-only specs) fall back to the cropped bounds center.
    Vec3        offset{};
    std::size_t includeCount = 0;
    for (const CropBox& b : spec.boxes) {
        if (b.role == BoxRole::Include) {
            offset = offset + b.box.center;
            ++includeCount;
        }
    }
    if (includeCount > 0) {
        offset = offset * (1.0f / static_cast<float>(includeCount));
    } else {
        const Aabb b = out.bounds();
        offset       = b.valid() ? (b.min + b.max) * 0.5f : Vec3{};
    }

    for (Vec3& p : out.positions()) p = p - offset;

    // Record the applied translation so it is traceable/reversible:
    // original coords == stored coords + crop_offset.
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%.9g %.9g %.9g", static_cast<double>(offset.x),
                  static_cast<double>(offset.y), static_cast<double>(offset.z));
    out.metadata()["crop_offset"] = buf;

    return out;
}

}  // namespace cc
