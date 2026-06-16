// Crop engine (docs/design/03 §3): box membership over a PointCloud.
// First slice is brute-force; an octree path slots in behind the same API later.
#pragma once

#include <cstdint>
#include <vector>

#include "cloudcropper/common/geometry.hpp"
#include "cloudcropper/core/index_policy.hpp"
#include "cloudcropper/core/point_cloud.hpp"

namespace cc {

enum class BoxRole { Include, Exclude };
enum class BoolOp { Union, Intersection };

struct CropBox {
    Obb     box;
    BoxRole role = BoxRole::Include;
};

struct CropSpec {
    std::vector<CropBox> boxes;
    BoolOp               combine = BoolOp::Union;  // over Include boxes; Exclude always subtracts
};

struct CropResult {
    std::vector<std::uint8_t> mask;  // 1 byte/point: 1 = kept
    std::uint64_t             inCount    = 0;
    std::uint64_t             totalCount = 0;
};

// Brute-force by default; builds an octree over the include-box union AABB when
// `policy` says to (size threshold), which is exactly result-equal but faster on
// large/selective crops.
CropResult                  crop(const PointCloud& pc, const CropSpec& spec,
                                 const IndexPolicy& policy = {});
std::vector<std::uint32_t>  selectedIndices(const CropResult& result);

// Crops then ALWAYS recenters the output so the crop's center moves to the
// world origin (0,0,0). The translation offset is the average of the Include
// box centers (the single-include box's center in the common case), or the
// cropped cloud's bounds center when the spec has no Include boxes. The applied
// offset is recorded in metadata under "crop_offset" as "<x> <y> <z>"; original
// coords == stored coords + crop_offset. The mask-only crop() stays world-space.
PointCloud                  cropToCloud(const PointCloud& pc, const CropSpec& spec);

}  // namespace cc
