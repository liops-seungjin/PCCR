// Normal estimation (docs/design/05 §4.1): kNN PCA over a self-contained uniform
// grid, so it runs in every build (no nanoflann dependency required). Writes a
// "normal" F32 arity-3 column so the existing PLY/PCD/NPZ writers emit normals.
#pragma once

#include <cstdint>
#include <optional>

#include "cloudcropper/common/geometry.hpp"
#include "cloudcropper/core/point_cloud.hpp"

namespace cc {

struct NormalParams {
    enum class Search { Knn, Radius, Hybrid };

    int                 k             = 30;     // neighbours for the local plane fit (Knn/Hybrid cap)
    Search              search        = Search::Knn;
    float               radius        = 0.0f;   // Radius/Hybrid support; 0 => 3 * point spacing
    std::optional<Vec3> viewpoint;              // if set, flip normals to face it
    bool                orientOutward = true;   // else flip away from the centroid
};

// Estimates a unit normal per point and stores it as the "normal" column
// (replacing any existing one). No-op-safe on tiny clouds.
void estimateNormals(PointCloud& pc, const NormalParams& params = {});

// Quick orientation/quality diagnostics over the estimated normals.
struct NormalStats {
    float flipRate        = 0.0f;  // fraction of neighbour pairs with opposite normals
    float outwardViolation = 0.0f;  // fraction facing away from the viewpoint (if given)
    float meanResidual    = 0.0f;  // mean point-to-local-plane distance
};
NormalStats normalDiagnostics(const PointCloud& pc, const std::optional<Vec3>& viewpoint = std::nullopt);

}  // namespace cc
