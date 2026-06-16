// Cloud analysis used to fill the template schema (docs/design/05): a PCA-derived
// canonical frame and a nominal point spacing.
#pragma once

#include "cloudcropper/common/geometry.hpp"
#include "cloudcropper/core/point_cloud.hpp"

namespace cc {

struct CanonicalFrame {
    Vec3 center{};  // centroid of the cloud
    Vec3 axis{};    // principal (largest-variance) direction, unit length
    Vec3 extents{}; // sqrt of the three eigenvalues (std-dev along each PCA axis)
};

// Centroid + covariance PCA. `axis` is the eigenvector of the largest eigenvalue.
CanonicalFrame pcaFrame(const PointCloud& pc);

// Nominal point spacing (meters): median nearest-neighbour distance over a sample.
float estimatePointSpacing(const PointCloud& pc);

}  // namespace cc
