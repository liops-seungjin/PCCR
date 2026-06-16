// Internal shared pieces for the registration backends: Vec3<->Eigen bridges,
// the 4x4 row-major array convention, and the backend-independent post-alignment
// metric (RMS nearest-neighbour distance + inlier count) so results from
// different algorithms are directly comparable.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "cloudcropper/core/point_cloud.hpp"
#include "cloudcropper/registration/registration.hpp"

namespace cc::reg::detail {

std::vector<Eigen::Vector3d> toEigenD(const std::vector<Vec3>& pts);
std::vector<Eigen::Vector3f> toEigenF(const std::vector<Vec3>& pts);

std::array<double, 16> fromIsometry(const Eigen::Isometry3d& T);  // row-major
Eigen::Isometry3d      toIsometry(const std::array<double, 16>& T);

// RMS source->target NN distance and inlier count (d < 3 * target spacing)
// after applying T to `source`. Samples up to ~50k points for speed.
struct AlignMetric {
    double      rmse    = 0.0;
    std::size_t inliers = 0;
};
AlignMetric alignmentMetric(const PointCloud& source, const PointCloud& target,
                            const Eigen::Isometry3d& T);

// Mean point spacing of `pc` (core estimator), clamped to a small positive value.
float safeSpacing(const PointCloud& pc);

}  // namespace cc::reg::detail
