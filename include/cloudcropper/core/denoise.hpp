// Statistical outlier removal (docs/design/05 §4.5): a denoise guard so a stray
// point can't blow up bounds / normals. kNN-mean-distance with a mu+ratio*sigma
// threshold (same idea as Open3D remove_statistical_outlier). Returns a filtered
// PointCloud (positions + every attribute gathered by the kept indices).
#pragma once

#include "cloudcropper/core/point_cloud.hpp"

namespace cc {

struct DenoiseParams {
    int   k        = 16;    // neighbours used for the mean-distance statistic
    float stdRatio = 2.0f;  // keep points with meanDist <= mu + stdRatio * sigma
};

PointCloud removeStatisticalOutliers(const PointCloud& pc, const DenoiseParams& params = {});

}  // namespace cc
