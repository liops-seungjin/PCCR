// small_gicp-backed local registration: ICP / point-to-plane ICP / GICP / VGICP.
// Also reused by the global backends (KISS-Matcher, gradient-SDF) as the
// refinement stage, via the `init` transform in RegOptions.
#pragma once

#include "cloudcropper/registration/registration.hpp"

namespace cc::reg::gicp {

// Handles RegAlgo::{Icp, PlaneIcp, Gicp, VGicp}. `opt.init` is the initial
// guess (target <- source). Auto-derives downsample/maxCorr from the target
// spacing when 0.
Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt);

}  // namespace cc::reg::gicp
