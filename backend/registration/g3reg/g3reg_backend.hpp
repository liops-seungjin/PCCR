// G3Reg: learning-free GLOBAL point-cloud registration (Gaussian Ellipsoid
// Model + Pyramid Compatibility Graph, IEEE T-ASE 2024, HKUST-Aerial-Robotics).
// G3Reg's dependency stack is heavily version-pinned (PCL<1.11 / GTSAM 4.1.1 /
// Eigen<3.4 / igraph) and conflicts with CloudCropper's toolchain, so we do NOT
// link it. Instead we call a standalone-built external CLI `cc_g3reg_cli` as a
// ONE-SHOT subprocess (zero build-time dependency): the clouds are handed off
// as temporary .pcd files written by CloudCropper's own PCD writer, and the
// three stdout lines (G3REG_TF / G3REG_INLIERS / G3REG_TIME) are parsed back.
// A missing binary/config or unparseable output fails cleanly with an
// ErrorCode. The G3RegGicp variant chains GICP (small_gicp, in C++) onto the
// coarse global result. Needs the PCD codec (CLOUDCROPPER_HAS_PCD) for the
// handoff; without it the backend returns Unsupported.
#pragma once

#include "cloudcropper/registration/registration.hpp"

namespace cc::reg::g3reg {

// Handles RegAlgo::G3Reg / RegAlgo::G3RegGicp.
Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt);

}  // namespace cc::reg::g3reg
