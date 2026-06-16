// Point-cloud registration backends. One dependency-free public API; the
// algorithms live in per-package subdirectories of backend/registration/:
//   common/           dispatcher, Vec3<->Eigen bridges, shared metrics (rmse)
//   gicp/             small_gicp wrapper: ICP / point-to-plane ICP / GICP / VGICP
//   kiss_matcher/     KISS-Matcher global registration (CLOUDCROPPER_HAS_KISS_MATCHER)
//   gradient_sdf_gpu/ gradient-SDF on the persistent Python worker (FFT init,
//                     uncertainty channel; the algorithm lives in the vendored
//                     gradient_sdf_registration package)
//
// Convention: `transform` maps SOURCE-frame points into the TARGET frame,
// row-major 4x4 (p_target = T * p_source).
#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "cloudcropper/common/result.hpp"
#include "cloudcropper/core/point_cloud.hpp"

namespace cc::reg {

enum class RegAlgo {
    Icp,             // point-to-point ICP        (local; needs a decent init)
    PlaneIcp,        // point-to-plane ICP        (local)
    Gicp,            // generalized ICP           (local; default)
    VGicp,           // voxelized GICP            (local)
    KissMatcher,     // KISS-Matcher              (global; no init needed)
    KissGicp,        // KISS-Matcher -> GICP      (global + local refine)
    GradientSdfGpu,  // gradient-SDF (Python worker; FFT init, CUDA/CPU)
    BufferX,         // BUFFER-X (Python worker; learning-based, global, no init)
    BufferXGicp,     // BUFFER-X -> GICP          (global + local refine)
    G3Reg,           // G3Reg (external CLI subprocess; learning-free, global, no init)
    G3RegGicp,       // G3Reg -> GICP             (global + local refine)
};

inline constexpr std::array<double, 16> kIdentity4 = {1, 0, 0, 0, 0, 1, 0, 0,
                                                      0, 0, 1, 0, 0, 0, 0, 1};

struct RegOptions {
    RegAlgo algo = RegAlgo::Gicp;

    // Shared knobs. 0 = derive automatically from the target's point spacing.
    float downsample = 0.0f;  // preprocess voxel/leaf size (meters)
    float maxCorr    = 0.0f;  // max correspondence distance (meters)
    int   threads    = 4;

    // KISS-Matcher: working resolution (its `voxel_size`); 0 = auto.
    float kissResolution = 0.0f;

    // gradient-SDF (worker): voxel grid resolution per axis (0 = take the yaml
    // value) and truncation distance as a multiple of the target's median
    // point spacing — feeds the variance channel below.
    int   sdfResolution = 100;
    float sdfTruncMul   = 4.0f;
    // gradient-SDF (worker): build a per-voxel uncertainty channel on the SDF
    // field (discrete GPIS-style query variance: data proximity + local
    // surface residual + quantization), weight the robust loss
    // heteroscedastically with it, and report confidence/normResidual.
    // Disabled = the classic uniform-Cauchy behavior, no trust score.
    bool sdfUncertainty = true;

    // BUFFER-X (worker): input downsample voxel size (0 = take the yaml value /
    // let the worker auto-derive via its scale normalization). The inference
    // device is decided by the yaml (`device`); every other BUFFER-X knob is
    // forwarded verbatim from config/bufferx.yaml to the worker.
    float bufferxVoxel = 0.0f;

    // For global methods (KISS-Matcher / gradient-SDF): refine the coarse
    // result with GICP afterwards (the authors' recommended pipeline).
    bool refine = true;

    // Initial guess for the local methods (row-major, target <- source).
    std::array<double, 16> init = kIdentity4;
};

struct RegResult {
    std::array<double, 16> transform = kIdentity4;  // row-major, target <- source
    bool        converged = false;
    double      rmse      = 0.0;  // RMS source->target nearest-neighbour distance AFTER alignment
    std::size_t inliers   = 0;    // matches within 3x target spacing after alignment
    double      seconds   = 0.0;
    std::string detail;           // backend-specific one-liner for the UI/CLI

    // Uncertainty-aware trust signals (gradient-SDF with the uncertainty channel
    // only; -1 = not available). At the FINAL pose each source residual r_i is
    // normalized by the field's own variance, u_i = r_i^2 / sigma^2(x_i):
    //   confidence   = fraction of points with u_i < 4 (~95% chi-square) — how
    //                  much of the source the field can "explain". Plausible-but-
    //                  wrong poses (symmetry / occlusion) score LOW here even
    //                  when rmse looks fine.
    //   normResidual = mean u_i (~1 means residuals consistent with the field).
    double confidence   = -1.0;
    double normResidual = -1.0;
};

const char* algoName(RegAlgo a);

// Registers `source` onto `target` (estimates T with p_target = T * p_source).
// Fails with ErrorCode::Unsupported when the requested backend was not built.
Result<RegResult> registerClouds(const PointCloud& source, const PointCloud& target,
                                 const RegOptions& opt = {});

// Applies a row-major 4x4 to every position (and rotates a "normal" column if
// present). Used by "apply to source" / "save aligned".
void applyTransform(PointCloud& pc, const std::array<double, 16>& T);

}  // namespace cc::reg
