// gradient-SDF: hands the clouds to the vendored PyTorch/CUDA implementation
// (python/gradient_sdf_registration) through a persistent worker process
// (python/gsdf_worker.py, JSON-lines + NPZ handoff, FFT exhaustive-grid
// initialization, optional GPIS uncertainty channel). Runtime requirements
// only (never build-time): python3 with torch + open3d + scipy — see
// python/requirements.txt. Every algorithm knob in config/gradient-sdf-gpu.yaml
// is forwarded to the worker verbatim; there is no native fallback.
#pragma once

#include "cloudcropper/registration/registration.hpp"

namespace cc::reg::gsdf_gpu {

// Handles RegAlgo::GradientSdfGpu.
Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt);

}  // namespace cc::reg::gsdf_gpu
