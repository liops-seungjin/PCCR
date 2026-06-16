// BUFFER-X: hands the clouds to the vendored PyTorch implementation
// (python/bufferx) through a persistent worker process
// (python/bufferx_worker.py, JSON-lines + NPZ handoff). BUFFER-X is a
// learning-based, zero-shot, GLOBAL registrar — no initial guess needed; a
// single (3DMatch-trained) checkpoint generalizes across sensors/scales via
// automatic density-aware scale normalization. Runtime requirements only
// (never build-time): python3 with torch + the BUFFER-X CUDA extensions
// (pointnet2_ops / KNN_CUDA / torch-batch-svd / cpp_wrappers, no sparse-conv
// backbone) + the BUFFER-X weights — see python/requirements.txt and
// python/download_weights.sh. Every algorithm knob in config/bufferx.yaml is
// forwarded to the worker verbatim; there is no native fallback. The
// BufferXGicp variant chains GICP (small_gicp, in C++) onto the coarse result.
#pragma once

#include "cloudcropper/registration/registration.hpp"

namespace cc::reg::bufferx {

// Handles RegAlgo::BufferX / BufferXGicp.
Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt);

}  // namespace cc::reg::bufferx
