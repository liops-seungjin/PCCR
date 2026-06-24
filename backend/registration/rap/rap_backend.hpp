// RAP (Register Any Point): hands the clouds to the vendored Rectified-Point-Flow
// implementation (python/rap_upstream) through a persistent worker process
// (python/rap_worker.py, JSON-lines + NPZ handoff). RAP is a learning-based,
// zero-shot, GLOBAL registrar built on flow matching — no initial guess needed;
// pairwise use feeds [target, source] as two "parts", the model generates the
// assembled cloud over N flow steps, per-part 4x4 poses are recovered by
// SVD/Procrustes and made relative to the target (T = inv(T_target) @ T_source ->
// row-major target<-source). Runtime requirements only (never build-time):
// python3 with torch 2.5.1 + the RAP core + the rap_model/mini_spinnet weights —
// see python/requirements.txt and python/download_weights.sh. DEPLOYMENT CAVEAT:
// the RAP core hard-depends on flash-attn (no SDPA fallback), whose pinned 2.7.4
// wheel does NOT support RTX 50xx/Blackwell (sm_120) — the single biggest
// deployment blocker for new GPUs. Every algorithm knob in config/rap.yaml is
// forwarded to the worker verbatim; there is no native fallback. The RapGicp
// variant chains GICP (small_gicp, in C++) onto the coarse result.
#pragma once

#include "cloudcropper/registration/registration.hpp"

namespace cc::reg::rap {

// Handles RegAlgo::Rap / RapGicp.
Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt);

}  // namespace cc::reg::rap
