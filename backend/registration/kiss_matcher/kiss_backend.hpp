// KISS-Matcher global registration (no initial guess needed). KissGicp chains
// the coarse global solution into the gicp package as the recommended refine.
#pragma once

#include "cloudcropper/registration/registration.hpp"

namespace cc::reg::kiss {

// Handles RegAlgo::{KissMatcher, KissGicp}.
Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt);

}  // namespace cc::reg::kiss
