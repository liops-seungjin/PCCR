// Per-package default settings, loaded from config/<package>.yaml:
//   config/gicp.yaml              ICP / Plane-ICP / GICP / VGICP
//   config/kiss-matcher.yaml      KISS-Matcher (+ GICP refine)
//   config/gradient-sdf-gpu.yaml  gradient-SDF (Python worker; every key is
//                                 forwarded to the worker)
// The files are a flat `key: value` subset of YAML (comments with #). Search
// order: $CLOUDCROPPER_CONFIG_DIR, ./config/, then a config/ directory next to
// (or up to a few levels above) the executable — so running from the repo or
// from build/<preset>/src/app both find the checked-in files.
#pragma once

#include <map>
#include <string>

#include "cloudcropper/registration/registration.hpp"

namespace cc::reg {

// The YAML file name that holds this algorithm's defaults.
const char* configFileFor(RegAlgo a);

// RegOptions for `algo` with the package YAML applied over the compiled
// defaults. Missing file or keys silently keep the compiled values; explicit
// CLI flags / UI edits are expected to override the result. Never fails.
RegOptions defaultsFor(RegAlgo algo);

// Raw key/value map of a config file (same search order); empty when absent.
// Used by packages with settings beyond RegOptions (e.g. the GPU bridge).
std::map<std::string, std::string> configValues(const char* filename);

}  // namespace cc::reg
