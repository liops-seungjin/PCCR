// gzip transport (docs/design/00 §4, 03 §4): RFC-1952 gzip framing for persisted
// `.gz` artifacts. Single codec = zlib (the design's accepted substitute for
// zlib-ng). Whole-buffer helpers — the CLI reads/writes whole clouds.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "cloudcropper/common/result.hpp"

namespace cc::transport {

Result<std::vector<std::byte>> gzipCompress(std::span<const std::byte> in, int level = 6);
Result<std::vector<std::byte>> gzipDecompress(std::span<const std::byte> in);

}  // namespace cc::transport
