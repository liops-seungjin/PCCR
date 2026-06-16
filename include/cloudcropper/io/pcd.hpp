// PCD codec (docs/design/01 §1.2): ascii + binary self-contained;
// binary_compressed (LZF) when liblzf is available (CLOUDCROPPER_HAS_LZF).
#pragma once

#include "cloudcropper/io/format.hpp"

namespace cc::io {

class PcdReader : public IReader {
public:
    [[nodiscard]] FormatInfo info() const override;
    [[nodiscard]] bool       can_handle(std::string_view              ext,
                                        std::span<const std::byte>    magic) const override;
    [[nodiscard]] Result<PointCloud> read(IByteSource&       src,
                                          const ReadOptions& opt) const override;
};

class PcdWriter : public IWriter {
public:
    [[nodiscard]] FormatInfo               info() const override;
    [[nodiscard]] std::vector<std::string> writable_fields(const PointCloud& pc) const override;
    [[nodiscard]] Result<void> write(const PointCloud& pc, IByteSink& sink,
                                     const WriteOptions& opt) const override;
};

}  // namespace cc::io
