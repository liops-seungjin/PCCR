// Self-contained PLY codec (ascii + binary_little_endian, vertex element).
// First-slice stand-in for happly (docs/design/01 §1.1); same IReader/IWriter
// surface, so swapping to happly later is invisible to callers.
#pragma once

#include "cloudcropper/io/format.hpp"

namespace cc::io {

class PlyReader : public IReader {
public:
    [[nodiscard]] FormatInfo info() const override;
    [[nodiscard]] bool       can_handle(std::string_view              ext,
                                        std::span<const std::byte>    magic) const override;
    [[nodiscard]] Result<PointCloud> read(IByteSource&       src,
                                          const ReadOptions& opt) const override;
};

class PlyWriter : public IWriter {
public:
    [[nodiscard]] FormatInfo               info() const override;
    [[nodiscard]] std::vector<std::string> writable_fields(const PointCloud& pc) const override;
    [[nodiscard]] Result<void> write(const PointCloud& pc, IByteSink& sink,
                                     const WriteOptions& opt) const override;
};

}  // namespace cc::io
