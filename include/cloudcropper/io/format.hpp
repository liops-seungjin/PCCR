// Pluggable reader/writer interfaces (docs/design/01 §4).
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cloudcropper/common/result.hpp"
#include "cloudcropper/core/point_cloud.hpp"
#include "cloudcropper/io/byte_stream.hpp"

namespace cc::io {

struct FormatInfo {
    std::string              id;          // "ply", "pcd", "npz"
    std::vector<std::string> extensions;  // {".ply"}
    bool                     can_read  = false;
    bool                     can_write = false;
};

struct ReadOptions {
    std::uint64_t maxPoints = 0;  // 0 = unlimited; else fail with CloudTooLarge if N exceeds it
};

enum class Encoding { Auto, Ascii, Binary, BinaryCompressed };

struct WriteOptions {
    std::vector<std::string> fields;                  // attribute names; empty => all writable
    Encoding                 encoding = Encoding::Auto;
    bool                     gzip     = false;        // persisted-file gzip (transport layer)
};

class IReader {
public:
    virtual ~IReader()                       = default;
    [[nodiscard]] virtual FormatInfo info() const = 0;
    [[nodiscard]] virtual bool       can_handle(std::string_view              ext,
                                                std::span<const std::byte>    magic) const = 0;
    [[nodiscard]] virtual Result<PointCloud> read(IByteSource&       src,
                                                  const ReadOptions& opt) const = 0;
};

class IWriter {
public:
    virtual ~IWriter()                       = default;
    [[nodiscard]] virtual FormatInfo info() const = 0;
    // Intersection of attributes present in `pc` and what this format can emit.
    [[nodiscard]] virtual std::vector<std::string> writable_fields(const PointCloud& pc) const = 0;
    [[nodiscard]] virtual Result<void>             write(const PointCloud&   pc, IByteSink& sink,
                                                         const WriteOptions& opt) const = 0;
};

}  // namespace cc::io
