// Format registry — dispatch by content magic then extension (docs/design/01 §4).
#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "cloudcropper/io/format.hpp"

namespace cc::io {

class FormatRegistry {
public:
    void registerReader(std::shared_ptr<IReader> reader);
    void registerWriter(std::shared_ptr<IWriter> writer);

    [[nodiscard]] std::shared_ptr<IReader> readerFor(std::string_view           ext,
                                                     std::span<const std::byte> magic) const;
    [[nodiscard]] std::shared_ptr<IWriter> writerForId(std::string_view id) const;
    [[nodiscard]] std::shared_ptr<IWriter> writerForExt(std::string_view ext) const;

    [[nodiscard]] std::vector<FormatInfo> available() const;

private:
    std::vector<std::shared_ptr<IReader>> readers_;
    std::vector<std::shared_ptr<IWriter>> writers_;
};

// Registers every format compiled into this build (guarded by CLOUDCROPPER_HAS_*).
// Explicit registration avoids the static-lib self-registration pitfall.
void registerBuiltinFormats(FormatRegistry& registry);

}  // namespace cc::io
