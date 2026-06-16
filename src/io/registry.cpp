#include "cloudcropper/io/registry.hpp"

#include <algorithm>
#include <cctype>

#if defined(CLOUDCROPPER_HAS_PLY)
#include "cloudcropper/io/ply.hpp"
#endif
#if defined(CLOUDCROPPER_HAS_PCD)
#include "cloudcropper/io/pcd.hpp"
#endif
#if defined(CLOUDCROPPER_HAS_NPZ)
#include "cloudcropper/io/npz.hpp"
#endif

namespace cc::io {
namespace {

std::string lower(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

bool hasExtension(const FormatInfo& info, std::string_view ext) {
    const std::string e = lower(ext);
    for (const std::string& x : info.extensions) {
        if (lower(x) == e) return true;
    }
    return false;
}

}  // namespace

void FormatRegistry::registerReader(std::shared_ptr<IReader> reader) {
    readers_.push_back(std::move(reader));
}

void FormatRegistry::registerWriter(std::shared_ptr<IWriter> writer) {
    writers_.push_back(std::move(writer));
}

std::shared_ptr<IReader> FormatRegistry::readerFor(std::string_view           ext,
                                                   std::span<const std::byte> magic) const {
    // Content magic first (robust to mis-named files), then extension.
    for (const auto& r : readers_) {
        if (r->can_handle({}, magic)) return r;
    }
    for (const auto& r : readers_) {
        if (hasExtension(r->info(), ext)) return r;
    }
    return nullptr;
}

std::shared_ptr<IWriter> FormatRegistry::writerForId(std::string_view id) const {
    for (const auto& w : writers_) {
        if (w->info().id == id) return w;
    }
    return nullptr;
}

std::shared_ptr<IWriter> FormatRegistry::writerForExt(std::string_view ext) const {
    for (const auto& w : writers_) {
        if (hasExtension(w->info(), ext)) return w;
    }
    return nullptr;
}

std::vector<FormatInfo> FormatRegistry::available() const {
    std::vector<FormatInfo> out;
    for (const auto& r : readers_) out.push_back(r->info());
    for (const auto& w : writers_) out.push_back(w->info());
    return out;
}

void registerBuiltinFormats(FormatRegistry& registry) {
#if defined(CLOUDCROPPER_HAS_PLY)
    registry.registerReader(std::make_shared<PlyReader>());
    registry.registerWriter(std::make_shared<PlyWriter>());
#endif
#if defined(CLOUDCROPPER_HAS_PCD)
    registry.registerReader(std::make_shared<PcdReader>());
    registry.registerWriter(std::make_shared<PcdWriter>());
#endif
#if defined(CLOUDCROPPER_HAS_NPZ)
    registry.registerReader(std::make_shared<NpzReader>());
    registry.registerWriter(std::make_shared<NpzWriter>());
#endif
    (void)registry;
}

}  // namespace cc::io
