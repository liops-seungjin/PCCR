// PointCloud — the one canonical SoA data model (docs/design/03 §2.2).
// xyz is mandatory; every other attribute is an optional named column.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "cloudcropper/common/geometry.hpp"
#include "cloudcropper/core/attribute.hpp"

namespace cc {

// Canonical attribute names (arbitrary names use the same paths).
namespace attr {
inline constexpr std::string_view kRGB       = "rgb";        // U8,  arity 3 (or 4)
inline constexpr std::string_view kNormal    = "normal";     // F32, arity 3
inline constexpr std::string_view kIntensity = "intensity";  // F32, arity 1
inline constexpr std::string_view kLabel     = "label";      // U32, arity 1
inline constexpr std::string_view kTimestamp = "timestamp";  // F64, arity 1
}  // namespace attr

class PointCloud {
public:
    [[nodiscard]] std::size_t size() const { return xyz_.size(); }

    [[nodiscard]] std::vector<Vec3>&       positions() { return xyz_; }
    [[nodiscard]] const std::vector<Vec3>& positions() const { return xyz_; }

    [[nodiscard]] bool                   has(std::string_view name) const;
    [[nodiscard]] AttributeColumn*       find(std::string_view name);
    [[nodiscard]] const AttributeColumn* find(std::string_view name) const;

    // Adds (or replaces) a column; returns a reference to the stored column.
    AttributeColumn& add(AttributeColumn col);

    [[nodiscard]] std::vector<AttributeColumn>&       attributes() { return attrs_; }
    [[nodiscard]] const std::vector<AttributeColumn>& attributes() const { return attrs_; }

    [[nodiscard]] std::map<std::string, std::string>&       metadata() { return meta_; }
    [[nodiscard]] const std::map<std::string, std::string>& metadata() const { return meta_; }

    [[nodiscard]] Aabb bounds() const;

    // Permute positions AND every attribute by the same indices (assemble step).
    [[nodiscard]] PointCloud gather(const std::vector<std::uint32_t>& indices) const;

private:
    std::vector<Vec3>                  xyz_;
    std::vector<AttributeColumn>       attrs_;
    std::map<std::string, std::string> meta_;
};

}  // namespace cc
