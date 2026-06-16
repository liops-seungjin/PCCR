#include "cloudcropper/core/point_cloud.hpp"

#include <cstring>

namespace cc {

bool PointCloud::has(std::string_view name) const { return find(name) != nullptr; }

AttributeColumn* PointCloud::find(std::string_view name) {
    for (auto& c : attrs_) {
        if (c.name() == name) return &c;
    }
    return nullptr;
}

const AttributeColumn* PointCloud::find(std::string_view name) const {
    for (const auto& c : attrs_) {
        if (c.name() == name) return &c;
    }
    return nullptr;
}

AttributeColumn& PointCloud::add(AttributeColumn col) {
    if (AttributeColumn* existing = find(col.name())) {
        *existing = std::move(col);
        return *existing;
    }
    attrs_.push_back(std::move(col));
    return attrs_.back();
}

Aabb PointCloud::bounds() const {
    Aabb b;
    for (const Vec3& p : xyz_) b.expand(p);
    return b;
}

PointCloud PointCloud::gather(const std::vector<std::uint32_t>& indices) const {
    PointCloud out;
    out.meta_ = meta_;

    out.xyz_.reserve(indices.size());
    for (std::uint32_t i : indices) out.xyz_.push_back(xyz_[i]);

    for (const AttributeColumn& src : attrs_) {
        AttributeColumn   dst(src.name(), src.type(), src.arity(), indices.size());
        const std::size_t stride = src.stride();
        const std::byte*  sp     = src.bytes().data();
        std::byte*        dp     = dst.bytes().data();
        for (std::size_t k = 0; k < indices.size(); ++k) {
            std::memcpy(dp + k * stride, sp + static_cast<std::size_t>(indices[k]) * stride, stride);
        }
        out.attrs_.push_back(std::move(dst));
    }
    return out;
}

}  // namespace cc
