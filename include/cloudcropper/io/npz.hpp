// NPZ codec (docs/design/01 §1.3): own NPY parser/serializer + miniz ZIP
// container. Compiled only when miniz is available (CLOUDCROPPER_HAS_NPZ).
#pragma once

#include <optional>
#include <string>

#include "cloudcropper/io/format.hpp"

namespace cc::io {

// Runtime "template" schema (docs/design/05): per-cloud metadata written
// alongside the per-point surface_points/surface_normals arrays. Units are meters.
struct TemplateMeta {
    Vec3                bbox_min{}, bbox_max{}, bbox_center{};
    Vec3                canonical_center{}, canonical_axis{};
    std::optional<Vec3> tip_local;        // tailstock_tip_local (template frame)
    std::optional<Vec3> bar_point_local;  // bar_axis_point_local
    std::optional<Vec3> bar_dir_local;    // bar_axis_dir_local (unit)
    std::optional<Vec3> object_pose_origin_local;  // generic object pose origin
    std::optional<Vec3> object_pose_dir_local;     // generic object pose direction (unit)
    float               point_spacing_m = 0.0f;
    std::string         units = "m";
    std::string         frame = "template";
};

// Writes the meters template schema: positions as `surface_points` (N,3),
// the "normal" column (if any) as `surface_normals` (N,3), the per-cloud
// metadata arrays, and a `__meta__` JSON entry. Read back by NpzReader (which
// recognises the aliases and keeps the metadata arrays in metadata()).
Result<void> writeTemplateNpz(const PointCloud& pc, const TemplateMeta& meta, IByteSink& sink);

class NpzReader : public IReader {
public:
    [[nodiscard]] FormatInfo info() const override;
    [[nodiscard]] bool       can_handle(std::string_view              ext,
                                        std::span<const std::byte>    magic) const override;
    [[nodiscard]] Result<PointCloud> read(IByteSource&       src,
                                          const ReadOptions& opt) const override;
};

class NpzWriter : public IWriter {
public:
    [[nodiscard]] FormatInfo               info() const override;
    [[nodiscard]] std::vector<std::string> writable_fields(const PointCloud& pc) const override;
    [[nodiscard]] Result<void> write(const PointCloud& pc, IByteSink& sink,
                                     const WriteOptions& opt) const override;
};

}  // namespace cc::io
