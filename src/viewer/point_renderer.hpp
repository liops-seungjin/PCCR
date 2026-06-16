// GL point renderer: one static interleaved VBO for the display set, a small
// dynamic buffer for per-point "kept" preview flags, colour modes by uniform.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace cc {
class PointCloud;
}

namespace cc::viewer {

enum class ColorMode : int { Flat = 0, Rgb = 1, Scalar = 2, Height = 3 };

// Builds GPU buffers from a (borrowed) PointCloud and draws GL_POINTS. The full
// cloud is uploaded for v1 (the decimation seam is noted in build()).
class PointRenderer {
public:
    ~PointRenderer();

    // Returns false (with `error`) if shader build fails. Must be called with a
    // live GL context.
    bool build(const PointCloud& cloud, std::string& error);

    // Upload per-point kept flags. `kept` is the FULL-cloud mask (size ==
    // fullCount()); it is mapped onto the displayed (possibly decimated) set.
    void updateKept(const std::vector<std::uint8_t>& kept);

    // `model` is an extra world transform (registration overlay), `tint`/`tintAmt`
    // blend a flat colour over the result, `highlight` toggles the green
    // box-membership colouring. Defaults preserve the original behaviour.
    void draw(const glm::mat4& view, const glm::mat4& proj, float pointSize, ColorMode mode,
              const glm::mat4& model = glm::mat4(1.0f), const glm::vec3& tint = {0, 0, 0},
              float tintAmt = 0.0f, bool highlight = true);

    // Free GL objects; must run while the GL context is alive (call before
    // glfwTerminate). The destructor also calls it (idempotent).
    void release();

    // LOD: cap the uploaded display set to ~`budget` points (decimate by stride
    // above it). Full cloud stays authoritative in core. Re-build to apply.
    void setBudget(std::size_t budget) { budget_ = budget; }

    [[nodiscard]] std::size_t count() const { return count_; }       // displayed points
    [[nodiscard]] std::size_t fullCount() const { return fullCount_; }
    [[nodiscard]] bool        hasRgb() const { return hasRgb_; }
    [[nodiscard]] bool        hasScalar() const { return hasScalar_; }
    [[nodiscard]] const std::string& scalarName() const { return scalarName_; }

private:
    GLuint      vao_       = 0;
    GLuint      vboGeom_   = 0;  // static: pos(3) rgb(3) scalar(1) height(1)
    GLuint      vboKept_   = 0;  // dynamic: u8 kept flag
    GLuint      program_   = 0;
    std::size_t count_     = 0;          // displayed (possibly decimated) point count
    std::size_t fullCount_ = 0;          // full cloud point count
    std::size_t budget_    = 8'000'000;  // display-set cap before decimation
    std::vector<std::uint32_t> displayIdx_;  // full-cloud indices uploaded (empty == identity)
    bool        hasRgb_    = false;
    bool        hasScalar_ = false;
    std::string scalarName_;
};

}  // namespace cc::viewer
