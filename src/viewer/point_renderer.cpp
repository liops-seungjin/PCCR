#include "point_renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "cloudcropper/core/attribute.hpp"
#include "cloudcropper/core/point_cloud.hpp"
#include "gl_util.hpp"

namespace cc::viewer {

namespace {

constexpr const char* kVert = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aRgb;
layout(location = 2) in float aScalar;  // normalized 0..1
layout(location = 3) in float aHeight;  // normalized 0..1
layout(location = 4) in float aKept;    // 0 or 1

uniform mat4 uView;
uniform mat4 uProj;
uniform mat4 uModel;      // extra transform (registration overlay); identity normally
uniform float uPointSize;
uniform int  uColorMode;  // 0 flat, 1 rgb, 2 scalar, 3 height

out vec3 vColor;
out float vKept;

// Simple turbo-ish colormap (blue->green->red).
vec3 colormap(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3(1.5 - abs(4.0 * t - 3.0),
                      1.5 - abs(4.0 * t - 2.0),
                      1.5 - abs(4.0 * t - 1.0)), 0.0, 1.0);
}

void main() {
    gl_Position = uProj * uView * uModel * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
    if (uColorMode == 1)      vColor = aRgb;
    else if (uColorMode == 2) vColor = colormap(aScalar);
    else if (uColorMode == 3) vColor = colormap(aHeight);
    else                      vColor = vec3(0.85, 0.85, 0.90);
    vKept = aKept;
}
)";

constexpr const char* kFrag = R"(#version 330 core
in vec3 vColor;
in float vKept;
out vec4 FragColor;

uniform int uHighlight;   // 1 == colour points by box membership
uniform vec3 uTint;       // mixed over the final colour (registration overlay)
uniform float uTintAmt;   // 0 = off

void main() {
    // round sprite
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    vec3 c = vColor;
    if (uHighlight == 1) {
        if (vKept > 0.5) c = mix(c, vec3(0.20, 1.0, 0.35), 0.65);   // inside box: green tint
        else             c = mix(c, vec3(0.16, 0.16, 0.19), 0.80);  // outside: dimmed
    }
    c = mix(c, uTint, uTintAmt);
    FragColor = vec4(c, 1.0);
}
)";

// Per-vertex interleaved record (8 floats).
struct Vertex {
    float px, py, pz;
    float r, g, b;
    float scalar;
    float height;
};

}  // namespace

void PointRenderer::release() {
    if (vboGeom_) { glDeleteBuffers(1, &vboGeom_); vboGeom_ = 0; }
    if (vboKept_) { glDeleteBuffers(1, &vboKept_); vboKept_ = 0; }
    if (vao_) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
    if (program_) { glDeleteProgram(program_); program_ = 0; }
}

PointRenderer::~PointRenderer() { release(); }

bool PointRenderer::build(const PointCloud& cloud, std::string& error) {
    // Idempotent: drop any previous GL objects so build() can reload a new cloud.
    if (vboGeom_) { glDeleteBuffers(1, &vboGeom_); vboGeom_ = 0; }
    if (vboKept_) { glDeleteBuffers(1, &vboKept_); vboKept_ = 0; }
    if (vao_) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
    if (program_) { glDeleteProgram(program_); program_ = 0; }
    count_ = 0;

    program_ = buildProgram(kVert, kFrag, error);
    if (!program_) return false;

    const auto& pos = cloud.positions();
    fullCount_      = pos.size();

    // LOD: decimate the display set by stride when above the budget. The full
    // cloud stays in core (crop/preview use it); only the GPU upload is reduced.
    displayIdx_.clear();
    if (budget_ > 0 && fullCount_ > budget_) {
        const std::size_t stride = (fullCount_ + budget_ - 1) / budget_;
        for (std::size_t i = 0; i < fullCount_; i += stride)
            displayIdx_.push_back(static_cast<std::uint32_t>(i));
        count_ = displayIdx_.size();
    } else {
        count_ = fullCount_;  // identity: displayIdx_ empty
    }
    auto srcOf = [&](std::size_t j) { return displayIdx_.empty() ? j : displayIdx_[j]; };

    // --- gather optional colour/scalar sources ---
    const AttributeColumn* rgb       = cloud.find(attr::kRGB);
    const AttributeColumn* intensity = cloud.find(attr::kIntensity);
    const AttributeColumn* label     = cloud.find(attr::kLabel);
    hasRgb_                          = (rgb != nullptr && rgb->arity() >= 3);

    const AttributeColumn* scalarCol = intensity ? intensity : label;
    hasScalar_                       = (scalarCol != nullptr);
    if (intensity)
        scalarName_ = "intensity";
    else if (label)
        scalarName_ = "label";

    // scalar normalization range
    double smin = 0.0, smax = 1.0;
    if (scalarCol) {
        smin = 1e300;
        smax = -1e300;
        for (std::size_t i = 0; i < fullCount_; ++i) {  // range over the FULL cloud
            const double v = readScalar(*scalarCol, i * scalarCol->arity());
            smin           = std::min(smin, v);
            smax           = std::max(smax, v);
        }
        if (smax <= smin) smax = smin + 1.0;
    }

    // height (z) range for the Height colour mode
    float zmin = 1e30f, zmax = -1e30f;
    for (const auto& p : pos) {
        zmin = std::min(zmin, p.z);
        zmax = std::max(zmax, p.z);
    }
    const float zrange = (zmax > zmin) ? (zmax - zmin) : 1.0f;

    // NOTE (LOD seam): for v1 we upload the whole cloud. A decimated display set
    // would be built here (stride/voxel) before filling `verts`; the crop preview
    // and authoritative crop are unaffected since they run over core data.
    std::vector<Vertex> verts(count_);
    for (std::size_t j = 0; j < count_; ++j) {
        const std::size_t i = srcOf(j);  // full-cloud index for displayed point j
        Vertex&           v = verts[j];
        v.px                = pos[i].x;
        v.py                = pos[i].y;
        v.pz                = pos[i].z;
        if (hasRgb_) {
            const double r = readScalar(*rgb, i * rgb->arity() + 0);
            const double g = readScalar(*rgb, i * rgb->arity() + 1);
            const double b = readScalar(*rgb, i * rgb->arity() + 2);
            // u8 colours come through readScalar as 0..255; floats assumed 0..1.
            const float  s = (rgb->type() == AttrType::U8) ? (1.0f / 255.0f) : 1.0f;
            v.r            = static_cast<float>(r) * s;
            v.g            = static_cast<float>(g) * s;
            v.b            = static_cast<float>(b) * s;
        } else {
            v.r = v.g = v.b = 0.85f;
        }
        if (scalarCol) {
            const double sv = readScalar(*scalarCol, i * scalarCol->arity());
            v.scalar        = static_cast<float>((sv - smin) / (smax - smin));
        } else {
            v.scalar = 0.0f;
        }
        v.height = (pos[i].z - zmin) / zrange;
    }

    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);

    glGenBuffers(1, &vboGeom_);
    glBindBuffer(GL_ARRAY_BUFFER, vboGeom_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts.size() * sizeof(Vertex)),
                 verts.data(), GL_STATIC_DRAW);
    const GLsizei stride = sizeof(Vertex);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(6 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(7 * sizeof(float)));

    // kept buffer: start all-kept.
    std::vector<float> kept(count_, 1.0f);
    glGenBuffers(1, &vboKept_);
    glBindBuffer(GL_ARRAY_BUFFER, vboKept_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(kept.size() * sizeof(float)), kept.data(),
                 GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(float), reinterpret_cast<void*>(0));

    glBindVertexArray(0);
    return true;
}

void PointRenderer::updateKept(const std::vector<std::uint8_t>& kept) {
    if (kept.size() != fullCount_ || !vboKept_) return;  // full-cloud mask
    std::vector<float> f(count_);
    for (std::size_t j = 0; j < count_; ++j) {
        const std::size_t src = displayIdx_.empty() ? j : displayIdx_[j];
        f[j]                  = kept[src] ? 1.0f : 0.0f;
    }
    glBindBuffer(GL_ARRAY_BUFFER, vboKept_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(f.size() * sizeof(float)), f.data());
}

void PointRenderer::draw(const glm::mat4& view, const glm::mat4& proj, float pointSize,
                         ColorMode mode, const glm::mat4& model, const glm::vec3& tint,
                         float tintAmt, bool highlight) {
    if (!program_ || count_ == 0) return;
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glUseProgram(program_);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uView"), 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uProj"), 1, GL_FALSE, &proj[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uModel"), 1, GL_FALSE, &model[0][0]);
    glUniform1f(glGetUniformLocation(program_, "uPointSize"), pointSize);
    glUniform1i(glGetUniformLocation(program_, "uColorMode"), static_cast<int>(mode));
    glUniform1i(glGetUniformLocation(program_, "uHighlight"), highlight ? 1 : 0);
    glUniform3f(glGetUniformLocation(program_, "uTint"), tint.x, tint.y, tint.z);
    glUniform1f(glGetUniformLocation(program_, "uTintAmt"), tintAmt);
    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(count_));
    glBindVertexArray(0);
}

}  // namespace cc::viewer
