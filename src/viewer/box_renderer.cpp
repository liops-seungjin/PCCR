#include "box_renderer.hpp"

#include <array>

#include "gl_util.hpp"

namespace cc::viewer {

namespace {

constexpr const char* kVert = R"(#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat4 uModel;
void main() { gl_Position = uProj * uView * uModel * vec4(aPos, 1.0); }
)";

constexpr const char* kFrag = R"(#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main() { FragColor = vec4(uColor, 1.0); }
)";

// 12 edges of the cube [-1,1]^3 as line-list vertices.
const std::array<glm::vec3, 24> kEdges = {{
    {-1, -1, -1}, {1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {1, 1, -1},   {-1, 1, -1},
    {-1, 1, -1},  {-1, -1, -1}, {-1, -1, 1}, {1, -1, 1}, {1, -1, 1},   {1, 1, 1},
    {1, 1, 1},    {-1, 1, 1},   {-1, 1, 1},  {-1, -1, 1}, {-1, -1, -1}, {-1, -1, 1},
    {1, -1, -1},  {1, -1, 1},   {1, 1, -1},  {1, 1, 1},   {-1, 1, -1},  {-1, 1, 1},
}};

}  // namespace

void BoxRenderer::release() {
    if (vbo_) { glDeleteBuffers(1, &vbo_); vbo_ = 0; }
    if (vao_) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
    if (program_) { glDeleteProgram(program_); program_ = 0; }
}

BoxRenderer::~BoxRenderer() { release(); }

bool BoxRenderer::build(std::string& error) {
    program_ = buildProgram(kVert, kFrag, error);
    if (!program_) return false;
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(kEdges.size() * sizeof(glm::vec3)),
                 kEdges.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(0));
    glBindVertexArray(0);
    return true;
}

void BoxRenderer::draw(const glm::mat4& view, const glm::mat4& proj, const glm::mat4& model,
                       const glm::vec3& color) {
    if (!program_) return;
    glUseProgram(program_);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uView"), 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uProj"), 1, GL_FALSE, &proj[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uModel"), 1, GL_FALSE, &model[0][0]);
    glUniform3fv(glGetUniformLocation(program_, "uColor"), 1, &color[0]);
    glBindVertexArray(vao_);
    glDrawArrays(GL_LINES, 0, 24);
    glBindVertexArray(0);
}

}  // namespace cc::viewer
