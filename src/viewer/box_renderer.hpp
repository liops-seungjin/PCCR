// Wireframe renderer for OBB boxes (12 edges of a unit cube, transformed by the
// box localToWorld matrix). Tiny dynamic geometry — one shared unit-cube VBO.
#pragma once

#include <string>

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace cc::viewer {

class BoxRenderer {
public:
    ~BoxRenderer();
    bool build(std::string& error);
    // Free GL objects; must run while the GL context is alive (call before
    // glfwTerminate). The destructor also calls it (idempotent).
    void release();
    // Draws one box: model maps the unit cube [-1,1]^3 to the OBB; selected
    // boxes render brighter.
    void draw(const glm::mat4& view, const glm::mat4& proj, const glm::mat4& model,
              const glm::vec3& color);

private:
    GLuint vao_ = 0, vbo_ = 0, program_ = 0;
    GLuint locVP_ = 0, locModel_ = 0, locColor_ = 0;
};

}  // namespace cc::viewer
