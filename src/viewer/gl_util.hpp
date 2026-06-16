// Small GL helpers used by the viewer: shader build + framebuffer screenshot.
// Private impl header; pulls in glad (the GL loader) so it must be included
// before any GLFW header that would otherwise drag in <GL/gl.h>.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glad/glad.h>

namespace cc::viewer {

// Compiles a vertex+fragment program; returns 0 and fills `log` on failure.
GLuint buildProgram(const char* vertexSrc, const char* fragmentSrc, std::string& log);

// Reads the current framebuffer (RGBA8) back to CPU; rows top-to-bottom.
std::vector<std::uint8_t> readFramebuffer(int width, int height);

// Writes RGB(A) pixels (top-to-bottom) to `path`. PNG if the path ends in
// ".png" (self-contained encoder, no external deps), otherwise binary PPM.
// Returns false on file error.
bool writeImage(const std::string& path, const std::uint8_t* rgba, int width, int height);

}  // namespace cc::viewer
