// Public entry point for the interactive 3D viewer (docs/design/02).
//
// The viewer links GL/GLFW/ImGui/ImGuizmo internally; this header deliberately
// pulls in none of them so the app (and any other caller) stays GL-free. The
// viewer starts empty and loads clouds at runtime (initial file, drag-and-drop,
// or the in-UI Open…/path box) through a caller-provided load callback, so the
// viewer never needs to know about the io/transport (gz) loading details.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "cloudcropper/common/result.hpp"
#include "cloudcropper/core/point_cloud.hpp"

namespace cc::io {
class FormatRegistry;
}

namespace cc::viewer {

// Loads a cloud from a path (handling format dispatch + optional .gz). Supplied
// by the app, which owns the io registry + transport.
using LoadCloudFn = std::function<Result<PointCloud>(const std::string&)>;

// Options controlling a viewer session. The defaults give a normal interactive
// window; the screenshot/frames knobs make the viewer scriptable for CI smoke
// tests (render N frames, optionally dump the framebuffer, then exit).
struct ViewerOptions {
    std::string title       = "CloudCropper";
    int         width       = 1280;
    int         height      = 800;

    // Headless/CI controls. If `frames > 0`, the loop exits after that many
    // rendered frames instead of waiting for the window to close. If
    // `screenshot` is non-empty, the framebuffer is written there (PNG by
    // extension, else PPM) just before exit.
    int         frames      = 0;      // 0 == run until window closed
    std::string screenshot;           // empty == no screenshot
    float       pointSize   = 2.0f;   // initial point size, pixels
    std::string initialPath;          // optional cloud to load on startup ("" == empty)
};

// Runs the viewer, starting empty. `load` is invoked for the initial file (if
// `options.initialPath` is set), for drag-and-dropped files, and for the in-UI
// Open…/path box. `registry` provides the writers for the "Crop + Export" action.
// Returns an error only if the GL context / window cannot be created (no display).
[[nodiscard]] Result<void> runViewer(const io::FormatRegistry& registry,
                                     const ViewerOptions&      options,
                                     LoadCloudFn               load);

}  // namespace cc::viewer
