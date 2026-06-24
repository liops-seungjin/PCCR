// Interactive 3D viewer main loop (docs/design/02). Owns the GLFW window + GL
// context, ImGui/ImGuizmo, the orbit camera, point + box renderers, the box
// edit/crop UI, and crop export through the io registry.
#include "cloudcropper/viewer/viewer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <glad/glad.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL  // for gtx/matrix_decompose
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <imgui.h>
#include <ImGuizmo.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// Native file-open dialog (header-only; shells to zenity/kdialog on Linux).
// Optional — guarded so the viewer still builds if the header isn't installed.
#if __has_include(<portable-file-dialogs.h>)
#include <portable-file-dialogs.h>
#define CLOUDCROPPER_VIEWER_PFD 1
#endif

#include "box_renderer.hpp"
#include "camera.hpp"
#include "cloudcropper/core/analysis.hpp"
#include "cloudcropper/core/crop.hpp"
#include "cloudcropper/core/point_cloud.hpp"
#include "cloudcropper/io/byte_stream.hpp"
#if defined(CLOUDCROPPER_HAS_NPZ)
#include "cloudcropper/io/npz.hpp"
#endif
#include "cloudcropper/io/registry.hpp"
#include "cloudcropper/io/rosbag.hpp"  // ROS2 bag topic/frame navigation (guarded by HAS_ROSBAG)
#if defined(CLOUDCROPPER_HAS_REGISTRATION)
#include "cloudcropper/registration/config.hpp"        // per-package YAML defaults
#include "cloudcropper/registration/registration.hpp"  // right-side registration panel
#endif
#include "gl_util.hpp"
#include "point_renderer.hpp"

namespace cc::viewer {

namespace {

// --- cc::Obb <-> glm interop -----------------------------------------------

glm::quat toGlm(const Quat& q) { return glm::quat(q.w, q.x, q.y, q.z); }
Quat      fromGlm(const glm::quat& q) { return Quat{q.x, q.y, q.z, q.w}; }

glm::mat4 obbToMatrix(const Obb& b) {
    glm::mat4 m = glm::translate(glm::mat4(1.0f), glm::vec3(b.center.x, b.center.y, b.center.z));
    m *= glm::mat4_cast(toGlm(b.rotation));
    m  = glm::scale(m, glm::vec3(b.halfExtents.x, b.halfExtents.y, b.halfExtents.z));
    return m;
}

Obb matrixToObb(const glm::mat4& m) {
    glm::vec3 scale{1.0f}, translation{0.0f}, skew{0.0f};
    glm::vec4 perspective{0.0f};
    glm::quat rot{1.0f, 0.0f, 0.0f, 0.0f};
    glm::decompose(m, scale, rot, translation, skew, perspective);
    Obb b;
    b.center      = {translation.x, translation.y, translation.z};
    b.halfExtents = {std::abs(scale.x), std::abs(scale.y), std::abs(scale.z)};
    b.rotation    = fromGlm(glm::normalize(rot));
    return b;
}

glm::vec3 boxSemanticUp(int upAxis) {
    return upAxis == 0   ? glm::vec3{1, 0, 0}
           : upAxis == 2 ? glm::vec3{0, 0, 1}
                         : glm::vec3{0, 1, 0};
}

glm::vec3 fallbackPlanarAxis(glm::vec3 up) {
    const glm::vec3 candidates[] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
    };
    glm::vec3 best{1, 0, 0};
    float     bestLen2 = -1.0f;
    for (glm::vec3 c : candidates) {
        const glm::vec3 projected = c - up * glm::dot(c, up);
        const float     len2      = glm::dot(projected, projected);
        if (len2 > bestLen2) {
            best     = projected;
            bestLen2 = len2;
        }
    }
    return bestLen2 > 1e-12f ? best * (1.0f / std::sqrt(bestLen2)) : glm::vec3{1, 0, 0};
}

Quat uprightBoxRotation(Quat q, int upAxis) {
    const glm::vec3 up = boxSemanticUp(upAxis);
    Vec3            rx = rotate(q, Vec3{1, 0, 0});
    glm::vec3       x{rx.x, rx.y, rx.z};
    x -= up * glm::dot(x, up);
    if (glm::length(x) <= 1e-6f) {
        rx = rotate(q, Vec3{0, 1, 0});
        x  = glm::vec3{rx.x, rx.y, rx.z};
        x -= up * glm::dot(x, up);
    }
    x = glm::length(x) > 1e-6f ? glm::normalize(x) : fallbackPlanarAxis(up);
    const glm::vec3 y = glm::normalize(glm::cross(up, x));

    glm::mat3 r(1.0f);
    r[0] = x;   // local X = longitudinal/front
    r[1] = y;   // local Y = lateral/left
    r[2] = up;  // local Z = viewer up axis (Y-up files, Z-up ROS by default)
    return fromGlm(glm::normalize(glm::quat_cast(r)));
}

void enforceUprightBox(Obb& ob, int upAxis) {
    ob.rotation = uprightBoxRotation(ob.rotation, upAxis);
}

Obb uprightObbFromAabb(const Aabb& b, int upAxis) {
    Obb o;
    o.center   = (b.min + b.max) * 0.5f;
    o.rotation = uprightBoxRotation(Quat{}, upAxis);

    const Vec3 worldHalf = (b.max - b.min) * 0.5f;
    auto halfAlong = [&](Vec3 localAxis) {
        const Vec3 axis = rotate(o.rotation, localAxis);
        return std::fabs(axis.x) * worldHalf.x + std::fabs(axis.y) * worldHalf.y +
               std::fabs(axis.z) * worldHalf.z;
    };
    o.halfExtents = {
        halfAlong(Vec3{1, 0, 0}),
        halfAlong(Vec3{0, 1, 0}),
        halfAlong(Vec3{0, 0, 1}),
    };
    return o;
}

struct EditBox {
    Obb     obb;
    BoxRole role    = BoxRole::Include;
    bool    enabled = true;
};

CropSpec buildSpec(const std::vector<EditBox>& boxes, BoolOp op) {
    CropSpec spec;
    spec.combine = op;
    for (const auto& b : boxes) {
        if (b.enabled) spec.boxes.push_back({b.obb, b.role});
    }
    return spec;
}

// Preview membership over the (full) display set — same OBB math as core.
std::vector<std::uint8_t> previewMask(const PointCloud& cloud, const CropSpec& spec) {
    const auto& pos = cloud.positions();
    std::vector<std::uint8_t> mask(pos.size(), spec.boxes.empty() ? 1 : 0);
    if (spec.boxes.empty()) return mask;
    for (std::size_t i = 0; i < pos.size(); ++i) {
        bool include = (spec.combine == BoolOp::Intersection);
        bool sawInc  = false;
        bool exclude = false;
        for (const auto& cb : spec.boxes) {
            const bool in = cb.box.contains(pos[i]);
            if (cb.role == BoxRole::Include) {
                sawInc = true;
                if (spec.combine == BoolOp::Union)
                    include = include || in;
                else
                    include = include && in;
            } else if (in) {
                exclude = true;
            }
        }
        if (!sawInc) include = true;  // exclude-only: keep everything not excluded
        mask[i] = (include && !exclude) ? 1 : 0;
    }
    return mask;
}

const char* extOf(const std::string& path) {
    const auto dot = path.find_last_of('.');
    return dot == std::string::npos ? "" : path.c_str() + dot;
}

glm::vec3 toGlm(Vec3 v) { return {v.x, v.y, v.z}; }
Vec3      fromGlm(const glm::vec3& v) { return {v.x, v.y, v.z}; }

Vec3 normalizedOr(Vec3 v, Vec3 fallback) {
    const float n2 = dot(v, v);
    if (n2 <= 1e-12f) return fallback;
    return v * (1.0f / std::sqrt(n2));
}

std::optional<glm::vec3> metadataVec3(const PointCloud& pc, const char* key) {
    const auto it = pc.metadata().find(key);
    if (it == pc.metadata().end()) return std::nullopt;
    float x = 0.0f, y = 0.0f, z = 0.0f;
    if (std::sscanf(it->second.c_str(), "%f %f %f", &x, &y, &z) == 3)
        return glm::vec3{x, y, z};
    return std::nullopt;
}

#if defined(CLOUDCROPPER_HAS_REGISTRATION)
struct GpuProbeInfo {
    bool        done = false;
    bool        usable = false;
    std::string gpuName = "-";
    std::string torchVersion = "-";
    std::string message = "checking...";
    std::string python = "python3";
};

std::string shellQuote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'')
            out += "'\\''";
        else
            out += c;
    }
    out += "'";
    return out;
}

std::string trimCopy(std::string s) {
    const char* ws = " \t\r\n";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return {};
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

std::string runPipe(const std::string& cmd) {
    std::array<char, 256> buf{};
    std::string          out;
    FILE*                pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {};
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe))
        out += buf.data();
    pclose(pipe);
    return out;
}

std::string firstNonEmptyLine(const std::string& text) {
    std::size_t pos = 0;
    while (pos <= text.size()) {
        const std::size_t end = text.find('\n', pos);
        std::string line = text.substr(pos, end == std::string::npos ? std::string::npos
                                                                      : end - pos);
        line = trimCopy(std::move(line));
        if (!line.empty()) return line;
        if (end == std::string::npos) break;
        pos = end + 1;
    }
    return {};
}

std::string probeNvidiaGpuName() {
    return firstNonEmptyLine(runPipe("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null"));
}

std::vector<std::string> gpuProbePythonCandidates() {
    std::vector<std::string> candidates;
    auto add = [&](std::string py) {
        py = trimCopy(std::move(py));
        if (py.empty()) return;
        for (const std::string& c : candidates)
            if (c == py) return;
        candidates.push_back(py);
    };
    add(cc::reg::configValues("rap.yaml")["python"]);
    add(cc::reg::configValues("bufferx.yaml")["python"]);
    add(cc::reg::configValues("gradient-sdf-gpu.yaml")["python"]);
    add("python3");
    return candidates;
}

GpuProbeInfo probeGpu() {
    GpuProbeInfo info;
    const std::string hardwareGpu = probeNvidiaGpuName();
    GpuProbeInfo firstTorchInfo;
    bool         hasTorchFallback = false;
    const char* code =
        "import sys\n"
        "try:\n"
        " import torch\n"
        " print('CCGPU_TORCH='+str(torch.__version__))\n"
        " avail=torch.cuda.is_available()\n"
        " print('CCGPU_CUDA=' + ('1' if avail else '0'))\n"
        " print('CCGPU_NAME=' + (torch.cuda.get_device_name(torch.cuda.current_device()) if avail else ''))\n"
        "except Exception as e:\n"
        " print('CCGPU_ERROR='+repr(e))\n"
        " sys.exit(1)\n";

    for (const std::string& py : gpuProbePythonCandidates()) {
        const std::string out = runPipe(shellQuote(py) + " -c " + shellQuote(code) + " 2>&1");
        bool sawTorch = false;
        bool sawCuda = false;
        std::string torch;
        std::string gpu;
        std::string err;
        std::size_t pos = 0;
        while (pos <= out.size()) {
            const std::size_t end = out.find('\n', pos);
            const std::string line = out.substr(pos, end == std::string::npos ? std::string::npos
                                                                              : end - pos);
            if (line.rfind("CCGPU_TORCH=", 0) == 0) {
                sawTorch = true;
                torch = line.substr(12);
            } else if (line.rfind("CCGPU_CUDA=", 0) == 0) {
                sawCuda = line.substr(11) == "1";
            } else if (line.rfind("CCGPU_NAME=", 0) == 0) {
                gpu = line.substr(11);
            } else if (line.rfind("CCGPU_ERROR=", 0) == 0) {
                err = line.substr(12);
            }
            if (end == std::string::npos) break;
            pos = end + 1;
        }
        if (sawTorch) {
            GpuProbeInfo candidate;
            candidate.done = true;
            candidate.python = py;
            candidate.torchVersion = torch.empty() ? "-" : torch;
            candidate.usable = sawCuda;
            candidate.gpuName = !gpu.empty() ? gpu : (!hardwareGpu.empty() ? hardwareGpu : "-");
            candidate.message = sawCuda ? "GPU using available" : "Unavailable: torch.cuda unavailable";
            if (sawCuda) return candidate;
            if (!hasTorchFallback) {
                firstTorchInfo = std::move(candidate);
                hasTorchFallback = true;
            }
            continue;
        }
        if (!err.empty()) info.message = err;
    }

    if (hasTorchFallback) return firstTorchInfo;

    info.done = true;
    info.usable = false;
    info.gpuName = hardwareGpu.empty() ? "-" : hardwareGpu;
    info.message = "Unavailable: PyTorch import failed";
    return info;
}

bool registrationAlgoCanUseGpu(cc::reg::RegAlgo a) {
    return a == cc::reg::RegAlgo::GradientSdfGpu ||
           a == cc::reg::RegAlgo::BufferX || a == cc::reg::RegAlgo::BufferXGicp ||
           a == cc::reg::RegAlgo::Rap || a == cc::reg::RegAlgo::RapGicp;
}

void drawSmallBadgeOnLastItem(const char* text, ImU32 bg, ImU32 fg) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 itemMin = ImGui::GetItemRectMin();
    const ImVec2 itemMax = ImGui::GetItemRectMax();
    const ImVec2 ts = ImGui::CalcTextSize(text);
    const ImVec2 pad{5.0f, 2.0f};
    const ImVec2 size{ts.x + pad.x * 2.0f, ts.y + pad.y * 2.0f};
    const float rowH = itemMax.y - itemMin.y;
    const float contentRight = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    const ImVec2 p{contentRight - size.x - 6.0f,
                   itemMin.y + std::max(0.0f, (rowH - size.y) * 0.5f)};
    dl->AddRectFilled(p, ImVec2{p.x + size.x, p.y + size.y}, bg, 4.0f);
    dl->AddText(ImVec2{p.x + pad.x, p.y + pad.y}, fg, text);
}
#endif

std::string withNpzExtension(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return path + ".npz";
    return path.substr(0, dot) + ".npz";
}

bool projectToScreen(const glm::mat4& view, const glm::mat4& proj, const glm::vec3& p,
                     int fbw, int fbh, ImVec2& out) {
    const glm::vec4 clip = proj * view * glm::vec4(p, 1.0f);
    if (clip.w <= 1e-6f) return false;
    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    if (ndc.z < -1.0f || ndc.z > 1.0f) return false;
    out.x = (ndc.x * 0.5f + 0.5f) * static_cast<float>(fbw);
    out.y = (1.0f - (ndc.y * 0.5f + 0.5f)) * static_cast<float>(fbh);
    return true;
}

glm::vec3 normalizeGlmOr(glm::vec3 v, glm::vec3 fallback) {
    const float len = glm::length(v);
    return len > 1e-6f ? v / len : fallback;
}

struct ViewControlFrame {
    glm::vec3 right{1, 0, 0};
    glm::vec3 forward{0, 0, -1};
    glm::vec3 up{0, 1, 0};
};

ViewControlFrame viewControlFrame(const Camera& cam, int upAxis) {
    ViewControlFrame f;
    f.up = boxSemanticUp(upAxis);
    auto projectMapPlane = [&](glm::vec3 v) {
        return v - f.up * glm::dot(v, f.up);
    };
    glm::vec3 fallbackRight = glm::cross(f.up, cam.forward());
    if (glm::length(fallbackRight) <= 1e-6f)
        fallbackRight = glm::cross(f.up, glm::vec3{0, 0, 1});
    if (glm::length(fallbackRight) <= 1e-6f)
        fallbackRight = glm::cross(f.up, glm::vec3{0, 1, 0});
    fallbackRight = normalizeGlmOr(fallbackRight, glm::vec3{1, 0, 0});

    f.right = normalizeGlmOr(projectMapPlane(cam.right()), fallbackRight);
    f.forward = normalizeGlmOr(glm::cross(f.up, f.right), cam.forward());
    const glm::vec3 viewForward = projectMapPlane(cam.forward());
    if (glm::length(viewForward) > 1e-6f && glm::dot(f.forward, viewForward) < 0.0f)
        f.forward = -f.forward;
    return f;
}

ViewControlFrame screenControlFrame(const Camera& cam) {
    ViewControlFrame f;
    f.right   = normalizeGlmOr(cam.right(), glm::vec3{1, 0, 0});
    f.up      = normalizeGlmOr(cam.screenUp(), glm::vec3{0, 1, 0});
    f.forward = normalizeGlmOr(cam.forward(), glm::vec3{0, 0, -1});
    return f;
}

enum class KeyHintMode { Move, Rotate, Resize };

glm::vec3 obbAxis(const Obb& ob, Vec3 localAxis) {
    const Vec3 a = rotate(ob.rotation, localAxis);
    return normalizeGlmOr(glm::vec3{a.x, a.y, a.z}, glm::vec3{1, 0, 0});
}

float& halfExtentByAxis(Obb& ob, int axis) {
    return axis == 0 ? ob.halfExtents.x : axis == 1 ? ob.halfExtents.y : ob.halfExtents.z;
}

const char* resizeAxisName(int axis) {
    return axis == 0 ? "length" : axis == 1 ? "width" : "height";
}

struct ResizeMatch {
    int axis = 0;
    int sign = 1;
};

ResizeMatch resizeMatchForDirection(const Obb& ob, glm::vec3 dir) {
    const glm::vec3 d = normalizeGlmOr(dir, glm::vec3{1, 0, 0});
    const glm::vec3 axes[] = {
        obbAxis(ob, Vec3{1, 0, 0}),
        obbAxis(ob, Vec3{0, 1, 0}),
        obbAxis(ob, Vec3{0, 0, 1}),
    };

    ResizeMatch match;
    float       best = -1.0f;
    for (int axis = 0; axis < 3; ++axis) {
        const float score = glm::dot(axes[axis], d);
        const float mag   = std::fabs(score);
        if (mag > best) {
            best       = mag;
            match.axis = axis;
            match.sign = score >= 0.0f ? 1 : -1;
        }
    }
    return match;
}

glm::quat normalizeQuatOr(glm::quat q, glm::quat fallback) {
    const float n2 = glm::dot(q, q);
    if (n2 <= 1e-12f || !std::isfinite(n2)) return fallback;
    return glm::normalize(q);
}

glm::quat poseRotationFromDirection(glm::vec3 dir, glm::vec3 preferredUp) {
    const glm::vec3 x = normalizeGlmOr(dir, glm::vec3{1, 0, 0});
    glm::vec3       y = preferredUp - x * glm::dot(preferredUp, x);
    if (glm::length(y) <= 1e-6f) y = fallbackPlanarAxis(x);
    y                 = normalizeGlmOr(y, glm::vec3{0, 1, 0});
    const glm::vec3 z = normalizeGlmOr(glm::cross(x, y), glm::vec3{0, 0, 1});
    y                 = normalizeGlmOr(glm::cross(z, x), y);

    glm::mat3 r(1.0f);
    r[0] = x;  // local +X is the exported object pose direction
    r[1] = y;
    r[2] = z;
    return glm::normalize(glm::quat_cast(r));
}

glm::vec3 poseDirection(const glm::quat& q) {
    return normalizeGlmOr(q * glm::vec3{1, 0, 0}, glm::vec3{1, 0, 0});
}

struct PoseGizmoVertex {
    glm::vec3 pos;
    glm::vec3 normal;
};

class PoseGizmoRenderer {
public:
    ~PoseGizmoRenderer() { release(); }

    bool build(std::string& error) {
        release();
        program_ = buildProgram(kVert, kFrag, error);
        if (!program_) return false;

        std::vector<PoseGizmoVertex> sphere;
        constexpr int   kStacks = 18;
        constexpr int   kSlices = 36;
        constexpr float kPi     = 3.14159265358979323846f;
        auto point = [](float phi, float theta) {
            const float c = std::cos(phi);
            return glm::vec3{c * std::cos(theta), std::sin(phi), c * std::sin(theta)};
        };
        sphere.reserve(kStacks * kSlices * 6);
        for (int stack = 0; stack < kStacks; ++stack) {
            const float phi0 = -0.5f * kPi + kPi * static_cast<float>(stack) / kStacks;
            const float phi1 = -0.5f * kPi + kPi * static_cast<float>(stack + 1) / kStacks;
            for (int slice = 0; slice < kSlices; ++slice) {
                const float th0 = 2.0f * kPi * static_cast<float>(slice) / kSlices;
                const float th1 = 2.0f * kPi * static_cast<float>(slice + 1) / kSlices;
                const glm::vec3 p00 = point(phi0, th0);
                const glm::vec3 p10 = point(phi1, th0);
                const glm::vec3 p11 = point(phi1, th1);
                const glm::vec3 p01 = point(phi0, th1);
                sphere.push_back({p00, p00});
                sphere.push_back({p10, p10});
                sphere.push_back({p11, p11});
                sphere.push_back({p00, p00});
                sphere.push_back({p11, p11});
                sphere.push_back({p01, p01});
            }
        }
        sphereCount_ = static_cast<GLsizei>(sphere.size());
        makeVao(sphereVao_, sphereVbo_, sphere);

        std::vector<PoseGizmoVertex> ring;
        constexpr int kRingSegments = 96;
        ring.reserve(kRingSegments + 1);
        for (int i = 0; i <= kRingSegments; ++i) {
            const float t = 2.0f * kPi * static_cast<float>(i) / kRingSegments;
            ring.push_back({glm::vec3{std::cos(t), std::sin(t), 0.0f}, glm::vec3{0.0f}});
        }
        ringCount_ = static_cast<GLsizei>(ring.size());
        makeVao(ringVao_, ringVbo_, ring);

        const std::vector<PoseGizmoVertex> axis = {
            {glm::vec3{-1.0f, 0.0f, 0.0f}, glm::vec3{0.0f}},
            {glm::vec3{ 1.0f, 0.0f, 0.0f}, glm::vec3{0.0f}},
        };
        makeVao(axisVao_, axisVbo_, axis);

        const std::vector<PoseGizmoVertex> segment = {
            {glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f}},
            {glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{0.0f}},
        };
        makeVao(segmentVao_, segmentVbo_, segment);

        std::vector<PoseGizmoVertex> cone;
        constexpr int kConeSegments = 36;
        cone.reserve(kConeSegments * 3);
        for (int i = 0; i < kConeSegments; ++i) {
            const float t0 = 2.0f * kPi * static_cast<float>(i) / kConeSegments;
            const float t1 = 2.0f * kPi * static_cast<float>(i + 1) / kConeSegments;
            const glm::vec3 p0{0.0f, std::cos(t0), std::sin(t0)};
            const glm::vec3 p1{0.0f, std::cos(t1), std::sin(t1)};
            const glm::vec3 tip{1.0f, 0.0f, 0.0f};
            cone.push_back({p0, glm::normalize(glm::vec3{1.0f, p0.y, p0.z})});
            cone.push_back({p1, glm::normalize(glm::vec3{1.0f, p1.y, p1.z})});
            cone.push_back({tip, glm::normalize(glm::vec3{1.0f, p0.y + p1.y, p0.z + p1.z})});
        }
        coneCount_ = static_cast<GLsizei>(cone.size());
        makeVao(coneVao_, coneVbo_, cone);
        return true;
    }

    void release() {
        if (sphereVbo_) { glDeleteBuffers(1, &sphereVbo_); sphereVbo_ = 0; }
        if (sphereVao_) { glDeleteVertexArrays(1, &sphereVao_); sphereVao_ = 0; }
        if (ringVbo_) { glDeleteBuffers(1, &ringVbo_); ringVbo_ = 0; }
        if (ringVao_) { glDeleteVertexArrays(1, &ringVao_); ringVao_ = 0; }
        if (axisVbo_) { glDeleteBuffers(1, &axisVbo_); axisVbo_ = 0; }
        if (axisVao_) { glDeleteVertexArrays(1, &axisVao_); axisVao_ = 0; }
        if (segmentVbo_) { glDeleteBuffers(1, &segmentVbo_); segmentVbo_ = 0; }
        if (segmentVao_) { glDeleteVertexArrays(1, &segmentVao_); segmentVao_ = 0; }
        if (coneVbo_) { glDeleteBuffers(1, &coneVbo_); coneVbo_ = 0; }
        if (coneVao_) { glDeleteVertexArrays(1, &coneVao_); coneVao_ = 0; }
        if (program_) { glDeleteProgram(program_); program_ = 0; }
        sphereCount_ = 0;
        ringCount_   = 0;
        coneCount_   = 0;
    }

    void draw(const glm::mat4& view, const glm::mat4& proj, glm::vec3 origin, glm::vec3 dir,
              const ViewControlFrame& frame, float poseLength, bool rotateMode) {
        if (!program_) return;

        const float r = std::max(poseLength * 0.28f, 0.04f);
        const glm::vec3 d = normalizeGlmOr(dir, frame.right);

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_CULL_FACE);

        glUseProgram(program_);
        glUniformMatrix4fv(glGetUniformLocation(program_, "uView"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(program_, "uProj"), 1, GL_FALSE, &proj[0][0]);

        drawSphere(origin, r, glm::vec4{0.10f, 0.55f, 1.0f, rotateMode ? 0.16f : 0.22f});

        const float axisR = r * 1.42f;
        const glm::vec4 rightCol{0.20f, 0.82f, 1.00f, rotateMode ? 0.28f : 0.78f};
        const glm::vec4 fwdCol{0.34f, 0.95f, 0.48f, rotateMode ? 0.24f : 0.70f};
        const glm::vec4 upCol{0.78f, 0.58f, 1.00f, rotateMode ? 0.24f : 0.70f};
        drawAxis(origin, frame.right, axisR, rightCol);
        drawAxis(origin, frame.forward, axisR, fwdCol);
        drawAxis(origin, frame.up, axisR, upCol);

        const float ringR = r * 1.18f;
        const glm::vec4 yawCol{1.00f, 0.70f, 0.20f, rotateMode ? 0.88f : 0.26f};
        const glm::vec4 pitchCol{1.00f, 0.48f, 0.28f, rotateMode ? 0.74f : 0.20f};
        const glm::vec4 rollCol{0.95f, 0.82f, 0.32f, rotateMode ? 0.70f : 0.18f};
        drawRing(origin, frame.right, frame.forward, ringR, yawCol);
        drawRing(origin, frame.forward, frame.up, ringR, pitchCol);
        drawRing(origin, frame.right, frame.up, ringR, rollCol);

        const float headLen = std::clamp(poseLength * 0.08f, r * 0.17f, r * 0.36f);
        const float shaftLen = std::max(poseLength - headLen * 0.78f, poseLength * 0.25f);
        drawSegment(origin, d, shaftLen, glm::vec4{1.00f, 0.86f, 0.20f, 0.96f});
        drawCone(origin + d * (poseLength - headLen), d, headLen, headLen * 0.42f,
                 glm::vec4{1.0f, 0.78f, 0.10f, 0.96f});
        drawSphere(origin, r * 0.095f, glm::vec4{1.0f, 1.0f, 1.0f, 0.96f});

        glLineWidth(1.0f);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }

private:
    static constexpr const char* kVert = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat4 uModel;
out vec3 vNormal;
void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vNormal = mat3(uModel) * aNormal;
    gl_Position = uProj * uView * world;
}
)";

    static constexpr const char* kFrag = R"(#version 330 core
in vec3 vNormal;
out vec4 FragColor;
uniform vec4 uColor;
uniform int uLit;
void main() {
    float shade = 1.0;
    if (uLit == 1) {
        vec3 n = normalize(vNormal);
        vec3 light = normalize(vec3(0.35, 0.55, 0.76));
        shade = 0.54 + 0.30 * max(dot(n, light), 0.0) + 0.16 * abs(dot(n, light));
    }
    FragColor = vec4(uColor.rgb * shade, uColor.a);
}
)";

    void makeVao(GLuint& vao, GLuint& vbo, const std::vector<PoseGizmoVertex>& verts) {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts.size() * sizeof(PoseGizmoVertex)),
                     verts.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PoseGizmoVertex),
                              reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(PoseGizmoVertex),
                              reinterpret_cast<void*>(sizeof(glm::vec3)));
        glBindVertexArray(0);
    }

    glm::mat4 sphereModel(glm::vec3 origin, float radius) const {
        glm::mat4 m(1.0f);
        m = glm::translate(m, origin);
        return glm::scale(m, glm::vec3(radius));
    }

    glm::mat4 axisModel(glm::vec3 origin, glm::vec3 axis, float radius) const {
        glm::mat4 m(1.0f);
        m[0] = glm::vec4(normalizeGlmOr(axis, glm::vec3{1, 0, 0}) * radius, 0.0f);
        m[3] = glm::vec4(origin, 1.0f);
        return m;
    }

    glm::mat4 arrowModel(glm::vec3 base, glm::vec3 axis, float length, float radius) const {
        const glm::vec3 x = normalizeGlmOr(axis, glm::vec3{1, 0, 0});
        const glm::vec3 ref = std::fabs(glm::dot(x, glm::vec3{0, 1, 0})) > 0.92f
                                  ? glm::vec3{0, 0, 1}
                                  : glm::vec3{0, 1, 0};
        const glm::vec3 z = normalizeGlmOr(glm::cross(x, ref), glm::vec3{0, 0, 1});
        const glm::vec3 y = normalizeGlmOr(glm::cross(z, x), glm::vec3{0, 1, 0});

        glm::mat4 m(1.0f);
        m[0] = glm::vec4(x * length, 0.0f);
        m[1] = glm::vec4(y * radius, 0.0f);
        m[2] = glm::vec4(z * radius, 0.0f);
        m[3] = glm::vec4(base, 1.0f);
        return m;
    }

    glm::mat4 ringModel(glm::vec3 origin, glm::vec3 a, glm::vec3 b, float radius) const {
        a = normalizeGlmOr(a, glm::vec3{1, 0, 0});
        b = normalizeGlmOr(b - a * glm::dot(a, b), glm::vec3{0, 1, 0});
        const glm::vec3 c = normalizeGlmOr(glm::cross(a, b), glm::vec3{0, 0, 1});
        glm::mat4 m(1.0f);
        m[0] = glm::vec4(a * radius, 0.0f);
        m[1] = glm::vec4(b * radius, 0.0f);
        m[2] = glm::vec4(c * radius, 0.0f);
        m[3] = glm::vec4(origin, 1.0f);
        return m;
    }

    void setMaterial(const glm::mat4& model, glm::vec4 color, bool lit) {
        glUniformMatrix4fv(glGetUniformLocation(program_, "uModel"), 1, GL_FALSE, &model[0][0]);
        glUniform4fv(glGetUniformLocation(program_, "uColor"), 1, &color[0]);
        glUniform1i(glGetUniformLocation(program_, "uLit"), lit ? 1 : 0);
    }

    void drawSphere(glm::vec3 origin, float radius, glm::vec4 color) {
        setMaterial(sphereModel(origin, radius), color, true);
        glBindVertexArray(sphereVao_);
        glDrawArrays(GL_TRIANGLES, 0, sphereCount_);
        glBindVertexArray(0);
    }

    void drawAxis(glm::vec3 origin, glm::vec3 axis, float radius, glm::vec4 color) {
        glLineWidth(2.4f);
        setMaterial(axisModel(origin, axis, radius), color, false);
        glBindVertexArray(axisVao_);
        glDrawArrays(GL_LINES, 0, 2);
        glBindVertexArray(0);
    }

    void drawSegment(glm::vec3 origin, glm::vec3 axis, float length, glm::vec4 color) {
        glLineWidth(4.0f);
        setMaterial(axisModel(origin, axis, length), color, false);
        glBindVertexArray(segmentVao_);
        glDrawArrays(GL_LINES, 0, 2);
        glBindVertexArray(0);
    }

    void drawCone(glm::vec3 base, glm::vec3 axis, float length, float radius, glm::vec4 color) {
        setMaterial(arrowModel(base, axis, length, radius), color, true);
        glBindVertexArray(coneVao_);
        glDrawArrays(GL_TRIANGLES, 0, coneCount_);
        glBindVertexArray(0);
    }

    void drawRing(glm::vec3 origin, glm::vec3 a, glm::vec3 b, float radius, glm::vec4 color) {
        glLineWidth(2.2f);
        setMaterial(ringModel(origin, a, b, radius), color, false);
        glBindVertexArray(ringVao_);
        glDrawArrays(GL_LINE_STRIP, 0, ringCount_);
        glBindVertexArray(0);
    }

    GLuint  program_ = 0;
    GLuint  sphereVao_ = 0, sphereVbo_ = 0;
    GLuint  ringVao_ = 0, ringVbo_ = 0;
    GLuint  axisVao_ = 0, axisVbo_ = 0;
    GLuint  segmentVao_ = 0, segmentVbo_ = 0;
    GLuint  coneVao_ = 0, coneVbo_ = 0;
    GLsizei sphereCount_ = 0;
    GLsizei ringCount_ = 0;
    GLsizei coneCount_ = 0;
};

void drawArrowLabel(ImDrawList* dl, const glm::mat4& view, const glm::mat4& proj,
                    int fbw, int fbh, const glm::vec3& anchor, const glm::vec3& dir,
                    float probeLen, const char* label, ImU32 color) {
    ImVec2 a, probe;
    if (!projectToScreen(view, proj, anchor, fbw, fbh, a))
        return;

    ImVec2 v{1.0f, 0.0f};
    if (projectToScreen(view, proj, anchor + normalizeGlmOr(dir, glm::vec3{1, 0, 0}) * probeLen,
                        fbw, fbh, probe)) {
        v = ImVec2{probe.x - a.x, probe.y - a.y};
    }
    float len = std::sqrt(v.x * v.x + v.y * v.y);
    if (len <= 3.0f) {
        const ImVec2 center{static_cast<float>(fbw) * 0.5f, static_cast<float>(fbh) * 0.5f};
        v   = ImVec2{a.x - center.x, a.y - center.y};
        len = std::sqrt(v.x * v.x + v.y * v.y);
    }
    if (len <= 3.0f) {
        v   = ImVec2{1.0f, 0.0f};
        len = 1.0f;
    }

    const ImVec2 n{v.x / len, v.y / len};
    const ImVec2 p{-n.y, n.x};
    const ImVec2 b{a.x + n.x * 46.0f, a.y + n.y * 46.0f};
    dl->AddCircleFilled(a, 3.5f, color);
    dl->AddLine(a, b, color, 2.0f);
    const float s = 7.0f;
    dl->AddTriangleFilled(b, ImVec2{b.x - n.x * s + p.x * s * 0.55f,
                                    b.y - n.y * s + p.y * s * 0.55f},
                          ImVec2{b.x - n.x * s - p.x * s * 0.55f,
                                 b.y - n.y * s - p.y * s * 0.55f},
                          color);

    const ImVec2 text = ImGui::CalcTextSize(label);
    const ImVec2 pos{b.x + n.x * 7.0f + p.x * 3.0f,
                     b.y + n.y * 7.0f + p.y * 3.0f - text.y * 0.5f};
    const ImVec2 pad{5.0f, 3.0f};
    dl->AddRectFilled(ImVec2{pos.x - pad.x, pos.y - pad.y},
                      ImVec2{pos.x + text.x + pad.x, pos.y + text.y + pad.y},
                      IM_COL32(18, 20, 24, 210), 4.0f);
    dl->AddText(pos, IM_COL32(245, 247, 250, 255), label);
}

void drawPillLabel(ImDrawList* dl, const glm::mat4& view, const glm::mat4& proj,
                   int fbw, int fbh, glm::vec3 pos3, const char* label, ImU32 bg, ImU32 fg) {
    ImVec2 p;
    if (!projectToScreen(view, proj, pos3, fbw, fbh, p))
        return;

    const ImVec2 text = ImGui::CalcTextSize(label);
    const ImVec2 pad{5.0f, 3.0f};
    const ImVec2 pos{p.x - text.x * 0.5f, p.y - text.y * 0.5f};
    dl->AddRectFilled(ImVec2{pos.x - pad.x, pos.y - pad.y},
                      ImVec2{pos.x + text.x + pad.x, pos.y + text.y + pad.y},
                      bg, 4.0f);
    dl->AddText(pos, fg, label);
}

void drawPoseMoveHints(const glm::mat4& view, const glm::mat4& proj, int fbw, int fbh,
                       glm::vec3 origin, const ViewControlFrame& frame, float radius) {
    struct Hint {
        const char* label;
        glm::vec3   dir;
    };
    const Hint hints[] = {
        {"D", frame.right},
        {"A", -frame.right},
        {"W", frame.forward},
        {"S", -frame.forward},
        {"Q", frame.up},
        {"E", -frame.up},
    };

    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImU32 bg = IM_COL32(18, 36, 48, 220);
    const ImU32 fg = IM_COL32(210, 245, 255, 255);
    const float labelRadius = std::max(radius * 0.44f, 0.06f);
    for (const Hint& h : hints) {
        const glm::vec3 d = normalizeGlmOr(h.dir, glm::vec3{1, 0, 0});
        drawPillLabel(dl, view, proj, fbw, fbh, origin + d * labelRadius, h.label, bg, fg);
    }
}

void drawPoseRotationSphere(const glm::mat4& view, const glm::mat4& proj, int fbw, int fbh,
                            glm::vec3 origin, const ViewControlFrame& frame, float radius) {
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const float r = std::max(radius * 0.45f, 0.07f);
    const ImU32 bg = IM_COL32(34, 24, 14, 225);
    const ImU32 fg = IM_COL32(255, 245, 220, 255);

    struct Hint {
        const char* label;
        glm::vec3   dir;
    };
    const Hint hints[] = {
        {"pitch W/S", frame.right},
        {"roll A/D", frame.forward},
        {"yaw Q/E", frame.up},
    };
    for (const Hint& h : hints)
        drawPillLabel(dl, view, proj, fbw, fbh,
                      origin + normalizeGlmOr(h.dir, glm::vec3{1, 0, 0}) * r,
                      h.label, bg, fg);
}

void drawBoxKeyHints(const Obb& ob, const glm::mat4& view, const glm::mat4& proj,
                     int fbw, int fbh, const ViewControlFrame& frame, KeyHintMode mode) {
    struct Control {
        const char* label;
        glm::vec3   dir;
    };
    struct Face {
        glm::vec3 normal;
        float     halfExtent;
        int       axis;
        int       sign;
    };

    const glm::vec3 x = obbAxis(ob, Vec3{1, 0, 0});
    const glm::vec3 y = obbAxis(ob, Vec3{0, 1, 0});
    const glm::vec3 z = obbAxis(ob, Vec3{0, 0, 1});
    const std::vector<Face> faces = {
        {x, ob.halfExtents.x, 0, 1},   {-x, ob.halfExtents.x, 0, -1},
        {y, ob.halfExtents.y, 1, 1},   {-y, ob.halfExtents.y, 1, -1},
        {z, ob.halfExtents.z, 2, 1},   {-z, ob.halfExtents.z, 2, -1},
    };
    std::vector<Control> controls;
    ImU32 color = IM_COL32(95, 205, 255, 235);

    if (mode == KeyHintMode::Resize) {
        color = IM_COL32(120, 230, 150, 240);
        controls = {{"D", frame.right}, {"A", -frame.right},
                    {"W", frame.forward}, {"S", -frame.forward},
                    {"Q", frame.up}, {"E", -frame.up}};
    } else if (mode == KeyHintMode::Rotate) {
        color = IM_COL32(255, 190, 80, 240);
        controls = {{"D: yaw", frame.right},    {"A: yaw", -frame.right},
                    {"W: pitch", frame.forward}, {"S: pitch", -frame.forward},
                    {"Q: roll", frame.up},       {"E: roll", -frame.up}};
    } else {
        controls = {{"D: right move", frame.right}, {"A: left move", -frame.right},
                    {"W: forward move", frame.forward}, {"S: back move", -frame.forward},
                    {"Q: up move", frame.up}, {"E: down move", -frame.up}};
    }

    const glm::vec3 center{ob.center.x, ob.center.y, ob.center.z};
    const float maxHalf = std::max({ob.halfExtents.x, ob.halfExtents.y, ob.halfExtents.z, 0.05f});
    const float pad = std::max(maxHalf * 0.35f, 0.03f);
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    for (const Face& face : faces) {
        const glm::vec3 normal = normalizeGlmOr(face.normal, glm::vec3{1, 0, 0});
        const char* label = nullptr;
        std::string resizeLabel;
        if (mode == KeyHintMode::Resize) {
            float best = -2.0f;
            for (const Control& c : controls) {
                const float score = glm::dot(normal, normalizeGlmOr(c.dir, glm::vec3{1, 0, 0}));
                if (score > best) {
                    best = score;
                    label = c.label;
                }
            }
            resizeLabel = std::string(label ? label : "?") + ": " + resizeAxisName(face.axis) +
                          (face.sign > 0 ? "+" : "-");
            label = resizeLabel.c_str();
        } else {
            float best = -2.0f;
            for (const Control& c : controls) {
                const float score = glm::dot(normal, normalizeGlmOr(c.dir, glm::vec3{1, 0, 0}));
                if (score > best) {
                    best = score;
                    label = c.label;
                }
            }
        }
        drawArrowLabel(dl, view, proj, fbw, fbh, center + normal * face.halfExtent,
                       normal, pad, label, color);
    }
}

void drawMetadataPose(PoseGizmoRenderer& poseGizmo, const PointCloud& pc, const glm::mat4& model,
                      const glm::mat4& view, const glm::mat4& proj,
                      const ViewControlFrame& frame) {
    auto origin = metadataVec3(pc, "object_pose_origin_local");
    auto dir    = metadataVec3(pc, "object_pose_dir_local");
    if (!origin || !dir) return;

    const glm::vec3 worldOrigin = glm::vec3(model * glm::vec4(*origin, 1.0f));
    const glm::vec3 worldDir =
        normalizeGlmOr(glm::mat3(model) * *dir, normalizeGlmOr(*dir, glm::vec3{1, 0, 0}));

    float length = 0.1f;
    const Aabb b = pc.bounds();
    if (b.valid()) {
        const glm::vec3 minp = glm::vec3(model * glm::vec4(b.min.x, b.min.y, b.min.z, 1.0f));
        const glm::vec3 maxp = glm::vec3(model * glm::vec4(b.max.x, b.max.y, b.max.z, 1.0f));
        length = std::max(glm::length(maxp - minp) * 0.35f, 0.1f);
    }

    poseGizmo.draw(view, proj, worldOrigin, worldDir, frame, length, false);
}

// Mouse drag state for the orbit camera.
struct InputState {
    bool   lDown = false, rDown = false, mDown = false;
    double lastX = 0, lastY = 0;
};

// Accumulated scroll delta, drained each frame (set from the GLFW callback).
double g_scroll = 0.0;
void   scrollCallback(GLFWwindow* w, double xoff, double yoff) {
    ImGui_ImplGlfw_ScrollCallback(w, xoff, yoff);  // keep ImGui scrolling working
    if (!ImGui::GetIO().WantCaptureMouse) g_scroll += yoff;
}

// Files dropped onto the window, drained each frame (set from the GLFW callback).
std::vector<std::string> g_dropped;
void                     dropCallback(GLFWwindow*, int count, const char** paths) {
    for (int i = 0; i < count; ++i) g_dropped.emplace_back(paths[i]);
}

// A clean, modern dark theme (rounded corners, generous spacing, soft palette
// with one indigo accent) so the viewer reads like a contemporary web app rather
// than default ImGui. Loads a proportional system font if one is available.
void applyModernTheme() {
    ImGuiStyle& s = ImGui::GetStyle();

    // geometry: rounded + roomy + (mostly) borderless
    s.WindowRounding          = 9.0f;
    s.ChildRounding           = 8.0f;
    s.FrameRounding           = 7.0f;
    s.PopupRounding           = 7.0f;
    s.ScrollbarRounding       = 9.0f;
    s.GrabRounding            = 7.0f;
    s.WindowBorderSize        = 0.0f;
    s.FrameBorderSize         = 0.0f;
    s.PopupBorderSize         = 1.0f;
    s.WindowPadding           = ImVec2(14.0f, 11.0f);
    s.FramePadding             = ImVec2(10.0f, 5.0f);
    s.CellPadding             = ImVec2(8.0f, 4.0f);
    s.ItemSpacing             = ImVec2(10.0f, 6.0f);
    s.ItemInnerSpacing        = ImVec2(8.0f, 5.0f);
    s.IndentSpacing           = 18.0f;
    s.ScrollbarSize           = 12.0f;
    s.GrabMinSize             = 11.0f;
    s.SeparatorTextBorderSize = 1.0f;
    s.SeparatorTextPadding    = ImVec2(18.0f, 6.0f);
    s.SeparatorTextAlign      = ImVec2(0.0f, 0.5f);
    s.DisabledAlpha           = 0.42f;

    // palette: soft dark surfaces, one indigo accent
    const ImVec4 bg      = ImVec4(0.105f, 0.110f, 0.130f, 1.00f);
    const ImVec4 panel   = ImVec4(0.165f, 0.173f, 0.200f, 1.00f);
    const ImVec4 panelHi = ImVec4(0.205f, 0.215f, 0.250f, 1.00f);
    const ImVec4 panelHo = ImVec4(0.235f, 0.245f, 0.285f, 1.00f);
    const ImVec4 accent  = ImVec4(0.380f, 0.470f, 0.960f, 1.00f);
    const ImVec4 accentH = ImVec4(0.470f, 0.555f, 1.000f, 1.00f);
    const ImVec4 textc   = ImVec4(0.900f, 0.910f, 0.935f, 1.00f);
    const ImVec4 textDim = ImVec4(0.470f, 0.490f, 0.555f, 1.00f);
    const ImVec4 border  = ImVec4(1.000f, 1.000f, 1.000f, 0.065f);
    const ImVec4 clear   = ImVec4(0.000f, 0.000f, 0.000f, 0.000f);

    ImVec4* c                        = s.Colors;
    c[ImGuiCol_Text]                 = textc;
    c[ImGuiCol_TextDisabled]         = textDim;
    c[ImGuiCol_WindowBg]             = bg;
    c[ImGuiCol_ChildBg]              = clear;
    c[ImGuiCol_PopupBg]              = ImVec4(0.140f, 0.147f, 0.172f, 0.98f);
    c[ImGuiCol_Border]               = border;
    c[ImGuiCol_BorderShadow]         = clear;
    c[ImGuiCol_FrameBg]              = panel;
    c[ImGuiCol_FrameBgHovered]       = panelHi;
    c[ImGuiCol_FrameBgActive]        = panelHo;
    c[ImGuiCol_TitleBg]              = bg;
    c[ImGuiCol_TitleBgActive]        = bg;
    c[ImGuiCol_TitleBgCollapsed]     = bg;
    c[ImGuiCol_MenuBarBg]            = bg;
    c[ImGuiCol_ScrollbarBg]          = clear;
    c[ImGuiCol_ScrollbarGrab]        = ImVec4(0.30f, 0.32f, 0.38f, 1.0f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.38f, 0.40f, 0.47f, 1.0f);
    c[ImGuiCol_ScrollbarGrabActive]  = accent;
    c[ImGuiCol_CheckMark]            = accent;
    c[ImGuiCol_SliderGrab]           = accent;
    c[ImGuiCol_SliderGrabActive]     = accentH;
    c[ImGuiCol_Button]               = panel;
    c[ImGuiCol_ButtonHovered]        = panelHi;
    c[ImGuiCol_ButtonActive]         = panelHo;
    c[ImGuiCol_Header]               = panel;
    c[ImGuiCol_HeaderHovered]        = panelHi;
    c[ImGuiCol_HeaderActive]         = panelHo;
    c[ImGuiCol_Separator]            = border;
    c[ImGuiCol_SeparatorHovered]     = accent;
    c[ImGuiCol_SeparatorActive]      = accent;
    c[ImGuiCol_ResizeGrip]           = ImVec4(0.30f, 0.32f, 0.38f, 0.5f);
    c[ImGuiCol_ResizeGripHovered]    = accent;
    c[ImGuiCol_ResizeGripActive]     = accentH;
    c[ImGuiCol_TextSelectedBg]       = ImVec4(accent.x, accent.y, accent.z, 0.35f);
    c[ImGuiCol_DragDropTarget]       = accent;
    c[ImGuiCol_TableHeaderBg]        = ImVec4(0.150f, 0.157f, 0.182f, 1.0f);
    c[ImGuiCol_TableRowBg]           = clear;
    c[ImGuiCol_TableRowBgAlt]        = ImVec4(1.0f, 1.0f, 1.0f, 0.022f);
    c[ImGuiCol_TableBorderLight]     = border;
    c[ImGuiCol_TableBorderStrong]    = border;

    // proportional system font (much cleaner than the default bitmap font)
    const char* fonts[] = {
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    };
    ImGuiIO& io   = ImGui::GetIO();
    io.IniFilename = nullptr;  // always use the designed default size; don't litter cwd
    for (const char* path : fonts) {
        if (FILE* fp = std::fopen(path, "rb")) {
            std::fclose(fp);
            io.Fonts->AddFontFromFileTTF(path, 16.0f);
            break;
        }
    }
}

}  // namespace

Result<void> runViewer(const io::FormatRegistry& registry, const ViewerOptions& options,
                       LoadCloudFn load) {
    if (!glfwInit()) {
        return makeError(ErrorCode::IoError, "glfwInit failed (no display?)");
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    // Hide the window in headless screenshot mode so a WM is not required.
    const bool headless = options.frames > 0;
    if (headless) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(options.width, options.height, options.title.c_str(),
                                          nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return makeError(ErrorCode::IoError, "glfwCreateWindow failed (no GL 3.3?)");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(headless ? 0 : 1);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return makeError(ErrorCode::IoError, "glad failed to load GL");
    }
    // Which device/driver is actually rendering (hybrid-GPU laptops often land
    // on the iGPU or even llvmpipe unless PRIME offload env vars are set).
    std::cerr << "viewer: GL renderer: "
              << reinterpret_cast<const char*>(glGetString(GL_RENDERER)) << "  ("
              << reinterpret_cast<const char*>(glGetString(GL_VERSION)) << ")\n";

    // --- renderers (the box geometry is cloud-independent; points rebuild per load) ---
    PointRenderer points;
    BoxRenderer   boxes;
    PoseGizmoRenderer poseGizmo;
    std::string   err;

    // --- crop-preview state (the OK/Cancel spinning preview before exporting) ---
    PointRenderer previewPoints;          // GL buffers for the cropped subset
    Camera        previewCam;             // independent camera, framed on the crop
    bool          cropPreview = false;    // true while the preview overlay is shown
    bool          cropPreviewFromPose = false;  // preview opened from Pose Setup, no commit
    bool          previewReturning = false;  // easing back to the initial pose after a drag
    glm::vec3     previewInitialOrbit{0.0f}; // stored yaw/pitch/roll targets of the initial pose
    glm::vec3     previewBmin{-1, -1, -1}, previewBmax{1, 1, 1};
    bool          previewLDown = false;   // LMB state inside the preview (for press->release edge)
    CropSpec      previewSpec;            // the spec captured when entering preview (reused on OK)

    // Optional object-pose workflow: crop first, then assign one origin+orientation
    // to the cropped cloud before writing the NPZ template. Export keeps the
    // current schema by deriving object_pose_dir_local from the quaternion.
    bool          addObjectPose = false;
    bool          poseSetup     = false;
    PointCloud    poseCloud;
    glm::vec3     poseOrigin{0.0f, 0.0f, 0.0f};
    glm::quat     poseRot{1.0f, 0.0f, 0.0f, 0.0f};
    float         poseMoveSpeed = 0.25f;
    if (!boxes.build(err)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return makeError(ErrorCode::IoError, "box shader: " + err);
    }
    if (!poseGizmo.build(err)) {
        boxes.release();
        glfwDestroyWindow(window);
        glfwTerminate();
        return makeError(ErrorCode::IoError, "pose gizmo shader: " + err);
    }

    // --- mutable scene state (starts empty; filled by applyCloud) ---
    PointCloud           cloud;            // the full-res cloud held here, swapped on load
    Camera               cam;
    Aabb                 sceneBounds;      // bounds of the current cloud (invalid if empty)
    glm::vec3            bmin{-1, -1, -1}, bmax{1, 1, 1};
    std::vector<EditBox> editBoxes;
    int                  selected   = 0;
    BoolOp               combineOp  = BoolOp::Union;
    int                  gizmoOp    = 0;  // 0 translate, 1 rotate, 2 scale
    float                pointSize  = options.pointSize;
    int                  budgetM    = 8;  // display point budget, millions (LOD)
    int                  colorMode  = static_cast<int>(ColorMode::Height);
    std::uint64_t        keptCount  = 0;
    bool                 dirty      = true;
    std::string          status;
    int                  upAxis     = 1;  // 0=X, 1=Y, 2=Z (world up for the camera)

    auto applyUp = [&](int ax) {
        upAxis = ax;
        cam.setUp(boxSemanticUp(upAxis));
        for (EditBox& b : editBoxes)
            enforceUprightBox(b.obb, upAxis);
        dirty = true;
    };

    // ROS2 bag navigation state (active only when a .db3/bag is loaded).
    bool                       bagMode = false;
    std::string                bagPath;
    std::vector<std::string>   bagTopics;   // PointCloud2 topic names
    std::vector<std::uint64_t> bagCounts;   // message count per topic
    int                        bagTopicIdx = 0;
    int                        bagFrame    = 0;
    bool                       bagPlay     = false;
    float                      bagFps      = 5.0f;
    double                     bagLastStep = 0.0;

#if defined(CLOUDCROPPER_HAS_REGISTRATION)
    // --- registration panel state (right side). Source = the open cloud;
    //     target = a separately loaded cloud (e.g. the last Crop+Export). The
    //     solve runs on a worker thread over COPIES so the UI stays live. ---
    PointCloud    targetCloud;
    PointRenderer targetPoints;
    bool          showTarget      = true;
    char          regTargetPath[512]  = {0};
    char          regAlignedPath[512] = "aligned.ply";
    std::string   lastExportPath;       // filled by a successful Crop+Export
    int           regAlgoIdx   = 0;     // index into kRegAlgos below
    float         regDownsample = 0.0f, regMaxCorr = 0.0f, regKissRes = 0.0f;
    float         regBufferxVoxel = 0.0f, regRapVoxel = 0.0f;
    int           regThreads = 4, regSdfRes = 100;
    bool          regRefine = true, regUncertainty = true;
    std::thread        regThread;
    std::atomic<int>   regState{0};  // 0 idle, 1 running, 2 done, 3 failed
    std::mutex         regMutex;     // guards regResult/regError handoff
    cc::reg::RegResult regResult;    // worker-owned; copied out under the mutex
    cc::reg::RegResult regShown;     // UI-owned copy displayed in the panel
    std::string        regError;
    glm::mat4          regT{1.0f};   // result transform for the live overlay
    bool               regHasResult = false;
    bool               regOverlay   = true;  // draw source through regT
    double             regStarted   = 0.0;
    std::string        regStatus;   // feedback shown INSIDE the panel
    std::thread        gpuProbeThread;
    std::mutex         gpuProbeMutex;
    GpuProbeInfo       gpuProbe;

    struct RegAlgoEntry { const char* name; cc::reg::RegAlgo algo; };
    static constexpr RegAlgoEntry kRegAlgos[] = {
        {"GICP", cc::reg::RegAlgo::Gicp},
        {"VGICP", cc::reg::RegAlgo::VGicp},
        {"ICP (point)", cc::reg::RegAlgo::Icp},
        {"ICP (plane)", cc::reg::RegAlgo::PlaneIcp},
#if defined(CLOUDCROPPER_HAS_KISS_MATCHER)
        {"KISS-Matcher", cc::reg::RegAlgo::KissMatcher},
        {"KISS + GICP", cc::reg::RegAlgo::KissGicp},
#endif
        {"gradient-SDF (GPU)", cc::reg::RegAlgo::GradientSdfGpu},
        {"BUFFER-X", cc::reg::RegAlgo::BufferX},
        {"BUFFER-X + GICP", cc::reg::RegAlgo::BufferXGicp},
        {"G3Reg", cc::reg::RegAlgo::G3Reg},
        {"G3Reg + GICP", cc::reg::RegAlgo::G3RegGicp},
        {"RAP", cc::reg::RegAlgo::Rap},
        {"RAP + GICP", cc::reg::RegAlgo::RapGicp},
    };
    constexpr int kRegAlgoCount = static_cast<int>(sizeof(kRegAlgos) / sizeof(kRegAlgos[0]));

    // Pull the package defaults (config/<pkg>.yaml) into the UI fields; runs at
    // startup and whenever the algorithm combo changes.
    auto loadRegDefaults = [&](int algoIdx) {
        const cc::reg::RegOptions d =
            cc::reg::defaultsFor(kRegAlgos[static_cast<std::size_t>(algoIdx)].algo);
        regDownsample  = d.downsample;
        regMaxCorr     = d.maxCorr;
        regThreads     = d.threads;
        regKissRes     = d.kissResolution;
        regSdfRes      = d.sdfResolution;
        regBufferxVoxel = d.bufferxVoxel;
        regRapVoxel    = d.rapVoxel;
        regRefine      = d.refine;
        regUncertainty = d.sdfUncertainty;
    };
    loadRegDefaults(regAlgoIdx);

    gpuProbeThread = std::thread([&gpuProbeMutex, &gpuProbe]() {
        GpuProbeInfo info = probeGpu();
        std::lock_guard<std::mutex> lk(gpuProbeMutex);
        gpuProbe = std::move(info);
    });
#endif

    // Re-derive everything that depends on the cloud, and rebuild the GPU buffers.
    // reframe=false keeps the camera + boxes (used when stepping bag frames so the
    // view doesn't jump every frame); only the point buffer + preview refresh.
    auto applyCloud = [&](PointCloud&& pc, bool reframe = true) {
        cloud = std::move(pc);
        std::string e;
        points.build(cloud, e);  // idempotent: safe to call on every load
        sceneBounds = cloud.bounds();
        if (sceneBounds.valid()) {
            bmin = {sceneBounds.min.x, sceneBounds.min.y, sceneBounds.min.z};
            bmax = {sceneBounds.max.x, sceneBounds.max.y, sceneBounds.max.z};
        } else {
            bmin = {-1, -1, -1};
            bmax = {1, 1, 1};
        }
        if (reframe) {
            cam.fit(bmin, bmax);
            EditBox b;
            if (sceneBounds.valid()) {
                b.obb             = uprightObbFromAabb(sceneBounds, upAxis);
                b.obb.halfExtents = b.obb.halfExtents * 0.5f;
            } else {
                b.obb.center      = {0, 0, 0};
                b.obb.halfExtents = {0.5f, 0.5f, 0.5f};
                enforceUprightBox(b.obb, upAxis);
            }
            editBoxes = {b};
            selected  = 0;
            colorMode = points.hasRgb() ? static_cast<int>(ColorMode::Rgb)
                                        : static_cast<int>(ColorMode::Height);
        }
        keptCount = cloud.size();
        dirty     = true;
    };

    // Load the current bag topic + frame (bag mode only). reframe=true re-fits the
    // camera (use on initial load / topic change); false keeps the view (stepping).
    auto loadBagFrame = [&](bool reframe) {
#if defined(CLOUDCROPPER_HAS_ROSBAG)
        if (bagTopics.empty()) return;
        cc::io::BagReadOptions o;
        o.topic = bagTopics[bagTopicIdx];
        o.frame = bagFrame;
        auto r  = cc::io::readRosbag(bagPath, o);
        if (!r) {
            status = "bag frame failed: " + r.error().message;
            return;
        }
        status = "frame " + std::to_string(bagFrame + 1) + "/" +
                 std::to_string(bagCounts[bagTopicIdx]) + "  " + bagTopics[bagTopicIdx];
        applyCloud(std::move(r.value()), reframe);
#else
        (void)reframe;
#endif
    };

    // Load a path: a ROS2 bag (→ topic/frame navigation) or a file (→ caller's loader).
    auto tryLoad = [&](const std::string& path) {
        bagMode = false;
        bagPath = path;
#if defined(CLOUDCROPPER_HAS_ROSBAG)
        if (cc::io::isRosbagPath(path)) {
            auto topics = cc::io::listBagTopics(path);
            if (!topics) {
                status = "bag: " + topics.error().message;
                return;
            }
            bagTopics.clear();
            bagCounts.clear();
            for (const auto& t : topics.value())
                if (t.type == "sensor_msgs/msg/PointCloud2") {
                    bagTopics.push_back(t.name);
                    bagCounts.push_back(t.count);
                }
            if (bagTopics.empty()) {
                status = "bag has no PointCloud2 topics";
                return;
            }
            bagMode     = true;
            bagTopicIdx = 0;
            bagFrame    = 0;
            bagPlay     = false;
            applyUp(2);          // ROS clouds are Z-up (REP-103) by default
            loadBagFrame(true);  // reframe on first load
            return;
        }
#endif
        applyUp(1);  // files default to Y-up
        auto r = load(path);
        if (!r) {
            status = "load failed: " + r.error().message;
            return;
        }
        status = "loaded " + std::to_string(r->size()) + " points";
        applyCloud(std::move(r.value()));
    };

    applyCloud(PointCloud{});                                  // start empty
    if (!options.initialPath.empty()) tryLoad(options.initialPath);

    // --- ImGui + ImGuizmo ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();  // sane baseline for any color applyModernTheme leaves unset
    applyModernTheme();        // modern rounded/soft-dark look + system font
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
    glfwSetScrollCallback(window, scrollCallback);  // chains to ImGui (above)
    glfwSetDropCallback(window, dropCallback);      // drag-and-drop file loading

    InputState input;
    char       exportPath[512] = "crop.ply";
    int        exportEncoding  = 0;    // 0 binary, 1 ascii
    char       pathBuf[512]    = "";   // manual path entry (Load button / fallback)

    auto forceNpzExportPath = [&]() {
        const std::string npz = withNpzExtension(exportPath);
        std::snprintf(exportPath, sizeof(exportPath), "%s", npz.c_str());
    };

#if defined(CLOUDCROPPER_HAS_NPZ)
    auto makeTemplateMeta = [&](const PointCloud& pc, Vec3 poseOriginLocal,
                                Vec3 poseDirLocal) {
        io::TemplateMeta tm;
        const Aabb       b = pc.bounds();
        if (b.valid()) {
            tm.bbox_min    = b.min;
            tm.bbox_max    = b.max;
            tm.bbox_center = (b.min + b.max) * 0.5f;
        }
        const CanonicalFrame cf = pcaFrame(pc);
        tm.canonical_center     = cf.center;
        tm.canonical_axis       = cf.axis;
        tm.point_spacing_m      = estimatePointSpacing(pc);
        tm.object_pose_origin_local = poseOriginLocal;
        tm.object_pose_dir_local    = normalizedOr(poseDirLocal, Vec3{1, 0, 0});
        return tm;
    };

    auto enterPoseSetup = [&](PointCloud&& cropped) {
        forceNpzExportPath();
        poseCloud  = std::move(cropped);
        poseOrigin = glm::vec3{0.0f, 0.0f, 0.0f};
        int boxIdx = -1;
        if (selected >= 0 && selected < static_cast<int>(editBoxes.size()) &&
            editBoxes[static_cast<std::size_t>(selected)].enabled &&
            editBoxes[static_cast<std::size_t>(selected)].role == BoxRole::Include) {
            boxIdx = selected;
        } else {
            for (int i = 0; i < static_cast<int>(editBoxes.size()); ++i) {
                if (editBoxes[static_cast<std::size_t>(i)].enabled &&
                    editBoxes[static_cast<std::size_t>(i)].role == BoxRole::Include) {
                    boxIdx = i;
                    break;
                }
            }
        }
        Vec3 d = boxIdx >= 0 ? rotate(editBoxes[static_cast<std::size_t>(boxIdx)].obb.rotation,
                                      Vec3{1, 0, 0})
                             : pcaFrame(poseCloud).axis;
        poseRot = poseRotationFromDirection(toGlm(normalizedOr(d, Vec3{1, 0, 0})),
                                            boxSemanticUp(upAxis));

        std::string pe;
        previewPoints.build(poseCloud, pe);
        Aabb pb = poseCloud.bounds();
        if (pb.valid()) {
            previewBmin = {pb.min.x, pb.min.y, pb.min.z};
            previewBmax = {pb.max.x, pb.max.y, pb.max.z};
        } else {
            previewBmin = {-1, -1, -1};
            previewBmax = {1, 1, 1};
        }
        previewCam.setUp(boxSemanticUp(upAxis));
        previewCam.fit(previewBmin, previewBmax);
        previewCam.update(0.0f);
        previewInitialOrbit = previewCam.orbitTarget();
        previewReturning    = false;
        previewLDown        = false;
        poseSetup           = true;
        status              = std::string("set object pose -> ") + exportPath;
    };
#endif

    auto isOverUi = [&]() { return ImGui::GetIO().WantCaptureMouse || ImGuizmo::IsOver(); };

    glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
    glEnable(GL_DEPTH_TEST);  // depth-order points + box wireframe (ImGui restores its own state)

    int    frameCount = 0;
    double lastTime   = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // ---- drag-and-drop: load the most recent dropped file ----
        if (!g_dropped.empty()) {
            tryLoad(g_dropped.back());
            g_dropped.clear();
        }

#if defined(CLOUDCROPPER_HAS_ROSBAG)
        // ---- bag playback: advance one frame at the chosen rate ----
        if (bagMode && bagPlay && !bagTopics.empty()) {
            const double t = glfwGetTime();
            if (t - bagLastStep > 1.0 / std::max(bagFps, 0.1f)) {
                const int total = static_cast<int>(bagCounts[bagTopicIdx]);
                bagFrame = (total > 0) ? (bagFrame + 1) % total : 0;
                loadBagFrame(false);
                bagLastStep = t;
            }
        }
#endif

        // ---- frame timestep (shared by camera easing + preview auto-spin) ----
        const double now = glfwGetTime();
        const float  dt  = static_cast<float>(now - lastTime);
        lastTime         = now;

        // ---- camera input (only when not interacting with UI/gizmo) ----
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        const bool l = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        const bool r = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        const bool m = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        const float dx = static_cast<float>(mx - input.lastX);
        const float dy = static_cast<float>(my - input.lastY);
        const bool alt = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                         glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
        if (cropPreview || poseSetup) {
            // ---- preview camera: auto-yaw spin with drag override + ease-back ----
            const bool overUi = isOverUi();
            const bool leftDragging = l && previewLDown && !overUi;
            bool previewOrbitDragging = false;
            if (poseSetup && !cropPreview) {
                if (leftDragging) {
                    previewCam.orbit(dx, dy, false);
                }
                if (((r && input.rDown) || (m && input.mDown)) && !overUi) {
                    previewCam.pan(dx, dy);
                }
            } else if (leftDragging) {
                previewCam.orbit(dx, dy, false);  // drag takes over; spin paused
                previewReturning = false;
                previewOrbitDragging = true;
            } else if (cropPreview && previewLDown && !l) {
                // press -> release edge: ease back to the initial preview pose.
                previewReturning = true;
                previewCam.setOrbitTarget(previewInitialOrbit);
            }
            if (cropPreview && previewReturning) {
                // Keep nudging the target back (cheap) until we settle, then resume.
                previewCam.setOrbitTarget(previewInitialOrbit);
                if (previewCam.nearOrbitTarget(1e-3f)) previewReturning = false;
            } else if (cropPreview && !previewOrbitDragging) {
                previewCam.nudgeYaw(0.4f * dt);
            }
            if (g_scroll != 0.0 && !overUi) previewCam.dolly(static_cast<float>(g_scroll));
            previewLDown = l;
            input.lDown = l;
            input.rDown = r;
            input.mDown = m;
        } else {
            if (!isOverUi()) {
                // Drag = orbit (horizontal yaw, vertical pitch); Alt+drag = roll.
                if (l && input.lDown) cam.orbit(dx, dy, alt);
                if ((r && input.rDown) || (m && input.mDown)) cam.pan(dx, dy);
            }
            input.lDown = l;
            input.rDown = r;
            input.mDown = m;
            if (g_scroll != 0.0 && !isOverUi()) cam.dolly(static_cast<float>(g_scroll));
            if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS && !ImGui::GetIO().WantCaptureKeyboard)
                cam.fit(bmin, bmax);
        }
        input.lastX = mx;
        input.lastY = my;
        g_scroll = 0.0;

        // ---- WASD / QE: move/rotate/resize the selected crop box.
        //      WASD = map-plane forward/left/back/right from the current view,
        //      Q/E = map up/down.
        //      With Alt held: A/D yaw, W/S pitch, Q/E roll around view-frame axes.
        //      With Ctrl held: resize the box axis whose face matches the view key
        //      direction (A/D left/right, W/S forward/back, Q/E up/down).
        //      Hold Shift to go faster.
        //      Speeds are per-second (dt-scaled) so they're frame-rate independent.
        if (!cropPreview && !poseSetup && !ImGui::GetIO().WantCaptureKeyboard && selected >= 0 &&
            selected < static_cast<int>(editBoxes.size()) &&
            editBoxes[static_cast<std::size_t>(selected)].enabled) {
            auto down = [&](int key) { return glfwGetKey(window, key) == GLFW_PRESS; };
            const float kx = (down(GLFW_KEY_D) ? 1.0f : 0.0f) - (down(GLFW_KEY_A) ? 1.0f : 0.0f);
            const float ky = (down(GLFW_KEY_W) ? 1.0f : 0.0f) - (down(GLFW_KEY_S) ? 1.0f : 0.0f);
            const float kz = (down(GLFW_KEY_Q) ? 1.0f : 0.0f) - (down(GLFW_KEY_E) ? 1.0f : 0.0f);
            const bool  altKey = down(GLFW_KEY_LEFT_ALT) || down(GLFW_KEY_RIGHT_ALT);
            const bool  ctrlKey = down(GLFW_KEY_LEFT_CONTROL) || down(GLFW_KEY_RIGHT_CONTROL);
            const bool  shift  = down(GLFW_KEY_LEFT_SHIFT) || down(GLFW_KEY_RIGHT_SHIFT);

            if (kx != 0.0f || ky != 0.0f || kz != 0.0f) {
                Obb&       ob = editBoxes[selected].obb;
                const Quat q  = ob.rotation;
                const ViewControlFrame frame = viewControlFrame(cam, upAxis);
                if (ctrlKey) {
                    const float maxHalf = std::max({ob.halfExtents.x, ob.halfExtents.y,
                                                    ob.halfExtents.z, 0.01f});
                    const float step = (shift ? 2.4f : 0.8f) * maxHalf * dt;
                    auto resizeToward = [&](glm::vec3 dir) {
                        const ResizeMatch match = resizeMatchForDirection(ob, dir);
                        float& h = halfExtentByAxis(ob, match.axis);
                        h = std::max(1e-4f, h + static_cast<float>(match.sign) * step);
                    };
                    if (down(GLFW_KEY_D)) resizeToward(frame.right);
                    if (down(GLFW_KEY_A)) resizeToward(-frame.right);
                    if (down(GLFW_KEY_W)) resizeToward(frame.forward);
                    if (down(GLFW_KEY_S)) resizeToward(-frame.forward);
                    if (down(GLFW_KEY_Q)) resizeToward(frame.up);
                    if (down(GLFW_KEY_E)) resizeToward(-frame.up);
                    dirty = true;
                } else if (altKey) {
                    const float rstep = (shift ? 3.6f : 1.2f) * dt;  // rad/sec
                    glm::quat rot = toGlm(q);
                    auto rotateAround = [&](float amount, glm::vec3 axis) {
                        if (amount == 0.0f) return;
                        axis = normalizeGlmOr(axis, frame.up);
                        rot = glm::angleAxis(amount * rstep, axis) * rot;
                    };
                    rotateAround(-kx, frame.up);
                    rotateAround(ky, frame.right);
                    rotateAround(-kz, frame.forward);
                    ob.rotation = fromGlm(glm::normalize(rot));
                    dirty       = true;
                } else {
                    const glm::vec3 gv = frame.right * kx + frame.forward * ky + frame.up * kz;
                    Vec3            mv{gv.x, gv.y, gv.z};
                    const float len2 = dot(mv, mv);
                    if (len2 > 0.0f) {
                        const float speed = (shift ? 4.0f : 1.0f) *
                                            std::max(cam.distance() * 0.30f, 1e-6f) * dt;  // units/sec
                        mv        = mv * (speed / std::sqrt(len2));
                        ob.center = ob.center + mv;
                        dirty     = true;
                    }
                }
            }
        }
        if (!cropPreview && poseSetup && !ImGui::GetIO().WantCaptureKeyboard) {
            auto down = [&](int key) { return glfwGetKey(window, key) == GLFW_PRESS; };
            const float kx = (down(GLFW_KEY_D) ? 1.0f : 0.0f) - (down(GLFW_KEY_A) ? 1.0f : 0.0f);
            const float ky = (down(GLFW_KEY_W) ? 1.0f : 0.0f) - (down(GLFW_KEY_S) ? 1.0f : 0.0f);
            const float kz = (down(GLFW_KEY_Q) ? 1.0f : 0.0f) - (down(GLFW_KEY_E) ? 1.0f : 0.0f);
            const bool  altKey = down(GLFW_KEY_LEFT_ALT) || down(GLFW_KEY_RIGHT_ALT);
            const bool  shift  = down(GLFW_KEY_LEFT_SHIFT) || down(GLFW_KEY_RIGHT_SHIFT);

            if (kx != 0.0f || ky != 0.0f || kz != 0.0f) {
                const ViewControlFrame frame = viewControlFrame(previewCam, upAxis);
                if (altKey) {
                    const ViewControlFrame rotFrame = screenControlFrame(previewCam);
                    const float rstep = (shift ? 3.6f : 1.2f) * dt;
                    glm::quat rot = poseRot;
                    auto rotateAround = [&](float amount, glm::vec3 axis) {
                        if (amount == 0.0f) return;
                        axis = normalizeGlmOr(axis, rotFrame.up);
                        rot = glm::angleAxis(amount * rstep, axis) * rot;
                    };
                    rotateAround(-ky, rotFrame.right);    // W pitch down, S pitch up
                    rotateAround(kx, rotFrame.forward);   // A roll left, D roll right
                    rotateAround(kz, rotFrame.up);        // Q yaw left, E yaw right
                    poseRot = normalizeQuatOr(rot, poseRot);
                } else {
                    const glm::vec3 gv = frame.right * kx + frame.forward * ky + frame.up * kz;
                    const float len = glm::length(gv);
                    if (len > 1e-6f) {
                        const float speed = (shift ? 4.0f : 1.0f) * poseMoveSpeed *
                                            std::max(previewCam.distance() * 0.30f, 1e-6f) * dt;
                        poseOrigin += (gv / len) * speed;
                    }
                }
            }
        }

        // ---- smooth the camera toward its target (slerp/lerp) ----
        cam.update(dt);
        if (cropPreview || poseSetup) previewCam.update(dt);

        int fbw, fbh;
        glfwGetFramebufferSize(window, &fbw, &fbh);
        if (fbw == 0 || fbh == 0) {
            fbw = options.width;
            fbh = options.height;
        }
        const float aspect = static_cast<float>(fbw) / static_cast<float>(std::max(fbh, 1));

        // ---- new frame ----
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();

        const glm::mat4 view = cam.viewMatrix();
        const glm::mat4 proj = cam.projMatrix(aspect);

        if (cropPreview) {
            // ===== crop preview overlay: render the cropped subset + OK/Cancel =====
            const glm::mat4 pview = previewCam.viewMatrix();
            const glm::mat4 pproj = previewCam.projMatrix(aspect);
            const bool      previewOnly = cropPreviewFromPose;

            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(300, 150), ImGuiCond_FirstUseEver);
            ImGui::Begin("Crop preview");
            ImGui::TextWrapped(previewOnly ? "Preview of the pose cloud (%zu pts). Drag to rotate."
                                           : "Preview of the cropped cloud (%zu pts). Drag to rotate; "
                                             "releases back to the spin.",
                               previewOnly ? poseCloud.size() : previewPoints.count());
            ImGui::SliderFloat("point size", &pointSize, 1.0f, 12.0f, "%.1f");
            bool ok = false, cancel = false;
            if (previewOnly) {
                cancel = ImGui::Button("Back to pose");
            } else {
                ok = ImGui::Button("OK");
                ImGui::SameLine();
                cancel = ImGui::Button("Cancel");
            }
            if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());
            ImGui::End();

            if (cancel) {
                cropPreview = false;
                cropPreviewFromPose = false;
            } else if (ok) {
                // Commit: authoritative crop on the FULL cloud + export.
                PointCloud out = cropToCloud(cloud, previewSpec);
                auto       w   = registry.writerForExt(extOf(exportPath));
                if (!w) {
                    status = std::string("no writer for ") + exportPath;
                } else {
                    io::FileByteSink sink(exportPath);
                    if (!sink.ok()) {
                        status = std::string("cannot create ") + exportPath;
                    } else {
                        io::WriteOptions wo;
                        wo.encoding =
                            (exportEncoding == 1) ? io::Encoding::Ascii : io::Encoding::Binary;
                        auto wr = w->write(out, sink, wo);
                        status  = wr ? ("wrote " + std::to_string(out.size()) + " pts -> " + exportPath)
                                     : ("write failed: " + wr.error().message);
#if defined(CLOUDCROPPER_HAS_REGISTRATION)
                        if (wr) lastExportPath = exportPath;  // "Use last export" target
#endif
                    }
                }
                cropPreview = false;
                cropPreviewFromPose = false;
            }

            // ---- render: ONLY the cropped subset, depth-tested like the main view ----
            glViewport(0, 0, fbw, fbh);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            previewPoints.draw(pview, pproj, pointSize, static_cast<ColorMode>(colorMode));
            if (previewOnly) {
                const float poseVectorLength =
                    std::max(glm::length(previewBmax - previewBmin) * 0.35f, 0.1f);
                const ViewControlFrame poseFrame = viewControlFrame(previewCam, upAxis);
                const glm::vec3 poseDir = poseDirection(poseRot);
                poseGizmo.draw(pview, pproj, poseOrigin, poseDir, poseFrame,
                               poseVectorLength, false);
                drawPoseMoveHints(pview, pproj, fbw, fbh, poseOrigin, poseFrame,
                                  poseVectorLength);
            }

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            ++frameCount;
            const bool lastFramePv = (options.frames > 0 && frameCount >= options.frames);
            glfwSwapBuffers(window);
            if (lastFramePv) break;
            continue;  // skip the editing panel / scene render entirely this frame
        }

        if (poseSetup) {
            const glm::mat4 pview = previewCam.viewMatrix();
            const glm::mat4 pproj = previewCam.projMatrix(aspect);
            const float poseVectorLength =
                std::max(glm::length(previewBmax - previewBmin) * 0.35f, 0.1f);

            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(370, 330), ImGuiCond_FirstUseEver);
            ImGui::Begin("Pose Setup");
            bool back = false;
            ImGui::Text("cropped cloud  %zu pts", poseCloud.size());
            ImGui::SameLine();
            const float backW = 72.0f;
            ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(),
                                          ImGui::GetWindowContentRegionMax().x - backW));
            back = ImGui::Button("Back", ImVec2(backW, 0));

            ImGui::TextUnformatted("move speed");
            ImGui::SameLine(92.0f);
            ImGui::SetNextItemWidth(165.0f);
            ImGui::SliderFloat("##pose_move_speed_slider", &poseMoveSpeed, 0.02f, 2.0f, "%.2fx");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(72.0f);
            ImGui::InputFloat("##pose_move_speed_input", &poseMoveSpeed, 0.0f, 0.0f, "%.3f");
            poseMoveSpeed = std::clamp(poseMoveSpeed, 0.005f, 10.0f);

            ImGui::DragFloat3("origin", glm::value_ptr(poseOrigin), 0.005f, -1e6f, 1e6f, "%.4f");
            float quatVals[4] = {poseRot.x, poseRot.y, poseRot.z, poseRot.w};
            if (ImGui::DragFloat4("quat xyzw", quatVals, 0.005f, -1.0f, 1.0f, "%.4f")) {
                poseRot = normalizeQuatOr(glm::quat(quatVals[3], quatVals[0], quatVals[1],
                                                    quatVals[2]),
                                          poseRot);
            }
            glm::vec3 poseDir = poseDirection(poseRot);
            float dirVals[3] = {poseDir.x, poseDir.y, poseDir.z};
            if (ImGui::DragFloat3("direction", dirVals, 0.005f, -1.0f, 1.0f, "%.4f")) {
                poseRot = poseRotationFromDirection(glm::vec3{dirVals[0], dirVals[1], dirVals[2]},
                                                    boxSemanticUp(upAxis));
                poseDir = poseDirection(poseRot);
            }
            if (ImGui::Button("Normalize quat")) {
                poseRot = normalizeQuatOr(poseRot, glm::quat{1.0f, 0.0f, 0.0f, 0.0f});
                poseDir = poseDirection(poseRot);
            }
            ImGui::SameLine();
            if (ImGui::Button("Use PCA")) {
                poseRot = poseRotationFromDirection(toGlm(normalizedOr(pcaFrame(poseCloud).axis,
                                                                        Vec3{1, 0, 0})),
                                                    boxSemanticUp(upAxis));
                poseDir = poseDirection(poseRot);
            }
            ImGui::TextDisabled("LMB drag rotate camera  |  MMB drag pan camera  |  WASD/QE move vector");
            if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());

            const float footerH = ImGui::GetFrameHeight() * 2.0f + ImGui::GetStyle().ItemSpacing.y;
            const float availY = ImGui::GetContentRegionAvail().y;
            if (availY > footerH)
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + availY - footerH);
            bool previewPose = ImGui::Button("Preview", ImVec2(-FLT_MIN, 0));
            bool exportPose = ImGui::Button("Export NPZ", ImVec2(-FLT_MIN, 0));
            ImGui::End();

            if (previewPose) {
                cropPreviewFromPose = true;
                cropPreview         = true;
                previewReturning    = false;
                previewLDown        = false;
                previewInitialOrbit = previewCam.orbitTarget();
            } else if (back) {
                poseSetup = false;
                status    = "pose setup cancelled";
            } else if (exportPose) {
#if defined(CLOUDCROPPER_HAS_NPZ)
                forceNpzExportPath();
                io::FileByteSink sink(exportPath);
                if (!sink.ok()) {
                    status = std::string("cannot create ") + exportPath;
                } else {
                    const Vec3 origin = fromGlm(poseOrigin);
                    const Vec3 dir    = normalizedOr(fromGlm(poseDirection(poseRot)),
                                                     Vec3{1, 0, 0});
                    const io::TemplateMeta tm = makeTemplateMeta(poseCloud, origin, dir);
                    auto wr = io::writeTemplateNpz(poseCloud, tm, sink);
                    status  = wr ? ("wrote pose template " + std::to_string(poseCloud.size()) +
                                     " pts -> " + exportPath)
                                 : ("write failed: " + wr.error().message);
                    if (wr) {
#if defined(CLOUDCROPPER_HAS_REGISTRATION)
                        lastExportPath = exportPath;
#endif
                        poseSetup = false;
                    }
                }
#else
                status = "pose export requires the NPZ build";
#endif
            }

            glViewport(0, 0, fbw, fbh);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            previewPoints.draw(pview, pproj, pointSize, static_cast<ColorMode>(colorMode));

            const ViewControlFrame poseFrame = viewControlFrame(previewCam, upAxis);
            const bool poseAltDown = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                                     glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
            const ViewControlFrame poseDrawFrame =
                poseAltDown ? screenControlFrame(previewCam) : poseFrame;
            poseGizmo.draw(pview, pproj, poseOrigin, poseDir, poseDrawFrame, poseVectorLength,
                           poseAltDown);
            if (poseAltDown)
                drawPoseRotationSphere(pview, pproj, fbw, fbh, poseOrigin, poseDrawFrame,
                                       poseVectorLength);
            else
                drawPoseMoveHints(pview, pproj, fbw, fbh, poseOrigin, poseFrame,
                                  poseVectorLength);

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            ++frameCount;
            const bool lastFramePose = (options.frames > 0 && frameCount >= options.frames);
            glfwSwapBuffers(window);
            if (lastFramePose) break;
            continue;
        }

        // ---- UI panel: function-grouped sections (Source -> View -> Boxes ->
        //      Crop&Export) in workflow order, aligned label column, load-first
        //      gating, and a status footer. ----
        ImGui::SetNextWindowPos(ImVec2(16, 16), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(390, 760), ImGuiCond_FirstUseEver);
        ImGui::Begin("CloudCropper");

        const float kLabelW = 86.0f;  // shared left label column
        auto        Row     = [&](const char* label) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(label);
            ImGui::SameLine(kLabelW);
            ImGui::SetNextItemWidth(-FLT_MIN);  // value widget fills to the right edge
        };
        auto Tip = [](const char* t) {
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", t);
        };

        // ===== SOURCE (always enabled) =====
        ImGui::SeparatorText("SOURCE");
#if defined(CLOUDCROPPER_VIEWER_PFD)
        if (ImGui::Button("Open...")) {
            auto sel = pfd::open_file("Open point cloud", ".",
                                      {"Point clouds",
                                       "*.ply *.pcd *.npz *.ply.gz *.pcd.gz *.npz.gz *.db3 *.mcap",
                                       "All files", "*"})
                           .result();
            if (!sel.empty()) tryLoad(sel[0]);
        }
        ImGui::SameLine();
#endif
        ImGui::SetNextItemWidth(180);
        ImGui::InputText("##path", pathBuf, sizeof(pathBuf));
        ImGui::SameLine();
        ImGui::BeginDisabled(pathBuf[0] == 0);
        if (ImGui::Button("Load")) tryLoad(pathBuf);
        ImGui::EndDisabled();

        if (cloud.size() == 0)
            ImGui::TextDisabled("Drag a .ply / .pcd / .npz / .db3 / .mcap here, or Open.../Load.");

#if defined(CLOUDCROPPER_HAS_ROSBAG)
        if (bagMode && !bagTopics.empty()) {
            ImGui::TextDisabled("ROS2 bag");
            ImGui::Indent(8.0f);
            Row("topic");
            if (ImGui::BeginCombo("##topic", bagTopics[bagTopicIdx].c_str())) {
                for (int i = 0; i < static_cast<int>(bagTopics.size()); ++i) {
                    const std::string label =
                        bagTopics[i] + " (" + std::to_string(bagCounts[i]) + ")";
                    if (ImGui::Selectable(label.c_str(), i == bagTopicIdx)) {
                        bagTopicIdx = i;
                        bagFrame    = 0;
                        loadBagFrame(true);  // new topic => reframe
                    }
                }
                ImGui::EndCombo();
            }
            const int total = static_cast<int>(bagCounts[bagTopicIdx]);
            if (ImGui::Button("|<")) { bagFrame = 0; loadBagFrame(false); }
            ImGui::SameLine();
            if (ImGui::Button("< prev") && bagFrame > 0) { --bagFrame; loadBagFrame(false); }
            ImGui::SameLine();
            if (ImGui::Button("next >") && bagFrame < total - 1) { ++bagFrame; loadBagFrame(false); }
            ImGui::SameLine();
            ImGui::Checkbox("play", &bagPlay);
            char ffmt[24];
            std::snprintf(ffmt, sizeof(ffmt), "%%d / %d", total > 0 ? total - 1 : 0);
            Row("frame");
            if (ImGui::SliderInt("##frame", &bagFrame, 0, total > 0 ? total - 1 : 0, ffmt))
                loadBagFrame(false);
            Row("fps");
            ImGui::SliderFloat("##fps", &bagFps, 0.5f, 30.0f, "%.1f");
            ImGui::Unindent(8.0f);
        }
#endif

        const bool hasCloud = cloud.size() > 0;

        // ===== VIEW =====
        ImGui::SeparatorText("VIEW");
        ImGui::BeginDisabled(!hasCloud);
        if (points.fullCount() != points.count())
            ImGui::Text("points  %zu / %zu (LOD)", points.count(), points.fullCount());
        else
            ImGui::Text("points  %zu", points.count());
        Row("size");
        ImGui::SliderFloat("##size", &pointSize, 1.0f, 12.0f, "%.1f");
        if (points.fullCount() > 1'000'000) {
            Row("budget");
            if (ImGui::SliderInt("##budget", &budgetM, 1, 50)) {
                points.setBudget(static_cast<std::size_t>(budgetM) * 1'000'000);
                std::string e;
                points.build(cloud, e);  // re-decimate the display set
                dirty = true;
            }
            Tip("Display point budget (millions); the full cloud is still cropped/exported");
        }
        const char* modes[] = {"flat", "rgb", "scalar", "height"};
        Row("color");
        ImGui::Combo("##color", &colorMode, modes, IM_ARRAYSIZE(modes));
        Tip("Point coloring mode");
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("up/fit");
        ImGui::SameLine(kLabelW);
        ImGui::SetNextItemWidth(90);
        const char* ups[] = {"X up", "Y up", "Z up"};
        if (ImGui::Combo("##up", &upAxis, ups, IM_ARRAYSIZE(ups))) {
            applyUp(upAxis);
            cam.fit(bmin, bmax);  // reframe with the new up
        }
        Tip("World up axis (reframes the camera)");
        ImGui::SameLine();
        if (ImGui::Button("Fit camera (F)")) cam.fit(bmin, bmax);
        Tip("Fit camera to the cloud (F)");
        ImGui::EndDisabled();

        // ===== BOXES =====
        ImGui::SeparatorText("BOXES");
        ImGui::BeginDisabled(!hasCloud);
        const char* ops[] = {"Union", "Intersection"};
        int         opi   = (combineOp == BoolOp::Union) ? 0 : 1;
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("combine");
        ImGui::SameLine(kLabelW);
        ImGui::SetNextItemWidth(130);
        if (ImGui::Combo("##combine", &opi, ops, IM_ARRAYSIZE(ops))) {
            combineOp = (opi == 0) ? BoolOp::Union : BoolOp::Intersection;
            dirty     = true;
        }
        Tip("How multiple boxes merge (Union / Intersection)");
        ImGui::SameLine();
        if (ImGui::Button("+ Add box")) {
            EditBox b;
            if (sceneBounds.valid()) {
                b.obb             = uprightObbFromAabb(sceneBounds, upAxis);
                b.obb.halfExtents = b.obb.halfExtents * 0.4f;
            } else {
                b.obb.halfExtents = {0.5f, 0.5f, 0.5f};
                enforceUprightBox(b.obb, upAxis);
            }
            editBoxes.push_back(b);
            selected = static_cast<int>(editBoxes.size()) - 1;
            dirty    = true;
        }
        if (!editBoxes.empty() &&
            ImGui::BeginTable("boxes", 4,
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH)) {
            ImGui::TableSetupColumn("on", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("box", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("role", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableHeadersRow();
            for (int i = 0; i < static_cast<int>(editBoxes.size()); ++i) {
                ImGui::PushID(i);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                bool en = editBoxes[i].enabled;
                if (ImGui::Checkbox("##en", &en)) {
                    editBoxes[i].enabled = en;
                    dirty                = true;
                }
                ImGui::TableSetColumnIndex(1);
                if (ImGui::RadioButton(("box " + std::to_string(i)).c_str(), selected == i))
                    selected = i;
                ImGui::TableSetColumnIndex(2);
                int         role    = (editBoxes[i].role == BoxRole::Include) ? 0 : 1;
                const char* roles[] = {"incl", "excl"};
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::Combo("##role", &role, roles, IM_ARRAYSIZE(roles))) {
                    editBoxes[i].role = (role == 0) ? BoxRole::Include : BoxRole::Exclude;
                    dirty             = true;
                }
                Tip("Include keeps points inside; Exclude removes them");
                ImGui::TableSetColumnIndex(3);
                ImGui::BeginDisabled(editBoxes.size() <= 1);
                const bool del = ImGui::SmallButton("x");
                ImGui::EndDisabled();
                if (del && editBoxes.size() > 1) {
                    editBoxes.erase(editBoxes.begin() + i);
                    if (selected >= static_cast<int>(editBoxes.size()))
                        selected = static_cast<int>(editBoxes.size()) - 1;
                    dirty = true;
                    ImGui::PopID();
                    break;
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("gizmo");
        ImGui::SameLine();
        ImGui::RadioButton("T", &gizmoOp, 0);
        ImGui::SameLine();
        ImGui::RadioButton("R", &gizmoOp, 1);
        ImGui::SameLine();
        ImGui::RadioButton("S", &gizmoOp, 2);

        if (selected >= 0 && selected < static_cast<int>(editBoxes.size())) {
            Obb&  ob = editBoxes[selected].obb;
            float c[3] = {ob.center.x, ob.center.y, ob.center.z};
            float h[3] = {ob.halfExtents.x, ob.halfExtents.y, ob.halfExtents.z};
            Row("center");
            if (ImGui::DragFloat3("##center", c, 0.01f)) {
                ob.center = {c[0], c[1], c[2]};
                dirty     = true;
            }
            Row("half-size");
            if (ImGui::DragFloat3("##half", h, 0.01f, 0.0f, 1e9f)) {
                ob.halfExtents = {std::max(h[0], 0.0f), std::max(h[1], 0.0f), std::max(h[2], 0.0f)};
                dirty          = true;
            }
            if (ImGui::Button("Snap to AABB")) {
                ob.rotation = uprightBoxRotation(Quat{}, upAxis);
                dirty       = true;
            }
            Tip("Reset the box rotation to axis-aligned");
        }
        ImGui::TextDisabled("Move: W/A/S/D map forward/left/back/right  Q/E up/down  -  Shift faster");
        ImGui::TextDisabled("Alt rotate: A/D yaw  W/S pitch  Q/E roll  -  Shift faster");
        ImGui::TextDisabled("Ctrl resize: A/D view-left/right  W/S view-forward/back  Q/E up/down");
        ImGui::TextDisabled("green = points inside");
        ImGui::EndDisabled();

        // ===== CROP & EXPORT =====
        ImGui::SeparatorText("CROP & EXPORT");
        ImGui::BeginDisabled(!hasCloud);
        ImGui::Text("kept  %llu / %zu", static_cast<unsigned long long>(keptCount), cloud.size());
        Row("out");
        ImGui::InputText("##out", exportPath, sizeof(exportPath));
        Row("enc");
        ImGui::Combo("##enc", &exportEncoding, "binary\0ascii\0");
        Tip("PLY/PCD encoding for the export");
#if defined(CLOUDCROPPER_HAS_NPZ)
        ImGui::Checkbox("Add object pose", &addObjectPose);
        Tip("Crop first, then assign one origin+direction pose and export an NPZ template");
#else
        bool poseDisabled = false;
        ImGui::BeginDisabled();
        ImGui::Checkbox("Add object pose", &poseDisabled);
        ImGui::EndDisabled();
        Tip("Requires an NPZ-enabled build");
#endif
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.38f, 0.47f, 0.96f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.47f, 0.555f, 1.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.32f, 0.40f, 0.86f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        const bool cropClicked = ImGui::Button("Crop + Export", ImVec2(-FLT_MIN, 0));
        ImGui::PopStyleColor(4);
        Tip(addObjectPose ? "Crop and enter Pose Setup; preview is available there"
                          : "Preview the crop (spins), then commit + write to 'out'");
        if (cropClicked) {
            CropSpec   spec        = buildSpec(editBoxes, combineOp);
            PointCloud previewCloud = cropToCloud(cloud, spec);
            if (previewCloud.size() == 0) {
                status = "crop is empty - nothing to export";
#if defined(CLOUDCROPPER_HAS_NPZ)
            } else if (addObjectPose) {
                enterPoseSetup(std::move(previewCloud));
#else
            } else if (addObjectPose) {
                status = "object pose export requires the NPZ build";
#endif
            } else {
                // Crop-only export still uses the spinning preview. OK commits
                // + exports; Cancel returns to editing.
                std::string pe;
                previewPoints.build(previewCloud, pe);  // idempotent rebuild
                Aabb pb = previewCloud.bounds();
                if (pb.valid()) {
                    previewBmin = {pb.min.x, pb.min.y, pb.min.z};
                    previewBmax = {pb.max.x, pb.max.y, pb.max.z};
                } else {
                    previewBmin = {-1, -1, -1};
                    previewBmax = {1, 1, 1};
                }
                previewCam.setUp(boxSemanticUp(upAxis));
                previewCam.fit(previewBmin, previewBmax);
                previewCam.update(0.0f);                 // snap live state to the fit targets
                previewInitialOrbit = previewCam.orbitTarget();
                previewSpec         = spec;
                previewReturning    = false;
                previewLDown        = false;
                cropPreviewFromPose = false;
                cropPreview         = true;
            }
        }
        ImGui::EndDisabled();

        // ===== status footer =====
        ImGui::Separator();
        if (status.empty()) {
            ImGui::TextDisabled("ready");
        } else {
            const bool isErr = status.find("fail") != std::string::npos ||
                               status.find("cannot") != std::string::npos ||
                               status.find("no writer") != std::string::npos ||
                               status.find("empty") != std::string::npos;
            if (isErr) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.90f, 0.35f, 0.30f, 1.0f));
            ImGui::TextWrapped("%s", status.c_str());
            if (isErr) ImGui::PopStyleColor();
        }
        ImGui::End();

#if defined(CLOUDCROPPER_HAS_REGISTRATION)
        // ===== REGISTRATION panel (right side) =====
        {
            // Collect a finished worker result on the UI thread.
            const int rs = regState.load();
            if (rs == 2 || rs == 3) {
                std::lock_guard<std::mutex> lk(regMutex);
                if (rs == 2) {
                    regShown = regResult;
                    // row-major array -> column-major glm
                    for (int row = 0; row < 4; ++row)
                        for (int col = 0; col < 4; ++col)
                            regT[col][row] = static_cast<float>(
                                regShown.transform[static_cast<std::size_t>(row * 4 + col)]);
                    regHasResult = true;
                    status       = std::string(regShown.converged ? "registered: " : "NOT converged: ") +
                             regShown.detail;
                    regStatus = status;
                } else {
                    status    = "registration failed: " + regError;
                    regStatus = status;
                }
                regState.store(0);
            }

            const ImGuiViewport* vp = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(ImVec2(vp->WorkSize.x - 396.0f, 16.0f), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(380, 520), ImGuiCond_FirstUseEver);
            ImGui::Begin("Registration");

            ImGui::SeparatorText("TARGET");
            // Loads `path` as the target cloud; feedback goes to regStatus,
            // which renders INSIDE this panel (the left footer is easy to miss).
            auto loadTarget = [&](const std::string& path) {
                auto lr = load(path);  // same loader as the source: files + bags
                if (lr) {
                    targetCloud = std::move(lr.value());
                    std::string te;
                    targetPoints.build(targetCloud, te);
                    regStatus = "target loaded: " + std::to_string(targetCloud.size()) +
                                " pts (" + path + ")";
                } else {
                    regStatus = "load failed: " + lr.error().message;
                }
            };
            Row("file");
            ImGui::InputText("##tgtpath", regTargetPath, sizeof(regTargetPath));
#if defined(CLOUDCROPPER_VIEWER_PFD)
            if (ImGui::Button("Open...##tgt")) {
                auto sel = pfd::open_file(
                               "Open target cloud", ".",
                               {"Point clouds",
                                "*.ply *.pcd *.npz *.ply.gz *.pcd.gz *.npz.gz *.db3 *.mcap",
                                "All files", "*"})
                               .result();
                if (!sel.empty()) {
                    std::snprintf(regTargetPath, sizeof(regTargetPath), "%s", sel[0].c_str());
                    loadTarget(sel[0]);
                }
            }
            ImGui::SameLine();
#endif
            if (ImGui::Button("Load target")) {
                if (regTargetPath[0])
                    loadTarget(regTargetPath);
                else
                    regStatus = "enter a path above (or use Open...)";
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(lastExportPath.empty());
            if (ImGui::Button("Use last export")) {
                std::snprintf(regTargetPath, sizeof(regTargetPath), "%s", lastExportPath.c_str());
                loadTarget(lastExportPath);
            }
            ImGui::EndDisabled();
            Tip("Load the cloud written by the last Crop + Export");
            if (!regStatus.empty()) {
                const bool bad = regStatus.find("failed") != std::string::npos ||
                                 regStatus.find("enter a path") != std::string::npos;
                if (bad) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.90f, 0.35f, 0.30f, 1.0f));
                ImGui::TextWrapped("%s", regStatus.c_str());
                if (bad) ImGui::PopStyleColor();
            }
            if (targetCloud.size() > 0)
                ImGui::Text("target  %zu pts", targetCloud.size());
            else
                ImGui::TextDisabled("no target loaded");
            ImGui::Checkbox("show target (cyan)", &showTarget);

            GpuProbeInfo gpuInfo;
            {
                std::lock_guard<std::mutex> lk(gpuProbeMutex);
                gpuInfo = gpuProbe;
            }
            ImGui::SeparatorText("GPU");
            ImGui::BeginChild("##gpu_status", ImVec2(0, 76), true);
            ImGui::Text("GPU: %s", gpuInfo.gpuName.c_str());
            ImGui::Text("PyTorch: %s", gpuInfo.torchVersion.c_str());
            if (!gpuInfo.done) {
                ImGui::TextDisabled("checking...");
            } else if (gpuInfo.usable) {
                ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%s",
                                   gpuInfo.message.c_str());
            } else {
                ImGui::TextColored(ImVec4(0.95f, 0.55f, 0.35f, 1.0f), "%s",
                                   gpuInfo.message.c_str());
            }
            ImGui::EndChild();

            ImGui::SeparatorText("ALGORITHM");
            Row("algo");
            if (ImGui::BeginCombo("##algo", kRegAlgos[regAlgoIdx].name)) {
                for (int i = 0; i < kRegAlgoCount; ++i) {
                    const bool selectedAlgo = i == regAlgoIdx;
                    const bool changed = ImGui::Selectable(kRegAlgos[i].name, selectedAlgo);
                    if (registrationAlgoCanUseGpu(kRegAlgos[i].algo)) {
                        drawSmallBadgeOnLastItem("GPU",
                                                 gpuInfo.usable ? IM_COL32(45, 150, 85, 230)
                                                                : IM_COL32(82, 86, 96, 220),
                                                 IM_COL32(245, 250, 248, 255));
                    }
                    if (changed && !selectedAlgo) {
                        regAlgoIdx = i;
                        loadRegDefaults(i);  // switch to that package's config defaults
                    }
                }
                ImGui::EndCombo();
            }
            Tip("Defaults come from config/<package>.yaml; edits here override them");
            const cc::reg::RegAlgo alg = kRegAlgos[regAlgoIdx].algo;
            Row("voxel");
            ImGui::InputFloat("##ds", &regDownsample, 0.0f, 0.0f, "%.4f");
            Tip("Preprocess leaf size in meters; 0 = auto from target spacing");
            Row("max corr");
            ImGui::InputFloat("##mc", &regMaxCorr, 0.0f, 0.0f, "%.4f");
            Tip("Max correspondence distance; 0 = auto");
            Row("threads");
            ImGui::SliderInt("##th", &regThreads, 1, 16);
            if (alg == cc::reg::RegAlgo::KissMatcher || alg == cc::reg::RegAlgo::KissGicp) {
                Row("kiss res");
                ImGui::InputFloat("##kr", &regKissRes, 0.0f, 0.0f, "%.4f");
                Tip("KISS-Matcher working resolution; 0 = auto (1.5x spacing)");
            }
            if (alg == cc::reg::RegAlgo::GradientSdfGpu) {
                Row("sdf res");
                ImGui::SliderInt("##sr", &regSdfRes, 32, 192);
                Tip("SDF voxel grid resolution per axis");
                ImGui::Checkbox("uncertainty", &regUncertainty);
                Tip("GPIS variance channel: heteroscedastic weighting plus the\n"
                    "confidence / norm-residual trust score on the result");
            }
            if (alg == cc::reg::RegAlgo::BufferX || alg == cc::reg::RegAlgo::BufferXGicp) {
                Row("voxel");
                ImGui::InputFloat("##bxv", &regBufferxVoxel, 0.0f, 0.0f, "%.4f");
                Tip("BUFFER-X input downsample voxel; 0 = auto (scale normalization)");
            }
            if (alg == cc::reg::RegAlgo::Rap || alg == cc::reg::RegAlgo::RapGicp) {
                Row("voxel");
                ImGui::InputFloat("##rapv", &regRapVoxel, 0.0f, 0.0f, "%.4f");
                Tip("RAP input downsample voxel; 0 = adaptive (voxel_ratio per-part counts)");
            }
            if (alg == cc::reg::RegAlgo::KissMatcher || alg == cc::reg::RegAlgo::KissGicp ||
                alg == cc::reg::RegAlgo::GradientSdfGpu ||
                alg == cc::reg::RegAlgo::BufferX || alg == cc::reg::RegAlgo::BufferXGicp ||
                alg == cc::reg::RegAlgo::G3Reg || alg == cc::reg::RegAlgo::G3RegGicp ||
                alg == cc::reg::RegAlgo::Rap || alg == cc::reg::RegAlgo::RapGicp) {
                ImGui::Checkbox("refine with GICP", &regRefine);
            }

            const bool running = regState.load() == 1;
            ImGui::BeginDisabled(running || cloud.size() == 0 || targetCloud.size() == 0);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.38f, 0.47f, 0.96f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.47f, 0.555f, 1.0f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.32f, 0.40f, 0.86f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
            const bool runClicked = ImGui::Button("Register", ImVec2(-FLT_MIN, 0));
            ImGui::PopStyleColor(4);
            ImGui::EndDisabled();
            Tip("Align the open cloud (source) onto the target");
            if (runClicked) {
                // Start from the yaml defaults so config-only knobs (e.g.
                // trunc_mul) survive, then apply the UI fields on top.
                cc::reg::RegOptions ro = cc::reg::defaultsFor(alg);
                ro.downsample     = regDownsample;
                ro.maxCorr        = regMaxCorr;
                ro.threads        = regThreads;
                ro.kissResolution = regKissRes;
                ro.sdfResolution  = regSdfRes;
                ro.bufferxVoxel   = regBufferxVoxel;
                ro.rapVoxel       = regRapVoxel;
                ro.refine         = regRefine;
                ro.sdfUncertainty = regUncertainty;
                // Source = the whole open cloud (boxes are crop-only).
                regStatus = "registering " + std::to_string(cloud.size()) +
                            " pts -> target " + std::to_string(targetCloud.size()) + " pts";
                if (regThread.joinable()) regThread.join();
                regState.store(1);
                regStarted = glfwGetTime();
                regThread  = std::thread(
                    [&regMutex, &regResult, &regError, &regState, ro, src = cloud,
                     tgt = targetCloud]() mutable {
                        auto rr = cc::reg::registerClouds(src, tgt, ro);
                        std::lock_guard<std::mutex> lk(regMutex);
                        if (rr) {
                            regResult = std::move(rr.value());
                            regState.store(2);
                        } else {
                            regError = rr.error().message;
                            regState.store(3);
                        }
                    });
            }
            if (running) {
                const char spin[] = {'|', '/', '-', '\\'};
                ImGui::Text("%c running %s... %.1fs",
                            spin[static_cast<int>(glfwGetTime() * 8.0) & 3],
                            kRegAlgos[regAlgoIdx].name, glfwGetTime() - regStarted);
            }

            ImGui::SeparatorText("RESULT");
            if (regHasResult) {
                ImGui::Text("%s   rmse %.5g   inliers %zu   %.2fs",
                            regShown.converged ? "converged" : "NOT converged",
                            regShown.rmse, regShown.inliers, regShown.seconds);
                if (regShown.confidence >= 0.0) {
                    const bool low = regShown.confidence < 0.6;
                    if (low)
                        ImGui::PushStyleColor(ImGuiCol_Text,
                                              ImVec4(0.95f, 0.65f, 0.25f, 1.0f));
                    ImGui::Text("confidence %.0f%%   norm-residual %.3g%s",
                                regShown.confidence * 100.0, regShown.normResidual,
                                low ? "  (check the result!)" : "");
                    if (low) ImGui::PopStyleColor();
                    Tip("How much of the source the SDF field can explain by its own\n"
                        "uncertainty (chi-square 95%). Low = plausible-but-wrong pose risk.");
                }
                for (int row = 0; row < 4; ++row)
                    ImGui::Text("[% .4f % .4f % .4f % .4f]",
                                regShown.transform[static_cast<std::size_t>(row * 4 + 0)],
                                regShown.transform[static_cast<std::size_t>(row * 4 + 1)],
                                regShown.transform[static_cast<std::size_t>(row * 4 + 2)],
                                regShown.transform[static_cast<std::size_t>(row * 4 + 3)]);
                ImGui::Checkbox("overlay aligned source", &regOverlay);
                Tip("Preview the source through the result transform (orange)");
                if (ImGui::Button("Apply to source")) {
                    PointCloud moved = std::move(cloud);
                    cc::reg::applyTransform(moved, regShown.transform);
                    applyCloud(std::move(moved), false);
                    regHasResult = false;
                    regT         = glm::mat4(1.0f);
                    status       = "applied the registration transform to the source";
                    regStatus    = status;
                }
                Tip("Bake the transform into the open cloud");
                ImGui::SameLine();
                if (ImGui::Button("Reset")) {
                    regHasResult = false;
                    regT         = glm::mat4(1.0f);
                }
                Row("save as");
                ImGui::InputText("##alnpath", regAlignedPath, sizeof(regAlignedPath));
                if (ImGui::Button("Save aligned source") && regAlignedPath[0]) {
                    PointCloud aligned = cloud;
                    cc::reg::applyTransform(aligned, regShown.transform);
                    auto w = registry.writerForExt(extOf(regAlignedPath));
                    if (!w) {
                        status = std::string("no writer for ") + regAlignedPath;
                    } else {
                        io::FileByteSink sink(regAlignedPath);
                        if (!sink.ok()) {
                            status = std::string("cannot create ") + regAlignedPath;
                        } else {
                            auto wr = w->write(aligned, sink, {});
                            status  = wr ? ("wrote aligned -> " + std::string(regAlignedPath))
                                         : ("write failed: " + wr.error().message);
                        }
                    }
                }
            } else {
                ImGui::TextDisabled(running ? "running..." : "no result yet");
            }
            ImGui::End();
        }
#endif

        // ---- ImGuizmo on the selected box. LOCAL mode keeps the transform axes
        //      attached to the box, so rotating the box rotates its coordinate frame too. ----
        if (selected >= 0 && selected < static_cast<int>(editBoxes.size()) &&
            editBoxes[static_cast<std::size_t>(selected)].enabled) {
            ImGuizmo::SetOrthographic(false);
            ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());
            ImGuizmo::SetRect(0, 0, static_cast<float>(fbw), static_cast<float>(fbh));
            glm::mat4              model = obbToMatrix(editBoxes[selected].obb);
            ImGuizmo::OPERATION    op    = gizmoOp == 0   ? ImGuizmo::TRANSLATE
                                          : gizmoOp == 1 ? ImGuizmo::ROTATE
                                                          : ImGuizmo::SCALE;
            if (ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj), op,
                                     ImGuizmo::LOCAL, glm::value_ptr(model))) {
                editBoxes[selected].obb = matrixToObb(model);
                enforceUprightBox(editBoxes[selected].obb, upAxis);
                dirty                   = true;
            }
        }

        // ---- recompute preview when boxes changed ----
        if (dirty) {
            CropSpec spec = buildSpec(editBoxes, combineOp);
            auto     mask = previewMask(cloud, spec);
            keptCount     = 0;
            for (auto v : mask) keptCount += v;
            points.updateKept(mask);
            dirty = false;
        }

        // ---- render ----
        glViewport(0, 0, fbw, fbh);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#if defined(CLOUDCROPPER_HAS_REGISTRATION)
        // Source: through the registration transform (orange) when previewing a
        // result; Target: cyan-tinted second cloud.
        const bool overlayOn = regHasResult && regOverlay;
        const glm::mat4 sourceModel = overlayOn ? regT : glm::mat4(1.0f);
        const ViewControlFrame poseFrame = viewControlFrame(cam, upAxis);
        points.draw(view, proj, pointSize, static_cast<ColorMode>(colorMode),
                    sourceModel,
                    glm::vec3(1.0f, 0.62f, 0.25f), overlayOn ? 0.35f : 0.0f);
        drawMetadataPose(poseGizmo, cloud, sourceModel, view, proj, poseFrame);
        if (showTarget && targetCloud.size() > 0) {
            targetPoints.draw(view, proj, pointSize, static_cast<ColorMode>(colorMode),
                              glm::mat4(1.0f), glm::vec3(0.25f, 0.85f, 0.95f), 0.55f,
                              /*highlight=*/false);
            drawMetadataPose(poseGizmo, targetCloud, glm::mat4(1.0f), view, proj, poseFrame);
        }
#else
        points.draw(view, proj, pointSize, static_cast<ColorMode>(colorMode));
#endif
        for (int i = 0; i < static_cast<int>(editBoxes.size()); ++i) {
            if (!editBoxes[i].enabled) continue;
            const glm::vec3 col = (i == selected) ? glm::vec3(1.0f, 0.85f, 0.2f)
                                : (editBoxes[i].role == BoxRole::Exclude)
                                    ? glm::vec3(0.9f, 0.3f, 0.3f)
                                    : glm::vec3(0.3f, 0.8f, 1.0f);
            boxes.draw(view, proj, obbToMatrix(editBoxes[i].obb), col);
        }
        if (selected >= 0 && selected < static_cast<int>(editBoxes.size()) &&
            editBoxes[static_cast<std::size_t>(selected)].enabled) {
            const bool ctrlDown = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                                  glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
            const bool altDown = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                                 glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
            const KeyHintMode mode = ctrlDown ? KeyHintMode::Resize
                                   : altDown  ? KeyHintMode::Rotate
                                              : KeyHintMode::Move;
            drawBoxKeyHints(editBoxes[static_cast<std::size_t>(selected)].obb, view, proj,
                             fbw, fbh, viewControlFrame(cam, upAxis), mode);
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // ---- screenshot / headless exit ----
        ++frameCount;
        const bool lastFrame =
            (options.frames > 0 && frameCount >= options.frames);
        if (lastFrame) {
            // Headless verification signal: framing + default-box preview result.
            if (sceneBounds.valid())
                std::fprintf(stderr,
                             "viewer: bounds [%.3f %.3f %.3f]..[%.3f %.3f %.3f]  kept %llu / %zu\n",
                             sceneBounds.min.x, sceneBounds.min.y, sceneBounds.min.z,
                             sceneBounds.max.x, sceneBounds.max.y, sceneBounds.max.z,
                             static_cast<unsigned long long>(keptCount), cloud.size());
            else
                std::fprintf(stderr, "viewer: empty (no cloud loaded)\n");
        }
        if (lastFrame && !options.screenshot.empty()) {
            glFinish();
            auto px = readFramebuffer(fbw, fbh);
            if (writeImage(options.screenshot, px.data(), fbw, fbh))
                std::fprintf(stderr, "viewer: wrote screenshot %s (%dx%d)\n",
                             options.screenshot.c_str(), fbw, fbh);
            else
                std::fprintf(stderr, "viewer: screenshot write failed: %s\n",
                             options.screenshot.c_str());
        }

        glfwSwapBuffers(window);
        if (lastFrame) break;
    }

    // Free GL objects while the context is still current (the renderers are stack
    // locals whose destructors would otherwise run AFTER glfwTerminate -> crash).
#if defined(CLOUDCROPPER_HAS_REGISTRATION)
    if (regThread.joinable()) regThread.join();  // worker references locals above
    if (gpuProbeThread.joinable()) gpuProbeThread.join();
    targetPoints.release();
#endif
    points.release();
    previewPoints.release();
    poseGizmo.release();
    boxes.release();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return {};
}

}  // namespace cc::viewer
