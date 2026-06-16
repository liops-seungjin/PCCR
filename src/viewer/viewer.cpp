// Interactive 3D viewer main loop (docs/design/02). Owns the GLFW window + GL
// context, ImGui/ImGuizmo, the orbit camera, point + box renderers, the box
// edit/crop UI, and crop export through the io registry.
#include "cloudcropper/viewer/viewer.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <mutex>
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
#include "cloudcropper/core/crop.hpp"
#include "cloudcropper/core/point_cloud.hpp"
#include "cloudcropper/io/byte_stream.hpp"
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
    std::string   err;

    // --- crop-preview state (the OK/Cancel spinning preview before exporting) ---
    PointRenderer previewPoints;          // GL buffers for the cropped subset
    Camera        previewCam;             // independent camera, framed on the crop
    bool          cropPreview = false;    // true while the preview overlay is shown
    bool          previewReturning = false;  // easing back to the initial pose after a drag
    glm::vec3     previewInitialOrbit{0.0f}; // stored yaw/pitch/roll targets of the initial pose
    glm::vec3     previewBmin{-1, -1, -1}, previewBmax{1, 1, 1};
    bool          previewLDown = false;   // LMB state inside the preview (for press->release edge)
    CropSpec      previewSpec;            // the spec captured when entering preview (reused on OK)
    if (!boxes.build(err)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return makeError(ErrorCode::IoError, "box shader: " + err);
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
        cam.setUp(ax == 0 ? glm::vec3{1, 0, 0} : ax == 2 ? glm::vec3{0, 0, 1} : glm::vec3{0, 1, 0});
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
    float         regBufferxVoxel = 0.0f;
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
        regRefine      = d.refine;
        regUncertainty = d.sdfUncertainty;
    };
    loadRegDefaults(regAlgoIdx);
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
                b.obb             = Obb::fromAabb(sceneBounds);
                b.obb.halfExtents = b.obb.halfExtents * 0.5f;
            } else {
                b.obb.center      = {0, 0, 0};
                b.obb.halfExtents = {0.5f, 0.5f, 0.5f};
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
        if (cropPreview) {
            // ---- preview camera: auto-yaw spin with drag override + ease-back ----
            const bool dragging = l && previewLDown && !isOverUi();
            if (dragging) {
                previewCam.orbit(dx, dy, false);  // drag takes over; spin paused
                previewReturning = false;
            } else if (previewLDown && !l) {
                // press -> release edge: ease back to the initial preview pose.
                previewReturning = true;
                previewCam.setOrbitTarget(previewInitialOrbit);
            }
            if (previewReturning) {
                // Keep nudging the target back (cheap) until we settle, then resume.
                previewCam.setOrbitTarget(previewInitialOrbit);
                if (previewCam.nearOrbitTarget(1e-3f)) previewReturning = false;
            } else if (!dragging) {
                previewCam.nudgeYaw(0.4f * dt);
            }
            previewLDown = l;
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

        // ---- WASD / QE: move (or, with Alt, rotate) the selected box along ITS
        //      OWN local axes. A/D = X, W/S = Y, Q/E = Z. Hold Shift to go faster.
        //      Speeds are per-second (dt-scaled) so they're frame-rate independent.
        if (!cropPreview && !ImGui::GetIO().WantCaptureKeyboard && selected >= 0 &&
            selected < static_cast<int>(editBoxes.size()) &&
            editBoxes[static_cast<std::size_t>(selected)].enabled) {
            auto down = [&](int key) { return glfwGetKey(window, key) == GLFW_PRESS; };
            const float kx = (down(GLFW_KEY_D) ? 1.0f : 0.0f) - (down(GLFW_KEY_A) ? 1.0f : 0.0f);
            const float ky = (down(GLFW_KEY_W) ? 1.0f : 0.0f) - (down(GLFW_KEY_S) ? 1.0f : 0.0f);
            const float kz = (down(GLFW_KEY_E) ? 1.0f : 0.0f) - (down(GLFW_KEY_Q) ? 1.0f : 0.0f);
            const bool  altKey = down(GLFW_KEY_LEFT_ALT) || down(GLFW_KEY_RIGHT_ALT);
            const bool  shift  = down(GLFW_KEY_LEFT_SHIFT) || down(GLFW_KEY_RIGHT_SHIFT);

            if (kx != 0.0f || ky != 0.0f || kz != 0.0f) {
                Obb&       ob = editBoxes[selected].obb;
                const Quat q  = ob.rotation;
                if (altKey) {
                    // rotate about the box's own axes: A/D yaw(Y), W/S pitch(X), Q/E roll(Z)
                    const float rstep = (shift ? 3.6f : 1.2f) * dt;  // rad/sec
                    glm::quat   dq    = glm::angleAxis(ky * rstep, glm::vec3{1, 0, 0}) *
                                   glm::angleAxis(kx * rstep, glm::vec3{0, 1, 0}) *
                                   glm::angleAxis(kz * rstep, glm::vec3{0, 0, 1});
                    ob.rotation = fromGlm(glm::normalize(toGlm(q) * dq));  // local-frame: right-multiply
                    dirty       = true;
                } else {
                    // translate along the box's world-space local axes
                    Vec3 mv = rotate(q, Vec3{1, 0, 0}) * kx + rotate(q, Vec3{0, 1, 0}) * ky +
                              rotate(q, Vec3{0, 0, 1}) * kz;
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

        // ---- smooth the camera toward its target (slerp/lerp) ----
        cam.update(dt);
        if (cropPreview) previewCam.update(dt);

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

            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(300, 150), ImGuiCond_FirstUseEver);
            ImGui::Begin("Crop preview");
            ImGui::TextWrapped("Preview of the cropped cloud (%zu pts). Drag to rotate; "
                               "releases back to the spin.",
                               previewPoints.count());
            ImGui::SliderFloat("point size", &pointSize, 1.0f, 12.0f, "%.1f");
            bool ok     = ImGui::Button("OK");
            ImGui::SameLine();
            bool cancel = ImGui::Button("Cancel");
            if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());
            ImGui::End();

            if (cancel) {
                cropPreview = false;
            } else if (ok) {
                // Commit: authoritative crop on the FULL cloud + export (old path).
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
            }

            // ---- render: ONLY the cropped subset, depth-tested like the main view ----
            glViewport(0, 0, fbw, fbh);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            previewPoints.draw(pview, pproj, pointSize, static_cast<ColorMode>(colorMode));

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            ++frameCount;
            const bool lastFramePv = (options.frames > 0 && frameCount >= options.frames);
            glfwSwapBuffers(window);
            if (lastFramePv) break;
            continue;  // skip the editing panel / scene render entirely this frame
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
                b.obb             = Obb::fromAabb(sceneBounds);
                b.obb.halfExtents = b.obb.halfExtents * 0.4f;
            } else {
                b.obb.halfExtents = {0.5f, 0.5f, 0.5f};
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
                ob.rotation = Quat{};
                dirty       = true;
            }
            Tip("Reset the box rotation to axis-aligned");
        }
        ImGui::TextDisabled("WASD/QE move box  -  Shift faster  -  Alt rotate");
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
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.38f, 0.47f, 0.96f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.47f, 0.555f, 1.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.32f, 0.40f, 0.86f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        const bool cropClicked = ImGui::Button("Crop + Export", ImVec2(-FLT_MIN, 0));
        ImGui::PopStyleColor(4);
        Tip("Preview the crop (spins), then commit + write to 'out'");
        if (cropClicked) {
            // Don't crop yet: build the cropped subset and enter the spinning
            // preview. OK commits + exports; Cancel returns to editing.
            CropSpec   spec        = buildSpec(editBoxes, combineOp);
            PointCloud previewCloud = cropToCloud(cloud, spec);
            if (previewCloud.size() == 0) {
                status = "crop is empty - nothing to preview";
            } else {
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
                previewCam.setUp(upAxis == 0   ? glm::vec3{1, 0, 0}
                                 : upAxis == 2 ? glm::vec3{0, 0, 1}
                                               : glm::vec3{0, 1, 0});
                previewCam.fit(previewBmin, previewBmax);
                previewCam.update(0.0f);                 // snap live state to the fit targets
                previewInitialOrbit = previewCam.orbitTarget();
                previewSpec         = spec;
                previewReturning    = false;
                previewLDown        = false;
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

            ImGui::SeparatorText("ALGORITHM");
            Row("algo");
            if (ImGui::BeginCombo("##algo", kRegAlgos[regAlgoIdx].name)) {
                for (int i = 0; i < kRegAlgoCount; ++i)
                    if (ImGui::Selectable(kRegAlgos[i].name, i == regAlgoIdx) &&
                        i != regAlgoIdx) {
                        regAlgoIdx = i;
                        loadRegDefaults(i);  // switch to that package's config defaults
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
            if (alg == cc::reg::RegAlgo::KissMatcher || alg == cc::reg::RegAlgo::KissGicp ||
                alg == cc::reg::RegAlgo::GradientSdfGpu ||
                alg == cc::reg::RegAlgo::BufferX || alg == cc::reg::RegAlgo::BufferXGicp) {
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

        // ---- ImGuizmo on the selected box (hidden while the box is disabled,
        //      matching the wireframe; re-enabling brings both back) ----
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
                                     ImGuizmo::WORLD, glm::value_ptr(model))) {
                editBoxes[selected].obb = matrixToObb(model);
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
        points.draw(view, proj, pointSize, static_cast<ColorMode>(colorMode),
                    overlayOn ? regT : glm::mat4(1.0f),
                    glm::vec3(1.0f, 0.62f, 0.25f), overlayOn ? 0.35f : 0.0f);
        if (showTarget && targetCloud.size() > 0)
            targetPoints.draw(view, proj, pointSize, static_cast<ColorMode>(colorMode),
                              glm::mat4(1.0f), glm::vec3(0.25f, 0.85f, 0.95f), 0.55f,
                              /*highlight=*/false);
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
    targetPoints.release();
#endif
    points.release();
    previewPoints.release();
    boxes.release();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return {};
}

}  // namespace cc::viewer
