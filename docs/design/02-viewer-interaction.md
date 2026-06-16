# 02 — Viewer & Interaction Layer Design

**Component:** CloudCropper Viewer/Interaction
**Author:** Viewer/Interaction architect
**Date:** 2026-06-04
**Status:** Proposed
**Scope:** Rendering tech, bounding-box interaction UX + math, camera controls, display performance/LOD, and viewer↔core coupling. Transport/serialization details are deferred to the core-architecture agent.

---

## 0. TL;DR (recommendations)

- **Renderer:** Custom **OpenGL 3.3+ core profile** + **Dear ImGui** (docking branch) + **ImGuizmo** for transform gizmos. Reject VTK/PCL Visualizer and Open3D-GUI as primary — they own the render loop and event handling, which fights our requirement for custom interactive box gizmos and a custom point pipeline.
- **Box model:** Boxes are **OBB-native** (center + 3×3 rotation + half-extents). AABB is the special case where rotation = identity, exposed as a UI lock. Creation = **2D screen rectangle → frustum slab → snapped OBB** for the fast path, plus **direct 3D handle drag**. Manipulation via **ImGuizmo** (translate/rotate/scale) + custom corner/face handles. Multiple boxes with a selection model; export = union or per-box.
- **Membership test:** transform point into box-local space, AABB compare against half-extents. O(1) per point, trivially SIMD/thread-parallel and GPU-portable.
- **Coupling:** **In-process, single binary, shared memory.** Viewer and core link into one executable; the viewer borrows a read-only `std::span<const PointXYZ...>` over the core's full-res cloud. The gzip "data transfer between components" requirement applies to **import/export and any future remote/headless mode**, not the hot viewer↔core path. Design the seam as an interface so a future out-of-process client is possible without touching viewer code.

---

## 1. Rendering technology choice

### 1.1 Requirements that drive the choice

1. **Custom interactive 3D widgets** — bounding-box gizmos with corner/face handles, hover highlight, drag-to-resize. This needs direct control over picking, the render loop, and overlay drawing.
2. **Point-cloud-first pipeline** — millions of points, custom shaders (color-by-attribute, point-size attenuation, future LOD/compute rasterization).
3. **Immediate-mode tool UI** — panels, sliders, attribute pickers, box list. Fast to iterate.
4. **Permissive license** — commercial product (`@liops.ai`); avoid copyleft and avoid VTK's heavyweight footprint.
5. **C++, performance-first** (the whole reason the project isn't Python).

### 1.2 Options compared

| Option | Control over render loop / picking | Point-cloud rendering | Custom 3D gizmos | UI toolkit | License | Verdict |
|---|---|---|---|---|---|---|
| **(a) Custom OpenGL + Dear ImGui + ImGuizmo** | Full — we own the loop, framebuffers, pick passes | Full — write our own VBO/shader pipeline, add compute rasterizer later | ImGuizmo (translate/rotate/scale/**bounds**) + our own handles | Dear ImGui (immediate mode, ideal for tooling) | MIT (ImGui, ImGuizmo) | **CHOSEN** |
| (b) VTK / PCL Visualizer | Low — VTK owns the pipeline & interactor; custom interactors are awkward (`vtkInteractorStyle`, picker classes) | Excellent out of the box; PCL viewer is built on VTK | Possible but heavy (`vtkBoxWidget2`) and not the look/feel we want | None native; bolt on Qt | VTK: BSD-3 (OK); PCL: BSD-3 (OK) but huge dependency tree | Reject as primary |
| (c) Open3D C++ GUI | Low–medium — GUI built on Google **Filament**; you script a scene, not the loop. Custom per-frame interaction limited | Good built-in point rendering | Limited; no first-class editable box gizmo, would fight the framework | Open3D `gui` module | MIT (Open3D), Apache-2.0 (Filament) | Reject as primary; good reference |
| (d-1) Magnum | Full (it's middleware you build on) | DIY, like OpenGL but nicer abstractions | DIY | Has ImGui integration | MIT | Strong runner-up |
| (d-2) bgfx | Full; multi-backend (D3D/Metal/Vulkan/GL) | DIY | DIY | ImGui integration exists | BSD-2 | Runner-up if multi-backend needed |
| (d-3) Qt3D / Qt Quick 3D | Medium; Qt owns a lot | OK | Awkward | Qt Widgets/QML | LGPL-3 / commercial — friction for a shipped C++ product | Reject |

### 1.3 Decision: custom OpenGL + Dear ImGui + ImGuizmo

**Why this and not VTK/PCL/Open3D:** all three are *frameworks* — they own the render loop and the interaction model. Our central feature is a **custom, editable bounding-box gizmo** with screen-rect creation and handle dragging, plus a point pipeline we want to push toward compute-shader rasterization for large clouds. That demands lower-level control. VTK/PCL/Open3D are excellent for "show me this cloud," poor for "let me build a bespoke 3D editing tool." We keep them in mind only as algorithm references.

**Why not Magnum/bgfx (the honest runner-ups):** both give the same low-level control with nicer abstractions, and **bgfx** specifically buys multi-backend (Metal/D3D/Vulkan) almost for free. The reason to start on **raw OpenGL** is pragmatic: ImGuizmo and the vast majority of point-cloud rendering references are written against GL; OpenGL 3.3 core is universally available; and our renderer is small enough that the abstraction layer of Magnum/bgfx isn't yet paying for itself. **Mitigation:** isolate all GL calls behind a thin `IRenderDevice` interface (see §6) so swapping to bgfx (for Metal/Vulkan) or adding a compute backend later is a contained change.

**Concrete stack:**
- Window/context/input: **GLFW** (zlib license).
- GL loader: **glad** (or glbinding) for GL 3.3 core.
- UI: **Dear ImGui** docking branch (MIT).
- Gizmos: **ImGuizmo** (MIT) — provides `Manipulate()` for T/R/S **and a Bounds mode** with draggable corner handles, exactly matching our box-editing need.
- Math: **GLM** (MIT) for vec/mat/quat.

License posture is clean: MIT/BSD/zlib throughout — safe for a commercial product.

---

## 2. Bounding-box interaction (UX + implementation)

### 2.1 Box representation — OBB-native

We store every box as an **Oriented Bounding Box**; an AABB is just an OBB with identity rotation. This avoids a dual code path and lets the user rotate a box to hug an angled object.

```cpp
struct Obb {
    glm::vec3 center;        // world-space center
    glm::quat orientation;   // box-local -> world rotation (identity == AABB)
    glm::vec3 halfExtents;   // >= 0 along local x,y,z

    // Cached for fast tests / rendering:
    glm::mat4 localToWorld() const;       // T(center) * R(orientation) * S(halfExtents... unit cube)
    glm::mat4 worldToLocal() const;       // inverse
    bool      axisAligned = false;        // UI lock; forces orientation = identity
};

struct BoxScene {
    std::vector<Obb>      boxes;
    int                   selected = -1;   // index, -1 == none
    enum class Op { Union, Intersect, PerBox } op = Op::Union;
};
```

**AABB vs OBB math (why OBB is cheap enough):**
- *AABB membership:* `min <= p <= max` componentwise — 6 compares.
- *OBB membership:* transform `p` into box-local frame (`worldToLocal * p`), then it's the same 6 compares against `±halfExtents`. One mat4×vec4 + 6 compares per point. Equivalently, project `(p - center)` onto each local axis and compare against the half-extent (no full matrix needed). This is the standard, well-understood test and is the only thing in the per-point hot loop.

### 2.2 Creating a box

Two complementary creation flows:

**A. Screen-rectangle → frustum slab (fast path, default).**
User left-drags a 2D rectangle over the cloud. We unproject the rectangle's 4 corners through the camera to build a view-frustum sub-volume, then fit a box:

```
screen rect (x0,y0)-(x1,y1)
      │ unproject 4 corners at near & far  → 8 world points (a frustum sub-volume)
      ▼
choose depth slab:
   - depth from points actually under the rect (median ± k·MAD), OR
   - whole visible depth range (fallback)
      ▼
fit OBB:
   - default: AABB in *world* axes covering the slab (axisAligned = true)
   - or: OBB whose local Z = camera forward (box aligned to the view) — nice for angled grabs
      ▼
new Obb pushed to scene, becomes selected, ready for gizmo edit
```
This is the most intuitive "lasso a region" gesture and works even before the user understands 3D handles.

**B. Direct 3D placement.** Click to drop a default unit box at the picked surface point (ray-cast hit, §2.4), or at the cloud centroid; then edit with the gizmo. Good for precise work.

### 2.3 Resize / rotate / translate

- **ImGuizmo** drives translate/rotate/scale on the box's `localToWorld()` matrix. World vs local space toggle; optional grid snapping (`Manipulate(..., snap)`).
- **ImGuizmo Bounds mode** gives draggable **corner/edge handles** that directly change `halfExtents` while keeping the opposite face fixed — the natural "drag the box face" gesture. We decompose the manipulated matrix back into `center / orientation / halfExtents` after each frame.
- **Scale handling:** ImGuizmo scale is folded into `halfExtents` (not stored as a separate scale in the matrix), so the box stays a clean OBB and the membership test needs no scale term.
- **Numeric panel** (ImGui): exact center/size/euler entry; "Snap to AABB" button zeroes rotation; per-box color and enabled checkbox.

```
 ImGui side panel                3D viewport
 ┌───────────────────┐    ┌──────────────────────────┐
 │ Boxes             │    │        ╔═══════╗  ← handles│
 │  [x] Box 0  (sel) │    │      ◄─╢       ╟─►          │
 │  [ ] Box 1        │    │        ║  pts  ║  gizmo axes│
 │ Mode: ⦿T ○R ○S    │    │        ╚═══════╝   ↑→↗      │
 │ Center  x y z     │    │                            │
 │ Size    x y z     │    │   orbit/pan/zoom camera     │
 │ [Snap AABB] [Del] │    └──────────────────────────┘
 │ Combine: Union ▾  │
 └───────────────────┘
```

### 2.4 Picking / ray-cast

Mouse → world ray via unproject of `(mouseX, mouseY, depth0/1)`:

```cpp
Ray r = camera.screenRay(mouse);   // origin + normalized dir in world space
```

- **Box selection (Ray vs OBB):** transform the ray into box-local space (`worldToLocal`), then run the **slab method** (Ray vs AABB) against `±halfExtents`. Pick the nearest box whose `tmin` is smallest. This is the standard ray-OBB trick: an OBB is an AABB in its own frame.
- **Handle picking:** ImGuizmo handles its own hit-testing; our custom handles (if any) are small screen-space quads tested in 2D.
- **Surface point picking (for placement):** for small clouds, brute-force nearest point along the ray (perpendicular-distance threshold). For large clouds, reuse the display LOD/spatial grid (§4) to limit candidates, or read back the GPU depth buffer at the click pixel and unproject — O(1) and resolution-independent. **Recommended:** depth-buffer readback for placement; keep brute force only as a fallback.

### 2.5 Box → crop semantics

- A point is "kept" if it passes the membership test for the active combine op:
  - **Union** (default): inside *any* enabled box.
  - **Intersect:** inside *all* enabled boxes.
  - **PerBox:** export one cloud per box.
- The viewer computes membership only for the **display set** to preview the selection (highlight kept points). The **authoritative crop runs in the core over the full-res cloud** (§5) using the identical OBB math, so preview and export agree.

---

## 3. Camera & display controls

### 3.1 Camera

- **Orbit/pan/zoom** turntable camera around a focus point:
  - LMB-drag (when not over a gizmo/box) — orbit (azimuth/elevation; clamp elevation to avoid gimbal flip).
  - MMB-drag or Shift+LMB — pan (move focus in camera plane).
  - Wheel — dolly/zoom toward cursor; zoom-to-cursor by unprojecting the pointer.
- **Fit-to-cloud:** compute cloud AABB once on load, place focus at its center, set distance so the bounding sphere fits the vertical FOV: `dist = radius / sin(fovY/2)`. Hotkey `F`.
- **Projection:** perspective default; orthographic toggle (helpful for axis-aligned box placement / precise framing).
- **View presets:** top/front/side/iso hotkeys snap orientation — very useful for AABB work.
- Near/far auto-fit from cloud bounds to preserve depth precision (matters for depth-buffer picking).

### 3.2 Point appearance

- **Point size:** slider in pixels; optional **perspective size attenuation** (`gl_PointSize = base * (focalLen / dist)`), clamped. Round-sprite option (discard outside radius in frag shader) for nicer dots.
- **Color-by-attribute:** shader uniform `colorMode`:
  - `RGB` — per-point color if present.
  - `Intensity` — scalar → colormap (viridis/turbo) in shader; min/max auto from percentiles + manual override.
  - `Label` — integer → categorical palette (hash to color); optional per-label show/hide.
  - `Height/Axis` — colormap by Z (or any axis) — handy when no color/intensity exists.
- **Selection shading:** kept points (inside boxes) drawn at full color; rejected points dimmed/desaturated, so the crop preview is obvious. Toggle "show only kept."
- **Background, axis triad, ground grid** toggles.

---

## 4. Performance: point budget, LOD, GPU buffers

### 4.1 Principle

> The **core holds the full-resolution cloud**. The **viewer renders a budgeted, decimated copy** for interactivity. LOD/decimation is **display-only** — it never affects the actual crop, which always runs on full-res data in the core.

### 4.2 Point budget

- Target **≤ ~5–10M points on screen** for a comfortable 60 FPS with `GL_POINTS` on mid GPUs; expose a "max display points" setting.
- Small clouds (under budget): upload the whole thing once, no LOD. This is the common case now and must be instant.
- Large clouds (over budget): build a display LOD (below).

### 4.3 LOD / decimation (display only)

Phased plan, simplest first:

1. **Now — uniform/random subsample to budget.** On load, if `N > budget`, take a deterministic stride/random sample for display; keep an index map back to full-res only if needed for preview (not required, since crop uses the core).
2. **Next — voxel-grid decimation.** Bin points into a voxel grid sized to the budget; render one representative per voxel. Density-uniform, looks better than random when zoomed out.
3. **Later — hierarchical LOD (octree / layered point cloud)** with view-dependent, GPU-driven culling and on-demand upload. The literature shows GPU-driven culling + compute-shader rasterization handling billions of points at ~80 FPS; this is the upgrade path when "large-cloud LOD" becomes real. Keep it behind the `IRenderDevice` seam so it's additive.

A **compute-shader point rasterizer** (rather than `GL_POINTS`) is the known order-of-magnitude win for very large clouds — earmark it for the large-cloud phase, not v1.

### 4.4 GPU buffer strategy

- **Interleaved VBO**: `position (3×f32) | color/attr`. For attribute-colored modes, keep attributes in the buffer and colormap in-shader (no re-upload on colormap change).
- **One static VBO** for the display set, uploaded once with `GL_STATIC_DRAW`; never re-upload per frame.
- Box geometry & gizmos are tiny — their own dynamic buffers / ImGui draw lists.
- Switching color mode / colormap / point size = **uniform changes only**, zero re-upload.
- For streaming LOD later: **persistent-mapped ring buffers** (`GL_MAP_PERSISTENT_BIT`) or buffer orphaning for on-demand chunk upload.

### 4.5 Staying responsive while core holds full cloud

- Decouple threads: **render thread never blocks on the core.** Loading/decimation/crop run on worker threads; the viewer shows progress and renders whatever display set is ready.
- Crop preview (highlight) is computed over the *display set* (cheap, parallel `std::for_each(par_unseq)` or a fragment-side test) so dragging a box updates instantly; the *authoritative* crop is dispatched to the core asynchronously and only on demand (export / explicit "apply").
- Double-buffer the display VBO if LOD swaps the visible set, so a swap never stalls a frame.

---

## 5. Coupling to the core / crop engine

### 5.1 Recommendation: in-process, shared memory

**Run the viewer in the same process as the core** as a single executable. The core owns the canonical full-res cloud in memory; the viewer receives a **read-only view** (`std::span` / pointer + count) and builds its own GPU display set from it. No copy, no serialization, no gzip on the hot path.

```
            ┌────────────────────────── one process ──────────────────────────┐
            │                                                                   │
  files ───►│  CORE                                ICloudView (read-only)       │
 (PLY/PCD/  │  ┌───────────────────┐   span<const Point>   ┌──────────────────┐ │
  NPZ,      │  │ full-res cloud     ├──────────────────────►│ VIEWER           │ │
  gzipped)  │  │ + spatial index    │                       │  display LOD/VBO │ │
            │  │ + crop engine (OBB)│◄───── BoxScene ───────┤  ImGui/ImGuizmo  │ │
            │  └─────────┬──────────┘   apply/export        │  camera, picking │ │
            │            │                                   └──────────────────┘ │
            │   gzipped export ◄── crop result (full-res)                        │
            └───────────────────────────────────────────────────────────────────┘
```

**Why in-process:** zero-copy access to potentially huge clouds is the entire performance argument for choosing C++. Serializing+gzipping millions of points just to hand them to a viewer in the same machine would dominate latency and defeat the point. Shared-memory/borrowed-span is the fastest possible coupling.

**Where gzip belongs:** the product's "gzip data transfer between components" requirement maps to (a) **import** of compressed point files and (b) **export** of the cropped result, and (c) a **future out-of-process / remote / headless mode** where the viewer is a thin client. None of those are the interactive viewer↔core inner loop.

### 5.2 Keep the seam swappable

Define the boundary as an interface so the viewer never assumes in-process:

```cpp
// Read side: viewer pulls geometry to display.
struct ICloudView {
    virtual std::size_t size() const = 0;
    virtual std::span<const PointXYZRGBI> points() const = 0;  // borrowed, read-only
    virtual Aabb bounds() const = 0;
    // Future remote impl: returns a locally-cached, gzip-decoded chunk.
};

// Write side: viewer pushes box edits / requests crop.
struct ICropController {
    virtual void   setBoxScene(const BoxScene&) = 0;            // boxes + combine op
    virtual CropStats previewCount() const = 0;                 // cheap estimate
    virtual std::future<CroppedCloud> applyCrop() = 0;          // full-res, async
    virtual void   exportCrop(const Path&, ExportOpts) = 0;     // core does gzip
};
```

- **v1:** `InProcessCloudView` wraps the core's buffer directly (no copy).
- **future:** `RemoteCloudView` receives gzipped chunks from a core service and decodes locally — viewer code is unchanged. Transport (sockets/shared-mem/IPC framing, gzip params) is the **core-architecture agent's** call; we only commit to the `ICloudView`/`ICropController` contract and to identical OBB math on both sides.

### 5.3 Implications

- **Single binary, simplest deploy, fastest** — chosen for v1.
- Trade-off: a crash in viewer or core takes down both, and you can't run the core headless on a server with a remote thin-client GUI *yet*. The interface seam above preserves that option without paying for it now.
- The viewer and core **must share the exact OBB membership implementation** (one header, used by both the GPU/preview path and the core's full-res crop) so preview === export.

---

## 6. Architecture sketch (viewer-internal)

```
Viewer
 ├─ App (GLFW window, GL context, main loop, ImGui frame)
 ├─ IRenderDevice            // thin GL wrapper; swappable for bgfx/compute later
 │   └─ GlDevice (OpenGL 3.3 core): VBO, shaders, draw
 ├─ PointRenderer            // builds/owns display VBO, colormap shaders, point-size
 ├─ LodManager               // budget, decimation; octree streaming (later)
 ├─ Camera                   // orbit/pan/zoom, fit, presets, screenRay()
 ├─ BoxScene + BoxEditor     // ImGuizmo integration, handles, create-from-rect
 ├─ Picker                   // ray vs OBB, depth-buffer readback for placement
 ├─ obb_math.hpp  ⟵ SHARED with core (membership, ray-OBB)
 └─ UiPanels (ImGui)         // box list, attribute/colormap, camera, perf settings
        │
        ▼ ICloudView / ICropController
      CORE
```

Key shared file: **`obb_math.hpp`** — `bool contains(const Obb&, vec3)`, `optional<float> rayObb(const Obb&, Ray)`, matrix builders. Compiled into both viewer (preview) and core (authoritative crop).

---

## 7. Risks & open questions

- **ImGuizmo Bounds mode ergonomics** for our exact box-edit gesture need a spike; if its bounds handles feel wrong, fall back to custom screen-space face handles (math is simple given the OBB).
- **Depth-buffer picking precision** depends on good near/far auto-fit (§3.1).
- **Decimation quality** for very sparse vs very dense clouds — voxel decimation (phase 2) should land before we market "large clouds."
- **Coordinate conventions** (Y-up vs Z-up; NPZ/PCD may differ) must be fixed once and shared with core.
- Final call on whether export/preview ever need full-res in the viewer process (e.g., screenshot of cropped result at full density) — currently assumed no; core renders/export handles it.

---

## 8. References

- Dear ImGui — https://github.com/ocornut/imgui (MIT)
- ImGuizmo (Manipulate / Bounds gizmo) — https://github.com/CedricGuillemet/ImGuizmo (MIT)
- Open3D (C++ GUI on Filament) — https://www.open3d.org/ (MIT; reference, not chosen)
- PCL Visualization (VTK-based) — https://pointclouds.org/documentation/group__visualization.html (BSD; reference)
- Magnum graphics middleware — https://magnum.graphics/ (MIT; runner-up)
- bgfx multi-backend renderer — comparison: https://dev.to/funatsufumiya/comparison-of-c-low-level-graphics-cross-platform-frameworks-and-libraries-58e5
- Ray-OBB picking — http://www.opengl-tutorial.org/miscellaneous/clicking-on-objects/picking-with-custom-ray-obb-function/
- AABB vs OBB / SAT — https://dev.to/pratyush_mohanty_6b8f2749/the-math-behind-bounding-box-collision-detection-aabb-vs-obbseparate-axis-theorem-1gdn
- GPU-Accelerated LOD Generation for Point Clouds (Schütz 2023) — https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14877
- Virtualized / GPU-driven point cloud rendering (2025) — https://pubmed.ncbi.nlm.nih.gov/40257869/
- Rendering Point Clouds with Compute Shaders — https://arxiv.org/pdf/1908.02681
