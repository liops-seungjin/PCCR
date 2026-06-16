# 04 — Build System, Dependency Management & Dev Tooling

**Component owner:** Build/Tooling architect
**Status:** Design proposal
**Scope:** C++ standard & compilers, physical layout, CMake target graph, dependency management, testing, dev tooling (format/tidy/sanitizers/presets/CI), and the initial scaffold.

---

## 0. Executive decisions (TL;DR)

| Decision | Choice | Rationale |
|---|---|---|
| C++ standard | **C++20** (no compiler extensions) | Concepts, `<span>`, `<filesystem>`, designated inits, `std::format` (with fallback). Universally available on the toolchains below. Not C++23 — `std::format`/`std::expected` support is still uneven across stable distro compilers. |
| Compilers | GCC ≥ 11, Clang ≥ 14, MSVC ≥ 19.3x (VS2022) | Linux is primary (GCC/Clang); keep MSVC green to stay portable. |
| Build system | **CMake ≥ 3.24** + **CMakePresets.json (v6)** | 3.24 gives `FIND_PACKAGE_ARGS` for FetchContent, `--preset` test/build presets, and dependency-provider hook for vcpkg. |
| Dependency manager | **vcpkg (manifest mode)** as primary; `find_package`-first so system/Conan still work | Lowest-friction for a CMake-only project; manifest = reproducible; binary caching makes heavy deps tolerable. |
| Heavy deps (PCL/VTK/Open3D) | **Optional, off by default**, isolated behind feature flags | They dominate build time and dependency surface; the product treats formats/GUI as pluggable, so the build must too. |
| GUI stack | OpenGL + GLFW + Dear ImGui + ImGuizmo + glm | Light, vcpkg-available, no VTK/Open3D mandatory. |
| Test framework | **GoogleTest** (via vcpkg), driven by CTest | Mature, fixtures + parameterized tests for IO round-trips, gtest-discover for CTest. doctest is the fallback if we want header-only/fast-compile. |
| Error handling | **`tl-expected`** → `cc::Result<T> = tl::expected<T, Error>` | C++20 target lacks reliable `std::expected`; single-header CC0 shim, one-line swap to `std::expected` on C++23. |
| Compression | **zlib-ng** (gzip transport/`.gz` codec) + **miniz** (NPZ ZIP container only) | One general codec (zlib-ng, RFC-1952 gzip framing); miniz is a per-format container handler, not a second codec. |
| Bench | `tests/bench` (micro-benchmarks) | Calibrates the `IndexPolicy` octree threshold (brute-force vs octree across N × selectivity). |
| Feature flags | `CLOUDCROPPER_BUILD_GUI`, `_WITH_PLY`, `_WITH_PCD`, `_WITH_NPZ`, `_WITH_PCL`, `_TRANSPORT_GZIP`, `_BUILD_TESTS`, `_BUILD_BENCH`, `_ENABLE_SANITIZERS` | Build-time pluggability matching the "optional" product goal. |

---

## 1. C++ standard, compilers, and project layout

### 1.1 Standard & compiler policy
- `CMAKE_CXX_STANDARD 20`, `CXX_STANDARD_REQUIRED ON`, `CXX_EXTENSIONS OFF` (use `-std=c++20`, not `gnu++20`, for portability).
- Set the standard **per target** via `target_compile_features(<tgt> PUBLIC cxx_std_20)` so consumers inherit it; avoid global `CMAKE_CXX_FLAGS`.
- Warnings as a curated set per target (`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`; `/W4` on MSVC). `-Werror` only in CI, not in default dev builds.

### 1.2 Component → module mapping
Five product components map to five static libraries plus one executable:

| Product component | CMake target | Kind |
|---|---|---|
| core (geometry, crop engine, KD-tree) | `cloudcropper_core` | static lib |
| io (PLY/PCD/NPZ readers/writers) | `cloudcropper_io` | static lib |
| transport (gzip / data transfer) | `cloudcropper_transport` | static lib |
| viewer (GL/ImGui/ImGuizmo) | `cloudcropper_viewer` | static lib (built only when GUI on) |
| app (CLI + wiring + entry point) | `cloudcropper` | executable |

All libraries are exported under the `cloudcropper::` namespace alias (`cloudcropper::core`, etc.) so the app and tests link by alias and intra-project deps look like external package deps.

### 1.3 Directory tree

```
cloudcropper/
├── CMakeLists.txt                  # top-level: project(), options, add_subdirectory()
├── CMakePresets.json               # shared configure/build/test presets (checked in)
├── CMakeUserPresets.json           # per-dev overrides (gitignored)
├── vcpkg.json                      # manifest: deps + features
├── vcpkg-configuration.json        # registry/baseline pin (reproducibility)
├── .clang-format
├── .clang-tidy
├── .gitignore
├── .gitmodules                     # only if vcpkg is vendored as a submodule
├── LICENSE
├── README.md
├── cmake/                          # reusable CMake helpers
│   ├── CompilerWarnings.cmake      # warning interface target
│   ├── Sanitizers.cmake            # asan/ubsan interface target + option glue
│   ├── ProjectOptions.cmake        # central option() declarations
│   └── modules/                    # custom Find*.cmake if ever needed
├── include/
│   └── cloudcropper/               # PUBLIC headers, mirrors module names
│       ├── core/                   #   crop.hpp, cloud.hpp, kdtree.hpp, bbox.hpp
│       ├── io/                     #   reader.hpp, writer.hpp, format.hpp
│       ├── transport/              #   gzip.hpp
│       └── viewer/                 #   viewer.hpp  (guarded by GUI flag)
├── src/
│   ├── core/
│   │   ├── CMakeLists.txt
│   │   ├── crop.cpp
│   │   ├── cloud.cpp
│   │   └── kdtree.cpp
│   ├── io/
│   │   ├── CMakeLists.txt
│   │   ├── ply_io.cpp              # compiled only if WITH_PLY
│   │   ├── pcd_io.cpp              # compiled only if WITH_PCD
│   │   ├── npz_io.cpp              # compiled only if WITH_NPZ
│   │   └── format_registry.cpp     # always-on dispatch / plugin registry
│   ├── transport/
│   │   ├── CMakeLists.txt
│   │   └── gzip.cpp
│   ├── viewer/
│   │   ├── CMakeLists.txt
│   │   └── viewer.cpp              # whole target gated by BUILD_GUI
│   └── app/
│       ├── CMakeLists.txt
│       └── main.cpp                # CLI parse → core/io; GUI optional
├── tests/
│   ├── CMakeLists.txt
│   ├── core/test_crop.cpp
│   ├── core/test_kdtree.cpp
│   ├── io/test_ply_roundtrip.cpp
│   ├── io/test_npz_roundtrip.cpp
│   ├── transport/test_gzip.cpp
│   ├── bench/bench_index.cpp        # brute-force vs octree sweep → IndexPolicy threshold
│   ├── fixtures/
│   │   ├── cube.ply               # tiny known clouds (≤ a few KB)
│   │   ├── plane.pcd
│   │   └── small.npz
│   └── support/TestData.hpp        # CLOUDCROPPER_TEST_DATA_DIR helper
├── third_party/                    # ONLY vendored bits not in vcpkg (e.g. ImGuizmo if pinned)
│   └── vcpkg/                       # optional: vcpkg as a submodule for hermetic CI
├── docs/
│   ├── design/                     # this file lives here
│   └── ...
└── .github/
    └── workflows/ci.yml
```

Rule: **public headers live in `include/cloudcropper/<module>/`**, private/impl headers live next to their `.cpp` in `src/<module>/`. Targets use `target_include_directories(<tgt> PUBLIC include PRIVATE src/<module>)`.

---

## 2. Build system: CMake skeletons

### 2.1 Top-level `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.24)

# Pull in the vcpkg toolchain via a dependency provider if present.
# (Preferred: pass it through CMAKE_TOOLCHAIN_FILE in a preset; this guard
#  keeps a plain `cmake -S . -B build` working when vcpkg is vendored.)
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "vcpkg toolchain")
endif()

project(CloudCropper
  VERSION 0.1.0
  DESCRIPTION "Interactive point cloud cropping tool"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---- Global hygiene -------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clang-tidy / IDEs
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Feature options (central) -------------------------------------------
include(ProjectOptions)   # declares the CLOUDCROPPER_* options below

# ---- Reusable interface targets ------------------------------------------
include(CompilerWarnings) # defines cloudcropper_warnings  (INTERFACE)
include(Sanitizers)       # defines cloudcropper_sanitizers (INTERFACE)

# ---- Dependencies (find_package; satisfied by vcpkg manifest) ------------
if(CLOUDCROPPER_WITH_PLY)
  find_package(tinyply CONFIG REQUIRED)
endif()
if(CLOUDCROPPER_WITH_NPZ)
  find_package(miniz CONFIG REQUIRED)        # NPZ ZIP container only
endif()
if(CLOUDCROPPER_TRANSPORT_GZIP)
  find_package(zlib-ng CONFIG REQUIRED)      # single gzip/.gz codec (RFC 1952)
endif()
find_package(tl-expected CONFIG REQUIRED)    # cc::Result<T>
find_package(nanoflann CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

if(CLOUDCROPPER_WITH_PCL)
  find_package(PCL CONFIG REQUIRED COMPONENTS common io)   # heavy; opt-in only
endif()

if(CLOUDCROPPER_BUILD_GUI)
  find_package(glfw3 CONFIG REQUIRED)
  find_package(imgui CONFIG REQUIRED)
  find_package(OpenGL REQUIRED)
  # ImGuizmo: vendored in third_party/ or via a vcpkg overlay port.
endif()

# ---- Subprojects ----------------------------------------------------------
add_subdirectory(src/core)
add_subdirectory(src/io)
add_subdirectory(src/transport)
if(CLOUDCROPPER_BUILD_GUI)
  add_subdirectory(src/viewer)
endif()
add_subdirectory(src/app)

if(CLOUDCROPPER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
```

### 2.2 `cmake/ProjectOptions.cmake`

```cmake
include_guard(GLOBAL)
option(CLOUDCROPPER_BUILD_GUI       "Build the interactive GL/ImGui viewer" ON)
option(CLOUDCROPPER_BUILD_TESTS     "Build unit tests"                      ON)
option(CLOUDCROPPER_BUILD_BENCH     "Build micro-benchmarks (index tuning)" OFF)
option(CLOUDCROPPER_WITH_PLY        "Enable PLY format support (tinyply)"   ON)
option(CLOUDCROPPER_WITH_PCD        "Enable PCD format support"             ON)
option(CLOUDCROPPER_WITH_NPZ        "Enable NPZ format support (miniz ZIP)" ON)
option(CLOUDCROPPER_WITH_PCL        "Use PCL for PCD/IO (heavy, opt-in)"    OFF)
option(CLOUDCROPPER_TRANSPORT_GZIP  "Enable gzip data transfer (zlib-ng)"   ON)
option(CLOUDCROPPER_ENABLE_SANITIZERS "Build with ASan+UBSan"               OFF)
option(CLOUDCROPPER_WARNINGS_AS_ERRORS "Treat warnings as errors"           OFF)
```

### 2.3 Per-module target — `src/io/CMakeLists.txt` (the interesting one)

```cmake
add_library(cloudcropper_io STATIC src/io/format_registry.cpp)
add_library(cloudcropper::io ALIAS cloudcropper_io)

target_compile_features(cloudcropper_io PUBLIC cxx_std_20)
target_include_directories(cloudcropper_io
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(cloudcropper_io
  PUBLIC  cloudcropper::core
  PRIVATE cloudcropper_warnings $<$<BOOL:${CLOUDCROPPER_ENABLE_SANITIZERS}>:cloudcropper_sanitizers>)

# --- Format plug-ins compiled & defined conditionally ---
if(CLOUDCROPPER_WITH_PLY)
  target_sources(cloudcropper_io PRIVATE ply_io.cpp)
  target_link_libraries(cloudcropper_io PRIVATE tinyply)
  target_compile_definitions(cloudcropper_io PUBLIC CLOUDCROPPER_HAS_PLY=1)
endif()

if(CLOUDCROPPER_WITH_NPZ)
  target_sources(cloudcropper_io PRIVATE npz_io.cpp)
  target_link_libraries(cloudcropper_io PRIVATE miniz::miniz)  # ZIP container only
  target_compile_definitions(cloudcropper_io PUBLIC CLOUDCROPPER_HAS_NPZ=1)
endif()

# `cloudcropper_io` also links cc::Result (tl::expected) transitively via core,
# which carries tl-expected as a PUBLIC dependency (see src/core/CMakeLists.txt).
# The gzip codec lives in `cloudcropper_transport`, which links zlib-ng:
#   target_link_libraries(cloudcropper_transport PRIVATE zlib-ng::zlib-ng)

if(CLOUDCROPPER_WITH_PCD)
  target_sources(cloudcropper_io PRIVATE pcd_io.cpp)
  target_compile_definitions(cloudcropper_io PUBLIC CLOUDCROPPER_HAS_PCD=1)
  if(CLOUDCROPPER_WITH_PCL)
    target_link_libraries(cloudcropper_io PRIVATE ${PCL_LIBRARIES})
    target_compile_definitions(cloudcropper_io PRIVATE CLOUDCROPPER_PCD_USE_PCL=1)
  endif()
endif()
```

`core`, `transport`, `viewer`, `app` follow the same shape. The app target:

```cmake
add_executable(cloudcropper src/app/main.cpp)
target_link_libraries(cloudcropper
  PRIVATE cloudcropper::core cloudcropper::io cloudcropper::transport
          CLI11::CLI11 cloudcropper_warnings
          $<$<BOOL:${CLOUDCROPPER_ENABLE_SANITIZERS}>:cloudcropper_sanitizers>)
if(CLOUDCROPPER_BUILD_GUI)
  target_link_libraries(cloudcropper PRIVATE cloudcropper::viewer)
  target_compile_definitions(cloudcropper PRIVATE CLOUDCROPPER_HAS_GUI=1)
endif()
```

This gives a clean **headless build** (`-DCLOUDCROPPER_BUILD_GUI=OFF`) that drops GLFW/ImGui/OpenGL entirely — important for servers/CI and the "viewer optional" goal.

---

## 3. Dependency management

### 3.1 Comparison

| Approach | Pros | Cons | Verdict for CloudCropper |
|---|---|---|---|
| **vcpkg (manifest)** | One `vcpkg.json` beside CMakeLists; transparent CMake toolchain integration; `find_package` "just works"; binary caching; baseline pin = reproducible; **features** map 1:1 onto our optional formats/GUI | Build-from-source first time is slow (mitigated by binary cache); registry curation lag for niche libs (ImGuizmo) | **PRIMARY.** Best fit for a CMake-only, Linux-first project with optional components. |
| **Conan 2** | Powerful lockfiles, profiles, prebuilt binaries from ConanCenter, great cross-compile/ABI handling | Extra non-CMake step (`conan install`); steeper ramp; two toolchains to learn | Strong alternative; pick if we later need rich ABI/profile matrices. Not needed now. |
| **FetchContent** | Zero extra tooling; fully hermetic, source-pinned; great for small header-only libs (tinyply, nanoflann, glm, doctest) | Slow clean builds; you build/patch heavy deps yourself; no binary cache | **Fallback** for tiny header-only deps via `FIND_PACKAGE_ARGS` so vcpkg is used when present, FetchContent otherwise. Never for PCL/VTK. |
| **git submodules** | Trivial, no package manager, exact pin | Manual transitive deps; manual build flags; painful for anything non-trivial | Use only to vendor **vcpkg itself** (hermetic CI) and one-off ports like ImGuizmo. |

**Recommendation: vcpkg manifest mode as the single source of truth**, written `find_package`-first so the same `CMakeLists.txt` also resolves against a Conan-generated toolchain or system packages. Heavy deps stay behind options so a default build never touches them.

### 3.2 `vcpkg.json` (manifest with features)

```json
{
  "$schema": "https://raw.githubusercontent.com/microsoft/vcpkg/master/scripts/vcpkg.schema.json",
  "name": "cloudcropper",
  "version": "0.1.0",
  "builtin-baseline": "<pinned-vcpkg-commit-sha>",
  "dependencies": [
    "nanoflann",
    "glm",
    "cli11",
    "tl-expected"
  ],
  "default-features": ["ply", "npz", "pcd", "gzip", "gui"],
  "features": {
    "ply":  { "description": "PLY support",  "dependencies": ["tinyply"] },
    "npz":  { "description": "NPZ support (ZIP container)", "dependencies": ["miniz"] },
    "gzip": { "description": "gzip data transfer codec",    "dependencies": ["zlib-ng"] },
    "pcd":  { "description": "PCD support",  "dependencies": [] },
    "pcl":  { "description": "Heavy PCL backend",
              "dependencies": [{ "name": "pcl", "features": ["core"] }] },
    "gui":  { "description": "GL/ImGui viewer",
              "dependencies": ["glfw3", { "name": "imgui",
                "features": ["glfw-binding", "opengl3-binding"] }] },
    "tests":{ "description": "Unit tests", "dependencies": ["gtest"] }
  }
}
```

The CMake `CLOUDCROPPER_*` options and these vcpkg `features` are selected together by presets (§5): a preset sets `VCPKG_MANIFEST_FEATURES` **and** the matching `-D` cache vars, so "GUI on" pulls the right libs *and* compiles the viewer target.

### 3.3 Heavy-dep strategy (PCL / VTK / Open3D)
- **Off by default.** `CLOUDCROPPER_WITH_PCL=OFF`; PCD parsing ships with a small custom reader so the common path needs no PCL.
- When enabled, isolate all PCL includes inside `src/io/pcd_io.cpp` behind `CLOUDCROPPER_PCD_USE_PCL` — never leak PCL headers into public `include/`.
- VTK/Open3D are **not** adopted; the GUI uses the lightweight GLFW/ImGui stack. If a future "advanced viewer" wants Open3D, it goes in its own optional target/feature, never in the default graph.
- CI runs the default (no-PCL) matrix on every push; a separate, manually-triggered/nightly job exercises `-DCLOUDCROPPER_WITH_PCL=ON` with vcpkg binary caching so the slow build doesn't block PRs.

---

## 4. Testing

- **Framework: GoogleTest** via vcpkg `tests` feature; registered with CTest using `gtest_discover_tests()`. (doctest is the swap-in if compile time becomes a concern — same CTest wiring.)
- One test executable per module (`core_tests`, `io_tests`, `transport_tests`) keeps link graphs small and lets headless CI skip viewer tests.

### `tests/CMakeLists.txt` (sketch)

```cmake
find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(io_tests io/test_ply_roundtrip.cpp io/test_npz_roundtrip.cpp)
target_link_libraries(io_tests PRIVATE
  cloudcropper::io GTest::gtest_main
  $<$<BOOL:${CLOUDCROPPER_ENABLE_SANITIZERS}>:cloudcropper_sanitizers>)
target_compile_definitions(io_tests PRIVATE
  CLOUDCROPPER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
gtest_discover_tests(io_tests)
```

### What to test
- **IO round-trips (the key invariant):** generate a known cloud → write PLY/PCD/NPZ → read back → assert point count, XYZ within float epsilon, and attributes (normals/colors/intensity) preserved. Cover ascii vs binary PLY and compressed vs raw NPZ. Use `TYPED_TEST`/parameterized cases across formats so one body covers all writers.
- **Crop engine:** axis-aligned and oriented bounding-box crops on a synthetic unit cube/grid; assert exactly the expected indices survive; test boundary inclusivity, empty result, and full-pass. Property test: crop(crop(X, B), B) == crop(X, B) (idempotence).
- **KD-tree:** radius/knn queries vs a brute-force reference on small random clouds.
- **transport/gzip:** compress→decompress round-trip on cloud byte buffers; corrupted-stream error path.
- **Fixtures:** keep tiny (≤ few KB) deterministic files in `tests/fixtures/`; large data is generated in-test, never committed. `CLOUDCROPPER_TEST_DATA_DIR` compile-def locates fixtures regardless of cwd.

### Benchmarks (`tests/bench`, gated by `CLOUDCROPPER_BUILD_BENCH`)
- `bench_index.cpp` sweeps synthetic clouds across **N (10k→50M) × box selectivity** comparing brute-force parallel scan vs octree build+query, to **calibrate `IndexPolicy::auto_point_threshold`** (doc 03 §3.5) on the target hardware. Off by default (not a PR gate); run manually and record the chosen threshold. GoogleTest's bench support or a tiny hand-rolled timer is sufficient — no extra dependency required.

---

## 5. Dev tooling

### 5.1 `.clang-format` (anchor)
Base `Google` (or `LLVM`), `ColumnLimit: 100`, `IndentWidth: 4`, `PointerAlignment: Left`, sorted/grouped includes. Enforced by a `format` make-style target and checked in CI (`clang-format --dry-run --Werror`).

### 5.2 `.clang-tidy`
Enable `bugprone-*, performance-*, modernize-*, cppcoreguidelines-*, readability-*`; disable the noisy `modernize-use-trailing-return-type` and `*-magic-numbers`. Runs over `compile_commands.json` (already emitted). CI runs it on changed files; locally `-DCMAKE_CXX_CLANG_TIDY=clang-tidy` is opt-in (it slows builds).

### 5.3 Sanitizers — `cmake/Sanitizers.cmake`

```cmake
include_guard(GLOBAL)
add_library(cloudcropper_sanitizers INTERFACE)
if(CLOUDCROPPER_ENABLE_SANITIZERS AND NOT MSVC)
  target_compile_options(cloudcropper_sanitizers INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(cloudcropper_sanitizers INTERFACE
    -fsanitize=address,undefined)
endif()
```
ASan + UBSan are the default *checked* build (run via the `debug-asan` preset); TSan is a separate, occasional lane. Targets opt in by linking the interface lib (already wired in §2/§4 via a generator expression).

### 5.4 `CMakePresets.json` (v6, checked in)

```json
{
  "version": 6,
  "cmakeMinimumRequired": { "major": 3, "minor": 24, "patch": 0 },
  "configurePresets": [
    {
      "name": "base", "hidden": true,
      "binaryDir": "${sourceDir}/build/${presetName}",
      "generator": "Ninja",
      "toolchainFile": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
      "cacheVariables": { "CMAKE_EXPORT_COMPILE_COMMANDS": "ON" }
    },
    {
      "name": "dev", "inherits": "base",
      "displayName": "Dev (RelWithDebInfo, full features)",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
        "CLOUDCROPPER_BUILD_GUI": "ON", "CLOUDCROPPER_BUILD_TESTS": "ON"
      },
      "environment": { "VCPKG_MANIFEST_FEATURES": "ply;npz;pcd;gzip;gui;tests" }
    },
    {
      "name": "headless", "inherits": "dev",
      "displayName": "Headless (no GUI)",
      "cacheVariables": { "CLOUDCROPPER_BUILD_GUI": "OFF" },
      "environment": { "VCPKG_MANIFEST_FEATURES": "ply;npz;pcd;gzip;tests" }
    },
    {
      "name": "debug-asan", "inherits": "dev",
      "displayName": "Debug + ASan/UBSan",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CLOUDCROPPER_ENABLE_SANITIZERS": "ON"
      }
    },
    {
      "name": "release", "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release", "CLOUDCROPPER_BUILD_TESTS": "OFF"
      },
      "environment": { "VCPKG_MANIFEST_FEATURES": "ply;npz;pcd;gzip;gui" }
    }
  ],
  "buildPresets": [
    { "name": "dev", "configurePreset": "dev" },
    { "name": "headless", "configurePreset": "headless" },
    { "name": "debug-asan", "configurePreset": "debug-asan" },
    { "name": "release", "configurePreset": "release" }
  ],
  "testPresets": [
    { "name": "dev", "configurePreset": "dev",
      "output": { "outputOnFailure": true } },
    { "name": "debug-asan", "configurePreset": "debug-asan",
      "output": { "outputOnFailure": true } }
  ]
}
```
`CMakeUserPresets.json` (gitignored) is where a dev overrides `VCPKG_ROOT`, generator, or compiler.

### 5.5 CI — `.github/workflows/ci.yml` (outline)

```yaml
name: ci
on: [push, pull_request]
jobs:
  build-test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        preset: [headless]            # headless = fast PR gate, no GL needed
    runs-on: ${{ matrix.os }}
    env:
      VCPKG_ROOT: ${{ github.workspace }}/third_party/vcpkg
      VCPKG_BINARY_SOURCES: "clear;x-gha,readwrite"   # GH Actions binary cache
    steps:
      - uses: actions/checkout@v4
        with: { submodules: recursive }   # vendored vcpkg
      - uses: lukka/run-vcpkg@v11
      - uses: lukka/get-cmake@latest
      - run: cmake --preset ${{ matrix.preset }}
      - run: cmake --build --preset ${{ matrix.preset }}
      - run: ctest --preset dev --test-dir build/${{ matrix.preset }}

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: clang-format --dry-run --Werror $(git ls-files '*.cpp' '*.hpp')
      # clang-tidy on changed files using build/headless/compile_commands.json

  sanitizers:                            # separate lane, Linux only
    runs-on: ubuntu-latest
    env: { VCPKG_ROOT: ${{ github.workspace }}/third_party/vcpkg }
    steps:
      - uses: actions/checkout@v4
        with: { submodules: recursive }
      - run: cmake --preset debug-asan && cmake --build --preset debug-asan
      - run: ctest --preset debug-asan --test-dir build/debug-asan

  pcl-nightly:                           # heavy, schedule-only
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    steps: [ "...configure with -DCLOUDCROPPER_WITH_PCL=ON, build, test..." ]
```
PR gate = headless build/test + format/tidy + ASan. GUI and PCL builds run on a slower schedule with binary caching so they never block PRs.

---

## 6. README / quickstart outline + initial scaffold

### 6.1 README outline
1. What it is (one paragraph) + screenshot.
2. Features matrix (formats, GUI, transport) and which are optional.
3. **Prerequisites:** CMake ≥ 3.24, Ninja, a C++20 compiler, `VCPKG_ROOT` set (or vendored submodule).
4. **Quickstart:**
   ```bash
   git clone --recursive <repo> && cd cloudcropper
   export VCPKG_ROOT=$PWD/third_party/vcpkg   # or your own checkout
   cmake --preset dev          # configure + resolve deps via vcpkg manifest
   cmake --build --preset dev  # build all targets
   ctest --preset dev          # run tests
   ./build/dev/cloudcropper --help
   ```
   Headless: `cmake --preset headless && cmake --build --preset headless`.
5. Build options table (the `CLOUDCROPPER_*` flags) and presets.
6. Dev workflow: format, tidy, `debug-asan` preset.
7. Layout map + contributing.

### 6.2 Initial scaffold files a dev needs to `cmake --preset … && build`
Minimum set to get a green configure/build/test:
- `CMakeLists.txt`, `cmake/{ProjectOptions,CompilerWarnings,Sanitizers}.cmake`
- `CMakePresets.json`, `.gitignore` (build/, CMakeUserPresets.json), `.clang-format`, `.clang-tidy`
- `vcpkg.json`, `vcpkg-configuration.json`, vcpkg submodule under `third_party/vcpkg`
- Per-module `src/<m>/CMakeLists.txt` + at least one `.cpp` and matching public header in `include/cloudcropper/<m>/`
- `src/app/main.cpp` (CLI stub that links core/io)
- `tests/CMakeLists.txt` + one passing test per module + `tests/fixtures/*`
- `.github/workflows/ci.yml`, `README.md`, `LICENSE`

A bootstrap script (`scripts/bootstrap.sh`) that clones the vcpkg submodule and runs `cmake --preset dev` is a nice-to-have for first-run onboarding.

---

## Sources
- [vcpkg vs Conan vs FetchContent — CMake Discourse](https://discourse.cmake.org/t/fetchcontent-vs-vcpkg-conan/6578)
- [The state of C++ package management — twdev.blog](https://twdev.blog/2024/08/cpp_pkgmng1/)
- [Configure and build with CMake Presets — Microsoft Learn](https://learn.microsoft.com/en-us/cpp/build/cmake-presets-vs?view=msvc-170)
- [The Complete C/C++ Sanitizers Handbook](https://gist.github.com/MangaD/3b46e4c5ef4c63e44a21bed39ae64093)
- [Using AddressSanitizer in a CMake project — Marek's blog](https://felsoci.sk/blog/using-address-sanitizer-asan-in-a-cmake-project.html)
