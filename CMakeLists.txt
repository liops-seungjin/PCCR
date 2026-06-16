cmake_minimum_required(VERSION 3.24)

# NOTE: The first vertical slice builds with zero external dependencies (pure
# C++20 + std) so it configures/builds/tests anywhere. The full vcpkg-based
# dependency wiring (happly, glm, zlib-ng, GTest, ...) from docs/design/04 lands
# as those libraries are integrated; targets and layout already match it.
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "vcpkg toolchain")
endif()

project(CloudCropper
  VERSION 0.1.0
  DESCRIPTION "Interactive point cloud cropping tool"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CLOUDCROPPER_BUILD_TESTS "Build unit tests"                ON)
option(CLOUDCROPPER_BUILD_GUI   "Build the interactive GL/ImGui viewer" OFF)
option(CLOUDCROPPER_WITH_PLY    "Enable PLY format support"       ON)
option(CLOUDCROPPER_WITH_PCD    "Enable PCD format support"       ON)
option(CLOUDCROPPER_WITH_KISS_MATCHER "Fetch + build the KISS-Matcher registration backend" OFF)
option(CLOUDCROPPER_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

include(CompilerWarnings)

# Optional dependencies — resolved by the vcpkg manifest when the toolchain is
# active (preset "vcpkg"), silently absent in the zero-dep "dev" preset. Written
# find_package-first (docs/design/04 §3) so the same tree builds either way.
find_package(tl-expected CONFIG QUIET)
find_package(glm CONFIG QUIET)
find_package(nanoflann CONFIG QUIET)
find_package(CLI11 CONFIG QUIET)
find_package(miniz CONFIG QUIET)              # NPZ ZIP container
find_package(unofficial-liblzf CONFIG QUIET)  # PCD binary_compressed (LZF)
find_package(ZLIB QUIET)                       # gzip transport (.gz)
find_package(unofficial-sqlite3 CONFIG QUIET) # ROS2 bag (.db3) reader
find_package(unofficial-mcap CONFIG QUIET)    # ROS2 bag (.mcap) reader
find_package(Eigen3 CONFIG QUIET)             # registration backends
find_package(small_gicp CONFIG QUIET)         # GICP family (vcpkg small-gicp)

# GUI viewer deps — only required when the viewer is being built (preset "gui").
if(CLOUDCROPPER_BUILD_GUI)
  find_package(glfw3 CONFIG REQUIRED)
  find_package(imgui CONFIG REQUIRED)
  find_package(imguizmo CONFIG REQUIRED)
  find_package(glad CONFIG REQUIRED)
  find_package(OpenGL REQUIRED)
endif()

add_subdirectory(src/common)
add_subdirectory(src/core)
add_subdirectory(src/io)
if(TARGET ZLIB::ZLIB)
  add_subdirectory(src/transport)
endif()
if(TARGET Eigen3::Eigen AND TARGET small_gicp::small_gicp)
  add_subdirectory(backend/registration)  # before the viewer, which links it
endif()
if(CLOUDCROPPER_BUILD_GUI)
  add_subdirectory(src/viewer)
endif()
add_subdirectory(src/app)

# ---- Wire found dependencies into the existing targets --------------------
if(TARGET tl::expected)
  target_link_libraries(cloudcropper_common INTERFACE tl::expected)
  target_compile_definitions(cloudcropper_common INTERFACE CLOUDCROPPER_USE_TL_EXPECTED=1)
  message(STATUS "CloudCropper: cc::Result backed by tl::expected")
else()
  message(STATUS "CloudCropper: cc::Result using std-only stand-in")
endif()
if(TARGET glm::glm)
  target_link_libraries(cloudcropper_common INTERFACE glm::glm)
  target_compile_definitions(cloudcropper_common INTERFACE CLOUDCROPPER_HAS_GLM=1)
endif()
if(TARGET nanoflann::nanoflann)
  target_link_libraries(cloudcropper_core PRIVATE nanoflann::nanoflann)
  target_compile_definitions(cloudcropper_core PRIVATE CLOUDCROPPER_HAS_NANOFLANN=1)
endif()
if(TARGET CLI11::CLI11)
  target_link_libraries(cloudcropper PRIVATE CLI11::CLI11)
  target_compile_definitions(cloudcropper PRIVATE CLOUDCROPPER_HAS_CLI11=1)
endif()
if(TARGET cloudcropper::transport)
  target_link_libraries(cloudcropper PRIVATE cloudcropper::transport)
endif()
if(TARGET cloudcropper::registration)
  target_link_libraries(cloudcropper PRIVATE cloudcropper::registration)
endif()
if(CLOUDCROPPER_BUILD_GUI)
  target_link_libraries(cloudcropper PRIVATE cloudcropper::viewer)
  target_compile_definitions(cloudcropper PRIVATE CLOUDCROPPER_HAS_GUI=1)
endif()

if(CLOUDCROPPER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
