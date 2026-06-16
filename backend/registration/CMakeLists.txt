# Registration backends, one package per subdirectory (common/, gicp/,
# kiss_matcher/, gradient_sdf_gpu/, bufferx/), compiled into a single static library
# behind the dependency-free public header include/cloudcropper/registration/.
add_library(cloudcropper_registration STATIC
  common/registration.cpp
  common/reg_common.cpp
  common/config.cpp
  common/python_worker.cpp
  gicp/gicp_backend.cpp
  gradient_sdf_gpu/gsdf_gpu.cpp
  bufferx/bufferx_backend.cpp)
add_library(cloudcropper::registration ALIAS cloudcropper_registration)

target_include_directories(cloudcropper_registration
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_SOURCE_DIR}/src/core)  # core-internal GridIndex for the shared metric
target_compile_features(cloudcropper_registration PUBLIC cxx_std_20)
target_compile_definitions(cloudcropper_registration PUBLIC CLOUDCROPPER_HAS_REGISTRATION=1)
target_link_libraries(cloudcropper_registration
  PUBLIC  cloudcropper::core
  PRIVATE cloudcropper_warnings
          cloudcropper::io   # NPZ handoff files for the gsdf-gpu bridge
          Eigen3::Eigen
          small_gicp::small_gicp)

# KISS-Matcher package: fetched from GitHub (no vcpkg port). Optional — gated
# so offline configures of the other backends still work.
if(CLOUDCROPPER_WITH_KISS_MATCHER)
  include(${CMAKE_CURRENT_SOURCE_DIR}/kiss_matcher/fetch_kiss_matcher.cmake)
endif()
