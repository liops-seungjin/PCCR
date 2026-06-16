add_library(cloudcropper_core STATIC
  attribute.cpp
  point_cloud.cpp
  crop.cpp
  normals.cpp
  denoise.cpp
  analysis.cpp)
add_library(cloudcropper::core ALIAS cloudcropper_core)

target_include_directories(cloudcropper_core
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(cloudcropper_core PUBLIC cxx_std_20)
target_link_libraries(cloudcropper_core
  PUBLIC  cloudcropper::common
  PRIVATE cloudcropper_warnings)
