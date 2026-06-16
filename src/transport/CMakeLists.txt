add_library(cloudcropper_transport STATIC gzip.cpp)
add_library(cloudcropper::transport ALIAS cloudcropper_transport)

target_include_directories(cloudcropper_transport
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(cloudcropper_transport PUBLIC cxx_std_20)
target_link_libraries(cloudcropper_transport
  PUBLIC  cloudcropper::common
  PRIVATE ZLIB::ZLIB cloudcropper_warnings)
# Public so anything linking transport (app, tests) can #ifdef on it.
target_compile_definitions(cloudcropper_transport PUBLIC CLOUDCROPPER_HAS_GZIP=1)
