# common is header-only: an INTERFACE library exposing the public include dir.
add_library(cloudcropper_common INTERFACE)
add_library(cloudcropper::common ALIAS cloudcropper_common)

target_include_directories(cloudcropper_common INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(cloudcropper_common INTERFACE cxx_std_20)
