add_library(cloudcropper_io STATIC
  byte_stream.cpp
  registry.cpp)
add_library(cloudcropper::io ALIAS cloudcropper_io)

target_include_directories(cloudcropper_io
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(cloudcropper_io PUBLIC cxx_std_20)
target_link_libraries(cloudcropper_io
  PUBLIC  cloudcropper::core
  PRIVATE cloudcropper_warnings)

# Format plug-ins compiled & advertised conditionally (matches doc 04).
if(CLOUDCROPPER_WITH_PLY)
  target_sources(cloudcropper_io PRIVATE ply.cpp)
  target_compile_definitions(cloudcropper_io PUBLIC CLOUDCROPPER_HAS_PLY=1)
endif()

if(CLOUDCROPPER_WITH_PCD)
  target_sources(cloudcropper_io PRIVATE pcd.cpp)
  target_compile_definitions(cloudcropper_io PUBLIC CLOUDCROPPER_HAS_PCD=1)
  # binary_compressed needs LZF; available only when liblzf was found (vcpkg).
  if(TARGET unofficial::liblzf::liblzf)
    target_link_libraries(cloudcropper_io PRIVATE unofficial::liblzf::liblzf)
    target_compile_definitions(cloudcropper_io PUBLIC CLOUDCROPPER_HAS_LZF=1)
  endif()
endif()

# NPZ requires miniz for the ZIP container (vcpkg npz feature).
if(TARGET miniz::miniz)
  target_sources(cloudcropper_io PRIVATE npz.cpp)
  target_link_libraries(cloudcropper_io PRIVATE miniz::miniz)
  target_compile_definitions(cloudcropper_io PUBLIC CLOUDCROPPER_HAS_NPZ=1)
endif()

# ROS2 bag (.db3) reader needs sqlite3 for the rosbag2 container (vcpkg rosbag feature).
if(TARGET unofficial::sqlite3::sqlite3)
  target_sources(cloudcropper_io PRIVATE rosbag.cpp)
  target_link_libraries(cloudcropper_io PRIVATE unofficial::sqlite3::sqlite3)
  target_compile_definitions(cloudcropper_io PUBLIC CLOUDCROPPER_HAS_ROSBAG=1)

  # .mcap container support (same rosbag feature). The mcap C++ reader is
  # header-only; MCAP_IMPLEMENTATION lives in its own TU (mcap_impl.cpp) where
  # the vendor header's warnings are silenced without relaxing our own flags.
  if(TARGET unofficial::mcap::mcap)
    target_sources(cloudcropper_io PRIVATE mcap_impl.cpp)
    target_link_libraries(cloudcropper_io PRIVATE unofficial::mcap::mcap)
    target_compile_definitions(cloudcropper_io PUBLIC CLOUDCROPPER_HAS_MCAP=1)
  endif()
endif()
