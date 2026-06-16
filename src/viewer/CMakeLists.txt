add_library(cloudcropper_viewer STATIC
  gl_util.cpp
  point_renderer.cpp
  box_renderer.cpp
  viewer.cpp)
add_library(cloudcropper::viewer ALIAS cloudcropper_viewer)

target_include_directories(cloudcropper_viewer
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(cloudcropper_viewer PUBLIC cxx_std_20)

target_link_libraries(cloudcropper_viewer
  PUBLIC  cloudcropper::core
  PRIVATE cloudcropper::io
          glfw
          glad::glad
          imgui::imgui
          imguizmo::imguizmo
          glm::glm
          OpenGL::GL
          cloudcropper_warnings)

# Registration panel (right side) — compiled in when the backends are built.
if(TARGET cloudcropper::registration)
  target_link_libraries(cloudcropper_viewer PRIVATE cloudcropper::registration)
endif()
