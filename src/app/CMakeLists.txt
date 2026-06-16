add_executable(cloudcropper main.cpp)
target_link_libraries(cloudcropper
  PRIVATE cloudcropper::core cloudcropper::io cloudcropper_warnings)
