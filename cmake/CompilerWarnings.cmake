include_guard(GLOBAL)

# Reusable INTERFACE target carrying the project's warning flags.
add_library(cloudcropper_warnings INTERFACE)

if(MSVC)
  target_compile_options(cloudcropper_warnings INTERFACE /W4)
  if(CLOUDCROPPER_WARNINGS_AS_ERRORS)
    target_compile_options(cloudcropper_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(cloudcropper_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
  if(CLOUDCROPPER_WARNINGS_AS_ERRORS)
    target_compile_options(cloudcropper_warnings INTERFACE -Werror)
  endif()
endif()
