# Fetches KISS-Matcher (MIT-SPARK, MIT license) from GitHub and adds the
# kiss_matcher package to cloudcropper_registration. Eigen/TBB come from vcpkg
# (USE_SYSTEM_*=ON); ROBIN is fetched by KISS-Matcher's own CMake. Needs network
# on the first configure — same constraint as the vcpkg install itself.
include(FetchContent)

set(USE_SYSTEM_EIGEN3 ON  CACHE BOOL "" FORCE)
set(USE_SYSTEM_TBB    ON  CACHE BOOL "" FORCE)
set(USE_SYSTEM_ROBIN  OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

FetchContent_Declare(kiss_matcher
  GIT_REPOSITORY https://github.com/MIT-SPARK/KISS-Matcher.git
  GIT_TAG        v1.0.2
  SOURCE_SUBDIR  cpp/kiss_matcher)
FetchContent_MakeAvailable(kiss_matcher)

# Upstream v1.0.2 forgets `#include <cassert>` in GncSolver.cpp (breaks on
# gcc 11's libstdc++ where nothing pulls it in transitively). Force-include it.
if(TARGET kiss_matcher_core AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
  target_compile_options(kiss_matcher_core PRIVATE -include cassert)
endif()

target_sources(cloudcropper_registration PRIVATE kiss_matcher/kiss_backend.cpp)
target_link_libraries(cloudcropper_registration PRIVATE kiss_matcher::kiss_matcher_core)
target_compile_definitions(cloudcropper_registration PUBLIC CLOUDCROPPER_HAS_KISS_MATCHER=1)
# Their headers are consumed as system includes inside kiss_backend.cpp pragmas.
get_target_property(_km_inc kiss_matcher::kiss_matcher_core INTERFACE_INCLUDE_DIRECTORIES)
if(_km_inc)
  target_include_directories(cloudcropper_registration SYSTEM PRIVATE ${_km_inc})
endif()
