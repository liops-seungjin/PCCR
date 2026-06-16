# Zero-dep test runner for the first slice (GoogleTest swaps in via doc 04 later).
add_executable(cc_tests cc_tests.cpp)
target_link_libraries(cc_tests PRIVATE cloudcropper::core cloudcropper::io)
if(TARGET cloudcropper::transport)
  target_link_libraries(cc_tests PRIVATE cloudcropper::transport)  # gzip test + HAS_GZIP define
endif()
if(TARGET cloudcropper::registration)
  target_link_libraries(cc_tests PRIVATE cloudcropper::registration)  # + HAS_REGISTRATION define
endif()
add_test(NAME cc_tests COMMAND cc_tests)
