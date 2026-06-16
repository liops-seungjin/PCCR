// The mcap C++ library is header-only and requires MCAP_IMPLEMENTATION to be
// defined in exactly one translation unit — this one. The vendor header is not
// clean under our warning set, so it is isolated here behind pragmas instead of
// relaxing the project flags.
#if defined(CLOUDCROPPER_HAS_MCAP)

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#define MCAP_IMPLEMENTATION
#include <mcap/reader.hpp>

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif  // CLOUDCROPPER_HAS_MCAP
