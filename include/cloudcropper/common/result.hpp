// cc::Result<T> — the project-wide value-or-error type.
//
// Resolved design decision (docs/design/00 §5): the public surface is the
// `tl::expected` shape. When the build provides tl-expected (vcpkg) we alias
// straight to it (CLOUDCROPPER_USE_TL_EXPECTED, set by CMake); otherwise we ship
// a tiny std-only stand-in with the same API. Call sites are identical in both
// modes: `return value;`, `return {};` (void success), `return makeError(...);`.
// Once the toolchain provides std::expected (C++23) this collapses to a one-line
// alias swap. Do NOT add features here that tl/std::expected lack.
#pragma once

#include <string>
#include <utility>

namespace cc {

enum class ErrorCode {
    Ok,
    NotFound,
    Unsupported,
    ParseError,
    IoError,
    InvalidArgument,
    CloudTooLarge,
};

struct Error {
    ErrorCode   code = ErrorCode::Ok;
    std::string message;
};

}  // namespace cc

#if defined(CLOUDCROPPER_USE_TL_EXPECTED)

#include <tl/expected.hpp>

namespace cc {

template <class T>
using Result = tl::expected<T, Error>;

// Returns an unexpected, which implicitly converts to any Result<T>.
inline tl::unexpected<Error> makeError(ErrorCode code, std::string message) {
    return tl::make_unexpected(Error{code, std::move(message)});
}

}  // namespace cc

#else  // ---- std-only stand-in -------------------------------------------------

#include <optional>

namespace cc {

inline Error makeError(ErrorCode code, std::string message) {
    return Error{code, std::move(message)};
}

template <class T>
class Result {
public:
    Result(T value) : value_(std::move(value)) {}      // NOLINT(*-explicit-*)
    Result(Error error) : error_(std::move(error)) {}  // NOLINT(*-explicit-*)

    [[nodiscard]] bool has_value() const { return value_.has_value(); }
    explicit           operator bool() const { return has_value(); }

    T&       value() { return *value_; }
    const T& value() const { return *value_; }
    T&       operator*() { return *value_; }
    const T& operator*() const { return *value_; }
    T*       operator->() { return &*value_; }
    const T* operator->() const { return &*value_; }

    [[nodiscard]] const Error& error() const { return error_; }

private:
    std::optional<T> value_;
    Error            error_;
};

template <>
class Result<void> {
public:
    Result() = default;
    Result(Error error) : ok_(false), error_(std::move(error)) {}  // NOLINT(*-explicit-*)

    [[nodiscard]] bool has_value() const { return ok_; }
    explicit           operator bool() const { return ok_; }

    [[nodiscard]] const Error& error() const { return error_; }

private:
    bool  ok_ = true;
    Error error_;
};

}  // namespace cc

#endif  // CLOUDCROPPER_USE_TL_EXPECTED
