// Persistent Python worker process + the mini JSON codec for its protocol.
//
// PythonWorker spawns `python script` once (lazily, on the first call), keeps
// it alive, and exchanges JSON-lines over stdin/stdout: one UTF-8 JSON object
// per line, one request in flight, responses in request order. The worker's
// stderr is redirected to a log file whose tail is attached to error messages.
// A dead worker (crash, timeout kill, failed spawn) is respawned on the NEXT
// call; the failed call itself returns an error (callers decide on fallback).
//
// Used by the gsdf-gpu backend; the JSON parser is public so the protocol
// pieces are unit-testable. Hand-rolled like the project's readFlatYaml/NPY
// codecs — no third-party JSON dependency, POSIX-only (fork/execvp).
#pragma once

#include <sys/types.h>

#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "cloudcropper/common/result.hpp"

namespace cc::reg {

// ---------------------------------------------------------------------------
// Mini JSON tree. Plain members instead of a variant: the protocol messages
// are tiny, simplicity beats compactness here.
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type                             type    = Type::Null;
    bool                             boolean = false;
    double                           number  = 0.0;
    std::string                      string;
    std::vector<JsonValue>           array;
    std::map<std::string, JsonValue> object;

    bool isNull() const { return type == Type::Null; }
    bool isObject() const { return type == Type::Object; }
    bool isArray() const { return type == Type::Array; }

    // Object member lookup; nullptr when absent or not an object.
    const JsonValue* find(const std::string& key) const;

    // Typed accessors with defaults (numbers also satisfy asBool 0/1 etc.).
    double      asDouble(double dflt = 0.0) const;
    bool        asBool(bool dflt = false) const;
    std::string asString(std::string dflt = {}) const;
};

// Parses exactly one JSON document (trailing whitespace allowed, anything else
// is a ParseError). Handles nesting, all escapes incl. \uXXXX (+ surrogate
// pairs -> UTF-8), exponent floats, true/false/null.
Result<JsonValue> parseJson(std::string_view text);

// "ab\"c" -> ab\"c with all mandatory JSON escapes applied (no quotes added).
std::string jsonEscape(std::string_view s);

// Last `n` non-empty lines of a text file (worker.log tails for error msgs).
std::string lastLines(const std::string& path, int n);

// ---------------------------------------------------------------------------
class PythonWorker {
  public:
    struct Options {
        std::string python = "python3";  // interpreter (argv[0] for execvp)
        std::string script;              // worker script path
        std::string logFile;             // worker stderr destination ("" = /dev/null)
        int         loadingTimeoutSec = 15;   // until {"event":"loading"}
        int         readyTimeoutSec   = 300;  // until {"event":"ready"} (cold torch import)
    };

    explicit PythonWorker(Options opt);
    ~PythonWorker();  // shutdown op -> wait 2s -> SIGTERM -> SIGKILL
    PythonWorker(const PythonWorker&)            = delete;
    PythonWorker& operator=(const PythonWorker&) = delete;

    // Sends {"id":<auto>,"op":"<op>"[,<paramsJsonFragment>]} and waits up to
    // timeoutSec for the response line. paramsJsonFragment is the raw inner
    // text, e.g. R"("device":"cuda","resolution":100)" (may be empty).
    // Returns the parsed response object on {"ok":true}; maps {"ok":false},
    // spawn/handshake failures, crashes and timeouts to errors. Thread-safe
    // (calls are serialized); a request timeout SIGKILLs the worker.
    Result<JsonValue> call(const std::string& op, const std::string& paramsJsonFragment,
                           int timeoutSec);

    bool alive() const { return pid_ > 0; }

  private:
    Result<void> ensureStarted();                       // spawn + handshake
    Result<std::string> readLine(double deadlineSec);   // protocol line w/ timeout
    void killWorker(int sig);                           // signal + reap + close fds
    std::string logTail() const;

    Options     opt_;
    std::mutex  mu_;
    pid_t       pid_   = -1;
    int         inFd_  = -1;  // write end: worker stdin
    int         outFd_ = -1;  // read end:  worker stdout
    std::string rdBuf_;       // partial-line carry between reads
    long        nextId_ = 0;
};

}  // namespace cc::reg
