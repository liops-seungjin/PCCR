#include "cloudcropper/registration/python_worker.hpp"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace cc::reg {

// ============================================================================
// Mini JSON parser (recursive descent, depth-limited).
// ============================================================================
namespace {

struct JsonParser {
    std::string_view s;
    std::size_t      i = 0;
    static constexpr int kMaxDepth = 64;

    bool atEnd() const { return i >= s.size(); }
    char peek() const { return s[i]; }

    void skipWs() {
        while (!atEnd() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
            ++i;
    }

    Result<JsonValue> fail(const std::string& what) {
        return makeError(ErrorCode::ParseError,
                         "json: " + what + " at offset " + std::to_string(i));
    }

    bool consume(std::string_view lit) {
        if (s.size() - i < lit.size() || s.substr(i, lit.size()) != lit) return false;
        i += lit.size();
        return true;
    }

    static void appendUtf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    bool hex4(unsigned& out) {
        if (s.size() - i < 4) return false;
        out = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[i + static_cast<std::size_t>(k)];
            out <<= 4;
            if (c >= '0' && c <= '9') out |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') out |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') out |= static_cast<unsigned>(c - 'A' + 10);
            else return false;
        }
        i += 4;
        return true;
    }

    Result<std::string> parseString() {
        ++i;  // opening quote (caller checked)
        std::string out;
        while (true) {
            if (atEnd()) return makeError(ErrorCode::ParseError, "json: unterminated string");
            const char c = s[i++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20)
                return makeError(ErrorCode::ParseError, "json: raw control char in string");
            if (c != '\\') {
                out += c;
                continue;
            }
            if (atEnd()) return makeError(ErrorCode::ParseError, "json: dangling escape");
            const char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = 0;
                    if (!hex4(cp)) return makeError(ErrorCode::ParseError, "json: bad \\u escape");
                    if (cp >= 0xD800 && cp <= 0xDBFF) {  // high surrogate
                        unsigned lo = 0;
                        if (consume("\\u") && hex4(lo) && lo >= 0xDC00 && lo <= 0xDFFF)
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        else
                            return makeError(ErrorCode::ParseError, "json: lone surrogate");
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default: return makeError(ErrorCode::ParseError, "json: unknown escape");
            }
        }
    }

    Result<JsonValue> parseValue(int depth) {
        if (depth > kMaxDepth) return fail("nesting too deep");
        skipWs();
        if (atEnd()) return fail("unexpected end of input");
        JsonValue v;
        const char c = peek();
        if (c == '{') {
            ++i;
            v.type = JsonValue::Type::Object;
            skipWs();
            if (!atEnd() && peek() == '}') { ++i; return v; }
            while (true) {
                skipWs();
                if (atEnd() || peek() != '"') return fail("expected object key");
                auto key = parseString();
                if (!key) return makeError(key.error().code, key.error().message);
                skipWs();
                if (atEnd() || s[i] != ':') return fail("expected ':'");
                ++i;
                auto val = parseValue(depth + 1);
                if (!val) return val;
                v.object[*key] = std::move(*val);
                skipWs();
                if (atEnd()) return fail("unterminated object");
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == '}') { ++i; return v; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++i;
            v.type = JsonValue::Type::Array;
            skipWs();
            if (!atEnd() && peek() == ']') { ++i; return v; }
            while (true) {
                auto el = parseValue(depth + 1);
                if (!el) return el;
                v.array.push_back(std::move(*el));
                skipWs();
                if (atEnd()) return fail("unterminated array");
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == ']') { ++i; return v; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            auto str = parseString();
            if (!str) return makeError(str.error().code, str.error().message);
            v.type   = JsonValue::Type::String;
            v.string = std::move(*str);
            return v;
        }
        if (consume("true")) { v.type = JsonValue::Type::Bool; v.boolean = true; return v; }
        if (consume("false")) { v.type = JsonValue::Type::Bool; v.boolean = false; return v; }
        if (consume("null")) { v.type = JsonValue::Type::Null; return v; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            // strtod accepts a superset (hex, inf): restrict to JSON number chars.
            const std::size_t start = i;
            if (s[i] == '-') ++i;
            while (!atEnd() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' ||
                                s[i] == 'E' || s[i] == '+' || s[i] == '-'))
                ++i;
            const std::string num(s.substr(start, i - start));
            char*             end = nullptr;
            errno             = 0;
            const double d = std::strtod(num.c_str(), &end);
            if (end != num.c_str() + num.size() || num == "-") {
                i = start;
                return fail("malformed number");
            }
            v.type   = JsonValue::Type::Number;
            v.number = d;
            return v;
        }
        return fail("unexpected character");
    }
};

}  // namespace

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

double JsonValue::asDouble(double dflt) const {
    if (type == Type::Number) return number;
    if (type == Type::Bool) return boolean ? 1.0 : 0.0;
    return dflt;
}

bool JsonValue::asBool(bool dflt) const {
    if (type == Type::Bool) return boolean;
    if (type == Type::Number) return number != 0.0;
    return dflt;
}

std::string JsonValue::asString(std::string dflt) const {
    return type == Type::String ? string : dflt;
}

Result<JsonValue> parseJson(std::string_view text) {
    JsonParser p{text};
    auto       v = p.parseValue(0);
    if (!v) return v;
    p.skipWs();
    if (!p.atEnd())
        return makeError(ErrorCode::ParseError,
                         "json: trailing garbage at offset " + std::to_string(p.i));
    return v;
}

std::string jsonEscape(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string lastLines(const std::string& path, int n) {
    std::ifstream            in(path);
    std::vector<std::string> lines;
    std::string              l;
    while (std::getline(in, l))
        if (!l.empty()) lines.push_back(l);
    std::string out;
    for (std::size_t i = lines.size() > static_cast<std::size_t>(n)
                             ? lines.size() - static_cast<std::size_t>(n)
                             : 0;
         i < lines.size(); ++i)
        out += lines[i] + "\n";
    return out;
}

// ============================================================================
// PythonWorker
// ============================================================================
namespace {

double nowSec() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

PythonWorker::PythonWorker(Options opt) : opt_(std::move(opt)) {}

std::string PythonWorker::logTail() const {
    const std::string tail = opt_.logFile.empty() ? "" : lastLines(opt_.logFile, 6);
    return tail.empty() ? "(worker log empty)" : tail;
}

void PythonWorker::killWorker(int sig) {
    if (pid_ > 0) {
        ::kill(pid_, sig);
        int status = 0;
        ::waitpid(pid_, &status, 0);
    }
    if (inFd_ >= 0) ::close(inFd_);
    if (outFd_ >= 0) ::close(outFd_);
    pid_   = -1;
    inFd_  = -1;
    outFd_ = -1;
    rdBuf_.clear();
}

Result<std::string> PythonWorker::readLine(double deadlineSec) {
    while (true) {
        const auto nl = rdBuf_.find('\n');
        if (nl != std::string::npos) {
            std::string line = rdBuf_.substr(0, nl);
            rdBuf_.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return line;
        }
        const double remain = deadlineSec - nowSec();
        if (remain <= 0)
            return makeError(ErrorCode::IoError, "python worker: response timed out");
        struct pollfd pfd { outFd_, POLLIN, 0 };
        const int     pr = ::poll(&pfd, 1, static_cast<int>(remain * 1000) + 1);
        if (pr < 0) {
            if (errno == EINTR) continue;
            return makeError(ErrorCode::IoError,
                             std::string("python worker: poll failed: ") + std::strerror(errno));
        }
        if (pr == 0)
            return makeError(ErrorCode::IoError, "python worker: response timed out");
        char          buf[4096];
        const ssize_t n = ::read(outFd_, buf, sizeof buf);
        if (n < 0) {
            if (errno == EINTR) continue;
            return makeError(ErrorCode::IoError,
                             std::string("python worker: read failed: ") + std::strerror(errno));
        }
        if (n == 0)
            return makeError(ErrorCode::IoError, "python worker: process closed its stdout");
        rdBuf_.append(buf, static_cast<std::size_t>(n));
    }
}

Result<void> PythonWorker::ensureStarted() {
    if (pid_ > 0) return {};

    static const bool sigpipeIgnored = [] {
        ::signal(SIGPIPE, SIG_IGN);  // a dead worker must not kill the app
        return true;
    }();
    (void)sigpipeIgnored;

    int inPipe[2], outPipe[2];  // in: parent->child stdin, out: child stdout->parent
    if (::pipe(inPipe) != 0)
        return makeError(ErrorCode::IoError, "python worker: pipe() failed");
    if (::pipe(outPipe) != 0) {
        ::close(inPipe[0]);
        ::close(inPipe[1]);
        return makeError(ErrorCode::IoError, "python worker: pipe() failed");
    }

    // Everything the child needs, prepared BEFORE fork: the child path in a
    // multithreaded process may only use async-signal-safe calls.
    const std::string& py  = opt_.python;
    const std::string& scr = opt_.script;
    const char* argv[3] = {py.c_str(), scr.c_str(), nullptr};
    const char* logPath = opt_.logFile.empty() ? "/dev/null" : opt_.logFile.c_str();

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(inPipe[0]); ::close(inPipe[1]);
        ::close(outPipe[0]); ::close(outPipe[1]);
        return makeError(ErrorCode::IoError, "python worker: fork() failed");
    }
    if (pid == 0) {  // child — async-signal-safe calls only
        ::dup2(inPipe[0], 0);
        ::dup2(outPipe[1], 1);
        const int logFd = ::open(logPath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (logFd >= 0) ::dup2(logFd, 2);
        ::close(inPipe[0]); ::close(inPipe[1]);
        ::close(outPipe[0]); ::close(outPipe[1]);
        if (logFd > 2) ::close(logFd);
        ::execvp(argv[0], const_cast<char* const*>(argv));
        _exit(127);
    }
    ::close(inPipe[0]);
    ::close(outPipe[1]);
    pid_   = pid;
    inFd_  = inPipe[1];
    outFd_ = outPipe[0];
    rdBuf_.clear();

    // Handshake: "loading" fast (proves exec + protocol bootstrap), then
    // "ready" slow (cold torch import) or "fatal" (import error).
    auto expectEvent = [&](const char* name, double deadline) -> Result<void> {
        auto line = readLine(deadline);
        if (!line) {
            const std::string why = line.error().message;
            killWorker(SIGKILL);
            return makeError(ErrorCode::NotFound, "python worker: no '" + std::string(name) +
                                                      "' handshake (" + why + ")\n" + logTail());
        }
        auto msg = parseJson(*line);
        if (!msg || !msg->isObject()) {
            killWorker(SIGKILL);
            return makeError(ErrorCode::ParseError,
                             "python worker: bad handshake line: " + *line);
        }
        const JsonValue* ev = msg->find("event");
        if (ev && ev->asString() == name) return {};
        if (ev && ev->asString() == "fatal") {
            const JsonValue* err  = msg->find("error");
            const std::string what =
                err && err->find("message") ? err->find("message")->asString() : "?";
            const std::string type =
                err && err->find("type") ? err->find("type")->asString() : "?";
            killWorker(SIGKILL);
            return makeError(ErrorCode::IoError,
                             "python worker: startup failed: " + type + ": " + what);
        }
        killWorker(SIGKILL);
        return makeError(ErrorCode::ParseError,
                         "python worker: unexpected handshake: " + *line);
    };
    if (auto r = expectEvent("loading", nowSec() + opt_.loadingTimeoutSec); !r) return r;
    if (auto r = expectEvent("ready", nowSec() + opt_.readyTimeoutSec); !r) return r;
    return {};
}

Result<JsonValue> PythonWorker::call(const std::string& op,
                                     const std::string& paramsJsonFragment, int timeoutSec) {
    std::lock_guard<std::mutex> lock(mu_);

    if (auto s = ensureStarted(); !s) return makeError(s.error().code, s.error().message);

    const long  id  = ++nextId_;
    std::string req = "{\"id\":" + std::to_string(id) + ",\"op\":\"" + jsonEscape(op) + "\"";
    if (!paramsJsonFragment.empty()) req += "," + paramsJsonFragment;
    req += "}\n";

    std::size_t off = 0;
    while (off < req.size()) {
        const ssize_t n = ::write(inFd_, req.data() + off, req.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            const std::string why = std::strerror(errno);
            killWorker(SIGKILL);
            return makeError(ErrorCode::IoError,
                             "python worker: died while writing request (" + why + ")\n" +
                                 logTail());
        }
        off += static_cast<std::size_t>(n);
    }

    auto line = readLine(nowSec() + timeoutSec);
    if (!line) {
        // Timeout or EOF: the only safe move for a hung/dead worker is kill +
        // reap; it respawns on the next call.
        const std::string why = line.error().message;
        killWorker(SIGKILL);
        return makeError(ErrorCode::IoError, "python worker ('" + op + "'): " + why + "\n" +
                                                 logTail());
    }
    auto msg = parseJson(*line);
    if (!msg)
        return makeError(ErrorCode::ParseError,
                         "python worker: unparseable response: " + msg.error().message);
    const JsonValue* ok = msg->find("ok");
    if (!msg->isObject() || !ok)
        return makeError(ErrorCode::ParseError, "python worker: malformed response: " + *line);
    if (const JsonValue* rid = msg->find("id");
        rid && static_cast<long>(rid->asDouble(-1)) != id)
        return makeError(ErrorCode::ParseError, "python worker: response id mismatch");
    if (!ok->asBool()) {
        const JsonValue*  err  = msg->find("error");
        const std::string type = err && err->find("type") ? err->find("type")->asString() : "?";
        const std::string what =
            err && err->find("message") ? err->find("message")->asString() : "?";
        return makeError(ErrorCode::IoError,
                         "python worker ('" + op + "'): " + type + ": " + what);
    }
    return msg;
}

PythonWorker::~PythonWorker() {
    std::lock_guard<std::mutex> lock(mu_);
    if (pid_ <= 0) return;

    // Polite first: shutdown op + closed stdin both ask the loop to exit.
    const std::string bye = "{\"id\":" + std::to_string(++nextId_) + ",\"op\":\"shutdown\"}\n";
    (void)!::write(inFd_, bye.data(), bye.size());
    ::close(inFd_);
    inFd_ = -1;

    const double deadline = nowSec() + 2.0;
    while (nowSec() < deadline) {
        int status = 0;
        if (::waitpid(pid_, &status, WNOHANG) == pid_) {
            pid_ = -1;
            break;
        }
        ::usleep(50 * 1000);
    }
    if (pid_ > 0) {
        ::kill(pid_, SIGTERM);
        ::usleep(200 * 1000);
        int status = 0;
        if (::waitpid(pid_, &status, WNOHANG) != pid_) {
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, &status, 0);
        }
        pid_ = -1;
    }
    if (outFd_ >= 0) ::close(outFd_);
    outFd_ = -1;
}

}  // namespace cc::reg
