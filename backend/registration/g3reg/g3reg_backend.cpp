#include "g3reg_backend.hpp"

#include <cstdlib>
#include <map>
#include <string>

#include "cloudcropper/registration/config.hpp"

#if defined(CLOUDCROPPER_HAS_PCD)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <thread>
#include <vector>

#include "../gicp/gicp_backend.hpp"
#include "cloudcropper/io/byte_stream.hpp"
#include "cloudcropper/io/pcd.hpp"
#include "cloudcropper/registration/python_worker.hpp"  // lastLines() for the error tail
#endif

namespace cc::reg::g3reg {

#if !defined(CLOUDCROPPER_HAS_PCD)

Result<RegResult> run(const PointCloud&, const PointCloud&, const RegOptions&) {
    return makeError(ErrorCode::Unsupported,
                     "g3reg: needs the PCD codec (vcpkg pcd feature) for the handoff");
}

#else

namespace fs = std::filesystem;

namespace {

double nowSec() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// gsdf_gpu.cpp:44-58 findScript() + config.cpp:40-56 findConfig() patterns:
// env override (must exist), then the yaml `bin` key, then a few executable-
// relative locations. Empty = not found -> NotFound at the call site.
fs::path findBin(const std::map<std::string, std::string>& cfg) {
    std::error_code ec;
    if (const char* e = std::getenv("CLOUDCROPPER_G3REG_BIN"))
        if (fs::exists(e, ec)) return e;
    if (const auto it = cfg.find("bin");
        it != cfg.end() && !it->second.empty() && fs::exists(it->second, ec))
        return it->second;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        fs::path d = exe.parent_path();
        for (int up = 0; up < 6 && !d.empty(); ++up, d = d.parent_path())
            for (const char* rel :
                 {"cc_g3reg_cli", "g3reg/bin/cc_g3reg_cli", "bin/cc_g3reg_cli"})
                if (fs::exists(d / rel, ec)) return d / rel;
    }
    return {};
}

// Path to the EXTERNAL G3Reg config yaml (cc_g3reg_cli's first argument). We do
// not parse it — its schema is PCL/GTSAM-side; we only forward the path. May be
// empty (the external CLI is then expected to use its own default config).
std::string g3regConfig(const std::map<std::string, std::string>& cfg) {
    if (const char* e = std::getenv("CLOUDCROPPER_G3REG_CONFIG")) return e;
    if (const auto it = cfg.find("g3reg_config"); it != cfg.end() && !it->second.empty())
        return it->second;
    return {};
}

// xyz-only PCD handoff (G3Reg consumes PointXYZ). A field filter that matches no
// attribute leaves only the always-emitted x/y/z (pcd.cpp:349-364). Binary
// encoding avoids ascii rounding and is faster for big clouds; units stay meters.
Result<void> writePcd(const PointCloud& pc, const fs::path& path) {
    io::FileByteSink sink(path.string());
    if (!sink.ok())
        return makeError(ErrorCode::IoError, "g3reg: cannot create " + path.string());
    io::WriteOptions opt;
    opt.fields   = {"__cc_xyz_only__"};  // matches nothing -> x/y/z only
    opt.encoding = io::Encoding::Binary;
    return io::PcdWriter{}.write(pc, sink, opt);
}

struct CliResult {
    int         exitCode = -1;
    std::string stdoutText;
    bool        timedOut = false;
};

// One-shot subprocess: argv[0]=bin, rest=args. stdout is captured through a
// pipe; the child's stderr (glog noise) goes to logPath so it never pollutes the
// 3-line protocol. Exceeding timeoutSec SIGKILLs the child. Always reaps.
Result<CliResult> runCli(const std::vector<std::string>& args, const std::string& logPath,
                         int timeoutSec) {
    int outPipe[2];
    if (::pipe(outPipe) != 0) return makeError(ErrorCode::IoError, "g3reg: pipe() failed");

    std::vector<const char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& a : args) argv.push_back(a.c_str());
    argv.push_back(nullptr);

    static const bool ign = [] {
        ::signal(SIGPIPE, SIG_IGN);
        return true;
    }();
    (void)ign;

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(outPipe[0]);
        ::close(outPipe[1]);
        return makeError(ErrorCode::IoError, "g3reg: fork() failed");
    }
    if (pid == 0) {  // child: async-signal-safe calls only
        ::close(outPipe[0]);
        ::dup2(outPipe[1], 1);  // stdout -> pipe
        const int logFd = ::open(logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (logFd >= 0) ::dup2(logFd, 2);  // stderr (glog) -> log file
        ::close(outPipe[1]);
        if (logFd > 2) ::close(logFd);
        const int devnull = ::open("/dev/null", O_RDONLY);
        if (devnull >= 0) ::dup2(devnull, 0);
        ::execvp(argv[0], const_cast<char* const*>(argv.data()));
        _exit(127);  // exec failed
    }
    ::close(outPipe[1]);
    const int outFd = outPipe[0];

    std::string  out;
    CliResult    cr;
    const double deadline = nowSec() + timeoutSec;
    for (;;) {
        const double remain = deadline - nowSec();
        if (remain <= 0) {
            ::kill(pid, SIGKILL);
            cr.timedOut = true;
            break;
        }
        struct pollfd pfd { outFd, POLLIN, 0 };
        const int     pr = ::poll(&pfd, 1, static_cast<int>(remain * 1000) + 1);
        if (pr < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (pr == 0) {
            ::kill(pid, SIGKILL);
            cr.timedOut = true;
            break;
        }
        char          buf[4096];
        const ssize_t n = ::read(outFd, buf, sizeof buf);
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (n == 0) break;  // EOF: child closed stdout
        out.append(buf, static_cast<std::size_t>(n));
    }
    ::close(outFd);
    int status = 0;
    // Bounded reap: EOF on stdout usually means the child exited, but a child
    // that closed stdout yet hangs (or a grandchild holding the write end) must
    // not let waitpid block past the deadline — that is the whole safety net.
    for (;;) {
        const pid_t w = ::waitpid(pid, &status, WNOHANG);
        if (w == pid || w < 0) break;  // reaped, or already gone
        if (cr.timedOut || nowSec() >= deadline) {
            ::kill(pid, SIGKILL);
            cr.timedOut = true;
            ::waitpid(pid, &status, 0);  // a SIGKILLed child reaps immediately
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));  // avoid busy-spin
    }
    cr.exitCode   = (!cr.timedOut && WIFEXITED(status)) ? WEXITSTATUS(status) : -1;
    cr.stdoutText = std::move(out);
    return cr;
}

// Parse the 3 protocol lines into a RegResult. transform is row-major
// target<-source, so it maps straight onto RegResult::transform (same convention
// as registration.hpp). inliers / solver-time go into `detail` only; the
// dispatcher recomputes rmse/inliers/seconds via the shared metric.
Result<RegResult> parseG3reg(const CliResult& cr, const std::string& logTail) {
    if (cr.timedOut)
        return makeError(ErrorCode::IoError, "g3reg: cc_g3reg_cli timed out\n" + logTail);
    if (cr.exitCode != 0)
        return makeError(ErrorCode::IoError,
                         "g3reg: cc_g3reg_cli exited " + std::to_string(cr.exitCode) + "\n" +
                             logTail);

    std::array<double, 16> tf      = kIdentity4;
    int                    tfCount = 0;
    long                   inliers = -1;
    double                 secs    = -1.0;
    std::istringstream     ss(cr.stdoutText);
    std::string            line;
    while (std::getline(ss, line)) {
        if (line.rfind("G3REG_TF:", 0) == 0) {
            std::istringstream ns(line.substr(9));
            double             v = 0.0;
            tfCount              = 0;
            while (tfCount < 16 && (ns >> v)) tf[static_cast<std::size_t>(tfCount++)] = v;
        } else if (line.rfind("G3REG_INLIERS:", 0) == 0) {
            inliers = std::strtol(line.substr(14).c_str(), nullptr, 10);
        } else if (line.rfind("G3REG_TIME:", 0) == 0) {
            secs = std::strtod(line.substr(11).c_str(), nullptr);
        }
    }
    if (tfCount != 16)
        return makeError(ErrorCode::ParseError,
                         "g3reg: no valid G3REG_TF (16 floats) in CLI stdout\n" + logTail);

    RegResult out;
    out.transform = tf;    // row-major, target<-source — verbatim
    out.converged = true;  // exit 0 + a full TF = converged
    // confidence / normResidual stay -1 (G3Reg does not provide them).
    std::ostringstream d;
    d << "G3Reg: " << (inliers >= 0 ? std::to_string(inliers) : std::string("?"))
      << " inliers";
    if (secs >= 0.0) {
        d.precision(3);
        d << ", " << secs << "s (solver)";
    }
    out.detail = d.str();  // e.g. "G3Reg: 1234 inliers, 0.42s (solver)"
    return out;
}

// Per-call temp dir; destroyed (with the handoff .pcd + worker.log) on return.
struct TempDirGuard {
    fs::path path;
    ~TempDirGuard() {
        std::error_code ec;
        if (!path.empty()) fs::remove_all(path, ec);
    }
};

}  // namespace

Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt) {
    const auto cfg = configValues("g3reg.yaml");

    const fs::path bin = findBin(cfg);
    if (bin.empty())
        return makeError(ErrorCode::NotFound,
                         "g3reg: cc_g3reg_cli not found (set CLOUDCROPPER_G3REG_BIN or "
                         "the `bin` key in config/g3reg.yaml)");

    int timeoutSec = 600;
    if (const auto it = cfg.find("timeout_sec"); it != cfg.end()) {
        try {
            timeoutSec = std::stoi(it->second);
        } catch (...) {}
    }

    // Unique per call (pid + atomic counter): the library API does not serialize
    // concurrent registerClouds, so a pid-only dir would race on the handoff .pcd
    // and one TempDirGuard could delete it out from under another call's CLI.
    static std::atomic<unsigned> callSeq{0};
    TempDirGuard tmp{fs::temp_directory_path() /
                     ("cc_g3reg_" + std::to_string(static_cast<long>(::getpid())) + "_" +
                      std::to_string(callSeq.fetch_add(1)))};
    std::error_code ec;
    fs::create_directories(tmp.path, ec);
    const fs::path srcPcd  = tmp.path / "source.pcd";
    const fs::path tgtPcd  = tmp.path / "target.pcd";
    const fs::path logPath = tmp.path / "worker.log";
    if (auto w = writePcd(source, srcPcd); !w)
        return makeError(w.error().code, w.error().message);
    if (auto w = writePcd(target, tgtPcd); !w)
        return makeError(w.error().code, w.error().message);

    const std::vector<std::string> args = {bin.string(), g3regConfig(cfg), srcPcd.string(),
                                           tgtPcd.string()};
    auto                           cli  = runCli(args, logPath.string(), timeoutSec);
    if (!cli) return makeError(cli.error().code, cli.error().message);

    const std::string logTail = lastLines(logPath.string(), 6);
    auto              coarse  = parseG3reg(*cli, logTail);
    if (!coarse) return coarse;

    // G3RegGicp: chain GICP seeded by the global transform (kiss_backend.cpp:61-70).
    if (opt.algo == RegAlgo::G3RegGicp && opt.refine) {
        RegOptions ro = opt;
        ro.algo       = RegAlgo::Gicp;
        ro.init       = coarse->transform;
        auto refined  = gicp::run(source, target, ro);
        if (refined) {
            refined->detail = coarse->detail + "  ->  " + refined->detail;
            return refined;
        }
        // refine failed: the global result is still valid -> return coarse.
    }
    return coarse;
}

#endif  // CLOUDCROPPER_HAS_PCD

}  // namespace cc::reg::g3reg
