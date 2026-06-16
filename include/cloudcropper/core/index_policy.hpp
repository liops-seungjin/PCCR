// IndexPolicy — resolved decision (docs/design/03 §3.5): the octree build
// decision is an explicit, tunable policy, never a hard-coded constant.
#pragma once

#include <cstdint>

namespace cc {

struct IndexPolicy {
    enum class Mode { Never, Always, Auto };

    Mode          mode                 = Mode::Auto;
    std::uint64_t auto_point_threshold = 200'000;  // PROVISIONAL — calibrate via tests/bench
};

// Two-factor heuristic: under Auto, build an index iff
//   N > threshold AND (interactive || expectedQueries > 1).
inline bool shouldBuildIndex(const IndexPolicy& p, std::uint64_t n, bool interactive,
                             unsigned expectedQueries) {
    switch (p.mode) {
        case IndexPolicy::Mode::Never:
            return false;
        case IndexPolicy::Mode::Always:
            return true;
        case IndexPolicy::Mode::Auto:
            return n > p.auto_point_threshold && (interactive || expectedQueries > 1);
    }
    return false;
}

}  // namespace cc
