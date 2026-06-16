// Minimal data-parallel helper: split [0, n) into contiguous ranges across the
// hardware threads and run `fn(begin, end)` on each, joining before return. Each
// range owns its own scratch (declare locals inside `fn`), so the hot loops in
// crop / normals / denoise parallelise with disjoint output writes and no locks.
#pragma once

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace cc {

template <class Fn>
void parallelForRange(std::size_t n, Fn fn) {
    if (n == 0) return;
    const unsigned    hw    = std::max(1u, std::thread::hardware_concurrency());
    // Stay serial for small n (thread setup would dominate).
    const std::size_t want  = std::max<std::size_t>(1, n / 4096);
    const std::size_t nthr  = std::min<std::size_t>(hw, want);
    if (nthr <= 1) {
        fn(std::size_t{0}, n);
        return;
    }
    const std::size_t       chunk = (n + nthr - 1) / nthr;
    std::vector<std::thread> pool;
    pool.reserve(nthr);
    for (std::size_t t = 0; t < nthr; ++t) {
        const std::size_t b = t * chunk;
        const std::size_t e = std::min(n, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&fn, b, e]() { fn(b, e); });
    }
    for (std::thread& th : pool) th.join();
}

}  // namespace cc
