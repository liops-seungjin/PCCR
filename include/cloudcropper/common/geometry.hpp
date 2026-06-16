// Minimal self-contained 3D math: Vec3, Quat, Aabb, Obb.
//
// Kept dependency-free for the first slice; doc 04 swaps in glm once vcpkg is
// wired, but the Aabb/Obb membership semantics here are the authority shared by
// the crop engine and (later) the viewer (docs/design/02 obb_math).
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace cc {

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

inline Vec3  operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3  operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3  operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3  cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 cmin(Vec3 a, Vec3 b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}
inline Vec3 cmax(Vec3 a, Vec3 b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

// Unit quaternion (x,y,z,w); identity maps local axes onto world axes.
struct Quat {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

inline Quat conjugate(Quat q) { return {-q.x, -q.y, -q.z, q.w}; }

// Rotate v by q (assumes q is unit). Standard optimized form.
inline Vec3 rotate(Quat q, Vec3 v) {
    const Vec3 u{q.x, q.y, q.z};
    const Vec3 t = cross(u, v) * 2.0f;
    return v + t * q.w + cross(u, t);
}

struct Aabb {
    Vec3 min{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
             std::numeric_limits<float>::infinity()};
    Vec3 max{-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
             -std::numeric_limits<float>::infinity()};

    [[nodiscard]] bool valid() const { return min.x <= max.x; }
    void               expand(Vec3 p) {
        min = cmin(min, p);
        max = cmax(max, p);
    }
    [[nodiscard]] bool contains(Vec3 p) const {
        return p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y && p.z >= min.z &&
               p.z <= max.z;
    }
};

// Oriented bounding box. AABB is the special case rotation == identity.
struct Obb {
    Vec3 center{};
    Vec3 halfExtents{};
    Quat rotation{};

    [[nodiscard]] bool contains(Vec3 p) const {
        const Vec3 local = rotate(conjugate(rotation), p - center);
        return std::fabs(local.x) <= halfExtents.x && std::fabs(local.y) <= halfExtents.y &&
               std::fabs(local.z) <= halfExtents.z;
    }

    static Obb fromAabb(const Aabb& b) {
        Obb o;
        o.center      = (b.min + b.max) * 0.5f;
        o.halfExtents = (b.max - b.min) * 0.5f;
        return o;
    }
};

}  // namespace cc
