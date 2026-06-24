// Orbit camera. Orientation is kept as independent yaw / pitch / roll *angles*
// (not a single accumulated quaternion), each smoothed toward a target and
// recomposed into a quaternion every frame:
//     orient = Yaw(worldUp) * Pitch(localRight) * Roll(localForward)
// Because the smoothing is on the scalar angles, horizontal drag is pure world
// yaw with ZERO roll at every instant (no quaternion-geodesic roll wobble), and
// roll only changes when explicitly asked (Alt-drag). No Euler is ever extracted
// from a matrix, so there's no gimbal lock in normal use; pitch is unclamped so
// the view can go over the pole smoothly.
#pragma once

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace cc::viewer {

class Camera {
public:
    // Frame the cloud (focus at centre, distance fits the bounding sphere in the
    // FOV) and reset to the default 3/4 view. First call snaps; later calls ease.
    void fit(const glm::vec3& bmin, const glm::vec3& bmax) {
        focusTarget_    = 0.5f * (bmin + bmax);
        const float diag = glm::length(bmax - bmin);
        radius_         = std::max(0.5f * diag, 1e-3f);
        distanceTarget_ = radius_ / std::sin(glm::radians(fovYDeg_) * 0.5f);
        yawTarget_      = 0.6f;
        pitchTarget_    = -0.5f;
        rollTarget_     = 0.0f;
        if (!initialized_) {
            yaw_         = yawTarget_;
            pitch_       = pitchTarget_;
            roll_        = rollTarget_;
            focus_       = focusTarget_;
            distance_    = distanceTarget_;
            initialized_ = true;
        }
        updateClip();
    }

    // Choose the world up axis (Y for most files, Z for ROS clouds). `base_` is
    // the rotation that aligns the camera's local up (Y) with this world up, so
    // the same yaw/pitch/roll model works for any up axis.
    void setUp(const glm::vec3& u) {
        up_                = glm::normalize(u);
        const glm::vec3 y{0, 1, 0};
        const float     d = glm::dot(y, up_);
        if (d > 0.9999f)
            base_ = glm::quat(1, 0, 0, 0);
        else if (d < -0.9999f)
            base_ = glm::angleAxis(3.14159265f, glm::vec3{1, 0, 0});
        else
            base_ = glm::angleAxis(std::acos(d), glm::normalize(glm::cross(y, up_)));
    }

    // Drag-orbit. `rollMode` (Alt held) routes horizontal drag to roll instead of
    // yaw. Vertical drag is always pitch.
    void orbit(float dx, float dy, bool rollMode = false) {
        const float s = 0.01f;
        if (rollMode)
            rollTarget_ -= dx * s;
        else
            yawTarget_ -= dx * s;
        pitchTarget_ -= dy * s;
    }

    // Add to the yaw TARGET only (used by the preview auto-spin). The live yaw
    // still eases toward it via update(), so the spin stays smooth.
    void nudgeYaw(float radians) { yawTarget_ += radians; }

    // Snapshot the current orientation TARGETS so they can be restored later.
    // Returns {yaw,pitch,roll} targets at call time.
    [[nodiscard]] glm::vec3 orbitTarget() const {
        return {yawTarget_, pitchTarget_, rollTarget_};
    }

    // Set the orientation TARGETS (yaw,pitch,roll). The live state eases toward
    // them in update(); used to return the preview camera to its initial pose.
    void setOrbitTarget(const glm::vec3& t) {
        yawTarget_   = t.x;
        pitchTarget_ = t.y;
        rollTarget_  = t.z;
    }

    // True when the live orientation is within `eps` (radians) of its targets,
    // i.e. the easing has essentially settled.
    [[nodiscard]] bool nearOrbitTarget(float eps) const {
        return std::abs(yaw_ - yawTarget_) < eps && std::abs(pitch_ - pitchTarget_) < eps &&
               std::abs(roll_ - rollTarget_) < eps;
    }

    // Pan moves the focus in the camera's screen plane.
    void pan(float dx, float dy) {
        const glm::quat o     = orientOf(yaw_, pitch_, roll_);
        const glm::vec3 right = o * glm::vec3{1, 0, 0};
        const glm::vec3 up    = o * glm::vec3{0, 1, 0};
        focusTarget_ += (-dx * right + dy * up) * (distance_ * 0.0015f);
    }

    void dolly(float ticks) {
        distanceTarget_ = std::max(distanceTarget_ * std::pow(0.9f, ticks), 1e-4f);
    }

    // Ease the live state toward the targets. `dt` in seconds.
    void update(float dt) {
        const float a = std::clamp(1.0f - std::exp(-smooth_ * std::max(dt, 0.0f)), 0.0f, 1.0f);
        yaw_      = glm::mix(yaw_, yawTarget_, a);
        pitch_    = glm::mix(pitch_, pitchTarget_, a);
        roll_     = glm::mix(roll_, rollTarget_, a);
        focus_    = glm::mix(focus_, focusTarget_, a);
        distance_ = glm::mix(distance_, distanceTarget_, a);
        updateClip();
    }

    [[nodiscard]] glm::vec3 eye() const {
        return focus_ + (orientOf(yaw_, pitch_, roll_) * glm::vec3{0, 0, 1}) * distance_;
    }

    [[nodiscard]] glm::mat4 viewMatrix() const {
        const glm::quat o = orientOf(yaw_, pitch_, roll_);
        return glm::mat4_cast(glm::conjugate(o)) * glm::translate(glm::mat4(1.0f), -eye());
    }

    [[nodiscard]] glm::mat4 projMatrix(float aspect) const {
        return glm::perspective(glm::radians(fovYDeg_), aspect, near_, far_);
    }

    [[nodiscard]] glm::vec3 right() const {
        return orientOf(yaw_, pitch_, roll_) * glm::vec3{1, 0, 0};
    }

    [[nodiscard]] glm::vec3 screenUp() const {
        return orientOf(yaw_, pitch_, roll_) * glm::vec3{0, 1, 0};
    }

    [[nodiscard]] glm::vec3 forward() const {
        return orientOf(yaw_, pitch_, roll_) * glm::vec3{0, 0, -1};
    }

    [[nodiscard]] float distance() const { return distance_; }

private:
    // orient (camera->world) = Yaw(world up) * base * Pitch(localRight) * Roll(localFwd).
    // Yawing left-multiplies a world-up rotation, so horizontal drag never rolls.
    glm::quat orientOf(float yaw, float pitch, float roll) const {
        return glm::normalize(glm::angleAxis(yaw, up_) * base_ *
                              glm::angleAxis(pitch, glm::vec3{1, 0, 0}) *
                              glm::angleAxis(roll, glm::vec3{0, 0, 1}));
    }
    void updateClip() {
        near_ = std::max(distance_ * 1e-3f, 1e-3f);
        far_  = distance_ + radius_ * 4.0f;
    }

    glm::vec3 up_{0, 1, 0};       // world up axis
    glm::quat base_{1, 0, 0, 0};  // rotation aligning local up (Y) with up_
    float     yaw_ = 0.6f, yawTarget_ = 0.6f;
    float     pitch_ = -0.5f, pitchTarget_ = -0.5f;
    float     roll_ = 0.0f, rollTarget_ = 0.0f;
    glm::vec3 focus_{0.0f}, focusTarget_{0.0f};
    float     distance_ = 5.0f, distanceTarget_ = 5.0f;
    float     radius_   = 1.0f;
    float     fovYDeg_  = 45.0f;
    float     near_     = 0.01f;
    float     far_      = 1000.0f;
    float     smooth_   = 14.0f;  // easing rate (1/sec)
    bool      initialized_ = false;
};

}  // namespace cc::viewer
