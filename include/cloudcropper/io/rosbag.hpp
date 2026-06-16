// ROS2 bag reader: parses sensor_msgs/msg/PointCloud2 messages out of a rosbag2
// SQLite (`.db3`) bag — or an `.mcap` bag when the mcap library is available
// (CLOUDCROPPER_HAS_MCAP) — with a self-contained CDR decoder; no ROS dependency
// in the build. Compiled only when sqlite3 is available (CLOUDCROPPER_HAS_ROSBAG).
//
// A loaded cloud carries whatever fields the message had (xyz, rgb, normals,
// intensity, …); if normals are absent they are filled in later by the normal
// estimator like any other input.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cloudcropper/common/result.hpp"
#include "cloudcropper/core/point_cloud.hpp"

namespace cc::io {

struct BagReadOptions {
    std::string   topic;          // empty => the only PointCloud2 topic (error if ambiguous)
    bool          merge = false;  // concatenate all messages on the topic (else one frame)
    int           frame = 0;      // message index when !merge (0 = first)
    std::uint64_t maxPoints = 0;  // 0 = unlimited; else fail with CloudTooLarge
};

struct BagTopic {
    std::string   name;
    std::string   type;
    std::uint64_t count = 0;
};

// True if `path` looks like a rosbag2 SQLite bag: a `.db3` file, or a directory
// containing one.
bool isRosbagPath(const std::string& path);

// Lists every topic in the bag (name, type, message count).
Result<std::vector<BagTopic>> listBagTopics(const std::string& path);

// Reads a PointCloud2 topic into a PointCloud (one frame, or all merged).
Result<PointCloud> readRosbag(const std::string& path, const BagReadOptions& opt = {});

}  // namespace cc::io
