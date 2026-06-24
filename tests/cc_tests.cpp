// Minimal self-contained test runner (no GTest dependency in the zero-dep
// slice). Doc 04 swaps in GoogleTest once vcpkg is wired; the assertions below
// migrate 1:1 to EXPECT_*.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "cloudcropper/core/analysis.hpp"
#include "cloudcropper/core/crop.hpp"
#include "cloudcropper/core/denoise.hpp"
#include "cloudcropper/core/normals.hpp"
#include "cloudcropper/io/byte_stream.hpp"
#include "cloudcropper/io/ply.hpp"
#if defined(CLOUDCROPPER_HAS_PCD)
#include "cloudcropper/io/pcd.hpp"
#endif
#if defined(CLOUDCROPPER_HAS_NPZ)
#include "cloudcropper/io/npz.hpp"
#endif
#if defined(CLOUDCROPPER_HAS_GZIP)
#include "cloudcropper/transport/gzip.hpp"
#endif
#if defined(CLOUDCROPPER_HAS_REGISTRATION)
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "cloudcropper/registration/config.hpp"
#include "cloudcropper/registration/python_worker.hpp"
#include "cloudcropper/registration/registration.hpp"
#endif

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "  FAIL: " << #cond << " @ " << __FILE__ << ":"       \
                      << __LINE__ << "\n";                                     \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

// Build a small synthetic cloud: xyz + rgb(u8x3) + intensity(f32).
cc::PointCloud makeCloud() {
    cc::PointCloud pc;
    auto&          pts = pc.positions();
    std::vector<std::uint8_t> rgbBytes;
    std::vector<float>        intenVals;
    for (int i = 0; i < 8; ++i) {
        pts.push_back({static_cast<float>(i), static_cast<float>(i) * 2.0f,
                       static_cast<float>(i) * 0.5f});
        rgbBytes.push_back(static_cast<std::uint8_t>(i * 10));
        rgbBytes.push_back(static_cast<std::uint8_t>(i * 10 + 1));
        rgbBytes.push_back(static_cast<std::uint8_t>(i * 10 + 2));
        intenVals.push_back(static_cast<float>(i) + 0.25f);
    }
    cc::AttributeColumn rgbFull(std::string(cc::attr::kRGB), cc::AttrType::U8, 3, 8);
    std::memcpy(rgbFull.bytes().data(), rgbBytes.data(), rgbBytes.size());
    cc::AttributeColumn intenFull(std::string(cc::attr::kIntensity), cc::AttrType::F32, 1, 8);
    std::memcpy(intenFull.bytes().data(), intenVals.data(), intenVals.size() * sizeof(float));
    pc.add(std::move(rgbFull));
    pc.add(std::move(intenFull));
    return pc;
}

bool cloudsEqual(const cc::PointCloud& a, const cc::PointCloud& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const cc::Vec3 p = a.positions()[i], q = b.positions()[i];
        if (std::fabs(p.x - q.x) > 1e-4f || std::fabs(p.y - q.y) > 1e-4f ||
            std::fabs(p.z - q.z) > 1e-4f)
            return false;
    }
    const auto* ar = a.find(cc::attr::kRGB);
    const auto* br = b.find(cc::attr::kRGB);
    if (!ar || !br || ar->bytes().size() != br->bytes().size()) return false;
    if (std::memcmp(ar->bytes().data(), br->bytes().data(), ar->bytes().size()) != 0) return false;
    const auto* ai = a.find(cc::attr::kIntensity);
    const auto* bi = b.find(cc::attr::kIntensity);
    if (!ai || !bi) return false;
    auto av = ai->as<float>(), bv = bi->as<float>();
    for (std::size_t i = 0; i < av.size(); ++i)
        if (std::fabs(av[i] - bv[i]) > 1e-5f) return false;
    return true;
}

std::optional<cc::Vec3> parseMetaVec3(const std::string& s) {
    std::istringstream iss(s);
    float              x = 0.0f, y = 0.0f, z = 0.0f;
    if (iss >> x >> y >> z) return cc::Vec3{x, y, z};
    return std::nullopt;
}

void checkNear(cc::Vec3 got, cc::Vec3 want, float eps = 1e-4f) {
    CHECK(std::fabs(got.x - want.x) < eps);
    CHECK(std::fabs(got.y - want.y) < eps);
    CHECK(std::fabs(got.z - want.z) < eps);
}

void roundTripVia(const char* label, const cc::io::IWriter& w, const cc::io::IReader& r,
                  cc::io::Encoding enc) {
    std::cerr << "[" << label << "]\n";
    const cc::PointCloud   original = makeCloud();
    cc::io::MemoryByteSink sink;
    cc::io::WriteOptions   opt;
    opt.encoding = enc;
    auto wr      = w.write(original, sink, opt);
    CHECK(static_cast<bool>(wr));
    if (!wr) {
        std::cerr << "    write error: " << wr.error().message << "\n";
        return;
    }
    cc::io::MemoryByteSource src(sink.data());
    auto                     rd = r.read(src, {});
    CHECK(static_cast<bool>(rd));
    if (!rd) {
        std::cerr << "    read error: " << rd.error().message << "\n";
        return;
    }
    CHECK(cloudsEqual(original, rd.value()));
}

void testPlyRoundTrip(bool ascii) {
    std::cerr << "[ply round-trip " << (ascii ? "ascii" : "binary") << "]\n";
    const cc::PointCloud original = makeCloud();

    cc::io::MemoryByteSink sink;
    cc::io::PlyWriter      writer;
    cc::io::WriteOptions   opt;
    opt.encoding = ascii ? cc::io::Encoding::Ascii : cc::io::Encoding::Binary;
    auto wr      = writer.write(original, sink, opt);
    CHECK(static_cast<bool>(wr));

    cc::io::MemoryByteSource src(sink.data());
    cc::io::PlyReader        reader;
    auto                     rd = reader.read(src, {});
    CHECK(static_cast<bool>(rd));
    if (rd) CHECK(cloudsEqual(original, rd.value()));
}

void testCropAabb() {
    std::cerr << "[crop aabb]\n";
    cc::PointCloud pc;
    for (int x = 0; x < 10; ++x)
        for (int y = 0; y < 10; ++y)
            pc.positions().push_back({static_cast<float>(x), static_cast<float>(y), 0.0f});

    cc::Aabb box;
    box.min = {2.0f, 2.0f, -1.0f};
    box.max = {4.0f, 4.0f, 1.0f};  // x,y in {2,3,4} => 3x3 = 9 points
    cc::CropSpec spec;
    spec.boxes.push_back({cc::Obb::fromAabb(box), cc::BoxRole::Include});

    const cc::CropResult r = cc::crop(pc, spec);
    CHECK(r.totalCount == 100);
    CHECK(r.inCount == 9);
    const cc::PointCloud out = cc::cropToCloud(pc, spec);
    CHECK(out.size() == 9);
}

void testCropRecenter() {
    std::cerr << "[crop recenter]\n";
    cc::PointCloud pc;
    for (int x = 0; x < 10; ++x)
        for (int y = 0; y < 10; ++y)
            pc.positions().push_back({static_cast<float>(x), static_cast<float>(y), 0.0f});

    cc::Aabb box;
    box.min = {2.0f, 2.0f, -1.0f};
    box.max = {4.0f, 4.0f, 1.0f};  // center (3,3,0); keeps x,y in {2,3,4} => 3x3
    cc::CropSpec spec;
    spec.boxes.push_back({cc::Obb::fromAabb(box), cc::BoxRole::Include});

    const cc::PointCloud out = cc::cropToCloud(pc, spec);
    CHECK(out.size() == 9);

    // Output is recentered: bounds center ~ origin, spanning [-1,-1]..[1,1].
    const cc::Aabb b = out.bounds();
    const cc::Vec3 c = (b.min + b.max) * 0.5f;
    CHECK(std::fabs(c.x) < 1e-4f && std::fabs(c.y) < 1e-4f && std::fabs(c.z) < 1e-4f);
    CHECK(std::fabs(b.min.x + 1.0f) < 1e-4f && std::fabs(b.max.x - 1.0f) < 1e-4f);
    CHECK(std::fabs(b.min.y + 1.0f) < 1e-4f && std::fabs(b.max.y - 1.0f) < 1e-4f);

    // Coordinate mean is ~ 0 as well.
    cc::Vec3 mean{};
    for (const cc::Vec3& p : out.positions()) mean = mean + p;
    mean = mean * (1.0f / static_cast<float>(out.size()));
    CHECK(std::fabs(mean.x) < 1e-4f && std::fabs(mean.y) < 1e-4f && std::fabs(mean.z) < 1e-4f);

    // The applied offset is recorded and parses to the box center (3,3,0).
    CHECK(out.metadata().count("crop_offset") == 1);
    auto it = out.metadata().find("crop_offset");
    if (it != out.metadata().end()) {
        double ox = 0.0, oy = 0.0, oz = 0.0;
        CHECK(std::sscanf(it->second.c_str(), "%lf %lf %lf", &ox, &oy, &oz) == 3);
        CHECK(std::fabs(ox - 3.0) < 1e-4 && std::fabs(oy - 3.0) < 1e-4 &&
              std::fabs(oz) < 1e-4);
    }
}

void testCropObbExclude() {
    std::cerr << "[crop obb + exclude]\n";
    cc::PointCloud pc;
    for (int x = -5; x <= 5; ++x)
        pc.positions().push_back({static_cast<float>(x), 0.0f, 0.0f});  // 11 points

    // Include everything within |x|<=5, then carve out |x|<=1 (3 points: -1,0,1).
    cc::Aabb inc;
    inc.min = {-5.5f, -1.0f, -1.0f};
    inc.max = {5.5f, 1.0f, 1.0f};
    cc::Aabb exc;
    exc.min = {-1.5f, -1.0f, -1.0f};
    exc.max = {1.5f, 1.0f, 1.0f};
    cc::CropSpec spec;
    spec.boxes.push_back({cc::Obb::fromAabb(inc), cc::BoxRole::Include});
    spec.boxes.push_back({cc::Obb::fromAabb(exc), cc::BoxRole::Exclude});

    const cc::CropResult r = cc::crop(pc, spec);
    CHECK(r.inCount == 8);  // 11 - 3
}

void testGatherAlignment() {
    std::cerr << "[gather alignment]\n";
    const cc::PointCloud pc = makeCloud();
    std::vector<std::uint32_t> idx = {1, 3, 5};
    const cc::PointCloud sub = pc.gather(idx);
    CHECK(sub.size() == 3);
    // intensity of original index 3 == 3.25 should land at sub index 1.
    const auto* inten = sub.find(cc::attr::kIntensity);
    CHECK(inten != nullptr);
    if (inten) CHECK(std::fabs(inten->as<float>()[1] - 3.25f) < 1e-5f);
    // rgb of original index 5 (50,51,52) at sub index 2.
    const auto* rgb = sub.find(cc::attr::kRGB);
    CHECK(rgb != nullptr);
    if (rgb) {
        auto b = rgb->as<std::uint8_t>();
        CHECK(b[2 * 3 + 0] == 50 && b[2 * 3 + 1] == 51 && b[2 * 3 + 2] == 52);
    }
}

void testNormalsPlane() {
    std::cerr << "[normals: plane => +Z]\n";
    cc::PointCloud pc;
    // 21x21 grid on the z=0 plane, jitter-free.
    for (int i = -10; i <= 10; ++i)
        for (int j = -10; j <= 10; ++j)
            pc.positions().push_back({static_cast<float>(i) * 0.1f, static_cast<float>(j) * 0.1f, 0.0f});

    cc::NormalParams np;
    np.k         = 16;
    np.viewpoint = cc::Vec3{0.0f, 0.0f, 10.0f};  // above the plane => normals point +Z
    cc::estimateNormals(pc, np);

    const auto* nc = pc.find(cc::attr::kNormal);
    CHECK(nc != nullptr);
    if (!nc) return;
    auto   nrm   = nc->as<float>();
    int    good  = 0;
    double maxXY = 0.0;
    for (std::size_t i = 0; i < pc.size(); ++i) {
        const float nx = nrm[i * 3 + 0], ny = nrm[i * 3 + 1], nz = nrm[i * 3 + 2];
        const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        CHECK(std::fabs(len - 1.0f) < 1e-3f);          // unit length
        if (nz > 0.99f) ++good;                        // points up
        maxXY = std::max<double>(maxXY, std::sqrt(double(nx) * nx + double(ny) * ny));
    }
    // Interior points must be essentially +Z; allow edge points to wobble.
    CHECK(good >= static_cast<int>(pc.size()) * 7 / 10);
}

void testNormalsSphere() {
    std::cerr << "[normals: sphere => radial]\n";
    cc::PointCloud pc;
    // Fibonacci sphere, radius 1, centered at origin.
    const int    n  = 2000;
    const double ga = 3.39996322972865332;  // golden angle
    for (int i = 0; i < n; ++i) {
        const double z = 1.0 - 2.0 * (i + 0.5) / n;
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double t = ga * i;
        pc.positions().push_back({static_cast<float>(r * std::cos(t)),
                                  static_cast<float>(r * std::sin(t)), static_cast<float>(z)});
    }
    cc::NormalParams np;
    np.k            = 24;
    np.orientOutward = true;  // centroid ~ origin => outward == radial
    cc::estimateNormals(pc, np);

    const auto* nc = pc.find(cc::attr::kNormal);
    CHECK(nc != nullptr);
    if (!nc) return;
    auto   nrm  = nc->as<float>();
    double meanDot = 0.0;
    for (std::size_t i = 0; i < pc.size(); ++i) {
        const cc::Vec3 p = pc.positions()[i];  // == outward radial unit (radius 1)
        const cc::Vec3 nn{nrm[i * 3 + 0], nrm[i * 3 + 1], nrm[i * 3 + 2]};
        meanDot += cc::dot(p, nn);  // should be ~ +1
    }
    meanDot /= pc.size();
    CHECK(meanDot > 0.95);  // normals align with the radial (outward) direction
}

void testDenoise() {
    std::cerr << "[denoise: drop far outliers]\n";
    cc::PointCloud pc;
    // 20x20 tight cluster on z=0 (spacing 0.05) ...
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            pc.positions().push_back({i * 0.05f, j * 0.05f, 0.0f});
    // ... plus 5 far outliers.
    for (int m = 0; m < 5; ++m)
        pc.positions().push_back({10.0f + static_cast<float>(m), 10.0f, 10.0f});
    CHECK(pc.size() == 405);

    const cc::PointCloud out = cc::removeStatisticalOutliers(pc, {16, 2.0f});
    CHECK(out.size() >= 395 && out.size() <= 401);  // ~400 cluster kept
    float maxc = 0.0f;
    for (const cc::Vec3& p : out.positions())
        maxc = std::max({maxc, std::fabs(p.x), std::fabs(p.y), std::fabs(p.z)});
    CHECK(maxc < 2.0f);  // every far outlier (coord >= 10) was removed
}

#if defined(CLOUDCROPPER_HAS_GZIP)
void testGzip() {
    std::cerr << "[gzip roundtrip]\n";
    std::string text;
    for (int i = 0; i < 2000; ++i) text += "CloudCropper gzip transport test. ";
    std::span<const std::byte> in(reinterpret_cast<const std::byte*>(text.data()), text.size());

    auto c = cc::transport::gzipCompress(in);
    CHECK(static_cast<bool>(c));
    if (!c) return;
    CHECK(c->size() < text.size());  // compressible payload shrinks

    auto d = cc::transport::gzipDecompress(std::span<const std::byte>(c->data(), c->size()));
    CHECK(static_cast<bool>(d));
    if (!d) return;
    CHECK(d->size() == text.size());
    CHECK(std::memcmp(d->data(), text.data(), text.size()) == 0);

    // PLY -> gzip -> gunzip -> PLY round-trip through the real codec path.
    const cc::PointCloud orig = makeCloud();
    cc::io::MemoryByteSink sink;
    cc::io::PlyWriter      w;
    CHECK(static_cast<bool>(w.write(orig, sink, {})));
    auto gz = cc::transport::gzipCompress(sink.data());
    CHECK(static_cast<bool>(gz));
    if (!gz) return;
    auto raw = cc::transport::gzipDecompress(std::span<const std::byte>(gz->data(), gz->size()));
    CHECK(static_cast<bool>(raw));
    if (!raw) return;
    cc::io::MemoryByteSource src(*raw);
    cc::io::PlyReader        r;
    auto                     back = r.read(src, {});
    CHECK(static_cast<bool>(back));
    if (back) CHECK(cloudsEqual(orig, back.value()));
}
#endif

void testCropOctreeParity() {
    std::cerr << "[crop octree parity]\n";
    cc::PointCloud pc;  // 40^3 = 64000-point grid
    for (int x = 0; x < 40; ++x)
        for (int y = 0; y < 40; ++y)
            for (int z = 0; z < 40; ++z)
                pc.positions().push_back(
                    {static_cast<float>(x) * 0.1f, static_cast<float>(y) * 0.1f, static_cast<float>(z) * 0.1f});

    cc::Aabb inc;
    inc.min = {0.5f, 0.5f, 0.5f};
    inc.max = {2.5f, 2.5f, 2.5f};
    cc::Aabb exc;
    exc.min = {1.0f, 1.0f, 1.0f};
    exc.max = {1.5f, 1.5f, 1.5f};
    cc::CropSpec spec;
    spec.boxes.push_back({cc::Obb::fromAabb(inc), cc::BoxRole::Include});
    spec.boxes.push_back({cc::Obb::fromAabb(exc), cc::BoxRole::Exclude});

    cc::IndexPolicy never;
    never.mode = cc::IndexPolicy::Mode::Never;
    cc::IndexPolicy always;
    always.mode = cc::IndexPolicy::Mode::Always;
    const cc::CropResult bf = cc::crop(pc, spec, never);   // brute-force
    const cc::CropResult ot = cc::crop(pc, spec, always);  // octree
    CHECK(bf.inCount > 0);
    CHECK(bf.inCount == ot.inCount);
    CHECK(bf.mask == ot.mask);  // identical result
}

void testPcaFrame() {
    std::cerr << "[pca frame]\n";
    cc::PointCloud pc;  // elongated along X
    for (int i = -50; i <= 50; ++i)
        pc.positions().push_back({static_cast<float>(i) * 0.1f,
                                  0.01f * static_cast<float>((i % 3) - 1), 0.0f});
    const cc::CanonicalFrame f = cc::pcaFrame(pc);
    CHECK(std::fabs(f.axis.x) > 0.99f);  // principal axis ~ ±X
}

#if defined(CLOUDCROPPER_HAS_REGISTRATION)
// A bumpy non-symmetric surface so ICP-family methods have unambiguous geometry.
cc::PointCloud regTestCloud() {
    cc::PointCloud pc;
    for (int x = 0; x < 40; ++x)
        for (int y = 0; y < 40; ++y) {
            const float fx = static_cast<float>(x) * 0.05f;
            const float fy = static_cast<float>(y) * 0.05f;
            const float fz = 0.3f * std::sin(fx * 3.1f) + 0.2f * std::cos(fy * 4.3f) +
                             0.05f * std::sin(fx * 11.0f + fy * 7.0f);
            pc.positions().push_back({fx, fy, fz});
        }
    return pc;
}

// Row-major 4x4: rotation about Y by `deg` plus translation t.
std::array<double, 16> makeT(double deg, cc::Vec3 t) {
    const double c = std::cos(deg * 3.14159265358979 / 180.0);
    const double s = std::sin(deg * 3.14159265358979 / 180.0);
    return {c, 0, s, t.x, 0, 1, 0, t.y, -s, 0, c, t.z, 0, 0, 0, 1};
}

void testRegistrationGicp() {
    std::cerr << "[registration gicp]\n";
    const cc::PointCloud target = regTestCloud();
    cc::PointCloud       source = target;
    // Move the source by a known transform; registration must find its inverse.
    const auto Tgt = makeT(12.0, {0.15f, -0.1f, 0.08f});
    cc::reg::applyTransform(source, Tgt);

    cc::reg::RegOptions opt;
    opt.algo = cc::reg::RegAlgo::Gicp;
    auto rr  = cc::reg::registerClouds(source, target, opt);
    CHECK(static_cast<bool>(rr));
    if (!rr) return;
    CHECK(rr->converged);
    // Composing recovered T with the applied motion must give identity:
    // p_target = T_rec * (Tgt * p) => T_rec*Tgt ~ I.
    const auto& R = rr->transform;
    double      maxOff = 0.0;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            double v = 0.0;
            for (int k = 0; k < 4; ++k)
                v += R[static_cast<std::size_t>(r * 4 + k)] *
                     Tgt[static_cast<std::size_t>(k * 4 + c)];
            const double want = (r == c) ? 1.0 : 0.0;
            maxOff             = std::max(maxOff, std::fabs(v - want));
        }
    CHECK(maxOff < 0.02);          // ~rot<0.5deg, trans<2cm on a 2m cloud
    CHECK(rr->rmse < 0.01);        // points land back on the target surface
    CHECK(rr->inliers > source.size() / 2);
}

void testApplyTransformPoseMetadata() {
    std::cerr << "[registration pose metadata transform]\n";
    cc::PointCloud pc;
    pc.positions().push_back({0.0f, 0.0f, 0.0f});
    pc.positions().push_back({1.0f, 2.0f, 3.0f});
    pc.metadata()["object_pose_origin_local"] = "1 2 3";
    pc.metadata()["object_pose_dir_local"]    = "1 0 0";
    pc.metadata()["bar_axis_point_local"]     = "0 0 1";
    pc.metadata()["bar_axis_dir_local"]       = "0 0 1";
    pc.metadata()["bbox_min"]                 = "0 0 0";
    pc.metadata()["bbox_max"]                 = "1 2 3";
    pc.metadata()["bbox_center"]              = "0.5 1 1.5";

    const auto T = makeT(90.0, {10.0f, 0.0f, 0.0f});
    cc::reg::applyTransform(pc, T);

    auto origin = parseMetaVec3(pc.metadata().at("object_pose_origin_local"));
    auto dir    = parseMetaVec3(pc.metadata().at("object_pose_dir_local"));
    auto axisP  = parseMetaVec3(pc.metadata().at("bar_axis_point_local"));
    auto axisD  = parseMetaVec3(pc.metadata().at("bar_axis_dir_local"));
    auto bmin   = parseMetaVec3(pc.metadata().at("bbox_min"));
    auto bmax   = parseMetaVec3(pc.metadata().at("bbox_max"));
    auto bcen   = parseMetaVec3(pc.metadata().at("bbox_center"));

    CHECK(static_cast<bool>(origin));
    CHECK(static_cast<bool>(dir));
    CHECK(static_cast<bool>(axisP));
    CHECK(static_cast<bool>(axisD));
    CHECK(static_cast<bool>(bmin));
    CHECK(static_cast<bool>(bmax));
    CHECK(static_cast<bool>(bcen));
    if (!origin || !dir || !axisP || !axisD || !bmin || !bmax || !bcen) return;

    checkNear(*origin, {13.0f, 2.0f, -1.0f});
    checkNear(*dir, {0.0f, 0.0f, -1.0f});
    checkNear(*axisP, {11.0f, 0.0f, 0.0f});
    checkNear(*axisD, {1.0f, 0.0f, 0.0f});
    checkNear(*bmin, {10.0f, 0.0f, -1.0f});
    checkNear(*bmax, {13.0f, 2.0f, 0.0f});
    checkNear(*bcen, {11.5f, 1.0f, -0.5f});
}
// Unique (non-periodic) geometry: fixed gaussian blobs on a plane, so the
// global solution is unambiguous.
cc::PointCloud regBlobsCloud() {
    cc::PointCloud pc;
    const float    cx[] = {0.5f, 2.1f, 1.2f, 2.6f, 0.9f};
    const float    cy[] = {0.7f, 0.4f, 2.2f, 2.4f, 1.5f};
    const float    am[] = {0.35f, -0.3f, 0.25f, -0.2f, 0.4f};
    for (int x = 0; x < 50; ++x)
        for (int y = 0; y < 50; ++y) {
            const float fx = static_cast<float>(x) * 0.06f;
            const float fy = static_cast<float>(y) * 0.06f;
            float       fz = 0.0f;
            for (int k = 0; k < 5; ++k) {
                const float dx = fx - cx[k], dy = fy - cy[k];
                fz += am[k] * std::exp(-(dx * dx + dy * dy) / 0.18f);
            }
            pc.positions().push_back({fx, fy, fz});
        }
    return pc;
}

// 90deg about Z + a large offset — defeats purely local methods.
std::array<double, 16> bigT() {
    return {0, -1, 0, 2.0, 1, 0, 0, -1.5, 0, 0, 1, 0.8, 0, 0, 0, 1};
}

double composedOffset(const std::array<double, 16>& A, const std::array<double, 16>& B) {
    double maxOff = 0.0;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            double v = 0.0;
            for (int k = 0; k < 4; ++k)
                v += A[static_cast<std::size_t>(r * 4 + k)] *
                     B[static_cast<std::size_t>(k * 4 + c)];
            maxOff = std::max(maxOff, std::fabs(v - ((r == c) ? 1.0 : 0.0)));
        }
    return maxOff;
}

void testRegConfigDefaults() {
    std::cerr << "[registration config defaults]\n";
    // Point the loader at a temp config dir and check the YAML wins over the
    // compiled defaults (and that a missing file keeps them).
    const char* dir = "/tmp/cc_cfg_test";
    std::filesystem::create_directories(dir);
    {
        std::ofstream f(std::string(dir) + "/gradient-sdf-gpu.yaml");
        f << "# test\nresolution: 128\ntrunc_mul: 6.5\nuncertainty: false\n"
             "refine: false\nthreads: 7\n";
    }
    setenv("CLOUDCROPPER_CONFIG_DIR", dir, 1);
    const cc::reg::RegOptions g = cc::reg::defaultsFor(cc::reg::RegAlgo::GradientSdfGpu);
    CHECK(g.sdfResolution == 128);
    CHECK(std::fabs(g.sdfTruncMul - 6.5f) < 1e-6f);
    CHECK(g.sdfUncertainty == false);
    CHECK(g.refine == false);
    CHECK(g.threads == 7);
    // kiss-matcher.yaml absent in the temp dir -> falls through the search to
    // the repo config/ (refine true) or compiled defaults; either way valid:
    const cc::reg::RegOptions k = cc::reg::defaultsFor(cc::reg::RegAlgo::KissGicp);
    CHECK(k.algo == cc::reg::RegAlgo::KissGicp);
    CHECK(k.threads >= 1);
    unsetenv("CLOUDCROPPER_CONFIG_DIR");
    std::filesystem::remove_all(dir);
}

void testJsonParse() {
    std::cerr << "[json parse]\n";
    // Nested object/array, escapes, exponent floats, bool/null.
    auto r = cc::reg::parseJson(
        R"({"id":2,"ok":true,"result":{"transform":[1,2.5,-3e-2],)"
        R"("msg":"a\"b\\c\nd","none":null,"neg":false}})");
    CHECK(static_cast<bool>(r));
    if (r) {
        CHECK(r->isObject());
        CHECK(r->find("id") && r->find("id")->asDouble() == 2.0);
        CHECK(r->find("ok") && r->find("ok")->asBool());
        const cc::reg::JsonValue* res = r->find("result");
        CHECK(res && res->isObject());
        if (res && res->isObject()) {
            const cc::reg::JsonValue* t = res->find("transform");
            CHECK(t && t->isArray() && t->array.size() == 3);
            if (t && t->array.size() == 3) {
                CHECK(t->array[1].asDouble() == 2.5);
                CHECK(std::fabs(t->array[2].asDouble() + 0.03) < 1e-12);
            }
            CHECK(res->find("msg") && res->find("msg")->asString() == "a\"b\\c\nd");
            CHECK(res->find("none") && res->find("none")->isNull());
            CHECK(res->find("neg") && !res->find("neg")->asBool(true));
        }
    }
    // \uXXXX escapes (incl. a surrogate pair) decode to UTF-8.
    auto u = cc::reg::parseJson("\"a\\u00e9\\ud83d\\ude00\"");
    CHECK(static_cast<bool>(u));
    if (u) CHECK(u->asString() == "a\xc3\xa9\xf0\x9f\x98\x80");
    // Round trip through the escaper.
    auto esc = cc::reg::parseJson("\"" + cc::reg::jsonEscape("p\"q\\r\nnul\x01") + "\"");
    CHECK(static_cast<bool>(esc));
    if (esc) CHECK(esc->asString() == "p\"q\\r\nnul\x01");
    // Malformed inputs must fail, not crash or half-parse.
    CHECK(!cc::reg::parseJson(""));
    CHECK(!cc::reg::parseJson(R"({"a":1)"));      // truncated object
    CHECK(!cc::reg::parseJson("{} x"));           // trailing garbage
    CHECK(!cc::reg::parseJson("[1,]"));           // dangling comma
    CHECK(!cc::reg::parseJson("\"abc"));          // unterminated string
    CHECK(!cc::reg::parseJson("1e"));             // malformed exponent
    CHECK(!cc::reg::parseJson("-"));              // lone minus
    CHECK(!cc::reg::parseJson(R"("\ud83d")"));    // lone surrogate
}

void testPythonWorkerProtocol() {
    std::cerr << "[python worker protocol]\n";
    if (std::system("command -v python3 >/dev/null 2>&1") != 0) {
        std::cerr << "  SKIP: python3 not on PATH\n";
        return;
    }
    namespace fs       = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "cc_worker_proto_test";
    fs::create_directories(dir);
    const fs::path script = dir / "fake_worker.py";
    {
        // stdlib-only stand-in that speaks the gsdf_worker.py protocol,
        // including the stdout hijack (the noise print must NOT reach us).
        std::ofstream f(script);
        f << "import json,os,sys,time\n"
             "proto=os.fdopen(os.dup(1),'w',buffering=1)\n"
             "os.dup2(2,1)\n"
             "proto.write(json.dumps({'event':'loading','pid':os.getpid()})+'\\n')\n"
             "print('stdout noise: must land in the log, not the protocol')\n"
             "proto.write(json.dumps({'event':'ready','device':'fake'})+'\\n')\n"
             "for line in sys.stdin:\n"
             "    r=json.loads(line); i=r.get('id'); op=r.get('op')\n"
             "    if op=='ping':\n"
             "        proto.write(json.dumps({'id':i,'ok':True,'result':{'device':'fake'}})+'\\n')\n"
             "    elif op=='boom':\n"
             "        proto.write(json.dumps({'id':i,'ok':False,"
             "'error':{'type':'ValueError','message':'boom'}})+'\\n')\n"
             "    elif op=='die': os._exit(1)\n"
             "    elif op=='sleep': time.sleep(30)\n"
             "    elif op=='shutdown':\n"
             "        proto.write(json.dumps({'id':i,'ok':True,'result':{}})+'\\n'); sys.exit(0)\n";
    }
    {
        cc::reg::PythonWorker::Options o;
        o.python          = "python3";
        o.script          = script.string();
        o.logFile         = (dir / "worker.log").string();
        o.readyTimeoutSec = 30;
        cc::reg::PythonWorker w(o);

        // Handshake + ping round trip, protocol clean despite the noise print.
        auto p = w.call("ping", "", 10);
        CHECK(static_cast<bool>(p));
        if (p) {
            const cc::reg::JsonValue* res = p->find("result");
            CHECK(res && res->find("device") && res->find("device")->asString() == "fake");
        }
        // ok:false propagates as an error and does NOT kill the worker.
        auto b = w.call("boom", "", 10);
        CHECK(!b);
        if (!b) CHECK(b.error().message.find("boom") != std::string::npos);
        CHECK(w.alive());
        // Crash mid-request -> error now, respawn on the next call.
        auto d = w.call("die", "", 10);
        CHECK(!d);
        CHECK(!w.alive());
        auto p2 = w.call("ping", "", 10);
        CHECK(static_cast<bool>(p2));
        // Hung request -> timeout kill -> error now, respawn on the next call.
        auto s = w.call("sleep", "", 1);
        CHECK(!s);
        if (!s) CHECK(s.error().message.find("timed out") != std::string::npos);
        auto p3 = w.call("ping", "", 10);
        CHECK(static_cast<bool>(p3));
    }  // destructor: shutdown -> reap (hangs here = teardown bug)
    std::error_code ec;
    fs::remove_all(dir, ec);
}

#if defined(CLOUDCROPPER_HAS_NPZ)
// End-to-end through gsdf_gpu::run with a FAKE worker (no torch needed):
// verifies the generic yaml->JSON forwarding (a sentinel n_steps from the
// temp config must arrive typed), the uncertainty plumbing (flag + target_key
// suffix), and the confidence/norm_residual -> RegResult mapping. Must be the
// process's FIRST gsdf-gpu call: the real worker + its spawn options are
// static and bind to this temp config.
void testGsdfGpuWorkerResult() {
    std::cerr << "[gsdf-gpu worker result]\n";
    if (std::system("command -v python3 >/dev/null 2>&1") != 0) {
        std::cerr << "  SKIP: python3 not on PATH\n";
        return;
    }
    namespace fs       = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "cc_gsdf_worker_cfg";
    fs::create_directories(dir);
    {
        std::ofstream f(dir / "gradient-sdf-gpu.yaml");
        f << "python: python3\nn_steps: 123\ntimeout_sec: 30\n";
    }
    const fs::path script = dir / "fake_gsdf_worker.py";
    {
        std::ofstream f(script);
        f << "import json,os,sys\n"
             "proto=os.fdopen(os.dup(1),'w',buffering=1)\n"
             "os.dup2(2,1)\n"
             "proto.write(json.dumps({'event':'loading','pid':os.getpid()})+'\\n')\n"
             "proto.write(json.dumps({'event':'ready','device':'fake'})+'\\n')\n"
             "for line in sys.stdin:\n"
             "    r=json.loads(line); i=r.get('id'); op=r.get('op')\n"
             "    if op=='shutdown':\n"
             "        proto.write(json.dumps({'id':i,'ok':True,'result':{}})+'\\n'); sys.exit(0)\n"
             "    bad=[]\n"
             "    if r.get('n_steps')!=123: bad.append('n_steps')\n"
             "    if r.get('uncertainty') is not True: bad.append('uncertainty')\n"
             "    if abs(float(r.get('trunc_mul',0))-4.0)>1e-6: bad.append('trunc_mul')\n"
             "    if ':u1:' not in r.get('target_key',''): bad.append('target_key')\n"
             "    if not (os.path.exists(r.get('source','')) and "
             "os.path.exists(r.get('target',''))): bad.append('npz')\n"
             "    if bad:\n"
             "        proto.write(json.dumps({'id':i,'ok':False,'error':{'type':"
             "'AssertionError','message':'bad request: '+','.join(bad)}})+'\\n')\n"
             "    else:\n"
             "        proto.write(json.dumps({'id':i,'ok':True,'result':{'transform':"
             "[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],'converged':1,'loss':0.001,'iou':0.5,"
             "'confidence':0.93,'norm_residual':1.2,'device':'fake','seconds':0.1,"
             "'cache_hit':0}})+'\\n')\n";
    }
    setenv("CLOUDCROPPER_CONFIG_DIR", dir.c_str(), 1);
    setenv("CLOUDCROPPER_GSDF_GPU_SCRIPT", script.c_str(), 1);

    const cc::PointCloud target = regBlobsCloud();
    cc::PointCloud       source = target;
    cc::reg::RegOptions  opt    = cc::reg::defaultsFor(cc::reg::RegAlgo::GradientSdfGpu);
    CHECK(opt.sdfUncertainty);  // compiled default survives the minimal config
    auto rr = cc::reg::registerClouds(source, target, opt);
    CHECK(static_cast<bool>(rr));
    if (!rr) {
        std::cerr << "  error: " << rr.error().message << "\n";
    } else {
        CHECK(rr->converged);
        CHECK(std::fabs(rr->confidence - 0.93) < 1e-9);
        CHECK(std::fabs(rr->normResidual - 1.2) < 1e-9);
        CHECK(rr->transform == cc::reg::kIdentity4);
        CHECK(rr->detail.find("gradient-SDF (worker, fake)") == 0);
    }
    unsetenv("CLOUDCROPPER_CONFIG_DIR");
    unsetenv("CLOUDCROPPER_GSDF_GPU_SCRIPT");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

// End-to-end through bufferx::run with a FAKE worker (no torch needed): verifies
// the generic yaml->JSON forwarding (a sentinel num_keypoints from the temp
// config must arrive typed), that `refine` is forced false to the worker (the
// GICP refine is chained in C++), the num_inliers/fitness -> detail mapping, and
// the BufferXGicp -> GICP chaining (detail is prepended before the GICP line).
// Must be the process's FIRST bufferx call: the real worker + its spawn options
// are static and bind to this temp config.
void testBufferxWorkerResult() {
    std::cerr << "[bufferx worker result]\n";
    if (std::system("command -v python3 >/dev/null 2>&1") != 0) {
        std::cerr << "  SKIP: python3 not on PATH\n";
        return;
    }
    namespace fs       = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "cc_bufferx_worker_cfg";
    fs::create_directories(dir);
    {
        std::ofstream f(dir / "bufferx.yaml");
        f << "python: python3\nnum_keypoints: 777\ntimeout_sec: 30\n";
    }
    const fs::path script = dir / "fake_bufferx_worker.py";
    {
        std::ofstream f(script);
        f << "import json,os,sys\n"
             "proto=os.fdopen(os.dup(1),'w',buffering=1)\n"
             "os.dup2(2,1)\n"
             "proto.write(json.dumps({'event':'loading','pid':os.getpid()})+'\\n')\n"
             "proto.write(json.dumps({'event':'ready','device':'fake','bufferx':0})+'\\n')\n"
             "for line in sys.stdin:\n"
             "    r=json.loads(line); i=r.get('id'); op=r.get('op')\n"
             "    if op=='shutdown':\n"
             "        proto.write(json.dumps({'id':i,'ok':True,'result':{}})+'\\n'); sys.exit(0)\n"
             "    bad=[]\n"
             "    if r.get('num_keypoints')!=777: bad.append('num_keypoints')\n"
             "    if r.get('refine') is not False: bad.append('refine')\n"
             "    if 'voxel_size' not in r: bad.append('voxel_size')\n"
             "    if 'cuda' not in r.get('target_key',''): bad.append('target_key')\n"
             "    if not (os.path.exists(r.get('source','')) and "
             "os.path.exists(r.get('target',''))): bad.append('npz')\n"
             "    if bad:\n"
             "        proto.write(json.dumps({'id':i,'ok':False,'error':{'type':"
             "'AssertionError','message':'bad request: '+','.join(bad)}})+'\\n')\n"
             "    elif float(r.get('voxel_size',0) or 0)>0:\n"
             "        proto.write(json.dumps({'id':i,'ok':True,'result':{'transform':"
             "[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],'converged':0,'device':'fake',"
             "'seconds':0.0,'cache_hit':0,'note':'weights missing'}})+'\\n')\n"
             "    else:\n"
             "        proto.write(json.dumps({'id':i,'ok':True,'result':{'transform':"
             "[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],'converged':1,'num_inliers':4321,"
             "'fitness':0.91,'device':'fake','seconds':0.1,'cache_hit':0}})+'\\n')\n";
    }
    setenv("CLOUDCROPPER_CONFIG_DIR", dir.c_str(), 1);
    setenv("CLOUDCROPPER_BUFFERX_SCRIPT", script.c_str(), 1);

    const cc::PointCloud target = regBlobsCloud();
    cc::PointCloud       source = target;  // identity-aligned: GICP refine converges

    // (1) Plain BUFFER-X: worker result -> RegResult mapping.
    {
        cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::BufferX);
        opt.algo               = cc::reg::RegAlgo::BufferX;
        auto rr                = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (!rr) {
            std::cerr << "  error: " << rr.error().message << "\n";
        } else {
            CHECK(rr->converged);
            CHECK(rr->confidence < 0.0);    // BUFFER-X does not provide it
            CHECK(rr->normResidual < 0.0);
            CHECK(rr->transform == cc::reg::kIdentity4);
            CHECK(rr->detail.find("BUFFER-X (worker, fake)") == 0);
            CHECK(rr->detail.find("4321 inliers") != std::string::npos);
            CHECK(rr->detail.find("fitness 0.91") != std::string::npos);
        }
    }
    // (2) BufferXGicp: the coarse line is prepended before the GICP refine line.
    {
        cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::BufferXGicp);
        opt.algo               = cc::reg::RegAlgo::BufferXGicp;
        opt.refine             = true;
        auto rr                = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (!rr) {
            std::cerr << "  error: " << rr.error().message << "\n";
        } else {
            CHECK(rr->converged);  // from the chained GICP
            CHECK(rr->detail.find("BUFFER-X (worker, fake)") == 0);
            CHECK(rr->detail.find("  ->  ") != std::string::npos);
        }
    }
    // (3) No-weights fallback honesty: the worker returns identity + converged:
    // false with a note and NO num_inliers/fitness. The bridge must NOT fabricate
    // quality numbers, must surface the note, and must report not-converged.
    // (bufferxVoxel>0 also exercises the voxel_size override path.)
    {
        cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::BufferX);
        opt.algo               = cc::reg::RegAlgo::BufferX;
        opt.bufferxVoxel       = 0.5f;  // triggers the fake worker's fallback shape
        auto rr                = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (rr) {
            CHECK(!rr->converged);
            CHECK(rr->detail.find("? inliers") != std::string::npos);
            CHECK(rr->detail.find("fitness") == std::string::npos);  // never fabricated
            CHECK(rr->detail.find("[weights missing]") != std::string::npos);
        }
    }

    unsetenv("CLOUDCROPPER_CONFIG_DIR");
    unsetenv("CLOUDCROPPER_BUFFERX_SCRIPT");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

// End-to-end through rap::run with a FAKE worker (no torch/flash-attn needed):
// verifies the generic yaml->JSON forwarding (a sentinel n_generations from the
// temp config must arrive typed), that `refine` is forced false to the worker
// (the GICP refine is chained in C++), the row-major target<-source transform
// round-trips, the RapGicp -> GICP chaining (coarse line prepended), and the
// identity-fallback HONESTY (note surfaced, no fabricated inliers, not
// converged). Must be the process's FIRST rap call: the real worker + its spawn
// options are static and bind to this temp config.
void testRapWorkerResult() {
    std::cerr << "[rap worker result]\n";
    if (std::system("command -v python3 >/dev/null 2>&1") != 0) {
        std::cerr << "  SKIP: python3 not on PATH\n";
        return;
    }
    namespace fs       = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "cc_rap_worker_cfg";
    fs::create_directories(dir);
    {
        std::ofstream f(dir / "rap.yaml");
        f << "python: python3\nn_generations: 3\ntimeout_sec: 30\n";
    }
    const fs::path script = dir / "fake_rap_worker.py";
    {
        std::ofstream f(script);
        f << "import json,os,sys\n"
             "proto=os.fdopen(os.dup(1),'w',buffering=1)\n"
             "os.dup2(2,1)\n"
             "proto.write(json.dumps({'event':'loading','pid':os.getpid()})+'\\n')\n"
             "proto.write(json.dumps({'event':'ready','device':'fake','rap':0})+'\\n')\n"
             "for line in sys.stdin:\n"
             "    r=json.loads(line); i=r.get('id'); op=r.get('op')\n"
             "    if op=='shutdown':\n"
             "        proto.write(json.dumps({'id':i,'ok':True,'result':{}})+'\\n'); sys.exit(0)\n"
             "    bad=[]\n"
             "    if r.get('n_generations')!=3: bad.append('n_generations')\n"
             "    if r.get('refine') is not False: bad.append('refine')\n"
             "    if 'voxel_size' not in r: bad.append('voxel_size')\n"
             "    if 'cuda' not in r.get('target_key',''): bad.append('target_key')\n"
             "    if not (os.path.exists(r.get('source','')) and "
             "os.path.exists(r.get('target',''))): bad.append('npz')\n"
             "    if bad:\n"
             "        proto.write(json.dumps({'id':i,'ok':False,'error':{'type':"
             "'AssertionError','message':'bad request: '+','.join(bad)}})+'\\n')\n"
             "    elif float(r.get('voxel_size',0) or 0)>0:\n"
             "        proto.write(json.dumps({'id':i,'ok':True,'result':{'transform':"
             "[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],'converged':0,'device':'fake',"
             "'seconds':0.0,'cache_hit':0,'note':'flash-attn missing'}})+'\\n')\n"
             "    else:\n"
             "        proto.write(json.dumps({'id':i,'ok':True,'result':{'transform':"
             "[1,0,0,5,0,1,0,0,0,0,1,0,0,0,0,1],'converged':1,'num_inliers':2468,"
             "'fitness':0.88,'device':'fake','seconds':0.1,'cache_hit':0}})+'\\n')\n";
    }
    setenv("CLOUDCROPPER_CONFIG_DIR", dir.c_str(), 1);
    setenv("CLOUDCROPPER_RAP_SCRIPT", script.c_str(), 1);

    const cc::PointCloud target = regBlobsCloud();
    cc::PointCloud       source = target;  // identity-aligned: GICP refine converges

    // (1) Plain RAP: worker result -> RegResult mapping. The fake worker returns
    // a row-major target<-source 4x4 with a +5 X translation, which must land
    // verbatim on RegResult::transform (index 3).
    {
        cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::Rap);
        opt.algo               = cc::reg::RegAlgo::Rap;
        auto rr                = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (!rr) {
            std::cerr << "  error: " << rr.error().message << "\n";
        } else {
            CHECK(rr->converged);
            CHECK(rr->confidence < 0.0);    // RAP does not provide it
            CHECK(rr->normResidual < 0.0);
            CHECK(std::fabs(rr->transform[3] - 5.0) < 1e-9);  // row-major target<-source
            CHECK(rr->detail.find("RAP (worker, fake)") == 0);
            CHECK(rr->detail.find("2468 inliers") != std::string::npos);
            CHECK(rr->detail.find("fitness 0.88") != std::string::npos);
        }
    }
    // (2) RapGicp: the coarse line is prepended before the GICP refine line.
    {
        cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::RapGicp);
        opt.algo               = cc::reg::RegAlgo::RapGicp;
        opt.refine             = true;
        auto rr                = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (!rr) {
            std::cerr << "  error: " << rr.error().message << "\n";
        } else {
            CHECK(rr->converged);  // from the chained GICP
            CHECK(rr->detail.find("RAP (worker, fake)") == 0);
            CHECK(rr->detail.find("  ->  ") != std::string::npos);
        }
    }
    // (3) No-flash-attn fallback honesty: the worker returns identity +
    // converged:false with a note and NO num_inliers/fitness. The bridge must
    // NOT fabricate quality numbers, must surface the note, and must report
    // not-converged. (rapVoxel>0 also exercises the voxel_size override path.)
    {
        cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::Rap);
        opt.algo               = cc::reg::RegAlgo::Rap;
        opt.rapVoxel           = 0.5f;  // triggers the fake worker's fallback shape
        auto rr                = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (rr) {
            CHECK(!rr->converged);
            CHECK(rr->detail.find("? inliers") != std::string::npos);
            CHECK(rr->detail.find("fitness") == std::string::npos);  // never fabricated
            CHECK(rr->detail.find("[flash-attn missing]") != std::string::npos);
        }
    }

    unsetenv("CLOUDCROPPER_CONFIG_DIR");
    unsetenv("CLOUDCROPPER_RAP_SCRIPT");
    std::error_code ec;
    fs::remove_all(dir, ec);
}
#endif

#if defined(CLOUDCROPPER_HAS_KISS_MATCHER)
void testRegistrationKiss() {
    std::cerr << "[registration kiss-matcher]\n";
    const cc::PointCloud target = regBlobsCloud();
    cc::PointCloud       source = target;
    const auto           Tgt    = bigT();
    cc::reg::applyTransform(source, Tgt);

    cc::reg::RegOptions opt;
    opt.algo = cc::reg::RegAlgo::KissGicp;  // global coarse + GICP refine
    auto rr  = cc::reg::registerClouds(source, target, opt);
    CHECK(static_cast<bool>(rr));
    if (!rr) return;
    CHECK(rr->converged);
    CHECK(composedOffset(rr->transform, Tgt) < 0.05);  // 90deg + offset recovered
    CHECK(rr->rmse < 0.01);
}
#endif

#if defined(CLOUDCROPPER_HAS_PCD)
// End-to-end through g3reg::run with a FAKE one-shot CLI (no PCL/GTSAM/igraph
// needed): a tiny python3 script stands in for cc_g3reg_cli and prints the three
// G3REG_* protocol lines on stdout (plus glog-style noise on stderr). Verifies
// that the backend (1) writes the src/tgt .pcd handoff files, (2) passes the
// external config path as the first argument, (3) keeps stderr noise out of the
// stdout parse, (4) maps the row-major TF onto RegResult::transform, (5) carries
// inliers/time into `detail` while leaving confidence/normResidual = -1, (6)
// chains GICP for G3RegGicp (coarse line prepended), and (7) fails cleanly when
// the binary is absent.
void testG3regCliResult() {
    std::cerr << "[g3reg cli result]\n";
    if (std::system("command -v python3 >/dev/null 2>&1") != 0) {
        std::cerr << "  SKIP: python3 not on PATH\n";
        return;
    }
    namespace fs       = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "cc_g3reg_cli_cfg";
    fs::create_directories(dir);

    // Thin CloudCropper-side layer. g3reg_config is a dummy path the fake CLI
    // only echoes back / checks; it is never parsed by CloudCropper.
    setenv("CLOUDCROPPER_CONFIG_DIR", dir.c_str(), 1);
    unsetenv("CLOUDCROPPER_G3REG_CONFIG");  // force the yaml key path
    {
        std::ofstream f(dir / "g3reg.yaml");
        f << "timeout_sec: 30\nrefine: true\ng3reg_config: " << (dir / "fake.yaml").string()
          << "\n";
    }
    { std::ofstream f(dir / "fake.yaml"); f << "# dummy external g3reg config\n"; }

    // Fake cc_g3reg_cli: argv = [config, src.pcd, tgt.pcd]. Asserts the handoff
    // files exist + the config path is ours, emits glog noise on stderr (must NOT
    // reach stdout), then prints the 3 protocol lines on stdout.
    const fs::path bin = dir / "fake_g3reg_cli.py";
    {
        std::ofstream f(bin);
        f << "#!/usr/bin/env python3\n"
             "import sys,os\n"
             "cfg,src,tgt = sys.argv[1], sys.argv[2], sys.argv[3]\n"
             "sys.stderr.write('I0616 glog noise on stderr\\n')\n"
             "assert os.path.exists(src) and os.path.exists(tgt), 'pcd missing'\n"
             "assert open(src,'rb').read(6)==b'# .PCD', 'not a pcd'\n"
             "assert cfg.endswith('fake.yaml'), 'config not forwarded'\n"
             "print('G3REG_TF: 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1')\n"
             "print('G3REG_INLIERS: 1234')\n"
             "print('G3REG_TIME: 0.42')\n";
    }
    fs::permissions(bin, fs::perms::owner_all, fs::perm_options::add);
    setenv("CLOUDCROPPER_G3REG_BIN", bin.c_str(), 1);

    const cc::PointCloud target = regBlobsCloud();
    cc::PointCloud       source = target;  // identity-aligned: GICP refine converges

    // (1) G3Reg alone: 3-line parse -> RegResult mapping.
    {
        cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::G3Reg);
        opt.algo                = cc::reg::RegAlgo::G3Reg;
        auto rr                 = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (!rr) {
            std::cerr << "  error: " << rr.error().message << "\n";
        } else {
            CHECK(rr->converged);
            CHECK(rr->confidence < 0.0);    // G3Reg does not provide it
            CHECK(rr->normResidual < 0.0);
            CHECK(rr->transform == cc::reg::kIdentity4);
            CHECK(rr->detail.find("G3Reg: 1234 inliers") == 0);
        }
    }
    // (2) G3RegGicp: the coarse line is prepended before the GICP refine line.
    {
        cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::G3RegGicp);
        opt.algo                = cc::reg::RegAlgo::G3RegGicp;
        opt.refine              = true;
        auto rr                 = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (!rr) {
            std::cerr << "  error: " << rr.error().message << "\n";
        } else {
            CHECK(rr->converged);  // from the chained GICP
            CHECK(rr->detail.find("G3Reg: 1234 inliers") == 0);
            CHECK(rr->detail.find("  ->  ") != std::string::npos);
        }
    }
    // (3) Missing binary -> clean failure (NotFound), no fabricated result.
    {
        setenv("CLOUDCROPPER_G3REG_BIN", (dir / "does_not_exist").c_str(), 1);
        cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::G3Reg);
        opt.algo                = cc::reg::RegAlgo::G3Reg;
        auto rr                 = cc::reg::registerClouds(source, target, opt);
        CHECK(!rr);
    }

    unsetenv("CLOUDCROPPER_CONFIG_DIR");
    unsetenv("CLOUDCROPPER_G3REG_BIN");
    std::error_code ec;
    fs::remove_all(dir, ec);
}
#endif
#endif

#if defined(CLOUDCROPPER_HAS_NPZ)
void testTemplateNpz() {
    std::cerr << "[npz template round-trip]\n";
    cc::PointCloud pc;
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y)
            pc.positions().push_back({static_cast<float>(x) * 0.1f, static_cast<float>(y) * 0.1f, 0.0f});
    cc::estimateNormals(pc, {});  // adds a "normal" column

    cc::io::TemplateMeta tm;
    const cc::Aabb       b = pc.bounds();
    tm.bbox_min            = b.min;
    tm.bbox_max            = b.max;
    tm.bbox_center         = (b.min + b.max) * 0.5f;
    const cc::CanonicalFrame cf = cc::pcaFrame(pc);
    tm.canonical_center         = cf.center;
    tm.canonical_axis           = cf.axis;
    tm.point_spacing_m          = cc::estimatePointSpacing(pc);
    tm.tip_local                = cc::Vec3{1, 2, 3};
    tm.object_pose_origin_local = cc::Vec3{0.1f, 0.2f, 0.3f};
    tm.object_pose_dir_local    = cc::Vec3{0.0f, 0.0f, 1.0f};

    cc::io::MemoryByteSink sink;
    CHECK(static_cast<bool>(cc::io::writeTemplateNpz(pc, tm, sink)));

    cc::io::MemoryByteSource src(sink.data());
    cc::io::NpzReader        r;
    auto                     rd = r.read(src, {});
    CHECK(static_cast<bool>(rd));
    if (!rd) return;
    const cc::PointCloud& out = rd.value();
    CHECK(out.size() == pc.size());                 // surface_points -> positions
    CHECK(out.find(cc::attr::kNormal) != nullptr);  // surface_normals -> normal column
    CHECK(out.metadata().count("bbox_min") == 1);
    CHECK(out.metadata().count("canonical_center") == 1);
    CHECK(out.metadata().count("point_spacing_m") == 1);
    CHECK(out.metadata().count("tailstock_tip_local") == 1);
    CHECK(out.metadata().count("object_pose_origin_local") == 1);
    CHECK(out.metadata().count("object_pose_dir_local") == 1);
    CHECK(out.metadata().count("units") == 1 && out.metadata().at("units") == "m");

    cc::io::MemoryByteSink genericSink;
    cc::io::NpzWriter      genericWriter;
    CHECK(static_cast<bool>(genericWriter.write(out, genericSink, {})));
    cc::io::MemoryByteSource genericSrc(genericSink.data());
    auto                     genericRd = r.read(genericSrc, {});
    CHECK(static_cast<bool>(genericRd));
    if (genericRd) {
        CHECK(genericRd->metadata().count("object_pose_origin_local") == 1);
        CHECK(genericRd->metadata().count("object_pose_dir_local") == 1);
    }
}
#endif

}  // namespace

int main() {
    testPlyRoundTrip(true);
    testPlyRoundTrip(false);
    testCropAabb();
    testCropRecenter();
    testCropObbExclude();
    testGatherAlignment();
    testNormalsPlane();
    testNormalsSphere();
    testDenoise();
    testCropOctreeParity();
    testPcaFrame();
#if defined(CLOUDCROPPER_HAS_GZIP)
    testGzip();
#endif
#if defined(CLOUDCROPPER_HAS_NPZ)
    testTemplateNpz();
#endif
#if defined(CLOUDCROPPER_HAS_REGISTRATION)
    testApplyTransformPoseMetadata();
    testRegistrationGicp();
    testRegConfigDefaults();
    testJsonParse();
    testPythonWorkerProtocol();
#if defined(CLOUDCROPPER_HAS_NPZ)
    testGsdfGpuWorkerResult();
    testBufferxWorkerResult();
    testRapWorkerResult();
#endif
#if defined(CLOUDCROPPER_HAS_PCD)
    testG3regCliResult();
#endif
#if defined(CLOUDCROPPER_HAS_KISS_MATCHER)
    testRegistrationKiss();
#endif
#endif

#if defined(CLOUDCROPPER_HAS_PCD)
    roundTripVia("pcd ascii", cc::io::PcdWriter{}, cc::io::PcdReader{}, cc::io::Encoding::Ascii);
    roundTripVia("pcd binary", cc::io::PcdWriter{}, cc::io::PcdReader{}, cc::io::Encoding::Binary);
#if defined(CLOUDCROPPER_HAS_LZF)
    roundTripVia("pcd binary_compressed", cc::io::PcdWriter{}, cc::io::PcdReader{},
                 cc::io::Encoding::BinaryCompressed);
#endif
#endif
#if defined(CLOUDCROPPER_HAS_NPZ)
    roundTripVia("npz", cc::io::NpzWriter{}, cc::io::NpzReader{}, cc::io::Encoding::Binary);
#endif

    if (g_failures == 0) {
        std::cerr << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cerr << g_failures << " CHECK(S) FAILED\n";
    return 1;
}
