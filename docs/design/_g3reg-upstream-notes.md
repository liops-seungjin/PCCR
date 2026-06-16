# G3Reg — Upstream Reconnaissance Notes

> Purpose: assess whether **G3Reg** (HKUST-Aerial-Robotics) is realistically integrable as a
> CloudCropper (PCCR) native C++ registration backend, and what it would cost.
>
> Method: web reconnaissance only (no shell / no clone). All claims sourced below; items I
> could **not** fully verify are explicitly flagged `⚠ UNVERIFIED`.
>
> Repo: https://github.com/HKUST-Aerial-Robotics/G3Reg
> Paper: G3Reg: Pyramid Graph-based Global Registration using Gaussian Ellipsoid Model,
> IEEE T-ASE 2024 — https://arxiv.org/abs/2308.11573

---

## TL;DR verdict

G3Reg is **NOT a ROS package** — it is a plain **CMake static library** (`add_library(g3reg STATIC ...)`)
with a clean C++ entry point:

```cpp
g3reg::FRGresult GlobalRegistration(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &src_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &tgt_cloud,
    std::tuple<int,int,int> pair_info = {0,0,0});   // returns .tf as Eigen::Matrix4d
```

So "no ROS" is a real win. **BUT** the dependency footprint is heavy and version-pinned:
**PCL 1.10 (must be < 1.11), GTSAM 4.1.1, Eigen 3.3.7 (< 3.4.0), glog, Boost, yaml-cpp, TBB,
OpenMP, optionally igraph 0.9.9.** These are pinned to *old* versions because the code relies on
`boost::make_shared` (broken in PCL ≥ 1.11) and GTSAM 4.2 ↔ Eigen 3.4 incompatibility.

**Blunt recommendation:** A **native static-lib link is technically possible** (the API is clean,
no ROS, learning-free → no model weights). The real risk is **dependency hell**: forcing PCL 1.10 +
GTSAM 4.1.1 + Eigen 3.3.7 into CloudCropper's existing toolchain, plus it requires a YAML config file
at runtime. For a first integration, a **subprocess/CLI wrapper around the `demo_reg` executable**
(PCD in → 4×4 transform out) is the lower-risk path; promote to native linking later if the dep
versions can be reconciled. See §8.

---

## 1. Dependencies (exact + versions)

From `docs/install.md` (manual install section):

- **Ubuntu packages:** `sudo apt install libboost-dev libyaml-cpp-dev libomp-dev libtbb-dev`
  → **Boost, yaml-cpp, OpenMP (libomp), Intel TBB.**
- **GTSAM 4.1.1** — *"GTSAM-4.2 is not compatible with Eigen-3.4.0."*
- **PCL 1.10** — *"PCL-1.11 and later versions remove support for `boost::make_shared` in favor of
  `std::make_shared`, which is not compatible with this project. Use a version below 1.11, such as
  PCL-1.10."*
- **Eigen < 3.4.0** (e.g. **3.3.7**) — tied to the GTSAM 4.1.1 constraint above.
- **GLOG** (glog) — install latest from source.
- **igraph 0.9.9** — only for "3DMAC" support; needs `sudo apt-get install flex bison`. ⚠ Appears
  optional (a specific solver/feature); core PAGOR path may not need it — `cmake/igraph.cmake` exists,
  but I could not confirm whether the build hard-fails without igraph. `⚠ UNVERIFIED`
- For PCL visualization: install **VTK and Qt** before PCL (only needed for the demo viewer, not the lib).

`CMakeLists.txt` pulls each dep via its own module under `cmake/`:
`boost.cmake, eigen.cmake, yaml.cmake, glog.cmake, pcl.cmake, openmp.cmake, gtsam.cmake, igraph.cmake`.

**No Ceres, no gflags** observed (glog is used; gflags not mentioned). `⚠ Ceres/gflags absence not
100% confirmed` — they are not in install.md nor the cmake module list, so almost certainly not used.

Source: https://github.com/HKUST-Aerial-Robotics/G3Reg/blob/main/docs/install.md ,
`CMakeLists.txt` (raw).

## 2. Build system

- **Plain CMake**, no catkin / no ROS. Build:
  ```
  mkdir build && cd build
  cmake ..
  make -j4
  ```
- **Docker provided** (`Dockerfile` in repo): `docker build -t g3reg .` then `docker run` with volume
  mount + X11 forwarding for the GUI demo.
- `CMakeLists.txt` declares: `add_library(${PROJECT_NAME} STATIC ${ALL_SRCS})` (static lib `g3reg`)
  plus executables `reg_bm`, `matching_bm`, `demo_reg`, `demo_seg` from `examples/`.
- **No catkin/ROS references** in `CMakeLists.txt` or install docs → **ROS-free build confirmed.**

Source: install.md, CMakeLists.txt (raw), docs/demo.md.

## 3. Public API (C++, no ROS)

Header: `include/back_end/reglib.h`. Two entry points:

```cpp
// Raw-cloud → transform (self-contained: does its own front-end)
FRGresult GlobalRegistration(const pcl::PointCloud<pcl::PointXYZ>::Ptr &src_cloud,
                             const pcl::PointCloud<pcl::PointXYZ>::Ptr &tgt_cloud,
                             std::tuple<int,int,int> pair_info = std::make_tuple(0,0,0));

// Correspondence-based (you supply matches)
FRGresult SolveFromCorresp(const Eigen::MatrixX3d &src_corresp,
                           const Eigen::MatrixX3d &tgt_corresp,
                           const Eigen::MatrixX3d &src_cloud,
                           const Eigen::MatrixX3d &tgt_cloud,
                           const Config &config_custom = config);
```

`FRGresult` (from `include/utils/evaluation.h`):
```cpp
class FRGresult {
    Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();   // <-- the 4x4 result
    std::vector<Eigen::Matrix4d> candidates;            // PAGOR multi-resolution candidates
    int plane_inliers, line_inliers, cluster_inliers;
    clique_solver::Association inliers;
    double feature_time, tf_solver_time, clique_time, graph_time, verify_time, total_time;
    bool valid = true;
};
```

**Input format:** PCL `pcl::PointCloud<pcl::PointXYZ>::Ptr` — **xyz only, no normals, no intensity**.
The demo loads from `.pcd` (`pcl::io::loadPCDFile<pcl::PointXYZ>`). `.ply` not used by the demo, but
any cloud you can get into a PCL `PointXYZ` cloud works.

**Demo entry point** `examples/demo_reg.cpp` (verbatim core):
```cpp
config.load_config(config_path, argv);                 // <-- YAML config REQUIRED
pcl::io::loadPCDFile<pcl::PointXYZ>(argv[2], *source);
pcl::io::loadPCDFile<pcl::PointXYZ>(argv[3], *target);
FRGresult solution = g3reg::GlobalRegistration(source, target);
Eigen::Matrix4d tf = solution.tf;
```
Run as: `./bin/demo_reg configs/<dataset>/gem_pagor.yaml source.pcd target.pcd`

**Important caveat:** the entry point reads a global `config` object that must be loaded from a YAML
file (`config.load_config(...)`) before calling `GlobalRegistration`. So it is *not* a zero-config
one-liner — a tuned `.yaml` (voxel sizes, segmentation params, solver settings) is needed per
sensor/scene. `⚠` Whether sane built-in defaults work without `load_config` is `UNVERIFIED`.

Sources: reglib.h (raw), evaluation.h (raw), examples/demo_reg.cpp (raw), docs/demo.md.

## 4. Input assumptions

- Designed for **outdoor LiDAR** global registration (paper abstract:
  *"global registration of outdoor LiDAR point clouds"*).
- Extracts geometric primitives — **planes, clusters, lines** — from the raw cloud, each modeled as a
  **Gaussian Ellipsoid Model (GEM)**. So performance depends on the scene having extractable
  planar/linear/cluster structure; segmentation params are in the per-dataset YAML.
- **xyz only** (PointXYZ) — no normals required, no intensity required.
- Sensor-specific configs ship under `configs/` — e.g. `configs/apollo_lc_bm/gem_pagor.yaml`
  (Velodyne) and **`configs/hit_ms/gem_pagor.yaml` for a Livox demo**
  (`examples/data/livox/source.pcd`). Livox = dense, near-range solid-state LiDAR → suggests it is
  **not strictly limited to sparse spinning LiDAR**, and there is a config tuned for denser/near-range
  data. Relevant for CloudCropper if its clouds are dense/near-range.
- Docs explicitly say: *"If the registration does not perform as expected on your point cloud, it is
  advisable to review the segmentation results"* → tuning the front-end voxel/segmentation params is
  expected per dataset.

Sources: arXiv abstract (2308.11573), docs/demo.md, docs/install.md.

## 5. Front-end (self-contained?)

**Yes — self-contained.** `GlobalRegistration(src, tgt)` runs the full pipeline internally:
primitive extraction (planes/clusters/lines → GEM) → Pyramid Compatibility Graph → distrust-and-verify
(PAGOR) → transform. No externally-supplied descriptors/correspondences are needed for the main path.
(`demo_seg` exists to debug just the segmentation stage.)

An **alternative correspondence-based** entry (`SolveFromCorresp`, Eigen matrices) exists if you want
to feed your own matches and use only the robust back-end solver.

Source: reglib.h, demo_reg.cpp, demo.md, arXiv abstract.

## 6. License & model weights

- **License: MIT** (per repo). ✅ Permissive — safe to vendor/link.
- **Learning-free** → **no model weights / no checkpoints needed.** Confirmed by design (geometric +
  graph-theoretic, no neural network). Only runtime artifact required is the **YAML config**, not weights.

Source: repo footer (MIT), arXiv abstract, install/demo docs (no weights mentioned anywhere).

## 7. Known build pain / forks

Closed GitHub issues indicate the main friction is **dependency-version compilation**, not ROS:
- #18 "Compilation Errors with PCL"
- #17 "I reinstalled gtsam and compiled successfully, but the command failed."
- #15 "Compilation error due to the implementation of a missing virtual function"
- #4  "Compilation problems"
- #19 "Installation Instructions Update Suggestions"
- Runtime crashes also reported: #11 "Got Segmentation Fault", #20 "double free or corruption",
  #10 / #6 demo runtime errors.

So the standalone (non-ROS) build is the *intended* build, but getting **PCL 1.10 + GTSAM 4.1.1 +
Eigen 3.3.7** lined up is the recurring headache. **Docker is the upstream-blessed escape hatch.**
No widely-known "easier" maintained fork surfaced in this recon. `⚠ Fork landscape UNVERIFIED`.
Related sibling repos from same lab: **Pagor** (IROS 2023) and **LiDAR-Registration-Benchmark**.

Source: https://github.com/HKUST-Aerial-Robotics/G3Reg/issues

---

## 8. Integration verdict for CloudCropper

**Feasible without ROS? — YES.** It's a CMake static lib with a clean
`GlobalRegistration(src, tgt) → Eigen::Matrix4d` API, MIT-licensed, no model weights.

**Realistic to link natively right now? — Risky.** The blocker is the **pinned, dated dependency
stack** (PCL <1.11, GTSAM 4.1.1, Eigen <3.4). If CloudCropper already uses a newer PCL/Eigen, ABI/version
conflicts are likely, and GTSAM 4.1.1 is a heavy add. Plus a per-scene YAML config is required at runtime.

**Recommended path:**
1. **Phase 1 (low risk): subprocess/CLI wrapper.** Build G3Reg in its own Docker image (upstream
   Dockerfile) or isolated toolchain, exposing the `demo_reg`-style flow: PCD source + PCD target +
   YAML → write 4×4 transform to stdout/file. CloudCropper shells out and parses the transform.
   Decouples the dependency stack entirely. Fastest to a working PCCR backend.
2. **Phase 2 (optional, higher effort): native static link.** Only if (a) we can pin PCL 1.10 /
   GTSAM 4.1.1 / Eigen 3.3.7 in CloudCropper's build, or vendor them isolated; (b) we wire a YAML
   config (start from `configs/hit_ms/gem_pagor.yaml` if our data is dense/near-range Livox-like, or
   `apollo_lc_bm` for spinning LiDAR) and tune segmentation params. Then call `GlobalRegistration`
   directly — no IPC, faster, but owns the dependency conflict risk.

**Open items to verify before committing (currently `⚠ UNVERIFIED`):**
- Does `GlobalRegistration` work with built-in defaults if `config.load_config` is skipped? (Likely no.)
- Is **igraph** mandatory or only for the 3DMAC solver variant?
- Confirm **no Ceres / gflags** (strongly implied absent).
- Behavior/quality on **dense near-range** clouds (CloudCropper's likely domain) vs the outdoor-LiDAR
  design target — needs an empirical test on real CloudCropper data.

---

### Sources
- Repo: https://github.com/HKUST-Aerial-Robotics/G3Reg
- Install: https://github.com/HKUST-Aerial-Robotics/G3Reg/blob/main/docs/install.md
- Demo: https://github.com/HKUST-Aerial-Robotics/G3Reg/blob/main/docs/demo.md
- API headers (raw): `include/back_end/reglib.h`, `include/utils/evaluation.h`, `examples/demo_reg.cpp`
- Issues: https://github.com/HKUST-Aerial-Robotics/G3Reg/issues
- Paper: https://arxiv.org/abs/2308.11573 (IEEE T-ASE 2024, IEEE Xplore doc 10518010)
