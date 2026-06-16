# G3Reg 외부 CLI 빌드 런북 (cc_g3reg_cli 재현)

> CloudCropper의 `g3reg` 백엔드는 **외부 one-shot 바이너리 `cc_g3reg_cli`** 를 subprocess로
> 부른다(우리 빌드엔 의존성 0). 그 바이너리는 G3Reg를 독립 빌드해 만든다 — 아래는 main agent가
> 2026-06-16 실제로 수행해 **검증까지 통과**한 정확한 절차다. (/tmp 산출물은 저장소에 커밋되지 않으므로
> 이 문서로 재현한다.)

## 환경(검증 시점)
Ubuntu 22.04, GCC 11.4, CMake 3.31, Boost 1.74, PCL **1.12**(시스템), Eigen 시스템, sudo 없음 → 로컬 prefix.

## 1. 의존성 빌드 → `/tmp/g3reg-deps` (로컬 prefix, 시스템 비오염)
```bash
mkdir -p /tmp/g3reg-deps
# GTSAM 4.1.1 (시스템 Eigen 사용, TBB off)
git clone --depth 1 --branch 4.1.1 https://github.com/borglab/gtsam.git /tmp/gtsam
cmake -S /tmp/gtsam -B /tmp/gtsam/build -DCMAKE_INSTALL_PREFIX=/tmp/g3reg-deps \
  -DCMAKE_BUILD_TYPE=Release -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_UNSTABLE=OFF \
  -DGTSAM_BUILD_PYTHON=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
cmake --build /tmp/gtsam/build -j && cmake --install /tmp/gtsam/build

# igraph 0.9.9 — RELEASE TARBALL (git clone는 flex/bison 필요; 타르볼은 파서 미리 생성됨)
wget https://github.com/igraph/igraph/releases/download/0.9.9/igraph-0.9.9.tar.gz -O /tmp/igraph-0.9.9.tar.gz
tar xzf /tmp/igraph-0.9.9.tar.gz -C /tmp
cmake -S /tmp/igraph-0.9.9 -B /tmp/igraph-0.9.9/build -DCMAKE_INSTALL_PREFIX=/tmp/g3reg-deps \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build /tmp/igraph-0.9.9/build -j && cmake --install /tmp/igraph-0.9.9/build
```
(PCL/glog/yaml-cpp/Boost/OpenMP는 시스템 패키지로 충족: `libpcl-dev libgoogle-glog-dev libyaml-cpp-dev libboost-all-dev`.)

## 2. G3Reg clone + **PCL 1.12 호환 패치**
```bash
git clone --depth 1 https://github.com/HKUST-Aerial-Robotics/G3Reg.git /tmp/G3Reg
cd /tmp/G3Reg
# (a) PCL 1.11+ 는 boost::shared_ptr/make_shared 제거 → std로 치환 (include/ + src/ 만)
grep -rl 'boost::shared_ptr\|boost::make_shared' include/ src/ | xargs -r \
  sed -i 's/boost::shared_ptr/std::shared_ptr/g; s/boost::make_shared/std::make_shared/g'
# (b) 누락 include 보강 (Boost 1.74 / PCL 1.12 에서 transitive include 끊김)
#   - Thirdparty/robot_utils/src/kitti_utils.cpp : #include <boost/filesystem.hpp>
#   - include/datasets/kitti_loader.h            : #include <boost/filesystem.hpp>
#   - src/back_end/pagor/geo_verify.cpp          : #include <pcl/common/transforms.h>
```
> 위 (a)(b)는 22개 boost ptr 사용처(대부분 `pcl::PointCloud`)와 3개 누락 include가 전부였다.
> GTSAM용 boost ptr와 섞이지 않아 안전(전부 PCL/자체 타입).

## 3. 파싱 가능한 CLI 추가
`examples/cc_g3reg_cli.cpp`(이 저장소 `docs/design/`에 사본 보관 권장)를 추가하고 `CMakeLists.txt`에
```cmake
add_executable(cc_g3reg_cli examples/cc_g3reg_cli.cpp ${BACKWARD_ENABLE})
add_backward(cc_g3reg_cli)
target_link_libraries(cc_g3reg_cli ${PROJECT_NAME})
```
CLI는 `g3reg::GlobalRegistration(src,tgt)`만 호출하고 stdout에 **딱 3줄**:
`G3REG_TF: <16 floats row-major target<-source>` / `G3REG_INLIERS: <n>` / `G3REG_TIME: <s>`
(glog는 `FLAGS_logtostderr=1`로 stderr로). 빌드:
```bash
cmake -S /tmp/G3Reg -B /tmp/G3Reg/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/tmp/g3reg-deps
cmake --build /tmp/G3Reg/build -j --target cc_g3reg_cli   # -> /tmp/G3Reg/bin/cc_g3reg_cli
```

## 4. 근거리(near-range) config — **객체 스케일 튜닝이 필수**
G3Reg 기본은 옥외 대형 장면용(plane resolution **1 m**)이라 sub-meter crop에선 primitive 0개 → identity.
객체 스케일로 낮춰야 동작한다(`/tmp/G3Reg/configs/cc_near.yaml` + `configs/sensors/near/`):
- `front_end: gem`, `cluster_mtd: euc`(ring 구조 가정 제거), `back_end: pagor`
- `min_range: 0.0`(crop은 원점 근처), `min_cluster_size: 8~10`
- `plane_extraction`: `resolution 1→0.05`, `distance_thresh 0.2→0.02`, `eigenvalue_thresh 30→5`

## 5. 실행(런타임 환경)
```bash
export CLOUDCROPPER_G3REG_BIN=/tmp/G3Reg/bin/cc_g3reg_cli
export CLOUDCROPPER_G3REG_CONFIG=/tmp/G3Reg/configs/cc_near.yaml
export LD_LIBRARY_PATH=/tmp/g3reg-deps/lib:$LD_LIBRARY_PATH   # libgtsam / libigraph
cloudcropper register <src.ply> <tgt.ply> --reg-algo g3reg     # 또는 g3reg-gicp
```

## 6. 검증 결과 (hap101_f0 → crop, 2026-06-16)
| | rmse(전체) | inlier RMSE | median | inliers | translation |
|---|---|---|---|---|---|
| g3reg | 3.03125 | **4.79 mm** | 4.45 mm | 10596 | [-1.983,-0.276,-0.085] |
| g3reg-gicp | 3.03049 | **1.61 mm** | 0.42 mm | 10598 | [-1.984,-0.272,-0.102] |

→ g3reg-gicp가 kiss-gicp(3.03052)·bufferx-gicp(3.03048)와 **동일 포즈로 수렴**(교차검증 통과).
학습-free 전역기로서 BUFFER-X/KISS와 동급. 전역단독 정밀도는 kiss(1.84) < g3reg(4.79) < bufferx(5.62)mm.
