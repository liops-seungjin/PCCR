# G3Reg 통합 계약서 (main agent가 /tmp/G3Reg 직접 조사 → 팀 공유용)

> subagent는 `/tmp/G3Reg`(프로젝트 밖)에 접근할 수 없으므로, 실제 업스트림에서 확인한
> 사실과 통합 전략을 여기에 고정한다. 웹 보강은 `_g3reg-upstream-notes.md` 참조.

## 1. G3Reg 실측 사실 (clone: HKUST-Aerial-Robotics/G3Reg)

- **공개 C++ API** (`include/back_end/reglib.h`):
  ```cpp
  namespace g3reg {
    FRGresult GlobalRegistration(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &src_cloud,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &tgt_cloud,
        std::tuple<int,int,int> pair_info = {0,0,0});
  }
  ```
- **FRGresult** (`include/utils/evaluation.h`): `Eigen::Matrix4d tf` (= **target<-source**,
  데모가 `pcl::transformPointCloud(src, out, tf)`로 src를 tgt에 정렬), `plane_inliers`,
  `line_inliers`, `cluster_inliers`, `double time`(+ feature/tf_solver/clique/graph time).
- **입력**: raw `pcl::PointXYZ` 클라우드(xyz만, 노멀 불필요). 자체 primitive(plane/line/
  cluster) 추출 → Gaussian Ellipsoid + Pyramid Compatibility Graph → max-clique. **외부
  correspondence/descriptor 불필요(완결형)**. config yaml(`configs/`)로 voxel/threshold 설정.
- **빌드 의존성**: PCL(✅시스템 1.12), Eigen(✅), yaml-cpp(✅), glog(✅), OpenMP(✅), Boost(✅1.74),
  **GTSAM 4.1.1**(소스 빌드 → `/tmp/g3reg-deps`), **igraph 0.9.9**(소스 빌드 → `/tmp/g3reg-deps`).
  Thirdparty(clique_solver/fpfh_matcher/robot_utils/backward-cpp)는 저장소 내 번들. 학습 없음 →
  가중치 0. License: 저장소 LICENSE 확인(MIT 계열로 보임 — 확정은 recon).

## 2. 통합 전략 — subprocess 래퍼 (우리 빌드에 의존성 0 추가)

G3Reg를 CloudCropper에 **링크하지 않는다**(GTSAM/igraph/PCL이 빌드에 딸려와 충돌·비대화).
대신:

1. **main agent가** G3Reg를 `/tmp/g3reg-deps`(CMAKE_PREFIX_PATH)로 독립 빌드하고, G3Reg의
   `examples/`에 **파싱 가능한 CLI** `cc_g3reg_cli.cpp`를 추가해 빌드한다:
   ```
   cc_g3reg_cli <config.yaml> <src.pcd> <tgt.pcd>
   stdout:  G3REG_TF: m00 m01 ... m33   (row-major 16)
            G3REG_INLIERS: <plane+line+cluster>
            G3REG_TIME: <seconds>
   ```
   (glog는 stderr로 보내고 stdout엔 위 3줄만 — 파싱 안정성.)
2. **CloudCropper `g3reg` 백엔드**(dev팀 구현, 프로세스 분리 철학은 gsdf/bufferx와 동일하나
   **persistent worker가 아니라 one-shot subprocess**):
   - `RegAlgo::G3Reg` (+ 선택적 `RegAlgo::G3RegGicp` = G3Reg→GICP 미세정합 체인, kiss_backend.cpp:61-70 패턴).
   - src/tgt를 임시 `.pcd`로 기록: CloudCropper의 **자체 PCD writer**(`src/io/pcd.cpp`, `io::registry`/`NpzWriter`처럼) 사용 — PCL 불필요.
   - CLI 바이너리 위치: 환경변수 `CLOUDCROPPER_G3REG_BIN`(+ `gsdf_gpu.cpp:findScript` 류의 상대경로 탐색). config 경로는 `CLOUDCROPPER_G3REG_CONFIG` 또는 `config/g3reg.yaml`의 키.
   - 호출: fork/exec 또는 `popen`으로 stdout 캡처(persistent PythonWorker는 과함 — one-shot). stdout 3줄 파싱 → `RegResult`(tf→transform, inliers 합→`detail` 및 `inliers`는 공통 metric이 재계산, confidence/normResidual = -1 유지).
   - `G3RegGicp`이면 C++에서 GICP refine 체인(small_gicp, BUFFER-X와 동일 방식).
   - 빌드 게이트: 순수 C++ subprocess라 **빌드타임 의존성 0**. 바이너리·config 부재 시 `ErrorCode`로 깔끔히 실패(gsdf 철학과 동일).

## 3. 변경 파일 체크리스트 (dev팀)

| 파일 | 변경 |
|---|---|
| `registration.hpp` | `RegAlgo::G3Reg`(+ `G3RegGicp`) enum, 필요한 옵션 필드(예: `g3regConfig` 경로는 yaml로 충분하면 생략) |
| `common/registration.cpp` | `algoName` + dispatcher case → `g3reg::run` |
| `common/config.cpp` (+ config.hpp) | `configFileFor`/`defaultsFor`에 `g3reg.yaml` |
| `backend/registration/g3reg/g3reg_backend.{hpp,cpp}` | 신규: PCD 기록 + subprocess 호출 + stdout 파싱 + (G3RegGicp) GICP 체인 |
| `backend/registration/CMakeLists.txt` | 소스 1줄 추가 |
| `config/g3reg.yaml` | binary 경로/탐색, config 경로, refine 등 |
| `src/app/main.cpp` | `--reg-algo g3reg\|g3reg-gicp`, usage |
| `src/viewer/viewer.cpp` | 콤보/패널 |
| `tests/cc_tests.cpp` | **fake CLI**(stdout 3줄 출력하는 stdlib 스크립트)로 PCD 기록·파싱·매핑·G3RegGicp 체인 검증(GTSAM/igraph 불필요) |

## 4. 검증 (main agent)

- GTSAM/igraph/G3Reg 빌드 성공 후 `cc_g3reg_cli`를 hap101_f0.pcd→crop.pcd에 직접 실행(저밀도 가정이라
  config voxel 튜닝 필요할 수 있음).
- CloudCropper `register --reg-algo g3reg[-gicp]`로 end-to-end.
- `scripts/inlier-rmse.py`로 mm급 + KISS-Matcher와 **A/B**(성공률·inlier·시간·중첩부 RMSE).

## 5. 리스크
- GTSAM 4.1.1 / igraph 0.9.9가 옛 버전이라 GCC 11.4에서 빌드 실패 가능(진행 중).
- G3Reg는 **옥외 LiDAR·저밀도 가정** → 근거리 고밀도 hap/hesai에서 primitive 추출 파라미터(config의 voxel/plane threshold) 튜닝이 필요할 수 있음(검증 단계에서 조정).
- PCD 입력 단위/스케일은 미터 그대로.
