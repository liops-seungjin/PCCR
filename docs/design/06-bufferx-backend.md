# 06 — BUFFER-X 글로벌 정합 백엔드 통합 설계

> 대상 저장소: `/home/sjjung/Project/cloudcropper`
> 새 백엔드: **BUFFER-X** (zero-shot point cloud registration, ICCV 2025 Highlight,
> MIT-SPARK, <https://github.com/MIT-SPARK/BUFFER-X>)
> 작성: 설계팀 / 구현은 별도. 본 문서는 **구현 지침서**이며 프로덕션 코드는 포함하지 않는다.
>
> ⚠️ 갱신(2026-06-16): §5.1의 미해결 항목 중 일부(특히 백본)는 이후 업스트림 정찰로
> 확정됐다 — BUFFER-X는 **sparse-conv 백본(MinkowskiEngine/spconv)을 쓰지 않는다**
> (pointnet2_ops / KNN_CUDA / torch-batch-svd / cpp_wrappers CUDA 확장 사용). 본문에
> 남은 sparse-conv 가정은 옛 추정이며, 정확한 사실은 `_bufferx-upstream-notes.md`와
> 실제 코드(`requirements.txt` / `VENDORED.md` / `bufferx_backend.hpp`)를 따른다.

---

## 0. 한 줄 요약

BUFFER-X는 학습 기반 **글로벌(초기 추정 불필요)** 정합기다. 기존
`RegAlgo::GradientSdfGpu`가 쓰는 **영속 Python 워커 패턴**(JSON-lines IPC +
NPZ 파일 핸드오프, 빌드타임 의존성 0)을 그대로 복제하여
`RegAlgo::BufferX` / `RegAlgo::BufferXGicp`로 추가한다. torch / sparse-conv /
사전학습 가중치는 **전부 런타임 관심사**이며, C++ 측은 얇은 브리지
(`backend/registration/bufferx/bufferx_backend.cpp`) 하나만 컴파일한다.

---

## 1. 아키텍처

### 1.1 BUFFER-X가 들어가는 위치

정합 백엔드는 `backend/registration/` 아래 **알고리즘 1개당 디렉터리 1개**
구조다 (`backend/registration/CMakeLists.txt:1-3` 주석). BUFFER-X는 학습 기반
글로벌 정합이므로 분류상 KISS-Matcher / gradient-SDF와 같은 "글로벌 초기화"
계열이다.

```
backend/registration/
├── include/cloudcropper/registration/registration.hpp   # 공개 API (Eigen 비의존)
├── common/            # dispatcher, Vec3<->Eigen, 공통 metric, PythonWorker
├── gicp/              # small_gicp 래퍼 (ICP/Plane-ICP/GICP/VGICP)
├── kiss_matcher/      # KISS-Matcher (FetchContent) — C++ 네이티브 글로벌
├── gradient_sdf_gpu/  # gradient-SDF: 영속 Python 워커 + 벤더링 패키지   ← 패턴 원본
└── bufferx/           # ★ 신규: BUFFER-X 영속 Python 워커 + 벤더링 패키지
```

흐름상 권장 파이프라인은 기존 글로벌 방식과 동일하다:

```
BUFFER-X (글로벌, 초기값 불필요)  →  GICP / gradient-SDF (mm 단위 미세 정합)
```

`RegAlgo::KissGicp`가 KISS → GICP를 체이닝하는 방식
(`backend/registration/kiss_matcher/kiss_backend.cpp:61-70`)을 그대로 따라
`RegAlgo::BufferXGicp`를 별도 enum case로 둔다.

### 1.2 영속 Python 워커 재사용

핵심 설계 결정: **새 IPC 인프라를 만들지 않는다.** gradient-SDF가 쓰는
`cc::reg::PythonWorker`
(`backend/registration/include/cloudcropper/registration/python_worker.hpp:65-104`)
는 알고리즘 독립적이다 — 스크립트 경로/인터프리터/로그만 옵션으로 받고
JSON-lines 프로토콜을 말한다. BUFFER-X 브리지는 이 클래스를 **그대로** 쓰고
워커 스크립트만 새로 작성한다.

`PythonWorker`의 핸드셰이크 타임아웃(`python_worker.hpp:71-72`)이 이미 cold
torch import(최대 300초)를 가정하고 있어 BUFFER-X(torch + MinkowskiEngine +
가중치 로드)에도 적합하다. 단 sparse-conv 백엔드 로드 + 체크포인트 로드가
더 무거울 수 있으므로 `readyTimeoutSec`를 워커 옵션에서 상향
조정한다(§3.2 참조, 600초 권장).

워커 라이프사이클:
- C++ 브리지의 첫 `register` 호출 시 **lazy**하게 1회 spawn
(`gsdf_gpu.cpp:121-127`의 `static PythonWorker worker(...)` 패턴).
- 토치/Minkowski import + 가중치 로드 비용을 **1회만** 지불, 이후 호출은
모델을 메모리에 유지.
- 앱 종료까지 생존, 소멸자가 shutdown → SIGTERM → SIGKILL
(`python_worker.hpp:76`).

### 1.3 IPC 프로토콜 (gradient-SDF와 동일)

`gsdf_worker.py:9-44`에 문서화된 프로토콜을 재사용한다:

1. **핸드셰이크** (요청 전): `{"event":"loading","pid":N}` 즉시 →
   import 성공 시 `{"event":"ready","pid":N,"device":"cuda"|"cpu",...}`,
   실패 시 `{"event":"fatal","error":{...}}` 후 exit 1.
2. **요청/응답**: stdin/stdout에 UTF-8 JSON 한 줄씩, 항상 한 요청만 in-flight.
3. stdout fd는 워커가 사설 dup로 보호하고 fd 1을 stderr로 리다이렉트
   (`gsdf_worker.py:289-295`) — torch/Minkowski가 fd 1에 찍는 로그가
   프로토콜을 오염시키지 않게 하는 **필수 패턴**. BUFFER-X 워커도 동일하게
   stdout을 가장 먼저 가로채야 한다.

### 1.4 NPZ 핸드오프

점군은 NPZ 임시 파일로 전달한다 (`gsdf_gpu.cpp:60-65`, `:129-132`).
CloudCropper의 `io::NpzWriter`가 키 `"xyz"`(N,3 f4), 선택적 `"normal"`로 export
하며 워커는 `np.load`로 읽는다 (`gsdf_worker.py:130-133, 156-160`). BUFFER-X는
**색상/노멀이 불필요**(좌표만 사용)하므로 `"xyz"`만 있으면 충분하다 — 기존
`writeNpz`를 그대로 재사용한다.

- 임시 디렉터리: `temp_directory_path()/("cc_bufferx_"+pid)`, `TempDirGuard`로
  앱 종료 시 정리 (`gsdf_gpu.cpp:91-97, 114-117`).
- 핸드오프 파일 `source.npz`/`target.npz`는 호출마다 덮어쓰기(호출은 워커가
  직렬화) (`gsdf_gpu.cpp:130`).

### 1.5 가중치(weights) 처리 — **런타임 전용**

BUFFER-X는 사전학습 체크포인트가 필요하다(zero-shot이므로 **단일 가중치
세트**가 모든 센서/스케일에 일반화 — 재학습/파인튜닝 불필요). 이것은
**오직 런타임 관심사**이며 빌드/CMake와 무관하다.

가중치 획득·배치 정책:

| 항목 | 정책 |
|---|---|
| 출처 | 업스템 릴리스/배포 링크 (저장소 `ckpt/` 또는 README의 다운로드 스크립트). 구현 시 정확한 URL/파일명 확정 필요 — §6 미해결. |
| 기본 위치 | `backend/registration/bufferx/python/weights/` (벤더링 패키지 옆). `.gitignore`에 추가하여 **저장소에 커밋하지 않음**. |
| 위치 오버라이드 | `config/bufferx.yaml`의 `weights_dir:` 키 → 워커로 forward. 미지정 시 워커가 스크립트 기준 `./weights/`를 탐색. |
| 다운로드 | `requirements.txt`와 나란히 `download_weights.sh`(또는 워커 `--fetch-weights` 모드) 제공. README/`config/bufferx.yaml` 주석에 1줄 안내. |
| 부재 시 동작 | 워커가 `{"event":"fatal",...}` 또는 op 에러로 명확한 메시지 반환 → C++가 `ErrorCode`로 표면화 (네이티브 폴백 없음, gradient-SDF와 동일 철학 `gsdf_gpu.hpp:6-7`). |

> 빌드타임에는 가중치가 **존재하지 않아도 된다.** 오프라인 빌드/CI는 C++
> 브리지만 컴파일하며 가중치는 첫 실행 시점의 사용자 책임이다.

---

## 2. 파일별 변경 목록 (경로 + 라인 앵커)

> 앵커는 현재(2026-06-16) 기준. enum/switch에 case를 **추가**하면 컴파일러가
> 비포괄 switch를 잡아주므로(아래 dispatcher/algoName/config 모두 enum 전수
> switch) 누락 지점이 자동 검출된다.

### 2.1 공개 헤더 — `backend/registration/include/cloudcropper/registration/registration.hpp`

**(a) enum 확장** (`registration.hpp:23-31`). `GradientSdfGpu` 뒤에 추가:

```cpp
enum class RegAlgo {
    Icp, PlaneIcp, Gicp, VGicp,
    KissMatcher,
    KissGicp,
    GradientSdfGpu,
    BufferX,       // BUFFER-X (Python 워커; 학습 기반, 글로벌, 초기값 불필요)
    BufferXGicp,   // BUFFER-X -> GICP (글로벌 + local refine)
};
```

**(b) 옵션 필드** (`registration.hpp:36-65`, `RegOptions`). gradient-SDF가
`sdfResolution`/`sdfTruncMul`/`sdfUncertainty`를 둔 자리
(`registration.hpp:47-57`) 뒤에 BUFFER-X 전용 노브를 추가. 대부분의 알고리즘
하이퍼파라미터는 **yaml에서 워커로 verbatim forward**되므로(§3) C++ 구조체에는
UI/CLI가 자주 만지는 소수만 둔다:

```cpp
    // BUFFER-X (worker): 다운샘플 voxel 크기(0 = 워커 yaml 값/자동),
    // 추론 디바이스는 yaml(device)에서 결정. 나머지 노브는 yaml -> 워커.
    float bufferxVoxel = 0.0f;
```

`refine`(`registration.hpp:61`)·`init`(`:64`)는 기존 필드를 그대로 공유한다
(BUFFER-X는 글로벌이라 `init`을 무시; `refine`은 BufferXGicp 체이닝에 사용).

`RegResult`(`registration.hpp:67-85`)는 **변경 불필요**. BUFFER-X는 gradient-SDF
의 `confidence`/`normResidual`(GPIS 신뢰도)를 만들지 않으므로 기본값 `-1`
(미제공)으로 둔다 — CSV/UI가 이미 `-1` 가드를 가짐
(`main.cpp:319`, `viewer.cpp:1176`). 대신 BUFFER-X의 RANSAC 인라이어 수는
공통 metric의 `inliers`와 별개로 `detail` 문자열에 싣는다(§3.3).

**(c) `algoName` 선언**(`registration.hpp:87`)은 시그니처 변경 없음.

### 2.2 디스패처 — `backend/registration/common/registration.cpp`

**(a) include 추가** (`registration.cpp:7-12` 근처):

```cpp
#include "../bufferx/bufferx_backend.hpp"
```

**(b) `algoName` switch** (`registration.cpp:16-27`). `GradientSdfGpu` case
(`:24`) 뒤에:

```cpp
        case RegAlgo::BufferX:     return "BUFFER-X";
        case RegAlgo::BufferXGicp: return "BUFFER-X + GICP";
```

**(c) dispatcher switch** (`registration.cpp:37-56`). `GradientSdfGpu` case
(`:53-55`) 뒤에:

```cpp
        case RegAlgo::BufferX:
        case RegAlgo::BufferXGicp:
            r = bufferx::run(source, target, opt);
            break;
```

> 디스패처 뒤의 **백엔드 독립 metric**(`registration.cpp:59-66`)이 rmse/inliers
> 를 재계산하므로 BUFFER-X도 자동으로 다른 알고리즘과 비교 가능해진다.
> 워커가 따로 metric을 만들 필요 없음.

### 2.3 새 백엔드 헤더/구현

**`backend/registration/bufferx/bufferx_backend.hpp`** — `gsdf_gpu.hpp`의
복제(`gsdf_gpu.hpp:1-18`). 네임스페이스 `cc::reg::bufferx`, 단일 진입점:

```cpp
namespace cc::reg::bufferx {
// RegAlgo::BufferX / BufferXGicp 처리.
Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt);
}  // namespace cc::reg::bufferx
```

**`backend/registration/bufferx/bufferx_backend.cpp`** — `gsdf_gpu.cpp`를
원본으로 한다(§3에 상세). 재사용/변경점:

- `#if !defined(CLOUDCROPPER_HAS_NPZ)` 가드 + `Unsupported` 스텁
  (`gsdf_gpu.cpp:32-37`) 그대로.
- `findScript()`(`gsdf_gpu.cpp:44-58`): 환경변수
  `CLOUDCROPPER_BUFFERX_SCRIPT`, 상대경로
  `backend/registration/bufferx/python/bufferx_worker.py`로 치환.
- `writeNpz`/`TempDirGuard`/`fnv1a64`/`jsonScalar`(`gsdf_gpu.cpp:60-97`) 동일.
- temp dir 접두사 `cc_bufferx_`(`gsdf_gpu.cpp:114-116`).
- 정합 후 `BufferXGicp`이면 GICP 체이닝
  (kiss_backend.cpp:61-70 패턴) **또는** 워커 내부에서 refine 수행 후
  `refine` 플래그만 forward — §3.4에서 GICP 체이닝을 **C++ 측**에서 하도록
  결정(이유: small_gicp는 C++에 이미 있고 워커 의존성을 줄임).

### 2.4 설정 — `config.hpp` / `common/config.cpp` + `config/bufferx.yaml`

**(a) `configFileFor`** (`config.cpp:85-92`). switch에 추가:

```cpp
        case RegAlgo::BufferX:
        case RegAlgo::BufferXGicp: return "bufferx.yaml";
```

**(b) `defaultsFor`** (`config.cpp:100-125`). 공통 키(`refine` 등)는
`:108-111`에서 이미 처리. BUFFER-X 전용 분기를 `:117-121`(gradient-SDF) 옆에
추가:

```cpp
        case RegAlgo::BufferX:
        case RegAlgo::BufferXGicp:
            opt.bufferxVoxel = getF(kv, "voxel_size", opt.bufferxVoxel);
            break;
```

**(c) 신규 `config/bufferx.yaml`** — `config/gradient-sdf-gpu.yaml`을 모델로.
워커로 forward되는 모든 알고리즘 노브를 여기에 둔다(주석 = 스키마):

```yaml
# BUFFER-X (zero-shot point cloud registration) 워커 기본값.
# backend/registration/bufferx/python/bufferx_worker.py 가 읽고, 영속 워커로
# 1회 spawn 된다. python/timeout_sec/weights_dir 외 모든 키는 워커로 verbatim
# forward 된다 (gradient-sdf-gpu.yaml 과 동일 규약).
#   pip install -r backend/registration/bufferx/python/requirements.txt
#   bash backend/registration/bufferx/python/download_weights.sh

python: python3          # 워커 인터프리터 — 변경 시 앱 재시작 필요
device: cuda             # cuda | cpu (cuda 불가 시 cpu 폴백)
timeout_sec: 600         # 정합 1회 한계; 초과 시 워커 kill
weights_dir:             # 빈 값 = 스크립트 기준 ./weights/ 탐색

# --- BUFFER-X 추론 노브 (업스템 config 키에 맞춰 확정) ---
voxel_size: 0.0          # 입력 다운샘플 voxel (0 = 자동/스케일 정규화에 위임)
num_keypoints: 5000      # patch 추출 키포인트 수
ransac_iters: 50000      # local-to-global pose 추정 반복
# scale_factor:          # 부재 = BUFFER-X 자동 스케일 정규화 (zero-shot 핵심)

refine: true             # 글로벌 결과를 GICP 로 미세 정합 (bufferx-gicp)
```

### 2.5 CLI — `src/app/main.cpp`

**(a) usage 문자열**(`main.cpp:249-255`)에 `bufferx`/`bufferx-gicp` 추가:

```cpp
" [--reg-algo icp|icp-plane|gicp|vgicp|kiss|kiss-gicp|gsdf|gsdf-gpu|bufferx|bufferx-gicp]\n"
```
그리고 옵션 줄에 `[--reg-bufferx-voxel V]` 추가.

**(b) `parseAlgo` 람다**(`main.cpp:261-272`). `gsdf` 분기(`:269`) 뒤에:

```cpp
            else if (v == "bufferx") out = cc::reg::RegAlgo::BufferX;
            else if (v == "bufferx-gicp") out = cc::reg::RegAlgo::BufferXGicp;
```

**(c) 플래그 파싱**(`main.cpp:282-300`). `--reg-no-refine`(`:293`) 근처에 추가:

```cpp
            else if (a == "--reg-bufferx-voxel") ro.bufferxVoxel = std::stof(next());
```

> stdout 포맷(`main.cpp:312-322`)·CSV 파싱(reg-bench)은 변경 불필요 —
> `confidence`가 `-1`이라 `main.cpp:319`의 `>= 0.0` 가드에서 자동 생략.

### 2.6 뷰어 패널 — `src/viewer/viewer.cpp`

**(a) 알고리즘 콤보**(`viewer.cpp:364-375`, `kRegAlgos[]`). gradient-SDF
엔트리(`:374`) 뒤에 추가. BUFFER-X도 빌드 게이트가 필요하면 매크로로 감싼다
(§4.3 — 기본은 항상 빌드, 런타임 가용성으로만 결정하므로 **무가드** 권장):

```cpp
        {"gradient-SDF (GPU)", cc::reg::RegAlgo::GradientSdfGpu},
        {"BUFFER-X", cc::reg::RegAlgo::BufferX},
        {"BUFFER-X + GICP", cc::reg::RegAlgo::BufferXGicp},
```

**(b) `loadRegDefaults`**(`viewer.cpp:380-390`). 신규 필드 동기화:

```cpp
        regBufferxVoxel = d.bufferxVoxel;   // 새 UI 상태 변수 (viewer.cpp:349-351 옆에 선언)
```
상태 변수는 `viewer.cpp:349-351`(regDownsample 등)과 같은 블록에 `float
regBufferxVoxel = 0.0f;` 추가.

**(c) 알고리즘별 파라미터 UI**(`viewer.cpp:1105-1121`). gradient-SDF
블록(`:1110-1117`) 뒤에:

```cpp
            if (alg == cc::reg::RegAlgo::BufferX ||
                alg == cc::reg::RegAlgo::BufferXGicp) {
                Row("voxel");
                ImGui::InputFloat("##bxv", &regBufferxVoxel, 0.0f, 0.0f, "%.4f");
                Tip("BUFFER-X 입력 다운샘플 voxel; 0 = 자동(스케일 정규화)");
            }
```
그리고 "refine with GICP" 체크박스 게이트(`viewer.cpp:1118-1121`)에
`|| alg == BufferX || alg == BufferXGicp` 추가.

**(d) Register 실행 시 옵션 채우기**(`viewer.cpp:1136-1143`):

```cpp
                ro.bufferxVoxel = regBufferxVoxel;
```

### 2.7 빌드/벤더링

- **`backend/registration/CMakeLists.txt:4-10`** — `add_library` 소스 목록에
  추가:
  ```cmake
    gradient_sdf_gpu/gsdf_gpu.cpp
    bufferx/bufferx_backend.cpp)
  ```
  추가 링크/패키지 **불필요** — BUFFER-X는 런타임 Python이라 C++ 측은
  기존 `cloudcropper::io`(NPZ)·`PythonWorker`만 의존
  (`CMakeLists.txt:19-24`에 이미 존재).
- **`backend/registration/bufferx/python/`** — 신규 디렉터리:
  - `bufferx_worker.py` (워커, §3)
  - `requirements.txt` (torch / MinkowskiEngine(또는 spconv) / open3d / numpy /
    easydict 등 — 업스템 확정)
  - `download_weights.sh` (가중치 다운로드)
  - `weights/` (`.gitignore` 처리)
  - `VENDORED.md` — gradient-SDF의 `python/VENDORED.md`와 같은 양식으로 출처
    커밋/벤더링 일자/로컬 수정 기록. **BUFFER-X 코어를 third_party가 아닌
    이 디렉터리에 벤더링**(아래 결정 참조).
- **`.gitignore`** — `backend/registration/bufferx/python/weights/` 추가
  (현재 `.gitignore`에 `build/` 등만 있음).

#### 벤더링 위치 결정: `bufferx/python/` (third_party 아님)

gradient-SDF 선례(`gradient_sdf_gpu/python/VENDORED.md:1-11`)를 따른다. 근거:

1. BUFFER-X는 순수 Python 추론 코드 + npy/ckpt 자산 → **빌드 산출물이 없음**.
   third_party(vcpkg/FetchContent)는 C++ 컴파일 의존성용이다.
2. 워커가 `sys.path[0]`(스크립트 옆)에서 import 하므로 **install 불필요**
   (`gsdf_worker.py:53-58`). 같은 디렉터리에 벤더링하면 그대로 동작.
3. KISS-Matcher만 FetchContent를 쓰는데, 그건 C++ 라이브러리라서다
   (`kiss_matcher/fetch_kiss_matcher.cmake`, `CMakeLists.txt:26-30`).

> 단, BUFFER-X 코어가 크고(모델 정의 + 여러 모듈) 라이선스가 MIT이므로,
> 통째 복사 대신 **git submodule**(가중치 제외) + `VENDORED.md`로 커밋 핀
> 고정하는 방식도 가능. 구현팀이 저장소 크기·라이선스 확인 후 선택(§6).

---

## 3. Python 워커 계약 (`bufferx_worker.py`)

`gsdf_worker.py`를 골격으로 한다. **stdlib만 모듈 레벨 import**, 무거운
import는 `_import_heavy()` 안에서 (`gsdf_worker.py:45-58, 280-286`) — 의존성
부재를 `fatal` 이벤트로 보고하기 위함.

### 3.1 핸드셰이크 / 라이프사이클 (변경 없음)

`gsdf_worker.py:289-308`과 동일: stdout dup 보호 → `loading` 이벤트 →
`_import_heavy()`(torch + MinkowskiEngine + BUFFER-X 모델 클래스 + **가중치
로드를 여기서 1회**) → `ready` 이벤트(device/torch 버전). import/가중치 로드
실패는 `fatal`.

```python
def _import_heavy(weights_dir):
    import numpy as np
    import torch
    import open3d as o3d
    # MinkowskiEngine 또는 spconv — 업스템 백본에 맞춰 확정
    from bufferx import BufferXModel        # 벤더링 패키지
    model = BufferXModel.load(weights_dir)  # 가중치 로드 (1회)
    model.eval()
    return Worker(np, torch, o3d, model)
```

### 3.2 Worker 상태 + 모델 캐시

`Worker`는 무거운 모듈과 **로드된 모델**을 보유(`gsdf_worker.py:107-118`
참조). gradient-SDF는 타깃 SDF 필드를 캐시했지만, BUFFER-X는 **모델 가중치
자체가 영속 상태**다 — 매 호출 재로드 금지. 타깃 특징(feature) 캐시는 선택:
같은 타깃 반복 정합 시 타깃 feature 추출을 캐시하면 가속되나, MVP에서는
생략 가능.

### 3.3 `register` op — 요청/응답 스키마

**요청** (C++ 브리지가 보냄, `gsdf_gpu.cpp:150-168` 형식):

```json
{
  "id": 7,
  "op": "register",
  "source": "/tmp/cc_bufferx_1234/source.npz",
  "target": "/tmp/cc_bufferx_1234/target.npz",
  "target_key": "<fnv1a64(target xyz)>:<voxel>:<device>",
  "device": "cuda",
  "voxel_size": 0.0,
  "refine": false,
  "num_keypoints": 5000,
  "ransac_iters": 50000
}
```

- `source`/`target`/`target_key`/`device`/`voxel_size`/`refine`는 워커가
  명시적으로 소비(`gsdf_worker.py:75-78`의 `_RESERVED` 패턴).
- 나머지 키는 **타입 테이블**(`_REG_KW` 같은 `name->caster` dict,
  `gsdf_worker.py:82-104`)로 캐스팅해 BUFFER-X 추론 함수에 `**kwargs`로 전달.
  미지의 키는 stderr 경고 후 무시(`gsdf_worker.py:194-196`). → **yaml에 노브를
  추가해도 C++ 변경 0**.

**응답** (성공):

```json
{
  "id": 7, "ok": true,
  "result": {
    "transform": [16개 float, row-major, target<-source],
    "converged": 1,
    "num_inliers": 1234,
    "fitness": 0.87,
    "device": "cuda",
    "seconds": 3.21,
    "cache_hit": 0
  }
}
```

C++ 브리지의 매핑(`gsdf_gpu.cpp:178-197`)과 정합되도록:

| 응답 키 | RegResult 필드 | 비고 |
|---|---|---|
| `transform` (16) | `transform` | 필수, 없으면 `ParseError`(`gsdf_gpu.cpp:180-181`) |
| `converged` | `converged` | BUFFER-X 성공/실패 판정(예: `num_inliers > 임계` 또는 fitness) |
| `num_inliers`, `fitness` | → `detail` 문자열 | `out.detail = "BUFFER-X (cuda): N inliers, fitness 0.87"` |
| `device`, `cache_hit` | `detail` 보조 | gradient-SDF와 동일 표기 |
| (없음) | `confidence`/`normResidual` | **-1 유지** (BUFFER-X 미제공) |

> **CSV 일관성** (`registration-results.csv` 헤더
> `date,label,source,target,algo,converged,rmse,inliers,time_s,confidence,detail,transform`,
> `scripts/reg-bench.py:28-29`): `rmse`/`inliers`/`time_s`는 C++ 공통 metric이
> 채우고(`registration.cpp:59-66`), `confidence`는 빈 칸, `detail`은 워커
> 한 줄이 들어간다. reg-bench 파싱(`reg-bench.py:58-77`)이 그대로 동작 —
> **스크립트 변경 불필요**, 단 `DEFAULT_ALGOS`/`--gpu` 안내만 갱신(§2 외).

### 3.4 GICP 체이닝 위치 결정

`BufferXGicp`의 미세 정합은 **C++ 디스패처/브리지 측**에서 수행한다
(`kiss_backend.cpp:61-70`처럼 `gicp::run`에 BUFFER-X 결과를 `init`으로 전달).

근거: (1) small_gicp가 C++에 이미 링크됨 → 워커에 GICP 의존성 추가 불필요,
(2) gradient-SDF는 워커 내부 small_gicp refine을 썼지만 그건 그쪽 패키지가
이미 small_gicp를 번들했기 때문. BUFFER-X 워커는 순수 추론만 담당하게 하여
가볍게 유지. 따라서 워커에 보내는 `refine`은 **항상 false**로 두고, 체이닝은
`bufferx_backend.cpp`에서:

```cpp
if (opt.algo == RegAlgo::BufferXGicp && opt.refine) {
    RegOptions ro = opt;
    ro.algo = RegAlgo::Gicp;
    ro.init = coarseFromWorker;       // BUFFER-X 4x4
    return gicp::run(source, target, ro);   // detail 앞에 BUFFER-X 한 줄 prepend
}
```

### 3.5 디버그 / 테스트 모드

`gsdf_worker.py:331-339`의 `--oneshot` 모드를 복제해 토치 없이도 CLI에서
워커를 단독 구동 가능하게 한다. 또한 C++ 단위 테스트는 **가짜 워커**
(`tests/cc_tests.cpp:648-675`)로 IPC/매핑만 검증 — torch/Minkowski/가중치
**불필요**. BUFFER-X용 가짜 워커 테스트를 같은 양식으로 추가(아래 §5-7).

---

## 4. 빌드 & 의존성 전략

### 4.1 빌드타임 = C++ 브리지뿐

`bufferx_backend.cpp` 한 파일을 `cloudcropper_registration` 정적 라이브러리에
추가(§2.7)하는 것이 빌드타임 변경의 **전부**다. 신규 C++ 의존성·vcpkg
포트·FetchContent **없음**. 기존 게이트가 그대로 적용된다:

- `registration` 백엔드는 vcpkg `registration` 피처(Eigen + small_gicp)가
  있을 때 빌드되고 `CLOUDCROPPER_HAS_REGISTRATION`을 정의
  (`backend/registration/CMakeLists.txt:18`, `CMakeLists.txt:66-68`).
- NPZ 핸드오프는 `CLOUDCROPPER_HAS_NPZ`(io 라이브러리, `src/io/CMakeLists.txt:34`)
  필요 — 없으면 BUFFER-X 브리지도 `Unsupported` 스텁(`gsdf_gpu.cpp:32-37` 복제).
- `vcpkg`/`gui` 프리셋(`CMakePresets.json`)의 `VCPKG_MANIFEST_FEATURES`에
  이미 `npz;registration` 포함 → **프리셋/매니페스트 변경 불필요**.
- `dev` 프리셋(외부 의존성 0)은 정합 자체를 컴파일에서 제외 → BUFFER-X도 자동
  제외.

### 4.2 런타임 = pip + 가중치

사용자가 1회 준비:

```bash
pip install -r backend/registration/bufferx/python/requirements.txt
bash backend/registration/bufferx/python/download_weights.sh   # 가중치
```

`config/bufferx.yaml`의 `python:`이 가리키는 인터프리터(venv 권장)에 설치.
MinkowskiEngine은 CUDA 빌드가 까다로우므로 README에 설치 노트 필요(§6).

### 4.3 선택성(optionality) 보장

- C++ 브리지는 항상 컴파일되지만, **런타임에** Python/torch/가중치가 없으면
  워커 spawn/핸드셰이크가 실패하고 `registerClouds`가 깔끔한 에러를 반환
  (`python_worker.hpp:6-8`의 "dead worker" 규약). → 정합 빌드를 했어도
  BUFFER-X를 안 쓰면 **아무 비용 없음**.
- 뷰어 콤보/CLI에는 항상 노출하되(KISS처럼 `CLOUDCROPPER_HAS_KISS_MATCHER`
  매크로 게이트가 **불필요** — 빌드타임 C++ 의존성이 없으므로), 실패는
  런타임 메시지로 안내. (KISS는 C++ 라이브러리라 게이트가 필요했음
  `viewer.cpp:370-373`.)

---

## 5. 리스크 / 미해결 + 구현 순서

### 5.1 리스크 / 미해결 (구현 착수 전 확정)

1. **sparse-conv 백엔드**: BUFFER-X가 MinkowskiEngine을 쓰는지 spconv/torchsparse
   인지 업스템 `requirements`/모델 코드로 확정. MinkowskiEngine은 최신 CUDA/torch
   에서 빌드가 어려움 — 도커/사전빌드 휠 안내 필요.
2. **가중치 출처/파일명/크기**: 정확한 다운로드 URL·체크포인트 파일명·SHA를
   확정해 `download_weights.sh`에 고정. zero-shot 단일 가중치인지, 데이터셋별
   다중 ckpt인지 확인(zero-shot이면 단일 예상).
3. **입력 규약**: BUFFER-X가 좌표 단위(m)·스케일·다운샘플 voxel에 민감한지.
   "zero-shot + 스케일 정규화"가 핵심 강점이지만, CloudCropper의 근접
   고밀도 스캔(예: `tests/data/crop.ply`)에서 voxel/keypoint 기본값 튜닝 필요
   가능성. KISS가 dense 클라우드에서 resolution 재시도를 넣은 선례
   (`kiss_backend.cpp:43-50`)와 유사한 자동 보정이 필요할 수 있음.
4. **노멀 필요 여부**: BUFFER-X 백본이 노멀/색상을 입력으로 받는지. 받으면
   NPZ `"normal"` 키 활용(이미 export 됨), 아니면 `"xyz"`만.
5. **타깃 feature 캐시**: 반복 정합 가속을 위한 캐시 도입 여부(MVP 생략 가능).
6. **벤더링 방식**: 코어 통째 복사 vs git submodule(가중치 제외) — 저장소
   크기·라이선스(MIT 확인) 검토 후 결정.
7. **GPU 메모리**: 큰 소스 클라우드에서 OOM 가능 — 워커가 voxel 다운샘플로
   방어(gradient-SDF의 `voxel_size 0 = auto` 주의사항 `gsdf_worker.py:200-202`
   참조).

### 5.2 구현 순서 (개발팀)

1. **업스템 조사** — BUFFER-X 저장소를 읽어 §5.1의 1~4(백본/가중치/입력/노멀)
   확정. 데모 스크립트로 (source, target) → 4x4를 뽑는 최소 추론 경로 파악.
2. **벤더링** — `backend/registration/bufferx/python/`에 코어 + `requirements.txt`
   + `download_weights.sh` + `VENDORED.md` 배치. `weights/`를 `.gitignore`에.
3. **워커 작성** — `bufferx_worker.py`: `gsdf_worker.py`에서 프로토콜/핸드셰이크/
   stdout 보호/`--oneshot`을 복제하고 `register`만 BUFFER-X 추론으로 교체.
   `--oneshot`으로 실제 점군에 단독 검증.
4. **C++ 브리지** — `bufferx_backend.{hpp,cpp}`를 `gsdf_gpu.*`에서 복제,
   스크립트 경로/temp 접두사/응답 매핑(§3.3)/GICP 체이닝(§3.4) 수정.
   `CMakeLists.txt:10`에 소스 추가.
5. **공개 API + 디스패처 + config** — §2.1~2.4 enum/switch/yaml. 컴파일러가
   비포괄 switch로 누락 지점 안내.
6. **CLI + 뷰어** — §2.5~2.6 플래그/콤보/파라미터 UI.
7. **테스트** — `tests/cc_tests.cpp`에 가짜 워커 테스트
   (`tests/cc_tests.cpp:635-698` 양식) 추가: yaml→JSON forward, 응답→RegResult
   매핑, `BufferXGicp` 체이닝 검증. torch 불필요.
8. **벤치/문서** — `scripts/reg-bench.py`에 `bufferx`/`bufferx-gicp`를 `--gpu`
   계열로 안내, `docs/design/06-registration.md`에 BUFFER-X 절 추가, README의
   런타임 설치 안내 갱신. 합성 known-transform ctest(90°+오프셋, 부분 중첩)에
   BUFFER-X 케이스 추가.

---

## 부록 A — 변경 파일 체크리스트

| 파일 | 변경 |
|---|---|
| `backend/registration/include/.../registration.hpp` | enum 2 case(`:30` 뒤), `bufferxVoxel` 필드(`:57` 뒤) |
| `backend/registration/common/registration.cpp` | include, `algoName`(`:24` 뒤), dispatcher(`:55` 뒤) |
| `backend/registration/common/config.cpp` | `configFileFor`(`:89` 옆), `defaultsFor`(`:121` 옆) |
| `backend/registration/bufferx/bufferx_backend.hpp` | 신규 (gsdf_gpu.hpp 복제) |
| `backend/registration/bufferx/bufferx_backend.cpp` | 신규 (gsdf_gpu.cpp 복제 + 매핑/체이닝) |
| `backend/registration/CMakeLists.txt` | 소스 1줄(`:10`) |
| `backend/registration/bufferx/python/` | 신규: worker / requirements / download_weights / VENDORED.md / weights(gitignore) |
| `config/bufferx.yaml` | 신규 (gradient-sdf-gpu.yaml 모델) |
| `src/app/main.cpp` | usage(`:250`), parseAlgo(`:269` 뒤), 플래그(`:293` 옆) |
| `src/viewer/viewer.cpp` | kRegAlgos(`:374` 뒤), 상태변수(`:351` 옆), loadRegDefaults(`:389` 옆), UI(`:1117` 뒤), 옵션채움(`:1143` 옆) |
| `.gitignore` | `backend/registration/bufferx/python/weights/` |
| `scripts/reg-bench.py` | `--gpu`/안내 문자열만(파싱 무변경) |
| `tests/cc_tests.cpp` | 가짜 워커 테스트 1개(`:635` 양식) |
| `docs/design/06-registration.md` | BUFFER-X 절 추가 |
