# 08 — RAP 글로벌 정합 백엔드 통합 설계

> 대상 저장소: `/home/sjjung/Project/cloudcropper`
> 새 백엔드: **RAP** (Register Any Point: Scaling 3D Point Cloud Registration by
> Flow Matching, arXiv:2512.01850, 2025, PRBonn, MIT) — Rectified Point Flow
> (RPF, NeurIPS 2025) 위에 구축. <https://github.com/PRBonn/RAP>
> 작성: 설계팀 / 구현은 별도. 본 문서는 **구현 지침서**이며 프로덕션 코드는 포함하지 않는다.
>
> 권위 입력:
> - `docs/design/_rap-integration-contract.md` — API·워커 전략·파일 체크리스트(따른다)
> - `docs/design/_rap-upstream-notes.md` — 업스트림 정찰(의존성 핀, MIT, flash-attn 리스크)
> - `docs/design/06-bufferx-backend.md` — **패턴 원본**(영속 워커 + NPZ)
> - `docs/design/07-g3reg-backend.md` — 직전 백엔드 선례(형식/구조)

---

## 0. 한 줄 요약

RAP은 학습 기반 **글로벌(초기 추정 불필요)** zero-shot 정합기다. torch+lightning
체크포인트 모델 + Python `demo.py`/`sample.py` API라는 형태가 **BUFFER-X와 동일**하므로,
G3Reg의 one-shot subprocess가 아니라 **BUFFER-X의 영속 PythonWorker 패턴**(JSON-lines
IPC + NPZ 핸드오프, 빌드타임 의존성 0)을 그대로 복제해 `RegAlgo::Rap` /
`RegAlgo::RapGicp`로 추가한다. torch / flash-attn / pytorch3d / 가중치는 **전부 런타임
관심사**이며 C++ 측은 얇은 브리지(`backend/registration/rap/rap_backend.cpp`) 하나만
컴파일한다.

핵심 제약 두 가지를 미리 고정한다:
1. **flash-attn HARD 의존성** — RPF 코어 `layer.py`가 `import flash_attn`을 try/except
   없이 호출하며 SDPA 폴백이 없다. **RTX 50xx/Blackwell(sm_120)은 핀된 2.7.4 wheel이
   동작하지 않는다** → 신형 GPU 배포의 단일 최대 리스크(§6).
2. **GRACEFUL DEGRADATION 필수** — 워커는 RAP 코어/가중치/flash-attn이 없어도 `ready`로
   떠야 하고, `register`는 **IDENTITY + `converged:false` + `note`** 를 반환한다. inlier
   수치는 **절대 날조하지 않는다.** 이 정직한 폴백 덕분에 무거운 RAP 스택 없이도 이
   환경에서 백엔드를 빌드·테스트할 수 있다.

### 0.1 왜 G3Reg가 아니라 BUFFER-X 패턴인가

직전 백엔드 G3Reg는 학습-프리 외부 바이너리라 one-shot subprocess가 맞았지만, RAP은
**가중치 워밍업이 있는 torch 모델**이라 정반대다. 대비표:

| 항목 | **RAP (본 설계 = BUFFER-X 패턴)** | G3Reg (07, one-shot) |
|---|---|---|
| 프로세스 | **영속 PythonWorker**(lazy spawn, 앱 종료까지 생존) | 호출마다 fork/exec → 종료 |
| 워밍업 비용 | torch import + flow 체크포인트 로드(수백 MB) **1회만** | 없음(학습-프리) → 매번 새 프로세스가 깔끔 |
| IPC | stdin/stdout **JSON-lines** | stdout 3줄 텍스트 파싱 |
| 점군 핸드오프 | `io::NpzWriter` → `*.npz`(xyz) | `io::PcdWriter` → `*.pcd` |
| 빌드 게이트 | `CLOUDCROPPER_HAS_NPZ` | `CLOUDCROPPER_HAS_PCD` |
| C++ 빌드 의존성 | 0 (런타임 python) | 0 (런타임 외부 바이너리) |
| 폴백 | **graceful**(identity+note, 워커가 항상 ready) | ErrorCode(바이너리 부재 시) |

RAP의 가중치 로드(+ 무거운 flash-attn/pytorch3d import)는 BUFFER-X보다도 비싸므로 매
호출 프로세스를 새로 띄우는 것은 비현실적이다 → 영속 워커가 유일하게 합리적이다.

`RegAlgo::RapGicp`의 미세정합 체이닝은 `kiss_backend.cpp:61-70` / `g3reg_backend.cpp:296-308`
패턴 그대로 **C++ 측**에서 한다(워커는 순수 추론만).

---

## 0.2 REALIZED 접근 (구현 완료분, 설계↔구현 일치 갱신 — 2026-06-17)

설계 시점의 "flash-attn HARD 의존성 + Blackwell blocker"(§6 #1)는 **순수-torch shim으로
해소**했다. flash-attn / pytorch3d를 **설치하지 않고** 다음을 실현했다. 본 절이 실제 구현의
권위 기술이며, 아래 §3/§4/§6의 일부 서술은 이 절로 갱신된 것으로 읽는다.

- **conda env `rap`** (`/home/sjjung/miniconda3/envs/rap/bin/python`, torch **2.7.0+cu128**)으로
  RTX 5060 / **Blackwell(sm_120)** 에서 동작. 핀된 torch 2.5.1 격자는 사용하지 않는다.
- **순수-torch shim** (`backend/registration/rap/python/_shims/`):
  - `flash_attn.py` — `flash_attn.flash_attn_varlen_qkvpacked_func`의 **SDPA drop-in**.
    FLASH/EFFICIENT `sdpa_kernel` 백엔드를 강제(기본 MATH 백엔드는 8GB에서 global attention
    OOM). 업스트림 `import flash_attn`이 이 shim으로 resolve되도록 워커가 `_shims`를
    `sys.path` **최상단**에 self-insert.
  - `pytorch3d/` — 실제 `ops.ball_query`, `ops.sample_farthest_points`, `loss/chamfer`;
    `structures`/`renderer`/`ops.iterative_closest_point`은 stub(viz/eval 전용, 추론 경로
    미사용). 따라서 §6 #3(pytorch3d 추론 의존)은 **shim으로 충족** — 별도 설치 불요.
- **points-only 기본 `rap_12_po`** (mini-SpinNet feature 추출 skip; checkpoint
  `rap_model_12.ckpt`). `config/rap.yaml`의 기본 `model`도 `rap_12_po`로 갱신.
- **영속 모델 로드 + 호출별 추론** (검증된 `demo.py` 파이프라인 재사용):
  - 시작(`_import_heavy`→`_load_rap_model`): `_shims`/`rap_upstream`/`rap_upstream/dataset_process`
    를 `sys.path`에 self-insert(PYTHONPATH env 불요), `PYTORCH_CUDA_ALLOC_CONF=
    expandable_segments:True` 설정, hydra `initialize_config_dir`로 `RAP_inference` compose
    (overrides: `model=rap_12_po`, `ckpt_path=<abs rap_model_12.ckpt>`, `n_generations`,
    `inference_sampling_steps`) 후 `sample.py:setup()`으로 **모델+체크포인트를 1회** 로드,
    상주. (hydra는 프로세스당 1회만 init → `GlobalHydra.clear()` 가드.)
  - 호출별(`register`→`infer_pair`): NPZ `"xyz"`를 PLY로 per-call sample dir에 기록 →
    `demo.py:process_point_clouds(feature_extraction_on=False, use_random_downsample=True,
    target_points_per_scan≈8000)`로 sample dir 구성 → **상주 모델**에 대해
    `trainer.test(model, datamodule)` 실행(모델 가중치 상주, **datamodule/trainer만** 재생성).
    출력 dir은 `trainer.log_dir`(= `default_root_dir`)에 baked-in이므로 **호출별 trainer를
    새로 인스턴스화**(저렴; 모델은 상주)해 per-call `log_dir`을 가리킨다. 그 후 두
    `*_<part>_transform.txt`(WORLD pose)를 읽어 `T = inv(T_target) @ T_source` 반환.
  - 8GB VRAM 대응: per-cloud point ≈8000, 호출 후 `torch.cuda.empty_cache()`.
  - 영속 디렉터리: per-call 입출력은 `/tmp`가 아닌 `python/rap_worker_runs/`(이 박스는
    전원 차단 시 `/tmp`를 비움) 아래에 두고 호출 종료 시 정리.
- **graceful degradation 유지**: shim/코어/가중치가 없으면 워커는 그대로 `ready`(`rap:0`),
  `register`는 identity+`converged:false`+`note` 반환. inlier 수치 날조 없음.

> **검증 상태**: `--oneshot`(Livox demo pair) end-to-end 및 C++ fake-worker 테스트는 검증
> 명령을 본 문서 갱신 시점에 실행하지 못함(이 환경에서 Python 실행이 권한 차단됨) — 사용자가
> §0.3의 명령으로 확인. 전체 mini-SpinNet(feature) 경로는 **미검증**(rap_12_po points-only만
> 검증 대상).

## 0.3 검증 명령 (실제 실행 필요)

```bash
# 1) Livox demo PLY → xyz-only NPZ (rap env)
/home/sjjung/miniconda3/envs/rap/bin/python \
  backend/registration/rap/python/rap_smoke/make_npz.py

# 2) oneshot: 실제 4x4 (converged true, 회전 ≈7-8°, 병진 ≈9 m 기대)
/home/sjjung/miniconda3/envs/rap/bin/python \
  backend/registration/rap/python/rap_worker.py --oneshot \
  backend/registration/rap/python/rap_smoke/source.npz \
  backend/registration/rap/python/rap_smoke/target.npz

# 3) graceful degradation: 가짜 ckpt를 가리키면 identity+converged:false+note
CLOUDCROPPER_RAP_WEIGHTS=/nonexistent \
  /home/sjjung/miniconda3/envs/rap/bin/python \
  backend/registration/rap/python/rap_worker.py --oneshot \
  backend/registration/rap/python/rap_smoke/source.npz \
  backend/registration/rap/python/rap_smoke/target.npz

# 4) C++ 빌드 + 테스트 (기존 [rap worker result] fake-worker 테스트가 통과해야 함)
cmake --build build/vcpkg --target cloudcropper cc_tests && \
  ./build/vcpkg/tests/cc_tests
```

---

## 1. 아키텍처

### 1.1 RAP이 들어가는 위치

정합 백엔드는 `backend/registration/` 아래 **알고리즘 1개당 디렉터리 1개** 구조다
(`backend/registration/CMakeLists.txt:1-3` 주석). RAP은 학습 기반 글로벌 정합이므로
분류상 BUFFER-X / gradient-SDF와 같은 "글로벌 초기화" 계열이다.

```
backend/registration/
├── include/cloudcropper/registration/registration.hpp   # 공개 API (Eigen 비의존)
├── common/            # dispatcher, Vec3<->Eigen, 공통 metric, PythonWorker
├── gicp/              # small_gicp 래퍼 (ICP/Plane-ICP/GICP/VGICP)
├── kiss_matcher/      # KISS-Matcher (FetchContent) — C++ 네이티브 글로벌
├── gradient_sdf_gpu/  # gradient-SDF: 영속 Python 워커 + NPZ
├── bufferx/           # BUFFER-X: 영속 Python 워커 + NPZ   ← 패턴 원본
├── g3reg/             # G3Reg: one-shot CLI subprocess + PCD
└── rap/               # ★ 신규: RAP 영속 Python 워커 + NPZ + flow-matching
```

권장 파이프라인은 다른 글로벌 방식과 동일하다:

```
RAP (글로벌, 초기값 불필요)  →  GICP (mm 단위 미세 정합)
```

### 1.2 영속 Python 워커 재사용

핵심 설계 결정: **새 IPC 인프라를 만들지 않는다.** BUFFER-X/gradient-SDF가 쓰는
`cc::reg::PythonWorker`
(`backend/registration/include/cloudcropper/registration/python_worker.hpp:65-104`)는
알고리즘 독립적이다 — 스크립트 경로/인터프리터/로그만 옵션으로 받고 JSON-lines
프로토콜을 말한다. RAP 브리지는 이 클래스를 **그대로** 쓰고 워커 스크립트만 새로 쓴다.

워커 라이프사이클(`bufferx_backend.cpp:121-134` 그대로):
- C++ 브리지의 첫 `register` 호출 시 **lazy**하게 1회 spawn(`static PythonWorker`).
- torch/flash-attn/pytorch3d import + flow 체크포인트 로드 비용을 **1회만** 지불, 이후
  호출은 모델을 메모리에 유지.
- 앱 종료까지 생존, 소멸자가 shutdown → SIGTERM → SIGKILL(`python_worker.hpp:76`).
- 핸드셰이크 타임아웃은 이미 cold torch import(최대 300초)를 가정(`python_worker.hpp:71-72`);
  flow 모델 로드가 더 무거우므로 `timeout_sec`(register 한계)을 **900초**로 올린다(§2.4).

### 1.3 IPC 프로토콜 (BUFFER-X와 동일)

`bufferx_worker.py:11-44`에 문서화된 프로토콜을 재사용한다:

1. **핸드셰이크**(요청 전): `{"event":"loading","pid":N}` 즉시 → import 성공 시
   `{"event":"ready","pid":N,"device":"cuda"|"cpu","torch":"<ver>","rap":0|1}`. numpy/torch
   **import 자체**가 실패할 때만 `{"event":"fatal","error":{...}}` 후 exit 1.
2. **요청/응답**: stdin/stdout에 UTF-8 JSON 한 줄씩, 항상 한 요청만 in-flight.
3. stdout fd는 워커가 사설 dup로 보호하고 fd 1을 stderr로 리다이렉트
   (`bufferx_worker.py:333-335`) — torch/flash-attn이 fd 1에 찍는 로그가 프로토콜을
   오염시키지 않게 하는 **필수 패턴**. RAP 워커도 동일하게 stdout을 가장 먼저 가로챈다.

> `rap:0|1` 플래그는 BUFFER-X의 `bufferx:0|1`과 동형 — RAP 코어+가중치가 실제로 로드됐는지
> (1) 아니면 graceful 폴백 모드인지(0)를 알린다.

### 1.4 NPZ 핸드오프

점군은 NPZ 임시 파일로 전달한다(`bufferx_backend.cpp:137-139`). CloudCropper의
`io::NpzWriter`가 키 `"xyz"`(N,3 f4)로 export하며 워커는 `np.load`로 읽는다. RAP의
`demo.py`도 좌표만으로 동작하므로(노멀/RGB는 viz 전용, notes §5) `"xyz"`만 있으면 충분 —
기존 `writeNpz`를 그대로 재사용한다.

- 임시 디렉터리: `temp_directory_path()/("cc_rap_"+pid)`, `TempDirGuard`로 앱 종료 시 정리.
- `TempDirGuard`는 static `PythonWorker` **앞에** 선언한다(`bufferx_backend.cpp:90-98,121-134`)
  — 워커(와 그 `worker.log`)가 디렉터리 제거 **전에** 소멸하도록 소멸 순서를 보장.
- 핸드오프 파일 `source.npz`/`target.npz`는 호출마다 덮어쓰기(호출은 워커가 직렬화).

### 1.5 pairwise pose 복원 — `[target, source]` → SVD → relative-to-target

이것이 RAP을 CloudCropper의 두 클라우드 규약에 맞추는 **핵심 어댑터**다(계약서 §1):

1. RAP은 본래 ≥2 part의 multi-view 조립기다. 우리는 **두 part `[target, source]`** 를
   넣는다.
2. RAP이 N번의 flow-matching step으로 정합된(assembled) 클라우드를 **생성**한다.
3. 각 part의 4×4 pose를 conditioning 클라우드와 생성 클라우드 사이의 **orthogonal
   Procrustes(SVD)** 로 복원한다(RPF의 마지막 추론 단계, notes §4의 6번; pytorch3d
   `corresponding_points_alignment` 또는 로컬 SVD).
4. pose를 **타깃(reference) 기준 상대값**으로 만든다:
   ```
   T = inv(T_target) @ T_source        # row-major target<-source
   ```
   이는 `demo.py`의 `T_part = inv(T_reference) @ T_part`(notes §4)와 동일하며,
   CloudCropper의 `RegResult::transform` 규약(`registration.hpp:10-11,78`,
   `p_target = T * p_source`)과 **정확히 일치** — 전치/추가 역행렬 불필요. 16개를 그대로
   `out.transform[i]`에 채운다.

> 이 어댑터는 **워커 내부**에서 수행한다(part 순서·reference 선택은 RAP 추론 세부라
> Python 쪽이 자연스럽다). C++ 브리지는 이미 target<-source 16개를 받는다고 가정한다.

### 1.6 백엔드 독립 metric (자동 획득)

디스패처는 백엔드 반환 후 `detail::alignmentMetric`으로 rmse/inliers/seconds를 **재계산**한다
(`registration.cpp:73-79`). 따라서 RAP도:
- `RegResult::rmse`, `inliers`, `seconds`는 **공통 metric이 채운다** → 다른 알고리즘과 바로
  비교 가능.
- 백엔드는 `transform` + `converged` + `detail`만 책임진다.
- `confidence`/`normResidual`은 gradient-SDF 전용(`registration.hpp:86-94`)이므로 RAP은
  **기본값 -1 유지**(미제공). CSV/UI가 이미 `-1` 가드를 가짐.

---

## 2. 파일별 변경 목록 (경로 + 라인 앵커)

> 앵커는 현재(2026-06-17) 기준. enum/switch에 case를 **추가**하면 컴파일러가 비포괄 switch를
> 잡아주므로(dispatcher/algoName/config 모두 enum 전수 switch) 누락 지점이 자동 검출된다.

### 2.1 공개 헤더 — `backend/registration/include/cloudcropper/registration/registration.hpp`

**(a) enum 확장**(`registration.hpp:23-35`). `G3RegGicp`(`:34`) **뒤**에 추가:

```cpp
    G3Reg,           // G3Reg (external CLI subprocess; learning-free, global, no init)
    G3RegGicp,       // G3Reg -> GICP             (global + local refine)
    Rap,             // RAP (Python worker; flow-matching, learning-based, global, no init)
    RapGicp,         // RAP -> GICP               (global + local refine)
```

**(b) 옵션 필드**(`RegOptions`, `registration.hpp:40-75`). `bufferxVoxel`(`:67`)을 **정확히
미러**하는 단일 노브를 그 뒤에 추가:

```cpp
    // RAP (worker): input downsample voxel size (0 = take the yaml value / let the
    // worker auto-derive: voxel_ratio drives the adaptive per-part point counts).
    // The inference device is decided by the yaml (`device`); every other RAP knob
    // is forwarded verbatim from config/rap.yaml to the worker.
    float rapVoxel = 0.0f;
```

`refine`(`:71`)·`init`(`:74`)는 기존 필드를 공유(RAP은 글로벌이라 `init` 무시; `refine`은
RapGicp 체이닝에 사용). **그 외 RAP 필드는 추가하지 않는다** — 나머지 노브는 전부 yaml로
워커에 forward(§3). `RegResult`(`registration.hpp:77-95`)는 **변경 불필요**.

### 2.2 디스패처 — `backend/registration/common/registration.cpp`

**(a) include 추가**(`registration.cpp:7-11` 블록):

```cpp
#include "../rap/rap_backend.hpp"
```

**(b) `algoName` switch**(`registration.cpp:18-33`). `G3RegGicp`(`:30`) 뒤에:

```cpp
        case RegAlgo::Rap: return "RAP";
        case RegAlgo::RapGicp: return "RAP + GICP";
```

**(c) dispatcher switch**(`registration.cpp:43-70`). `G3Reg/G3RegGicp`(`:66-69`) 뒤에:

```cpp
        case RegAlgo::Rap:
        case RegAlgo::RapGicp:
            r = rap::run(source, target, opt);
            break;
```

> 그 뒤 백엔드 독립 metric(`registration.cpp:73-79`)이 rmse/inliers/seconds를 재계산하므로
> RAP도 자동으로 비교 가능해진다(§1.6).

### 2.3 새 백엔드 헤더/구현

**`backend/registration/rap/rap_backend.hpp`** — `bufferx_backend.hpp` 형식 복제.
네임스페이스 `cc::reg::rap`, 단일 진입점. doc-comment에 **flash-attn/Blackwell 배포 caveat
+ 영속 워커 + NPZ 핸드오프 + RapGicp가 C++에서 GICP 체인**을 명시:

```cpp
// RAP (Register Any Point): hands the clouds to the vendored Rectified-Point-Flow
// implementation (python/rap_upstream) through a persistent worker process
// (python/rap_worker.py, JSON-lines + NPZ handoff). RAP is a learning-based,
// zero-shot, GLOBAL registrar — no initial guess: it feeds [target, source] as
// two "parts", generates the assembled cloud via N flow-matching steps, and
// recovers per-part 4x4 poses by SVD/Procrustes (made relative to the target).
// Runtime requirements only (never build-time): python3 with torch 2.5.1 + a
// HARD flash-attn dependency (no SDPA fallback) — NOTE: the pinned flash-attn
// 2.7.4 wheel does NOT support RTX 50xx / Blackwell (sm_120); see
// python/requirements.txt and python/download_weights.sh. Every algorithm knob
// in config/rap.yaml is forwarded to the worker verbatim; there is no native
// fallback (the worker degrades gracefully to identity instead). The RapGicp
// variant chains GICP (small_gicp, in C++) onto the coarse result.
#pragma once

#include "cloudcropper/registration/registration.hpp"

namespace cc::reg::rap {

// Handles RegAlgo::Rap / RapGicp.
Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt);

}  // namespace cc::reg::rap
```

**`backend/registration/rap/rap_backend.cpp`** — `bufferx_backend.cpp`를 **가깝게 미러**(§3에
상세). 재사용/변경점:
- `#if !defined(CLOUDCROPPER_HAS_NPZ)` 가드 + `Unsupported` 스텁(`bufferx_backend.cpp:33-39`),
  메시지 `"rap: needs the NPZ codec (vcpkg npz feature) for the handoff"`.
- `findScript()`: env `CLOUDCROPPER_RAP_SCRIPT`, 상대경로
  `backend/registration/rap/python/rap_worker.py`로 치환.
- `writeNpz`/`TempDirGuard`/`fnv1a64`/`jsonScalar`/`cfgGet`(`bufferx_backend.cpp:25-98`) 동일.
- temp dir 접두사 `cc_rap_`.
- 응답 매핑(§3.3)·GICP 체이닝(§3.4)은 BUFFER-X와 동형, detail 접두만 `"RAP (worker, ...)"`.

### 2.4 설정 — `config.hpp` / `common/config.cpp` + `config/rap.yaml`

**(a) `configFileFor`**(`config.cpp:85-95`). `G3Reg/G3RegGicp`(`:92-93`) 옆에:

```cpp
        case RegAlgo::Rap:
        case RegAlgo::RapGicp: return "rap.yaml";
```

**(b) `defaultsFor`**(`config.cpp:104-135`). 공통 `refine`은 `:115`에서 이미 처리됨.
`BufferX`(`:126-129`)와 동형으로 switch에 case 추가:

```cpp
        case RegAlgo::Rap:
        case RegAlgo::RapGicp:
            opt.rapVoxel = getF(kv, "voxel_size", opt.rapVoxel);
            break;
```

**(c) 신규 `config/rap.yaml`** — `bufferx.yaml`과 같은 flat 형식. 워커로 forward되는 모든
알고리즘 노브를 여기에 둔다(주석 = 스키마). 주석은 반드시 **forward 규약 + flash-attn HARD
의존성 + Blackwell blocker**를 경고한다:

```yaml
# Defaults for RAP: Register Any Point — zero-shot, learning-based GLOBAL point
# cloud registration by flow matching (arXiv:2512.01850, 2025, PRBonn), built on
# Rectified Point Flow (NeurIPS 2025). Run by a persistent Python worker process
# (backend/registration/rap/python/rap_worker.py) spawned lazily on the first
# registration and kept alive, so the torch import and the flow-model weight-load
# cost is paid once. Pairwise use feeds [target, source] as two parts; per-part
# 4x4 poses are recovered by SVD and made relative to the target.
# Once the RAP core is vendored next to the worker, install deps + weights:
#   pip install -r backend/registration/rap/python/requirements.txt
#   bash backend/registration/rap/python/download_weights.sh
#
# EVERY key here except `python`, `timeout_sec` and `weights_dir` is forwarded to
# the worker verbatim; unknown keys are ignored with a warning in
# <tmpdir>/worker.log. CLI flags / viewer edits override: voxel_size
# (--reg-rap-voxel / voxel field), refine (--reg-no-refine / checkbox).
#
# DEPLOYMENT WARNING: RAP has a HARD flash-attn dependency (no SDPA fallback in
# the RPF core). The pinned flash-attn 2.7.4 wheel does NOT support consumer
# RTX 50xx / Blackwell (sm_120) — on those GPUs the worker comes up but the
# register op returns identity (converged=false) with a note. See
# docs/design/_rap-upstream-notes.md §1.

python: python3          # interpreter for the worker — changing this needs an app restart
device: cuda             # cuda | cpu (falls back to cpu if cuda is unavailable)
timeout_sec: 900         # per-registration limit; the flow model is heavy (> bufferx's 600)
weights_dir:             # empty = the worker searches ./weights/ next to the script

# --- RAP inference knobs (forwarded to the worker; see rap_worker.py _RAP_KW) ---
voxel_size: 0.0          # input downsample voxel (0 = adaptive: voxel_ratio drives counts)
voxel_ratio: 0.05        # adaptive per-part target = voxel_ratio * voxel coverage
des_r: 5.0               # mini-SpinNet feature radius (meters)
n_generations: 1         # multi-hypothesis count
inference_sampling_steps: 10   # flow steps
model: rap_12            # rap_10 (M) | rap_12 (L) | rap_12_po (points-only, skips SpinNet)
allocation_method: voxel_adaptive   # | point_count | spatial_coverage
min_points_per_part: 200
max_points_per_part: 20000

refine: true             # refine the global result with GICP (rap-gicp)
```

### 2.5 CLI — `src/app/main.cpp`

**(a) usage 문자열**(`main.cpp:249-253`). `--reg-algo` 줄에 `rap|rap-gicp` 추가, 플래그 줄에
`--reg-rap-voxel V` 추가:

```cpp
" [--reg-algo ...|g3reg|g3reg-gicp|rap|rap-gicp]\n"
"         [--reg-downsample S] ... [--reg-bufferx-voxel V] [--reg-rap-voxel V] [--reg-no-refine]\n"
```

**(b) `parseAlgo` 람다**(`main.cpp:261-276`). g3reg 케이스(`:272-273`) 옆에:

```cpp
            else if (v == "rap") out = cc::reg::RegAlgo::Rap;
            else if (v == "rap-gicp") out = cc::reg::RegAlgo::RapGicp;
```

**(c) 플래그 파싱**(`main.cpp:286-306`). `--reg-bufferx-voxel`(`:297`) 옆에:

```cpp
            else if (a == "--reg-rap-voxel") ro.rapVoxel = std::stof(next());
```

> stdout 포맷(confidence `-1` 가드)·CSV 파싱(reg-bench)은 변경 불필요.

### 2.6 뷰어 패널 — `src/viewer/viewer.cpp`

**(a) 알고리즘 콤보**(`viewer.cpp:366-380`, `kRegAlgos[]`). G3Reg 엔트리(`:378-379`) 뒤에:

```cpp
        {"RAP", cc::reg::RegAlgo::Rap},
        {"RAP + GICP", cc::reg::RegAlgo::RapGicp},
```
(빌드타임 C++ 의존성이 없으므로 KISS류 매크로 게이트 불필요 — 항상 노출. `kRegAlgoCount`는
`sizeof` 자동 계산이라 무변경.)

**(b) 상태 변수 / `loadRegDefaults`**(`viewer.cpp:385-396`). `regBufferxVoxel`을 미러하는
`regRapVoxel` state를 같은 블록에 선언(`float regRapVoxel = 0.0f;`)하고:

```cpp
        regRapVoxel = d.rapVoxel;   // loadRegDefaults, regBufferxVoxel(:393) 옆
```

**(c) voxel-slider 게이트**(`viewer.cpp:1124-1128`). 현재 BufferX/BufferXGicp 블록에
Rap/RapGicp를 추가해 rap voxel override 슬라이더를 노출:

```cpp
            if (alg == cc::reg::RegAlgo::BufferX || alg == cc::reg::RegAlgo::BufferXGicp ||
                alg == cc::reg::RegAlgo::Rap || alg == cc::reg::RegAlgo::RapGicp) {
                Row("voxel");
                ImGui::InputFloat("##rapv", &regRapVoxel, 0.0f, 0.0f, "%.4f");
                Tip("RAP/BUFFER-X input downsample voxel; 0 = auto");
            }
```

> 구현 메모: BufferX와 Rap이 같은 슬라이더 위젯을 공유하면 `ro.bufferxVoxel`/`ro.rapVoxel`을
> 둘 다 같은 state로 채워도 무방하지만(서로 다른 알고리즘이 동시에 안 돌므로), 계약서 §11의
> "regBufferxVoxel을 미러하는 regRapVoxel 추가" 권장안을 따라 **별도 state**로 두면 더
> 명시적이다. 어느 쪽이든 bufferxVoxel이 loadRegDefaults→호출 전 적용되는 방식과 동일하게
> 배선한다.

**(d) "global, no init/no max-corr" 게이트 블록**: refine 체크박스 게이트
(`viewer.cpp:1129-1134`)에 bufferx와 나란히 Rap/RapGicp를 추가:

```cpp
            if (... || alg == cc::reg::RegAlgo::BufferX || alg == cc::reg::RegAlgo::BufferXGicp ||
                alg == cc::reg::RegAlgo::G3Reg || alg == cc::reg::RegAlgo::G3RegGicp ||
                alg == cc::reg::RegAlgo::Rap || alg == cc::reg::RegAlgo::RapGicp) {
                ImGui::Checkbox("refine with GICP", &regRefine);
            }
```

**(e) Register 실행 옵션 채우기**(`viewer.cpp:1149-1157`). `ro.bufferxVoxel`(`:1155`) 옆에:

```cpp
                ro.rapVoxel = regRapVoxel;
```

### 2.7 빌드 — `backend/registration/CMakeLists.txt`

`add_library` 소스 목록(`CMakeLists.txt:4-12`)에 1줄 추가하고, 선두 주석에 `rap/` 언급:

```cmake
  g3reg/g3reg_backend.cpp
  rap/rap_backend.cpp)
```

- 추가 링크/패키지 **불필요** — RAP 백엔드는 NPZ writer(`cloudcropper::io`, 이미 PRIVATE
  링크)와 `PythonWorker`(이미 존재)만 쓴다.
- NPZ 핸드오프는 `CLOUDCROPPER_HAS_NPZ` 필요 — `vcpkg`/`gui` 프리셋의
  `VCPKG_MANIFEST_FEATURES`에 이미 `npz;registration` 포함 → **프리셋/매니페스트 변경 불필요**.
  없으면 백엔드가 `Unsupported` 스텁(§3.0).
- `dev` 프리셋(외부 의존성 0)은 정합 자체를 컴파일에서 제외 → RAP도 자동 제외.

### 2.8 Python 디렉터리 — `backend/registration/rap/python/`

신규 디렉터리(BUFFER-X의 `python/`과 같은 구성, §4):
- `rap_worker.py`(워커, §3)
- `requirements.txt`(headless-trim된 의존성, §4.2)
- `download_weights.sh`(가중치 다운로드, §4.2)
- `VENDORED.md`(벤더링 출처/MIT/flash-attn caveat, §4.2)
- `weights/`(`.gitignore` 처리), `rap_upstream/`(벤더링 코어, `.gitignore` 또는 submodule)

### 2.9 테스트 — `tests/cc_tests.cpp`

§5의 **fake-worker 테스트** 1개 추가(`testBufferxWorkerResult`, `cc_tests.cpp:705-812`
양식). stdlib fake `rap_worker.py`가 `ready`(`rap:0`)와 register 응답을 emit → IPC/forward/
매핑/RapGicp 체이닝/identity-fallback 정직성을 검증. torch/flash-attn/가중치 불필요.

---

## 3. Python 워커 계약 (`rap_worker.py`)

`bufferx_worker.py`를 골격으로 한다. **stdlib만 모듈 레벨 import**(json/os/sys/time/
traceback) — 의존성 부재에서도 프로토콜 부트스트랩이 동작해야 함
(`bufferx_worker.py:49-55`).

### 3.1 핸드셰이크 / 라이프사이클 / graceful degradation

`bufferx_worker.py:329-349`와 동일: stdout dup 보호 → `loading` 이벤트 →
`_import_heavy()`(numpy/torch import는 필수; RAP 코어 + flash-attn + pytorch3d + flow
체크포인트는 **best-effort**) → `ready` 이벤트(`device`/`torch`/`rap:0|1`).

핵심: **graceful degradation은 의무**(`bufferx_worker.py:282-326,104-122`).
- numpy/torch import **자체**가 실패할 때만 `fatal` + exit 1.
- RAP 코어(벤더링 패키지 + CUDA 확장)·**flash-attn**·가중치가 없으면 → 워커는 그래도
  `ready`로 뜨고(`rap:0`), `register`는 **IDENTITY + `converged:false` + `note`** 를
  반환한다. **inlier/quality 수치를 절대 날조하지 않는다**(폴백 응답에는 그 키들이 단순히
  부재).
- 이 환경(torch/flash-attn/GPU 없음)에서는 항상 이 폴백 경로가 타며, 그것이 C++/프로토콜
  통합을 테스트 가능하게 만든다.

```python
def _import_heavy(weights_dir=None):
    import numpy as np
    import torch                       # 여기까지 실패하면 fatal
    model = None; model_error = None
    try:
        import flash_attn             # HARD dep — 없으면(또는 sm_120) 여기서 실패
        from rectified_point_flow import RAPModel   # 벤더링 rap_upstream/
        model = RAPModel.load(weights_dir or _WEIGHTS)
        model.eval()
    except BaseException as exc:       # 코어/가중치/flash-attn 부재 → 폴백
        model, model_error = None, f"{type(exc).__name__}: {exc}"
    return Worker(np, torch, model, model_error)
```

### 3.2 `register` op — 요청/응답 스키마 + pose 복원

**요청**(C++ 브리지가 보냄, `bufferx_backend.cpp:154-168` 형식):

```json
{
  "id": 7, "op": "register",
  "source": "/tmp/cc_rap_1234/source.npz",
  "target": "/tmp/cc_rap_1234/target.npz",
  "target_key": "<fnv1a64(target xyz)>:<voxel>:<device>:<model>",
  "device": "cuda",
  "voxel_size": 0.0,
  "refine": false,
  "voxel_ratio": 0.05, "des_r": 5.0, "n_generations": 1,
  "inference_sampling_steps": 10, "model": "rap_12", ...
}
```

- `source`/`target`/`target_key`/`device`/`voxel_size`/`refine`/`weights_dir`는 워커가
  명시적으로 소비(`_RESERVED`, `bufferx_worker.py:86-87`).
- 나머지 키는 **타입 테이블** `_RAP_KW`(`bufferx_worker.py:92-95`의 `_BX_KW` 패턴)로 캐스팅:

```python
_RAP_KW = {
    "n_generations": int,
    "inference_sampling_steps": int,   # flow steps
    "des_r": float,                    # mini-SpinNet feature radius (m)
    "voxel_ratio": float,
    "allocation_method": str,          # voxel_adaptive | point_count | spatial_coverage
    "min_points_per_part": int,
    "max_points_per_part": int,
    "model": str,                      # rap_10 | rap_12 | rap_12_po (points-only)
}
```
미지의 키는 stderr 경고 후 무시(`bufferx_worker.py:230-232`). → **yaml에 노브를 추가해도
C++ 변경 0**. 브리지에서 오는 `refine`은 **항상 false**(GICP는 C++ 측, §3.4).

> lock-step caveat(BUFFER-X의 `num_scales` 주석 `bufferx_worker.py:90-91`처럼): `model`이
> `rap_12_po`(points-only)면 mini-SpinNet feature 추출 단계(`des_r`)를 **건너뛴다** — `model`
> 변경 시 feature 경로 분기가 따라 바뀜을 `_RAP_KW` 주석에 명시.

**추론 파이프라인**(`_run_rap`, notes §4):
1. `np.load` source/target `"xyz"`.
2. voxel downsample(`voxel_size>0`이면 그 값, 아니면 `voxel_ratio` 기반 adaptive).
3. points-only 모델이 아니면 mini-SpinNet feature 추출(`des_r`).
4. `[target, source]` 두 part로 flow sampling(`inference_sampling_steps`).
5. **SVD pose recovery** → part별 4×4.
6. **relative-to-target**: `T = inv(T_target) @ T_source`(§1.5) → row-major target<-source.

**응답**(성공):

```json
{ "id": 7, "ok": true, "result": {
    "transform": [16 float, row-major, target<-source],
    "converged": 1,
    "device": "cuda", "seconds": 12.3, "cache_hit": 0,
    "num_inliers": 1234            // 진짜 quality 키는 사용 가능할 때만
}}
```

C++ 브리지의 매핑(`bufferx_backend.cpp:178-203`)과 정합:

| 응답 키 | RegResult 필드 | 비고 |
|---|---|---|
| `transform`(16) | `transform` | 필수, 없거나 ≠16이면 `ParseError` |
| `converged` | `converged` | RAP 성공/실패(폴백 시 false) |
| `device`/`cache_hit` | → `detail` | `cache_hit` → `", cached target"` |
| `num_inliers`/`seconds`(있으면) | → `detail` | 공통 metric `inliers`와 별개 |
| `note`(있으면) | → `detail` `[...]` | 폴백 사유 표면화(identity를 정합으로 오인 방지) |
| (없음) | `confidence`/`normResidual` | **-1 유지** |

detail 형식: `"RAP (worker, <device>): ..."` (BUFFER-X와 동형).

### 3.3 fallback 정직성 (절대 날조 금지)

`bufferx_worker.py:256-261`의 규약을 그대로 따른다: **진짜 quality 수치만** 응답에 싣는다.
폴백(model is None) 경로의 응답은 `transform=identity16`, `converged:0`, `note`만 있고
`num_inliers` 키는 **부재**. C++ 브리지는 그 키가 없으면 `"? inliers"`로 표기하고
(`bufferx_backend.cpp:192-194`), `note`를 `[...]`로 표면화한다. → identity 결과가 진짜
정합으로 오인되지 않는다(§5 테스트 (3)이 이를 검증).

### 3.4 GICP 체이닝 위치 결정

`RapGicp`의 미세 정합은 **C++ 디스패처/브리지 측**에서 수행한다(`bufferx_backend.cpp:207-217`
패턴). 워커에 보내는 `refine`은 **항상 false**, 체이닝은 `rap_backend.cpp`에서:

```cpp
if (opt.algo == RegAlgo::RapGicp && opt.refine) {
    RegOptions ro = opt;
    ro.algo       = RegAlgo::Gicp;
    ro.init       = out.transform;             // RAP 4x4 를 초기값으로
    auto refined  = gicp::run(source, target, ro);
    if (refined) {
        refined->detail = out.detail + "  ->  " + refined->detail;  // coarse 한 줄 prepend
        return refined;
    }
    // refine 실패 시 coarse 반환(글로벌 결과는 유효)
}
return out;
```

근거: small_gicp가 C++에 이미 링크됨 → 워커에 GICP 의존성 추가 불필요. 워커는 순수
추론만 담당해 가볍게 유지.

### 3.5 디버그 / 테스트 모드

`bufferx_worker.py:372-380`의 `--oneshot <source.npz> <target.npz>` 모드를 복제해 토치
없이도(폴백 경로) CLI에서 워커를 단독 구동 가능하게 한다. C++ 단위 테스트는 **가짜 워커**로
IPC/매핑만 검증(§5).

---

## 4. 빌드 & 의존성 전략

### 4.1 빌드타임 = C++ 브리지뿐 (의존성 0 추가)

`rap_backend.cpp` 한 파일을 `cloudcropper_registration` 정적 라이브러리에 추가(§2.7)하는
것이 빌드타임 변경의 **전부**다. 신규 C++ 의존성·vcpkg 포트·FetchContent **없음**.

- NPZ 핸드오프는 `CLOUDCROPPER_HAS_NPZ`(io 라이브러리) 필요 — 없으면 `Unsupported`
  스텁(§3.0). `vcpkg`/`gui` 프리셋은 `npz;registration` 포함 → **프리셋/매니페스트 무변경**.
- `PythonWorker`·`io::NpzWriter`는 이미 라이브러리에 존재.

### 4.2 런타임 = pip + 가중치 + 벤더링 (notes §1-3, 6)

`requirements.txt`(headless-trim — gradio/mitsuba/wandb/matplotlib **제거**):

| 패키지 | 버전/출처 |
|---|---|
| torch / torchvision | 2.5.1 / 0.20.1, `--index-url .../whl/cu124` |
| flash-attn | **2.7.4.post1** — GitHub release prebuilt wheel(notes §1 URL) |
| pytorch3d | 0.7.8 (`py310_cu121_pyt251` wheel) |
| diffusers / lightning / torchmetrics | 0.33.0 / 2.5.2 / (pin) |
| hydra-core | 1.3.2 |
| torch-scatter/sparse/cluster/spline-conv | PyG (torch-2.5.1+cu124) |
| open3d / huggingface-hub / scipy / h5py / addict / trimesh / laspy | (notes §2) |

`download_weights.sh`: HF `YuePanEdward/RAP`(`rap_model_10.ckpt` 327MB,
`rap_model_12.ckpt` 356MB, `mini_spinnet_t.pth` 3.67MB) 또는 ipb.uni-bonn.de
`weights.zip` → `./weights/`(notes §3).

`VENDORED.md`: `rap_upstream/`에 클론할 것(PRBonn/RAP `rectified_point_flow` 패키지), **MIT**,
flash-attn/Blackwell caveat, 그리고 코어+가중치가 준비될 때까지 워커가 graceful하게
degrade한다는 점.

사용자 1회 준비:
```bash
pip install -r backend/registration/rap/python/requirements.txt
bash backend/registration/rap/python/download_weights.sh
# rap_upstream/ 에 PRBonn/RAP rectified_point_flow 패키지 벤더링 (VENDORED.md)
```

### 4.3 선택성(optionality)

- C++ 브리지는 (NPZ가 있으면) 항상 컴파일되지만, **런타임에** Python/torch가 없으면 워커
  spawn/핸드셰이크가, 코어/가중치/flash-attn이 없으면 **graceful 폴백**(identity+note)이
  동작 → 정합 빌드를 했어도 RAP을 안 쓰면 비용 0.
- 뷰어 콤보/CLI에는 항상 노출(빌드타임 C++ 의존성 없음), 미설치/Blackwell은 `note`로 안내.

---

## 5. FAKE-WORKER 단위 테스트 (`tests/cc_tests.cpp`)

`testBufferxWorkerResult`(`cc_tests.cpp:705-812`)와 **같은 양식**. stdlib fake `rap_worker.py`가
`{"event":"ready",...,"rap":0}`와 transform[16]+converged+device(+note) register 응답을
emit. torch/flash-attn/가중치 불필요.

```cpp
// 반드시 프로세스의 FIRST rap 호출: PythonWorker(+spawn 옵션)는 함수 로컬 static 이라
// 이 임시 config 에 1회 바인딩된다 (bufferx 테스트 :705-706 과 동일 제약).
void testRapWorkerResult() {
    std::cerr << "[rap worker result]\n";
    if (std::system("command -v python3 >/dev/null 2>&1") != 0) {
        std::cerr << "  SKIP: python3 not on PATH\n"; return;
    }
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "cc_rap_worker_cfg";
    fs::create_directories(dir);
    { std::ofstream f(dir / "rap.yaml");
      f << "python: python3\nn_generations: 3\ntimeout_sec: 30\n"; }
    const fs::path script = dir / "fake_rap_worker.py";
    {   std::ofstream f(script);
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
             "    if r.get('refine') is not False: bad.append('refine')\n"    // 워커는 항상 false 수신
             "    if 'voxel_size' not in r: bad.append('voxel_size')\n"
             "    if not (os.path.exists(r.get('source','')) and "
             "os.path.exists(r.get('target',''))): bad.append('npz')\n"
             "    if bad:\n"
             "        proto.write(json.dumps({'id':i,'ok':False,'error':{'type':"
             "'AssertionError','message':'bad: '+','.join(bad)}})+'\\n')\n"
             "    elif float(r.get('voxel_size',0) or 0)>0:\n"          // identity 폴백 shape
             "        proto.write(json.dumps({'id':i,'ok':True,'result':{'transform':"
             "[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],'converged':0,'device':'fake',"
             "'seconds':0.0,'cache_hit':0,'note':'flash-attn missing'}})+'\\n')\n"
             "    else:\n"
             "        proto.write(json.dumps({'id':i,'ok':True,'result':{'transform':"
             "[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],'converged':1,'num_inliers':555,"
             "'device':'fake','seconds':0.1,'cache_hit':0}})+'\\n')\n"; }
    setenv("CLOUDCROPPER_CONFIG_DIR", dir.c_str(), 1);
    setenv("CLOUDCROPPER_RAP_SCRIPT", script.c_str(), 1);

    const cc::PointCloud target = regBlobsCloud();
    cc::PointCloud       source = target;  // identity-aligned: GICP refine converges

    // (1) 평문 RAP: 워커 결과 -> RegResult 매핑.
    {   cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::Rap);
        opt.algo = cc::reg::RegAlgo::Rap;
        auto rr = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (rr) {
            CHECK(rr->converged);
            CHECK(rr->confidence < 0.0);          // RAP 미제공
            CHECK(rr->normResidual < 0.0);
            CHECK(rr->transform == cc::reg::kIdentity4);
            CHECK(rr->detail.find("RAP (worker, fake)") == 0);
            CHECK(rr->detail.find("555 inliers") != std::string::npos);
        } else std::cerr << "  error: " << rr.error().message << "\n";
    }
    // (2) RapGicp: coarse 라인이 GICP refine 라인 앞에 prepend.
    {   cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::RapGicp);
        opt.algo = cc::reg::RegAlgo::RapGicp; opt.refine = true;
        auto rr = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (rr) {
            CHECK(rr->converged);                 // 체인된 GICP 에서
            CHECK(rr->detail.find("RAP (worker, fake)") == 0);
            CHECK(rr->detail.find("  ->  ") != std::string::npos);
        }
    }
    // (3) identity-fallback 정직성: identity + converged:false + note, num_inliers 부재.
    {   cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::Rap);
        opt.algo = cc::reg::RegAlgo::Rap; opt.rapVoxel = 0.5f;  // 폴백 shape + voxel override
        auto rr = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (rr) {
            CHECK(!rr->converged);
            CHECK(rr->detail.find("? inliers") != std::string::npos);
            CHECK(rr->detail.find("[flash-attn missing]") != std::string::npos);
        }
    }
    unsetenv("CLOUDCROPPER_CONFIG_DIR");
    unsetenv("CLOUDCROPPER_RAP_SCRIPT");
    std::error_code ec; fs::remove_all(dir, ec);
}
```

호출 등록은 bufferx 테스트와 같은 블록에 `testRapWorkerResult();`. 가드는
`#if defined(CLOUDCROPPER_HAS_NPZ) && defined(CLOUDCROPPER_HAS_REGISTRATION)`.

> **검증 항목**: ① 워커가 받은 `refine:false`(가짜 워커의 `bad` 검사), ② yaml `n_generations`
> forward, ③ src/tgt `.npz` 실제 기록, ④ TF row-major 매핑(detail이 `"RAP (worker, fake)"`로
> 시작), ⑤ RapGicp 체이닝 prepend(`  ->  `), ⑥ identity-fallback 정직성(note 표면화,
> num_inliers 날조 없음), ⑦ confidence/normResidual=-1. **torch/flash-attn/GPU 불필요.**

---

## 6. 리스크 / 미해결

1. **flash-attn HARD 의존성 + Blackwell blocker**(notes §1,7 / VERDICT): RPF 코어가
   `import flash_attn`을 try/except 없이 호출하고 SDPA 폴백이 없다. **RTX 50xx/Blackwell
   (sm_120)은 핀된 2.7.4 wheel이 동작하지 않는다.** 신형 GPU 배포 시 (a) 더 새 flash-attn
   (FA3/2.8+, CUDA 12.8+) + 더 새 torch(2.5.1 핀 파괴) 또는 (b) SDPA 패치 spike가 필요.
   본 설계는 이를 **graceful 폴백**으로 흡수(워커는 떠도 identity+note 반환) — 통합/빌드는
   영향 없고, 실제 정합만 미동작. 배포 전 **타깃 GPU arch 확인 필수.**
2. **타이트한 버전 격자**: Python 3.10 + torch 2.5.1 + cu124 + pytorch3d 0.7.8(cu121 wheel)
   + flash-attn cp310. 핀을 벗어나면 wheel이 깨짐 → conda env 권장(notes §7).
3. **pytorch3d 추론 의존 여부**: SVD pose recovery에 `corresponding_points_alignment`를
   쓰면 pytorch3d가 **추론 의존**(notes §2). 깔끔한 로컬 SVD로 대체하면 제거 가능 — 구현팀
   확인(MVP는 pytorch3d 가정).
4. **part 순서 / reference 선택**: `[target, source]` 순서와 `inv(T_target)` reference가
   §1.5대로 일관돼야 한다. 워커 `--oneshot`으로 known-transform에 대해 방향(전치/역행렬)을
   검증 후 고정.
5. **points-only 변종**: `model=rap_12_po`는 mini-SpinNet을 건너뛰어 워커가 단순해지나
   정확도 delta는 미확인(notes §7) — feature 추출이 취약하면 대안.
6. **벤더링 크기**: RPF/RAP 코어 통째 복사 vs git submodule(가중치 제외) — 저장소 크기/MIT
   확인 후 선택(BUFFER-X §벤더링과 동일 고민).

---

## 7. 구현 순서 (개발팀, 8단계)

1. **공개 API + 디스패처** — §2.1 enum 2 case(`registration.hpp:34` 뒤) + `rapVoxel` 필드,
   §2.2 include + `algoName`(`:30` 뒤) + dispatcher(`:69` 뒤). 컴파일러가 비포괄 switch로
   누락 안내.
2. **config 배선** — §2.4 `configFileFor`(`config.cpp:93` 옆) + `defaultsFor` case
   (`:129` 옆) + 신규 `config/rap.yaml`(flash-attn/Blackwell 경고 포함).
3. **벤더링** — `backend/registration/rap/python/`에 코어 + `requirements.txt`(headless-trim)
   + `download_weights.sh` + `VENDORED.md`. `weights/`·`rap_upstream/`을 `.gitignore`에.
4. **워커 작성** — `rap_worker.py`: `bufferx_worker.py`에서 프로토콜/핸드셰이크/stdout 보호/
   `_RESERVED`/`_RAP_KW`/graceful degradation/`--oneshot`을 복제하고 `register`만 RAP
   추론(downsample→feature→flow→SVD→relative-to-target)으로 교체. `--oneshot`으로 방향 검증.
5. **C++ 브리지** — `rap/rap_backend.{hpp,cpp}`를 `bufferx_backend.*`에서 복제, 스크립트
   경로/temp 접두사(`cc_rap_`)/캐시 키(+model)/timeout 900/응답 매핑(§3.2)/GICP 체이닝(§3.4)
   수정. `CMakeLists.txt`에 소스 1줄(§2.7).
6. **CLI + 뷰어** — §2.5 usage + `parseAlgo` 2 case + `--reg-rap-voxel`; §2.6 `kRegAlgos`
   2 엔트리 + `regRapVoxel` state + voxel/refine 게이트 + 옵션 채움.
7. **테스트** — §5 fake-worker 테스트(`testRapWorkerResult`)를 `cc_tests.cpp`에 추가(반드시
   첫 rap 호출)하고 같은 블록에서 호출. IPC/forward/매핑/체이닝/폴백 정직성을 torch 없이 검증.
8. **빌드/벤치** — `cmake --build build/vcpkg --target cloudcropper cc_tests` →
   `./build/vcpkg/tests/cc_tests`(새 테스트 + 기존 전부 통과). 이후 (지원 GPU에서) main agent가
   `register --reg-algo rap[-gicp]` end-to-end, `scripts/inlier-rmse.py`로 BUFFER-X/G3Reg와
   A/B(성공률·시간·중첩부 RMSE). README에 런타임 설치 + Blackwell 경고.

---

## 부록 A — 변경 파일 체크리스트

| 파일 | 변경 |
|---|---|
| `backend/registration/include/.../registration.hpp` | enum 2 case(`:34` 뒤), `rapVoxel` 필드(`:67` 옆) |
| `backend/registration/common/registration.cpp` | include, `algoName`(`:30` 뒤), dispatcher(`:69` 뒤) |
| `backend/registration/common/config.cpp` | `configFileFor`(`:93` 옆), `defaultsFor` case(`:129` 옆) |
| `backend/registration/rap/rap_backend.hpp` | 신규(bufferx_backend.hpp 복제 + flash-attn caveat) |
| `backend/registration/rap/rap_backend.cpp` | 신규(bufferx_backend.cpp 미러 + 매핑/체이닝) |
| `backend/registration/CMakeLists.txt` | 소스 1줄 + 주석 `rap/` |
| `backend/registration/rap/python/` | 신규: rap_worker.py / requirements.txt / download_weights.sh / VENDORED.md / weights·rap_upstream(gitignore) |
| `config/rap.yaml` | 신규(bufferx.yaml 모델 + forward/flash-attn/Blackwell 경고) |
| `src/app/main.cpp` | usage, parseAlgo 2 case, `--reg-rap-voxel` 플래그 |
| `src/viewer/viewer.cpp` | kRegAlgos 2 엔트리, `regRapVoxel` state, loadRegDefaults, voxel/refine 게이트, 옵션 채움 |
| `.gitignore` | `backend/registration/rap/python/weights/`(+ rap_upstream/) |
| `tests/cc_tests.cpp` | fake-worker 테스트 1개(`:705` 양식) + 호출 |
| `scripts/reg-bench.py` | 안내 문자열만(파싱 무변경) |

> **빌드 게이트**: `CLOUDCROPPER_HAS_NPZ`(io) — `vcpkg`/`gui` 프리셋에 이미 포함. **프리셋/
> 매니페스트 변경 불필요.** RAP은 빌드타임 의존성 0(순수 런타임 Python 워커).
