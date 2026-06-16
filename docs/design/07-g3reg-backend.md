# 07 — G3Reg 글로벌 정합 백엔드 통합 설계

> 대상 저장소: `/home/sjjung/Project/cloudcropper`
> 새 백엔드: **G3Reg** (learning-free global registration, Pyramid Graph + Gaussian
> Ellipsoid Model, IEEE T-ASE 2024, HKUST-Aerial-Robotics,
> <https://github.com/HKUST-Aerial-Robotics/G3Reg>)
> 작성: 설계팀 / 구현은 별도. 본 문서는 **구현 지침서**이며 프로덕션 코드는 포함하지 않는다.
>
> 권위 입력:
> - `docs/design/_g3reg-integration-contract.md` — API·subprocess 전략·파일 체크리스트(따른다)
> - `docs/design/_g3reg-upstream-notes.md` — 업스트림 정찰(의존성 핀, MIT, FRGresult)
> - `docs/design/06-bufferx-backend.md` — 직전 백엔드 선례(형식/구조)

---

## 0. 한 줄 요약

G3Reg는 **학습이 필요 없는 글로벌(초기 추정 불필요) 정합기**다. 의존성 스택이
**버전 핀(PCL<1.11 / GTSAM 4.1.1 / Eigen<3.4 / igraph)** 으로 무겁고
CloudCropper의 최신 toolchain과 충돌하므로, **링크하지 않는다.** 대신 G3Reg를
독립 빌드한 **one-shot CLI `cc_g3reg_cli`** 를 호출하는 subprocess 래퍼로 통합한다.

gsdf-gpu / BUFFER-X는 **영속 PythonWorker + JSON-lines IPC + NPZ 핸드오프**를
쓰지만, G3Reg는 그게 **과하다**: 모델 가중치 워밍업이 없는(학습-프리) 단발 호출이므로
**프로세스를 살려둘 이유가 없다.** 따라서 패턴을 그대로 베끼지 **않고**:

| 항목 | gsdf-gpu / BUFFER-X | **G3Reg (본 설계)** |
|---|---|---|
| 프로세스 | 영속 PythonWorker(lazy spawn, 앱 종료까지 생존) | **호출마다 fork/exec → 종료**(one-shot) |
| IPC | stdin/stdout JSON-lines | **stdout 3줄 텍스트 파싱**(인자=파일 경로) |
| 점군 핸드오프 | `io::NpzWriter` → `*.npz` | **`io::PcdWriter` → `*.pcd`** (xyz만) |
| 빌드 게이트 | `CLOUDCROPPER_HAS_NPZ` | **`CLOUDCROPPER_HAS_PCD`** |
| C++ 빌드 의존성 | 0 (런타임 python) | **0** (런타임 외부 바이너리) |

`RegAlgo::G3Reg`(글로벌 단독)와 `RegAlgo::G3RegGicp`(G3Reg→GICP 미세정합 체인)을
추가한다. 체이닝은 `kiss_backend.cpp:61-70` 패턴 그대로 **C++ 측**에서 한다.

---

## 1. 아키텍처

### 1.1 G3Reg가 들어가는 위치

정합 백엔드는 `backend/registration/` 아래 **알고리즘 1개당 디렉터리 1개** 구조다
(`backend/registration/CMakeLists.txt:1-3`). G3Reg는 글로벌 초기화 계열이므로
KISS-Matcher / gradient-SDF / BUFFER-X와 같은 분류다.

```
backend/registration/
├── include/cloudcropper/registration/registration.hpp   # 공개 API (Eigen 비의존)
├── common/            # dispatcher, Vec3<->Eigen, 공통 metric, PythonWorker
├── gicp/              # small_gicp 래퍼 (ICP/Plane-ICP/GICP/VGICP)
├── kiss_matcher/      # KISS-Matcher (FetchContent) — C++ 네이티브 글로벌
├── gradient_sdf_gpu/  # gradient-SDF: 영속 Python 워커 + NPZ
├── bufferx/           # BUFFER-X: 영속 Python 워커 + NPZ
└── g3reg/             # ★ 신규: one-shot CLI subprocess + PCD 핸드오프
```

권장 파이프라인은 다른 글로벌 방식과 동일하다:

```
G3Reg (글로벌, 초기값 불필요)  →  GICP (mm 단위 미세 정합)
```

### 1.2 영속 워커를 쓰지 **않는** 이유 (핵심 결정)

`gsdf_gpu.cpp`는 `static PythonWorker worker(...)`를 lazy하게 1회 spawn해 앱 종료까지
유지한다(`gsdf_gpu.cpp:121-127`). 이는 **torch import + 가중치 로드(최대 300초)** 비용을
1회만 내려는 설계다(`python_worker.hpp:71-72`). G3Reg는:

1. **학습-프리** → 로드할 가중치가 없다. cold-start 비용은 거의 프로세스 exec 비용뿐.
2. 호출이 **완전 stateless**(타깃 SDF 캐시 같은 재사용 상태가 없다 — primitive 추출은
   매 호출 새로 한다).
3. 외부 바이너리는 **glog/GTSAM 전역 상태**를 들고 있어 한 프로세스 안에서 config를
   바꿔 재호출하기보다 **매번 새 프로세스가 깔끔**하다.

→ 따라서 **PythonWorker도 NPZ도 JSON IPC도 도입하지 않는다.** 대신 `gsdf_gpu.cpp`에서
**파일 탐색(`findScript`)·temp-dir 수명(`TempDirGuard`)·config 로딩(`configValues`)**
관용구만 빌려오고, 호출부는 fork/exec 한 번으로 대체한다.

### 1.3 subprocess 계약 (외부 `cc_g3reg_cli`)

main agent가 G3Reg `examples/`에 추가·빌드하는 CLI의 규약(계약서 §2):

```
cc_g3reg_cli <config.yaml> <src.pcd> <tgt.pcd>

stdout (정확히 이 3줄, 다른 출력 금지):
    G3REG_TF: m00 m01 m02 m03 m10 ... m33     (row-major 16 floats, target<-source)
    G3REG_INLIERS: <plane_inliers + line_inliers + cluster_inliers>
    G3REG_TIME: <seconds>
stderr: glog 등 모든 로그 (파싱 대상 아님 → worker.log 류로 리다이렉트해 에러 tail에 사용)
exit code: 0 성공, != 0 실패
```

- **TF 방향**: FRGresult.tf는 데모가 `pcl::transformPointCloud(src, out, tf)`로 src를 tgt에
  맞추는 데 쓰므로 **target<-source** (= `p_target = T * p_source`). 이는 CloudCropper의
  `RegResult::transform` 규약(`registration.hpp:10-11, 76`)과 **정확히 일치** — 전치/역행렬
  불필요. 16개를 그대로 `out.transform[i]`에 채운다.
- **INLIERS**: plane/line/cluster inlier 합. 공통 metric의 `inliers`(아래 §1.5)와는 별개의
  G3Reg 내부 지표이므로 **`detail` 문자열에만** 싣는다.
- **TIME**: G3Reg 자체 측정 시간. `RegResult::seconds`는 디스패처가 wall-clock으로
  덮어쓰므로(아래 §1.5) G3REG_TIME 역시 **`detail`에만** 참고로 싣는다.

### 1.4 바이너리·config 위치 결정

`gsdf_gpu.cpp:44-58` `findScript()` + `config.cpp:40-56` `findConfig()` 패턴을 결합한다.

**바이너리** `cc_g3reg_cli`:
1. 환경변수 `CLOUDCROPPER_G3REG_BIN`이 가리키는 실행 파일(존재 시 우선).
2. 없으면 `PATH` 탐색을 위해 `execvp`가 받는 단순 이름 `"cc_g3reg_cli"`로 폴백 가능 —
   단, **존재 확인을 위해** `gsdf_gpu.cpp:51-56`처럼 실행 파일 기준 상대경로
   (`<exe_dir>/cc_g3reg_cli`, 그리고 몇 단계 상위의 `g3reg/bin/cc_g3reg_cli` 등)도 탐색한다.
3. 어디에서도 못 찾으면 `ErrorCode::NotFound`로 **깔끔히 실패**(메시지에
   `set CLOUDCROPPER_G3REG_BIN` 안내). gsdf 철학과 동일(`gsdf_gpu.cpp:104-106`).

**config** `g3reg.yaml`:
- `CLOUDCROPPER_G3REG_CONFIG`(절대경로) 우선, 없으면 `findConfig("g3reg.yaml")`
  (= `config.cpp:40-56`의 `$CLOUDCROPPER_CONFIG_DIR` → `./config/` → exe 인접 `config/`).
- 이 yaml은 **두 종류 키**를 담는다:
  - CloudCropper 측 키: `bin`(바이너리 경로 오버라이드), `g3reg_config`(외부 G3Reg가 읽는
    *진짜* 정합 config yaml의 경로 — voxel/plane threshold 등은 그쪽 스키마), `timeout_sec`,
    `refine`.
  - 즉 `cc_g3reg_cli`의 첫 인자로 넘기는 `<config.yaml>`은 **외부 G3Reg config**이고,
    CloudCropper의 `config/g3reg.yaml`은 그 경로와 바이너리 위치를 가리키는 **얇은
    레이어**다. (둘을 분리하는 이유: G3Reg config 스키마는 PCL/GTSAM 쪽이라 우리가
    파싱하지 않고 **경로로만** 외부 바이너리에 위임한다.)

> 비교: gsdf/bufferx는 yaml 키를 전부 워커로 forward했지만, G3Reg는 yaml 본문을 **읽지
> 않고** 파일 경로째로 외부 바이너리에 넘긴다 — 파싱 책임이 외부에 있다.

### 1.5 백엔드 독립 metric (자동 획득)

디스패처는 백엔드 반환 후 `detail::alignmentMetric`으로 rmse/inliers/seconds를 **재계산**한다
(`registration.cpp:66-72`). 따라서 G3Reg도:
- `RegResult::rmse`, `RegResult::inliers`, `RegResult::seconds`는 **공통 metric이 채운다**
  → 다른 알고리즘과 바로 비교 가능.
- 백엔드는 `transform` + `converged` + `detail`만 책임진다.
- `confidence`/`normResidual`은 gradient-SDF 전용(`registration.hpp:83-92`)이므로 G3Reg는
  **기본값 -1 유지**(미제공). CSV/UI가 이미 `-1` 가드를 가짐(`main.cpp` stdout 포맷,
  `viewer.cpp` 표시).

---

## 2. 파일별 변경 목록 (경로 + 라인 앵커)

> 앵커는 현재(2026-06-16) 기준. enum/switch에 case를 **추가**하면 컴파일러가 비포괄 switch를
> 잡아주므로(아래 dispatcher/algoName/config 모두 enum 전수 switch) 누락 지점이 자동 검출된다.

### 2.1 공개 헤더 — `backend/registration/include/cloudcropper/registration/registration.hpp`

**(a) enum 확장**(`registration.hpp:23-33`). `BufferXGicp`(`:32`) 뒤에 추가:

```cpp
enum class RegAlgo {
    Icp, PlaneIcp, Gicp, VGicp,
    KissMatcher, KissGicp,
    GradientSdfGpu,
    BufferX, BufferXGicp,
    G3Reg,         // G3Reg (외부 CLI subprocess; 학습-프리, 글로벌, 초기값 불필요)
    G3RegGicp,     // G3Reg -> GICP (글로벌 + local refine)
};
```

**(b) 옵션 필드**: **추가 불필요.** G3Reg의 정합 노브(voxel/plane threshold 등)는 전부
외부 G3Reg config yaml에 있고 CloudCropper가 파싱하지 않는다. UI/CLI가 만질 G3Reg 전용
스칼라가 없으므로 `RegOptions`(`registration.hpp:38-73`)는 그대로 둔다. `refine`(`:69`)·
`init`(`:72`)만 공유한다(G3Reg는 글로벌이라 `init` 무시; `refine`은 G3RegGicp 체이닝에 사용).

> 비교: BUFFER-X는 `bufferxVoxel`(`registration.hpp:65`)을 두었지만 그건 viewer 슬라이더가
> 자주 만지는 값이라서다. G3Reg는 그런 단일 핫노브가 없어 구조체 오염을 피한다(계약서 §3 주석:
> "yaml로 충분하면 생략").

**(c) `RegResult`**(`registration.hpp:75-93`): **변경 불필요.** confidence/normResidual 기본
-1 유지.

### 2.2 디스패처 — `backend/registration/common/registration.cpp`

**(a) include 추가**(`registration.cpp:7-13` 블록):

```cpp
#include "../g3reg/g3reg_backend.hpp"
```

**(b) `algoName` switch**(`registration.cpp:17-30`). `BufferXGicp`(`:27`) 뒤에:

```cpp
        case RegAlgo::G3Reg:     return "G3Reg";
        case RegAlgo::G3RegGicp: return "G3Reg + GICP";
```

**(c) dispatcher switch**(`registration.cpp:40-63`). `BufferX/BufferXGicp`(`:59-62`) 뒤에:

```cpp
        case RegAlgo::G3Reg:
        case RegAlgo::G3RegGicp:
            r = g3reg::run(source, target, opt);
            break;
```

> 그 뒤 백엔드 독립 metric(`registration.cpp:66-72`)이 rmse/inliers/seconds를 재계산하므로
> G3Reg도 자동으로 비교 가능해진다(§1.5).

### 2.3 새 백엔드 헤더/구현

**`backend/registration/g3reg/g3reg_backend.hpp`** — `gsdf_gpu.hpp` 형식 복제. 네임스페이스
`cc::reg::g3reg`, 단일 진입점:

```cpp
// G3Reg: learning-free global registration (Gaussian Ellipsoid Model + Pyramid
// Compatibility Graph, IEEE T-ASE 2024). G3Reg는 PCL<1.11 / GTSAM 4.1.1 / Eigen<3.4
// 의존성 핀이 무거워 링크하지 않고, 독립 빌드한 외부 CLI `cc_g3reg_cli`를 one-shot
// subprocess로 호출한다(빌드타임 의존성 0). 점군은 임시 .pcd 로 핸드오프하고 stdout
// 3줄(G3REG_TF/INLIERS/TIME)을 파싱한다. 바이너리·config 부재는 ErrorCode로 실패.
#pragma once

#include "cloudcropper/registration/registration.hpp"

namespace cc::reg::g3reg {
// Handles RegAlgo::G3Reg / RegAlgo::G3RegGicp.
Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt);
}  // namespace cc::reg::g3reg
```

**`backend/registration/g3reg/g3reg_backend.cpp`** — §3에 상세. `gsdf_gpu.cpp`에서 빌리는 것:
`findScript`류 탐색, `TempDirGuard`(`gsdf_gpu.cpp:91-97`), `configValues`/`cfgGet`,
`CLOUDCROPPER_HAS_*` 가드 스텁(`gsdf_gpu.cpp:32-37`). **새로 쓰는 것**: PCD 기록(§3.2),
one-shot subprocess 실행+stdout 캡처(§3.3), 3줄 파서(§3.4), GICP 체이닝(§3.5).

### 2.4 설정 — `config.hpp` / `common/config.cpp` + `config/g3reg.yaml`

**(a) `configFileFor`**(`config.cpp:85-94`). `BufferX/BufferXGicp`(`:90-91`) 옆에 추가:

```cpp
        case RegAlgo::G3Reg:
        case RegAlgo::G3RegGicp: return "g3reg.yaml";
```

**(b) `defaultsFor`**(`config.cpp:102-131`). 공통 `refine`은 `:113`에서 이미 처리됨.
G3Reg 전용 스칼라가 `RegOptions`에 없으므로 **switch에 case를 추가하되 본문은 비운다**
(`break;`)? — 아니다. switch는 enum 전수라서 case를 빼면 컴파일은 되지만(default 있음)
**명시성**을 위해 추가하고, 파싱할 RegOptions 필드가 없으므로 `refine`만 공통 처리에 맡긴다.
`g3reg.yaml`의 `bin`/`g3reg_config`/`timeout_sec`는 `RegOptions`가 아니라 백엔드가
`configValues("g3reg.yaml")`로 직접 읽으므로(§3.1) `defaultsFor`에서 다룰 필요가 없다.
따라서 `defaultsFor`의 알고리즘별 switch(`config.cpp:114-129`)에 다음을 추가:

```cpp
        case RegAlgo::G3Reg:
        case RegAlgo::G3RegGicp:
            break;  // G3Reg 노브는 외부 yaml; refine 은 위 공통 처리에서 로드됨
```

**(c) 신규 `config/g3reg.yaml`** — gsdf/bufferx yaml과 같은 flat 형식. 단 본문 키는 워커로
forward되는 게 아니라 **CloudCropper 백엔드가 직접 읽는 얇은 레이어**임을 주석으로 명시:

```yaml
# Defaults for G3Reg: learning-free GLOBAL point cloud registration (Pyramid Graph
# + Gaussian Ellipsoid Model, IEEE T-ASE 2024, HKUST-Aerial-Robotics). G3Reg는
# PCL<1.11 / GTSAM 4.1.1 / Eigen<3.4 의존성 핀이 무거워 CloudCropper에 링크하지 않고,
# 독립 빌드한 외부 one-shot CLI `cc_g3reg_cli <g3reg_config> <src.pcd> <tgt.pcd>` 를
# subprocess로 호출한다(빌드타임 의존성 0). 점군은 자체 PCD writer 로 임시 .pcd 핸드오프.
#
# 이 파일의 키는 (gsdf/bufferx 와 달리) 외부로 forward 되지 않고 CloudCropper 백엔드가
# 직접 읽는다. 실제 정합 파라미터(voxel/plane/cluster threshold 등)는 `g3reg_config` 가
# 가리키는 *외부 G3Reg* yaml 에 있다(스키마는 G3Reg 저장소 configs/ 참조).

# 외부 CLI 바이너리 경로. 비우면 $CLOUDCROPPER_G3REG_BIN, 그다음 실행파일 인접/PATH 탐색.
bin:
# 외부 G3Reg 정합 config(cc_g3reg_cli 의 첫 인자). 비우면 $CLOUDCROPPER_G3REG_CONFIG.
# 근거리 고밀도(hap/hesai) 스캔은 G3Reg configs/hit_ms(Livox) 계열에서 시작해 튜닝 권장.
g3reg_config:
timeout_sec: 600         # 정합 1회 한계(초); 초과 시 subprocess 를 kill
refine: true             # 글로벌 결과를 GICP 로 미세 정합 (g3reg-gicp)
```

### 2.5 CLI — `src/app/main.cpp`

**(a) usage 문자열**(`main.cpp:249-253`)의 `--reg-algo` 줄에 `g3reg|g3reg-gicp` 추가:

```cpp
" [--reg-algo icp|icp-plane|gicp|vgicp|kiss|kiss-gicp|gsdf|gsdf-gpu|bufferx|bufferx-gicp|g3reg|g3reg-gicp]\n"
```
(G3Reg 전용 스칼라 플래그는 없으므로 옵션 줄(`:251-253`)은 변경 불필요.)

**(b) `parseAlgo` 람다**(`main.cpp:261-273`). `bufferx-gicp`(`:271`) 뒤에:

```cpp
            else if (v == "g3reg") out = cc::reg::RegAlgo::G3Reg;
            else if (v == "g3reg-gicp") out = cc::reg::RegAlgo::G3RegGicp;
```

**(c) 플래그 파싱**(`main.cpp:289-298`): **변경 불필요**(전용 노브 없음). `--reg-no-refine`
(`:296`)이 G3RegGicp의 체이닝 토글로 그대로 동작. stdout 포맷(confidence `-1` 가드) 무변경.

### 2.6 뷰어 패널 — `src/viewer/viewer.cpp`

**(a) 알고리즘 콤보**(`viewer.cpp:366-379`, `kRegAlgos[]`). `BUFFER-X + GICP`(`:377`) 뒤에:

```cpp
        {"G3Reg", cc::reg::RegAlgo::G3Reg},
        {"G3Reg + GICP", cc::reg::RegAlgo::G3RegGicp},
```
(빌드타임 C++ 의존성이 없으므로 KISS류 매크로 게이트 **불필요** — 항상 노출, 런타임 가용성으로만
결정. `kRegAlgoCount`(`:379`)는 `sizeof` 자동 계산이라 무변경.)

**(b) 상태 변수 / `loadRegDefaults`**: G3Reg 전용 UI 상태가 없으므로 **변경 불필요**
(`viewer.cpp:348-351`, `:383-391`).

**(c) 알고리즘별 파라미터 UI**(`viewer.cpp:1100-1131`): G3Reg 전용 입력칸이 없으므로 추가
블록 불필요. 단 "refine with GICP" 체크박스 게이트(`viewer.cpp:1127-1131`)에 G3Reg를 추가해
체크박스를 노출:

```cpp
            if (alg == cc::reg::RegAlgo::KissMatcher || alg == cc::reg::RegAlgo::KissGicp ||
                alg == cc::reg::RegAlgo::GradientSdfGpu ||
                alg == cc::reg::RegAlgo::BufferX || alg == cc::reg::RegAlgo::BufferXGicp ||
                alg == cc::reg::RegAlgo::G3Reg || alg == cc::reg::RegAlgo::G3RegGicp) {
                ImGui::Checkbox("refine with GICP", &regRefine);
            }
```

**(d) Register 실행 옵션 채우기**(`viewer.cpp:1146-1154`): 신규 필드가 없으므로 변경 불필요
(`ro.refine = regRefine`이 이미 G3RegGicp 체이닝을 제어).

### 2.7 빌드 — `backend/registration/CMakeLists.txt`

`add_library` 소스 목록(`CMakeLists.txt:4-11`)에 1줄 추가:

```cmake
  bufferx/bufferx_backend.cpp
  g3reg/g3reg_backend.cpp)
```

- 추가 링크/패키지 **불필요**. G3Reg 백엔드는 PCD writer(`cloudcropper::io`, 이미
  `CMakeLists.txt:23`에서 PRIVATE 링크)와 POSIX(fork/exec, 표준)만 쓴다.
- PCD 핸드오프는 `CLOUDCROPPER_HAS_PCD` 필요 — `vcpkg`/`gui` 프리셋의
  `VCPKG_MANIFEST_FEATURES`에 이미 `pcd` 포함(`CMakePresets.json:25,39`) → **프리셋/매니페스트
  변경 불필요**. 없으면 백엔드가 `Unsupported` 스텁(§3.0).
- `dev` 프리셋(외부 의존성 0)은 정합 자체를 컴파일에서 제외 → G3Reg도 자동 제외.

### 2.8 테스트 — `tests/cc_tests.cpp`

§5의 **fake CLI 테스트** 1개 추가. stdlib(python3 또는 sh) 스크립트가 G3REG 3줄을 stdout으로
찍게 하고, `CLOUDCROPPER_G3REG_BIN`/`CLOUDCROPPER_G3REG_CONFIG`로 주입 → PCD 기록·subprocess·
파싱·매핑·G3RegGicp 체이닝을 검증. GTSAM/igraph/실제 빌드 불필요.

---

## 3. 백엔드 구현 계약 (`g3reg_backend.cpp`)

### 3.0 빌드 게이트 스텁

`gsdf_gpu.cpp:32-37`과 동형. 단 가드는 **`CLOUDCROPPER_HAS_PCD`**:

```cpp
#if !defined(CLOUDCROPPER_HAS_PCD)
Result<RegResult> run(const PointCloud&, const PointCloud&, const RegOptions&) {
    return makeError(ErrorCode::Unsupported,
                     "g3reg: needs the PCD codec (vcpkg pcd feature) for the handoff");
}
#else
// ... 실제 구현 ...
#endif
```

### 3.1 바이너리·config 해석

```cpp
namespace fs = std::filesystem;

// gsdf_gpu.cpp:44-58 findScript() 패턴 + PATH 폴백.
fs::path findBin(const std::map<std::string,std::string>& cfg) {
    std::error_code ec;
    if (const char* e = std::getenv("CLOUDCROPPER_G3REG_BIN"))
        if (fs::exists(e, ec)) return e;
    if (auto it = cfg.find("bin"); it != cfg.end() && !it->second.empty()
            && fs::exists(it->second, ec)) return it->second;
    // 실행 파일 인접 / 상위 탐색 (gsdf_gpu.cpp:51-56 형태)
    fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        fs::path d = exe.parent_path();
        for (int up = 0; up < 6 && !d.empty(); ++up, d = d.parent_path())
            for (const char* rel : {"cc_g3reg_cli", "g3reg/bin/cc_g3reg_cli", "bin/cc_g3reg_cli"})
                if (fs::exists(d / rel, ec)) return d / rel;
    }
    return {};  // 못 찾음 → execvp("cc_g3reg_cli") PATH 폴백 또는 NotFound (호출부 결정)
}
```

config 경로(외부 G3Reg yaml):

```cpp
std::string g3regConfig(const std::map<std::string,std::string>& cfg) {
    if (const char* e = std::getenv("CLOUDCROPPER_G3REG_CONFIG")) return e;
    if (auto it = cfg.find("g3reg_config"); it != cfg.end() && !it->second.empty())
        return it->second;
    return {};  // 비어도 호출 — 외부 CLI 가 자체 기본 config 를 쓸 수 있음(빈 인자 처리 합의)
}
```

> 바이너리를 못 찾으면(반환 빈 경로 & PATH에도 없음) `ErrorCode::NotFound`로 실패하고
> 메시지에 `set CLOUDCROPPER_G3REG_BIN`을 안내한다(`gsdf_gpu.cpp:104-106` 톤).

### 3.2 임시 PCD 기록 (PCL 불필요)

CloudCropper **자체** PCD writer를 쓴다(`io::PcdWriter` + `io::FileByteSink`). gsdf의
`writeNpz`(`gsdf_gpu.cpp:60-65`)와 동형이되 NPZ → PCD:

```cpp
#include "cloudcropper/io/byte_stream.hpp"   // FileByteSink
#include "cloudcropper/io/pcd.hpp"           // PcdWriter

Result<void> writePcd(const PointCloud& pc, const fs::path& path) {
    io::FileByteSink sink(path.string());
    if (!sink.ok())
        return makeError(ErrorCode::IoError, "g3reg: cannot create " + path.string());
    io::WriteOptions opt;
    opt.fields   = {};                       // xyz만 쓰고 싶으면 아래 주석 참조
    opt.encoding = io::Encoding::Binary;     // 빠르고 정밀(ascii 반올림 회피)
    return io::PcdWriter{}.write(pc, sink, opt);
}
```

- **xyz만 필요**(G3Reg는 PointXYZ; 노멀/색상 불필요, notes §4). `PcdWriter`는 x/y/z를 **항상**
  먼저 emit하고(`pcd.cpp:349-351`), `opt.fields`가 비면 모든 속성을 추가한다(`pcd.cpp:341-346`).
  여분 속성(rgb/normal)이 붙어도 G3Reg `loadPCDFile<PointXYZ>`가 무시하므로 무해하지만,
  **핸드오프 용량/시간 최소화**를 위해 `opt.fields`에 존재하지 않는 더미("")를 주는 대신
  — 더 깔끔하게는 **위치만 복사한 임시 PointCloud**를 만들거나, `opt.fields = {"__none__"}`로
  매칭 0개를 강제(그러면 x/y/z만 남음, `pcd.cpp:352-364`의 `selected()`가 전부 false). 후자를
  권장: 코드 한 줄로 xyz-only 보장.
- 인코딩은 **`Binary`**(AoS) 권장: ascii는 `%.9g`(`pcd.cpp:430`)라 손실 적지만 느리고 크다.
  G3Reg 데모가 `loadPCDFile`로 binary PCD를 읽으므로 호환.
- 단위는 **미터 그대로**(계약서 §5). 스케일 변환 없음.

### 3.3 one-shot subprocess 실행 + stdout 캡처

**기존 헬퍼 없음** — `PythonWorker`는 영속·양방향 IPC라 부적합. `python_worker.cpp`의
fork/pipe/dup2/poll 관용구(`python_worker.cpp:344-428`)를 **단순화**한 self-contained 함수를
백엔드 내부 익명 네임스페이스에 둔다(다른 백엔드가 쓰지 않으므로 공용화하지 않음):

```cpp
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

struct CliResult { int exitCode = -1; std::string stdoutText; bool timedOut = false; };

// argv: {bin, g3regConfig, srcPcd, tgtPcd, nullptr}. stderr -> logPath(=worker.log)
// stdout -> 파이프로 캡처. timeoutSec 초과 시 SIGKILL.
Result<CliResult> runCli(const std::vector<std::string>& args,
                         const std::string& logPath, int timeoutSec) {
    int outPipe[2];
    if (::pipe(outPipe) != 0)
        return makeError(ErrorCode::IoError, "g3reg: pipe() failed");

    // fork 전에 C 문자열 준비(child 는 async-signal-safe 만, python_worker.cpp:362-367 참조)
    std::vector<const char*> argv;
    for (auto& a : args) argv.push_back(a.c_str());
    argv.push_back(nullptr);

    static const bool ign = []{ ::signal(SIGPIPE, SIG_IGN); return true; }();
    (void)ign;

    const pid_t pid = ::fork();
    if (pid < 0) { ::close(outPipe[0]); ::close(outPipe[1]);
                   return makeError(ErrorCode::IoError, "g3reg: fork() failed"); }
    if (pid == 0) {  // child
        ::close(outPipe[0]);
        ::dup2(outPipe[1], 1);                              // stdout -> pipe
        const int logFd = ::open(logPath.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0644);
        if (logFd >= 0) ::dup2(logFd, 2);                   // stderr(glog) -> log
        ::close(outPipe[1]); if (logFd > 2) ::close(logFd);
        // stdin 은 /dev/null
        const int devnull = ::open("/dev/null", O_RDONLY); if (devnull >= 0) ::dup2(devnull, 0);
        ::execvp(argv[0], const_cast<char* const*>(argv.data()));
        _exit(127);                                         // exec 실패
    }
    ::close(outPipe[1]);
    const int outFd = outPipe[0];

    // poll 루프로 stdout 전부 읽기 + 타임아웃 (python_worker.cpp:310-342 단순화)
    std::string out; CliResult cr;
    const double deadline = nowSec() + timeoutSec;
    for (;;) {
        const double remain = deadline - nowSec();
        if (remain <= 0) { ::kill(pid, SIGKILL); cr.timedOut = true; break; }
        struct pollfd pfd{ outFd, POLLIN, 0 };
        const int pr = ::poll(&pfd, 1, static_cast<int>(remain*1000)+1);
        if (pr < 0) { if (errno == EINTR) continue; break; }
        if (pr == 0) { ::kill(pid, SIGKILL); cr.timedOut = true; break; }
        char buf[4096];
        const ssize_t n = ::read(outFd, buf, sizeof buf);
        if (n < 0) { if (errno == EINTR) continue; break; }
        if (n == 0) break;                                   // child closed stdout (EOF)
        out.append(buf, static_cast<std::size_t>(n));
    }
    ::close(outFd);
    int status = 0; ::waitpid(pid, &status, 0);              // 반드시 reap
    cr.exitCode   = (!cr.timedOut && WIFEXITED(status)) ? WEXITSTATUS(status) : -1;
    cr.stdoutText = std::move(out);
    return cr;
}
```

설계 포인트:
- **stdout만** 파이프로 캡처, **stderr(glog)** 는 `<tmpdir>/worker.log`로 — 프로토콜 오염
  방지(gsdf의 stdout 보호와 같은 철학). 에러 시 `lastLines(logPath, 6)`
  (`python_worker.hpp:62`, 공개 함수)로 tail을 메시지에 붙인다.
- **타임아웃 → SIGKILL → waitpid reap**: 좀비 방지. `timeoutSec`은 `g3reg.yaml`의
  `timeout_sec`(기본 600).
- `nowSec()`는 `python_worker.cpp:281-285`와 동일한 steady_clock 헬퍼를 백엔드 내부에 복제.

### 3.4 stdout 3줄 파싱 → RegResult 매핑

```cpp
Result<RegResult> parseG3reg(const CliResult& cr) {
    if (cr.timedOut)
        return makeError(ErrorCode::IoError, "g3reg: cc_g3reg_cli timed out\n" + logTail);
    if (cr.exitCode != 0)
        return makeError(ErrorCode::IoError,
            "g3reg: cc_g3reg_cli exited " + std::to_string(cr.exitCode) + "\n" + logTail);

    std::array<double,16> tf = kIdentity4; bool gotTf = false;
    long inliers = -1; double secs = -1.0;
    std::istringstream ss(cr.stdoutText); std::string line;
    while (std::getline(ss, line)) {
        if (line.rfind("G3REG_TF:", 0) == 0) {
            std::istringstream ns(line.substr(9));
            for (int i = 0; i < 16 && (ns >> tf[i]); ++i) ;       // 16개 파싱
            gotTf = ns ? true : (/* 16개 정확히 읽었는지 카운트 검증 */ true);
        } else if (line.rfind("G3REG_INLIERS:", 0) == 0) {
            inliers = std::strtol(line.substr(14).c_str(), nullptr, 10);
        } else if (line.rfind("G3REG_TIME:", 0) == 0) {
            secs = std::strtod(line.substr(11).c_str(), nullptr);
        }
    }
    if (!gotTf)
        return makeError(ErrorCode::ParseError, "g3reg: no G3REG_TF line in CLI stdout");

    RegResult out;
    out.transform = tf;                 // row-major, target<-source — 그대로(§1.3)
    out.converged = true;               // exit 0 + TF 파싱 성공 = 수렴 간주
    // confidence / normResidual 은 -1 유지 (G3Reg 미제공)
    std::ostringstream d;
    d << "G3Reg: " << (inliers >= 0 ? std::to_string(inliers) : "?") << " inliers";
    if (secs >= 0) { d.precision(3); d << ", " << secs << "s (solver)"; }
    out.detail = d.str();               // 예: "G3Reg: 1234 inliers, 0.42s (solver)"
    return out;
}
```

> **16개 카운트 검증**: 위 의사코드의 `gotTf`는 구현 시 "정확히 16개를 읽었는가"로
> 판정한다(루프 카운터 `i==16` 확인). 16개 미만이면 `ParseError`. gsdf의 `t->array.size()!=16`
> 가드(`gsdf_gpu.cpp:180-181`)와 같은 엄격함.

매핑 표:

| stdout 라인 | RegResult 필드 | 비고 |
|---|---|---|
| `G3REG_TF:` (16 float) | `transform` | 필수. 없거나 ≠16개면 `ParseError` |
| (exit 0 + TF) | `converged = true` | exit≠0 / timeout → 에러 반환 |
| `G3REG_INLIERS:` | → `detail` 문자열 | 공통 metric `inliers`와 별개(§1.5) |
| `G3REG_TIME:` | → `detail` (solver 시간) | `seconds`는 디스패처 wall-clock이 덮음 |
| (없음) | `confidence`/`normResidual` | **-1 유지** |
| `rmse`/`inliers`/`seconds` | (공통 metric) | `registration.cpp:66-72`가 채움 |

### 3.5 GICP 체이닝 (G3RegGicp)

`kiss_backend.cpp:61-70`과 **동일 패턴**. 워커가 아니라 C++ 디스패처/브리지에서 GICP를
`init`으로 체이닝:

```cpp
Result<RegResult> run(const PointCloud& source, const PointCloud& target,
                      const RegOptions& opt) {
    // ... findBin / writePcd(src,tgt) / runCli / parseG3reg -> RegResult coarse ...
    if (!coarse) return coarse;

    if (opt.algo == RegAlgo::G3RegGicp && opt.refine) {
        RegOptions ro = opt;
        ro.algo = RegAlgo::Gicp;
        ro.init = coarse->transform;              // G3Reg 4x4 를 초기값으로
        auto refined = gicp::run(source, target, ro);
        if (refined) {
            refined->detail = coarse->detail + "  ->  " + refined->detail;  // 한 줄 prepend
            return refined;
        }
        // refine 실패 시 coarse 를 그대로 반환(글로벌 결과는 유효)
    }
    return coarse;
}
```

- `gicp::run`은 `#include "../gicp/gicp_backend.hpp"` 필요(kiss_backend.cpp:20 참조).
- `refine` 플래그(`--reg-no-refine` / 뷰어 체크박스)로 끄면 글로벌 단독 결과 반환.

---

## 4. 빌드 & 의존성 전략

### 4.1 빌드타임 = C++ 브리지뿐 (의존성 0 추가)

`g3reg_backend.cpp` 한 파일을 `cloudcropper_registration` 정적 라이브러리에 추가
(§2.7)하는 것이 빌드타임 변경의 **전부**다. 신규 C++ 의존성·vcpkg 포트·FetchContent **없음**.

- PCD 핸드오프는 `CLOUDCROPPER_HAS_PCD`(io, `src/io/CMakeLists.txt:20-22`) 필요 — 없으면
  `Unsupported` 스텁(§3.0). `vcpkg`/`gui` 프리셋은 `pcd` feature 포함(`CMakePresets.json:25,39`).
- subprocess는 POSIX fork/exec(표준). GTSAM/igraph/PCL은 **외부 바이너리 안에만** 존재.

### 4.2 런타임 = 외부 G3Reg 빌드 (main agent 책임)

main agent가 G3Reg를 `/tmp/g3reg-deps`(CMAKE_PREFIX_PATH)로 독립 빌드하고 `cc_g3reg_cli`를
산출한 뒤, 사용자가 1회:

```bash
export CLOUDCROPPER_G3REG_BIN=/path/to/cc_g3reg_cli
export CLOUDCROPPER_G3REG_CONFIG=/path/to/g3reg/configs/<tuned>/gem_pagor.yaml
# 또는 config/g3reg.yaml 의 bin / g3reg_config 키에 기입
```

바이너리/config 부재 시 백엔드가 `ErrorCode`로 깔끔히 실패(빌드는 영향 없음).

### 4.3 선택성(optionality)

- C++ 브리지는 (PCD가 있으면) 항상 컴파일되지만, **런타임에** 바이너리/config가 없으면
  `g3reg::run`이 `NotFound`/`IoError`를 반환 → 정합 빌드를 했어도 G3Reg를 안 쓰면 **비용 0**.
- 뷰어 콤보/CLI에는 항상 노출(빌드타임 C++ 의존성이 없으므로 KISS류 매크로 게이트 불필요),
  실패는 런타임 메시지로 안내.

---

## 5. FAKE-CLI 단위 테스트 (`tests/cc_tests.cpp`)

gsdf/bufferx의 fake-worker 테스트(`cc_tests.cpp:635-698`, `:707-812`)와 **같은 양식**이되,
영속 워커가 아니라 **one-shot CLI**를 흉내 내는 stdlib 스크립트를 만든다. GTSAM/igraph/실제
G3Reg 불필요. 핵심: G3REG 3줄을 stdout으로 찍는 스크립트를 `CLOUDCROPPER_G3REG_BIN`으로 주입.

```cpp
void testG3regCliResult() {
    std::cerr << "[g3reg cli result]\n";
    if (std::system("command -v python3 >/dev/null 2>&1") != 0) {
        std::cerr << "  SKIP: python3 not on PATH\n"; return;
    }
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "cc_g3reg_cli_cfg";
    fs::create_directories(dir);

    // config/g3reg.yaml (얇은 레이어). g3reg_config 는 더미 경로면 충분(가짜 CLI가 무시).
    setenv("CLOUDCROPPER_CONFIG_DIR", dir.c_str(), 1);
    { std::ofstream f(dir / "g3reg.yaml");
      f << "timeout_sec: 30\nrefine: true\ng3reg_config: " << (dir/"fake.yaml").string() << "\n"; }
    { std::ofstream f(dir / "fake.yaml"); f << "# dummy g3reg config\n"; }

    // 가짜 cc_g3reg_cli: argv = [config, src.pcd, tgt.pcd].
    //  - src/tgt .pcd 가 실제로 존재하는지 확인(백엔드가 PCD를 썼다는 증거)
    //  - 첫 인자(config)가 우리가 준 경로인지 확인
    //  - glog 흉내로 stderr 에 노이즈 출력(stdout 오염 안 됨을 검증)
    //  - stdout 에 G3REG 3줄 출력
    const fs::path bin = dir / "fake_g3reg_cli.py";
    { std::ofstream f(bin);
      f << "import sys,os\n"
           "cfg,src,tgt = sys.argv[1], sys.argv[2], sys.argv[3]\n"
           "sys.stderr.write('I0616 glog noise on stderr\\n')\n"   // stdout 오염 금지 검증
           "assert os.path.exists(src) and os.path.exists(tgt), 'pcd missing'\n"
           "assert open(src,'rb').read(5)[:1]==b'#' or True\n"      // PCD 헤더 존재
           "print('G3REG_TF: 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1')\n"
           "print('G3REG_INLIERS: 1234')\n"
           "print('G3REG_TIME: 0.42')\n"; }
    // execvp 가 직접 실행할 수 있게 shebang 래퍼(또는 bin 키를 "python3 fake.py" 로 분해).
    // 가장 단순하게는 CLOUDCROPPER_G3REG_BIN 을 'python3' 로, 첫 인자 앞에 스크립트를
    // 끼우는 대신, 실행권한 + shebang 스크립트를 바이너리로 주입한다:
    fs::permissions(bin, fs::perms::owner_exec, fs::perm_options::add);
    // (shebang '#!/usr/bin/env python3' 를 첫 줄에 넣어야 함 — 위 ofstream 첫 줄로 추가)
    setenv("CLOUDCROPPER_G3REG_BIN", bin.c_str(), 1);

    const cc::PointCloud target = regBlobsCloud();
    cc::PointCloud       source = target;     // identity-aligned

    // (1) G3Reg 단독: 3줄 파싱 -> RegResult 매핑
    {
        cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::G3Reg);
        opt.algo = cc::reg::RegAlgo::G3Reg;
        auto rr = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (rr) {
            CHECK(rr->converged);
            CHECK(rr->confidence < 0.0);                       // 미제공
            CHECK(rr->normResidual < 0.0);
            CHECK(rr->transform == cc::reg::kIdentity4);
            CHECK(rr->detail.find("G3Reg: 1234 inliers") == 0);
        } else std::cerr << "  error: " << rr.error().message << "\n";
    }
    // (2) G3RegGicp: coarse 라인이 GICP refine 라인 앞에 prepend
    {
        cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::G3RegGicp);
        opt.algo = cc::reg::RegAlgo::G3RegGicp; opt.refine = true;
        auto rr = cc::reg::registerClouds(source, target, opt);
        CHECK(static_cast<bool>(rr));
        if (rr) {
            CHECK(rr->converged);                              // 체인된 GICP 에서
            CHECK(rr->detail.find("G3Reg: 1234 inliers") == 0);
            CHECK(rr->detail.find("  ->  ") != std::string::npos);
        }
    }
    // (3) 바이너리 부재 시 깔끔한 실패
    {
        setenv("CLOUDCROPPER_G3REG_BIN", (dir/"does_not_exist").c_str(), 1);
        cc::reg::RegOptions opt = cc::reg::defaultsFor(cc::reg::RegAlgo::G3Reg);
        opt.algo = cc::reg::RegAlgo::G3Reg;
        auto rr = cc::reg::registerClouds(source, target, opt);
        CHECK(!rr);                                            // NotFound
    }

    unsetenv("CLOUDCROPPER_CONFIG_DIR");
    unsetenv("CLOUDCROPPER_G3REG_BIN");
    std::error_code ec; fs::remove_all(dir, ec);
}
```

호출 등록은 gsdf/bufferx 테스트와 같은 블록(`cc_tests.cpp:900-901` 인근)에 `testG3regCliResult();`
추가. 가드는 `#if defined(CLOUDCROPPER_HAS_PCD) && defined(CLOUDCROPPER_HAS_REGISTRATION)`.

> **검증 항목**: ① 백엔드가 src/tgt `.pcd`를 실제로 기록했는가(가짜 CLI의 `os.path.exists`),
> ② config 경로가 첫 인자로 전달됐는가, ③ stderr glog 노이즈가 stdout 파싱을 오염시키지
> 않는가, ④ TF row-major 매핑, ⑤ inliers/time → detail, ⑥ confidence/normResidual=-1,
> ⑦ G3RegGicp 체이닝 prepend, ⑧ 바이너리 부재 시 에러. **torch/GTSAM/igraph 불필요.**
>
> 구현 메모: fake CLI를 `execvp`가 직접 실행하려면 스크립트 첫 줄에 shebang
> `#!/usr/bin/env python3` + 실행권한이 필요하다(위 코드의 `fs::permissions`). 또는
> `runCli`가 `bin`을 공백 분해해 `{"python3","fake.py",...}`로 실행하도록 설계해도 되지만,
> **단순 단일 실행 파일** 가정이 프로덕션과 일치하므로 shebang 방식을 권장한다.

---

## 6. 리스크 / 미해결

1. **외부 빌드(main agent)**: GTSAM 4.1.1 / igraph 0.9.9가 옛 버전이라 GCC 11.4에서 빌드 실패
   가능(계약서 §5). 본 C++ 설계와는 분리된 리스크 — 바이너리만 나오면 통합은 영향 없음.
2. **저밀도 가정**: G3Reg는 옥외 LiDAR·저밀도 설계(notes §4). 근거리 고밀도 hap/hesai에선
   primitive 추출 파라미터(외부 config의 voxel/plane threshold) 튜닝 필요 — `g3reg_config`로
   `configs/hit_ms`(Livox dense) 계열에서 시작(notes §4).
3. **빈 config 인자**: `cc_g3reg_cli`가 빈 `<config.yaml>` 인자를 어떻게 처리할지(자체 기본값
   사용 vs 에러) main agent와 합의 필요 — 본 설계는 "비면 외부 기본 config"를 가정(§3.1).
4. **수렴 판정**: 현재 "exit 0 + TF 파싱 성공 = converged=true"로 단순화. G3Reg가 실패를
   exit code로 신호하지 않고 identity TF + 낮은 inliers로 신호한다면, `cc_g3reg_cli`가
   `FRGresult.valid`를 exit code(또는 4번째 라인 `G3REG_VALID:`)로 노출하도록 계약 확장 고려.
   그 경우 §3.4에 `G3REG_VALID:` 파싱 + `out.converged = valid` 추가(현 설계는 inliers를
   detail로만 보유하므로 공통 metric의 rmse/inliers로 사후 판별 가능).
5. **PCD 용량**: 큰 소스 클라우드의 binary PCD 기록/read I/O 비용. 필요 시 `RegOptions::downsample`
   기반 사전 다운샘플을 백엔드에서 적용해 핸드오프를 줄이는 최적화 가능(MVP는 생략).

---

## 7. 구현 순서 (개발팀, 8단계)

1. **외부 CLI 계약 고정** — main agent와 `cc_g3reg_cli <config> <src.pcd> <tgt.pcd>` →
   stdout 3줄(G3REG_TF/INLIERS/TIME, glog→stderr, exit code) 규약을 확정(§1.3). 빈 config
   인자·수렴 신호(리스크 §3,4) 합의.
2. **공개 API + 디스패처** — §2.1 enum 2 case(`registration.hpp:32` 뒤), §2.2 include +
   `algoName`(`:27` 뒤) + dispatcher(`:62` 뒤). 컴파일러가 비포괄 switch로 누락 안내.
3. **config 배선** — §2.4 `configFileFor`(`config.cpp:91` 옆) + `defaultsFor` case + 신규
   `config/g3reg.yaml`(bin/g3reg_config/timeout_sec/refine).
4. **새 백엔드** — `g3reg/g3reg_backend.{hpp,cpp}` 작성: §3.0 가드 → §3.1 findBin/config →
   §3.2 `writePcd`(io::PcdWriter) → §3.3 `runCli`(fork/exec/poll) → §3.4 3줄 파서 →
   §3.5 GICP 체이닝. `CMakeLists.txt:11`에 소스 1줄(§2.7).
5. **CLI** — §2.5 usage 문자열 + `parseAlgo` 2 case(`main.cpp:271` 뒤). 전용 플래그 없음.
6. **뷰어** — §2.6 `kRegAlgos` 2 엔트리(`viewer.cpp:377` 뒤) + refine 체크박스 게이트 확장
   (`:1127-1131`). 전용 파라미터 UI 불필요.
7. **테스트** — §5 fake-CLI 테스트(`testG3regCliResult`)를 `cc_tests.cpp`에 추가하고
   `:900` 인근에서 호출. PCD 기록·subprocess·파싱·매핑·체이닝·바이너리부재를 GTSAM 없이 검증.
8. **벤치/문서** — main agent가 외부 빌드 후 `register --reg-algo g3reg[-gicp]` end-to-end,
   `scripts/reg-bench.py`에 `g3reg`/`g3reg-gicp` 안내(파싱 무변경), `scripts/inlier-rmse.py`로
   KISS-Matcher와 A/B(성공률·inlier·시간·중첩부 RMSE). README에 `CLOUDCROPPER_G3REG_BIN/CONFIG`
   런타임 안내.

---

## 부록 A — 변경 파일 체크리스트

| 파일 | 변경 |
|---|---|
| `backend/registration/include/.../registration.hpp` | enum 2 case(`:32` 뒤). RegOptions/RegResult **무변경** |
| `backend/registration/common/registration.cpp` | include(`:9` 블록), `algoName`(`:27` 뒤), dispatcher(`:62` 뒤) |
| `backend/registration/common/config.cpp` | `configFileFor`(`:91` 옆), `defaultsFor` case(`:127` 옆, 본문 `break;`) |
| `backend/registration/g3reg/g3reg_backend.hpp` | 신규(gsdf_gpu.hpp 형식) |
| `backend/registration/g3reg/g3reg_backend.cpp` | 신규: PCD 기록 + one-shot subprocess + 3줄 파싱 + GICP 체이닝 |
| `backend/registration/CMakeLists.txt` | 소스 1줄(`:11`) |
| `config/g3reg.yaml` | 신규(bin/g3reg_config/timeout_sec/refine — 얇은 레이어) |
| `src/app/main.cpp` | usage(`:250`), parseAlgo 2 case(`:271` 뒤). 플래그 무변경 |
| `src/viewer/viewer.cpp` | kRegAlgos 2 엔트리(`:377` 뒤), refine 체크박스 게이트(`:1127-1131`) |
| `tests/cc_tests.cpp` | fake-CLI 테스트 1개(`:707` 양식) + 호출(`:901` 인근) |
| `scripts/reg-bench.py` | 안내 문자열만(파싱 무변경) |

> **빌드 게이트**: `CLOUDCROPPER_HAS_PCD`(io, `src/io/CMakeLists.txt:22`) — `vcpkg`/`gui` 프리셋에
> 이미 포함(`CMakePresets.json:25,39`). **프리셋/매니페스트 변경 불필요.** G3Reg는 빌드타임
> 의존성 0(순수 subprocess).
