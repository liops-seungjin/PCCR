# RAP 통합 계약서 (AUTHORITATIVE — 모든 팀이 이 문서를 그대로 따른다)

> 본 문서는 dev팀·review팀이 공유하는 **권위 사양**이다. 설계 문서
> `docs/design/08-rap-backend.md`는 본 계약과 **정확히 일치**해야 한다(enum 이름,
> 옵션 이름, 파일 목록, 워커 프로토콜, config 키, 폴백 동작). 업스트림 정찰 사실은
> `docs/design/_rap-upstream-notes.md` 참조.

## 1. 알고리즘

RAP = *"Register Any Point: Scaling 3D Point Cloud Registration by Flow Matching"*
(arXiv:2512.01850, 2025, PRBonn, MIT). **Rectified Point Flow (RPF, NeurIPS 2025)**
위에 구축. 학습 기반, zero-shot, **글로벌 정합기**(초기 추정 불필요).

CloudCropper의 pairwise 사용:
- `[target, source]`를 두 개의 "part"로 모델에 넣는다.
- RAP가 N번의 flow-matching step으로 정합된(assembled) 클라우드를 생성한다.
- part별 4×4 pose를 **SVD/Procrustes**로 복원한다.
- pose를 **타깃(reference) 기준 상대값**으로 만든다:
  `T = inv(T_target) @ T_source` → row-major **target<-source**, 즉
  `RegResult::transform` 규약(`p_target = T * p_source`)과 정확히 일치.

## 2. 통합 방식 — BUFFER-X 패턴 복제 (영속 Python 워커)

RAP는 torch+lightning 체크포인트 모델 + Python demo.py/sample.py API → **BUFFER-X와
같은 형태**다. 따라서:

- `PythonWorker`(영속, lazy-spawn, JSON-lines) + **NPZ 핸드오프**를 쓴다.
- **G3Reg의 one-shot-subprocess 패턴이 아니다.**
- **GRACEFUL DEGRADATION 필수**(bufferx_worker.py와 동일): RAP 코어 / 가중치 /
  flash-attn이 설치되지 **않아도** 워커는 `ready`로 떠야 하고, 그때 `register` op은
  **IDENTITY transform + `converged:false` + 설명용 `note`** 를 반환한다.
  inlier/quality 수치를 **절대 날조하지 않는다.**
- **네이티브 C++ 폴백은 없다.** 이 graceful degradation 덕분에 무거운 RAP 스택 없이도
  이 환경에서 백엔드를 빌드·테스트할 수 있다.

## 3. enum / dispatcher

- `registration.hpp`: `RegAlgo::Rap`, `RegAlgo::RapGicp` 추가 (`G3RegGicp` **뒤**에 배치).
- `registration.cpp` `algoName()`: `"RAP"`, `"RAP + GICP"`.
- `registration.cpp` dispatcher: `case Rap/RapGicp -> rap::run(source, target, opt)`.

## 4. RegOptions

추가 필드(단 하나):
```cpp
float rapVoxel = 0.0f;  // input downsample voxel override (0 = take yaml / worker auto)
```
- `bufferxVoxel`을 **정확히 미러**한다. 구조체 주석도 `bufferxVoxel`처럼 문서화.
- 그 외 RAP 필드는 **추가하지 않는다** — 나머지 노브는 전부 yaml로 흐른다.

## 5. config (config.cpp / config.hpp)

- `configFileFor()`: `Rap/RapGicp -> "rap.yaml"`.
- `defaultsFor()`: 공통 refine은 switch 위에서 이미 로드됨 + switch에 case 추가:
  ```cpp
  case RegAlgo::Rap:
  case RegAlgo::RapGicp:
      opt.rapVoxel = getF(kv, "voxel_size", opt.rapVoxel);
      break;
  ```

## 6. 백엔드 파일: `backend/registration/rap/`

### 6.1 `rap_backend.hpp`
- 네임스페이스 `cc::reg::rap`.
- `Result<RegResult> run(const PointCloud&, const PointCloud&, const RegOptions&)`.
- 헤더 doc-comment는 `bufferx_backend.hpp` 스타일: flash-attn/Blackwell 배포 caveat +
  영속 워커 + NPZ 핸드오프 + RapGicp가 C++에서 GICP 체인 명시.

### 6.2 `rap_backend.cpp` — `bufferx_backend.cpp`를 가깝게 미러
- `CLOUDCROPPER_HAS_NPZ` 게이트. `#if !defined ->` Unsupported 에러
  `"rap: needs the NPZ codec (vcpkg npz feature) for the handoff"`.
- `findScript()`: env `CLOUDCROPPER_RAP_SCRIPT`(존재해야 함), 그다음 상대경로
  `"backend/registration/rap/python/rap_worker.py"`, 그다음 exe 기준 탐색(6단계 상위).
- yaml `weights_dir` 키 -> lazy spawn 전에 `setenv CLOUDCROPPER_RAP_WEIGHTS`.
- static `TempDirGuard`(`cc_rap_<pid>`)를 static `PythonWorker` **앞에** 선언 → 워커(및
  worker.log)가 디렉터리 제거 전에 소멸.
- static `PythonWorker`: `python=get("python","python3")`, script, `logFile=dir/worker.log`.
- `writeNpz(source/target)` -> `source.npz`/`target.npz` (bufferx의 writeNpz 관용구 재사용).
- `voxelSize = opt.rapVoxel>0 ? to_string(opt.rapVoxel) : get("voxel_size","0.0")`.
- target 캐시 키: `hex(fnv1a64(target positions bytes)) + ":" + voxelSize + ":" + device
  + ":" + model-variant` (RAP 모델이 바뀌면 캐시 키도 바뀌도록).
- params JSON: `"source"`, `"target"`, `"target_key"`, `"device"`, `"voxel_size"`,
  `<voxelSize>`, `"refine":false`, 그다음 reserved가 아니고 비어있지 않은 **모든** yaml 키를
  `jsonScalar()`로 verbatim forward. `reserved = {python, timeout_sec, weights_dir,
  device, voxel_size, refine}`.
- `timeoutSec`는 `get("timeout_sec","900")` (flow 모델은 무거움; bufferx의 600보다 큰 기본값).
- `worker.call("register", params, timeoutSec)`; `result.transform`(16 double) 파싱 ->
  `RegResult.transform`; `converged`는 `result.converged`; confidence/normResidual은 **-1 유지**.
  detail은 `"RAP (worker, <device>): ..."` + 선택적 inliers/seconds/note, bufferx처럼
  `result.note` 표면화. `cache_hit` -> `", cached target"`.
- `RapGicp + opt.refine`: `gicp::run`을 `out.transform`으로 seed해 체인; 성공 시 detail 앞에
  `"  ->  "` prepend 후 refined 반환; 실패 시 coarse 반환.

## 6.5 REALIZED 갱신 (구현 완료, 2026-06-17 — 설계↔구현 일치)

설계의 "flash-attn HARD blocker"는 **순수-torch shim**으로 해소되었고 실제 추론이
구현되었다. 본 절이 §7-§9의 실현 사실을 갱신한다(자세히는 `08-rap-backend.md` §0.2).

- conda env `rap`(`/home/sjjung/miniconda3/envs/rap/bin/python`, **torch 2.7.0+cu128**)으로
  RTX 5060/Blackwell(sm_120)에서 동작. flash-attn / pytorch3d는 **설치하지 않는다**.
- 순수-torch shim `backend/registration/rap/python/_shims/`: SDPA drop-in `flash_attn.py`
  (FLASH/EFFICIENT 강제) + 순수-torch `pytorch3d/`(ball_query/FPS/chamfer 실제, viz/eval stub).
  워커가 `_shims`/`rap_upstream`/`rap_upstream/dataset_process`를 `sys.path`에 self-insert →
  **C++에서 PYTHONPATH env 불요**. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 설정.
- 기본 모델 **`rap_12_po`**(points-only, mini-SpinNet skip), checkpoint `rap_model_12.ckpt`.
- 영속 모델 로드(1회) + 호출별 추론: hydra compose `RAP_inference`(overrides model=rap_12_po,
  ckpt_path=abs) → `sample.py:setup()`로 모델 상주. 호출별로 PLY 기록 →
  `demo.py:process_point_clouds(feature_extraction_on=False, use_random_downsample=True,
  target_points_per_scan≈8000)` → 호출별 datamodule+trainer 재생성 후 `trainer.test` →
  `*_<part>_transform.txt` 읽어 `inv(T_target)@T_source` 반환. per-call I/O는 `/tmp`가 아닌
  `python/rap_worker_runs/`에.
- graceful degradation 유지(shim/코어/가중치 부재 시 identity+converged:false+note).

## 7. 워커: `backend/registration/rap/python/rap_worker.py` — `bufferx_worker.py` 미러

- 모듈 레벨: **STDLIB ONLY**(json/os/sys/time/traceback). torch 없이도 프로토콜
  부트스트랩이 동작해야 함.
- stdout hijack 최우선(dup fd1 -> private proto; fd1 -> stderr).
- 핸드셰이크 이벤트: `{"event":"loading","pid"}` 그다음
  `{"event":"ready","pid","device","torch","rap":0|1}`;
  numpy/torch **import 자체가** 실패할 때만 `{"event":"fatal","error"}` + exit 1.
- Ops:
  - `ping` -> `{device, torch, cuda, rap}`.
  - `register` -> `{transform[16], converged, device, seconds, cache_hit,
    + 사용 가능할 때만 진짜 quality 키}`.
  - `shutdown` -> `{}` 후 exit 0.
  - 루프는 op 실패에도 생존(`{"ok":false,...}`); stdin EOF -> exit 0.
- `register`: `np.load` source/target `"xyz"`; RAP 파이프라인 best-effort 실행:
  voxel downsample -> (points-only 모델이 아니면 mini-SpinNet feature) -> flow sampling ->
  SVD pose recovery -> 4×4 relative-to-target. 코어/가중치/flash-attn 부재 시 ->
  identity + `converged:false` + note (**inlier 날조 금지**).
- `_RESERVED` set은 bufferx 미러 + RAP 전용 reserved 키.
- `_RAP_KW`: SAFE per-call override의 타입 테이블. 예:
  `n_generations:int, inference_sampling_steps:int, des_r:float, voxel_ratio:float,
  allocation_method:str, min_points_per_part:int, max_points_per_part:int, model:str`.
  (bufferx의 num_scales 같은 lock-step caveat가 있으면 문서화.) 미지의 키 -> stderr 경고,
  무시. 브리지에서 오는 `refine`은 **항상 false**(GICP는 C++ 측).
- 벤더링 업스트림은 `./rap_upstream/`(sys.path)에, 가중치는 `./rap_upstream/weights/`에 기대
  (REALIZED: `_shims`/`rap_upstream`/`rap_upstream/dataset_process`를 워커가 self-insert).
- `--oneshot <source.npz> <target.npz>` 디버그 모드.
- REALIZED: `_RAP_KW`에 `target_points_per_scan:int` 추가(8GB용 per-cloud 다운샘플 예산).
  `model` per-call override는 상주 checkpoint를 바꾸지 못하므로(시작 시 1회 로드) 실제
  추론에는 시작 `model`과 lock-step 유지 — 다른 variant 지정 시 fallback note.

## 8. config/rap.yaml — bufferx.yaml 스타일 헤더 주석 + 키

REALIZED 기본값(실제 `config/rap.yaml`):
```yaml
python: /home/sjjung/miniconda3/envs/rap/bin/python   # conda env `rap` (torch 2.7+cu128, sm_120)
device: cuda
timeout_sec: 900
weights_dir:           # 비우면 워커가 ./rap_upstream/weights/ 탐색
voxel_size: 0.0        # 0 = adaptive (voxel_ratio drives per-part counts)
voxel_ratio: 0.05
des_r: 5.0             # mini-SpinNet feature radius (meters) — rap_12_po에서는 미사용
n_generations: 1
inference_sampling_steps: 10   # flow steps
model: rap_12_po       # 기본 points-only (rap_10 M | rap_12 L | rap_12_po PO) — REALIZED default
target_points_per_scan: 8000   # per-cloud 랜덤 다운샘플 예산 (8GB용)
allocation_method: voxel_adaptive
min_points_per_part: 200
max_points_per_part: 20000
refine: true
```
- 주석은 `python`/`timeout_sec`/`weights_dir`를 제외한 **모든** 키가 워커로 verbatim
  forward됨을 경고. **flash-attn/Blackwell blocker는 순수-torch shim으로 해소**(§6.5)되어
  flash-attn/pytorch3d를 설치하지 않는다는 사실로 갱신.

## 9. python/requirements.txt + download_weights.sh + VENDORED.md

REALIZED(실제 사용 환경): conda env `rap` = `/home/sjjung/miniconda3/envs/rap/bin/python`,
**torch 2.7.0+cu128**(RTX 5060/Blackwell sm_120), + diffusers/lightning/hydra/omegaconf/
open3d/h5py/scipy/trimesh/sklearn/einops. **flash-attn / pytorch3d는 의도적으로 미설치** —
순수-torch shim(`_shims/`)으로 대체. (설계 시점의 torch 2.5.1/cu124 + flash-attn 2.7.4.post1
+ pytorch3d 0.7.8 wheel 격자는 **사용하지 않는다**.)
- **HEADLESS-TRIM**: gradio, mitsuba, wandb, matplotlib 제거.
- `download_weights.sh`: HF `YuePanEdward/RAP`(`rap_model_10.ckpt`, `rap_model_12.ckpt`,
  `mini_spinnet_t.pth`) 또는 ipb.uni-bonn.de `weights.zip` → `rap_upstream/weights/`.
- `VENDORED.md`: PRBonn/RAP 전체가 `rap_upstream/`에 벤더링됨, MIT, 순수-torch shim 접근
  (flash-attn/pytorch3d 미설치), 코어/가중치/shim 부재 시 워커 graceful degrade 설명.

## 10. CLI: src/app/main.cpp

- `parseAlgo()`: `else if (v == "rap") out = RegAlgo::Rap;` 및
  `"rap-gicp" -> RegAlgo::RapGicp;` (g3reg 케이스 옆에 배치).
- usage 문자열(알고리즘 목록 줄 + 플래그 줄): 알고리즘 목록에 `rap|rap-gicp` 추가,
  플래그 줄에 `--reg-rap-voxel V` 추가.
- 옵션 파싱 루프: `else if (a == "--reg-rap-voxel") ro.rapVoxel = std::stof(next());`.

## 11. Viewer: src/viewer/viewer.cpp

- `kRegAlgos[]`: G3Reg 엔트리 **뒤**에 `{"RAP", RegAlgo::Rap}`,
  `{"RAP + GICP", RegAlgo::RapGicp}` 추가.
- voxel-slider 게이트(현재 BufferX/BufferXGicp)에 Rap/RapGicp 추가 → rap voxel override
  슬라이더 노출; `ro.rapVoxel`을 regBufferxVoxel-상당 상태에 연결(regBufferxVoxel을 미러하는
  `regRapVoxel` state 추가, 또는 기존 voxel state 재사용 — bufferxVoxel이 loadRegDefaults에서
  로드되고 호출 전에 적용되는 방식 그대로 따른다).
- "global, no init/no max-corr" 게이트 블록에 bufferx와 나란히 Rap/RapGicp 추가.

## 12. Tests: tests/cc_tests.cpp

bufferx fake-worker e2e 테스트를 미러(torch 불필요): stdlib fake `rap_worker.py`가
`{"event":"ready",...,"rap":0}`와 transform[16]+converged+device(+note)를 담은 register
응답을 emit. 검증:
1. 평문 Rap 워커 -> RegResult 매핑 (detail이 `"RAP (worker, fake)"`로 시작).
2. RapGicp가 coarse 라인을 GICP refine 라인 앞에 prepend.
3. identity-fallback 정직성(note 표면화, 실제 정합으로 오인 금지).
4. 워커가 받는 `refine:false`.
- **반드시 프로세스의 첫 rap 호출**(PythonWorker는 함수 로컬 static).
- 이 파일의 기존 bufferx 워커 테스트 구조를 정확히 따른다.

## 13. CMake: backend/registration/CMakeLists.txt

`rap/rap_backend.cpp`를 `cloudcropper_registration` STATIC 소스 목록에 추가하고
앞쪽 주석에 `rap/` 언급.

## 14. 빌드 / 검증 (이 환경)

- Configure preset은 `"vcpkg"`; build dir `build/vcpkg` 이미 존재.
- 빌드: `cmake --build build/vcpkg --target cloudcropper cc_tests` (incremental).
- 실행: `./build/vcpkg/tests/cc_tests` — 새 RAP fake-worker 테스트 + 기존 모든 테스트 통과.
- C++ fake-worker 테스트는 torch 없이 프로토콜/매핑을 커버(불변).
- REALIZED: 실제 RAP 경로는 conda env `rap`로 검증한다 — `08-rap-backend.md` §0.3의
  `--oneshot`(Livox demo pair, 회전 ≈7-8°/병진 ≈9 m 기대) 명령으로 확인.
- 기존 테스트 회귀 금지. bufferx/g3reg 백엔드의 주석 밀도·네이밍·관용구 스타일을 정확히 일치.
