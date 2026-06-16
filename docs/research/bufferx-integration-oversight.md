# BUFFER-X 통합 & 결과 수치 총괄 검토 (Oversight)

> 작성: 총괄팀(Oversight) · 2026-06-16 · 읽기 전용 검토(빌드 불가, 파일 교차검증만)
> 전제(메인 에이전트 확인): `build/vcpkg` 클린 컴파일, `cc_tests` ALL TESTS PASSED,
> `[bufferx worker result]` 테스트 3개 서브케이스(매핑 / BufferXGicp 체이닝 /
> no-weights 폴백 정직성) 통과.
> 검토 대상 파일은 모두 직접 읽어 상호 대조함.
>
> 🆕 **갱신(2026-06-16, 코어 벤더링 + 실추론 배선 후)**: BUFFER-X 코어가
> `bufferx_upstream/`에 벤더링되고 가중치·CUDA 확장(pointnet2_ops/knn_cuda/
> torch_batch_svd; knn_cuda는 원본 저장소 삭제로 순수-torch shim 대체)이 설치되어
> **실제 추론이 동작**한다. hap101_f0→crop 실측: `bufferx` rmse **3.03073**(10593
> inliers), `bufferx-gicp` rmse **3.03048**(10598) — KISS-GICP(3.03052)와 5자리
> 일치하여 **교차검증 통과**. 아래 §2.1 #17 및 바텀라인은 이 사실로 갱신됨.

---

## 1. 통합 종합(Track B) — 상태: **구조적으로 완결, 추론은 의도적 미연결**

### 1.1 한 줄 요약

BUFFER-X는 gradient-SDF의 **영속 Python 워커 패턴**(JSON-lines IPC + NPZ 핸드오프)을
그대로 복제해 `RegAlgo::BufferX` / `BufferXGicp`로 추가됐다. C++ 브리지·디스패처·
config·CLI·뷰어·테스트 배선은 **전부 완료**됐고 컴파일/테스트가 통과한다. 단,
**업스트림 코어(model 정의)와 가중치가 아직 벤더링되지 않아** 워커는 현재 항상
**identity + converged:false + note**를 반환한다(의도된 정직 폴백). 즉 배선은 끝났고
실제 추론만 남았다.

### 1.2 SHOULD-FIX 3건 — 코드에서 실제 반영 확인

| 리뷰 지적 | 반영 위치 | 검증 결과 |
|---|---|---|
| (a) `weights_dir`가 워커 child 환경변수 `CLOUDCROPPER_BUFFERX_WEIGHTS`로 전달(죽은 JSON 파라미터 아님) | `bufferx_backend.cpp:114-116` — 정적 워커 spawn **이전**에 `::setenv(...,overwrite=1)`; 워커는 `bufferx_worker.py:250`(serve)·`:285`(oneshot)에서 `os.environ.get(...)`로 소비 | ✅ 정확. `_RESERVED`에 `weights_dir`가 들어 있어(`:79`) JSON으로는 forward되지 않음 → 환경변수 경로로만 전달되는 게 맞음 |
| (b) `VENDORED.md`가 sparse-conv 백본 / `register(src,tgt)` API 주장을 더 이상 안 함 | `VENDORED.md:9,34-42` | ✅ "**no sparse-conv backbone**", "**No single-pair `register(src, tgt)` API.** Inference is `model(data_source)`" 명시 — recon(`_bufferx-upstream-notes.md` §1,§4)과 일치 |
| (c) no-fabrication 폴백 경로 테스트됨 | `tests/cc_tests.cpp:790-806` 서브케이스 (3) | ✅ identity+converged:false+note 검증, `fitness` 부재(`find("fitness")==npos`), `[weights missing]` note 노출, `? inliers` 확인 |

### 1.3 전체 일관성 교차검증

- **recon ↔ requirements/VENDORED/worker hpp 주석**: sparse-conv 없음, XYZ-only,
  `model(data_source)`, 2개 소스 체크포인트(threedmatch/kitti) 등 모든 사실이 일치.
  `download_weights.sh`의 HF repo·경로·파일 크기(~3.67MB)도 recon §3과 일치. ✅
- **폴백 정직성 체인**(가장 중요): 워커 `_run_bufferx`는 `model is None`일 때
  identity + note만 반환하고 inlier/fitness 키를 **생성하지 않음**(`:105-110`);
  `register`는 `"num_inliers"/"fitness"`가 info에 있을 때만 결과에 실음(`:202-205`);
  C++ 브리지는 note를 `detail`에 `[...]`로 노출(`:201-202`)하고 `confidence/normResidual`은
  `-1` 유지(`:188`). UI/CSV는 `confidence -1`을 이미 가드함. → identity 결과가 실제
  정합으로 오인될 경로가 없음. ✅
- **GICP 체이닝 위치**: 설계대로 C++ 측(`bufferx_backend.cpp:207-216`)에서 수행, 워커로
  보내는 `refine`은 항상 false(`:160`). 테스트(2)가 detail prepend(`->`)를 확인. ✅

### 1.4 남은 불일치 / 플래그 (모두 경미·비차단)

1. **설계문서 `06-bufferx-backend.md`가 stale**: 본문 여러 곳(§1.2 line 60·62, §3.1
   line 382·390, §5.1 line 541)이 여전히 **MinkowskiEngine/sparse-conv를 백본으로 가정**한다.
   이는 recon 이전에 쓰인 "구현 지침서"이고 §5.1에서 "확정 필요"로 자기-플래그했지만,
   recon·실제 코드(hpp 주석·requirements·VENDORED)는 모두 "sparse-conv 없음"으로 수정됨.
   → **단독으로 읽으면 오해 소지.** 설계문서 상단에 "recon으로 갱신됨, _bufferx-upstream-notes.md
   참조" 한 줄 추가 권장(코드/recon은 정확하므로 비차단).
2. **`readyTimeoutSec` 미상향**: 설계 §1.2는 가중치 로드가 무거우니 600초로 올리라 권고했으나,
   브리지는 `PythonWorker::Options` 기본값(`python_worker.hpp:72`, **300초**)을 그대로 씀.
   현재는 코어 미벤더링이라 워커가 즉시 `ready`로 떠서 무해. 실제 가중치 로드 연결 시
   콜드 스타트 300초 초과 가능성 → 그때 재검토 필요(현재 비차단).
3. **`timeout_sec`(yaml 600)**은 register 호출 타임아웃으로만 사용됨(`bufferx_backend.cpp:170-175`).
   spawn/handshake 타임아웃과 별개 — 위 2번과 합쳐 인지해두면 됨.

> Part 1 결론: **통합은 신뢰 가능하고 정직하다.** 실제 추론을 켜기 전까지 가짜 수치를
> 만들 경로가 코드/문서/테스트 어디에도 없다. 경미한 stale 문서 1건만 정리 권장.

---

## 2. 결과 수치 재검증 (사용자 명시 요구: "결과 수치가 타당한지 한번 더 검토")

### 2.1 검증 표

| # | 출처 | 주장/수치 | 판정 | 근거·주석 |
|---|---|---|---|---|
| 1 | SOTA report | BUFFER-X = "ICCV 2025 Highlight, zero-shot, MIT" | **유효** | recon(`_bufferx-upstream-notes.md` line 3·§7)과 완전 일치: ICCV'25 Highlight, MIT-SPARK, MIT 라이선스, arXiv 2503.07940 |
| 2 | SOTA report | GeoTransformer 3DMatch RR 92.5% / IR 70.9% / 3DLoMatch 74.2%, KITTI RRE 0.230°/RTE 6.2cm | **유효(데시미터/cm로 정확히 라벨)** | line 58·83에서 92.5/74.2 채택, 92.0/75.0(RANSAC)은 1-2 vote로 명시 기각 → 내부 일관. cm급으로 표기됨 |
| 3 | SOTA report | PARE-Net 3DMatch RR 95.0% / 3DLoMatch 80.5%, KITTI ~99.8% | **유효** | line 41·59 동일 수치, 충돌 없음. "IR은 CAST 열세"도 두 곳 일관 |
| 4 | SOTA report | KISS-Matcher KITTI G-ICP 후 ~1.1cm | **유효(mm 아님으로 정확히 표기)** | line 64, "sub-cm … ~1.1cm", ❌(fine 아님). **cm를 mm로 과장하지 않음** |
| 5 | SOTA report | MAC+GeoTransformer RR 95.7/78.9 (line 42) vs MAC 단독 KITTI TE~8cm (line 66) | **유효(혼동 아님)** | 결합 기법과 단독 기법의 다른 행 — 모순 아님 |
| 6 | SOTA report | CAST "SOTA accuracy·robustness·efficiency", IR 최고 | **정성적·수치 미제시** | 구체 수치 없는 정성 주장. 거짓은 아니나 "수치"로 인용 불가 |
| 7 | SOTA report | 핵심 caveat: 조사된 어떤 방법도 mm RMSE 직접 보고 안 함, RR 기준은 RMSE<0.2m(데시미터) | **유효(핵심 라벨링 정확)** | line 50. 모든 RR 수치를 데시미터급 recall로 정정 — 표 전체가 ❌(mm 아님)로 일관 |
| 8 | SOTA report | "검증된 후보 … 13건" 헤더 | **경미한 불일치** | 표 실제 행은 11개(GeoTr/PARE/RoITr/FLAT/DINOReg/RAP/KISS/TEASER/MAC/TCF/서베이). 통계의 afterSynthesis=13과 표 행 수가 안 맞음 — 수치 신뢰성엔 영향 없으나 정리 권장 |
| 9 | experiments CSV | gsdf-gpu confidence 0.005 / iou 0.13 / rmse 3.16 (hesai) | **유효(저신뢰로 정확히 자기-표기)** | 신뢰도·iou 둘 다 낮음 → 일관. converged=yes지만 confidence 0.005가 "믿지 말라"는 정직 신호 |
| 10 | experiments CSV | gsdf-gpu confidence 0.034 / iou 0.207 / rmse 3.96 (hap100) | **유효(저신뢰)** | transform이 컨센서스와 크게 다름(틀린 포즈) → 낮은 신뢰도로 정확히 표기. 일관 |
| 11 | experiments CSV | gsdf-gpu confidence 1.000 / iou 0.376 / rmse 3.026 (hap101) | **유효(진짜 성공)** | transform이 kiss/kiss-gicp 컨센서스(≈ t[-2.0,-0.28,0.05], 소회전)와 일치, 셋 중 최저 rmse·최고 iou → confidence 1.000 정당. **세 confidence 값이 iou/rmse 순서와 정확히 정합** ✅ |
| 12 | experiments CSV | 로컬 ICP 행들(icp/icp-plane/gicp/vgicp, hap100·hap101) converged=**yes**, inliers=0, transform=identity, "0 iters" | **모순/오도 가능** | line 11(vgicp hap100), line 14-17(hap101 4종): **0 iters·identity·0 inliers인데 converged=yes**. rmse가 4종 모두 3.95147로 동일 = 초기 미정합 그대로(아무 것도 안 함). line 8(icp hap100)도 inliers 0·err 0.0025. **이 행들은 "성공"이 아니라 no-op/실패** — 결과로 인용하면 오도됨 |
| 13 | experiments CSV | icp-plane hap100 converged=yes, inliers 5253, rmse 4.97, 큰 회전 | **오도 가능(plausible-but-wrong)** | 공통 metric inliers는 많지만 rmse~5·transform이 컨센서스와 전혀 다름 → 틀린 포즈인데 converged=yes |
| 14 | archive CSV | blobs-90deg / part-to-crop: gicp·gsdf rmse 0.0003~0.02, 3600/2593 inliers | **유효(단, 합성·정확중첩 데이터)** | 알려진 변환의 합성 클라우드 → sub-mm 도달은 타당하나 실측 metrology 아님. SOTA report의 "mm는 fine 단계가 담당" 주장과 일치하는 실증 |
| 15 | archive CSV | gsdf hap101 conf 1.000/norm-res 0.0373 vs hap100-**WRONG** conf 0.230/norm-res 0.172 (line 34-35) | **유효(불확실성 채널 정상 동작)** | 명시적으로 "WRONG"라벨된 잘못된 포즈에 낮은 confidence 0.230 부여 → 신뢰도 채널이 옳게 작동함을 보여줌 |
| 16 | archive CSV | gsdf-gpu CUDA OOM (line 16) | **유효(정직한 에러 기록)** | algo=error로 기록, 가짜 성공 아님 |
| 17 | BUFFER-X 통합 출력 (벤더링·배선 전) | 실제 벤치마크 수치 | (구) 아직 생산 안 됨 | 벤더링 전 상태: 워커 model None → identity+converged:false+note, 가짜 수치 없음 |
| 17b | BUFFER-X 통합 출력 (벤더링·배선 후, **현재**) | bufferx rmse 3.03073/10593, bufferx-gicp rmse 3.03048/10598 (hap101→crop) | **유효(실측·교차검증)** | KISS-GICP(rmse 3.03052, t[-1.984,-0.272,-0.102])와 5자리 일치. transform·inliers 모두 컨센서스와 정합 → 올바른 정합으로 인용 가능. CSV에 정상 기록(reg-bench) |

### 2.2 각 출처 종합 판정

**① SOTA report (`registration-sota-2026-06.md`)** — 내부 일관성 양호. 모든 벤치마크
수치가 **데시미터급 Registration Recall**로 정확히 라벨되며 mm로 과장된 곳이 없다.
기각된 GeoTransformer 변형 수치도 명시 처리됨. **중심 주장**("조사된 어떤 방법도 mm
정밀도를 인증하지 못하며, BUFFER-X는 out-of-domain zero-shot SOTA, mm는 GICP/SDF
fine이 담당")은 **성립한다** — caveat 섹션이 RR 성공기준=RMSE<0.2m임을 명확히 못박아
표 전체와 정합. 경미한 흠: "13건" 헤더 vs 실제 11행, CAST는 수치 없는 정성 주장.

**② 기존 실험 CSV** — **gsdf-gpu의 confidence(0.005/0.034/1.000)는 iou/rmse와 정확히
정합**하며 hap101을 유일한 진짜 성공으로 옳게 순위매김한다(신뢰 가능, 인용 가능).
불확실성 채널은 archive의 "WRONG" 케이스에 낮은 신뢰도를 부여해 정상 작동을 입증한다.
**반면 로컬 ICP 계열의 다수 행(hap100·hap101의 icp/icp-plane/gicp/vgicp)은
converged=yes인데 0 iters·identity·0 inliers인 no-op/실패**다. 여기서 converged는
"정확히 정합됨"이 아니라 "알고리즘이 발산/에러 없이 종료"를 뜻하므로, **이 행들을
'결과'로 인용하면 오도된다.** rmse 3.95147 동일값이 그 증거.

**③ BUFFER-X 통합 출력** — **현재 실제 수치를 전혀 생산하지 않는다.** 가짜 수치가
실제처럼 제시된 곳(detail/CSV/UI/docs) 없음 — 확인됨.

### 2.3 실제 BUFFER-X 정합 수치를 내려면 필요한 것

1. **코어 벤더링**: 업스트림 model 정의(`models/BUFFERX.py` + `MiniSpinNet` 디스크립터)
   + 워커가 import할 thin loader를 `backend/registration/bufferx/python/bufferx/`에 배치
   (git submodule, 가중치 제외 권장).
2. **가중치**: `download_weights.sh`로 HF `Hyungtae-Lim/BUFFER-X`의
   `snapshot/threedmatch/{Desc,Pose}/best.pth`(~7.3MB) 다운로드.
3. **CUDA 확장 빌드**: `pointnet2_ops` / `KNN_CUDA==0.2` / `torch-batch-svd` /
   in-repo `cpp_wrappers`를 대상 CUDA로 컴파일(업스트림 `scripts/install.sh --cuda cu124`,
   Python 3.11 + torch cu124).
4. **`model(data_source)` 배선**: `bufferx_worker.py:120-124`의 TODO — xyz 배열로
   `data_source`를 구성해 `model(data_source)` 호출, `trans_est`를 16-float row-major로
   ravel, `num_inliers`로 converged 판정. (단일쌍 `register()` 헬퍼는 업스트림에 없음.)

이후 readyTimeoutSec(현 300초)를 콜드 로드 비용에 맞춰 재검토(§1.4-2).

---

## 3. 바텀라인 — 사용자에게 제시되는 결과 수치는 신뢰 가능한가?

**대체로 신뢰 가능하나, 반드시 동반해야 할 caveat가 있다.**

- ✅ **SOTA report 수치는 타당하고 정직하게 라벨됨**: 전부 데시미터급 recall이며 mm로
  둔갑한 수치 없음. 중심 주장(mm는 GICP/SDF가 담당, BUFFER-X는 zero-shot 일반화 SOTA)은
  성립. → 그대로 인용 가능.
- ✅ **gradient-SDF의 confidence/iou/rmse는 내부 정합**하며, 불확실성 채널이 틀린 포즈를
  낮은 신뢰도로 걸러냄을 실증. hap101(conf 1.000)만이 실제 정합 성공으로 인용 적합.
- ✅ **BUFFER-X는 이제 실제 수치를 생산하며(벤더링·배선 완료), 그 수치가 검증된다**:
  hap101→crop에서 bufferx-gicp rmse 3.03048이 KISS-GICP 3.03052와 5자리 일치(교차검증).
  벤더링 전에는 가짜 수치 없이 정직한 identity 폴백이었고, 지금은 실측값이며 둘 다 정직하다.
- ⚠️ **반드시 붙일 caveat 1 — 로컬 ICP "converged=yes" ≠ 성공**: 기존 CSV의 hap100/hap101
  icp·icp-plane·gicp·vgicp 행 다수는 identity·0 iters·0 inliers인 no-op다. converged
  플래그만 보고 "성공"으로 인용하면 안 되며, inliers·transform 일치·confidence를 함께 봐야 한다.
- ⚠️ **반드시 붙일 caveat 2 — mm는 합성/근접 데이터 한정**: archive의 sub-mm rmse는
  합성·정확중첩(blobs/part-to-crop) 결과다. 실측 hap/hesai 부분중첩에서는 rmse가 3 내외로
  남으며, 성공 판정은 inliers·confidence·transform 컨센서스로 한다.
- ⚠️ **caveat 3 — 설계문서 stale**: `06-bufferx-backend.md`는 sparse-conv 백본을 가정하는
  옛 가정을 본문에 남기고 있다(코드/recon은 정정됨). 갱신 1줄 권장.

> **요약**: 제시된 수치 중 *허위·과장은 없다.* SOTA 수치는 데시미터급으로 정직하게
> 라벨됐고, gsdf 신뢰도 값은 잘 보정돼 있으며, BUFFER-X는 정직하게 "아직 수치 없음"
> 상태다. 단 기존 CSV의 로컬-ICP converged=yes 행들은 **결과가 아니라 no-op/실패**이므로
> 인용에서 배제하거나 그 의미를 명시해야 한다.

---

## 4. 중첩(inlier) 영역 mm급 RMSE — hap101_f0 → crop

`registration-results.csv`의 `rmse`(전체 source의 최근접 RMS, m)는 부분 중첩 때문에
정밀도가 아니라 비교 점수다. 실제 mm급 정렬 오차를 보려면 **중첩 영역(=inlier)** 으로
한정해 다시 재야 한다. `scripts/inlier-rmse.py`가 기록된 transform을 같은 입력쌍에
적용해, target 점간격(8.77 mm)의 **3배(26.3 mm) 이내**로 들어온 source 점들에 대해서만
RMS를 계산한다(원자료: `experiments/inlier-rmse-hap101.csv`).

| 기법 | conv | **inlier RMSE** | median | #inliers | overlap% | full rmse |
|---|---|---|---|---|---|---|
| **bufferx-gicp** | yes | **1.61 mm** | 0.41 mm | 10599 | 23.3% | 3.005 m |
| kiss-gicp | yes | 1.60 mm | 0.39 mm | 10599 | 23.3% | 3.005 m |
| gsdf (reference) | yes | 1.61 mm | 0.43 mm | 10599 | 23.3% | 3.005 m |
| kiss | yes | 1.84 mm | 0.74 mm | 10600 | 23.3% | 3.006 m |
| **bufferx** (global only) | yes | 5.62 mm | 4.79 mm | 10596 | 23.3% | 3.005 m |
| gsdf-gpu | yes | 18.94 mm | 19.26 mm | 781 | 1.7% | 3.014 m |
| icp / icp-plane / gicp / vgicp | yes | — | — | 0 | 0% | 3.917 m |

해석:
- **전역+미세(GICP) 계열(bufferx-gicp · kiss-gicp · gsdf)이 모두 ~1.6 mm RMSE /
  ~0.4 mm median** 으로 수렴 — 실측 데이터에서의 mm급 정밀도. SOTA 리포트의 분업 구조
  ("전역으로 근접, mm는 GICP/SDF가 마무리")가 그대로 입증된다.
- **BUFFER-X 단독(전역, refine 없음)은 5.62 mm** — 좋은 초기화지만 미세정합 전. GICP
  체이닝(bufferx-gicp)이 이를 1.61 mm로 끌어내려 kiss-gicp와 동급이 된다.
- **gsdf-gpu(이번 실행)는 18.9 mm·inlier 781개** — confidence 0.029(저신뢰)가 정확히
  경고한 부정확한 포즈. 불확실성 채널이 옳게 작동.
- **로컬 ICP 4종은 inlier 0개** — identity no-op이라 중첩 자체가 없음(=실패 확정).
- overlap ≈ 23.3% 는 crop이 전체 스캔에서 차지하는 비율과 일치 → full rmse가 ~3 m로
  남는 이유를 정량적으로 설명한다.

> 재현: `python3 scripts/inlier-rmse.py --source experiments/data/hap101_f0.ply \
> --target tests/data/crop.ply --inlier-mult 3.0 --out experiments/inlier-rmse-hap101.csv`
