# Point Cloud Registration SOTA 조사 (2026-06-16 복원본)

> 출처: 2026-06-16 `deep-research` 하베스트(21 sources → 101 claims → 25 검증 → 24 confirmed/1 killed, 103 agent calls) +
> 후속 타깃 검색(BUFFER-X / CAST). 컴퓨터 종료로 유실된 세션 `023cf142` 기록을 트랜스크립트에서 복원.

## 조사 질문 / 범위

최신(2023~2025) 포인트 클라우드 정합 중 **mm/sub-mm 정밀도**를 제시한 SOTA를 조사.
- 시나리오: 초기 추정값 없이 **전역(global) → 미세(fine)** 한 파이프라인으로 mm 도달.
- 데이터: 근거리 고밀도 LiDAR/스캔, 부분 중첩(partial overlap). (CloudCropper = hap/hesai 커스텀 데이터)
- 우선순위: **성능 SOTA 우선** (통합 난이도 무관).

---

## 🏆 결론: 당신의 시나리오에서 SOTA = **BUFFER-X**

**BUFFER-X** — *Towards Zero-Shot Point Cloud Registration in Diverse Scenes*, **ICCV 2025 Highlight**, MIT-SPARK (Minkyun Seo, Hyungtae Lim, Kanghee Lee, Luca Carlone, Jaesik Park).
📄 [arXiv 2503.07940](https://arxiv.org/abs/2503.07940) · 💻 [github.com/MIT-SPARK/BUFFER-X](https://github.com/MIT-SPARK/BUFFER-X)

**왜 "당신에게" SOTA인가** — 벤치마크 점수 1등이 아니라, **재학습·튜닝 없이 처음 보는 센서/장면에 바로 동작(zero-shot generalization)**하는 것이 핵심. CloudCropper는 hap/hesai 같은 **out-of-domain 커스텀 데이터**를 다루므로 3DMatch에 과적합된 모델은 실전에서 무너진다. BUFFER-X는 그 문제를 정조준한다.

- **핵심 아이디어**: 학습 정합의 일반화를 막는 3요소 제거 — ① 환경별 voxel/search radius 의존 → **적응적 자동 결정**, ② 학습 keypoint detector의 도메인 취약성 → **FPS로 우회**, ③ raw 좌표 사용 → **patch별 scale 정규화**. + multi-scale patch descriptor & 계층적 inlier search.
- **검증 범위**: 3DMatch/3DLoMatch, ScanNet++, KITTI, ETH, Oxford, 이종 센서(hetero)를 **단일 모델 무재학습**으로 평가 — 데이터 일반화의 직접 근거.
- **실용성**: MIT 라이선스, Python + CUDA. **KISS-Matcher와 같은 연구실(MIT-SPARK)** 계보 → 현 백엔드와 통합 친화적. 단, 학습 기반이라 torch + sparse-conv + **사전학습 가중치 다운로드** 필요(런타임).

### 권장 파이프라인

> **BUFFER-X (강건 전역, zero-shot) → gradient-SDF / GICP (mm 미세정밀화)**

현 구조(KISS-Matcher → GICP/SDF)에서 **전역 단계만 BUFFER-X로 업그레이드**하는 그림. mm 정밀도는 여전히 `small_gicp` / `gradient-SDF`가 담당한다.

---

## 순수 벤치마크 정확도 SOTA (참고)

데이터 일반화가 아니라 3DMatch 점수만 본다면:

| 기법 | 강점 | 수치 | 코드 |
|---|---|---|---|
| **CAST** (NeurIPS'24) | **Inlier Ratio(대응 정밀도) 최고** — mm 추구에 가장 관련 | "SOTA accuracy·robustness·efficiency" | [Python/CUDA, MIT](https://github.com/RenlangHuang/CAST) |
| **PARE-Net** (ECCV'24) | **전체 Registration Recall 최고** (3DMatch 95.0 / LoMatch 80.5) | recall SOTA, IR은 CAST에 열세 | [Python/CUDA](https://github.com/yaorz97/PARENet) |
| **MAC+GeoTransformer** | 강건한 학습-free 추정 결합 | RR 95.7 / 78.9 | 공개 |

→ **correspondence 정밀도(=mm로 가는 길)** 관점에선 **CAST**가 PARE-Net보다 낫다.

---

## ⚠️ 가장 중요한 한계 (검증된 caveat)

조사된 **어떤 학습/로버스트 논문도 mm/sub-mm RMSE를 직접 보고하지 않았다.** 3DMatch/3DLoMatch의 Registration Recall은 성공 기준이 **RMSE < 0.2 m(200 mm, 데시미터급)**, KITTI는 cm~m 단위. 따라서 이 SOTA 수치들은 mm 정밀도를 *인증*하지 못하며 **강건성/recall 비교**로 읽어야 한다. mm는 전역 정합 뒤의 fine 단계(small_gicp / gradient-SDF)가 담당한다. → **현 CloudCropper의 '강건 전역 + GICP/SDF 미세' 분업 구조가 문헌의 표준 패턴과 일치.**

---

## 검증된 후보 기법 비교 (deep-research: 13개 검증 findings → 11개 distinct 기법)

| 기법 | 분류 | 보고 정밀도 / 평가 | mm? | 코드 | 신뢰도 |
|---|---|---|---|---|---|
| **GeoTransformer** | 학습 특징매칭 (keypoint-free C2F, RANSAC-free LGR, ~100× 가속) | 3DMatch RR 92.5%/IR 70.9%, 3DLoMatch RR 74.2%; KITTI RRE 0.230°/RTE 6.2cm | ❌ cm급 | [repo](https://github.com/qinzheng93/GeoTransformer) | high |
| **PARE-Net** (ECCV'24) | 학습, position-aware rotation-equivariant + hypothesis proposer | 3DMatch RR 95.0%, 3DLoMatch 80.5%, KITTI ~99.8% RR | ❌ (IR은 CAST 열세) | [repo](https://github.com/yaorz97/PARENet) | high |
| **RoITr** (CVPR'23) | 학습, rotation-invariant Transformer (실내 RGB-D) | 3DMatch RR 91.9%(rot 94.7%), 3DLoMatch 74.8% | ❌ decimeter | — | high |
| **FLAT** (2025) | 학습, Gromov-Wasserstein cross-attention C2F | IR 0.1m / RR<0.2m, KITTI RTE<2m | ❌ m~dm | — | high |
| **DINOReg** (2025) | 멀티모달(DINOv2+geom), **RGB-D 필수** | RR<0.2m, KITTI RTE 9.8cm | ❌ + 이미지필수 → 부적합 | [repo](https://github.com/ccjccjccj/DINOReg) | high |
| **RAP** (PRBonn'25) | flow matching conditional generation, direct multi-view | 신규 패러다임, mm 수치 미검증 | ❓ 미확인 | [repo](https://github.com/PRBonn/RAP) | medium |
| **KISS-Matcher** (MIT-SPARK'24) | 로버스트 전역 (Faster-PFH + k-core pruning, C++) | sub-degree/sub-cm; KITTI G-ICP 후 ~1.1cm | ❌ (fine 아님) | [repo](https://github.com/MIT-SPARK/KISS-Matcher) | high |
| **TEASER++** | 로버스트 전역 (>99% outlier, SDP certifiable) | 강건 전역 baseline, fine 아님 | ❌ | [repo](https://github.com/MIT-SPARK/TEASER-plusplus) | high |
| **MAC** (CVPR'23) | 학습-free 전역 (maximal clique) | 3DMatch RE≤15°/TE≤30cm, KITTI TE~8cm | ❌ cm/multi-deg | [repo](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques) | high |
| **TCF** (RAL'24) | 로버스트 전역 (단계적 RANSAC + IRLS, 실시간) | MAC 대비 최대 3자릿수 가속, KITTI/ETH 度·m/dm | ❌ | [repo](https://github.com/ShiPC-AI/TCF) | high (2-1) |
| DL 정합 서베이 (arXiv 2404.13830) | 통제 벤치마크 (cross-method 비교 레퍼런스) | — | — | — | high |

**종합 판단 (medium)**: 조사된 어떤 단일 학습 논문도 mm 정밀도 측면에서 현 백엔드(KISS-Matcher 전역 + small_gicp fine + gradient-SDF)를 *즉시 대체*하지 못함. 학습 특징매칭은 "초기값 없는 저중첩 전역 정합의 강건성"을 보완하는 옵션으로만 가치. deep-research 자체 상위 추천은 (1) PARE-Net (2) GeoTransformer (3) KISS-Matcher 유지였고, 후속 타깃 검색에서 **out-of-domain 일반화 기준 BUFFER-X가 최종 SOTA로 선정**됨.

---

## Open Questions

1. 근거리 고밀도/object-scale에서 자체 metrology 데이터로 **sub-mm RMSE를 직접 보고**한 2023-2025 논문이 별도로 존재하는가? (industrial inspection 특화 정합 — 본 조사 범위 밖 가능성)
2. GeoTransformer/PARE-Net의 출력 transform을 small_gicp/gradient-SDF fine과 결합 시, KISS-Matcher 대비 **저중첩(<30%) 초기정합 성공률**이 실제로 개선되는가?
3. RAP(flow matching)/implicit·SDF 정합이 ICP/GICP 대비 **연속 좌표 sub-mm 정밀화**에서 실측 우위가 있는가?
4. coarse-to-fine에서 sub-mm를 얻는 메커니즘(GP/uncertainty, SDF gradient, sub-voxel 회귀)별 정확도-속도 트레이드오프 직접 비교 벤치마크가 있는가?

## Refuted (1 killed)

- "GeoTransformer가 3DMatch RR 92.0% / 3DLoMatch 75.0% (RANSAC), LGR로 91.5%/74.0%" — vote 1-2, 기각. (다른 finding의 92.5%/74.2% 수치 채택)

## 검증 통계

`angles 5 · sourcesFetched 21 · claimsExtracted 101 · claimsVerified 25 · confirmed 24 · killed 1 · afterSynthesis 13 · agentCalls 103`

## 주요 출처

- BUFFER-X: [arXiv 2503.07940](https://arxiv.org/abs/2503.07940) · [ICCV2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Seo_BUFFER-X_Towards_Zero-Shot_Point_Cloud_Registration_in_Diverse_Scenes_ICCV_2025_paper.pdf) · [repo](https://github.com/MIT-SPARK/BUFFER-X)
- PARE-Net [2407.10142](https://arxiv.org/abs/2407.10142) · GeoTransformer [2202.06688](https://arxiv.org/abs/2202.06688) · RoITr [2303.08231](https://arxiv.org/abs/2303.08231) · FLAT [2502.08285](https://arxiv.org/html/2502.08285) · DINOReg [2509.24370](https://arxiv.org/pdf/2509.24370) · RAP [2512.01850](https://arxiv.org/abs/2512.01850)
- KISS-Matcher [2409.15615](https://arxiv.org/abs/2409.15615) · TEASER++ [2001.07715](https://arxiv.org/abs/2001.07715) · MAC [CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_3D_Registration_With_Maximal_Cliques_CVPR_2023_paper.pdf) · TCF [2410.15682](https://arxiv.org/abs/2410.15682)
- CAST [repo](https://github.com/RenlangHuang/CAST) · DL registration survey [2404.13830](https://arxiv.org/html/2404.13830v3)
