# 전역(initial-pose-free) Point Cloud Registration SOTA 조사 (2026-06-16)

> deep-research 하베스트(22 sources · 108 claims → 25 검증 · 23 confirmed/2 killed · 14 findings · 104 agents).
> 범위: **전역(coarse) 단계의 강건성/성공률**(미세정합 제외). 대상: CloudCropper hap/hesai 근거리 고밀도
> crop→전체스캔, **부분중첩 ~20~30% + OOD 커스텀 센서 + 초기추정 없음**. 코드 유무 무관.
> 현재 전역 백엔드 = KISS-Matcher / BUFFER-X.

## 핵심 결론

2024~2026 전역 정합 SOTA는 세 갈래:
1. **생성/flow 기반 — RAP** (가장 직결): zero-shot 일반화 + 저중첩 강건성 + 코드 공개를 모두 충족, BUFFER-X를 직접 능가하는 근거 있음.
2. **학습-free 클리크/그래프 로버스트 추정 — G3Reg · MAC++ · CLIPPER+**: 학습 가중치가 없어 본질적으로 도메인 무관, 초기값 불필요 → KISS-Matcher 계열 직접 대체/병행.
3. **학습 특징매칭 — PSReg · Decision PCR · UGP · HyperGCT**: 실내 벤치 정확도는 높으나 cross-sensor 일반화 근거 약함. Decision PCR/HyperGCT는 전역 백엔드가 아니라 **검증/이상치제거 보조 모듈**.

## 비교표

| 기법 | 분류 | 전역 강건성 근거 | OOD/zero-shot 일반화 | 코드 | CloudCropper 가치 |
|---|---|---|---|---|---|
| **RAP** (Register Any Point, arXiv:2512.01850) | 생성/flow matching (multi-view) | 20% 중첩 **>80% 성공**(베이스라인 <40%), 50m 뷰간격서 BUFFER-X·Predator 능가; 3DMatch RSR 95.9 / 3DLoMatch 78.78 | **강함** — Livox/Velodyne/**Hesai**/Ouster/RGB-D/TLS/항공 60+ 데이터셋 zero-shot | ✅ 공개 (PRBonn) | **1순위 대체** — OOD+저중첩 직결 |
| **G3Reg** (IEEE T-ASE'24) | 학습-free, Gaussian-Ellipsoid + Pyramid Compatibility Graph (max-clique) | TEASER/Quatro 능가 강건성·실시간 | 학습 없음 → 도메인 무관(옥외 **LiDAR 전용** 설계) | ✅ 공개 | **2순위 대체** — LiDAR 도메인 최적합 |
| **MAC++** (3DV'25) | 학습-free, seed-voting clique-pool + 점진 평가 | **<1% inlier** 극저중첩서 MAC 실패영역 처리, 3DMatch +27.1% / 3DLoMatch +30.6% RR | 학습 없음 → 도메인 무관 | (미확정) | **2순위 대체/병행** — 극저중첩 강건 |
| **CLIPPER+** (arXiv:2402.15464) | 학습-free, max-clique + k-core pruning | **>99% 이상치** 환경서도 성공 | 학습 없음 → 도메인 무관 | ✅ 공개 | KISS-Matcher 동계열 대체 후보 |
| **Decision PCR** (arXiv:2507.14965) | 학습 검증기(binary "옳은 정합?") | GeoTransformer 결합 **3DLoMatch 86.97% RR** (새 SOTA 주장) | ETH 일반화 보임(미peer-review) | (미확정) | **3순위 보완** — 후단 후보 검증기 |
| **HyperGCT** (ICCV'25) | 학습 이상치제거(dynamic Hyper-GNN) | 고차 일관성 기반 가설 생성 | correspondence 입력(descriptor 아님) | ✅ 공개 | KISS-Matcher graph-pruning 대체 보조 |
| **PSReg** (AAAI'25) | 학습 특징매칭(prior-guided Sparse MoE) | 3DMatch 95.7 / 3DLoMatch 79.3 RR | **약함** — 실내 3DMatch/ModelNet 한정 | (미확정) | 우선순위 낮음(OOD 미검증) |
| **UGP** (CVPR'25 Highlight) | 학습 특징매칭(cross-attn 제거) | (주장된 일반화 수치 일부 **반증 0-3**) | 불확실 | (미확정) | 근거 불확실 |
| ~~ZeroReg/ZeroReg++~~ | 2D foundation model | — | RGB **이미지 필수** | ✅ | **제외** — image-free LiDAR 비호환 |
| ~~RARE~~ | zero-shot **미세정합** | T_init 필요 | — | ✅ | **제외** — 전역 아님(refine 단계) |

## 상위 추천 (KISS-Matcher/BUFFER-X를 능가할 가능성)

1. **RAP** — 일반화·저중첩·코드를 모두 충족하는 유일 후보. CloudCropper의 OOD 커스텀 LiDAR + <30% 중첩에 가장 직결. BUFFER-X처럼 Python 워커로 통합 가능. **단, 가장 강한 수치가 저자 자체 벤치마크 기반 → 실측 hap/hesai 자체 검증 필수.**
2. **G3Reg(LiDAR 전용) / MAC++(극저중첩)** — 학습-free라 재학습 없이 도메인 무관, KISS-Matcher와 같은 슬롯에 직접 대체. 단 입력이 correspondence라 descriptor(Faster-PFH/FCGF/BUFFER-X 특징) 선택이 따라옴.
3. **Decision PCR** — 대체가 아닌 **보완**: KISS-Matcher/BUFFER-X가 낸 다중 가설을 재랭킹·검증해 성공률을 끌어올림(저중첩에서 Maximum-Inlier-Count 한계 보완).

## CloudCropper 통합 한 줄 코멘트

- **RAP**: 전역 백엔드 완전 대체 후보 — `backend/registration/rap/`에 BUFFER-X와 동일한 영속 Python 워커 패턴으로 얹고, 실측 데이터로 BUFFER-X 대비 성공률·속도 A/B.
- **G3Reg / CLIPPER+**: 학습-free C++ 라이브러리 → small_gicp/KISS처럼 네이티브 백엔드로 통합 가능(빌드타임 의존성 추가).
- **Decision PCR**: KISS/BUFFER-X 후단 검증기로 래핑(전역 단계에 가설 선택기 추가).

## 주의 (검증 caveat)

1. **시간 민감성**: PSReg(95.7/79.3)는 2025 중반 Decision PCR(86.97)·DINOReg에 이미 추월. RAP가 가장 신선(2025-12, v2 2026-03)하나 독립 3자 재현 없음.
2. **학습-free "도메인 무관성"은 추론**: 가중치가 없다는 사실에서 도출된 것이며, **근거리 고밀도 hap/hesai 실성공률을 직접 평가한 논문은 없다.**
3. **correspondence 입력 계열**(MAC++/CLIPPER+/HyperGCT): 전역 백엔드 교체가 **descriptor 선택까지 포함**될 수 있음.
4. RAP의 "20% 중첩 >80%"는 저자 cross-domain 벤치마크 기반(canonical 3DLoMatch pairwise에선 PARE-Net 80.50 > RAP 78.78). "동일 환경 부분중첩" 가정 — CloudCropper crop→scan과는 부합.
5. RAP/MAC++/HyperGCT의 추론 속도·GPU 메모리·라이선스는 본 조사에서 미확정.

## Open Questions (실증 필요)

- RAP을 실측 hap/hesai 20~30% crop에 zero-shot 적용 시 성공률·속도·GPU 메모리?
- G3Reg(옥외·저밀도 가정)가 근거리 **고밀도** crop-to-scan에서 primitive 추출이 잘 되는가?
- MAC++/CLIPPER+는 어떤 descriptor와 결합해야 최적인가(전역 교체 = descriptor 교체?)
- Decision PCR을 커스텀 LiDAR 후단 검증기로 붙일 때 재학습 없이 동작하는가?

## 주요 출처

RAP [2512.01850](https://arxiv.org/abs/2512.01850) ([repo](https://github.com/PRBonn/RAP)) · G3Reg (IEEE T-ASE'24, repo 공개) · MAC++ [3DV'25](https://openreview.net/forum?id=dOpxroaprM) · CLIPPER+ [2402.15464](https://arxiv.org/abs/2402.15464) ([repo](https://github.com/ariarobotics/clipperp)) · Decision PCR [2507.14965](https://arxiv.org/abs/2507.14965) · PSReg [2501.07762](https://arxiv.org/abs/2501.07762) · HyperGCT (ICCV'25) · CL-PCR (Sensors 2024, doi:10.3390/s24175499) · ZeroReg [2312.03032](https://arxiv.org/abs/2312.03032) · RARE [2507.19950](https://arxiv.org/abs/2507.19950)
