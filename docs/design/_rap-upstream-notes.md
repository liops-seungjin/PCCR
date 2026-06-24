# RAP (Register Any Point) — Upstream Reconnaissance Notes

> Scope: feasibility notes for integrating **RAP / Rectified Point Flow** as a
> CloudCropper registration backend (persistent-Python-worker style, like
> BUFFER-X). Web-sourced only (no local clone). Date: 2026-06-16.
> Anything not directly read from source is marked **[UNVERIFIED]**.

## Identity & sources

- **RAP** = *Register Any Point: Scaling 3D Point Cloud Registration by Flow
  Matching*, Pan, Sun, Zhu, Nunes, Armeni, Behley, Stachniss — **arXiv:2512.01850**
  (2025). PRBonn (Uni Bonn).
  - Repo: https://github.com/PRBonn/RAP — **MIT license**.
  - Project page: https://register-any-point.github.io/
  - Model: HuggingFace **YuePanEdward/RAP** (MIT).
- **RAP is built on Rectified Point Flow (RPF)** — *Generic Point Cloud Pose
  Estimation*, Tao Sun et al., **NeurIPS 2025 Spotlight**, arXiv:2506.05282;
  repo https://github.com/GradientSpaces/Rectified-Point-Flow (HF
  `gradient-spaces/Rectified-Point-Flow`). The PRBonn `rectified_point_flow/`
  package is a fork/extension of RPF, so RPF docs are authoritative for the
  flow-model internals (attention, SVD pose recovery).

---

## 1. flash-attn dependency — **HARD requirement, no fallback**

`rectified_point_flow/flow_model/layer.py` does a **bare `import flash_attn`** at
module top (no `try/except`) and calls
`flash_attn.flash_attn_varlen_qkvpacked_func(...)` in two places (part-wise and
global attention), e.g.:

```python
out = flash_attn.flash_attn_varlen_qkvpacked_func(
    qkv=qkv.to(self.attn_dtype),
    cu_seqlens=cu_seqlens_part,
    max_seqlen=self.max_points_per_part,
    softcap=self.softcap,
)
```

- **No documented SDPA mode, config flag, or alternative attention backend.**
  Confirmed by reading the raw `layer.py`: no `try/except` block, no SDPA path,
  no configuration flag, no alternative attention backend — only a module-level
  note about DDP compatibility.
- This is the **varlen, qkv-packed** API using `cu_seqlens` (ragged batches of
  variable-length point sets). A drop-in `torch.nn.functional.scaled_dot_product_attention`
  replacement is **non-trivial**: SDPA has no native varlen/`cu_seqlens` form, so
  a patch must either (a) pad each part to `max_seqlen` + build an attention
  mask, or (b) loop per-segment. Either is a real porting effort (and the softcap
  arg has no SDPA equivalent pre-padding). **[UNVERIFIED that any community SDPA
  patch for RPF/RAP exists]** — none found in search.

### Pinned install (from `scripts/install.sh`)

flash-attn is installed as a **prebuilt wheel**, not from source:

```
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

So on a matching stack (Linux x86_64, **Python 3.10, torch 2.5.1, CUDA 12.x,
cxx11abi=FALSE**) **no from-source build is required** — pip pulls the wheel.
The build-from-source path (`MAX_JOBS=4 pip install flash-attn --no-build-isolation`)
is the fallback only when no matching wheel exists.

### New-GPU / Blackwell gotcha

- flash-attn **2.7.4 does NOT support consumer Blackwell (sm_120, RTX 50xx)**.
  Dao-AILab issue #1638: the highest supported compute capability in that line is
  8.9; sm_120 raises a runtime error from a hardcoded arch check. Users
  hand-patch the arch list; **no official 2.7.x wheel works on RTX 50xx**. The
  same symptom is reported in NVIDIA Isaac-GR00T #309 (SM120 unsupported).
- On Blackwell you'd need a much newer flash-attn (FA3 / 2.8+ built against
  **CUDA 12.8+**) AND a newer torch — which breaks the torch==2.5.1 pin RPF/RAP
  build against. **This is the single biggest deployment risk for new GPUs.**
- On Hopper/Ampere/Ada (A100, RTX 30/40, sm_80–sm_90) the pinned wheel works.

---

## 2. Full dependency list + versions

From `scripts/install.sh` (PRBonn/RAP) + `requirements_other.txt`:

| Package | Version | Source |
|---|---|---|
| torch / torchvision / torchaudio | **2.5.1 / 0.20.1 / 2.5.1** | `--index-url .../whl/cu124` |
| CUDA | **cu124** (12.4) | wheel index |
| diffusers | **0.33.0** | pip |
| pytorch3d | **0.7.8** | fbaipublicfiles wheel `py310_cu121_pyt251` |
| flash-attn | **2.7.4.post1** | GitHub release wheel (see §1) |
| torch-scatter/sparse/cluster/spline-conv | (PyG, torch-2.5.1+cu124) | for PointTransformerV3 |
| lightning (pytorch-lightning) | **2.5.2** | pip |
| torchmetrics | **1.6.3** | pip |
| trimesh | **4.6.4** | pip |
| Muon optimizer | git `KellerJordan/Muon` | (training only) |
| ninja | latest | build helper |
| hydra-core | **1.3.2** | requirements_other |
| open3d | unpinned | requirements_other |
| huggingface-hub | unpinned | requirements_other |
| scipy | 1.15.2 | requirements_other |
| h5py | 3.13.0 | requirements_other |
| addict | 2.4.0 | requirements_other |
| wandb | 0.20.1 | requirements_other (training/logging) |
| mitsuba | 3.6.4 | requirements_other (rendering/viz) |
| matplotlib | 3.10.3 | requirements_other (viz) |
| rich, tqdm, natsort | various | requirements_other |
| laspy | unpinned | requirements_other (LAS/LAZ I/O) |
| gradio | **>=6.0.0** | requirements_other (app.py UI only) |

Notes:
- **omegaconf / numpy** not explicitly pinned (pulled transitively by hydra/torch).
- The **upstream RPF** `install.sh` additionally lists `xformers==0.0.29` and
  `spconv-cu120==2.3.6`; PRBonn/RAP's script did not surface those — **[UNVERIFIED
  whether RAP needs spconv/xformers]** (likely pulled by PointTransformerV3 path).

### Is **pytorch3d** needed for inference?

- pytorch3d is installed unconditionally in `install.sh`. RPF/RAP recovers each
  part's rigid pose by **solving the orthogonal Procrustes problem via SVD**
  between the conditioning cloud and the generated (assembled) cloud — this is
  exactly what `pytorch3d.ops.corresponding_points_alignment` provides.
  **[UNVERIFIED — likely required for INFERENCE]**, not just eval/viz, because
  pose recovery is the final inference step. Treat pytorch3d as an inference dep
  unless a clean local SVD is substituted. (mitsuba, matplotlib, wandb, gradio
  are viz/logging/UI-only and droppable for a headless worker.)

---

## 3. Checkpoints

Two distribution channels:

- **HuggingFace `YuePanEdward/RAP`** (read from file tree):
  - `rap_model_10.ckpt` — **327 MB** → app.py label **"M (rap_10)"**, config `rap_10`
  - `rap_model_12.ckpt` — **356 MB** → app.py label **"L (rap_12)"**, config `rap_12`
  - `mini_spinnet_t.pth` — **3.67 MB** (the SpinNet local-feature extractor, see §4)
  - README.md (2.51 kB), .gitattributes
  - **Total ≈ 686 MB.** README says latest model version **v1.1** (v1.5 planned).
- **ipb.uni-bonn.de** (used by `scripts/download_weights_and_demo_data.sh`):
  ```
  wget https://www.ipb.uni-bonn.de/html/projects/rap/weights.zip ; unzip ; rm
  wget https://www.ipb.uni-bonn.de/html/projects/rap/demo_example_data.zip ; unzip ; rm
  ```
  `weights.zip` unpacks into `./weights/` (the `rap_model_*.ckpt` +
  `mini_spinnet_t.pth`); individual sizes not listed in the script.
- So: **two flow-model checkpoints (M / L)** + **one feature checkpoint**
  (mini_spinnet). `demo.py` default is `./weights/rap_model.ckpt`; `app.py`
  defaults to `./weights/rap_model_12.ckpt` (the L model).

---

## 4. Single-pair / multi-view inference flow

Two entry points; `app.py` (Gradio) is a thin wrapper that **subprocesses
`demo.py`**. `demo.py` is the real CLI.

**Pipeline (`demo.py`):**
1. **Load** each cloud: `o3d.io.read_point_cloud(ply_path)` → XYZ (+ optional
   normals, RGB). Optional `COORDINATE_TRANSFORM`. Optional global-shift
   de-biasing when coords exceed 1e5 (large UTM-style values).
2. **Voxel downsample**: `dataset_utils.downsample_points(..., method="voxel",
   voxel_size)` (default **0.25 m**).
3. **Per-part point allocation** (`allocation_method`, default `voxel_adaptive`):
   `target = voxel_ratio * voxel_coverage`, clamped to
   **[min_points_per_part=200, max_points_per_part=20000]**; FPS (or random)
   sampling of keypoints. (`allocated_voxel_size = 4.0 * voxel_size`.)
4. **Feature extraction**: `FeatureExtractor({'num_points_per_patch':512,
   'des_r':des_r}, mini_spinnet_t.pth)` — **mini-SpinNet** computes local
   geometric descriptors (~32-D) around each sampled keypoint. (Skippable with
   `--model rap_12_po`, the "points-only" variant.)
5. **Flow-matching inference** (`sample.py`, hydra config): the model generates
   the registered/assembled point set via N flow steps.
6. **Pose recovery**: per RPF, **Procrustes via SVD** between the condition
   cloud and the generated cloud → a **4×4 transform per part**, saved as
   `*_{part}_transform.txt`. `demo.py` then makes them **relative to a reference
   part**: `T_part = inv(T_reference) @ T_part`, and applies
   `pcd.transform(T_part)`. Registered PLYs written to `registered/`.

**Quoted demo usage** (README):
```bash
python app.py                          # interactive Gradio
bash ./scripts/test_script_example.sh  # batch (edit configs first)
```
**`demo.py` key CLI flags (with defaults):**
```
--input (required)                 folder of point-cloud files
--output / --log_dir
--voxel_size            0.25       (meters)
--feature_extraction_checkpoint    ./weights/mini_spinnet_t.pth
--des_r                 5.0        (feature radius, meters)
--remove_outliers       True
--allocation_method     voxel_adaptive  (| point_count | spatial_coverage)
--voxel_ratio           0.05
--min_points_per_part   200
--max_points_per_part   20000
--use_random_downsample False      (else FPS)
--flow_model_checkpoint ./weights/rap_model.ckpt
--n_generations         1          (multi-hypothesis)
--skip_inference        False
--output_generated      False      (save generated vs. transformed clouds)
--visualize             False
# app.py adds: --config <rap_10|rap_12>, --model, inference_sampling_steps (flow steps, default 10)
```

---

## 5. Input convention

- **XYZ required**; normals optional (used if present); RGB optional (viz only).
- `app.py` accepts `.ply .pcd .las .laz .pts .e57 .ptx .obj`; meshes are
  surface-sampled to ~100k points. `demo.py` core path reads **PLY via Open3D**
  (other formats converted upstream in app.py).
- **Voxel size 0.25 m default** but described as "[overwritten by adaptive
  parameters]"; `voxel_ratio` drives adaptive per-part counts.
- **Points per part: 200 (min) – 20000 (max)**; FPS-sampled keypoints, not the
  raw cloud.
- **≥2 parts/views required** (app.py validation: "at least 2 point cloud files").
  Handles pairwise AND multi-view in one shot.
- **Scale generalization**: paper claims zero-shot across scales/sensor
  modalities. Practically realized via the adaptive voxel/`des_r` (meters) +
  optional global-shift normalization — so **inputs should be in metric units**
  and `voxel_size`/`des_r` tuned to scene scale.

---

## 6. License

**MIT** — both PRBonn/RAP repo and HF `YuePanEdward/RAP` (and upstream RPF).
Permissive; compatible with vendoring into CloudCropper like BUFFER-X/G3Reg.

---

## 7. Known gotchas / forks for standalone inference

- **flash-attn is mandatory and arch-sensitive** (§1). On the target GPU's arch
  you must have a matching prebuilt wheel; **RTX 50xx/Blackwell currently
  unsupported by the pinned 2.7.4 wheel** → blocker without a newer FA + newer
  torch (which breaks the torch 2.5.1 pin) or an SDPA patch.
- **Tight version lattice**: Python 3.10 + torch 2.5.1 + cu124 + pytorch3d 0.7.8
  (cu121 wheel) + flash-attn cu12torch2.5 cp310. Deviating from any pin tends to
  break a wheel. This is a conda env, not a casual pip install.
- pytorch3d wheel is Linux-x86_64/py310 only — **no easy Windows/macOS path**.
- `app.py` shells out to `demo.py` (subprocess + temp dirs + log scraping) — for
  a CloudCropper worker we'd call the **`demo.py` / `sample.py` Python API
  directly**, not the Gradio layer.
- Heavy extra deps (mitsuba, gradio, wandb, matplotlib) are **viz/UI/logging
  only** and can be dropped from a headless worker requirements file.
- `--model rap_12_po` ("points-only") **skips mini-SpinNet** — simpler worker if
  feature extraction proves fragile. **[UNVERIFIED accuracy delta]**.

---

## VERDICT

**Is RAP inference runnable WITHOUT a from-source flash-attn build?**
**Yes — on supported GPU archs (sm_80/86/89/90: A100, RTX 30/40, Hopper).** The
install uses a **prebuilt flash-attn 2.7.4.post1 wheel** (cu12/torch2.5/py310);
no compilation needed there. **BUT flash-attn is non-optional** — there is no
SDPA fallback in `layer.py`, and it uses the varlen/`cu_seqlens` packed API that
SDPA can't drop-in replace without a real padding+masking patch. **On Blackwell /
RTX 50xx (sm_120) the pinned wheel does NOT work**, and there is no clean
prebuilt path — you'd need a newer flash-attn built against CUDA 12.8+ (forcing a
torch upgrade off the 2.5.1 pin) or you'd have to write+validate an SDPA patch.
So: runnable without source-build *only if* (a) the deployment GPU is pre-Blackwell
**and** (b) you accept the exact torch2.5.1/cu124/py310 lattice.

**Is it integrable as a CloudCropper persistent-Python-worker backend like
BUFFER-X?** **Architecturally yes, and it's a strong fit** — RAP is a
torch+lightning checkpoint model with a Python `demo.py`/`sample.py` API, zero-shot
across scales/sensors (same selling point as BUFFER-X), MIT-licensed, outputs the
exact artifact CloudCropper wants (**per-view 4×4 transforms**, recovered by
SVD/Procrustes). It maps cleanly onto the existing
`backend/registration/.../python_worker.hpp` pattern: lazy-spawn, load
checkpoint once, stream point sets in / poses out, forward knobs
(voxel_size, voxel_ratio, des_r, n_generations, flow steps) via config like
`bufferx.yaml`. **The integration risk is entirely the dependency stack**, not
the interface: the brittle Python-3.10/torch-2.5.1/cu124/pytorch3d/flash-attn
lattice, ~686 MB weights, and especially the **flash-attn hard dependency** which
makes Blackwell support a blocker. Recommend: (1) confirm target GPU arch before
committing; (2) if Blackwell is in scope, scope an SDPA-patch spike or wait for an
upstream non-flash path; (3) vendor + headless-trim deps (drop gradio/mitsuba/
wandb) as done for BUFFER-X.

---

## 통합 결정 (확정)

위 정찰 결과를 바탕으로, RAP 백엔드는 다음 형태로 통합한다(전 팀 공통 기준).

- **연동 방식 — BUFFER-X 패턴(persistent Python worker) + NPZ 핸드오프.**
  RAP는 torch+lightning 체크포인트 모델에 `demo.py`/`sample.py` Python API를
  가진, BUFFER-X와 동일한 모양이다. 따라서 G3Reg의 one-shot subprocess가 아니라
  **`PythonWorker`**(lazy-spawn, JSON-lines)로 워커를 한 번 띄워 체크포인트를 상주
  로드하고, 점군은 **NPZ 파일로 주고받는다**(`bufferx_worker.py`의 `writeNpz` 관용
  그대로). pairwise는 `[target, source]` 두 part를 넣어 part별 4×4 포즈를 SVD로
  복원하고, **타깃 기준 상대 변환** `T = inv(T_target) @ T_source`로 만들어
  row-major target←source(`RegResult::transform`) 규약에 맞춘다.

- **Graceful degradation — 필수.** 워커는 RAP 코어/weights/flash-attn이 **설치되지
  않아도** `ready`로 올라와야 하며(`bufferx_worker.py`와 동일), 이 상태에서
  `register`는 **identity 변환 + `converged:false` + 설명 `note`**를 반환한다.
  inlier/품질 수치는 **절대 지어내지 않는다**. 네이티브 C++ fallback은 없다. 이
  덕분에 무거운 RAP 스택 없이도 이 환경에서 빌드·테스트가 가능하다(fake worker
  e2e 테스트로 C++/프로토콜 경로 전체를 검증).

- **enum — `RegAlgo::Rap` / `RegAlgo::RapGicp`** (G3Reg 항목 뒤). `algoName()`은
  "RAP" / "RAP + GICP", dispatcher는 `rap::run(source, target, opt)`. `RapGicp`는
  C++ 측에서 GICP refine 체인을 잇는다(BUFFER-X와 동일 방식).

- **config — `config/rap.yaml`, 키는 워커로 verbatim 전달.** `python`/
  `timeout_sec`/`weights_dir`/`device`/`voxel_size`/`refine`만 예약(reserved)이고,
  그 외 모든 비어있지 않은 키(`voxel_ratio`, `des_r`, `n_generations`,
  `inference_sampling_steps`, `model`, `allocation_method`,
  `min/max_points_per_part` 등)는 워커로 **그대로 forward**한다. RegOptions에는
  `bufferxVoxel`을 미러한 `rapVoxel`(입력 다운샘플 voxel override, 0=yaml/워커
  auto) 한 개만 추가하고, 나머지 노브는 전부 yaml 경유다. `timeout_sec` 기본은
  flow 모델이 무거우므로 BUFFER-X(600)보다 큰 **900**.

- **#1 배포 리스크 — flash-attn / Blackwell.** 정합 인터페이스 자체는 위험이 없고,
  리스크는 전적으로 의존성 스택이다. 그중에서도 **flash-attn 하드 의존성과
  RTX 50xx/Blackwell(sm_120) 미지원**(§1, §7)이 **단연 1순위 배포 블로커**다.
  배포 GPU 아키텍처를 커밋 전에 반드시 확인하고, Blackwell이 범위에 들어오면
  SDPA 패치 스파이크 또는 상위 non-flash 경로를 별도로 스코핑한다. 이 블로커가
  바로 graceful degradation을 필수로 만드는 이유이기도 하다.
