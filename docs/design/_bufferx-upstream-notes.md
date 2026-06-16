# BUFFER-X Upstream Reconnaissance Notes

> Source repo: <https://github.com/MIT-SPARK/BUFFER-X> (ICCV 2025 Highlight) · Paper: <https://arxiv.org/abs/2503.07940>
> Reconnaissance date: 2026-06-16. All claims below are sourced with the URL and a direct quote.
> Items that could **not** be verified from the upstream source are explicitly flagged.

---

## 1. Sparse-conv backbone — which library?

**Answer: NONE of MinkowskiEngine / spconv / torchsparse.** BUFFER-X does **not** use a sparse-convolution backbone at all.
It inherits BUFFER's two-stage design:
- **Point-wise / pose stage** uses C++ grid subsampling + KNN + PointNet2 ops (KPConv-lineage tooling, `cpp_wrappers/`), **not** an `nn.Module` KPConv layer.
- **Descriptor stage** is a **SpinNet-style cylindrical descriptor `MiniSpinNet`** built from plain `nn.Conv2d` / `nn.Conv3d`.

Evidence:

- `models/BUFFERX.py` import (source: <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/models/BUFFERX.py>):
  > `from models.patch_embedder import MiniSpinNet` … `self.Desc = MiniSpinNet(config)`
  > "No KPConv, MinkowskiEngine, spconv, or torchsparse imports are present in this file."

- `models/patchnet.py` (source: <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/models/patchnet.py>):
  > Imports: `import torch.nn as nn`, `import torch`, `import utils.common`.
  > "Convolution Types: standard PyTorch `Conv2d` and `Conv3d`. **KPConv: Not used.** **MinkowskiEngine/spconv/torchsparse: Not used.**"
  > Classes: `Cylindrical_Net`, `Cylindrical_UNet`, `CostNet` — operate on cylindrical/volumetric feature maps.

- `models/patch_embedder.py` imports (source: <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/models/patch_embedder.py>):
  > `import models.patchnet as pn`, `import pointnet2_ops.pointnet2_utils as pnt2`, `from utils.SE3 import *`

- `scripts/install.sh` (source: <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/scripts/install.sh>) installs CUDA extensions:
  > "Pointnet2_PyTorch (pointnet2_ops); KNN_CUDA 0.2; cpp_wrappers compilation; torch-batch-svd. The script notably **omits sparse convolution libraries entirely**."

**Implication for CloudCropper:** no MinkowskiEngine/spconv toolchain to provision. Instead the build needs `pointnet2_ops`, `KNN_CUDA`, `torch-batch-svd`, and compilation of the in-repo `cpp_wrappers/` (KPConv-style C++ grid subsampling). These are CUDA-compiled extensions → a CUDA toolchain is still required for GPU inference.

---

## 2. Python / torch / CUDA versions + dependency list

**Recommended environment:** Python **3.11**, installed via `scripts/install.sh --cuda cu124` (PyTorch CUDA 12.4 wheels).

- README (source: <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/README.md>):
  > ```bash
  > conda create -n bufferx python=3.11 -y
  > conda activate bufferx
  > ./scripts/install.sh --cuda cu124 --with-hub
  > ```

- `INSTALL.md` CUDA matrix (source: <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/INSTALL.md>):
  > `--cuda cu124`: "PyTorch CUDA 12.4 wheels (recommended)"
  > `--cuda cu118`: "PyTorch CUDA 11.8 wheels"
  > `--cuda cu111`: "legacy PyTorch 1.9.1 and CUDA 11.1"
  > `--cuda cpu --skip-cuda-extensions`: CPU-only, **no inference capability**
  > Legacy per-combo installers exist: `install_py3_8_cuda11_1.sh`, `install_py3_10_cuda11_8.sh`, `install_py3_11_cuda12_4.sh`.

**Dependency list** (what goes in requirements). Requirements are split into files (source dir: <https://github.com/MIT-SPARK/BUFFER-X/tree/main/requirements>): `base.txt`, `dev.txt`, `hub.txt`, `scannetpp.txt`.

`requirements/base.txt` verbatim (source: <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/requirements/base.txt>):
```
easydict>=1.13
einops>=0.7
kornia>=0.7
matplotlib>=3.7
nibabel>=5.0
numpy==1.26.3
open3d==0.18.0
scikit-learn>=1.3
tabulate>=0.9
tensorboard>=2.14
tensorboardX>=2.6
tqdm>=4.66
```

`requirements/hub.txt` verbatim (source: <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/requirements/hub.txt>):
```
huggingface_hub>=0.31
```

**Plus** (from `install.sh`, not in the `.txt` files — installed as torch + CUDA extensions):
`torch`, `torchvision`, `torchaudio` (cu124 index), `pointnet2_ops` (Pointnet2_PyTorch), `KNN_CUDA==0.2`, `torch-batch-svd`, in-repo `cpp_wrappers` compilation; build via `pip install -e ".[runtime,hub,kiss,scannetpp,dev]"`.

> **NOT verified:** exact contents of `dev.txt` and `scannetpp.txt` were not fetched. `kiss` extra implies an optional `kiss-matcher` dependency (see §4 pose estimator) but its pin was not confirmed.

---

## 3. Pretrained weights — single zero-shot or multiple?

**Answer: TWO source-trained checkpoint sets, each used zero-shot.** They are organized by *training source* (3DMatch indoor, KITTI outdoor), and each has a `Desc` and a `Pose` stage `.pth`. The paper's headline zero-shot generalist is the **3DMatch-trained** model evaluated on all other datasets without retraining.

Hosted on HuggingFace: `Hyungtae-Lim/BUFFER-X`.

Download command (source: README <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/README.md>):
> ```bash
> python scripts/download_pretrained_models.py --source hf --repo-id Hyungtae-Lim/BUFFER-X
> ```
> "Pretrained checkpoints are hosted at Hyungtae-Lim/BUFFER-X."
> `download_pretrained_models.py` also supports `--source` Dropbox (hardcoded `snapshot.zip`); files extract into a local `snapshot/` directory. (source: <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/scripts/download_pretrained_models.py>)

Exact files + sizes (source: HF API <https://huggingface.co/api/models/Hyungtae-Lim/BUFFER-X/tree/main/snapshot?recursive=true>):
```
snapshot/kitti/Desc/best.pth        3,670,575 bytes  (~3.67 MB)
snapshot/kitti/Pose/best.pth        3,670,575 bytes  (~3.67 MB)
snapshot/threedmatch/Desc/best.pth  3,671,729 bytes  (~3.67 MB)
snapshot/threedmatch/Pose/best.pth  3,671,729 bytes  (~3.67 MB)
```
Total HF repo size ~14.7 MB (source: <https://huggingface.co/Hyungtae-Lim/BUFFER-X/tree/main>). Loaded at runtime from `snapshot/{experiment_id}/{stage}/best.pth` where stage ∈ {`Desc`,`Pose`} (source: test.py analysis below).

**Implication:** the checkpoint chosen is selected by `--experiment_id` (e.g. `threedmatch`). For CloudCropper's general/indoor zero-shot use, ship `snapshot/threedmatch/{Desc,Pose}/best.pth` (~7.3 MB total). Tiny weights — trivial to bundle.

---

## 4. Inference entry point — single-pair API?

**Answer: There is NO standalone single-pair demo script.** Inference goes through `test.py`, which operates over a **dataset/dataloader**, not a bare `(src, tgt)` pair. There is no `demo.py` (verified: repo top-level has `train.py`, `test.py`, `trainer.py` only — source: <https://github.com/MIT-SPARK/BUFFER-X/tree/main>).

CLI (source: README <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/README.md>):
> ```bash
> python test.py --dataset 3DMatch TIERS Oxford MIT --experiment_id threedmatch --verbose
> ```
> Args include `--dataset`, `--experiment_id`, `--pose_estimator` (`ransac` or `kiss_matcher`), `--gpu`, plus ablation overrides `--num_points_per_patch`, `--num_scales`, `--num_fps`, `--src_sensor`, `--tgt_sensor`, `--verbose`.

Core registration call (source: test.py analysis, <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/test.py>):
```python
test_loader = get_dataloader(dataset=load_dataset, split="test", config=cfg,
                             shuffle=False, num_workers=cfg.train.num_workers)
# per batch `data_source` (fields: "src_id", "tgt_id", "relt_pose" = GT pose):
(trans_est, times, num_inliers, num_mutual_inliers,
 num_inlier_ind, scales_used) = model(data_source)
```
Model is `BufferX` (`models/BUFFERX.py`), weights loaded from `snapshot/{experiment_id}/{stage}/best.pth`.

**Implication for CloudCropper:** to register an arbitrary `(source, target)` pair we must **construct a `data_source` dict ourselves** (mimicking the dataset loader's output) and call `model(data_source)` directly — there is no turnkey `register(src_pcd, tgt_pcd)` function upstream. The forward signature `model(data_source) -> (trans, times, inliers...)` is the integration seam. This is the main integration cost.

---

## 5. Input convention — normals/colors/xyz, scale normalization, voxel defaults

**Input = XYZ only.** No normals, no colors required.

- `MiniSpinNet.forward` (source: <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/models/patch_embedder.py>):
  > `def forward(self, pts, kpts, des_r, is_aligned_to_global_z, z_axis=None, is_aug=False)`
  > "MiniSpinNet requires only XYZ coordinates … no color or normal channels."
- Local reference frame is computed **internally via PCA**, not from supplied normals:
  > `z_axis = utils.common.cal_Z_axis(delta_x, ref_point=center)` (when `is_aligned_to_global_z=False`); else global `[0,0,1]`.

**Scale normalization (the zero-shot mechanism):** automatic, two parts —
1. **Per-patch radius normalization** inside the descriptor:
   > `def normalize(self, pts, radius): delta_x = pts / (torch.ones_like(pts) * radius)` — points divided by descriptor radius `des_r`.
2. **Density-aware radius estimation** in `BUFFERX.py` picks `des_r` per scale automatically:
   > `def density_aware_radius_estimation(src_fds_pts, src_kpts, tgt_fds_pts, tgt_kpts, min_r=0.0, max_r=5.0, tolerance=0.01, thresholds=[5, 2, 0.5])`
   > Loops over `num_scales`, computing `des_r` per scale from local point-density percentages.
   → **Scale normalization is invoked automatically inside `model.forward`**; no manual scale flag needed. This is what makes it claim zero-shot across indoor/outdoor without unit assumptions.

**Voxel / downsampling defaults** (source: `config/indoor_config.py`, <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/config/indoor_config.py>):
```
voxel_size_0        = 0.035     # (voxel_size_1 = voxel_size_0)
downsample          = 0.02      # metres
num_points_per_patch= 512
num_scales          = 3
num_fps             = 1500      # farthest-point-sampled keypoints
des_r               = 0.3       # descriptor radius (training value; inference des_r is auto via density-aware estimation)
pose_estimator      = "ransac"  # alternative: "kiss_matcher"
```
Normal estimation: **none** ("No normal estimation parameters are defined").
Dataset note (source: <https://raw.githubusercontent.com/MIT-SPARK/BUFFER-X/main/dataset/README.md>): 3DMatch training fragments were preprocessed at **1.5 cm voxel size**.

> **NOTE:** defaults above are from the *indoor* base config. Outdoor/KITTI (`outdoor_config.py`) and per-dataset configs override these (different voxel/downsample). Not all per-dataset overrides were fetched — flagged as partially verified.

**Implication:** CloudCropper feeds raw cropped XYZ (metres) → BUFFER-X auto-handles scale. We do NOT need to compute/export normals for registration (consistent with [CloudCropper scope](cloudcropper-scope.md): normals are an export concern, not needed here). Coordinate units matter only insofar as density-aware estimation has `max_r=5.0` — extremely large scenes may need pre-scaling (unverified edge case).

---

## 6. Registration output — what does it return?

From `model(data_source)` (source: test.py analysis):
```python
(trans_est,            # 4x4 estimated transformation matrix (source->target)
 times,                # timing breakdown (descriptor / pose-est / refine)
 num_inliers,          # correspondence inlier count
 num_mutual_inliers,   # mutual-NN inlier count
 num_inlier_ind,       # inlier indices
 scales_used)          # which multi-scale descriptor scale(s) were selected
```
No explicit `fitness` scalar is returned (unlike Open3D); inlier counts are the confidence proxy.

`test.py` then computes evaluation metrics (not part of the model output):
> RTE = `compute_rte(trans_est, trans)`, RRE = `compute_rre(trans_est, trans)`, `success = rte < rte_thresh and rre < rre_thresh`.
Disk outputs: per-sample `.txt` (success, RTE, RRE, inlier counts, timing) and summary `.csv` (recall, RTE/RRE mean±std, timing). (source: README + test.py)

**Implication:** CloudCropper gets the 4×4 transform + inlier counts directly. If we need a normalized fitness/confidence we must derive it (e.g. `num_inliers / num_correspondences`).

---

## 7. License

**MIT License.** (source: README badge + `LICENSE` file <https://github.com/MIT-SPARK/BUFFER-X> ; HF mirror lists `LICENSE` 1.07 kB MIT.)
→ Permissive; compatible with bundling/redistribution for CloudCropper integration. (Confirm any third-party sub-component licenses — e.g. SpinNet/BUFFER lineage, KISS-Matcher — separately if vendored.)

---

## Quick reference summary (the 7 answers)

| # | Question | Answer |
|---|----------|--------|
| 1 | Sparse-conv backbone | **None.** No Minkowski/spconv/torchsparse/KPConv-layer. Uses SpinNet-style `MiniSpinNet` (Conv2d/3d) + pointnet2_ops/KNN_CUDA/cpp_wrappers (KPConv-lineage C++ tooling). |
| 2 | Py/torch/CUDA + deps | Python 3.11, torch cu124 (or cu118/cu111-legacy 1.9.1). Deps: base.txt (open3d 0.18, numpy 1.26.3, einops, kornia, scikit-learn, tensorboard…) + huggingface_hub + torch + pointnet2_ops + KNN_CUDA 0.2 + torch-batch-svd + cpp_wrappers. |
| 3 | Pretrained weights | **Two source models** (`threedmatch`, `kitti`), each {Desc,Pose}/best.pth ≈3.67 MB (~7.3 MB per model). HF `Hyungtae-Lim/BUFFER-X` via `scripts/download_pretrained_models.py`. 3DMatch model = zero-shot generalist. |
| 4 | Inference entry | **No single-pair demo.** `python test.py --dataset … --experiment_id threedmatch`; internally `model(data_source) -> (trans, times, inliers…)`. Must build `data_source` dict ourselves to register an arbitrary pair. |
| 5 | Input convention | **XYZ only** (no normals/colors). PCA-based local frame. Scale handled **automatically** (density-aware radius est. `max_r=5.0` + per-patch radius normalize). Indoor defaults: downsample 0.02 m, num_points_per_patch 512, num_scales 3, num_fps 1500. |
| 6 | Output | `(trans_est 4×4, times, num_inliers, num_mutual_inliers, num_inlier_ind, scales_used)`. No fitness scalar. |
| 7 | License | **MIT.** |

### Open items / NOT fully verified
- `dev.txt` / `scannetpp.txt` exact contents (and the `kiss-matcher` pin) not fetched.
- Per-dataset config overrides beyond `indoor_config.py` (e.g. outdoor voxel/downsample) only partially checked.
- Behavior on very large coordinate extents vs `max_r=5.0` density-radius cap — untested assumption.
- `kiss_matcher` pose estimator path vs default `ransac` — only confirmed it exists; not traced.
