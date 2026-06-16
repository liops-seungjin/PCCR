"""
Registration pipeline: PCA 초기화 + Gradient‑SDF 최적화 + (선택) GICP + 3 D IoU 평가
"""

from __future__ import annotations

import sys
import io
import time
import math
from typing import Callable, Dict, Optional, Tuple, List, Sequence
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import open3d as o3d
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import parallel_apply

from .gradient_sdf import GradientSDFField
from .pca_transform import PCABatchSE3Transform
from .robust_loss import RobustSDFLoss

# -----------------------------------------------------------------------------#
# Optional small‑gicp                                                             #
# -----------------------------------------------------------------------------#
try:
    import small_gicp

    SMALL_GICP_AVAILABLE = True
except ImportError:  # pragma: no cover
    SMALL_GICP_AVAILABLE = False


def _orient_normals_to_centroid(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    if points is None or normals is None or len(points) == 0 or len(normals) == 0:
        return normals
    center = points.mean(axis=0)
    dirs = points - center.reshape(1, 3)
    dots = np.einsum("ij,ij->i", normals, dirs)
    oriented = normals.copy()
    oriented[dots < 0.0] *= -1.0
    return oriented


def _estimate_source_normals_for_loss(
    source_points: np.ndarray,
    *,
    radius: float,
    max_nn: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if source_points is None or len(source_points) < 10:
        return None, "too_few_source_points_for_normals"
    try:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(float(radius), 1.0e-6),
                max_nn=max(3, int(max_nn)),
            )
        )
        normals = np.asarray(pcd.normals, dtype=np.float32)
        if normals.shape != source_points.shape or not np.isfinite(normals).all():
            return None, "invalid_source_normals"
        normals = _orient_normals_to_centroid(
            np.asarray(source_points, dtype=np.float32),
            normals,
        )
        return normals.astype(np.float32, copy=False), None
    except Exception as exc:  # pragma: no cover - Open3D failures are environment-specific
        return None, f"normal_estimation_failed: {exc}"


# CloudCropper: fetch sdf/grad plus the normalized, DETACHED variance u from
# the field's uncertainty channel (when present and enabled). u must be
# detached: the heteroscedastic rho is decreasing in u, so differentiating
# through it would reward pushing points into uncertain (unobserved) space.
def _query_with_uncertainty(sdf_field, pts, enabled):
    if enabled and getattr(sdf_field, "has_uncertainty", False):
        sdf, grad, var = sdf_field.query_sdf_and_gradient(pts, return_variance=True)
        if var is not None:
            u = (
                (var.float() / sdf_field.median_variance)
                .clamp_min(1e-6)
                .detach()
            )
            return sdf, grad, u
        return sdf, grad, None
    sdf, grad = sdf_field.query_sdf_and_gradient(pts)
    return sdf, grad, None


def _rotation_matrices_from_axis_angle(rotation_params: torch.Tensor) -> torch.Tensor:
    batch_size = rotation_params.shape[0]
    angle = torch.norm(rotation_params, dim=1, keepdim=True)
    axis = rotation_params / (angle + 1.0e-8)

    K = torch.zeros(
        (batch_size, 3, 3),
        device=rotation_params.device,
        dtype=rotation_params.dtype,
    )
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    I = torch.eye(3, device=rotation_params.device, dtype=rotation_params.dtype).unsqueeze(0)
    I = I.expand(batch_size, -1, -1)
    angle_expanded = angle.unsqueeze(-1)
    return I + torch.sin(angle_expanded) * K + (1.0 - torch.cos(angle_expanded)) * torch.matmul(K, K)


def _rotate_source_normals(source_normals: torch.Tensor, batch_tf) -> torch.Tensor:
    rotations = _rotation_matrices_from_axis_angle(batch_tf.rotation_params)
    normals_expanded = source_normals.unsqueeze(0).expand(rotations.shape[0], -1, -1)
    return F.normalize(torch.matmul(normals_expanded, rotations.transpose(1, 2)), p=2, dim=-1, eps=1.0e-8)


# -----------------------------------------------------------------------------#
# Helper: Batch transform for multiple targets                                 #
# -----------------------------------------------------------------------------#
class PCABatchSE3TransformGroup(nn.Module):
    """
    Multiple PCA-based SE(3) transforms for batch processing multiple targets.
    
    This allows processing multiple CAD targets in parallel on GPU.
    """
    
    def __init__(
        self,
        source_points: np.ndarray,
        target_points_list: Sequence[np.ndarray],
        n_candidates: int,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.transforms = nn.ModuleList([
            PCABatchSE3Transform(source_points, tgt_pts, n_candidates, device=device)
            for tgt_pts in target_points_list
        ])
    
    def forward(self, src: torch.Tensor) -> List[torch.Tensor]:
        """Return list of transformed points for each target"""
        return [tf(src) for tf in self.transforms]
    
    def get_best_transforms(self, batch_losses_list: Sequence[torch.Tensor]) -> List[np.ndarray]:
        """Get best transform for each target"""
        best_tfs = []
        for tf, losses in zip(self.transforms, batch_losses_list):
            best_tf, _ = tf.get_best_transform(losses)
            best_tfs.append(best_tf)
        return best_tfs


class _RegisterBatchShard(nn.Module):
    """Lightweight worker module for multi-GPU parallel_apply."""

    def __init__(
        self,
        engine_kwargs: Dict,
        register_kwargs: Dict,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.engine_kwargs = engine_kwargs
        self.register_kwargs = register_kwargs
        self.device = device

    def forward(
        self,
        source_points: np.ndarray,
        sdf_fields: Sequence[GradientSDFField],
        target_points_list: Optional[Sequence[Optional[np.ndarray]]] = None,
        precomputed_src_dense: Optional[np.ndarray] = None,
        precomputed_downsampled: Optional[np.ndarray] = None,
        precomputed_src_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        engine = PCARegistration(
            device=self.device,
            data_parallel_devices=None,
            **self.engine_kwargs,
        )
        return engine.register_batch(
            source_points,
            sdf_fields,
            target_points_list=target_points_list,
            precomputed_src_dense=precomputed_src_dense,
            precomputed_downsampled=precomputed_downsampled,
            precomputed_src_tensor=precomputed_src_tensor,
            use_data_parallel=False,
            data_parallel_devices=None,
            **self.register_kwargs,
        )


# -----------------------------------------------------------------------------#
# Main registration class                                                      #
# -----------------------------------------------------------------------------#
def _axis_angle_batch_to_matrices(rotvecs: torch.Tensor) -> torch.Tensor:
    """(B,3) axis-angle -> (B,3,3) rotation matrices (Rodrigues)."""
    angle = torch.norm(rotvecs, dim=1, keepdim=True)
    eps = 1e-8
    axis = rotvecs / (angle + eps)
    B = rotvecs.shape[0]
    K = torch.zeros((B, 3, 3), device=rotvecs.device, dtype=rotvecs.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    I = torch.eye(3, device=rotvecs.device, dtype=rotvecs.dtype).unsqueeze(0).expand(B, -1, -1)
    a = angle.unsqueeze(-1)
    return I + torch.sin(a) * K + (1 - torch.cos(a)) * (K @ K)


class PCARegistration:
    """
    PCA + Gradient‑SDF 기반 강체 정합.

    Args:
        n_candidates: PCA 회전 후보 개수
        cauchy_c: Cauchy scale 파라미터
        learning_rate: Adam 초기 학습률
        device: torch.device
        use_gradient_weighting: Gradient 크기로 가중치 적용 여부
    """

    def __init__(
        self,
        n_candidates: int = 48,
        cauchy_c: float = 0.5,
        learning_rate: float = 1.0,
        device: Optional[torch.device] = None,
        data_parallel_devices: Optional[Sequence[torch.device | str]] = None,
        use_gradient_weighting: bool = True,
        use_amp: bool = True,
        pin_memory: bool = True,
    ) -> None:
        self.n_candidates = int(n_candidates)
        self.cauchy_c = float(cauchy_c)
        self.learning_rate = float(learning_rate)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if data_parallel_devices:
            self.data_parallel_devices = [torch.device(d) for d in data_parallel_devices]
        else:
            self.data_parallel_devices = None
        self.use_gradient_weighting = bool(use_gradient_weighting)
        self.use_amp = bool(use_amp) and self.device.type == 'cuda'  # Only use AMP on GPU
        self.pin_memory = bool(pin_memory)
    
    # --------------------------------------------------------------------- #
    # Helper methods for code reuse                                         #
    # --------------------------------------------------------------------- #
    def _prepare_source_points(
        self, 
        source_points: np.ndarray, 
        voxel_size: Optional[float] = 0.5,
        verbose: bool = True
    ) -> Tuple[np.ndarray, torch.Tensor, np.ndarray]:
        """Prepare source points: downsample and convert to tensor.
        
        Returns:
            src_dense: Original dense points for IoU computation
            src_tensor: Downsampled points as torch tensor
            src_downsampled: Downsampled points as numpy array
        """
        src_dense = source_points.copy()  # IoU 계산용 (deep copy로 원본 보존)
        src_downsampled = source_points
        
        if voxel_size and len(source_points) > 1_000:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
            src_downsampled = np.asarray(pcd.voxel_down_sample(voxel_size).points)
            if verbose:
                print(f"   ↳ downsample {len(src_dense)} → {len(src_downsampled)} (voxel={voxel_size} m)")
        
        src_tensor = torch.as_tensor(src_downsampled, dtype=torch.float32, device=self.device)
        return src_dense, src_tensor, src_downsampled
    
    def _create_optimizer_and_scheduler(
        self, 
        parameters
    ) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Create optimizer and scheduler with standard settings."""
        optimizer = optim.RAdam(parameters, lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.9,
            patience=100,
            min_lr=1.0e-4,
            threshold=1.0e-5,
            cooldown=20,
        )
        return optimizer, scheduler
    
    def _perform_optimization_step(
        self,
        batch_tf,
        src: torch.Tensor,
        sdf_field: GradientSDFField,
        loss_fn,
        opt: optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Perform a single optimization step with optional AMP."""
        if self.use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                tf_src = batch_tf(src)
                sdf_vals, grads = sdf_field.query_sdf_and_gradient(tf_src)
                loss_dict = loss_fn(sdf_vals, grads)
                batch_losses = loss_dict["loss"]
                total_loss = loss_dict["mean_loss"]
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(batch_tf.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            tf_src = batch_tf(src)
            sdf_vals, grads = sdf_field.query_sdf_and_gradient(tf_src)
            loss_dict = loss_fn(sdf_vals, grads)
            batch_losses = loss_dict["loss"]
            total_loss = loss_dict["mean_loss"]
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(batch_tf.parameters(), 1.0)
            opt.step()
        
        return batch_losses, total_loss, loss_dict
    
    def _perform_early_stopping_check(
        self,
        best_loss_hist: List[float],
        patience: int,
        min_improvement: float,
        early_stop_patience: int,
        loss_threshold: float,
        step: int,
        min_steps: int
    ) -> Tuple[int, bool]:
        """Check early stopping criteria.
        
        Returns:
            Updated patience and whether to stop
        """
        should_stop = False
        
        if len(best_loss_hist) > 1:
            improv = best_loss_hist[-2] - best_loss_hist[-1]
            if improv < min_improvement:
                patience += 3 if improv < 0 else 2
            else:
                patience = max(0, patience - 3)
        
        if len(best_loss_hist) > 10:
            long_improv = best_loss_hist[-10] - best_loss_hist[-1]
            if long_improv < min_improvement * 10:
                patience += 1
        
        if step >= min_steps and (patience >= early_stop_patience or best_loss_hist[-1] < loss_threshold):
            should_stop = True
        
        return patience, should_stop
    
    def _compute_post_optimization_metrics(
        self,
        source_points: np.ndarray,
        src_dense: np.ndarray,
        target_points: np.ndarray,
        sdf_field: GradientSDFField,
        transform: np.ndarray,
        use_gicp_refinement: bool,
        gicp_voxel_size: float,
        gicp_num_threads: Optional[int],
        compute_iou: bool,
        iou_voxel_size: float,
        iou_max_points: int,
        distance_threshold: Optional[float] = None,
        normal_threshold: float = 0.9,
        normal_weight: float = 1.0,
        normal_radius: Optional[float] = None,
        use_normal_score: bool = True,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float, bool, Optional[float], float]:
        """Compute GICP refinement and IoU after optimization.
        
        Returns:
            final_transform, gicp_time, gicp_converged, iou_val, iou_time
        """
        # GICP refinement
        gicp_time = 0.0
        gicp_conv = False
        if use_gicp_refinement and SMALL_GICP_AVAILABLE:
            if verbose:
                print("   ▶ GICP refinement ...", end="", flush=True)
            t1 = time.time()
            transform, gicp_conv = _perform_gicp_refinement(
                source_points, target_points, transform, gicp_voxel_size, gicp_num_threads
            )
            gicp_time = time.time() - t1
            if verbose:
                print(f" done ({gicp_time:.2f}s, converged={gicp_conv})")
        
        # IoU computation
        iou_val = None
        iou_time = 0.0
        if compute_iou:
            if verbose:
                print("   ▶ 3 D IoU ...", end="", flush=True)
            t2 = time.time()
            iou_val = compute_3d_iou(
                src_dense,
                sdf_field.mesh,
                transform,
                voxel_size=iou_voxel_size,
                max_points=iou_max_points,
                verbose=verbose,
                distance_threshold=distance_threshold,
                normal_threshold=normal_threshold,
                normal_weight=normal_weight,
                normal_radius=normal_radius,
                use_normal_score=use_normal_score,
                device=self.device
            )
            iou_time = time.time() - t2
            if verbose:
                print(f" IoU={iou_val:.3f} ({iou_time:.2f}s)")
        
        return transform, gicp_time, gicp_conv, iou_val, iou_time

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def register(
        self,
        source_points: np.ndarray,
        sdf_field: GradientSDFField,
        target_points: Optional[np.ndarray] = None,
        *,
        n_steps: int = 200,
        early_stop_patience: int = 30,
        min_steps: int = 50,
        min_improvement: float = 1.0e-4,
        loss_threshold: float = 1.0e-2,
        voxel_size: Optional[float] = 0.5,
        use_gicp_refinement: bool = True,
        gicp_voxel_size: float = 0.01,  # 1cm for tighter refinement
        gicp_num_threads: Optional[int] = None,
        compute_iou: bool = True,
        iou_voxel_size: float = 0.3,
        iou_max_points: int = 250_000,
        distance_threshold: Optional[float] = None,  # 매칭 거리 임계값
        normal_threshold: float = 0.9,  # 법선 정렬 임계값
        normal_weight: float = 1.0,
        normal_radius: Optional[float] = None,
        normal_loss_weight: float = 0.0,
        normal_loss_radius: Optional[float] = None,
        normal_loss_max_nn: int = 30,
        use_normal_score: bool = True,
        init_mode: str = "fft",  # "fft" (exhaustive grid search) | "pca"
        fft_voxel_size: float = 0.5,
        fft_rotation_choice: str = "AA_ICO162_S10",
        fft_topk: int = 16,
        fft_target_samples: int = 100_000,
        fft_peaks_per_rotation: int = 4,
        fft_min_peak_separation_m: float = 10.0,
        fft_refine_steps: int = 50,
        fft_expand_translation_frac: float = 0.15,  # of target max extent
        fft_expand_yaw_deg: float = 15.0,
        fft_expand_tilt_deg: float = 3.0,
        fft_max_candidates: int = 128,
        disambiguation_mesh=None,
        yaw_prior_deg: Optional[float] = None,
        yaw_prior_tolerance_deg: float = 90.0,
        callback: Optional[Callable[[Dict], None]] = None,
        use_uncertainty: bool = False,  # CloudCropper: heteroscedastic loss
    ) -> Tuple[np.ndarray, Dict]:
        """
        Registration 수행.

        Returns:
            (4×4) 최종 변환 행렬, info 딕셔너리
        """
        print(f"\n▶ Registration on {self.device} | src={len(source_points)} pts")

        # ----------------------------------------------------------------- #
        # 0) 원본 보존 / 다운샘플                                            #
        # ----------------------------------------------------------------- #
        src_dense = source_points.copy()  # IoU 계산용 (deep copy로 원본 보존)
        if voxel_size and len(source_points) > 1_000:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
            source_points = np.asarray(pcd.voxel_down_sample(voxel_size).points)
            print(f"   ↳ downsample {len(src_dense)} → {len(source_points)} (voxel={voxel_size} m)")

        # Fast host→device transfer (pinned + non_blocking)
        src_cpu = torch.as_tensor(source_points, dtype=torch.float32)
        if self.device.type == "cuda" and self.pin_memory and src_cpu.device.type == "cpu":
            src_cpu = src_cpu.pin_memory()
        src = src_cpu.to(self.device, non_blocking=self.device.type == "cuda")
        normal_loss_weight = max(0.0, float(normal_loss_weight or 0.0))
        normal_loss_radius_value = (
            float(normal_loss_radius)
            if normal_loss_radius is not None
            else max(float(voxel_size or 0.25) * 2.0, 1.0e-6)
        )
        source_normals = None
        normal_loss_reason = None
        if normal_loss_weight > 0.0:
            source_normals_np, normal_loss_reason = _estimate_source_normals_for_loss(
                source_points,
                radius=normal_loss_radius_value,
                max_nn=normal_loss_max_nn,
            )
            if source_normals_np is not None:
                normals_cpu = torch.as_tensor(source_normals_np, dtype=torch.float32)
                if self.device.type == "cuda" and self.pin_memory and normals_cpu.device.type == "cpu":
                    normals_cpu = normals_cpu.pin_memory()
                source_normals = normals_cpu.to(self.device, non_blocking=self.device.type == "cuda")
                print(
                    f"   ↳ normal loss enabled weight={normal_loss_weight:.3f} "
                    f"radius={normal_loss_radius_value:.3f}m nn={int(normal_loss_max_nn)}"
                )
            else:
                print(f"   ↳ normal loss skipped ({normal_loss_reason})")

        # ----------------------------------------------------------------- #
        # 1) 초기화 — FFT exhaustive grid search 또는 PCA                   #
        # ----------------------------------------------------------------- #
        if init_mode == "fft":
            from scipy.spatial.transform import Rotation as _ScipyRotation

            from .exhaustive_grid import exhaustive_grid_topk

            if target_points is None:
                target_points = np.asarray(
                    sdf_field.mesh.sample_points_uniformly(int(fft_target_samples)).points
                )
            # scale candidate count with how many CAD-sized footprints fit in
            # the source extent, so scene-sized clusters keep full coverage
            src_extent = np.asarray(source_points).max(axis=0) - np.asarray(source_points).min(axis=0)
            tgt_extent = np.asarray(target_points).max(axis=0) - np.asarray(target_points).min(axis=0)
            footprint_ratio = (src_extent[0] * src_extent[1]) / max(
                float(tgt_extent[0] * tgt_extent[1]), 1.0e-6
            )
            effective_topk = int(np.clip(np.ceil(footprint_ratio) * 8, int(fft_topk), 48))
            t_fft = time.time()
            fft_results = exhaustive_grid_topk(
                np.asarray(source_points, dtype=np.float64),
                np.asarray(target_points, dtype=np.float64),
                voxel_size=float(fft_voxel_size),
                rotation_choice=str(fft_rotation_choice),
                topk=effective_topk,
                peaks_per_rotation=int(fft_peaks_per_rotation),
                min_peak_separation_m=float(fft_min_peak_separation_m),
                device=self.device,
            )
            # yaw prior: in-yard block orientation is an operational
            # convention, and 180-degree yaw modes of near-symmetric hulls
            # are geometrically undecidable from a truncated single-sided
            # scan (verified: full-mesh IoU differences are within noise).
            if yaw_prior_deg is not None:
                def _candidate_yaw(T_c: np.ndarray) -> float:
                    return float(np.degrees(np.arctan2(T_c[1, 0], T_c[0, 0])))

                filtered = [
                    (T_c, s_c) for T_c, s_c in fft_results
                    if abs((_candidate_yaw(T_c) - float(yaw_prior_deg) + 180.0) % 360.0 - 180.0)
                    <= float(yaw_prior_tolerance_deg)
                ]
                if filtered:
                    print(
                        f"   ↳ yaw prior {yaw_prior_deg:+.0f}±{yaw_prior_tolerance_deg:.0f}deg: "
                        f"{len(fft_results)} -> {len(filtered)} candidates"
                    )
                    fft_results = filtered
                else:
                    print("   ↳ yaw prior filtered everything — keeping unfiltered candidates")
            # local multi-start expansion around each FFT seed: the FFT grid
            # is quantized (voxel translation, rotation preset step), so
            # sample extra starts — translations within a fraction of the
            # target extent, rotations about the seed's own CAD placement —
            # and let the batch SDF refine pull each into its local optimum.
            if int(fft_max_candidates) > len(fft_results) and fft_results:
                _rng = np.random.default_rng(0)
                _tgt_pts = np.asarray(target_points, dtype=np.float64)
                _extent = _tgt_pts.max(axis=0) - _tgt_pts.min(axis=0)
                _radius = float(fft_expand_translation_frac) * float(np.max(_extent))
                _c_local = _tgt_pts.mean(axis=0)
                _seeds = list(fft_results)
                _per_seed = max(1, (int(fft_max_candidates) - len(_seeds)) // len(_seeds))

                def _delta_rotation(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
                    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
                    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
                    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
                    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
                    Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
                    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
                    return Rz @ Ry @ Rx

                expanded = list(_seeds)
                for T0, s0 in _seeds:
                    for _ in range(_per_seed):
                        dt = _rng.uniform(-_radius, _radius, 3)
                        dt[2] *= 0.3  # ground-aligned scenes: keep z perturbation small
                        Rd = _delta_rotation(
                            np.deg2rad(_rng.uniform(-fft_expand_yaw_deg, fft_expand_yaw_deg)),
                            np.deg2rad(_rng.uniform(-fft_expand_tilt_deg, fft_expand_tilt_deg)),
                            np.deg2rad(_rng.uniform(-fft_expand_tilt_deg, fft_expand_tilt_deg)),
                        )
                        P = np.eye(4)
                        P[:3, :3] = Rd
                        P[:3, 3] = _c_local - Rd @ _c_local + dt
                        expanded.append((P @ T0, s0))
                fft_results = expanded[: int(fft_max_candidates)]
                print(
                    f"   ↳ multi-start expansion: {len(_seeds)} seeds -> "
                    f"{len(fft_results)} candidates (t±{_radius:.1f}m, "
                    f"yaw±{fft_expand_yaw_deg:.0f}deg)"
                )
            print(
                f"   ↳ FFT grid init: top-{len(fft_results)} of "
                f"{fft_rotation_choice} rotations (voxel={fft_voxel_size}m, "
                f"best score={fft_results[0][1]:.0f}, {time.time() - t_fft:.1f}s)"
            )
            _tgt_center = np.asarray(target_points, dtype=np.float64).mean(axis=0)
            for _b, (_T, _s) in enumerate(fft_results[:8]):
                _Ti = np.linalg.inv(_T)
                _cw = _Ti[:3, :3] @ _tgt_center + _Ti[:3, 3]
                print(
                    f"   ↳ fft init[{_b}] score={_s:.0f} "
                    f"center=({_cw[0]:.1f},{_cw[1]:.1f},{_cw[2]:.1f})"
                )
            batch_tf = PCABatchSE3Transform(
                source_points, None, len(fft_results), device=self.device, skip_pca=True
            )
            init_rotvecs = np.stack([
                _ScipyRotation.from_matrix(T[:3, :3]).as_rotvec() for T, _ in fft_results
            ])
            init_translations = np.stack([T[:3, 3] for T, _ in fft_results])
            with torch.no_grad():
                batch_tf.rotation_params.data = torch.tensor(
                    init_rotvecs, device=self.device, dtype=torch.float32
                )
                batch_tf.translation.data = torch.tensor(
                    init_translations, device=self.device, dtype=torch.float32
                )

            # per-candidate independent SDF refine. The legacy shared-batch
            # descent (shared scheduler, shared early-stop, global gradient
            # clipping) couples candidates and was verified to slide
            # near-optimal FFT poses onto 180-degree flips, so each coupling
            # source is removed: fixed lr, fixed step count, per-candidate
            # clipping, summed (not averaged) batch loss, and a loss mask
            # frozen at the init poses.
            if int(fft_refine_steps) > 0:
                refine_opt = optim.RAdam(batch_tf.parameters(), lr=self.learning_rate)
                refine_loss_fn = RobustSDFLoss(
                    cauchy_c=self.cauchy_c,
                    use_gradient_weighting=self.use_gradient_weighting,
                )
                with torch.no_grad():
                    tf0 = batch_tf(src)
                    refine_mask = (
                        (tf0 >= sdf_field.grid_min) & (tf0 <= sdf_field.grid_max)
                    ).all(dim=-1)
                init_rot = batch_tf.rotation_params.detach().clone()
                init_tr = batch_tf.translation.detach().clone()
                t_refine = time.time()
                for _ in range(int(fft_refine_steps)):
                    refine_opt.zero_grad(set_to_none=True)
                    tf_src_refine = batch_tf(src)
                    # CloudCropper: variance-aware query (u is None when off).
                    sdf_vals_refine, grads_refine, u_refine = _query_with_uncertainty(
                        sdf_field, tf_src_refine, use_uncertainty
                    )
                    refine_losses = refine_loss_fn(
                        sdf_vals_refine, grads_refine, valid_mask=refine_mask,
                        variances=u_refine,
                    )["loss"]
                    refine_losses.sum().backward()
                    with torch.no_grad():
                        grads_cat = torch.cat(
                            [batch_tf.rotation_params.grad, batch_tf.translation.grad],
                            dim=1,
                        )
                        clip_scale = 1.0 / grads_cat.norm(dim=1, keepdim=True).clamp(min=1.0)
                        batch_tf.rotation_params.grad.mul_(clip_scale)
                        batch_tf.translation.grad.mul_(clip_scale)
                    refine_opt.step()
                # trust region around the FFT init: the saturated Cauchy loss
                # has no restoring force along flat surface directions, so a
                # candidate can drift loss-neutrally. FFT poses are accurate
                # to ~voxel/rotation-step — anything that moved further than
                # that drifted, not converged: roll it back.
                with torch.no_grad():
                    rot_delta = (batch_tf.rotation_params - init_rot).norm(dim=1)
                    tr_delta = (batch_tf.translation - init_tr).norm(dim=1)
                    runaway = (tr_delta > 2.0) | (rot_delta > float(np.deg2rad(20.0)))
                    if bool(runaway.any()):
                        batch_tf.rotation_params.data[runaway] = init_rot[runaway]
                        batch_tf.translation.data[runaway] = init_tr[runaway]
                print(
                    f"   ↳ fft per-candidate SDF refine: {int(fft_refine_steps)} steps "
                    f"({time.time() - t_refine:.1f}s, rolled back {int(runaway.sum())} drifters)"
                )
        else:
            if target_points is None:
                target_points = np.asarray(sdf_field.mesh.sample_points_uniformly(10_000).points)
            batch_tf = PCABatchSE3Transform(
                source_points, target_points, self.n_candidates, device=self.device
            )

        # ----------------------------------------------------------------- #
        # 2) Optimizer & 손실                                              #
        # ----------------------------------------------------------------- #
        opt = optim.RAdam(batch_tf.parameters(), lr=self.learning_rate)
        loss_fn = RobustSDFLoss(
            cauchy_c=self.cauchy_c,
            use_gradient_weighting=self.use_gradient_weighting,
            normal_loss_weight=normal_loss_weight if source_normals is not None else 0.0,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.9,
            patience=100,
            min_lr=1.0e-4,
            threshold=1.0e-5,
            cooldown=20,
        )
        
        # AMP setup
        scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        autocast_enabled = self.use_amp and self.device.type == "cuda"

        fft_coverage_voxel = max(float(fft_voxel_size), 0.25)
        fft_target_voxel_count = 1
        if init_mode == "fft":
            _tgt_vox = np.unique(
                np.floor(np.asarray(target_points) / fft_coverage_voxel).astype(np.int64),
                axis=0,
            )
            fft_target_voxel_count = max(1, int(_tgt_vox.shape[0]))

        def _candidate_selection_losses(batch_losses: torch.Tensor) -> torch.Tensor:
            # Scene-sized sources reward dense-but-wrong placements under raw
            # SDF loss, so in fft mode rank candidates by inlier precision
            # inside the target bbox * CAD surface coverage and return a
            # pseudo-loss. Precision alone admits poses slid along large flat
            # faces; the coverage factor punishes them.
            if init_mode != "fft":
                return batch_losses
            with torch.no_grad():
                tf_src = batch_tf(src)
                sdf_vals, _ = sdf_field.query_sdf_and_gradient(tf_src)
                bbox_lo = torch.as_tensor(
                    np.asarray(sdf_field.mesh.get_min_bound()) - 1.0,
                    device=self.device, dtype=torch.float32,
                )
                bbox_hi = torch.as_tensor(
                    np.asarray(sdf_field.mesh.get_max_bound()) + 1.0,
                    device=self.device, dtype=torch.float32,
                )
                inside = ((tf_src >= bbox_lo) & (tf_src <= bbox_hi)).all(dim=-1)
                tau = max(float(fft_voxel_size) * 0.5, 0.25)
                inlier = inside & (sdf_vals.abs() < tau)
                inside_count = inside.sum(dim=-1)
                precision = inlier.sum(dim=-1).float() / inside_count.clamp(min=1).float()

                # CAD surface coverage: unique coverage-voxels hit by inliers
                coverage = torch.zeros_like(precision)
                vox = (tf_src / fft_coverage_voxel).floor().long()
                for b in range(vox.shape[0]):
                    if int(inlier[b].sum()) == 0:
                        continue
                    hit = torch.unique(vox[b][inlier[b]], dim=0).shape[0]
                    coverage[b] = float(hit) / float(fft_target_voxel_count)

                score = precision * coverage.clamp(max=1.0)
                score = torch.where(
                    inside_count >= 200, score, torch.zeros_like(score)
                )
                # world position of the CAD center per candidate: forward is
                # p_local = R @ p_world + t, so p_world = R^T (c_local - t)
                center_local = torch.as_tensor(
                    np.asarray(sdf_field.mesh.get_center()),
                    device=self.device, dtype=torch.float32,
                )
                rot_mats = _axis_angle_batch_to_matrices(batch_tf.rotation_params.detach())
                centers_world = torch.einsum(
                    "bij,bi->bj", rot_mats, center_local.unsqueeze(0) - batch_tf.translation.detach()
                )
                for b in range(score.shape[0]):
                    cw = centers_world[b].tolist()
                    print(
                        f"   ↳ fft cand[{b}] score={float(score[b]):.3f} "
                        f"prec={float(precision[b]):.3f} cov={float(coverage[b]):.3f} "
                        f"inside={int(inside_count[b])} "
                        f"center=({cw[0]:.1f},{cw[1]:.1f},{cw[2]:.1f})"
                    )
            return -score

        best_loss_hist: list[float] = []
        patience = 0
        t0 = time.time()
        final_tf: Optional[np.ndarray] = None

        # fft init poses are already inside the GICP convergence basin; the
        # shared-batch SDF descent (shared scheduler/early-stop, scene-sized
        # sources) corrupts them — verified to slide candidates onto
        # 180-degree flips. Skip it and go straight to per-candidate
        # GICP + cropped-IoU selection.
        n_descent_steps = 0 if init_mode == "fft" else n_steps
        step = -1
        batch_losses = torch.zeros(batch_tf.batch_size, device=self.device)
        loss_dict = {
            "loss": batch_losses,
            "sdf_loss": batch_losses,
            "normal_loss": batch_losses,
            "inlier_ratio": torch.tensor(0.0),
            "mean_loss": torch.tensor(0.0),
        }
        best_loss_hist.append(0.0)

        # ----------------------------------------------------------------- #
        # 3) 최적화 루프                                                   #
        # ----------------------------------------------------------------- #
        for step in range(n_descent_steps):
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', enabled=autocast_enabled):
                tf_src = batch_tf(src)  # (B,N,3)
                # CloudCropper: variance-aware query (u is None when off).
                sdf_vals, grads, u_main = _query_with_uncertainty(
                    sdf_field, tf_src, use_uncertainty
                )
                tf_normals = _rotate_source_normals(source_normals, batch_tf) if source_normals is not None else None
                loss_dict = loss_fn(sdf_vals, grads, source_normals=tf_normals,
                                    variances=u_main)
                batch_losses = loss_dict["loss"]
                total_loss = loss_dict["mean_loss"]
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(batch_tf.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            # ------------------------------------------------------------- #
            # 스케줄러 / early‑stop                                         #
            # ------------------------------------------------------------- #
            with torch.no_grad():
                best_loss = batch_losses.min()
                best_idx = int(batch_losses.argmin())
                scheduler.step(best_loss)

                best_loss_hist.append(best_loss.item())
                if len(best_loss_hist) > 1:
                    improv = best_loss_hist[-2] - best_loss_hist[-1]
                    if improv < min_improvement:
                        patience += 3 if improv < 0 else 2
                    else:
                        patience = max(0, patience - 3)

                if len(best_loss_hist) > 10:
                    long_improv = best_loss_hist[-10] - best_loss_hist[-1]
                    if long_improv < min_improvement * 10:
                        patience += 1

                if (
                    step >= min_steps
                    and (patience >= early_stop_patience or best_loss < loss_threshold)
                ):
                    print(f"   ↳ early‑stop @ {step}  loss={best_loss:.6f}")
                    final_tf, _ = batch_tf.get_best_transform(batch_losses)
                    break

            # ------------------------------------------------------------- #
            # user 콜백 / 로그                                              #
            # ------------------------------------------------------------- #
            if callback and step % 2 == 0:
                cur_tf, _ = batch_tf.get_best_transform(batch_losses)
                callback(
                    {
                        "step": step,
                        "best_loss": best_loss.item(),
                        "mean_loss": total_loss.item(),
                        "best_idx": best_idx,
                        "lr": opt.param_groups[0]["lr"],
                        "inlier_ratio": loss_dict["inlier_ratio"].item(),
                        "transform": cur_tf,
                    }
                )
            if step % 20 == 0:
                print(
                    f"   step {step:4d} | loss={best_loss:.6f}  "
                    f"inlier={loss_dict['inlier_ratio']:.1%}"
                )

        optim_time = time.time() - t0

        def _crop_to_target_bbox(points, tf, margin: float = 2.0):
            local = np.asarray(points) @ tf[:3, :3].T + tf[:3, 3]
            bbox_lo = np.asarray(sdf_field.mesh.get_min_bound()) - margin
            bbox_hi = np.asarray(sdf_field.mesh.get_max_bound()) + margin
            mask = np.all((local >= bbox_lo) & (local <= bbox_hi), axis=1)
            if int(mask.sum()) >= 500:
                return np.asarray(points)[mask]
            return np.asarray(points)

        def _gicp_and_iou(tf):
            conv = False
            if use_gicp_refinement and SMALL_GICP_AVAILABLE:
                gicp_src = _crop_to_target_bbox(source_points, tf) if init_mode == "fft" else source_points
                tf, conv = _perform_gicp_refinement(
                    gicp_src, target_points, tf, gicp_voxel_size, gicp_num_threads
                )
            iou = None
            if compute_iou:
                iou_src = _crop_to_target_bbox(src_dense, tf) if init_mode == "fft" else src_dense
                iou = compute_3d_iou(
                    iou_src,
                    sdf_field.mesh,
                    tf,
                    voxel_size=iou_voxel_size,
                    max_points=iou_max_points,
                    distance_threshold=distance_threshold,
                    normal_threshold=normal_threshold,
                    normal_weight=normal_weight,
                    normal_radius=normal_radius,
                    use_normal_score=use_normal_score,
                    device=self.device
                )
            return tf, conv, iou

        gicp_time = 0.0
        gicp_conv = False
        iou_val = None
        iou_time = 0.0
        if init_mode == "fft":
            # ------------------------------------------------------------- #
            # 4+5) finalist 후보 각각 GICP + cropped IoU -> 최고 IoU 선택    #
            # ------------------------------------------------------------- #
            t1 = time.time()
            scores = -_candidate_selection_losses(batch_losses)
            # flip mis-ranking is handled by the yaw prior, so the cheap
            # pre-rank is reliable within the band — cap the expensive
            # GICP+IoU stage to the top candidates
            n_finalists = min(max(int(fft_topk) // 2, 8), int(scores.shape[0]))
            finalist_order = torch.argsort(scores, descending=True)[:n_finalists].tolist()
            evaluated = []
            for b in finalist_order:
                onehot = torch.ones_like(batch_losses)
                onehot[b] = 0.0
                cand_tf, _ = batch_tf.get_best_transform(onehot)
                cand_tf, conv_b, iou_b = _gicp_and_iou(cand_tf)
                cand_yaw = float(np.degrees(np.arctan2(cand_tf[1, 0], cand_tf[0, 0])))
                print(
                    f"   > finalist[{b}] gicp_conv={conv_b} iou={iou_b} yaw={cand_yaw:.1f}"
                )
                evaluated.append((iou_b if iou_b is not None else -1.0, cand_tf, conv_b, iou_b))
            evaluated.sort(key=lambda e: e[0], reverse=True)
            # 180-degree yaw flips of near-symmetric hulls score nearly
            # identical IoU (the voxel intersection is flip-blind and scan
            # normals are unoriented). Break near-ties by oriented-normal
            # agreement: both clouds' normals oriented away from their own
            # centroid, signed cosine at matched surface points — a flipped
            # pose mates the scan with the far-side faces and goes negative.
            def _yaw_of(entry) -> float:
                tf_e = entry[1]
                return float(np.degrees(np.arctan2(tf_e[1, 0], tf_e[0, 0])))

            best_iou_value = evaluated[0][0]
            best_yaw = _yaw_of(evaluated[0])
            tie_band = [evaluated[0]]
            for e in evaluated[1:]:
                if e[0] <= 0:
                    continue
                yaw_gap = abs((_yaw_of(e) - best_yaw + 180.0) % 360.0 - 180.0)
                near_tie = e[0] >= 0.85 * best_iou_value
                # the hull-only IoU is flip-biased on truncated single-sided
                # scans, so the best candidate of an opposite yaw mode gets a
                # full-mesh rematch even when the hull IoU gap is large
                opposite_mode = yaw_gap > 90.0 and e[0] >= 0.4 * best_iou_value
                if near_tie or opposite_mode:
                    tie_band.append(e)
                if len(tie_band) >= 4:
                    break
            if len(tie_band) > 1 and disambiguation_mesh is not None:
                # 180-degree yaw modes of a near-symmetric hull are
                # geometrically self-consistent — only asymmetric detail
                # (e.g. outfitting parts excluded from the hull-only
                # registration mesh) can break the tie, so re-score the
                # near-tied poses against the full mesh.
                tie_scores = []
                for e in tie_band:
                    src_crop = _crop_to_target_bbox(src_dense, e[1])
                    tie_iou = compute_3d_iou(
                        src_crop,
                        disambiguation_mesh,
                        e[1],
                        voxel_size=iou_voxel_size,
                        max_points=iou_max_points,
                        distance_threshold=distance_threshold,
                        normal_threshold=normal_threshold,
                        normal_weight=normal_weight,
                        normal_radius=normal_radius,
                        use_normal_score=use_normal_score,
                        device=self.device,
                    )
                    tie_scores.append(float(tie_iou or 0.0))
                    print(f"   > tie-break: hull_iou={e[0]:.3f} full_mesh_iou={tie_iou:.3f}")
                best_pack = tie_band[int(np.argmax(tie_scores))]
            else:
                best_pack = evaluated[0]
            _, final_tf, gicp_conv, iou_val = best_pack
            gicp_time = time.time() - t1
        else:
            if final_tf is None:
                final_tf, _ = batch_tf.get_best_transform(batch_losses)

            if use_gicp_refinement and SMALL_GICP_AVAILABLE:
                print("   > GICP refinement ...", end="", flush=True)
                t1 = time.time()
                final_tf, gicp_conv = _perform_gicp_refinement(
                    source_points, target_points, final_tf, gicp_voxel_size, gicp_num_threads
                )
                gicp_time = time.time() - t1
                print(f" done ({gicp_time:.2f}s, converged={gicp_conv})")

            if compute_iou:
                print("   > 3D IoU ...", end="", flush=True)
                t2 = time.time()
                iou_val = compute_3d_iou(
                    src_dense,
                    sdf_field.mesh,
                    final_tf,
                    voxel_size=iou_voxel_size,
                    max_points=iou_max_points,
                    distance_threshold=distance_threshold,
                    normal_threshold=normal_threshold,
                    normal_weight=normal_weight,
                    normal_radius=normal_radius,
                    use_normal_score=use_normal_score,
                    device=self.device
                )
                iou_time = time.time() - t2
                print(f" IoU={iou_val:.3f} ({iou_time:.2f}s)")


        # ----------------------------------------------------------------- #
        # 6) 결과 정보                                                     #
        # ----------------------------------------------------------------- #
        info = {
            "final_loss": best_loss_hist[-1],
            "total_steps": step + 1,
            "optimization_time": optim_time,
            "converged": patience < early_stop_patience,
            "inlier_ratio": loss_dict["inlier_ratio"].item(),
            "gicp_applied": use_gicp_refinement and SMALL_GICP_AVAILABLE,
            "gicp_converged": gicp_conv,
            "gicp_time": gicp_time,
            "iou": iou_val,
            "iou_time": iou_time,
            "normal_weight": normal_weight,
            "normal_loss_weight": normal_loss_weight,
            "normal_loss_radius": normal_loss_radius_value,
            "normal_loss_max_nn": int(normal_loss_max_nn),
            "normal_loss_applied": source_normals is not None,
            "normal_loss_reason": normal_loss_reason,
            "final_sdf_loss": float(loss_dict.get("sdf_loss", batch_losses).min().item()),
            "final_normal_loss": float(loss_dict.get("normal_loss", batch_losses * 0.0).min().item()),
            "total_time": optim_time + gicp_time + iou_time,
            # CloudCropper: whether the heteroscedastic loss was in effect.
            "uncertainty_applied": bool(
                use_uncertainty and getattr(sdf_field, "has_uncertainty", False)
            ),
        }
        return final_tf, info
    
    # --------------------------------------------------------------------- #
    # Batch registration for multiple targets                               #
    # --------------------------------------------------------------------- #
    def register_batch(
        self,
        source_points: np.ndarray,
        sdf_fields: Sequence[GradientSDFField],
        *,
        target_points_list: Optional[Sequence[Optional[np.ndarray]]] = None,
        n_steps: int = 200,
        early_stop_patience: int = 30,
        min_steps: int = 50,
        min_improvement: float = 1.0e-4,
        loss_threshold: float = 1.0e-2,
        voxel_size: Optional[float] = 0.5,
        use_gicp_refinement: bool = True,
        gicp_voxel_size: float = 0.01,  # 1cm for tighter refinement
        gicp_num_threads: Optional[int] = None,
        compute_iou: bool = True,
        iou_voxel_size: float = 0.3,
        iou_max_points: int = 250_000,
        distance_threshold: Optional[float] = None,  # 매칭 거리 임계값
        normal_threshold: float = 0.9,  # 법선 정렬 임계값
        normal_weight: float = 1.0,
        normal_radius: Optional[float] = None,
        normal_loss_weight: float = 0.0,
        normal_loss_radius: Optional[float] = None,
        normal_loss_max_nn: int = 30,
        use_normal_score: bool = True,
        precomputed_src_dense: Optional[np.ndarray] = None,
        precomputed_downsampled: Optional[np.ndarray] = None,
        precomputed_src_tensor: Optional[torch.Tensor] = None,
        use_data_parallel: bool = False,
        data_parallel_devices: Optional[Sequence[torch.device | str]] = None,
        callback: Optional[Callable[[Dict], None]] = None,
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Batch registration against multiple targets simultaneously.

        Process multiple CAD targets in parallel on GPU for efficiency.

        Args:
            source_points: Source point cloud
            sdf_fields: List of target SDF fields
            target_points_list: Optional list of target point clouds
            gicp_top_k: Only apply GICP to top-k results by loss (None = all)
            precomputed_src_dense: Optional dense source points (for shared reuse in multi-GPU path)
            precomputed_downsampled: Optional pre-downsampled source points (skips voxel downsample here)
            precomputed_src_tensor: Optional pre-built tensor for the downsampled source (CPU pinned or device)

        Returns:
            List of transforms and info dicts for each target
        """
        n_targets = len(sdf_fields)
        assert n_targets > 0, "Need at least one target"
        
        print(f"\n▶ Batch Registration on {self.device} | src={len(source_points)} pts | {n_targets} targets")

        dp_devices = data_parallel_devices or self.data_parallel_devices
        use_dp = use_data_parallel or (dp_devices is not None and len(dp_devices) > 1)
        if use_dp and torch.cuda.is_available() and dp_devices and len(dp_devices) > 1:
            return self._register_batch_data_parallel(
                source_points,
                sdf_fields,
                target_points_list=target_points_list,
                data_parallel_devices=dp_devices,
                n_steps=n_steps,
                early_stop_patience=early_stop_patience,
                min_steps=min_steps,
                min_improvement=min_improvement,
                loss_threshold=loss_threshold,
                voxel_size=voxel_size,
                use_gicp_refinement=use_gicp_refinement,
                gicp_voxel_size=gicp_voxel_size,
                compute_iou=compute_iou,
                iou_voxel_size=iou_voxel_size,
                iou_max_points=iou_max_points,
                distance_threshold=distance_threshold,
                normal_threshold=normal_threshold,
                normal_weight=normal_weight,
                normal_radius=normal_radius,
                normal_loss_weight=normal_loss_weight,
                normal_loss_radius=normal_loss_radius,
                normal_loss_max_nn=normal_loss_max_nn,
                use_normal_score=use_normal_score,
                callback=callback,
            )
        
        # Downsample source once (reused for all targets)
        src_dense = precomputed_src_dense.copy() if precomputed_src_dense is not None else source_points.copy()
        if precomputed_downsampled is not None:
            source_points = precomputed_downsampled
        elif voxel_size and len(source_points) > 1_000:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
            source_points = np.asarray(pcd.voxel_down_sample(voxel_size).points)
            print(f"   ↳ downsample {len(src_dense)} → {len(source_points)} (voxel={voxel_size} m)")
        
        # Build source tensor with pinned memory for fast H2D copies
        if precomputed_src_tensor is not None:
            src = precomputed_src_tensor
            if src.device != self.device:
                src = src.to(self.device, non_blocking=self.device.type == "cuda")
        else:
            src_cpu = torch.as_tensor(source_points, dtype=torch.float32)
            if self.device.type == "cuda" and self.pin_memory and src_cpu.device.type == "cpu":
                src_cpu = src_cpu.pin_memory()
            src = src_cpu.to(self.device, non_blocking=self.device.type == "cuda")
        normal_loss_weight = max(0.0, float(normal_loss_weight or 0.0))
        normal_loss_radius_value = (
            float(normal_loss_radius)
            if normal_loss_radius is not None
            else max(float(voxel_size or 0.25) * 2.0, 1.0e-6)
        )
        source_normals = None
        normal_loss_reason = None
        if normal_loss_weight > 0.0:
            source_normals_np, normal_loss_reason = _estimate_source_normals_for_loss(
                source_points,
                radius=normal_loss_radius_value,
                max_nn=normal_loss_max_nn,
            )
            if source_normals_np is not None:
                normals_cpu = torch.as_tensor(source_normals_np, dtype=torch.float32)
                if self.device.type == "cuda" and self.pin_memory and normals_cpu.device.type == "cpu":
                    normals_cpu = normals_cpu.pin_memory()
                source_normals = normals_cpu.to(self.device, non_blocking=self.device.type == "cuda")
            else:
                print(f"   ↳ normal loss skipped ({normal_loss_reason})")
        
        # Initialize batch transforms
        if target_points_list is None:
            target_points_list = [
                np.asarray(sdf.mesh.sample_points_uniformly(10_000).points)
                for sdf in sdf_fields
            ]
        
        batch_tf = PCABatchSE3TransformGroup(
            source_points, target_points_list, self.n_candidates, device=self.device
        )
        
        # Create separate optimizer for each target
        optimizers = []
        schedulers = []
        for tf in batch_tf.transforms:
            opt = optim.RAdam(tf.parameters(), lr=self.learning_rate)
            optimizers.append(opt)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.9, patience=100, min_lr=1.0e-4
            )
            schedulers.append(scheduler)
        
        loss_fn = RobustSDFLoss(
            cauchy_c=self.cauchy_c,
            use_gradient_weighting=self.use_gradient_weighting,
            normal_loss_weight=normal_loss_weight if source_normals is not None else 0.0,
        )
        
        # AMP setup
        scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        autocast_enabled = self.use_amp and self.device.type == "cuda"
        
        best_loss_hist = []
        patience = 0
        t0 = time.time()
        final_tfs = None
        
        # Per-target tracking
        last_inlier_ratios = [0.0] * n_targets
        
        # Optimization loop
        for step in range(n_steps):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            
            tf_src_list = batch_tf(src)  # List of (B,N,3)
            
            batch_losses_list: list[torch.Tensor] = []
            mean_losses: list[torch.Tensor] = []
            
            with torch.amp.autocast(device_type='cuda', enabled=autocast_enabled):
                for i, (tf_src, sdf_field) in enumerate(zip(tf_src_list, sdf_fields)):
                    sdf_vals, grads = sdf_field.query_sdf_and_gradient(tf_src)
                    tf_normals = (
                        _rotate_source_normals(source_normals, batch_tf.transforms[i])
                        if source_normals is not None
                        else None
                    )
                    loss_dict = loss_fn(sdf_vals, grads, source_normals=tf_normals)
                    batch_losses_list.append(loss_dict["loss"].detach())
                    mean_losses.append(loss_dict["mean_loss"])
                    last_inlier_ratios[i] = loss_dict["inlier_ratio"].item()
            
            # Single backward/scale for all targets to keep GPU busy
            total_loss = torch.stack(mean_losses).sum()
            scaler.scale(total_loss).backward()
            
            for opt, tf_module in zip(optimizers, batch_tf.transforms):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(tf_module.parameters(), 1.0)
                scaler.step(opt)
            scaler.update()
            
            # Early stopping
            with torch.no_grad():
                best_losses = [bl.min() for bl in batch_losses_list]
                best_losses_tensor = torch.stack(best_losses)
                best_loss, best_idx_tensor = best_losses_tensor.min(dim=0)
                best_idx = int(best_idx_tensor)
                
                for scheduler, bl in zip(schedulers, batch_losses_list):
                    scheduler.step(bl.min())
                
                best_loss_hist.append(best_loss.item())
                if len(best_loss_hist) > 1:
                    improv = best_loss_hist[-2] - best_loss_hist[-1]
                    if improv < min_improvement:
                        patience += 3 if improv < 0 else 2
                    else:
                        patience = max(0, patience - 3)
                
                if step >= min_steps and (patience >= early_stop_patience or best_loss < loss_threshold):
                    print(f"   ↳ early‑stop @ {step}  loss={best_loss:.6f}")
                    final_tfs = batch_tf.get_best_transforms(batch_losses_list)
                    break
            
            if step % 20 == 0:
                worst_loss = best_losses_tensor.max().item()
                best_inlier = last_inlier_ratios[best_idx] if last_inlier_ratios else 0.0
                print(f"   step {step:4d} | best={best_loss:.6f}  (worst={worst_loss:.6f}) inlier={best_inlier:.1%}")
        
        if final_tfs is None:
            final_tfs = batch_tf.get_best_transforms(batch_losses_list)
        
        optim_time = time.time() - t0

        # Compute per-target results (GICP + IoU)
        infos = []
        gicp_workers = min(
            n_targets,
            max(1, mp.cpu_count() - 2)
        ) if use_gicp_refinement and SMALL_GICP_AVAILABLE else 0
        per_job_threads = max(1, (mp.cpu_count() - 2) // gicp_workers) if gicp_workers else 0

        def _run_gicp(idx: int) -> Tuple[int, np.ndarray, float, bool]:
            tf_i = final_tfs[idx]
            gicp_time = 0.0
            gicp_conv = False
            if use_gicp_refinement and SMALL_GICP_AVAILABLE:
                t1 = time.time()
                tf_i, gicp_conv = _perform_gicp_refinement(
                    source_points, target_points_list[idx], tf_i, gicp_voxel_size, per_job_threads
                )
                gicp_time = time.time() - t1
                print(f"   ▶ GICP [{idx+1}/{n_targets}] voxel={gicp_voxel_size}m → {'✓' if gicp_conv else '✗'} ({gicp_time:.2f}s)")
            return idx, tf_i, gicp_time, gicp_conv

        gicp_results: List[Tuple[int, np.ndarray, float, bool]] = []
        if gicp_workers and n_targets > 1:
            with ThreadPoolExecutor(max_workers=gicp_workers) as pool:
                futures = [pool.submit(_run_gicp, idx) for idx in range(n_targets)]
                for fut in as_completed(futures):
                    gicp_results.append(fut.result())
        else:
            for idx in range(n_targets):
                gicp_results.append(_run_gicp(idx))

        gicp_results.sort(key=lambda x: x[0])

        for idx, tf_i, gicp_time, gicp_conv in gicp_results:
            iou_val = None
            iou_time = 0.0
            if compute_iou:
                t2 = time.time()
                iou_val = compute_3d_iou(
                    src_dense, sdf_fields[idx].mesh, tf_i,
                    voxel_size=iou_voxel_size, max_points=iou_max_points, verbose=False,
                    distance_threshold=distance_threshold,
                    normal_threshold=normal_threshold,
                    normal_weight=normal_weight,
                    normal_radius=normal_radius,
                    use_normal_score=use_normal_score,
                    device=self.device
                )
                iou_time = time.time() - t2

            infos.append({
                "final_loss": best_losses[idx].item(),
                "total_steps": step + 1,
                "optimization_time": optim_time,
                "converged": patience < early_stop_patience,
                "inlier_ratio": last_inlier_ratios[idx],
                "gicp_applied": use_gicp_refinement and SMALL_GICP_AVAILABLE,
                "gicp_converged": gicp_conv,
                "gicp_time": gicp_time,
                "iou": iou_val,
                "iou_time": iou_time,
                "normal_weight": normal_weight,
                "normal_loss_weight": normal_loss_weight,
                "normal_loss_radius": normal_loss_radius_value,
                "normal_loss_max_nn": int(normal_loss_max_nn),
                "normal_loss_applied": source_normals is not None,
                "normal_loss_reason": normal_loss_reason,
                "total_time": optim_time + gicp_time + iou_time,
            })

            final_tfs[idx] = tf_i  # Update with GICP result
        
        print(f"   ✓ Batch completed in {optim_time:.2f}s")
        return final_tfs, infos


    def _register_batch_data_parallel(
        self,
        source_points: np.ndarray,
        sdf_fields: Sequence[GradientSDFField],
        *,
        target_points_list: Optional[Sequence[Optional[np.ndarray]]] = None,
        data_parallel_devices: Optional[Sequence[torch.device | str]] = None,
        **register_kwargs,
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Run register_batch across multiple CUDA devices using torch parallel_apply.
        Targets are evenly sharded per device to avoid thread-safety issues.
        """
        devices = data_parallel_devices or self.data_parallel_devices
        if not devices or len(devices) <= 1 or not torch.cuda.is_available():
            # Fallback to single-device path
            register_kwargs.pop("use_data_parallel", None)
            register_kwargs.pop("data_parallel_devices", None)
            single_engine = PCARegistration(
                n_candidates=self.n_candidates,
                cauchy_c=self.cauchy_c,
                learning_rate=self.learning_rate,
                device=self.device,
                data_parallel_devices=None,
                use_gradient_weighting=self.use_gradient_weighting,
                use_amp=self.use_amp,
                pin_memory=self.pin_memory,
            )
            return single_engine.register_batch(
                source_points,
                sdf_fields,
                target_points_list=target_points_list,
                use_data_parallel=False,
                data_parallel_devices=None,
                **register_kwargs,
            )

        # Pre-downsample/pin source once to avoid repeated CPU work per shard
        voxel_size = register_kwargs.get("voxel_size", None)
        src_dense = source_points.copy()
        src_downsampled = source_points
        if voxel_size and len(source_points) > 1_000:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
            src_downsampled = np.asarray(pcd.voxel_down_sample(voxel_size).points)
            print(f"   ↳ downsample {len(src_dense)} → {len(src_downsampled)} (voxel={voxel_size} m)")
        src_tensor_cpu = torch.as_tensor(src_downsampled, dtype=torch.float32)
        if torch.cuda.is_available() and src_tensor_cpu.device.type == "cpu":
            src_tensor_cpu = src_tensor_cpu.pin_memory()

        device_list = [torch.device(d) for d in devices]
        shard_size = math.ceil(len(sdf_fields) / len(device_list))
        sdf_shards = [
            sdf_fields[i : i + shard_size] for i in range(0, len(sdf_fields), shard_size)
        ]
        tp_shards = None
        if target_points_list is not None:
            tp_shards = [
                target_points_list[i : i + shard_size]
                for i in range(0, len(target_points_list), shard_size)
            ]

        # Strip parallel flags for per-device workers
        worker_kwargs = register_kwargs.copy()
        worker_kwargs.pop("use_data_parallel", None)
        worker_kwargs.pop("data_parallel_devices", None)

        engine_params = {
            "n_candidates": self.n_candidates,
            "cauchy_c": self.cauchy_c,
            "learning_rate": self.learning_rate,
            "use_gradient_weighting": self.use_gradient_weighting,
            "use_amp": self.use_amp,
            "pin_memory": self.pin_memory,
        }

        modules = []
        inputs = []
        for idx, shard in enumerate(sdf_shards):
            device = device_list[idx % len(device_list)]
            moved_fields = []
            for field in shard:
                if getattr(field, "device", None) and str(field.device) == str(device):
                    moved_fields.append(field)
                elif hasattr(field, "copy_to_device"):
                    moved_fields.append(field.copy_to_device(device))
                else:
                    moved_fields.append(field)

            shard_targets = tp_shards[idx] if tp_shards is not None else None

            modules.append(_RegisterBatchShard(engine_params, worker_kwargs, device))
            inputs.append(
                (
                    source_points,
                    moved_fields,
                    shard_targets,
                    src_dense,
                    src_downsampled,
                    src_tensor_cpu,
                )
            )

        results = parallel_apply(
            modules,
            inputs,
            devices=device_list[: len(modules)],
        )

        transforms: List[np.ndarray] = []
        infos: List[Dict] = []
        for tfs, info in results:
            transforms.extend(tfs)
            infos.extend(info)

        return transforms, infos



# -----------------------------------------------------------------------------#
# Multi-target registration for CAD matching                                   #
# -----------------------------------------------------------------------------#
class MultiTargetRegistration(PCARegistration):
    """
    Multiple target registration for finding best CAD match.
    
    Performs registration of each source point cloud against all target CADs
    and selects the best match based on 3D IoU.
    """
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def best_match_registration(
        self,
        source_point_clouds: list[np.ndarray],
        target_sdf_fields: list[GradientSDFField],
        *,
        target_names: Optional[list[str]] = None,
        chunk_size: int = 720,
        verbose: bool = True,
        **register_kwargs
    ) -> list[Dict]:
        """
        Find best CAD match for each source point cloud using batch processing.
        
        Args:
            source_point_clouds: List of source point clouds
            target_sdf_fields: List of pre-built GradientSDFField objects for each CAD
            target_names: Optional names for each target CAD
            chunk_size: Number of targets to process in parallel (default: 8)
            verbose: Print progress
            **register_kwargs: Additional arguments passed to register()
            
        Returns:
            List of dictionaries, one per source point cloud, containing:
                - best_target_idx: Index of best matching CAD
                - best_target_name: Name of best matching CAD (if provided)
                - best_transform: Best transformation matrix
                - best_iou: Best IoU score
                - all_results: Registration results for all targets
        """
        if not target_names:
            target_names = [f"Target_{i}" for i in range(len(target_sdf_fields))]
        
        all_results = []
        n_targets = len(target_sdf_fields)
        
        # Process each source point cloud
        for src_idx, source_points in enumerate(source_point_clouds):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Processing source point cloud {src_idx + 1}/{len(source_point_clouds)}")
                print(f"{'='*60}")
            
            best_iou = -1.0
            best_target_idx = -1
            best_transform = None
            best_info = None
            target_results = []
            
            # Process targets in chunks for GPU efficiency
            for chunk_start in range(0, n_targets, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_targets)
                chunk_sdf_fields = target_sdf_fields[chunk_start:chunk_end]
                chunk_names = target_names[chunk_start:chunk_end]
                
                if verbose:
                    print(f"\n▶ Processing chunk [{chunk_start}:{chunk_end}] ({len(chunk_sdf_fields)} targets)")
                
                try:
                    # Always compute IoU for comparison
                    kwargs = register_kwargs.copy()
                    kwargs['compute_iou'] = True
                    
                    # Batch registration for this chunk
                    transforms, infos = self.register_batch(
                        source_points,
                        chunk_sdf_fields,
                        **kwargs
                    )
                    
                    # Process results from this chunk
                    for local_idx, (transform, info) in enumerate(zip(transforms, infos)):
                        global_idx = chunk_start + local_idx
                        iou = info.get('iou', 0.0)
                        
                        target_results.append({
                            'target_idx': global_idx,
                            'target_name': target_names[global_idx],
                            'transform': transform,
                            'iou': iou,
                            'info': info
                        })
                        
                        # Update best match
                        if iou > best_iou:
                            best_iou = iou
                            best_target_idx = global_idx
                            best_transform = transform
                            best_info = info
                        
                        if verbose:
                            print(f"   {target_names[global_idx]:20s} | IoU: {iou:.3f} {'<-- BEST' if global_idx == best_target_idx else ''}")

                except Exception as e:
                    print(f"   ERROR: Batch registration failed - {str(e)}")
                    # Add error entries for this chunk
                    for local_idx in range(len(chunk_sdf_fields)):
                        global_idx = chunk_start + local_idx
                        target_results.append({
                            'target_idx': global_idx,
                            'target_name': target_names[global_idx],
                            'transform': None,
                            'iou': 0.0,
                            'info': {'error': str(e)}
                        })
                finally:
                    # Clear CUDA cache after each chunk to prevent OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Compile results for this source
            result = {
                'source_idx': src_idx,
                'best_target_idx': best_target_idx,
                'best_target_name': target_names[best_target_idx] if best_target_idx >= 0 else None,
                'best_transform': best_transform,
                'best_iou': best_iou,
                'best_info': best_info,
                'all_results': target_results
            }
            
            if verbose:
                print(f"\n🎯 Best match for source {src_idx + 1}: {result['best_target_name']} (IoU: {best_iou:.3f})")
            
            all_results.append(result)
        
        return all_results

# -----------------------------------------------------------------------------#
# Fast 3D IoU (Normal-aware Voxel Hashing)                                     #
# -----------------------------------------------------------------------------#
from ._iou_func import (
    NormalAwareIoU,
    compute_normal_aware_iou,
    SoftVoxelNormalIoU,
    compute_soft_normal_iou,
    compute_normal_aligned_score,
    compute_normal_aligned_score_gpu,
)


def compute_3d_iou(
    source_points: np.ndarray,
    target_mesh: o3d.geometry.TriangleMesh,
    transform: np.ndarray,
    *,
    voxel_size: float = 0.05,  # 5cm voxel for good resolution
    max_points: int = 100_000,
    verbose: bool = True,
    use_normal_score: bool = True,  # If False, skip normal filtering (geometric voxel IoU)
    use_gpu_score: bool = True,
    distance_threshold: Optional[float] = None,  # Legacy, mapped to voxel_size
    normal_threshold: float = 0.9,  # Cosine threshold (~25 degrees)
    normal_weight: float = 1.0,
    normal_score_downsample_voxel: Optional[float] = None,
    normal_radius: Optional[float] = None,
    device: Optional[str | torch.device] = None,
    gpu_chunk_n: int = 2048,  # Ignored (legacy)
    gpu_chunk_m: int = 2048,  # Ignored (legacy)
) -> float:
    """
    Compute Normal-aware 3D IoU between source point cloud and target mesh.

    Uses GPU-optimized voxel hashing with normal vector similarity filtering:
    1. Voxelizes both point clouds using spatial hashing
    2. Computes average normal per voxel
    3. Filters intersection by normal similarity (parallel or anti-parallel)
    4. IoU = valid_intersection / union

    Args:
        source_points: (N, 3) Source point cloud
        target_mesh: Target mesh (Open3D TriangleMesh)
        transform: (4, 4) Transformation matrix
        voxel_size: Voxel grid size in meters (default: 0.10m = 10cm)
        max_points: Maximum points to sample
        normal_threshold: Cosine similarity threshold (0.9 = ~25 degrees)
        normal_radius: Radius for normal estimation (default: voxel_size*2)
        device: 'cuda' or 'cpu'

    Returns:
        IoU score (0~1)
    """
    # Handle legacy distance_threshold parameter
    # Downsample scale for normals (fixed small voxel unless overridden)
    downsample_voxel = normal_score_downsample_voxel if normal_score_downsample_voxel is not None else 0.05
    # IoU voxel size follows distance_threshold if provided, otherwise the explicit voxel_size argument
    effective_voxel_size = distance_threshold if distance_threshold is not None else voxel_size
    # Normal radius defaults to 2x normal-downsample (e.g., 0.025 → 0.05) to stay local to the normal grid
    radius = normal_radius if normal_radius is not None else downsample_voxel * 2.0

    # Determine device
    dev = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Compute IoU (normal-aware if enabled, otherwise geometric)
    iou, _ = compute_normal_aware_iou(
        source_points,
        target_mesh,
        transform,
        voxel_size=effective_voxel_size,
        cos_thresh=None,  # always use 1+cos without threshold
        normal_radius=radius,
        max_points=max_points,
        device=dev,
        verbose=verbose,
        downsample_voxel=downsample_voxel,
        use_normal_score=use_normal_score,
        normal_weight=normal_weight,
    )
    return iou



# -----------------------------------------------------------------------------#
# (선택) small‑gicp refinement                                                 #
# -----------------------------------------------------------------------------#
def _perform_gicp_refinement(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    init_tf: np.ndarray,
    down_res: float = 0.01,  # 1cm default for fine refinement
    max_corr_dist: Optional[float] = None,  # Auto-calculated if None
    num_threads: Optional[int] = None,
) -> Tuple[np.ndarray, bool]:
    if not SMALL_GICP_AVAILABLE:
        return init_tf, False

    try:
        # Auto-calculate max correspondence distance: 10x voxel size
        if max_corr_dist is None:
            max_corr_dist = max(down_res * 10, 0.05)  # Minimum 5cm
        if num_threads is None:
            num_threads = max(1, mp.cpu_count() - 2)

        src_trans = (init_tf[:3, :3] @ source_pts.T).T + init_tf[:3, 3]
        tgt, tgt_tree = small_gicp.preprocess_points(target_pts, downsampling_resolution=down_res)
        src, src_tree = small_gicp.preprocess_points(src_trans, downsampling_resolution=down_res)

        res = small_gicp.align(
            tgt,
            src,
            tgt_tree,
            registration_type="GICP",
            max_correspondence_distance=max_corr_dist,
            num_threads=num_threads,
            max_iterations=100,  # More iterations for fine alignment
        )
        refined_tf = res.T_target_source @ init_tf
        return refined_tf, bool(res.converged)
    except Exception:  # pragma: no cover
        return init_tf, False
