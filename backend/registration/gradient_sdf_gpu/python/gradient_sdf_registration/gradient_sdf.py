"""
Gradient‑SDF 필드: SDF + 정규화 gradient 를 가지는 4‑채널 3‑D 그리드
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch


class GradientSDFField:
    """
    정규화 gradient 를 포함한 SDF 3‑D 그리드.

    Args:
        mesh: Open3D TriangleMesh
        resolution: 격자 해상도 (한 변)
        device: torch.device
        padding_ratio: 모델 대각선 대비 padding 비율
    """

    def __init__(
        self,
        mesh: o3d.geometry.TriangleMesh,
        *,
        resolution: int = 100,
        device: Optional[torch.device] = None,
        padding_ratio: float = 0.2,
        use_amp: bool = True,
    ) -> None:
        self.mesh = mesh
        self.resolution = int(resolution)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding_ratio = float(padding_ratio)
        self.use_amp = bool(use_amp) and self.device.type == 'cuda'  # Only use AMP on GPU

        print(f"▶ build Gradient‑SDF {resolution}³ on {self.device}")
        t0 = time.time()
        self._build_gradient_sdf_grid()
        print(f"   done in {time.time() - t0:.2f}s  | voxel={self.voxel_size.cpu().numpy()}")

    # --------------------------------------------------------------------- #
    # 내부: 그리드 구축                                                    #
    # --------------------------------------------------------------------- #
    def _build_gradient_sdf_grid(self) -> None:
        # a) Bounding box + padding
        bmin, bmax = self.mesh.get_min_bound(), self.mesh.get_max_bound()
        diag = np.linalg.norm(bmax - bmin)
        pad = np.clip(diag * self.padding_ratio, 1.0, 10.0)
        self.grid_min = torch.as_tensor(bmin - pad, dtype=torch.float32, device=self.device)
        self.grid_max = torch.as_tensor(bmax + pad, dtype=torch.float32, device=self.device)
        self.voxel_size = (self.grid_max - self.grid_min) / (self.resolution - 1)

        # b) 격자 좌표 생성 (CPU → Tensor)
        axes = [np.linspace(self.grid_min[i].cpu(), self.grid_max[i].cpu(), self.resolution) for i in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        grid_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)

        # c) Open3D raycasting 으로 SDF
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        sdf = scene.compute_signed_distance(o3c.Tensor(grid_pts, o3c.float32)).numpy().reshape(
            self.resolution, self.resolution, self.resolution
        )

        # d) 중앙차분 gradient + unit normalization
        gx, gy, gz = np.gradient(sdf)
        grad_mag = np.sqrt(gx**2 + gy**2 + gz**2) + 1.0e-8
        gx, gy, gz = gx / grad_mag, gy / grad_mag, gz / grad_mag

        # e) 4‑채널 Tensor [SDF, gx, gy, gz]
        self.grid = torch.tensor(
            np.stack((sdf, gx, gy, gz), axis=-1), dtype=torch.float32, device=self.device
        )
        # CloudCropper: optional 5th channel (GPIS variance), see
        # add_uncertainty_channel(); absent until requested.
        self.has_uncertainty = False
        self.median_variance = 1.0
        self.uncertainty_trunc = 0.0

    # --------------------------------------------------------------------- #
    # CloudCropper: uncertainty (variance) channel                           #
    # --------------------------------------------------------------------- #
    def add_uncertainty_channel(
        self, points: np.ndarray, normals: np.ndarray, *, trunc: float, spacing: float
    ) -> None:
        """CloudCropper: GPIS-style per-voxel variance channel, ported from the
        removed native C++ backend (see docs/design/06-registration.md).

        Per voxel center v with nearest observed point p (normal n) at
        distance r:
            var = sf2*(1 - rho_32(r)) + s2(p) + quant
        with rho_32(r) = (1 + sqrt(3) r/ell) exp(-sqrt(3) r/ell)  (Matern-3/2),
        ell = 3*spacing, sf2 = trunc^2, s2(p) = mean squared point-to-plane
        distance of p's 8 nearest neighbours (self included) to plane (p, n),
        and quant = (0.25 h)^2 the voxel quantization floor (h = max step).
        """
        from scipy.spatial import cKDTree

        res = self.resolution
        gmin = self.grid_min.cpu().numpy().astype(np.float64)
        gmax = self.grid_max.cpu().numpy().astype(np.float64)
        axes = [np.linspace(gmin[i], gmax[i], res) for i in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (res^3, 3)

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        tree = cKDTree(pts)

        # Per-TARGET-POINT local plane residual s2: 8-NN including self (the
        # self term is 0 but counted in the mean, matching the C++ port).
        k = min(8, len(pts))
        _, nn8 = tree.query(pts, k=k, workers=-1)
        nn8 = nn8.reshape(len(pts), k)
        pd = np.einsum("pkj,pj->pk", pts[nn8] - pts[:, None, :], nrm)
        s2 = np.mean(pd * pd, axis=1)  # (P,)

        ell = 3.0 * max(float(spacing), 1e-9)
        sf2 = float(trunc) * float(trunc)
        h = float(self.voxel_size.max().cpu())
        quant = (0.25 * h) ** 2

        dist, j = tree.query(centers, k=1, workers=-1)  # (res^3,)
        a = np.sqrt(3.0) * dist / ell
        rho = (1.0 + a) * np.exp(-a)  # Matern-3/2 data proximity
        var = sf2 * (1.0 - rho) + s2[j] + quant

        self.median_variance = float(max(np.median(var), 1e-12))
        self.uncertainty_trunc = float(trunc)
        var_t = torch.tensor(
            var.reshape(res, res, res, 1).astype(np.float32), device=self.device
        )
        self.grid = torch.cat([self.grid[..., :4], var_t], dim=-1)  # idempotent
        self.has_uncertainty = True

    # --------------------------------------------------------------------- #
    # Public API: SDF & gradient 질의                                      #
    # --------------------------------------------------------------------- #
    def query_sdf_and_gradient(
        self, points: torch.Tensor, return_variance: bool = False
    ):
        """
        Trilinear interpolation 기반 SDF / gradient 반환.

        points: (B,N,3) 또는 (N,3)
        return_variance: CloudCropper — True면 (sdf, grad, var)를 반환한다.
            var는 5번째 채널(불확실성)의 보간값이며 채널이 없으면 None.
        """
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                return self._query_sdf_and_gradient_impl(points, return_variance)
        else:
            return self._query_sdf_and_gradient_impl(points, return_variance)

    def _query_sdf_and_gradient_impl(
        self, points: torch.Tensor, return_variance: bool = False
    ):
        orig_shape = points.shape
        batched = len(orig_shape) == 3
        pts = points.reshape(-1, 3) if batched else points  # (M,3)

        # 격자 좌표
        gcoord = (pts - self.grid_min) / self.voxel_size
        gcoord = torch.clamp(gcoord, 0, self.resolution - 1.001)

        g0 = gcoord.floor().long()  # (M,3)
        frac = gcoord - g0.float()

        # index clamp
        g0 = torch.clamp(g0, 0, self.resolution - 2)
        g1 = g0 + 1

        # Corner gather
        x0, y0, z0 = g0.T
        x1, y1, z1 = g1.T
        v000 = self.grid[x0, y0, z0]
        v001 = self.grid[x0, y0, z1]
        v010 = self.grid[x0, y1, z0]
        v011 = self.grid[x0, y1, z1]
        v100 = self.grid[x1, y0, z0]
        v101 = self.grid[x1, y0, z1]
        v110 = self.grid[x1, y1, z0]
        v111 = self.grid[x1, y1, z1]

        wx, wy, wz = frac.T.unsqueeze(-1)  # broadcast (M,1)

        interp = (
            (1 - wx)
            * (1 - wy)
            * (1 - wz)
            * v000
            + (1 - wx)
            * (1 - wy)
            * wz
            * v001
            + (1 - wx)
            * wy
            * (1 - wz)
            * v010
            + (1 - wx)
            * wy
            * wz
            * v011
            + wx
            * (1 - wy)
            * (1 - wz)
            * v100
            + wx
            * (1 - wy)
            * wz
            * v101
            + wx
            * wy
            * (1 - wz)
            * v110
            + wx
            * wy
            * wz
            * v111
        )

        sdf = interp[:, 0]
        grad = interp[:, 1:4]
        # CloudCropper: interpolated variance channel (clamped positive).
        var = (
            interp[:, 4].clamp_min(1e-12)
            if return_variance and interp.shape[-1] > 4
            else None
        )

        if batched:
            B, N = orig_shape[0], orig_shape[1]
            sdf = sdf.view(B, N)
            grad = grad.view(B, N, 3)
            if var is not None:
                var = var.view(B, N)
        if return_variance:
            return sdf, grad, var
        return sdf, grad

    # --------------------------------------------------------------------- #
    # Info                                                                  #
    # --------------------------------------------------------------------- #
    def get_grid_info(self) -> dict:
        return {
            "resolution": self.resolution,
            "grid_min": self.grid_min.cpu().numpy(),
            "grid_max": self.grid_max.cpu().numpy(),
            "voxel_size": self.voxel_size.cpu().numpy(),
            "device": str(self.device),
        }

    # --------------------------------------------------------------------- #
    # Device Transfer                                                       #
    # --------------------------------------------------------------------- #
    def copy_to_device(self, target_device) -> "GradientSDFField":
        """
        Create a copy of this SDF field on the target device.

        Args:
            target_device: Target device (torch.device or string like "cuda:1")

        Returns:
            New GradientSDFField instance with tensors on target device
        """
        # Normalize to torch.device
        if not isinstance(target_device, torch.device):
            target_device = torch.device(target_device)

        # Check if already on target device
        if self.device == target_device or str(self.device) == str(target_device):
            return self

        # Create shallow copy without rebuilding the SDF grid
        new_field = object.__new__(GradientSDFField)
        new_field.mesh = self.mesh
        new_field.resolution = self.resolution
        new_field.device = target_device
        new_field.padding_ratio = self.padding_ratio
        new_field.use_amp = self.use_amp and target_device.type == 'cuda'

        # Copy tensors to target device
        new_field.grid_min = self.grid_min.to(target_device)
        new_field.grid_max = self.grid_max.to(target_device)
        new_field.voxel_size = self.voxel_size.to(target_device)
        new_field.grid = self.grid.to(target_device)
        # CloudCropper: uncertainty-channel metadata travels with the grid.
        new_field.has_uncertainty = getattr(self, "has_uncertainty", False)
        new_field.median_variance = getattr(self, "median_variance", 1.0)
        new_field.uncertainty_trunc = getattr(self, "uncertainty_trunc", 0.0)

        return new_field
