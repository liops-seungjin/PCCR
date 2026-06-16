"""
Normal-aware IoU computation for point cloud registration evaluation.
GPU-optimized voxel hashing with normal vector similarity filtering.
"""
import time
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F


class NormalAwareIoU:
    """
    GPU-optimized Normal-aware IoU using voxel hashing.

    Algorithm:
    1. Voxelize both point clouds using spatial hashing
    2. Compute average normal per voxel (scatter add + normalize)
    3. Use sorted search to find geometric intersection
    4. Filter by normal similarity (parallel or anti-parallel)
    5. IoU = valid_intersection / union
    """

    def __init__(self, voxel_size=0.05, cos_thresh=None, device='cuda', normal_weight: float = 1.0):
        self.voxel_size = voxel_size
        self.cos_thresh = cos_thresh  # None = soft 1+cos scoring
        self.device = device
        self.normal_weight = max(0.0, float(normal_weight or 0.0))

    def compute(self, points1, normals1, points2, normals2):
        """
        Compute Normal-aware IoU between two point clouds.

        Args:
            points1: (N, 3) Point coordinates of cloud 1
            normals1: (N, 3) Normal vectors of cloud 1
            points2: (M, 3) Point coordinates of cloud 2
            normals2: (M, 3) Normal vectors of cloud 2

        Returns:
            iou: Normal-filtered IoU value (0~1)
            intersection_count: Number of valid intersection voxels
            union_count: Number of union voxels
        """
        # 1. Voxelize & Average Normals
        k1, n1 = self._voxelize(points1, normals1)
        k2, n2 = self._voxelize(points2, normals2)

        # 2. Sorted Search for Intersection
        # k1, k2는 이미 unique하고 정렬되어 있지 않을 수 있으므로 정렬 수행
        k1, idx_sort1 = torch.sort(k1)
        n1 = n1[idx_sort1]

        k2, idx_sort2 = torch.sort(k2)
        n2 = n2[idx_sort2]

        # searchsorted: k1의 요소들이 k2의 어디에 들어갈지 찾음
        idx_in_k2 = torch.searchsorted(k2, k1)

        # 3. Valid Intersection Indexing (버그 수정됨)
        # 인덱스가 범위를 벗어나지 않도록 클램핑 후 비교 (실제 유효성 검사는 아래에서)
        idx_in_k2_clamped = idx_in_k2.clamp(max=k2.size(0) - 1)

        # 교집합 조건:
        # 1) 인덱스가 범위 내에 있어야 함 (searchsorted 결과가 len(k2)면 매칭 실패)
        # 2) 실제 키 값이 같아야 함 (k2[idx] == k1)
        match_mask = (idx_in_k2 < k2.size(0)) & (k2[idx_in_k2_clamped] == k1)

        # 교집합이 하나도 없으면 조기 종료
        if match_mask.sum() == 0:
            return 0.0, 0, (k1.size(0) + k2.size(0))

        # 4. Normal weighting: 교집합 voxel의 cos 유사도를 가중치로 사용 (부호 일치 기준)
        matched_n1 = n1[match_mask]
        matched_n2 = n2[idx_in_k2_clamped[match_mask]]
        cos_sim = (matched_n1 * matched_n2).sum(dim=1)

        # Clamp negatives to zero to avoid negative weights
        cos_sim = torch.clamp(cos_sim, min=0.0)

        # Softer weighting preserves geometric intersection count while boosting aligned normals.
        # normal_weight=1.0 keeps the legacy 1+cos behavior.
        valid_intersection = (1.0 + self.normal_weight * cos_sim).sum().item()

        # 5. Union Calculation
        # Union = |V1| + |V2| - |Geom_Intersection|
        geom_intersection_count = match_mask.sum().item()
        union_count = k1.size(0) + k2.size(0) - geom_intersection_count

        iou = valid_intersection / union_count if union_count > 0 else 0.0

        return iou, valid_intersection, union_count

    def _voxelize(self, points, normals):
        """
        Voxelize points and compute average normal per voxel.

        Args:
            points: (N, 3) Point coordinates
            normals: (N, 3) Normal vectors

        Returns:
            unique_keys: (K,) Unique voxel hash keys
            avg_normals: (K, 3) Averaged and normalized normals per voxel
        """
        # 좌표 정수화
        grid_coords = torch.floor(points / self.voxel_size).long()

        # 해시 키 생성 (3D -> 1D)
        # 2cm~5cm 복셀이라면 좌표 범위가 꽤 크므로 int64 범위 활용
        # 소수 Multiplier를 사용하여 충돌 최소화
        keys = (grid_coords[:, 0] * 73856093 +
                grid_coords[:, 1] * 19349663 +
                grid_coords[:, 2] * 83492791)

        # 유니크 복셀 추출 및 Inverse Index 획득
        unique_keys, inverse_indices = torch.unique(keys, return_inverse=True)

        # Voxel별 Normal 합산 후 정규화
        n_voxels = unique_keys.size(0)
        sum_normals = torch.zeros((n_voxels, 3), device=points.device, dtype=normals.dtype)

        # index_add_로 합산 후 정규화하여 대표 방향 생성
        sum_normals.index_add_(0, inverse_indices, normals)
        avg_normals = F.normalize(sum_normals, p=2, dim=1)

        return unique_keys, avg_normals


# =============================================================================
# SoftVoxelNormalIoU - Union-based voxel IoU with soft weighting
# =============================================================================

class SoftVoxelNormalIoU:
    """
    Soft Normal-aware IoU using 3D voxel coordinates and union-based computation.

    Unlike NormalAwareIoU which uses hash-based intersection search, this method:
    1. Uses torch.unique on 3D coordinates directly (no hash collision risk)
    2. Computes true voxel union by concatenating both voxel sets
    3. Supports soft IoU where normal similarity acts as continuous weight
    4. Supports anisotropic voxel sizes (different for Z vs XY)

    Algorithm:
    1. Voxelize both point clouds (3D coords -> unique voxels + avg normals)
    2. Concatenate voxel sets and find unique union voxels
    3. Identify shared voxels (count == 2)
    4. Filter by normal cosine similarity (parallel condition)
    5. Hard IoU: valid_intersection / union_count
       Soft IoU: weighted_intersection / weighted_union
    """

    def __init__(self, voxel_size: float = 0.05, cos_thresh: Optional[float] = None,
                 soft: bool = False, device: str | torch.device = 'cuda',
                 voxel_size_z: Optional[float] = None, voxel_size_xy: Optional[float] = None):
        """
        Args:
            voxel_size: Voxel grid size in meters (default: 10cm). Used as fallback if voxel_size_z/xy not set.
            cos_thresh: Cosine similarity threshold for parallel normals.
                        None = use soft 1+cos scoring without threshold.
            soft: If True, use soft IoU with continuous normal weights
            device: 'cuda' or 'cpu'
            voxel_size_z: Voxel size for Z-axis (vertical). If None, uses voxel_size.
            voxel_size_xy: Voxel size for X and Y axes. If None, uses voxel_size.
        """
        self.voxel_size = voxel_size
        # Anisotropic voxel support: Z can be finer than XY
        self.voxel_size_z = voxel_size_z if voxel_size_z is not None else voxel_size
        self.voxel_size_xy = voxel_size_xy if voxel_size_xy is not None else voxel_size
        self.cos_thresh = cos_thresh
        self.soft = soft
        self.device = torch.device(device)

    def compute(self, points1: torch.Tensor, normals1: torch.Tensor,
                points2: torch.Tensor, normals2: torch.Tensor) -> Tuple[float, int, int]:
        """
        Compute Normal-aware IoU between two point clouds.

        Args:
            points1: (N, 3) Point coordinates of cloud 1
            normals1: (N, 3) Normal vectors of cloud 1 (unit vectors)
            points2: (M, 3) Point coordinates of cloud 2
            normals2: (M, 3) Normal vectors of cloud 2 (unit vectors)

        Returns:
            iou: IoU value (0~1)
            valid_intersection: Number of valid intersection voxels
            union_count: Number of union voxels
        """
        device = self.device

        # Ensure tensors on correct device
        points1 = points1.to(device)
        normals1 = normals1.to(device)
        points2 = points2.to(device)
        normals2 = normals2.to(device)

        # 1. Voxelize both point clouds
        keys1, n1v = self._voxelize_with_normals(points1, normals1)
        keys2, n2v = self._voxelize_with_normals(points2, normals2)

        M1 = keys1.shape[0]
        M2 = keys2.shape[0]

        # 2. Concatenate and find unique union voxels
        all_keys = torch.cat([keys1, keys2], dim=0)  # (M1+M2, 3)
        uniq, inv, counts = torch.unique(all_keys, return_inverse=True,
                                         return_counts=True, dim=0)
        # uniq: (K, 3) - all unique voxel coordinates
        # inv: (M1+M2,) - mapping to uniq index
        # counts: (K,) - 1 if only one set, 2 if both sets

        K = uniq.shape[0]

        # 3. Map each set's voxels to union indices
        inv1 = inv[:M1]
        inv2 = inv[M1:]

        # Global to local index mapping
        g2l1 = torch.full((K,), -1, device=device, dtype=torch.long)
        g2l2 = torch.full((K,), -1, device=device, dtype=torch.long)

        g2l1[inv1] = torch.arange(M1, device=device)
        g2l2[inv2] = torch.arange(M2, device=device)

        # 4. Find shared voxels (exist in both sets)
        shared_mask = (counts == 2)
        shared_g_idx = shared_mask.nonzero(as_tuple=False).squeeze(1)

        if shared_g_idx.numel() == 0:
            # No intersection at all
            return 0.0, 0, K

        # Get local indices for shared voxels
        idx1 = g2l1[shared_g_idx]
        idx2 = g2l2[shared_g_idx]

        # 5. Normal cosine similarity for shared voxels
        cos_sim = F.cosine_similarity(n1v[idx1], n2v[idx2], dim=1)
        cond = torch.ones_like(cos_sim, dtype=torch.bool)

        if self.soft:
            # Soft IoU: normal similarity as continuous weight
            # Map cos from [-1, 1] to [0, 1]
            w = (cos_sim.clamp(-1, 1) + 1.0) / 2.0
            w = w * cond.float()  # cond is all True

            I = w.sum()

            # Weighted union calculation
            o1 = (g2l1 >= 0).float()
            o2 = (g2l2 >= 0).float()

            w_full = torch.zeros(K, device=device)
            w_full[shared_g_idx] = w

            U = (o1 + o2 - o1 * o2 * w_full).sum()

            valid_intersection = int(I.item())
            union_count = int(U.item())
        else:
            # Hard IoU: count valid intersection voxels
            valid_intersection = int(cond.sum().item())
            union_count = K

        if union_count == 0:
            return 0.0, 0, 0

        iou = valid_intersection / union_count
        return iou, valid_intersection, union_count

    def _voxelize_with_normals(self, points: torch.Tensor,
                                normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Voxelize points and compute average normal per voxel.
        Supports anisotropic voxel sizes (different for Z vs XY).

        Args:
            points: (N, 3) Point coordinates
            normals: (N, 3) Normal vectors (unit vectors)

        Returns:
            keys: (M, 3) Unique voxel indices (integer coordinates)
            avg_normals: (M, 3) Per-voxel averaged normals (normalized)
        """
        device = points.device

        # 1. Quantize to integer voxel coordinates (anisotropic: XY vs Z)
        # voxel_sizes: [voxel_xy, voxel_xy, voxel_z]
        voxel_sizes = torch.tensor(
            [self.voxel_size_xy, self.voxel_size_xy, self.voxel_size_z],
            device=device, dtype=points.dtype
        )
        coords = torch.floor(points / voxel_sizes).long()

        # 2. Find unique voxels and inverse mapping
        keys, inv = torch.unique(coords, return_inverse=True, dim=0)
        M = keys.shape[0]

        # 3. Accumulate normals per voxel (scatter_add)
        normals_sum = torch.zeros(M, 3, device=device, dtype=points.dtype)
        normals_sum.scatter_add_(0, inv.view(-1, 1).expand(-1, 3), normals)

        # 4. Count points per voxel
        counts = torch.bincount(inv, minlength=M).float().unsqueeze(1)
        normals_avg = normals_sum / torch.clamp(counts, min=1.0)

        # 5. Re-normalize
        normals_avg = F.normalize(normals_avg, dim=1)

        # 6. Orient normals outward from origin (approx) using voxel center vector
        centers = (keys.float() + 0.5) * voxel_sizes  # (M,3) - use anisotropic sizes
        dot = (centers * normals_avg).sum(dim=1, keepdim=True)
        flip_mask = dot < 0
        normals_avg = torch.where(flip_mask, -normals_avg, normals_avg)

        return keys, normals_avg


def compute_soft_normal_iou(
    source_points: np.ndarray,
    target_mesh: o3d.geometry.TriangleMesh,
    transform: np.ndarray,
    *,
    voxel_size: float = 0.05,
    cos_thresh: Optional[float] = None,
    soft: bool = False,
    normal_radius: Optional[float] = None,
    max_points: int = 100_000,
    device: str | torch.device = 'cuda',
    verbose: bool = True,
    downsample_voxel: Optional[float] = None,
) -> Tuple[float, dict]:
    """
    Compute Soft Normal-aware IoU using union-based voxel computation.

    This is an alternative to compute_normal_aware_iou that:
    - Uses 3D voxel coordinates directly (no hash collisions)
    - Computes true voxel union
    - Optionally supports soft IoU with continuous normal weights

    Args:
        source_points: (N, 3) Source point cloud
        target_mesh: Target mesh (Open3D TriangleMesh)
        transform: (4, 4) Transformation matrix to apply to source
        voxel_size: Voxel grid size in meters (default: 0.10m = 10cm)
        cos_thresh: Cosine similarity threshold. None = soft 1+cos scoring.
        soft: If True, use soft IoU with continuous weights
        normal_radius: Radius for normal estimation on source points (default: voxel_size*2 if None)
        max_points: Maximum points to sample
        device: 'cuda' or 'cpu'
        verbose: Print progress info
        downsample_voxel: Optional voxel size for pre-downsampling

    Returns:
        iou: IoU value (0~1)
        info: Dictionary with detailed metrics
    """
    timings = {}
    t_start = time.time()

    try:
        # 0. Input validation
        n_points = len(source_points)
        if n_points == 0:
            if verbose:
                print(f"\n     IoU: Empty source points")
            return 0.0, {"error": "empty_source"}

        # 1. Sample source points if too many
        t0 = time.time()
        if n_points > max_points:
            idx = np.random.choice(n_points, max_points, replace=False)
            source_points = source_points[idx]
        timings["sampling"] = time.time() - t0

        # 2. Apply transform to source points
        t1 = time.time()
        R, t = transform[:3, :3], transform[:3, 3]
        src_transformed = (R @ source_points.T).T + t
        timings["transform"] = time.time() - t1

        # 3. Build source point cloud and estimate normals
        t2 = time.time()
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_transformed)

        if downsample_voxel and downsample_voxel > 0:
            src_pcd = src_pcd.voxel_down_sample(voxel_size=float(downsample_voxel))

        if len(src_pcd.points) < 10:
            if verbose:
                print(f"\n     IoU: Too few source points ({len(src_pcd.points)})")
            return 0.0, {"error": "too_few_points"}

        radius = float(normal_radius) if normal_radius is not None else float(voxel_size) * 2.0
        src_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=30
            )
        )
        src_points_np = np.asarray(src_pcd.points)
        src_normals_np = np.asarray(src_pcd.normals)
        timings["source_normals"] = time.time() - t2

        # 4. Sample target mesh and get normals
        t3 = time.time()
        if not target_mesh.has_vertex_normals():
            target_mesh.compute_vertex_normals()

        n_target_samples = min(max_points * 2, 200_000)
        target_pcd = target_mesh.sample_points_uniformly(number_of_points=n_target_samples)
        tgt_points_np = np.asarray(target_pcd.points)
        tgt_normals_np = np.asarray(target_pcd.normals)
        timings["target_sampling"] = time.time() - t3

        # 5. Convert to torch tensors
        t4 = time.time()
        dev = torch.device(device)

        src_pts_t = torch.as_tensor(src_points_np, dtype=torch.float32, device=dev)
        src_norms_t = torch.as_tensor(src_normals_np, dtype=torch.float32, device=dev)
        tgt_pts_t = torch.as_tensor(tgt_points_np, dtype=torch.float32, device=dev)
        tgt_norms_t = torch.as_tensor(tgt_normals_np, dtype=torch.float32, device=dev)

        src_norms_t = F.normalize(src_norms_t, p=2, dim=1)
        tgt_norms_t = F.normalize(tgt_norms_t, p=2, dim=1)
        timings["to_tensor"] = time.time() - t4

        # 6. Compute Soft Normal-aware IoU
        t5 = time.time()
        evaluator = SoftVoxelNormalIoU(
            voxel_size=voxel_size,
            cos_thresh=cos_thresh,
            soft=soft,
            device=dev
        )
        iou, valid_intersection, union_count = evaluator.compute(
            src_pts_t, src_norms_t,
            tgt_pts_t, tgt_norms_t
        )
        timings["iou_compute"] = time.time() - t5

        total_time = time.time() - t_start

        # Get mesh surface area for debugging
        try:
            mesh_surface_area = target_mesh.get_surface_area()
        except Exception:
            mesh_surface_area = None

        info = {
            "iou": iou,
            "valid_intersection": valid_intersection,
            "union_count": union_count,
            "source_points": len(src_points_np),
            "target_points": len(tgt_points_np),
            "target_samples_used": n_target_samples,
            "mesh_surface_area": mesh_surface_area,
            "voxel_size": voxel_size,
            "cos_thresh": cos_thresh,
            "soft": soft,
            "timings": timings,
            "total_time": total_time,
        }

        if verbose:
            mode_str = "Soft" if soft else "Hard"
            area_str = f"{mesh_surface_area:.1f}m²" if mesh_surface_area else "N/A"
            print(f"\n     {mode_str} Voxel IoU: "
                  f"src={len(src_points_np)} tgt={len(tgt_points_np)} "
                  f"(mesh_area={area_str}, samples={n_target_samples}) | "
                  f"valid_inter={valid_intersection} union={union_count} | "
                  f"IoU={iou:.3f} ({total_time:.2f}s)")

        return iou, info

    except Exception as e:
        if verbose:
            print(f"\n     IoU error: {e}")
        return 0.0, {"error": str(e)}


# =============================================================================
# API Functions for registration.py
# =============================================================================

def compute_normal_aware_iou(
    source_points: np.ndarray,
    target_mesh: o3d.geometry.TriangleMesh,
    transform: np.ndarray,
    *,
    voxel_size: float = 0.05,
    cos_thresh: Optional[float] = None,
    normal_radius: Optional[float] = None,
    max_points: int = 100_000,
    device: str | torch.device = 'cuda',
    verbose: bool = True,
    downsample_voxel: Optional[float] = None,
    use_normal_score: bool = True,
    normal_weight: float = 1.0,
) -> Tuple[float, dict]:
    """
    Compute Normal-aware IoU between transformed source points and target mesh.

    Args:
        source_points: (N, 3) Source point cloud
        target_mesh: Target mesh (Open3D TriangleMesh)
        transform: (4, 4) Transformation matrix to apply to source
        voxel_size: Voxel grid size in meters (default: 0.10m = 10cm)
        cos_thresh: Cosine similarity threshold. None = soft 1+cos scoring.
        normal_radius: Radius for normal estimation on source points (default: voxel_size*2 if None)
        max_points: Maximum points to sample
        device: 'cuda' or 'cpu'
        verbose: Print progress info
        downsample_voxel: Optional voxel size for pre-downsampling

    Returns:
        iou: IoU value (0~1)
        info: Dictionary with detailed metrics
    """
    def _orient_normals_to_centroid(points_np: np.ndarray, normals_np: np.ndarray) -> np.ndarray:
        if points_np is None or normals_np is None or len(points_np) == 0 or len(normals_np) == 0:
            return normals_np
        center = points_np.mean(axis=0)
        dirs = points_np - center
        dots = np.einsum("ij,ij->i", normals_np, dirs)
        flip_mask = dots < 0
        normals_np[flip_mask] *= -1
        return normals_np
    timings = {}
    t_start = time.time()

    try:
        # 0. Input validation
        n_points = len(source_points)
        if n_points == 0:
            if verbose:
                print(f"\n     IoU: Empty source points")
            return 0.0, {"error": "empty_source"}

        # 1. Apply transform to source points
        t1 = time.time()
        R, t = transform[:3, :3], transform[:3, 3]
        src_transformed = (R @ source_points.T).T + t
        timings["transform"] = time.time() - t1

        # 2. Voxel downsample (preferred over random sampling) before any cap
        if downsample_voxel and downsample_voxel > 0:
            t2 = time.time()
            src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_transformed))
            src_pcd = src_pcd.voxel_down_sample(voxel_size=float(downsample_voxel))
            src_transformed = np.asarray(src_pcd.points)
            timings["source_downsample"] = time.time() - t2

        # 3. Cap by max_points only if still too dense (post-downsample)
        t0 = time.time()
        if len(src_transformed) > max_points:
            idx = np.random.choice(len(src_transformed), max_points, replace=False)
            src_transformed = src_transformed[idx]
        timings["sampling"] = time.time() - t0

        dev = torch.device(device)

        # Fast path: skip normal filtering and just use geometric voxel union
        if not use_normal_score:
            src_points_np = src_transformed
            if len(src_points_np) < 10:
                if verbose:
                    print(f"\n     IoU: Too few source points ({len(src_points_np)})")
                return 0.0, {"error": "too_few_points"}

            t3 = time.time()
            n_target_samples = min(max_points * 2, 200_000)
            target_pcd = target_mesh.sample_points_uniformly(number_of_points=n_target_samples)
            tgt_points_np = np.asarray(target_pcd.points)
            timings["target_sampling"] = time.time() - t3

            # Torch tensors
            t4 = time.time()
            src_pts_t = torch.as_tensor(src_points_np, dtype=torch.float32, device=dev)
            tgt_pts_t = torch.as_tensor(tgt_points_np, dtype=torch.float32, device=dev)
            timings["to_tensor"] = time.time() - t4

            # Voxelize and compute geometric IoU (no normal filter)
            t5 = time.time()
            def _voxel_keys(pts_t: torch.Tensor) -> torch.Tensor:
                grid = torch.floor(pts_t / voxel_size).long()
                keys = (grid[:, 0] * 73856093 +
                        grid[:, 1] * 19349663 +
                        grid[:, 2] * 83492791)
                return torch.unique(keys)

            k1 = torch.sort(_voxel_keys(src_pts_t)).values
            k2 = torch.sort(_voxel_keys(tgt_pts_t)).values
            idx_in_k2 = torch.searchsorted(k2, k1)
            idx_clamped = idx_in_k2.clamp(max=k2.size(0) - 1)
            match_mask = (idx_in_k2 < k2.size(0)) & (k2[idx_clamped] == k1)
            geom_intersection = match_mask.sum().item()
            union_count = k1.numel() + k2.numel() - geom_intersection
            iou = geom_intersection / union_count if union_count > 0 else 0.0
            timings["iou_compute"] = time.time() - t5

            total_time = time.time() - t_start
            # Get mesh surface area for debugging
            try:
                mesh_surface_area = target_mesh.get_surface_area()
            except Exception:
                mesh_surface_area = None
            info = {
                "iou": iou,
                "valid_intersection": geom_intersection,
                "union_count": union_count,
                "source_points": len(src_points_np),
                "target_points": len(tgt_points_np),
                "target_samples_used": n_target_samples,
                "mesh_surface_area": mesh_surface_area,
                "source_voxels": int(k1.numel()),
                "target_voxels": int(k2.numel()),
                "voxel_size": voxel_size,
                "cos_thresh": cos_thresh,
                "timings": timings,
                "total_time": total_time,
                "normal_filtered": False,
                "normal_weight": normal_weight,
            }
            if verbose:
                print(f"\n     Geometric IoU: src={len(src_points_np)} tgt={len(tgt_points_np)} "
                      f"(mesh_area={mesh_surface_area:.1f}m², samples={n_target_samples}) | "
                      f"src_vox={k1.numel()} tgt_vox={k2.numel()} | "
                      f"inter={geom_intersection} union={union_count} | IoU={iou:.3f} ({total_time:.2f}s)")
            return iou, info

        # 3. Build source point cloud and estimate normals
        t2 = time.time()
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_transformed)
        if len(src_pcd.points) < 10:
            if verbose:
                print(f"\n     IoU: Too few source points ({len(src_pcd.points)})")
            return 0.0, {"error": "too_few_points"}

        # Estimate normals for source
        radius = float(normal_radius) if normal_radius is not None else float(voxel_size) * 2.0
        src_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=30
            )
        )
        src_points_np = np.asarray(src_pcd.points)
        src_normals_np = np.asarray(src_pcd.normals)
        src_normals_np = _orient_normals_to_centroid(src_points_np, src_normals_np)
        timings["source_normals"] = time.time() - t2

        # 4. Sample target mesh and get normals
        t3 = time.time()
        if not target_mesh.has_vertex_normals():
            target_mesh.compute_vertex_normals()

        # Sample points from target for coverage (denser to stabilize union)
        n_target_samples = min(max_points * 4, 400_000)
        target_pcd = target_mesh.sample_points_uniformly(number_of_points=n_target_samples)
        tgt_points_np = np.asarray(target_pcd.points)
        tgt_normals_np = np.asarray(target_pcd.normals)
        tgt_normals_np = _orient_normals_to_centroid(tgt_points_np, tgt_normals_np)
        timings["target_sampling"] = time.time() - t3

        # 5. Convert to torch tensors
        t4 = time.time()
        src_pts_t = torch.as_tensor(src_points_np, dtype=torch.float32, device=dev)
        src_norms_t = torch.as_tensor(src_normals_np, dtype=torch.float32, device=dev)
        tgt_pts_t = torch.as_tensor(tgt_points_np, dtype=torch.float32, device=dev)
        tgt_norms_t = torch.as_tensor(tgt_normals_np, dtype=torch.float32, device=dev)

        # Normalize normals (safety)
        src_norms_t = F.normalize(src_norms_t, p=2, dim=1)
        tgt_norms_t = F.normalize(tgt_norms_t, p=2, dim=1)
        timings["to_tensor"] = time.time() - t4

        # 6. Compute Normal-aware IoU
        t5 = time.time()
        evaluator = NormalAwareIoU(
            voxel_size=voxel_size,
            cos_thresh=cos_thresh,
            device=dev,
            normal_weight=normal_weight,
        )
        iou, valid_intersection, union_count = evaluator.compute(
            src_pts_t, src_norms_t,
            tgt_pts_t, tgt_norms_t
        )
        timings["iou_compute"] = time.time() - t5

        total_time = time.time() - t_start

        # Get mesh surface area for debugging
        try:
            mesh_surface_area = target_mesh.get_surface_area()
        except Exception:
            mesh_surface_area = None

        # Build info dict
        info = {
            "iou": iou,
            "valid_intersection": valid_intersection,
            "union_count": union_count,
            "source_points": len(src_points_np),
            "target_points": len(tgt_points_np),
            "target_samples_used": n_target_samples,
            "mesh_surface_area": mesh_surface_area,
            "voxel_size": voxel_size,
            "cos_thresh": cos_thresh,
            "timings": timings,
            "total_time": total_time,
            "normal_filtered": True,
            "normal_weight": normal_weight,
        }

        if verbose:
            area_str = f"{mesh_surface_area:.1f}m²" if mesh_surface_area else "N/A"
            print(f"\n     Normal-aware IoU: "
                  f"src={len(src_points_np)} tgt={len(tgt_points_np)} "
                  f"(mesh_area={area_str}, samples={n_target_samples}) | "
                  f"valid_inter={valid_intersection} union={union_count} | "
                  f"IoU={iou:.3f} ({total_time:.2f}s)")

        return iou, info

    except Exception as e:
        if verbose:
            print(f"\n     IoU error: {e}")
        return 0.0, {"error": str(e)}


# =============================================================================
# Legacy API Compatibility
# =============================================================================

def compute_normal_aligned_score(
    source_points: np.ndarray,
    target_mesh: o3d.geometry.TriangleMesh,
    transform: np.ndarray,
    *,
    normal_threshold: float = 0.9,
    distance_threshold: float = 0.05,
    normal_radius: Optional[float] = None,
    max_points: int = 100_000,
    device: str | torch.device = 'cpu',
    verbose: bool = True,
    downsample_voxel: float = 0.025,
) -> float:
    """
    Legacy API wrapper - redirects to NormalAwareIoU.
    distance_threshold is mapped to voxel_size.
    """
    iou, _ = compute_normal_aware_iou(
        source_points,
        target_mesh,
        transform,
        voxel_size=distance_threshold,
        cos_thresh=normal_threshold,
        normal_radius=normal_radius,
        max_points=max_points,
        device=device,
        verbose=verbose,
        downsample_voxel=downsample_voxel,
    )
    return iou


def compute_normal_aligned_score_gpu(
    source_points: np.ndarray,
    target_mesh: o3d.geometry.TriangleMesh,
    transform: np.ndarray,
    *,
    normal_threshold: float = 0.9,
    distance_threshold: float = 0.05,
    normal_radius: Optional[float] = None,
    max_points: int = 100_000,
    device: str | torch.device = 'cuda',
    verbose: bool = True,
    downsample_voxel: float = 0.025,
    chunk_n: int = 2048,  # Ignored
    chunk_m: int = 2048,  # Ignored
) -> float:
    """
    Legacy API wrapper - redirects to NormalAwareIoU (GPU version).
    """
    iou, _ = compute_normal_aware_iou(
        source_points,
        target_mesh,
        transform,
        voxel_size=distance_threshold,
        cos_thresh=normal_threshold,
        normal_radius=normal_radius,
        max_points=max_points,
        device=device,
        verbose=verbose,
        downsample_voxel=downsample_voxel,
    )
    return iou
