"""
PCA-based batch SE(3) transformation for initial alignment.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Optional, List
import itertools


class PCABatchSE3Transform(nn.Module):
    """
    PCA-based batch SE(3) transformation module.
    
    Generates multiple rotation candidates by aligning principal components
    of source and target point clouds with different sign combinations.
    
    Args:
        source_points: Source point cloud (Nx3)
        target_points: Target point cloud (Mx3) - optional for manual mode
        n_pca_candidates: Number of PCA rotation candidates (default: 8)
        device: PyTorch device
        skip_pca: Skip PCA and manually set parameters (default: False)
    """
    
    def __init__(self,
                 source_points: np.ndarray,
                 target_points: Optional[np.ndarray] = None,
                 n_pca_candidates: int = 8,
                 device: Optional[torch.device] = None,
                 skip_pca: bool = False,
                 source_center: Optional[np.ndarray] = None,
                 target_center: Optional[np.ndarray] = None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Resolve / fall back to data means so the centroid-aligning translation
        # initialisation below has something to subtract from. Without these, the
        # PCA rotations get applied to non-centered points and source ends up far
        # outside the target SDF (inlier=0 even at step 0).
        resolved_source_center = (
            np.asarray(source_center, dtype=np.float64).reshape(3)
            if source_center is not None else
            np.asarray(source_points, dtype=np.float64).mean(axis=0)
        ) if source_points is not None else np.zeros(3)
        resolved_target_center = (
            np.asarray(target_center, dtype=np.float64).reshape(3)
            if target_center is not None else (
                np.asarray(target_points, dtype=np.float64).mean(axis=0)
                if target_points is not None else np.zeros(3)
            )
        )

        if not skip_pca and target_points is not None:
            # Generate PCA rotation candidates
            self.rotation_candidates = self._generate_pca_rotations(
                source_points, target_points, n_pca_candidates,
                source_center, target_center
            )
        else:
            # Manual mode - will be set externally
            self.rotation_candidates = np.zeros((n_pca_candidates, 3))

        self.batch_size = len(self.rotation_candidates)

        # Initialize parameters
        self.rotation_params = nn.Parameter(
            torch.tensor(self.rotation_candidates, device=self.device, dtype=torch.float32)
        )
        # Centroid-aligning translation init: forward applies R @ p + t, so we
        # set t = target_center - R @ source_center per candidate.
        rot_mats = np.stack(
            [self._axis_angle_to_rotation_matrix(rv) for rv in self.rotation_candidates],
            axis=0,
        )  # (B, 3, 3)
        translation_init = (
            resolved_target_center.reshape(1, 3)
            - np.einsum("bij,j->bi", rot_mats, resolved_source_center)
        )  # (B, 3)
        self.translation = nn.Parameter(
            torch.tensor(translation_init, device=self.device, dtype=torch.float32)
        )
        
        print(f"PCABatchSE3Transform initialized with {self.batch_size} candidates on {self.device}")
    
    def _generate_pca_rotations(self, source_points: np.ndarray, target_points: np.ndarray,
                                n_candidates: int,
                                source_center: Optional[np.ndarray] = None,
                                target_center: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate rotation candidates using PCA alignment.

        Args:
            source_points: Source point cloud
            target_points: Target point cloud
            n_candidates: Number of rotation candidates to generate
            source_center: Optional custom center for source (if None, use mean)
            target_center: Optional custom center for target (if None, use mean)

        Returns:
            Array of axis-angle representations (n_candidates, 3)
        """
        # Center point clouds using provided centers or compute from data
        source_center = source_center if source_center is not None else source_points.mean(axis=0)
        target_center = target_center if target_center is not None else target_points.mean(axis=0)

        source_centered = source_points - source_center
        target_centered = target_points - target_center
        
        # Compute PCA
        pca_source = PCA(n_components=3)
        pca_target = PCA(n_components=3)
        
        pca_source.fit(source_centered)
        pca_target.fit(target_centered)
        
        # Get principal axes
        axes_source = pca_source.components_  # (3, 3)
        axes_target = pca_target.components_  # (3, 3)
        
        rotation_candidates = []
        
        # Basic 8 combinations: sign flips for each axis
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    # Apply sign flips
                    axes_s = axes_source.copy()
                    axes_s[0] *= sx
                    axes_s[1] *= sy
                    axes_s[2] *= sz
                    
                    # Compute rotation: R @ axes_s.T = axes_target.T
                    # Therefore: R = axes_target.T @ axes_s
                    R = axes_s.T @ axes_target
                    
                    # Convert to axis-angle
                    rotation_vec = self._rotation_matrix_to_axis_angle(R)
                    rotation_candidates.append(rotation_vec)
        
        # Additional candidates with axis permutations
        if n_candidates > 8:
            perms = list(itertools.permutations([0, 1, 2]))
            sign_patterns = [
                (1, 1, -1), (1, -1, 1), (-1, 1, 1), 
                (1, -1, -1), (-1, 1, -1), (-1, -1, 1),
                (-1, -1, -1)
            ]
            
            for perm in perms:
                for sign_pattern in sign_patterns:
                    if len(rotation_candidates) >= n_candidates:
                        break
                        
                    axes_s_perm = axes_source[list(perm)].copy()
                    axes_s_perm[0] *= sign_pattern[0]
                    axes_s_perm[1] *= sign_pattern[1]
                    axes_s_perm[2] *= sign_pattern[2]
                    
                    R = axes_s_perm.T @ axes_target
                    rotation_vec = self._rotation_matrix_to_axis_angle(R)
                    rotation_candidates.append(rotation_vec)
        
        # Add small random perturbations if needed
        if len(rotation_candidates) < n_candidates:
            base_candidates = rotation_candidates[:8].copy()
            for base in base_candidates:
                if len(rotation_candidates) >= n_candidates:
                    break
                # Add ±10 degree perturbation
                perturb = np.random.randn(3) * 0.174  # 0.174 rad = 10 degrees
                rotation_candidates.append(base + perturb)
        
        return np.array(rotation_candidates[:n_candidates])
    
    @staticmethod
    def _axis_angle_to_rotation_matrix(rotation_vec: np.ndarray) -> np.ndarray:
        """Convert axis-angle (Rodrigues) to a 3x3 rotation matrix."""
        rv = np.asarray(rotation_vec, dtype=np.float64).reshape(3)
        theta = float(np.linalg.norm(rv))
        if theta < 1e-8:
            return np.eye(3, dtype=np.float64)
        axis = rv / theta
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ], dtype=np.float64)
        return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle representation"""
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        
        if angle < 1e-6:
            # Near identity
            return np.zeros(3)
        elif angle > np.pi - 1e-6:
            # Near 180 degrees - find axis from diagonal
            idx = np.argmax(np.diag(R))
            axis = np.zeros(3)
            axis[idx] = 1
            return axis * angle
        else:
            # General case - extract axis from skew-symmetric part
            K = (R - R.T) / (2 * np.sin(angle))
            axis = np.array([K[2, 1], K[0, 2], K[1, 0]])
            return axis * angle
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply batch transformation to points.
        
        Args:
            points: (M, 3) tensor of points
            
        Returns:
            (N, M, 3) batch of transformed points
        """
        batch_size = self.rotation_params.shape[0]
        
        # Compute rotation matrices using Rodrigues formula
        angle = torch.norm(self.rotation_params, dim=1, keepdim=True)  # (N, 1)
        eps = 1e-8
        axis = self.rotation_params / (angle + eps)  # (N, 3)
        
        # Build skew-symmetric matrices
        K = torch.zeros((batch_size, 3, 3), device=self.device, dtype=torch.float32)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        I = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1)
        angle_expanded = angle.unsqueeze(-1)  # (N, 1, 1)
        R = I + torch.sin(angle_expanded) * K + (1 - torch.cos(angle_expanded)) * torch.matmul(K, K)
        
        # Apply transformation
        if points.device != self.device:
            points = points.to(self.device)
        points_expanded = points.unsqueeze(0).expand(batch_size, -1, -1)
        transformed = torch.matmul(points_expanded, R.transpose(1, 2)) + self.translation.unsqueeze(1)
        
        return transformed
    
    def get_best_transform(self, losses: torch.Tensor) -> Tuple[np.ndarray, int]:
        """
        Get the transformation matrix for the candidate with lowest loss.
        
        Args:
            losses: (N,) tensor of losses for each candidate
            
        Returns:
            4x4 transformation matrix and best candidate index
        """
        best_idx = torch.argmin(losses)
        
        # Extract best parameters
        best_rotation = self.rotation_params[best_idx].detach()
        best_translation = self.translation[best_idx].detach()
        
        # Convert to rotation matrix
        angle = torch.norm(best_rotation)
        if angle > 0:
            axis = best_rotation / angle
            K = torch.zeros((3, 3), device=self.device)
            K[0, 1] = -axis[2]
            K[0, 2] = axis[1]
            K[1, 0] = axis[2]
            K[1, 2] = -axis[0]
            K[2, 0] = -axis[1]
            K[2, 1] = axis[0]
            R = torch.eye(3, device=self.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
        else:
            R = torch.eye(3, device=self.device)
        
        # Build 4x4 transformation matrix
        T = torch.eye(4, device=self.device)
        T[:3, :3] = R
        T[:3, 3] = best_translation
        
        return T.cpu().numpy(), best_idx.item()
    
    def get_all_transforms(self) -> List[np.ndarray]:
        """Get all transformation matrices"""
        transforms = []
        for i in range(self.batch_size):
            rotation = self.rotation_params[i].detach()
            translation = self.translation[i].detach()
            
            # Convert to matrix
            angle = torch.norm(rotation)
            if angle > 0:
                axis = rotation / angle
                K = torch.zeros((3, 3), device=self.device)
                K[0, 1] = -axis[2]
                K[0, 2] = axis[1]
                K[1, 0] = axis[2]
                K[1, 2] = -axis[0]
                K[2, 0] = -axis[1]
                K[2, 1] = axis[0]
                R = torch.eye(3, device=self.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
            else:
                R = torch.eye(3, device=self.device)
            
            T = torch.eye(4, device=self.device)
            T[:3, :3] = R
            T[:3, 3] = translation
            transforms.append(T.cpu().numpy())
            
        return transforms