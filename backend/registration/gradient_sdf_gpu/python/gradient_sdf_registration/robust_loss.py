"""
Robust loss functions for SDF-based registration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class RobustSDFLoss(nn.Module):
    """
    Robust SDF loss with Cauchy M-estimator for outlier rejection.
    
    The Cauchy M-estimator is defined as:
    ρ(e) = (c²/2) * log(1 + (e/c)²)
    
    where c controls the sensitivity to outliers.
    
    Args:
        cauchy_c: Scale parameter for Cauchy distribution (default: 1.0)
        use_gradient_weighting: Weight loss by gradient magnitude (default: True)
        inlier_threshold: Threshold for counting inliers (default: 0.5)
        normal_loss_weight: Weight for source-normal vs SDF-gradient alignment
    """
    
    def __init__(self, 
                 cauchy_c: float = 1.0, 
                 use_gradient_weighting: bool = True,
                 inlier_threshold: float = 0.5,
                 normal_loss_weight: float = 0.0):
        super().__init__()
        self.cauchy_c = cauchy_c
        self.use_gradient_weighting = use_gradient_weighting
        self.inlier_threshold = inlier_threshold
        self.normal_loss_weight = max(0.0, float(normal_loss_weight or 0.0))
        
    def cauchy_loss(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Compute Cauchy M-estimator loss.
        
        Args:
            residuals: Squared residuals (e²)
            
        Returns:
            Robust loss values
        """
        return (self.cauchy_c**2 / 2) * torch.log(1 + (residuals / self.cauchy_c**2))
    
    def compute_weights(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Compute influence weights for visualization/analysis.
        
        The weight function is: w(e) = ψ(e)/e = 1/(1 + (e/c)²)
        
        Args:
            residuals: Squared residuals
            
        Returns:
            Weights in [0, 1]
        """
        return 1.0 / (1.0 + (residuals / self.cauchy_c**2))
    
    def forward(
        self,
        sdf_values: torch.Tensor,
        gradients: Optional[torch.Tensor] = None,
        source_normals: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        variances: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute robust loss for batch SDF values.

        Args:
            sdf_values: (N, M) batch SDF values
            gradients: (N, M, 3) batch gradient vectors (optional)
            source_normals: (N, M, 3) transformed source normals (optional)
            valid_mask: (N, M) bool mask of points to include (optional).
                Must stay FIXED across optimization steps: a mask recomputed
                per step lets the masked mean reward poses for pushing points
                out of the SDF grid.
            variances: (N, M) CloudCropper — normalized field variance u
                (var / median_var), ALREADY DETACHED by the caller. Enables the
                heteroscedastic Cauchy: confident voxels (u < 1) count more,
                uncertain ones (occlusion, sparsity) are downweighted.

        Returns:
            Dictionary containing:
                - loss: (N,) per-batch losses
                - weights: (N, M) influence weights
                - inlier_ratio: Scalar inlier ratio
                - mean_loss: Scalar mean loss
        """
        # Compute squared residuals
        residuals = sdf_values**2

        if variances is not None:
            # CloudCropper: heteroscedastic Cauchy (GPIS variance channel).
            # rho = (c^2/2) log1p(r^2 / (c^2 u)); autograd of this rho yields
            # the IRLS weight (1/u)/(1 + r^2/(c^2 u)) automatically. u must be
            # detached upstream: rho is decreasing in u, so an attached u would
            # reward pushing points into high-variance (unobserved) space.
            c2 = self.cauchy_c**2
            u = variances.clamp_min(1e-6)
            cauchy_losses = (c2 / 2) * torch.log1p(residuals / (c2 * u))
            weights = (1.0 / u) / (1.0 + residuals / (c2 * u))
        else:
            # Apply Cauchy M-estimator
            cauchy_losses = self.cauchy_loss(residuals)
            # Compute weights for analysis
            weights = self.compute_weights(residuals)

        # Optional: weight by gradient magnitude (confidence)
        if self.use_gradient_weighting and gradients is not None:
            grad_magnitude = torch.norm(gradients, dim=-1)
            # Higher gradient magnitude = more confident (near surface)
            # Use sigmoid to map gradient magnitude to [0, 1]
            gradient_weights = torch.sigmoid(5 * (grad_magnitude - 0.5))
            cauchy_losses = cauchy_losses * gradient_weights
            weights = weights * gradient_weights

        # Compute per-batch loss (mean over points)
        if valid_mask is not None:
            mask = valid_mask.to(cauchy_losses.dtype)
            cauchy_losses = cauchy_losses * mask
            weights = weights * mask
            sdf_batch_losses = cauchy_losses.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            sdf_batch_losses = torch.mean(cauchy_losses, dim=1)
        normal_losses = torch.zeros_like(sdf_batch_losses)
        if self.normal_loss_weight > 0 and gradients is not None and source_normals is not None:
            src_normals = F.normalize(source_normals, p=2, dim=-1, eps=1.0e-8)
            sdf_normals = F.normalize(gradients, p=2, dim=-1, eps=1.0e-8)
            cos_sim = (src_normals * sdf_normals).sum(dim=-1).clamp(0.0, 1.0)
            normal_residual = 1.0 - cos_sim
            normal_weights = weights.detach()
            denom = normal_weights.sum(dim=1).clamp_min(1.0e-8)
            normal_losses = (normal_residual * normal_weights).sum(dim=1) / denom

        batch_losses = sdf_batch_losses + self.normal_loss_weight * normal_losses
        
        # Compute inlier ratio
        inlier_ratio = (weights > self.inlier_threshold).float().mean()
        
        return {
            'loss': batch_losses,
            'sdf_loss': sdf_batch_losses,
            'normal_loss': normal_losses,
            'weights': weights,
            'inlier_ratio': inlier_ratio,
            'mean_loss': torch.mean(batch_losses)
        }


class TukeyBiweightLoss(nn.Module):
    """
    Alternative robust loss using Tukey's biweight function.
    
    More aggressive outlier rejection compared to Cauchy.
    """
    
    def __init__(self, c: float = 4.685, use_gradient_weighting: bool = True):
        super().__init__()
        self.c = c
        self.use_gradient_weighting = use_gradient_weighting
        
    def forward(self, sdf_values: torch.Tensor, gradients: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute Tukey biweight loss"""
        residuals = sdf_values**2
        scaled_residuals = torch.sqrt(residuals) / self.c
        
        # Tukey biweight: ρ(e) = (c²/6) * [1 - (1 - (e/c)²)³] if |e| <= c, else c²/6
        mask = scaled_residuals <= 1.0
        
        losses = torch.zeros_like(residuals)
        losses[mask] = (self.c**2 / 6) * (1 - (1 - scaled_residuals[mask]**2)**3)
        losses[~mask] = self.c**2 / 6
        
        # Weights
        weights = torch.zeros_like(residuals)
        weights[mask] = (1 - scaled_residuals[mask]**2)**2
        
        # Optional gradient weighting
        if self.use_gradient_weighting and gradients is not None:
            grad_magnitude = torch.norm(gradients, dim=-1)
            gradient_weights = torch.sigmoid(5 * (grad_magnitude - 0.5))
            losses = losses * gradient_weights
            weights = weights * gradient_weights
        
        batch_losses = torch.mean(losses, dim=1)
        inlier_ratio = (weights > 0.5).float().mean()
        
        return {
            'loss': batch_losses,
            'weights': weights,
            'inlier_ratio': inlier_ratio,
            'mean_loss': torch.mean(batch_losses)
        }
