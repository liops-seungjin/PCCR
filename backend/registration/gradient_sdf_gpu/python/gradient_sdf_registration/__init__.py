"""
Gradient-SDF Registration Library
================================

A robust point cloud registration library using Gradient-SDF fields and PCA-based initialization.
"""

from .gradient_sdf import GradientSDFField
from .robust_loss import RobustSDFLoss
from .pca_transform import PCABatchSE3Transform
from .registration import (
    PCARegistration,
    MultiTargetRegistration,
    compute_3d_iou,
    NormalAwareIoU,
    compute_normal_aware_iou,
    SoftVoxelNormalIoU,
    compute_soft_normal_iou,
)

__version__ = "0.1.0"
__all__ = [
    "GradientSDFField",
    "RobustSDFLoss",
    "PCABatchSE3Transform",
    "PCARegistration",
    "MultiTargetRegistration",
    "compute_3d_iou",
    "NormalAwareIoU",
    "compute_normal_aware_iou",
    "SoftVoxelNormalIoU",
    "compute_soft_normal_iou",
]