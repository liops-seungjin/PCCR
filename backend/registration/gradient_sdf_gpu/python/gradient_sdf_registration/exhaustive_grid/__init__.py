"""
Exhaustive grid search FFT registration, vendored from
https://github.com/DavidBoja/exhaustive-grid-search (MVA 2024, Bojanic et al.).

"Addressing the generalization of 3D registration methods with a featureless
baseline and an unbiased benchmark", Machine Vision and Applications 2024.

Used as the initialization stage ahead of gradient-SDF refinement: uniform
SO(3) rotation candidates, translation solved per rotation by FFT
cross-correlation of voxel occupancy grids, top-k poses returned.
"""

from .fft_init import exhaustive_grid_topk
from .rot_utils import load_rotations

__all__ = ["exhaustive_grid_topk", "load_rotations"]
