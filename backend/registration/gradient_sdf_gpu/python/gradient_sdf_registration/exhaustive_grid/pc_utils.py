"""
Vendored from https://github.com/DavidBoja/exhaustive-grid-search
(utils/pc_utils.py) — only the functions needed for FFT grid registration.
"""

import operator

import torch


def unravel_index_pytorch(flat_index, shape):
    flat_index = operator.index(flat_index)
    res = []

    # Short-circuits on zero dim tensors
    if shape == torch.Size([]):
        return 0

    for size in shape[::-1]:
        res.append(flat_index % size)
        flat_index = flat_index // size

    if len(res) == 1:
        return res[0]

    return tuple(res[::-1])


def voxelize(points, voxel_size, fill_positive=1, fill_negative=0):
    """
    Voxelize points to voxel_size.

    Input:  points: (torch) Nx3 points to voxelize
            voxel_size: (int) scalar that determines size of one voxel
            fill_positive: (int) number put in place of filled voxels
            fill_negative: (int) number put in place of emtpy voxels
    Returns: voxels (torch): voxelized points of dim
                            NR_VOXELS[0] x NR_VOXELS[1] x NR_VOXELS[2]
             NR_VOXELS: (torch) tensor of voxel dimensions, dim3
    """

    # max of input by ax
    max_ax_input = torch.max(points, dim=0)[0]
    NR_VOXELS = (torch.floor(max_ax_input / voxel_size) + 1).type(torch.int64)

    voxels = torch.zeros(tuple(NR_VOXELS.tolist())) + fill_negative
    voxel_indices = torch.floor(points / voxel_size).long()
    voxels[voxel_indices[:, 0],
           voxel_indices[:, 1],
           voxel_indices[:, 2]] = fill_positive

    return voxels, NR_VOXELS
