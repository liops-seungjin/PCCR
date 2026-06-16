"""
Exhaustive grid search initialization — top-k (R, t) candidates.

Core loop vendored from https://github.com/DavidBoja/exhaustive-grid-search
register.py (MVA 2024, Bojanic et al.) with two changes:
  * returns the top-k scoring poses instead of the single argmax
  * voxel grids are cast to float32 before fft_conv (newer torch versions
    reject integer inputs to torch.fft.rfftn)
"""

from typing import List, Tuple

import numpy as np
import torch

from .data_utils import preprocess_pcj_B1
from .fft_conv import fft_conv
from .padding import padding_options
from .pc_utils import unravel_index_pytorch, voxelize
from .rot_utils import create_T_estim_matrix, load_rotations


def _suppress_neighborhood(volume: torch.Tensor, index: Tuple[int, int, int], radius: int) -> None:
    i1, i2, i3 = index
    volume[max(0, i1 - radius):i1 + radius + 1,
           max(0, i2 - radius):i2 + radius + 1,
           max(0, i3 - radius):i3 + radius + 1] = float("-inf")


def _relative_rotation_angle_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> float:
    trace = float(torch.trace(Ra @ Rb.T))
    cos = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(np.rad2deg(np.arccos(cos)))


def exhaustive_grid_topk(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    voxel_size: float = 0.5,
    rotation_choice: str = "AA_ICO42_S15",
    topk: int = 8,
    pv: int = 5,
    nv: int = -1,
    ppv: int = -1,
    padding: str = "same",
    num_workers: int = 4,
    peaks_per_rotation: int = 3,
    min_peak_separation_m: float = 5.0,
    min_rotation_separation_deg: float = 30.0,
    device: torch.device | None = None,
) -> List[Tuple[np.ndarray, float]]:
    """Score every precomputed rotation by FFT cross-correlation and return
    the top-k poses.

    source_points: (N,3) scan points to be registered onto target_points.
    target_points: (M,3) points sampled from the registration target (CAD).

    The raw correlation score favors the densest region of the source scene,
    so a single argmax collapses onto the biggest structure even when the
    target belongs elsewhere. Instead, ``peaks_per_rotation`` spatially
    separated correlation peaks are kept per rotation (non-max suppression by
    ``min_peak_separation_m``), and the final top-k is picked greedily with
    the same spatial diversity constraint (two candidates closer than the
    separation are considered duplicates unless their rotations differ by
    more than ``min_rotation_separation_deg``).

    Returns a list of (T, score) with T a 4x4 source->target transform,
    sorted by descending correlation score.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pci = torch.from_numpy(np.asarray(target_points, dtype=np.float64))
    pcj = torch.from_numpy(np.asarray(source_points, dtype=np.float64))

    R_batch = load_rotations(rotation_choice)

    #### PREPROCESS pci ##########################################################
    # 1. make pci positive for voxelization
    make_pci_posit_translation = torch.min(pci, axis=0)[0]
    pci = pci - make_pci_posit_translation

    # 2. voxelize pci
    pci_voxel, NR_VOXELS_PCI = voxelize(pci, voxel_size,
                                        fill_positive=pv,
                                        fill_negative=nv)

    # find indices of the pci central voxel
    CENTRAL_VOXEL_PCI = torch.where(NR_VOXELS_PCI % 2 == 0,  # check if even
                                    (NR_VOXELS_PCI / 2) - 1,  # if even take one voxel to the left
                                    torch.floor(NR_VOXELS_PCI / 2)).int()  # else just take middle voxel
    # find central voxel in xyz coordinates
    central_voxel_center = CENTRAL_VOXEL_PCI * voxel_size + (0.5 * voxel_size)

    # 3. move pci on cuda -- dims needed 1 x 1 x Vx x Vy x Vz
    weight_to_fftconv3d = pci_voxel.type(torch.float32).to(device)[None, None, :, :, :]

    #### PREPROCESS pcj ##########################################################
    # define padding (z,y,x) axis is the order for padding
    pp, pp_xyz = padding_options(padding,
                                 CENTRAL_VOXEL_PCI,
                                 NR_VOXELS_PCI)

    my_data, my_dataloader = preprocess_pcj_B1(pcj,
                                               R_batch,
                                               voxel_size,
                                               pp,
                                               num_workers,
                                               pv,
                                               nv,
                                               ppv)

    #### PROCESS (FFT) ###########################################################
    # peaks_per_rotation spatially separated peaks per rotation instead of the
    # vendored single argmax
    suppression_radius = max(1, int(round(min_peak_separation_m / voxel_size)))
    peak_candidates: List[Tuple[float, int, Tuple[int, int, int]]] = []
    minimas = torch.empty(R_batch.shape[0], 3)

    for ind_dataloader, (voxelized_pts_padded, mins, orig_input_shape) in enumerate(my_dataloader):
        minimas[ind_dataloader, :] = mins

        input_to_fftconv3d = voxelized_pts_padded.type(torch.float32).to(device)

        out = fft_conv(input_to_fftconv3d,
                       weight_to_fftconv3d, bias=None)

        volume = out[0, 0]
        for _ in range(max(1, int(peaks_per_rotation))):
            flat_index = int(torch.argmax(volume))
            peak_index = unravel_index_pytorch(flat_index, volume.shape)
            score = float(volume[peak_index])
            if not np.isfinite(score):
                break
            peak_candidates.append((score, ind_dataloader, tuple(int(i) for i in peak_index)))
            _suppress_neighborhood(volume, peak_index, suppression_radius)

    #### POST-PROCESS ############################################################
    def _pose_for(rotation_index: int, peak_index: Tuple[int, int, int]) -> np.ndarray:
        ind1, ind2, ind3 = peak_index
        R = R_batch[rotation_index]

        # translation -- translate for padding pp_xyz and CENTRAL_VOXEL_PCI
        t = torch.Tensor([-(pp_xyz[0] * voxel_size) +
                          ((CENTRAL_VOXEL_PCI[0]) * voxel_size) +
                          (ind1 * voxel_size) +
                          (0.5 * voxel_size),

                          -(pp_xyz[2] * voxel_size) +
                          ((CENTRAL_VOXEL_PCI[1]) * voxel_size) +
                          (ind2 * voxel_size) +
                          (0.5 * voxel_size),

                          -(pp_xyz[4] * voxel_size) +
                          ((CENTRAL_VOXEL_PCI[2]) * voxel_size) +
                          (ind3 * voxel_size) +
                          (0.5 * voxel_size)
                          ])

        center_pcj_translation = my_data.center
        make_pcj_posit_translation = minimas[rotation_index]
        estim_T_baseline = create_T_estim_matrix(center_pcj_translation,
                                                 R,
                                                 make_pcj_posit_translation,
                                                 central_voxel_center,
                                                 t,
                                                 make_pci_posit_translation
                                                 )
        return estim_T_baseline.numpy().astype(np.float64)

    # greedy top-k with spatial diversity measured where the target center
    # lands in the source (world) frame — T maps source->target, so the
    # target sits at T^-1 @ target_center
    target_center_local = np.asarray(target_points, dtype=np.float64).mean(axis=0)

    def _target_center_world(T: np.ndarray) -> np.ndarray:
        T_inv = np.linalg.inv(T)
        return T_inv[:3, :3] @ target_center_local + T_inv[:3, 3]

    # spatial quota: at most max_per_cell candidates per XY cell of size
    # min_peak_separation_m, so a single dense region cannot fill every slot
    # and remote (possibly truncated) structures keep candidates
    max_per_cell = 2
    peak_candidates.sort(key=lambda c: c[0], reverse=True)
    results: List[Tuple[np.ndarray, float]] = []
    cell_counts: dict = {}
    cell_rotations: dict = {}
    for score, rotation_index, peak_index in peak_candidates:
        T = _pose_for(rotation_index, peak_index)
        center_world = _target_center_world(T)
        cell = (int(np.floor(center_world[0] / min_peak_separation_m)),
                int(np.floor(center_world[1] / min_peak_separation_m)))
        if cell_counts.get(cell, 0) >= max_per_cell:
            continue
        too_similar = False
        for rot_sel in cell_rotations.get(cell, []):
            angle = _relative_rotation_angle_deg(R_batch[rotation_index].double(),
                                                 R_batch[rot_sel].double())
            if angle < min_rotation_separation_deg:
                too_similar = True
                break
        if too_similar:
            continue
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        cell_rotations.setdefault(cell, []).append(rotation_index)
        results.append((T, float(score)))
        if len(results) >= topk:
            break

    return results
