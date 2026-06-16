"""
Vendored from https://github.com/DavidBoja/exhaustive-grid-search
(utils/data_utils.py) — RotatePCB1 dataset + weighted preprocess_pcj_B1.
Only the "weighted" voxelization path is kept.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from .pc_utils import voxelize


class RotatePCB1(Dataset):
    """
    Rotates a given centered point cloud by a given batch of rotations.
    __getitem__ first rotates the point cloud and then makes it positive by
    translating the minimal bounding box point to the origin.
    """

    def __init__(self, pts, R_batch, voxel_size, pp,
                 subsampling_indices=None, center=torch.zeros(3),
                 voxelization_option=None,
                 fill_positive=5, fill_negative=-1, fill_padding=-1):

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]
        self.voxel_size = voxel_size
        self.pp = pp
        self.fill_positive = fill_positive
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding

        self.center = torch.mean(self.pts, axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]

    def __len__(self):
        return self.K

    def __getitem__(self, idx):

        # rotate
        points = torch.matmul(self.R_batch[idx],  # 3 x 3
                              self.points_preprocessed.T).T  # 3 X N

        # make positive by translating min bounding box point to origin
        minima = torch.min(points, dim=0)[0]
        points = points - minima

        # voxelize
        voxelized_pts, orig_shape = voxelize(points,
                                             self.voxel_size,
                                             self.fill_positive,
                                             self.fill_negative)  # Vx x Vy x Vz

        # pad
        voxelized_pts_padded = torch.nn.functional.pad(voxelized_pts.type(torch.int32),
                                                       self.pp,
                                                       mode='constant',
                                                       value=self.fill_padding)  # Vx x Vy x Vz

        return voxelized_pts_padded.unsqueeze(0), minima, orig_shape  # 1 x Vx x Vy x Vz


def preprocess_pcj_B1(pcj, R_batch, voxel_size, pp, num_workers,
                      fill_positive, fill_negative, fill_padding, **kwargs):
    """
    Create a dataloader that loads batch_size batches of rotated, voxelized and padded pcj
    points for a given dataset.

    The batches are voxelized and paded pcj rotated with a number of
    rotations from R_batch.

    Input:  pcj: (torch) Nx3 points
            R_batch: (torch) NX3x3 rotations
            voxel_size: (float) size of voxel side
            pp: (tuple) 6dim tuple for padding -- deterimend in padding.padding_options
            num_workers: (scalar) torch dataloader num workers
    Returns: my_data: (torch.Dataset) dataset that returns rotated pcj for given index of R_batch
             my_dataloader: (torch.DataLoader) datalodaer that loads batch_size batches of rotated,
                                                voxelized and padded pcj points for a given dataset.
    """

    rot_pbb1_data = RotatePCB1(pcj, R_batch, voxel_size, pp,
                               fill_positive=fill_positive,
                               fill_negative=fill_negative,
                               fill_padding=fill_padding)

    rot_pcb1_loader = DataLoader(rot_pbb1_data,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=num_workers
                                 )

    return rot_pbb1_data, rot_pcb1_loader
