# CloudCropper substitute for the (now-removed) unlimblue/KNN_CUDA package.
#
# BUFFER-X's only inference-path use of KNN is exact k-nearest-neighbour search
# over feature descriptors / keypoints (models/BUFFERX.py: mutual_matching,
# get_matching_indices). The original `knn_cuda` shipped a custom CUDA kernel,
# but its GitHub release asset AND repository were removed (404 as of
# 2026-06-16), so it can no longer be installed from the upstream source.
#
# This is a drop-in replacement with the SAME interface and SAME result: a
# brute-force k-NN on the active device (torch.cdist + topk) returns the exact
# Euclidean nearest neighbours. The descriptor/keypoint counts on the inference
# path are small (~num_fps per scale), so the O(M*N) distance matrix is cheap.
#
# It is only used when the real `knn_cuda` is NOT installed (the worker appends
# this dir to sys.path as a fallback), so a genuine install always wins.
import torch


class KNN:
    """Mimics knn_cuda.KNN(k, transpose_mode=True).__call__(ref, query).

    transpose_mode=True  -> ref/query are (B, N, C); returns (dist, idx) as
                            (B, M, k) with idx indexing into ref's N axis.
    transpose_mode=False -> ref/query are (B, C, N) (channels first).
    """

    def __init__(self, k, transpose_mode=True):
        self.k = int(k)
        self.transpose_mode = transpose_mode

    def __call__(self, ref, query):
        if not self.transpose_mode:
            ref = ref.transpose(-1, -2)
            query = query.transpose(-1, -2)
        # ref: (B, N, C), query: (B, M, C)
        dist = torch.cdist(query.float(), ref.float())  # (B, M, N)
        d, idx = dist.topk(self.k, dim=-1, largest=False)  # (B, M, k)
        return d, idx
