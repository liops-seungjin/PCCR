# Pure-torch drop-in for pointnet2_ops.pointnet2_utils (inference subset).
#
# Implements the four ops BUFFER-X calls at test time with the SAME signatures
# and SAME results as the CUDA package, so the vendored upstream core runs
# unmodified on a Blackwell (sm_120) box where the compiled extension is
# unavailable. Memory is bounded by chunking the brute-force distance work;
# the inference-path tensors are small (num_fps ~1500 keypoints, 420-voxel SPT
# grid), so the O(M*N) distance matrices are cheap.
#
# References (real package behaviour this reproduces):
#   furthest_point_sample(xyz(B,N,3), npoint)      -> idx (B, npoint)
#   gather_operation(feat(B,C,N), idx(B,M))        -> (B, C, M)
#   ball_query(radius, nsample, xyz(B,N,3),
#              new_xyz(B,M,3))                      -> idx (B, M, nsample)
#   grouping_operation(feat(B,C,N), idx(B,M,S))    -> (B, C, M, S)
import torch

# Cap for the element count of any single (rows x N) distance tensor materialised
# inside ball_query, ~32M floats = 128 MB. Keeps peak memory bounded regardless
# of cloud size while staying a single chunk for the typical inference shapes.
_DIST_BUDGET = 32_000_000


def furthest_point_sample(xyz, npoint):
    """Iterative farthest-point sampling. xyz (B, N, 3) -> idx (B, npoint) long.

    Matches the CUDA kernel: the first sampled point is index 0, each subsequent
    pick maximises the min-distance to the already-selected set.
    """
    B, N, _ = xyz.shape
    npoint = int(npoint)
    device = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device, dtype=xyz.dtype)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch = torch.arange(B, device=device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch, farthest, :].unsqueeze(1)          # (B, 1, 3)
        d = torch.sum((xyz - centroid) ** 2, dim=-1)             # (B, N)
        dist = torch.minimum(dist, d)
        farthest = torch.max(dist, dim=-1).indices
    return idx


def gather_operation(features, idx):
    """features (B, C, N), idx (B, M) -> (B, C, M)."""
    B, C, N = features.shape
    M = idx.shape[1]
    idx_exp = idx.long().unsqueeze(1).expand(B, C, M)
    return torch.gather(features, 2, idx_exp).contiguous()


def grouping_operation(features, idx):
    """features (B, C, N), idx (B, M, S) -> (B, C, M, S)."""
    B, C, N = features.shape
    _, M, S = idx.shape
    idx_flat = idx.long().reshape(B, 1, M * S).expand(B, C, M * S)
    out = torch.gather(features, 2, idx_flat).reshape(B, C, M, S)
    return out.contiguous()


def ball_query(radius, nsample, xyz, new_xyz):
    """Group up to `nsample` points of `xyz` within `radius` of each `new_xyz`.

    Returns idx (B, M, nsample) long, reproducing the CUDA semantics exactly:
    indices are the in-range points in ascending point order, truncated to
    `nsample`; when fewer than `nsample` are found the remaining slots are filled
    with the FIRST in-range index, and when none are found the row is all zeros.
    """
    radius = float(radius)
    nsample = int(nsample)
    B, N, _ = xyz.shape
    M = new_xyz.shape[1]
    device = xyz.device

    out = torch.empty(B, M, nsample, dtype=torch.long, device=device)
    arange_n = torch.arange(N, device=device)
    pos = torch.arange(nsample, device=device).unsqueeze(0)      # (1, nsample)
    chunk = max(1, min(M, _DIST_BUDGET // max(1, N)))

    for b in range(B):
        p = xyz[b].float()                                       # (N, 3)
        m0 = 0
        while m0 < M:
            q = new_xyz[b, m0:m0 + chunk].float()                # (cm, 3)
            dist = torch.cdist(q, p)                             # (cm, N)
            within = dist < radius
            # in-range -> point index; out-of-range -> pushed past every in-range
            # index so argsort orders all in-range first (ascending), then the
            # rest; keys are unique so ordering is stable without stable=True.
            key = torch.where(within, arange_n, N + arange_n)    # (cm, N)
            sorted_idx = key.argsort(dim=-1)[:, :nsample]        # (cm, nsample)
            cnt = within.sum(dim=-1, keepdim=True)               # (cm, 1)
            first = sorted_idx[:, :1]                            # (cm, 1)
            valid = pos < cnt                                    # (cm, nsample)
            out[b, m0:m0 + chunk] = torch.where(valid, sorted_idx, first)
            m0 += chunk
    return out
