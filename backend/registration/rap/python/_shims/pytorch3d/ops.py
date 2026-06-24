"""Pure-torch reimplementations of the two PyTorch3D ops RAP/RPF uses.

Both match PyTorch3D 0.7.8's I/O conventions closely enough for RAP's mini-SpinNet and
FPS allocation:

- ``sample_farthest_points(points, lengths=None, K=50, random_start_point=False)`` ->
  ``(selected_points (B,maxK,3), selected_idx (B,maxK))``; iterative farthest-point
  sampling, padded with -1 indices when a batch element asks for more than it has.
- ``ball_query(p1, p2, K, radius, return_nn=True)`` ->
  ``(dists (B,P1,K), idx (B,P1,K), nn (B,P1,K,3))``; for each query in ``p1`` the FIRST
  ``K`` points of ``p2`` (in array order) whose squared distance is ``<= radius**2``,
  padded with idx ``-1`` / dist ``0`` / nn ``0`` — exactly PyTorch3D's fixed-radius,
  not nearest-K, behavior (RAP shuffles ``p2`` before calling, so order == random).
"""
from __future__ import annotations

from typing import Optional, Union

import torch

__all__ = [
    "ball_query",
    "sample_farthest_points",
    "iterative_closest_point",
    "_PointsNeighbors",
]


def iterative_closest_point(*args, **kwargs):
    """Eval-only stub (RAP uses it in metrics, never on the inference path)."""
    raise NotImplementedError(
        "pytorch3d.ops.iterative_closest_point is an eval-only stub in the RAP worker; "
        "install real pytorch3d to use it."
    )


def _as_per_batch_int(x, B, device, dflt):
    if x is None:
        return torch.full((B,), int(dflt), dtype=torch.long, device=device)
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long).expand(B) if x.dim() == 0 else x.to(
            device=device, dtype=torch.long
        )
    return torch.full((B,), int(x), dtype=torch.long, device=device)


def sample_farthest_points(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, torch.Tensor] = 50,
    random_start_point: bool = False,
):
    """Iterative farthest point sampling. Returns (selected_points, selected_idx)."""
    B, P, D = points.shape
    device = points.device
    lengths = (
        torch.full((B,), P, dtype=torch.long, device=device)
        if lengths is None
        else lengths.to(device=device, dtype=torch.long)
    )
    Ks = _as_per_batch_int(K, B, device, 50)
    maxK = int(Ks.max().item()) if B > 0 else 0

    idx = torch.full((B, maxK), -1, dtype=torch.long, device=device)
    for b in range(B):
        n = int(lengths[b].item())
        k = min(int(Ks[b].item()), n)
        if n <= 0 or k <= 0:
            continue
        pts = points[b, :n]                                   # (n, D)
        dist = torch.full((n,), float("inf"), device=device)
        start = (
            int(torch.randint(0, n, (1,), device=device).item())
            if random_start_point
            else 0
        )
        sel = idx[b]
        sel[0] = start
        last = start
        for i in range(1, k):
            d = ((pts - pts[last]) ** 2).sum(-1)              # (n,)
            dist = torch.minimum(dist, d)
            last = int(torch.argmax(dist).item())
            sel[i] = last
    sel_clamped = idx.clamp(min=0)
    selected = torch.gather(points, 1, sel_clamped.unsqueeze(-1).expand(B, maxK, D))
    selected = torch.where((idx >= 0).unsqueeze(-1), selected, torch.zeros_like(selected))
    return selected, idx


class _PointsNeighbors(tuple):
    """Mimics pytorch3d's namedtuple-ish return (dists, idx, knn) while staying a tuple."""

    @property
    def dists(self):
        return self[0]

    @property
    def idx(self):
        return self[1]

    @property
    def knn(self):
        return self[2]


def ball_query(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Optional[torch.Tensor] = None,  # noqa: ARG001 — accepted for parity
    lengths2: Optional[torch.Tensor] = None,
    K: int = 500,
    radius: float = 0.2,
    return_nn: bool = True,
):
    """Fixed-radius neighbor query matching PyTorch3D's (dists, idx, nn) convention."""
    B, P1, D = p1.shape
    P2 = p2.shape[1]
    device = p1.device
    r2 = float(radius) * float(radius)

    d2 = torch.cdist(p1.float(), p2.float()) ** 2                  # (B, P1, P2)
    if lengths2 is not None:
        ar = torch.arange(P2, device=device).view(1, 1, P2)
        valid_len = ar < lengths2.to(device).view(B, 1, 1)
        d2 = d2.masked_fill(~valid_len, float("inf"))
    within = d2 <= r2                                              # (B, P1, P2)

    order = torch.arange(P2, device=device).view(1, 1, P2).expand(B, P1, P2)
    key = torch.where(within, order, torch.full_like(order, P2))  # invalid sorts last
    sel = key.sort(dim=-1).values[..., :K]                        # (B, P1, K)
    valid = sel < P2
    sel_c = sel.clamp(max=P2 - 1)

    idx = torch.where(valid, sel_c, torch.full_like(sel_c, -1))
    dists = torch.gather(d2, 2, sel_c).to(p1.dtype)
    dists = torch.where(valid, dists, torch.zeros_like(dists))
    nn = None
    if return_nn:
        gather_idx = sel_c.unsqueeze(-1).expand(B, P1, K, D)
        nn = torch.gather(p2.unsqueeze(1).expand(B, P1, P2, D), 2, gather_idx)
        nn = torch.where(valid.unsqueeze(-1), nn, torch.zeros_like(nn))
    return _PointsNeighbors((dists, idx, nn))
