"""Pure-torch ``chamfer_distance`` matching the subset of PyTorch3D's API RAP eval uses.

RAP's ``eval/metrics.py:compute_cd`` calls
``chamfer_distance(x=(1,N,3), y=(1,N,3), single_directional=False, norm=2,
point_reduction="mean")`` and then takes ``(0.5 * cd).sqrt()``. With ``norm=2``
PyTorch3D uses squared-L2 nearest-neighbor distances, which is what we reproduce here.
Returns ``(loss, loss_normals)`` like PyTorch3D (normals unused → None).
"""
from __future__ import annotations

import torch


def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    x_lengths=None,  # noqa: ARG001 — parity
    y_lengths=None,  # noqa: ARG001
    single_directional: bool = False,
    norm: int = 2,
    point_reduction: str = "mean",
    batch_reduction: str = "mean",
    **_ignored,
):
    if norm != 2:
        raise NotImplementedError(f"chamfer shim supports norm=2 only, got norm={norm}")
    d2 = torch.cdist(x.float(), y.float()) ** 2          # (B, N, M) squared-L2
    fwd = d2.min(dim=2).values                            # (B, N)
    if point_reduction == "sum":
        loss = fwd.sum(dim=1)
    else:
        loss = fwd.mean(dim=1)
    if not single_directional:
        bwd = d2.min(dim=1).values                        # (B, M)
        loss = loss + (bwd.sum(dim=1) if point_reduction == "sum" else bwd.mean(dim=1))
    if batch_reduction == "mean":
        loss = loss.mean()
    elif batch_reduction == "sum":
        loss = loss.sum()
    return loss.to(x.dtype), None
