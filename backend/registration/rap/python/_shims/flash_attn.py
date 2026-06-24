"""SDPA-based drop-in shim for the slice of flash-attn that RAP/RPF uses.

RAP's ``rectified_point_flow/flow_model/layer.py`` does a bare ``import flash_attn``
and calls ``flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen,
softcap)`` in its part-wise and global attention. The pinned flash-attn 2.7.4 wheel
does NOT support consumer Blackwell (sm_120, e.g. RTX 50xx) and there is no SDPA path
upstream. We therefore put this module FIRST on sys.path so ``import flash_attn``
resolves here — zero upstream edits (same trick as BUFFER-X's knn_cuda shim).

It reimplements the varlen, qkv-packed attention with
``torch.nn.functional.scaled_dot_product_attention`` (which has a flash/mem-efficient
backend on sm_120 under torch>=2.7). Attention is computed independently within each
segment described by ``cu_seqlens`` (no cross-segment leakage), matching flash-attn's
varlen semantics. RAP runs with ``softcap=0`` (the default, never overridden), which
maps to plain SDPA; a non-zero softcap falls back to an explicit attention that applies
``softcap * tanh(logits / softcap)`` before the softmax (heavier, but correct).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # torch>=2.3
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # Prefer the O(L)-memory kernels; fall back to math only as a last resort. The
    # default selection sometimes picks the math backend (which materializes the full
    # L×L scores and OOMs on the global attention), so we pin the preference order.
    _SDPA_BACKENDS = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
except Exception:  # pragma: no cover - very old torch
    sdpa_kernel = None
    _SDPA_BACKENDS = None

__all__ = ["flash_attn_varlen_qkvpacked_func", "__version__"]

# Advertise a version so any ``flash_attn.__version__`` probe does not crash.
__version__ = "0.0.0-sdpa-shim"


def _sdpa(q, k, v, *, dropout_p, causal, scale):
    """SDPA on (H,L,D) inputs, forcing memory-efficient kernels when available."""
    q4, k4, v4 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)  # (1,H,L,D) for flash
    if sdpa_kernel is not None:
        with sdpa_kernel(_SDPA_BACKENDS):
            o = F.scaled_dot_product_attention(
                q4, k4, v4, dropout_p=dropout_p, is_causal=causal, scale=scale
            )
    else:
        o = F.scaled_dot_product_attention(
            q4, k4, v4, dropout_p=dropout_p, is_causal=causal, scale=scale
        )
    return o.squeeze(0)  # (H,L,D)


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int | None = None,  # noqa: ARG001 — accepted for API parity, unused by SDPA
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    softcap: float = 0.0,
    **_ignored,
) -> torch.Tensor:
    """varlen packed-QKV self-attention via SDPA.

    Args:
        qkv: ``(T, 3, H, D)`` packed query/key/value for all tokens, where T is the
            total token count across every segment.
        cu_seqlens: ``(S + 1,)`` int cumulative segment boundaries (``cu[0] == 0``,
            ``cu[-1] == T``). Attention is restricted to within each ``[cu[i], cu[i+1])``.
        softcap: tanh logit soft-cap; ``0`` disables it (the RAP default → fast SDPA).

    Returns:
        ``(T, H, D)`` attention output, same dtype/device as ``qkv``.
    """
    if qkv.dim() != 4 or qkv.shape[1] != 3:
        raise ValueError(f"expected qkv of shape (T,3,H,D), got {tuple(qkv.shape)}")
    T, _, H, D = qkv.shape
    scale = softmax_scale if softmax_scale is not None else (D ** -0.5)
    use_softcap = softcap is not None and softcap != 0.0

    out = torch.empty((T, H, D), dtype=qkv.dtype, device=qkv.device)
    bounds = cu_seqlens.detach().to("cpu", torch.long).tolist()
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e <= s:
            continue
        seg = qkv[s:e]                       # (L, 3, H, D)
        q, k, v = seg.unbind(dim=1)          # each (L, H, D)
        # SDPA wants (..., L, D) with the head dim batched: (H, L, D).
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()
        if use_softcap:
            attn = torch.matmul(q, k.transpose(-1, -2)) * scale          # (H, L, L)
            attn = softcap * torch.tanh(attn / softcap)
            if causal:
                L = e - s
                mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=qkv.device), 1)
                attn = attn.masked_fill(mask, float("-inf"))
            o = torch.matmul(attn.softmax(dim=-1), v)                    # (H, L, D)
        else:
            o = _sdpa(q, k, v, dropout_p=dropout_p, causal=causal, scale=scale)
        out[s:e] = o.transpose(0, 1)          # back to (L, H, D)
    return out
