"""Minimal pure-torch shim for the slice of PyTorch3D that RAP/RPF imports.

RAP's mini-SpinNet feature extractor (``dataset_process/utils/spinnet/...``) and its
FPS allocation (``dataset_process/utils/point_sampling_utils.py``) do
``from pytorch3d.ops import ball_query, sample_farthest_points`` at module top, so the
package must be importable even for the points-only path (where they are never called).
The pinned PyTorch3D 0.7.8 wheel is py310/cu121/torch2.5.1 only — it does not match this
box's torch-2.7/cu128 stack, and a source build is heavy/brittle. We therefore shadow
``pytorch3d`` with this shim (same trick as the flash_attn SDPA shim), implementing only
``ops.ball_query`` and ``ops.sample_farthest_points`` in pure torch with PyTorch3D's
exact I/O conventions. See ``ops.py`` for the semantics.
"""
__all__ = ["ops"]
