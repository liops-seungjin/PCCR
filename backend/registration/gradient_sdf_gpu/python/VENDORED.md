# Vendored: gradient_sdf_registration

| | |
|---|---|
| Origin | `/home/sjjung/Project/HD-samho-dt/HD-samho-DT-webviewer/third-part/gradient_sdf_registration/` (git submodule of HD-samho-DT-webviewer) |
| Commit | `f4da1a09a4c017eae54c8f888ce640d17d0cd360` (2026-06-10, merge of `feat/fft-exhaustive-grid-init`) |
| Vendored | 2026-06-11 |
| Contents | the inner `gradient_sdf_registration/` package only: `__init__.py`, `gradient_sdf.py`, `registration.py`, `pca_transform.py`, `robust_loss.py`, `_iou_func.py`, `exhaustive_grid/` (incl. `rotations/*.npy`, 11 files ≈2.7 MB) |
| Excluded | `.git`, `.gitignore`, `__pycache__/`, `scripts/` (PyQt5/pyvista GUI demo), `setup.py`, `README.md`, `requirements.txt` (superseded by the one here), `CLAUDE.md`, `REFACTORING_SUMMARY.md` |
| Local modifications | yes — see below; every edit is marked with a `# CloudCropper:` comment |

## Local modifications (2026-06-11)

Uncertainty/variance channel ported from CloudCropper's (since removed) native
C++ gradient-SDF backend:

- `gradient_sdf.py` — new `add_uncertainty_channel(points, normals, *, trunc,
  spacing)` method (optional 5th grid channel: GPIS variance = Matern-3/2 data
  proximity + local plane residual + voxel quantization floor, plus
  `median_variance`/`uncertainty_trunc`/`has_uncertainty` attributes);
  `query_sdf_and_gradient(..., return_variance=False)` kwarg; attribute copy in
  `copy_to_device`.
- `robust_loss.py` — `RobustSDFLoss.forward(..., variances=None)`: when given
  the normalized (and pre-detached) variance `u`, uses the heteroscedastic
  Cauchy `rho = (c²/2)·log1p(r²/(c²u))` instead of the plain one.
- `registration.py` — `_query_with_uncertainty()` helper (detaches `u`);
  `register(..., use_uncertainty=False)` kwarg wired into both the fft
  per-candidate refine loop and the main descent loop;
  `info["uncertainty_applied"]`.

The package is pure Python (no compiled extensions); the rotation presets are
loaded relative to `__file__`, so the copy is relocation-safe. It is imported
by `gsdf_worker.py` in this directory (same dir ⇒ `sys.path[0]`, no install
needed). To update: re-copy the inner package from the origin at a newer
commit, refresh this table, and re-run the gsdf-gpu tests.
