# CloudCropper pure-torch substitute for the `torch_batch_svd` package.
#
# Upstream BUFFER-X does `from torch_batch_svd import svd` and calls it once on
# the inference path (utils/common.cal_Z_axis: batched 3x3 covariance SVD for the
# local reference-frame Z axis). The real package (KinglittleQ/torch-batch-svd)
# is a CUDA extension; torch.linalg.svd is itself batched and fast for these tiny
# 3x3 matrices, so this is an exact drop-in. Only used when the real package is
# absent (the worker appends ./_shims LAST), so a genuine install still wins.
import torch


def svd(input, some=True, compute_uv=True):
    """Batched SVD returning (U, S, V) like torch.svd / torch_batch_svd.svd.

    Note the V (not V^H) convention: torch.linalg.svd returns Vh, so we transpose
    it back. cal_Z_axis uses U's last column, but we return the full triple to
    match the original interface.
    """
    U, S, Vh = torch.linalg.svd(input, full_matrices=not some)
    V = Vh.transpose(-2, -1).conj()
    return U, S, V
