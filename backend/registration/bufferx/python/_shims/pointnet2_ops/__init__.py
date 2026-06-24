# CloudCropper pure-torch substitute for the `pointnet2_ops` package.
#
# Upstream BUFFER-X imports `pointnet2_ops.pointnet2_utils as pnt2` and uses four
# of its CUDA-compiled ops on the inference path: furthest_point_sample,
# gather_operation, ball_query, grouping_operation. The real package
# (erikwijmans/Pointnet2_PyTorch) needs a per-arch CUDA build that is not
# available for Blackwell (sm_120) wheels, so this directory provides exact
# pure-torch equivalents. It is only on sys.path as a fallback (the worker
# appends ./_shims LAST), so a genuine install always wins.
from . import pointnet2_utils  # noqa: F401
