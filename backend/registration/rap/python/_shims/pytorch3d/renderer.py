"""Import-only stubs for ``pytorch3d.renderer`` (RAP renderer, never run headless).

RAP's ``utils/render.py`` imports these names at module top for PyTorch3D-based
visualization. The inference worker never renders, so each is a stub that raises on use.
"""


def _stub(name):
    def _factory(*args, **kwargs):
        raise NotImplementedError(
            f"pytorch3d.renderer.{name} is a viz-only stub in the RAP worker; "
            "install real pytorch3d to render."
        )

    _factory.__name__ = name
    return _factory


look_at_view_transform = _stub("look_at_view_transform")
FoVPerspectiveCameras = _stub("FoVPerspectiveCameras")
PointsRasterizationSettings = _stub("PointsRasterizationSettings")
PointsRasterizer = _stub("PointsRasterizer")
PointsRenderer = _stub("PointsRenderer")
AlphaCompositor = _stub("AlphaCompositor")
