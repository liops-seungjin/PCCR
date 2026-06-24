"""Import-only stub for ``pytorch3d.structures`` (used solely by RAP's renderer).

RAP's ``utils/render.py`` imports ``Pointclouds`` at module top, but rendering is never
invoked by the headless inference worker. This stub keeps the import working; touching it
raises, so any accidental use is loud rather than silently wrong.
"""


class Pointclouds:  # noqa: D101 — stub
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "pytorch3d.structures.Pointclouds is a viz-only stub in the RAP worker; "
            "install real pytorch3d to render."
        )
