"""Surface-builder layer.

An abstract :class:`~lidarrig.surfaces.base.SurfaceBuilder` turns a stream of
point batches into a reconstructed surface.  The concrete implementation shipped
now is :class:`~lidarrig.surfaces.kalman_heightmap.KalmanHeightmap`.

Nothing here imports a visualization library at module load time — meshes are
produced as plain NumPy arrays (:class:`~lidarrig.surfaces.base.MeshData`), and
the optional Open3D conversion lives behind a lazily-imported helper so a
headless ROS 2 node can use this layer untouched.

See the README section "Adding a new surface-reconstruction algorithm" and
:mod:`lidarrig.surfaces.example_stub` for how to add another builder.
"""

from rocklabel.live.surfaces.base import MeshData, SurfaceBuilder
from rocklabel.live.surfaces.kalman_heightmap import KalmanHeightmap

__all__ = ["MeshData", "SurfaceBuilder", "KalmanHeightmap", "make_surface_builder"]


def make_surface_builder(config) -> SurfaceBuilder:  # type: ignore[no-untyped-def]
    """Factory for the configured surface builder.

    Currently only ``KalmanHeightmap`` ships; add an ``elif`` here when you add a
    new :class:`SurfaceBuilder` subclass.
    """
    return KalmanHeightmap(config)
