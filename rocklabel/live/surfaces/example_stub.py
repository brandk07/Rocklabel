"""Template for adding a new surface-reconstruction algorithm.

Copy this file, rename the class, implement the three abstract methods, and
register it in :func:`lidarrig.surfaces.make_surface_builder`.  The rest of the
app (sources, filters, visualizer, CLI) needs no changes.

Only ``add_points`` / ``get_mesh_arrays`` / ``reset`` are required.  Keeping the
class free of Open3D imports (use :class:`MeshData` NumPy arrays) means it stays
reusable from a headless ROS 2 node.
"""

from __future__ import annotations

import numpy as np

from rocklabel.live.config import AppConfig
from rocklabel.live.surfaces.base import MeshData, SurfaceBuilder


class MovingAverageHeightmap(SurfaceBuilder):
    """Minimal example: a running-mean heightmap (no variance tracking).

    This is intentionally simpler than :class:`KalmanHeightmap`; it exists to
    show the plumbing. It is *not* wired into the CLI by default — add it to
    :func:`lidarrig.surfaces.make_surface_builder` to try it.
    """

    def __init__(self, config: AppConfig) -> None:
        g = config.grid
        self._x0, self._y0 = float(g.origin[0]), float(g.origin[1])
        self._cell = float(g.cell_size)
        self._nx = max(1, int(np.ceil(g.extent[0] / self._cell)))
        self._ny = max(1, int(np.ceil(g.extent[1] / self._cell)))
        self._sum = np.zeros((self._ny, self._nx))
        self._count = np.zeros((self._ny, self._nx), dtype=np.int64)

    def add_points(self, points: np.ndarray, intensity: np.ndarray | None = None) -> None:
        if points.size == 0:
            return
        pts = np.asarray(points, dtype=np.float64)
        ix = np.floor((pts[:, 0] - self._x0) / self._cell).astype(np.int64)
        iy = np.floor((pts[:, 1] - self._y0) / self._cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self._nx) & (iy >= 0) & (iy < self._ny)
        # np.add.at handles multiple points landing in the same cell correctly.
        np.add.at(self._sum, (iy[ok], ix[ok]), pts[ok, 2])
        np.add.at(self._count, (iy[ok], ix[ok]), 1)

    def get_mesh_arrays(self) -> MeshData:
        # Left as an exercise — reuse KalmanHeightmap._build_mesh's approach:
        # compute height = sum / count where count > 0, then triangulate.
        raise NotImplementedError("Fill in mesh construction for your algorithm.")

    def reset(self) -> None:
        self._sum.fill(0.0)
        self._count.fill(0)
