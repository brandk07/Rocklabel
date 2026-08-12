"""The :class:`SurfaceBuilder` abstract base class and the :class:`MeshData`
plain-array mesh container.

Design note — *why two mesh accessors*:

The prompt requires ``get_mesh() -> o3d.geometry.TriangleMesh``, but the
sources/surfaces layers must stay free of visualization imports so a future
ROS 2 node can reuse them.  We reconcile this by making
:meth:`SurfaceBuilder.get_mesh_arrays` (returning plain NumPy) the abstract
primitive that subclasses implement, and providing a concrete
:meth:`SurfaceBuilder.get_mesh` on the base that *lazily* imports Open3D only
when called.  Import Open3D never happens at module load, so ``import
lidarrig.surfaces`` is viz-free; a headless node simply calls
``get_mesh_arrays`` and never triggers the Open3D import.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # for type checkers only; never imported at runtime here
    import open3d as o3d


@dataclass
class HeightRaster:
    """A top-down height image of a surface, in world coordinates.

    ``heights`` is ``(h, w)`` with NaN where nothing has been measured. Row 0 is
    the low-y edge, so ``heights[j, i]`` is the cell whose center sits at
    ``(x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell)`` — the same convention the
    grid itself uses, which keeps a consumer from having to guess an origin or
    a flip.
    """

    heights: np.ndarray
    x0: float
    y0: float
    cell: float

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.heights.shape[0]), int(self.heights.shape[1]))


@dataclass
class MeshData:
    """A triangle mesh as plain NumPy arrays (no Open3D dependency).

    Attributes:
        vertices: ``(V, 3)`` float array of vertex positions (m).
        triangles: ``(T, 3)`` int array of vertex indices.
        vertex_colors: optional ``(V, 3)`` float array of RGB in ``[0, 1]``.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    vertex_colors: np.ndarray | None = None

    def is_empty(self) -> bool:
        return self.vertices.shape[0] == 0 or self.triangles.shape[0] == 0

    def to_open3d(self) -> "o3d.geometry.TriangleMesh":
        """Convert to an Open3D ``TriangleMesh`` (imports Open3D lazily)."""
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            np.ascontiguousarray(self.vertices, dtype=np.float64)
        )
        mesh.triangles = o3d.utility.Vector3iVector(
            np.ascontiguousarray(self.triangles, dtype=np.int32)
        )
        if self.vertex_colors is not None and self.vertex_colors.shape[0] == self.vertices.shape[0]:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.ascontiguousarray(self.vertex_colors, dtype=np.float64)
            )
        mesh.compute_vertex_normals()
        return mesh


class SurfaceBuilder(ABC):
    """Incremental surface reconstruction from streaming point batches.

    Subclasses own *all* logic for how new points modify the existing surface.
    The app loop only ever calls :meth:`add_points`, :meth:`get_mesh` (or
    :meth:`get_mesh_arrays`), and :meth:`reset`.
    """

    @abstractmethod
    def add_points(self, points: np.ndarray, intensity: np.ndarray | None = None) -> None:
        """Incrementally integrate a ``(N, 3)`` batch into the surface.

        Implementations must handle an empty batch gracefully and must not
        assume any particular ordering or that points fall inside the surface's
        region of interest.  ``intensity`` is an optional parallel ``(N,)``
        array (reflectivity/RSSI); builders that don't use it may ignore it.
        """

    @abstractmethod
    def get_mesh_arrays(self) -> MeshData:
        """Return the current reconstructed surface as plain-NumPy mesh arrays.

        This is the visualization-free primitive; a ROS 2 node can publish from
        it without importing Open3D.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state, returning to the initial surface."""

    def get_mesh(self) -> "o3d.geometry.TriangleMesh":
        """Return the current surface as an Open3D ``TriangleMesh``.

        Provided on the base class so every builder satisfies the required
        interface; it lazily converts :meth:`get_mesh_arrays`, so importing this
        module never imports Open3D.
        """
        return self.get_mesh_arrays().to_open3d()

    def height_raster(self, max_side: int = 256) -> "HeightRaster | None":
        """A top-down height image of the surface, for an overhead view.

        Optional: builders that hold no regular grid return ``None`` and their
        callers simply draw no terrain. Implementations should crop to the
        region they actually have data for and downsample so neither side
        exceeds ``max_side`` — this is display data, sent over a socket several
        times a second, not the surface itself.
        """
        return None
