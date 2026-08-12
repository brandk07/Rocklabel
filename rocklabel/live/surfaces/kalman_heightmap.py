"""``KalmanHeightmap`` — a 2.5D heightmap fused with per-cell 1D Kalman filters.

Each cell of a regular x/y grid stores a height estimate, its variance, and a
hit count.  An incoming batch is:

1. optionally pre-filtered for statistical outliers,
2. binned into grid cells,
3. aggregated per cell (multiple points in the same cell within one batch are
   fused correctly — never clobbered), and
4. folded into each touched cell with a vectorized 1D Kalman update.

Kalman update per cell (state = height ``h``, variance ``P``):

    predict:  P <- P + Q                         (process noise Q)
    gain:     K  = P / (P + R_eff)               (R_eff = sensor_var / m)
    update:   h <- h + K (z_bar - h)
              P <- (1 - K) P

where ``m`` measurements landed in the cell this batch and ``z_bar`` is their
mean.  Fusing ``m`` equal-variance measurements of a (locally static) height is
exactly a single update with the mean and variance ``R/m`` — so the per-cell
aggregation is not an approximation, it is the correct batch update.
"""

from __future__ import annotations

import math
import threading
import warnings

import numpy as np

from rocklabel.live.colormap import apply_colormap, normalize
from rocklabel.live.config import AppConfig
from rocklabel.live.filters import filter_keep_mask
from rocklabel.live.surfaces.base import HeightRaster, MeshData, SurfaceBuilder

#: Cap on the per-cell intensity running-mean weight, so reflectivity stays
#: responsive when the scene changes instead of freezing at the all-time mean.
_INTEN_W_CAP = 200.0


class KalmanHeightmap(SurfaceBuilder):
    """A regular 2.5D grid fused with independent per-cell Kalman filters."""

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        g = config.grid
        self._x0, self._y0 = float(g.origin[0]), float(g.origin[1])
        self._cell = float(g.cell_size)
        self._nx = max(1, int(math.ceil(g.extent[0] / self._cell)))
        self._ny = max(1, int(math.ceil(g.extent[1] / self._cell)))

        k = config.kalman
        self._R = float(k.sensor_variance)
        self._Q = float(k.process_noise)
        self._P0 = float(k.initial_variance)
        self._max_var = float(k.max_render_variance)
        self._max_edge_dz = float(k.max_edge_dz)
        self._outlier_cfg = config.outlier
        #: Mesh color source: "height" or "reflectivity" (toggled live via V).
        self._color_mode = (
            "reflectivity" if config.display.color_mode == "reflectivity" else "height"
        )

        self._lock = threading.Lock()
        self._allocate()

        # Mesh cache (rebuilt lazily, only when new data has arrived).
        self._mesh_dirty = True
        self._mesh_cache = MeshData(
            vertices=np.empty((0, 3)), triangles=np.empty((0, 3), np.int32)
        )

        # Precompute cell-center coordinates (constant for the grid's lifetime).
        xs = self._x0 + (np.arange(self._nx) + 0.5) * self._cell
        ys = self._y0 + (np.arange(self._ny) + 0.5) * self._cell
        self._cx, self._cy = np.meshgrid(xs, ys)  # (ny, nx) each

    # -- grid state ---------------------------------------------------------- #
    def _allocate(self) -> None:
        shape = (self._ny, self._nx)
        self._height = np.zeros(shape, dtype=np.float64)
        self._variance = np.full(shape, self._P0, dtype=np.float64)
        self._hits = np.zeros(shape, dtype=np.int64)
        # Per-cell reflectivity: capped-weight running mean (NaN = no data yet).
        self._inten = np.full(shape, np.nan, dtype=np.float64)
        self._inten_w = np.zeros(shape, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape as ``(ny, nx)``."""
        return (self._ny, self._nx)

    def cells_occupied(self) -> int:
        """Number of cells that have received at least one measurement."""
        return int(np.count_nonzero(self._hits))

    # -- SurfaceBuilder contract -------------------------------------------- #
    def add_points(self, points: np.ndarray, intensity: np.ndarray | None = None) -> None:
        """Integrate one ``(N, 3)`` batch into the heightmap (thread-safe).

        ``intensity`` is an optional parallel ``(N,)`` reflectivity/RSSI array;
        cells additionally accumulate a running-mean reflectivity used when the
        mesh color mode is ``"reflectivity"``.
        """
        if points.size == 0:
            return
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"expected (N, 3) points, got {pts.shape!r}")
        inten = None
        if intensity is not None and intensity.shape[0] == pts.shape[0]:
            inten = np.asarray(intensity, dtype=np.float64)

        # 1. Outlier rejection (config-toggled); same mask keeps intensity aligned.
        keep = filter_keep_mask(pts, self._outlier_cfg)
        if keep is not None:
            pts = pts[keep]
            if inten is not None:
                inten = inten[keep]
        if pts.shape[0] == 0:
            return

        # 2. Bin into grid cells; drop points outside the grid.
        ix = np.floor((pts[:, 0] - self._x0) / self._cell).astype(np.int64)
        iy = np.floor((pts[:, 1] - self._y0) / self._cell).astype(np.int64)
        in_bounds = (ix >= 0) & (ix < self._nx) & (iy >= 0) & (iy < self._ny)
        if not np.any(in_bounds):
            return
        ix = ix[in_bounds]
        iy = iy[in_bounds]
        z = pts[in_bounds, 2]
        if inten is not None:
            inten = inten[in_bounds]

        # 3. Aggregate points per cell. np.unique + np.add.at guarantee that
        #    duplicate cell indices are summed, never overwritten.
        flat = iy * self._nx + ix
        uniq, inverse, counts = np.unique(
            flat, return_inverse=True, return_counts=True
        )
        sums = np.zeros(uniq.shape[0], dtype=np.float64)
        np.add.at(sums, inverse, z)
        z_bar = sums / counts
        m = counts.astype(np.float64)

        uy, ux = np.divmod(uniq, self._nx)

        # 3b. Aggregate reflectivity per cell over the finite samples only
        #     (sources without RSSI mark points NaN — they must not poison means).
        iy_i = ix_i = i_bar = m_i = None
        if inten is not None:
            finite = np.isfinite(inten)
            if np.any(finite):
                uniq_i, inv_i, cnt_i = np.unique(
                    flat[finite], return_inverse=True, return_counts=True
                )
                isums = np.zeros(uniq_i.shape[0], dtype=np.float64)
                np.add.at(isums, inv_i, inten[finite])
                i_bar = isums / cnt_i
                m_i = cnt_i.astype(np.float64)
                iy_i, ix_i = np.divmod(uniq_i, self._nx)

        with self._lock:
            self._kalman_update(uy, ux, z_bar, m)
            if i_bar is not None:
                self._intensity_update(iy_i, ix_i, i_bar, m_i)
            self._mesh_dirty = True

    def _kalman_update(
        self, uy: np.ndarray, ux: np.ndarray, z_bar: np.ndarray, m: np.ndarray
    ) -> None:
        """Vectorized 1D Kalman update on the given (already-unique) cells."""
        h = self._height[uy, ux]
        P = self._variance[uy, ux] + self._Q  # predict
        R_eff = self._R / m  # fusing m measurements => variance R/m
        K = P / (P + R_eff)
        self._height[uy, ux] = h + K * (z_bar - h)
        self._variance[uy, ux] = (1.0 - K) * P
        self._hits[uy, ux] += m.astype(np.int64)

    def _intensity_update(
        self, uy: np.ndarray, ux: np.ndarray, i_bar: np.ndarray, m: np.ndarray
    ) -> None:
        """Capped-weight running mean of per-cell reflectivity (NaN = unseen)."""
        w_old = self._inten_w[uy, ux]
        old = np.where(w_old > 0.0, self._inten[uy, ux], 0.0)
        self._inten[uy, ux] = (old * w_old + i_bar * m) / (w_old + m)
        self._inten_w[uy, ux] = np.minimum(w_old + m, _INTEN_W_CAP)

    def set_color_mode(self, mode: str) -> None:
        """Switch mesh vertex coloring: "height" or "reflectivity" (thread-safe)."""
        mode = "reflectivity" if mode == "reflectivity" else "height"
        with self._lock:
            if mode != self._color_mode:
                self._color_mode = mode
                self._mesh_dirty = True  # recolor on the next mesh fetch

    def get_mesh_arrays(self) -> MeshData:
        """Return the cached surface mesh, rebuilding only if new data arrived."""
        with self._lock:
            if not self._mesh_dirty:
                return self._mesh_cache
            # Snapshot under the lock, build the mesh outside heavy contention.
            height = self._height.copy()
            variance = self._variance.copy()
            hits = self._hits.copy()
            inten = self._inten.copy()
            self._mesh_dirty = False
        self._mesh_cache = self._build_mesh(height, variance, hits, inten)
        return self._mesh_cache

    def height_raster(self, max_side: int = 256) -> HeightRaster | None:
        """Top-down height image of the fused surface (see the base class).

        Uses the same validity rule as the 3D mesh — measured, and settled
        enough that its variance is under the render cap — so the overhead view
        and the 3D window agree about what exists.

        Cropped to the occupied bounding box, because the grid is 20 m square
        and a room fills a corner of it; then block-reduced by an integer stride
        so neither side exceeds ``max_side``. Reduction takes the *maximum* of
        each block, not the mean: on a floor with rocks on it, averaging a rock
        with the floor around it is how a rock disappears at low zoom.
        """
        with self._lock:
            height = self._height.copy()
            variance = self._variance.copy()
            hits = self._hits.copy()

        valid = (hits > 0) & (variance <= self._max_var)
        if not np.any(valid):
            return None

        rows = np.flatnonzero(valid.any(axis=1))
        cols = np.flatnonzero(valid.any(axis=0))
        j0, j1 = int(rows[0]), int(rows[-1]) + 1
        i0, i1 = int(cols[0]), int(cols[-1]) + 1

        z = np.where(valid[j0:j1, i0:i1], height[j0:j1, i0:i1], np.nan)
        cell = self._cell
        stride = max(1, int(np.ceil(max(z.shape) / float(max(1, max_side)))))
        if stride > 1:
            ny, nx = z.shape
            # Pad to a whole number of blocks with NaN, which np.nanmax ignores.
            pad_y = (-ny) % stride
            pad_x = (-nx) % stride
            if pad_y or pad_x:
                z = np.pad(z, ((0, pad_y), (0, pad_x)), constant_values=np.nan)
            blocks = z.reshape(z.shape[0] // stride, stride, z.shape[1] // stride, stride)
            with warnings.catch_warnings():
                # An all-NaN block is empty ground, not a problem worth a warning.
                warnings.simplefilter("ignore", RuntimeWarning)
                z = np.nanmax(blocks, axis=(1, 3))
            cell = self._cell * stride

        return HeightRaster(
            heights=z.astype(np.float32),
            x0=self._x0 + i0 * self._cell,
            y0=self._y0 + j0 * self._cell,
            cell=cell,
        )

    def reset(self) -> None:
        """Clear all fused state back to the initial (empty) surface."""
        with self._lock:
            self._allocate()
            self._mesh_dirty = True
            self._mesh_cache = MeshData(
                vertices=np.empty((0, 3)), triangles=np.empty((0, 3), np.int32)
            )

    # -- mesh construction --------------------------------------------------- #
    def _build_mesh(
        self,
        height: np.ndarray,
        variance: np.ndarray,
        hits: np.ndarray,
        inten: np.ndarray,
    ) -> MeshData:
        """Triangulate valid cells (2 triangles per quad); color per the active
        mode (height colormap, or per-cell reflectivity when in RSSI mode)."""
        valid = (hits > 0) & (variance <= self._max_var)  # (ny, nx)
        if not np.any(valid):
            return MeshData(
                vertices=np.empty((0, 3)), triangles=np.empty((0, 3), np.int32)
            )

        ny, nx = self._ny, self._nx
        # Full lattice of vertices at cell centers (compacted below).
        verts_full = np.stack(
            [self._cx.ravel(), self._cy.ravel(), height.ravel()], axis=1
        )  # (ny*nx, 3)

        # A quad (ix,iy) is meshed only if all four corner cells are valid.
        v00 = valid[:-1, :-1]
        v10 = valid[:-1, 1:]
        v01 = valid[1:, :-1]
        v11 = valid[1:, 1:]
        quad_ok = v00 & v10 & v01 & v11  # (ny-1, nx-1)

        # Edge guard: refuse to span quads whose corner heights differ by more
        # than max_edge_dz — a heightmap cannot represent vertical structure
        # (walls, people), and triangulating across it renders as giant spikes.
        if self._max_edge_dz > 0.0:
            h00 = height[:-1, :-1]
            h10 = height[:-1, 1:]
            h01 = height[1:, :-1]
            h11 = height[1:, 1:]
            hmax = np.maximum(np.maximum(h00, h10), np.maximum(h01, h11))
            hmin = np.minimum(np.minimum(h00, h10), np.minimum(h01, h11))
            quad_ok &= (hmax - hmin) <= self._max_edge_dz
        if not np.any(quad_ok):
            return MeshData(
                vertices=np.empty((0, 3)), triangles=np.empty((0, 3), np.int32)
            )

        qy, qx = np.nonzero(quad_ok)
        # Flat vertex indices of the four corners of each valid quad.
        i00 = qy * nx + qx
        i10 = qy * nx + (qx + 1)
        i01 = (qy + 1) * nx + qx
        i11 = (qy + 1) * nx + (qx + 1)
        # Two triangles per quad (consistent winding for outward normals).
        tris = np.empty((qy.shape[0] * 2, 3), dtype=np.int64)
        tris[0::2] = np.stack([i00, i10, i11], axis=1)
        tris[1::2] = np.stack([i00, i11, i01], axis=1)

        # Compact to only the vertices actually referenced by triangles.
        used = np.unique(tris)
        remap = np.full(verts_full.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.shape[0])
        vertices = verts_full[used]
        triangles = remap[tris].astype(np.int32)

        if self._color_mode == "reflectivity":
            colors = self._reflectivity_colors(inten.ravel()[used])
        else:
            colors = apply_colormap(
                normalize(vertices[:, 2]), self._cfg.display.colormap
            )
        return MeshData(vertices=vertices, triangles=triangles, vertex_colors=colors)

    def _reflectivity_colors(self, vals: np.ndarray) -> np.ndarray:
        """Fixed full-scale reflectivity colors (matches the point clouds:
        RSSI / 65535, so identical materials keep identical colors across
        frames and retroreflectors stay distinct; cells with no RSSI data
        render mid-gray)."""
        finite = np.isfinite(vals)
        v = np.full(vals.shape, 0.5)
        if np.any(finite):
            v[finite] = np.clip(vals[finite] / 65535.0, 0.0, 1.0)
        return apply_colormap(v, self._cfg.display.reflectivity_colormap)
