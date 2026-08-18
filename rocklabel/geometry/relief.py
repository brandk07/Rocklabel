"""Height above the local ground, so small objects stop hiding inside big terrain.

Colouring a fused cloud by its raw height only works when the ground is flat and
the colour ramp is tight around it. On real arena data neither holds: an outdoor
pitch sags and crowns by half a metre, the sensor sees it over a 10 m span, and
a 25 cm rock ends up a few percent of the ramp — the same shade as the sand
around it. Levelling the frame fixes the *tilt*, but not the sag.

So measure each point against the ground *underneath it* rather than against a
global zero. The ground surface is estimated the way it is in classic LiDAR
terrain filtering: build a coarse height grid, then take a greyscale **opening**
of it — shrink every high spot away (a minimum filter), then grow the surface
back (a maximum filter). Anything narrower than the filter window disappears and
does not come back, while slopes, sag, ruts and berms wider than the window
survive untouched. Subtract that surface and a rock is 0.25 m tall on a scale
that only goes to 0.25 m, whatever the ground beneath it was doing.

The window size is the one number that matters: it must be comfortably wider
than the widest thing you want to *keep*, and narrower than the terrain you want
to *lose*. Set it below a rock's width and the rock becomes its own ground and
vanishes.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (distance_transform_edt, maximum_filter, median_filter,
                           minimum_filter, uniform_filter)

#: Cells with fewer points than this hold no trustworthy height.
_MIN_CELL_POINTS = 2
#: Largest grid side allowed before the cell size is coarsened to fit.
_MAX_GRID_SIDE = 1200


def ground_surface(
    xyz: np.ndarray,
    cell_m: float = 0.15,
    open_m: float = 1.5,
    smooth_m: float = 0.6,
    floor_pct: float = 25.0,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """Estimate the ground height under a cloud as a regular grid.

    Args:
        xyz: ``(N, 3)`` points. Any frame will do — the estimate is local, so it
            tolerates a tilted one (though the rest of the labeler does not).
        cell_m: grid resolution.
        open_m: width of the opening window. Objects narrower than this are
            removed from the ground and therefore stand out; terrain wider than
            it is kept and therefore does not.
        smooth_m: final averaging window, to stop cell-to-cell noise printing
            itself onto every relief value.
        floor_pct: percentile of the heights in a cell taken as its height. Not
            the minimum: one stray low return per cell would otherwise punch a
            hole through the ground everywhere.

    Returns:
        ``(grid (ni, nj), origin (x0, y0), cell_m)``, ready for
        :func:`sample_grid`.
    """
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros((1, 1)), (0.0, 0.0), cell_m

    x0 = float(pts[:, 0].min())
    y0 = float(pts[:, 1].min())
    # A run that wandered across a car park would otherwise ask for a grid of
    # tens of millions of cells. Coarsen instead of stalling; the opening
    # window is given in metres, so it keeps meaning the same thing.
    span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    cell_m = max(cell_m, span / _MAX_GRID_SIDE)
    ij = np.floor((pts[:, :2] - (x0, y0)) / cell_m).astype(np.int64)
    ni = int(ij[:, 0].max()) + 1
    nj = int(ij[:, 1].max()) + 1
    flat = ij[:, 0] * nj + ij[:, 1]

    # Per-cell low percentile, computed by sorting once: with the cell index as
    # the primary key and height as the secondary, every cell's heights land in
    # one contiguous, already-sorted run, so the percentile is an index lookup.
    order = np.lexsort((pts[:, 2], flat))
    cells = flat[order]
    z = pts[order, 2]
    starts = np.r_[0, np.flatnonzero(np.diff(cells)) + 1]
    counts = np.diff(np.r_[starts, cells.shape[0]])
    picks = starts + np.minimum(counts - 1, (counts * floor_pct / 100.0).astype(np.int64))

    grid = np.full(ni * nj, np.nan)
    enough = counts >= _MIN_CELL_POINTS
    grid[cells[starts[enough]]] = z[picks[enough]]
    grid = grid.reshape(ni, nj)

    # Empty cells (occlusion shadows, the gap under the sensor) would otherwise
    # poison the filters. Fill each from its nearest measured neighbour first.
    if np.isnan(grid).all():
        return np.zeros_like(grid), (x0, y0), cell_m
    grid = _fill_holes(grid)

    # Take the overall slope out before opening, and put it back after. Two
    # reasons. The filters see whatever is inside their window, so on a 30°
    # ramp a 1.5 m window spans nearly a metre of legitimate height and the
    # estimate becomes a fight between the slope and the objects. And at the
    # edges the window can only look inward — downhill on the uphill side —
    # so the ground is pulled low there and the whole rim reads as relief.
    # Against a subtracted plane both problems are simply gone, which also
    # means this works on a recording nobody remembered to level.
    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    design = np.column_stack([ii.ravel(), jj.ravel(), np.ones(ni * nj)])
    coef, *_ = np.linalg.lstsq(design, grid.ravel(), rcond=None)
    plane = (design @ coef).reshape(ni, nj)
    flat = grid - plane

    k = max(3, int(round(open_m / cell_m)) | 1)  # odd, so the window is centred
    ground = maximum_filter(minimum_filter(flat, size=k, mode="nearest"),
                            size=k, mode="nearest")
    s = max(1, int(round(smooth_m / cell_m)) | 1)
    if s > 1:
        ground = uniform_filter(ground, size=s, mode="nearest")
    return ground + plane, (x0, y0), cell_m


def _fill_holes(grid: np.ndarray) -> np.ndarray:
    """Replace NaN cells with their nearest measured neighbour."""
    holes = np.isnan(grid)
    if not holes.any():
        return grid
    if holes.all():
        return np.zeros_like(grid)
    _, idx = distance_transform_edt(holes, return_indices=True)
    return grid[tuple(idx)]


def _cell_index(xy: np.ndarray, origin: tuple[float, float], cell_m: float,
                shape: tuple[int, int]) -> np.ndarray:
    """Flat index of the cell each ``(N, 2)`` position falls in."""
    ij = np.floor((np.asarray(xy, dtype=np.float64) - origin) / cell_m).astype(np.int64)
    np.clip(ij[:, 0], 0, shape[0] - 1, out=ij[:, 0])
    np.clip(ij[:, 1], 0, shape[1] - 1, out=ij[:, 1])
    return ij[:, 0] * shape[1] + ij[:, 1]


def sample_grid(grid: np.ndarray, origin: tuple[float, float], cell_m: float,
                xy: np.ndarray) -> np.ndarray:
    """Read a grid at arbitrary ``(N, 2)`` positions (nearest cell)."""
    ij = np.floor((np.asarray(xy, dtype=np.float64) - origin) / cell_m).astype(np.int64)
    np.clip(ij[:, 0], 0, grid.shape[0] - 1, out=ij[:, 0])
    np.clip(ij[:, 1], 0, grid.shape[1] - 1, out=ij[:, 1])
    return grid[ij[:, 0], ij[:, 1]]


def relief_above_ground(xyz: np.ndarray, cell_m: float = 0.15, open_m: float = 1.5,
                        smooth_m: float = 0.6, floor_pct: float = 25.0) -> np.ndarray:
    """Height of every point above the ground beneath it, in metres.

    Flat ground lands near zero whatever it is doing globally; a rock lands at
    roughly its own height. Arguments are :func:`ground_surface`'s.
    """
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros(0)
    grid, origin, cell = ground_surface(pts, cell_m, open_m, smooth_m, floor_pct)
    rel = pts[:, 2] - sample_grid(grid, origin, cell, pts[:, :2])

    # Re-centre the ground on zero. The opening's shrink step takes the *lowest*
    # height in its window, so the surface it leaves sits at the bottom of the
    # measurement noise rather than in the middle of it — and ordinary ground
    # then reads as half a noise band of relief. That band widens with range
    # (fewer, more slanted returns further out), so without this correction the
    # ground appears to climb steadily as you look away from the sensor, which
    # is exactly where a relief threshold stops being usable.
    #
    # The correction is the typical amount that ground *around here* sits above
    # the opened surface, measured over the same window the opening used. A
    # rock is far smaller than that window, so it barely moves the typical
    # value and keeps its full height.
    k = max(3, int(round(open_m / cell)) | 1)
    flat = _cell_index(pts[:, :2], origin, cell, grid.shape)
    order = np.lexsort((rel, flat))
    cells, sorted_rel = flat[order], rel[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cells)) + 1]
    counts = np.diff(np.r_[starts, cells.shape[0]])
    per_cell = np.full(grid.size, np.nan)
    per_cell[cells[starts]] = sorted_rel[starts + counts // 2]  # per-cell median
    bias = median_filter(_fill_holes(per_cell.reshape(grid.shape)), size=k, mode="nearest")
    return rel - sample_grid(bias, origin, cell, pts[:, :2])
