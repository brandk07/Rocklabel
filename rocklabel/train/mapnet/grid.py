"""The accumulated map: a 2.5 cm height histogram of every return seen so far.

A cell keeps how many returns landed in each 1 cm height slice between 10 cm
below and 60 cm above the measured floor (the rig's operational band), plus the
sum and maximum of their normalised brightness. That is everything the feature
channels need, it can be updated one sweep at a time in O(points), and it never
has to remember an individual point - so it is what a robot would keep live.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

CELL_M = 0.025
DZ_M = 0.01
#: Height band, relative to the measured floor. Matches ``map_eval.FLOOR_BAND``.
Z_LO, Z_HI = -0.10, 0.60
NBINS = int(round((Z_HI - Z_LO) / DZ_M))

#: Feature channels, in order. Heights are relative to a local ground estimate
#: rather than the global floor, because a crater or a sloping court moves the
#: floor by more than a rock is tall.
CHANNELS = ("observed", "density", "rel_q10", "rel_q50", "rel_q90", "rel_max",
            "spread", "ground", "bright_mean", "bright_max", "has_bright")
#: Side of the square the local ground is estimated over, metres.
GROUND_WINDOW_M = 1.5
#: Which percentile of the median-height surface counts as ground.
GROUND_PERCENTILE = 30
#: The density channel stops counting here. Past a few dozen returns a cell's
#: statistics no longer change, and a 35-minute run is ~50x denser than a
#: 45-second one; saturating keeps the two on one scale.
DENSITY_CAP = 64


class HeightGrid:
    """Height histogram over a fixed rectangle of the (levelled) world frame."""

    def __init__(self, x0: float, y0: float, width: int, height: int,
                 floor_z: float, cell_m: float = CELL_M):
        self.x0, self.y0 = float(x0), float(y0)
        self.w, self.h = int(width), int(height)
        self.floor_z = float(floor_z)
        self.cell = float(cell_m)
        self.hist = np.zeros((self.h * self.w, NBINS), np.uint32)
        self.bsum = np.zeros(self.h * self.w, np.float32)
        self.bmax = np.full(self.h * self.w, -np.inf, np.float32)
        self.has_bright = True

    @classmethod
    def around(cls, xy_lo, xy_hi, floor_z: float, cell_m: float = CELL_M,
               margin_m: float = 0.0) -> "HeightGrid":
        lo = np.asarray(xy_lo, float) - margin_m
        hi = np.asarray(xy_hi, float) + margin_m
        w = int(np.ceil((hi[0] - lo[0]) / cell_m))
        h = int(np.ceil((hi[1] - lo[1]) / cell_m))
        return cls(lo[0], lo[1], w, h, floor_z, cell_m)

    def add(self, xyz: np.ndarray, brightness: np.ndarray | None) -> None:
        """Add returns. ``brightness`` is already normalised per recording
        (see :func:`normalise_brightness`); ``None`` for a sensor without it."""
        if len(xyz) == 0:
            return
        c = np.floor((xyz[:, 0] - self.x0) / self.cell).astype(np.int64)
        r = np.floor((xyz[:, 1] - self.y0) / self.cell).astype(np.int64)
        b = np.floor((xyz[:, 2] - self.floor_z - Z_LO) / DZ_M).astype(np.int64)
        ok = (c >= 0) & (c < self.w) & (r >= 0) & (r < self.h) & (b >= 0) & (b < NBINS)
        if not ok.any():
            return
        cell = (r * self.w + c)[ok]
        flat = cell * NBINS + b[ok]
        # One sweep touches a few thousand cells of a grid holding millions:
        # scatter into those. A whole recording at once is cheaper as a count.
        sparse = len(flat) * 50 < self.hist.size
        if sparse:
            np.add.at(self.hist.reshape(-1), flat, 1)
        else:
            self.hist += np.bincount(flat, minlength=self.hist.size).reshape(
                self.hist.shape).astype(np.uint32)
        if brightness is None:
            self.has_bright = False
            return
        v = np.asarray(brightness, np.float32)[ok]
        if sparse:
            np.add.at(self.bsum, cell, v)
        else:
            self.bsum += np.bincount(cell, weights=v, minlength=self.bsum.size).astype(np.float32)
        np.maximum.at(self.bmax, cell, v)

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x0 + (np.arange(self.w) + 0.5) * self.cell
        ys = self.y0 + (np.arange(self.h) + 0.5) * self.cell
        return xs, ys


def normalise_brightness(intensity: np.ndarray) -> np.ndarray:
    """Per-recording robust scaling, so two sensors' raw scales do not matter:
    zero at the recording's median return, one per 10th-90th percentile span."""
    v = np.asarray(intensity, np.float32)
    lo, med, hi = np.percentile(v, [10, 50, 90])
    return (v - med) / max(float(hi - lo), 1e-6)


def _quantile(cum: np.ndarray, total: np.ndarray, q: float) -> np.ndarray:
    target = np.maximum(np.ceil(total * q), 1)[:, None]
    return np.argmax(cum >= target, axis=1)


def _fill_nearest(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all() or not valid.any():
        return np.where(valid, a, 0.0)
    idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                         return_indices=True)
    return a[tuple(idx)]


def local_ground(q50: np.ndarray, valid: np.ndarray, cell_m: float) -> np.ndarray:
    """A smooth ground surface under every cell, in metres above the floor.

    The median-height surface is reduced to 10 cm blocks, a low percentile is
    taken over a 1.5 m window (wide enough that a 60 cm rock cannot lift it,
    narrow enough to follow a crater wall), and the result is smoothed back up
    to the grid.
    """
    k = max(int(round(0.10 / cell_m)), 1)
    h, w = q50.shape
    hh, ww = -(-h // k), -(-w // k)
    pad = np.full((hh * k, ww * k), np.nan, np.float32)
    pad[:h, :w] = np.where(valid, q50, np.nan)
    blocks = pad.reshape(hh, k, ww, k).transpose(0, 2, 1, 3).reshape(hh, ww, k * k)
    with np.errstate(all="ignore"):
        cnt = np.isfinite(blocks).sum(-1)
        coarse = np.nanmedian(np.where(cnt[..., None] > 0, blocks, 0.0), axis=-1)
    cvalid = cnt > 0
    coarse = _fill_nearest(coarse.astype(np.float32), cvalid)
    size = max(int(round(GROUND_WINDOW_M / (k * cell_m))) | 1, 3)
    g = ndimage.percentile_filter(coarse, GROUND_PERCENTILE, size=size, mode="nearest")
    g = ndimage.uniform_filter(g, size=3, mode="nearest")
    up = np.repeat(np.repeat(g, k, axis=0), k, axis=1)[:h, :w]
    return ndimage.uniform_filter(up, size=k, mode="nearest").astype(np.float32)


#: Optional extra channels: the share of a cell's returns in each 2.5 cm slice
#: from 5 cm below to 35 cm above the local ground. A rock's vertical faces
#: spread returns over many slices; a lump of soil or a noisy patch of floor
#: piles them into one or two.
PROFILE_LO, PROFILE_HI, PROFILE_STEP = -0.05, 0.35, 0.025
PROFILE_CHANNELS = tuple(f"profile_{i}" for i in
                         range(int(round((PROFILE_HI - PROFILE_LO) / PROFILE_STEP))))


def channels(feature_set: str = "base") -> tuple[str, ...]:
    return CHANNELS + (PROFILE_CHANNELS if feature_set == "profile" else ())


def _profile(hist: np.ndarray, total: np.ndarray, ground: np.ndarray) -> np.ndarray:
    """Height profile relative to ``ground`` (metres above floor), ``[K, cells]``."""
    per = int(round(PROFILE_STEP / DZ_M))
    k_fine = len(PROFILE_CHANNELS) * per
    offset = np.round((ground + PROFILE_LO - Z_LO) / DZ_M).astype(np.int64)
    src = offset[:, None] + np.arange(k_fine)[None, :]
    inside = (src >= 0) & (src < NBINS)
    fine = np.take_along_axis(hist, np.clip(src, 0, NBINS - 1), axis=1) * inside
    coarse = fine.reshape(len(hist), -1, per).sum(-1).astype(np.float32)
    return (coarse / np.maximum(total, 1)[:, None]).T


def features(grid: HeightGrid, feature_set: str = "base") -> tuple[np.ndarray, dict]:
    """Image channels ``[C, H, W]`` (see :func:`channels`) and per-cell stats.

    The stats dict carries the absolute ``q90`` height (metres, world frame) of
    every cell so the grader can place a representative point on the surface.
    """
    hist = grid.hist
    total = hist.sum(axis=1).astype(np.float32)
    valid = total > 0
    cum = np.cumsum(hist, axis=1)
    top = np.where(valid, NBINS - 1 - np.argmax(hist[:, ::-1] > 0, axis=1), 0)

    def z(b):
        return (Z_LO + (b.astype(np.float32) + 0.5) * DZ_M)

    q10 = z(_quantile(cum, total, 0.10))
    q50 = z(_quantile(cum, total, 0.50))
    q90 = z(_quantile(cum, total, 0.90))
    zmax = z(top)
    shape = (grid.h, grid.w)
    valid2 = valid.reshape(shape)
    ground = local_ground(q50.reshape(shape), valid2, grid.cell)
    g = ground.reshape(-1)
    bmean = np.where(valid, grid.bsum / np.maximum(total, 1), 0.0)
    bmax = np.where(valid & np.isfinite(grid.bmax), grid.bmax, 0.0)
    if not grid.has_bright:
        bmean = np.zeros_like(bmean)
        bmax = np.zeros_like(bmax)
    v = valid.astype(np.float32)
    # Heights are scaled to tenths of a metre so every channel sits near unit
    # range; unobserved cells read zero everywhere and are flagged by channel 0.
    chans = [
        v,
        np.log1p(np.minimum(total, DENSITY_CAP)) / 4.0,
        (q10 - g) * 10 * v,
        (q50 - g) * 10 * v,
        (q90 - g) * 10 * v,
        (zmax - g) * 10 * v,
        (q90 - q10) * 10 * v,
        g * 10,
        np.clip(bmean, -3, 3) * v,
        np.clip(bmax, -3, 3) * v,
        np.full_like(v, 1.0 if grid.has_bright else 0.0),
    ]
    if feature_set == "profile":
        chans.extend(_profile(hist, total, g) * 4.0)
    x = np.stack(chans).reshape(len(chans), *shape).astype(np.float32)
    stats = {"valid": valid2, "q90_abs": (q90 + grid.floor_z).reshape(shape),
             "ground_abs": (ground + grid.floor_z), "count": total.reshape(shape)}
    return x, stats
