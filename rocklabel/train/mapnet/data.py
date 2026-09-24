"""Training crops of the accumulated map, built on the fly from dumped clouds.

A sample is a square of map as it stood at some moment of a recording: every
return up to a random time, optionally thinned, then moved rigidly (rotation,
mirror), stretched (rocks of another size), and laid over a synthetic smooth
terrain (tilt and broad mounds or craters) with extra height noise. Labels are
the rock footprints put through the same transform. Augmenting the points
rather than the finished image keeps every channel honest - the histogram, the
ground estimate and the brightness are rebuilt from the moved points exactly
as they would be on the robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ...dataset.labeling import distance_to_polygon_xy, points_in_polygon_xy
from .clouds import Cloud
from .grid import CELL_M, HeightGrid, features

IGNORE = -1


@dataclass
class Aug:
    """How hard to push each augmentation. ``Aug(off=True)`` for evaluation."""
    off: bool = False
    rotate: bool = True
    scale_xy: tuple = (0.9, 1.5)
    scale_z: tuple = (0.85, 1.4)
    thin_p: float = 0.5
    thin: tuple = (0.15, 1.0)
    prefix_p: float = 0.5
    prefix: tuple = (0.05, 1.0)
    terrain_p: float = 0.7
    bump_amp_m: float = 0.12
    tilt: float = 0.05
    jitter_m: tuple = (0.0, 0.015)
    bright_drop_p: float = 0.15
    bright_gain: tuple = (0.7, 1.4)
    #: Rock transplants (GPU pipeline only): chance per crop, most per crop,
    #: and size change.
    paste_p: float = 0.0
    paste_max: int = 3
    paste_scale: tuple = (0.8, 1.25)


def rasterize_labels(labels, xform, grid: HeightGrid, observed: np.ndarray,
                     shell_m: float = 0.05, region=None) -> np.ndarray:
    """Per-cell target: 1 on a rock footprint, 0 clear, ``IGNORE`` otherwise.

    ``xform`` maps world xy to the grid's xy. Cells inside the ignore shell
    around a rock, unobserved cells, cells outside the arena, and cells outside
    ``region`` (a callable on world xy, for spatial hold-outs) are ignored.
    """
    xs, ys = grid.cell_centers()
    gx, gy = np.meshgrid(xs, ys)
    cxy = np.c_[gx.ravel(), gy.ravel()]
    y = np.zeros(len(cxy), np.int8)
    inv = xform.inverse(cxy)
    for rock in labels.rocks:
        v = xform.forward(np.asarray(rock.vertices, float)) if rock.shape == "polygon" else None
        if v is None:
            # Old sphere/box labels: a disc of the rock's radius.
            c = xform.forward(np.asarray(rock.center, float)[None, :2])[0]
            d = np.linalg.norm(cxy - c, axis=1)
            r = float(rock.radius) * xform.scale
            y[d <= r + shell_m] = np.where(d[d <= r + shell_m] <= r, 1, IGNORE)
            continue
        lo, hi = v.min(0) - shell_m, v.max(0) + shell_m
        near = ((cxy[:, 0] >= lo[0]) & (cxy[:, 0] <= hi[0])
                & (cxy[:, 1] >= lo[1]) & (cxy[:, 1] <= hi[1]))
        if not near.any():
            continue
        sub = cxy[near]
        inside = points_in_polygon_xy(sub, v)
        lab = np.where(inside, 1, 0).astype(np.int8)
        out = ~inside
        if out.any():
            dist = distance_to_polygon_xy(sub[out], v)
            lab[np.nonzero(out)[0][dist <= shell_m]] = IGNORE
        cur = y[near]
        # A rock claim wins over another rock's shell; a shell wins over clear.
        y[near] = np.where(lab == 1, 1, np.where((lab == IGNORE) & (cur != 1), IGNORE, cur))
    keep = observed.ravel().copy()
    if labels.arena is not None:
        keep &= points_in_polygon_xy(inv, labels.arena)
    if region is not None:
        keep &= region(inv)
    y[~keep] = IGNORE
    return y.reshape(grid.h, grid.w)


class Xform:
    """Similarity transform on xy: world -> crop frame (crop centre at 0)."""

    def __init__(self, center, angle: float, flip: bool, scale: float):
        self.c = np.asarray(center, float)
        ca, sa = np.cos(angle), np.sin(angle)
        m = np.array([[ca, -sa], [sa, ca]]) * scale
        if flip:
            m = m @ np.diag([1.0, -1.0])
        self.m = m
        self.minv = np.linalg.inv(m)
        self.scale = float(scale)

    def forward(self, xy):
        return (np.asarray(xy, float)[:, :2] - self.c) @ self.m.T

    def inverse(self, xy):
        return np.asarray(xy, float) @ self.minv.T + self.c


def _smooth_terrain(xy: np.ndarray, rng, aug: Aug, half: float) -> np.ndarray:
    """Tilt plus a few broad bumps and dips, metres, at crop-frame ``xy``."""
    dz = xy @ rng.uniform(-aug.tilt, aug.tilt, 2)
    for _ in range(rng.integers(1, 4)):
        c = rng.uniform(-half, half, 2)
        s = rng.uniform(0.4, 1.5)
        a = rng.uniform(-aug.bump_amp_m, aug.bump_amp_m)
        dz = dz + a * np.exp(-((xy - c) ** 2).sum(1) / (2 * s * s))
    return dz


class MapCrops(Dataset):
    """Random map crops from a set of clouds.

    ``sources`` is a list of ``(cloud, weight, region)``; ``region`` restricts
    both where crops are centred and which cells carry labels (None = all).
    """

    def __init__(self, sources, size: int = 192, length: int = 4000,
                 aug: Aug | None = None, rock_frac: float = 0.5, seed: int = 0,
                 feature_set: str = "base"):
        self.sources = sources
        w = np.array([s[1] for s in sources], float)
        self.p = w / w.sum()
        self.size = int(size)
        self.length = int(length)
        self.aug = aug or Aug()
        self.rock_frac = rock_frac
        self.seed = seed
        self.feature_set = feature_set
        self.epoch = 0

    def __len__(self):
        return self.length

    def _center(self, cloud: Cloud, region, rng, t_max):
        rocks = cloud.labels.rocks if cloud.labels is not None else []
        for _ in range(50):
            if rocks and rng.random() < self.rock_frac:
                r = rocks[rng.integers(len(rocks))]
                c = np.asarray(r.center[:2], float) + rng.uniform(-1.2, 1.2, 2)
            else:
                i = rng.integers(len(cloud.xyz))
                if cloud.t[i] > t_max:
                    i = rng.integers(max(int(np.searchsorted(cloud.t, t_max)), 1))
                c = cloud.xyz[i, :2].astype(float)
            if region is None or region(c[None])[0]:
                return c
        return c

    def sample(self, k: int, rng) -> tuple[np.ndarray, np.ndarray]:
        aug = self.aug
        ci = rng.choice(len(self.sources), p=self.p)
        cloud, _, region = self.sources[ci]
        t_max = cloud.duration
        if not aug.off and rng.random() < aug.prefix_p:
            t_max = cloud.duration * rng.uniform(*aug.prefix)
        center = self._center(cloud, region, rng, t_max)
        scale = 1.0 if aug.off else rng.uniform(*aug.scale_xy)
        angle = 0.0 if (aug.off or not aug.rotate) else rng.uniform(0, 2 * np.pi)
        flip = False if aug.off else bool(rng.random() < 0.5)
        xf = Xform(center, angle, flip, scale)
        half_out = self.size * CELL_M / 2
        idx = cloud.select(center, half_out * 1.5 / scale, t_max)
        if not aug.off and rng.random() < aug.thin_p and len(idx):
            idx = idx[rng.random(len(idx)) < rng.uniform(*aug.thin)]
        pts = cloud.xyz[idx].astype(np.float64)
        bright = None if cloud.brightness is None else cloud.brightness[idx].copy()
        xy = xf.forward(pts)
        z = pts[:, 2] - cloud.floor_z
        if not aug.off:
            z = z * rng.uniform(*aug.scale_z)
            if rng.random() < aug.terrain_p:
                z = z + _smooth_terrain(xy, rng, aug, half_out)
            z = z + rng.normal(0, rng.uniform(*aug.jitter_m), len(z))
            if bright is not None:
                if rng.random() < aug.bright_drop_p:
                    bright = None
                else:
                    bright = bright * rng.uniform(*aug.bright_gain) + rng.normal(0, 0.1)
        grid = HeightGrid(-half_out, -half_out, self.size, self.size, 0.0)
        grid.add(np.c_[xy, z], bright)
        if bright is None:
            grid.has_bright = False
        x, st = features(grid, self.feature_set)
        y = rasterize_labels(cloud.labels, xf, grid, st["valid"], region=region)
        return x, y

    def __getitem__(self, k):
        rng = np.random.default_rng([self.seed, self.epoch, k])
        x, y = self.sample(k, rng)
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))
