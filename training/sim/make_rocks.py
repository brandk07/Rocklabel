"""Procedural rock meshes for the Gazebo arena.

Writes binary STL. The sensor is a LiDAR, so surface colour and texture are
invisible to it - only the silhouette and how the shape breaks the ground
plane ever reach the model. STL carries exactly that and nothing else, and
Gazebo takes it for both the visual and the collision geometry.

Sizes are drawn from the 69 rocks labelled across the eleven volleyball
recordings (bounding radius 0.11-0.34 m, median 0.23), using rocklabel's own
definition of that radius: half the diagonal of the axis-aligned extent.
"""

from __future__ import annotations

import struct
import numpy as np
from scipy.spatial import ConvexHull

# Measured radius histogram, 2.5 cm bins from 0.100 m (labels/volleyball/*).
_BIN_LO = 0.100
_BIN_W = 0.025
_HIST = np.array([2, 3, 5, 11, 12, 16, 9, 5, 1, 5], float)

# Three form families. Real rubble is not one shape: hull_pts controls how
# faceted the result is (few directions = sharp shard, many = rounded cobble)
# and flatten is the vertical squash that makes a slab.
FORMS = {
    "cobble":  dict(hull_pts=(70, 140), lumps=(3, 5), amp=(0.10, 0.22), flatten=(0.80, 1.00)),
    "angular": dict(hull_pts=(10, 22),  lumps=(2, 4), amp=(0.18, 0.38), flatten=(0.65, 1.00)),
    "slab":    dict(hull_pts=(24, 60),  lumps=(2, 4), amp=(0.12, 0.28), flatten=(0.32, 0.55)),
}


def draw_radius(rng: np.random.Generator) -> float:
    """A bounding radius from the measured distribution, smoothed inside its bin."""
    i = rng.choice(len(_HIST), p=_HIST / _HIST.sum())
    return float(_BIN_LO + (i + rng.random()) * _BIN_W)


def _sphere_dirs(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_rock(rng: np.random.Generator, radius: float, form: str) -> np.ndarray:
    """Return an [F, 3, 3] triangle-soup rock, resting on z = 0."""
    f = FORMS[form]
    n = rng.integers(*f["hull_pts"])
    pts = _sphere_dirs(int(n), rng)

    # Low-frequency radial lumps: a few random directions each pull the surface
    # in or out. This is what stops every rock looking like the same potato.
    scale = np.ones(len(pts))
    for _ in range(int(rng.integers(*f["lumps"]))):
        d = _sphere_dirs(1, rng)[0]
        a = rng.uniform(*f["amp"]) * rng.choice([-1.0, 1.0])
        scale += a * (pts @ d)
    pts = pts * np.clip(scale, 0.35, None)[:, None]

    pts[:, 2] *= rng.uniform(*f["flatten"])
    # Mild horizontal anisotropy so footprints are not circles.
    pts[:, 0] *= rng.uniform(0.78, 1.0)
    pts[:, 1] *= rng.uniform(0.78, 1.0)
    pts = pts @ _rot_z(rng.uniform(0, 2 * np.pi))

    hull = ConvexHull(pts)
    v = hull.points

    # Match rocklabel's bounding radius: half the diagonal of the extent.
    ext = v.max(0) - v.min(0)
    v = v * (radius / (np.linalg.norm(ext) / 2.0))

    # Bury a little, then rest the remainder on the ground.
    v[:, 2] -= v[:, 2].min()
    v[:, 2] -= (v[:, 2].max() - v[:, 2].min()) * rng.uniform(0.0, 0.18)
    return v[hull.simplices]


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def write_stl(tris: np.ndarray, path: str) -> None:
    """Binary STL. Winding is not fixed up - a LiDAR ray hits the surface
    either way, and Gazebo's mesh collision does not care."""
    n = len(tris)
    with open(path, "wb") as fh:
        fh.write(b"rocklabel procedural rock".ljust(80, b"\0"))
        fh.write(struct.pack("<I", n))
        for t in tris:
            e1, e2 = t[1] - t[0], t[2] - t[0]
            nz = np.cross(e1, e2)
            ln = np.linalg.norm(nz)
            nz = nz / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
            fh.write(struct.pack("<3f", *nz))
            for p in t:
                fh.write(struct.pack("<3f", *p))
            fh.write(b"\0\0")


def scatter(rng: np.random.Generator, n: int, bounds, min_gap: float = 0.6):
    """Positions for ``n`` rocks inside an xy box, no two closer than min_gap.

    Returns fewer than ``n`` if the box cannot hold them. The gap keeps two
    rocks from merging into one candidate ball, which would label a single
    blob twice.
    """
    (x0, y0), (x1, y1) = bounds
    out: list[tuple[float, float]] = []
    for _ in range(n * 40):
        if len(out) >= n:
            break
        p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_gap ** 2 for q in out):
            out.append(p)
    return out
