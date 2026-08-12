"""Per-point ground-truth from shaped labels (spheres, boxes, extruded polygons).

Labels: 1 = rock (inside the shape), -1 = ignore (inside the boundary shell,
shape grown by shell_m, but not the shape), 0 = clear. In uint8 mask outputs
ignore is encoded as 255.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

LABEL_CLEAR = 0
LABEL_ROCK = 1
LABEL_IGNORE = -1
MASK_IGNORE = 255


# -- exact shape membership ---------------------------------------------------

def points_in_polygon_xy(xy: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Vectorized even-odd point-in-polygon test. xy [N, 2], verts [M, 2]."""
    xy = np.asarray(xy, float)
    verts = np.asarray(verts, float)
    inside = np.zeros(len(xy), bool)
    x, y = xy[:, 0], xy[:, 1]
    x0, y0 = verts[:, 0], verts[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for i in range(len(verts)):
        crosses = (y0[i] > y) != (y1[i] > y)
        if not crosses.any():
            continue
        xi = (x1[i] - x0[i]) * (y[crosses] - y0[i]) / (y1[i] - y0[i]) + x0[i]
        hit = np.zeros(len(xy), bool)
        hit[crosses] = x[crosses] < xi
        inside ^= hit
    return inside


def distance_to_polygon_xy(xy: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Min 2D distance from each point to the polygon boundary. xy [N, 2]."""
    xy = np.asarray(xy, float)
    verts = np.asarray(verts, float)
    d2 = np.full(len(xy), np.inf)
    for a, b in zip(verts, np.roll(verts, -1, axis=0)):
        ab = b - a
        denom = float(ab @ ab)
        t = np.clip((xy - a) @ ab / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
        proj = a + np.outer(np.atleast_1d(t), ab)
        d2 = np.minimum(d2, ((xy - proj) ** 2).sum(axis=1))
    return np.sqrt(d2)


def points_in_rock(xyz: np.ndarray, rock, grow_m: float = 0.0) -> np.ndarray:
    """Boolean mask of points inside the rock's shape, optionally grown by
    grow_m on every side (used for the ignore shell)."""
    xyz = np.asarray(xyz, float)
    if len(xyz) == 0:
        return np.zeros(0, bool)
    shape = getattr(rock, "shape", "sphere")
    if shape == "box":
        half = np.asarray(rock.size, float) / 2.0 + grow_m
        return (np.abs(xyz - np.asarray(rock.center, float)) <= half).all(axis=1)
    if shape == "polygon":
        z0, z1 = rock.z_range
        verts = np.asarray(rock.vertices, float)
        mask = (xyz[:, 2] >= z0 - grow_m) & (xyz[:, 2] <= z1 + grow_m)
        if mask.any():
            sub = xyz[mask, :2]
            in_xy = points_in_polygon_xy(sub, verts)
            if grow_m > 0.0:
                out = ~in_xy
                if out.any():
                    in_xy[out] = distance_to_polygon_xy(sub[out], verts) <= grow_m
            mask[mask] = in_xy
        return mask
    dist = np.linalg.norm(xyz - np.asarray(rock.center, float), axis=1)
    return dist <= float(rock.radius) + grow_m


# -- labeling -----------------------------------------------------------------

def label_rocks(xyz: np.ndarray, rocks, shell_m: float) -> np.ndarray:
    """Label each point against shaped rocks. Returns int8 [N] in {0, 1, -1}.

    A KD-tree over the points is queried once per rock using its bounding
    radius (grown enough to cover the shell in the worst case), then the
    candidates get an exact shape test.
    """
    labels = np.zeros(len(xyz), np.int8)
    rocks = list(rocks)
    if len(xyz) == 0 or not rocks:
        return labels
    tree = cKDTree(xyz)
    rock_mask = np.zeros(len(xyz), bool)
    shell_mask = np.zeros(len(xyz), bool)
    # sqrt(3) covers a box grown by shell_m on all sides in the worst corner
    reach = shell_m * float(np.sqrt(3.0)) + 1e-9
    for rock in rocks:
        idx = tree.query_ball_point(np.asarray(rock.center, float),
                                    float(rock.radius) + reach)
        if not idx:
            continue
        idx = np.asarray(idx, dtype=np.intp)
        sub = xyz[idx]
        inside = points_in_rock(sub, rock)
        near = points_in_rock(sub, rock, grow_m=shell_m)
        rock_mask[idx[inside]] = True
        shell_mask[idx[near & ~inside]] = True
    labels[shell_mask] = LABEL_IGNORE
    labels[rock_mask] = LABEL_ROCK
    return labels


def label_points(xyz: np.ndarray, centers: np.ndarray, radii: np.ndarray, shell_m: float) -> np.ndarray:
    """Sphere-only convenience wrapper around label_rocks (kept for callers
    that carry plain centers/radii arrays)."""
    from types import SimpleNamespace

    rocks = [SimpleNamespace(shape="sphere", center=c, radius=r)
             for c, r in zip(np.asarray(centers, float), np.asarray(radii, float))]
    return label_rocks(xyz, rocks, shell_m)
