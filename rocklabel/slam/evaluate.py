"""How sharp is the reconstructed surface?

There is no ground-truth trajectory for these recordings, so trajectory error
cannot be measured directly. What *can* be measured is the thing that actually
matters: if the poses are right, every pass of the sensor over the same patch of
sand lands on top of the previous pass and the surface comes out thin. If the
poses drift, the same patch gets laid down at slightly different heights and the
surface comes out as a thick fuzzy slab — and a rock standing 5-10 cm proud
disappears into that fuzz.

So the score is: chop the ground into small columns, and measure how tall the
spread of points is inside each one. Real sand roughness plus sensor noise is
about 5-10 mm. Anything well above that is pose error.
"""

from __future__ import annotations

import numpy as np


def detilt_rotation(up: np.ndarray) -> np.ndarray:
    """Rotation taking ``up`` onto +z, so ground columns are truly vertical."""
    a = np.asarray(up, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, b)
    c = float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def surface_sharpness(
    points: np.ndarray,
    up: np.ndarray,
    cell: float = 0.10,
    radius: float = 8.0,
    slab: float = 1.0,
    min_points: int = 20,
) -> dict:
    """Measure ground-surface thickness in millimetres.

    Args:
        points: ``(N, 3)`` accumulated world-frame cloud.
        up: gravity up-vector in that same frame.
        cell: column footprint (m). 10 cm is about one rock across.
        radius: only score within this horizontal distance of the origin, so
            the number reflects the court rather than the far tree line.
        slab: keep points within this distance of the modal ground height.
        min_points: a column needs this many points before it is scored.

    Returns:
        dict with ``median_mm``, ``p90_mm``, ``cells`` and ``points``.
    """
    R = detilt_rotation(up)
    Q = points @ R.T
    if Q.shape[0] == 0:
        return {"median_mm": float("nan"), "p90_mm": float("nan"),
                "cells": 0, "points": 0}

    # The ground is whatever height most points sit at.
    hist, edges = np.histogram(Q[:, 2], bins=400)
    z0 = edges[int(hist.argmax())]
    Q = Q[np.abs(Q[:, 2] - z0) < slab]
    r = np.linalg.norm(Q[:, :2], axis=1)
    Q = Q[r < radius]
    if Q.shape[0] == 0:
        return {"median_mm": float("nan"), "p90_mm": float("nan"),
                "cells": 0, "points": 0}

    ij = np.floor(Q[:, :2] / cell).astype(np.int64)
    key = ij[:, 0] * 1000003 + ij[:, 1]
    order = np.argsort(key, kind="stable")
    key = key[order]
    z = Q[order, 2]
    bounds = np.flatnonzero(np.diff(key)) + 1
    counts = np.diff(np.concatenate([[0], bounds, [len(z)]]))
    groups = np.split(z, bounds)
    stds = np.array([g.std() for g, c in zip(groups, counts) if c >= min_points])
    if stds.size == 0:
        return {"median_mm": float("nan"), "p90_mm": float("nan"),
                "cells": 0, "points": int(Q.shape[0])}
    return {
        "median_mm": float(np.median(stds) * 1000.0),
        "p90_mm": float(np.percentile(stds, 90) * 1000.0),
        "cells": int(stds.size),
        "points": int(Q.shape[0]),
    }


def accumulate(frames, positions=None, quats=None, stride: int = 1) -> np.ndarray:
    """Stack every batch into one world-frame cloud.

    With ``positions``/``quats`` omitted the poses stored in the frames are
    used, which is what makes an old-vs-new comparison a one-liner.
    """
    from rocklabel.live.motion import quat_to_matrix

    out = []
    for i in range(0, len(frames), stride):
        fr = frames[i]
        if positions is None:
            R = quat_to_matrix(fr.pose_quat)
            t = fr.pose_position
        else:
            R = quat_to_matrix(quats[i])
            t = positions[i]
        out.append(fr.points.astype(np.float64) @ R.T + t)
    return np.concatenate(out) if out else np.zeros((0, 3))


# --------------------------------------------------------------------------- #
# Splitting the error in two
# --------------------------------------------------------------------------- #
# `surface_sharpness` gives one number, and one number cannot tell you whether a
# thick surface is the sensor's fault or the trajectory's. These do: they
# compare a patch of ground against *itself, seen at a different time*.
#
#   * spread inside a single look  = sensor noise + real sand roughness (a floor)
#   * spread between separate looks = pose error (what better SLAM can remove)
#
# Chasing the wrong one wastes a lot of time — ask before optimising.

def _ground_with_time(frames, positions, quats, up, radius, slab, stride):
    """Detilted ground points plus each point's timestamp."""
    from rocklabel.live.motion import quat_to_matrix

    R = detilt_rotation(up)
    t0 = frames[0].timestamp
    pts, tms = [], []
    for i in range(0, len(frames), stride):
        f = frames[i]
        rot = quat_to_matrix(f.pose_quat if positions is None else quats[i])
        pos = f.pose_position if positions is None else positions[i]
        pts.append(f.points.astype(np.float64) @ rot.T + pos)
        tms.append(np.full(len(f.points), f.timestamp - t0))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    P = np.concatenate(pts) @ R.T
    T = np.concatenate(tms)
    hist, edges = np.histogram(P[:, 2], bins=400)
    z0 = edges[int(hist.argmax())]
    keep = (np.abs(P[:, 2] - z0) < slab) & (np.linalg.norm(P[:, :2], axis=1) < radius)
    return P[keep], T[keep]


def _visit_means(P, T, cell, bin_sec, min_points):
    """Mean height of each (ground cell, time slice) pair that has enough points."""
    ij = np.floor(P[:, :2] / cell).astype(np.int64)
    ck = ij[:, 0] * 1000003 + ij[:, 1]
    tb = np.floor(T / bin_sec).astype(np.int64)
    key = ck * 100003 + tb
    order = np.argsort(key, kind="stable")
    ks, zs, cs, ts = key[order], P[order, 2], ck[order], tb[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    visits, within = {}, []
    for g in np.split(np.arange(len(zs)), bounds):
        if g.size < min_points:
            continue
        z = zs[g]
        within.append(z.var())
        visits.setdefault(int(cs[g[0]]), []).append((int(ts[g[0]]), float(z.mean())))
    return visits, within


def revisit_error(frames, positions=None, quats=None, up=None, cell=0.10,
                  bin_sec=2.0, radius=8.0, slab=1.0, min_points=6,
                  stride=2) -> dict:
    """Split ground-surface thickness into sensor noise and pose error (mm).

    Returns dict with ``within_mm`` (the floor), ``between_mm`` (what a better
    trajectory could remove) and ``cells`` (how many patches were revisited
    often enough to score).
    """
    if up is None:
        raise ValueError("revisit_error needs the gravity up-vector")
    P, T = _ground_with_time(frames, positions, quats, up, radius, slab, stride)
    if P.shape[0] == 0:
        return {"within_mm": float("nan"), "between_mm": float("nan"), "cells": 0}
    visits, within = _visit_means(P, T, cell, bin_sec, min_points)
    between = [np.var([z for _t, z in v]) for v in visits.values() if len(v) >= 3]
    return {
        "within_mm": float(np.sqrt(np.mean(within)) * 1000) if within else float("nan"),
        "between_mm": float(np.sqrt(np.mean(between)) * 1000) if between else float("nan"),
        "cells": len(between),
    }


def error_vs_gap(frames, positions=None, quats=None, up=None, cell=0.10,
                 bin_sec=1.0, radius=8.0, slab=1.0, min_points=6, stride=2,
                 buckets=((0, 2), (2, 5), (5, 10), (10, 20), (20, 35), (35, 60)),
                 min_pairs=200) -> list:
    """Pose disagreement as a function of the delay between two looks.

    This is the test that says which kind of improvement is worth building:

    * **rising with the gap** -> drift is accumulating; loop closure / global
      optimisation is the fix.
    * **flat** -> no accumulating drift left; the remaining error is per-window
      registration accuracy, and loop closure will buy nothing.

    Returns a list of ``(low_s, high_s, median_mm, pairs)``.
    """
    if up is None:
        raise ValueError("error_vs_gap needs the gravity up-vector")
    P, T = _ground_with_time(frames, positions, quats, up, radius, slab, stride)
    if P.shape[0] == 0:
        return []
    visits, _ = _visit_means(P, T, cell, bin_sec, min_points)
    gaps: dict[int, list] = {}
    for v in visits.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                gaps.setdefault(abs(v[j][0] - v[i][0]), []).append(abs(v[j][1] - v[i][1]))
    out = []
    for lo, hi in buckets:
        vals = [x for g, lst in gaps.items() if lo <= g * bin_sec < hi for x in lst]
        if len(vals) < min_pairs:
            continue
        out.append((lo, hi, float(np.median(vals) * 1000), len(vals)))
    return out
