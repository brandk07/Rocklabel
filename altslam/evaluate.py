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
