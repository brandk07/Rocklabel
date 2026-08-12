"""Dataset format A: per-point neighborhood samples around voxel-grid candidate centers."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .accumulate import voxel_downsample_centroids
from .labeling import LABEL_CLEAR, LABEL_IGNORE, LABEL_ROCK, label_rocks


def build_neighborhood_samples(
    xyz: np.ndarray,
    intensity: np.ndarray,
    rocks: list,
    gcfg: dict,
    rng: np.random.Generator,
) -> dict | None:
    """Build format-A samples for one cropped, odom-frame frame cloud.

    Returns dict with neighborhoods [S, P, 4] f32, labels [S] i8,
    true_counts [S] i16, centers_odom [S, 3] f32 — or None if no sample
    survives filtering/subsampling.
    """
    n_points = int(gcfg["neighborhood_points"])
    cand = voxel_downsample_centroids(xyz, gcfg["centers_voxel_m"])
    if len(cand) == 0:
        return None

    cand_labels = label_rocks(cand, rocks, gcfg["boundary_shell_m"])
    keep = cand_labels == LABEL_ROCK
    clear = cand_labels == LABEL_CLEAR
    keep |= clear & (rng.random(len(cand)) < gcfg["negative_keep_prob"])
    cand, cand_labels = cand[keep], cand_labels[keep]
    if len(cand) == 0:
        return None

    tree = cKDTree(xyz)
    neighbor_lists = tree.query_ball_point(cand, gcfg["neighborhood_radius_m"])

    neighborhoods, labels, true_counts, centers_out = [], [], [], []
    for center, label, idx in zip(cand, cand_labels, neighbor_lists):
        k = len(idx)
        if k < gcfg["min_neighbors"]:
            continue
        idx = np.asarray(idx, dtype=np.intp)
        pts = xyz[idx]
        local = np.empty((k, 4), np.float32)
        local[:, 0] = pts[:, 0] - center[0]
        local[:, 1] = pts[:, 1] - center[1]
        local[:, 2] = pts[:, 2] - pts[:, 2].min()  # local ground sits near z=0
        local[:, 3] = intensity[idx]
        if k > n_points:
            local = local[rng.choice(k, n_points, replace=False)]
        elif k < n_points:
            pad = local[rng.choice(k, n_points - k, replace=True)]
            local = np.concatenate([local, pad])
        neighborhoods.append(local)
        labels.append(label)
        true_counts.append(min(k, np.iinfo(np.int16).max))
        centers_out.append(center)

    if not neighborhoods:
        return None
    return {
        "neighborhoods": np.stack(neighborhoods).astype(np.float32),
        "labels": np.asarray(labels, np.int8),
        "true_counts": np.asarray(true_counts, np.int16),
        "centers_odom": np.stack(centers_out).astype(np.float32),
    }


def build_inference_samples(
    xyz: np.ndarray,
    intensity: np.ndarray,
    gcfg: dict,
    rng: np.random.Generator,
    max_centers: int | None = None,
) -> dict | None:
    """Label-free variant of :func:`build_neighborhood_samples` for running a
    trained model on unlabeled recordings: every candidate center is kept (no
    rock filter, no negative subsampling), but the canonicalization - dx/dy
    center-relative, dz min-relative, subsample/pad-by-repeat to
    neighborhood_points with real points first - is byte-for-byte the same
    contract the training data used.

    ``max_centers`` randomly subsamples the candidate centers when there are
    more — a hard memory/latency bound for live scoring, where the input cloud
    is an accumulated map rather than a single scan.
    """
    n_points = int(gcfg["neighborhood_points"])
    cand = voxel_downsample_centroids(xyz, gcfg["centers_voxel_m"])
    if len(cand) == 0:
        return None
    if max_centers is not None and len(cand) > max_centers:
        cand = cand[rng.choice(len(cand), int(max_centers), replace=False)]

    tree = cKDTree(xyz)
    neighbor_lists = tree.query_ball_point(
        cand, gcfg["neighborhood_radius_m"], workers=-1)

    neighborhoods, true_counts, centers_out = [], [], []
    for center, idx in zip(cand, neighbor_lists):
        k = len(idx)
        if k < gcfg["min_neighbors"]:
            continue
        # Same sampling contract as build_neighborhood_samples, but subsample
        # the *indices* before gathering point rows: the ball can hold
        # thousands of points on dense clouds and only n_points survive, so
        # gathering all of them first dominated live-scoring passes. The ball
        # z-min is still taken over the full ball, before any subsampling.
        idx = np.asarray(idx, dtype=np.intp)
        z_min = xyz[idx, 2].min()
        if k > n_points:
            idx = idx[rng.choice(k, n_points, replace=False)]
        elif k < n_points:
            idx = np.concatenate([idx, idx[rng.choice(k, n_points - k, replace=True)]])
        pts = xyz[idx]
        local = np.empty((n_points, 4), np.float32)
        local[:, 0] = pts[:, 0] - center[0]
        local[:, 1] = pts[:, 1] - center[1]
        local[:, 2] = pts[:, 2] - z_min
        local[:, 3] = intensity[idx]
        neighborhoods.append(local)
        true_counts.append(min(k, np.iinfo(np.int16).max))
        centers_out.append(center)

    if not neighborhoods:
        return None
    return {
        "neighborhoods": np.stack(neighborhoods).astype(np.float32),
        "true_counts": np.asarray(true_counts, np.int16),
        "centers_odom": np.stack(centers_out).astype(np.float32),
    }
