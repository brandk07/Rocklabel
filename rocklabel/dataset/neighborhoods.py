"""Dataset format A: per-point neighborhood samples around voxel-grid candidate centers."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.accumulate import voxel_downsample_centroids
from .labeling import LABEL_CLEAR, LABEL_IGNORE, LABEL_ROCK, inside_arena, label_rocks

#: Channels of the per-point sample row, in the order both builders below
#: write them. Models select a subset of these by name (see train/models.py);
#: the stored tensor always holds all four, so which channels a model reads is
#: a training setting rather than a property of the dataset.
FEATURES = ("dx", "dy", "dz", "intensity")
#: The three that carry geometry. Anything that samples or groups by position
#: (PointNet++, the T-Nets, the z-rotation augmentation) needs all of them.
GEOMETRY = ("dx", "dy", "dz")


def resolve_features(features: list[str] | tuple[str, ...] | None) -> list[str]:
    """Validate a feature selection and put it back in storage order.

    Canonical order is not cosmetic: the augmentation rotates channels 0/1 of
    the stored tensor and the T-Nets slice the leading three, so 'dx, dy first'
    has to be a property of the selection rather than of how it was typed.

    Lives here rather than with the models so the plain CLI and the run-naming
    helpers can reach it without importing torch.
    """
    if features is None:
        return list(FEATURES)
    chosen = [str(f).strip() for f in features if str(f).strip()]
    unknown = [f for f in chosen if f not in FEATURES]
    if unknown:
        raise ValueError(f"unknown feature(s) {unknown}; pick from {list(FEATURES)}")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"duplicate feature in {chosen}")
    if not chosen:
        raise ValueError(f"select at least one feature from {list(FEATURES)}")
    return [f for f in FEATURES if f in chosen]


def build_neighborhood_samples(
    xyz: np.ndarray,
    intensity: np.ndarray,
    rocks: list,
    gcfg: dict,
    rng: np.random.Generator,
    arena: np.ndarray | None = None,
) -> dict | None:
    """Build format-A samples for one cropped, odom-frame frame cloud.

    Returns dict with neighborhoods [S, P, 4] f32, labels [S] i8,
    true_counts [S] i16, centers_odom [S, 3] f32 — or None if no sample
    survives filtering/subsampling.

    ``arena`` (odom-frame xy polygon, see labels.LabelSet) restricts which
    *centers* become samples. It deliberately does not restrict which points
    build a neighborhood: a rock sitting on the arena boundary still needs its
    full 0.5 m ball of context, half of which lies outside the line, and
    dropping those points would corrupt exactly the samples nearest the edge.
    """
    n_points = int(gcfg["neighborhood_points"])
    cand = voxel_downsample_centroids(xyz, gcfg["centers_voxel_m"])
    if len(cand) == 0:
        return None
    cand = cand[inside_arena(cand, arena)]
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


def build_segmentation_frame(
    xyz: np.ndarray,
    intensity: np.ndarray,
    point_labels: np.ndarray,
    base: np.ndarray,
    gcfg: dict,
    rng: np.random.Generator,
    arena: np.ndarray | None = None,
) -> dict | None:
    """Dataset format C: one whole cropped frame, labeled per point.

    Formats A and C answer the same question from opposite directions. A cuts
    the frame into thousands of overlapping 0.5 m balls and asks "is this ball's
    *center* on a rock", so the canonicalization hands the model a pre-centered,
    pre-levelled patch and every ball costs a separate forward pass. C hands
    over the whole frame once and asks for a label on every point, so one pass
    labels everything - but the model has to learn the position invariance that
    A gets for free from its preprocessing.

    Canonicalization mirrors A where it can: dx/dy/dz are relative to the robot
    base (which is exactly what the crop box is defined against, so the values
    are bounded by crop_down_m..crop_up_m), intensity is passed through, real
    points come first and padding repeats them. ``true_count`` is the number of
    real rows, so the same ``arange(N) < count`` mask works for both formats.

    Points outside ``arena`` are dropped outright here, unlike format A which
    only drops *centers*: a per-point model is scored on every point it is
    given, so an out-of-arena point is not context, it is a labeled example.
    """
    n = int(gcfg["segmentation_points"])
    if arena is not None:
        keep = inside_arena(xyz, arena)
        xyz, intensity, point_labels = xyz[keep], intensity[keep], point_labels[keep]
    k = len(xyz)
    if k < int(gcfg["segmentation_min_points"]):
        return None
    if k > n:
        idx = rng.choice(k, n, replace=False)
    else:
        idx = np.concatenate([np.arange(k), rng.choice(k, n - k, replace=True)])
    pts = xyz[idx]
    local = np.empty((n, 4), np.float32)
    local[:, 0] = pts[:, 0] - base[0]
    local[:, 1] = pts[:, 1] - base[1]
    local[:, 2] = pts[:, 2] - base[2]
    local[:, 3] = intensity[idx]
    return {
        "points": local,
        "labels": point_labels[idx].astype(np.int8),
        "true_count": np.int32(min(k, n)),
        "base_odom": np.asarray(base, np.float32),
    }


def build_inference_samples(
    xyz: np.ndarray,
    intensity: np.ndarray,
    gcfg: dict,
    rng: np.random.Generator,
    max_centers: int | None = None,
    arena: np.ndarray | None = None,
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
    # Arena first, then the cap: subsampling before the boundary test would
    # spend the budget on centers that are about to be discarded.
    cand = cand[inside_arena(cand, arena)]
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
