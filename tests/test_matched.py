"""Tests for the shared-population comparison of a segmenter and a classifier.

The thing worth guarding here is the geometry: the two dataset formats store
positions in different frames (candidate centers in odom, segmented points
relative to the robot base), so a sign error or a missed frame alignment would
silently pair every center with the wrong points and still produce a
plausible-looking table.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rocklabel.train import matched as MT


def _build(tmp_path, *, base=np.array([10.0, -5.0, 1.0]), frame_ids=(3, 7),
           center_offset=0.0, seg_frame_ids=None):
    """A two-frame cache + predictions where the right answer is known.

    Each frame holds four points; the classifier's candidate centers sit exactly
    on two of them. The segmenter is given probability 1.0 on the rock point and
    0.0 everywhere else, so a correct matching reproduces the labels exactly.
    """
    cache = tmp_path / "cache"
    run = "runA"
    (cache / run).mkdir(parents=True)

    n_pts = 4
    local = np.zeros((len(frame_ids), n_pts, 4), np.float32)
    seg_probs = np.zeros((len(frame_ids), n_pts), np.float64)
    seg_labels = np.zeros((len(frame_ids), n_pts), np.int8)
    centers, clabels, cframe = [], [], []
    for f in range(len(frame_ids)):
        # Four well-separated points, in robot-base-relative coordinates.
        local[f, :, :3] = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                                    [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
        seg_probs[f] = [0.9, 0.1, 0.2, 0.05]   # point 0 is the confident rock
        seg_labels[f] = [1, 0, 0, 0]
        # Centers on points 0 (rock) and 3 (clear), expressed in odom.
        for p, lab in ((0, 1), (3, 0)):
            centers.append(local[f, p, :3] + base + center_offset)
            clabels.append(lab)
            cframe.append(frame_ids[f])

    np.save(cache / run / "seg_points.npy", local)
    meta = {"config_hash": "x", "generator": {}, "runs": {run: {"n": len(clabels)}}}
    (cache / "meta.json").write_text(json.dumps(meta))

    root = tmp_path / "ablate"
    for arm, payload in (
        ("clf", dict(probs=np.array([0.8, 0.2] * len(frame_ids)),
                     labels=np.asarray(clabels, np.int8),
                     frame=np.asarray(cframe, np.int64),
                     centers=np.asarray(centers, np.float64),
                     counts=np.zeros(len(clabels)))),
        ("seg", dict(probs=seg_probs, labels=seg_labels,
                     counts=np.full(len(frame_ids), n_pts, np.int64),
                     frame=np.asarray(seg_frame_ids or frame_ids, np.int64),
                     centers=np.tile(base, (len(frame_ids), 1)))),
    ):
        d = root / "suite" / arm / f"loro_{run}"
        d.mkdir(parents=True)
        np.savez_compressed(d / "predictions.npz", **payload)
    return str(cache), str(root), run


def test_centers_are_matched_to_the_points_at_their_own_position(tmp_path):
    cache, root, run = _build(tmp_path)
    m = MT.match_fold(cache, root, "suite", "clf", "seg", run, radius=0.15)
    assert m["matched"] == 4 and m["unmatched"] == 0
    # Rock centers sit on the point scored 0.9, clear centers on the one at 0.05.
    assert m["seg"][m["labels"] == 1] == pytest.approx(0.9)
    assert m["seg"][m["labels"] == 0] == pytest.approx(0.05)


def test_matching_survives_a_base_far_from_the_origin(tmp_path):
    """Segmented points are stored base-relative; centers are absolute.

    Dropping the base offset would still match *something* when the robot is
    near the origin, so the guard has to use a base that is plainly not zero.
    """
    cache, root, run = _build(tmp_path, base=np.array([120.0, -80.0, 3.0]))
    m = MT.match_fold(cache, root, "suite", "clf", "seg", run, radius=0.15)
    assert m["matched"] == 4 and m["unmatched"] == 0
    assert m["seg"][m["labels"] == 1] == pytest.approx(0.9)


def test_a_center_with_no_nearby_point_is_dropped_from_both_sides(tmp_path):
    """Centers pushed half a metre off every point must not be silently paired
    with the nearest thing available - they carry no segmenter opinion at all."""
    cache, root, run = _build(tmp_path, center_offset=0.5)
    m = MT.match_fold(cache, root, "suite", "clf", "seg", run, radius=0.15)
    assert m is None or m["matched"] == 0


def test_a_frame_the_segmenter_never_saw_is_dropped_not_misaligned(tmp_path):
    """Sparse frames produce no segmentation frame at all. Those centers must be
    counted as unmatched rather than paired with some other frame's points."""
    cache, root, run = _build(tmp_path, frame_ids=(3, 7), seg_frame_ids=(3, 99))
    m = MT.match_fold(cache, root, "suite", "clf", "seg", run, radius=0.15)
    assert m["matched"] == 2      # only frame 3 has both sides
    assert m["unmatched"] == 2


def test_frames_are_aligned_by_index_not_by_position(tmp_path):
    """Frame ids are not row numbers; the seg rows must be looked up by id."""
    cache, root, run = _build(tmp_path, frame_ids=(11, 4))
    m = MT.match_fold(cache, root, "suite", "clf", "seg", run, radius=0.15)
    assert m["matched"] == 4 and m["unmatched"] == 0
    assert m["seg"][m["labels"] == 1] == pytest.approx(0.9)


def test_missing_arm_reports_nothing_rather_than_guessing(tmp_path):
    cache, root, run = _build(tmp_path)
    assert MT.match_fold(cache, root, "suite", "clf", "absent", run) is None


def test_aggregation_choices_change_the_pooled_score(tmp_path):
    cache, root, run = _build(tmp_path)
    got = {a: MT.match_fold(cache, root, "suite", "clf", "seg", run,
                            radius=2.5, aggregation=a)["seg"][0]
           for a in MT.AGGREGATIONS}
    # Radius 2.5 m pulls in every point of the frame, so max/mean/nearest must
    # disagree - if they did not, the aggregation argument would be dead code.
    assert got["max"] == pytest.approx(0.9)
    assert got["nearest"] == pytest.approx(0.9)
    # The fourth point is 2.83 m from the center, outside the 2.5 m radius.
    assert got["mean"] == pytest.approx(np.mean([0.9, 0.1, 0.2]))


def test_default_pairs_finds_the_cross_task_contrast():
    pairs = MT.default_pairs("fullsweep")
    assert ("pointnet2-geom", "pointnet2seg-geom") in pairs
    # A suite with no segmenter has nothing to match.
    assert MT.default_pairs("reflectivity") == []
