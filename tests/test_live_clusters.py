"""Tests for the rock-outline builder (:mod:`rocklabel.live.clusters`).

The unit under test is the step between "the model said these points are rock"
and "there are three rocks over there": grouping detections by proximity,
dropping the clumps too small to be real, and wrapping the survivors in a
polygon. Everything the two viewers draw comes from here, so this is where the
behaviour is pinned — no Open3D, no Flask, no model.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocklabel.live.clusters import (
    find_rocks,
    outline_wireframe,
    polygon_area,
)


def _blob(cx: float, cy: float, n: int = 12, spread: float = 0.08,
          seed: int = 0) -> np.ndarray:
    """A tight clump of detections around (cx, cy), like one rock's worth."""
    rng = np.random.default_rng(seed)
    xy = rng.normal(0.0, spread, size=(n, 2))
    z = rng.normal(0.0, 0.02, size=(n, 1))
    return np.column_stack([xy[:, 0] + cx, xy[:, 1] + cy, z])


def test_two_separated_clumps_become_two_rocks():
    pts = np.vstack([_blob(0.0, 0.0, seed=1), _blob(3.0, 0.0, seed=2)])
    probs = np.full(len(pts), 0.95)
    found = find_rocks(pts, probs, link_m=0.3, min_points=4)
    assert len(found.rocks) == 2
    assert found.noise_points == 0
    # Each outline sits over the clump it came from.
    xs = sorted(round(float(r.center[0])) for r in found.rocks)
    assert xs == [0, 3]


def test_a_lone_point_is_noise_not_a_rock():
    """The whole reason the noise gate exists: one stray center above the
    threshold looks exactly like a rock until you count how many there are."""
    pts = np.vstack([_blob(0.0, 0.0, n=10, seed=3), [[5.0, 5.0, 0.0]]])
    probs = np.full(len(pts), 0.99)
    found = find_rocks(pts, probs, link_m=0.3, min_points=5)
    assert len(found.rocks) == 1
    assert found.noise_points == 1 and found.noise_groups == 1
    assert "1 of 11 detections" in found.noise_note()


def test_min_points_is_the_knob_that_decides():
    pts = _blob(0.0, 0.0, n=6, seed=4)
    probs = np.full(len(pts), 0.9)
    assert len(find_rocks(pts, probs, link_m=0.3, min_points=6).rocks) == 1
    strict = find_rocks(pts, probs, link_m=0.3, min_points=7)
    assert strict.rocks == [] and strict.noise_points == 6
    assert strict.describe() == "no rocks"


def test_link_distance_decides_one_rock_or_two():
    """Two clumps 0.5 m apart: a short link keeps them separate, a long one
    merges them. This is the knob that draws the boundary between rocks."""
    pts = np.vstack([_blob(0.0, 0.0, spread=0.05, seed=5),
                     _blob(0.5, 0.0, spread=0.05, seed=6)])
    probs = np.full(len(pts), 0.95)
    assert len(find_rocks(pts, probs, link_m=0.15, min_points=4).rocks) == 2
    assert len(find_rocks(pts, probs, link_m=0.9, min_points=4).rocks) == 1


def test_outline_covers_its_detections_and_has_real_area():
    pts = _blob(1.0, -2.0, n=20, spread=0.1, seed=7)
    found = find_rocks(pts, np.full(len(pts), 0.9), link_m=0.5, min_points=4,
                       pad_m=0.05)
    rock = found.rocks[0]
    ring = rock.polygon
    assert rock.area_m2 > 0.0
    # Every detection is inside the ring (winding test, no extra dependency).
    for p in pts[:, :2]:
        rel = ring - p
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        step = np.diff(np.concatenate([ang, ang[:1]]))
        step = (step + np.pi) % (2 * np.pi) - np.pi
        assert abs(step.sum()) > 3.0, "a detection fell outside its own outline"
    # The polygon is inflated, so it covers more ground than the bare spread.
    span = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
    assert rock.area_m2 > 0.25 * float(span[0] * span[1])


def test_degenerate_clumps_still_produce_a_drawable_polygon():
    """Three detections in a perfectly straight line have no area at all — the
    padding is what keeps that from becoming an undrawable shape."""
    pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    found = find_rocks(pts, np.full(3, 0.9), link_m=0.3, min_points=3, pad_m=0.05)
    ring = found.rocks[0].polygon
    assert len(ring) >= 3
    assert polygon_area(ring) > 0.0


def test_probabilities_are_summarized_per_rock():
    pts = _blob(0.0, 0.0, n=8, seed=8)
    probs = np.linspace(0.6, 1.0, len(pts))
    rock = find_rocks(pts, probs, link_m=0.4, min_points=4).rocks[0]
    assert rock.points == 8
    assert rock.prob_max == pytest.approx(1.0)
    assert rock.prob_mean == pytest.approx(float(probs.mean()))


def test_rocks_come_back_biggest_first():
    pts = np.vstack([_blob(0.0, 0.0, n=8, spread=0.05, seed=9),
                     _blob(4.0, 0.0, n=30, spread=0.3, seed=10)])
    probs = np.full(len(pts), 0.9)
    rocks = find_rocks(pts, probs, link_m=0.4, min_points=4).rocks
    assert len(rocks) == 2
    assert rocks[0].area_m2 >= rocks[1].area_m2


def test_height_is_not_flattened_when_grouping():
    """Linking happens in 3D: a detection two metres overhead is not part of
    the rock underneath it, however well the two line up from above."""
    ground = _blob(0.0, 0.0, n=8, seed=11)
    overhead = ground.copy()
    overhead[:, 2] += 2.0
    pts = np.vstack([ground, overhead])
    found = find_rocks(pts, np.full(len(pts), 0.9), link_m=0.3, min_points=4)
    assert len(found.rocks) == 2


def test_an_empty_map_is_not_an_error():
    found = find_rocks(np.empty((0, 3)), np.empty((0,)))
    assert found.rocks == [] and found.input_points == 0
    assert found.describe() == "no detections yet"
    assert found.noise_note() == "—"


def test_the_input_cap_keeps_the_confident_detections():
    pts = np.column_stack([np.arange(50.0), np.zeros(50), np.zeros(50)])
    probs = np.linspace(0.5, 1.0, 50)
    found = find_rocks(pts, probs, link_m=0.05, min_points=1, max_points=10)
    assert found.capped is True and found.input_points == 10
    kept = sorted(r.prob_max for r in found.rocks)
    assert min(kept) == pytest.approx(float(np.sort(probs)[-10]))


def test_the_wireframe_closes_every_ring():
    pts = np.vstack([_blob(0.0, 0.0, seed=12), _blob(3.0, 3.0, seed=13)])
    found = find_rocks(pts, np.full(len(pts), 0.9), link_m=0.3, min_points=4)
    verts, lines = outline_wireframe(found.rocks, lift=0.02)
    assert len(verts) and len(lines)
    # A prism: bottom ring + top ring + one riser per vertex, and every vertex
    # is used exactly three times — anything else is an open outline.
    used = np.bincount(lines.ravel(), minlength=len(verts))
    assert set(used.tolist()) == {3}
    # The two rings straddle the detections' own height band.
    assert verts[:, 2].min() < min(r.z_min for r in found.rocks)


def test_no_rocks_means_no_wireframe():
    verts, lines = outline_wireframe([])
    assert verts.shape == (0, 3) and lines.shape == (0, 2)
