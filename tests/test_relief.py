"""Height above the local ground (rocklabel.geometry.relief) and the labeler's use of it.

The scene these tests use is the one that broke labelling in the field: ground
that is neither flat nor level — a slope with a broad dip in it — plus rocks
small enough that plain height colouring buries them inside the terrain.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocklabel.geometry.relief import ground_surface, relief_above_ground, sample_grid
from rocklabel.gui.viewer import _LabelerApp, relief_colors


def _terrain(rocks=((1.0, 1.0, 0.25), (-2.0, 2.5, 0.30)), tilt=0.35, seed=0,
             n=120_000, sag=0.4):
    """Sloping, sagging ground with a little surface texture, plus rocks.

    ``tilt`` is metres of rise per metre travelled; ``sag`` is the depth of a
    broad bowl. Both are far larger than the rocks, which is the whole problem.
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-5.0, 5.0, size=(n, 2))
    r2 = (xy ** 2).sum(axis=1)
    z = tilt * xy[:, 0] - sag * np.exp(-r2 / 8.0)
    z += rng.normal(0.0, 0.01, n)                       # sensor noise
    z += 0.02 * np.sin(xy[:, 0] * 6.0) * np.cos(xy[:, 1] * 5.0)  # footprints
    is_rock = np.zeros(n, bool)
    for cx, cy, h in rocks:
        d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
        on = d < 0.18
        z[on] += h
        is_rock |= on
    return np.column_stack([xy, z]), is_rock


def test_relief_finds_rocks_that_height_alone_cannot():
    pts, is_rock = _terrain()
    rel = relief_above_ground(pts)

    # Ground reads as ground wherever it is and whatever it is doing.
    assert np.percentile(rel[~is_rock], 99) < 0.09
    # Rocks read as their own height.
    assert np.median(rel[is_rock]) > 0.20

    # And the point of the exercise: raw height cannot separate them, because
    # the terrain moves through metres while a rock is a quarter of one.
    z = pts[:, 2]
    assert z.max() - z.min() > 2.0
    assert np.percentile(z[~is_rock], 90) > np.median(z[is_rock])


def test_relief_clip_keeps_the_rocks_and_drops_the_ground():
    """The labelling workflow: hide everything under ~10 cm, label what's left."""
    pts, is_rock = _terrain()
    rel = relief_above_ground(pts)
    kept = rel >= 0.10
    assert kept[is_rock].mean() > 0.9          # nearly every rock point survives
    assert kept[~is_rock].mean() < 0.005       # nearly all the ground is gone


def test_ground_surface_keeps_terrain_wider_than_the_window():
    """A dip metres across is terrain, not an object — it must survive."""
    pts, _ = _terrain(rocks=(), sag=0.4)
    grid, origin, cell = ground_surface(pts)
    at_centre = sample_grid(grid, origin, cell, np.array([[0.0, 0.0]]))[0]
    at_edge = sample_grid(grid, origin, cell, np.array([[4.0, 0.0]]))[0]
    assert at_edge - at_centre == pytest.approx(0.4 + 0.35 * 4.0, abs=0.12)


def test_ground_surface_swallows_objects_narrower_than_the_window():
    """Shrink the window under a rock's width and the rock becomes its own
    ground — the failure mode worth knowing about, pinned here."""
    pts, is_rock = _terrain()
    swallowed = relief_above_ground(pts, cell_m=0.05, open_m=0.10, smooth_m=0.05)
    assert np.median(swallowed[is_rock]) < 0.05


def test_relief_handles_empty_and_single_cell_clouds():
    assert relief_above_ground(np.zeros((0, 3))).shape == (0,)
    assert relief_above_ground(np.zeros((5, 3))).shape == (5,)


def test_relief_colors_are_fixed_scale_not_percentile():
    """10 cm must be the same colour whatever else is on screen."""
    alone = relief_colors(np.array([0.10]), high_m=0.25)
    crowded = relief_colors(np.array([0.10, 0.0, 0.24, 0.01]), high_m=0.25)[:1]
    assert np.allclose(alone, crowded)
    # Past the top of the ramp saturates rather than re-scaling everything.
    assert np.allclose(relief_colors(np.array([0.25]), 0.25),
                       relief_colors(np.array([9.9]), 0.25))


def _visibility_host(pts, relief, relief_min=0.0, z_min=-99.0, z_max=99.0):
    from types import SimpleNamespace

    return SimpleNamespace(xyz=pts, relief=relief, relief_min=relief_min,
                           z_min=z_min, z_max=z_max)


def test_labeler_visible_mask_applies_the_relief_clip():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    rel = np.array([0.01, 0.15, 0.30])
    host = _visibility_host(pts, rel, relief_min=0.10)
    assert _LabelerApp._visible_mask(host).tolist() == [False, True, True]
    # Off by default: nothing is hidden until you ask for it.
    host.relief_min = 0.0
    assert _LabelerApp._visible_mask(host).tolist() == [True, True, True]


def test_labeler_relief_clip_stacks_with_the_z_clip():
    pts = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 0.0]])
    host = _visibility_host(pts, np.array([0.30, 0.30]), relief_min=0.10, z_max=1.0)
    assert _LabelerApp._visible_mask(host).tolist() == [False, True]


def test_relief_does_not_creep_upward_with_range():
    """The complaint that made this correction necessary.

    The opening's shrink step lands on the *bottom* of the measurement noise,
    so ordinary ground reads as half a noise band of relief — and the band
    widens with range as returns get sparser and more slanted. Left alone, the
    ground appears to climb steadily as you look away from the sensor and any
    single relief threshold stops working somewhere in the middle of the map.
    """
    rng = np.random.default_rng(1)
    n = 200_000
    xy = rng.uniform(-8.0, 8.0, size=(n, 2))
    r = np.hypot(xy[:, 0], xy[:, 1])
    # Flat ground, but noise that grows with range, exactly as a LiDAR does.
    z = rng.normal(0.0, 0.004 + 0.010 * r, n)
    rel = relief_above_ground(np.column_stack([xy, z]))

    near = rel[(r > 1.0) & (r < 3.0)]
    far = rel[(r > 6.0) & (r < 8.0)]
    assert abs(np.median(near)) < 0.02
    assert abs(np.median(far)) < 0.02
    # And a single threshold means the same thing at both ends of the map.
    assert (far >= 0.10).mean() - (near >= 0.10).mean() < 0.15


def test_relief_survives_the_correction_on_a_rock():
    """The correction is measured over the opening window, which a rock is far
    smaller than — so it must not eat the rock it is there to reveal."""
    pts, is_rock = _terrain()
    rel = relief_above_ground(pts)
    assert np.median(rel[is_rock]) > 0.20
