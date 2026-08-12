"""Gravity levelling of the live world frame (rocklabel.live.leveling)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rocklabel.live.config import CropConfig, LevelConfig
from rocklabel.live.filters import crop_band, crop_mask
from rocklabel.live.leveling import (
    GroundLeveler,
    fit_ground_plane,
    level_rotation_for_up,
    mount_rotation,
    roll_pitch_deg,
    rotation_between,
    tilt_deg,
)


def _tilted_room(pitch_deg: float, floor_z: float = -0.45, n: int = 6000, seed: int = 0):
    """A flat floor plus four walls, expressed in a frame tilted by ``pitch_deg``.

    Mirrors the real failure: the sensor is bolted nose-up on a mast, so the
    world frame is rotated and the floor arrives as a ramp.
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-5.0, 5.0, size=(n, 2))
    floor = np.column_stack([xy, np.full(n, floor_z)])
    # Walls: vertical planes at x = ±5, y = ±5 (the fit must ignore these).
    m = n // 2
    wall = np.concatenate([
        np.column_stack([np.full(m, 5.0), rng.uniform(-5, 5, m), rng.uniform(floor_z, 2.0, m)]),
        np.column_stack([rng.uniform(-5, 5, m), np.full(m, -5.0), rng.uniform(floor_z, 2.0, m)]),
    ])
    level = np.concatenate([floor, wall])
    rot = mount_rotation(0.0, pitch_deg)
    return level @ rot  # world = level rotated by the mount => p_world = R^T p_level


# --------------------------------------------------------------------------- #
# Rotation helpers
# --------------------------------------------------------------------------- #
def test_rotation_between_is_minimal_and_exact():
    a = np.array([0.3, -0.4, 0.86])
    b = np.array([0.0, 0.0, 1.0])
    r = rotation_between(a, b)
    assert np.allclose(r @ (a / np.linalg.norm(a)), b, atol=1e-9)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    # Minimal rotation adds no yaw: the axis lies in the horizontal plane.
    assert r[0, 1] == pytest.approx(r[1, 0], abs=1e-9)


def test_rotation_between_handles_antiparallel():
    r = rotation_between(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(r @ np.array([0.0, 0.0, -1.0]), [0.0, 0.0, 1.0], atol=1e-9)


def test_mount_rotation_matches_imu_convention():
    """--mount-pitch θ must be readable straight off the IMU's pitch."""
    rot = mount_rotation(0.0, 31.0)
    roll, pitch = roll_pitch_deg(rot)
    assert roll == pytest.approx(0.0, abs=1e-6)
    assert pitch == pytest.approx(31.0, abs=1e-6)
    assert tilt_deg(rot) == pytest.approx(31.0, abs=1e-6)
    # A sensor pitched nose-up by θ sees the ground normal at (-sin θ, 0, cos θ).
    normal = np.array([-math.sin(math.radians(31.0)), 0.0, math.cos(math.radians(31.0))])
    assert np.allclose(rot @ normal, [0.0, 0.0, 1.0], atol=1e-9)


def test_level_rotation_for_up_ignores_normal_sign():
    up = np.array([-0.5, 0.03, 0.86])
    assert np.allclose(level_rotation_for_up(up), level_rotation_for_up(-up))


# --------------------------------------------------------------------------- #
# Ground-plane fit
# --------------------------------------------------------------------------- #
def test_fit_ground_plane_finds_the_floor_not_the_walls():
    pts = _tilted_room(31.0)
    normal, d, frac = fit_ground_plane(pts, thresh=0.03, iterations=200)
    rot = level_rotation_for_up(normal)
    assert tilt_deg(rot) == pytest.approx(31.0, abs=0.5)
    assert -d == pytest.approx(-0.45, abs=0.02)  # perpendicular floor distance
    assert frac > 0.3


def test_fit_ground_plane_rejects_planes_beyond_the_tilt_gate():
    """A wall is the largest plane in many indoor scans; the gate must veto it."""
    rng = np.random.default_rng(1)
    wall = np.column_stack([
        np.full(4000, 2.0), rng.uniform(-5, 5, 4000), rng.uniform(-1, 2, 4000)])
    assert fit_ground_plane(wall, thresh=0.03, iterations=200, max_tilt_deg=50.0) is None


def test_fit_ground_plane_survives_noise_and_rocks():
    rng = np.random.default_rng(2)
    xy = rng.uniform(-4, 4, size=(5000, 2))
    z = np.full(5000, -0.4) + rng.normal(0.0, 0.01, 5000)
    rocks = rng.random(5000) < 0.15
    z[rocks] += rng.uniform(0.05, 0.3, int(rocks.sum()))  # rocks sit on top
    normal, d, _ = fit_ground_plane(np.column_stack([xy, z]), thresh=0.04, iterations=200)
    assert tilt_deg(level_rotation_for_up(normal)) < 0.5
    assert -d == pytest.approx(-0.4, abs=0.02)


# --------------------------------------------------------------------------- #
# GroundLeveler state machine
# --------------------------------------------------------------------------- #
def _feed(leveler, points, *, batches=40, dt=0.1):
    locked_at = None
    for i in range(batches):
        if leveler.observe(points @ leveler.rotation.T, i * dt) and locked_at is None:
            locked_at = i
    return locked_at


def test_leveler_ground_only_recovers_the_mount_angle():
    """No IMU (a replay, or --no-imu): the floor alone must level the frame."""
    cfg = LevelConfig(mode="ground", calib_sec=0.5, min_points=1000, plane_thresh=0.03)
    lev = GroundLeveler(cfg)
    pts = _tilted_room(31.0)
    assert _feed(lev, pts) is not None
    assert lev.locked and lev.state == GroundLeveler.LOCKED
    assert tilt_deg(lev.rotation) == pytest.approx(31.0, abs=0.5)
    assert lev.floor_z == pytest.approx(-0.45, abs=0.02)
    # The whole point: floor points come out flat (the walls, which also dip
    # below z=0, are excluded by taking a thin slab around the fitted floor).
    levelled = pts @ lev.rotation.T
    floor = levelled[np.abs(levelled[:, 2] - lev.floor_z) < 0.1]
    assert floor.shape[0] > 3000
    assert floor[:, 2].std() < 0.02


def test_leveler_imu_seed_is_immediate_and_refined_by_the_ground():
    """auto: the IMU gets you to ~2°, the ground fit gets you the rest."""
    cfg = LevelConfig(mode="auto", calib_sec=0.5, min_points=1000, plane_thresh=0.03)
    lev = GroundLeveler(cfg)
    # R(q_ref) for a sensor pitched 28.8° nose-up. The world frame *is* that
    # sensor frame, so R(q_ref) is exactly how far the world is tipped, and
    # seeding must undo it.
    lev.seed_from_imu(mount_rotation(0.0, 28.8))
    assert lev.state == GroundLeveler.COLLECTING
    assert tilt_deg(lev.rotation) == pytest.approx(28.8, abs=0.1)  # seeded instantly

    pts = _tilted_room(31.0)
    assert _feed(lev, pts) is not None
    assert tilt_deg(lev.rotation) == pytest.approx(31.0, abs=0.5)  # refined
    assert lev.floor_z == pytest.approx(-0.45, abs=0.02)


def test_leveler_imu_mode_keeps_the_imu_angle_but_measures_the_floor():
    cfg = LevelConfig(mode="imu", calib_sec=0.5, min_points=1000, plane_thresh=0.03)
    lev = GroundLeveler(cfg)
    lev.seed_from_imu(mount_rotation(0.0, 31.0))
    pts = _tilted_room(31.0)
    assert _feed(lev, pts) is not None
    assert tilt_deg(lev.rotation) == pytest.approx(31.0, abs=1e-6)  # untouched
    assert lev.floor_z == pytest.approx(-0.45, abs=0.02)


def test_leveler_manual_mode_uses_the_given_angle():
    cfg = LevelConfig(mode="manual", mount_pitch_deg=31.0, calib_sec=0.5,
                      min_points=1000, plane_thresh=0.03)
    lev = GroundLeveler(cfg)
    assert tilt_deg(lev.rotation) == pytest.approx(31.0, abs=1e-6)
    lev.seed_from_imu(mount_rotation(0.0, 5.0))  # ignored in manual mode
    assert tilt_deg(lev.rotation) == pytest.approx(31.0, abs=1e-6)
    assert _feed(lev, _tilted_room(31.0)) is not None
    assert lev.floor_z == pytest.approx(-0.45, abs=0.02)  # height still measured


def test_leveler_off_is_the_identity():
    lev = GroundLeveler(LevelConfig(mode="off"))
    assert not lev.active and lev.locked
    assert np.allclose(lev.rotation, np.eye(3))
    assert lev.floor_z is None


def test_leveler_times_out_when_no_floor_is_visible():
    """Pointed at a wall with nothing fittable: lock anyway, and say why."""
    cfg = LevelConfig(mode="ground", calib_sec=0.2, calib_timeout_sec=0.5,
                      min_points=100)
    lev = GroundLeveler(cfg)
    rng = np.random.default_rng(3)
    wall = np.column_stack([
        np.full(3000, 2.0), rng.uniform(-3, 3, 3000), rng.uniform(-1, 1, 3000)])
    assert _feed(lev, wall, batches=20, dt=0.1) is not None
    assert lev.locked
    assert np.allclose(lev.rotation, np.eye(3))
    assert "failed" in lev.status()


def test_leveler_recalibrate_reopens_collection():
    cfg = LevelConfig(mode="ground", calib_sec=0.5, min_points=1000, plane_thresh=0.03)
    lev = GroundLeveler(cfg)
    _feed(lev, _tilted_room(31.0))
    assert lev.locked
    lev.recalibrate()
    assert lev.state == GroundLeveler.COLLECTING and lev.floor_z is None
    _feed(lev, _tilted_room(12.0))
    assert tilt_deg(lev.rotation) == pytest.approx(12.0, abs=0.5)


# --------------------------------------------------------------------------- #
# Crop band + range gate
# --------------------------------------------------------------------------- #
def test_crop_band_is_floor_anchored_when_asked():
    cfg = CropConfig(z_min=-0.1, z_max=0.6, floor_relative=True)
    assert crop_band(cfg, -0.45) == pytest.approx((-0.55, 0.15))
    # No measured floor yet: stay absolute rather than guess.
    assert crop_band(cfg, None) == pytest.approx((-0.1, 0.6))
    assert crop_band(CropConfig(z_min=-1.5, z_max=-0.5), -0.45) == pytest.approx((-1.5, -0.5))


def test_crop_range_gate_follows_the_sensor():
    """The gate must mean 'near the robot', not 'near where it started'."""
    cfg = CropConfig(z_min=-5.0, z_max=5.0, range_max=2.0)
    pts = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    keep = crop_mask(pts, cfg, origin=np.array([10.0, 0.0, 0.0]))
    assert keep.tolist() == [True, True, False]
    # Anchored at the origin (the old behaviour) it keeps exactly the opposite.
    assert crop_mask(pts, cfg).tolist() == [False, False, True]


def test_crop_mask_applies_the_floor_relative_band():
    cfg = CropConfig(z_min=-0.1, z_max=0.6, floor_relative=True)
    pts = np.array([[0.0, 0.0, -0.45], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    keep = crop_mask(pts, cfg, floor_z=-0.45)
    assert keep.tolist() == [True, True, False]
