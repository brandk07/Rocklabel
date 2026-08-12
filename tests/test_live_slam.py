"""Tests for the scan-to-map odometry: voxel hash map, ICP registration, and
the end-to-end "moving sensor keeps the room fixed" property."""

from __future__ import annotations

import math

import numpy as np

from rocklabel.live.config import MotionConfig, SlamConfig
from rocklabel.live.slam import SlamTracker, VoxelHashMap, register, voxel_downsample


def _room_cloud(n_per_surface: int = 1500, seed: int = 0) -> np.ndarray:
    """A synthetic room: floor at z=-1, two perpendicular walls (x=5, y=3).

    Three mutually orthogonal planes fully constrain a rigid transform.
    """
    rng = np.random.default_rng(seed)
    floor = np.column_stack(
        [rng.uniform(0, 5, n_per_surface), rng.uniform(-3, 3, n_per_surface),
         np.full(n_per_surface, -1.0)]
    )
    wall_x = np.column_stack(
        [np.full(n_per_surface, 5.0), rng.uniform(-3, 3, n_per_surface),
         rng.uniform(-1, 1.5, n_per_surface)]
    )
    wall_y = np.column_stack(
        [rng.uniform(0, 5, n_per_surface), np.full(n_per_surface, 3.0),
         rng.uniform(-1, 1.5, n_per_surface)]
    )
    return np.concatenate([floor, wall_x, wall_y])


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# VoxelHashMap
# --------------------------------------------------------------------------- #
def test_voxel_downsample_reduces_and_centers():
    pts = np.array([[0.01, 0.01, 0.0], [0.09, 0.09, 0.0], [1.0, 1.0, 1.0]])
    out = voxel_downsample(pts, 0.2)
    assert out.shape == (2, 3)
    # the two near points collapse to their mean
    assert np.allclose(sorted(out[:, 0]), [0.05, 1.0], atol=1e-9)


def test_voxel_map_insert_and_query():
    vmap = VoxelHashMap(0.2)
    pts = _room_cloud(400)
    vmap.insert(pts)
    assert vmap.size > 100

    # Querying slightly perturbed copies of inserted points matches nearby.
    q = pts[:200] + 0.03
    valid, targets = vmap.query(q, max_dist=0.3)
    assert valid.mean() > 0.9
    d = np.linalg.norm(q[valid] - targets[valid], axis=1)
    assert d.max() <= 0.3 + 1e-9

    # Points far from anything must not match.
    far = np.full((10, 3), 100.0)
    valid_far, _ = vmap.query(far, max_dist=0.3)
    assert not valid_far.any()


def test_voxel_map_empty_query():
    vmap = VoxelHashMap(0.2)
    valid, _ = vmap.query(np.zeros((5, 3)), 0.3)
    assert not valid.any()


# --------------------------------------------------------------------------- #
# ICP registration
# --------------------------------------------------------------------------- #
def test_register_recovers_known_transform():
    """A scan displaced by a known (yaw, translation) must be recovered."""
    room = _room_cloud()
    vmap = VoxelHashMap(0.15)
    vmap.insert(room)

    true_yaw = math.radians(2.5)
    true_t = np.array([0.12, -0.08, 0.04])
    # The scan arrives at a *wrong* predicted pose: displaced by the inverse.
    d_r_true = _rz(true_yaw)
    scan = voxel_downsample(_room_cloud(seed=1), 0.15)
    displaced = (scan - true_t) @ d_r_true  # = R^T (x - t): inverse transform

    d_r, d_t, ratio, ok = register(
        displaced, vmap, iterations=8, dist_start=0.45, dist_end=0.15,
        rotation_mode="yaw", min_matches=100,
    )
    assert ok and ratio > 0.5
    # Recovered correction must undo the displacement.
    recovered = displaced @ d_r.T + d_t
    err = np.abs(recovered - scan).mean()
    assert err < 0.03, f"mean alignment error {err:.3f} m"
    yaw_rec = math.atan2(d_r[1, 0], d_r[0, 0])
    assert abs(yaw_rec - true_yaw) < math.radians(0.5)


def test_register_translation_only_mode():
    room = _room_cloud()
    vmap = VoxelHashMap(0.15)
    vmap.insert(room)
    true_t = np.array([0.15, 0.1, -0.05])
    scan = voxel_downsample(_room_cloud(seed=2), 0.15) - true_t
    d_r, d_t, ratio, ok = register(
        scan, vmap, 8, 0.45, 0.15, rotation_mode="none", min_matches=100
    )
    assert ok
    assert np.allclose(d_r, np.eye(3))
    assert np.allclose(d_t, true_t, atol=0.03)


def test_register_unmapped_area_reports_failure():
    vmap = VoxelHashMap(0.15)
    vmap.insert(_room_cloud())
    scan = _room_cloud(seed=3) + 50.0  # nowhere near the map
    _, _, ratio, ok = register(scan, vmap, 5, 0.45, 0.15, "yaw", 100)
    assert not ok
    assert ratio < 0.05


# --------------------------------------------------------------------------- #
# End-to-end: a translating sensor keeps the room fixed in the world frame
# --------------------------------------------------------------------------- #
def test_moving_sensor_room_stays_fixed():
    """Walk the sensor ~1 m through a synthetic room; wall/floor points must
    keep landing on the same world-frame planes (the whole point of SLAM).

    Mirrors real usage: the map anchors during a brief stationary start, then
    the sensor moves at a walking pace.
    """
    cfg = SlamConfig(window_sec=0.1, voxel_size=0.15, max_corr_dist=0.45)
    tracker = SlamTracker(cfg, MotionConfig())
    rng = np.random.default_rng(4)
    room = _room_cloud(4000, seed=5)
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

    t_now = 1000.0
    pos = np.zeros(3)
    step = np.array([0.02, 0.01, 0.0])  # ~2.2 cm/batch ≈ 1.1 m/s walking pace
    outputs = []
    for i in range(65):
        # Sensor at `pos`, unrotated: sensor-frame cloud = world - pos.
        sample = room[rng.choice(room.shape[0], 900, replace=False)]
        sensor_pts = sample - pos
        world_out = tracker.process(sensor_pts, identity_quat, timestamp=t_now)
        outputs.append(world_out)
        if i >= 15:  # stationary anchor phase, then start walking
            pos = pos + step
        t_now += 0.021  # ~5 batches per 0.1 s window

    # Pose must track the true position (allow a few cm of odometry error).
    final_true = pos - step  # position of the last processed batch
    pose_err = np.linalg.norm(tracker.position - final_true)
    assert pose_err < 0.10, f"pose error {pose_err:.3f} m (true {final_true})"

    # Late-run points must still land on the room's planes: check the x=5 wall
    # (mean offset = world-frame consistency, std = wall crispness). Exclude
    # floor (z=-1) and the y=3 wall so only true x-wall points are measured.
    late = np.concatenate(outputs[-5:])
    on_wall = (
        (np.abs(late[:, 0] - 5.0) < 0.3) & (late[:, 1] < 2.7) & (late[:, 2] > -0.85)
    )
    wall_pts = late[on_wall]
    assert wall_pts.shape[0] > 50
    assert np.abs(wall_pts[:, 0] - 5.0).mean() < 0.05
    assert wall_pts[:, 0].std() < 0.05

    # Sanity: without correction the drift would be ~the full path length.
    assert tracker.windows_registered >= 5


def test_slam_reset_clears_map_and_pose():
    cfg = SlamConfig()
    tracker = SlamTracker(cfg, MotionConfig())
    q = np.array([1.0, 0.0, 0.0, 0.0])
    tracker.process(_room_cloud(500), q, timestamp=10.0)
    tracker.process(_room_cloud(500), q, timestamp=10.2)  # closes a window
    assert tracker.map.size > 0
    tracker.reset()
    assert tracker.map.size == 0
    assert np.allclose(tracker.position, 0.0)
