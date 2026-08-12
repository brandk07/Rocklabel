"""Tests for the quaternion math and IMU orientation compensation.

The end-to-end test is the one that matters: a fixed wall observed from a
rotating sensor must land in the same world-frame location at every yaw angle
once the tracker de-rotates the points.
"""

from __future__ import annotations

import math

import numpy as np

from rocklabel.live.motion import (
    OrientationTracker,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
    yaw_from_matrix,
)


def _yaw_quat(yaw: float) -> np.ndarray:
    """Quaternion (w, x, y, z) for a rotation of ``yaw`` about +z."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def test_quat_to_matrix_identity():
    assert np.allclose(quat_to_matrix(np.array([1.0, 0, 0, 0])), np.eye(3))


def test_quat_to_matrix_90deg_yaw_maps_x_to_y():
    rot = quat_to_matrix(_yaw_quat(math.pi / 2))
    assert np.allclose(rot @ np.array([1.0, 0, 0]), [0.0, 1.0, 0.0], atol=1e-9)


def test_quat_multiply_composes_yaws():
    q = quat_multiply(_yaw_quat(0.3), _yaw_quat(0.4))
    assert np.allclose(quat_to_matrix(q), quat_to_matrix(_yaw_quat(0.7)), atol=1e-9)


def test_yaw_from_matrix():
    assert yaw_from_matrix(quat_to_matrix(_yaw_quat(0.5))) == np.float64(0.5).item() or \
        abs(yaw_from_matrix(quat_to_matrix(_yaw_quat(0.5))) - 0.5) < 1e-9


def test_tracker_first_quat_becomes_reference():
    """The first update returns identity — the world frame starts aligned with
    the sensor frame regardless of the sensor's absolute IMU orientation."""
    tr = OrientationTracker()
    rot = tr.update(_yaw_quat(1.234))  # arbitrary nonzero startup orientation
    assert np.allclose(rot, np.eye(3), atol=1e-9)
    assert tr.has_reference
    assert abs(tr.yaw_deg) < 1e-6


def test_tracker_relative_rotation():
    tr = OrientationTracker()
    tr.update(_yaw_quat(0.2))  # reference
    rot = tr.update(_yaw_quat(0.2 + math.pi / 2))
    assert np.allclose(rot, quat_to_matrix(_yaw_quat(math.pi / 2)), atol=1e-9)
    assert abs(tr.yaw_deg - 90.0) < 1e-6


def test_rotating_sensor_sees_fixed_wall_stationary():
    """THE core property: as the sensor spins, a fixed world object observed in
    the (rotating) sensor frame must map back to the same world coordinates."""
    tr = OrientationTracker()
    # A wall segment fixed in the world at bearing 0, 3-5 m out.
    wall_world = np.column_stack(
        [np.linspace(3.0, 5.0, 50), np.zeros(50), np.full(50, -1.0)]
    )
    reconstructed = []
    for yaw in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
        # Sensor yawed by `yaw`: it sees the wall rotated by -yaw in its frame.
        rot_inv = quat_to_matrix(_yaw_quat(-yaw))
        wall_sensor = wall_world @ rot_inv.T
        out = tr.transform(wall_sensor, _yaw_quat(yaw))
        reconstructed.append(out)
    for out in reconstructed:
        assert np.allclose(out, wall_world, atol=1e-9)


def test_yaw_only_ignores_tilt():
    """With yaw_only=True, a pure-tilt (roll) change leaves points untouched."""
    roll = 0.2
    q_roll = np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0])
    tr = OrientationTracker(yaw_only=True)
    tr.update(np.array([1.0, 0, 0, 0]))  # reference = identity
    rot = tr.update(q_roll)
    assert np.allclose(rot, np.eye(3), atol=1e-9)


def test_transform_without_quat_reuses_last_rotation():
    tr = OrientationTracker()
    tr.update(_yaw_quat(0.0))
    tr.update(_yaw_quat(math.pi / 2))
    pts = np.array([[1.0, 0.0, 0.0]])
    out = tr.transform(pts, None)  # scan arrived with no fresh IMU sample
    assert np.allclose(out, [[0.0, 1.0, 0.0]], atol=1e-9)


def test_quat_normalize_rejects_zero():
    import pytest

    with pytest.raises(ValueError):
        quat_normalize(np.zeros(4))
