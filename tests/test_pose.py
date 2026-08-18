"""PoseBuffer interpolation and chaining against analytic poses."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rocklabel.recording.pose import PoseBuffer, PoseUnavailable, make_matrix


def _yaw(deg):
    return Rotation.from_euler("z", deg, degrees=True).as_quat()


@pytest.fixture
def buffer_with_motion():
    pb = PoseBuffer(tolerance_s=0.15)
    # odom->base: moves +1 m in x and yaws 90 deg between t=0 and t=1.
    pb.add_dynamic("odom", "base_link", 0.0, [0, 0, 0], _yaw(0))
    pb.add_dynamic("odom", "base_link", 1.0, [1, 0, 0], _yaw(90))
    pb.add_static("base_link", "lidar_link", [1, 0, 0], _yaw(0))
    pb.finalize()
    return pb


def test_midpoint_interpolation(buffer_with_motion):
    m = buffer_with_motion.lookup("odom", "base_link", 0.5)
    expected = make_matrix([0.5, 0, 0], _yaw(45))  # lerp translation, slerp rotation
    np.testing.assert_allclose(m, expected, atol=1e-12)


def test_chained_lookup(buffer_with_motion):
    m = buffer_with_motion.lookup("odom", "lidar_link", 0.5)
    expected = make_matrix([0.5, 0, 0], _yaw(45)) @ make_matrix([1, 0, 0], _yaw(0))
    np.testing.assert_allclose(m, expected, atol=1e-12)
    # lidar origin in odom: base at (0.5,0,0), +1 m along base x rotated 45 deg.
    np.testing.assert_allclose(m[:3, 3], [0.5 + np.cos(np.pi / 4), np.sin(np.pi / 4), 0], atol=1e-12)


def test_inverse_lookup_roundtrip(buffer_with_motion):
    fwd = buffer_with_motion.lookup("odom", "lidar_link", 0.3)
    inv = buffer_with_motion.lookup("lidar_link", "odom", 0.3)
    np.testing.assert_allclose(fwd @ inv, np.eye(4), atol=1e-12)


def test_tolerance_clamps_then_raises(buffer_with_motion):
    near = buffer_with_motion.lookup("odom", "base_link", 1.1)  # within 0.15 s: clamp to t=1
    np.testing.assert_allclose(near, make_matrix([1, 0, 0], _yaw(90)), atol=1e-12)
    with pytest.raises(PoseUnavailable):
        buffer_with_motion.lookup("odom", "base_link", 1.2)
    with pytest.raises(PoseUnavailable):
        buffer_with_motion.lookup("odom", "base_link", -0.2)


def test_unsorted_and_duplicate_samples():
    pb = PoseBuffer()
    pb.add_dynamic("odom", "base_link", 2.0, [2, 0, 0], _yaw(0))
    pb.add_dynamic("odom", "base_link", 0.0, [0, 0, 0], _yaw(0))
    pb.add_dynamic("odom", "base_link", 2.0, [99, 0, 0], _yaw(0))  # duplicate stamp dropped
    pb.add_dynamic("odom", "base_link", 1.0, [1, 0, 0], _yaw(0))
    pb.finalize()
    m = pb.lookup("odom", "base_link", 1.5)
    np.testing.assert_allclose(m[:3, 3], [1.5, 0, 0], atol=1e-12)


def test_missing_chain_raises():
    pb = PoseBuffer()
    pb.add_dynamic("odom", "base_link", 0.0, [0, 0, 0], _yaw(0))
    pb.finalize()
    with pytest.raises(PoseUnavailable):
        pb.lookup("odom", "lidar_link", 0.0)
