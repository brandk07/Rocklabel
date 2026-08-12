"""IMU-based orientation tracking (viz-free, pure NumPy).

The multiScan reports a fused orientation quaternion with every IMU telegram.
When the sensor itself is being rotated (e.g. sitting on a chair that you spin
to sweep the room), points must be rotated *out* of the moving sensor frame and
into a fixed world frame before fusion — otherwise the room smears into rings
around the sensor.

:class:`OrientationTracker` captures the first valid quaternion as the
reference "world" orientation and, for each subsequent quaternion, produces the
rotation matrix that maps current-sensor-frame points into that fixed frame:

    p_world = R(q_ref)^T  R(q_now)  p_sensor

so the world frame coincides with the sensor frame at startup — the same frame
the grid, crop band, and camera were already using.

Quaternions are ``(w, x, y, z)``, matching the Compact-format IMU telegram.
"""

from __future__ import annotations

import math

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return the unit quaternion (w, x, y, z); raises on near-zero norm."""
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        raise ValueError("cannot normalize a zero quaternion")
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quaternions) of (w, x, y, z)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b, both (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) of the unit quaternion (w, x, y, z)."""
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_from_matrix(rot: np.ndarray) -> float:
    """Yaw (rotation about +z, radians) of a rotation matrix."""
    return math.atan2(rot[1, 0], rot[0, 0])


def matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix (Shepperd's method)."""
    m = np.asarray(rot, dtype=np.float64)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        )
    elif m[1, 1] >= m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array(
            [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        )
    return quat_normalize(q)


def yaw_matrix(yaw: float) -> np.ndarray:
    """Pure-yaw rotation matrix about +z."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class OrientationTracker:
    """Turns a stream of IMU orientation quaternions into world-frame rotations.

    Args:
        yaw_only: if True, only the yaw component of the relative rotation is
            applied (useful when the sensor pivots about z, e.g. on a chair,
            and you want to ignore small tilt/wobble noise).
    """

    def __init__(self, yaw_only: bool = False) -> None:
        self._yaw_only = yaw_only
        self._ref_conj: np.ndarray | None = None  # conjugate of the reference quat
        self._last_rot = np.eye(3)
        self._last_yaw = 0.0

    @property
    def has_reference(self) -> bool:
        """True once at least one valid quaternion has been received."""
        return self._ref_conj is not None

    @property
    def yaw_deg(self) -> float:
        """Yaw of the last update relative to the reference, in degrees."""
        return math.degrees(self._last_yaw)

    @property
    def rotation(self) -> np.ndarray:
        """Last sensor→world rotation matrix (identity before any sample)."""
        return self._last_rot

    @property
    def reference_rotation(self) -> np.ndarray | None:
        """``R(q_ref)`` — the gravity-referenced rotation of the quaternion that
        defined the world frame, or None before the first sample.

        The multiScan's quaternion carries absolute roll/pitch (gravity) with
        arbitrary yaw, so this is what tells you how far the world frame is
        tipped away from level. See :mod:`rocklabel.live.leveling`.
        """
        if self._ref_conj is None:
            return None
        return quat_to_matrix(quat_conjugate(self._ref_conj))

    def reset(self) -> None:
        """Drop the reference; the next quaternion re-zeros the world frame."""
        self._ref_conj = None
        self._last_rot = np.eye(3)
        self._last_yaw = 0.0

    def update(self, quat_wxyz: np.ndarray) -> np.ndarray:
        """Feed the latest orientation; return the sensor→world rotation (3, 3).

        The first valid quaternion becomes the reference, so the returned
        matrix is identity at startup and thereafter undoes any rotation the
        sensor has undergone since.
        """
        q = quat_normalize(quat_wxyz)
        if self._ref_conj is None:
            self._ref_conj = quat_conjugate(q)
        rel = quat_multiply(self._ref_conj, q)
        rot = quat_to_matrix(rel)
        self._last_yaw = yaw_from_matrix(rot)
        if self._yaw_only:
            rot = yaw_matrix(self._last_yaw)
        self._last_rot = rot
        return rot

    def transform(self, points: np.ndarray, quat_wxyz: np.ndarray | None) -> np.ndarray:
        """Rotate ``(N, 3)`` sensor-frame points into the fixed world frame.

        If ``quat_wxyz`` is None the last known rotation is reused (scan
        telegrams arrive far more often than the pose changes meaningfully).
        """
        rot = self.update(quat_wxyz) if quat_wxyz is not None else self._last_rot
        return points @ rot.T
