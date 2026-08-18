"""Native lidarrig recording support (``/lidar/frames``, schema ``lidarrig/Frame``).

These recordings come from the lidarrig live rig: each message is one raw
ingest batch holding the **sensor-frame** points, per-point reflectivity
(RSSI), and the **world pose** the rig's SLAM/IMU pipeline computed for that
batch. No TF tree exists or is needed — the pose is embedded per frame.

Frame payload v1 (little-endian)::

    u16 version | u16 flags | u32 n | f64 timestamp
    f64[4] imu quat (w, x, y, z)                  -- valid if flags & ORIENT
    f64[3] pose position | f64[4] pose quat wxyz  -- valid if flags & POSE
    f32[n*3] points (sensor frame)
    f32[n]   intensity                            -- present if flags & INTEN

The codec here is self-contained (mirrors lidarrig/recording.py) so rocklabel
has no dependency on the lidarrig package.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

TOPIC = "/lidar/frames"
SCHEMA_NAME = "lidarrig/Frame"

_VERSION = 1
_FLAG_INTEN = 1 << 0
_FLAG_ORIENT = 1 << 1
_FLAG_POSE = 1 << 2

_HEADER = struct.Struct("<HHId4d3d4d")  # version, flags, n, timestamp, quat, pos, pose-quat
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass
class LidarrigFrame:
    """One decoded ``/lidar/frames`` message."""

    points: np.ndarray            # (N, 3) float32, sensor frame
    intensity: np.ndarray | None  # (N,) float32 (RSSI counts), or None
    timestamp: float              # source/batch timestamp (s)
    pose_position: np.ndarray     # (3,) world position of the sensor
    pose_quat: np.ndarray         # (4,) world orientation quat (w, x, y, z)
    has_pose: bool = False
    log_time_ns: int = 0

    def pose_matrix(self) -> np.ndarray:
        """4x4 sensor->world transform from the recorded pose."""
        m = np.eye(4)
        m[:3, :3] = quat_wxyz_to_matrix(self.pose_quat)
        m[:3, 3] = self.pose_position
        return m

    def world_points(self) -> np.ndarray:
        """Sensor-frame points transformed by the recorded pose (float32 [N,3])."""
        rot = quat_wxyz_to_matrix(self.pose_quat).astype(np.float32)
        return self.points @ rot.T + self.pose_position.astype(np.float32)


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion (normalized internally)."""
    w, x, y, z = np.asarray(q, float)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def decode_frame(data: bytes, log_time_ns: int = 0) -> LidarrigFrame:
    """Parse a v1 frame payload back into arrays."""
    version, flags, n, ts, *rest = _HEADER.unpack_from(data)
    if version != _VERSION:
        raise ValueError(f"unsupported lidarrig frame version {version}")
    pos = np.array(rest[4:7])
    pquat = np.array(rest[7:11])
    off = _HEADER.size
    pts = np.frombuffer(data, dtype="<f4", count=n * 3, offset=off).reshape(n, 3)
    off += n * 3 * 4
    inten = None
    if flags & _FLAG_INTEN:
        inten = np.frombuffer(data, dtype="<f4", count=n, offset=off)
    has_pose = bool(flags & _FLAG_POSE)
    return LidarrigFrame(
        points=pts,
        intensity=inten,
        timestamp=ts,
        pose_position=pos if has_pose else np.zeros(3),
        pose_quat=pquat if has_pose else _IDENTITY_QUAT.copy(),
        has_pose=has_pose,
        log_time_ns=log_time_ns,
    )


def encode_frame(
    points: np.ndarray,
    intensity: np.ndarray | None,
    timestamp: float,
    pose_position: np.ndarray | None,
    pose_quat: np.ndarray | None,
) -> bytes:
    """Serialize one frame (used by tests and synthetic recordings)."""
    pts = np.ascontiguousarray(points[:, :3], dtype=np.float32)
    n = pts.shape[0]
    flags = 0
    pos = np.zeros(3)
    pquat = _IDENTITY_QUAT
    if pose_position is not None and pose_quat is not None:
        flags |= _FLAG_POSE
        pos = np.asarray(pose_position, dtype=np.float64)
        pquat = np.asarray(pose_quat, dtype=np.float64)
    inten_bytes = b""
    if intensity is not None and len(intensity) == n:
        flags |= _FLAG_INTEN
        inten_bytes = np.ascontiguousarray(intensity, dtype=np.float32).tobytes()
    header = _HEADER.pack(_VERSION, flags, n, float(timestamp), *np.zeros(4), *pos, *pquat)
    return header + pts.tobytes() + inten_bytes


def iter_frames(path: str, topic: str = TOPIC):
    """Yield decoded LidarrigFrame messages in log-time order.

    Falls back to a linear (unindexed) scan for recordings that were never
    finalized, recovering every complete frame before the truncation point.
    """
    from mcap.exceptions import McapError
    from mcap.reader import NonSeekingReader, make_reader

    with open(path, "rb") as fh:
        try:
            reader = make_reader(fh)
            summary = reader.get_summary()
        except Exception:
            reader, summary = None, None
        if summary is not None and summary.statistics and summary.statistics.message_count:
            it = reader.iter_messages(topics=[topic])
        else:  # unfinalized recording: linear scan, salvage what's readable
            fh.seek(0)
            it = NonSeekingReader(fh).iter_messages(topics=[topic], log_time_order=False)
        while True:
            try:
                _schema, _channel, message = next(it)
            except StopIteration:
                return
            except (McapError, EOFError, ValueError):
                return  # truncated tail: everything before it was delivered
            yield decode_frame(message.data, message.log_time)


def read_embedded_config(path: str) -> str | None:
    """Return the lidarrig config YAML embedded as mcap metadata, if present."""
    from mcap.reader import make_reader

    try:
        with open(path, "rb") as fh:
            for metadata in make_reader(fh).iter_metadata():
                if metadata.name == "lidarrig":
                    return metadata.metadata.get("config_yaml")
    except Exception:
        pass
    return None
