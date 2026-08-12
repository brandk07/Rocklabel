"""Minimal ROS 2 bag support so ``--play`` accepts rosbag MCAPs (viz-free).

Competition/robot recordings are rosbag2 MCAPs (profile ``ros2``): CDR-encoded
``sensor_msgs/msg/PointCloud2`` clouds plus ``tf2_msgs/msg/TFMessage`` poses,
instead of lidarrig's own ``/lidar/frames``. This module decodes exactly those
two message types with a small hand-rolled CDR reader — no ROS installation is
required to replay a bag.

Only little-endian CDR is supported (what every real ROS 2 system emits);
big-endian payloads raise. Alignment follows XCDR1: primitives align to their
size, counted from the byte after the 4-byte encapsulation header.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from rocklabel.live.motion import matrix_to_quat, quat_to_matrix

POINTCLOUD2_SCHEMA = "sensor_msgs/msg/PointCloud2"
TFMESSAGE_SCHEMA = "tf2_msgs/msg/TFMessage"

#: PointField.datatype -> numpy dtype (sensor_msgs/msg/PointField constants).
_DATATYPES = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}

#: Field names accepted as per-point intensity, in preference order.
_INTENSITY_FIELDS = ("intensity", "reflectivity", "reflective", "rssi")


class _Cdr:
    """Sequential little-endian CDR reader over one serialized message."""

    __slots__ = ("data", "off")

    def __init__(self, data: bytes) -> None:
        # Encapsulation header: {0x00, 0x01} = CDR little-endian (+2 options).
        if len(data) < 4 or not data[1] & 1:
            raise ValueError("unsupported CDR encapsulation (not little-endian)")
        self.data = data
        self.off = 4

    def _align(self, size: int) -> None:
        pad = (self.off - 4) % size
        if pad:
            self.off += size - pad

    def u8(self) -> int:
        v = self.data[self.off]
        self.off += 1
        return v

    def u32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from("<I", self.data, self.off)
        self.off += 4
        return v

    def i32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from("<i", self.data, self.off)
        self.off += 4
        return v

    def f64n(self, n: int) -> tuple[float, ...]:
        self._align(8)
        v = struct.unpack_from(f"<{n}d", self.data, self.off)
        self.off += 8 * n
        return v

    def string(self) -> str:
        n = self.u32()  # length including the NUL terminator
        s = self.data[self.off : self.off + n - 1] if n else b""
        self.off += n
        return s.decode("utf-8")


@dataclass
class CloudFrame:
    """One decoded PointCloud2: sensor-frame points + optional intensity."""

    frame_id: str
    stamp: float  # header stamp (s)
    points: np.ndarray  # (N, 3) float32, sensor frame
    intensity: np.ndarray | None  # (N,) float32


def decode_pointcloud2(data: bytes) -> CloudFrame:
    """Decode a CDR ``sensor_msgs/msg/PointCloud2`` into numpy arrays.

    x/y/z come from the named fields at their declared offsets/dtypes, so any
    point_step/layout works. The first field named like an intensity channel
    (see ``_INTENSITY_FIELDS``) is returned as float32 intensity, unscaled.
    """
    c = _Cdr(data)
    sec, nsec = c.i32(), c.u32()
    frame_id = c.string()
    height, width = c.u32(), c.u32()
    fields: dict[str, tuple[int, int]] = {}  # name -> (offset, datatype)
    for _ in range(c.u32()):
        name = c.string()
        off, datatype, _count = c.u32(), c.u8(), c.u32()
        fields[name] = (off, datatype)
    is_bigendian = c.u8()
    point_step, row_step = c.u32(), c.u32()
    nbytes = c.u32()
    if is_bigendian:
        raise ValueError("big-endian PointCloud2 data is not supported")
    buf = memoryview(c.data)[c.off : c.off + nbytes]

    n = height * width
    if height > 1 and row_step != width * point_step:
        # Row padding: compact rows so points are contiguous at point_step.
        rows = np.frombuffer(buf, np.uint8, count=height * row_step)
        buf = np.ascontiguousarray(
            rows.reshape(height, row_step)[:, : width * point_step]
        ).reshape(-1).data

    def column(name: str) -> np.ndarray:
        off, datatype = fields[name]
        return np.ndarray((n,), _DATATYPES[datatype], buf, off, (point_step,))

    pts = np.column_stack([column("x"), column("y"), column("z")]).astype(
        np.float32, copy=False
    )
    inten = None
    for name in _INTENSITY_FIELDS:
        if name in fields:
            inten = column(name).astype(np.float32)
            break
    return CloudFrame(frame_id, sec + nsec * 1e-9, pts, inten)


def decode_tfmessage(data: bytes) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Decode a CDR ``tf2_msgs/msg/TFMessage``.

    Returns ``(parent_frame, child_frame, position (3,), quat wxyz (4,))`` per
    transform (ROS stores quaternions xyzw; reordered here to lidarrig's wxyz).
    """
    c = _Cdr(data)
    out = []
    for _ in range(c.u32()):
        c.i32(), c.u32()  # header stamp (unused: latest-value TF store)
        parent = c.string()
        child = c.string()
        pos = np.array(c.f64n(3))
        x, y, z, w = c.f64n(4)
        out.append((parent, child, pos, np.array([w, x, y, z])))
    return out


class TfTree:
    """Latest-value TF store: composes a frame's pose up to the tree root.

    Each edge holds the most recent parent->child transform seen (no time
    interpolation — at bag rates the odometry is far denser than the clouds,
    so latest-value error is a fraction of one frame's motion).
    """

    def __init__(self) -> None:
        self._edges: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}

    def update(self, parent: str, child: str, pos: np.ndarray, quat: np.ndarray) -> None:
        self._edges[child] = (parent, pos, quat_to_matrix(quat))

    def pose(self, frame: str, max_depth: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """World pose ``(position, quat wxyz)`` of ``frame`` via its ancestors.

        Walks parent links until the root (or a not-yet-seen edge); missing
        edges early in a bag simply mean an identity contribution.
        """
        pos = np.zeros(3)
        rot = np.eye(3)
        for _ in range(max_depth):
            edge = self._edges.get(frame)
            if edge is None:
                break
            frame, t, r = edge[0], edge[1], edge[2]
            pos = r @ pos + t
            rot = r @ rot
        return pos, matrix_to_quat(rot)


def find_lidar_topics(summary) -> tuple[str, list[str]] | None:
    """Locate the cloud + tf topics in a rosbag's MCAP summary.

    Returns ``(cloud_topic, tf_topics)`` — the PointCloud2 channel with the
    most messages plus every TFMessage channel — or None if the bag has no
    point clouds.
    """
    counts = {}
    if summary.statistics is not None:
        counts = summary.statistics.channel_message_counts
    clouds, tfs = [], []
    for cid, ch in summary.channels.items():
        schema = summary.schemas.get(ch.schema_id)
        if schema is None:
            continue
        if schema.name == POINTCLOUD2_SCHEMA:
            clouds.append((counts.get(cid) or 0, ch.topic))
        elif schema.name == TFMESSAGE_SCHEMA:
            tfs.append(ch.topic)
    if not clouds:
        return None
    return max(clouds)[1], sorted(tfs)
