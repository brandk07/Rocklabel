"""Reading rosbag2 mcap files and decoding PointCloud2 messages.

No ROS 2 installation required: messages are decoded from the schemas embedded
in the mcap file via mcap-ros2-support.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"
LIDARRIG_FRAME = "lidarrig/Frame"

# ROS PointField datatype constant -> numpy format character
_PF_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}

# Intensity channel candidates, tried in this order.
_INTENSITY_NAMES = ("intensity", "reflectivity", "i")


class McapFormatError(Exception):
    pass


@dataclass
class McapInfo:
    topics: dict = field(default_factory=dict)  # topic -> (schema_name, count)
    start_time_ns: int = 0
    end_time_ns: int = 0

    @property
    def duration_s(self) -> float:
        return (self.end_time_ns - self.start_time_ns) / 1e9

    def pointcloud_topics(self) -> list[str]:
        return [t for t, (s, _) in self.topics.items() if s == POINTCLOUD2]

    def lidarrig_topics(self) -> list[str]:
        return [t for t, (s, _) in self.topics.items() if s == LIDARRIG_FRAME]

    @property
    def is_lidarrig(self) -> bool:
        """True for native lidarrig recordings (/lidar/frames, pose embedded)."""
        return bool(self.lidarrig_topics())

    def message_count(self, topic: str) -> int:
        return self.topics.get(topic, (None, 0))[1]


_RECOVERY_HINT = (
    "the file is likely truncated or was never finalized (recorder killed "
    "mid-write). Salvage the readable part with:\n"
    "    rocklabel trim --mcap {path} --out fixed.mcap\n"
    "(which also drops non-LiDAR topics), or use the official `mcap recover` CLI."
)


def read_info(path: str) -> McapInfo:
    info = McapInfo()
    with open(path, "rb") as f:
        try:
            summary = make_reader(f).get_summary()
        except Exception as e:
            raise McapFormatError(
                f"{path}: cannot read mcap summary ({type(e).__name__}: {e}); "
                + _RECOVERY_HINT.format(path=path)
            ) from e
        if summary is None:
            raise McapFormatError(
                f"{path}: mcap file has no summary section; " + _RECOVERY_HINT.format(path=path)
            )
        stats = summary.statistics
        counts = stats.channel_message_counts if stats else {}
        for cid, channel in summary.channels.items():
            schema = summary.schemas.get(channel.schema_id)
            name = schema.name if schema else "<unknown>"
            info.topics[channel.topic] = (name, counts.get(cid, 0))
        if stats:
            info.start_time_ns = stats.message_start_time
            info.end_time_ns = stats.message_end_time
    return info


def resolve_pointcloud_topic(info: McapInfo, configured: str) -> str:
    """Validate the configured PointCloud2 topic against what the file contains."""
    candidates = info.pointcloud_topics()
    if not candidates:
        raise McapFormatError(
            "No sensor_msgs/msg/PointCloud2 topics (and no native lidarrig "
            "/lidar/frames channel) found in this mcap - run 'rocklabel inspect' "
            "to see what it contains"
        )
    if configured in info.topics:
        schema_name = info.topics[configured][0]
        if schema_name != POINTCLOUD2:
            raise McapFormatError(
                f"Topic {configured!r} has type {schema_name}, not {POINTCLOUD2}. "
                f"PointCloud2 topics in this file: {candidates}"
            )
        return configured
    raise McapFormatError(
        f"Configured pointcloud topic {configured!r} not found in this mcap. "
        f"PointCloud2 topics in this file: {candidates}. "
        "Set topics.pointcloud_topic in your config (see 'rocklabel inspect')."
    )


def iter_decoded(path: str, topics: list[str]):
    """Yield (topic, log_time_ns, decoded_message) in log-time order."""
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=topics, log_time_order=True
        ):
            yield channel.topic, message.log_time, decoded


def structured_dtype(fields, point_step: int, is_bigendian: bool) -> np.dtype:
    """Build a numpy structured dtype straight from PointCloud2 field descriptors."""
    endian = ">" if is_bigendian else "<"
    names, formats, offsets = [], [], []
    for f in fields:
        if f.datatype not in _PF_DTYPES:
            raise McapFormatError(f"Unsupported PointField datatype {f.datatype} for {f.name!r}")
        base = endian + _PF_DTYPES[f.datatype]
        count = getattr(f, "count", 1) or 1
        names.append(f.name)
        formats.append(base if count == 1 else (base, (count,)))
        offsets.append(f.offset)
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step})


def decode_pointcloud2(msg) -> tuple[np.ndarray, np.ndarray, bool]:
    """Decode a PointCloud2 message.

    Returns (xyz float32 [N,3], intensity float32 [N], intensity_available).
    Non-finite points are dropped. Organized clouds (height > 1) are flattened,
    honoring row_step padding. Integer intensity channels are normalized to
    [0, 1] by their dtype max; missing intensity yields zeros.
    """
    n = int(msg.width) * int(msg.height)
    if n == 0:
        return np.empty((0, 3), np.float32), np.empty(0, np.float32), False
    dt = structured_dtype(msg.fields, int(msg.point_step), bool(msg.is_bigendian))
    for req in ("x", "y", "z"):
        if req not in dt.names:
            raise McapFormatError(f"PointCloud2 missing required field {req!r}")

    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    row_bytes = int(msg.width) * int(msg.point_step)
    if msg.height > 1 and int(msg.row_step) != row_bytes:
        raw = np.ascontiguousarray(
            raw[: int(msg.height) * int(msg.row_step)]
            .reshape(int(msg.height), int(msg.row_step))[:, :row_bytes]
        ).reshape(-1)
    points = np.frombuffer(raw.tobytes() if not raw.flags.c_contiguous else raw, dtype=dt, count=n)

    xyz = np.empty((n, 3), np.float32)
    for i, name in enumerate(("x", "y", "z")):
        xyz[:, i] = points[name].astype(np.float32)  # astype handles byte order

    intensity_available = False
    inten = np.zeros(n, np.float32)
    for name in _INTENSITY_NAMES:
        if name in dt.names:
            col = points[name]
            kind, itemsize = col.dtype.kind, col.dtype.itemsize
            if col.ndim > 1:
                col = col[..., 0]
            if kind == "u":
                inten = col.astype(np.float32) / float(2 ** (8 * itemsize) - 1)
            else:
                inten = col.astype(np.float32)
            intensity_available = True
            break

    finite = np.isfinite(xyz).all(axis=1)
    if not finite.all():
        xyz, inten = xyz[finite], inten[finite]
    return xyz, inten, intensity_available


def stamp_to_seconds(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9
