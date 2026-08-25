"""`rocklabel selfhits`: drop the robot's own bodywork out of a recording.

On a competition run the LiDAR is bolted to the robot, so part of every single
sweep lands on the machine carrying it. Those returns sit at a **fixed spot in
sensor coordinates** — the sensor never moves relative to its own mount — but
once each sweep is placed into the world they smear into a long trail that
follows the robot everywhere it drove, sitting on top of the terrain you
actually want to look at.

The test that separates bodywork from terrain is distance to the sensor. On the
Lunabotics rig the LiDAR is mounted ~0.57 m above the ground, so the closest a
real ground return can ever be is ~0.5 m, while every persistent return
measures under ~0.35 m. Anything inside a small sphere around the sensor is
therefore the robot, and nothing else can be.

The input file is opened read-only and never modified: this writes a **new**
recording, copying every other topic and every other message through byte for
byte, so labels, poses and timing are untouched. Both formats are handled —
ROS 2 bags (``PointCloud2``, re-encoded with the surviving points) and native
lidarrig recordings (``/lidar/frames``).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np
from mcap.records import Channel, Message, Metadata, Schema
from mcap.stream_reader import StreamReader
from mcap.writer import Writer as McapWriter
from tqdm import tqdm

from .. import __version__

POINTCLOUD2_SCHEMA = "sensor_msgs/msg/PointCloud2"
LIDARRIG_SCHEMA = "lidarrig/Frame"

#: Default sphere radius (m) around the sensor. Everything inside it is the
#: robot: see the module docstring for why nothing real can reach in this far.
DEFAULT_RADIUS = 0.45


class SelfHitError(Exception):
    """Raised when a recording cannot be filtered (bad geometry, no clouds)."""


# --------------------------------------------------------------------------- #
# The mask
# --------------------------------------------------------------------------- #
@dataclass
class SelfHitMask:
    """Which sensor-frame points count as the robot's own body.

    ``radius`` is the sphere around the sensor origin. ``boxes`` are extra
    axis-aligned regions in sensor coordinates, for a part (a mast, a raised
    blade) that pokes out beyond the sphere; each is
    ``(x0, x1, y0, y1, z0, z1)`` in meters.
    """

    radius: float = DEFAULT_RADIUS
    boxes: list[tuple[float, float, float, float, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise SelfHitError("radius must be >= 0")
        for b in self.boxes:
            if len(b) != 6:
                raise SelfHitError(
                    f"a box needs 6 numbers (x0 x1 y0 y1 z0 z1), got {len(b)}"
                )
            if b[0] > b[1] or b[2] > b[3] or b[4] > b[5]:
                raise SelfHitError(f"box has a reversed edge (min > max): {b}")
        if self.radius == 0 and not self.boxes:
            raise SelfHitError("nothing to remove: radius is 0 and no --box was given")

    def hits(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask, True where a sensor-frame point is the robot itself."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.size == 0:
            return np.zeros(len(pts), dtype=bool)
        bad = ~np.isfinite(pts).all(axis=1)
        # Squared distance: avoids a sqrt over millions of points per sweep.
        d2 = np.einsum("ij,ij->i", pts, pts)
        out = d2 <= self.radius * self.radius
        for x0, x1, y0, y1, z0, z1 in self.boxes:
            out |= (
                (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
                & (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
            )
        out &= ~bad  # never judge a non-finite point; it is passed through
        return out

    def describe(self) -> str:
        parts = []
        if self.radius > 0:
            parts.append(f"sphere r={self.radius:g} m around the sensor")
        for b in self.boxes:
            parts.append(
                "box x[{:g},{:g}] y[{:g},{:g}] z[{:g},{:g}]".format(*b)
            )
        return "; ".join(parts)


# --------------------------------------------------------------------------- #
# PointCloud2 surgery
# --------------------------------------------------------------------------- #
@dataclass
class _CloudLayout:
    """Byte offsets inside one CDR PointCloud2 needed to rewrite it in place."""

    off_height: int
    off_width: int
    off_row_step: int
    off_data_len: int
    data_start: int
    data_end: int
    height: int
    width: int
    point_step: int
    row_step: int
    fields: dict          # name -> (offset, datatype)
    is_bigendian: int


_DATATYPES = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}


class _Cdr:
    """Little-endian CDR reader that also reports where each field started."""

    def __init__(self, data: bytes) -> None:
        if len(data) < 4 or not data[1] & 1:
            raise SelfHitError("unsupported CDR encapsulation (not little-endian)")
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

    def u32_at(self) -> tuple[int, int]:
        """Return ``(value, offset_it_started_at)``."""
        self._align(4)
        at = self.off
        (v,) = struct.unpack_from("<I", self.data, at)
        self.off += 4
        return v, at

    def u32(self) -> int:
        return self.u32_at()[0]

    def i32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from("<i", self.data, self.off)
        self.off += 4
        return v

    def string(self) -> str:
        n = self.u32()
        s = self.data[self.off : self.off + n - 1] if n else b""
        self.off += n
        return s.decode("utf-8", "replace")


def _parse_cloud(data: bytes) -> _CloudLayout:
    """Walk a PointCloud2 far enough to know where to cut and what to patch."""
    c = _Cdr(data)
    c.i32(), c.u32()          # header stamp
    c.string()                # frame_id
    height, off_height = c.u32_at()
    width, off_width = c.u32_at()
    fields: dict = {}
    for _ in range(c.u32()):
        name = c.string()
        off, datatype, _count = c.u32(), c.u8(), c.u32()
        fields[name] = (off, datatype)
    is_bigendian = c.u8()
    point_step = c.u32()
    row_step, off_row_step = c.u32_at()
    nbytes, off_data_len = c.u32_at()
    data_start = c.off
    data_end = data_start + nbytes
    if data_end > len(data):
        raise SelfHitError("PointCloud2 data runs past the end of the message")
    return _CloudLayout(
        off_height=off_height, off_width=off_width, off_row_step=off_row_step,
        off_data_len=off_data_len, data_start=data_start, data_end=data_end,
        height=height, width=width, point_step=point_step, row_step=row_step,
        fields=fields, is_bigendian=is_bigendian,
    )


def _cloud_records(data: bytes, lay: _CloudLayout) -> np.ndarray:
    """The point payload as a compact ``(n, point_step)`` byte array.

    Some publishers pad each row, so rows are copied out one at a time when
    ``row_step`` is wider than the points it carries.
    """
    n = lay.height * lay.width
    blob = np.frombuffer(data, np.uint8, count=lay.data_end - lay.data_start,
                         offset=lay.data_start)
    if lay.height > 1 and lay.row_step != lay.width * lay.point_step:
        rows = blob[: lay.height * lay.row_step].reshape(lay.height, lay.row_step)
        blob = np.ascontiguousarray(rows[:, : lay.width * lay.point_step]).reshape(-1)
    return blob[: n * lay.point_step].reshape(n, lay.point_step)


def _cloud_xyz(records: np.ndarray, lay: _CloudLayout) -> np.ndarray:
    """Pull x/y/z out of the raw records at their declared offsets/dtypes."""
    cols = []
    for name in ("x", "y", "z"):
        if name not in lay.fields:
            raise SelfHitError(f"PointCloud2 has no '{name}' field")
        off, datatype = lay.fields[name]
        dt = _DATATYPES.get(datatype)
        if dt is None:
            raise SelfHitError(f"unsupported PointField datatype {datatype}")
        size = np.dtype(dt).itemsize
        cols.append(records[:, off : off + size].copy().view(dt).reshape(-1))
    return np.column_stack(cols).astype(np.float64)


def filter_pointcloud2(data: bytes, mask: SelfHitMask) -> tuple[bytes, int, int]:
    """Return ``(new_message, kept, removed)`` with self-hit points dropped.

    Every per-point field the sensor wrote (intensity, range, azimuth, layer,
    echo, ...) rides along untouched, because whole ``point_step`` records are
    kept or dropped rather than re-serialized field by field.
    """
    lay = _parse_cloud(data)
    if lay.is_bigendian:
        raise SelfHitError("big-endian PointCloud2 data is not supported")
    records = _cloud_records(data, lay)
    n = records.shape[0]
    if n == 0:
        return data, 0, 0
    hit = mask.hits(_cloud_xyz(records, lay))
    removed = int(hit.sum())
    if removed == 0:
        return data, n, 0
    kept_records = records[~hit]
    m = int(kept_records.shape[0])

    head = bytearray(data[: lay.data_start])
    # The cloud stops being an organized grid once arbitrary points are gone,
    # so it is re-declared as a single row of m points.
    struct.pack_into("<I", head, lay.off_height, 1)
    struct.pack_into("<I", head, lay.off_width, m)
    struct.pack_into("<I", head, lay.off_row_step, m * lay.point_step)
    struct.pack_into("<I", head, lay.off_data_len, m * lay.point_step)
    return (
        bytes(head) + np.ascontiguousarray(kept_records).tobytes() + data[lay.data_end :],
        m,
        removed,
    )


def filter_lidarrig_frame(data: bytes, mask: SelfHitMask) -> tuple[bytes, int, int]:
    """Same, for a native lidarrig ``/lidar/frames`` payload."""
    from ..live.recording import decode_frame, encode_frame

    fr = decode_frame(data)
    n = int(fr.points.shape[0])
    if n == 0:
        return data, 0, 0
    hit = mask.hits(fr.points)
    removed = int(hit.sum())
    if removed == 0:
        return data, n, 0
    keep = ~hit
    inten = None if fr.intensity is None else np.asarray(fr.intensity)[keep]
    return (
        encode_frame(fr.points[keep], inten, fr.timestamp, fr.orientation,
                     fr.pose_position, fr.pose_quat),
        int(keep.sum()),
        removed,
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _pick_cloud_topics(in_path: str, configured: str) -> tuple[set[str], set[str]]:
    """(``PointCloud2`` topics to filter, native ``lidarrig`` topics to filter).

    Prefers the topic named in the config; falls back to the busiest cloud
    topic in the file so a bag from someone else's rig still works.
    """
    from mcap.reader import make_reader

    clouds: dict[str, int] = {}
    native: set[str] = set()
    try:
        with open(in_path, "rb") as fh:
            summary = make_reader(fh).get_summary()
    except Exception:
        summary = None
    if summary is None:
        return ({configured}, {"/lidar/frames"})
    counts = {}
    if summary.statistics is not None:
        counts = summary.statistics.channel_message_counts
    for cid, ch in summary.channels.items():
        schema = summary.schemas.get(ch.schema_id)
        if schema is None:
            continue
        if schema.name == POINTCLOUD2_SCHEMA:
            clouds[ch.topic] = counts.get(cid) or 0
        elif schema.name == LIDARRIG_SCHEMA:
            native.add(ch.topic)
    if configured in clouds:
        return ({configured}, native)
    if clouds:
        return ({max(clouds, key=lambda t: clouds[t])}, native)
    return (set(), native)


def run_selfhits(in_path: str, out_path: str, cfg: dict,
                 radius: float = DEFAULT_RADIUS,
                 boxes: list[tuple[float, ...]] | None = None,
                 dry_run: bool = False) -> None:
    """Write ``out_path``: ``in_path`` minus the points that hit the robot."""
    if not dry_run and os.path.abspath(in_path) == os.path.abspath(out_path):
        raise SelfHitError("selfhits: --out must differ from the input file")
    mask = SelfHitMask(radius=radius, boxes=[tuple(b) for b in (boxes or [])])
    cloud_topics, native_topics = _pick_cloud_topics(
        in_path, cfg["topics"]["pointcloud_topic"]
    )
    if not cloud_topics and not native_topics:
        raise SelfHitError(
            f"{in_path}: no PointCloud2 or /lidar/frames topics - nothing to filter"
        )

    schemas: dict[int, Schema] = {}
    channels: dict[int, Channel] = {}
    new_schema_ids: dict[int, int] = {}
    new_channel_ids: dict[int, int] = {}
    kept_pts = removed_pts = 0
    frames = 0
    per_frame: list[float] = []
    read_error: Exception | None = None

    total = os.path.getsize(in_path)
    fout = None if dry_run else open(out_path, "wb")
    try:
        writer = None
        if fout is not None:
            writer = McapWriter(fout)
            writer.start(profile="ros2", library=f"rocklabel {__version__}")
        with open(in_path, "rb") as fin:
            records = StreamReader(fin, emit_chunks=False).records
            progress = tqdm(total=total, unit="B", unit_scale=True, desc="selfhits")
            pos = 0
            while True:
                try:
                    rec = next(records)
                except StopIteration:
                    break
                except Exception as e:  # truncated input: salvage what we have
                    read_error = e
                    break
                here = fin.tell()
                if here > pos:
                    progress.update(here - pos)
                    pos = here
                if isinstance(rec, Schema):
                    schemas[rec.id] = rec
                    continue
                if isinstance(rec, Channel):
                    channels[rec.id] = rec
                    continue
                if isinstance(rec, Metadata):
                    if writer is not None:
                        writer.add_metadata(rec.name, dict(rec.metadata))
                    continue
                if not isinstance(rec, Message):
                    continue
                channel = channels.get(rec.channel_id)
                if channel is None:
                    continue
                data = rec.data
                if channel.topic in cloud_topics or channel.topic in native_topics:
                    fn = (filter_lidarrig_frame if channel.topic in native_topics
                          else filter_pointcloud2)
                    try:
                        data, kept, removed = fn(rec.data, mask)
                    except SelfHitError:
                        raise
                    except Exception as e:
                        raise SelfHitError(
                            f"could not filter a message on {channel.topic}: {e}"
                        ) from e
                    kept_pts += kept
                    removed_pts += removed
                    frames += 1
                    if kept + removed:
                        per_frame.append(100.0 * removed / (kept + removed))
                if writer is None:
                    continue
                new_cid = new_channel_ids.get(rec.channel_id)
                if new_cid is None:
                    schema = schemas.get(channel.schema_id)
                    if schema is None:
                        continue
                    new_sid = new_schema_ids.get(channel.schema_id)
                    if new_sid is None:
                        new_sid = writer.register_schema(
                            schema.name, schema.encoding, schema.data
                        )
                        new_schema_ids[channel.schema_id] = new_sid
                    new_cid = writer.register_channel(
                        channel.topic, channel.message_encoding, new_sid,
                        dict(channel.metadata),
                    )
                    new_channel_ids[rec.channel_id] = new_cid
                writer.add_message(new_cid, rec.log_time, data, rec.publish_time,
                                   rec.sequence)
            progress.close()
        if writer is not None:
            writer.add_metadata("rocklabel_selfhits", {
                "source": os.path.basename(in_path),
                "mask": mask.describe(),
                "radius_m": str(mask.radius),
                "points_removed": str(removed_pts),
            })
            writer.finish()
    finally:
        if fout is not None:
            fout.close()

    label = "would remove" if dry_run else "removed"
    print(f"\n=== selfhits summary: {out_path if not dry_run else '(dry run)'} ===")
    print(f"  robot mask:      {mask.describe()}")
    print(f"  scans filtered:  {frames}")
    seen = kept_pts + removed_pts
    if seen:
        print(f"  points {label}:  {removed_pts:,} of {seen:,} "
              f"({100.0 * removed_pts / seen:.2f}%)")
        print(f"  points kept:     {kept_pts:,}")
    if per_frame:
        pf = np.array(per_frame)
        print(f"  per scan:        median {np.median(pf):.1f}%  "
              f"min {pf.min():.1f}%  max {pf.max():.1f}%")
    if removed_pts == 0 and frames:
        print("  NOTHING was removed - the sensor may not be robot-mounted, or "
              "the radius is smaller than the machine. Try --dry-run with a "
              "larger --radius.")
    if not dry_run:
        print(f"  output size:     {os.path.getsize(out_path) / 1e6:.1f} MB "
              f"(input {total / 1e6:.1f} MB)")
        print(f"  input untouched: {in_path}")
    if read_error is not None:
        print(
            f"  NOTE: input ended early at byte {pos} of {total} "
            f"({100 * pos / max(total, 1):.0f}%): {type(read_error).__name__}. "
            "Everything before that point was kept; the output file is valid."
        )


def measure(in_path: str, cfg: dict, sample: int = 150,
            max_range: float = 3.0) -> dict:
    """Measure how close persistent, sensor-fixed structure gets.

    Bins a sample of scans by direction and asks, per direction, whether a
    return keeps coming back at the same distance. Bodywork does (it is bolted
    on); ground and rocks do not (they slide past as the robot drives). The
    furthest such return is what the mask radius has to cover.
    """
    from collections import defaultdict

    from mcap.reader import make_reader

    cloud_topics, native_topics = _pick_cloud_topics(
        in_path, cfg["topics"]["pointcloud_topic"]
    )
    topics = sorted(cloud_topics | native_topics)
    if not topics:
        raise SelfHitError(f"{in_path}: no cloud topics to measure")

    def clouds():
        with open(in_path, "rb") as fh:
            reader = make_reader(fh)
            summary = reader.get_summary()
            total = 0
            if summary is not None and summary.statistics is not None:
                total = summary.statistics.message_count
            step = max(1, total // (sample * 4)) if total else 1
            for i, (_s, ch, m) in enumerate(reader.iter_messages(topics=topics)):
                if i % step:
                    continue
                if ch.topic in native_topics:
                    from ..live.recording import decode_frame
                    yield np.asarray(decode_frame(m.data).points, dtype=np.float64)
                else:
                    lay = _parse_cloud(m.data)
                    yield _cloud_xyz(_cloud_records(m.data, lay), lay)

    AB, EB = 720, 180
    seen: dict[int, list[float]] = defaultdict(list)
    used = 0
    for pts in clouds():
        pts = pts[np.isfinite(pts).all(axis=1)]
        d = np.linalg.norm(pts, axis=1)
        ok = (d > 1e-3) & (d < max_range)
        pts, d = pts[ok], d[ok]
        if not len(d):
            continue
        az = (np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) + 180.0) % 360.0
        el = np.degrees(np.arcsin(np.clip(pts[:, 2] / d, -1, 1)))
        ai = np.clip((az / 360.0 * AB).astype(np.int32), 0, AB - 1)
        ei = np.clip(((el + 90.0) / 180.0 * EB).astype(np.int32), 0, EB - 1)
        idx = ai * EB + ei
        order = np.argsort(d)
        idx_s, d_s = idx[order], d[order]
        first = np.unique(idx_s, return_index=True)[1]
        for k, v in zip(idx_s[first].tolist(), d_s[first].tolist()):
            seen[k].append(v)
        used += 1
        if used >= sample:
            break

    rigid = []
    for v in seen.values():
        if len(v) < 0.9 * used:
            continue
        arr = np.array(v)
        med = float(np.median(arr))
        if float((np.abs(arr - med) < 0.05).mean()) >= 0.9:
            rigid.append(med)
    return {
        "scans": used,
        "directions": len(seen),
        "rigid_directions": len(rigid),
        "max_rigid_range_m": max(rigid) if rigid else 0.0,
        "p95_rigid_range_m": float(np.percentile(rigid, 95)) if rigid else 0.0,
        "suggested_radius_m": round(max(rigid) + 0.1, 2) if rigid else 0.0,
    }


def print_measure(result: dict) -> None:
    print("\n=== how far out does the robot reach? ===")
    print(f"  scans looked at:            {result['scans']}")
    print(f"  directions with a return:   {result['directions']}")
    print(f"  ...fixed to the sensor:     {result['rigid_directions']}")
    if not result["rigid_directions"]:
        print("\n  No sensor-fixed structure found. Either this sensor is not "
              "robot-mounted, or it never sees the machine it rides on.")
        return
    print(f"  furthest fixed return:      {result['max_rigid_range_m']:.3f} m")
    print(f"  95% of them are within:     {result['p95_rigid_range_m']:.3f} m")
    print(f"\n  suggested --radius:         {result['suggested_radius_m']:.2f}"
          "   (furthest fixed return + 10 cm)")
