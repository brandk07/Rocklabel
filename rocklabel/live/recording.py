"""MCAP recording and replay for the lidarrig pipeline (viz-free).

Recording writes one MCAP channel, ``/lidar/frames``, holding every raw batch
the engine ingested: the **sensor-frame** points, per-point intensity (RSSI),
the batch timestamp, the IMU orientation quaternion, and the **world pose**
(position + orientation) the live SLAM/IMU pipeline computed for that batch.
The full :class:`~lidarrig.config.AppConfig` is embedded as MCAP metadata so a
replay reconstructs with the exact settings that were live.

Replay (:class:`McapReplaySource`) is a drop-in :class:`PointSource`: it feeds
the recorded batches back through the same pipeline, already transformed into
the world frame by the *recorded* pose (SLAM is not re-run, so playback is
deterministic and seek-rebuilds are fast). Play/pause, forward seeks, and
backward seeks (rewind + fast re-fuse from the start, via the ``on_rewind``
callback that resets the downstream engine) are all driven through
:meth:`read` on the consumer thread, so no cross-thread geometry races exist.

Frame payload v1 (little-endian)::

    u16 version | u16 flags | u32 n | f64 timestamp
    f64[4] orientation quat (w, x, y, z)          -- valid if flags & ORIENT
    f64[3] pose position | f64[4] pose quat wxyz  -- valid if flags & POSE
    f32[n*3] points (sensor frame)
    f32[n]   intensity                            -- present if flags & INTEN
"""

from __future__ import annotations

import datetime as _dt
import os
import struct
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import yaml

from rocklabel.live.config import AppConfig
from rocklabel.live.motion import quat_to_matrix
from rocklabel.live.sources.base import PointBatch, PointSource

TOPIC = "/lidar/frames"
SCHEMA_NAME = "lidarrig/Frame"
_METADATA_NAME = "lidarrig"

_VERSION = 1
_FLAG_INTEN = 1 << 0
_FLAG_ORIENT = 1 << 1
_FLAG_POSE = 1 << 2

_HEADER = struct.Struct("<HHId4d3d4d")  # version, flags, n, timestamp, quat, pos, pose-quat

_SCHEMA_DOC = (
    "little-endian: u16 version, u16 flags(1=intensity,2=orientation,4=pose), "
    "u32 n, f64 timestamp, f64[4] imu quat wxyz, f64[3] pose position, "
    "f64[4] pose quat wxyz, f32[n*3] sensor-frame points, [f32[n] intensity]"
)

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# Frame codec
# --------------------------------------------------------------------------- #
@dataclass
class RecordedFrame:
    """One decoded ``/lidar/frames`` message."""

    points: np.ndarray  # (N, 3) float32, sensor frame
    intensity: np.ndarray | None  # (N,) float32
    timestamp: float  # source/batch timestamp (s)
    orientation: np.ndarray | None  # (4,) IMU quat wxyz
    pose_position: np.ndarray  # (3,) world position of the sensor
    pose_quat: np.ndarray  # (4,) world orientation quat wxyz
    log_time_ns: int = 0  # wall-clock record time (set by the reader)

    def world_points(self) -> np.ndarray:
        """Sensor-frame points transformed by the recorded pose, as float64."""
        pts = self.points.astype(np.float64)
        rot = quat_to_matrix(self.pose_quat)
        return pts @ rot.T + self.pose_position


def encode_frame(
    points: np.ndarray,
    intensity: np.ndarray | None,
    timestamp: float,
    orientation: np.ndarray | None,
    pose_position: np.ndarray | None,
    pose_quat: np.ndarray | None,
) -> bytes:
    """Serialize one batch into the v1 frame payload."""
    pts = np.ascontiguousarray(points[:, :3], dtype=np.float32)
    n = pts.shape[0]
    flags = 0
    quat = np.zeros(4)
    if orientation is not None:
        flags |= _FLAG_ORIENT
        quat = np.asarray(orientation, dtype=np.float64)
    pos = np.zeros(3)
    pquat = _IDENTITY_QUAT
    if pose_position is not None and pose_quat is not None:
        flags |= _FLAG_POSE
        pos = np.asarray(pose_position, dtype=np.float64)
        pquat = np.asarray(pose_quat, dtype=np.float64)
    inten_bytes = b""
    if intensity is not None and intensity.shape[0] == n:
        flags |= _FLAG_INTEN
        inten_bytes = np.ascontiguousarray(intensity, dtype=np.float32).tobytes()
    header = _HEADER.pack(_VERSION, flags, n, float(timestamp), *quat, *pos, *pquat)
    return header + pts.tobytes() + inten_bytes


def decode_frame(data: bytes, log_time_ns: int = 0) -> RecordedFrame:
    """Parse a v1 frame payload back into arrays."""
    version, flags, n, ts, *rest = _HEADER.unpack_from(data)
    if version != _VERSION:
        raise ValueError(f"unsupported lidarrig frame version {version}")
    quat = np.array(rest[0:4])
    pos = np.array(rest[4:7])
    pquat = np.array(rest[7:11])
    off = _HEADER.size
    pts = np.frombuffer(data, dtype=np.float32, count=n * 3, offset=off).reshape(n, 3)
    off += n * 3 * 4
    inten = None
    if flags & _FLAG_INTEN:
        inten = np.frombuffer(data, dtype=np.float32, count=n, offset=off)
    return RecordedFrame(
        points=pts,
        intensity=inten,
        timestamp=ts,
        orientation=quat if flags & _FLAG_ORIENT else None,
        pose_position=pos if flags & _FLAG_POSE else np.zeros(3),
        pose_quat=pquat if flags & _FLAG_POSE else _IDENTITY_QUAT.copy(),
        log_time_ns=log_time_ns,
    )


# --------------------------------------------------------------------------- #
# Config <-> YAML metadata
# --------------------------------------------------------------------------- #
def config_to_yaml(cfg: AppConfig) -> str:
    """Serialize an AppConfig to YAML (tuples become lists)."""

    def clean(v: object) -> object:
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v

    return yaml.safe_dump(clean(asdict(cfg)), sort_keys=False)


def default_recording_path(directory: str) -> str:
    """Timestamped recording filename inside ``directory`` (created if needed)."""
    os.makedirs(directory, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(directory, f"lidar_{stamp}.mcap")


def normalize_recording_path(path: str, directory: str) -> str:
    """Resolve a user-supplied recording path to a real ``.mcap`` under a dir.

    A bare name ("myrun") is a name, not a path: it lands in ``directory``
    rather than the working directory, and it gets the ``.mcap`` extension.
    Without this an autostart recording writes an extension-less file into
    whatever directory the process was launched from, where the dashboard
    inventory — which scans ``recordings/`` for ``*.mcap`` — cannot see it.
    An explicit path ("out/run.mcap", "/tmp/x.mcap") is left where it points.
    """
    if os.path.dirname(path) == "":
        path = os.path.join(directory, path)
    if os.path.splitext(path)[1].lower() != ".mcap":
        path += ".mcap"
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #
class McapRecorder:
    """Writes ``/lidar/frames`` messages to an MCAP file (thread-safe).

    ``write_frame`` is called from the ingest thread; ``close`` from the UI
    thread — a lock serializes them. Files are chunked + indexed by the mcap
    writer, which is what makes seeking during replay efficient.
    """

    def __init__(self, path: str, config: AppConfig | None = None) -> None:
        from mcap.writer import Writer

        self.path = path
        self._fh = open(path, "wb")
        self._writer = Writer(self._fh)
        self._writer.start(profile="x-lidarrig", library="lidarrig")
        if config is not None:
            self._writer.add_metadata(
                _METADATA_NAME,
                {
                    "config_yaml": config_to_yaml(config),
                    "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                },
            )
        schema_id = self._writer.register_schema(
            name=SCHEMA_NAME, encoding="x-lidarrig", data=_SCHEMA_DOC.encode()
        )
        self._channel = self._writer.register_channel(
            topic=TOPIC, message_encoding="x-lidarrig-frame", schema_id=schema_id
        )
        self._lock = threading.Lock()
        self._closed = False
        self._seq = 0
        self.started_at = time.monotonic()
        self.points_total = 0

    @property
    def batches(self) -> int:
        return self._seq

    @property
    def elapsed_sec(self) -> float:
        return time.monotonic() - self.started_at

    def write_frame(
        self,
        points: np.ndarray,
        intensity: np.ndarray | None,
        timestamp: float,
        orientation: np.ndarray | None,
        pose_position: np.ndarray | None,
        pose_quat: np.ndarray | None,
    ) -> None:
        data = encode_frame(
            points, intensity, timestamp, orientation, pose_position, pose_quat
        )
        now_ns = time.time_ns()
        with self._lock:
            if self._closed:
                return
            self._writer.add_message(
                channel_id=self._channel,
                log_time=now_ns,
                publish_time=now_ns,
                data=data,
                sequence=self._seq,
            )
            self._seq += 1
            self.points_total += int(points.shape[0])

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._writer.finish()
            self._fh.close()


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def _guarded(it):
    """Yield from a reader iterator, treating a truncated tail as end-of-file.

    A recording that was never finalized (crash, SIGKILL, power loss) ends in
    a partial record; everything before it is still perfectly readable.
    """
    from mcap.exceptions import McapError

    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except (McapError, EOFError, ValueError):
            return


def read_recording_config(path: str) -> AppConfig | None:
    """Recover the AppConfig embedded in a recording, or None if absent.

    Tries the indexed reader first, then falls back to a linear scan so
    unfinalized (crashed) recordings still yield their config.
    """
    from mcap.reader import NonSeekingReader, make_reader

    for make in (make_reader, NonSeekingReader):
        try:
            with open(path, "rb") as fh:
                for metadata in _guarded(make(fh).iter_metadata()):
                    if metadata.name == _METADATA_NAME:
                        text = metadata.metadata.get("config_yaml")
                        if text:
                            return AppConfig.from_dict(yaml.safe_load(text) or {})
        except Exception:
            continue
    return None


class McapReplaySource(PointSource):
    """Replays a recording as a :class:`PointSource`, with transport controls.

    Two file formats are auto-detected on :meth:`start`:

    * **native** — lidarrig's own ``/lidar/frames`` recordings;
    * **ros2** — rosbag2 MCAPs (e.g. competition robot logs): the busiest
      ``sensor_msgs/msg/PointCloud2`` channel supplies the points (any
      intensity/reflectivity field included, rescaled to the u16 RSSI range
      the display expects), and ``/tf`` + ``/tf_static`` supply the pose that
      brings each cloud into the world frame. See :mod:`lidarrig.ros2bag`.

    Emitted batches are already in the **world frame** (recorded pose applied)
    with ``orientation=None``, so the downstream engine performs no additional
    motion compensation — the reconstruction matches the live run exactly.

    All file iteration happens inside :meth:`read` (the engine thread).
    :meth:`play`/:meth:`pause`/:meth:`seek` may be called from any thread; they
    only flip control flags. A backward seek sets a rewind flag: :meth:`read`
    reopens the file, invokes ``on_rewind`` (the engine's reset, running safely
    *between* batches on the consumer thread), then fast-forwards — delivering
    batches unpaced so the surface re-fuses as fast as the engine can go.
    """

    is_replay = True  # recording from a replay is disabled engine-side

    def __init__(
        self,
        path: str,
        on_rewind: Callable[[], None] | None = None,
        autoplay: bool = True,
        speed: float = 1.0,
    ) -> None:
        self.path = path
        self.on_rewind = on_rewind
        self._speed = max(1e-3, float(speed))
        self._lock = threading.Lock()
        self._playing = autoplay
        self._seek_target_ns: int | None = None
        self._rewind = False
        self._finished = False

        self._fh = None
        self._reader = None
        self._scan_fh = None  # linear-scan handle for unfinalized recordings
        self._indexed = False
        self._mode = "native"  # "native" (/lidar/frames) or "ros2" (rosbag)
        self._cloud_topic = ""
        self._tf_topics: list[str] = []
        self._inten_scale = 1.0  # ros2: intensity -> u16 RSSI display range
        self._iter = None
        #: prefetched (log_time, payload); payload is raw bytes (native) or an
        #: already-decoded RecordedFrame (ros2 — pose must be captured at
        #: iteration time, before the tf tree advances past this cloud).
        self._next: tuple[int, object] | None = None

        self._t0_ns = 0
        self._t1_ns = 0
        self._pos_ns = 0
        # Pacing anchor: wall time that corresponds to _anchor_log_ns.
        self._anchor_wall = 0.0
        self._anchor_log_ns = 0

        self._last_pose = (np.zeros(3), _IDENTITY_QUAT.copy())

    # -- PointSource lifecycle ---------------------------------------------- #
    def start(self) -> None:
        from mcap.reader import make_reader

        self._fh = open(self.path, "rb")
        self._indexed = False
        try:
            self._reader = make_reader(self._fh)
            summary = self._reader.get_summary()
            stats = summary.statistics if summary is not None else None
        except Exception:
            self._reader, stats = None, None
        if stats is not None and stats.message_count > 0:
            self._indexed = True
            self._t0_ns = stats.message_start_time
            self._t1_ns = stats.message_end_time
            topics = {ch.topic for ch in summary.channels.values()}
            if TOPIC not in topics:
                from rocklabel.live.ros2bag import find_lidar_topics

                found = find_lidar_topics(summary)
                if found is None:
                    raise ValueError(
                        f"{self.path}: no {TOPIC!r} channel and no PointCloud2 "
                        "topics — nothing to replay"
                    )
                self._mode = "ros2"
                self._cloud_topic, self._tf_topics = found
                self._inten_scale = self._probe_intensity_scale()
                print(
                    f"[rocklabel] ROS 2 bag: replaying {self._cloud_topic!r} "
                    f"(poses from {', '.join(self._tf_topics) or 'none: identity'}; "
                    f"intensity scale x{self._inten_scale:g})",
                    flush=True,
                )
        else:
            # Unfinalized recording (crash / kill): no summary index. Linear
            # pre-scan for the time range; iteration will also scan linearly.
            self._reader = None
            for log_time, _data in self._linear_iter():
                if self._t0_ns == 0:
                    self._t0_ns = log_time
                self._t1_ns = log_time
            self._close_scan()
        self._pos_ns = self._t0_ns
        self._open_iter()
        self._reanchor()

    def stop(self) -> None:
        self._iter = None
        self._reader = None
        self._close_scan()
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _close_scan(self) -> None:
        if self._scan_fh is not None:
            self._scan_fh.close()
            self._scan_fh = None

    def _linear_iter(self):
        """(log_time, data) over a fresh linear pass (no index required).

        ``log_time_order=False`` is essential for truncated files: the ordered
        variant buffers messages and loses them all when the partial tail
        record raises. Our writer emits one channel sequentially, so file
        order *is* log-time order.
        """
        from mcap.reader import NonSeekingReader

        self._close_scan()
        self._scan_fh = open(self.path, "rb")
        reader = NonSeekingReader(self._scan_fh)
        return (
            (m.log_time, m.data)
            for _s, _c, m in _guarded(
                reader.iter_messages(topics=[TOPIC], log_time_order=False)
            )
        )

    def _open_iter(self) -> None:
        if self._mode == "ros2":
            self._iter = self._ros2_iter()
        elif self._indexed:
            assert self._reader is not None
            self._iter = (
                (m.log_time, m.data)
                for _s, _c, m in _guarded(self._reader.iter_messages(topics=[TOPIC]))
            )
        else:
            self._iter = self._linear_iter()
        self._next = None
        self._advance()

    def _probe_intensity_scale(self) -> float:
        """Map the bag's intensity range onto u16 RSSI (display expects 0-65535).

        Drivers publish reflectivity as normalized 0-1 floats, u8, or raw u16;
        probe the first clouds' max instead of trusting the dtype.
        """
        from rocklabel.live.ros2bag import decode_pointcloud2

        assert self._reader is not None
        peak, probed = 0.0, 0
        it = _guarded(self._reader.iter_messages(topics=[self._cloud_topic]))
        for _s, _c, m in it:
            inten = decode_pointcloud2(m.data).intensity
            if inten is None:
                return 1.0
            finite = inten[np.isfinite(inten)]
            if finite.size:
                peak = max(peak, float(finite.max()))
            probed += 1
            if probed >= 25:
                break
        if peak <= 1.5:  # normalized 0-1 (or a boolean reflector flag)
            return 65535.0
        if peak <= 255.0:  # 8-bit reflectivity
            return 257.0
        return 1.0

    def _ros2_iter(self):
        """(log_time, RecordedFrame) over a rosbag's clouds, tf applied.

        TF messages are consumed inline (never yielded), so the pose composed
        for a cloud reflects every transform logged before it — including
        during seek fast-forwards, which iterate this same generator.
        """
        from rocklabel.live.ros2bag import TfTree, decode_pointcloud2, decode_tfmessage

        assert self._reader is not None
        tf = TfTree()
        topics = [self._cloud_topic, *self._tf_topics]
        for _s, ch, m in _guarded(self._reader.iter_messages(topics=topics)):
            if ch.topic != self._cloud_topic:
                for parent, child, pos, quat in decode_tfmessage(m.data):
                    tf.update(parent, child, pos, quat)
                continue
            cloud = decode_pointcloud2(m.data)
            pts, inten = cloud.points, cloud.intensity
            keep = np.isfinite(pts).all(axis=1)
            if not keep.all():  # drop no-return points (NaN/inf from the driver)
                pts = pts[keep]
                if inten is not None:
                    inten = inten[keep]
            if pts.shape[0] == 0:
                continue
            pos, quat = tf.pose(cloud.frame_id)
            yield (
                m.log_time,
                RecordedFrame(
                    points=pts,
                    intensity=None if inten is None else inten * self._inten_scale,
                    timestamp=cloud.stamp,
                    orientation=None,
                    pose_position=pos,
                    pose_quat=quat,
                    log_time_ns=m.log_time,
                ),
            )

    def _advance(self) -> None:
        """Prefetch the next message into ``self._next`` (None at EOF)."""
        assert self._iter is not None
        self._next = next(self._iter, None)

    # -- transport controls (any thread) ------------------------------------- #
    @property
    def duration_sec(self) -> float:
        return max(0.0, (self._t1_ns - self._t0_ns) / 1e9)

    @property
    def position_sec(self) -> float:
        return max(0.0, (self._pos_ns - self._t0_ns) / 1e9)

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def seeking(self) -> bool:
        return self._seek_target_ns is not None

    def play(self) -> None:
        with self._lock:
            if self._finished:  # play at the end = restart from the top
                self._rewind = True
                self._seek_target_ns = self._t0_ns
                self._finished = False
            self._playing = True
            self._reanchor()

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def toggle_play(self) -> bool:
        if self._playing:
            self.pause()
        else:
            self.play()
        return self._playing

    def seek(self, t_sec: float) -> None:
        """Jump to ``t_sec`` (relative). Backward jumps trigger a rewind+refuse."""
        with self._lock:
            t = max(0.0, min(float(t_sec), self.duration_sec))
            if t >= self.duration_sec:
                target = self._t1_ns  # exact: float→ns rounding must not drop the tail
            else:
                target = self._t0_ns + int(round(t * 1e9))
            if target < self._pos_ns or self._finished:
                self._rewind = True
                self._finished = False
            self._seek_target_ns = target

    def current_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World pose (position, quat wxyz) of the last delivered frame."""
        return self._last_pose

    def _reanchor(self) -> None:
        self._anchor_wall = time.monotonic()
        self._anchor_log_ns = self._pos_ns

    # -- consumer side (engine thread) ---------------------------------------- #
    def read(self, timeout: float | None = None) -> PointBatch | None:
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            with self._lock:
                if self._rewind:
                    self._rewind = False
                    do_rewind = True
                else:
                    do_rewind = False
                seek_target = self._seek_target_ns
                playing = self._playing

            if do_rewind:
                self._open_iter()
                self._pos_ns = self._t0_ns
                self._last_pose = (np.zeros(3), _IDENTITY_QUAT.copy())
                if self.on_rewind is not None:
                    self.on_rewind()  # engine reset, safely between batches

            if self._next is None:  # end of recording
                with self._lock:
                    self._playing = False
                    self._seek_target_ns = None
                self._finished = True
                self._pos_ns = self._t1_ns
                return self._sleep_out(deadline)

            log_time, data = self._next

            if seek_target is not None:
                if log_time > seek_target:  # reached the target
                    with self._lock:
                        # Only clear if no newer seek arrived meanwhile.
                        if self._seek_target_ns == seek_target:
                            self._seek_target_ns = None
                    self._reanchor()
                    continue
                return self._deliver()  # fast-forward: unpaced delivery

            if not playing:
                return self._sleep_out(deadline)

            due = self._anchor_wall + (log_time - self._anchor_log_ns) / 1e9 / self._speed
            now = time.monotonic()
            if now >= due:
                return self._deliver()
            wait = due - now
            if deadline is not None:
                remaining = deadline - now
                if remaining <= 0:
                    return None
                if wait > remaining:
                    time.sleep(remaining)
                    return None
            time.sleep(min(wait, 0.05))

    def _sleep_out(self, deadline: float | None) -> None:
        """Idle politely (paused / finished) without spinning the engine loop."""
        if deadline is None:
            time.sleep(0.05)
        else:
            time.sleep(max(0.0, min(0.05, deadline - time.monotonic())))
        return None

    def _deliver(self) -> PointBatch:
        assert self._next is not None
        log_time, data = self._next
        frame = (
            data if isinstance(data, RecordedFrame) else decode_frame(data, log_time)
        )
        self._pos_ns = log_time
        self._last_pose = (frame.pose_position, frame.pose_quat)
        self._advance()
        inten = (
            np.asarray(frame.intensity, dtype=np.float32)
            if frame.intensity is not None
            else None
        )
        return PointBatch(
            points=frame.world_points(),
            intensity=inten,
            timestamp=frame.timestamp,
            orientation=None,  # already world-frame: engine must not re-rotate
        )
