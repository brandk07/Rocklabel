"""Shared scan replay: decode -> pose -> odom/world-frame points.

``ScanStream(path, cfg, ...)`` auto-detects the recording format and returns
the right iterator, so the labeler, driftcheck, and the dataset generator all
see identical geometry regardless of where the data came from:

* **ROS 2 bags** (``sensor_msgs/msg/PointCloud2`` + ``/tf``): clouds are
  decoded and transformed into the odom frame via the TF tree.
* **Native lidarrig recordings** (``/lidar/frames``): each frame already
  carries the world pose its SLAM/IMU pipeline computed, so no TF is needed —
  points are simply transformed by the recorded pose.

When ``level.mode`` is not ``"off"`` the stream is additionally wrapped in a
:class:`LevelledScanStream`, which rotates a tilted sensor mount out of every
scan (see :mod:`rocklabel.geometry.leveling`). Doing it here rather than in the labeler
is the whole point: labels are stored in world coordinates, so the labeler and
the generator must agree on the frame or every rock center is misplaced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from . import lidarrig_io
from .mcap_io import (
    INTENSITY_PROBE_FRAMES,
    McapFormatError,
    decode_pointcloud2,
    intensity_scale_for_peak,
    iter_decoded,
    read_info,
    resolve_pointcloud_topic,
    stamp_to_seconds,
)
from .pose import PoseUnavailable, build_pose_buffer

# A header stamp is treated as garbage if zero or absurdly far from log time.
_STAMP_SANITY_S = 3600.0


@dataclass
class ScanCounters:
    scans_seen: int = 0
    scans_used: int = 0
    skipped_pose: int = 0
    skipped_stride: int = 0
    stamp_fallbacks: int = 0
    intensity_available: bool | None = None

    def summary_lines(self) -> list[str]:
        lines = [
            f"scans seen:            {self.scans_seen}",
            f"scans used:            {self.scans_used}",
            f"skipped (no pose):     {self.skipped_pose}",
            f"skipped (stride):      {self.skipped_stride}",
            f"bad header stamps:     {self.stamp_fallbacks} (fell back to log time)",
        ]
        if self.intensity_available is False:
            lines.append("intensity:             NOT AVAILABLE (all zeros written)")
        return lines


@dataclass
class OdomScan:
    index: int          # index among pointcloud messages, before any stride
    time_s: float       # sanitized scan time (header stamp, or log time fallback)
    xyz_odom: np.ndarray
    intensity: np.ndarray
    T_odom_base: np.ndarray
    T_odom_lidar: np.ndarray


class WindowedScanStream:
    """Wrap a scan stream, merging consecutive scans into time-window frames.

    Each yielded OdomScan concatenates the points of every scan whose time
    falls within ``window_s`` of the window's first scan; the pose, time, and
    index are the *last* member's. Turns the ~4 ms raw batches of a native
    lidarrig recording into dense frames.
    """

    def __init__(self, stream, window_s: float):
        self.stream = stream
        self.window_s = float(window_s)

    @property
    def counters(self):
        return self.stream.counters

    @property
    def info(self):
        return self.stream.info

    @property
    def scan_count(self) -> int:
        return self.stream.scan_count

    @property
    def format_name(self) -> str:
        return getattr(self.stream, "format_name", "")

    @staticmethod
    def _merge(buf: list[OdomScan]) -> OdomScan:
        last = buf[-1]
        if len(buf) == 1:
            return last
        return OdomScan(
            index=last.index,
            time_s=last.time_s,
            xyz_odom=np.concatenate([s.xyz_odom for s in buf]),
            intensity=np.concatenate([s.intensity for s in buf]),
            T_odom_base=last.T_odom_base,
            T_odom_lidar=last.T_odom_lidar,
        )

    def __iter__(self):
        buf: list[OdomScan] = []
        for scan in self.stream:
            if buf and scan.time_s - buf[0].time_s >= self.window_s:
                yield self._merge(buf)
                buf = []
            buf.append(scan)
        if buf:
            yield self._merge(buf)


class LevelledScanStream:
    """Wrap a scan stream, rotating a tilted sensor mount out of every scan.

    The rotation is a single constant measured once up front (see
    :func:`rocklabel.geometry.leveling.solve_level`) and applied to the points *and*
    both pose matrices, so a consumer that crops around ``T_odom_base`` gets a
    box aligned with the levelled floor rather than with the mast.
    """

    def __init__(self, stream, solution):
        self.stream = stream
        self.solution = solution

    @property
    def counters(self):
        return self.stream.counters

    @property
    def info(self):
        return self.stream.info

    @property
    def scan_count(self) -> int:
        return self.stream.scan_count

    @property
    def format_name(self) -> str:
        return getattr(self.stream, "format_name", "")

    def __iter__(self):
        rot = self.solution.rotation.astype(np.float32)
        m4 = self.solution.matrix4
        for scan in self.stream:
            yield OdomScan(
                index=scan.index,
                time_s=scan.time_s,
                xyz_odom=scan.xyz_odom @ rot.T,
                intensity=scan.intensity,
                T_odom_base=m4 @ scan.T_odom_base,
                T_odom_lidar=m4 @ scan.T_odom_lidar,
            )


def ScanStream(mcap_path: str, cfg: dict, stride: int = 1, progress: bool = True,
               desc: str = "scans"):
    """Open a recording as a stream of odom-frame scans, whatever its format.

    With levelling enabled this makes one extra pass over the recording to
    measure the mount angle before the caller's iteration begins.
    """
    info = read_info(mcap_path)

    def build(desc_: str):
        if info.is_lidarrig:
            return LidarrigScanStream(mcap_path, cfg, stride=stride, progress=progress,
                                      desc=desc_, info=info)
        return Ros2ScanStream(mcap_path, cfg, stride=stride, progress=progress,
                              desc=desc_, info=info)

    from ..geometry.leveling import solve_level  # deferred: leveling imports back through us

    solution = solve_level(cfg, lambda: build("level fit"))
    if solution is None:
        return build(desc)
    print("\n=== levelling ===")
    for line in solution.summary_lines():
        print("  " + line)
    return LevelledScanStream(build(desc), solution)


class Ros2ScanStream:
    """Iterate a rosbag2 recording's point clouds transformed into the odom frame."""

    format_name = "ros2"

    def __init__(self, mcap_path: str, cfg: dict, stride: int = 1, progress: bool = True,
                 desc: str = "scans", info=None):
        self.mcap_path = mcap_path
        self.cfg = cfg
        self.stride = max(int(stride), 1)
        self.progress = progress
        self.desc = desc
        self.counters = ScanCounters()
        self.info = info if info is not None else read_info(mcap_path)
        self.topic = resolve_pointcloud_topic(self.info, cfg["topics"]["pointcloud_topic"])
        # The chain check happens here against the scans' real frame_id, so
        # build_pose_buffer's config-frame check is skipped.
        self.pose_buffer = build_pose_buffer(mcap_path, cfg["topics"], check_lidar_mount=False)
        self._warned_intensity = False
        self._verified_frame: str | None = None
        self._inten_scale: float | None = None

    @property
    def scan_count(self) -> int:
        return self.info.message_count(self.topic)

    def _probe_intensity_scale(self) -> float:
        """Bring this bag's intensity into [0, 1].

        :func:`decode_pointcloud2` divides *integer* intensity channels by
        their dtype max, but a driver is free to publish RSSI counts in a
        float32 field - the SICK multiScan does exactly that, writing 0-65535
        as floats - and those arrive unscaled. Probing the first messages
        catches that case; already-normalized input probes at <= 1.5 and is
        left alone, so this never double-scales the integer path.
        """
        peak, probed = 0.0, 0
        for _topic, _log_ns, msg in iter_decoded(self.mcap_path, [self.topic]):
            _xyz, inten, has_intensity = decode_pointcloud2(msg)
            if not has_intensity:
                return 1.0
            finite = inten[np.isfinite(inten)]
            if finite.size:
                peak = max(peak, float(finite.max()))
            probed += 1
            if probed >= INTENSITY_PROBE_FRAMES:
                break
        return intensity_scale_for_peak(peak)

    def _source_frame(self, msg) -> str:
        """The frame the cloud's points are expressed in: the message's own
        frame_id, falling back to the configured lidar frame. Verified against
        the TF tree once so a wrong frame fails loudly, not as silent skips."""
        frame = (msg.header.frame_id or self.cfg["topics"]["lidar_frame"]).lstrip("/")
        if frame != self._verified_frame:
            odom = self.cfg["topics"]["odom_frame"]
            if not self.pose_buffer.has_path(frame, odom):
                raise SystemExit(
                    f"Point clouds on {self.topic!r} are in frame {frame!r}, but there is "
                    f"no transform chain from {frame!r} to the {odom!r} frame. "
                    f"Frames seen in TF: {sorted(self.pose_buffer.frames())}. "
                    "Check topics.odom_frame, or provide the lidar mount via "
                    "topics.static_lidar_to_base (applied as base_frame -> topics.lidar_frame)."
                )
            self._verified_frame = frame
        return frame

    def __iter__(self):
        topics = self.cfg["topics"]
        odom, base = topics["odom_frame"], topics["base_frame"]
        it = iter_decoded(self.mcap_path, [self.topic])
        if self.progress:
            it = tqdm(it, total=self.scan_count, desc=self.desc, unit="scan")
        for index, (_topic, log_time_ns, msg) in enumerate(it):
            self.counters.scans_seen += 1
            if index % self.stride != 0:
                self.counters.skipped_stride += 1
                continue

            t = stamp_to_seconds(msg.header)
            log_t = log_time_ns / 1e9
            if t <= 0.0 or abs(t - log_t) > _STAMP_SANITY_S:
                t = log_t
                self.counters.stamp_fallbacks += 1

            source = self._source_frame(msg)
            try:
                T_odom_base = self.pose_buffer.lookup(odom, base, t)
                T_odom_lidar = self.pose_buffer.lookup(odom, source, t)
            except PoseUnavailable:
                self.counters.skipped_pose += 1
                continue

            xyz, inten, has_intensity = decode_pointcloud2(msg)
            if self.counters.intensity_available is None:
                self.counters.intensity_available = has_intensity
                if not has_intensity and not self._warned_intensity:
                    self._warned_intensity = True
                    tqdm.write(
                        "WARNING: no intensity/reflectivity field found in the point cloud; "
                        "intensity will be 0 for all points and outputs will record "
                        '"intensity_available": false.'
                    )
            if self._inten_scale is None:
                self._inten_scale = self._probe_intensity_scale()
                if self._inten_scale != 1.0:
                    tqdm.write(f"intensity: rescaling by {self._inten_scale:g} "
                               "(driver publishes raw counts in a float field)")
            if self._inten_scale != 1.0:
                inten = inten * self._inten_scale
            xyz_odom = xyz @ T_odom_lidar[:3, :3].T.astype(np.float32) + T_odom_lidar[:3, 3].astype(np.float32)

            min_range = topics.get("min_range_m") or 0.0
            if min_range > 0.0 and len(xyz_odom):
                base_xy = T_odom_base[:2, 3].astype(np.float32)
                far = np.linalg.norm(xyz_odom[:, :2] - base_xy, axis=1) >= min_range
                xyz_odom, inten = xyz_odom[far], inten[far]

            self.counters.scans_used += 1
            yield OdomScan(index, t, xyz_odom, inten, T_odom_base, T_odom_lidar)


#: A sensor path this far off horizontal is a tilted frame, not terrain.
_PATH_TILT_WARN_DEG = 4.0
#: …but only believe it when the height really does track the position: this
#: much of the height variation has to be explained by where the rig was.
_PATH_TILT_MIN_R2 = 0.6
#: …and only after it has travelled far enough sideways to tell the difference.
_PATH_TILT_MIN_SPAN_M = 0.5


def path_tilt_deg(positions: np.ndarray) -> tuple[float, float] | None:
    """Measure how far a recording's world frame is off level from the path alone.

    A rig walking over flat ground traces a flat path, so its recorded height
    should not care where it is horizontally. Fit ``z = a·x + b·y + c`` over the
    path: a steep, well-explained fit means the frame itself is a ramp — the
    mount tilt was never taken out at capture time.

    Returns ``(tilt_deg, metres climbed per metre travelled)``, or None when the
    path is level, too short, or too wobbly to call.
    """
    p = np.asarray(positions, dtype=np.float64)
    if p.shape[0] < 3:
        return None
    if np.linalg.norm(p[:, :2] - p[:, :2].mean(axis=0), axis=1).max() < _PATH_TILT_MIN_SPAN_M:
        return None
    a = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    coef, *_ = np.linalg.lstsq(a, p[:, 2], rcond=None)
    spread = float(np.var(p[:, 2]))
    if spread <= 1e-9:
        return None
    r2 = 1.0 - float(np.var(p[:, 2] - a @ coef)) / spread
    slope = float(math.hypot(coef[0], coef[1]))
    tilt = math.degrees(math.atan(slope))
    if tilt < _PATH_TILT_WARN_DEG or r2 < _PATH_TILT_MIN_R2:
        return None
    return tilt, slope


class LidarrigScanStream:
    """Iterate a native lidarrig recording's frames in the world (odom) frame.

    Each frame's recorded SLAM pose plays the role both of T_odom_base and
    T_odom_lidar (the sensor *is* the base on the handheld rig). Reflectivity
    (RSSI counts) is normalized to [0, 1] using an auto-probed scale so it
    matches what the ROS 2 path produces for integer intensity channels.
    """

    format_name = "lidarrig"

    def __init__(self, mcap_path: str, cfg: dict, stride: int = 1, progress: bool = True,
                 desc: str = "scans", info=None):
        self.mcap_path = mcap_path
        self.cfg = cfg
        self.stride = max(int(stride), 1)
        self.progress = progress
        self.desc = desc
        self.counters = ScanCounters()
        self.info = info if info is not None else read_info(mcap_path)
        topics = self.info.lidarrig_topics()
        if not topics:
            raise McapFormatError(f"{mcap_path}: no lidarrig/Frame channel found")
        self.topic = lidarrig_io.TOPIC if lidarrig_io.TOPIC in topics else topics[0]
        self._inten_scale: float | None = None
        self._warned_pose = False
        #: Sensor positions collected as scans stream past, for the tilt check
        #: below. Free — the poses are already decoded.
        self._path: list[np.ndarray] = []
        self._warned_tilt = False

    @property
    def scan_count(self) -> int:
        return self.info.message_count(self.topic)

    def _probe_intensity_scale(self) -> float:
        """Normalize recorded intensity to [0, 1] whatever units the rig wrote.

        The multiScan RSSI arrives as u16 counts; older recordings may hold
        normalized floats or u8. Probe the first frames' max instead of
        trusting anything (see :func:`intensity_scale_for_peak`).
        """
        peak, probed = 0.0, 0
        for frame in lidarrig_io.iter_frames(self.mcap_path, self.topic):
            if frame.intensity is None:
                return 1.0
            finite = frame.intensity[np.isfinite(frame.intensity)]
            if finite.size:
                peak = max(peak, float(finite.max()))
            probed += 1
            if probed >= INTENSITY_PROBE_FRAMES:
                break
        return intensity_scale_for_peak(peak)

    def __iter__(self):
        min_range = self.cfg["topics"].get("min_range_m") or 0.0
        it = lidarrig_io.iter_frames(self.mcap_path, self.topic)
        if self.progress:
            it = tqdm(it, total=self.scan_count, desc=self.desc, unit="scan")
        for index, frame in enumerate(it):
            self.counters.scans_seen += 1
            if index % self.stride != 0:
                self.counters.skipped_stride += 1
                continue
            if len(frame.points) == 0:
                continue
            if not frame.has_pose:
                # Pose-less frames (SLAM had no fix yet) are identity-posed;
                # count them like TF-less scans and keep going.
                if self.counters.scans_used and not self._warned_pose:
                    self._warned_pose = True
                    tqdm.write("WARNING: frames without a recorded pose found mid-stream; "
                               "they are placed at the identity pose.")

            if self.counters.intensity_available is None:
                self.counters.intensity_available = frame.intensity is not None
                if frame.intensity is None:
                    tqdm.write("WARNING: this recording has no reflectivity (RSSI) data; "
                               "intensity will be 0 for all points.")
            if self._inten_scale is None:
                self._inten_scale = self._probe_intensity_scale()

            xyz = frame.world_points()
            if frame.intensity is not None:
                inten = frame.intensity.astype(np.float32) * self._inten_scale
            else:
                inten = np.zeros(len(xyz), np.float32)

            pose = frame.pose_matrix()
            if min_range > 0.0:
                base_xy = pose[:2, 3].astype(np.float32)
                far = np.linalg.norm(xyz[:, :2] - base_xy, axis=1) >= min_range
                xyz, inten = xyz[far], inten[far]

            t = frame.timestamp
            log_t = frame.log_time_ns / 1e9
            if t <= 0.0:
                t = log_t
                self.counters.stamp_fallbacks += 1

            self.counters.scans_used += 1
            self._check_tilt(pose[:3, 3])
            yield OdomScan(index, t, xyz, inten, pose, pose)

    def _check_tilt(self, position: np.ndarray) -> None:
        """Warn when the recording's world frame is not level, using only poses.

        A rig walking over flat ground traces a flat path. If the recorded
        sensor height instead tracks its horizontal position — z climbing as x
        and y do — then the frame the recording was made in is a ramp, and the
        mount tilt was never taken out at capture time. That single fact makes
        the height map span metres instead of centimetres, so it is worth
        saying out loud rather than leaving to be discovered by eye.

        Costs nothing: the poses are already decoded, and the fit runs once.
        """
        if self._warned_tilt or self.cfg["level"].get("mode", "off") != "off":
            return
        self._path.append(position[:3].copy())
        if len(self._path) < 300:
            return
        self._warned_tilt = True  # one verdict per stream, pass or fail
        measured = path_tilt_deg(np.asarray(self._path))
        if measured is None:
            return
        tilt, slope = measured
        tqdm.write(
            f"WARNING: this recording's world frame is tilted about {tilt:.0f}° — "
            f"the sensor's own path climbs {slope:.2f} m for every metre it "
            f"travels sideways, which flat ground cannot do.\n"
            f"         The mount tilt was not taken out when it was captured, so "
            f"the ground reads as a ramp metres tall and rocks disappear into it.\n"
            f"         Re-run with 'Undo mount tilt' set to ground (--level ground) "
            f"and use the same setting for every command on this recording."
        )
