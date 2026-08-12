"""Ingestion runtime — the viz-free glue between a source and a surface builder.

Runs the integration loop in a background thread so it is fully decoupled from
the render loop: however fast points arrive, they are filtered and fused here,
while the visualizer samples the latest state at its own frame rate.

Contains no visualization imports, so it can be exercised headlessly (see the
end-to-end convergence check) and, later, wrapped by a ROS 2 node.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from rocklabel.live.config import AppConfig
from rocklabel.live.filters import crop_mask
from rocklabel.live.leveling import GroundLeveler
from rocklabel.live.motion import OrientationTracker, matrix_to_quat, quat_to_matrix
from rocklabel.live.slam import SlamTracker
from rocklabel.live.sources.base import PointSource
from rocklabel.live.surfaces.base import SurfaceBuilder

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


class RawPointBuffer:
    """Fixed-capacity ring buffer of recent points + intensity (thread-safe).

    Stores an ``(capacity, 3)`` position array and a parallel ``(capacity,)``
    intensity array with a rolling write head so the newest points always
    overwrite the oldest. Intensity is NaN for points whose source provided
    none (the viewer falls back to neutral coloring there).
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, int(capacity))
        self._buf = np.zeros((self._capacity, 3), dtype=np.float64)
        self._inten = np.full(self._capacity, np.nan, dtype=np.float32)
        self._head = 0
        self._filled = 0
        self._lock = threading.Lock()

    def add(self, points: np.ndarray, intensity: np.ndarray | None = None) -> None:
        if points.size == 0:
            return
        pts = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        n = pts.shape[0]
        if intensity is not None and intensity.shape[0] == n:
            inten = np.asarray(intensity, dtype=np.float32)
        else:
            inten = np.full(n, np.nan, dtype=np.float32)
        with self._lock:
            if n >= self._capacity:
                self._buf[:] = pts[-self._capacity :]
                self._inten[:] = inten[-self._capacity :]
                self._head = 0
                self._filled = self._capacity
                return
            end = self._head + n
            if end <= self._capacity:
                self._buf[self._head : end] = pts
                self._inten[self._head : end] = inten
            else:
                first = self._capacity - self._head
                self._buf[self._head :] = pts[:first]
                self._inten[self._head :] = inten[:first]
                self._buf[: n - first] = pts[first:]
                self._inten[: n - first] = inten[first:]
            self._head = end % self._capacity
            self._filled = min(self._capacity, self._filled + n)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of the buffered ``(points (M, 3), intensity (M,))``."""
        with self._lock:
            m = self._filled if self._filled < self._capacity else self._capacity
            return self._buf[:m].copy(), self._inten[:m].copy()

    def clear(self) -> None:
        with self._lock:
            self._head = 0
            self._filled = 0


class FrameAccumBuffer:
    """Accumulated world-frame cloud whose retention is counted in FRAMES.

    The display cloud used to be a point-sized ring buffer, which made "how
    long does the accumulated cloud stay up" a function of scan density: a
    dense source evicted everything older after a handful of frames, a sparse
    one kept half a minute. Keeping whole frames instead makes the retention
    the operator sets ("Accum frames" in the viewer) mean the same thing on
    every source.

    ``max_points`` remains as a memory / render-cost ceiling — whichever limit
    is hit first drops the oldest frame. Frames also carry their source
    timestamp so the viewer can report the retained time span.
    """

    def __init__(self, max_frames: int, max_points: int) -> None:
        self._max_frames = max(1, int(max_frames))
        self._max_points = max(1, int(max_points))
        self._frames: deque[tuple[np.ndarray, np.ndarray, float]] = deque()
        self._n_points = 0
        #: Concatenated snapshot, rebuilt lazily: the viewer samples far more
        #: often than frames arrive, and concatenating 500k points is not free.
        self._cache: tuple[np.ndarray, np.ndarray] | None = None
        self._lock = threading.Lock()

    @property
    def max_frames(self) -> int:
        return self._max_frames

    def set_max_frames(self, value: int) -> None:
        """Change the retention; trims at once so the view reacts immediately."""
        with self._lock:
            self._max_frames = max(1, int(value))
            self._trim()

    @property
    def max_points(self) -> int:
        return self._max_points

    def set_max_points(self, value: int) -> None:
        """Change the point ceiling (trims immediately, like the frame limit)."""
        with self._lock:
            self._max_points = max(1, int(value))
            self._trim()

    def add(
        self,
        points: np.ndarray,
        intensity: np.ndarray | None = None,
        timestamp: float = 0.0,
    ) -> None:
        if points.size == 0:
            return
        pts = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        n = pts.shape[0]
        if intensity is not None and intensity.shape[0] == n:
            inten = np.asarray(intensity, dtype=np.float32)
        else:
            inten = np.full(n, np.nan, dtype=np.float32)
        with self._lock:
            self._frames.append((pts, inten, float(timestamp)))
            self._n_points += n
            self._trim()

    def _trim(self) -> None:
        """Drop oldest frames until both limits hold (caller holds the lock)."""
        while self._frames and (
            len(self._frames) > self._max_frames or self._n_points > self._max_points
        ):
            pts, _, _ = self._frames.popleft()
            self._n_points -= pts.shape[0]
        self._cache = None

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Retained ``(points (M, 3), intensity (M,))``, oldest frame first.

        Returns the buffer's cached concatenation — treat it as read-only
        (the viewer only ever masks or copies it).
        """
        with self._lock:
            if self._cache is None:
                if self._frames:
                    self._cache = (
                        np.concatenate([p for p, _, _ in self._frames]),
                        np.concatenate([i for _, i, _ in self._frames]),
                    )
                else:
                    self._cache = (
                        np.zeros((0, 3), dtype=np.float64),
                        np.zeros(0, dtype=np.float32),
                    )
            return self._cache

    def stats(self) -> tuple[int, int, float]:
        """``(frames, points, span_sec)`` currently held."""
        with self._lock:
            if not self._frames:
                return 0, 0, 0.0
            span = self._frames[-1][2] - self._frames[0][2]
            return len(self._frames), self._n_points, max(0.0, span)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._n_points = 0
            self._cache = None


class Stats:
    """Rolling runtime statistics for the on-screen / console readout."""

    def __init__(self, window_sec: float = 1.0) -> None:
        self._window = window_sec
        self._events: deque[tuple[float, int]] = deque()  # (t, n_points)
        self.points_total = 0
        self.batches_total = 0
        self.last_add_latency_ms = 0.0  # time for surface.add_points on last batch
        self.cells_occupied = 0
        self._lock = threading.Lock()

    def record_batch(self, n_points: int, add_latency_ms: float) -> None:
        now = time.perf_counter()
        with self._lock:
            self.points_total += n_points
            self.batches_total += 1
            self.last_add_latency_ms = add_latency_ms
            self._events.append((now, n_points))
            self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self._window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def points_per_sec(self) -> float:
        now = time.perf_counter()
        with self._lock:
            self._trim(now)
            if not self._events:
                return 0.0
            total = sum(n for _, n in self._events)
            span = max(1e-6, now - self._events[0][0])
            return total / span


class IngestEngine:
    """Owns the source, the surface builder, the raw buffer, and the stats, and
    drives the background integration loop.

    The visualizer calls :meth:`raw_snapshot`, :meth:`surface.get_mesh_arrays`,
    and the pause/reset controls; it never touches the source directly.
    """

    def __init__(
        self,
        source: PointSource,
        surface: SurfaceBuilder,
        config: AppConfig,
    ) -> None:
        self.source = source
        self.surface = surface
        d = config.display
        self.raw = RawPointBuffer(d.raw_buffer_size)
        #: Long-lived, subsampled world-frame cloud for visual verification.
        #: Retention is in frames (viewer knob); the point count is only a cap.
        self.accum = FrameAccumBuffer(d.accum_frames, d.accum_buffer_size)
        self._accum_stride = max(1, int(d.accum_subsample))
        self.stats = Stats()
        #: Rolling window of recent world-frame batches at FULL resolution,
        #: for live model scoring: the model trains on single-scan clouds, so
        #: the scorer needs fresh raw scans, not the fused/accumulated map.
        self._recent: deque[tuple[float, np.ndarray, np.ndarray | None]] = deque()
        self._recent_lock = threading.Lock()
        self._recent_max_sec = 5.0
        self._recent_max_batches = 512
        self._crop = config.crop
        self._use_imu = config.motion.use_imu
        self.tracker = OrientationTracker(yaw_only=config.motion.yaw_only)
        self._imu_seen = False
        #: Gravity levelling of the world frame — see :mod:`rocklabel.live.leveling`.
        #: Applied *after* IMU/SLAM, as a constant rotation, so the fused
        #: heightmap and the SLAM map never shift underneath themselves.
        self.leveler = GroundLeveler(config.level)
        #: Scan-to-map odometry; engages per-batch once IMU orientation arrives.
        self.slam: SlamTracker | None = (
            SlamTracker(config.slam, config.motion) if config.slam.enabled else None
        )
        self._slam_active = False
        self._record_cfg = config.record
        #: Live MCAP recorder; swapped atomically by start/stop_recording.
        self._recorder = None
        self._config = config

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()

    # -- lifecycle ----------------------------------------------------------- #
    def start(self) -> None:
        if self._thread is not None:
            return
        self.source.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="ingest-loop", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.source.stop()
        self.stop_recording()

    # -- MCAP recording (controls called from the UI thread) ------------------ #
    @property
    def recorder(self):
        """The active :class:`~lidarrig.recording.McapRecorder`, or None."""
        return self._recorder

    def start_recording(self, path: str | None = None) -> str | None:
        """Begin writing every ingested batch (+ pose) to an MCAP file.

        Returns the file path, or None when recording is not possible
        (already recording, or replaying — replays are never re-recorded).
        """
        if self._recorder is not None or self.source.is_replay:
            return None
        from rocklabel.live.recording import (McapRecorder, default_recording_path,
                                              normalize_recording_path)

        if not path:
            path = self._record_cfg.path or default_recording_path(self._record_cfg.dir)
        path = normalize_recording_path(path, self._record_cfg.dir)
        self._recorder = McapRecorder(path, config=self._config)
        return path

    def stop_recording(self) -> str | None:
        """Finalize and close the active recording; returns its path."""
        rec, self._recorder = self._recorder, None
        if rec is None:
            return None
        rec.close()
        return rec.path

    def current_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Sensor pose in the levelled world frame: ``(position, quat wxyz)``.

        Replay sources report the recorded pose; live, SLAM provides full pose,
        IMU-only provides rotation about a fixed position, else identity. The
        levelling rotation is applied last, so the pose lives in the same frame
        as the points every consumer sees.
        """
        src_pose = getattr(self.source, "current_pose", None)
        if callable(src_pose):
            pos, quat = src_pose()
        elif self._slam_active and self.slam is not None:
            pos, quat = self.slam.position, matrix_to_quat(self.slam.rotation)
        elif self._use_imu and self._imu_seen:
            pos, quat = np.zeros(3), matrix_to_quat(self.tracker.rotation)
        else:
            pos, quat = np.zeros(3), _IDENTITY_QUAT.copy()
        if not self.leveler.active:
            return pos, quat
        rot = self.leveler.rotation
        return rot @ pos, matrix_to_quat(rot @ quat_to_matrix(quat))

    def floor_z(self) -> float | None:
        """Measured floor height in the levelled frame (None until known)."""
        return self.leveler.floor_z

    # -- controls (called from the UI thread) -------------------------------- #
    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def set_paused(self, value: bool) -> None:
        if value:
            self._paused.set()
        else:
            self._paused.clear()

    def toggle_pause(self) -> bool:
        self.set_paused(not self.paused)
        return self.paused

    def reset_surface(self, reset_slam: bool = True) -> None:
        """Drop every fused/buffered point.

        ``reset_slam=False`` keeps the odometry map: levelling uses it that way
        because the SLAM map lives in the *pre*-levelling frame and is
        unaffected by the rotation — re-anchoring it there would throw away a
        good pose (and re-zero the IMU reference the levelling was solved for).
        """
        self.surface.reset()
        self.raw.clear()
        self.accum.clear()
        with self._recent_lock:
            self._recent.clear()
        if reset_slam and self.slam is not None:
            self.slam.reset()

    def raw_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Recent (cropped, fused) points + intensity for display."""
        return self.raw.snapshot()

    def accum_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Accumulated full world-frame cloud + intensity for display."""
        return self.accum.snapshot()

    def accum_stats(self) -> tuple[int, int, float]:
        """``(frames, points, span_sec)`` held by the accumulated cloud."""
        return self.accum.stats()

    @property
    def accum_frames(self) -> int:
        """How many recent frames the accumulated cloud keeps."""
        return self.accum.max_frames

    def set_accum_frames(self, value: int) -> None:
        """Set the accumulated cloud's retention, in frames (UI thread)."""
        self.accum.set_max_frames(value)

    @property
    def accum_max_points(self) -> int:
        """Point ceiling guarding the accumulated cloud."""
        return self.accum.max_points

    def set_accum_max_points(self, value: int) -> None:
        """Raise/lower that ceiling (UI thread)."""
        self.accum.set_max_points(value)

    def recent_snapshot(self, window_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """World-frame points + intensity of the freshest scans, full resolution.

        ``window_s <= 0`` returns only the single latest batch — the same
        per-scan cloud the training datasets were generated from. Larger
        windows concatenate every batch within that many seconds of the
        newest one. Intensity is NaN where the source provided none.
        """
        with self._recent_lock:
            if not self._recent:
                return np.empty((0, 3)), np.empty(0, np.float32)
            newest = self._recent[-1][0]
            if window_s <= 0.0:
                batches = [self._recent[-1]]
            else:
                batches = [b for b in self._recent if newest - b[0] <= window_s]
        pts = np.concatenate([b[1] for b in batches])
        inten = np.concatenate([
            b[2] if b[2] is not None and b[2].shape[0] == b[1].shape[0]
            else np.full(b[1].shape[0], np.nan, np.float32)
            for b in batches
        ])
        return pts, inten

    def pose_status(self) -> str:
        """Human-readable pose / IMU state for the stats readout."""
        if self.source.is_replay:
            x, y, z = self.current_pose()[0]
            return f"replay x{x:+.2f} y{y:+.2f} z{z:+.2f} m"
        if self._slam_active and self.slam is not None:
            return self.slam.status()
        if not self._use_imu:
            return "imu disabled"
        if not self._imu_seen:
            return "none (no IMU telegrams)"
        return f"yaw {self.tracker.yaw_deg:+.1f}° (imu only)"

    # -- levelling ----------------------------------------------------------- #
    def recalibrate_level(self) -> None:
        """Re-measure the mount tilt from scratch (GUI / L key)."""
        self.leveler.recalibrate()

    def _warn_if_band_misses_floor(self) -> None:
        """Say so, loudly, when the configured crop band excludes the ground.

        A band tuned for a hand-held rig (sensor ~1 m up) keeps nothing at all
        on a robot whose sensor rides 0.4 m up: the heightmap stays empty and
        the model reports "no rocks" rather than "no data", which is a very
        expensive thing to debug in the field.
        """
        from rocklabel.live.filters import crop_band

        floor = self.leveler.floor_z
        if floor is None or not self._crop.enabled:
            return
        z_min, z_max = crop_band(self._crop, floor)
        if z_min <= floor <= z_max:
            return
        lo = round(floor - 0.1, 2)
        hi = round(floor + 0.6, 2)
        print(
            f"[rocklabel] WARNING: floor is at z{floor:+.2f} m but the crop band is "
            f"{z_min:+.2f}..{z_max:+.2f} m — the ground is outside it, so the "
            f"heightmap and the model will see (almost) nothing.\n"
            f"[rocklabel]          use --floor-band -0.10 0.60, or "
            f"--z-min {lo:g} --z-max {hi:g}",
            flush=True,
        )

    def _level(self, points: np.ndarray, timestamp: float) -> np.ndarray:
        """Rotate one world-frame batch into the levelled frame, advancing the
        calibration while it is still collecting."""
        if not self.leveler.locked:
            # Seed from the IMU's own gravity reference: the world frame is the
            # sensor frame at the reference quaternion, so that quaternion's
            # rotation is exactly what tips it away from level.
            if self._imu_seen and self._use_imu:
                tracker = self.slam.imu if self._slam_active and self.slam else self.tracker
                ref = getattr(tracker, "reference_rotation", None)
                if ref is not None:
                    self.leveler.seed_from_imu(ref)
            levelled = points @ self.leveler.rotation.T
            if self.leveler.observe(levelled, timestamp):
                # The rotation just changed; anything fused with the old one is
                # stale, so drop it rather than leaving a tilted ghost in the
                # map. Re-levels this batch with the final rotation.
                print(f"[rocklabel] {self.leveler.status()}", flush=True)
                self._warn_if_band_misses_floor()
                self.reset_surface(reset_slam=False)
                return points @ self.leveler.rotation.T
            return levelled
        return points @ self.leveler.rotation.T

    # -- integration loop ---------------------------------------------------- #
    def _loop(self) -> None:
        while not self._stop.is_set():
            batch = self.source.read(timeout=0.1)
            if batch is None:
                continue
            if self._paused.is_set():
                continue
            points = batch.points
            inten = batch.intensity
            # 1. Bring points into the fixed world frame. With SLAM: full pose
            #    (IMU rotation prior + scan-to-map ICP translation) so the
            #    sensor may move through the room. Without: IMU rotation only.
            #    Point order is preserved, so intensity stays aligned.
            if batch.orientation is not None:
                self._imu_seen = True
                if self.slam is not None:
                    self._slam_active = True
                    points = self.slam.process(
                        points, batch.orientation, batch.timestamp
                    )
                elif self._use_imu:
                    points = self.tracker.transform(points, batch.orientation)
            # 1b. Gravity-level that world frame. The IMU's reference rotation
            #     is gravity-referenced, so it seeds the estimate on the very
            #     first telegram; a ground-plane fit then refines it and
            #     measures the floor height. Everything downstream — accum,
            #     scoring window, crop band, heightmap — sees levelled points.
            if self.leveler.active:
                points = self._level(points, batch.timestamp)
            pos, quat = self.current_pose()
            # Record the RAW sensor-frame batch plus the pose just computed for
            # it, so a replay reproduces this exact world-frame reconstruction.
            rec = self._recorder
            if rec is not None:
                rec.write_frame(
                    batch.points, batch.intensity, batch.timestamp,
                    batch.orientation, pos, quat,
                )
            # The accumulated cloud keeps the FULL world-frame view (walls and
            # all) — the map you check while moving the sensor.
            self.accum.add(
                points[:: self._accum_stride],
                inten[:: self._accum_stride] if inten is not None else None,
                batch.timestamp,
            )
            # Fresh-scan window for live model scoring (full resolution).
            with self._recent_lock:
                self._recent.append((batch.timestamp, points, inten))
                while (
                    len(self._recent) > self._recent_max_batches
                    or batch.timestamp - self._recent[0][0] > self._recent_max_sec
                ):
                    self._recent.popleft()
            # 2. Region-of-interest crop (z-band / range gate) BEFORE fusion so
            #    wall/ceiling returns from a 360° scan never enter the heightmap.
            #    The range gate is measured from the sensor, not the origin, so
            #    it keeps meaning the same thing once the robot drives off.
            keep = crop_mask(points, self._crop, origin=pos,
                             floor_z=self.leveler.floor_z)
            if keep is not None:
                points = points[keep]
                if inten is not None:
                    inten = inten[keep]
            if points.shape[0] == 0:
                self.stats.record_batch(0, 0.0)
                continue
            t0 = time.perf_counter()
            self.surface.add_points(points, inten)
            add_ms = (time.perf_counter() - t0) * 1000.0
            self.raw.add(points, inten)
            self.stats.record_batch(len(batch), add_ms)
            # Cheap-ish; only KalmanHeightmap exposes this.
            occ = getattr(self.surface, "cells_occupied", None)
            if callable(occ):
                self.stats.cells_occupied = occ()
