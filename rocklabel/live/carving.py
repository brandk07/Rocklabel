"""Asynchronous ray-carving accumulated cloud for the live view.

One worker owns the mutable map. Ingest queues ray-preserving observations;
the UI reads immutable published snapshots and never waits for carving work.
Logical evidence groups remain independent from the scheduling interval.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..geometry.carve import CarvingEvidence, KFCMap, RayBatch


@dataclass(frozen=True)
class _Observation:
    points: np.ndarray
    intensity: np.ndarray
    timestamp: float
    origin: np.ndarray
    pose_uncertainty_m: float


class CarvingAccum:
    """Carved world-frame cloud with a bounded asynchronous work queue."""

    def __init__(
        self,
        max_points: int,
        voxel: float = 0.05,
        interval_s: float = 0.4,
        delete_max_range: float = 8.0,
        evidence_group_s: float = 0.05,
        confirm_observations: int = 2,
        tentative_contradictions: int = 2,
        confirmed_contradictions: int = 3,
        queue_limit: int = 256,
        assumed_pose_uncertainty_m: float | None = None,
    ) -> None:
        self._max_points = max(1, int(max_points))
        self._interval_s = float(interval_s)
        if not np.isfinite(self._interval_s) or self._interval_s <= 0.0:
            raise ValueError("carving interval must be finite and positive")
        self._evidence_group_s = float(evidence_group_s)
        if not np.isfinite(self._evidence_group_s) or self._evidence_group_s <= 0.0:
            raise ValueError(
                "carving evidence-group interval must be finite and positive"
            )
        self._map_kwargs = {
            "voxel": voxel,
            "delete_max_range": delete_max_range,
            "add_max_range": delete_max_range,
            "confirm_observations": confirm_observations,
            "tentative_contradictions": tentative_contradictions,
            "confirmed_contradictions": confirmed_contradictions,
        }
        self._map = KFCMap(**self._map_kwargs)
        if assumed_pose_uncertainty_m is not None and (
            not np.isfinite(assumed_pose_uncertainty_m)
            or assumed_pose_uncertainty_m < 0.0
        ):
            raise ValueError("assumed pose uncertainty must be non-negative")
        self._assumed_pose_uncertainty_m = assumed_pose_uncertainty_m
        self._pending: list[_Observation] = []
        self._origin = np.zeros(3)
        self._last_fold: float | None = None
        self._last_stamp = 0.0
        self._first_stamp: float | None = None
        self._next_group = 0
        self._last_evidence_group: int | None = None
        self._groups_folded = 0

        empty_points = np.empty((0, 3), np.float32)
        empty_intensity = np.empty(0, np.float32)
        empty_observations = np.empty(0, np.int64)
        empty_support = np.empty(0, np.uint8)
        empty_contradictions = np.empty(0, np.uint8)
        empty_confirmed = np.empty(0, bool)
        empty_points.setflags(write=False)
        empty_intensity.setflags(write=False)
        for value in (
            empty_observations, empty_support, empty_contradictions,
            empty_confirmed,
        ):
            value.setflags(write=False)
        self._published = (empty_points, empty_intensity)
        self._published_evidence = (
            empty_points,
            CarvingEvidence(
                empty_observations, empty_support,
                empty_contradictions, empty_confirmed,
            ),
        )
        self._published_stats = (0, 0, 0.0)
        self._published_deleted = 0
        self._published_evicted = 0

        self._state_lock = threading.Lock()
        self._condition = threading.Condition()
        self._queue: deque[
            tuple[str, object | None, threading.Event | None, float]
        ] = deque()
        self._queue_limit = max(1, int(queue_limit))
        self._max_queued = 0
        self._producer_waits = 0
        self._last_work_ms = 0.0
        self._worker_error: BaseException | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._run, name="carving-map", daemon=True
        )
        self._worker.start()

    @property
    def deleted(self) -> int:
        with self._state_lock:
            return self._published_deleted

    @property
    def evicted(self) -> int:
        with self._state_lock:
            return self._published_evicted

    @property
    def max_frames(self) -> int:
        return 0

    def set_max_frames(self, value: int) -> None:
        """No-op: carving retains geometry by evidence, not frame count."""

    @property
    def max_points(self) -> int:
        with self._state_lock:
            return self._max_points

    def set_max_points(self, value: int) -> None:
        value = max(1, int(value))
        with self._state_lock:
            self._max_points = value
        self._enqueue("set_max_points", value)

    def add(
        self,
        points: np.ndarray,
        intensity: np.ndarray | None = None,
        timestamp: float = 0.0,
        origin: np.ndarray | None = None,
        pose_uncertainty_m: float | None = None,
    ) -> None:
        if points.size == 0:
            return
        # This is an asynchronous ownership boundary.  ``ascontiguousarray``
        # may return the caller's float64 buffer unchanged, so copy explicitly
        # before returning while the worker still owns the observation.
        pts = np.ascontiguousarray(points[:, :3], dtype=np.float64).copy()
        n = len(pts)
        if intensity is not None and np.asarray(intensity).shape == (n,):
            inten = np.asarray(intensity, dtype=np.float64).copy()
        else:
            inten = np.full(n, np.nan, dtype=np.float64)
        view = (
            np.asarray(origin, dtype=np.float64).reshape(-1)[:3].copy()
            if origin is not None else np.zeros(3, np.float64)
        )
        if view.shape != (3,):
            raise ValueError("origin must contain three coordinates")
        uncertainty = (
            (np.nan if self._assumed_pose_uncertainty_m is None
             else float(self._assumed_pose_uncertainty_m))
            if pose_uncertainty_m is None else float(pose_uncertainty_m)
        )
        self._enqueue(
            "add", _Observation(pts, inten, float(timestamp), view, uncertainty)
        )

    def flush(self) -> None:
        """Wait until all queued observations and the final group are published."""
        if self._closed:
            self._raise_worker_error()
            return
        self._enqueue("flush", None, wait=True)

    def sync(self) -> None:
        """Drain queued work without finalizing the open evidence group.

        Pause and a completed forward seek are resumable barriers: another
        observation may still arrive in the same time bucket.  Finalizing it
        here would either double-count that bucket or reject the resumed data.
        EOF, export, and shutdown use :meth:`flush` instead.
        """
        if self._closed:
            self._raise_worker_error()
            return
        self._enqueue("sync", None, wait=True)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the latest immutable published render snapshot."""
        with self._state_lock:
            return self._published

    def evidence_snapshot(self):
        with self._state_lock:
            return self._published_evidence

    def stats(self) -> tuple[int, int, float]:
        """Return ``(evidence groups, points, source span seconds)``."""
        with self._state_lock:
            return self._published_stats

    def backlog_stats(self) -> tuple[int, int, int, float]:
        """Return ``(queued, high-water, producer waits, last work ms)``."""
        with self._condition:
            queued = len(self._queue)
            high = self._max_queued
            waits = self._producer_waits
        with self._state_lock:
            work_ms = self._last_work_ms
        return queued, high, waits, work_ms

    def clear(self) -> None:
        self._enqueue("clear", None, wait=True)

    def close(self) -> None:
        if self._closed:
            return
        self._enqueue("close", None, wait=True)
        self._worker.join()

    def _enqueue(
        self,
        kind: str,
        payload: object | None,
        *,
        wait: bool = False,
    ) -> None:
        event = threading.Event() if wait else None
        with self._condition:
            self._raise_worker_error()
            if self._closed:
                if kind == "close":
                    return
                raise RuntimeError("carving accumulator is closed")
            while len(self._queue) >= self._queue_limit:
                self._producer_waits += 1
                self._condition.wait(timeout=0.1)
                self._raise_worker_error()
            self._queue.append((kind, payload, event, time.perf_counter()))
            self._max_queued = max(self._max_queued, len(self._queue))
            self._condition.notify()
        if event is not None:
            event.wait()
            self._raise_worker_error()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("carving worker failed") from self._worker_error

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                kind, payload, event, _queued_at = self._queue.popleft()
                self._condition.notify_all()
            started = time.perf_counter()
            measured_mapping_work = False
            try:
                if kind == "add":
                    assert isinstance(payload, _Observation)
                    measured_mapping_work = self._owner_add(payload)
                elif kind == "sync":
                    self._owner_sync()
                    measured_mapping_work = True
                elif kind == "flush":
                    self._owner_flush()
                    measured_mapping_work = True
                elif kind == "set_max_points":
                    self._map.evict_farthest(int(payload), self._origin)
                    self._publish_owner()
                    measured_mapping_work = True
                elif kind == "clear":
                    self._owner_clear()
                    measured_mapping_work = True
                elif kind == "close":
                    self._owner_flush()
                    measured_mapping_work = True
                    with self._condition:
                        self._closed = True
                    return
                else:
                    raise RuntimeError(f"unknown carving command {kind!r}")
            except BaseException as exc:
                with self._condition:
                    self._worker_error = exc
                    self._closed = True
                    while self._queue:
                        _kind, _payload, queued_event, _time = self._queue.popleft()
                        if queued_event is not None:
                            queued_event.set()
                    self._condition.notify_all()
                return
            finally:
                if measured_mapping_work:
                    with self._state_lock:
                        self._last_work_ms = (
                            time.perf_counter() - started
                        ) * 1000.0
                if event is not None:
                    event.set()

    def _owner_add(self, observation: _Observation) -> bool:
        self._pending.append(observation)
        self._origin = observation.origin
        self._last_stamp = observation.timestamp
        if self._first_stamp is None:
            self._first_stamp = observation.timestamp
        if (
            self._last_fold is None
            or observation.timestamp - self._last_fold >= self._interval_s
        ):
            self._fold_owner()
            self._last_fold = observation.timestamp
            return True
        return False

    def _fold_owner(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        groups: list[tuple[int, list[_Observation]]] = []
        for observation in pending:
            if np.isfinite(observation.timestamp):
                group_id = int(np.floor(
                    observation.timestamp / self._evidence_group_s
                ))
            else:
                self._next_group += 1
                group_id = -self._next_group
            if not groups or group_id != groups[-1][0]:
                groups.append((group_id, []))
            groups[-1][1].append(observation)
        for group_id, observations in groups:
            points = np.concatenate([item.points for item in observations])
            intensity = np.concatenate([item.intensity for item in observations])
            origins = np.concatenate([
                np.broadcast_to(item.origin, (len(item.points), 3))
                for item in observations
            ])
            timestamps = np.concatenate([
                np.full(len(item.points), item.timestamp, np.float64)
                for item in observations
            ])
            uncertainties = np.concatenate([
                np.full(len(item.points), item.pose_uncertainty_m, np.float64)
                for item in observations
            ])
            self._map.update_batch(RayBatch(
                endpoints=points,
                origins=origins,
                timestamps=timestamps,
                groups=np.full(len(points), group_id),
                intensity=intensity,
                floater=np.zeros(len(points), bool),
                pose_uncertainty_m=uncertainties,
            ))
            if group_id != self._last_evidence_group:
                self._groups_folded += 1
                self._last_evidence_group = group_id
        self._map.evict_farthest(self._max_points, self._origin)
        self._publish_owner()

    def _owner_flush(self) -> None:
        self._fold_owner()
        self._map.flush()
        self._map.evict_farthest(self._max_points, self._origin)
        self._publish_owner()

    def _owner_sync(self) -> None:
        self._fold_owner()
        self._map.evict_farthest(self._max_points, self._origin)
        self._publish_owner()

    def _publish_owner(self) -> None:
        points, intensity, _hits = self._map.result()
        points.setflags(write=False)
        intensity.setflags(write=False)
        evidence = self._map.evidence()
        for value in (
            evidence.observations, evidence.support,
            evidence.contradictions, evidence.confirmed,
        ):
            value.setflags(write=False)
        evidence_points = self._map.pts.copy()
        evidence_points.setflags(write=False)
        span = 0.0
        if self._first_stamp is not None:
            span = max(0.0, self._last_stamp - self._first_stamp)
        with self._state_lock:
            self._published = (points, intensity)
            self._published_evidence = (evidence_points, evidence)
            self._published_stats = (self._groups_folded, len(points), span)
            self._published_deleted = self._map.n_deleted
            self._published_evicted = self._map.n_evicted

    def _owner_clear(self) -> None:
        self._map = KFCMap(**self._map_kwargs)
        self._pending.clear()
        self._origin = np.zeros(3)
        self._first_stamp = None
        self._last_fold = None
        self._last_stamp = 0.0
        self._next_group = 0
        self._last_evidence_group = None
        self._groups_folded = 0
        self._publish_owner()
