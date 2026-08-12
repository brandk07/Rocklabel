"""TF pose buffering and time-interpolated transform lookup (no ROS required)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .mcap_io import iter_decoded, stamp_to_seconds


class PoseUnavailable(Exception):
    """Raised when a transform cannot be evaluated at the requested time."""


def make_matrix(translation, quat_xyzw) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    m[:3, 3] = translation
    return m


def invert(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    r = m[:3, :3].T
    out[:3, :3] = r
    out[:3, 3] = -r @ m[:3, 3]
    return out


@dataclass
class _Series:
    times: np.ndarray  # [N] float seconds, strictly increasing
    trans: np.ndarray  # [N, 3]
    rots: Rotation     # length N
    slerp: Slerp | None


class PoseBuffer:
    """Holds static and time-sorted dynamic transforms; interpolating lookup.

    Convention: a stored (parent, child) transform T maps points expressed in
    the child frame into the parent frame. ``lookup(target, source, t)``
    returns the 4x4 matrix mapping source-frame points into the target frame,
    chaining across the tree and interpolating dynamic transforms in time
    (lerp for translation, slerp for rotation).
    """

    def __init__(self, tolerance_s: float = 0.15):
        self.tolerance_s = float(tolerance_s)
        self._static: dict[tuple[str, str], np.ndarray] = {}
        self._raw: dict[tuple[str, str], list] = {}
        self._dynamic: dict[tuple[str, str], _Series] = {}
        self._paths: dict[tuple[str, str], list] = {}
        self._finalized = False

    def add_static(self, parent: str, child: str, translation, quat_xyzw) -> None:
        self._static[(parent, child)] = make_matrix(translation, quat_xyzw)
        self._paths.clear()

    def add_dynamic(self, parent: str, child: str, t: float, translation, quat_xyzw) -> None:
        self._raw.setdefault((parent, child), []).append(
            (float(t), np.asarray(translation, float), np.asarray(quat_xyzw, float))
        )
        self._finalized = False

    def finalize(self) -> None:
        """Sort dynamic samples by time, drop duplicate timestamps, build interpolators."""
        for pair, samples in self._raw.items():
            times = np.array([s[0] for s in samples])
            order = np.argsort(times, kind="stable")
            times = times[order]
            keep = np.concatenate(([True], np.diff(times) > 0))  # dedupe equal stamps
            idx = order[keep]
            times = times[keep]
            trans = np.stack([samples[i][1] for i in idx])
            rots = Rotation.from_quat(np.stack([samples[i][2] for i in idx]))
            slerp = Slerp(times, rots) if len(times) >= 2 else None
            self._dynamic[pair] = _Series(times, trans, rots, slerp)
        self._paths.clear()
        self._finalized = True

    def frames(self) -> set[str]:
        out = set()
        for parent, child in list(self._static) + list(self._raw):
            out.update((parent, child))
        return out

    def pairs(self) -> dict[tuple[str, str], str]:
        """(parent, child) -> 'static' | 'dynamic(N)' for inspection output."""
        out = {pair: "static" for pair in self._static}
        for pair, samples in self._raw.items():
            out[pair] = f"dynamic({len(samples)})"
        return out

    def time_range(self, parent: str, child: str) -> tuple[float, float]:
        s = self._dynamic[(parent, child)]
        return float(s.times[0]), float(s.times[-1])

    # -- lookup ------------------------------------------------------------

    def _find_path(self, source: str, target: str) -> list[tuple[tuple[str, str], bool]]:
        """BFS over the frame graph. Each step is ((parent, child), up) where
        up=True means we move child->parent (multiply by T_parent_child)."""
        cached = self._paths.get((source, target))
        if cached is not None:
            return cached
        adj: dict[str, list] = {}
        for pair in list(self._static) + list(self._dynamic):
            parent, child = pair
            adj.setdefault(child, []).append((parent, pair, True))
            adj.setdefault(parent, []).append((child, pair, False))
        prev: dict[str, tuple] = {source: None}
        queue = [source]
        while queue:
            frame = queue.pop(0)
            if frame == target:
                break
            for nxt, pair, up in adj.get(frame, []):
                if nxt not in prev:
                    prev[nxt] = (frame, pair, up)
                    queue.append(nxt)
        if target not in prev:
            raise PoseUnavailable(
                f"No transform chain from {source!r} to {target!r}; known frames: {sorted(self.frames())}"
            )
        path = []
        frame = target
        while prev[frame] is not None:
            came_from, pair, up = prev[frame]
            path.append((pair, up))
            frame = came_from
        path.reverse()
        self._paths[(source, target)] = path
        return path

    def _edge_matrix(self, pair: tuple[str, str], t: float) -> np.ndarray:
        series = self._dynamic.get(pair)
        if series is None:
            return self._static[pair]
        t0, t1 = series.times[0], series.times[-1]
        if t < t0 - self.tolerance_s or t > t1 + self.tolerance_s:
            raise PoseUnavailable(
                f"Transform {pair[0]!r}->{pair[1]!r} unavailable at t={t:.3f}s "
                f"(recorded range [{t0:.3f}, {t1:.3f}], tolerance {self.tolerance_s}s)"
            )
        tc = min(max(t, t0), t1)
        if series.slerp is None:
            return make_matrix(series.trans[0], series.rots.as_quat().reshape(-1, 4)[0])
        i = int(np.searchsorted(series.times, tc, side="right"))
        i = min(max(i, 1), len(series.times) - 1)
        ta, tb = series.times[i - 1], series.times[i]
        w = 0.0 if tb == ta else (tc - ta) / (tb - ta)
        trans = (1.0 - w) * series.trans[i - 1] + w * series.trans[i]
        rot = series.slerp(tc)
        m = np.eye(4)
        m[:3, :3] = rot.as_matrix()
        m[:3, 3] = trans
        return m

    def has_path(self, frame_a: str, frame_b: str) -> bool:
        """True if a transform chain exists between the frames (time-independent)."""
        if frame_a == frame_b:
            return True
        if not self._finalized:
            self.finalize()
        try:
            self._find_path(frame_a, frame_b)
            return True
        except PoseUnavailable:
            return False

    def lookup(self, target_frame: str, source_frame: str, t: float) -> np.ndarray:
        """4x4 matrix mapping points in source_frame to target_frame at time t."""
        if not self._finalized:
            self.finalize()
        if target_frame == source_frame:
            return np.eye(4)
        m = np.eye(4)
        for pair, up in self._find_path(source_frame, target_frame):
            edge = self._edge_matrix(pair, t)
            m = (edge if up else invert(edge)) @ m
        return m


def build_pose_buffer(path: str, topics_cfg: dict, check_lidar_mount: bool = True) -> PoseBuffer:
    """One pass over /tf and /tf_static building a PoseBuffer.

    If the recording provides no transform chain between the base and lidar
    frames, the topics.static_lidar_to_base config fallback is used; if
    neither source provides it, PoseUnavailable is raised immediately (unless
    check_lidar_mount is False, as in `rocklabel inspect`).
    """
    pb = PoseBuffer(tolerance_s=topics_cfg["pose_tolerance_s"])
    tf_topic = topics_cfg["tf_topic"]
    tf_static_topic = topics_cfg["tf_static_topic"]
    for topic, _log_time, msg in iter_decoded(path, [tf_topic, tf_static_topic]):
        for tr in msg.transforms:
            trans = (tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z)
            quat = (tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w)
            parent = tr.header.frame_id
            child = tr.child_frame_id
            if topic == tf_static_topic:
                pb.add_static(parent, child, trans, quat)
            else:
                pb.add_dynamic(parent, child, stamp_to_seconds(tr.header), trans, quat)

    pb.finalize()
    base, lidar = topics_cfg["base_frame"], topics_cfg["lidar_frame"]
    if base != lidar:
        try:
            pb._find_path(lidar, base)
        except PoseUnavailable:
            fallback = topics_cfg.get("static_lidar_to_base")
            if fallback is not None:
                pb.add_static(base, lidar, fallback["translation"], fallback["quaternion"])
            elif check_lidar_mount:
                raise PoseUnavailable(
                    f"No transform chain between {base!r} and {lidar!r}: the recording "
                    "provides no lidar mount on /tf_static (or /tf), and no "
                    "topics.static_lidar_to_base fallback is configured. "
                    f"Frames seen in TF: {sorted(pb.frames())}"
                ) from None
    return pb
