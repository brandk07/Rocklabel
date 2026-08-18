"""Offline multi-pass pose solver.

Running from a file instead of from the live sensor buys two things the stock
tracker cannot have, and this module spends both:

**Time.** The stock tracker re-solves the whole 41 s recording in under 2 s, so
it is using about 5% of one core's realtime budget. Offline there is no budget
at all, so we can afford point-to-plane matching, 25 iterations per window
instead of 5, a 3x3x3 correspondence search, and points out to 30 m.

**Hindsight.** Pass 1 is causal: window 3 is aligned against a map built from
windows 1-2, which is nearly empty, so the early poses are the worst ones and
they get baked into the map permanently. Every later pass rebuilds the map from
the *finished* trajectory and re-aligns every window against it, so window 3
finally gets to see the whole court. That alone removes most of the start-up
error.

The output is one pose per original batch, interpolated smoothly across window
boundaries rather than extrapolated on velocity, so there is no step in the
trajectory every 0.1 s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rocklabel.slam.config import AltSlamConfig
from rocklabel.slam.register import register
from rocklabel.slam.voxelmap import NormalVoxelMap, voxel_downsample
from rocklabel.live.motion import (
    matrix_to_quat,
    quat_conjugate,
    quat_multiply,
    quat_to_matrix,
)


@dataclass
class Window:
    """One registration window: pooled geometry plus the pose being solved."""

    t_ref: float                      # reference timestamp (window midpoint)
    first: int                        # index of first source batch
    last: int                         # index of last source batch (inclusive)
    pts: np.ndarray                   # (M,3) IMU-rotated, downsampled: ICP source
    dts: np.ndarray                   # (M,) mean time offset from t_ref
    raw: np.ndarray                   # (K,3) IMU-rotated, full density: map input
    raw_dts: np.ndarray               # (K,) time offset from t_ref
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))   # ICP correction
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world position
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ok: bool = False
    rmse: float = float("nan")
    ratio: float = 0.0
    suppressed: int = 0

    def _place(self, pts: np.ndarray, dts: np.ndarray) -> np.ndarray:
        p = pts @ self.rot.T + self.pos
        # De-skew: the sensor keeps moving during the window (a few cm at
        # walking pace), so shift each point by where the sensor was when it
        # was actually measured.
        if self.vel.any():
            p = p + self.vel * dts[:, None]
        return p

    def world(self) -> np.ndarray:
        """Downsampled points in the world frame — what ICP aligns."""
        return self._place(self.pts, self.dts)

    def world_dense(self) -> np.ndarray:
        """Full-density points in the world frame — what feeds the map.

        The map needs density, not thrift: a voxel only yields a surface normal
        once enough points have landed in it, and feeding it one downsampled
        point per window starves it. This is the difference between 16% and 90%
        of a window finding correspondences.
        """
        return self._place(self.raw, self.raw_dts)


@dataclass
class SolveStats:
    """What the solve did, for the report."""

    windows: int = 0
    registered: int = 0
    failed: int = 0
    rmse_median: float = float("nan")
    ratio_median: float = 0.0
    suppressed_mean: float = 0.0
    path_length: float = 0.0
    pass_rmse: list = field(default_factory=list)


class OfflineSolver:
    """Solves a whole recording's trajectory in several passes."""

    def __init__(self, cfg: AltSlamConfig | None = None) -> None:
        self.cfg = cfg or AltSlamConfig()
        self.windows: list[Window] = []
        self.stats = SolveStats()
        self.up = np.array([0.0, 0.0, 1.0])

    # -- setup -------------------------------------------------------------- #
    def build_windows(self, frames) -> None:
        """Pool batches into windows, pre-rotating each by its own IMU sample.

        The IMU part of the pose never changes between passes, so it is applied
        once here; a pass then only has to apply the window's correction
        rotation and position.
        """
        cfg = self.cfg
        # World frame = sensor frame at startup, matching the live convention.
        ref_q = None
        for fr in frames:
            if fr.orientation is not None:
                ref_q = np.asarray(fr.orientation, dtype=np.float64)
                break
        if ref_q is None:
            raise ValueError("recording has no IMU orientation; cannot solve")
        ref_conj = quat_conjugate(ref_q / np.linalg.norm(ref_q))
        # Gravity up-vector expressed in that startup frame (the IMU
        # quaternion is gravity-referenced, so this is absolute).
        self.up = quat_to_matrix(ref_q)[2, :].copy()

        t0 = frames[0].timestamp
        self.windows = []
        buf_pts: list[np.ndarray] = []
        buf_ts: list[np.ndarray] = []
        win_start = None
        first_idx = 0
        last_rot = np.eye(3)

        def flush(idx_last: int) -> None:
            if not buf_pts:
                return
            pts = np.concatenate(buf_pts)
            ts = np.concatenate(buf_ts)
            t_ref = float(ts.mean())
            # Downsample for the ICP source, carrying each cell's mean timestamp
            # along so the de-skew term survives the reduction. The full-density
            # copy is kept separately for map building.
            cen, tm = voxel_downsample_timed(pts, ts, cfg.voxel_size)
            self.windows.append(
                Window(t_ref=t_ref, first=first_idx, last=idx_last,
                       pts=cen, dts=tm - t_ref,
                       raw=pts, raw_dts=ts - t_ref)
            )

        for i, fr in enumerate(frames):
            if fr.orientation is not None:
                q = np.asarray(fr.orientation, dtype=np.float64)
                last_rot = quat_to_matrix(quat_multiply(ref_conj, q / np.linalg.norm(q)))
            p = np.asarray(fr.points, dtype=np.float64)
            if p.shape[0]:
                r2 = np.einsum("ij,ij->i", p, p)
                sel = (r2 >= cfg.reg_range_min ** 2) & (r2 <= cfg.reg_range_max ** 2)
                if np.any(sel):
                    buf_pts.append(p[sel] @ last_rot.T)
                    buf_ts.append(np.full(int(sel.sum()), fr.timestamp - t0))
            if win_start is None:
                win_start = fr.timestamp
                first_idx = i
            if fr.timestamp - win_start >= cfg.window_sec:
                flush(i)
                buf_pts, buf_ts = [], []
                win_start = None
        flush(len(frames) - 1)
        self.stats.windows = len(self.windows)

    # -- passes ------------------------------------------------------------- #
    def solve(self, progress=None) -> SolveStats:
        """Run the configured number of passes and return the statistics."""
        if not self.windows:
            raise ValueError("call build_windows() first")
        self._pass_forward(progress)
        for p in range(2, self.cfg.passes + 1):
            self._pass_refine(p, progress)
        self._finalize_stats()
        return self.stats

    def _pass_forward(self, progress) -> None:
        """Pass 1: causal solve, growing the map as it goes."""
        cfg = self.cfg
        vmap = NormalVoxelMap(cfg.voxel_size, cfg.map_max_points,
                              cfg.min_points_normal, cfg.map_capacity)
        prev: Window | None = None
        vel = np.zeros(3)
        for k, w in enumerate(self.windows):
            if prev is not None:
                # Constant-velocity prediction from the previous window.
                w.rot = prev.rot.copy()
                w.pos = prev.pos + vel * (w.t_ref - prev.t_ref)
                w.vel = vel
            if vmap.size == 0:
                vmap.insert(w.world_dense())
                w.ok = True
                prev = w
                continue
            res = register(w.world(), w.pos, vmap, cfg, self.up)
            if res.ok:
                w.rot = res.rotation @ w.rot
                w.pos = w.pos + res.translation
            w.ok, w.rmse, w.ratio, w.suppressed = (
                res.ok, res.rmse, res.ratio, res.suppressed
            )
            vmap.insert(w.world_dense())
            if prev is not None and w.t_ref > prev.t_ref:
                v_inst = (w.pos - prev.pos) / (w.t_ref - prev.t_ref)
                vel = v_inst if not vel.any() else 0.5 * vel + 0.5 * v_inst
                speed = float(np.linalg.norm(vel))
                if speed > 2.0:
                    vel *= 2.0 / speed
            prev = w
            if progress:
                progress("pass 1", k + 1, len(self.windows))

    def _pass_refine(self, number: int, progress) -> None:
        """Later passes: rebuild the map from the current trajectory, then
        re-align every window against that finished map."""
        cfg = self.cfg
        vmap = NormalVoxelMap(cfg.voxel_size, cfg.map_max_points,
                              cfg.min_points_normal, cfg.map_capacity)
        for w in self.windows:
            vmap.insert(w.world_dense())
        vmap.refresh_normals()
        for k, w in enumerate(self.windows):
            res = register(w.world(), w.pos, vmap, cfg, self.up)
            if res.ok:
                w.rot = res.rotation @ w.rot
                w.pos = w.pos + res.translation
            w.ok, w.rmse, w.ratio, w.suppressed = (
                res.ok, res.rmse, res.ratio, res.suppressed
            )
            if progress:
                progress(f"pass {number}", k + 1, len(self.windows))
        self._reestimate_velocities()

    def _reestimate_velocities(self) -> None:
        """Recompute each window's de-skew velocity from the solved trajectory."""
        n = len(self.windows)
        for i, w in enumerate(self.windows):
            a = self.windows[max(0, i - 1)]
            b = self.windows[min(n - 1, i + 1)]
            dt = b.t_ref - a.t_ref
            w.vel = (b.pos - a.pos) / dt if dt > 1e-6 else np.zeros(3)

    def _finalize_stats(self) -> None:
        s = self.stats
        ok = [w for w in self.windows if w.ok]
        s.registered = len(ok)
        s.failed = len(self.windows) - len(ok)
        if ok:
            s.rmse_median = float(np.nanmedian([w.rmse for w in ok]))
            s.ratio_median = float(np.median([w.ratio for w in ok]))
            s.suppressed_mean = float(np.mean([w.suppressed for w in ok]))
        pos = np.array([w.pos for w in self.windows])
        s.path_length = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())

    # -- output ------------------------------------------------------------- #
    def batch_poses(self, frames) -> tuple[np.ndarray, np.ndarray]:
        """One ``(position, quaternion)`` per input batch.

        Positions are interpolated between window solutions rather than
        extrapolated on velocity, so the trajectory has no step at each window
        boundary — those steps are what smear the fused surface.
        """
        n = len(frames)
        ref_q = None
        for fr in frames:
            if fr.orientation is not None:
                ref_q = np.asarray(fr.orientation, dtype=np.float64)
                break
        ref_conj = quat_conjugate(ref_q / np.linalg.norm(ref_q))

        wt = np.array([w.t_ref for w in self.windows])
        wpos = np.array([w.pos for w in self.windows])
        wquat = np.array([matrix_to_quat(w.rot) for w in self.windows])

        positions = np.zeros((n, 3))
        quats = np.zeros((n, 4))
        last_rot = np.eye(3)
        t0 = frames[0].timestamp
        for i, fr in enumerate(frames):
            if fr.orientation is not None:
                q = np.asarray(fr.orientation, dtype=np.float64)
                last_rot = quat_to_matrix(quat_multiply(ref_conj, q / np.linalg.norm(q)))
            t = fr.timestamp - t0
            if self.cfg.smooth_batch_poses:
                pos, corr = _interp_pose(wt, wpos, wquat, t)
            else:
                j = int(np.clip(np.searchsorted(wt, t), 0, len(wt) - 1))
                pos, corr = wpos[j], quat_to_matrix(wquat[j])
            positions[i] = pos
            quats[i] = matrix_to_quat(corr @ last_rot)
        return positions, quats


def voxel_downsample_timed(points: np.ndarray, times: np.ndarray, voxel: float):
    """Voxel-downsample, returning each cell's centroid *and* mean timestamp."""
    if points.shape[0] == 0:
        return points, times
    from rocklabel.slam.voxelmap import encode

    keys = encode(np.floor(points / voxel).astype(np.int64))
    uniq, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv, minlength=uniq.shape[0]).astype(np.float64)
    cen = np.empty((uniq.shape[0], 3))
    for k in range(3):
        cen[:, k] = np.bincount(inv, weights=points[:, k], minlength=uniq.shape[0]) / cnt
    tm = np.bincount(inv, weights=times, minlength=uniq.shape[0]) / cnt
    return cen, tm


def _interp_pose(wt: np.ndarray, wpos: np.ndarray, wquat: np.ndarray, t: float):
    """Linear position / SLERP rotation between the two bracketing windows."""
    j = int(np.searchsorted(wt, t))
    if j <= 0:
        return wpos[0], quat_to_matrix(wquat[0])
    if j >= len(wt):
        return wpos[-1], quat_to_matrix(wquat[-1])
    t0, t1 = wt[j - 1], wt[j]
    a = 0.0 if t1 <= t0 else float((t - t0) / (t1 - t0))
    pos = wpos[j - 1] * (1.0 - a) + wpos[j] * a
    return pos, quat_to_matrix(_slerp(wquat[j - 1], wquat[j], a))


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Shortest-arc spherical interpolation between two unit quaternions."""
    d = float(q0 @ q1)
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:  # nearly identical: straight blend is numerically safer
        q = q0 * (1.0 - a) + q1 * a
        return q / np.linalg.norm(q)
    th = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(th)
    return (q0 * math.sin((1.0 - a) * th) + q1 * math.sin(a * th)) / s
