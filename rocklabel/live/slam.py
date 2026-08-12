"""Scan-to-map LiDAR odometry (lightweight SLAM) — viz-free, pure NumPy.

Lets the sensor *move* through the room (translation + rotation) while points
keep landing in a fixed world frame, in the spirit of KISS-ICP:

* A sparse **voxel hash map** holds a downsampled point cloud of everything
  registered so far (one running-mean centroid per voxel, vectorized lookups
  via ``np.searchsorted`` on sorted int64 keys — no Python dicts in the hot
  path).
* The sensor's **IMU quaternion is the rotation prior** (excellent short-term),
  so ICP only has to solve for translation and slow yaw drift.
* Incoming batches are pooled into short **registration windows** (~0.1 s of
  data, one or two sensor revolutions' worth of geometry); each window is
  voxel-downsampled and aligned to the map with a few iterations of
  point-to-voxel-centroid ICP, then merged into the map.

Registration runs on the *full* 3D cloud — walls and ceiling are exactly the
structure that constrains x/y translation — while the caller keeps cropping to
the floor band for heightmap fusion.

Limitations (documented, deliberate): no loop closure, so pose drifts slowly
over long runs; dynamic objects (people walking through) get baked into the map
until a reset; and the motion between two windows must stay within the ICP
correspondence radius (~0.4 m), i.e. move the sensor at a walking pace or
slower.
"""

from __future__ import annotations

import math
import time

import numpy as np

from rocklabel.live.config import MotionConfig, SlamConfig
from rocklabel.live.motion import OrientationTracker, yaw_from_matrix

# 21 bits per axis packed into an int64; supports |coord| < 2^20 voxels.
_BITS = 21
_OFF = 1 << 20
_LIM = (1 << _BITS) - 1

#: Face-adjacent neighborhood searched during correspondence lookup, expressed
#: directly as int64 key deltas: for packed keys, the neighbor at (dx, dy, dz)
#: is ``key + dx*2^42 + dy*2^21 + dz`` — no re-encode needed per offset.
_NEIGHBOR_DELTAS = tuple(
    int(dx * (1 << (2 * _BITS)) + dy * (1 << _BITS) + dz)
    for dx, dy, dz in
    [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
)


def _encode(ijk: np.ndarray) -> np.ndarray:
    """Pack integer voxel coords ``(N, 3)`` into sortable int64 keys."""
    c = np.clip(ijk + _OFF, 0, _LIM)
    return (c[:, 0] << (2 * _BITS)) | (c[:, 1] << _BITS) | c[:, 2]


def _group_means(points: np.ndarray, inv: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-group centroid + count via ``bincount`` (much faster than add.at)."""
    counts = np.bincount(inv, minlength=n_groups)
    sums = np.empty((n_groups, 3), dtype=np.float64)
    for k in range(3):
        sums[:, k] = np.bincount(inv, weights=points[:, k], minlength=n_groups)
    return sums / counts[:, None].astype(np.float64), counts


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Reduce ``(N, 3)`` points to one centroid per ``voxel``-sized cell."""
    if points.shape[0] == 0:
        return points
    ijk = np.floor(points / voxel).astype(np.int64)
    keys = _encode(ijk)
    uniq, inv = np.unique(keys, return_inverse=True)
    means, _ = _group_means(points, inv, uniq.shape[0])
    return means


class VoxelHashMap:
    """Sparse world model: one running-mean centroid + hit count per voxel.

    Keys are kept sorted so both :meth:`query` and :meth:`insert` are fully
    vectorized (``searchsorted`` + merges); no per-point Python loops.
    """

    def __init__(self, voxel_size: float, max_hits: int = 30, capacity: int = 1_000_000) -> None:
        self._vs = float(voxel_size)
        self._max_hits = int(max_hits)
        self._capacity = int(capacity)
        self._keys = np.empty(0, dtype=np.int64)  # sorted
        self._centroids = np.empty((0, 3), dtype=np.float64)
        self._counts = np.empty(0, dtype=np.int64)

    @property
    def size(self) -> int:
        """Number of occupied voxels."""
        return int(self._keys.shape[0])

    def query(self, points: np.ndarray, max_dist: float) -> tuple[np.ndarray, np.ndarray]:
        """Nearest map centroid per point within ``max_dist``.

        Searches the point's own voxel plus the 6 face neighbors.

        Returns:
            ``(valid (N,) bool, targets (N, 3))``; ``targets`` rows are only
            meaningful where ``valid`` is True.
        """
        n = points.shape[0]
        targets = np.zeros((n, 3), dtype=np.float64)
        if n == 0 or self.size == 0:
            return np.zeros(n, dtype=bool), targets

        base = np.floor(points / self._vs).astype(np.int64)
        best_d2 = np.full(n, np.inf)
        best_idx = np.full(n, -1, dtype=np.int64)
        last = self.size - 1
        base_keys = _encode(base)
        for delta in _NEIGHBOR_DELTAS:
            keys = base_keys if delta == 0 else base_keys + delta
            pos = np.minimum(np.searchsorted(self._keys, keys), last)
            # Only compute distances for actual key hits (most neighbor offsets
            # miss for most points — skip their distance math entirely).
            hit_idx = np.nonzero(self._keys[pos] == keys)[0]
            if hit_idx.size == 0:
                continue
            cand = pos[hit_idx]
            diff = points[hit_idx] - self._centroids[cand]
            d2 = np.einsum("ij,ij->i", diff, diff)
            better = d2 < best_d2[hit_idx]
            upd = hit_idx[better]
            best_d2[upd] = d2[better]
            best_idx[upd] = cand[better]

        valid = (best_idx >= 0) & (best_d2 <= max_dist * max_dist)
        targets[valid] = self._centroids[best_idx[valid]]
        return valid, targets

    def insert(self, points: np.ndarray) -> None:
        """Merge points into the map (running-mean per voxel, capped hits)."""
        if points.shape[0] == 0 or self.size >= self._capacity:
            return
        ijk = np.floor(points / self._vs).astype(np.int64)
        keys = _encode(ijk)
        uniq, inv = np.unique(keys, return_inverse=True)
        means, cnts = _group_means(points, inv, uniq.shape[0])

        if self.size:
            pos = np.searchsorted(self._keys, uniq)
            pos_c = np.minimum(pos, self.size - 1)
            exists = self._keys[pos_c] == uniq
        else:
            pos_c = np.zeros(uniq.shape[0], dtype=np.int64)
            exists = np.zeros(uniq.shape[0], dtype=bool)

        # Update existing voxels with a count-weighted running mean; freeze
        # centroids once max_hits is reached so the map stays stable.
        if np.any(exists):
            idx = pos_c[exists]
            old_c = self._counts[idx].astype(np.float64)
            add_c = cnts[exists].astype(np.float64)
            upd = self._counts[idx] < self._max_hits
            w_new = add_c / (old_c + add_c)
            blended = self._centroids[idx] * (1.0 - w_new[:, None]) + means[exists] * w_new[:, None]
            self._centroids[idx[upd]] = blended[upd]
            self._counts[idx] = np.minimum(
                self._counts[idx] + cnts[exists], self._max_hits
            )

        # Splice brand-new voxels into the sorted arrays in O(n) — both sides
        # are already sorted, so a positional insert beats a full re-sort.
        new_mask = ~exists
        if np.any(new_mask):
            new_keys = uniq[new_mask]  # sorted (np.unique output)
            ins = np.searchsorted(self._keys, new_keys)
            self._keys = np.insert(self._keys, ins, new_keys)
            self._centroids = np.insert(self._centroids, ins, means[new_mask], axis=0)
            self._counts = np.insert(self._counts, ins, cnts[new_mask])


def register(
    points_world: np.ndarray,
    vmap: VoxelHashMap,
    iterations: int,
    dist_start: float,
    dist_end: float,
    rotation_mode: str,
    min_matches: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Align ``points_world`` (already at the predicted pose) to the map.

    Runs a fixed number of ICP iterations with a shrinking correspondence
    radius, estimating a *correction* transform ``(dR, dt)`` such that
    ``dR @ p + dt`` best matches the map.

    Args:
        rotation_mode: ``"yaw"`` (2D rotation about z + 3D translation, default
            — the IMU already pins roll/pitch via gravity), ``"full"`` (SE(3)
            Kabsch), or ``"none"`` (translation only).

    Returns:
        ``(dR (3,3), dt (3,), matched_ratio, converged)``. When too few
        correspondences exist (entering unmapped space), returns identity with
        ``converged=False``.
    """
    R = np.eye(3)
    t = np.zeros(3)
    ratio = 0.0
    n = points_world.shape[0]
    if n < min_matches or vmap.size == 0:
        return R, t, 0.0, False

    for dist in np.linspace(dist_start, dist_end, max(1, iterations)):
        p = points_world @ R.T + t
        valid, targets = vmap.query(p, dist)
        m = int(valid.sum())
        ratio = m / n
        if m < min_matches:
            return R, t, ratio, False
        ps = p[valid]
        qs = targets[valid]
        mu_p = ps.mean(axis=0)
        mu_q = qs.mean(axis=0)
        a = ps - mu_p
        b = qs - mu_q

        if rotation_mode == "none":
            dR = np.eye(3)
        elif rotation_mode == "full":
            h = a.T @ b
            u, _, vt = np.linalg.svd(h)
            d = np.sign(np.linalg.det(vt.T @ u.T))
            dR = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        else:  # "yaw": planar rotation about +z (2D Kabsch on x/y)
            s = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
            c = float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
            th = math.atan2(s, c)
            ct, st = math.cos(th), math.sin(th)
            dR = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])

        dt = mu_q - dR @ mu_p
        R = dR @ R
        t = dR @ t + dt

    return R, t, ratio, True


class SlamTracker:
    """Full pose tracking: IMU rotation prior + scan-to-map ICP for translation.

    Per batch, :meth:`process` transforms sensor-frame points into the world
    frame at the current pose estimate and pools a downsampled copy into the
    active registration window; when the window closes, ICP against the voxel
    map refines the pose (and the correction is applied to future batches).

    Pose model: ``R_world = C · R_imu``, where ``R_imu`` comes from the IMU
    relative to startup and ``C`` is the slowly-evolving ICP yaw/rotation
    correction; translation ``t`` comes entirely from ICP.
    """

    def __init__(self, slam_cfg: SlamConfig, motion_cfg: MotionConfig) -> None:
        self._cfg = slam_cfg
        self._use_imu = motion_cfg.use_imu
        self.imu = OrientationTracker(yaw_only=False)
        self.map = VoxelHashMap(
            slam_cfg.voxel_size, slam_cfg.map_max_hits, slam_cfg.map_capacity
        )
        self._C = np.eye(3)  # ICP rotation correction applied on top of the IMU
        self._t = np.zeros(3)  # anchor position, valid at self._t_time
        self._t_time: float | None = None
        self._v = np.zeros(3)  # constant-velocity model between window closes
        self._last_close_pos: np.ndarray | None = None
        self._last_close_time: float | None = None
        self._last_R = np.eye(3)
        self.match_ratio = 0.0
        self.windows_registered = 0
        self.windows_skipped = 0
        self._win_pts: list[np.ndarray] = []
        self._win_start: float | None = None
        self._last_now: float = 0.0

    # -- pose accessors ------------------------------------------------------ #
    @property
    def position(self) -> np.ndarray:
        """Sensor position (3,) in the world frame at the last processed batch
        (velocity-predicted between window closes)."""
        return self._pos_at(self._last_now) if self._last_now else self._t.copy()

    def _pos_at(self, now: float) -> np.ndarray:
        """Position predicted at time ``now`` via the constant-velocity model."""
        if self._t_time is None:
            return self._t
        return self._t + self._v * max(0.0, now - self._t_time)

    @property
    def yaw_deg(self) -> float:
        """Current sensor yaw in degrees (world frame)."""
        return math.degrees(yaw_from_matrix(self._last_R))

    @property
    def rotation(self) -> np.ndarray:
        """Sensor→world rotation matrix at the last processed batch."""
        return self._last_R

    def status(self) -> str:
        """One-line pose summary for the stats readout."""
        x, y, z = self.position
        return (
            f"x{x:+.2f} y{y:+.2f} z{z:+.2f} m  yaw {self.yaw_deg:+.1f}°  "
            f"match {self.match_ratio * 100.0:3.0f}%  map {self.map.size // 1000}k vox"
        )

    def reset(self) -> None:
        """Forget the map and pose; the next window re-anchors the world frame."""
        self.map = VoxelHashMap(
            self._cfg.voxel_size, self._cfg.map_max_hits, self._cfg.map_capacity
        )
        self.imu.reset()
        self._C = np.eye(3)
        self._t = np.zeros(3)
        self._t_time = None
        self._v = np.zeros(3)
        self._last_close_pos = None
        self._last_close_time = None
        self._last_R = np.eye(3)
        self.match_ratio = 0.0
        self._win_pts.clear()
        self._win_start = None
        self._last_now = 0.0

    # -- main entry ------------------------------------------------------------ #
    def process(
        self,
        points: np.ndarray,
        quat_wxyz: np.ndarray | None,
        timestamp: float = 0.0,
    ) -> np.ndarray:
        """Transform one sensor-frame batch into the world frame; advance SLAM.

        Args:
            points: ``(N, 3)`` sensor-frame points (uncropped — walls included).
            quat_wxyz: IMU orientation for this batch, or None to reuse the last.
            timestamp: batch timestamp in seconds (any monotonic-ish clock);
                falls back to ``time.time()`` when 0.

        Returns:
            ``(N, 3)`` world-frame points at the current pose estimate.
        """
        now = timestamp if timestamp else time.time()
        self._last_now = now

        if self._use_imu and quat_wxyz is not None:
            r_imu = self.imu.update(quat_wxyz)
        else:
            r_imu = self._imu_last()
        r_pred = self._C @ r_imu
        self._last_R = r_pred
        if self._t_time is None:
            self._t_time = now
        world = points @ r_pred.T + self._pos_at(now)

        # Pool a range-limited copy for registration. Downsampling happens once
        # per window at close time — one big np.unique beats hundreds of small
        # per-batch ones (this was ~1/4 of the whole SLAM budget).
        r2 = np.einsum("ij,ij->i", points, points)
        sel = (r2 >= self._cfg.reg_range_min**2) & (r2 <= self._cfg.reg_range_max**2)
        if np.any(sel):
            self._win_pts.append(world[sel])
            if self._win_start is None:
                self._win_start = now

        if self._win_start is not None and now - self._win_start >= self._cfg.window_sec:
            self._close_window(now)

        return world

    def _imu_last(self) -> np.ndarray:
        """Last IMU rotation (identity before any IMU sample)."""
        # OrientationTracker keeps its last rotation internally; reuse it via
        # a zero-point transform to avoid exposing private state.
        return self.imu._last_rot  # noqa: SLF001 — same package, hot path

    def _close_window(self, now: float) -> None:
        """Register the pooled window against the map, merge it in, and update
        the pose anchor + velocity model."""
        src = voxel_downsample(np.concatenate(self._win_pts, axis=0), self._cfg.voxel_size)
        self._win_pts.clear()
        self._win_start = None
        if src.shape[0] == 0:
            return

        pos_pred = self._pos_at(now)

        if self.map.size == 0:  # first window anchors the world frame
            self.map.insert(src)
            self._t, self._t_time = pos_pred, now
            self._last_close_pos, self._last_close_time = pos_pred.copy(), now
            return

        d_r, d_t, ratio, ok = register(
            src,
            self.map,
            iterations=self._cfg.icp_iterations,
            dist_start=self._cfg.max_corr_dist,
            dist_end=self._cfg.voxel_size,
            rotation_mode=self._cfg.rotation_mode,
            min_matches=self._cfg.min_matches,
        )
        self.match_ratio = ratio
        if ok:
            self._C = d_r @ self._C
            new_pos = d_r @ pos_pred + d_t
            self.map.insert(src @ d_r.T + d_t)
            self.windows_registered += 1
            # Update the constant-velocity model from successive window poses
            # (lightly smoothed, magnitude-capped for safety).
            if self._last_close_time is not None and now > self._last_close_time:
                v_inst = (new_pos - self._last_close_pos) / (now - self._last_close_time)
                # Warm-start: adopt the first estimate outright, then smooth.
                v = v_inst if not self._v.any() else 0.5 * self._v + 0.5 * v_inst
                speed = float(np.linalg.norm(v))
                if speed > 2.0:  # cap at a brisk walk; anything more is noise
                    v *= 2.0 / speed
                self._v = v
            self._last_close_pos, self._last_close_time = new_pos.copy(), now
            self._t, self._t_time = new_pos, now
        else:
            # Too little overlap (fast move into unmapped space): coast on the
            # prediction and still grow the map so we can re-acquire.
            self.map.insert(src)
            self.windows_skipped += 1
            self._t, self._t_time = pos_pred, now
