"""Gravity leveling of the live world frame (viz-free, pure NumPy).

Without this, the world frame is simply *the sensor frame at startup*
(:class:`~rocklabel.live.motion.OrientationTracker` takes the first IMU
quaternion as its reference). That is harmless for a hand-held rig held
roughly upright, but on a robot whose LiDAR is bolted to a slanted mast the
whole world is permanently tilted by the mount angle, and everything
downstream breaks in the same way:

* the **z-band crop** slices a diagonal wedge out of the floor instead of a
  horizontal slab, so most of the ground never reaches the heightmap;
* the **2.5D Kalman heightmap** sees the floor as a ramp — one height per
  (x, y) column is the wrong model for a tilted plane, so the surface folds
  into the saddle/spike mess a tilted run produces;
* the **model** gets neighborhoods whose ``dz`` (height above the ball's
  lowest point) is dominated by the tilt: a flat floor inside a 0.5 m ball
  tilted 30° looks like 0.25 m of relief, i.e. like a rock.

So the world frame must be levelled *before* any of that. Two independent
sources of truth are available and this module uses both:

1. **The IMU.** The multiScan's orientation quaternion is gravity-referenced
   (absolute roll/pitch, arbitrary yaw), so the up-vector expressed in the
   startup world frame is ``R(q_ref)^T ẑ`` — available from the very first
   telegram, and it keeps tracking as the robot pitches over obstacles
   because the *dynamic* part of the rotation is already handled upstream.
2. **The ground itself.** A RANSAC plane fit over a few seconds of pooled
   points measures the floor directly. On this rig the IMU seed leaves a
   consistent ~2° residual (the IMU's axes are not perfectly aligned with the
   optical frame), which the fit removes — and it also yields the floor
   height, which is what the crop band actually needs to be anchored to.

The result is a single constant rotation applied to every world-frame batch,
plus the measured floor height. Constant matters: SLAM's map and the fused
heightmap are built in the levelled frame, so the transform may not drift
underneath them — hence the explicit collect → fit → **lock** state machine
(re-run it deliberately with :meth:`GroundLeveler.recalibrate`).
"""

from __future__ import annotations

import math
import threading

import numpy as np

from rocklabel.live.config import LevelConfig

_Z = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# Rotation helpers
# --------------------------------------------------------------------------- #
def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit vector ``a`` onto unit vector ``b``.

    "Minimal" = rotation about ``a × b``, so no spurious yaw is introduced:
    levelling must not spin the map around.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        # Antiparallel: rotate 180° about any axis perpendicular to a.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def level_rotation_for_up(up: np.ndarray) -> np.ndarray:
    """Rotation that brings the measured up-vector ``up`` onto +z (yaw-free)."""
    u = np.asarray(up, dtype=np.float64)
    n = float(np.linalg.norm(u))
    if n < 1e-9:
        return np.eye(3)
    u = u / n
    if u[2] < 0.0:
        u = -u
    return rotation_between(u, _Z)


def mount_rotation(roll_deg: float, pitch_deg: float) -> np.ndarray:
    """Levelling rotation for a sensor bolted at ``roll``/``pitch`` degrees.

    The angles use the same convention as the IMU's roll/pitch readout: a
    sensor pitched nose-up by θ about its own +y axis needs ``Ry(θ)`` to bring
    its points back to level, so ``--mount-pitch`` can be read straight off the
    IMU (or off :meth:`GroundLeveler.status`).
    """
    r = math.radians(float(roll_deg))
    p = math.radians(float(pitch_deg))
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, math.cos(r), -math.sin(r)],
                   [0.0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0.0, math.sin(p)],
                   [0.0, 1.0, 0.0],
                   [-math.sin(p), 0.0, math.cos(p)]])
    return ry @ rx


def tilt_deg(rot: np.ndarray) -> float:
    """How far ``rot`` tips the z axis, in degrees."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))


def roll_pitch_deg(rot: np.ndarray) -> tuple[float, float]:
    """``(roll, pitch)`` in degrees of a levelling rotation (ZYX convention)."""
    pitch = math.asin(max(-1.0, min(1.0, float(-rot[2, 0]))))
    roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
    return math.degrees(roll), math.degrees(pitch)


# --------------------------------------------------------------------------- #
# Ground-plane fit
# --------------------------------------------------------------------------- #
def fit_ground_plane(
    points: np.ndarray,
    *,
    thresh: float = 0.04,
    iterations: int = 128,
    max_tilt_deg: float = 50.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, float] | None:
    """RANSAC + least-squares ground-plane fit of ``(N, 3)`` points.

    Only planes whose normal is within ``max_tilt_deg`` of +z are considered,
    which is what keeps the fit off walls: indoors a wall usually has *more*
    coplanar points than the floor, so an unconstrained "largest plane" fit
    happily locks onto one.

    Returns ``(normal, d, inlier_fraction)`` with ``normal`` unit-length and
    pointing up (``normal · p + d = 0`` on the plane), or None if no plane
    passes the gate.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n < 3:
        return None
    rng = rng or np.random.default_rng(0)
    cos_gate = math.cos(math.radians(max_tilt_deg))

    # Score every hypothesis in one shot: sample all triples up front, keep the
    # ones whose normal survives the tilt gate, then evaluate them as a single
    # (n, k) distance matrix (a Python loop over iterations was ~10x slower).
    tri = rng.integers(0, n, size=(iterations, 3))
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    normals = np.cross(b - a, c - a)
    lens = np.linalg.norm(normals, axis=1)
    ok = lens > 1e-9
    if not np.any(ok):
        return None
    normals = normals[ok] / lens[ok, None]
    normals[normals[:, 2] < 0.0] *= -1.0  # orient up
    ok2 = normals[:, 2] >= cos_gate
    if not np.any(ok2):
        return None
    normals = normals[ok2]
    offsets = -np.einsum("ij,ij->i", normals, a[ok][ok2])

    # Cap the scoring set: the fit is over-determined long before this matters,
    # and the distance matrix is (n_score, n_hypotheses).
    score_pts = pts if n <= 20_000 else pts[rng.choice(n, 20_000, replace=False)]
    dist = np.abs(score_pts @ normals.T + offsets[None, :])
    counts = np.count_nonzero(dist < thresh, axis=0)
    best = int(np.argmax(counts))
    if counts[best] < 3:
        return None
    normal, d = normals[best], float(offsets[best])

    # Refine on the inliers (total least squares) — RANSAC picks the right
    # points, the SVD gets the angle right to a fraction of a degree.
    for _ in range(3):
        inl = np.abs(pts @ normal + d) < thresh
        if int(inl.sum()) < 3:
            break
        q = pts[inl]
        centroid = q.mean(axis=0)
        _, _, vt = np.linalg.svd(q - centroid, full_matrices=False)
        cand = vt[2]
        if cand[2] < 0.0:
            cand = -cand
        if cand[2] < cos_gate:
            break
        normal, d = cand, float(-cand @ centroid)

    frac = float(np.count_nonzero(np.abs(pts @ normal + d) < thresh) / n)
    return normal, d, frac


# --------------------------------------------------------------------------- #
# Leveler
# --------------------------------------------------------------------------- #
class GroundLeveler:
    """Estimates, then holds, the constant world→level rotation + floor height.

    Lifecycle (all driven from the ingest thread):

    ``seed_from_imu`` (optional, instant) → ``observe`` per batch while
    collecting → automatic fit + **lock** once enough data has arrived. Readers
    (``rotation``, ``floor_z``) are lock-protected and safe from any thread.
    """

    OFF = "off"
    COLLECTING = "collecting"
    LOCKED = "locked"

    def __init__(self, cfg: LevelConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(0)
        self._rot = np.eye(3)
        self._floor_z: float | None = None
        self._inlier_frac = 0.0
        self._source = "none"
        self._seeded = False
        self._buf: list[np.ndarray] = []
        self._buf_n = 0
        self._t_start: float | None = None
        self._next_fit = -np.inf
        self._note = ""

        mode = (cfg.mode or "auto").lower()
        self._mode = mode
        #: Whether the ground fit is allowed to set the *angle*. When False
        #: ("imu", "manual") the fit still runs, but only to measure the floor
        #: height — which no IMU and no tape measure of mount angle can give
        #: you, and which is what a --floor-band is anchored to.
        self._fit_angle = mode not in ("imu", "manual")
        if mode == "off":
            self._state = self.OFF
        else:
            self._state = self.COLLECTING
            if mode == "manual":
                self._rot = mount_rotation(cfg.mount_roll_deg, cfg.mount_pitch_deg)
                self._source = "manual"
                self._seeded = True

    # -- readers ------------------------------------------------------------- #
    @property
    def rotation(self) -> np.ndarray:
        """Current world→level rotation (identity until anything is known)."""
        with self._lock:
            return self._rot

    @property
    def floor_z(self) -> float | None:
        """Height of the fitted floor plane in the levelled frame, or None."""
        with self._lock:
            return self._floor_z

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def locked(self) -> bool:
        with self._lock:
            return self._state in (self.LOCKED, self.OFF)

    @property
    def active(self) -> bool:
        """False only in ``mode="off"`` (nothing is ever rotated)."""
        return self._state != self.OFF

    def status(self) -> str:
        """One-line summary for the stats readout."""
        with self._lock:
            state, rot, fz = self._state, self._rot, self._floor_z
            src, frac, note = self._source, self._inlier_frac, self._note
        if state == self.OFF:
            return "level: off (world frame = sensor frame at start)"
        roll, pitch = roll_pitch_deg(rot)
        floor = f"floor z{fz:+.2f} m" if fz is not None else "floor unknown"
        if state == self.COLLECTING:
            seed = f"seed roll{roll:+.1f}° pitch{pitch:+.1f}°" if self._seeded else "no seed yet"
            return f"level: collecting… ({seed})"
        extra = f" {int(frac * 100)}% inliers" if src == "ground" else ""
        tail = f"  [{note}]" if note else ""
        return (
            f"level: {src} roll{roll:+.1f}° pitch{pitch:+.1f}° "
            f"(tilt {tilt_deg(rot):.1f}°){extra}, {floor}{tail}"
        )

    # -- estimation ---------------------------------------------------------- #
    def recalibrate(self) -> None:
        """Drop the current estimate and collect a fresh one (GUI 'L' key)."""
        if self._mode in ("off", "manual"):
            return
        with self._lock:
            self._state = self.COLLECTING
            self._seeded = False
            self._floor_z = None
            self._source = "none"
            self._note = ""
            self._buf.clear()
            self._buf_n = 0
            self._t_start = None
            self._next_fit = -np.inf

    def seed_from_imu(self, quat_ref_rotation: np.ndarray) -> None:
        """Seed the levelling rotation from the IMU's startup orientation.

        ``quat_ref_rotation`` is ``R(q_ref)``, the gravity-referenced rotation
        of the *first* IMU quaternion — the one that defines the world frame.
        Gravity-up expressed in that frame is its third row, so the seed is the
        minimal rotation carrying that back onto +z.
        """
        if self._mode in ("off", "manual", "ground"):
            return
        with self._lock:
            if self._seeded or self._state != self.COLLECTING:
                return
            up_in_world = np.asarray(quat_ref_rotation, dtype=np.float64).T @ _Z
            self._rot = level_rotation_for_up(up_in_world)
            self._seeded = True
            self._source = "imu"

    def observe(self, level_points: np.ndarray, timestamp: float) -> bool:
        """Pool one batch of already-levelled world points; fit when ready.

        Returns True on the transition to LOCKED, so the caller can flush any
        state that was fused with a stale (or identity) rotation.
        """
        cfg = self._cfg
        with self._lock:
            if self._state != self.COLLECTING:
                return False
            if self._t_start is None:
                self._t_start = timestamp

        # Keep only a plausible ground shell: far enough out to be floor rather
        # than the robot's own frame, near enough to be dense and flat.
        pts = np.asarray(level_points, dtype=np.float64)
        if pts.shape[0] == 0:
            return False
        pts = pts[:: max(1, int(cfg.sample_stride))]
        r2 = np.einsum("ij,ij->i", pts, pts)
        sel = (r2 >= cfg.range_min**2) & (r2 <= cfg.range_max**2)
        with self._lock:
            seeded = self._seeded
        if seeded:
            # Frame is already within a couple of degrees of level, so "below
            # the sensor" is a meaningful and very effective ceiling reject.
            # Unseeded, the mount angle can put real floor points metres above
            # z=0 — the tilt gate and the below-the-sensor check on the fitted
            # plane do that job instead.
            sel &= pts[:, 2] < cfg.ceiling_margin
        pts = pts[sel]

        with self._lock:
            if pts.shape[0]:
                self._buf.append(pts)
                self._buf_n += pts.shape[0]
                # Bound the pool: a fit that keeps failing must not grow it
                # without limit while the app runs.
                while self._buf_n > cfg.max_points and len(self._buf) > 1:
                    self._buf_n -= self._buf.pop(0).shape[0]
            # `or` would be wrong here: a t_start of exactly 0.0 is falsy, and
            # a clock that starts at zero would then never advance past it.
            elapsed = timestamp - (timestamp if self._t_start is None else self._t_start)
            timed_out = elapsed >= cfg.calib_timeout_sec
            ready = (
                elapsed >= cfg.calib_sec
                and self._buf_n >= cfg.min_points
                and timestamp >= self._next_fit
            )
            if not ready and not timed_out:
                return False
            pooled = np.concatenate(self._buf) if self._buf else np.empty((0, 3))
            # Keep the pool on failure and retry later with *more* data, rather
            # than refitting a single fresh batch on every one of the ~240
            # batches a second the sensor delivers.
            self._next_fit = timestamp + cfg.calib_sec

        fit = fit_ground_plane(
            pooled,
            thresh=cfg.plane_thresh,
            iterations=cfg.ransac_iters,
            max_tilt_deg=cfg.max_tilt_deg,
            rng=self._rng,
        ) if pooled.shape[0] >= 3 else None

        # A ceiling is just as planar and just as level as a floor, so accept a
        # fit only when it passes below the sensor.
        if fit is not None:
            nz = float(fit[0][2])
            plane_z = -fit[1] / nz if abs(nz) > 1e-6 else float("inf")
            if plane_z > cfg.ceiling_margin:
                fit = None

        accepted = (
            fit is not None
            and fit[2] >= cfg.min_inlier_frac
            and fit[2] * pooled.shape[0] >= cfg.min_inliers
        )
        with self._lock:
            # A floor height read off a frame that is still tilted is not a
            # height at all, so height-only modes need a settled angle first.
            usable = accepted and (self._fit_angle or self._seeded)
            if usable:
                normal, d, frac = fit
                if self._fit_angle:
                    # Compose: the fit was measured in the already-seeded frame.
                    self._rot = level_rotation_for_up(normal) @ self._rot
                    self._source = "ground"
                    self._inlier_frac = frac
                # The residual rotates about the origin, which preserves the
                # plane's perpendicular distance from it; once the normal is
                # +z that distance *is* the floor height.
                self._floor_z = float(-d)
                self._state = self.LOCKED
                self._buf.clear()
                self._buf_n = 0
                return True
            if timed_out:
                self._state = self.LOCKED
                self._buf.clear()
                self._buf_n = 0
                self._note = (
                    "ground fit failed — kept the seeded angle" if self._seeded
                    else "ground fit failed — frame left unlevelled"
                )
                return True
            return False
