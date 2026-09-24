"""Causal scan history and radius policies for format-A neighborhoods.

The original builder (:mod:`rocklabel.dataset.neighborhoods`) cuts a fixed
0.5 m ball out of one assembled sweep. This module generalises the two things
that fixes:

* **Which points a ball may use.** Candidates still come from the current
  sweep only - that keeps every arm scoring the same locations - but the points
  a ball is built from can also come from older sweeps, picked by a fixed list
  of desired ages. Selection is causal (never a sweep newer than the one being
  scored), each sweep is used once, and a slot the recording cannot fill within
  a tolerance stays empty rather than borrowing a neighbour's sweep.
* **How big a ball is.** Fixed, as before, or adaptive: the distance to the
  K-th nearest support point, clipped to a band. The minimum keeps floor and
  edges around a dense rock; the maximum stops a sparse area from collecting
  distant floor just to fill the tensor.

There is one implementation of each step and every consumer calls it -
dataset generation, the visual audit, the whole-recording map evaluation and
the live scorer - so a checkpoint sees the same tensor at training and at
inference. Legacy checkpoints (``preprocessing_version`` 1) never reach this
module; they keep :func:`~rocklabel.dataset.neighborhoods.build_inference_samples`.

Torch-free, like the rest of ``rocklabel.dataset``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.accumulate import voxel_downsample_centroids
from .labeling import inside_arena
from .neighborhoods import QUERY_HEIGHT_CHANNEL, SAMPLE_CHANNELS

#: The builder this module implements, as recorded in configs and checkpoints.
PREPROCESSING_VERSION = 2
#: Radius slack, in metres, applied to adaptive ball queries. An adaptive
#: radius *is* the distance to a real point - the K-th neighbour - so that
#: point sits exactly on the boundary, and rounding between the distance query
#: and the ball query must not drop it. Fixed balls keep version 1's exact
#: test: a tolerance there would admit the odd point a micron outside 0.5 m,
#: shift every later random draw, and break bit-for-bit parity of the fresh
#: fixed-0.50 m baseline with the historical builder.
TIE_TOLERANCE_M = 1e-6
#: Voxel edge used to count *distinct* support locations in a ball. Two sweeps
#: of a stationary sensor land on the same spots; the measured-point count
#: doubles and this one does not.
DUPLICATE_VOXEL_M = 0.02
#: A sensor that moves further than this between consecutive sweeps has been
#: relocalised or replayed from elsewhere, not driven. History is dropped.
POSE_JUMP_M = 1.0
#: Stored row layout: the four FEATURES then the query height. Unchanged from
#: version 1, so every model and augmentation reads version-2 samples as is.
FEATURE_SCHEMA = "dx,dy,dz,intensity|query_height"
#: Sampling RNG stream id, mixed into the per-sweep seed. Candidate selection
#: uses ``[seed, index]`` exactly as version 1 did; point sampling uses
#: ``[seed, index, SAMPLE_STREAM]``, so a bigger ball consuming more random
#: numbers can never move which negatives were kept.
SAMPLE_STREAM = 1


def is_legacy(gcfg: dict) -> bool:
    """True for a generator config (or checkpoint) built by the version-1 path."""
    return int(gcfg.get("preprocessing_version", 1)) < PREPROCESSING_VERSION


@dataclass(frozen=True)
class RadiusPolicy:
    """How big a candidate's ball is."""

    mode: str
    radius_m: float
    min_m: float
    max_m: float
    k: int
    min_neighbors: int
    n_points: int

    @classmethod
    def from_generator(cls, g: dict) -> "RadiusPolicy":
        return cls(mode=str(g.get("neighborhood_mode", "fixed")),
                   radius_m=float(g["neighborhood_radius_m"]),
                   min_m=float(g.get("adaptive_radius_min_m", 0.2)),
                   max_m=float(g.get("adaptive_radius_max_m", 0.5)),
                   k=int(g.get("adaptive_k", 256)),
                   min_neighbors=int(g["min_neighbors"]),
                   n_points=int(g["neighborhood_points"]))

    @property
    def reach_m(self) -> float:
        """The largest ball this policy can ever cut."""
        return self.radius_m if self.mode == "fixed" else self.max_m

    def describe(self) -> str:
        if self.mode == "fixed":
            return f"fixed {self.radius_m:.2f} m"
        return (f"adaptive {self.min_m:.2f}-{self.max_m:.2f} m, "
                f"target {self.k} real points")


@dataclass(frozen=True)
class HistoryPolicy:
    """Which older sweeps supply neighborhood points."""

    ages_s: tuple[float, ...]
    tolerance_s: float

    @classmethod
    def from_generator(cls, g: dict) -> "HistoryPolicy":
        return cls(ages_s=tuple(float(a) for a in (g.get("history_ages_s") or [0.0])),
                   tolerance_s=float(g.get("history_tolerance_s", 0.10)))

    @property
    def max_age_s(self) -> float:
        return float(max(self.ages_s))

    @property
    def uses_history(self) -> bool:
        return len(self.ages_s) > 1

    @property
    def retention_s(self) -> float:
        """How long a buffer must keep a sweep to fill every slot."""
        return self.max_age_s + self.tolerance_s + 0.5

    def key(self) -> tuple:
        return (self.ages_s, self.tolerance_s)

    def describe(self) -> str:
        if not self.uses_history:
            return "current sweep only"
        return ("sweeps aged " + ", ".join(f"{a:g}" for a in self.ages_s)
                + f" s (tolerance {self.tolerance_s:g} s)")


def input_contract(gcfg: dict) -> dict:
    """Everything a checkpoint's input depends on, for reports and manifests."""
    if is_legacy(gcfg):
        return {"preprocessing_version": 1,
                "radius": f"fixed {float(gcfg['neighborhood_radius_m']):.2f} m",
                "history": "current sweep only",
                "frame_window_s": float(gcfg.get("frame_window_s") or 0.0),
                "feature_schema": FEATURE_SCHEMA}
    r, h = RadiusPolicy.from_generator(gcfg), HistoryPolicy.from_generator(gcfg)
    return {"preprocessing_version": PREPROCESSING_VERSION,
            "radius": r.describe(), "neighborhood_mode": r.mode,
            "radius_m": r.radius_m, "adaptive_radius_min_m": r.min_m,
            "adaptive_radius_max_m": r.max_m, "adaptive_k": r.k,
            "history": h.describe(), "history_ages_s": list(h.ages_s),
            "history_tolerance_s": h.tolerance_s,
            "frame_window_s": float(gcfg.get("frame_window_s") or 0.0),
            "feature_schema": FEATURE_SCHEMA}


def sample_rng(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(index), SAMPLE_STREAM])


def candidate_rng(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(index)])


# --------------------------------------------------------------------------- #
# Sweeps and history
# --------------------------------------------------------------------------- #
@dataclass
class Sweep:
    """One assembled sensor sweep in the world frame."""

    sweep_id: int
    time_s: float
    xyz: np.ndarray
    intensity: np.ndarray
    #: Sensor position the sweep was measured from.
    origin: np.ndarray
    #: Heading of the robot base, radians. For view-diversity diagnostics.
    yaw: float = 0.0

    @classmethod
    def from_scan(cls, scan) -> "Sweep":
        """From an :class:`~rocklabel.recording.pipeline.OdomScan`."""
        rot = np.asarray(scan.T_odom_base)[:3, :3]
        return cls(int(scan.index), float(scan.time_s), scan.xyz_odom,
                   np.asarray(scan.intensity, np.float32),
                   np.asarray(scan.T_odom_lidar)[:3, 3].astype(np.float64),
                   float(math.atan2(rot[1, 0], rot[0, 0])))

    @property
    def n_points(self) -> int:
        return int(len(self.xyz))


@dataclass
class HistorySlot:
    """One desired age and the sweep that filled it, if any."""

    target_age_s: float
    sweep: Sweep | None
    #: ``now - sweep.time_s``; NaN when the slot is empty.
    age_s: float
    #: ``ok``, ``before_start`` (the recording has not run that long),
    #: ``gap`` (nearest older sweep is beyond tolerance) or ``duplicate``
    #: (that sweep already fills a younger slot).
    status: str


def select_slots(sweeps: list[Sweep], ages_s, tolerance_s: float) -> list[HistorySlot]:
    """Causal, unique history selection with the newest sweep as "now".

    ``sweeps`` must be in time order and end with the current sweep. For each
    desired age the latest sweep at or before ``now - age`` is taken; it must be
    no more than ``tolerance_s`` older than asked for, and no sweep fills two
    slots. The current sweep is always slot 0.
    """
    if not sweeps:
        raise ValueError("history selection needs at least the current sweep")
    now = sweeps[-1].time_s
    times = np.fromiter((s.time_s for s in sweeps), float, len(sweeps))
    used: set[int] = set()
    slots: list[HistorySlot] = []
    for age in ages_s:
        age = float(age)
        if age == 0.0:
            cur = sweeps[-1]
            used.add(cur.sweep_id)
            slots.append(HistorySlot(0.0, cur, 0.0, "ok"))
            continue
        target = now - age
        # At or before the target. The epsilon only absorbs float noise in
        # "exactly age seconds ago"; it can never admit a newer sweep than that.
        j = int(np.searchsorted(times, target + 1e-9, side="right")) - 1
        if j < 0:
            slots.append(HistorySlot(age, None, float("nan"), "before_start"))
            continue
        s = sweeps[j]
        actual = now - s.time_s
        if actual - age > tolerance_s + 1e-9:
            slots.append(HistorySlot(age, None, float("nan"), "gap"))
        elif s.sweep_id in used:
            slots.append(HistorySlot(age, None, float("nan"), "duplicate"))
        else:
            used.add(s.sweep_id)
            slots.append(HistorySlot(age, s, float(actual), "ok"))
    return slots


class SweepHistory:
    """Bounded, causal buffer of assembled sweeps.

    Keeps every sweep younger than ``retention_s`` behind the newest one - any
    of them can become a target once the clock moves on - plus a hard count cap
    so a misbehaving source cannot grow it without bound. Time running
    backwards, a repeated stamp or a pose jump empties it: a replay seek or a
    relocalisation is a new history, never a continuation of the old one.
    """

    def __init__(self, retention_s: float, max_sweeps: int = 4096,
                 pose_jump_m: float = POSE_JUMP_M) -> None:
        self.retention_s = float(retention_s)
        self.max_sweeps = int(max_sweeps)
        self.pose_jump_m = float(pose_jump_m)
        self._sweeps: deque[Sweep] = deque()
        #: How many times the buffer was emptied for a discontinuity.
        self.resets = 0

    def __len__(self) -> int:
        return len(self._sweeps)

    @property
    def points_held(self) -> int:
        return sum(s.n_points for s in self._sweeps)

    @property
    def newest(self) -> Sweep | None:
        return self._sweeps[-1] if self._sweeps else None

    def reset(self) -> None:
        self._sweeps.clear()

    def push(self, sweep: Sweep) -> bool:
        """Append the newest sweep; True when a discontinuity reset the buffer."""
        reset = False
        last = self.newest
        if last is not None and (
                sweep.time_s <= last.time_s
                or float(np.linalg.norm(np.asarray(sweep.origin) - last.origin))
                > self.pose_jump_m):
            self._sweeps.clear()
            self.resets += 1
            reset = True
        self._sweeps.append(sweep)
        horizon = sweep.time_s - self.retention_s
        while self._sweeps and (self._sweeps[0].time_s < horizon
                                or len(self._sweeps) > self.max_sweeps):
            self._sweeps.popleft()
        return reset

    def select(self, policy: HistoryPolicy) -> list[HistorySlot]:
        return select_slots(list(self._sweeps), policy.ages_s, policy.tolerance_s)


class SweepAssembler:
    """Group raw sensor batches into sweeps, as WindowedScanStream does offline.

    A sweep closes when a batch arrives ``window_s`` or more after the sweep's
    first batch; its time and pose are the last member's. Used by the live
    engine, which receives batches rather than an iterator.
    """

    def __init__(self, window_s: float) -> None:
        self.window_s = float(window_s)
        self._buf: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
        self._next_id = 0

    def reset(self) -> None:
        self._buf = []

    def add(self, time_s: float, xyz: np.ndarray, intensity: np.ndarray | None,
            origin: np.ndarray, yaw: float = 0.0) -> Sweep | None:
        """Add one batch; returns the sweep this batch completed, if any."""
        done = None
        if self._buf and (time_s < self._buf[-1][0]):
            self._buf = []                     # clock went backwards: drop it
        if self._buf and time_s - self._buf[0][0] >= self.window_s:
            done = self._close()
        # NaN where the source gave no intensity, as the live scoring window
        # marks it; the scorer zeroes non-finite values before inference.
        inten = (np.full(len(xyz), np.nan, np.float32)
                 if intensity is None or len(intensity) != len(xyz)
                 else np.asarray(intensity, np.float32))
        self._buf.append((float(time_s), np.asarray(xyz), inten,
                          np.asarray(origin, np.float64), float(yaw)))
        if self.window_s <= 0.0:
            done = self._close()
        return done

    def _close(self) -> Sweep:
        buf, self._buf = self._buf, []
        last = buf[-1]
        sweep = Sweep(self._next_id, last[0],
                      np.concatenate([b[1] for b in buf]),
                      np.concatenate([b[2] for b in buf]), last[3], last[4])
        self._next_id += 1
        return sweep


# --------------------------------------------------------------------------- #
# Support clouds
# --------------------------------------------------------------------------- #
@dataclass
class Support:
    """The points a batch of balls is cut from, with where each came from."""

    xyz: np.ndarray
    intensity: np.ndarray
    #: Seconds between the point's sweep and the current sweep.
    age: np.ndarray
    #: Which history slot supplied the point (0 = current sweep).
    slot: np.ndarray
    #: Per slot: the slot record and how many of its points survived the crop.
    slots: list[HistorySlot] = field(default_factory=list)
    slot_points: list[int] = field(default_factory=list)

    @property
    def current(self) -> np.ndarray:
        return self.slot == 0

    def __len__(self) -> int:
        return int(len(self.xyz))


def assemble_support(slots: list[HistorySlot], crop) -> Support:
    """Concatenate the filled slots, each cropped by ``crop(xyz) -> mask``.

    ``crop`` is built from the *current* pose: an older sweep contributes only
    what lies in today's box, transformed by its own pose (already world-frame).
    """
    xyz, inten, age, slot, counts = [], [], [], [], []
    for i, s in enumerate(slots):
        if s.sweep is None or s.sweep.n_points == 0:
            counts.append(0)
            continue
        keep = crop(s.sweep.xyz)
        n = int(np.count_nonzero(keep))
        counts.append(n)
        if not n:
            continue
        # Kept in the sweep's own precision: voxel centroids and crop tests on
        # float32 input are what version 1 computed, and the h1 arms must match it.
        xyz.append(np.asarray(s.sweep.xyz[keep]))
        inten.append(np.asarray(s.sweep.intensity[keep], np.float32))
        age.append(np.full(n, s.age_s, np.float32))
        slot.append(np.full(n, i, np.uint8))
    if not xyz:
        return Support(np.empty((0, 3)), np.empty(0, np.float32), np.empty(0, np.float32),
                       np.empty(0, np.uint8), list(slots), counts)
    return Support(np.concatenate(xyz), np.concatenate(inten), np.concatenate(age),
                   np.concatenate(slot), list(slots), counts)


def single_sweep_support(xyz: np.ndarray, intensity: np.ndarray) -> Support:
    """A support cloud made of the current sweep alone (history policy h1)."""
    n = len(xyz)
    return Support(np.asarray(xyz), np.asarray(intensity, np.float32),
                   np.zeros(n, np.float32), np.zeros(n, np.uint8),
                   [HistorySlot(0.0, None, 0.0, "ok")], [n])


# --------------------------------------------------------------------------- #
# Neighborhoods
# --------------------------------------------------------------------------- #
def candidate_centers(xyz: np.ndarray, gcfg: dict,
                      arena: np.ndarray | None = None) -> np.ndarray:
    """Voxel-grid candidate centers of the current sweep, inside the arena."""
    cand = voxel_downsample_centroids(np.asarray(xyz), gcfg["centers_voxel_m"])
    if len(cand) == 0:
        return cand
    return cand[inside_arena(cand, arena)]


def ball_radii(tree: cKDTree, centers: np.ndarray, policy: RadiusPolicy) -> np.ndarray:
    """Per-center radius under ``policy``, before the tie tolerance."""
    if policy.mode == "fixed" or len(centers) == 0:
        return np.full(len(centers), policy.radius_m, np.float64)
    k = min(policy.k, max(tree.n, 1))
    d, _ = tree.query(centers, k=k, distance_upper_bound=policy.max_m + TIE_TOLERANCE_M,
                      workers=-1)
    d = np.asarray(d, float).reshape(len(centers), -1)
    # Fewer than K points inside the maximum (inf), or fewer than K points in
    # the whole cloud (k was shortened): the ball goes to the maximum.
    kth = d[:, -1] if k == policy.k else np.full(len(centers), np.inf)
    return np.where(np.isfinite(kth), np.clip(kth, policy.min_m, policy.max_m),
                    policy.max_m)


def build_neighborhoods(centers: np.ndarray, support: Support, policy: RadiusPolicy,
                        rng: np.random.Generator, diagnostics: bool = True,
                        tree: cKDTree | None = None) -> dict:
    """Cut one ball per center out of ``support`` and canonicalise it.

    Returns a dict with ``scorable`` (bool per input center) and, for the
    scorable ones only: ``neighborhoods`` [S, P, 5] f32 (FEATURES then query
    height, exactly the version-1 layout), ``true_counts`` [S] i16,
    ``centers_odom`` [S, 3] f32, ``radius`` [S] f32, ``ball_count`` [S] i32
    (real points before sampling) and ``point_age`` [S, P] f16. With
    ``diagnostics`` also ``ball_voxels`` [S] i32 (distinct 2 cm cells in the
    ball) and ``ball_sweeps`` [S] u8 (distinct sweeps contributing).

    The sampling contract is version 1's: above ``n_points`` a uniform draw
    without replacement from the *whole* ball; below it every real point then
    repeats as padding; the height reference is the full ball's minimum,
    taken before any sampling. Coordinates stay in metres.
    """
    n_points = policy.n_points
    centers = np.asarray(centers, np.float64).reshape(-1, 3)
    n = len(centers)
    empty = {"scorable": np.zeros(n, bool),
             "neighborhoods": np.empty((0, n_points, SAMPLE_CHANNELS), np.float32),
             "true_counts": np.empty(0, np.int16), "centers_odom": np.empty((0, 3), np.float32),
             "radius": np.empty(0, np.float32), "ball_count": np.empty(0, np.int32),
             "point_age": np.empty((0, n_points), np.float16),
             "radius_all": np.full(n, np.nan, np.float32),
             "ball_count_all": np.zeros(n, np.int32)}
    if diagnostics:
        empty.update(ball_voxels=np.empty(0, np.int32), ball_sweeps=np.empty(0, np.uint8))
    if n == 0 or len(support) == 0:
        return empty

    xyz = support.xyz
    if tree is None:
        tree = cKDTree(xyz)
    radii = ball_radii(tree, centers, policy)
    slack = TIE_TOLERANCE_M if policy.mode == "adaptive" else 0.0
    lists = tree.query_ball_point(centers, radii + slack, workers=-1)
    vox_keys = None
    if diagnostics:
        v = np.floor(xyz / DUPLICATE_VOXEL_M).astype(np.int64)
        v -= v.min(axis=0)
        vox_keys = (v[:, 0] * 1_000_003 + v[:, 1]) * 1_000_003 + v[:, 2]

    scorable = np.zeros(n, bool)
    counts_all = np.zeros(n, np.int32)
    rows, counts, out_centers, out_radius, ball_counts, ages = [], [], [], [], [], []
    voxels, sweeps = [], []
    int16_max = np.iinfo(np.int16).max
    for i, (center, idx) in enumerate(zip(centers, lists)):
        k = len(idx)
        counts_all[i] = k
        if k < policy.min_neighbors:
            continue
        scorable[i] = True
        idx = np.asarray(idx, dtype=np.intp)
        z_min = xyz[idx, 2].min()
        if diagnostics:
            voxels.append(len(np.unique(vox_keys[idx])))
            sweeps.append(len(np.unique(support.slot[idx])))
        if k > n_points:
            sel = idx[rng.choice(k, n_points, replace=False)]
        elif k < n_points:
            sel = np.concatenate([idx, idx[rng.choice(k, n_points - k, replace=True)]])
        else:
            sel = idx
        pts = xyz[sel]
        local = np.empty((n_points, SAMPLE_CHANNELS), np.float32)
        local[:, 0] = pts[:, 0] - center[0]
        local[:, 1] = pts[:, 1] - center[1]
        local[:, 2] = pts[:, 2] - z_min
        local[:, 3] = support.intensity[sel]
        local[:, QUERY_HEIGHT_CHANNEL] = center[2] - z_min
        rows.append(local)
        counts.append(min(k, int16_max))
        out_centers.append(center)
        out_radius.append(radii[i])
        ball_counts.append(k)
        ages.append(support.age[sel])

    out = dict(empty)
    out["scorable"] = scorable
    out["radius_all"] = radii.astype(np.float32)
    out["ball_count_all"] = counts_all
    if not rows:
        return out
    out.update(
        neighborhoods=np.stack(rows).astype(np.float32),
        true_counts=np.asarray(counts, np.int16),
        centers_odom=np.stack(out_centers).astype(np.float32),
        radius=np.asarray(out_radius, np.float32),
        ball_count=np.asarray(ball_counts, np.int32),
        point_age=np.stack(ages).astype(np.float16))
    if diagnostics:
        out.update(ball_voxels=np.asarray(voxels, np.int32),
                   ball_sweeps=np.asarray(sweeps, np.uint8))
    return out


def slot_diagnostics(slots: list[HistorySlot], current: Sweep | None) -> dict:
    """Per-slot arrays describing one anchor's history, for manifests."""
    n = len(slots)
    ids = np.full(n, -1, np.int64)
    ages = np.full(n, np.nan, np.float32)
    shift = np.full(n, np.nan, np.float32)
    yaw = np.full(n, np.nan, np.float32)
    status = []
    for i, s in enumerate(slots):
        status.append(s.status)
        if s.sweep is None:
            continue
        ids[i] = s.sweep.sweep_id
        ages[i] = s.age_s
        if current is not None:
            shift[i] = float(np.linalg.norm(s.sweep.origin - current.origin))
            d = (s.sweep.yaw - current.yaw + math.pi) % (2 * math.pi) - math.pi
            yaw[i] = abs(math.degrees(d))
    return {"history_target_ages": np.asarray([s.target_age_s for s in slots], np.float32),
            "history_sweep_ids": ids, "history_ages": ages,
            "history_origin_shift": shift, "history_yaw_delta_deg": yaw,
            "history_status": np.asarray(status)}
