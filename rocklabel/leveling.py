"""Offline gravity levelling: undo a tilted sensor mount before anything sees it.

The live rig levels at capture time (:mod:`rocklabel.live.leveling`), but a
recording made with ``--level off`` has the mount angle baked permanently into
its poses: a native lidarrig frame's world frame *is* the sensor frame at
startup, so a LiDAR on a slanted mast tilts every point in the file. Nothing
downstream can compensate, and everything downstream cares:

* the labeler's **z clip** slices a diagonal wedge out of the floor rather than
  a horizontal slab, which makes the "drag z-max down to the floor" trick -
  the fastest way to spot rocks - useless;
* the generator's **crop box** is axis-aligned, so it keeps a wedge too;
* the model's neighborhoods see ``dz`` dominated by the tilt: a flat floor
  inside a 0.5 m ball tilted 40 degrees looks like 0.3 m of relief, i.e. like
  a rock.

Unlike the live path there is no IMU here - the recorded frames keep only the
fused pose - so the ground itself is the only measurement available, and this
module fits it directly. The result is a single constant rotation applied by
:class:`rocklabel.pipeline.LevelledScanStream` to every scan, which keeps the
labeler, the generator, and driftcheck in one shared frame. Constant matters:
labels are stored in world coordinates, so a rotation that drifted between
``label`` and ``generate`` would silently misplace every rock.

Because it changes the geometry of every output, the levelling settings are
part of the config hash, so a levelled dataset can never be mixed into an
unlevelled one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .accumulate import VoxelAccumulator
from .live.leveling import (
    fit_ground_plane,
    level_rotation_for_up,
    mount_rotation,
    roll_pitch_deg,
    tilt_deg,
)

#: Below this much measured tilt a recording counts as already level and is
#: left alone: the live rig levelled it at capture time, and re-fitting would
#: only add noise (and, indoors, risk locking onto a ceiling).
ALREADY_LEVEL_DEG = 2.0
#: How far the sensor must travel horizontally before its path can be trusted
#: to reveal the tilt.
_PATH_MIN_SPAN_M = 0.5
#: …and how much of its height variation that travel must explain.
_PATH_MIN_R2 = 0.5
#: Tilt gate for a ground fit that runs in an already-seeded frame, where the
#: floor is supposed to be nearly level already.
_SEEDED_TILT_GATE_DEG = 15.0

MODES = ("auto", "off", "ground", "manual")


class LevelError(Exception):
    """Raised when a requested levelling mode cannot be satisfied."""


def mode_of(cfg: dict) -> str:
    """The configured levelling mode, normalized. Never raises.

    Bare ``off`` in YAML parses as the boolean ``False``, not the string, so
    an unquoted ``mode: off`` would otherwise miss every comparison here and
    silently level a recording the user asked to leave alone.
    """
    raw = (cfg.get("level") or {}).get("mode", "auto")
    if isinstance(raw, bool):
        raw = "off" if raw is False else "on"
    return str(raw or "off").lower()


@dataclass
class LevelSolution:
    """The measured world->levelled rotation, plus what it was measured from."""

    rotation: np.ndarray            # (3, 3) world -> levelled
    source: str                     # "auto" | "ground" | "manual"
    floor_z: float | None = None    # floor height in the levelled frame
    inlier_frac: float = 0.0
    pooled_points: int = 0
    note: str = ""                  # why the fit fell back, when it did

    @property
    def matrix4(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation
        return m

    def summary_lines(self) -> list[str]:
        roll, pitch = roll_pitch_deg(self.rotation)
        lines = [
            f"levelling:             {self.source}",
            f"mount roll / pitch:    {roll:+.2f} deg / {pitch:+.2f} deg "
            f"(tilt {tilt_deg(self.rotation):.2f} deg)",
        ]
        if self.source in ("auto", "ground") and self.inlier_frac > 0.0:
            lines.append(f"ground fit:            {self.inlier_frac:.1%} inliers "
                         f"of {self.pooled_points} pooled points")
        if self.floor_z is not None:
            lines.append(f"floor (levelled):      z = {self.floor_z:+.3f} m")
        if self.note:
            lines.append(f"note:                  {self.note}")
        return lines


def level_record(stream) -> dict | None:
    """Serializable description of the frame ``stream`` yields, for label files.

    None for an unlevelled stream, which is also what a pre-levelling label
    file carries - so the two compare equal and old work keeps loading.
    """
    solution = getattr(stream, "solution", None)
    if solution is None:
        return None
    roll, pitch = roll_pitch_deg(solution.rotation)
    record = {"mode": solution.source,
              "roll_deg": round(roll, 4),
              "pitch_deg": round(pitch, 4)}
    if solution.floor_z is not None:
        record["floor_z"] = round(solution.floor_z, 4)
    return record


def pin_level_to_labels(cfg: dict, labelled: dict | None) -> dict:
    """Reproduce a label file's frame exactly instead of re-measuring it.

    A ``"ground"`` fit is repeatable to about half a degree - it pools
    whatever scans the caller's stride kept, and ``label`` and ``generate``
    rarely use the same one. Half a degree sounds harmless until you put it at
    the far edge of a 6 m crop box, where it is ~10 cm of vertical error: the
    radius of a small rock. The label file already records the angle its
    centers were picked at, so replay that verbatim.

    Only the *measured* modes are pinned. An explicit ``"manual"`` angle is the
    user overriding the measurement on purpose, and ``"off"`` has nothing to
    pin - both are left alone for :func:`check_level_match` to judge.

    A label file with no frame recorded at all predates levelling, so its
    centers live in the recording's own frame: that pins to ``"off"``. Without
    this, turning levelling on by default would quietly invalidate every label
    made before it existed.

    Returns a copy; the caller's cfg must stay untouched because it is what
    gets hashed into the dataset manifest, and the pinned angle differs from
    one recording to the next.
    """
    if mode_of(cfg) not in ("auto", "ground"):
        return cfg
    out = copy.deepcopy(cfg)
    if not labelled:
        # Labels exist but carry no frame: they were picked before levelling
        # existed, i.e. in the recording's own tilted frame. Measuring a fresh
        # angle now would slide every one of those centers sideways, so replay
        # the frame they were actually picked in.
        out["level"]["mode"] = "off"
        return out
    out["level"]["mode"] = "manual"
    out["level"]["mount_roll_deg"] = float(labelled.get("roll_deg", 0.0))
    out["level"]["mount_pitch_deg"] = float(labelled.get("pitch_deg", 0.0))
    return out


def _record_angles(record: dict | None) -> tuple[float, float]:
    if not record:
        return 0.0, 0.0
    return float(record.get("roll_deg", 0.0)), float(record.get("pitch_deg", 0.0))


def check_level_match(labelled: dict | None, current: dict | None,
                      labels_path: str, tolerance_deg: float = 0.5) -> None:
    """Refuse to mix a levelled frame with an unlevelled one.

    Rock centers are stored in world coordinates, so a rotation applied when
    labelling but not when generating (or vice versa) misplaces every single
    one of them - and nothing downstream would ever complain. A tolerance is
    allowed because a ``"ground"`` fit pools whatever scans its stride kept,
    and ``label`` and ``generate`` rarely use the same stride; anything past
    it means the two runs genuinely disagree about which way is up.
    """
    l_roll, l_pitch = _record_angles(labelled)
    c_roll, c_pitch = _record_angles(current)
    if max(abs(l_roll - c_roll), abs(l_pitch - c_pitch)) <= tolerance_deg:
        return

    def describe(record, angles):
        if not record:
            return "unlevelled (level.mode='off')"
        return f"{record.get('mode', '?')} roll{angles[0]:+.2f} deg pitch{angles[1]:+.2f} deg"

    fix = ("--level off" if not labelled else
           f"--level manual --mount-roll {l_roll:.2f} --mount-pitch {l_pitch:.2f}")
    raise LevelError(
        f"{labels_path} was labelled in a different frame than this run is generating: "
        f"labels are {describe(labelled, (l_roll, l_pitch))}, this run is "
        f"{describe(current, (c_roll, c_pitch))}. Rock centers are world coordinates, so "
        "generating now would misplace every one of them. Re-run with the labels' own "
        f"frame ({fix}), or re-label this recording."
    )


def _pool_ground_candidates(lcfg: dict, raw_stream_factory) -> tuple[np.ndarray, np.ndarray]:
    """Pool floor-candidate points, still in the recording's tilted frame.

    Returns ``(points, sensor_positions)``. The range band is applied per scan
    relative to that scan's sensor origin, so it follows a rig that walks -
    a band around the world origin would drop the floor the moment the rig
    left its starting spot.

    No ceiling cut is applied here. The live path can cut on "below the
    sensor" because its IMU seed has already brought the frame within a couple
    of degrees of level; unseeded, a 40 degree mount puts real floor points
    metres above the sensor. The tilt gate inside the fit and the
    below-the-sensor check on the fitted *plane* do that job instead.
    """
    limit = int(lcfg.get("fit_scans") or 0)
    r_min = float(lcfg["range_min_m"])
    r_max = float(lcfg["range_max_m"])
    acc = VoxelAccumulator(float(lcfg["fit_voxel_m"]))
    sensors: list[np.ndarray] = []

    for n, scan in enumerate(raw_stream_factory()):
        if limit and n >= limit:
            break
        origin = scan.T_odom_lidar[:3, 3]
        sensors.append(origin.copy())
        pts = scan.xyz_odom
        if len(pts) == 0:
            continue
        rel = pts - origin.astype(pts.dtype)
        r2 = np.einsum("ij,ij->i", rel, rel)
        keep = (r2 >= r_min * r_min) & (r2 <= r_max * r_max)
        n_keep = int(keep.sum())
        if n_keep:
            acc.add(pts[keep], np.zeros(n_keep, np.float32))

    pooled, _inten, _counts = acc.result()
    if not sensors:
        raise LevelError("levelling found no scans in the recording")
    return pooled.astype(np.float64), np.asarray(sensors, dtype=np.float64)


def _floor_plane_z(levelled_pts: np.ndarray, lcfg: dict) -> float | None:
    """Height of the floor in an already-levelled cloud, or None if unclear.

    Used for ``mode="manual"``, where the angle is given but the height still
    has to be measured - no tape measure of mount angle tells you where the
    floor is, and the height is what a z clip is actually anchored to. The
    tilt gate is tight here because the frame is supposed to be level already.
    """
    if len(levelled_pts) < 3:
        return None
    fit = fit_ground_plane(
        levelled_pts,
        thresh=float(lcfg["plane_thresh_m"]),
        iterations=int(lcfg["ransac_iters"]),
        max_tilt_deg=10.0,
    )
    if fit is None or fit[2] < float(lcfg["min_inlier_frac"]):
        return None
    return float(-fit[1])


def path_level_rotation(sensors: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Measure the mount tilt from the sensor's own path through the recording.

    A rig walking over flat ground traces a flat path, so any *consistent*
    climb in its recorded height as it moves horizontally is the frame being a
    ramp, not the rig going uphill. Fitting ``z = a·x + b·y + c`` over the path
    gives that ramp's normal directly — no points, no RANSAC, no risk of
    locking onto a wall or a ceiling, and it costs nothing because the poses
    are already in hand.

    It is a *seed*, not an answer: it is only as good as the odometry, and it
    needs the rig to have moved. But it lands within a few degrees, and that is
    enough to make "below the sensor" mean something again — which is what lets
    the ground fit that follows tell a floor from a ceiling.

    Returns ``(rotation, r_squared)``, or None when the path is too short or
    too wobbly to read.
    """
    p = np.asarray(sensors, dtype=np.float64)
    if p.shape[0] < 3:
        return None
    if np.linalg.norm(p[:, :2] - p[:, :2].mean(axis=0), axis=1).max() < _PATH_MIN_SPAN_M:
        return None
    design = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    coef, *_ = np.linalg.lstsq(design, p[:, 2], rcond=None)
    spread = float(np.var(p[:, 2]))
    if spread <= 1e-9:
        return None
    r2 = 1.0 - float(np.var(p[:, 2] - design @ coef)) / spread
    if r2 < _PATH_MIN_R2:
        return None
    # The path lies in the plane z = a·x + b·y + c, whose upward normal is
    # (-a, -b, 1); rotating that onto +z is what flattens the path.
    return level_rotation_for_up(np.array([-coef[0], -coef[1], 1.0])), r2


def _refine_on_ground(pooled: np.ndarray, sensors: np.ndarray, lcfg: dict,
                      seed: np.ndarray | None) -> tuple[np.ndarray, float, float] | None:
    """Fit the floor, optionally in a seeded frame. ``(normal, d, frac)`` or None.

    With a seed the frame is already within a few degrees of level, which buys
    two things an unseeded fit cannot have: a ceiling cut that works (real
    floor points can no longer sit metres *above* the sensor) and a tight tilt
    gate. Without one, neither is safe — a 40 degree mount really does put
    floor points overhead — so the gate stays as configured and every pooled
    point is offered to the fit, which is the behaviour ``mode="ground"`` has
    always had.
    """
    if seed is None:
        if pooled.shape[0] < 3:
            return None
        return fit_ground_plane(
            pooled,
            thresh=float(lcfg["plane_thresh_m"]),
            iterations=int(lcfg["ransac_iters"]),
            max_tilt_deg=float(lcfg["max_tilt_deg"]),
        )
    pts = pooled @ seed.T
    sensor_z = float(np.median((sensors @ seed.T)[:, 2]))
    below = pts[:, 2] < sensor_z - float(lcfg["ceiling_margin_m"])
    if int(below.sum()) < 3:
        return None
    return fit_ground_plane(
        pts[below],
        thresh=float(lcfg["plane_thresh_m"]),
        iterations=int(lcfg["ransac_iters"]),
        max_tilt_deg=min(_SEEDED_TILT_GATE_DEG, float(lcfg["max_tilt_deg"])),
    )


def _solve_auto(pooled: np.ndarray, sensors: np.ndarray,
                lcfg: dict) -> LevelSolution | None:
    """Measure the tilt without being told anything, and never raise.

    Seed from the sensor path, refine on the floor, and if the total comes to
    less than :data:`ALREADY_LEVEL_DEG` decide the recording was levelled at
    capture time and leave it alone. Never raising is the point of this mode:
    it is the default, so a recording it cannot read must still open.
    """
    path = path_level_rotation(sensors)
    seed = path[0] if path is not None else None
    note = "" if path is not None else "sensor path too short to seed from"

    fit = _refine_on_ground(pooled, sensors, lcfg, seed)
    if seed is None:
        seed = np.eye(3)
    if fit is not None and fit[2] >= float(lcfg["min_inlier_frac"]):
        normal, d, frac = fit
        rot = level_rotation_for_up(normal) @ seed
        floor_z = float(-d)
        source, inlier = "auto", frac
    else:
        rot, floor_z, frac = seed, None, 0.0
        source, inlier = "auto", 0.0
        note = (note + "; " if note else "") + "no usable floor fit — kept the path seed"

    if tilt_deg(rot) < ALREADY_LEVEL_DEG:
        print(f"\n=== levelling ===\n  already level ({tilt_deg(rot):.1f} deg) — "
              f"nothing to undo")
        return None
    return LevelSolution(rot, source, floor_z=floor_z, inlier_frac=inlier,
                         pooled_points=len(pooled), note=note)


def solve_level(cfg: dict, raw_stream_factory) -> LevelSolution | None:
    """Measure the recording's levelling rotation. None when levelling is off.

    ``raw_stream_factory`` must return a *fresh* unlevelled scan stream; it is
    consumed in a pass of its own, before the caller iterates for real.
    """
    lcfg = cfg.get("level") or {}
    mode = mode_of(cfg)
    if mode == "off":
        return None
    if mode not in MODES:
        raise LevelError(
            f"unknown level.mode {mode!r} (expected one of {', '.join(MODES)}). "
            'Quote the value in YAML - bare off/on/no/yes parse as booleans.'
        )

    pooled, sensors = _pool_ground_candidates(lcfg, raw_stream_factory)

    if mode == "auto":
        return _solve_auto(pooled, sensors, lcfg)

    if mode == "manual":
        rot = mount_rotation(lcfg["mount_roll_deg"], lcfg["mount_pitch_deg"])
        floor_z = _floor_plane_z(pooled @ rot.T, lcfg)
        return LevelSolution(rot, "manual", floor_z=floor_z, pooled_points=len(pooled))

    if len(pooled) < 3:
        raise LevelError(
            f"level.mode='ground' pooled only {len(pooled)} points - nothing to fit. "
            "Widen level.range_min_m / level.range_max_m, or set level.mode='manual' "
            "with a known level.mount_roll_deg / level.mount_pitch_deg."
        )
    # Seed from the sensor path when it is readable, exactly as "auto" does:
    # an unseeded fit indoors happily locks onto a ceiling, because a ceiling
    # is every bit as planar and as level as a floor.
    path = path_level_rotation(sensors)
    seed = path[0] if path is not None else None
    fit = _refine_on_ground(pooled, sensors, lcfg, seed)
    if fit is None and seed is not None:
        seed = None  # the seeded, gated fit found nothing: try it wide open
        fit = _refine_on_ground(pooled, sensors, lcfg, None)
    if seed is None:
        seed = np.eye(3)
    if fit is None:
        raise LevelError(
            f"level.mode='ground' found no plane within {lcfg['max_tilt_deg']} deg of "
            f"+z in {len(pooled)} pooled points. Set level.mode='manual' with a known "
            "mount angle instead."
        )
    normal, d, frac = fit
    min_frac = float(lcfg["min_inlier_frac"])
    if frac < min_frac:
        raise LevelError(
            f"level.mode='ground' fit kept only {frac:.1%} of {len(pooled)} pooled points "
            f"(need {min_frac:.0%}) - it is probably not the floor. Lower "
            "level.min_inlier_frac if you trust it, or use level.mode='manual'."
        )

    rot = level_rotation_for_up(normal) @ seed
    # The rotation is about the origin, which preserves the plane's
    # perpendicular distance from it; once the normal is +z that distance
    # *is* the floor height.
    floor_z = float(-d)

    # A ceiling is just as planar and just as level as a floor, so accept the
    # fit only when it passes below the sensor - checked against the median
    # sensor height so one bad pose cannot veto a good fit.
    sensor_z = float(np.median((sensors @ rot.T)[:, 2]))
    margin = float(lcfg["ceiling_margin_m"])
    if floor_z > sensor_z - margin:
        raise LevelError(
            f"level.mode='ground' fitted a plane at z={floor_z:+.2f} m, at or above the "
            f"sensor (median z={sensor_z:+.2f} m) - that is the ceiling, not the floor. "
            "Use level.mode='manual' with a known mount angle."
        )

    return LevelSolution(rot, "ground", floor_z=floor_z, inlier_frac=frac,
                         pooled_points=len(pooled))
