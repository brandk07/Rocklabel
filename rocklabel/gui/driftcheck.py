"""`rocklabel driftcheck`: visualize odometry drift around one labeled rock.

Accumulates only the first 10% and last 10% of scans, crops both to a 1 m box
around the chosen rock, and renders them overlaid in different colors. If
odometry drifted, the rock appears doubled or smeared and the labels from
`rocklabel label` cannot be trusted for that run.

The overlay is the honest answer, but "do blue and orange trace the same
surface" is a question a number can answer too, so the same comparison is also
measured and printed: how far apart the early and late surfaces sit. That makes
the check usable over SSH, in a script, and as a regression test after a
trajectory is re-solved -- ``--report-only`` skips the window entirely.
"""

from __future__ import annotations

import numpy as np

from ..geometry.accumulate import VoxelAccumulator
from ..geometry.leveling import check_level_match, level_record, pin_level_to_labels
from ..labels import load_labels
from ..recording.pipeline import ScanStream

BOX_HALF_M = 0.5     # minimum: 1 m box around the rock center
BOX_MARGIN_M = 0.3   # grows with the rock so big spheres keep context around them
SPARSE_VOXELS = 300  # below this the comparison is probably inconclusive
DRIFT_CELL_M = 0.05  # column footprint for the measured early-vs-late offset
DRIFT_MIN_CELLS = 8  # fewer shared columns than this and the number means nothing


def surface_offset(early: np.ndarray, late: np.ndarray,
                   cell: float = DRIFT_CELL_M) -> dict:
    """How far apart the early and late surfaces sit, in millimetres.

    Chops the box into small columns, takes the top of the surface in each one
    (the 90th percentile of height, so a few stray points below cannot move
    it), and compares columns both halves saw. A rock that stayed put gives a
    few millimetres; a rock that moved gives the size of the move.

    Height is used rather than a full 3D match because that is the direction
    the sensor measures best and the direction a smeared surface actually
    opens up in. ``centroid_mm`` catches sideways movement alongside it.
    """
    out = {"cells": 0, "median_mm": float("nan"), "p90_mm": float("nan"),
           "centroid_mm": float("nan")}
    if len(early) == 0 or len(late) == 0:
        return out
    out["centroid_mm"] = float(
        np.linalg.norm(early.mean(axis=0) - late.mean(axis=0)) * 1000.0)

    def tops(pts):
        ij = np.floor(pts[:, :2] / cell).astype(np.int64)
        key = ij[:, 0] * 1000003 + ij[:, 1]
        order = np.argsort(key, kind="stable")
        key, z = key[order], pts[order, 2]
        bounds = np.flatnonzero(np.diff(key)) + 1
        return {int(key[g[0]]): float(np.percentile(z[g], 90))
                for g in np.split(np.arange(len(z)), bounds)}

    a, b = tops(np.asarray(early, float)), tops(np.asarray(late, float))
    shared = [abs(a[k] - b[k]) for k in a.keys() & b.keys()]
    if len(shared) < DRIFT_MIN_CELLS:
        return out
    out["cells"] = len(shared)
    out["median_mm"] = float(np.median(shared) * 1000.0)
    out["p90_mm"] = float(np.percentile(shared, 90) * 1000.0)
    return out


def run_driftcheck(mcap_path: str, labels_path: str, rock_id: int, cfg: dict,
                   report_only: bool = False) -> dict:
    labelset = load_labels(labels_path)
    rock = labelset.get(rock_id)
    if rock is None:
        raise SystemExit(
            f"Rock id {rock_id} not found in {labels_path}; available ids: "
            f"{[r.id for r in labelset.rocks]}"
        )

    # Replay the frame the centers were picked in rather than re-measuring it,
    # exactly as `generate` does. A ground fit is only repeatable to about half
    # a degree, and half a degree at the far edge of the box is centimetres --
    # which is the same size as the drift this command exists to detect.
    cfg = pin_level_to_labels(cfg, labelset.level)
    stream = ScanStream(mcap_path, cfg, stride=1, progress=True, desc="driftcheck")
    check_level_match(labelset.level, level_record(stream), labels_path)
    n = stream.scan_count
    head_end = max(int(np.ceil(n * 0.1)), 1)
    tail_start = n - head_end

    voxel = cfg["labeler"]["accumulator_voxel_m"]
    acc_early, acc_late = VoxelAccumulator(voxel), VoxelAccumulator(voxel)
    range_early = [None, None]
    range_late = [None, None]
    half = max(BOX_HALF_M, rock.radius + BOX_MARGIN_M)
    lo = rock.center - half
    hi = rock.center + half

    for scan in stream:
        if scan.index < head_end:
            acc, rng = acc_early, range_early
        elif scan.index >= tail_start:
            acc, rng = acc_late, range_late
        else:
            continue
        inside = ((scan.xyz_odom >= lo) & (scan.xyz_odom <= hi)).all(axis=1)
        acc.add(scan.xyz_odom[inside], scan.intensity[inside])
        rng[0] = scan.time_s if rng[0] is None else rng[0]
        rng[1] = scan.time_s

    early_xyz, _, _ = acc_early.result()
    late_xyz, _, _ = acc_late.result()

    def _fmt(rng):
        return "empty" if rng[0] is None else f"{rng[0]:.3f} -> {rng[1]:.3f} s ({rng[1] - rng[0]:.1f} s)"

    print(f"\nrock {rock.id}: center {np.round(rock.center, 3).tolist()}, radius {rock.radius} m")
    print(f"early 10% ({head_end} scans):  time {_fmt(range_early)},  {len(early_xyz)} voxels in box")
    print(f"late  10% ({n - tail_start} scans):  time {_fmt(range_late)},  {len(late_xyz)} voxels in box")
    if len(early_xyz) == 0 or len(late_xyz) == 0:
        raise SystemExit(
            "One of the two accumulations has no points near this rock - the robot may "
            "not have seen it early and late in the run. Try another rock id."
        )
    if min(len(early_xyz), len(late_xyz)) < SPARSE_VOXELS:
        print(
            "CAUTION: very few points near this rock in the early/late windows - the "
            "LiDAR probably didn't have it in view at the start and/or end of the run. "
            "The comparison may be inconclusive; try a rock the robot saw both early "
            "and late (e.g. one near its start position)."
        )
    off = surface_offset(early_xyz, late_xyz)
    if off["cells"]:
        print(f"measured offset: early vs late surface differs by "
              f"{off['median_mm']:.1f} mm (median over {off['cells']} shared "
              f"columns), {off['p90_mm']:.1f} mm at the 90th percentile; "
              f"centroids {off['centroid_mm']:.1f} mm apart.")
        # Sensor noise alone is ~11 mm inside 6 m, so a rock that held still
        # cannot come out much under that and anything near a rock radius
        # (150-260 mm) means the label no longer sits on the rock.
        if off["median_mm"] < 20.0:
            print("  -> that is sensor-noise territory: the rock held still and "
                  "the labels are trustworthy.")
        elif off["median_mm"] < 60.0:
            print("  -> some smearing, but well under a rock radius: labels "
                  "still usable, worth a look at the overlay.")
        else:
            print("  -> that is a real displacement. Check the overlay; these "
                  "labels may need moving.")
    else:
        print("measured offset: too few shared columns to measure - use the overlay.")

    if report_only:
        return off

    print("viewer: early scans = blue, late scans = orange, label sphere = red wireframe.")
    print("If blue and orange trace the SAME rock surface, odometry held and the labels "
          "are trustworthy; if the rock appears twice (blue copy offset from orange "
          "copy), odometry drifted by that offset.")

    from . import viewer

    viewer.show_driftcheck(early_xyz, late_xyz, rock.center, rock.radius)
    return off
