"""`rocklabel driftcheck`: visualize odometry drift around one labeled rock.

Accumulates only the first 10% and last 10% of scans, crops both to a 1 m box
around the chosen rock, and renders them overlaid in different colors. If
odometry drifted, the rock appears doubled or smeared and the labels from
`rocklabel label` cannot be trusted for that run.
"""

from __future__ import annotations

import numpy as np

from ..geometry.accumulate import VoxelAccumulator
from ..labels import load_labels
from ..recording.pipeline import ScanStream

BOX_HALF_M = 0.5     # minimum: 1 m box around the rock center
BOX_MARGIN_M = 0.3   # grows with the rock so big spheres keep context around them
SPARSE_VOXELS = 300  # below this the comparison is probably inconclusive


def run_driftcheck(mcap_path: str, labels_path: str, rock_id: int, cfg: dict) -> None:
    labelset = load_labels(labels_path)
    rock = labelset.get(rock_id)
    if rock is None:
        raise SystemExit(
            f"Rock id {rock_id} not found in {labels_path}; available ids: "
            f"{[r.id for r in labelset.rocks]}"
        )

    stream = ScanStream(mcap_path, cfg, stride=1, progress=True, desc="driftcheck")
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
    print("viewer: early scans = blue, late scans = orange, label sphere = red wireframe.")
    print("If blue and orange trace the SAME rock surface, odometry held and the labels "
          "are trustworthy; if the rock appears twice (blue copy offset from orange "
          "copy), odometry drifted by that offset.")

    from . import viewer

    viewer.show_driftcheck(early_xyz, late_xyz, rock.center, rock.radius)
