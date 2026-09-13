"""`rocklabel label`: accumulate the whole recording into one odom-frame cloud,
then hand off to the interactive viewer for sphere labeling."""

from __future__ import annotations

import os

import numpy as np

from ..geometry.accumulate import VoxelAccumulator, write_ply
from ..labels import LabelSet, load_labels
from ..geometry.leveling import level_record, pin_level_to_labels
from ..recording.pipeline import ScanStream


def default_labels_path(mcap_path: str, labels_root: str = "labels",
                        recordings_root: str = "recordings") -> str:
    """Default label file for a recording, under ``labels/``.

    ``recordings/`` and ``labels/`` are foldered by project now
    (``recordings/volleyball/reslam/X.mcap``), so this does two things a flat
    join could not. It first looks for an existing label file with this stem
    anywhere under ``labels/`` and returns that wherever it sits - relabelling
    must reopen the file that already holds the work, not start an empty one
    beside it. Failing that it mirrors the recording's own folder, so a
    volleyball recording's labels land in ``labels/volleyball/``.
    """
    stem = os.path.splitext(os.path.basename(mcap_path))[0]
    basename = stem + ".labels.json"

    for dirpath, _dirs, names in os.walk(labels_root):
        if basename in names:
            return os.path.join(dirpath, basename)

    # No existing file: mirror the recording's folder under labels/. A
    # recording outside recordings/ (an absolute path, say) lands at the top.
    sub = ""
    rec_dir = os.path.dirname(os.path.abspath(mcap_path))
    rec_root = os.path.abspath(recordings_root)
    if rec_dir.startswith(rec_root + os.sep):
        parts = os.path.relpath(rec_dir, rec_root).split(os.sep)
        # "volleyball/reslam" and "volleyball/raw" share one label folder: the
        # rocks are in the same place either way, only the poses differ.
        sub = os.path.join(*parts[:1]) if parts and parts[0] != "." else ""
    out_dir = os.path.join(labels_root, sub) if sub else labels_root
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, basename)


def accumulate_cloud(mcap_path: str, cfg: dict, stride: int,
                     min_hits: int = 1, carve: bool = False,
                     carve_assumed_pose_uncertainty: float | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, "ScanStream"]:
    """Fuse all (strided) scans into one voxel-accumulated odom-frame cloud.

    ``min_hits`` throws away every voxel that fewer than that many scans ever
    hit. The accumulator counts hits anyway, and the count is the sharpest
    signal available for telling terrain from stray returns: ground and rocks
    come back in the same voxel every time the sensor looks at them, while a
    mixed pixel or a grazing-angle return lands somewhere new each scan. That
    asymmetry means a couple of percent of bad returns can be a third of the
    voxels you actually see on screen.

    ``carve`` swaps the add-only accumulator for the ray-carving map in
    :mod:`~rocklabel.geometry.carve`, which also deletes points a later beam is
    repeatedly seen to pass through. It remains opt-in until the labelled,
    per-rock Lance audit validates it.
    """
    voxel = cfg["labeler"]["accumulator_voxel_m"]
    stream = ScanStream(mcap_path, cfg, stride=stride, progress=True, desc="accumulate")
    if carve:
        from ..geometry.carve import KFCMap, RayBatch
        pose_uncertainty = (
            np.nan if carve_assumed_pose_uncertainty is None
            else float(carve_assumed_pose_uncertainty)
        )
        if np.isfinite(pose_uncertainty) and pose_uncertainty < 0.0:
            raise ValueError("carve_assumed_pose_uncertainty must be non-negative")
        acc = KFCMap(voxel=voxel)
        for scan in stream:
            acc.update_batch(RayBatch.one_group(
                scan.xyz_odom,
                scan.T_odom_lidar[:3, 3],
                intensity=scan.intensity,
                group=scan.index,
                timestamp=scan.time_s,
                pose_uncertainty_m=pose_uncertainty,
            ))
        acc.flush()
        xyz, inten, _raw_counts = acc.result()
        # --min-hits is documented in source observations, not raw returns.
        # A dense scan may put several returns in one voxel but still supplies
        # only one independent support event.
        counts = acc.observations.copy()
    else:
        acc = VoxelAccumulator(voxel)
        for scan in stream:
            acc.add(scan.xyz_odom, scan.intensity)
        xyz, inten, counts = acc.result()

    print("\n=== accumulation summary ===")
    for line in stream.counters.summary_lines():
        print("  " + line)
    print(f"  voxels:                {len(xyz)} @ {voxel} m")
    if carve:
        print(f"  ray carving:           deleted {acc.n_deleted} map points "
              f"after repeated free-space evidence")
        if carve_assumed_pose_uncertainty is None:
            print("  pose uncertainty:      unknown; destructive votes disabled")
        else:
            print("  pose uncertainty:      "
                  f"assumed {float(carve_assumed_pose_uncertainty):.3f} m")
    if min_hits > 1 and len(xyz):
        keep = counts >= int(min_hits)
        dropped = len(xyz) - int(keep.sum())
        print(f"  min-hits filter:       kept {int(keep.sum())} voxels hit {min_hits}+ "
              f"times, dropped {dropped} ({dropped / len(xyz) * 100:.1f}%)")
        if not keep.any():
            raise SystemExit(
                f"--min-hits {min_hits} removed every voxel. The recording is short, "
                "the stride is large, or the value is too high - try a smaller one."
            )
        xyz, inten, counts = xyz[keep], inten[keep], counts[keep]
    if len(xyz):
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        print(f"  bounding box:          x [{lo[0]:.2f}, {hi[0]:.2f}]  y [{lo[1]:.2f}, {hi[1]:.2f}]  z [{lo[2]:.2f}, {hi[2]:.2f}]")
    print(f"  recording duration:    {stream.info.duration_s:.1f} s")
    return xyz, inten, counts, stream


def run_label(mcap_path: str, cfg: dict, labels_path: str | None, stride: int | None,
              z_min: float | None, z_max: float | None,
              dump_accumulated: str | None = None, fallback_viewer: bool = False,
              min_hits: int | None = None, carve: bool = False,
              carve_assumed_pose_uncertainty: float | None = None) -> None:
    lcfg = cfg["labeler"]
    stride = stride if stride is not None else lcfg["stride"]
    z_min = z_min if z_min is not None else lcfg["z_min"]
    z_max = z_max if z_max is not None else lcfg["z_max"]
    min_hits = 1 if min_hits is None else int(min_hits)

    # Resume before accumulating, not after: rock centers are world
    # coordinates, so a levelling angle measured now that differs from the one
    # they were picked in would move every existing rock off its rock. Pin the
    # frame to whatever the label file says before a single scan is read.
    labels_path = labels_path or default_labels_path(mcap_path)
    resumed = load_labels(labels_path) if os.path.exists(labels_path) else None
    if resumed is not None and resumed.rocks:
        cfg = pin_level_to_labels(cfg, resumed.level)

    xyz, inten, _counts, stream = accumulate_cloud(
        mcap_path, cfg, stride, min_hits, carve,
        carve_assumed_pose_uncertainty,
    )
    if len(xyz) == 0:
        raise SystemExit("No points accumulated - check topic/frame configuration with 'rocklabel inspect'.")

    if dump_accumulated:
        write_ply(dump_accumulated, xyz, inten)
        print(f"Wrote accumulated cloud ({len(xyz)} points) to {dump_accumulated}; exiting without viewer.")
        return

    if resumed is not None:
        labelset = resumed
        print(f"Resuming {len(labelset.rocks)} existing labels from {labels_path}")
    else:
        labelset = LabelSet()
        print(f"Starting new label set; will save to {labels_path}")
    labelset.mcap_file = os.path.basename(mcap_path)
    labelset.run_id = os.path.splitext(os.path.basename(mcap_path))[0]
    labelset.odom_frame = ("world" if getattr(stream, "format_name", "") == "lidarrig"
                           else cfg["topics"]["odom_frame"])
    labelset.intensity_available = bool(stream.counters.intensity_available)
    labelset.accumulator_voxel_m = lcfg["accumulator_voxel_m"]
    # Stamp the frame these centers are about to be picked in, so `generate`
    # can refuse to replay the recording the other way up.
    labelset.level = level_record(stream)

    from . import viewer  # Open3D import deferred so headless paths never touch it

    if fallback_viewer:
        viewer.run_labeler_fallback(xyz, inten, labelset, labels_path, cfg)
    else:
        viewer.run_labeler_gui(xyz, inten, labelset, labels_path, cfg, z_min=z_min, z_max=z_max)
