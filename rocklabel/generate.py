"""`rocklabel generate`: replay a recording, project sphere labels into every
kept frame, and write both dataset formats plus a config-hash-guarded manifest."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np

from . import __version__
from .bev import rasterize_bev
from .config import config_hash
from .labeling import LABEL_CLEAR, LABEL_IGNORE, LABEL_ROCK, label_rocks
from .labels import load_labels
from .leveling import check_level_match, level_record, pin_level_to_labels
from .neighborhoods import build_neighborhood_samples, build_segmentation_frame
from .pipeline import ScanStream, WindowedScanStream

MANIFEST_NAME = "manifest.json"


class ManifestConflict(Exception):
    pass


def _load_manifest(out_dir: str) -> dict | None:
    path = os.path.join(out_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def check_manifest(out_dir: str, cfg: dict) -> dict:
    """Return the existing manifest (or a fresh one). Refuse mixed configs."""
    new_hash = config_hash(cfg)
    manifest = _load_manifest(out_dir)
    if manifest is None:
        return {
            "tool_version": __version__,
            "config": cfg,
            "config_hash": new_hash,
            "runs": {},
        }
    if manifest.get("config_hash") != new_hash:
        raise ManifestConflict(
            f"Dataset directory {out_dir!r} was generated with a different config "
            f"(hash {manifest.get('config_hash', '?')[:12]}... vs {new_hash[:12]}...). "
            "Mixing configs in one dataset is not allowed - choose a new --out directory."
        )
    return manifest


def run_generate(mcap_path: str, labels_path: str, out_dir: str, cfg: dict) -> dict:
    """Generate both dataset formats for one recording. Returns the run's manifest entry."""
    gcfg = cfg["generator"]
    labelset = load_labels(labels_path)
    run_id = labelset.run_id or os.path.splitext(os.path.basename(mcap_path))[0]

    os.makedirs(out_dir, exist_ok=True)
    manifest = check_manifest(out_dir, cfg)

    points_dir = os.path.join(out_dir, "points", run_id)
    bev_dir = os.path.join(out_dir, "bev", run_id)
    seg_dir = os.path.join(out_dir, "seg", run_id)
    for d in (points_dir, bev_dir, seg_dir):  # re-generating a run_id replaces its files
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # With frame_window_s, every scan is decoded and merged into time-window
    # frames first, then frame_stride keeps every Nth *window*. Without it,
    # ScanStream's stride skips scans before decoding, avoiding the
    # decode/transform cost of dropped frames.
    # Replay in the exact frame the centers were picked in, not a fresh fit of
    # it. Deliberately not fed to check_manifest above: the pinned angle is
    # per-recording, so hashing it would split one dataset into a directory
    # per run.
    stream_cfg = pin_level_to_labels(cfg, labelset.level)

    window_s = gcfg.get("frame_window_s") or 0.0
    if window_s > 0.0:
        base_stream = ScanStream(mcap_path, stream_cfg, stride=1, progress=True,
                                 desc=f"generate {run_id}")
        stream = WindowedScanStream(base_stream, window_s)
        frames = (s for k, s in enumerate(stream) if k % gcfg["frame_stride"] == 0)
    else:
        base_stream = ScanStream(mcap_path, stream_cfg, stride=gcfg["frame_stride"],
                                 progress=True, desc=f"generate {run_id}")
        stream = base_stream
        frames = stream

    # Both sides must be the same way up before a single center is projected.
    check_level_match(labelset.level, level_record(base_stream), labels_path)

    # The z clip the labeler was left at, or None for "no height restriction".
    z_band = labelset.z_band
    if z_band is None:
        print(f"  height band:           none - vertical extent is the crop box "
              f"({gcfg['crop_down_m']:.2f} m below to {gcfg['crop_up_m']:.2f} m above the sensor)")
    else:
        print(f"  height band:           z {z_band[0]:.2f} to {z_band[1]:.2f} m "
              f"(from the labeler's z clip; replaces the crop box's vertical limits)")

    stats = {
        "frames_kept": 0,
        "frames_skipped_pose": 0,
        "frames_skipped_empty": 0,
        "point_samples": 0,
        "bev_frames": 0,
        "seg_frames": 0,
        "seg_points": 0,
        "seg_point_labels": {"rock": 0, "clear": 0, "ignore": 0},
        "sample_labels": {"rock": 0, "clear": 0},
        "point_labels": {"rock": 0, "clear": 0, "ignore": 0},
        "bev_cells": {"rock": 0, "clear": 0, "ignore": 0},
    }

    for scan in frames:
        base = scan.T_odom_base[:3, 3]
        xyz, inten = scan.xyz_odom, scan.intensity

        # Axis-aligned odom-frame crop centered on the robot base position.
        # Deliberately not rotated with heading (see README).
        lo = np.array([base[0] - gcfg["crop_backward_m"], base[1] - gcfg["crop_right_m"], base[2] - gcfg["crop_down_m"]], np.float32)
        hi = np.array([base[0] + gcfg["crop_forward_m"], base[1] + gcfg["crop_left_m"], base[2] + gcfg["crop_up_m"]], np.float32)
        # A labeled height band replaces the crop's vertical limits outright
        # rather than intersecting with them. The band is a world-frame slab
        # picked by eye on the fused cloud; the crop's is a slab that rides up
        # and down with the sensor. Intersecting would let crop_up_m silently
        # eat the top of a band the user could see themselves setting.
        if z_band is not None:
            lo[2], hi[2] = z_band
        inside = ((xyz >= lo) & (xyz <= hi)).all(axis=1)
        if not inside.any():
            stats["frames_skipped_empty"] += 1
            continue
        xyz, inten = xyz[inside], inten[inside]

        pt_labels = label_rocks(xyz, labelset.rocks, gcfg["boundary_shell_m"])
        stats["point_labels"]["rock"] += int((pt_labels == LABEL_ROCK).sum())
        stats["point_labels"]["clear"] += int((pt_labels == LABEL_CLEAR).sum())
        stats["point_labels"]["ignore"] += int((pt_labels == LABEL_IGNORE).sum())

        common_meta = {
            "frame_time": np.float64(scan.time_s),
            "robot_pose": scan.T_odom_base.astype(np.float64),
        }

        # Seed per frame so output is reproducible regardless of skipped frames.
        rng = np.random.default_rng([int(gcfg["seed"]), scan.index])
        samples = build_neighborhood_samples(xyz, inten, labelset.rocks, gcfg, rng,
                                             arena=labelset.arena)
        if samples is not None:
            np.savez_compressed(
                os.path.join(points_dir, f"frame_{scan.index:06d}.npz"), **samples, **common_meta
            )
            stats["point_samples"] += len(samples["labels"])
            stats["sample_labels"]["rock"] += int((samples["labels"] == LABEL_ROCK).sum())
            stats["sample_labels"]["clear"] += int((samples["labels"] == LABEL_CLEAR).sum())

        seg = build_segmentation_frame(xyz, inten, pt_labels, base, gcfg, rng,
                                       arena=labelset.arena)
        if seg is not None:
            np.savez_compressed(
                os.path.join(seg_dir, f"frame_{scan.index:06d}.npz"), **seg, **common_meta
            )
            stats["seg_frames"] += 1
            stats["seg_points"] += int(seg["true_count"])
            real = seg["labels"][:int(seg["true_count"])]
            stats["seg_point_labels"]["rock"] += int((real == LABEL_ROCK).sum())
            stats["seg_point_labels"]["clear"] += int((real == LABEL_CLEAR).sum())
            stats["seg_point_labels"]["ignore"] += int((real == LABEL_IGNORE).sum())

        channels, mask = rasterize_bev(xyz, inten, pt_labels, base, gcfg)
        np.savez_compressed(
            os.path.join(bev_dir, f"frame_{scan.index:06d}.npz"),
            channels=channels, label_mask=mask, **common_meta,
        )
        stats["bev_frames"] += 1
        stats["bev_cells"]["rock"] += int((mask == 1).sum())
        stats["bev_cells"]["clear"] += int((mask == 0).sum())
        stats["bev_cells"]["ignore"] += int((mask == 255).sum())
        stats["frames_kept"] += 1

    stats["frames_skipped_pose"] = stream.counters.skipped_pose
    stats["stamp_fallbacks"] = stream.counters.stamp_fallbacks

    entry = {
        "run_id": run_id,
        "mcap_path": os.path.abspath(mcap_path),
        "labels_path": os.path.abspath(labels_path),
        "rock_count": len(labelset.rocks),
        # Recorded so a dataset says whether its samples were arena-bounded;
        # pooling a bounded run with an unbounded one mixes two different
        # definitions of "clear" and the manifest is the only place that shows.
        "arena_vertices": (None if labelset.arena is None
                           else int(len(labelset.arena))),
        # Same reason as arena_vertices, for the vertical half of the bound:
        # pooling a height-banded run with an unbanded one mixes two different
        # definitions of "clear", and the manifest is where that shows.
        "z_band": (None if z_band is None
                   else [float(z_band[0]), float(z_band[1])]),
        # The mount rotation this run's geometry was produced under, for the
        # same reason as arena_vertices: the manifest is where a dataset's
        # provenance shows.
        "level": level_record(base_stream),
        "intensity_available": bool(stream.counters.intensity_available),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **stats,
    }
    manifest["runs"][run_id] = entry
    manifest["tool_version"] = __version__
    manifest["generated"] = entry["generated"]
    tmp = os.path.join(out_dir, MANIFEST_NAME + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, os.path.join(out_dir, MANIFEST_NAME))

    _print_summary(manifest)

    if labelset.rocks and stats["sample_labels"]["rock"] == 0 and stats["bev_cells"]["rock"] == 0:
        print(
            "\n" + "!" * 78 + "\n"
            f"WARNING: run {run_id!r} has {len(labelset.rocks)} labeled rocks but ZERO rock-labeled\n"
            "samples were produced. This almost always means the labels and the frames are\n"
            "misaligned (odometry drift, wrong frames, or labels from a different run).\n"
            f"Check alignment with:  rocklabel driftcheck --mcap {mcap_path} --labels {labels_path} --rock-id {labelset.rocks[0].id}\n"
            + "!" * 78
        )
    return entry


def _print_summary(manifest: dict) -> None:
    rows = []
    for run_id, e in sorted(manifest["runs"].items()):
        total = e["sample_labels"]["rock"] + e["sample_labels"]["clear"]
        frac = e["sample_labels"]["rock"] / total if total else 0.0
        rows.append((run_id, e["frames_kept"], e["point_samples"], e["bev_frames"], frac))
    total_frames = sum(r[1] for r in rows)
    total_pts = sum(r[2] for r in rows)
    total_bev = sum(r[3] for r in rows)
    all_rock = sum(m["sample_labels"]["rock"] for m in manifest["runs"].values())
    all_n = sum(m["sample_labels"]["rock"] + m["sample_labels"]["clear"] for m in manifest["runs"].values())

    print("\n=== dataset summary ===")
    header = f"{'run':<28} {'frames':>7} {'pt samples':>11} {'bev frames':>11} {'rock frac':>10}"
    print(header)
    print("-" * len(header))
    for run_id, frames, pts, bev, frac in rows:
        print(f"{run_id:<28} {frames:>7} {pts:>11} {bev:>11} {frac:>10.3f}")
    print("-" * len(header))
    total_frac = all_rock / all_n if all_n else 0.0
    print(f"{'TOTAL':<28} {total_frames:>7} {total_pts:>11} {total_bev:>11} {total_frac:>10.3f}")
