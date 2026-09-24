"""Version-2 dataset generation: format A with causal history and adaptive radii.

Reached from :func:`rocklabel.dataset.generate.run_generate` whenever the
config says ``generator.preprocessing_version: 2``. Differences from the
version-1 path, all deliberate:

* Every assembled sweep is pushed into a :class:`~.history.SweepHistory`;
  every ``frame_stride``-th one is an *anchor*. Longer history never costs an
  anchor - the intervening sweeps are kept for history, not thrown away.
* Candidates, their labels and the negatives kept come from the anchor's own
  sweep with ``[seed, index]`` - the same draws for every arm - and the ball
  around each is cut from the anchor plus its selected history with a separate
  sampling RNG. So two arms differ only in the points their balls are made of.
* Only format A is written.
* Every sample carries its radius, pre-sampling count, distinct-voxel count,
  contributing-sweep count and per-point age; every frame carries which sweeps
  filled which slot. Nothing is summarised away before it reaches the cache.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone

import numpy as np

from .. import __version__
from ..config import validate_generator
from ..geometry.leveling import check_level_match, level_record, pin_level_to_labels
from ..labels import load_labels
from ..profiles import identify as identify_profile
from .history import (FEATURE_SCHEMA, PREPROCESSING_VERSION, HistoryPolicy, RadiusPolicy,
                      Sweep, SweepHistory, assemble_support, build_neighborhoods,
                      candidate_centers, candidate_rng, input_contract, sample_rng,
                      slot_diagnostics)
from .labeling import LABEL_CLEAR, LABEL_ROCK, label_rocks


def _quantiles(values, qs=(0, 5, 25, 50, 75, 95, 100)) -> dict | None:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return {f"p{q}": float(np.percentile(v, q)) for q in qs} | {"mean": float(v.mean()),
                                                                 "n": int(len(v))}


def crop_box(base: np.ndarray, gcfg: dict, z_band) -> tuple[np.ndarray, np.ndarray]:
    """The version-1 operational crop around the robot base, as (lo, hi).

    float32, as version 1 built it: the h1 arms must cut the identical cloud.
    """
    lo = np.array([base[0] - gcfg["crop_backward_m"], base[1] - gcfg["crop_right_m"],
                   base[2] - gcfg["crop_down_m"]], np.float32)
    hi = np.array([base[0] + gcfg["crop_forward_m"], base[1] + gcfg["crop_left_m"],
                   base[2] + gcfg["crop_up_m"]], np.float32)
    if z_band is not None:
        lo[2], hi[2] = z_band
    return lo, hi


def run_generate_history(mcap_path: str, labels_path: str, out_dir: str, cfg: dict,
                         profile: str | None = None, sweep_cache: str | None = None) -> dict:
    """Generate format A for one recording under a version-2 config."""
    from .generate import MANIFEST_NAME, _print_summary, check_manifest

    gcfg = cfg["generator"]
    validate_generator(gcfg)
    if int(gcfg.get("preprocessing_version", 1)) != PREPROCESSING_VERSION:
        raise ValueError("run_generate_history needs generator.preprocessing_version: 2")
    radius = RadiusPolicy.from_generator(gcfg)
    history = HistoryPolicy.from_generator(gcfg)
    labelset = load_labels(labels_path)
    run_id = labelset.run_id or os.path.splitext(os.path.basename(mcap_path))[0]

    os.makedirs(out_dir, exist_ok=True)
    manifest = check_manifest(out_dir, cfg, profile)
    points_dir = os.path.join(out_dir, "points", run_id)
    if os.path.isdir(points_dir):
        shutil.rmtree(points_dir)
    os.makedirs(points_dir)

    window_s = float(gcfg.get("frame_window_s") or 0.0)
    stream_cfg = pin_level_to_labels(cfg, labelset.level)
    if sweep_cache:
        from .sweep_cache import CachedSweeps, cache_identity, check_cache
        ok, why = check_cache(sweep_cache, cache_identity(mcap_path, labels_path, stream_cfg))
        if not ok:
            raise SystemExit(f"sweep cache {sweep_cache!r} does not match this "
                             f"recording/config: {why}")
        sweeps = CachedSweeps(sweep_cache)
        level = sweeps.meta["level"]
        check_level_match(labelset.level, level, labels_path)
        counters = {"skipped_pose": sweeps.meta.get("skipped_pose", 0),
                    "stamp_fallbacks": sweeps.meta.get("stamp_fallbacks", 0),
                    "intensity_available": sweeps.meta.get("intensity_available")}
        source = {"sweep_cache": os.path.abspath(sweep_cache)}
    else:
        from ..recording.pipeline import ScanStream, WindowedScanStream
        base_stream = ScanStream(mcap_path, stream_cfg, stride=1, progress=True,
                                 desc=f"generate {run_id}")
        check_level_match(labelset.level, level_record(base_stream), labels_path)
        sweeps = (WindowedScanStream(base_stream, window_s) if window_s > 0
                  else base_stream)
        level, counters, source = None, None, {"sweep_cache": None}

    z_band = labelset.z_band
    buffer = SweepHistory(history.retention_s)
    stats = {"frames_kept": 0, "frames_skipped_empty": 0, "point_samples": 0,
             "sample_labels": {"rock": 0, "clear": 0},
             "candidates_total": 0, "candidates_selected": {"rock": 0, "clear": 0},
             "unscorable": {"rock": 0, "clear": 0}, "history_resets": 0,
             "sweeps_seen": 0}
    radii, ball_counts, ball_voxels, ball_sweeps, support_pts = [], [], [], [], []
    slot_ages = [[] for _ in history.ages_s]
    slot_status = [dict() for _ in history.ages_s]
    slot_shift = [[] for _ in history.ages_s]
    slot_yaw = [[] for _ in history.ages_s]
    slot_points = [[] for _ in history.ages_s]
    anchor_times = []
    seed = int(gcfg["seed"])

    for k, scan in enumerate(sweeps):
        stats["sweeps_seen"] += 1
        current = Sweep.from_scan(scan)
        if buffer.push(current):
            stats["history_resets"] += 1
        if k % int(gcfg["frame_stride"]):
            continue
        base = scan.T_odom_base[:3, 3]
        lo, hi = crop_box(base, gcfg, z_band)

        def crop(xyz, lo=lo, hi=hi):
            return ((xyz >= lo) & (xyz <= hi)).all(axis=1)

        slots = buffer.select(history)
        support = assemble_support(slots, crop)
        cur = support.current
        if not cur.any():
            stats["frames_skipped_empty"] += 1
            continue

        # Candidates and negatives: the anchor sweep only, with the version-1
        # draw order, so every arm keeps the identical candidate set.
        crng = candidate_rng(seed, scan.index)
        cand = candidate_centers(support.xyz[cur], gcfg, labelset.arena)
        stats["candidates_total"] += int(len(cand))
        if len(cand) == 0:
            continue
        cand_labels = label_rocks(cand, labelset.rocks, gcfg["boundary_shell_m"])
        keep = cand_labels == LABEL_ROCK
        keep |= (cand_labels == LABEL_CLEAR) & (crng.random(len(cand)) < gcfg["negative_keep_prob"])
        cand, cand_labels = cand[keep], cand_labels[keep]
        if len(cand) == 0:
            continue
        stats["candidates_selected"]["rock"] += int((cand_labels == LABEL_ROCK).sum())
        stats["candidates_selected"]["clear"] += int((cand_labels == LABEL_CLEAR).sum())

        built = build_neighborhoods(cand, support, radius, sample_rng(seed, scan.index))
        ok = built["scorable"]
        stats["unscorable"]["rock"] += int(((cand_labels == LABEL_ROCK) & ~ok).sum())
        stats["unscorable"]["clear"] += int(((cand_labels == LABEL_CLEAR) & ~ok).sum())

        diag = slot_diagnostics(slots, current)
        for i, s in enumerate(slots):
            slot_status[i][s.status] = slot_status[i].get(s.status, 0) + 1
            slot_points[i].append(support.slot_points[i])
            if s.sweep is not None:
                slot_ages[i].append(s.age_s)
                slot_shift[i].append(diag["history_origin_shift"][i])
                slot_yaw[i].append(diag["history_yaw_delta_deg"][i])
        support_pts.append(len(support))
        anchor_times.append(float(scan.time_s))
        if not ok.any():
            continue
        labels = cand_labels[ok].astype(np.int8)
        np.savez_compressed(
            os.path.join(points_dir, f"frame_{scan.index:06d}.npz"),
            neighborhoods=built["neighborhoods"], labels=labels,
            true_counts=built["true_counts"], centers_odom=built["centers_odom"],
            radius=built["radius"], ball_count=built["ball_count"],
            ball_voxels=built["ball_voxels"], ball_sweeps=built["ball_sweeps"],
            point_age=built["point_age"],
            frame_time=np.float64(scan.time_s),
            robot_pose=np.asarray(scan.T_odom_base, np.float64),
            support_points=np.int64(len(support)),
            slot_points=np.asarray(support.slot_points, np.int64),
            **{k2: v for k2, v in diag.items() if k2 != "history_status"},
            history_status=diag["history_status"].astype("U12"))
        stats["frames_kept"] += 1
        stats["point_samples"] += int(len(labels))
        stats["sample_labels"]["rock"] += int((labels == LABEL_ROCK).sum())
        stats["sample_labels"]["clear"] += int((labels == LABEL_CLEAR).sum())
        radii.append(built["radius"])
        ball_counts.append(built["ball_count"])
        ball_voxels.append(built["ball_voxels"])
        ball_sweeps.append(built["ball_sweeps"])

    if not sweep_cache:
        # Only known once the decode has run to the end.
        level = level_record(base_stream)
        counters = {"skipped_pose": base_stream.counters.skipped_pose,
                    "stamp_fallbacks": base_stream.counters.stamp_fallbacks,
                    "intensity_available": base_stream.counters.intensity_available}
    cat = (lambda xs, dt: np.concatenate(xs) if xs else np.empty(0, dt))
    r, bc = cat(radii, np.float32), cat(ball_counts, np.int32)
    bv, bs = cat(ball_voxels, np.int32), cat(ball_sweeps, np.uint8)
    t0 = anchor_times[0] if anchor_times else 0.0
    diagnostics = {
        "input_contract": input_contract(gcfg),
        "radius_m": _quantiles(r),
        "radius_at_min_frac": (float(np.mean(np.isclose(r, radius.min_m)))
                               if radius.mode == "adaptive" and len(r) else None),
        "radius_at_max_frac": (float(np.mean(np.isclose(r, radius.max_m)))
                               if radius.mode == "adaptive" and len(r) else None),
        "shortfall_below_k_frac": (float(np.mean(bc < radius.k))
                                   if radius.mode == "adaptive" and len(bc) else None),
        "below_n_points_frac": float(np.mean(bc < radius.n_points)) if len(bc) else None,
        "ball_count": _quantiles(bc),
        "ball_voxels": _quantiles(bv),
        "points_per_voxel": _quantiles(bc / np.maximum(bv, 1)) if len(bv) else None,
        "ball_sweeps": _quantiles(bs),
        "support_points": _quantiles(support_pts),
        "anchor_span_s": (anchor_times[-1] - t0) if anchor_times else 0.0,
        "anchors_before_full_history": (int(sum(t - t0 < history.max_age_s
                                                for t in anchor_times))),
        "slots": [{"target_age_s": float(a), "status": slot_status[i],
                   "fill_rate": (slot_status[i].get("ok", 0)
                                 / max(sum(slot_status[i].values()), 1)),
                   "actual_age_s": _quantiles(slot_ages[i]),
                   "origin_shift_m": _quantiles(slot_shift[i]),
                   "yaw_delta_deg": _quantiles(slot_yaw[i]),
                   "points_in_crop": _quantiles(slot_points[i])}
                  for i, a in enumerate(history.ages_s)],
    }
    entry = {
        "run_id": run_id,
        "mcap_path": os.path.abspath(mcap_path),
        "labels_path": os.path.abspath(labels_path),
        "rock_count": len(labelset.rocks),
        "arena_vertices": None if labelset.arena is None else int(len(labelset.arena)),
        "z_band": None if z_band is None else [float(z_band[0]), float(z_band[1])],
        "level": level,
        "intensity_available": bool(counters["intensity_available"]),
        "frames_skipped_pose": int(counters["skipped_pose"]),
        "stamp_fallbacks": int(counters["stamp_fallbacks"]),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "feature_schema": FEATURE_SCHEMA,
        "formats": ["points"],
        **source, **stats, "diagnostics": diagnostics,
        # Fields the version-1 summary and cache expect; this builder writes none.
        "bev_frames": 0, "seg_frames": 0,
    }
    manifest["runs"][run_id] = entry
    manifest["tool_version"] = __version__
    manifest["profile"] = profile or manifest.get("profile") or identify_profile(cfg)
    manifest["generated"] = entry["generated"]
    tmp = os.path.join(out_dir, MANIFEST_NAME + ".tmp")
    import json
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, os.path.join(out_dir, MANIFEST_NAME))
    _print_summary(manifest)
    if labelset.rocks and stats["sample_labels"]["rock"] == 0:
        print(f"WARNING: run {run_id!r} has {len(labelset.rocks)} labeled rocks but "
              "produced zero rock samples - check label alignment.")
    return entry
