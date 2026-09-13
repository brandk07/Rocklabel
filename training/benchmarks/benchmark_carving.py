#!/usr/bin/env python3
"""Repeatable warmed-map benchmark for the ray-carving core.

Input is a ray-preserving NPZ with ``xyz`` (concatenated endpoints), ``org``
(one origin per sweep), and ``counts`` (endpoint counts per sweep). Optional
per-sweep ``timestamps`` and ``groups`` are retained; otherwise the manifest
marks the nominal values synthesized by this harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
import tracemalloc
from pathlib import Path

import numpy as np

from rocklabel.geometry.carve import KFCMap, RayBatch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_version(root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _snapshot(map_: KFCMap) -> dict[str, np.ndarray]:
    order = np.argsort(map_._keys_by_row)
    return {
        name: np.asarray(getattr(map_, name))[order]
        for name in (
            "_keys_by_row", "pts", "hits", "observations", "support",
            "contradictions", "confirmed", "intensity_sum", "intensity_hits",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--warm-sweeps", type=int, default=240)
    parser.add_argument("--timed-sweeps", type=int, default=60)
    parser.add_argument("--nominal-group-s", type=float, default=0.05)
    parser.add_argument("--voxel", type=float, default=0.05)
    parser.add_argument("--max-range", type=float, default=8.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--snapshot-out", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--tracemalloc", action="store_true")
    args = parser.parse_args()

    cache = args.cache.resolve()
    loaded = np.load(cache)
    xyz = np.asarray(loaded["xyz"], np.float64)
    origins = np.asarray(loaded["org"], np.float64)
    counts = np.asarray(loaded["counts"], np.int64)
    if counts.sum() != len(xyz) or origins.shape != (len(counts), 3):
        raise SystemExit("cache xyz/org/counts are not aligned")
    boundaries = np.r_[0, counts.cumsum()]
    n_needed = args.warm_sweeps + args.timed_sweeps
    if n_needed > len(counts):
        raise SystemExit(f"requested {n_needed} sweeps; cache has {len(counts)}")

    have_timestamps = "timestamps" in loaded and len(loaded["timestamps"]) == len(counts)
    have_groups = "groups" in loaded and len(loaded["groups"]) == len(counts)
    timestamps = (
        np.asarray(loaded["timestamps"], np.float64) if have_timestamps
        else np.arange(len(counts), dtype=np.float64) * args.nominal_group_s
    )
    groups = (
        np.asarray(loaded["groups"]) if have_groups
        else np.arange(len(counts), dtype=np.int64)
    )
    settings = {
        "voxel": args.voxel,
        "delete_max_range": args.max_range,
        "add_max_range": args.max_range,
    }
    map_ = KFCMap(**settings)

    def step(index: int) -> dict[str, object]:
        start = time.perf_counter()
        map_.update_batch(RayBatch.one_group(
            xyz[boundaries[index]:boundaries[index + 1]],
            origins[index],
            timestamp=float(timestamps[index]),
            group=groups[index],
            pose_uncertainty_m=0.0,
        ))
        map_.flush()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {"source_index": index, "wall_ms": elapsed_ms, **map_.last_metrics}

    for index in range(args.warm_sweeps):
        step(index)
    warm_voxels = len(map_.pts)
    rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if args.tracemalloc:
        tracemalloc.start()
    records = [step(index) for index in range(args.warm_sweeps, n_needed)]
    python_peak_bytes = tracemalloc.get_traced_memory()[1] if args.tracemalloc else None
    if args.tracemalloc:
        tracemalloc.stop()
    wall = np.asarray([row["wall_ms"] for row in records], np.float64)
    root = Path(__file__).resolve().parents[2]
    report = {
        "cache": {
            "path": str(cache),
            "sha256": _sha256(cache),
            "sweeps": len(counts),
            "rays": int(counts.sum()),
            "timestamps": "source" if have_timestamps else "synthesized nominal",
            "groups": "source" if have_groups else "synthesized sweep index",
        },
        "code": _git_version(root) | {
            "carve_sha256": _sha256(root / "rocklabel/geometry/carve.py"),
        },
        "settings": settings | {
            "warm_sweeps": args.warm_sweeps,
            "timed_sweeps": args.timed_sweeps,
            "nominal_group_s": args.nominal_group_s,
            "rendering": False,
            "inference": False,
            "decoding": False,
        },
        "summary": {
            "warm_voxels": warm_voxels,
            "final_voxels": len(map_.pts),
            "total_s": float(wall.sum() / 1000.0),
            "mean_ms": float(wall.mean()),
            "median_ms": float(np.median(wall)),
            "p95_ms": float(np.quantile(wall, 0.95)),
            "max_ms": float(wall.max()),
            "nominal_speed": float(
                args.timed_sweeps * args.nominal_group_s / (wall.sum() / 1000.0)
            ),
            "rss_high_water_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_before_timed_kib": rss_before_kib,
            "python_peak_bytes": python_peak_bytes,
            "deleted_total": map_.n_deleted,
            "invalid_total": map_.n_invalid,
        },
        "groups": records,
    }

    snapshot = _snapshot(map_)
    if args.compare:
        reference = np.load(args.compare)
        mismatch = [
            name for name, value in snapshot.items()
            if name not in reference or not np.array_equal(
                value, reference[name], equal_nan=True
            )
        ]
        report["comparison"] = {
            "path": str(args.compare.resolve()), "exact": not mismatch,
            "mismatched_arrays": mismatch,
        }
        if mismatch:
            print(json.dumps(report, indent=2))
            raise SystemExit("finalized map differs from comparison snapshot")
    if args.snapshot_out:
        args.snapshot_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.snapshot_out, **snapshot)
    encoded = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + os.linesep)
    print(encoded)


if __name__ == "__main__":
    main()
