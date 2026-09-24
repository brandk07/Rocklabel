"""Decode a recording once into assembled sweeps that every arm can reuse.

A neighborhood/history campaign builds the same eleven recordings under
twenty-four policies. Each policy needs the *same* sweeps - same levelling,
same window assembly, same poses - and the old sampled datasets cannot supply
them: they hold 256-point balls, not the clouds the balls were cut from. So
the sweeps themselves are cached, with everything a history builder needs:
time, source index, both poses and the full uncropped geometry.

The cache records what it was built from (recording and label hashes, the
levelling it was replayed under, the topic/level/window settings) and a reader
refuses one whose identity does not match, so a relabelled recording or a
changed window can never be paired with stale sweeps.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import numpy as np

from ..recording.pipeline import OdomScan

SCHEMA = 1
ARRAYS = ("xyz", "intensity", "offsets", "times", "index", "T_odom_base", "T_odom_lidar")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_identity(mcap_path: str, labels_path: str, cfg: dict) -> dict:
    """What a sweep cache's contents depend on - and nothing it does not."""
    g = cfg["generator"]
    return {"schema": SCHEMA,
            "recording_sha256": file_sha256(mcap_path),
            "labels_sha256": file_sha256(labels_path),
            "topics": cfg["topics"], "level": cfg["level"],
            "frame_window_s": float(g.get("frame_window_s") or 0.0)}


def check_cache(cache_dir: str, identity: dict) -> tuple[bool, str]:
    path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(path):
        return False, "no sweep cache"
    with open(path) as f:
        meta = json.load(f)
    changed = sorted(k for k in identity if meta.get("identity", {}).get(k) != identity[k])
    if changed:
        return False, "built from different inputs: " + ", ".join(changed)
    missing = [a for a in ARRAYS if not os.path.exists(os.path.join(cache_dir, f"{a}.npy"))]
    if missing:
        return False, "incomplete: missing " + ", ".join(missing)
    return True, ""


def build_cache(mcap_path: str, labels_path: str, cfg: dict, cache_dir: str) -> dict:
    """Decode, level and window one recording; publish atomically."""
    from ..geometry.leveling import check_level_match, level_record, pin_level_to_labels
    from ..labels import load_labels
    from ..recording.pipeline import ScanStream, WindowedScanStream

    labelset = load_labels(labels_path)
    window_s = float(cfg["generator"].get("frame_window_s") or 0.0)
    stream = ScanStream(mcap_path, pin_level_to_labels(cfg, labelset.level), stride=1,
                        progress=False)
    check_level_match(labelset.level, level_record(stream), labels_path)
    sweeps = WindowedScanStream(stream, window_s) if window_s > 0 else stream

    xyz, inten, offsets, times, index, base, lidar = [], [], [0], [], [], [], []
    for scan in sweeps:
        xyz.append(np.asarray(scan.xyz_odom, np.float32))
        inten.append(np.asarray(scan.intensity, np.float32))
        offsets.append(offsets[-1] + len(scan.xyz_odom))
        times.append(float(scan.time_s))
        index.append(int(scan.index))
        base.append(np.asarray(scan.T_odom_base, np.float64))
        lidar.append(np.asarray(scan.T_odom_lidar, np.float64))
    if not times:
        raise SystemExit(f"{mcap_path}: no sweeps decoded")
    arrays = {"xyz": np.concatenate(xyz), "intensity": np.concatenate(inten),
              "offsets": np.asarray(offsets, np.int64), "times": np.asarray(times),
              "index": np.asarray(index, np.int64), "T_odom_base": np.stack(base),
              "T_odom_lidar": np.stack(lidar)}
    dt = np.diff(arrays["times"])
    meta = {"identity": cache_identity(mcap_path, labels_path, cfg),
            "recording": os.path.abspath(mcap_path),
            "labels": os.path.abspath(labels_path),
            "level": level_record(stream),
            "intensity_available": bool(stream.counters.intensity_available),
            "skipped_pose": int(stream.counters.skipped_pose),
            "stamp_fallbacks": int(stream.counters.stamp_fallbacks),
            "sweeps": len(times), "points": int(offsets[-1]),
            "duration_s": float(times[-1] - times[0]),
            "sweep_interval_s": {"median": float(np.median(dt)) if len(dt) else None,
                                 "p99": float(np.percentile(dt, 99)) if len(dt) else None,
                                 "max": float(dt.max()) if len(dt) else None}}
    staging = cache_dir.rstrip("/") + ".partial"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    for key, arr in arrays.items():
        np.save(os.path.join(staging, f"{key}.npy"), arr)
    with open(os.path.join(staging, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(cache_dir)), exist_ok=True)
    os.replace(staging, cache_dir)
    return meta


class CachedSweeps:
    """Iterate a sweep cache as OdomScan objects, in recorded order."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)
        for key in ARRAYS:
            setattr(self, "_" + key, np.load(os.path.join(cache_dir, f"{key}.npy"),
                                             mmap_mode="r"))

    def __len__(self) -> int:
        return int(len(self._times))

    def __iter__(self):
        for i in range(len(self)):
            a, b = int(self._offsets[i]), int(self._offsets[i + 1])
            yield OdomScan(index=int(self._index[i]), time_s=float(self._times[i]),
                           xyz_odom=np.asarray(self._xyz[a:b]),
                           intensity=np.asarray(self._intensity[a:b]),
                           T_odom_base=np.asarray(self._T_odom_base[i]),
                           T_odom_lidar=np.asarray(self._T_odom_lidar[i]))
