"""Pool format-A runs into a flat cache and produce leakage-safe splits.

Why a cache: one epoch over ~75k samples would otherwise reopen thousands of
npz files. `build_cache` concatenates each run once into plain .npy arrays
(~100 MB/run) that load instantly and can be memory-mapped.

Why the split rules live here and not in the training script: candidate
centers sit on a 5 cm grid while neighborhoods span a 50 cm radius, and
consecutive frames barely move, so random sample splits leak near-duplicates
into the test set. The only splits this module hands out are (a) whole-run
holdouts (leave-one-run-out) and (b) contiguous frame blocks separated by a
temporal gap.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..generate import MANIFEST_NAME
from ..neighborhoods import FEATURES, resolve_features

# The four comparable "my room" runs (all generated with the same config).
DEFAULT_DATASETS = [
    "datasets/myroomdataset1",
    "datasets/myroomdataset2",
    "datasets/myroomdataset3",
    "datasets/myroomdataset4",
]

CACHE_ARRAYS = ("points", "labels", "counts", "centers", "frame")


class DataError(SystemExit):
    pass


def _run_entries(dataset_dirs: list[str]) -> list[tuple[str, str, dict, dict]]:
    """[(dataset_dir, run_id, manifest_entry, manifest)] for every run found."""
    out = []
    for d in dataset_dirs:
        path = os.path.join(d, MANIFEST_NAME)
        if not os.path.exists(path):
            raise DataError(f"{d!r} has no {MANIFEST_NAME} - not a generated dataset")
        with open(path) as f:
            manifest = json.load(f)
        for run_id, entry in sorted(manifest["runs"].items()):
            out.append((d, run_id, entry, manifest))
    return out


def build_cache(dataset_dirs: list[str], cache_dir: str) -> dict:
    """Concatenate every run's points/*.npz into cache_dir/<run_id>/*.npy.

    Refuses to pool datasets with differing config_hash (their samples were
    built with different neighborhood geometry and are not comparable), and
    verifies the concatenated label counts against each manifest's
    sample_labels totals. Returns the cache meta dict.
    """
    entries = _run_entries(dataset_dirs)
    hashes = {m["config_hash"] for _, _, _, m in entries}
    if len(hashes) != 1:
        detail = ", ".join(f"{d}={m['config_hash'][:12]}" for d, _, _, m in entries)
        raise DataError(f"refusing to pool datasets with mixed config hashes: {detail}")
    config_hash = hashes.pop()
    gcfg = entries[0][3]["config"]["generator"]

    runs_meta = {}
    for d, run_id, entry, _ in entries:
        if run_id in runs_meta:
            raise DataError(f"run_id {run_id!r} appears in more than one dataset")
        src = os.path.join(d, "points", run_id)
        files = sorted(f for f in os.listdir(src) if f.startswith("frame_") and f.endswith(".npz"))
        pts, labels, counts, centers, frame_idx, times = [], [], [], [], [], {}
        for name in files:
            with np.load(os.path.join(src, name)) as z:
                fi = int(name[6:12])
                pts.append(z["neighborhoods"])
                labels.append(z["labels"])
                counts.append(z["true_counts"])
                centers.append(z["centers_odom"])
                frame_idx.append(np.full(len(z["labels"]), fi, np.int32))
                times[fi] = float(z["frame_time"])
        arrays = {
            "points": np.concatenate(pts).astype(np.float32),
            "labels": np.concatenate(labels).astype(np.int8),
            "counts": np.concatenate(counts).astype(np.int16),
            "centers": np.concatenate(centers).astype(np.float32),
            "frame": np.concatenate(frame_idx),
        }
        want = entry["sample_labels"]
        got_rock = int((arrays["labels"] == 1).sum())
        got_clear = int((arrays["labels"] == 0).sum())
        if got_rock != want["rock"] or got_clear != want["clear"]:
            raise DataError(
                f"run {run_id}: cached counts (rock {got_rock}, clear {got_clear}) "
                f"disagree with manifest sample_labels {want} - regenerate the dataset"
            )
        run_dir = os.path.join(cache_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        for key, arr in arrays.items():
            np.save(os.path.join(run_dir, f"{key}.npy"), arr)
        runs_meta[run_id] = {
            "dataset_dir": os.path.abspath(d),
            "n": len(arrays["labels"]),
            "rock": got_rock,
            "clear": got_clear,
            "frames": len(files),
            "frame_times": times,
        }
        print(f"cached {run_id}: {len(arrays['labels'])} samples "
              f"({got_rock} rock / {got_clear} clear) from {len(files)} frames "
              f"- matches manifest")

    meta = {"config_hash": config_hash, "generator": gcfg, "runs": runs_meta}
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    total = sum(r["n"] for r in runs_meta.values())
    rock = sum(r["rock"] for r in runs_meta.values())
    print(f"pooled {len(runs_meta)} runs: {total} samples, {rock} rock ({rock / total:.1%})")
    return meta


class RunData:
    """One cached run, loaded into RAM (a run is ~100 MB)."""

    def __init__(self, cache_dir: str, run_id: str):
        d = os.path.join(cache_dir, run_id)
        if not os.path.isdir(d):
            raise DataError(f"no cache for run {run_id!r} - run 'rocklabel-train cache' first")
        self.run_id = run_id
        for key in CACHE_ARRAYS:
            setattr(self, key, np.load(os.path.join(d, f"{key}.npy")))

    def __len__(self) -> int:
        return len(self.labels)


def load_cache_meta(cache_dir: str) -> dict:
    path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(path):
        raise DataError(f"{cache_dir!r} has no meta.json - run 'rocklabel-train cache' first")
    with open(path) as f:
        return json.load(f)


def run_suffix(features: list[str] | None) -> str:
    """Name fragment identifying a non-default input-channel selection.

    Empty for the full channel set, so directories written before the setting
    existed keep their names and keep resuming. Anything else is tagged, which
    is what lets 'same fold, different channels' experiments sit side by side
    instead of colliding on one run directory - the whole point of being able
    to train with and without reflectivity off one cache.
    """
    chosen = resolve_features(features)
    return "" if chosen == list(FEATURES) else "_" + "-".join(chosen)


def run_dir_name(model: str, fold_name: str, features: list[str] | None = None) -> str:
    """Directory name for one trained fold, e.g. ``pointnet_loro_run3_dx-dy-dz``."""
    return f"{model}_{fold_name}{run_suffix(features)}"


def loro_folds(run_ids: list[str]) -> list[dict]:
    """Leave-one-run-out: each fold tests on one whole run, trains on the rest."""
    runs = sorted(run_ids)
    return [{"name": f"loro_{test}", "test": test,
             "train": [r for r in runs if r != test]} for test in runs]


def block_val_mask(frame: np.ndarray, val_frac: float = 0.15,
                   gap_frames: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """(train_mask, val_mask) over one run's samples, split by contiguous frames.

    The last val_frac of the run's frames become validation; the gap_frames
    frames just before them belong to neither side, so no neighborhood in
    train shares points (or a near-identical robot pose) with one in val.
    """
    uniq = np.unique(frame)
    n_val = max(int(round(len(uniq) * val_frac)), 1)
    val_start = uniq[len(uniq) - n_val]
    gap_start = uniq[max(len(uniq) - n_val - gap_frames, 0)]
    val = frame >= val_start
    train = frame < gap_start
    return train, val


def check_no_frame_overlap(train_runs: dict[str, np.ndarray],
                           test_runs: dict[str, np.ndarray]) -> None:
    """Assert no (run, frame) pair appears on both sides of a split."""
    for run_id, frames in test_runs.items():
        if run_id in train_runs:
            common = np.intersect1d(np.unique(train_runs[run_id]), np.unique(frames))
            if len(common):
                raise DataError(f"split leak: run {run_id} frames {common[:5]} on both sides")
