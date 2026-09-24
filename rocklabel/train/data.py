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

from ..dataset.generate import MANIFEST_NAME
from ..dataset.neighborhoods import FEATURES, resolve_features
from ..profiles import DEFAULT_PROFILE, identify as identify_profile

#: Where datasets live, one folder per generation profile.
DATASETS_ROOT = "datasets"


def default_datasets(profile: str = DEFAULT_PROFILE,
                     root: str = DATASETS_ROOT) -> list[str]:
    """Every dataset built with ``profile``, which is what `cache` pools by default.

    Read off disk rather than hardcoded. The old hardcoded list still named
    four "my room" datasets long after the project had moved to the volleyball
    court, so `cache` with no arguments quietly pooled the wrong recordings.
    """
    base = os.path.join(root, profile)
    if not os.path.isdir(base):
        return []
    return [os.path.join(base, n) for n in sorted(os.listdir(base))
            if os.path.exists(os.path.join(base, n, MANIFEST_NAME))]

CACHE_ARRAYS = ("points", "labels", "counts", "centers", "frame")
#: Format C (whole-frame segmentation). Cached beside format A from the same
#: dataset, so one `cache` call serves both training tasks and a fold's
#: train/test split means the same thing for either.
SEG_ARRAYS = ("seg_points", "seg_labels", "seg_counts", "seg_frame", "seg_base")
#: Per-sample diagnostics the version-2 builder writes beside format A (see
#: dataset/history.py). Cached when present so a run can be stratified by ball
#: size, support and scan age without regenerating; nothing trains on them.
EXTRA_ARRAYS = ("radius", "ball_count", "ball_voxels", "ball_sweeps", "point_age")


class DataError(SystemExit):
    pass


def _profile_of(manifest: dict) -> str:
    """The profile a dataset names, or the one its config hash matches."""
    name = manifest.get("profile") or identify_profile(manifest.get("config_hash", ""))
    return name or f"an unnamed config ({manifest.get('config_hash', '?')[:12]})"


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
    if not dataset_dirs:
        raise DataError(
            f"no datasets to pool. Generate some first "
            f"(rocklabel generate --profile {DEFAULT_PROFILE} ...), or name them "
            "explicitly with --datasets.")
    entries = _run_entries(dataset_dirs)
    hashes = {m["config_hash"] for _, _, _, m in entries}
    if len(hashes) != 1:
        # Name the profiles, not just the hashes: "raw-burst and full-sweep"
        # says what went wrong, where two hex strings only say that something
        # did. One line per dataset, not per run inside it.
        by_dir = {d: _profile_of(m) for d, _, _, m in entries}
        named = sorted(set(by_dir.values()))
        raise DataError(
            "refusing to pool datasets built different ways - their frames hold "
            "different numbers of points, so their samples are not comparable "
            f"and a score across them would mean nothing. Found {' and '.join(named)}. "
            "Build one cache per profile.\n  "
            + "\n  ".join(f"{d} = {p}" for d, p in sorted(by_dir.items())))
    config_hash = hashes.pop()
    profile = _profile_of(entries[0][3])
    gcfg = entries[0][3]["config"]["generator"]
    # One cache directory is one population. Writing a different config's
    # runs over an existing cache would leave the old run folders beside the
    # new meta.json, and a resumed training run would read them as the same
    # data. Refuse rather than overwrite.
    old_meta = os.path.join(cache_dir, "meta.json")
    if os.path.exists(old_meta):
        with open(old_meta) as f:
            previous = json.load(f).get("config_hash")
        if previous and previous != config_hash:
            raise DataError(
                f"{cache_dir!r} already holds a cache built from a different dataset "
                f"config ({previous[:12]} vs {config_hash[:12]}). Pick another "
                "--cache-dir, or delete that folder first.")

    runs_meta = {}
    for d, run_id, entry, _ in entries:
        if run_id in runs_meta:
            raise DataError(f"run_id {run_id!r} appears in more than one dataset")
        src = os.path.join(d, "points", run_id)
        files = sorted(f for f in os.listdir(src) if f.startswith("frame_") and f.endswith(".npz"))
        pts, labels, counts, centers, frame_idx, times = [], [], [], [], [], {}
        extras: dict[str, list] = {k: [] for k in EXTRA_ARRAYS}
        history: dict[int, dict] = {}
        for name in files:
            with np.load(os.path.join(src, name)) as z:
                fi = int(name[6:12])
                pts.append(z["neighborhoods"])
                labels.append(z["labels"])
                counts.append(z["true_counts"])
                centers.append(z["centers_odom"])
                frame_idx.append(np.full(len(z["labels"]), fi, np.int32))
                times[fi] = float(z["frame_time"])
                for key in EXTRA_ARRAYS:
                    if key in z.files:
                        extras[key].append(z[key])
                # Which sweeps each anchor's balls were cut from, so a split
                # can be checked for shared source sweeps across its sides.
                if "history_sweep_ids" in z.files:
                    history[fi] = {"sweep_ids": z["history_sweep_ids"].tolist(),
                                   "ages": [None if not np.isfinite(a) else float(a)
                                            for a in z["history_ages"]]}
        arrays = {
            "points": np.concatenate(pts).astype(np.float32),
            "labels": np.concatenate(labels).astype(np.int8),
            "counts": np.concatenate(counts).astype(np.int16),
            "centers": np.concatenate(centers).astype(np.float32),
            "frame": np.concatenate(frame_idx),
        }
        for key, parts in extras.items():
            if parts and len(parts) == len(files):
                arrays[key] = np.concatenate(parts)
        want = entry["sample_labels"]
        got_rock = int((arrays["labels"] == 1).sum())
        got_clear = int((arrays["labels"] == 0).sum())
        if got_rock != want["rock"] or got_clear != want["clear"]:
            raise DataError(
                f"run {run_id}: cached counts (rock {got_rock}, clear {got_clear}) "
                f"disagree with manifest sample_labels {want} - regenerate the dataset"
            )
        # Format C, when the dataset has it (datasets generated before the
        # segmentation format existed simply have no seg/ directory).
        seg_src = os.path.join(d, "seg", run_id)
        seg_arrays: dict[str, np.ndarray] = {}
        if os.path.isdir(seg_src):
            sp, sl, sc, sf, sb = [], [], [], [], []
            for name in sorted(f for f in os.listdir(seg_src)
                               if f.startswith("frame_") and f.endswith(".npz")):
                with np.load(os.path.join(seg_src, name)) as z:
                    sp.append(z["points"])
                    sl.append(z["labels"])
                    sc.append(z["true_count"])
                    sb.append(z["base_odom"])
                    sf.append(int(name[6:12]))
            if sp:
                seg_arrays = {
                    "seg_points": np.stack(sp).astype(np.float32),
                    "seg_labels": np.stack(sl).astype(np.int8),
                    "seg_counts": np.asarray(sc, np.int32),
                    "seg_frame": np.asarray(sf, np.int32),
                    "seg_base": np.stack(sb).astype(np.float32),
                }

        run_dir = os.path.join(cache_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        for key, arr in {**arrays, **seg_arrays}.items():
            np.save(os.path.join(run_dir, f"{key}.npy"), arr)
        runs_meta[run_id] = {
            "dataset_dir": os.path.abspath(d),
            "n": len(arrays["labels"]),
            "rock": got_rock,
            "clear": got_clear,
            "frames": len(files),
            "frame_times": times,
            "seg_frames": int(len(seg_arrays["seg_counts"])) if seg_arrays else 0,
            "extra_arrays": sorted(k for k in EXTRA_ARRAYS if k in arrays),
            "z_band": entry.get("z_band"),
            "level": entry.get("level"),
            "source_sweep_cache": entry.get("sweep_cache"),
            "diagnostics": entry.get("diagnostics"),
        }
        if history:
            with open(os.path.join(run_dir, "history.json"), "w") as f:
                json.dump({str(k): v for k, v in sorted(history.items())}, f)
        print(f"cached {run_id}: {len(arrays['labels'])} samples "
              f"({got_rock} rock / {got_clear} clear) from {len(files)} frames "
              f"- matches manifest")

    # The cache says which profile and which datasets it came from, so a run
    # trained off it can be traced back to how its frames were cut.
    from ..dataset.history import input_contract
    meta = {"config_hash": config_hash, "profile": profile,
            "datasets": [os.path.abspath(d) for d in dataset_dirs],
            "generator": gcfg, "input_contract": input_contract(gcfg),
            "runs": runs_meta}
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    total = sum(r["n"] for r in runs_meta.values())
    rock = sum(r["rock"] for r in runs_meta.values())
    print(f"pooled {len(runs_meta)} runs from profile {profile!r}: "
          f"{total} samples, {rock} rock ({rock / total:.1%})")
    return meta


class RunData:
    """One cached run, loaded into RAM (a run is ~100 MB).

    ``task="segment"`` loads format C instead and re-exposes it under the
    task-neutral names the training engine uses (points / labels / counts /
    frame), so one Split implementation serves both.
    """

    def __init__(self, cache_dir: str, run_id: str, task: str = "classify"):
        d = os.path.join(cache_dir, run_id)
        if not os.path.isdir(d):
            raise DataError(f"no cache for run {run_id!r} - run 'rocklabel-train cache' first")
        self._dir = d
        self.run_id = run_id
        self.task = task
        if task == "segment":
            missing = [k for k in SEG_ARRAYS
                       if not os.path.exists(os.path.join(d, f"{k}.npy"))]
            if missing:
                raise DataError(
                    f"run {run_id!r} has no segmentation data in {cache_dir!r}. It was "
                    "cached from a dataset generated before format C existed - "
                    "regenerate with 'rocklabel generate' and re-run "
                    "'rocklabel-train cache'.")
            for key in SEG_ARRAYS:
                setattr(self, key, np.load(os.path.join(d, f"{key}.npy")))
            self.points, self.labels = self.seg_points, self.seg_labels
            self.counts, self.frame = self.seg_counts, self.seg_frame
            self.centers = self.seg_base
        else:
            for key in CACHE_ARRAYS:
                setattr(self, key, np.load(os.path.join(d, f"{key}.npy")))

    def __len__(self) -> int:
        return len(self.labels)

    def extra(self, key: str) -> np.ndarray | None:
        """A per-sample diagnostic array (see EXTRA_ARRAYS), or None if absent."""
        path = os.path.join(self._dir, f"{key}.npy")
        return np.load(path) if os.path.exists(path) else None

    def append_point_age(self) -> None:
        """Append each row's age in seconds as channel AGE_CHANNEL of ``points``.

        For models that read it (``reads_point_age``). Only version-2 datasets
        record ages, so a cache without them is refused rather than filled with
        zeros - a model trained on zeros would silently learn nothing from the
        channel and still claim to be the age model.
        """
        from ..dataset.neighborhoods import AGE_CHANNEL
        age = self.extra("point_age")
        if age is None:
            raise DataError(f"run {self.run_id!r} has no point ages - only history "
                            "(preprocessing version 2) caches record them")
        if self.points.shape[-1] != AGE_CHANNEL:
            raise DataError(f"run {self.run_id!r}: expected {AGE_CHANNEL} stored "
                            f"channels before the age, found {self.points.shape[-1]}")
        self.points = np.concatenate(
            [self.points, age.astype(np.float32)[..., None]], axis=-1)


def check_split_runs(train_runs: list[str], val_runs: list[str],
                     test_run: str = "") -> None:
    """Refuse a whole-recording split whose sides share a recording.

    Whole-recording validation is what makes a 30-second history safe: two
    recordings share no sweep, so no ball in one side can contain a point the
    other side was built from. That only holds if the lists are disjoint.
    """
    both = sorted(set(train_runs) & set(val_runs))
    if both:
        raise DataError(f"split leak: {both} are both training and validation runs")
    if test_run and test_run in set(train_runs) | set(val_runs):
        raise DataError(f"split leak: test run {test_run!r} is also trained or validated on")
    if len(set(val_runs)) != len(val_runs):
        raise DataError(f"duplicate validation run in {val_runs}")


def check_no_shared_sweeps(cache_dir: str, train_runs: list[str],
                           val_runs: list[str]) -> None:
    """Assert no (run, source sweep) feeds both sides of a split.

    Trivially true for whole-recording splits; kept as an executable check so
    a future temporal split with long history cannot quietly share sweeps.
    """
    def sweeps_of(runs):
        out = set()
        for r in runs:
            path = os.path.join(cache_dir, r, "history.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for frame in json.load(f).values():
                    out.update((r, int(i)) for i in frame["sweep_ids"] if int(i) >= 0)
        return out
    shared = sweeps_of(train_runs) & sweeps_of(val_runs)
    if shared:
        raise DataError(f"split leak: {len(shared)} source sweeps feed both sides, "
                        f"e.g. {sorted(shared)[:3]}")


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
                   gap_frames: int = 25, times: dict | None = None,
                   gap_seconds: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(train_mask, val_mask) over one run's samples, split by contiguous frames.

    The last val_frac of the run's frames become validation; the frames just
    before them belong to neither side, so no neighborhood in train shares
    points (or a near-identical robot pose) with one in val.

    Sizing that buffer in frames alone is a trap. ``gap_frames`` counts *kept*
    frames, and on this sensor (a ~225 Hz multiScan, stride 5) 25 of them span
    0.54 s — during which the robot barely moves, so the two sides stayed full
    of near-duplicates and validation read optimistically. Pass ``times``
    (run_id frame index -> wall clock, straight out of the cache meta) with
    ``gap_seconds`` to size the buffer in seconds instead; the frame count
    stays as a floor.
    """
    uniq = np.unique(frame)
    n_val = max(int(round(len(uniq) * val_frac)), 1)
    val_start = uniq[len(uniq) - n_val]
    n_gap = int(gap_frames)
    if times and gap_seconds:
        t = {int(k): float(v) for k, v in times.items()}
        cutoff = t[int(val_start)] - float(gap_seconds)
        before = uniq[:len(uniq) - n_val]
        n_gap = max(n_gap, int(sum(t[int(f)] > cutoff for f in before)))
    gap_start = uniq[max(len(uniq) - n_val - n_gap, 0)]
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
