"""Neighborhood-size x scan-history screen for the PointNet classifier.

Trains on the eleven Volleyball recordings only and judges on the Lance
competition recording, under a frozen recording-level split (VB4 and VB6
validate, the other nine train). Twenty-four arms: six neighborhood policies
crossed with four history policies, every other setting frozen.

Phases, in order (each resumable, each writes atomic per-job status):

``plan``        print the exact job list, configs, commands and output paths.
``preflight``   check inputs, GPU and disk; hash every input and the source
                tree; record label levelling, frames and height bands.
``prepare``     decode each recording once into a sweep cache, then build every
                arm's dataset (all eleven recordings) and cache from it.
``smoke``       one recording through all 24 policies, two tiny fits, and a
                short Lance replay proving the audit applies each contract.
``train``       one fit per arm for ``--seed``, one trainer on the GPU.
``evaluate``    whole-recording Lance map evaluation, visual audits against
                the historical reference and the fresh baseline, and a
                startup-window audit for history arms.
``alignment``   surface thickness / double-surface diagnostics per history.
``summarize``   machine-readable comparison tables and results.md.

Nothing here reads Lance labels during training, and no epoch or threshold is
chosen on Lance: best.pt is the Volleyball-validation epoch and its stored
threshold is the Volleyball-validation best-F1 threshold. Lance is consulted
repeatedly while the campaign is steered, which makes it a development
benchmark rather than an untouched test set - reports say so.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

CAMPAIGN = "neighborhood-history-v1"
REPORT_ROOT = os.path.join("training", "reports", CAMPAIGN)
DATASET_ROOT = os.path.join("datasets", CAMPAIGN)
CACHE_ROOT = os.path.join("training", "caches", CAMPAIGN)
EXPERIMENT_ROOT = os.path.join("training", "experiments", CAMPAIGN)
SWEEP_ROOT = os.path.join(CACHE_ROOT, "_sweeps")
MAPEVAL_FRAMES = os.path.join(CACHE_ROOT, "_mapeval-lance")
PROFILE = "full-sweep"

VB_RUNS = [f"VolleyBallTest{n}.reslam" for n in range(2, 13)]
VAL_RUNS = ["VolleyBallTest4.reslam", "VolleyBallTest6.reslam"]
TRAIN_RUNS = [r for r in VB_RUNS if r not in VAL_RUNS]
LANCE_RECORDING = ("recordings/archive/misc/"
                   "lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.mcap")
LANCE_LABELS = ("labels/archive/"
                "lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json")
REFERENCE = "training/experiments/deploy/cls-stray/trainall/best.pt"


def vb_recording(run: str) -> str:
    return f"recordings/volleyball/reslam/{run}.mcap"


def vb_labels(run: str) -> str:
    return f"labels/volleyball/{run}.labels.json"


#: Six neighborhood policies. Order is not priority; see PRIORITY.
NEIGHBORHOODS: dict[str, dict] = {
    "fixed-r020": {"neighborhood_mode": "fixed", "neighborhood_radius_m": 0.20},
    "fixed-r030": {"neighborhood_mode": "fixed", "neighborhood_radius_m": 0.30},
    "fixed-r050": {"neighborhood_mode": "fixed", "neighborhood_radius_m": 0.50},
    "fixed-r075": {"neighborhood_mode": "fixed", "neighborhood_radius_m": 0.75},
    "adaptive-r020-r050-k256": {"neighborhood_mode": "adaptive",
                                "adaptive_radius_min_m": 0.20,
                                "adaptive_radius_max_m": 0.50, "adaptive_k": 256},
    "adaptive-r020-r075-k256": {"neighborhood_mode": "adaptive",
                                "adaptive_radius_min_m": 0.20,
                                "adaptive_radius_max_m": 0.75, "adaptive_k": 256},
}
#: Four history policies: desired sweep ages in seconds.
HISTORIES: dict[str, list[float]] = {
    "h1": [0.0],
    "h3-4s": [0.0, 2.0, 4.0],
    "h5-8s": [0.0, 2.0, 4.0, 6.0, 8.0],
    "h5-30s": [0.0, 7.5, 15.0, 22.5, 30.0],
}
HISTORY_TOLERANCE_S = 0.10
BASELINE = "fixed-r050__h1"
LONGEST = "adaptive-r020-r075-k256__h5-30s"
SMOKE_RUN = "VolleyBallTest6.reslam"


def arm_id(nb: str, h: str) -> str:
    return f"{nb}__{h}"


ARMS = [arm_id(nb, h) for nb in NEIGHBORHOODS for h in HISTORIES]
#: Training order: the baseline first, then the pairs that answer one question
#: each against it (history at the reference radius, radius without history,
#: adaptive without history), then the rest. Stopping early still leaves the
#: most informative comparisons finished.
PRIORITY = ([BASELINE]
            + [arm_id("fixed-r050", h) for h in ("h3-4s", "h5-8s", "h5-30s")]
            + [arm_id(nb, "h1") for nb in ("fixed-r030", "adaptive-r020-r050-k256",
                                           "fixed-r075", "adaptive-r020-r075-k256",
                                           "fixed-r020")]
            + [LONGEST])
PRIORITY += [a for a in ARMS if a not in PRIORITY]

#: Feature follow-ups: a model change on an existing arm's dataset, named
#: ``<arm>+<variant>``. Run on the two configurations that led the three-seed
#: confirmation at equal false area. Query height is the stored fifth channel;
#: point age is each row's seconds-before-now, which only history arms record.
FOLLOWUP_VARIANTS = {"qz": {"model": "pointnet_qz"}, "age": {"model": "pointnet_age"}}
FOLLOWUP_BASES = ("fixed-r075__h5-8s", "fixed-r050__h5-30s")
FOLLOWUPS = [f"{b}+{v}" for b in FOLLOWUP_BASES for v in FOLLOWUP_VARIANTS]
ALL_ARMS = PRIORITY + FOLLOWUPS


def base_arm(arm: str) -> str:
    """The dataset arm a follow-up is trained on (itself for a plain arm)."""
    return arm.split("+")[0]


def variant_settings(arm: str) -> dict:
    return FOLLOWUP_VARIANTS[arm.split("+")[1]] if "+" in arm else {}

#: Frozen stage-1 training settings. Every other TRAIN_DEFAULTS value is
#: resolved and saved with each fit.
TRAIN_SETTINGS = {
    "model": "pointnet", "features": ["dx", "dy", "dz"], "tnet": False,
    "epochs": 60, "patience": 15, "batch": 256, "lr": 1e-3, "weight_decay": 1e-4,
    "augment": True, "aug_thin_min": 0.5, "aug_stray_frac": 0.05,
    "aug_stray_reach": 1.0, "aug_phantom_frac": 0.0,
}
SCREEN_SEED = 42
CONFIRM_SEEDS = (43, 44)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_json(path: str, value) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _sha256(path: str) -> str:
    from ..dataset.sweep_cache import file_sha256
    return file_sha256(path)


def status_path(phase: str, job: str) -> str:
    return os.path.join(REPORT_ROOT, "status", phase, f"{job}.json")


def log_path(phase: str, job: str) -> str:
    return os.path.join(REPORT_ROOT, "logs", phase, f"{job}.log")


def set_status(phase: str, job: str, state: str, **extra) -> dict:
    path = status_path(phase, job)
    old = _read_json(path) or {}
    old.update(state=state, updated=_now(), **extra)
    if state == "running":
        old["started"] = old["updated"]
    _atomic_json(path, old)
    return old


def job_done(phase: str, job: str) -> bool:
    return (_read_json(status_path(phase, job)) or {}).get("state") == "done"


def source_tree_hash() -> str:
    """Hash every Python source under rocklabel/, tracked or not."""
    h = hashlib.sha256()
    for path in sorted(glob.glob("rocklabel/**/*.py", recursive=True)):
        h.update(path.encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def git_provenance() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
        except OSError:
            return ""
    diff = run("git", "diff", "HEAD")
    return {"head": run("git", "rev-parse", "HEAD").strip(),
            "dirty_files": [ln for ln in run("git", "status", "--porcelain").splitlines()],
            "dirty_patch_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "source_tree_sha256": source_tree_hash()}


# --------------------------------------------------------------------------- #
# Arm configuration
# --------------------------------------------------------------------------- #
def arm_overrides(arm: str) -> dict:
    nb, h = base_arm(arm).split("__")
    return {"preprocessing_version": 2, "formats": ["points"],
            **NEIGHBORHOODS[nb], "history_ages_s": list(HISTORIES[h]),
            "history_tolerance_s": HISTORY_TOLERANCE_S}


def arm_yaml_path(arm: str) -> str:
    return os.path.join(REPORT_ROOT, "configs", f"{base_arm(arm)}.yaml")


def write_arm_yaml(arm: str) -> str:
    import yaml
    path = arm_yaml_path(arm)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = yaml.safe_dump({"generator": arm_overrides(arm)}, sort_keys=True)
    with open(path, "w") as f:
        f.write(f"# {CAMPAIGN} arm {arm}. Generated by neighborhood_campaign; "
                "applied under --profile full-sweep.\n" + body)
    return path


def resolved_config(arm: str) -> dict:
    """The exact config generation runs under: defaults <- arm YAML <- profile."""
    from ..config import load_config
    from ..profiles import apply_profile
    return apply_profile(load_config(write_arm_yaml(arm)), PROFILE)


def sweep_config() -> dict:
    """The config sweep caches are decoded under (topics, levelling, window)."""
    from ..config import load_config
    from ..profiles import apply_profile
    return apply_profile(load_config(None), PROFILE)


def dataset_dir(arm: str) -> str:
    return os.path.join(DATASET_ROOT, base_arm(arm))


def cache_dir(arm: str) -> str:
    return os.path.join(CACHE_ROOT, base_arm(arm))


def run_dir(arm: str, seed: int) -> str:
    return os.path.join(EXPERIMENT_ROOT, arm, f"seed-{seed}")


def sweep_dir(run: str) -> str:
    return os.path.join(SWEEP_ROOT, run)


def train_config(arm: str, seed: int, epochs: int | None = None,
                 patience: int | None = None, cache: str | None = None) -> dict:
    from .engine import default_config
    s = {**TRAIN_SETTINGS, **variant_settings(arm)}
    if epochs is not None:
        s["epochs"] = epochs
    if patience is not None:
        s["patience"] = patience
    return default_config(**s, cache_dir=cache or cache_dir(arm), train_runs=list(TRAIN_RUNS),
                          val_runs=list(VAL_RUNS), test_run="", seed=seed, device="cuda")


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def phase_plan(args) -> dict:
    from ..config import config_hash
    py = ".venv/bin/python"
    arms = []
    hashes = {}
    for arm in PRIORITY:
        cfg = resolved_config(arm)
        hashes[arm] = config_hash(cfg)
        g = cfg["generator"]
        arms.append({
            "arm": arm, "priority": PRIORITY.index(arm) + 1,
            "config_hash": hashes[arm][:12], "yaml": arm_yaml_path(arm),
            "generator": {k: g[k] for k in ("preprocessing_version", "neighborhood_mode",
                                            "neighborhood_radius_m", "adaptive_radius_min_m",
                                            "adaptive_radius_max_m", "adaptive_k",
                                            "history_ages_s", "history_tolerance_s",
                                            "frame_window_s", "frame_stride",
                                            "neighborhood_points", "min_neighbors",
                                            "negative_keep_prob", "formats")},
            "dataset": dataset_dir(arm), "cache": cache_dir(arm),
            "runs": {str(s): run_dir(arm, s) for s in (SCREEN_SEED, *CONFIRM_SEEDS)},
            "commands": {
                "generate": [f"{py} -m rocklabel.cli generate {vb_recording(r)} "
                             f"--labels {vb_labels(r)} --profile {PROFILE} "
                             f"--config {arm_yaml_path(arm)} --sweep-cache {sweep_dir(r)} "
                             f"--out {dataset_dir(arm)}" for r in VB_RUNS],
                "cache": (f"{py} -m rocklabel.train.cli cache --datasets {dataset_dir(arm)} "
                          f"--cache-dir {cache_dir(arm)}"),
                "train": (f"{py} -m rocklabel.train.neighborhood_campaign --phase train "
                          f"--arms {arm} --seed {SCREEN_SEED}"),
                "train_equivalent": (
                    f"{py} -m rocklabel.train.cli train --model pointnet --features dx dy dz "
                    f"--test-run all --val-runs {' '.join(VAL_RUNS)} "
                    f"--cache-dir {cache_dir(arm)} --epochs 60 --patience 15 --batch 256 "
                    f"--lr 0.001 --weight-decay 0.0001 --aug-thin-min 0.5 "
                    f"--aug-stray-frac 0.05 --aug-stray-reach 1.0 --aug-phantom-frac 0 "
                    f"--seed {SCREEN_SEED} --device cuda"),
            },
        })
    if len(set(hashes.values())) != len(hashes):
        raise SystemExit("two arms resolve to the same dataset config hash")
    plan = {"campaign": CAMPAIGN, "created": _now(), "profile": PROFILE,
            "train_runs": TRAIN_RUNS, "val_runs": VAL_RUNS, "baseline": BASELINE,
            "reference": REFERENCE, "lance": {"recording": LANCE_RECORDING,
                                              "labels": LANCE_LABELS},
            "train_settings": train_config(BASELINE, SCREEN_SEED),
            "screen_seed": SCREEN_SEED, "confirm_seeds": list(CONFIRM_SEEDS),
            "fits_in_stage_1": len(ARMS), "arms": arms,
            "sweep_caches": {r: sweep_dir(r) for r in VB_RUNS},
            "evaluation": {
                "mapeval_frames": MAPEVAL_FRAMES,
                "mapeval_out": os.path.join(REPORT_ROOT, "mapeval", "<arm>", "seed-<seed>"),
                "audit_out": os.path.join(REPORT_ROOT, "audits", "<arm>", "seed-<seed>",
                                          "<vs-reference|vs-baseline|startup>"),
                "audit_settings": "--floor-band -0.10 0.60 --max-range 8 --stride 10 "
                                  "--accum-seconds 5 --candidates-per-rock 3 --cell 0.10"}}
    _atomic_json(os.path.join(REPORT_ROOT, "plan.json"), plan)
    print(f"{CAMPAIGN}: {len(ARMS)} stage-1 fits (seed {SCREEN_SEED}), "
          f"train {len(TRAIN_RUNS)} recordings, validate on {', '.join(VAL_RUNS)}")
    print(f"{'#':>2}  {'arm':34} {'hash':12}  radius / history")
    for a in arms:
        g = a["generator"]
        radius = (f"fixed {g['neighborhood_radius_m']:.2f}" if g["neighborhood_mode"] == "fixed"
                  else f"adaptive {g['adaptive_radius_min_m']:.2f}-"
                       f"{g['adaptive_radius_max_m']:.2f} k{g['adaptive_k']}")
        print(f"{a['priority']:>2}  {a['arm']:34} {a['config_hash']}  {radius} / "
              f"ages {g['history_ages_s']}")
    print(f"\nexample commands for {BASELINE}:")
    first = arms[0]["commands"]
    print("  " + first["generate"][0] + "   (x11 recordings)")
    print("  " + first["cache"])
    print("  " + first["train_equivalent"])
    print(f"\nwrote {os.path.join(REPORT_ROOT, 'plan.json')}")
    return plan


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def phase_preflight(args) -> dict:
    import torch

    from ..labels import load_labels

    problems = []
    inputs = {}
    for run in VB_RUNS:
        for kind, path in (("recording", vb_recording(run)), ("labels", vb_labels(run))):
            if not os.path.exists(path):
                problems.append(f"missing {kind} {path}")
                continue
            inputs[path] = {"sha256": _sha256(path), "bytes": os.path.getsize(path)}
    for path in (LANCE_RECORDING, LANCE_LABELS, REFERENCE):
        if not os.path.exists(path):
            problems.append(f"missing {path}")
        else:
            inputs[path] = {"sha256": _sha256(path), "bytes": os.path.getsize(path)}

    labels = {}
    for run in VB_RUNS + ["lance"]:
        path = LANCE_LABELS if run == "lance" else vb_labels(run)
        if not os.path.exists(path):
            continue
        ls = load_labels(path)
        level = ls.level or {}
        floor = level.get("floor_z")
        band = ls.z_band
        labels[run] = {
            "run_id": ls.run_id, "rocks": len(ls.rocks),
            "arena_vertices": None if ls.arena is None else int(len(ls.arena)),
            "level": level, "z_band": None if band is None else list(band),
            # How far the label height band reaches above the floor. The band
            # replaces the crop's vertical limits in generation, so a narrow
            # one removes context every arm trains on.
            "band_above_floor_m": (None if band is None or floor is None
                                   else [round(band[0] - floor, 3), round(band[1] - floor, 3)]),
        }
        if run != "lance" and ls.run_id != run:
            problems.append(f"{path}: run_id {ls.run_id!r} != {run!r}")
        if not level:
            problems.append(f"{path}: labels carry no pinned levelling")

    stale = []
    m = _read_json("datasets/full-sweep/volleyball/manifest.json") or {}
    for run, e in (m.get("runs") or {}).items():
        for key in ("mcap_path", "labels_path"):
            if e.get(key) and not os.path.exists(e[key]):
                stale.append(f"{run}.{key}: {e[key]}")

    cuda = torch.cuda.is_available()
    disk = shutil.disk_usage(".")
    others = subprocess.run(["pgrep", "-af", "rocklabel.train|rocklabel-train"],
                            capture_output=True, text=True).stdout.splitlines()
    others = [ln for ln in others if str(os.getpid()) not in ln and "pgrep" not in ln
              and "--phase preflight" not in ln]
    result = {
        "checked": _now(), "problems": problems, "inputs": inputs, "labels": labels,
        "historical_manifest_stale_paths": stale,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "cuda": cuda,
                        "gpu": torch.cuda.get_device_name(0) if cuda else None,
                        "gpu_memory_mb": (torch.cuda.get_device_properties(0).total_memory
                                          // 2**20 if cuda else None),
                        "cpus": os.cpu_count(),
                        "disk_free_gb": round(disk.free / 1e9, 1)},
        "other_training_processes": others,
        "provenance": git_provenance(),
    }
    if not cuda:
        problems.append("no CUDA device")
    if disk.free < 50e9:
        problems.append(f"only {disk.free / 1e9:.0f} GB free")
    _atomic_json(os.path.join(REPORT_ROOT, "preflight.json"), result)
    print(f"inputs hashed: {len(inputs)}; GPU {result['environment']['gpu']}; "
          f"{result['environment']['disk_free_gb']} GB free")
    for run, lab in labels.items():
        print(f"  {run:26s} rocks {lab['rocks']:2d}  level {lab['level'].get('mode')}"
              f"  band above floor {lab['band_above_floor_m']}")
    if stale:
        print(f"  historical manifest names {len(stale)} stale paths "
              "(expected; this campaign resolves current locations)")
    if others:
        print("  other training processes running:\n    " + "\n    ".join(others))
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    print(f"preflight ok -> {os.path.join(REPORT_ROOT, 'preflight.json')}")
    return result


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def _peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def job_sweeps(run: str) -> dict:
    """Decode one recording into its sweep cache (skipped when current)."""
    from ..dataset.sweep_cache import build_cache, cache_identity, check_cache
    from ..geometry.leveling import pin_level_to_labels
    from ..labels import load_labels

    cfg = pin_level_to_labels(sweep_config(), load_labels(vb_labels(run)).level)
    ok, why = check_cache(sweep_dir(run), cache_identity(vb_recording(run), vb_labels(run), cfg))
    if ok:
        return _read_json(os.path.join(sweep_dir(run), "meta.json"))
    t0 = time.monotonic()
    meta = build_cache(vb_recording(run), vb_labels(run), cfg, sweep_dir(run))
    print(f"sweeps {run}: {meta['sweeps']} sweeps, {meta['points']} points, "
          f"{time.monotonic() - t0:.1f}s")
    return meta


def _arm_complete(arm: str) -> tuple[bool, str]:
    from ..config import config_hash
    want = config_hash(resolved_config(arm))
    man = _read_json(os.path.join(dataset_dir(arm), "manifest.json"))
    if not man:
        return False, "no dataset"
    if man.get("config_hash") != want:
        return False, "dataset built under a different config"
    missing = [r for r in VB_RUNS if r not in man.get("runs", {})]
    if missing:
        return False, f"dataset missing {missing}"
    meta = _read_json(os.path.join(cache_dir(arm), "meta.json"))
    if not meta or meta.get("config_hash") != want or sorted(meta["runs"]) != sorted(VB_RUNS):
        return False, "cache missing or stale"
    return True, ""


def validate_arm(arm: str) -> dict:
    """Fail visibly on a wrong, empty or degenerate arm; return its summary."""
    man = _read_json(os.path.join(dataset_dir(arm), "manifest.json"))
    g = man["config"]["generator"]
    want = arm_overrides(arm)
    bad = [k for k, v in want.items() if g.get(k) != v]
    if g.get("frame_window_s") != 0.05 or g.get("frame_stride") != 4:
        bad.append("profile (0.05 s sweeps, stride 4)")
    if bad:
        raise SystemExit(f"{arm}: resolved manifest does not hold the intended policy: {bad}")
    rows = {"warnings": []}
    for run, e in man["runs"].items():
        rock, clear = e["sample_labels"]["rock"], e["sample_labels"]["clear"]
        sel = e["candidates_selected"]
        uns = e["unscorable"]
        rows[run] = {"samples": rock + clear, "rock": rock,
                     "unscorable_rock_frac": uns["rock"] / max(sel["rock"], 1),
                     "unscorable_clear_frac": uns["clear"] / max(sel["clear"], 1)}
        if rock == 0 or clear == 0:
            raise SystemExit(f"{arm}/{run}: degenerate - {rock} rock, {clear} clear samples")
        # Not a refusal. How many rock candidates a small single-sweep ball
        # cannot score at all is one of the things this campaign measures (at
        # the historical 0.5 m geometry VB6 already loses 63%), and those
        # candidates stay in every Lance coverage denominator. Said loudly.
        if rows[run]["unscorable_rock_frac"] > 0.5:
            msg = (f"{arm}/{run}: {rows[run]['unscorable_rock_frac']:.0%} of rock "
                   f"candidates unscorable, {rock} rock samples kept")
            rows["warnings"].append(msg)
            print("WARNING " + msg)
    return rows


def job_prepare_arm(arm: str) -> dict:
    """Generate all eleven recordings for one arm, then cache and validate."""
    from ..dataset.generate import run_generate
    from .data import build_cache

    ok, why = _arm_complete(arm)
    if ok:
        return {"skipped": True, "validation": validate_arm(arm)}
    cfg = resolved_config(arm)
    t0 = time.monotonic()
    man = _read_json(os.path.join(dataset_dir(arm), "manifest.json")) or {}
    from ..config import config_hash
    if man and man.get("config_hash") != config_hash(cfg):
        raise SystemExit(f"{dataset_dir(arm)} holds another config; move it aside first")
    timings = {}
    for run in VB_RUNS:
        if run in (man.get("runs") or {}):
            continue
        t = time.monotonic()
        run_generate(vb_recording(run), vb_labels(run), dataset_dir(arm), cfg,
                     profile=PROFILE, sweep_cache=sweep_dir(run))
        timings[run] = round(time.monotonic() - t, 1)
    gen_s = time.monotonic() - t0
    if os.path.exists(os.path.join(cache_dir(arm), "meta.json")):
        meta = _read_json(os.path.join(cache_dir(arm), "meta.json"))
        if meta.get("config_hash") != config_hash(cfg) or sorted(meta["runs"]) != sorted(VB_RUNS):
            shutil.rmtree(cache_dir(arm))
    t1 = time.monotonic()
    build_cache([dataset_dir(arm)], cache_dir(arm))
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(cache_dir(arm), "*", "*")))
    ds_size = sum(os.path.getsize(p) for p in glob.glob(
        os.path.join(dataset_dir(arm), "points", "*", "*")))
    return {"generate_s": round(gen_s, 1), "per_recording_s": timings,
            "cache_s": round(time.monotonic() - t1, 1),
            "cache_mb": round(size / 2**20, 1), "dataset_mb": round(ds_size / 2**20, 1),
            "peak_rss_mb": _peak_rss_mb(), "validation": validate_arm(arm)}


def _run_job(phase: str, job: str, argv: list[str], force: bool = False) -> int:
    """Run one campaign job in a subprocess with its own log and status."""
    if job_done(phase, job) and not force:
        print(f"  skip {phase}/{job} (done)")
        return 0
    log = log_path(phase, job)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    set_status(phase, job, "running", log=log, argv=argv,
               source_tree_sha256=source_tree_hash())
    t0 = time.monotonic()
    with open(log, "a") as f:
        f.write(f"\n===== {_now()} {' '.join(argv)}\n")
        f.flush()
        rc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode
    state = "done" if rc == 0 else "failed"
    result = None
    with open(log) as f:
        for line in f:
            if line.startswith("JOB RESULT "):
                try:
                    result = json.loads(line[len("JOB RESULT "):])
                except ValueError:
                    result = line[len("JOB RESULT "):].strip()
    set_status(phase, job, state, returncode=rc, seconds=round(time.monotonic() - t0, 1),
               result=result)
    print(f"  {state:6s} {phase}/{job} in {time.monotonic() - t0:.0f}s"
          + (f"  (see {log})" if rc else ""))
    return rc


def _self(*extra: str) -> list[str]:
    return [sys.executable, "-m", "rocklabel.train.neighborhood_campaign", *extra]


def _pool(jobs, workers: int) -> list[int]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(lambda j: _run_job(*j), jobs))


def phase_prepare(args) -> None:
    arms = args.arms or PRIORITY
    print(f"decoding {len(VB_RUNS)} recordings into sweep caches")
    rc = _pool([("sweeps", run, _self("--phase", "job-sweeps", "--run", run), args.force)
                for run in VB_RUNS], args.workers)
    if any(rc):
        raise SystemExit("sweep decoding failed; see logs/sweeps/")
    for arm in arms:
        write_arm_yaml(arm)
    print(f"building {len(arms)} arm datasets and caches with {args.workers} workers")
    rc = _pool([("prepare", arm, _self("--phase", "job-prepare", "--arm", arm), args.force)
                for arm in arms], args.workers)
    failed = [a for a, r in zip(arms, rc) if r]
    if failed:
        raise SystemExit(f"prepare failed for {failed}; see logs/prepare/")


# --------------------------------------------------------------------------- #
# train / evaluate jobs
# --------------------------------------------------------------------------- #
def job_train(arm: str, seed: int, epochs: int | None = None, patience: int | None = None,
              out: str | None = None) -> dict:
    import torch

    from .engine import train_fold
    ok, why = _arm_complete(arm)
    if not ok:
        raise SystemExit(f"{arm} is not prepared: {why}")
    cfg = train_config(arm, seed, epochs, patience)
    rd = out or run_dir(arm, seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    summary = train_fold(cfg, rd)
    hist = []
    if os.path.exists(os.path.join(rd, "history.csv")):
        import csv
        with open(os.path.join(rd, "history.csv")) as f:
            hist = list(csv.DictReader(f))
    best = max(hist, key=lambda r: float(r["val_pr_auc"])) if hist else {}
    info = {"arm": arm, "seed": seed, "run_dir": rd, "seconds": round(time.monotonic() - t0, 1),
            "epochs_run": len(hist), "best_epoch": int(best.get("epoch", -1)),
            "val_pr_auc": summary.get("pr_auc"), "val_roc_auc": summary.get("roc_auc"),
            "threshold": summary.get("val_threshold"),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_rss_mb": _peak_rss_mb()}
    _atomic_json(os.path.join(rd, "campaign.json"), info)
    return info


def audit_dir(arm: str, seed: int, kind: str) -> str:
    return os.path.join(REPORT_ROOT, "audits", arm, f"seed-{seed}", kind)


def mapeval_dir(arm: str, seed: int) -> str:
    return os.path.join(REPORT_ROOT, "mapeval", arm, f"seed-{seed}")


def job_mapeval(checkpoint: str, out: str) -> dict:
    """Whole-recording Lance map evaluation of one checkpoint."""
    from .map_eval import (_sha256 as sha, build_geometry, build_scores, check_geometry_cache,
                           check_scores_cache, evaluate)
    geo = os.path.join(MAPEVAL_FRAMES, "geometry")
    ok, why = check_geometry_cache(geo, LANCE_RECORDING, LANCE_LABELS, 10, 0.05)
    if not ok:
        raise SystemExit(f"Lance geometry cache not ready ({why}); run --phase evaluate, "
                         "which builds it first")
    scores = os.path.join(MAPEVAL_FRAMES, "scores", sha(checkpoint)[:12])
    ok, why = check_scores_cache(scores, geo, checkpoint)
    if not ok:
        build_scores(geo, checkpoint, scores, device="cuda")
    result = evaluate(geo, scores, out, None)
    return {k: v for k, v in result["maps"]["control"].items() if not isinstance(v, dict)}


def job_geometry() -> dict:
    from .map_eval import build_geometry, check_geometry_cache
    geo = os.path.join(MAPEVAL_FRAMES, "geometry")
    ok, why = check_geometry_cache(geo, LANCE_RECORDING, LANCE_LABELS, 10, 0.05)
    if ok:
        return {"reused": True}
    return build_geometry(LANCE_RECORDING, LANCE_LABELS, geo, stride=10, window_s=0.05)


def job_audit(model_a: str, model_b: str, out: str, start: float | None = None,
              end: float | None = None, candidates_per_rock: int = 3,
              accum_seconds: float = 5.0) -> dict:
    from .visual_audit import run_visual_audit
    res = run_visual_audit(LANCE_RECORDING, LANCE_LABELS, model_a, model_b, out,
                           floor_band=(-0.10, 0.60), max_range=8.0, stride=10,
                           accum_seconds=accum_seconds, candidates_per_rock=candidates_per_rock,
                           cell_m=0.10, start_s=start, end_s=end, device="cuda", batch=256)
    return {"verdict": res["comparison"]["verdict"], "rocks": len(res["per_rock"])}


def _trained(arm: str, seed: int) -> str | None:
    p = os.path.join(run_dir(arm, seed), "best.pt")
    return p if os.path.exists(os.path.join(run_dir(arm, seed), "val_metrics.json")) else None


def phase_train(args) -> None:
    arms = args.arms or PRIORITY
    for seed in args.seeds:
        print(f"training {len(arms)} arms, seed {seed}")
        for arm in arms:
            _run_job("train", f"{arm}__seed-{seed}",
                     _self("--phase", "job-train", "--arm", arm, "--seed", str(seed)),
                     args.force)


def phase_evaluate(args) -> None:
    arms = args.arms or PRIORITY
    if _run_job("evaluate", "lance-geometry", _self("--phase", "job-geometry")):
        raise SystemExit("Lance geometry cache failed")
    jobs = [("evaluate", "reference__mapeval",
             _self("--phase", "job-mapeval", "--checkpoint", REFERENCE,
                   "--out", os.path.join(REPORT_ROOT, "mapeval", "reference")), args.force)]
    for seed in args.seeds:
        base = _trained(BASELINE, seed)
        for arm in arms:
            ck = _trained(arm, seed)
            if ck is None:
                print(f"  not trained yet: {arm} seed {seed}")
                continue
            tag = f"{arm}__seed-{seed}"
            jobs.append(("evaluate", f"{tag}__mapeval",
                         _self("--phase", "job-mapeval", "--checkpoint", ck,
                               "--out", mapeval_dir(arm, seed)), args.force))
            jobs.append(("evaluate", f"{tag}__vs-reference",
                         _self("--phase", "job-audit", "--model-a", REFERENCE,
                               "--model-b", ck, "--out",
                               audit_dir(arm, seed, "vs-reference")), args.force))
            if arm != BASELINE and base:
                jobs.append(("evaluate", f"{tag}__vs-baseline",
                             _self("--phase", "job-audit", "--model-a", base,
                                   "--model-b", ck, "--out",
                                   audit_dir(arm, seed, "vs-baseline")), args.force))
            if "+" in arm:
                # A follow-up is judged against the arm it modifies, same seed.
                parent = _trained(base_arm(arm), seed)
                if parent:
                    jobs.append(("evaluate", f"{tag}__vs-parent",
                                 _self("--phase", "job-audit", "--model-a", parent,
                                       "--model-b", ck, "--out",
                                       audit_dir(arm, seed, "vs-parent")), args.force))
            elif not arm.endswith("__h1") and base:
                # True recording startup: history is empty at 0 s and fills
                # over the first 30. Every interval of the first 46 s is kept,
                # two seconds long so none straddles the 4 s and 30 s stage
                # boundaries (a 0-5 s interval mixed startup with partial).
                jobs.append(("evaluate", f"{tag}__startup-2s",
                             _self("--phase", "job-audit", "--model-a", base,
                                   "--model-b", ck, "--out",
                                   audit_dir(arm, seed, "startup-2s"),
                                   "--start", "0", "--end", "46",
                                   "--accum-seconds", "2",
                                   "--candidates-per-rock", "20"), args.force))
    print(f"{len(jobs)} evaluation jobs, {args.workers} at a time")
    _pool(jobs, args.workers)


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def phase_smoke(args) -> dict:
    """Generation for every policy on one recording, two tiny fits, a replay."""
    from ..dataset.generate import run_generate

    job_sweeps(SMOKE_RUN)
    out = {"recording": SMOKE_RUN, "policies": {}}
    for arm in PRIORITY:
        d = os.path.join(DATASET_ROOT, "_smoke", arm)
        if os.path.isdir(d):
            shutil.rmtree(d)
        t0 = time.monotonic()
        e = run_generate(vb_recording(SMOKE_RUN), vb_labels(SMOKE_RUN), d,
                         resolved_config(arm), profile=PROFILE,
                         sweep_cache=sweep_dir(SMOKE_RUN))
        diag = e["diagnostics"]
        out["policies"][arm] = {
            "seconds": round(time.monotonic() - t0, 1), "samples": e["point_samples"],
            "labels": e["sample_labels"], "candidates": e["candidates_selected"],
            "unscorable": e["unscorable"], "radius_m": diag["radius_m"],
            "radius_at_min_frac": diag["radius_at_min_frac"],
            "radius_at_max_frac": diag["radius_at_max_frac"],
            "shortfall_below_k_frac": diag["shortfall_below_k_frac"],
            "ball_count": diag["ball_count"], "ball_voxels": diag["ball_voxels"],
            "ball_sweeps": diag["ball_sweeps"],
            "slot_fill": [s["fill_rate"] for s in diag["slots"]],
            "slot_age_p50": [None if s["actual_age_s"] is None else s["actual_age_s"]["p50"]
                             for s in diag["slots"]]}
        p = out["policies"][arm]
        print(f"  {arm:34s} {p['samples']:6d} samples  {p['seconds']:5.1f}s  "
              f"unscorable r/c {p['unscorable']['rock']}/{p['unscorable']['clear']}  "
              f"slot fill {[round(f, 2) for f in p['slot_fill']]}")
    base = out["policies"][BASELINE]["candidates"]
    for arm, p in out["policies"].items():
        if p["candidates"] != base:
            raise SystemExit(f"smoke: {arm} selected a different candidate set than the baseline")

    for arm in (BASELINE, LONGEST):
        ok, why = _arm_complete(arm)
        if not ok:
            if _run_job("prepare", arm, _self("--phase", "job-prepare", "--arm", arm)):
                raise SystemExit(f"smoke: could not prepare {arm}")
        rd = os.path.join(EXPERIMENT_ROOT, "_smoke", arm, f"seed-{SCREEN_SEED}")
        if os.path.isdir(rd):
            shutil.rmtree(rd)
        out.setdefault("fits", {})[arm] = job_train(arm, SCREEN_SEED, epochs=2, patience=2,
                                                    out=rd)
    ck = {a: os.path.join(EXPERIMENT_ROOT, "_smoke", a, f"seed-{SCREEN_SEED}", "best.pt")
          for a in (BASELINE, LONGEST)}
    # A short replay 40-60 s into Lance with the longest-history checkpoint,
    # which needs 30 s of warm-up from before the window.
    rep = os.path.join(REPORT_ROOT, "smoke", "lance-replay")
    job_audit(ck[BASELINE], ck[LONGEST], rep, start=40.0, end=60.0, candidates_per_rock=5)
    audit = _read_json(os.path.join(rep, "audit.json"))
    a, b = audit["checkpoints"]
    fa = max(a["scoring"].get("frames", 0), 1)
    fb = max(b["scoring"].get("frames", 0), 1)
    out["replay"] = {"out": rep,
                     "baseline_support_per_frame": a["scoring"]["support_points"] / fa,
                     "longest_support_per_frame": b["scoring"]["support_points"] / fb,
                     "contracts": [a["input_contract"], b["input_contract"]]}
    if not out["replay"]["longest_support_per_frame"] > 1.5 * out["replay"]["baseline_support_per_frame"]:
        raise SystemExit("smoke: the history checkpoint was not given its history support")
    _atomic_json(os.path.join(REPORT_ROOT, "smoke", "smoke.json"), out)
    print(f"smoke ok -> {os.path.join(REPORT_ROOT, 'smoke', 'smoke.json')}")
    return out


# --------------------------------------------------------------------------- #
# alignment diagnostics
# --------------------------------------------------------------------------- #
def _patch_thickness(xyz: np.ndarray, centers: np.ndarray, r: float = 0.15,
                     min_pts: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Per patch: robust residual spread off a fitted plane, and a double-surface
    score (gap between the two largest residual clusters)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    spread, gap = [], []
    for idx in tree.query_ball_point(centers, r):
        if len(idx) < min_pts:
            continue
        p = xyz[idx].astype(np.float64)
        a = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
        coef, *_ = np.linalg.lstsq(a, p[:, 2], rcond=None)
        res = p[:, 2] - a @ coef
        lo, hi = np.percentile(res, [10, 90])
        spread.append(hi - lo)
        s = np.sort(res)
        gap.append(float(np.max(np.diff(s[len(s) // 10: len(s) - len(s) // 10]))) if len(s) > 20 else 0.0)
    return np.asarray(spread), np.asarray(gap)


def phase_alignment(args) -> dict:
    """Does stacking sweeps thicken the floor or split it into two surfaces?

    Measured on clear floor patches (and rock patches) of every Volleyball
    recording and the first ten minutes of Lance, for each history policy,
    against that policy's own support cloud cropped exactly as generation
    crops it.
    """
    from ..dataset.history import HistoryPolicy, Sweep, SweepHistory, assemble_support
    from ..dataset.history_generate import crop_box
    from ..dataset.labeling import LABEL_CLEAR, LABEL_ROCK, label_rocks
    from ..dataset.sweep_cache import CachedSweeps
    from ..labels import load_labels

    rng = np.random.default_rng(0)
    g = sweep_config()["generator"]
    rows = []
    sources = [(r, sweep_dir(r), vb_labels(r)) for r in VB_RUNS]
    for name, sdir, lpath in sources:
        labels = load_labels(lpath)
        sweeps = list(CachedSweeps(sdir))
        anchors = list(range(0, len(sweeps), 4))
        pick = set(rng.choice(anchors, min(12, len(anchors)), replace=False).tolist())
        for hname, ages in HISTORIES.items():
            pol = HistoryPolicy(tuple(ages), HISTORY_TOLERANCE_S)
            buf = SweepHistory(pol.retention_s)
            acc = {"floor_spread": [], "floor_gap": [], "rock_spread": []}
            for k, scan in enumerate(sweeps):
                buf.push(Sweep.from_scan(scan))
                if k not in pick:
                    continue
                lo, hi = crop_box(scan.T_odom_base[:3, 3], g, labels.z_band)
                sup = assemble_support(buf.select(pol),
                                       lambda x, lo=lo, hi=hi: ((x >= lo) & (x <= hi)).all(1))
                cur = sup.xyz[sup.current]
                if len(cur) < 100:
                    continue
                lab = label_rocks(cur, labels.rocks, 0.05)
                floor_c = cur[lab == LABEL_CLEAR]
                rock_c = cur[lab == LABEL_ROCK]
                if len(floor_c):
                    fc = floor_c[rng.choice(len(floor_c), min(60, len(floor_c)), replace=False)]
                    s, gp = _patch_thickness(sup.xyz, fc)
                    acc["floor_spread"] += s.tolist()
                    acc["floor_gap"] += gp.tolist()
                if len(rock_c):
                    rc = rock_c[rng.choice(len(rock_c), min(30, len(rock_c)), replace=False)]
                    s, _ = _patch_thickness(sup.xyz, rc, r=0.08, min_pts=15)
                    acc["rock_spread"] += s.tolist()
            rows.append({"source": name, "history": hname,
                         "floor_spread_p50_cm": _p(acc["floor_spread"], 50),
                         "floor_spread_p90_cm": _p(acc["floor_spread"], 90),
                         "floor_double_surface_frac": (float(np.mean(np.asarray(acc["floor_gap"]) > 0.03))
                                                       if acc["floor_gap"] else None),
                         "rock_spread_p50_cm": _p(acc["rock_spread"], 50),
                         "patches": len(acc["floor_spread"])})
        print(f"  alignment {name}: " + ", ".join(
            f"{r['history']} {r['floor_spread_p50_cm']}" for r in rows if r["source"] == name))
    rows += _lance_alignment(rng)
    _atomic_json(os.path.join(REPORT_ROOT, "alignment.json"), rows)
    _write_csv(os.path.join(REPORT_ROOT, "alignment.csv"), rows)
    return {"rows": len(rows)}


def _p(v, q):
    return None if not len(v) else round(float(np.percentile(v, q)) * 100, 2)


def _lance_alignment(rng) -> list[dict]:
    from ..config import load_config
    from ..dataset.history import HistoryPolicy, Sweep, SweepHistory, assemble_support
    from ..dataset.labeling import LABEL_CLEAR, label_rocks
    from ..geometry.leveling import pin_level_to_labels
    from ..labels import load_labels
    from ..recording.pipeline import ScanStream, WindowedScanStream

    labels = load_labels(LANCE_LABELS)
    floor = float(labels.level["floor_z"])
    cfg = pin_level_to_labels(load_config(None), labels.level)
    pols = {h: HistoryPolicy(tuple(a), HISTORY_TOLERANCE_S) for h, a in HISTORIES.items()}
    buf = SweepHistory(max(p.retention_s for p in pols.values()))
    acc = {h: {"s": [], "g": []} for h in pols}
    stream = WindowedScanStream(ScanStream(LANCE_RECORDING, cfg, progress=False), 0.05)
    t0 = None
    for k, scan in enumerate(stream):
        buf.push(Sweep.from_scan(scan))
        t0 = t0 if t0 is not None else scan.time_s
        if scan.time_s - t0 > 600:
            break
        if k % 200 or scan.time_s - t0 < 35:
            continue
        base = scan.T_odom_base[:3, 3]
        for h, pol in pols.items():
            def crop(x, base=base):
                d = x[:, :2] - base[:2]
                return ((x[:, 2] >= floor - 0.10) & (x[:, 2] <= floor + 0.60)
                        & ((d * d).sum(1) <= 64.0))
            sup = assemble_support(buf.select(pol), crop)
            cur = sup.xyz[sup.current]
            lab = label_rocks(cur, labels.rocks, 0.05)
            fc = cur[(lab == LABEL_CLEAR) & (cur[:, 2] < floor + 0.08)]
            if len(fc) < 10:
                continue
            fc = fc[rng.choice(len(fc), min(60, len(fc)), replace=False)]
            s, gp = _patch_thickness(sup.xyz, fc)
            acc[h]["s"] += s.tolist()
            acc[h]["g"] += gp.tolist()
    return [{"source": "lance (first 10 min, mature history only)", "history": h,
             "floor_spread_p50_cm": _p(a["s"], 50), "floor_spread_p90_cm": _p(a["s"], 90),
             "floor_double_surface_frac": (float(np.mean(np.asarray(a["g"]) > 0.03))
                                           if a["g"] else None),
             "rock_spread_p50_cm": None, "patches": len(a["s"])} for h, a in acc.items()]


def _write_csv(path: str, rows: list[dict]) -> None:
    import csv
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                        for k, v in r.items()})


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def _stray_escape(arm: str, n: int = 4000) -> float | None:
    """Share of stray-augmented points that end up outside their own ball.

    The augmentation throws a point along its centre-relative direction by an
    exponential distance (median ~0.35 m at reach 1.0). With a 0.2 m ball most
    throws leave the ball entirely. Recorded, not corrected, in stage 1.
    """
    from .data import RunData
    try:
        run = RunData(cache_dir(arm), TRAIN_RUNS[0])
    except SystemExit:
        return None
    radius = run.extra("radius")
    if radius is None:
        return None
    rng = np.random.default_rng(0)
    idx = rng.choice(len(run), min(n, len(run)), replace=False)
    pts = run.points[idx, :, :3].astype(np.float64)
    cnt = np.minimum(run.counts[idx], pts.shape[1])
    # dz is relative to the ball minimum; recentre it on the candidate.
    qz = run.points[idx, 0, 4][:, None]
    xyz = pts.copy()
    xyz[..., 2] -= qz
    valid = np.arange(pts.shape[1])[None, :] < cnt[:, None]
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True).clip(1e-6)
    mag = -np.log(rng.random(xyz.shape[:2]).clip(1e-6)) * 0.5 * TRAIN_SETTINGS["aug_stray_reach"]
    sign = np.where(rng.random(xyz.shape[:2]) < 0.5, -1.0, 1.0)
    moved = xyz + (mag * sign)[..., None] * xyz / norm
    out = np.linalg.norm(moved, axis=-1) > radius[idx][:, None]
    return float(out[valid].mean())


def _stratify_scores(arm_score_dir: str, threshold: float) -> list[dict]:
    """Candidate-level rock recall and clear false-positive rate on Lance, by
    stratum: range from the robot, ball support, time since start and how far
    the robot moved across the history.

    The denominator is the **common opportunity set**: every candidate center
    of the current sweep, which depends only on the sweep and the voxel size
    and so is identical for every arm. It is rebuilt from the geometry cache
    and matched to the saved scores by voxel. A candidate the arm could not
    score counts as a miss in ``*_opportunity`` rates and is left out of the
    ``*_scored`` ones, so a small ball that rejects most of the rocks cannot
    look good by only reporting the ones it kept.
    """
    from ..dataset.history import candidate_centers
    from ..dataset.labeling import LABEL_CLEAR, LABEL_ROCK, label_rocks
    from ..labels import load_labels

    geo = os.path.join(MAPEVAL_FRAMES, "geometry")
    labels = load_labels(LANCE_LABELS)
    voxel = float(_read_json(os.path.join(arm_score_dir, "meta.json"))["generator"]
                  ["centers_voxel_m"])
    names = {"range": ["<2 m", "2-4 m", "4-6 m", ">6 m"],
             "support": ["<64 pts", "64-256", "256-1024", ">1024", "unscorable"],
             "stage": ["startup 0-4 s", "partial 4-30 s", "mature >30 s"],
             "pose_diversity": ["moved <5 cm", "5-25 cm", "25-100 cm", ">1 m"]}
    # [rock opportunities, rock scored, rock hot, clear opps, clear scored, clear hot]
    rows: dict = {}
    for name in sorted(f for f in os.listdir(arm_score_dir) if f.startswith("frame-")):
        with np.load(os.path.join(arm_score_dir, name)) as s, \
                np.load(os.path.join(geo, name)) as g:
            pos, prob = s["positions"], s["probabilities"]
            bc = s["ball_count"] if "ball_count" in s.files else None
            shift = float(np.nanmax(s["slot_shift"])) if "slot_shift" in s.files and \
                len(s["slot_shift"]) > 1 else 0.0
            base, t = g["base"], float(g["time_s"])
            cand = candidate_centers(g["xyz"].astype(np.float64), {"centers_voxel_m": voxel})
        if not len(cand):
            continue
        # A candidate is its voxel's centroid, so the voxel key identifies it
        # whatever precision each path computed the centroid in.
        where = {k: i for i, k in enumerate(map(tuple, np.floor(pos / voxel).astype(np.int64)))}
        idx = np.array([where.get(k, -1) for k in
                        map(tuple, np.floor(cand / voxel).astype(np.int64))])
        scored = idx >= 0
        hot = np.zeros(len(cand), bool)
        hot[scored] = prob[idx[scored]] >= threshold
        lab = label_rocks(cand, labels.rocks, 0.05)
        support = np.full(len(cand), 4)
        if bc is not None:
            support[scored] = np.digitize(bc[idx[scored]], [64, 256, 1024])
        elif scored.any():
            support[scored] = 0
        rng_m = np.linalg.norm(cand[:, :2] - base[:2], axis=1)
        strata = {
            "range": np.digitize(rng_m, [2.0, 4.0, 6.0]),
            "support": support,
            "stage": np.full(len(cand), 0 if t < 4 else (1 if t < 30 else 2)),
            # How far the sensor moved between the oldest used sweep and now:
            # time separation only buys a new viewpoint if the robot moved.
            "pose_diversity": np.full(len(cand), int(np.digitize(
                0.0 if not np.isfinite(shift) else shift, [0.05, 0.25, 1.0]))),
        }
        rock, clear = lab == LABEL_ROCK, lab == LABEL_CLEAR
        for kind, bins in strata.items():
            for b in np.unique(bins):
                r = rows.setdefault((kind, names[kind][int(b)]), [0] * 6)
                m = bins == b
                for j, cls in enumerate((rock, clear)):
                    r[3 * j] += int((m & cls).sum())
                    r[3 * j + 1] += int((m & cls & scored).sum())
                    r[3 * j + 2] += int((m & cls & hot).sum())

    def ratio(a, b):
        return a / b if b else None

    return [{"stratum": k, "bin": b,
             "rock_opportunities": r[0], "rock_scored": r[1],
             "rock_recall_opportunity": ratio(r[2], r[0]),
             "rock_recall_scored": ratio(r[2], r[1]),
             "clear_opportunities": r[3], "clear_scored": r[4],
             "clear_fp_rate_opportunity": ratio(r[5], r[3]),
             "clear_fp_rate_scored": ratio(r[5], r[4])}
            for (k, b), r in sorted(rows.items())]


def _map_row(summary: dict | None) -> dict:
    if not summary:
        return {}
    c = summary["maps"]["control"]
    budgets = summary.get("budgets", {}).get("control", {})
    row = {"map_macro_coverage_3d": c.get("macro_coverage"),
           "map_worst_rock_3d": c.get("worst_rock_coverage"),
           "map_macro_coverage_footprint": c.get("macro_coverage_footprint"),
           "map_worst_rock_footprint": c.get("worst_rock_coverage_footprint"),
           "map_false_cells_3d": c.get("false_cells_3d"),
           "map_false_cells_footprint": c.get("false_cells_footprint"),
           "map_rocks_never_covered": c.get("rocks_never_covered"),
           "map_false_cell_seconds": c.get("false_cell_seconds"),
           "map_median_first_covered_s": c.get("median_first_covered_s")}
    for b in ("2200", "1700", "1000", "500"):
        r = budgets.get(b)
        row[f"budget{b}_macro_coverage_3d"] = None if r is None else r["macro_coverage"]
        row[f"budget{b}_worst_rock_3d"] = None if r is None else r["worst_rock_coverage"]
        row[f"budget{b}_threshold_oracle"] = None if r is None else r["threshold"]
    s = summary.get("scores", {})
    lat = s.get("latency") or {}
    for k in ("preprocess_s", "network_s", "total_s"):
        v = lat.get(k) or {}
        row[f"latency_{k[:-2]}_p50_ms"] = v.get("p50_ms")
        row[f"latency_{k[:-2]}_p95_ms"] = v.get("p95_ms")
    row.update(lance_candidates=s.get("candidates"), lance_unscorable=s.get("unscorable"),
               lance_support_p50=(s.get("support_points") or {}).get("p50"),
               lance_slot_fill=s.get("history_slot_fill"),
               eval_peak_vram_mb=s.get("peak_vram_mb"), eval_peak_rss_mb=s.get("peak_rss_mb"))
    return row


def _audit_row(path: str, prefix: str) -> dict:
    a = _read_json(os.path.join(path, "audit.json"))
    if not a:
        return {}
    keys = ("model_a", "model_b")
    med = {k: [r[k]["median_coverage"] for r in a["per_rock"]] for k in keys}
    comp = a["comparison"]
    return {f"{prefix}_rocks": len(a["per_rock"]),
            f"{prefix}_a_macro_median": float(np.mean(med["model_a"])) if med["model_a"] else None,
            f"{prefix}_b_macro_median": float(np.mean(med["model_b"])) if med["model_b"] else None,
            f"{prefix}_a_false_cells_median": comp["false_cells"]["model_a"]["median_per_interval"],
            f"{prefix}_b_false_cells_median": comp["false_cells"]["model_b"]["median_per_interval"],
            f"{prefix}_b_leads": comp["coverage_leads"]["model_b"],
            f"{prefix}_a_leads": comp["coverage_leads"]["model_a"],
            f"{prefix}_verdict": comp["verdict"]}


def phase_summarize(args) -> dict:
    rows, per_rock, strata = [], [], []
    ref = _read_json(os.path.join(REPORT_ROOT, "mapeval", "reference", "summary.json"))
    if ref:
        rows.append({"arm": "reference (deploy/cls-stray, trained on all 11 with tail "
                            "validation; different provenance)", "seed": None,
                     **_map_row(ref)})
    seeds = sorted({int(p.split("seed-")[-1]) for p in
                    glob.glob(os.path.join(EXPERIMENT_ROOT, "*", "seed-*"))
                    if "_smoke" not in p})
    for seed in seeds:
        for arm in ALL_ARMS:
            rd = run_dir(arm, seed)
            info = _read_json(os.path.join(rd, "campaign.json"))
            if not info:
                continue
            meta = _read_json(os.path.join(cache_dir(arm), "meta.json")) or {}
            runs = meta.get("runs", {})
            diag = [r.get("diagnostics") or {} for r in runs.values()]
            nb, h = base_arm(arm).split("__")
            row = {"arm": arm, "neighborhood": nb, "history": h, "seed": seed,
                   "model": {**TRAIN_SETTINGS, **variant_settings(arm)}["model"],
                   "samples": sum(r["n"] for r in runs.values()),
                   "rock_samples": sum(r["rock"] for r in runs.values()),
                   **{k: info.get(k) for k in ("epochs_run", "best_epoch", "val_pr_auc",
                                               "val_roc_auc", "threshold", "seconds",
                                               "peak_vram_mb")}}
            man = _read_json(os.path.join(dataset_dir(arm), "manifest.json")) or {}
            entries = list((man.get("runs") or {}).values())
            sel = sum(e["candidates_selected"]["rock"] + e["candidates_selected"]["clear"]
                      for e in entries)
            uns = sum(e["unscorable"]["rock"] + e["unscorable"]["clear"] for e in entries)
            row["train_unscorable_frac"] = uns / sel if sel else None
            row["radius_p50"] = np.median([d["radius_m"]["p50"] for d in diag if d.get("radius_m")]) if diag else None
            row["radius_at_min_frac"] = _mean([d.get("radius_at_min_frac") for d in diag])
            row["radius_at_max_frac"] = _mean([d.get("radius_at_max_frac") for d in diag])
            row["shortfall_below_k_frac"] = _mean([d.get("shortfall_below_k_frac") for d in diag])
            row["ball_count_p50"] = _mean([(d.get("ball_count") or {}).get("p50") for d in diag])
            row["points_per_voxel_p50"] = _mean([(d.get("points_per_voxel") or {}).get("p50")
                                                 for d in diag])
            row["slot_fill"] = [round(_mean([d["slots"][i]["fill_rate"] for d in diag]), 3)
                                for i in range(len(HISTORIES[h]))] if diag else None
            row["stray_escape_frac"] = _stray_escape(arm)
            summary = _read_json(os.path.join(mapeval_dir(arm, seed), "summary.json"))
            row.update(_map_row(summary))
            row.update(_audit_row(audit_dir(arm, seed, "vs-reference"), "vs_ref"))
            row.update(_audit_row(audit_dir(arm, seed, "vs-baseline"), "vs_base"))
            row.update(_audit_row(audit_dir(arm, seed, "vs-parent"), "vs_parent"))
            st = _read_json(os.path.join(audit_dir(arm, seed, "startup-2s"), "audit.json"))
            if st:
                for s in st.get("history_stages", []):
                    if s["stage"].startswith("straddling"):
                        continue
                    key = s["stage"].split()[0]
                    row[f"startup_{key}_base"] = s.get("model_a_macro_median_coverage")
                    row[f"startup_{key}_arm"] = s.get("model_b_macro_median_coverage")
            rows.append(row)
            pr = os.path.join(mapeval_dir(arm, seed), "per-rock.csv")
            if os.path.exists(pr):
                import csv
                with open(pr) as f:
                    for r in csv.DictReader(f):
                        if r["map"] == "control":
                            per_rock.append({"arm": arm, "seed": seed, "rock_id": r["rock_id"],
                                             "coverage_3d": r["coverage"],
                                             "coverage_footprint": r["occupied_coverage"],
                                             "components": r["components"],
                                             "first_covered_s": r["first_covered_s"]})
            if summary and args.stratify:
                sd = os.path.join(MAPEVAL_FRAMES, "scores",
                                  summary["scores"]["checkpoint_sha256"][:12])
                for s in _stratify_scores(sd, float(summary["threshold"])):
                    strata.append({"arm": arm, "seed": seed, **s})
    _atomic_json(os.path.join(REPORT_ROOT, "comparison.json"), rows)
    _write_csv(os.path.join(REPORT_ROOT, "comparison.csv"), rows)
    _write_csv(os.path.join(REPORT_ROOT, "per-rock.csv"), per_rock)
    if strata:
        _write_csv(os.path.join(REPORT_ROOT, "strata.csv"), strata)
    _write_results_md(rows, per_rock)
    print(f"summarized {len(rows)} rows -> {os.path.join(REPORT_ROOT, 'comparison.csv')}")
    return {"rows": len(rows)}


def _mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def _fmt(v, p=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{p}f}"
    return str(v)


def _write_results_md(rows: list[dict], per_rock: list[dict]) -> None:
    lines = [f"# {CAMPAIGN}: machine-generated results\n\n",
             "Generated by `neighborhood_campaign --phase summarize`; the written "
             "interpretation is in summary.md. Lance is a repeatedly consulted "
             "development benchmark, not an untouched test set. Map columns are the "
             "whole-recording `mapeval` control map at each checkpoint's stored "
             "(Volleyball-validation) threshold; `3d` is 3D attribution, `fp` the "
             "occupied footprint. Budget columns read the finished map at an oracle "
             "threshold fitted on Lance and are diagnostics only.\n\n",
             "| arm | seed | val PR-AUC | epochs | map cov 3d | worst 3d | cov fp | "
             "false 3d | false fp | cov3d @1000 fp | cov3d @500 fp | worst @500 | "
             "p50 ms | p95 ms | unscorable |\n",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"]
    for r in rows:
        unsc = (None if not r.get("lance_candidates") else
                r["lance_unscorable"] / r["lance_candidates"])
        lines.append(
            f"| {r['arm']} | {_fmt(r.get('seed'))} | {_fmt(r.get('val_pr_auc'))} | "
            f"{_fmt(r.get('epochs_run'))} | {_fmt(r.get('map_macro_coverage_3d'))} | "
            f"{_fmt(r.get('map_worst_rock_3d'))} | {_fmt(r.get('map_macro_coverage_footprint'))} | "
            f"{_fmt(r.get('map_false_cells_3d'))} | {_fmt(r.get('map_false_cells_footprint'))} | "
            f"{_fmt(r.get('budget1000_macro_coverage_3d'))} | "
            f"{_fmt(r.get('budget500_macro_coverage_3d'))} | "
            f"{_fmt(r.get('budget500_worst_rock_3d'))} | "
            f"{_fmt(r.get('latency_total_p50_ms'), 0)} | {_fmt(r.get('latency_total_p95_ms'), 0)} | "
            f"{_fmt(unsc)} |\n")
    lines += _paired_lines(rows)
    with open(os.path.join(REPORT_ROOT, "results.md"), "w") as f:
        f.write("".join(lines))


#: Paired-difference columns: (label, comparison.csv key).
PAIRED = (("val PR-AUC", "val_pr_auc"), ("map cov 3d", "map_macro_coverage_3d"),
          ("worst 3d", "map_worst_rock_3d"), ("false 3d", "map_false_cells_3d"),
          ("cov3d @1000", "budget1000_macro_coverage_3d"),
          ("cov3d @500", "budget500_macro_coverage_3d"))


def _paired_lines(rows: list[dict]) -> list[str]:
    """Each arm minus the baseline trained with the same seed, for arms with
    more than one seed: the mean difference, and the smallest and largest of
    the per-seed differences. Seed-to-seed spread on this project has been as
    large as most recipe effects, so an unpaired mean is not a result.
    Follow-ups get a second table against the arm they modify."""
    out = _paired_table(rows, [a for a in ALL_ARMS if "+" not in a],
                        lambda a: BASELINE, "the baseline",
                        f"(arm - {BASELINE})")
    out += _paired_table(rows, FOLLOWUPS, base_arm, "the arm each follow-up modifies",
                         "(follow-up - its own arm)")
    return out


def _paired_table(rows: list[dict], candidates: list[str], ref, title: str,
                  what: str) -> list[str]:
    by = {(r["arm"], r.get("seed")): r for r in rows}
    arms = [a for a in candidates if sum(1 for (x, _s) in by if x == a) > 1]
    if not arms:
        return []
    out = [f"\n## Paired against {title}, same seed\n\n",
           f"Each cell is mean [min, max] over seeds of {what}.\n\n",
           "| arm | seeds | " + " | ".join(l for l, _ in PAIRED) + " |\n",
           "|---|---:|" + "---:|" * len(PAIRED) + "\n"]
    for arm in arms:
        r0 = ref(arm)
        seeds = sorted(s for (x, s) in by if x == arm and (r0, s) in by)
        cells = []
        for _label, key in PAIRED:
            d = [by[(arm, s)].get(key) - by[(r0, s)].get(key) for s in seeds
                 if by[(arm, s)].get(key) is not None
                 and by[(r0, s)].get(key) is not None]
            cells.append("-" if not d else
                         f"{np.mean(d):+.3f} [{min(d):+.3f}, {max(d):+.3f}]"
                         if key != "map_false_cells_3d" else
                         f"{np.mean(d):+.0f} [{min(d):+.0f}, {max(d):+.0f}]")
        out.append(f"| {arm} | {len(seeds)} | " + " | ".join(cells) + " |\n")
    return out


# --------------------------------------------------------------------------- #
# latency
# --------------------------------------------------------------------------- #
#: Settings benchmarked, seed 42, plus the deployed reference.
LATENCY_ARMS = ("fixed-r050__h1", "fixed-r075__h1", "fixed-r050__h5-8s",
                "fixed-r050__h5-30s", "fixed-r075__h5-8s",
                "adaptive-r020-r050-k256__h5-8s", "adaptive-r020-r075-k256__h3-4s",
                "fixed-r075__h5-8s+age")
#: Lance windows timed: 600 of them (5 minutes of driving) starting 100 s in,
#: so every 30 s history slot can already be filled.
LATENCY_FRAMES = (200, 800)
LATENCY_WARMUP = 20


def job_latency(checkpoint: str, out: str) -> dict:
    """Time one checkpoint's scoring pass on an otherwise idle GPU.

    The evaluation phase's latency columns were measured three jobs at a time
    and, for history checkpoints, with the recording being re-decoded inside
    the timer. Here the replay is untimed, and each window's cost is split
    into what the robot pays every pass: selecting and cropping history
    (``assemble``), cutting and sampling balls (``preprocess``) and the forward
    pass (``network``). No candidate cap, as in map evaluation; the candidate
    and support counts are recorded beside the times.
    """
    import torch

    from .map_eval import _history_supports
    from .policy_scoring import history_policy, score_classifier
    from .visual_audit import _load_checkpoint

    geo_dir = os.path.join(MAPEVAL_FRAMES, "geometry")
    geometry = _read_json(os.path.join(geo_dir, "settings.json"))
    frames = sorted(f for f in os.listdir(geo_dir) if f.startswith("frame-"))
    frames = frames[LATENCY_FRAMES[0]:LATENCY_FRAMES[1]]
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
    loaded = _load_checkpoint(checkpoint, dev)
    g = loaded["generator"]
    hist = history_policy(g)
    supports = _history_supports(geometry, geo_dir, frames, hist) if hist else None
    rows = []
    for i, name in enumerate(frames):
        with np.load(os.path.join(geo_dir, name)) as z:
            xyz = z["xyz"].astype(np.float64) if not g.get("preprocessing_version", 1) >= 2 \
                else z["xyz"]
            inten = z["intensity"].astype(np.float32)
            index = int(z["index"])
        support, assemble_s = (next(supports) if supports is not None else (None, 0.0))
        if support is not None:
            cur = support.current
            xyz, inten = support.xyz[cur], support.intensity[cur]
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        _pos, _prob, diag = score_classifier(xyz, inten, g, loaded["model"], dev, index,
                                             256, support=support)
        torch.cuda.synchronize(dev)
        total = time.perf_counter() - t0 + assemble_s
        if i >= LATENCY_WARMUP:
            rows.append({"assemble_ms": 1e3 * assemble_s,
                         "preprocess_ms": 1e3 * diag["preprocess_s"],
                         "network_ms": 1e3 * diag["network_s"], "total_ms": 1e3 * total,
                         "candidates": diag["candidates"],
                         "support_points": diag["support_points"]})

    def q(key, p):
        return round(float(np.percentile([r[key] for r in rows], p)), 1)

    result = {"checkpoint": checkpoint, "input_contract": loaded.get("input_contract"),
              "windows": len(rows), "frames": list(LATENCY_FRAMES),
              "device": torch.cuda.get_device_name(dev),
              **{f"{k}_p{p}": q(k, p) for k in ("assemble_ms", "preprocess_ms",
                                                "network_ms", "total_ms")
                 for p in (50, 95)},
              "candidates_p50": q("candidates", 50),
              "support_points_p50": q("support_points", 50),
              "peak_vram_mb": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1),
              "peak_rss_mb": _peak_rss_mb()}
    _atomic_json(out, result)
    return result


def phase_latency(args) -> None:
    """Benchmark scoring cost one checkpoint at a time; refuses to share the GPU."""
    # Desktop programs (the compositor, a browser, the editor) always hold a
    # few megabytes of the GPU; what disqualifies a timing run is real work.
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.splitlines()
    heavy = [a for a in apps if a.strip() and float(a.split(",")[1]) > 500]
    util = float(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True).stdout.split()[0])
    if heavy or util > 10:
        raise SystemExit(f"GPU busy ({util:.0f}% used; {heavy}); latency needs it to itself")
    targets = [("reference", REFERENCE)] + [
        (arm, os.path.join(run_dir(arm, SCREEN_SEED), "best.pt")) for arm in LATENCY_ARMS]
    rows = []
    for tag, ck in targets:
        out = os.path.join(REPORT_ROOT, "latency", f"{tag}.json")
        _run_job("latency", tag, _self("--phase", "job-latency", "--checkpoint", ck,
                                       "--out", out), args.force)
        r = _read_json(out)
        if r:
            rows.append({"setting": tag, **{k: v for k, v in r.items()
                                            if k not in ("checkpoint", "input_contract")}})
    _write_csv(os.path.join(REPORT_ROOT, "latency.csv"), rows)
    for r in rows:
        print(f"{r['setting']:34s} total p50 {r['total_ms_p50']:6.1f} ms  "
              f"p95 {r['total_ms_p95']:6.1f} ms  (assemble {r['assemble_ms_p50']:.1f}, "
              f"prep {r['preprocess_ms_p50']:.1f}, net {r['network_ms_p50']:.1f})")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
PHASES = ("plan", "preflight", "prepare", "smoke", "train", "evaluate", "alignment",
          "latency", "summarize", "status")
_JOBS = ("job-sweeps", "job-prepare", "job-train", "job-geometry", "job-mapeval",
         "job-audit", "job-latency")


def phase_status(args) -> None:
    for phase in ("sweeps", "prepare", "train", "evaluate", "latency"):
        files = sorted(glob.glob(os.path.join(REPORT_ROOT, "status", phase, "*.json")))
        states: dict[str, int] = {}
        for f in files:
            s = (_read_json(f) or {}).get("state", "?")
            states[s] = states.get(s, 0) + 1
        print(f"{phase:9s} {states}")
        for f in files:
            s = _read_json(f) or {}
            if s.get("state") in ("failed", "running"):
                print(f"   {s['state']:7s} {os.path.basename(f)[:-5]}  {s.get('log', '')}")


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--phase", required=True, choices=PHASES + _JOBS,
                   help="which stage to run: " + ", ".join(PHASES))
    p.add_argument("--arms", nargs="+", default=None, choices=ALL_ARMS, metavar="ARM",
                   help="restrict prepare/train/evaluate to these arms (default: all "
                        "24, baseline and the most informative pairs first)")
    p.add_argument("--seed", type=int, action="append", dest="seeds", default=None,
                   help=f"training/evaluation seed; repeat for several (default: "
                        f"{SCREEN_SEED}; confirmation seeds are "
                        f"{', '.join(map(str, CONFIRM_SEEDS))})")
    p.add_argument("--workers", type=int, default=6,
                   help="parallel CPU jobs for prepare and evaluate (default: 6). "
                        "Training always runs one fit at a time on the GPU.")
    p.add_argument("--force", action="store_true",
                   help="rerun jobs whose status says done")
    p.add_argument("--no-stratify", dest="stratify", action="store_false",
                   help="summarize without the per-candidate Lance strata (faster)")
    # Internal job arguments.
    p.add_argument("--arm", help=argparse.SUPPRESS)
    p.add_argument("--run", help=argparse.SUPPRESS)
    p.add_argument("--checkpoint", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    p.add_argument("--model-a", help=argparse.SUPPRESS)
    p.add_argument("--model-b", help=argparse.SUPPRESS)
    p.add_argument("--start", type=float, help=argparse.SUPPRESS)
    p.add_argument("--end", type=float, help=argparse.SUPPRESS)
    p.add_argument("--candidates-per-rock", type=int, default=3, help=argparse.SUPPRESS)
    p.add_argument("--accum-seconds", type=float, default=5.0, help=argparse.SUPPRESS)


def run(args) -> int:
    args.seeds = args.seeds or [SCREEN_SEED]
    phase = args.phase
    if phase.startswith("job-"):
        detail = {"job-sweeps": lambda: job_sweeps(args.run),
                  "job-prepare": lambda: job_prepare_arm(args.arm),
                  "job-train": lambda: job_train(args.arm, args.seeds[0]),
                  "job-geometry": job_geometry,
                  "job-mapeval": lambda: job_mapeval(args.checkpoint, args.out),
                  "job-latency": lambda: job_latency(args.checkpoint, args.out),
                  "job-audit": lambda: job_audit(args.model_a, args.model_b, args.out,
                                                 args.start, args.end,
                                                 args.candidates_per_rock,
                                                 args.accum_seconds)}[phase]()
        print("JOB RESULT " + json.dumps(detail, default=_jsonable))
        return 0
    {"plan": phase_plan, "preflight": phase_preflight, "prepare": phase_prepare,
     "smoke": phase_smoke, "train": phase_train, "evaluate": phase_evaluate,
     "alignment": phase_alignment, "latency": phase_latency,
     "summarize": phase_summarize,
     "status": phase_status}[phase](args)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="neighborhood_campaign", description=__doc__.split("\n")[0])
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
