"""Reviewable, time-bounded classifier campaign. Default: print commands only.

Run with ``python -m rocklabel.train.lance_campaign``. Only --execute starts
work. Lance is evaluation-only; no model is automatically promoted or exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from .data import run_dir_name

ROOT = Path(__file__).resolve().parents[2]
RECORDING = "recordings/archive/misc/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.mcap"
LABELS = "labels/archive/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json"
REFERENCE = "training/experiments/deploy/cls-stray/trainall/best.pt"
FEATURES = ["dx", "dy", "dz"]
ARMS = [("pointnet-stray", "pointnet", 0.0),
        ("pointnet-matched", "pointnet", 0.20),
        ("stats-stray", "pointnet_stats", 0.0),
        ("stats-matched", "pointnet_stats", 0.20)]


def plan(run_root: Path) -> list[dict]:
    tasks = []
    for seed in (42, 43, 44):
        for fold in ("VolleyBallTest3.reslam", "VolleyBallTest4.reslam", "VolleyBallTest6.reslam"):
            for arm, model, phantom in ARMS:
                parent = run_root / arm / f"seed-{seed}"
                directory = parent / run_dir_name(model, f"loro_{fold}", FEATURES)
                command = [sys.executable, "-m", "rocklabel.train.cli", "train",
                           "--model", model, "--features", *FEATURES,
                           "--cache-dir", "training/caches/full-sweep",
                           "--runs-root", str(parent), "--test-run", fold,
                           "--epochs", "60", "--patience", "15", "--batch", "256",
                           "--seed", str(seed), "--device", "cuda",
                           "--aug-stray-frac", "0.05", "--aug-thin-min", "0.5",
                           "--aug-phantom-mode", "matched", "--aug-phantom-frac", str(phantom)]
                tasks.append({"arm": arm, "seed": seed, "fold": fold,
                              "directory": str(directory), "command": command})
    return tasks


def audit_command(checkpoint: str, output: str) -> list[str]:
    return [sys.executable, "-m", "rocklabel.train.cli", "visual-audit", RECORDING,
            "--labels", LABELS, "--model-a", REFERENCE, "--model-b", checkpoint,
            "--out", output, "--floor-band", "-0.10", "0.60", "--max-range", "8",
            "--stride", "10", "--accum-seconds", "5", "--candidates-per-rock", "3",
            "--device", "cuda", "--batch", "256"]


def run_child(command, log: Path, deadline: float, source: Path | None = None) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return "budget_exhausted"
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    if source is not None:
        env["PYTHONPATH"] = str(source)
        # Safe-path mode prevents cwd's mutable source overriding this snapshot.
        command = [command[0], "-P", *command[1:]]
    with log.open("w") as output:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=remaining)
        except subprocess.TimeoutExpired:
            return "budget_exhausted"
    return "complete" if result.returncode == 0 else f"failed:{result.returncode}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="start the printed campaign")
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--train-hours", type=float, default=8.0,
                        help="maximum training allocation; remaining time is for audits")
    parser.add_argument("--run-root", type=Path,
                        default=ROOT / "training/experiments/lance-campaign-v1")
    args = parser.parse_args(argv)
    if not 0 < args.train_hours < args.hours:
        parser.error("require 0 < train-hours < hours")
    root = args.run_root.expanduser().resolve()
    tasks = plan(root)
    if not args.execute:
        print(f"DRY RUN: {len(tasks)} fits; at most {args.train_hours:g} training hours, "
              f"{args.hours:g} total hours. No processes started or files written.")
        for task in tasks:
            print(shlex.join(task["command"]))
        print("Each completed fit then receives a full-recording, sampled Lance audit:")
        print(shlex.join(audit_command("CHECKPOINT/best.pt", "REPORT_DIRECTORY")))
        return 0

    # Refuse to silently fall back to CPU or compete with the existing run.
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; no training started")
    active = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True)
    if active.stdout.strip():
        parser.error("GPU compute processes are active; finish/stop the other run before launching")
    meta = json.loads((ROOT / "training/caches/full-sweep/meta.json").read_text())
    if any(not name.startswith("VolleyBallTest") for name in meta["runs"]):
        parser.error("campaign requires a volleyball-only cache")
    for path in (RECORDING, LABELS, REFERENCE):
        if not (ROOT / path).is_file():
            parser.error(f"missing {path}")
    if root.exists():
        parser.error("run-root already exists; choose a new directory to preserve previous results")
    root.mkdir(parents=True)
    source = root / "source"
    shutil.copytree(ROOT / "rocklabel", source / "rocklabel",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source_hash = hashlib.sha256()
    for path in sorted((source / "rocklabel").rglob("*.py")):
        source_hash.update(str(path.relative_to(source)).encode())
        source_hash.update(path.read_bytes())
    manifest = {"source_sha256": source_hash.hexdigest(), "cache_hash": meta["config_hash"],
                "hours": args.hours, "train_hours": args.train_hours,
                "lance_training": False, "tasks": tasks}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.monotonic()
    results = []
    def save_results():
        (root / "campaign-status.json").write_text(json.dumps(results, indent=2) + "\n")
    for i, task in enumerate(tasks):
        print(f"Train {i + 1}/{len(tasks)}: {task['arm']} {task['fold']} seed {task['seed']}", flush=True)
        status = run_child(task["command"], root / f"train-{i:02d}.log",
                           start + args.train_hours * 3600, source)
        results.append({"task": i, "training": status})
        save_results()
        if status == "budget_exhausted":
            break
        if status != "complete":
            print(f"Stopping after {status}; inspect {root / f'train-{i:02d}.log'}", flush=True)
            return 1
    for row in results:
        if row["training"] != "complete":
            continue
        i = row["task"]
        checkpoint = str(Path(tasks[i]["directory"]) / "best.pt")
        print(f"Audit task {i}: {checkpoint}", flush=True)
        row["audit"] = run_child(audit_command(checkpoint, str(root / f"audit-{i:02d}")),
                                  root / f"audit-{i:02d}.log", start + args.hours * 3600, source)
        save_results()
        if row["audit"] == "budget_exhausted":
            break
        if row["audit"] != "complete":
            return 1
    print(f"Results: {root}. Inspect every per-rock table and selected case before choosing a model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
