"""Score every available best checkpoint on the fixed labelled Lance cache.

This is a sampled, native-unit ranking benchmark, not an accumulated-map audit.
Classifier AP is per candidate center; segmenter AP is per scorable point.
Results are kept separate from each checkpoint's volleyball test_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import data, engine, metrics
from .models import build_model_from_config, model_task


def input_differences(generator: dict, benchmark: dict, task: str) -> dict:
    """Native input settings changed by the fixed benchmark cache."""
    keys = ["frame_window_s", "crop_forward_m", "crop_backward_m", "crop_left_m",
            "crop_right_m", "crop_up_m", "crop_down_m"]
    if task == "classify":
        keys += ["centers_voxel_m", "neighborhood_radius_m", "min_neighbors",
                 "neighborhood_points"]
    else:
        keys += ["segmentation_points", "segmentation_min_points", "bev_cell_m"]
    return {key: {"checkpoint": generator.get(key), "benchmark": benchmark.get(key)}
            for key in keys if generator.get(key) != benchmark.get(key)}


#: Defaults, named here so the CLI layer and the dashboard quote the same ones
#: rather than two copies that drift.
DEFAULT_EXPERIMENTS = "training/experiments"
DEFAULT_CACHE = os.path.join("training", "caches", "lance-arena")
DEFAULT_OUT = os.path.join("training", "reports", "lance-checkpoints", "results.json")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The benchmark's flags, shared by ``rocklabel-train lancebench`` and by
    running this module directly."""
    parser.add_argument("--experiments", default=DEFAULT_EXPERIMENTS,
                        help="folder to search for checkpoints, recursively "
                             f"(default: {DEFAULT_EXPERIMENTS})")
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help="the labelled arena cache to score against "
                             f"(default: {DEFAULT_CACHE}); it must hold exactly "
                             "one run")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"where the scores are written (default: {DEFAULT_OUT}). "
                             "Existing entries are kept and only checkpoints "
                             "that changed, or that failed last time, are rescored")
    parser.add_argument("--pattern", default="best.pt",
                        help="which checkpoint files to score, as a glob "
                             "matched recursively under --experiments "
                             "(default: best.pt). Use 'epochs/epoch-*.pt' to "
                             "score every saved epoch of a run trained with "
                             "--save-every-epoch, which is how the question "
                             "'did the validation score keep the right epoch' "
                             "gets answered")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto)")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N checkpoints - a smoke "
                             "check, never a result")


def run(args) -> None:
    if not getattr(args, "device", None):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = data.load_cache_meta(args.cache)
    if len(meta["runs"]) != 1:
        raise ValueError("Lance benchmark expects exactly one labelled run")
    run_id = next(iter(meta["runs"]))
    run_meta = meta["runs"][run_id]
    device = torch.device(args.device)
    torch.set_num_threads(4)
    root = Path(args.experiments)
    paths = sorted(root.rglob(getattr(args, "pattern", None) or "best.pt"))
    if args.limit:
        paths = paths[:args.limit]
    splits = {
        task: engine.Split([data.RunData(args.cache, run_id, task=task)])
        for task in ("classify", "segment")
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prior = json.loads(out.read_text()) if out.exists() else {}
    rows = {r["checkpoint"]: r for r in prior.get("results", [])}
    cache_identity = {"config_hash": meta["config_hash"], "run_id": run_id,
                      "samples": run_meta["n"], "frames": run_meta["frames"],
                      "seg_frames": run_meta["seg_frames"]}
    if prior and prior.get("cache") != cache_identity:
        raise ValueError("existing results belong to a different Lance cache")

    def save() -> None:
        payload = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                   "cache": cache_identity,
                   "definition": "Step-integrated average precision on the fixed labelled Lance cache. "
                                 "Classifier unit: candidate center. Segmenter unit: scorable point. "
                                 "Uses each checkpoint's stored validation threshold for F1/precision/recall "
                                 "(0.5 fallback when absent); "
                                 "PR-AUC is threshold-free. The 109 classifier frames and 106 segmenter "
                                 "frames are sparse samples, "
                                 "not a full-recording deployment audit.",
                   "results": [rows[k] for k in sorted(rows)]}
        temp = out.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temp, out)

    for index, path in enumerate(paths, 1):
        key = str(path)
        stat = path.stat()
        old = rows.get(key)
        # An unchanged checkpoint is skipped - unless last time it failed. A
        # failure is usually about the environment (a cache missing a channel,
        # a driver, a full disk), not about the checkpoint, so a stored error
        # must not make a model permanently unscoreable: it is retried, and
        # succeeds the moment whatever broke has been fixed.
        unchanged = (old and old.get("checkpoint_size") == stat.st_size
                     and old.get("checkpoint_mtime_ns") == stat.st_mtime_ns)
        if unchanged and "error" not in old:
            if ("input_differences" not in old
                    or "lance_used_in_selection" not in old):
                if "input_differences" not in old:
                    ck = torch.load(path, map_location="cpu", weights_only=False)
                    old["input_differences"] = input_differences(
                        ck.get("generator", {}), meta["generator"], old["task"])
                old["training_uses_lance"] = (old.get("training_uses_lance", False)
                                              or "lance-target-negatives-v1/adapt/" in key)
                old["lance_used_in_selection"] = "lance-target-negatives-v1/" in key
                save()
            continue
        print(f"[{index}/{len(paths)}] {key}", flush=True)
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ck["config"]
            task = model_task(cfg["model"])
            model = build_model_from_config(cfg)
            model.load_state_dict(ck["model"])
            model.to(device).eval()
            split = splits[task]
            batch = 4 if task == "segment" else (32 if cfg["model"] == "pointnet2" else 256)
            probs = engine.predict(model, split, device, batch=batch)
            labels = split.labels.numpy().astype(np.int8)
            if task == "segment":
                labels, probs = engine.seg_flatten(labels, probs, split.counts.numpy())
            threshold = float(ck.get("threshold", 0.5))
            score = metrics.summarize(labels, probs, threshold)
            del model, probs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rows[key] = {"checkpoint": key, "checkpoint_size": stat.st_size,
                         "checkpoint_mtime_ns": stat.st_mtime_ns,
                         "model": cfg["model"], "task": task,
                         "training_uses_lance": (
                             any("lance" in str(x).lower() for x in cfg.get("train_runs", []))
                             or "lance-target-negatives-v1/adapt/" in key),
                         "lance_used_in_selection": "lance-target-negatives-v1/" in key,
                         "source_cache": cfg.get("cache_dir"),
                         "native_segmentation_points": ck.get("generator", {}).get("segmentation_points"),
                         "input_differences": input_differences(
                             ck.get("generator", {}), meta["generator"], task),
                         "threshold_source": "checkpoint" if "threshold" in ck else "fallback_0.5",
                         **score}
            print(f"  PR-AUC {score['pr_auc']:.4f}  F1@{threshold:.3f} {score['f1']:.4f}", flush=True)
        except Exception as exc:
            rows[key] = {"checkpoint": key, "checkpoint_size": stat.st_size,
                         "checkpoint_mtime_ns": stat.st_mtime_ns,
                         "error": f"{type(exc).__name__}: {exc}"}
            print(f"  ERROR {rows[key]['error']}", flush=True)
        save()
    print(f"Saved {len(rows)} results to {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
