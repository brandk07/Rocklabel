"""The control for the arena-fold map models: the per-ball classifier, given
the same arena labels.

The map model trained on volleyball plus one arena rock group beats every
earlier model on the other group's ground. Two things changed at once, though:
the model reads the map, and it has seen arena labels. This trains the
single-sweep 0.5 m PointNet baseline of the neighbourhood-history campaign
(same settings, same nine volleyball recordings, same validation runs) with
the arena's labelled candidate balls from one rock group's ground added as a
tenth training run, grades it with ``mapeval`` on the whole recording, and
re-grades that map on the other group's ground only - exactly as the map model
is graded.

The arena samples come from the labelled competition cache (109 frames spread
over the 35 minutes, every candidate kept). Clear ones are thinned to
``negative_keep`` so the arena run is not all floor.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from ...labels import load_labels
from . import grade as G
from .compare import grade_prior, load_map, restrict
from .train import LANCE_GEOMETRY, LANCE_LABELS, fold_region

LANCE_CACHE = ("training/caches/lance-arena/"
               "lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits")
BASE_ARM = "fixed-r050__h1"


def build_cache(fold: str, out: str, negative_keep: float, seed: int = 0) -> str:
    """VB runs of the baseline arm (linked) plus one arena run on ``fold``'s ground."""
    from .. import neighborhood_campaign as NC
    src = NC.cache_dir(BASE_ARM)
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(src, "meta.json")))
    for run in NC.VB_RUNS:
        link = os.path.join(out, run)
        if not os.path.exists(link):
            os.symlink(os.path.abspath(os.path.join(src, run)), link)
    labels = load_labels(LANCE_LABELS)
    region = fold_region(fold, labels, True, buffer_m=0.25)
    arr = {k: np.load(os.path.join(LANCE_CACHE, f"{k}.npy"))
           for k in ("points", "labels", "counts", "centers", "frame")}
    keep = region(arr["centers"][:, :2].astype(float))
    rng = np.random.default_rng(seed)
    clear = arr["labels"] == 0
    keep &= ~clear | (rng.random(len(clear)) < negative_keep)
    # Samples in a rock's ignore shell carry label -1 in some caches; drop them.
    keep &= arr["labels"] >= 0
    run = f"LanceFold{fold}"
    d = os.path.join(out, run)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for k, v in arr.items():
        np.save(os.path.join(d, f"{k}.npy"), v[keep])
    n, rock = int(keep.sum()), int((arr["labels"][keep] == 1).sum())
    meta["runs"][run] = {"dataset_dir": LANCE_CACHE, "n": n, "rock": rock,
                         "clear": n - rock, "frames": int(len(np.unique(arr["frame"][keep]))),
                         "note": f"arena fold {fold} ground, clear kept at {negative_keep}"}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{run}: {n} samples, {rock} rock", flush=True)
    return run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", choices=["A", "B"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative-keep", type=float, default=0.25)
    ap.add_argument("--root", default="training/experiments/mapnet-v1/ball-control")
    args = ap.parse_args(argv)

    from .. import neighborhood_campaign as NC
    from ..engine import train_fold
    name = f"fold{args.fold}-s{args.seed}"
    cache = os.path.join("training/caches/mapnet-v1/ball-control", f"fold{args.fold}")
    run = build_cache(args.fold, cache, args.negative_keep)
    cfg = NC.train_config(BASE_ARM, args.seed, cache=cache)
    cfg["train_runs"] = list(cfg["train_runs"]) + [run]
    rd = os.path.join(args.root, name)
    train_fold(cfg, rd)
    ck = os.path.join(rd, "best.pt")
    out = os.path.join(args.root, name, "mapeval")
    NC.job_mapeval(ck, out)
    labels = load_labels(LANCE_LABELS)
    vis = G.visible_from_geometry(LANCE_GEOMETRY, labels)
    held = fold_region(args.fold, labels, False)
    cells, v = restrict(load_map(out), vis, labels, held)
    res = G.grade(cells, labels, v, 0.5)
    G.save(res, os.path.join(args.root, name, "graded-held-out"))
    print("held-out ground:", G.summarize(res), flush=True)
    prior = grade_prior(labels, vis, held)
    with open(os.path.join(args.root, name, "summary.json"), "w") as f:
        json.dump({"fold": args.fold, "checkpoint": ck,
                   "result": {k: v for k, v in res.items() if k != "frontier"},
                   "prior": {k: {"budgets": p["budgets"]} for k, p in prior.items()}},
                  f, indent=1, default=float)


if __name__ == "__main__":
    main()
