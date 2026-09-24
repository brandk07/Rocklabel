"""One cross-validated arena map from the two spatial folds.

The fold-A model has never seen a label on fold B's ground and vice versa. So
each cell of the arena takes its probability from the model that did not train
on it, and the stitched map is graded on every rock exactly like any other
full-arena map. Each side can also carry an ensemble (several seeds).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import grade as G
from .compare import grade_prior
from .evaluate import load
from .gpu import features_from_grid
from .train import LANCE_GEOMETRY, fold_region, load_cloud


_LOADED: dict = {}


def _load(paths, device):
    key = tuple(paths)
    if key not in _LOADED:
        _LOADED[key] = load(paths, device)
    return _LOADED[key]


def stitched_prob(models_a, models_b, grid, device, tta: bool, labels):
    """Probability map where fold B's ground is read by the fold-A model(s)
    and fold A's by the fold-B model(s)."""
    ma, fsa, _ = _load(models_a, device)
    mb, fsb, _ = _load(models_b, device)
    pa, st = G.predict(ma, grid, device, fsa, tta)
    pb, _ = G.predict(mb, grid, device, fsb, tta)
    xs, ys = grid.cell_centers()
    gx, gy = np.meshgrid(xs, ys)
    xy = np.c_[gx.ravel(), gy.ravel()]
    on_a_ground = fold_region("A", labels, True)(xy).reshape(grid.h, grid.w)
    # Fold A's own ground is read by the model that trained on fold B.
    prob = np.where(on_a_ground, pb, pa)
    return prob, st


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--a", nargs="+", default=None, help="checkpoint(s) trained on fold A")
    ap.add_argument("--b", nargs="+", default=None, help="checkpoint(s) trained on fold B")
    for g in ("g1", "g2", "g3", "g4"):
        ap.add_argument(f"--{g}", nargs="+", default=None,
                        help=f"four-way split: checkpoint(s) that held out group {g}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", action="store_true",
                    help="also read the map rotated and mirrored, and average")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--timeline", type=float, default=0.0,
                    help="also grade the stitched map every this many seconds")


def stitched_prob_groups(models: dict, grid, device, tta: bool, labels):
    """Four-way version: ``models`` maps a group name to the checkpoint(s)
    that held that group out; each group's ground is read by its own model."""
    from .train import LANCE_GROUPS4
    xs, ys = grid.cell_centers()
    gx, gy = np.meshgrid(xs, ys)
    xy = np.c_[gx.ravel(), gy.ravel()]
    centres = {r.id: np.asarray(r.center[:2], float) for r in labels.rocks}
    names = sorted(LANCE_GROUPS4)
    # Each cell belongs to the group of its nearest rock.
    ids = np.array([r.id for r in labels.rocks])
    d = np.stack([np.linalg.norm(xy - centres[i], axis=1) for i in ids], 1)
    nearest = ids[d.argmin(1)]
    owner = np.array([next(n for n in names if rid in LANCE_GROUPS4[n]) for rid in nearest])
    prob = np.zeros(len(xy), np.float32)
    st = None
    for n in names:
        m, fs, _ = _load(models[n], device)
        p, st = G.predict(m, grid, device, fs, tta)
        take = owner == n
        prob[take] = p.ravel()[take]
    return prob.reshape(grid.h, grid.w), st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    return run(ap.parse_args(argv))


def run(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lance = load_cloud("LANCE")
    labels = lance.labels
    vis = G.visible_from_geometry(LANCE_GEOMETRY, labels)
    g = G.map_grid(lance)
    groups = {n: getattr(args, n) for n in ("g1", "g2", "g3", "g4")}
    four = all(groups.values())
    if not four and not (args.a and args.b):
        raise SystemExit("give --a and --b, or all four of --g1 .. --g4")

    def stitch(grid):
        if four:
            return stitched_prob_groups(groups, grid, dev, args.tta, labels)
        return stitched_prob(args.a, args.b, grid, dev, args.tta, labels)
    prob, st = stitch(g)
    cells = G.to_cells(prob, st, g)
    res = G.grade(cells, labels, vis, args.threshold)
    os.makedirs(args.out, exist_ok=True)
    G.save(res, args.out)
    np.savez_compressed(os.path.join(args.out, "map-control.npz"), positions=cells.positions,
                        probabilities=cells.probabilities, cell_ids=cells.cell_ids)
    x, _, _ = features_from_grid(g, dev, "base")
    for name, thr in [("threshold", args.threshold)] + [
            (f"at{b}", res["budgets"][b]["threshold"]) for b in ("500", "200", "100")
            if res["budgets"].get(b)]:
        G.plot(os.path.join(args.out, f"map-{name}.png"), cells, labels, thr,
               f"cross-validated arena map — {name} ({thr:.3f})",
               x[0, 4].cpu().numpy() / 10, g)
    print(G.summarize(res), flush=True)
    prior = grade_prior(labels, vis)
    summary = {"a": args.a, "b": args.b, "groups": groups if four else None, "tta": args.tta,
               "result": {k: v for k, v in res.items() if k != "frontier"},
               "prior": {k: {"budgets": v["budgets"], "macro_coverage": v["macro_coverage"],
                             "false_cells_footprint": v["false_cells_footprint"]}
                         for k, v in prior.items()}}
    if args.timeline:
        from . import timeline as TL
        rows = TL.visible_until(LANCE_GEOMETRY, labels)
        summary["timeline"] = TL.run(
            None, dev, lance, args.threshold, args.timeline, rows,
            predict=stitch)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)


if __name__ == "__main__":
    main()
