"""Grade finished map-model checkpoints on the arena, as ``mapeval`` would.

For each checkpoint: the whole-recording map (graded on every rock, or on one
fold's ground if the checkpoint was trained on the other fold), optionally the
map as it stood every N seconds, and a picture. Averaging several checkpoints'
logits (an ensemble) is one flag.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import grade as G
from . import timeline as TL
from .compare import grade_prior, restrict
from .grid import channels
from .model import build
from .train import LANCE_GEOMETRY, fold_region, load_cloud


class Ensemble(torch.nn.Module):
    """Mean logit of several checkpoints sharing one feature set."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        return torch.stack([m(x) for m in self.models]).mean(0)


def load(paths, device):
    models, fs, cfgs = [], None, []
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        f = ck.get("feature_set", "base")
        if fs is not None and f != fs:
            raise SystemExit("an ensemble needs one feature set")
        fs = f
        m = build(len(channels(f)), ck["arch"]).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
        cfgs.append(ck.get("config", {}))
    model = models[0] if len(models) == 1 else Ensemble(models)
    return model, fs, cfgs


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("checkpoints", nargs="+",
                    help="one checkpoint, or several to average as an ensemble")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", action="store_true",
                    help="also read the map rotated and mirrored, and average")
    ap.add_argument("--timeline", type=float, default=0.0,
                    help="also grade the map every this many seconds")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--fold", default=None,
                    help="grade only the ground of the fold NOT named here "
                         "(default: the checkpoint's own --lance-fold, if any)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    return run(ap.parse_args(argv))


def run(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fs, cfgs = load(args.checkpoints, dev)
    fold = args.fold or cfgs[0].get("lance_fold")
    holdout = cfgs[0].get("lance_holdout")
    lance = load_cloud("LANCE")
    labels = lance.labels
    vis = G.visible_from_geometry(LANCE_GEOMETRY, labels)
    region = (fold_region(holdout, labels, True) if holdout and not args.fold
              else fold_region(fold, labels, False) if fold else None)
    os.makedirs(args.out, exist_ok=True)

    g = G.map_grid(lance)
    prob, st = G.predict(model, g, dev, fs, args.tta)
    cells = G.to_cells(prob, st, g)
    cells_r, vis_r = restrict(cells, vis, labels, region)
    res = G.grade(cells_r, labels, vis_r, args.threshold)
    G.save(res, args.out)
    np.savez_compressed(os.path.join(args.out, "map-control.npz"), positions=cells.positions,
                        probabilities=cells.probabilities, cell_ids=cells.cell_ids)
    np.savez_compressed(os.path.join(args.out, "prob.npz"), prob=prob.astype(np.float16),
                        x0=g.x0, y0=g.y0, cell=g.cell)
    from .gpu import features_from_grid
    x, _, _ = features_from_grid(g, dev, fs)
    for name, thr in [("threshold", args.threshold)] + [
            (f"at{b}", res["budgets"][b]["threshold"]) for b in ("500", "200")
            if res["budgets"].get(b)]:
        G.plot(os.path.join(args.out, f"map-{name}.png"), cells_r, labels, thr,
               f"{os.path.basename(args.out)} — {name} ({thr:.3f})",
               x[0, 4].cpu().numpy() / 10, g)
    print(G.summarize(res), flush=True)
    prior = grade_prior(labels, vis, region)
    summary = {"checkpoints": [os.path.abspath(p) for p in args.checkpoints],
               "feature_set": fs, "tta": args.tta, "graded_fold_region": fold,
               "result": {k: v for k, v in res.items() if k != "frontier"},
               "prior": {k: {"budgets": v["budgets"],
                             "macro_coverage": v["macro_coverage"],
                             "false_cells_footprint": v["false_cells_footprint"]}
                         for k, v in prior.items()}}
    if args.timeline:
        rows = TL.visible_until(LANCE_GEOMETRY, labels)
        summary["timeline"] = TL.run(model, dev, lance, args.threshold, args.timeline,
                                     rows, feature_set=fs, tta=args.tta,
                                     region=region)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)


if __name__ == "__main__":
    main()
