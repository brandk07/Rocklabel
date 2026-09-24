"""Comparison tables for the map-model report.

Every row is graded by :func:`grade.grade` on identical terms: the same rocks,
the same visible-cell denominators, the same false-area definition. Rows for
earlier models are their saved ``mapeval`` maps re-graded; rows for map models
are computed here from checkpoints. Both coverage measures are printed, never
one standing in for the other.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import grade as G
from .compare import PRIOR, STORED_THRESHOLD, load_map, restrict
from .evaluate import load
from .stitch import stitched_prob
from .train import LANCE_GEOMETRY, fold_region, load_cloud

BUDGETS = ("1000", "500", "200", "100", "50")


def _row(name: str, res: dict) -> dict:
    b = res["budgets"]
    out = {"model": name, "claimed@0.5": res["claimed_cells"],
           "false@0.5": res["false_cells_footprint"], "cov@0.5": res["macro_coverage"],
           "foot@0.5": float(np.mean([r["occupied"] for r in res["per_rock"]]))}
    for k in BUDGETS:
        r = b.get(k)
        out[f"attr@{k}"] = None if r is None else r["macro_coverage"]
        out[f"foot@{k}"] = None if r is None else r["macro_coverage_footprint"]
        out[f"worst@{k}"] = None if r is None else r["worst_rock_coverage"]
    out["per_rock"] = {r["rock_id"]: round(r["coverage"], 3) for r in res["per_rock"]}
    return out


def fuse_maps(a, b):
    """Per 10 cm cell, the mean of two maps' probabilities (a cell missing from
    one map counts as 0 there); the representative comes from ``a`` when it has
    the cell. Both maps are graded on the same cells afterwards."""
    from ..visual_audit import CellMap
    ka = {tuple(k): i for i, k in enumerate(a.cell_ids)}
    kb = {tuple(k): i for i, k in enumerate(b.cell_ids)}
    pos, prob, ids = [], [], []
    for k in sorted(set(ka) | set(kb)):
        pa = a.probabilities[ka[k]] if k in ka else 0.0
        pb = b.probabilities[kb[k]] if k in kb else 0.0
        pos.append(a.positions[ka[k]] if k in ka else b.positions[kb[k]])
        prob.append(0.5 * (pa + pb))
        ids.append(k)
    return CellMap(np.asarray(pos), np.asarray(prob), np.asarray(ids, np.int64))


def _stored_threshold(folder: str) -> float:
    """The threshold ``mapeval`` ran the saved map at; 0.5 if it did not say."""
    for name in ("summary.json", "grade.json"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            t = json.load(open(p)).get("threshold")
            if t is not None:
                return float(t)
    return 0.5


_SECTIONS = {
    "full": ("Whole arena, models graded on ground they may have trained on",
             "Earlier models and the volleyball-only map model saw no arena labels, so "
             "the whole arena is held out for them."),
    "stitched": ("Whole arena, cross-validated",
                 "Every cell graded by a model that never saw a label on that ground "
                 "(two-way rock groups A and B)."),
    "four_way": ("Whole arena, cross-validated four ways",
                 "Each model trained on eight or nine arena rocks, graded on the rest."),
    "ground_A": ("Group A's ground only (rocks 5, 8, 9, 10, 11, 12: the rough east side)",
                 "Graded on the cells nearest those rocks."),
    "ground_B": ("Group B's ground only (rocks 1, 3, 4, 6, 7, 13)",
                 "Graded on the cells nearest those rocks."),
}


def write_markdown(rows: dict, path: str) -> None:
    """``results.md``: every row, both coverage measures, every budget."""
    def f(v):
        return "-" if v is None else f"{v:.3f}"
    lines = ["# mapnet-v1: machine-generated results", "",
             "Written by `python -m rocklabel.train.mapnet.tables`; the interpretation "
             "is in summary.md. Lance is development data. `attr` is 3D attribution "
             "(the claimed cell lies on the rock), `foot` occupied footprint (the cell "
             "is claimed at all), each at N wrongly-claimed footprint cells, read off "
             "the threshold frontier. The last three columns are at the model's own "
             "threshold (0.5 for map models and the per-ball control, the stored one for "
             "the earlier models).", ""]
    for key in ("stitched", "four_way", "full", "ground_A", "ground_B"):
        if not rows.get(key):
            continue
        title, blurb = _SECTIONS[key]
        lines += [f"## {title}", "", blurb, "",
                  "| model | " + " | ".join(f"attr @{b}" for b in BUDGETS) + " | "
                  + " | ".join(f"foot @{b}" for b in BUDGETS)
                  + " | coverage | footprint | false cells |",
                  "|---|" + "---:|" * (2 * len(BUDGETS) + 3)]
        for r in rows[key]:
            lines.append(f"| {r['model']} | " + " | ".join(f(r[f'attr@{b}']) for b in BUDGETS)
                         + " | " + " | ".join(f(r[f'foot@{b}']) for b in BUDGETS)
                         + f" | {f(r['cov@0.5'])} | {f(r.get('foot@0.5'))} | {r['false@0.5']} |")
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold-a", nargs="*", default=[], action="append",
                    help="checkpoints trained on fold A (repeat the flag per model row)")
    ap.add_argument("--fold-b", nargs="*", default=[], action="append")
    ap.add_argument("--full", nargs="*", default=[], action="append",
                    help="checkpoints graded on the whole arena (volleyball-only models)")
    ap.add_argument("--four", nargs=4, default=[], action="append", metavar=("G1", "G2", "G3", "G4"),
                    help="one checkpoint per four-way hold-out group, in order g1..g4 "
                         "(repeat the flag per model row)")
    ap.add_argument("--stitch-maps", nargs=2, default=[], action="append",
                    metavar=("A_DIR", "B_DIR"),
                    help="saved mapeval maps (folders holding map-control.npz) of a "
                         "model trained on fold A and one trained on fold B, stitched "
                         "the same way (the per-ball control)")
    ap.add_argument("--fuse", default=None,
                    help="also add, for the last stitched map-model row, a row with its "
                         "probabilities averaged with this earlier model's saved map "
                         "(a name from compare.PRIOR)")
    ap.add_argument("--names", nargs="*", default=[],
                    help="row names: one per --full, then one per stitched pair")
    args = ap.parse_args(argv)
    dev = torch.device("cuda")
    lance = load_cloud("LANCE")
    labels = lance.labels
    vis = G.visible_from_geometry(LANCE_GEOMETRY, labels)
    grid = G.map_grid(lance)
    names = list(args.names)
    rows = {"full": [], "stitched": [], "ground_A": [], "ground_B": []}

    for name, folder in PRIOR.items():
        cells = load_map(folder)
        # Each earlier model at its own stored threshold, which is what it runs at.
        thr = STORED_THRESHOLD.get(name) or _stored_threshold(folder)
        rows["full"].append(_row(name, G.grade(cells, labels, vis, thr)))
        for g, fold in (("ground_A", "B"), ("ground_B", "A")):
            c, v = restrict(cells, vis, labels, fold_region(fold, labels, False))
            rows[g].append(_row(name, G.grade(c, labels, v, thr)))

    for cks in [c for c in args.full if c]:
        m, fs, _ = load(cks, dev)
        prob, st = G.predict(m, grid, dev, fs)
        rows["full"].append(_row(names.pop(0), G.grade(G.to_cells(prob, st, grid),
                                                       labels, vis, 0.5)))

    pairs = [(a, b) for a, b in zip([c for c in args.fold_a if c], [c for c in args.fold_b if c])]
    for a, b in pairs:
        name = names.pop(0)
        prob, st = stitched_prob(a, b, grid, dev, False, labels)
        cells = G.to_cells(prob, st, grid)
        rows["stitched"].append(_row(name, G.grade(cells, labels, vis, 0.5)))
        for g, fold in (("ground_A", "B"), ("ground_B", "A")):
            c, v = restrict(cells, vis, labels, fold_region(fold, labels, False))
            rows[g].append(_row(name, G.grade(c, labels, v, 0.5)))

    from ..visual_audit import CellMap
    if args.fuse and pairs:
        prior = load_map(PRIOR[args.fuse])
        fused = fuse_maps(cells, prior)
        name = f"{rows['stitched'][-1]['model']} + {args.fuse}, averaged"
        rows["stitched"].append(_row(name, G.grade(fused, labels, vis, 0.5)))
        for g, fold in (("ground_A", "B"), ("ground_B", "A")):
            c, v = restrict(fused, vis, labels, fold_region(fold, labels, False))
            rows[g].append(_row(name, G.grade(c, labels, v, 0.5)))
    for a_dir, b_dir in [p for p in args.stitch_maps if p]:
        name = names.pop(0)
        a, b = load_map(a_dir), load_map(b_dir)
        on_a = fold_region("A", labels, True)
        # Fold A's ground is read by the map of the model trained on fold B.
        # Split by the 10 cm cell's centre, so no cell can come from both maps.
        ka = ~on_a((a.cell_ids + 0.5) * G.CELL_OUT_M)
        kb = on_a((b.cell_ids + 0.5) * G.CELL_OUT_M)
        cells = CellMap(np.concatenate([a.positions[ka], b.positions[kb]]),
                        np.concatenate([a.probabilities[ka], b.probabilities[kb]]),
                        np.concatenate([a.cell_ids[ka], b.cell_ids[kb]]))
        rows["stitched"].append(_row(name, G.grade(cells, labels, vis, 0.5)))
        for g, fold in (("ground_A", "B"), ("ground_B", "A")):
            c, v = restrict(cells, vis, labels, fold_region(fold, labels, False))
            rows[g].append(_row(name, G.grade(c, labels, v, 0.5)))

    from .stitch import stitched_prob_groups
    from .train import LANCE_GROUPS4
    rows["four_way"] = []
    for quad in [q for q in args.four if q]:
        name = names.pop(0)
        groups = {g: [c] for g, c in zip(sorted(LANCE_GROUPS4), quad)}
        prob, st = stitched_prob_groups(groups, grid, dev, False, labels)
        cells = G.to_cells(prob, st, grid)
        rows["four_way"].append(_row(name, G.grade(cells, labels, vis, 0.5)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1, default=float)
    write_markdown(rows, os.path.splitext(args.out)[0].replace("tables", "results") + ".md"
                   if "tables" in os.path.basename(args.out)
                   else os.path.splitext(args.out)[0] + ".md")
    for g, rs in rows.items():
        print(f"== {g}")
        for r in rs:
            fmt = lambda v: " -  " if v is None else f"{v:.2f}"   # noqa: E731
            print(f"  {r['model']:30s} attr " + " ".join(fmt(r[f'attr@{k}']) for k in BUDGETS)
                  + " | foot " + " ".join(fmt(r[f'foot@{k}']) for k in BUDGETS)
                  + f" | false@0.5 {r['false@0.5']}")


if __name__ == "__main__":
    main()
