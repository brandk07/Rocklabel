"""Train a map model and grade it every epoch.

Everything a run needs is in one folder: ``config.json``, ``log.csv`` (one row
per epoch with the training loss, the held-out volleyball maps' numbers and the
arena's), a checkpoint per epoch, and ``final.pt``.

The arena is graded every epoch so the learning curve can be read afterwards;
it is never used to choose anything inside a run. Which epoch to keep is fixed
before training (the last one), per the project's epoch-selection finding.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import grade as G
from .clouds import Cloud
from .data import IGNORE, Aug, MapCrops
from .grid import channels
from .model import build

CLOUDS = "training/caches/mapnet-v1/clouds"
LANCE_LABELS = "labels/archive/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json"
LANCE_GEOMETRY = "training/caches/mapeval-lance/geometry"
VB_RUNS = [f"VB{i}" for i in range(2, 14)]


def vb_labels(run: str) -> str:
    return f"labels/volleyball/VolleyBallTest{run[2:]}.reslam.labels.json"


def load_cloud(name: str) -> Cloud:
    if name == "LANCE":
        return Cloud(os.path.join(CLOUDS, "LANCE.npz"), LANCE_LABELS)
    return Cloud(os.path.join(CLOUDS, f"{name}.npz"), vb_labels(name))


def half_plane(axis: int, split: float, keep_low: bool, buffer_m: float = 0.0):
    """Region callable on world xy: one side of an axis-aligned line."""
    def region(xy):
        v = np.asarray(xy, float)[:, axis]
        return v < split - buffer_m if keep_low else v >= split + buffer_m
    return region


#: Spatial folds of the arena, for measuring what in-domain labels are worth.
#: Each fold is a set of rocks, and owns every piece of ground nearer to one of
#: its rocks than to any rock of the other fold. Rocks that touch (4-3-1-13 and
#: 9-10-11) are kept together, so no rock's outline is ever cut by the split.
LANCE_FOLDS = {"A": (5, 8, 9, 10, 11, 12), "B": (1, 3, 4, 6, 7, 13)}


#: A four-way split for held-out estimates closer to "the whole arena is
#: labelled": each model trains on eight or nine rocks and is graded on the
#: other three or four. Touching rocks stay together here too.
LANCE_GROUPS4 = {"g1": (9, 10, 11), "g2": (1, 3, 4, 13), "g3": (5, 8, 12), "g4": (6, 7)}


def group_rocks(name: str) -> tuple:
    return LANCE_FOLDS[name] if name in LANCE_FOLDS else LANCE_GROUPS4[name]


def fold_region(fold: str, labels, inside: bool, buffer_m: float = 0.0):
    """Region callable on world xy: the ground belonging to ``fold`` (or, with
    ``inside=False``, to every other rock), shrunk by ``buffer_m`` from the line
    between them. ``fold`` names a two-way fold (A, B) or a four-way group."""
    mine = set(group_rocks(fold))
    a = np.array([r.center[:2] for r in labels.rocks if (r.id in mine) == inside], float)
    b = np.array([r.center[:2] for r in labels.rocks if (r.id in mine) != inside], float)

    def region(xy):
        xy = np.asarray(xy, float)
        da = np.sqrt(((xy[:, None, :] - a[None]) ** 2).sum(-1)).min(1)
        db = np.sqrt(((xy[:, None, :] - b[None]) ** 2).sum(-1)).min(1)
        # (db - da) / 2 is roughly the distance to the bisector between them.
        return (db - da) / 2 > buffer_m
    return region


def loss_fn(logit, y, pos_weight: float, dice_w: float):
    valid = y != IGNORE
    if not valid.any():
        return logit.sum() * 0.0
    t = (y == 1).float()
    lg = logit[:, 0]
    bce = F.binary_cross_entropy_with_logits(
        lg[valid], t[valid], pos_weight=torch.tensor(pos_weight, device=lg.device))
    if not dice_w:
        return bce
    p = torch.sigmoid(lg) * valid
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + (t * valid).sum() + 1)
    return bce + dice_w * dice


_GRIDS: dict = {}


def _grid(cloud: Cloud):
    """The finished map never changes between epochs: build it once."""
    if cloud.name not in _GRIDS:
        _GRIDS[cloud.name] = G.map_grid(cloud)
    return _GRIDS[cloud.name]


def evaluate_maps(model, device, vals: list[Cloud], lance: Cloud | None, lance_vis,
                  lance_region=None, feature_set: str = "base", tta: bool = False,
                  cpu_features: bool = False) -> dict:
    out = {}
    for c in vals:
        g = _grid(c)
        prob, st = G.predict(model, g, device, feature_set, tta, cpu_features)
        r = G.grade(G.to_cells(prob, st, g), c.labels, G.visible_from_cloud(c), 0.5)
        out[c.name] = r
    if lance is not None:
        g = _grid(lance)
        prob, st = G.predict(model, g, device, feature_set, tta, cpu_features)
        cells = G.to_cells(prob, st, g)
        labels = lance.labels
        vis = lance_vis
        if lance_region is not None:
            # Grade only the held-out side: cells and rocks outside it drop out.
            from .compare import restrict
            cells, vis = restrict(cells, vis, labels, lance_region)
        out["LANCE"] = G.grade(cells, labels, vis, 0.5)
        out["_lance_cells"] = cells
    return out


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", required=True,
                    help="run folder: config.json, log.csv, one checkpoint per epoch, final.pt")
    ap.add_argument("--train", nargs="+", default=[r for r in VB_RUNS if r not in ("VB4", "VB6")],
                    help="clouds to train on, by name (VB2..VB13; default: all but VB4 and VB6)")
    ap.add_argument("--val", nargs="*", default=["VB4", "VB6"],
                    help="volleyball clouds graded every epoch for the learning curve")
    ap.add_argument("--lance-fold", choices=sorted(LANCE_FOLDS), default=None,
                    help="also train on this group of the arena's rocks (and the ground "
                         "nearest them); grade on the other group's ground only")
    ap.add_argument("--lance-holdout", choices=sorted(LANCE_GROUPS4), default=None,
                    help="four-way split: train on every arena rock except this group "
                         "(and the ground nearest them); grade on this group's ground only")
    ap.add_argument("--lance-all", action="store_true",
                    help="deployment fit: also train on the whole arena. Its arena "
                         "numbers are then in-sample and say nothing about accuracy")
    ap.add_argument("--lance-weight", type=float, default=0.3,
                    help="share of training crops drawn from the arena with --lance-fold")
    ap.add_argument("--no-lance-eval", action="store_true",
                    help="skip grading the arena every epoch")
    ap.add_argument("--arch", default="unet32", choices=["unet16", "unet32", "unet48", "unet64"],
                    help="network width")
    ap.add_argument("--size", type=int, default=192, help="crop side in 2.5 cm cells")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--samples", type=int, default=2000, help="crops per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--pos-weight", type=float, default=3.0,
                    help="how much a rock cell outweighs a clear one in the loss")
    ap.add_argument("--dice", type=float, default=0.5, help="weight of the overlap (Dice) loss")
    ap.add_argument("--workers", type=int, default=10, help="CPU crop workers (--cpu-data only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", default="{}", help="JSON overrides for data.Aug")
    ap.add_argument("--drop-channels", nargs="*", default=[],
                    help="zero these input channels (ablation)")
    ap.add_argument("--init", default=None, help="start from this checkpoint")
    ap.add_argument("--features", default="profile", choices=["base", "profile"],
                    help="input channels: heights and brightness only, or also the "
                         "height profile (share of returns per 2.5 cm slice)")
    ap.add_argument("--tta", action="store_true", help="grade with rotation/mirror averaging")
    ap.add_argument("--cpu-data", action="store_true", help="build crops on the CPU (reference)")
    ap.add_argument("--wall-clear-m", type=float, default=0.0,
                    help="label this band outside the arena's outline as clear ground "
                         "(arena crops only), so its walls are learned as not-rock. "
                         "It made the arena hold-outs worse; mask walls with the "
                         "arena outline instead")
    ap.add_argument("--ema", type=float, default=0.998,
                    help="decay of the running weight average that is graded and "
                         "saved (0 turns it off). One epoch's weights swing the "
                         "arena score by 0.1-0.2; the average does not")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    return run(ap.parse_args(argv))


def run(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aug = Aug(**json.loads(args.aug))
    sources = [(load_cloud(n), 1.0, None) for n in args.train]
    lance = (None if args.no_lance_eval and not (args.lance_fold or args.lance_holdout
                                                  or args.lance_all)
             else load_cloud("LANCE"))
    eval_region = None
    if sum(bool(x) for x in (args.lance_fold, args.lance_holdout, args.lance_all)) > 1:
        raise SystemExit("--lance-fold, --lance-holdout and --lance-all are different "
                         "splits; pick one")
    if args.lance_all:
        total = sum(w for _, w, _ in sources)
        sources.append((lance, args.lance_weight * total / (1 - args.lance_weight), None))
    if args.lance_fold and args.lance_holdout:
        raise SystemExit("--lance-fold and --lance-holdout are two different splits; pick one")
    if args.lance_holdout:
        total = sum(w for _, w, _ in sources)
        sources.append((lance, args.lance_weight * total / (1 - args.lance_weight),
                        fold_region(args.lance_holdout, lance.labels, False, buffer_m=0.25)))
        eval_region = fold_region(args.lance_holdout, lance.labels, True)
    if args.lance_fold:
        # Train on this fold's ground, with a buffer so no label reaches the
        # line; grade on the other fold's ground only.
        total = sum(w for _, w, _ in sources)
        sources.append((lance, args.lance_weight * total / (1 - args.lance_weight),
                        fold_region(args.lance_fold, lance.labels, True, buffer_m=0.25)))
        eval_region = fold_region(args.lance_fold, lance.labels, False)
    vals = [load_cloud(n) for n in args.val]
    lance_vis = (G.visible_from_geometry(LANCE_GEOMETRY, lance.labels)
                 if lance is not None else None)
    CHANNELS = channels(args.features)
    drop = [CHANNELS.index(c) for c in args.drop_channels]

    if args.cpu_data:
        ds = MapCrops(sources, size=args.size, length=args.samples, aug=aug, seed=args.seed,
                      feature_set=args.features)
    else:
        from .gpu import GpuCrops, GpuSource
        from .gpu import RockBank
        gsrc = [GpuSource(c, w, r, dev,
                          outside_clear_m=args.wall_clear_m if c.name == "LANCE" else 0.0)
                for c, w, r in sources]
        bank = RockBank(gsrc, dev) if aug.paste_p > 0 else None
        ds = GpuCrops(gsrc, args.size, aug, dev, args.features, bank=bank)
        brng = np.random.default_rng(args.seed)
    model = build(len(CHANNELS), args.arch).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * (args.samples // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    import copy
    ema = copy.deepcopy(model).eval() if args.ema else None
    if ema is not None:
        for p_ in ema.parameters():
            p_.requires_grad_(False)
    log_path = os.path.join(args.out, "log.csv")
    rows = []
    for epoch in range(args.epochs):
        if args.cpu_data:
            ds.epoch = epoch
            dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            drop_last=True, persistent_workers=False, prefetch_factor=4)
        else:
            dl = (ds.batch(args.batch, brng) for _ in range(args.samples // args.batch))
        model.train()
        t0 = time.monotonic()
        losses = []
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if drop:
                x = x.clone()
                x[:, drop] = 0
            with torch.autocast(device_type=dev.type, dtype=torch.float16,
                                enabled=dev.type == "cuda"):
                logit = model(x)
            loss = loss_fn(logit.float(), y, args.pos_weight, args.dice)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
            losses.append(float(loss.detach()))
        train_s = time.monotonic() - t0
        graded = ema if ema is not None else model
        state = {"model": graded.state_dict(), "channels": list(CHANNELS), "arch": args.arch,
                 "epoch": epoch, "config": vars(args), "drop": drop,
                 "feature_set": args.features}
        torch.save(state, os.path.join(args.out, f"epoch-{epoch:03d}.pt"))
        model_eval = _DropWrap(graded, drop) if drop else graded
        res = evaluate_maps(model_eval, dev, vals, lance, lance_vis, eval_region,
                            feature_set=args.features, tta=args.tta,
                            cpu_features=args.cpu_data)
        row = {"epoch": epoch, "loss": round(float(np.mean(losses)), 4),
               "train_s": round(train_s, 1)}
        for name, r in res.items():
            if name.startswith("_"):
                continue
            row[f"{name}_cov@0.5"] = round(r["macro_coverage"], 3)
            row[f"{name}_false@0.5"] = r["false_cells_footprint"]
            for b in ("1000", "500", "200", "100"):
                v = r["budgets"].get(b)
                row[f"{name}_cov@{b}"] = None if v is None else round(v["macro_coverage"], 3)
            row[f"{name}_worst@500"] = (None if r["budgets"].get("500") is None
                                        else round(r["budgets"]["500"]["worst_rock_coverage"], 3))
        rows.append(row)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        print(json.dumps(row), flush=True)
        if "LANCE" in res:
            G.save(res["LANCE"], os.path.join(args.out, f"lance-epoch-{epoch:03d}"))
    torch.save(state, os.path.join(args.out, "final.pt"))
    if "LANCE" in res:
        G.plot(os.path.join(args.out, "lance-final.png"), res["_lance_cells"], lance.labels,
               0.5, f"{os.path.basename(args.out)} final, threshold 0.5")


class _DropWrap(torch.nn.Module):
    def __init__(self, model, drop):
        super().__init__()
        self.model, self.drop = model, drop

    def forward(self, x):
        x = x.clone()
        x[:, self.drop] = 0
        return self.model(x)


if __name__ == "__main__":
    main()
