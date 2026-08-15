"""`rocklabel-train`: cache, train, compare, plot, export, and view.

Kept separate from the base `rocklabel` CLI so the core tool never imports
torch. Typical session:

    rocklabel-train cache                 # pool the four myroom runs, verify
    rocklabel-train compare               # both models x 4 LORO folds + figures
    rocklabel-train view datasets/myroomdataset2 \\
        --checkpoint training/runs/pointnet2_loro_myroom2/best.pt
    rocklabel-train export training/runs/pointnet2_loro_myroom2/best.pt
"""

from __future__ import annotations

import argparse
import os

from ..neighborhoods import FEATURES
from .data import DEFAULT_DATASETS, run_dir_name, run_suffix

DEFAULT_ROOT = "training"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache-dir", default=os.path.join(DEFAULT_ROOT, "cache"))
    p.add_argument("--runs-root", default=os.path.join(DEFAULT_ROOT, "runs"))
    p.add_argument("--results-dir", default=None,
                   help=f"default: {DEFAULT_ROOT}/results, tagged with the input "
                        "channels when they are not the full set, so one "
                        "channel selection's figures never overwrite another's")


def _features_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--features", nargs="+", default=None, choices=list(FEATURES),
                   metavar="CHANNEL",
                   help="channels of the stored sample tensor to feed the model "
                        f"(default: all of {' '.join(FEATURES)}). Drop 'intensity' "
                        "to train on shape alone - reflectivity is the channel "
                        "least likely to transfer between arenas. Selection is a "
                        "model setting, so no dataset regeneration is needed; a "
                        "non-default selection is tagged into the run directory "
                        "name so it sits beside the runs it is compared against. "
                        "pointnet2 needs dx dy dz (it groups by position).")


def _results_dir(args) -> str:
    """Explicit --results-dir wins; otherwise tag the default by channel set."""
    if args.results_dir:
        return args.results_dir
    return os.path.join(DEFAULT_ROOT, "results") + run_suffix(args.features)


def _add_train_args(p: argparse.ArgumentParser) -> None:
    _features_arg(p)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--gap-frames", type=int, default=25,
                   help="frames dropped between the train and val blocks")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--tnet", action="store_true",
                   help="enable PointNet input+feature T-Nets (data is already "
                        "canonicalized, so default off)")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--fresh", action="store_true", help="ignore an existing last.pt")


def _train_cfg(args, model: str, train_runs: list[str], test_run: str) -> dict:
    from .engine import default_config
    return default_config(
        model=model, features=args.features, tnet=args.tnet,
        dropout=args.dropout, cache_dir=args.cache_dir,
        train_runs=train_runs, test_run=test_run, val_frac=args.val_frac,
        gap_frames=args.gap_frames, epochs=args.epochs, batch=args.batch, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience, augment=args.augment,
        seed=args.seed, device=args.device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rocklabel-train",
        description="PointNet / PointNet++ rock classifiers on format-A datasets. "
                    "Evaluation is leave-one-run-out by design: random sample "
                    "splits would leak near-duplicate neighborhoods.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cache", help="pool dataset runs into a flat .npy cache "
                                     "(validates config hashes and manifest counts)")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    _add_common(p)

    p = sub.add_parser("train", help="train one model on one leave-one-run-out fold")
    p.add_argument("--model", choices=["pointnet", "pointnet2"], required=True)
    p.add_argument("--test-run", required=True, help="run held out for testing")
    _add_common(p)
    _add_train_args(p)

    p = sub.add_parser("compare", help="train both models on every LORO fold, "
                                       "then render all comparison figures")
    p.add_argument("--models", nargs="+", default=["pointnet", "pointnet2"])
    _add_common(p)
    _add_train_args(p)

    p = sub.add_parser("report", help="regenerate figures/tables from existing runs")
    p.add_argument("--models", nargs="+", default=["pointnet", "pointnet2"])
    _add_common(p)
    _features_arg(p)  # names which channel selection's runs to report on

    p = sub.add_parser("export", help="export a checkpoint to TorchScript + ONNX + metadata")
    p.add_argument("checkpoint", help="path to a best.pt")
    p.add_argument("--out", default=None,
                   help="output dir (default: training/exported/<run name>)")

    p = sub.add_parser("replay", help="3D viewer: run the model live on any .mcap "
                                      "recording (no labels or dataset needed)")
    p.add_argument("mcap", help="path to the .mcap recording (either format)")
    p.add_argument("--checkpoint", required=True, help="path to a best.pt")
    p.add_argument("--config", default=None,
                   help="YAML config for the topics section (default: built-ins; "
                        "neighborhood geometry always comes from the checkpoint)")
    p.add_argument("--stride", type=int, default=None,
                   help="keep every Nth frame (default: the training config's)")
    p.add_argument("--window-s", type=float, default=None,
                   help="merge scans into time-window frames first (for native "
                        "lidarrig recordings; default: the training config's)")
    p.add_argument("--z-min", type=float, default=None,
                   help="score only points with z >= sensor z + this (m) - skip "
                        "the floor-band, e.g. sensor 1 m up: --z-min -1.5 --z-max -0.5")
    p.add_argument("--z-max", type=float, default=None,
                   help="score only points with z <= sensor z + this (m)")
    p.add_argument("--max-range", type=float, default=None,
                   help="score only points within this horizontal distance of the "
                        "sensor (m) - big speedup on wall/ceiling-heavy recordings")
    p.add_argument("--device", default=None)
    p.add_argument("--dump", default=None,
                   help="write frame/centers/probs to this .npz and exit (no window)")

    p = sub.add_parser("view", help="3D viewer: replay a run colored by model confidence")
    p.add_argument("dataset_dir", help="dataset directory (e.g. datasets/myroomdataset2)")
    p.add_argument("--checkpoint", required=True, help="path to a best.pt")
    p.add_argument("--run", default=None, help="run_id if the dataset has several")
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "cache":
        from .data import build_cache
        build_cache(args.datasets, args.cache_dir)
        return 0

    if args.command == "train":
        from .data import load_cache_meta
        from .engine import train_fold
        runs = sorted(load_cache_meta(args.cache_dir)["runs"])
        if args.test_run not in runs:
            raise SystemExit(f"test run {args.test_run!r} not in cache; available: {runs}")
        cfg = _train_cfg(args, args.model, [r for r in runs if r != args.test_run],
                         args.test_run)
        run_dir = os.path.join(args.runs_root,
                               run_dir_name(args.model, f"loro_{args.test_run}",
                                            args.features))
        train_fold(cfg, run_dir, resume=not args.fresh)
        return 0

    if args.command == "compare":
        from .data import load_cache_meta, loro_folds
        from .engine import train_fold
        from .plots import render_all
        runs = sorted(load_cache_meta(args.cache_dir)["runs"])
        folds = loro_folds(runs)
        for model in args.models:
            for fold in folds:
                run_dir = os.path.join(
                    args.runs_root, run_dir_name(model, fold["name"], args.features))
                if os.path.exists(os.path.join(run_dir, "test_metrics.json")) and not args.fresh:
                    print(f"skip {run_dir} (already evaluated)")
                    continue
                cfg = _train_cfg(args, model, fold["train"], fold["test"])
                train_fold(cfg, run_dir, resume=not args.fresh)
        render_all(args.runs_root, _results_dir(args), args.models,
                   [f["name"] for f in folds], features=args.features)
        return 0

    if args.command == "report":
        from .data import load_cache_meta, loro_folds
        from .plots import render_all
        runs = sorted(load_cache_meta(args.cache_dir)["runs"])
        render_all(args.runs_root, _results_dir(args), args.models,
                   [f["name"] for f in loro_folds(runs)], features=args.features)
        return 0

    if args.command == "export":
        from .export import export_model
        out = args.out or os.path.join(
            DEFAULT_ROOT, "exported", os.path.basename(os.path.dirname(args.checkpoint)))
        export_model(args.checkpoint, out)
        return 0

    if args.command == "view":
        from .confview import run_confview
        run_confview(args.dataset_dir, args.run, args.checkpoint,
                     device=args.device, frame=args.frame)
        return 0

    if args.command == "replay":
        from ..config import load_config
        from .mcapview import run_mcap_replay
        run_mcap_replay(args.mcap, args.checkpoint, load_config(args.config),
                        device=args.device, stride=args.stride,
                        window_s=args.window_s, dump=args.dump,
                        z_min=args.z_min, z_max=args.z_max,
                        max_range=args.max_range)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
