"""`rocklabel-train`: cache, train, compare, plot, export, and view.

Kept separate from the base `rocklabel` CLI so the core tool never imports
torch. Typical session:

    rocklabel-train cache                 # pool every full-sweep dataset, verify
    rocklabel-train compare               # both models x every LORO fold + figures
    rocklabel-train view datasets/full-sweep/volleyball \\
        --checkpoint training/experiments/compare/pointnet2_loro_VolleyBallTest4.reslam/best.pt
    rocklabel-train export training/experiments/compare/pointnet2_loro_VolleyBallTest4.reslam/best.pt
"""

from __future__ import annotations

import argparse
import json
import os

from ..dataset.neighborhoods import FEATURES
from ..profiles import DEFAULT_PROFILE
from .models_meta import MODELS
from . import TRAIN_DEFAULTS
from .ablate import (DEFAULT_REPORT_ROOT as REPORT_ROOT, DEFAULT_ROOT as ABLATE_ROOT,
                     SUITES)
from .matched import AGGREGATIONS, DEFAULT_RADIUS_M
from .data import default_datasets, run_dir_name, run_suffix

DEFAULT_ROOT = "training"

#: Where `cache` writes and every training command reads. One cache per
#: generation profile, because a cache built from full-sweep frames and one
#: built from raw bursts hold populations that must never be pooled.
DEFAULT_CACHE = os.path.join(DEFAULT_ROOT, "caches", DEFAULT_PROFILE)

#: `compare`/`train` keep flat ``<model>_loro_<run>`` directories, so they get
#: an experiment folder of their own rather than being scattered.
DEFAULT_RUNS_ROOT = os.path.join(ABLATE_ROOT, "compare")

#: Training settings the ablation sweep passes through to every arm. An arm's
#: own overrides win over these - the arm's overrides are the thing under test.
ABLATE_PASSTHROUGH = ("epochs", "batch", "lr", "weight_decay", "patience",
                      "val_frac", "gap_frames", "gap_seconds", "augment",
                      "aug_intensity_gain", "aug_intensity_shift", "aug_thin_min",
                      "dropout", "tnet", "seed", "device")


def _settings_match(run_dir: str, cfg: dict) -> bool:
    """True when run_dir was produced by exactly the settings we are asking for.

    ``compare`` skips folds that already carry a test_metrics.json, and that
    check used to run *before* anything compared configs — so a fold left over
    from an earlier dataset name or an earlier hyperparameter was kept silently
    and reported as part of the new sweep. (It happened: a PointNet fold-1 run
    held out ``ConforterTest1`` and survived the rename.) Checking here means a
    stale directory is retrained rather than trusted.
    """
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        old = json.load(f)
    old.setdefault("features", list(FEATURES))  # predates the channel setting
    return old == cfg


def _archive_stale(run_dir: str) -> str:
    """Move a run directory aside so a retrain can use its name.

    Renamed rather than deleted: the old checkpoints are the only record of
    what the previous settings scored, and ``train_fold`` refuses to write into
    a directory whose config disagrees with it — which is the guard working, not
    something to override.
    """
    import time

    dest = f"{run_dir}.superseded-{time.strftime('%Y%m%dT%H%M%S')}"
    os.rename(run_dir, dest)
    return dest


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache-dir", default=DEFAULT_CACHE,
                   help=f"pooled cache to train from (default: {DEFAULT_CACHE}). "
                        "There is one per generation profile; pointing this at "
                        "the wrong one trains on differently-cut frames.")
    p.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT,
                   help=f"where 'train' and 'compare' write their fold "
                        f"directories (default: {DEFAULT_RUNS_ROOT})")
    p.add_argument("--results-dir", default=None,
                   help=f"default: {REPORT_ROOT}/compare, tagged with the input "
                        "channels when they are not the full set, so one "
                        "channel selection's figures never overwrite another's")


def _add_gpu_arg(p: argparse.ArgumentParser) -> None:
    """Cap this process's VRAM share, for two sweeps sharing one card."""
    p.add_argument("--gpu-fraction", type=float, default=None, metavar="FRAC",
                   help="most of the graphics card's memory this run may ever "
                        "take, as a fraction (0.45 = 45%%). With two sweeps on "
                        "one card the one that asks for memory second is the "
                        "one that crashes, which is usually not the one at "
                        "fault; capping means an overreaching run fails on its "
                        "own account and never knocks over a sweep already "
                        "going. Leave empty for no cap.")


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
    return os.path.join(REPORT_ROOT, "compare") + run_suffix(args.features)


def _add_train_args(p: argparse.ArgumentParser) -> None:
    # Every value here defaults to None so TRAIN_DEFAULTS stays the only place
    # a default is written down; default_config drops the Nones. Hardcoding
    # them a second time here is how --patience silently stayed at 6.
    def opt(flag, **kw):
        d = TRAIN_DEFAULTS[flag.lstrip("-").replace("-", "_")]
        kw["help"] = f"{kw.get('help', '').rstrip()} (default: {d})".lstrip()
        p.add_argument(flag, default=None, **kw)

    _features_arg(p)
    opt("--epochs", type=int)
    opt("--batch", type=int)
    opt("--lr", type=float)
    opt("--weight-decay", type=float)
    opt("--patience", type=int,
        help="stop after this many epochs with no val PR-AUC gain; keep it "
             "long enough that the cosine LR schedule can finish annealing")
    opt("--val-frac", type=float)
    opt("--gap-frames", type=int,
        help="minimum kept frames dropped between the train and val blocks")
    opt("--gap-seconds", type=float,
        help="wall-clock buffer between the train and val blocks; overrides "
             "--gap-frames when it implies a wider gap")
    opt("--aug-intensity-gain", type=float,
        help="half-width of the per-sample reflectivity gain jitter (0 = off)")
    opt("--aug-intensity-shift", type=float,
        help="half-width of the per-sample reflectivity offset jitter (0 = off)")
    opt("--aug-thin-min", type=float,
        help="smallest fraction of a neighborhood's real points kept by the "
             "density augmentation (1.0 = off)")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--tnet", action="store_true",
                   help="enable PointNet input+feature T-Nets (data is already "
                        "canonicalized, so default off)")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--seed", type=int, default=None,
                   help=f"(default: {TRAIN_DEFAULTS['seed']})")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--fresh", action="store_true", help="ignore an existing last.pt")


def _train_cfg(args, model: str, train_runs: list[str], test_run: str) -> dict:
    from .engine import default_config
    return default_config(
        model=model, features=args.features, tnet=args.tnet,
        dropout=args.dropout, cache_dir=args.cache_dir,
        train_runs=train_runs, test_run=test_run, val_frac=args.val_frac,
        gap_frames=args.gap_frames, gap_seconds=args.gap_seconds,
        epochs=args.epochs, batch=args.batch, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience, augment=args.augment,
        aug_intensity_gain=args.aug_intensity_gain,
        aug_intensity_shift=args.aug_intensity_shift,
        aug_thin_min=args.aug_thin_min,
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
    p.add_argument("--datasets", nargs="+", default=None,
                   help="dataset directories to pool (default: every dataset "
                        f"under datasets/{DEFAULT_PROFILE}/)")
    _add_common(p)

    p = sub.add_parser("train", help="train one model on one leave-one-run-out fold")
    p.add_argument("--model", choices=sorted(MODELS), required=True,
                   help="; ".join(f"{k} = {v[1]}" for k, v in MODELS.items()))
    p.add_argument("--test-run", required=True, help="run held out for testing")
    _add_common(p)
    _add_train_args(p)
    _add_gpu_arg(p)

    p = sub.add_parser("compare", help="train both models on every LORO fold, "
                                       "then render all comparison figures")
    p.add_argument("--models", nargs="+", default=["pointnet", "pointnet2"],
                   choices=sorted(MODELS), metavar="MODEL",
                   help="models to train/report on; " +
                        "; ".join(f"{k} = {v[1]}" for k, v in MODELS.items()))
    _add_common(p)
    _add_train_args(p)
    _add_gpu_arg(p)

    p = sub.add_parser("ablate", help="controlled A/B sweep: train every arm of a "
                                      "suite on every fold, then report paired "
                                      "per-fold differences")
    p.add_argument("--suite", default="reflectivity", choices=sorted(SUITES),
                   help="which set of arms to run; " +
                        "; ".join(f"{k} = {v['title']}" for k, v in SUITES.items()))
    p.add_argument("--arms", nargs="+", default=None, metavar="ARM",
                   help="run only these arms of the suite (default: all of them, "
                        "in the order they are declared - which is priority "
                        "order, so stopping early still leaves the headline "
                        "comparison finished)")
    p.add_argument("--ablate-root", default=ABLATE_ROOT,
                   help=f"where each arm's runs live (default: {ABLATE_ROOT}). "
                        "One directory per arm, which is what lets two arms "
                        "differing only in an augmentation setting coexist.")
    p.add_argument("--report-only", action="store_true",
                   help="skip training; rebuild the tables and figures from "
                        "whatever folds have already finished")
    _add_common(p)
    _add_train_args(p)
    _add_gpu_arg(p)

    p = sub.add_parser("matched", help="score a segmenter and a sliding-window "
                                       "classifier on one shared set of candidate "
                                       "centers, so their numbers can be compared")
    p.add_argument("--suite", default="fullsweep", choices=sorted(SUITES),
                   help="which sweep's arms to read; only suites holding both a "
                        "segmentation arm and a classifier arm have anything to "
                        "compare")
    p.add_argument("--ablate-root", default=ABLATE_ROOT,
                   help=f"where the sweep's runs live (default: {ABLATE_ROOT})")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--out", default=None,
                   help=f"output dir (default: {REPORT_ROOT}/<suite>/matched)")
    p.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M,
                   help="how far from a candidate center a segmented point may sit "
                        f"and still describe it (default: {DEFAULT_RADIUS_M} m)")
    p.add_argument("--aggregation", default="max", choices=list(AGGREGATIONS),
                   help="how the per-point probabilities near a center become one "
                        "number for it (default: max)")

    p = sub.add_parser("reflect", help="measure what the reflectivity channel "
                                       "actually carries, straight off the cache "
                                       "(no training)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--out", default=os.path.join(REPORT_ROOT, "reflect"),
                   help="directory for the figures and tables "
                        f"(default: {REPORT_ROOT}/reflect)")

    p = sub.add_parser("report", help="regenerate figures/tables from existing runs")
    p.add_argument("--models", nargs="+", default=["pointnet", "pointnet2"],
                   choices=sorted(MODELS), metavar="MODEL",
                   help="models to train/report on; " +
                        "; ".join(f"{k} = {v[1]}" for k, v in MODELS.items()))
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

    # Before anything imports torch and allocates: set_per_process_memory_fraction
    # only binds allocations made after it.
    if getattr(args, "gpu_fraction", None):
        from .gpu import cap_gpu
        cap_gpu(args.gpu_fraction)

    if args.command == "cache":
        from .data import build_cache
        build_cache(args.datasets or default_datasets(), args.cache_dir)
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
                cfg = _train_cfg(args, model, fold["train"], fold["test"])
                if os.path.isdir(run_dir) and not _settings_match(run_dir, cfg):
                    dest = _archive_stale(run_dir)
                    print(f"{run_dir} holds a run with different settings -> "
                          f"moved to {os.path.basename(dest)}, retraining")
                elif os.path.exists(os.path.join(run_dir, "test_metrics.json")) and not args.fresh:
                    print(f"skip {run_dir} (already evaluated)")
                    continue
                train_fold(cfg, run_dir, resume=not args.fresh)
        render_all(args.runs_root, _results_dir(args), args.models,
                   [f["name"] for f in folds], features=args.features)
        return 0

    if args.command == "ablate":
        from .ablate import run_suite
        from .ablate_report import render_ablation
        if not args.report_only:
            extra = {k: getattr(args, k) for k in ABLATE_PASSTHROUGH}
            run_suite(args.suite, args.cache_dir, args.ablate_root, args.arms,
                      extra, fresh=args.fresh)
        out = args.results_dir or os.path.join(REPORT_ROOT, args.suite)
        render_ablation(args.ablate_root, args.suite, out)
        return 0

    if args.command == "matched":
        from .matched import render_matched
        out = args.out or os.path.join(REPORT_ROOT, args.suite, "matched")
        render_matched(args.cache_dir, args.ablate_root, args.suite, out,
                       radius=args.radius, aggregation=args.aggregation)
        return 0

    if args.command == "reflect":
        from .reflect import render_reflectivity
        render_reflectivity(args.cache_dir, args.out)
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
