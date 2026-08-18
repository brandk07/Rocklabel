"""Controlled experiments: does a channel (or a model, or an augmentation)
actually change the score?

`compare` answers "how do the models do". This answers "does *this one thing*
matter", which needs a different shape of run. Two rules make the difference:

* **One arm, one run root.** ``compare`` names a run directory after its model,
  fold and channel selection — so two arms that differ only in an augmentation
  setting would land on the same directory, and the second would archive the
  first as stale. Every arm here gets ``<root>/<arm>/`` to itself, so any two
  settings can be compared, not just channel selections.
* **Paired by fold.** Arms are compared fold by fold, never as two pooled
  averages. Fold-to-fold spread on this data is far larger than any channel
  effect (PR-AUC ranges roughly 0.5-0.95 across runs), so an unpaired
  comparison drowns the thing being measured in which-run-was-held-out noise.

The seed-repeat arms exist for the same reason. A single training run is a
random draw; without knowing how far two runs of the *same* setting land apart,
a delta between two different settings cannot be called real. Repeats of one
arm under different seeds measure exactly that floor.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..dataset.neighborhoods import FEATURES, GEOMETRY

#: Where a sweep's per-fold run directories live: one folder per experiment,
#: then per arm, then per fold. Renamed from the old ``training/ablate`` (and
#: its ad-hoc twin ``training/ablate_vb``) so that every trained thing in the
#: project sits under one root that says which experiment produced it.
DEFAULT_ROOT = os.path.join("training", "experiments")

#: Where the rendered tables and figures for an experiment go.
DEFAULT_REPORT_ROOT = os.path.join("training", "reports")

#: Metrics carried into every table, primary first. PR-AUC leads because the
#: data is 5-31% rock depending on the run: accuracy is nearly free and ROC-AUC
#: is optimistic when negatives dominate.
METRICS = ("pr_auc", "roc_auc", "f1", "precision", "recall")
PRIMARY = "pr_auc"


@dataclass
class Arm:
    """One training setting, trained on every fold.

    ``overrides`` goes straight into the training config, so an arm can change
    anything ``default_config`` accepts — not just channels.
    """

    name: str
    model: str
    features: list[str]
    label: str          # short name for figures
    what: str           # plain-English description, shown in the report
    overrides: dict = field(default_factory=dict)
    seed: int | None = None

    def config(self, cfg_fn, train_runs: list[str], test_run: str) -> dict:
        kw = dict(self.overrides)
        if self.seed is not None:
            kw["seed"] = self.seed
        return cfg_fn(model=self.model, features=list(self.features),
                      train_runs=train_runs, test_run=test_run, **kw)


_ALL = list(FEATURES)
_GEOM = list(GEOMETRY)
#: Turning the reflectivity jitter off. The default augmentation deliberately
#: swamps the absolute intensity level (gain +/-25%, offset +/-0.10 on a channel
#: whose entire useful range is about 0.10 wide), so a model trained with it on
#: cannot use the absolute level even if the level were informative. Any honest
#: test of "does reflectivity help" has to include an arm where it is allowed to.
_NO_JITTER = {"aug_intensity_gain": 0.0, "aug_intensity_shift": 0.0}


#: The reflectivity question, as a set of arms. Order is priority order: the
#: sweep runs top to bottom, so if it is stopped early the headline comparison
#: is the part that finished.
REFLECTIVITY_ARMS = [
    Arm("pointnet-geom", "pointnet", _GEOM, "PointNet · shape only",
        "PointNet with the reflectivity channel removed entirely. The control: "
        "whatever this scores is what pure geometry is worth."),
    Arm("pointnet-refl", "pointnet", _ALL, "PointNet · shape + reflectivity",
        "PointNet with reflectivity included, using the standard augmentation "
        "(reflectivity randomly rescaled and shifted each sample). This is the "
        "setting the existing runs used."),
    Arm("pointnet2-geom", "pointnet2", _GEOM, "PointNet++ · shape only",
        "PointNet++ with no reflectivity. The shape-only control for the "
        "hierarchical model."),
    Arm("pointnet2-refl", "pointnet2", _ALL, "PointNet++ · shape + reflectivity",
        "PointNet++ with reflectivity included and the standard augmentation."),
    Arm("pointnet-refl-raw", "pointnet", _ALL, "PointNet · reflectivity unjittered",
        "PointNet with reflectivity included and the reflectivity augmentation "
        "switched off, so the model may use the raw absolute brightness. If "
        "reflectivity helps anywhere, it helps most here — and the gap between "
        "this and the jittered arm is the size of the cue the augmentation "
        "deliberately destroys.",
        overrides=_NO_JITTER),
    Arm("pointnet2-refl-raw", "pointnet2", _ALL, "PointNet++ · reflectivity unjittered",
        "PointNet++ with reflectivity included and its augmentation off.",
        overrides=_NO_JITTER),
    Arm("pointnet-refl-only", "pointnet", ["intensity"], "PointNet · reflectivity only",
        "PointNet fed nothing but reflectivity — no coordinates at all. It "
        "cannot see shape, so its score is a direct read of how much the "
        "brightness numbers alone can separate rock from sand.",
        overrides=_NO_JITTER),
    # Seed repeats of the two headline arms: the yardstick for "is a delta real".
    Arm("pointnet-geom-s43", "pointnet", _GEOM, "PointNet · shape only (seed 43)",
        "Same setting as the shape-only arm, different random seed. Exists only "
        "to measure how far two identical settings land apart.", seed=43),
    Arm("pointnet-refl-s43", "pointnet", _ALL, "PointNet · shape + reflectivity (seed 43)",
        "Seed repeat of the shape+reflectivity arm.", seed=43),
    Arm("pointnet-geom-s44", "pointnet", _GEOM, "PointNet · shape only (seed 44)",
        "Second seed repeat of the shape-only arm.", seed=44),
    Arm("pointnet-refl-s44", "pointnet", _ALL, "PointNet · shape + reflectivity (seed 44)",
        "Second seed repeat of the shape+reflectivity arm.", seed=44),
]

#: Pairs the report tests head to head, as (baseline, variant, question).
REFLECTIVITY_CONTRASTS = [
    ("pointnet-geom", "pointnet-refl",
     "PointNet: does adding reflectivity beat shape alone?"),
    ("pointnet2-geom", "pointnet2-refl",
     "PointNet++: does adding reflectivity beat shape alone?"),
    ("pointnet-geom", "pointnet-refl-raw",
     "PointNet: does reflectivity help when its augmentation is switched off?"),
    ("pointnet2-geom", "pointnet2-refl-raw",
     "PointNet++: does reflectivity help when its augmentation is switched off?"),
    ("pointnet-refl", "pointnet-refl-raw",
     "PointNet: how much does the reflectivity augmentation cost?"),
    ("pointnet-geom", "pointnet2-geom",
     "Shape only: is PointNet++ better than PointNet?"),
    ("pointnet-refl", "pointnet2-refl",
     "Shape + reflectivity: is PointNet++ better than PointNet?"),
    ("pointnet-geom", "pointnet-geom-s43",
     "Noise floor: the same shape-only setting, two different seeds."),
    ("pointnet-refl", "pointnet-refl-s43",
     "Noise floor: the same shape+reflectivity setting, two different seeds."),
]

#: The full-sweep question, as a set of arms. Two things change at once
#: relative to the reflectivity suite, on purpose:
#:
#: * the dataset is built from whole 20 Hz sensor rotations instead of single
#:   ~4 ms sensor batches, so a frame carries ~1250 points in the crop box
#:   instead of ~110, and
#: * a third model joins in - the per-point segmenter, which could not be
#:   trained at all on the batch-sized frames (they fell below the 512-point
#:   floor, so the generator produced zero segmentation frames).
#:
#: Order is priority order: the segmenter-vs-classifier headline runs first,
#: then reflectivity, then the seed repeats that say how big a meaningless
#: difference looks. Stopping early still leaves the headline finished.
FULLSWEEP_ARMS = [
    Arm("pointnet2-geom", "pointnet2", _GEOM, "PointNet++ · shape only",
        "PointNet++ scoring one 0.5 m ball at a time, with the reflectivity "
        "channel removed. This is the arm to line up against the same-named "
        "arm of the reflectivity suite: same model, same settings, same folds - "
        "the only difference is that a frame here is a whole sensor rotation."),
    Arm("pointnet2seg-geom", "pointnet2_seg", _GEOM, "Segmentation · shape only",
        "PointNet++ labelling every point of a whole frame in one pass, shape "
        "only. The headline arm: it answers a rock question in one forward pass "
        "per frame instead of one per candidate ball."),
    Arm("pointnet-geom", "pointnet", _GEOM, "PointNet · shape only",
        "Plain PointNet on single balls, shape only - the cheapest model, kept "
        "as the floor everything else has to beat."),
    Arm("pointnet2-refl", "pointnet2", _ALL, "PointNet++ · shape + reflectivity",
        "PointNet++ on single balls with reflectivity added back, using the "
        "standard brightness jitter."),
    Arm("pointnet2seg-refl", "pointnet2_seg", _ALL, "Segmentation · shape + reflectivity",
        "The whole-frame segmenter with reflectivity added back. Denser frames "
        "give the segmenter far more brightness context than a single ball has, "
        "so this is where reflectivity has its best chance of paying off."),
    Arm("pointnet-refl", "pointnet", _ALL, "PointNet · shape + reflectivity",
        "Plain PointNet with reflectivity included."),
    Arm("pointnet2-geom-s43", "pointnet2", _GEOM, "PointNet++ · shape only (seed 43)",
        "Same setting as the shape-only PointNet++ arm, different random seed. "
        "Exists only to measure how far two identical settings land apart, so a "
        "difference between two real arms can be called real or not.", seed=43),
    Arm("pointnet2seg-geom-s43", "pointnet2_seg", _GEOM, "Segmentation · shape only (seed 43)",
        "Seed repeat of the shape-only segmentation arm - the noise floor for "
        "the segmenter.", seed=43),
]

#: Head-to-head questions for the full-sweep suite.
#:
#: NOTE the first two are scored per point for the segmenter and per candidate
#: ball for the classifiers, which are different populations with very
#: different rock prevalence (~1% of points vs ~19% of balls). The paired table
#: still says which arm won each fold, but the raw PR-AUC gap between a
#: segmenter and a classifier is not a like-for-like number - `vb_compare.py`
#: re-scores both at the same candidate centers for that.
FULLSWEEP_CONTRASTS = [
    ("pointnet2-geom", "pointnet2seg-geom",
     "Shape only: does whole-frame segmentation beat the sliding-window classifier?"),
    ("pointnet2-refl", "pointnet2seg-refl",
     "Shape + reflectivity: does whole-frame segmentation beat the classifier?"),
    ("pointnet-geom", "pointnet2-geom",
     "Shape only: is PointNet++ better than plain PointNet?"),
    ("pointnet2-geom", "pointnet2-refl",
     "PointNet++: does adding reflectivity beat shape alone?"),
    ("pointnet2seg-geom", "pointnet2seg-refl",
     "Segmentation: does adding reflectivity beat shape alone?"),
    ("pointnet-geom", "pointnet-refl",
     "PointNet: does adding reflectivity beat shape alone?"),
    ("pointnet2-geom", "pointnet2-geom-s43",
     "Noise floor: the same PointNet++ setting, two different seeds."),
    ("pointnet2seg-geom", "pointnet2seg-geom-s43",
     "Noise floor: the same segmentation setting, two different seeds."),
]

SUITES: dict[str, dict] = {
    "reflectivity": {
        "arms": REFLECTIVITY_ARMS,
        "contrasts": REFLECTIVITY_CONTRASTS,
        "title": "Does reflectivity help?",
        "blurb": "PointNet and PointNet++, with and without the reflectivity "
                 "channel, plus seed repeats that show how big a meaningless "
                 "difference looks.",
    },
    "fullsweep": {
        "arms": FULLSWEEP_ARMS,
        "contrasts": FULLSWEEP_CONTRASTS,
        "title": "Full sensor sweeps, and per-point segmentation",
        "blurb": "Trained on frames built from whole 20 Hz sensor rotations "
                 "(~1250 points in the crop box) instead of single ~4 ms sensor "
                 "batches (~110 points). Adds the whole-frame segmenter, which "
                 "the batch-sized frames were too sparse to train at all.",
    },
}


def arms_of(suite: str) -> list[Arm]:
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r} (pick from {sorted(SUITES)})")
    return SUITES[suite]["arms"]


def arm_dir(root: str, suite: str, arm: Arm, test_run: str) -> str:
    return os.path.join(root, suite, arm.name, f"loro_{test_run}")


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def run_suite(suite: str, cache_dir: str, root: str, only_arms: list[str] | None,
              extra: dict, fresh: bool = False) -> None:
    """Train every arm of ``suite`` on every leave-one-run-out fold.

    Folds already carrying a test_metrics.json are skipped, so an interrupted
    sweep resumes where it stopped. ``extra`` are config overrides applied to
    every arm (epochs, device, ...); an arm's own overrides win over them, since
    the arm's overrides are the thing being tested.
    """
    from .data import load_cache_meta, loro_folds
    from .engine import default_config, train_fold

    arms = arms_of(suite)
    if only_arms:
        unknown = [a for a in only_arms if a not in {x.name for x in arms}]
        if unknown:
            raise SystemExit(f"unknown arm(s) {unknown} in suite {suite!r}; "
                             f"available: {[a.name for a in arms]}")
        arms = [a for a in arms if a.name in only_arms]

    runs = sorted(load_cache_meta(cache_dir)["runs"])
    folds = loro_folds(runs)
    total = len(arms) * len(folds)
    done = 0
    print(f"suite {suite!r}: {len(arms)} arms x {len(folds)} folds = {total} runs")

    for arm in arms:
        for fold in folds:
            done += 1
            run_dir = arm_dir(root, suite, arm, fold["test"])
            tag = f"[{done}/{total}] {arm.name} / {fold['test']}"
            if os.path.exists(os.path.join(run_dir, "test_metrics.json")) and not fresh:
                print(f"{tag}: already evaluated, skipping")
                continue
            cfg = arm.config(
                lambda **kw: default_config(cache_dir=cache_dir, **{**extra, **kw}),
                fold["train"], fold["test"])
            print(f"\n=== {tag} ===")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "arm.json"), "w") as f:
                json.dump({"suite": suite, "arm": arm.name, "label": arm.label,
                           "what": arm.what, "model": arm.model,
                           "features": list(arm.features),
                           "overrides": arm.overrides, "seed": arm.seed}, f, indent=2)
            train_fold(cfg, run_dir, resume=not fresh)


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #
def wilcoxon_signed_rank(deltas: np.ndarray) -> tuple[float, float]:
    """(statistic, two-sided exact p) for the signed-rank test on paired deltas.

    Exact rather than normal-approximated because a leave-one-run-out sweep has
    ~10 pairs, where the normal approximation is not trustworthy. Zero deltas
    are dropped (the standard Wilcoxon handling); with fewer than one nonzero
    pair the p-value is 1.0 by definition — no evidence either way.

    Written out here rather than pulled from scipy because scipy is not a
    dependency of this project and one 20-line exact test is cheaper than
    making it one.
    """
    d = np.asarray([x for x in np.asarray(deltas, float) if x != 0.0])
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, float)
    ranks[order] = np.arange(1, n + 1)
    # Average the ranks of tied magnitudes, as the test requires.
    mag = np.abs(d)
    for value in np.unique(mag):
        tied = mag == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    stat = min(w_plus, w_minus)
    if n > 22:  # 2**n enumerations stop being cheap; normal approximation
        mu = n * (n + 1) / 4.0
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        from math import erfc, sqrt
        z = (stat - mu) / max(sigma, 1e-12)
        return stat, float(min(1.0, erfc(abs(z) / sqrt(2))))
    # Exact: enumerate every assignment of signs to the ranks.
    totals = np.zeros(1)
    for r in ranks:
        totals = np.concatenate([totals, totals + r])
    tail = float((totals <= stat + 1e-9).mean())
    return stat, float(min(1.0, 2.0 * tail))


def _fold_metrics(root: str, suite: str, arm_name: str) -> dict[str, dict]:
    """{test_run: test_metrics dict} for every finished fold of one arm."""
    base = os.path.join(root, suite, arm_name)
    out = {}
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name, "test_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            out[m.get("test_run", name.replace("loro_", "", 1))] = m
    return out


def collect(root: str, suite: str) -> dict:
    """Everything the report needs, read off disk. No torch, no retraining."""
    arms = arms_of(suite)
    per_arm = {a.name: _fold_metrics(root, suite, a.name) for a in arms}
    folds = sorted({f for m in per_arm.values() for f in m})
    rows = []
    for a in arms:
        m = per_arm[a.name]
        row = {"arm": a.name, "label": a.label, "what": a.what, "model": a.model,
               "features": list(a.features), "seed": a.seed,
               "overrides": a.overrides, "folds_done": len(m), "folds": list(m)}
        for k in METRICS:
            vals = np.array([m[f][k] for f in sorted(m)], float)
            row[k] = {"mean": float(vals.mean()) if len(vals) else None,
                      "std": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                      "per_fold": {f: float(m[f][k]) for f in sorted(m)}}
        rows.append(row)

    contrasts = []
    for base_name, var_name, question in SUITES[suite]["contrasts"]:
        a, b = per_arm.get(base_name, {}), per_arm.get(var_name, {})
        shared = sorted(set(a) & set(b))
        entry = {"baseline": base_name, "variant": var_name, "question": question,
                 "n_folds": len(shared), "folds": shared}
        for k in METRICS:
            if not shared:
                entry[k] = None
                continue
            d = np.array([b[f][k] - a[f][k] for f in shared], float)
            stat, p = wilcoxon_signed_rank(d)
            entry[k] = {
                "mean_delta": float(d.mean()),
                "std_delta": float(d.std(ddof=1)) if len(d) > 1 else None,
                "median_delta": float(np.median(d)),
                "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                "ties": int((d == 0).sum()),
                "statistic": stat, "p_value": p,
                "per_fold": {f: float(x) for f, x in zip(shared, d)},
            }
        contrasts.append(entry)

    return {"suite": suite, "title": SUITES[suite]["title"],
            "blurb": SUITES[suite]["blurb"], "folds": folds,
            "arms": rows, "contrasts": contrasts}
