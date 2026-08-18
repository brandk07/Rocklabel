"""Score a per-point segmenter and a sliding-window classifier on one shared
population, so their numbers can honestly be put side by side.

The problem this solves: the two tasks are scored in different units. A
classifier is graded on candidate 0.5 m balls, of which ~19% sit on a rock. A
segmenter is graded on individual points, of which ~1% sit on a rock. PR-AUC
moves with prevalence - a no-skill model scores 0.19 on the first population
and 0.01 on the second - so the raw PR-AUC of a segmenter and a classifier are
simply not the same measurement, and reading them off one table ranks the
segmenter far below where it belongs.

The fix is to grade both on the classifier's population. Every candidate center
already carries a rock/clear label; the classifier emits a probability for it
directly, and the segmenter is asked for one by pooling its per-point
probabilities over the points sitting within ``radius`` of that center. Same
centers, same labels, same prevalence - so a difference in the resulting score
is a difference in the models, which is the only thing worth comparing.

Both dataset formats are written from the same kept frames of the same
recording (see generate.run_generate), so a center and the points it is matched
against come from one moment in time, not from two different scans.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial import cKDTree

from . import metrics as M
from .ablate import METRICS, PRIMARY, wilcoxon_signed_rank

#: How far from a candidate center a segmented point may sit and still count as
#: describing that center. Candidate centers are centroids of a 5 cm voxel, so
#: their own constituent points are within ~4 cm; 15 cm keeps those plus the
#: immediate surround while staying far below the 30 cm scale of a rock.
DEFAULT_RADIUS_M = 0.15

#: How the per-point probabilities inside that radius become one number for the
#: center. "max" asks "does the segmenter think anything here is rock", which is
#: the question the classifier's label actually poses.
AGGREGATIONS = ("max", "mean", "nearest")


def _pred_path(ablate_root: str, suite: str, arm: str, test_run: str) -> str:
    return os.path.join(ablate_root, suite, arm, f"loro_{test_run}", "predictions.npz")


def has_fold(ablate_root: str, suite: str, arm: str, test_run: str) -> bool:
    return os.path.exists(_pred_path(ablate_root, suite, arm, test_run))


def match_fold(cache_dir: str, ablate_root: str, suite: str, clf_arm: str,
               seg_arm: str, test_run: str, radius: float = DEFAULT_RADIUS_M,
               aggregation: str = "max") -> dict:
    """One fold's shared population: labels, classifier probs, segmenter probs.

    Returns ``None`` when either side has not been trained yet, so a report can
    be built off a sweep that is still running.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
    if not (has_fold(ablate_root, suite, clf_arm, test_run)
            and has_fold(ablate_root, suite, seg_arm, test_run)):
        return None

    with np.load(_pred_path(ablate_root, suite, clf_arm, test_run), allow_pickle=True) as z:
        clf_probs = z["probs"].astype(np.float64)
        clf_labels = z["labels"].astype(np.int8)
        clf_frame = z["frame"].astype(np.int64)
        centers = z["centers"].astype(np.float64)
    with np.load(_pred_path(ablate_root, suite, seg_arm, test_run), allow_pickle=True) as z:
        seg_probs = z["probs"].astype(np.float64)      # [F, N]
        seg_counts = z["counts"].astype(np.int64)      # [F]
        seg_frame = z["frame"].astype(np.int64)        # [F]
        seg_base = z["centers"].astype(np.float64)     # [F, 3] robot base in odom

    # The segmenter stores points relative to the robot base; the classifier
    # stores centers in odom. Put them both in odom before matching.
    local = np.load(os.path.join(cache_dir, test_run, "seg_points.npy"))
    seg_xyz = local[:, :, :3].astype(np.float64) + seg_base[:, None, :]

    row_of_frame = {int(f): i for i, f in enumerate(seg_frame)}

    y, p_clf, p_seg = [], [], []
    unmatched = 0
    for fi in np.unique(clf_frame):
        row = row_of_frame.get(int(fi))
        sel = clf_frame == fi
        if row is None:            # frame had too few points for a seg frame
            unmatched += int(sel.sum())
            continue
        n_real = int(seg_counts[row])
        tree = cKDTree(seg_xyz[row, :n_real])
        near = tree.query_ball_point(centers[sel], radius)
        probs_here = seg_probs[row, :n_real]
        for k, idx in enumerate(near):
            if not idx:
                unmatched += 1
                continue
            vals = probs_here[idx]
            if aggregation == "max":
                agg = float(vals.max())
            elif aggregation == "mean":
                agg = float(vals.mean())
            else:
                d, j = cKDTree(seg_xyz[row, :n_real][idx]).query(centers[sel][k])
                agg = float(vals[j])
            p_seg.append(agg)
            p_clf.append(float(clf_probs[sel][k]))
            y.append(int(clf_labels[sel][k]))

    if not y:
        return None
    return {"test_run": test_run, "labels": np.asarray(y, np.int8),
            "clf": np.asarray(p_clf), "seg": np.asarray(p_seg),
            "unmatched": unmatched, "matched": len(y)}


def _score(y: np.ndarray, p: np.ndarray) -> dict:
    """Threshold-free headline plus the best F1 this model could reach.

    The stored per-model threshold was tuned on a different population, so
    reusing it here would hand one model a threshold fitted for a different
    prevalence. Picking each model's own best F1 on the shared set treats both
    the same way.
    """
    out = M.summarize(y, p, M.best_f1_threshold(y, p))
    return {k: float(out[k]) for k in
            ("pr_auc", "roc_auc", "f1", "precision", "recall", "rock_frac",
             "baseline_pr_auc", "threshold")}


def compare(cache_dir: str, ablate_root: str, suite: str, clf_arm: str,
            seg_arm: str, test_runs: list[str], radius: float = DEFAULT_RADIUS_M,
            aggregation: str = "max") -> dict:
    """Paired per-fold comparison of one classifier arm against one segmenter arm."""
    folds, per_fold = [], []
    for run in test_runs:
        m = match_fold(cache_dir, ablate_root, suite, clf_arm, seg_arm, run,
                       radius, aggregation)
        if m is None:
            continue
        folds.append(run)
        per_fold.append({"test_run": run, "matched": m["matched"],
                         "unmatched": m["unmatched"],
                         "classifier": _score(m["labels"], m["clf"]),
                         "segmenter": _score(m["labels"], m["seg"])})

    deltas = {}
    for k in METRICS:
        if not per_fold:
            deltas[k] = None
            continue
        d = np.array([f["segmenter"][k] - f["classifier"][k] for f in per_fold])
        stat, p = wilcoxon_signed_rank(d)
        deltas[k] = {"mean_delta": float(d.mean()),
                     "median_delta": float(np.median(d)),
                     "std_delta": float(d.std(ddof=1)) if len(d) > 1 else None,
                     "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                     "statistic": stat, "p_value": p,
                     "per_fold": {f["test_run"]: float(x)
                                  for f, x in zip(per_fold, d)}}
    return {"suite": suite, "classifier_arm": clf_arm, "segmenter_arm": seg_arm,
            "radius_m": radius, "aggregation": aggregation,
            "folds": folds, "per_fold": per_fold, "deltas": deltas}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def default_pairs(suite: str) -> list[tuple[str, str]]:
    """(classifier arm, segmenter arm) for every contrast that spans both tasks.

    Read off the suite's own declared contrasts rather than hardcoded here, so
    a suite that gains a segmentation arm gains a matched comparison with it.
    """
    from .ablate import SUITES, arms_of
    from .models_meta import model_task

    task = {a.name: model_task(a.model) for a in arms_of(suite)}
    pairs = []
    for base, var, _q in SUITES[suite]["contrasts"]:
        if task.get(base) == "classify" and task.get(var) == "segment":
            pairs.append((base, var))
        elif task.get(base) == "segment" and task.get(var) == "classify":
            pairs.append((var, base))
    return pairs


def _labels_of(suite: str) -> dict[str, str]:
    from .ablate import arms_of
    return {a.name: a.label for a in arms_of(suite)}


def _fmt(x: float | None, nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _plot(result: dict, labels: dict[str, str], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = [f["test_run"] for f in result["per_fold"]]
    clf = [f["classifier"][PRIMARY] for f in result["per_fold"]]
    seg = [f["segmenter"][PRIMARY] for f in result["per_fold"]]
    base = [f["classifier"]["baseline_pr_auc"] for f in result["per_fold"]]
    x = np.arange(len(folds))

    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(folds)), 4.6))
    ax.bar(x - 0.2, clf, 0.38, label=labels.get(result["classifier_arm"],
                                                result["classifier_arm"]))
    ax.bar(x + 0.2, seg, 0.38, label=labels.get(result["segmenter_arm"],
                                                result["segmenter_arm"]))
    ax.plot(x, base, "k_", ms=18, label="no-skill baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("VolleyBall", "VB").replace(".reslam", "")
                        for f in folds], rotation=30, ha="right")
    ax.set_ylabel("PR-AUC at shared candidate centers")
    ax.set_title("Segmenter vs classifier, scored on the same centers\n"
                 f"({result['aggregation']} of segmenter points within "
                 f"{result['radius_m']:.2f} m)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def render_matched(cache_dir: str, ablate_root: str, suite: str, out_dir: str,
                   radius: float = DEFAULT_RADIUS_M, aggregation: str = "max",
                   pairs: list[tuple[str, str]] | None = None) -> dict:
    """Write the matched-population comparison as JSON, Markdown and figures."""
    from .data import load_cache_meta

    os.makedirs(out_dir, exist_ok=True)
    runs = sorted(load_cache_meta(cache_dir)["runs"])
    labels = _labels_of(suite)
    pairs = pairs or default_pairs(suite)
    if not pairs:
        raise SystemExit(f"suite {suite!r} has no classifier/segmenter contrast to match")

    results = []
    for clf_arm, seg_arm in pairs:
        r = compare(cache_dir, ablate_root, suite, clf_arm, seg_arm, runs,
                    radius, aggregation)
        if not r["folds"]:
            print(f"skip {clf_arm} vs {seg_arm}: no fold has both sides trained yet")
            continue
        results.append(r)
        _plot(r, labels, os.path.join(out_dir, f"matched_{clf_arm}__{seg_arm}.png"))

    out = {"suite": suite, "radius_m": radius, "aggregation": aggregation,
           "comparisons": results}
    with open(os.path.join(out_dir, "matched.json"), "w") as f:
        json.dump(out, f, indent=2)

    md = [f"# Segmentation vs sliding-window, scored the same way\n",
          "Both models are graded on **one shared set of candidate centers**, using ",
          "the centers' own rock/clear labels. The classifier scores each center ",
          "directly; the segmenter scores it by taking the ",
          f"**{aggregation}** of its per-point probabilities within ",
          f"**{radius:.2f} m** of that center.\n\n",
          "This matters because the two tasks are otherwise graded on different ",
          "populations - candidate balls are about 19% rock, individual points ",
          "about 1% - and PR-AUC moves with that prevalence, so the raw numbers in ",
          "the ablation report are not a like-for-like comparison. These are.\n\n"]

    for r in results:
        cl = labels.get(r["classifier_arm"], r["classifier_arm"])
        sl = labels.get(r["segmenter_arm"], r["segmenter_arm"])
        d = r["deltas"][PRIMARY]
        n_match = sum(f["matched"] for f in r["per_fold"])
        n_un = sum(f["unmatched"] for f in r["per_fold"])
        md.append(f"## {cl}  vs  {sl}\n\n")
        md.append(f"{len(r['folds'])} folds, {n_match} shared centers "
                  f"({n_un} centers had no segmented point nearby and were dropped "
                  f"from both sides).\n\n")
        if d:
            better = "the segmenter" if d["mean_delta"] > 0 else "the classifier"
            md.append(f"**{better} wins on average**: mean PR-AUC difference "
                      f"{d['mean_delta']:+.4f} (segmenter minus classifier), "
                      f"segmenter ahead on {d['wins']} of {len(r['folds'])} folds, "
                      f"signed-rank p = {d['p_value']:.3f}.\n\n")
        md.append("| held-out run | classifier PR-AUC | segmenter PR-AUC | difference | "
                  "no-skill | classifier F1 | segmenter F1 |\n")
        md.append("|---|---|---|---|---|---|---|\n")
        for f in r["per_fold"]:
            diff = f["segmenter"][PRIMARY] - f["classifier"][PRIMARY]
            md.append(f"| {f['test_run']} | {_fmt(f['classifier'][PRIMARY])} | "
                      f"{_fmt(f['segmenter'][PRIMARY])} | {diff:+.4f} | "
                      f"{_fmt(f['classifier']['baseline_pr_auc'], 3)} | "
                      f"{_fmt(f['classifier']['f1'])} | {_fmt(f['segmenter']['f1'])} |\n")
        cm = float(np.mean([f["classifier"][PRIMARY] for f in r["per_fold"]]))
        sm = float(np.mean([f["segmenter"][PRIMARY] for f in r["per_fold"]]))
        md.append(f"| **mean** | **{cm:.4f}** | **{sm:.4f}** | **{sm - cm:+.4f}** | | | |\n\n")

        md.append("Other metrics, as mean difference across folds "
                  "(positive = segmenter ahead):\n\n")
        md.append("| metric | mean difference | segmenter wins | p |\n|---|---|---|---|\n")
        for k in METRICS:
            dk = r["deltas"][k]
            if dk:
                md.append(f"| {k} | {dk['mean_delta']:+.4f} | "
                          f"{dk['wins']}/{len(r['folds'])} | {dk['p_value']:.3f} |\n")
        md.append("\n")

    path = os.path.join(out_dir, "matched.md")
    with open(path, "w") as f:
        f.write("".join(md))
    print(f"wrote {path}")
    return out
