"""Publication-style PNGs regenerated from run artifacts (history.csv,
predictions.npz, test_metrics.json) - nothing here needs a retrain.

Colors follow the validated reference palette: categorical slots in fixed
order (PointNet=blue, PointNet++=green; fold accents use slots 1-4), a single
blue sequential ramp for the confusion matrices, recessive gray chrome.
"""

from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics as M

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]  # slots 1-4, fixed order
MODEL_COLOR = {"pointnet": SERIES[0], "pointnet2": SERIES[1],
               "pointnet2_seg": SERIES[2]}
#: Short names for figures. The task matters more than the backbone here, so
#: every label says which of the two approaches it is: a classifier scored per
#: candidate center, or a segmenter scored per point.
MODEL_LABEL = {"pointnet": "PointNet (classifier)",
               "pointnet2": "PointNet++ (classifier)",
               "pointnet2_seg": "PointNet++ (segmentation)"}
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 150,
})


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def _read_history(run_dir: str) -> dict[str, np.ndarray]:
    with open(os.path.join(run_dir, "history.csv")) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _read_preds(run_dir: str):
    """(labels, probs, threshold) as flat 1-D arrays, whatever the task.

    A segmenter stores [frames, points] plus a per-frame count; flattening to
    the points it was actually scored on (real, non-boundary-shell) is what
    makes every curve and confusion matrix below task-agnostic.
    """
    z = np.load(os.path.join(run_dir, "predictions.npz"), allow_pickle=True)
    labels, probs = z["labels"], z["probs"]
    if labels.ndim == 2:
        n = labels.shape[1]
        keep = (np.arange(n)[None, :] < z["counts"][:, None]) & (labels >= 0)
        labels, probs = labels[keep], probs[keep]
    return labels, probs, float(z["threshold"])


def plot_history(run_dir: str, out_path: str | None = None) -> None:
    """Loss + validation metric curves for one training run."""
    h = _read_history(run_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(h["epoch"], h["train_loss"], color=SERIES[0], lw=2, label="train")
    ax1.plot(h["epoch"], h["val_loss"], color=SERIES[1], lw=2, label="validation")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(h["epoch"], h["val_pr_auc"], color=SERIES[0], lw=2, label="PR-AUC")
    ax2.plot(h["epoch"], h["val_roc_auc"], color=SERIES[1], lw=2, label="ROC-AUC")
    ax2.plot(h["epoch"], h["val_f1_at_0.5"], color=SERIES[2], lw=2, label="F1 @ 0.5")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("validation metric"); ax2.set_ylim(0, 1.02)
    ax2.set_title("Validation metrics"); ax2.legend(loc="lower right")
    fig.suptitle(os.path.basename(run_dir), y=1.0, fontsize=10, color=MUTED)
    _save(fig, out_path or os.path.join(run_dir, "curves.png"))


def plot_roc_pr(fold_dirs: dict[str, str], fold_name: str, out_path: str) -> None:
    """ROC + PR curves overlaying both models for one held-out run."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    prevalence = None
    for model, run_dir in fold_dirs.items():
        labels, probs, _ = _read_preds(run_dir)
        prevalence = float((labels == 1).mean())
        color = MODEL_COLOR[model]
        fpr, tpr, _ = M.roc_curve(labels, probs)
        ax1.plot(fpr, tpr, color=color, lw=2,
                 label=f"{MODEL_LABEL[model]}  (AUC {M.roc_auc(labels, probs):.3f})")
        p, r, _ = M.pr_curve(labels, probs)
        ax2.plot(r, p, color=color, lw=2,
                 label=f"{MODEL_LABEL[model]}  (AP {M.average_precision(labels, probs):.3f})")
    ax1.plot([0, 1], [0, 1], color=BASE, lw=1, ls="--")
    ax1.set_xlabel("false positive rate"); ax1.set_ylabel("true positive rate")
    ax1.set_title(f"ROC - held-out {fold_name}"); ax1.legend(loc="lower right")
    if prevalence is not None:  # no-skill PR baseline = rock prevalence
        ax2.axhline(prevalence, color=BASE, lw=1, ls="--")
        ax2.annotate(f"no skill ({prevalence:.2f})", (0.02, prevalence + 0.015),
                     color=MUTED, fontsize=8)
    ax2.set_xlabel("recall"); ax2.set_ylabel("precision"); ax2.set_ylim(0, 1.02)
    ax2.set_title(f"Precision-recall - held-out {fold_name}"); ax2.legend(loc="lower left")
    _save(fig, out_path)


def plot_confusions(fold_dirs: dict[str, str], fold_name: str, out_path: str) -> None:
    """Both models' confusion matrices at their val-chosen thresholds."""
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    fig, axes = plt.subplots(1, len(fold_dirs), figsize=(4.2 * len(fold_dirs), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (model, run_dir) in zip(axes, fold_dirs.items()):
        labels, probs, thr = _read_preds(run_dir)
        c = M.confusion(labels, probs, thr)
        mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], float)
        shade = mat / mat.max()
        ax.imshow(shade, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ink = "white" if shade[i, j] > 0.55 else INK
                name = [["TN", "FP"], ["FN", "TP"]][i][j]
                frac = mat[i, j] / mat.sum()
                ax.text(j, i, f"{name}\n{int(mat[i, j]):,}\n{frac:.1%}",
                        ha="center", va="center", color=ink, fontsize=10)
        ax.set_xticks([0, 1], ["pred clear", "pred rock"])
        ax.set_yticks([0, 1], ["true clear", "true rock"])
        ax.set_title(f"{MODEL_LABEL[model]} @ thr {thr:.2f}\n"
                     f"P {c['precision']:.3f}  R {c['recall']:.3f}  F1 {c['f1']:.3f}")
        ax.grid(False)
    fig.suptitle(f"held-out {fold_name}", fontsize=10, color=MUTED)
    _save(fig, out_path)


def plot_threshold_sweep(fold_dirs: dict[str, str], fold_name: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, len(fold_dirs), figsize=(4.5 * len(fold_dirs), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (model, run_dir) in zip(axes, fold_dirs.items()):
        labels, probs, thr = _read_preds(run_dir)
        s = M.threshold_sweep(labels, probs)
        ax.plot(s["threshold"], s["precision"], color=SERIES[0], lw=2, label="precision")
        ax.plot(s["threshold"], s["recall"], color=SERIES[1], lw=2, label="recall")
        ax.plot(s["threshold"], s["f1"], color=SERIES[2], lw=2, label="F1")
        ax.axvline(thr, color=BASE, lw=1, ls="--")
        ax.annotate(f"val-chosen {thr:.2f}", (thr + 0.02, 0.05), color=MUTED, fontsize=8)
        ax.set_xlabel("decision threshold")
        ax.set_title(MODEL_LABEL[model])
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("metric on held-out run")
    axes[-1].legend(loc="lower left")
    fig.suptitle(f"held-out {fold_name}", fontsize=10, color=MUTED)
    _save(fig, out_path)


def plot_comparison(results: dict[str, dict[str, dict]], out_path: str) -> None:
    """Grouped bars per held-out run + mean±std, for the headline metrics.

    results[model][test_run] = test_metrics dict. This doubles as the per-run
    breakdown: each LORO fold's test set IS one run/room.
    """
    metrics = ["pr_auc", "roc_auc", "f1", "recall"]
    titles = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
              "f1": "F1 @ val threshold", "recall": "Recall @ val threshold"}
    models = list(results)
    runs = sorted(next(iter(results.values())))
    # Width tracks the fold count: this figure was laid out for four myroom
    # folds and the labels collided into an unreadable smear at eight.
    fig, axes = plt.subplots(2, 2, figsize=(max(10.0, 1.3 * (len(runs) + 1) + 3.0), 7.0),
                             sharex=True)
    width = 0.36
    x = np.arange(len(runs) + 1)  # +1 slot for mean±std
    for ax, metric in zip(axes.flat, metrics):
        for k, model in enumerate(models):
            vals = np.array([results[model][r][metric] for r in runs])
            off = (k - (len(models) - 1) / 2) * width
            bars = ax.bar(x[:-1] + off, vals, width * 0.94, color=MODEL_COLOR[model],
                          label=MODEL_LABEL[model])
            ax.bar_label(bars, fmt="%.2f", fontsize=6, color=MUTED, padding=1)
            # ddof=1: this is a sample of folds standing in for the spread over
            # runs, and it is printed as an error bar - the population form
            # would quietly understate it.
            mb = ax.bar(x[-1] + off, vals.mean(), width * 0.94, color=MODEL_COLOR[model],
                        yerr=vals.std(ddof=1), error_kw={"ecolor": INK, "capsize": 3, "lw": 1})
            ax.bar_label(mb, fmt="%.2f", fontsize=6, color=MUTED, padding=1)
        if metric == "pr_auc":
            # no-skill PR-AUC = prevalence of the held-out run
            prev = [results[models[0]][r]["rock_frac"] for r in runs]
            ax.scatter(x[:-1], prev, marker="_", s=220, color=INK, zorder=3,
                       label="no-skill baseline")
        ax.set_title(titles[metric])
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x, [r.replace("loro_", "") for r in runs] + ["mean±std"],
                      rotation=30, ha="right", fontsize=8)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper right", ncols=3, fontsize=8,
               bbox_to_anchor=(0.99, 0.965))  # figure-level: bars fill every panel
    names = ", ".join(MODEL_LABEL.get(m, m) for m in models)
    fig.suptitle(f"Leave-one-run-out: {names} "
                 f"({len(runs)} folds, each testing one unseen recording)",
                 x=0.02, ha="left")
    _save(fig, out_path)


def write_summary_table(results: dict[str, dict[str, dict]], out_path: str) -> None:
    """Markdown summary with per-fold rows and mean±std, plus baselines."""
    models = list(results)
    runs = sorted(next(iter(results.values())))
    cols = ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"]
    lines = ["| model | held-out run | " + " | ".join(c.upper().replace("_", "-") for c in cols)
             + " | baseline acc | no-skill PR-AUC |",
             "|" + "---|" * (len(cols) + 4)]
    for model in models:
        vals = {c: [] for c in cols}
        for r in runs:
            m = results[model][r]
            for c in cols:
                vals[c].append(m[c])
            lines.append(f"| {MODEL_LABEL[model]} | {r} | "
                         + " | ".join(f"{m[c]:.3f}" for c in cols)
                         + f" | {m['baseline_accuracy']:.3f} | {m['rock_frac']:.3f} |")
        lines.append(f"| **{MODEL_LABEL[model]} mean±std** | all | "
                     + " | ".join(f"{np.mean(v):.3f}±{np.std(v, ddof=1):.3f}"
                                  for v in vals.values())
                     + " | | |")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def render_all(runs_root: str, results_dir: str, models: list[str], folds: list[str],
               features: list[str] | None = None) -> dict:
    """Regenerate every figure from artifacts on disk; returns pooled results.

    ``features`` selects *which* runs to report on: a non-default input-channel
    selection is tagged into the run directory name, so reporting has to look
    under the same tag the training wrote.
    """
    from .data import run_dir_name

    results: dict[str, dict[str, dict]] = {}
    for model in models:
        results[model] = {}
        for fold in folds:
            run_dir = os.path.join(runs_root, run_dir_name(model, fold, features))
            with open(os.path.join(run_dir, "test_metrics.json")) as f:
                results[model][fold] = json.load(f)
            plot_history(run_dir)
    for fold in folds:
        dirs = {m: os.path.join(runs_root, run_dir_name(m, fold, features)) for m in models}
        plot_roc_pr(dirs, fold, os.path.join(results_dir, f"roc_pr_{fold}.png"))
        plot_confusions(dirs, fold, os.path.join(results_dir, f"confusion_{fold}.png"))
        plot_threshold_sweep(dirs, fold, os.path.join(results_dir, f"threshold_{fold}.png"))
    plot_comparison(results, os.path.join(results_dir, "comparison.png"))
    write_summary_table(results, os.path.join(results_dir, "summary.md"))
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results
