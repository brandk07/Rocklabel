"""Binary classification metrics in plain numpy (no sklearn dependency).

Accuracy is deliberately not the headline anywhere: the datasets are ~29%
rock, so 71% accuracy is free. Every summary carries the majority-class
baseline so 'did it learn anything' is answerable at a glance.
"""

from __future__ import annotations

import numpy as np


def roc_curve(labels: np.ndarray, scores: np.ndarray):
    """(fpr, tpr, thresholds), thresholds descending."""
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(np.float64)
    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    # keep only the last index of each distinct score (step points)
    distinct = np.r_[np.diff(scores[order]) != 0, True]
    tps, fps, thr = tps[distinct], fps[distinct], scores[order][distinct]
    p, n = max(y.sum(), 1e-12), max(len(y) - y.sum(), 1e-12)
    return np.r_[0.0, fps] / n, np.r_[0.0, tps] / p, np.r_[np.inf, thr]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.trapezoid(tpr, fpr))


def pr_curve(labels: np.ndarray, scores: np.ndarray):
    """(precision, recall, thresholds), recall ascending order reversed to match sweep."""
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(np.float64)
    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    distinct = np.r_[np.diff(scores[order]) != 0, True]
    tps, fps, thr = tps[distinct], fps[distinct], scores[order][distinct]
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / max(y.sum(), 1e-12)
    return np.r_[1.0, precision], np.r_[0.0, recall], np.r_[np.inf, thr]


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC as step-integrated average precision (sklearn's definition)."""
    precision, recall, _ = pr_curve(labels, scores)
    return float(np.sum(np.diff(recall) * precision[1:]))


def confusion(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = probs >= threshold
    pos = labels == 1
    tp = int((pred & pos).sum())
    fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & ~pos).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / max(len(labels), 1)}


def threshold_sweep(labels: np.ndarray, probs: np.ndarray,
                    thresholds: np.ndarray | None = None) -> dict:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = [confusion(labels, probs, t) for t in thresholds]
    return {k: np.array([r[k] for r in rows]) for k in
            ("threshold", "precision", "recall", "f1", "accuracy")}


def best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Maximize validation F1 over all distinct observed probability scores.

    A fixed .01..99 grid cannot resolve confident models or scores clustered
    near zero. Keep tied scores together; prefer higher recall on an F1 tie.
    This chooses an operating point, not a probability calibration.
    """
    labels, probs = np.asarray(labels), np.asarray(probs)
    if labels.ndim != 1 or probs.shape != labels.shape or not len(labels):
        raise ValueError("threshold selection needs aligned, nonempty 1D arrays")
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("threshold selection needs finite probabilities in [0, 1]")
    if not np.isin(labels, [0, 1]).all() or not np.any(labels == 1):
        raise ValueError("threshold selection needs binary labels and at least one rock")
    precision, recall, thresholds = pr_curve(labels, probs)
    f1 = 2 * precision[1:] * recall[1:] / np.maximum(precision[1:] + recall[1:], 1e-12)
    # Thresholds descend; the last maximizing point has the highest recall.
    i = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[1:][i])


def summarize(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Everything the reports need, including the do-nothing baseline."""
    rock_frac = float((labels == 1).mean()) if len(labels) else 0.0
    out = confusion(labels, probs, threshold)
    out.update({
        "n": int(len(labels)),
        "rock_frac": rock_frac,
        "roc_auc": roc_auc(labels, probs),
        "pr_auc": average_precision(labels, probs),
        # predict-all-clear baseline: its accuracy is the bar to beat, and a
        # no-skill classifier's PR-AUC equals the positive prevalence
        "baseline_accuracy": max(rock_frac, 1.0 - rock_frac),
        "baseline_pr_auc": rock_frac,
    })
    out["norm_pr_auc"] = normalized_pr_auc(out["pr_auc"], rock_frac)
    return out


def normalized_pr_auc(pr_auc: float, prevalence: float) -> float:
    """PR-AUC rescaled so guessing scores 0 and a perfect model scores 1.

    ``(AP - prevalence) / (1 - prevalence)``. Raw PR-AUC starts at the positive
    rate, so a run with many rocks looks better than a run with few even when
    both models are equally good. Rock share spans 6.3%-31.6% across the eleven
    volleyball recordings, a 5x spread, which is enough that a raw per-fold
    table partly measures how many rocks a recording has.

    Measured, the reordering is mild - on full-sweep data it swaps three
    adjacent pairs of folds and leaves both ends of the table alone, and for the
    segmenter (every fold ~1% rock) it changes nothing at all. It is reported
    because it is honest and free, not because it overturns rankings.

    Only comparable within one population: a segmenter's per-point score and a
    classifier's per-ball score are still two different measurements after
    normalizing, because normalizing fixes the floor, not the unit. Use
    ``rocklabel-train matched`` for that comparison.
    """
    if prevalence >= 1.0:
        return 0.0
    return float((pr_auc - prevalence) / (1.0 - prevalence))
