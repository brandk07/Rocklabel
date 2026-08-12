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
        thresholds = np.linspace(0.02, 0.98, 49)
    rows = [confusion(labels, probs, t) for t in thresholds]
    return {k: np.array([r[k] for r in rows]) for k in
            ("threshold", "precision", "recall", "f1", "accuracy")}


def best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    sweep = threshold_sweep(labels, probs)
    return float(sweep["threshold"][int(np.argmax(sweep["f1"]))])


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
    return out
