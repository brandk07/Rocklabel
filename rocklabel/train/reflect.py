"""What does the reflectivity channel actually carry? Measured off the cache,
no training involved.

A trained network answering "reflectivity did not help" leaves the reason open:
the channel could be empty, or full but unusable, or usable but swamped by the
augmentation. This module goes at the data directly and separates those cases.

It scores each labeled neighborhood with a handful of scalar *measurements* of
its brightness — the plain average, but also several ways of reading brightness
*relatively*, which is the part that might survive a change of surface:

* contrast between the middle of the neighborhood and its outer ring (a rock
  sits in the middle, sand fills the ring),
* contrast between the high points and the low points (a rock is the tall
  thing), and
* how strongly brightness tracks height within one neighborhood.

Each measurement is scored with ROC-AUC per run: 0.5 means the measurement
cannot tell rock from clear at all, 1.0 means it separates them perfectly, and
below 0.5 means it separates them backwards. The same measurements are computed
on the height channel as a reference, because the honest question is never "is
there any signal" but "is there signal that shape does not already have".
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import metrics as M
from .plots import BASE, GRID, INK, MUTED, SERIES, _save

#: Neighborhood radius used to split "middle" from "outer ring". The generator
#: builds a 0.5 m ball; 0.15 m is about the radius of the rocks being labeled,
#: so the core is roughly "on the rock" for a positive sample.
CORE_R = 0.15
RING_R = 0.30

#: Every measurement, with the plain-English name the report prints. Order is
#: the order they appear in the tables and figures: brightness first, then the
#: shape measurements they are being compared against.
MEASUREMENTS: list[tuple[str, str, str]] = [
    ("i_mean", "brightness · average", "intensity"),
    ("i_std", "brightness · spread", "intensity"),
    ("i_max", "brightness · brightest point", "intensity"),
    ("i_min", "brightness · darkest point", "intensity"),
    ("i_p90_p10", "brightness · robust spread (90th - 10th pct)", "intensity"),
    ("i_core", "brightness · middle only", "intensity"),
    ("i_ring", "brightness · outer ring only", "intensity"),
    ("i_core_minus_ring", "brightness · middle minus ring", "intensity"),
    ("i_high_minus_low", "brightness · tall points minus low points", "intensity"),
    ("i_corr_z", "brightness · how well it tracks height", "intensity"),
    ("z_mean", "shape · average height", "geometry"),
    ("z_max", "shape · tallest point", "geometry"),
    ("z_std", "shape · height spread", "geometry"),
    ("z_core_minus_ring", "shape · middle minus ring height", "geometry"),
    ("n_real", "shape · how many points landed here", "geometry"),
]
KEYS = [k for k, _, _ in MEASUREMENTS]
LABELS = {k: label for k, label, _ in MEASUREMENTS}
KIND = {k: kind for k, _, kind in MEASUREMENTS}


def _weighted(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Row-wise mean of ``values`` over the rows ``w`` marks, NaN where empty."""
    n = w.sum(1)
    out = np.where(n > 0, (values * w).sum(1) / np.maximum(n, 1), np.nan)
    return out


def measure_run(cache_dir: str, run_id: str, chunk: int = 2000
                ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(labels, {measurement: value per sample}) for one cached run.

    Padded rows are excluded everywhere: they are duplicates of real points, so
    counting them would quietly weight dense neighborhoods' own points twice.
    """
    d = os.path.join(cache_dir, run_id)
    pts = np.load(os.path.join(d, "points.npy"), mmap_mode="r")
    labels = np.load(os.path.join(d, "labels.npy")).astype(np.int8)
    counts = np.load(os.path.join(d, "counts.npy")).astype(np.int64)
    n = pts.shape[1]
    cols: list[np.ndarray] = []

    for i in range(0, len(labels), chunk):
        p = np.asarray(pts[i:i + chunk], np.float32)
        c = np.minimum(counts[i:i + chunk], n)
        valid = np.arange(n)[None, :] < c[:, None]
        w = valid.astype(np.float32)
        inten, z = p[..., 3], p[..., 2]
        r = np.hypot(p[..., 0], p[..., 1])

        i_mean = _weighted(inten, w)
        i_var = _weighted((inten - i_mean[:, None]) ** 2, w)
        z_mean = _weighted(z, w)
        z_var = _weighted((z - z_mean[:, None]) ** 2, w)
        # -inf/+inf on the padded rows so they lose every max and min.
        i_max = np.where(valid, inten, -np.inf).max(1)
        i_min = np.where(valid, inten, np.inf).min(1)
        z_max = np.where(valid, z, -np.inf).max(1)

        # Percentiles over ragged rows: push the padding to +inf, sort, then
        # index each row at its own percentile position.
        srt = np.sort(np.where(valid, inten, np.inf), axis=1)
        idx10 = np.clip((0.10 * (c - 1)).astype(int), 0, n - 1)
        idx90 = np.clip((0.90 * (c - 1)).astype(int), 0, n - 1)
        rows = np.arange(len(c))
        i_p90_p10 = srt[rows, idx90] - srt[rows, idx10]

        core = (valid & (r < CORE_R)).astype(np.float32)
        ring = (valid & (r > RING_R)).astype(np.float32)
        i_core, i_ring = _weighted(inten, core), _weighted(inten, ring)
        z_core, z_ring = _weighted(z, core), _weighted(z, ring)

        # "Tall" and "low" split at the neighborhood's own median height, so
        # the measurement is about brightness-versus-height inside one sample
        # and never about how high the sample sits in the world.
        zs = np.sort(np.where(valid, z, np.inf), axis=1)
        z_med = zs[rows, np.clip((0.5 * (c - 1)).astype(int), 0, n - 1)][:, None]
        high = (valid & (z >= z_med)).astype(np.float32)
        low = (valid & (z < z_med)).astype(np.float32)
        i_high_minus_low = _weighted(inten, high) - _weighted(inten, low)

        # Correlation of brightness with height inside each neighborhood.
        cov = _weighted((inten - i_mean[:, None]) * (z - z_mean[:, None]), w)
        denom = np.sqrt(np.maximum(i_var, 0) * np.maximum(z_var, 0))
        i_corr_z = np.where(denom > 1e-12, cov / np.maximum(denom, 1e-12), 0.0)

        cols.append(np.column_stack([
            i_mean, np.sqrt(np.maximum(i_var, 0)), i_max, i_min, i_p90_p10,
            i_core, i_ring, i_core - i_ring, i_high_minus_low, i_corr_z,
            z_mean, z_max, np.sqrt(np.maximum(z_var, 0)), z_core - z_ring,
            c.astype(np.float32),
        ]))

    A = np.concatenate(cols)
    return labels, {k: A[:, j] for j, k in enumerate(KEYS)}


def _auc(y: np.ndarray, v: np.ndarray) -> float:
    """ROC-AUC, ignoring samples where the measurement is undefined."""
    good = np.isfinite(v)
    if good.sum() < 2 or len(np.unique(y[good])) < 2:
        return float("nan")
    return M.roc_auc(y[good], v[good])


# --------------------------------------------------------------------------- #
# A cheap "does brightness add anything on top of shape" check
# --------------------------------------------------------------------------- #
def _standardize(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.where(sd > 1e-9, sd, 1.0)


def logistic_fit(X: np.ndarray, y: np.ndarray, iters: int = 400,
                 lr: float = 0.5, l2: float = 1e-3) -> np.ndarray:
    """Plain gradient-descent logistic regression, positives up-weighted.

    Deliberately small: this is a sanity check on the hand-computed
    measurements, not a model anyone deploys. Rewriting it here rather than
    adding scikit-learn keeps the training extra to torch, numpy and matplotlib.
    """
    X = np.column_stack([_standardize(X), np.ones(len(X))])
    w = np.zeros(X.shape[1])
    pos = y == 1
    # Class weights, so a 5%-rock run does not train a predict-nothing model.
    cw = np.where(pos, 0.5 / max(pos.mean(), 1e-6), 0.5 / max(1 - pos.mean(), 1e-6))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (cw * (p - y)) / len(y) + l2 * w
        w -= lr * grad
    return w


def logistic_predict(X: np.ndarray, w: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu, sd = ref.mean(0), ref.std(0)
    Xs = np.column_stack([(X - mu) / np.where(sd > 1e-9, sd, 1.0), np.ones(len(X))])
    return 1.0 / (1.0 + np.exp(-np.clip(Xs @ w, -30, 30)))


def leave_one_run_out_probe(per_run: dict, keys: list[str]) -> dict:
    """Fit on every run but one, score the one, for one set of measurements."""
    runs = sorted(per_run)
    out = {}
    for held in runs:
        tr = [r for r in runs if r != held]
        Xtr = np.concatenate([np.column_stack([per_run[r][1][k] for k in keys]) for r in tr])
        ytr = np.concatenate([per_run[r][0] for r in tr]).astype(float)
        good = np.isfinite(Xtr).all(1)
        Xtr, ytr = Xtr[good], ytr[good]
        w = logistic_fit(Xtr, ytr)
        Xte = np.column_stack([per_run[held][1][k] for k in keys])
        yte = per_run[held][0].astype(float)
        ok = np.isfinite(Xte).all(1)
        probs = logistic_predict(Xte[ok], w, Xtr)
        out[held] = {"pr_auc": M.average_precision(yte[ok].astype(np.int8), probs),
                     "roc_auc": M.roc_auc(yte[ok].astype(np.int8), probs)}
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _fig_auc_bars(table: dict, runs: list[str], out_path: str) -> None:
    import matplotlib.pyplot as plt

    keys = list(KEYS)
    means = [np.nanmean([table[r][k] for r in runs]) for k in keys]
    fig, ax = plt.subplots(figsize=(9, 6))
    ypos = np.arange(len(keys))[::-1]
    colors = [SERIES[3] if KIND[k] == "intensity" else SERIES[1] for k in keys]
    ax.barh(ypos, [m - 0.5 for m in means], left=0.5, color=colors, height=0.7)
    for r in runs:
        ax.scatter([table[r][k] for k in keys], ypos, s=9, color=INK, alpha=0.25,
                   zorder=3, linewidths=0)
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.set_yticks(ypos, [LABELS[k] for k in keys], fontsize=9)
    ax.set_xlim(0.25, 1.0)
    ax.set_xlabel("ability to tell rock from clear (0.5 = coin flip, 1.0 = perfect)")
    ax.set_title("What each single measurement is worth\n"
                 "bars are the average over runs, dots are the individual runs",
                 loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, out_path)


def _fig_intensity_hist(per_run: dict, out_path: str) -> None:
    import matplotlib.pyplot as plt

    y = np.concatenate([per_run[r][0] for r in sorted(per_run)])
    v = np.concatenate([per_run[r][1]["i_mean"] for r in sorted(per_run)])
    good = np.isfinite(v)
    y, v = y[good], v[good]
    lo, hi = np.percentile(v, [0.5, 99.5])
    bins = np.linspace(lo, hi, 80)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(v[y == 0], bins=bins, density=True, color=BASE, label="clear (sand)")
    ax.hist(v[y == 1], bins=bins, density=True, histtype="step", lw=2.0,
            color=SERIES[3], label="rock")
    ax.set_xlabel("average brightness of the neighborhood")
    ax.set_ylabel("share of samples")
    ax.set_title(f"Rock and sand brightness, all runs pooled\n"
                 f"rock average {v[y == 1].mean():.4f}   sand average {v[y == 0].mean():.4f}"
                 f"   gap {v[y == 1].mean() - v[y == 0].mean():+.4f}", loc="left")
    ax.legend()
    _save(fig, out_path)


def _fig_run_drift(per_run: dict, out_path: str) -> None:
    import matplotlib.pyplot as plt

    runs = sorted(per_run)
    rock = [np.nanmean(per_run[r][1]["i_mean"][per_run[r][0] == 1]) for r in runs]
    clear = [np.nanmean(per_run[r][1]["i_mean"][per_run[r][0] == 0]) for r in runs]
    x = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(x, clear, "o-", color=BASE, lw=2, label="sand")
    ax.plot(x, rock, "o-", color=SERIES[3], lw=2, label="rock")
    ax.set_xticks(x, [r.replace(".reslam", "").replace("VolleyBall", "") for r in runs],
                  rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("average brightness")
    spread = float(np.ptp(clear))
    gap = float(np.mean(np.array(rock) - np.array(clear)))
    ax.set_title("Brightness level, run by run\n"
                 f"the sand level moves {spread:.4f} between runs; "
                 f"rock sits {gap:+.4f} from sand within a run", loc="left")
    ax.legend()
    _save(fig, out_path)


def _fig_probe(probe: dict, out_path: str) -> None:
    import matplotlib.pyplot as plt

    runs = sorted(probe["shape only"])
    names = ["shape only", "brightness only", "shape + brightness"]
    colors = [SERIES[1], SERIES[3], SERIES[0]]
    x = np.arange(len(runs))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [probe[name][r]["pr_auc"] for r in runs]
        ax.bar(x + (i - 1) * width, vals, width, color=color,
               label=f"{name} (avg {np.mean(vals):.3f})")
    ax.set_xticks(x, [r.replace(".reslam", "").replace("VolleyBall", "") for r in runs],
                  rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("PR-AUC on the held-out run")
    ax.set_title("Simple formula on the hand-made measurements, one run held out each time\n"
                 "if brightness carried anything extra, the blue bars would beat the green",
                 loc="left")
    ax.legend()
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render_reflectivity(cache_dir: str, out_dir: str) -> dict:
    from .data import load_cache_meta

    meta = load_cache_meta(cache_dir)
    runs = sorted(meta["runs"])
    print(f"measuring the reflectivity channel across {len(runs)} runs")
    per_run = {}
    for r in runs:
        y, s = measure_run(cache_dir, r)
        per_run[r] = (y, s)
        print(f"  {r}: {len(y)} samples, {int((y == 1).sum())} rock")

    table = {r: {k: _auc(per_run[r][0], per_run[r][1][k]) for k in KEYS} for r in runs}

    levels = {}
    for r in runs:
        y, s = per_run[r]
        v = s["i_mean"]
        levels[r] = {
            "rock_mean": float(np.nanmean(v[y == 1])),
            "clear_mean": float(np.nanmean(v[y == 0])),
            "gap": float(np.nanmean(v[y == 1]) - np.nanmean(v[y == 0])),
            "clear_std": float(np.nanstd(v[y == 0])),
        }
    clear_levels = np.array([levels[r]["clear_mean"] for r in runs])

    geom_keys = [k for k in KEYS if KIND[k] == "geometry"]
    int_keys = [k for k in KEYS if KIND[k] == "intensity"]
    print("running the leave-one-run-out formula probe")
    probe = {
        "shape only": leave_one_run_out_probe(per_run, geom_keys),
        "brightness only": leave_one_run_out_probe(per_run, int_keys),
        "shape + brightness": leave_one_run_out_probe(per_run, geom_keys + int_keys),
    }

    os.makedirs(out_dir, exist_ok=True)
    _fig_auc_bars(table, runs, os.path.join(out_dir, "measurement_power.png"))
    _fig_intensity_hist(per_run, os.path.join(out_dir, "brightness_histogram.png"))
    _fig_run_drift(per_run, os.path.join(out_dir, "brightness_drift.png"))
    _fig_probe(probe, os.path.join(out_dir, "formula_probe.png"))

    summary = {
        "cache_dir": cache_dir, "runs": runs,
        "measurements": [{"key": k, "label": LABELS[k], "kind": KIND[k]} for k in KEYS],
        "auc_per_run": table,
        "auc_mean": {k: float(np.nanmean([table[r][k] for r in runs])) for k in KEYS},
        "levels": levels,
        "clear_level_spread": float(np.ptp(clear_levels)),
        "clear_level_std": float(np.std(clear_levels)),
        "mean_within_run_gap": float(np.mean([levels[r]["gap"] for r in runs])),
        "probe": probe,
        "probe_mean": {name: {m: float(np.mean([v[m] for v in d.values()]))
                              for m in ("pr_auc", "roc_auc")}
                       for name, d in probe.items()},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_markdown(summary, os.path.join(out_dir, "summary.md"))
    return summary


def _write_markdown(s: dict, path: str) -> None:
    runs = s["runs"]
    L = ["# What the reflectivity channel carries", "",
         f"Measured on `{s['cache_dir']}` — {len(runs)} runs, no training involved.",
         "",
         "ROC-AUC below is the ability of one measurement, on its own, to tell a "
         "rock neighborhood from a clear one. 0.5 is a coin flip; 1.0 is perfect; "
         "below 0.5 means it separates them backwards.", "",
         "## Each measurement on its own", "",
         "| measurement | average | " + " | ".join(
             r.replace(".reslam", "").replace("VolleyBall", "") for r in runs) + " |",
         "|---|---|" + "---|" * len(runs)]
    for k in KEYS:
        L.append(f"| {LABELS[k]} | **{s['auc_mean'][k]:.3f}** | "
                 + " | ".join(f"{s['auc_per_run'][r][k]:.3f}" for r in runs) + " |")
    L += ["", "## Absolute brightness level, run by run", "",
          "| run | rock | sand | gap |", "|---|---|---|---|"]
    for r in runs:
        v = s["levels"][r]
        L.append(f"| {r} | {v['rock_mean']:.4f} | {v['clear_mean']:.4f} | {v['gap']:+.4f} |")
    L += ["",
          f"The sand level moves **{s['clear_level_spread']:.4f}** between runs, while "
          f"rock sits **{s['mean_within_run_gap']:+.4f}** from sand inside one run. "
          "When the second number is smaller than the first, an absolute brightness "
          "threshold cannot be carried from one run to the next.", "",
          "## A simple formula on these measurements, one run held out at a time", "",
          "| inputs | PR-AUC | ROC-AUC |", "|---|---|---|"]
    for name, v in s["probe_mean"].items():
        L.append(f"| {name} | {v['pr_auc']:.3f} | {v['roc_auc']:.3f} |")
    L += ["", "![](measurement_power.png)", "", "![](brightness_histogram.png)", "",
          "![](brightness_drift.png)", "", "![](formula_probe.png)", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {path}")
