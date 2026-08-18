"""Figures and tables for an ablation suite. Reads run artifacts only — no
torch, no retraining, safe to run while the sweep is still going.

Everything here is paired. A bar chart of arm averages is the one figure that
could mislead on this data: fold-to-fold spread is an order of magnitude larger
than any channel effect, so two arms' averages can differ by more than the
effect purely through which folds happened to finish. Every comparison is
therefore drawn and tested fold by fold, and the seed-repeat contrasts are
printed alongside as the yardstick for what a difference of nothing looks like.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .ablate import METRICS, PRIMARY, SUITES, collect
from .plots import BASE, GRID, INK, MUTED, SERIES, _save

#: Contrasts whose baseline and variant are the same setting under two seeds.
#: Their spread is the noise floor every other contrast is judged against.
def _is_seed_repeat(entry: dict) -> bool:
    return entry["question"].lower().startswith("noise floor")


def _short(run: str) -> str:
    return run.replace(".reslam", "").replace("VolleyBall", "")


def _arm_color(i: int) -> str:
    return SERIES[i % len(SERIES)]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_arm_ranking(data: dict, out_path: str) -> None:
    import matplotlib.pyplot as plt

    arms = [a for a in data["arms"] if a["folds_done"]]
    if not arms:
        return
    order = sorted(arms, key=lambda a: a[PRIMARY]["mean"])
    y = np.arange(len(order))
    # The axis has to cover every fold, not just the averages: the spread of
    # the dots is the figure's actual message, and clipping it would hide the
    # one thing worth seeing here.
    high = max(max(a[PRIMARY]["per_fold"].values()) for a in order)
    limit = min(1.0, high * 1.05)
    fig, ax = plt.subplots(figsize=(10, 0.52 * len(order) + 2.4))
    for i, a in enumerate(order):
        color = SERIES[1] if "geom" in a["arm"] else SERIES[0]
        if "refl-only" in a["arm"]:
            color = SERIES[3]
        ax.barh(i, a[PRIMARY]["mean"], color=color, height=0.62)
        vals = list(a[PRIMARY]["per_fold"].values())
        ax.scatter(vals, np.full(len(vals), i), s=14, color=INK, alpha=0.45,
                   zorder=3, linewidths=0)
    ax.set_yticks(y, [f"{a['label']}  ({a['folds_done']} folds)" for a in order],
                  fontsize=9)
    ax.set_xlabel("PR-AUC on the held-out run (higher is better)")
    ax.set_xlim(0, limit)
    # Averages in their own column past the plot, so the number never sits on
    # top of the dots it summarizes.
    for i, a in enumerate(order):
        ax.text(1.012, i, f"{a[PRIMARY]['mean']:.3f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=9, color=INK)
    ax.text(1.012, len(order) - 0.35, "average", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=8, color=MUTED)
    ax.set_title("Every setting, averaged over the folds that finished\n"
                 "dots are the individual folds — note how far they spread",
                 loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    _save(fig, out_path)


def fig_paired_deltas(data: dict, out_path: str) -> None:
    """One row per contrast: the per-fold difference, with zero marked."""
    import matplotlib.pyplot as plt

    entries = [c for c in data["contrasts"] if c["n_folds"] and c[PRIMARY]]
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(entries) + 2.4))
    for i, c in enumerate(entries[::-1]):
        d = np.array(list(c[PRIMARY]["per_fold"].values()))
        seed = _is_seed_repeat(c)
        color = MUTED if seed else (SERIES[1] if d.mean() > 0 else SERIES[2])
        ax.scatter(d, np.full(len(d), i), s=34, color=color, alpha=0.75,
                   zorder=3, linewidths=0)
        ax.plot([d.mean(), d.mean()], [i - 0.3, i + 0.3], color=INK, lw=2.2, zorder=4)
        ax.text(0.0, i + 0.42, "", fontsize=8)
    ax.axvline(0.0, color=INK, lw=1.2)
    labels = [f"{c['variant']}\n  vs {c['baseline']}   (p={c[PRIMARY]['p_value']:.3f}, "
              f"{c[PRIMARY]['wins']}W/{c[PRIMARY]['losses']}L)"
              for c in entries[::-1]]
    ax.set_yticks(np.arange(len(entries)), labels, fontsize=8)
    ax.set_xlabel("change in PR-AUC on the held-out run "
                  "(right of the line = the variant won that fold)")
    ax.set_title("Every comparison, fold by fold\n"
                 "black bar is the average. Grey rows are one setting under two\n"
                 "seeds — that spread is what a difference of nothing looks like.",
                 loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    _save(fig, out_path)


def fig_per_fold_lines(data: dict, out_path: str, arms: list[str] | None = None) -> None:
    """PR-AUC per held-out run for the headline arms, so fold difficulty shows."""
    import matplotlib.pyplot as plt

    wanted = arms or ["pointnet-geom", "pointnet-refl", "pointnet2-geom", "pointnet2-refl"]
    rows = [a for a in data["arms"] if a["arm"] in wanted and a["folds_done"]]
    if not rows:
        return
    folds = sorted({f for a in rows for f in a[PRIMARY]["per_fold"]})
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, a in enumerate(rows):
        vals = [a[PRIMARY]["per_fold"].get(f, np.nan) for f in folds]
        ax.plot(x, vals, "o-", lw=2, color=_arm_color(i), label=a["label"], ms=5)
    ax.set_xticks(x, [_short(f) for f in folds], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("PR-AUC on the held-out run")
    ax.set_title("Which run was held out matters far more than which channels were used",
                 loc="left")
    ax.legend(fontsize=8)
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _noise_floor(data: dict) -> float | None:
    """Typical size of a fold-level difference between two identical settings."""
    vals = []
    for c in data["contrasts"]:
        if _is_seed_repeat(c) and c["n_folds"] and c[PRIMARY]:
            vals.extend(abs(v) for v in c[PRIMARY]["per_fold"].values())
    return float(np.mean(vals)) if vals else None


def _verdict(c: dict, floor: float | None) -> str:
    """Plain-English reading of one contrast.

    Significance and size are separate questions and the wording has to keep
    them apart. A tiny effect repeated across every fold can be significant and
    still be worth nothing in practice, so "consistent" and "big enough to care
    about" are said separately rather than collapsed into one word.
    """
    m = c[PRIMARY]
    if not m:
        return "not run yet"
    if _is_seed_repeat(c):
        return (f"yardstick — the same setting twice, so this spread "
                f"({m['mean_delta']:+.4f}) is what no difference looks like")
    d, p = m["mean_delta"], m["p_value"]
    direction = "helps" if d > 0 else "hurts"
    if p >= 0.05:
        return f"no measurable difference ({d:+.4f}, p={p:.2f})"
    if floor is not None and abs(d) < floor:
        return (f"consistently {direction}, but by less than changing the random "
                f"seed does ({d:+.4f} vs a {floor:.4f} noise floor) — real, "
                f"not worth acting on")
    times = "" if floor is None else f", {abs(d) / max(floor, 1e-9):.1f}x the noise floor"
    return f"{direction} ({d:+.4f}, p={p:.3f}{times})"


def write_markdown(data: dict, path: str) -> None:
    floor = _noise_floor(data)
    L = [f"# {data['title']}", "", data["blurb"], "",
         f"Leave-one-run-out over {len(data['folds'])} recordings: every setting is "
         "trained on all but one run and scored on the run it never saw. "
         "PR-AUC is the headline number — it is the one that stays honest when "
         "rocks are a small share of the samples.", ""]
    if floor is not None:
        L += [f"**A difference of nothing looks like {floor:.4f} PR-AUC on this data.** "
              "That is the average fold-level gap between two runs of the *same* "
              "setting with only the random seed changed. Any effect smaller than "
              "that is noise, whatever the average says.", ""]

    L += ["## Every setting", "",
          "| setting | folds | PR-AUC | ROC-AUC | F1 | what it is |",
          "|---|---|---|---|---|---|"]
    for a in sorted(data["arms"], key=lambda r: -(r[PRIMARY]["mean"] or 0)):
        if not a["folds_done"]:
            continue
        def cell(k):
            m = a[k]
            s = f"{m['mean']:.3f}"
            return s + (f" ± {m['std']:.3f}" if m["std"] is not None else "")
        L.append(f"| {a['label']} | {a['folds_done']} | **{cell('pr_auc')}** | "
                 f"{cell('roc_auc')} | {cell('f1')} | {a['what']} |")

    L += ["", "## Head to head, paired fold by fold", "",
          "Each row trains two settings on the exact same folds and compares them "
          "one fold at a time. `W/L` counts folds won and lost. The p-value is a "
          "Wilcoxon signed-rank test: below 0.05 means the pattern of wins is "
          "unlikely to be chance.", "",
          "| comparison | folds | change in PR-AUC | W/L | p | verdict |",
          "|---|---|---|---|---|---|"]
    for c in data["contrasts"]:
        if not c["n_folds"]:
            L.append(f"| {c['question']} | 0 | — | — | — | not run yet |")
            continue
        m = c[PRIMARY]
        L.append(f"| {c['question']} | {c['n_folds']} | "
                 f"{m['mean_delta']:+.4f} ± {(m['std_delta'] or 0):.4f} | "
                 f"{m['wins']}/{m['losses']} | {m['p_value']:.3f} | "
                 f"{_verdict(c, floor)} |")

    L += ["", "## Per-fold detail", "",
          "| setting | " + " | ".join(_short(f) for f in data["folds"]) + " |",
          "|---|" + "---|" * len(data["folds"])]
    for a in data["arms"]:
        if not a["folds_done"]:
            continue
        cells = [f"{a[PRIMARY]['per_fold'].get(f, float('nan')):.3f}"
                 if f in a[PRIMARY]["per_fold"] else "–" for f in data["folds"]]
        L.append(f"| {a['label']} | " + " | ".join(cells) + " |")

    L += ["", "![](arm_ranking.png)", "", "![](paired_deltas.png)", "",
          "![](per_fold.png)", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {path}")


def render_ablation(root: str, suite: str, out_dir: str) -> dict:
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r} (pick from {sorted(SUITES)})")
    data = collect(root, suite)
    done = sum(a["folds_done"] for a in data["arms"])
    if not done:
        print(f"no finished folds under {os.path.join(root, suite)} yet — nothing to report")
        return data
    os.makedirs(out_dir, exist_ok=True)
    data["noise_floor_pr_auc"] = _noise_floor(data)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(data, f, indent=2)
    fig_arm_ranking(data, os.path.join(out_dir, "arm_ranking.png"))
    fig_paired_deltas(data, os.path.join(out_dir, "paired_deltas.png"))
    fig_per_fold_lines(data, os.path.join(out_dir, "per_fold.png"))
    write_markdown(data, os.path.join(out_dir, "summary.md"))
    print(f"reported {done} finished folds across "
          f"{sum(1 for a in data['arms'] if a['folds_done'])} settings")
    return data
