"""Report figures for the map model."""

from __future__ import annotations

import csv
import json
import os

import numpy as np


def timeline_figure(path: str, stitched_summary: str, prior_timelines: dict[str, str]) -> None:
    """False ground and rock coverage over the recording: the map model's
    stitched map (every 60 s) beside earlier models' maps at their stored
    thresholds (``mapeval``'s timeline.csv)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tl = json.load(open(stitched_summary))["timeline"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    t = [r["time_s"] / 60 for r in tl]
    ax[0].plot(t, [r["false_cells_footprint"] for r in tl], "C3-", lw=2,
               label="map model (cross-validated), threshold 0.5")
    ax[1].plot(t, [r["macro_coverage"] or 0 for r in tl], "C3-", lw=2, label="map model")
    for i, (name, p) in enumerate(prior_timelines.items()):
        rows = [r for r in csv.DictReader(open(p)) if r["map"] == "control"]
        tt = [float(r["time_s"]) / 60 for r in rows]
        ax[0].plot(tt, [int(r["false_cells_footprint"]) for r in rows], color=f"C{i}",
                   lw=1, label=f"{name}, stored threshold")
    ax[0].set_ylabel("wrongly-claimed 10 cm cells")
    ax[0].set_xlabel("minutes into the recording")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=8)
    ax[1].set_ylabel("mean 3D rock coverage (rocks seen so far)")
    ax[1].set_xlabel("minutes into the recording")
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def frontier_figure(path: str, curves: dict[str, str], title: str,
                    key: str = "macro_coverage") -> None:
    """Coverage against wrongly-claimed area, one line per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, p) in enumerate(curves.items()):
        rows = [r for r in csv.DictReader(open(p)) if r["map"] in ("control", "mapnet")]
        x = np.array([int(r["false_cells_footprint"]) for r in rows])
        y = np.array([float(r[key] or 0) for r in rows])
        keep = x > 0
        ax.plot(x[keep], y[keep], lw=2 if i == 0 else 1, color=f"C{3 if i == 0 else i - 1}",
                label=name)
    ax.set_xscale("log")
    ax.set_xlim(10, 3000)
    ax.set_ylim(0, 1)
    for b in (50, 100, 200, 500, 1000):
        ax.axvline(b, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("wrongly-claimed 10 cm cells (threshold swept)")
    ax.set_ylabel("mean 3D rock coverage" if key == "macro_coverage"
                  else "mean occupied-footprint coverage")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
