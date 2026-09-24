"""Grade a map model exactly the way ``mapeval`` grades the per-ball models.

The network's 2.5 cm probabilities are reduced to 10 cm ground cells by taking,
per cell, the sub-cell it is most sure of; that sub-cell's representative point
is its own measured surface (the 90th-percentile height of the returns in it),
so the 3D attribution test asks the same question it asks of a voxel centroid:
does the thing being claimed lie on the rock? The frontier, the budgets and the
per-rock coverage then come from :mod:`rocklabel.train.map_eval` unchanged,
against the same visible-cell denominators its geometry cache recorded.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from ...dataset.labeling import inside_arena, points_in_rock
from ..map_eval import at_budget, threshold_frontier
from ..visual_audit import CellMap, cell_ids, cell_set, projected_rock_labels
from .clouds import Cloud
from .grid import CELL_M, HeightGrid, features

CELL_OUT_M = 0.10
BUDGETS = (2200, 1700, 1000, 500, 200, 100, 50)


class _Frozen:
    def __init__(self, name, cells):
        self.name = name
        self._c = cells

    def cells(self):
        return self._c


def map_grid(cloud: Cloud, t_max: float | None = None, margin_m: float = 1.5,
             bright: bool = True) -> HeightGrid:
    """The whole recording's map (up to ``t_max``) over its arena, plus margin."""
    labels = cloud.labels
    if labels is not None and labels.arena is not None:
        lo, hi = labels.arena.min(0), labels.arena.max(0)
    else:
        lo, hi = cloud.xyz[:, :2].min(0), cloud.xyz[:, :2].max(0)
    grid = HeightGrid.around(lo, hi, cloud.floor_z, CELL_M, margin_m)
    n = len(cloud.xyz) if t_max is None else int(np.searchsorted(cloud.t, t_max, "right"))
    step = 4_000_000
    use_b = bright and cloud.brightness is not None
    for a in range(0, n, step):
        b = min(a + step, n)
        grid.add(cloud.xyz[a:b].astype(np.float64),
                 cloud.brightness[a:b] if use_b else None)
    if not use_b:
        grid.has_bright = False
    return grid


@torch.no_grad()
def predict(model, grid: HeightGrid, device, feature_set: str = "base",
            tta: bool = False, cpu_features: bool = False) -> tuple[np.ndarray, dict]:
    """Rock probability for every 2.5 cm cell of ``grid``, and its stats.

    With ``tta`` the map is also read rotated and mirrored (the eight
    symmetries of the square grid) and the logits averaged: the arena has no
    preferred direction, so a model's answer should not have one either.
    """
    if cpu_features or device.type != "cuda":
        x, st = features(grid, feature_set)
        t = torch.from_numpy(x)[None].to(device)
    else:
        from .gpu import features_from_grid
        t, valid, q90 = features_from_grid(grid, device, feature_set)
        st = {"valid": valid, "q90_abs": q90}
    model.eval()
    views = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
    acc = None
    for k, flip in views:
        v = torch.rot90(t, k, (2, 3))
        if flip:
            v = torch.flip(v, (3,))
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            out = model(v).float()
        if flip:
            out = torch.flip(out, (3,))
        out = torch.rot90(out, -k, (2, 3))
        acc = out if acc is None else acc + out
    prob = torch.sigmoid(acc / len(views))[0, 0].cpu().numpy()
    return prob, st


def to_cells(prob: np.ndarray, st: dict, grid: HeightGrid) -> CellMap:
    """Reduce 2.5 cm probabilities to 10 cm cells, best sub-cell per cell."""
    rows, cols = np.nonzero(st["valid"])
    xs, ys = grid.cell_centers()
    pos = np.c_[xs[cols], ys[rows], st["q90_abs"][rows, cols]]
    p = prob[rows, cols].astype(np.float64)
    ids = cell_ids(pos[:, :2], CELL_OUT_M)
    key = ids[:, 0].astype(np.int64) * 1_000_003 + ids[:, 1]
    order = np.lexsort((-p, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    take = order[first]
    return CellMap(pos[take], p[take], ids[take])


def visible_from_geometry(geometry_dir: str, labels) -> dict[int, set]:
    """Each rock's visible ground cells, exactly as ``mapeval`` recorded them."""
    visible = {r.id: set() for r in labels.rocks}
    for name in sorted(f for f in os.listdir(geometry_dir) if f.startswith("frame-")):
        with np.load(os.path.join(geometry_dir, name)) as z:
            for rid, i, j in z["raw_rock_cells"]:
                visible[int(rid)].add((int(i), int(j)))
    return visible


def visible_from_cloud(cloud: Cloud, t_max: float | None = None) -> dict[int, set]:
    """Same definition from the dumped cloud, for recordings with no mapeval cache."""
    n = len(cloud.xyz) if t_max is None else int(np.searchsorted(cloud.t, t_max, "right"))
    visible = {}
    for rock in cloud.labels.rocks:
        v = np.asarray(rock.vertices) if rock.shape == "polygon" else None
        c = np.asarray(rock.center[:2]) if v is None else v.mean(0)
        r = float(rock.radius) + 0.1
        p = cloud.xyz[:n]
        near = (np.abs(p[:, 0] - c[0]) <= r) & (np.abs(p[:, 1] - c[1]) <= r)
        on = p[near][points_in_rock(p[near].astype(np.float64), rock)]
        visible[rock.id] = cell_set(on[:, :2], CELL_OUT_M) if len(on) else set()
    return visible


def grade(cells: CellMap, labels, visible, threshold: float, shell: float = 0.05,
          name: str = "mapnet") -> dict:
    """Frontier, budgets and per-rock coverage at ``threshold``."""
    m = _Frozen(name, cells)
    curve = threshold_frontier(m, labels, shell, visible)
    arena = inside_arena(cells.positions, labels.arena)
    hot = (cells.probabilities >= threshold) & arena
    fp = projected_rock_labels(cells.positions, labels.rocks, shell)
    rocks = []
    for rock in labels.rocks:
        seen = visible[rock.id]
        if not seen:
            continue
        occ = {tuple(k) for k in cells.cell_ids[hot]} & seen
        on = hot & points_in_rock(cells.positions, rock)
        att = {tuple(k) for k in cells.cell_ids[on]} & seen
        rocks.append({"rock_id": rock.id, "visible_cells": len(seen),
                      "occupied": len(occ) / len(seen), "coverage": len(att) / len(seen)})
    cov = [r["coverage"] for r in rocks]
    budgets = {}
    for b in BUDGETS:
        row = at_budget(curve, b)
        budgets[str(b)] = None if row is None else {
            k: row[k] for k in ("threshold", "macro_coverage", "worst_rock_coverage",
                                "macro_coverage_footprint", "false_cells_footprint")}
    return {"threshold": threshold,
            "false_cells_footprint": int((hot & (fp == 0)).sum()),
            "claimed_cells": int(hot.sum()),
            "macro_coverage": float(np.mean(cov)) if cov else None,
            "worst_rock_coverage": float(min(cov)) if cov else None,
            "per_rock": rocks, "budgets": budgets, "frontier": curve}


def plot(path: str, cells: CellMap, labels, threshold: float, title: str,
         background: np.ndarray | None = None, grid: HeightGrid | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ...dataset.labeling import LABEL_CLEAR, label_rocks
    keep = inside_arena(cells.positions, labels.arena)
    pos, prob = cells.positions[keep], cells.probabilities[keep]
    hot = prob >= threshold
    att = label_rocks(pos, labels.rocks, 0.05)
    fig, ax = plt.subplots(figsize=(10, 8))
    if background is not None and grid is not None:
        ax.imshow(background, origin="lower", cmap="gray", vmin=-0.1, vmax=0.4,
                  extent=[grid.x0, grid.x0 + grid.w * grid.cell,
                          grid.y0, grid.y0 + grid.h * grid.cell])
    wrong = hot & (att == LABEL_CLEAR)
    right = hot & (att != LABEL_CLEAR)
    ax.scatter(pos[wrong, 0], pos[wrong, 1], s=5, c="#d1495b",
               label=f"claimed, no rock ({int(wrong.sum())})")
    ax.scatter(pos[right, 0], pos[right, 1], s=6, c="#2e8b57",
               label=f"claimed, on a rock ({int(right.sum())})")
    for rock in labels.rocks:
        if rock.shape == "polygon":
            v = np.asarray(rock.vertices)
            v = np.vstack([v, v[:1]])
            ax.plot(v[:, 0], v[:, 1], color="#4cc9f0", lw=1)
        ax.annotate(str(rock.id), np.asarray(rock.center[:2]), color="#f72585", fontsize=9)
    if labels.arena is not None:
        a = np.vstack([labels.arena, labels.arena[:1]])
        ax.plot(a[:, 0], a[:, 1], color="y", lw=1)
        lo, hi = labels.arena.min(0), labels.arena.max(0)
        ax.set_xlim(lo[0] - 0.3, hi[0] + 0.3)
        ax.set_ylim(lo[1] - 0.3, hi[1] + 0.3)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, markerscale=3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def summarize(result: dict) -> str:
    b = result["budgets"]

    def f(k):
        r = b.get(k)
        return "-" if r is None else f"{r['macro_coverage']:.3f}/{r['worst_rock_coverage']:.2f}"
    return (f"thr {result['threshold']:.2f}: cov {result['macro_coverage']:.3f} "
            f"worst {result['worst_rock_coverage']:.2f} false {result['false_cells_footprint']} | "
            f"@1000 {f('1000')} @500 {f('500')} @200 {f('200')} @100 {f('100')}")


def save(result: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    slim = {k: v for k, v in result.items() if k != "frontier"}
    with open(os.path.join(out_dir, "grade.json"), "w") as f:
        json.dump(slim, f, indent=1, default=float)
    import csv
    rows = result["frontier"]
    if rows:
        with open(os.path.join(out_dir, "threshold-frontier.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
