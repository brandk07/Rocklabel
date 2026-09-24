"""Grade a map model as the map grows, not only once the recording has ended.

The map is replayed in time order. Every ``every_s`` seconds the model is run on
the map exactly as it stood then - only returns already measured - and graded
against the rocks visible so far. This is what the robot would have on its
screen at each moment, and it is where a model that only works on a finished
map would be caught out.
"""

from __future__ import annotations

import time

import numpy as np

from . import grade as G
from .clouds import Cloud
from .grid import CELL_M, HeightGrid


def visible_until(geometry_dir: str, labels) -> list[tuple[float, int, tuple]]:
    """(time, rock, cell) for every visible rock cell in the mapeval cache."""
    import os
    rows = []
    for name in sorted(f for f in os.listdir(geometry_dir) if f.startswith("frame-")):
        with np.load(os.path.join(geometry_dir, name)) as z:
            t = float(z["time_s"])
            for rid, i, j in z["raw_rock_cells"]:
                rows.append((t, int(rid), (int(i), int(j))))
    return rows


def run(model, device, cloud: Cloud, threshold: float, every_s: float = 30.0,
        visible_rows=None, times=None, feature_set: str = "base",
        tta: bool = False, region=None, predict=None) -> list[dict]:
    """``region`` grades one fold's ground only; ``predict(grid)`` replaces the
    model call (the stitched two-model map uses it)."""
    labels = cloud.labels
    lo, hi = labels.arena.min(0), labels.arena.max(0)
    grid = HeightGrid.around(lo, hi, cloud.floor_z, CELL_M, 1.5)
    if times is None:
        times = list(np.arange(every_s, cloud.duration, every_s)) + [cloud.duration]
    out = []
    a = 0
    ever: dict[int, set] = {r.id: set() for r in labels.rocks}
    first: dict[int, float] = {}
    for t in times:
        b = int(np.searchsorted(cloud.t, t, "right"))
        for s in range(a, b, 4_000_000):
            e = min(s + 4_000_000, b)
            grid.add(cloud.xyz[s:e].astype(np.float64),
                     None if cloud.brightness is None else cloud.brightness[s:e])
        a = b
        t0 = time.perf_counter()
        if predict is not None:
            prob, st = predict(grid)
        else:
            prob, st = G.predict(model, grid, device, feature_set, tta)
        infer_s = time.perf_counter() - t0
        cells = G.to_cells(prob, st, grid)
        # Visible so far: the geometry cache's clock starts at its first
        # window, the cloud's at its first scan; both are the recording start.
        vis = {r.id: set() for r in labels.rocks}
        for tv, rid, key in visible_rows:
            if tv <= t:
                vis[rid].add(key)
        if region is not None:
            from .compare import restrict
            cells, vis = restrict(cells, vis, labels, region)
        res = G.grade(cells, labels, vis, threshold)
        for r in res["per_rock"]:
            if r["coverage"] > 0 and r["rock_id"] not in first:
                first[r["rock_id"]] = t
        out.append({"time_s": round(float(t), 1),
                    "rocks_visible": sum(1 for v in vis.values() if v),
                    "macro_coverage": res["macro_coverage"],
                    "worst_rock_coverage": res["worst_rock_coverage"],
                    "false_cells_footprint": res["false_cells_footprint"],
                    "inference_s": round(infer_s, 3),
                    "per_rock": {r["rock_id"]: round(r["coverage"], 3) for r in res["per_rock"]}})
        print(f"  t={t:7.1f}s rocks {out[-1]['rocks_visible']:2d} "
              f"cov {res['macro_coverage'] if res['macro_coverage'] is not None else float('nan'):.3f} "
              f"false {res['false_cells_footprint']}", flush=True)
    return out
