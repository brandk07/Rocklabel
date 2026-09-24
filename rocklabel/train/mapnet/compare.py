"""Put the map model and the per-ball models side by side on identical terms.

Every earlier model's finished map was saved by ``mapeval`` as
``map-control.npz``. This re-grades those saved maps with the same function the
map model is graded with, optionally restricted to part of the arena (for the
spatial hold-outs), so every row of a comparison table shares its rocks, its
denominators and its false-area definition.
"""

from __future__ import annotations

import os

import numpy as np

from ..visual_audit import CellMap
from . import grade as G

#: Earlier models worth comparing against, by the name the reports use.
PRIOR = {
    "deployed": "training/reports/epoch-selection-v1/incumbent-map-comparison/deployed-0.5399",
    "incumbent-0.7112": "training/reports/epoch-selection-v1/incumbent-map-comparison/incumbent-0.7112",
    "cls-both-s44": "training/reports/epoch-selection-v1/incumbent-map-comparison/clutter-s44-0.7044",
    **{f"baseline-r050-h1-s{s}": f"training/reports/neighborhood-history-v1/mapeval/fixed-r050__h1/seed-{s}"
       for s in (42, 43, 44)},
    **{f"history-r075-h5-8s-s{s}": f"training/reports/neighborhood-history-v1/mapeval/fixed-r075__h5-8s/seed-{s}"
       for s in (42, 43, 44)},
    **{f"history-r050-h5-30s-s{s}": f"training/reports/neighborhood-history-v1/mapeval/fixed-r050__h5-30s/seed-{s}"
       for s in (42, 43, 44)},
}


#: Stored thresholds of the reference checkpoints whose saved map folders do
#: not record one (read from each checkpoint's own "threshold").
STORED_THRESHOLD = {"deployed": 0.71, "incumbent-0.7112": 0.75, "cls-both-s44": 0.722}


def load_map(folder: str) -> CellMap:
    with np.load(os.path.join(folder, "map-control.npz")) as z:
        return CellMap(z["positions"].astype(np.float64), z["probabilities"].astype(np.float64),
                       z["cell_ids"].astype(np.int64))


def restrict(cells: CellMap, visible: dict, labels, region) -> tuple[CellMap, dict]:
    """Keep only cells, and rocks, on the ``region`` side of the arena."""
    if region is None:
        return cells, visible
    keep = region(cells.positions[:, :2])
    centre = {r.id: np.asarray(r.center[:2], float) for r in labels.rocks}
    vis = {rid: (s if region(centre[rid][None])[0] else set()) for rid, s in visible.items()}
    return CellMap(cells.positions[keep], cells.probabilities[keep], cells.cell_ids[keep]), vis


def grade_prior(labels, visible, region=None, names=None) -> dict[str, dict]:
    out = {}
    for name, folder in PRIOR.items():
        if names and name not in names:
            continue
        if not os.path.exists(os.path.join(folder, "map-control.npz")):
            continue
        cells, vis = restrict(load_map(folder), visible, labels, region)
        out[name] = G.grade(cells, labels, vis, 0.5)
    return out
