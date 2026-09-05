"""Model registry, importable without torch.

The CLI has to name every model in its --help and validate --models before
anything heavy loads, but rocklabel-train's whole reason for existing
separately is that the base tool never imports torch. So the name -> (task,
label) table lives here and models.py re-exports it.
"""

from __future__ import annotations

#: model name -> (task, human-readable label). ``task`` selects the dataset
#: format a run consumes ("classify" = format A neighborhoods, "segment" =
#: format C whole frames) and how it is scored.
MODELS: dict[str, tuple[str, str]] = {
    "pointnet":      ("classify", "PointNet (sliding-window classifier)"),
    "pointnet2":     ("classify", "PointNet++ (sliding-window classifier)"),
    "pointnet2_seg": ("segment",  "PointNet++ (per-point segmentation)"),
    "bev_cnn":       ("segment",  "BEV CNN (rasterized, per-point segmentation)"),
}


def model_task(name: str) -> str:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r} (pick from {sorted(MODELS)})")
    return MODELS[name][0]

#: The BEV CNN's per-cell input channels, in the order dataset/bev.py writes
#: them, so a selection here means the same thing as it does in a stored
#: raster. Lives beside the model table rather than with the models because the
#: ablation sweep names these channels and must not import torch to do it.
BEV_CHANNELS = ("occupied", "count", "z_max", "z_min", "z_span", "z_std",
                "intensity_mean", "intensity_max")

#: The two that say how many returns made a cell. A stray return sitting on no
#: surface is a cell with one return where a surface would have many, and no
#: point-based model here can see that: the point tensor is padded by repeating
#: real rows and the true count is used only to build a validity mask.
BEV_DENSITY = ("occupied", "count")
