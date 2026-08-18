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
}


def model_task(name: str) -> str:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r} (pick from {sorted(MODELS)})")
    return MODELS[name][0]
