"""Portable model export: TorchScript + ONNX + a metadata JSON that pins the
full preprocessing contract, so the model can be used without this repo.

Exported signature: (points [B, 256, 4] float32, counts [B] int64) -> rock
probability [B] float32 (sigmoid applied inside). If true neighbor counts are
unknown at inference time, pass counts=256: PointNet is exactly invariant to
that and PointNet++ degrades only mildly (all pooling is max).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import torch

from .models import FEATURES, build_model_from_config, model_task

EXAMPLE = '''\
"""Standalone scoring example - needs only torch (or onnxruntime), not rocklabel.

Read preprocessing_contract in metadata.json: classifiers use neighborhoods;
segmenters use whole frames relative to the robot base. Do not interchange
these layouts. Valid points must precede padding, and counts must be supplied.
"""
import json

import numpy as np
import torch

meta = json.load(open("metadata.json"))
model = torch.jit.load("model.torchscript.pt").eval()

n, p = 8, meta["input"]["points_per_sample"]
points = np.random.rand(n, p, 4).astype(np.float32)  # stand-in for real samples
counts = np.full(n, p, dtype=np.int64)

with torch.no_grad():
    probs = model(torch.from_numpy(points), torch.from_numpy(counts)).numpy()
rock = probs >= meta["decision_threshold"]
print("rock probability:", np.round(probs, 3))
print("is rock @ threshold", meta["decision_threshold"], ":", rock)

# ONNX alternative:
#   import onnxruntime as ort
#   sess = ort.InferenceSession("model.onnx")
#   probs = sess.run(None, {"points": points, "counts": counts})[0]
'''


class InferenceModel(torch.nn.Module):
    """Trained classifier wrapped to emit probabilities."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, points: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(points, counts))


def export_model(checkpoint_path: str, out_dir: str) -> None:
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg, gcfg = ck["config"], ck["generator"]
    model = build_model_from_config(cfg)
    model.load_state_dict(ck["model"])
    wrapped = InferenceModel(model).eval()

    os.makedirs(out_dir, exist_ok=True)
    task = model_task(cfg["model"])
    n_pts = int(gcfg["segmentation_points" if task == "segment" else "neighborhood_points"])
    ex_pts = torch.zeros(2, n_pts, 4)
    ex_cnt = torch.full((2,), n_pts, dtype=torch.long)

    # Trace rather than script: control flow (FPS loop, T-Net toggles) is
    # static for a fixed architecture, and tracing keeps the batch dim dynamic.
    with torch.no_grad():
        ts = torch.jit.trace(wrapped, (ex_pts, ex_cnt))
    ts_path = os.path.join(out_dir, "model.torchscript.pt")
    ts.save(ts_path)

    onnx_path = os.path.join(out_dir, "model.onnx")
    torch.onnx.export(
        wrapped, (ex_pts, ex_cnt), onnx_path,
        input_names=["points", "counts"], output_names=["rock_prob"],
        dynamic_axes={"points": {0: "batch"}, "counts": {0: "batch"}, "rock_prob": {0: "batch"}},
        dynamo=False,
    )

    used = list(model.features)
    ignored = [f for f in FEATURES if f not in used]
    meta = {
        "model": cfg["model"],
        "task": "binary rock classification (per neighborhood sample)",
        "input": {
            "points_per_sample": n_pts,
            "points": f"[batch, {n_pts}, 4] float32, channels [dx, dy, dz, intensity]",
            "counts": "[batch] int64, number of real (non-padded) points; pass "
                      f"{n_pts} if unknown",
            "neighborhood_points": n_pts,
            "neighborhood_radius_m": gcfg["neighborhood_radius_m"],
            "centers_voxel_m": gcfg["centers_voxel_m"],
            "min_neighbors": gcfg["min_neighbors"],
            "features_used": used,
            "features_ignored": ignored,
        },
        "preprocessing_contract": (
            "Neighborhood = all points within neighborhood_radius_m of a candidate "
            "center from a centers_voxel_m voxel grid. dx/dy = point xy minus center "
            "xy (odom frame, axis-aligned); dz = point z minus the neighborhood's "
            "minimum z (local ground ~ 0); intensity = LiDAR reflectivity as recorded "
            "(observed range ~[0, 1], passed through unchanged). If fewer than "
            "neighborhood_points points: pad by repeating real points AFTER them "
            "(real points come first); if more: random subsample. Inputs are already "
            "canonicalized - do not re-center or re-normalize."
            + ("" if not ignored else
               f" This model was trained on {used} only: the shape stays [B, "
               f"{n_pts}, 4] but {ignored} is selected out inside the model, so "
               "you may pass anything (zeros included) in those channels.")
        ),
        "output": "rock probability in [0, 1] (sigmoid applied)",
        "decision_threshold": ck.get("threshold", 0.5),
        "source_config_hash": ck["config_hash"],
        "training": {
            "train_runs": cfg["train_runs"],
            "held_out_run": cfg["test_run"],
            "epochs_trained": ck["epoch"] + 1,
            "seed": cfg["seed"],
            "tnet": cfg["tnet"],
            "class_weighting": "BCE pos_weight = n_clear / n_rock of the train split",
            "augmentation": "random z-rotation + xy mirror" if cfg["augment"] else "none",
        },
        "exported": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if task == "segment":
        meta["task"] = "binary rock segmentation (per point in a whole frame)"
        for key in ("neighborhood_points", "neighborhood_radius_m", "centers_voxel_m", "min_neighbors"):
            meta["input"].pop(key)
        meta["input"]["segmentation_points"] = n_pts
        meta["input"]["segmentation_min_points"] = gcfg["segmentation_min_points"]
        meta["input"]["counts"] = "[batch] int64, valid rows before repeat padding; ignore padded outputs"
        meta["preprocessing_contract"] = (
            "One whole cropped frame per sample. xyz = world-frame point minus "
            "robot-base xyz; intensity uses the training stream's [0, 1] scale. "
            "Subsample without replacement above segmentation_points; otherwise "
            "append repeat padding after all valid rows. Keep the selected point "
            "indices to map outputs back to world coordinates. The model applies "
            "its saved height/coordinate reference internally. Never pass "
            "classifier neighborhoods or subtract a neighborhood minimum.")
        meta["output"] = f"[batch, {n_pts}] rock probabilities; only the first counts[i] rows are valid"
    meta["generator"] = gcfg
    meta["model_config"] = cfg
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, "infer_example.py"), "w") as f:
        f.write(EXAMPLE)

    # Round-trip check: TorchScript must reproduce the eager model.
    with torch.no_grad():
        pts = torch.randn(3, n_pts, 4)
        cnt = torch.tensor([min(40, n_pts), n_pts, min(100, n_pts)])
        a, b = wrapped(pts, cnt), ts(pts, cnt)
    if not torch.allclose(a, b, atol=1e-5):
        raise SystemExit("TorchScript output diverged from the eager model")
    print(f"exported {cfg['model']} [{', '.join(used)}] -> {ts_path}, {onnx_path}, "
          f"metadata.json, infer_example.py (torchscript round-trip OK)")
