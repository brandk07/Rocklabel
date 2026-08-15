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

from .models import FEATURES, build_model

EXAMPLE = '''\
"""Standalone scoring example - needs only torch (or onnxruntime), not rocklabel.

Input contract (see metadata.json): raw [N, 256, 4] float32 neighborhoods
exactly as produced by the rocklabel generator - [dx, dy, dz, intensity] with
dx/dy relative to the sample center, dz relative to the neighborhood's lowest
point, already canonicalized (do NOT re-center or re-normalize). counts[i] is
the number of real (non-padded) points; use 256 if unknown.
"""
import json

import numpy as np
import torch

meta = json.load(open("metadata.json"))
model = torch.jit.load("model.torchscript.pt").eval()

n, p = 8, meta["input"]["neighborhood_points"]
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
    model = build_model(cfg["model"], tnet=cfg["tnet"], dropout=cfg.get("dropout"),
                        features=cfg.get("features"))
    model.load_state_dict(ck["model"])
    wrapped = InferenceModel(model).eval()

    os.makedirs(out_dir, exist_ok=True)
    n_pts = int(gcfg["neighborhood_points"])
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
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, "infer_example.py"), "w") as f:
        f.write(EXAMPLE)

    # Round-trip check: TorchScript must reproduce the eager model.
    with torch.no_grad():
        pts = torch.randn(3, n_pts, 4)
        cnt = torch.tensor([40, n_pts, 100])
        a, b = wrapped(pts, cnt), ts(pts, cnt)
    if not torch.allclose(a, b, atol=1e-5):
        raise SystemExit("TorchScript output diverged from the eager model")
    print(f"exported {cfg['model']} [{', '.join(used)}] -> {ts_path}, {onnx_path}, "
          f"metadata.json, infer_example.py (torchscript round-trip OK)")
