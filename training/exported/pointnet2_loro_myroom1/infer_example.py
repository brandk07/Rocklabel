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
