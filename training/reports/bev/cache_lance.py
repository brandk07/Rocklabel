"""Pull 100 competition frames off the bag once and keep them, so the
evaluation can be re-run over many checkpoints without re-reading the mcap."""
import sys, pickle
sys.path.insert(0, "/home/brandon/Documents/perception-2026-testing/rocklabel")
from bev_lance import frames_of, LANCE
frames, vis, LS, gc, arena = frames_of(*LANCE, 100, 25)
import numpy as np
print(f"{len(frames)} frames, ~{np.mean([len(f[0]) for f in frames]):.0f} pts each, "
      f"{sum(len(v) for v in vis)} rock sightings")
with open("/tmp/claude-1000/-home-brandon-Documents-perception-2026-testing/2865bc6a-44fc-4cae-8ca6-d3c5ef5181ec/scratchpad/lance100.pkl", "wb") as f:
    pickle.dump({"frames": frames, "vis": vis, "LS": LS, "gc": gc, "arena": arena}, f)
print("cached")
