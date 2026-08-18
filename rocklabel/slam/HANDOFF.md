# altslam — handoff notes

> **Where this lives:** `rocklabel/slam/` (it moved in from a top-level
> `altslam/` folder, internals unchanged). Run: `python -m rocklabel.slam`.
> Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_slam.py -q`.

For whoever picks this up next. What was measured, what worked, what didn't,
and where the remaining headroom is. Numbers are from
`recordings/VolleyBallTest1.mcap` (41.5 s, hand-swept sand volleyball court)
unless noted.

Read `README.md` first for what the thing does. This file is about *why the
decisions were made* and *what not to waste time on*.

---

## 1. State of play

The stock tracker (`rocklabel/live/slam.py`) left the sand surface 32.7 mm
thick. altslam gets it to 17.1 mm. Across all 13 volleyball recordings the gain
ranges 1.86x-4.06x, mean 2.44x, every run landing at 17-23 mm.

**Pose is no longer the dominant error.** Current budget, measured:

| term | size | can software fix it |
|---|---|---|
| sensor noise, 1-6 m | ~11 mm | no |
| sensor noise, 6-8 m | ~27 mm | no (but you can crop it out) |
| pose error | ~8 mm | yes, with work |
| resulting surface | 17 mm | — |

A *perfect* trajectory gets you to roughly 12 mm. That is the whole remaining
prize from SLAM work. Budget your effort accordingly.

---

## 2. Why the stock tracker fails here

Worth understanding before changing anything, because the failure is
counter-intuitive.

**It reports 99% of points matched and zero dropped windows for the entire
run.** It is not losing lock — it is confidently wrong. On this scene the match
ratio carries no information about whether the answer is right, so do not use it
as a health signal and do not trust a variant that only improves it.

Three separate causes:

1. **A flat sand court is geometrically degenerate.** A plane constrains height
   and nothing else. Point-to-point ICP will happily slide the scan sideways
   forever while the match count stays at 99%.
2. **The 12 m range cap threw away the only fix.** Everything within 12 m is
   sand. The grass edge / fence / tree line at 15-25 m is the only structure
   that constrains sideways motion, and it was excluded.
3. **The IMU's "down" is wrong exactly when moving.** Tilt comes from feeling
   gravity; swinging the sensor by hand adds acceleration the IMU cannot
   distinguish from gravity. `rotation_mode="yaw"` treats tilt as ground truth.
   On these recordings this is the single largest error term.

---

## 3. What worked, ranked by measured impact

| change | effect | notes |
|---|---|---|
| **Let ICP correct roll/pitch** (`lock_roll_pitch=False`) | 34.6 -> 17.9 mm | The big one, ~2x alone. Counter-intuitive: the IMU is gravity-referenced so tilt "should" be absolute. It isn't, when hand-held. |
| **Gate correspondences on along-normal distance**, not distance to voxel centroid | match 16% -> 58% | See traps below. Was a genuine bug in my first draft. |
| **Feed the map full-density points**, ICP only the downsampled ones | normals form at all | Downsampled insertion starves voxels; they never reach the point count needed for a covariance. |
| **Normal-space balancing** (`normal_balance`) | 49.8 -> 35.1 mm at `min_points_normal=8` | Much smaller effect at `minN=5`, where more voxels are usable anyway. Keep it; it is cheap insurance. |
| Point-to-plane + degeneracy guard + 30 m range | enabling changes | Hard to separate individually — they were introduced together as the baseline. Do not assume each is independently large. |
| `min_points_normal=5` over `8` | 35.6 -> 35.1 mm | More usable voxels beats stricter normals on sparse scans. |
| `voxel_size=0.20` over `0.30` | 36.9 -> 35.1 mm | |
| Multi-pass refinement | 35.1 -> 34.6 mm | Marginal. See below. |

---

## 4. What did NOT work — do not repeat these

- **Loop closure / pose graph.** *Measured, not assumed.* Disagreement between
  two looks at the same patch: 5.5 mm under 2 s, 8.0 mm at 5-10 s, 7.5 mm at
  20-35 s. **It stops growing after ~5 s**, so there is no accumulating drift
  left to close. The multi-pass refinement already made the trajectory globally
  consistent. This was on my own "next steps" list until I measured it. Use
  `evaluate.error_vs_gap()` before building any global-consistency machinery.
- **More refinement passes.** 1 -> 2 -> 3 -> 5 gives 35.1 / 34.9 / 34.6 /
  ~34.6 mm. Each pass costs ~40 s. `passes=1` is a fine default if you want
  speed; the gain is real but tiny.
- **Shorter windows.** `window_sec=0.05` *diverges badly* — 98.3 mm, far worse
  than doing nothing. Too few points per window. 0.10 and 0.20 are both fine
  (17.9 / 18.2 mm).
- **Larger voxels** (0.30 m): consistently slightly worse than 0.20 m.
- **Trusting the IMU for tilt** — the intuitive choice, and 2x worse.

---

## 5. Traps that cost real time

- **`pytest` needs `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`** in this environment. A
  ROS `launch_testing` plugin imports `lark`, which is missing, and crashes
  collection before any test runs. Nothing to do with this package.
- **The Bash tool times out at 120 s.** A `passes=3` solve takes ~110 s and a
  sweep takes far longer. Run long jobs in the background and poll.
- **Synthetic test scenes must be dense.** At ~1.3 points per voxel no normals
  form and every registration test fails with `res.ok == False`, which looks
  like a solver bug and is not. `scene_points()` now generates 160 k points for
  this reason. Same trap in `surface_sharpness`, which needs `min_points=20`
  per 10 cm cell and returns `nan` **silently** if starved.
- **The correspondence-radius trap.** Point-to-plane must reject on the
  *along-normal* residual. Rejecting on distance-to-voxel-centroid is wrong:
  with 20 cm voxels a perfectly aligned point can legitimately sit 17 cm from
  the centroid. Search generously (`1.5 * voxel`), gate on `|n . (p - q)|`.
- **`surface_sharpness` alone is too blunt to steer by.** It moved 35.1 -> 34.6
  across changes that were doing quite different things. Use `revisit_error()`
  to see whether you are fighting sensor noise or pose error.

---

## 6. How to measure

Three tools in `rocklabel/slam/evaluate.py`, each answering a different question.

```python
from rocklabel.slam.evaluate import surface_sharpness, revisit_error, error_vs_gap
```

| function | question it answers | when to use |
|---|---|---|
| `surface_sharpness` | how thick is the surface | headline number; tracks rock detectability |
| `revisit_error` | is that thickness the sensor or the poses | before optimising anything |
| `error_vs_gap` | is the pose error drift or jitter | decides loop-closure vs better registration |

`error_vs_gap` is the decision procedure: **rising with the gap** means drift is
accumulating and global optimisation pays; **flat** means it will not. It is
currently flat.

There is no ground-truth trajectory for these recordings, which is why all three
metrics are self-consistency measures.

Quick check of any change:

```bash
python -m rocklabel.slam recordings/VolleyBallTest1.mcap --score-only --passes 1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_slam.py -q     # 32 tests
```

---

## 7. Where to go next, in order

1. **Solve all poses jointly.** `solver._pass_refine` moves one window at a time
   against a frozen map — coordinate descent, which settles into a local
   optimum (hence passes 2-5 buying almost nothing). Real bundle adjustment over
   every window pose simultaneously is the main remaining algorithmic gap. This
   is the one worth building.
2. **Distribution-to-distribution matching** (GICP / NDT) in `register.py`.
   Point-to-plane reduces a rough sand patch to one normal. The map already
   stores full covariances (`voxelmap.NormalVoxelMap` keeps `_sumsq`), so most
   of the plumbing exists.
3. **Continuous-time trajectory.** Inside a 0.1 s window the rotation still
   comes only from the IMU — the component known to be unreliable during a hand
   sweep. A spline over the whole run would fix the one place the IMU is still
   trusted blindly.

**Not** loop closure. See section 4.

---

## 8. Bigger levers that are not SLAM

Worth saying out loud to whoever owns the data collection:

- **Crop the fusion range, not the registration range.** Noise is ~11 mm inside
  6 m and ~27 mm beyond it. The pipeline is already structured correctly —
  registration uses the full cloud, fusion uses a cropped band — so this is a
  tuning change, not a rewrite. Aim with the far scenery, build the surface from
  the near points.
- **Sweep the same ground more times.** Noise averages down; pose error does
  not. More passes over a patch is free thickness.
- **Get the sensor off a hand.** A tripod, monopod or slow cart removes the
  accelerometer corruption at source — the largest single error found. Fixing it
  in the recording beats correcting it afterwards.
- **Record raw gyro/accelerometer.** Only the *fused* quaternion is stored
  (see the frame layout in `rocklabel/live/recording.py`), and fusion is exactly
  what the hand-swinging corrupts. Raw gyro is immune and would enable proper
  tight coupling. Needs a recording-format change.

---

## 9. Why this matters downstream

From the user's own leave-one-run-out results
(`training/results_ablate/reflectivity/summary.json`, PointNet shape-only):

- Per-fold PR-AUC correlates **-0.56** with how hard that run is to solve for
  pose. Harder run -> worse model.
- The two collapsing folds (Test6 at 0.424, Test4 at 0.438, against 0.85-0.93
  for the good ones) are exactly the two hardest runs.
- Fold spread is 0.42-0.93, std 0.166 — against a same-setting seed-noise floor
  of 0.0078. A large share of that spread is pose quality, not model quality.

Mechanisms: pose fuzz buries the 5-10 cm rock relief the model keys on; it
displaces hand-placed sphere labels so points near the sphere edge get the wrong
class; and pose glitches create ridges that look exactly like rocks, which the
model learns to fire on and which never repeat on held-out runs.

Note the benefit only reaches the models once datasets are regenerated from the
improved recordings.

---

## 10. Housekeeping

- **Nothing outside `rocklabel/slam/` was modified.** It only reads from the rest of the package. The
  isolation was requested explicitly.
- **Not wired into the dashboard.** `CLAUDE.md` asks for every CLI change to get
  an entry in `rocklabel/dashboard/spec.py`; that was deliberately skipped to
  honour the isolation request. The *output* recordings do appear in the
  dashboard automatically, since its inventory scans `recordings/*.mcap`.
  Wiring the command in is a small, self-contained follow-up.
- **Output naming** is `<stem>.reslam.mcap`, matching the convention already
  used by the datasets, labels and training runs.
- Outputs preserve points, brightness and IMU samples byte-identically; only
  the per-batch pose changes. Verified by test and on the real files.
