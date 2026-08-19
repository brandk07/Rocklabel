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
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_slam.py -q     # 37 tests
```

---

## 7. Where to go next, in order

> **Read §9 first.** Pose quality was measured against model accuracy across all
> 13 recordings and does not predict it, so none of the items below are on the
> critical path for detection. They are worth roughly 14.7 mm -> 12 mm of
> surface thickness, and nothing more. The ordering is still right if you do
> pick this up.

1. **Solve all poses jointly.** `solver._pass_refine` moves one window at a time
   against a frozen map — coordinate descent, which settles into a local
   optimum (hence passes 2-5 buying almost nothing). Real bundle adjustment over
   every window pose simultaneously is the main remaining algorithmic gap. This
   is the one worth building — and §9 narrows the target: the failure it would
   fix shows up as a wrecked **first few seconds**, where the map is still
   empty, not as error spread over the whole trajectory.
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

## 9. Why this matters downstream — RETRACTED, and what replaced it

**This section used to claim that better poses would lift model accuracy, citing
a -0.56 correlation between per-fold PR-AUC and pose difficulty. That claim is
wrong. Do not build on it.** Full working in
[`handoff/02-slam-RESULT.md`](../../handoff/02-slam-RESULT.md).

Two things were wrong with it:

- The PR-AUC it correlated was **raw**, and PR-AUC scales with the positive
  rate. Rock prevalence runs 6.3%-31.6% across the 11 recordings, so "which fold
  is hard" was partly "which fold has few rocks". Normalize as
  `(AP - prevalence) / (1 - prevalence)` before comparing folds.
- It used **four spot-checked recordings**. Measured across all 13, ten of the
  eleven folds sit inside 16.7-17.9 mm of surface thickness — a 1.2 mm spread —
  and only VolleyBallTest4 is outside it at 23.1 mm. **Drop VB4 and the
  correlation goes from -0.61 to +0.004.** It was one leverage point, not a
  trend. Bootstrap CIs on every pose metric span zero.

The decisive number: `revisit_error`'s `between_mm` — the term that *is* pose
error, and the only one better SLAM can remove — correlates **+0.02** with fold
score. If poses drove accuracy, that is where it would show.

**What actually explains the two collapsing folds.** Both VolleyBallTest4 and
VolleyBallTest6 have exactly one label the model rejects outright:

| fold | label | samples | recall | mean prob | shape |
|---|---|---|---|---|---|
| VB4 | rock 8 | 1053 (46% of the fold's positives) | 0.03 | 0.07 | 47 cm across, 6.5 cm proud, on 5.1 cm-rough sand |
| VB6 | rock 8 | 95 | 0.00 | 0.04 | 10.6 cm across, 3.4 cm proud, on 3.5 cm-rough sand |

Removing them takes VB4 from 0.355 to **0.583** normalized and VB6 from 0.513 to
**0.655** — an order of magnitude more than every SLAM change on this project
combined. VB6's pose is meanwhile among the best of the 13 (its rock 8 patch
holds still to 9.2 mm, pure sensor noise). So the old "the two collapsing folds
are the two hardest runs" had the right folds for the wrong reason.

**VolleyBallTest4's real pose defect, since it is worth knowing.** Its damage is
confined to the **first five seconds** (67.5 mm surface thickness, against
12.5-14.0 mm for the remaining 50 s). Cause: `_pass_forward` starts from an
empty map, so the earliest windows align against almost nothing, and their poses
are then inserted *into* that map. Every `_pass_refine` rebuilds the map from
those same windows, so the error is self-confirming. `passes=5` gives 67.6 mm —
no improvement whatever. This makes the "joint bundle adjustment" item in §7 a
much smaller target than it looked: the failure is a **start-up** problem, not a
whole-trajectory one.

Re-scoring VB4 with the damaged frames dropped moves it only 0.355 -> 0.384
(dropping 35% of the data), which is why this is documented rather than fixed.

**How much thickness the model actually sees.** The 17 mm headline is scored to
8 m; the generator's crop box only keeps ~4 m. Inside that the surface is
**14.7 mm**. So §8's "crop the fusion range, not the registration range" is
already in place — registration reaches 30 m, the dataset is built from the near
band — and the remaining prize is 14.7 mm -> ~12 mm, not 17 -> 12.

## 10. Housekeeping

- **Nothing outside `rocklabel/slam/` was modified.** It only reads from the rest of the package. The
  isolation was requested explicitly.
- **Now wired into the dashboard** (agent 2). `rocklabel slam` is a real
  subcommand of the main CLI and a card in the `slam` stage of
  `rocklabel/dashboard/spec.py`. Both spellings share `add_slam_args`, so the
  catalog-drift test in `tests/test_dashboard.py` covers the solver's flags too.
- **Output folder fix** (agent 2): a recording read out of a `raw/` folder is
  written to the `reslam/` folder beside it. Before this, a batch re-solve wrote
  its outputs back into `recordings/<project>/raw/` alongside the originals.
- **Output naming** is `<stem>.reslam.mcap`, matching the convention already
  used by the datasets, labels and training runs.
- **The solve is exactly repeatable.** Re-solving VolleyBallTest4 from the raw
  file with the stored settings reproduces the shipped `.reslam` trajectory to
  0.000 mm, and its labelled rock centres move 0.0 mm. Labels only ever need
  checking if the *settings* change, never from a re-run.
- Outputs preserve points, brightness and IMU samples byte-identically; only
  the per-batch pose changes. Verified by test and on the real files.
