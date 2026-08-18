# Agent 2 — SLAM, and keeping Brandon's labels

**Read [README.md](README.md) first, then `rocklabel/slam/HANDOFF.md`** — the previous SLAM agent's
notes are good and mostly stand. This brief says what changed since, and what to do
differently.

Work inside the layout agent 1 left behind (see `handoff/01-cleanup-RESULT.md`).

## 1. Before building anything: check the premise

`rocklabel/slam/HANDOFF.md` §9 argues that better poses will lift model accuracy, citing a −0.56
correlation between per-fold PR-AUC and pose difficulty. **That correlation is not safe to
build on**, for two reasons:

- It was computed on raw PR-AUC, which is confounded by rock prevalence (6.3%–31.1% across
  the 11 recordings — a 5× spread).
- It was computed on the old raw-burst data, whose problems have since been partly fixed by
  the full-sweep change.

Normalizing the full-sweep results as `(AP − prevalence) / (1 − prevalence)` produces two
direct counterexamples:

| fold | normalized score | pose quality |
|---|---|---|
| VB4 | **0.355 — worst of 11** | 4.2% off-level — **the cleanest run of the 11** |
| VB12 | 0.487 | 26.8% off-level — consistent |
| VB6 | 0.513 | 15.2% off-level — consistent |
| VB11 | 0.766 — 6th of 11 | **24.3% off-level — the shakiest run** |

Two runs support the pose story, two contradict it. So **your first task is a measurement,
not a change** — exactly the discipline that saved the last agent from building loop closure:

1. Compute pose-quality metrics (`surface_sharpness`, `revisit_error`, off-level fraction)
   for **all 11 recordings**, not the four spot-checked.
2. Correlate against the normalized per-fold scores from
   `training/ablate_vb/fullsweep/pointnet2-geom/` (or wherever agent 1 moved it).
3. Report the correlation with its confidence interval, and say plainly whether it holds.

**If it holds**, the expensive work in §4 is justified. **If it collapses**, say so loudly —
it means model accuracy is limited by something other than pose, agent 3 should not wait on
you, and the remaining SLAM prize (17 mm → ~12 mm surface thickness) is worth having on its
own merits but is not the lever on detection accuracy.

Either way, **resolve the VB4 contradiction** (README, "open contradiction"). VB4 is the
worst fold in every sweep run so far and is reportedly the geometrically cleanest recording.
If its pose is genuinely fine, the cause is elsewhere — labelling, or its 7 rocks being
unusually small or flat — and that is worth knowing before anyone optimises anything.

## 2. Do not make Brandon relabel. This is a hard requirement.

Labels are hours of manual work and he has asked explicitly to keep them. A label file
stores rock centres, the arena polygon and the height band **in world coordinates**, plus
the mount roll/pitch it was labelled under. Change the trajectory and all of that moves.

**Work in this order — cheapest first:**

**Step 1: measure the damage before building anything.** Re-solve one recording, then
measure how far the labelled rock centres actually move between the old and new
reconstruction. Rocks have a ~0.15 m default radius, and the generator adds a 0.05 m
boundary shell. **If the median displacement is well under a few centimetres, the labels
transfer as-is** and Brandon nudges the odd one — no tool needed. Say so and stop here.

**Step 2, only if displacement is large: build a label migration tool.** Migrate
per-point rather than by one rigid transform, because the difference between two
trajectories is time-varying, not a fixed offset:

1. Replay the old recording with the old poses. For each rock sphere, gather the points
   inside it and record, for each, its **scan timestamp and its sensor-frame coordinate**
   (before the pose transform).
2. Replay the re-solved recording. Push each of those points through the **new** pose at the
   same timestamp to get its new world position.
3. The new rock centre is the robust centroid of those re-projected points; keep the radius
   unless a re-fit is clearly better.
4. The arena polygon and height band are static world geometry with no points of their own —
   transform them with a rigid alignment fitted between the old and new trajectories
   (Umeyama over the pose translations). Coarse is fine; the arena is a 4-corner box.

**Step 3, in either case: update the level record.** `pin_level_to_labels` replays the
mount angle stored in the label file, and `check_level_match` **hard-errors** if that angle
disagrees with the stream. If your changes alter the levelling, the migration must rewrite
that field or every downstream `generate` will fail with a confusing error.

**Validate** with `rocklabel driftcheck`, which overlays the first and last 10% of scans
around one rock and is exactly the right tool for "did this rock stay put". Confirm on a
recording with many rocks (VB4 has 7) before migrating the rest. Keep the originals — write
migrated labels alongside, never in place.

## 3. Cheap wins, worth doing regardless of §1

- **Re-solve all 13 volleyball recordings**, including `VolleyBallTest1` and
  `VolleyBallTest13`, which are missing from the current datasets. VB13 already has labels.
- **Crop the fusion range, not the registration range.** Sensor noise is ~11 mm inside 6 m
  and ~27 mm beyond it. The pipeline is already structured for this — registration uses the
  full cloud, fusion uses a cropped band — so it is a tuning change, not a rewrite. Aim with
  the far scenery, build the surface from the near points.
- **Wire altslam into the dashboard.** It was deliberately skipped for isolation; agent 1
  has now made a place for it. Per `CLAUDE.md`, if it is not reachable from the dashboard it
  does not exist for Brandon.

## 4. Algorithmic work, if §1 justifies it

Keep the previous agent's ordering — it is well reasoned:

1. **Joint bundle adjustment over all window poses.** `solver._pass_refine` moves one window
   at a time against a frozen map, which is coordinate descent and settles into a local
   optimum — that is why passes 2–5 buy almost nothing. This is the main remaining gap.
2. **Distribution-to-distribution matching (GICP/NDT)** in `register.py`. The map already
   stores full covariances in `voxelmap.NormalVoxelMap._sumsq`, so most of the plumbing
   exists.
3. **Continuous-time trajectory.** Inside a 0.1 s window the rotation still comes only from
   the IMU — the component known to be unreliable during a hand sweep.

**Do not build loop closure.** It was measured, not assumed: revisit error is flat against
time gap (5.5 mm under 2 s, 7.5 mm at 20–35 s), so there is no accumulating drift to close.

**Do not repeat these either:** trusting the IMU for tilt (2× worse), `window_sec=0.05`
(diverges to 98 mm), voxels larger than 0.20 m, or more than ~2 refinement passes.

## 5. Say this out loud to Brandon

From `rocklabel/slam/HANDOFF.md` §8, and worth repeating because it outranks everything above:
the single largest error source is that the sensor is **swung by hand**, which corrupts the
IMU's sense of "down". A tripod, monopod or slow cart removes it at the source. Recording
raw gyro/accelerometer instead of only the fused quaternion would enable proper tight
coupling. Both are recording-side fixes that beat anything achievable in software.

## Definition of done

- The §1 measurement, reported plainly, with a clear verdict on whether pose quality
  predicts model accuracy.
- Labels usable on the re-solved recordings without manual relabelling, validated with
  `driftcheck`.
- All 13 recordings re-solved into the layout agent 1 built.
- Tests pass (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest`), including `tests/test_slam.py` (32).
- `handoff/02-slam-RESULT.md`: what changed, the new surface-thickness numbers per
  recording, and whether agent 3 should regenerate datasets from the new recordings.
