# Agent 3 — The next training suite

**Read [README.md](README.md) first**, plus `handoff/01-cleanup-RESULT.md` and
`handoff/02-slam-RESULT.md`. Work inside the structure agent 1 built.

Two full sweeps are already done (121 folds on raw-burst data, 88 on full-sweep data).
Your job is not to repeat them. It is to chase the specific headroom they exposed.

## What is already answered — do not spend folds on it

Reflectivity (settled twice, both directions), PointNet vs PointNet++ (no difference,
twice), and raw bursts vs full sweeps (full sweeps win, decisively). See the README table.
**Drop reflectivity from the default channel set** and use `--features dx dy dz`.

## The headroom, in priority order

### 1. Segmentation never finished training — this is the cheapest real win

Every one of the 11 segmentation folds ran the full 30-epoch cap and **not one
early-stopped**. Median best epoch was 28 of 30; 7 of 11 peaked in the final 5 epochs. The
0.767 headline is a **floor, not a ceiling**.

Run segmentation at **60–80 epochs** with patience raised to match. High confidence, low
cost. Do this before anything clever.

### 2. Segmentation is starved of frames, not of points

- It trains on ~2,165 frames per fold against the classifier's 94,519 samples — **~40× fewer
  training items**. This, not points-per-frame, is the binding constraint.
- Meanwhile **43% of every frame is padding**: median 1,145 real points against a
  `segmentation_points` cap of 2,048, with only 9 of 2,381 frames hitting the cap.

So: **lower `frame_stride` from 4 to 1** (~4× more frames, ~9,500), and **lower
`segmentation_points` to ~1,280**, which cuts wasted compute at essentially no information
loss and buys back the epochs §1 needs. These pull in the same direction — do them together.

### 3. The segmentation model's geometry is hardcoded and probably wrong for rocks

In `PointNetPPSeg` (`rocklabel/train/models.py:353`) the three levels are fixed at
`npoints=(512, 128, 32)` and `radii=(0.25, 0.6, 1.4)` m, with no way to change them
without editing the file. Two things look wrong for this problem:

- It discards three-quarters of the points at the very first level (2,048 → 512).
- Its **finest radius is 0.25 m while the rocks are ~20–30 cm** — the smallest scale it
  looks at is about the size of the whole object it is hunting.

**Promote both to config**, then run arms at finer settings (e.g. radii `(0.1, 0.3, 0.8)`,
npoints `(1024, 256, 64)`). This is the highest-upside item here and the only one needing
real code.

### 4. A longer sweep window — one arm, tempered expectations

Worth testing, but do not expect a repeat of the last gain. Going from ~110 to ~1,250 points
fixed genuine starvation (frames below the 512-point floor; neighbourhoods with ~34 points).
Going from 1,250 to ~2,500 is diminishing returns, and at the same stride it **halves the
frame count** — trading away the scarce resource (§2) for one that already has 43% slack.
If you test `frame_window_s: 0.1`, halve `frame_stride` to hold the frame count and raise
`segmentation_points` to ~2,560. Past ~0.2 s you begin smearing anything that moves
(people on the court), since only the static world is motion-compensated.

### 5. Verify two old data problems are already fixed

The reflectivity agent found, on **raw-burst** data:
- **Effective sample range ~2 m** — 88% of rock samples within 2 m, max 2.76 m, despite a
  6 m crop.
- **12 of 63 labelled rocks produced zero samples** (VB8 only 1 of 4; VB6 4 of 7; VB12 3 of 6).

Both were caused by `frame_window_s: 0.0` starving `min_neighbors: 20` inside a 0.5 m ball.
The full-sweep change raised points-per-frame ~11× and sample count 63k → 94.5k, so both are
**probably already fixed — but nobody has re-measured them.** This is cheap and it matters:
if rocks are still invisible past 2 m, that caps everything else you do. Re-run those two
measurements on the full-sweep cache before designing the suite, and report the numbers.

### 6. Investigate VB4

VB4 is the worst fold in every sweep (normalized 0.355, next worst 0.487) *and* reportedly
the geometrically cleanest recording. Segmentation helps it more than any other fold
(0.483 → 0.594). Something specific is wrong with it. Its 7 rocks are the most of any run —
check whether they are unusually small, flat, or clustered such that the ±0.15 m nearest-rock
assignment double-counts them (a known approximation in the per-rock counts).

## How to report — two traps, both already caught once

1. **Any classifier-vs-segmenter comparison must go through `rocklabel-train matched`.**
   Raw PR-AUC across the two tasks is meaningless — the default report shows segmentation
   losing by −0.37, which is entirely an artifact of 1% vs 19% positive rates. The matched
   comparison is validated at 99.4–99.9% label agreement and holds across pooling choices
   (max/mean/nearest) at 0.10–0.15 m radii.
2. **Report per-fold scores normalized as `(AP − prevalence) / (1 − prevalence)`.** Rock
   prevalence spans 6.3%–31.1%, so raw per-fold PR-AUC partly measures how many rocks a
   recording has. This is not yet implemented anywhere — add it to the report and it will
   change which recordings look hard (README has the reordered table). Keep raw PR-AUC too,
   for continuity with the two existing sweeps.

Also: **pair by fold, never pool** (fold difficulty spans 0.42–0.93, far larger than any
effect under test); use the exact Wilcoxon signed-rank already in `ablate.py`; and clear the
right noise floor — **0.0128 for classifiers, 0.0207 for segmentation** on full-sweep data
(the older 0.0078 belongs to the raw-burst cache).

## Suggested suite

Budget realistically: on one GPU a classifier fold is ~10 min and a segmentation fold ~8–12
min at 30 epochs, so ~2 h per 11-fold classifier arm and more once epochs double. The last
88-fold sweep took ~15 h while sharing the GPU. **Order arms by priority** so an interrupted
sweep still leaves the headline finished — `run_suite` already runs top to bottom and skips
finished folds.

1. `seg-long` — segmentation, 80 epochs, `frame_stride: 1`, `segmentation_points: 1280`
2. `seg-fine` — as above plus the finer radii/npoints from §3
3. `pointnet2-geom` and `pointnet-geom` — classifier baselines on the same data, so the
   matched comparison has a partner
4. `seg-long-s43` — seed repeat, for the noise floor at the new settings
5. `window-0.1` — the longer-window arm from §4, if time allows

If agent 2 delivers re-solved recordings and migrated labels, regenerate the datasets from
those and add a paired arm against the current `.reslam` data — that is the clean test of
whether SLAM quality moves model accuracy, and it directly settles the open question in
[02-slam.md](02-slam.md) §1.

## Definition of done

- Tests pass; dashboard updated for any new flag or profile (`tests/test_dashboard.py`
  catches drift).
- Reports include both raw and prevalence-normalized per-fold scores.
- A plain-English summary per `CLAUDE.md`: what changed, what it means, what Brandon will
  notice — and honest about anything that did not work.
