# Neighborhood size and scan history for the PointNet classifier

September 2026. Trained on Volleyball recordings VB2-VB12 (validation on VB4 and
VB6, training on the other nine). Graded on the Lance competition recording,
which has now been consulted for model selection many times over and is
**development data, not a test set**.

## The answer

**Giving the classifier older scans helps. A wider ball on its own does not,
and neither does the adaptive radius.** The one setting that improved the map
on all three training seeds is a **fixed 0.75 m ball fed the current sweep plus
four older ones at 2, 4, 6 and 8 seconds** (`fixed-r075__h5-8s`). Paired
against the plain 0.5 m single-sweep baseline trained with the same seed:

- At an equal amount of wrongly-claimed ground (500 false footprint cells) it
  covers **+0.104** more of each rock's surface on average (+0.081 to +0.118
  across the three seeds). At 1,000 cells, +0.043 (+0.028 to +0.056).
- At its own stored threshold it claims **412 fewer false cells** (260 to 539
  fewer) for the same average coverage (-0.002).

That is more than twice the largest seed-to-seed spread this project has
recorded (0.049), and all three seeds agree. Against the 0.7112 incumbent,
recomputed with the corrected frontier, it covers 0.609, 0.616 and 0.610 at 500
false cells against the incumbent's 0.539, and 0.680-0.686 at 1,000 against
0.645. That comparison is not like for like: the incumbent was trained on all
eleven recordings, these on nine.

**It is not ready to deploy, for three reasons:**

1. **It costs 2.6x the scoring time.** On an otherwise idle RTX 3060 Ti a
   scoring pass takes 310 ms at the median and 493 ms at the 95th percentile,
   against 120 ms and 164 ms for the baseline. The network is the same speed
   (57 ms against 51 ms). The extra time is cutting and sampling balls from
   about four times as many points on the CPU. At its 95th percentile it
   nearly fills the rig's half-second scoring interval.
2. **The weakest rock does not reliably improve.** Worst-rock coverage moved
   -0.134 to +0.091 across seeds. Rock 12, a small six-cell rock near the right
   edge, loses 12-18 points on every seed, and rock 7 loses 9-18. Rock 5 gains
   14-17 on every seed.
3. **Every number here is from the arena that has been used to choose
   everything else.**

The 0.5 m ball with 30 seconds of history (`fixed-r050__h5-30s`) is the runner-up
and is cheaper, at 182 ms median and 267 ms at the 95th percentile. Its
equal-area gain is as large (+0.097, from +0.086 to +0.108), but it cuts
almost no false ground at its stored threshold (-32 cells). The pictures show
why: it recovers the same sparse rocks, but lights up more open floor near the
far wall.

## What did not work

- **The adaptive radius.** With K = 256 the rule clips at its maximum most of
  the time. On all eleven recordings the 0.20-0.50 m version sits there 99% of
  the time without history, which makes it the fixed 0.5 m ball again. The
  0.20-0.75 m version is genuinely adaptive with history (43-57% at the
  maximum), but neither version improved the map at equal false area on
  three seeds (-0.017 and +0.014 at 500 cells). It is fast: 117-125 ms,
  because its balls are smaller.
- **Small balls.** At 0.20 m and 0.30 m, most rock candidates have too few
  neighbours to be scored at all: 74-100% of rock candidates were unscorable
  per recording for `fixed-r020__h1`. Over the full candidate set,
  `fixed-r020__h1` recalls 15% of rocks at 2-4 m. Among the candidates it could
  score, it recalls 77%. At 500 false cells (seed 42 only):
  - the 0.20 m settings scored 0.30-0.38;
  - the 0.30 m ball without history scored 0.41;
  - the 0.30 m ball with history scored 0.53-0.58, close to the baseline's
    0.529 but not clearly above it.

  These settings were not carried to more seeds.
- **A wider ball without history.** `fixed-r075__h1` was worse than the
  baseline on all three seeds at every budget (-0.014 at 500 cells), and it
  lost rocks 1, 12 and 13 badly in the pictures. The width only pays when
  history fills it.
- **Telling the model how high its candidate sits** (`+qz`). It was slightly
  worse on both leading settings at equal area, on all three seeds: -0.017 and
  -0.045 at 500 cells.
- **Telling the model how old each point is** (`+age`, a new sixth input
  channel). It made no difference at equal area (0.000 and -0.016). Worst-rock
  coverage rose by 0.027 and 0.013, but not on every seed.
- **Volleyball validation as a guide.** It ranked the settings differently from
  Lance again:
  - The top Volleyball scorer, adaptive 0.20-0.75 m with 4 s of history
    (+0.139), was flat on Lance.
  - Both feature follow-ups raised Volleyball scores and did nothing or harm
    on Lance.
  - The wider ball alone gained +0.058 on Volleyball and lost on Lance.
- **Measuring the startup period.** The 0-4 s and 4-30 s stages cannot be
  tested on this recording. Only one rock comes into view in its first
  46 seconds, and none in the first four. Every other rock is first seen after
  the 30-second history is already full, so the startup table rests on one
  rock and says nothing.

## What the review found, and what changed

An independent review found six problems. None affected training, and all six
are fixed:

1. **The equal-false-area table credited coverage only to rock 1.** It compared
   a rock's ID with the rock/clear/ignore *class* label. This bug predates this
   work and affected every `threshold-frontier.csv` in the project. All 19
   saved ones have been recomputed from their saved maps. Stored-threshold
   numbers were never affected.

   One claim in CLAUDE.md changes as a result. It says the incumbent "leads at
   every false-area budget down to 500 cells". Recomputed, it ties
   `clutter/cls-both-s44` at 1,000 cells (0.645 each) and trails it at 500
   (0.539 against 0.552).
2. **The recording viewer ignored the new input contract.** It now scores every
   checkpoint through the shared scorer, with its own causal sweep history.
3. **The live scorer refused sparse current sweeps.** It returned when the
   newest sweep had fewer than 20 points, even when older sweeps filled the
   balls. The minimum now applies per ball, after history is added.
4. **The stratified recall table dropped unscorable candidates.** It is now
   computed over the common candidate set, which is identical for every
   setting, and reports recall over all candidates and among scored ones side
   by side.
5. **The live scorer drew candidates and point samples from one random
   stream.** A wider ball changed which candidates the next pass picked. There
   are now three streams, and a test drives the real live entry point to prove
   it.
6. **Startup results never reached the summary.** They are now saved. An
   interval counts in a stage only if it lies wholly inside it, and the startup
   audit uses 2-second intervals so none straddles 4 s or 30 s.

The latency columns in the evaluation were also unfair: they were measured
three jobs at a time, and for history models they included re-decoding the
recording. `--phase latency` now measures each checkpoint alone and times only
what the robot pays. The table above uses it.

## Numbers

Three seeds (42, 43, 44), each paired with the baseline of the same seed.
"Coverage" is 3D attribution: the claimed cell lies on that rock's own
surface. Budget columns read the finished map at an oracle threshold chosen on
Lance and are diagnostics, not operating points. Full tables are in
[results.md](results.md) and [comparison.csv](comparison.csv).

| setting | coverage at 500 false cells | at 1,000 | false cells at stored threshold | worst rock | median / 95th pct ms |
|---|---|---|---|---|---|
| 0.5 m, current sweep (baseline) | 0.529 / 0.498 / 0.496 | 0.657 / 0.625 / 0.635 | 2,245 / 2,362 / 2,372 | 0.53 / 0.55 / 0.36 | 120 / 164 |
| 0.75 m, 8 s history | **+0.104** [+0.081, +0.118] | +0.043 [+0.028, +0.056] | **-412** [-539, -260] | -0.053 [-0.134, +0.091] | 310 / 493 |
| 0.5 m, 30 s history | **+0.097** [+0.086, +0.108] | +0.043 [+0.029, +0.068] | -32 [-67, -11] | -0.038 [-0.176, +0.107] | 182 / 267 |
| 0.5 m, 8 s history | +0.049 [+0.024, +0.066] | +0.011 [-0.023, +0.029] | -19 [-103, +114] | -0.070 [-0.166, +0.091] | 209 / 301 |
| 0.75 m, current sweep | -0.014 [-0.022, -0.005] | -0.033 [-0.059, -0.001] | -165 [-222, -95] | -0.127 [-0.176, -0.070] | 149 / 213 |
| adaptive 0.20-0.50 m, 8 s | -0.017 [-0.072, +0.014] | -0.022 [-0.069, +0.007] | +84 [+9, +167] | -0.176 [-0.273, 0.000] | 125 / 166 |
| adaptive 0.20-0.75 m, 4 s | +0.014 [-0.048, +0.046] | -0.011 [-0.030, +0.005] | -406 [-522, -290] | -0.116 [-0.182, 0.000] | 117 / 156 |

Differences are mean [min, max] over seeds. The deployed reference
checkpoint scores 0.418 at 500 cells, 0.573 at 1,000, with 2,211 false cells
and a worst rock of 0.294.

**Where the gain comes from: range.** Rock recall over every candidate, mean of
three seeds at stored thresholds:

| distance from robot | baseline | 0.75 m + 8 s | 0.5 m + 30 s |
|---|---|---|---|
| under 2 m | 0.87 | 0.88 | 0.93 |
| 2-4 m | 0.54 | 0.69 | 0.83 |
| 4-6 m | 0.02 | 0.56 | 0.40 |

Beyond 4 m a single sweep leaves almost every rock candidate unscorable or
unrecognised. Stacked sweeps fill those balls. The 0.75 m setting also flags
fewer clear candidates: 21% under 2 m, against the baseline's 36%. The 30 s
setting flags more: 44%.

**Stacking sweeps does not smear the floor.** Floor thickness grows from a
median of 3.1 to 3.3 cm on Volleyball and from 1.8 to 2.2-2.6 cm on Lance;
under 1% of patches show a double surface. Rock-surface thickness under
history was not measured.

## Pictures

I looked at all twelve selected cases for the two leading settings against the
fresh baseline, and read the per-rock tables for every setting. The same story
shows every time:

- **History recovers sparse rocks.** Rocks the baseline missed or barely touched
  with 17-90 returns in a five-second window are covered 60-100% with history:
  rocks 3, 5, 8, 9, 10 and 13.
- **It loses some thin edges.** Seen from one side, the 0.75 m + 8 s model is
  more cautious (stored threshold 0.91 against 0.82) and drops what the
  baseline half-covered: rocks 1, 6 and 12.
- **The false ground sits in two places in every model.** One is the bottom
  edge line of the arena (y about -4.1). The other is an open patch in front
  of the far wall at the upper right. History lights the second up much more
  densely, and the 30 s setting worst.

## Recommendation

1. **Retrain `fixed-r075__h5-8s` on all eleven recordings** with the per-run
   tail validation, as CLAUDE.md prescribes, and grade it against the incumbent
   with `mapeval`. Only then is the incumbent comparison fair.
2. **Make ball cutting faster before any deployment.** The network is not the
   cost. The next thing to measure is the 95th-percentile pass time on the
   robot's own computer.
3. **Look at rock 12.** It is the one rock every history setting makes worse.
4. **Stop exploring the adaptive radius and the two extra inputs.** K = 128 was
   the planned next adaptive test. It is still untested, but nothing here
   suggests the radius rule is the lever.
5. **Record a fresh competition-like run.** It is the only thing that would say
   whether any of this generalises.

## How it was run

Everything runs from the dashboard's **Neighborhood x history campaign** card
on the Train stage, or:

```
rocklabel-train nhcampaign --phase prepare
rocklabel-train nhcampaign --phase train --seed 42
rocklabel-train nhcampaign --phase train --seed 43 --seed 44 --arms <shortlist>
rocklabel-train nhcampaign --phase train --seed 42 --seed 43 --seed 44 --arms fixed-r075__h5-8s+qz fixed-r075__h5-8s+age fixed-r050__h5-30s+qz fixed-r050__h5-30s+age
rocklabel-train nhcampaign --phase evaluate --seed 42 --seed 43 --seed 44
rocklabel-train nhcampaign --phase latency
rocklabel-train nhcampaign --phase summarize
```

Every job's command, log and result are under `status/` and `logs/`. Every
input is hashed in `preflight.json`. Across 50 fits and 207 evaluation jobs,
none failed.
