# Reading the map instead of the sweep

September 2026. Graded on the Lance competition recording, which has driven
model selection many times over and is **development data, not a test set**.
Every number that uses arena labels comes from a spatial hold-out inside that
one recording. How the model works and exactly how it was graded:
[HOW-IT-WORKS.md](HOW-IT-WORKS.md). Every table: [results.md](results.md).

## The answer

**A model that reads the robot's accumulated map, trained with arena labels,
keeps the arena map clean in a way no earlier model came close to.** Below,
every 10 cm cell of the arena is graded by a model that never saw a label on
that ground (a cross-validated map):

| wrongly-claimed cells allowed | map model | map model + history classifier, averaged | best earlier model | earlier model given the same arena labels |
|---|---|---|---|---|
| 1,000 | 0.68 | **0.76** | 0.69 | 0.56-0.66 |
| 500 | 0.64 | **0.71** | 0.63 | 0.43-0.54 |
| 200 | 0.60 | **0.62** | 0.50 | 0.29-0.39 |
| 100 | **0.58** | **0.58** | 0.42 | 0.21-0.30 |
| 50 | **0.55** | **0.55** | 0.32 | 0.15-0.23 |

Mean 3D rock coverage; the map model is three training seeds averaged. "Best
earlier model" takes the best of twelve earlier maps separately at each
budget, which flatters it. Single seeds land within 0.03 of each other at 100
false cells and above (0.07 at 50); the gain at 100 and 200 false cells is
0.10-0.16.

- **At its default threshold of 0.5 it claims 42 wrongly-claimed cells**, and
  between 29 and 73 at every point of the 35 minutes when graded as the map
  grew. Every earlier model claims 1,800-2,400 at its own stored threshold.
  The price is coverage at that threshold: 0.54 of each rock's surface and
  0.65 of its footprint, against their 0.65-0.72 and 0.98-0.99 - they claim
  nearly every rock cell because they claim nearly every cell. A lower
  threshold trades back: at 200 false cells the map model claims 0.73 of the
  footprint.
- **Eight rocks are covered at 0.75-0.90 with 100 false cells allowed** (1, 3,
  5, 6, 7, 8, 9, 11), against 0.37-0.64 for the best earlier map.
- **The gain is from reading the map, not from the arena labels.** The
  single-sweep classifier given the same arena labels got no better than
  without them (0.21-0.30 at 100 false cells over three seeds). On the rough,
  dug-over east side it covers 0.31-0.35 there; the map model 0.63.
- **More arena labels helped.** Trained on eight or nine arena rocks instead
  of six (a four-way split), a single seed reaches 0.72 / 0.68 / 0.63 / 0.56 /
  0.49 at the five budgets - ahead of every earlier model at every budget,
  1,000 included.
- **Averaged with the 0.5 m + 30 s history classifier** from the last campaign
  it is the best at every budget. The history classifier still partly finds
  the three rocks the map model misses; the map model supplies the clean
  ground.

**It is fast.** Adding a sweep to the map costs 0.17 ms. A whole-arena pass
costs 29 ms on the RTX 3060 Ti (31 ms at the 95th percentile) and 410 ms on
four CPU threads. The history classifier costs 182 ms a pass, the baseline
120 ms.

**Training**: 25 seconds an epoch, 18 minutes for the 40-epoch runs used here.

## What did not work

- **Volleyball alone.** Trained only on the volleyball runs it covers 0.47 at
  500 false cells and 0.26 at 100 - worse than every earlier model except the
  deployed one (0.50-0.63 and 0.19-0.42). The volleyball rocks are small and
  round, the court has no walls, and the model is badly under-confident on the
  arena (the threshold that allows 500 false cells is 0.006). Dropping
  brightness (0.36 at 500) and stretching rock heights (0.14) made it worse.
- **Three rocks.** Rock 13 is a sharp-cornered, flat-topped block 50-60 cm
  tall, unlike any training rock; rock 4 barely rises off the ground (90% of
  its returns are within 7 cm of it); rock 12 stands against the east wall.
  All three are near zero at 100 false cells, and they are why the map model
  alone only ties the earlier models past about 400 false cells.
- **The plain channels.** Heights, spread and brightness without the height
  profile (the share of each cell's returns per 2.5 cm slice) were far behind
  from the first epochs.
- **Rock transplants** (real rocks' returns pasted onto other floors): a little
  better at 500 false cells, 0.04-0.14 worse at 100, and three times the false
  ground on one fold.
- **Labelling the arena walls as clear ground**: 0.41 against 0.55 at 100
  false cells (seed 42). It made tall rocks near walls harder. Walls are better
  masked with the arena outline, which the robot already publishes on
  `/arena_zones`.
- **Taller training rocks, rotation averaging, epoch averaging and longer
  training**: no gain. The arena score peaks within five to eight epochs.

## What it looks like

- [frontier-3d.png](frontier-3d.png): coverage against false area for the map
  model and the leading earlier models;
  [frontier-footprint.png](frontier-footprint.png) the same for occupied
  footprint.
- [timeline.png](timeline.png): false ground over the 35 minutes. The earlier
  models climb to about 2,000 cells; the map model stays under 75.
- [stitched-3seeds/map-at100.png](stitched-3seeds/map-at100.png): the
  cross-validated map at 100 false cells. The false ground that remains sits on
  a real smooth mound, one small object, beside rock 3 and in a thin halo
  around rocks 5, 10 and 11 - the model draws rocks a little larger than their
  bumps.
- [view/](view/): the deployment model (volleyball plus the whole arena) on the
  12 and 14 May arena recordings, which nobody has labelled. Inside each arena
  it marks only the obvious raised objects; outside, it lights up the walls.

## Caveats

1. **One arena, one session.** The hold-outs never grade a model on ground it
   trained on, but the held-out ground was recorded the same day, by the same
   robot, under the same SLAM. A new arena will do worse.
2. **The control's arena labels come in a different form.** The map model
   learns from labelled ground; the per-ball control learns from the labelled
   candidate cache (109 frames, clear candidates thinned to a quarter). Both see
   the same rocks and the same ground.
3. **Coverage has two meanings.** 3D attribution (quoted above) and occupied
   footprint are both in [results.md](results.md); the map model leads on both
   at 200 false cells and below.
4. **The three hard rocks and the walls** were diagnosed by looking at this
   arena's held-out errors. Nothing in the quoted numbers was tuned to them.

## What to do next

1. **Label the 12 and 14 May arena recordings** (a few rocks each). That gives
   a second arena to grade on - the only honest test of whether any of this
   generalises - and more arena rocks to train on, which already helped.
2. **Run it on the robot.** It needs the map the robot already keeps, the
   arena outline and 30 ms of GPU a pass. The deployment checkpoint is
   `training/experiments/mapnet-v1/deploy-plain-s42/final.pt` (volleyball plus
   the whole arena; its arena numbers are in-sample and measure nothing).
3. **Keep the history classifier alongside** if tall, box-like obstacles
   matter; averaged, the two are the best at every budget.

## On the dashboard

Five new cards, all `rocklabel-train` commands:

| card | stage | what it does |
|---|---|---|
| **Map model: stack a recording** (`mapnet-cloud`) | Dataset | decode a recording once into the stacked cloud the map model uses |
| **Map model: train** (`mapnet-train`) | Train | train it; "Arena rock group" or the four-way hold-out uses arena labels, "Deployment fit" trains on the whole arena |
| **Map model: grade on the arena** (`mapnet-eval`) | Deploy | the `mapeval`-style grade, pictures and a timeline, with every earlier model re-graded beside it |
| **Map model: stitch folds** (`mapnet-stitch`) | Deploy | one cross-validated arena map from the per-group models |
| **Map model: look at a recording** (`mapnet-view`) | Deploy | draw where a model thinks the rocks are on any recording, labelled or not |

The stacked recordings (`training/caches/mapnet-v1/clouds/` and
`.../unlabelled/`) and trained runs (`training/experiments/mapnet-v1/`) appear
in the pickers, and this report's figures on the Overview page under "Map
model".

## How it was run

```
rocklabel-train mapnet-cloud --recording RUN.mcap --labels RUN.labels.json --out training/caches/mapnet-v1/clouds/NAME.npz
rocklabel-train mapnet-train --out training/experiments/mapnet-v1/RUN --train VB2 ... VB13 --val --lance-fold A --seed 42
rocklabel-train mapnet-stitch --a A_s42.pt A_s43.pt A_s44.pt --b B_s42.pt B_s43.pt B_s44.pt --out training/reports/mapnet-v1/stitched-3seeds --timeline 60
```

The tables come from `training/experiments/mapnet-v1/final_tables.sh`, the
per-ball control from `python -m rocklabel.train.mapnet.ball_control`. Every
run's settings are in `training/experiments/mapnet-v1/<run>/config.json` and
its per-epoch arena numbers in `<run>/log.csv`.
