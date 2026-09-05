# Reading the ground as a picture

*Sweep run overnight 3-4 September 2026. Eight settings, six recordings each,
44 trained models. Every number that matters here is measured on the
competition arena, which none of them was trained on.*

## The one-paragraph version

A model that draws the ground as a top-down picture and reads it with a
convolutional network **can be made to work, and one setting scores better on
the competition arena than the classifier we deploy** — but it still finds
fewer rocks, so nothing changes about what to deploy. The useful finding is
elsewhere: **how much a rock is allowed to outweigh bare ground while training
decides whether these models work at all.** Left at the raw imbalance, more
than half the trained models go blank on the arena. Capped, none of them do.
That setting had never been tried, and the same knob is set the same wrong way
for the whole-frame segmenter.

## What to beat, and whether anything did

The bar is the sliding-window classifier trained against stray returns
(`training/experiments/deploy/cls-stray/trainall/best.pt`). Scored on 100
arena frames carrying 112 rock sightings, on a shared 10 cm ground grid so
every model becomes one measurement:

| setting | what it changes | rocks found | rocks at equal cost | confidence gap | grid score | blank runs |
|---|---|---|---|---|---|---|
| **the deployed classifier** | — | **109/112** | **109/112** | **+0.90** | 0.540 | 0 of 1 |
| `bev-capped-relative` | capped weight + scale-free density | 73% | 83% | +0.81 ± 0.22 | **0.604 ± 0.068** | **0 of 6** |
| `bev-capped-nostray` | capped weight only | 68% | 79% | +0.81 ± 0.29 | 0.593 ± 0.048 | **0 of 6** |
| `bev-relative` | scale-free density only | 36% | 47% | +0.61 ± 0.30 | 0.272 ± 0.200 | **0 of 6** |
| `bev-base` | nothing (the control) | 32% | 72% | +0.47 ± 0.43 | 0.385 ± 0.165 | 3 of 6 |
| `bev-base-s43` | the control, different seed | 20% | 51% | +0.32 ± 0.39 | 0.315 ± 0.137 | 4 of 6 |
| `bev-local` | half the sight line | 23% | 58% | +0.35 ± 0.36 | 0.353 ± 0.125 | 3 of 6 |
| `bev-stray` | clutter training | 4% | 49% | +0.02 ± 0.02 | 0.267 ± 0.081 | 6 of 6 |
| `bev-capped` | clutter + capped weight | 8% | 42% | +0.01 ± 0.01 | 0.297 ± 0.082 | 2 of 2 |

"Blank" means the gap between the confidence a model puts on real rock and on
bare ground fell below 0.20. A blank model still ranks rocks above ground
correctly; it just emits so little confidence that nothing lights up on
screen. Every score this project tracked before September is blind to it.

**The best grid setting beats the classifier on the grid score (0.604 against
0.540) and loses badly on rocks found (83% against 97%).** Rocks found is what
a person watching the live view sees, so the classifier stays deployed.

## The finding worth keeping

**The loss weighting decides everything.** Training weights a rock example
against a clear one by the raw imbalance in the data. For any model that
labels every point that is about **93**; for the sliding-window classifier it
is about **4.3**, and the whole difference comes from the dataset builder
throwing away 95% of clear candidates for one format and none for the other.

| | blank runs |
|---|---|
| every setting left at the raw weight of 93 | **10 of 18** |
| every setting with the weight capped at 10 (clutter training off) | **0 of 12** |

Nothing else came close to mattering this much. Halving how far the network
can see changed nothing (3 of 6 blank, same as the control). Making the
density channel scale-free also stopped the blanking, but left the model
firing everywhere on half its folds. Capping the weight is the change that
makes this family behave.

**This is worth testing on the segmenter.** The segmenter carries the same ~99
weight and shows the same symptom — ranking rocks correctly on the arena while
its confidence collapses, swinging from +0.02 to +0.96 across checkpoints of
one setting. That knob was listed as untested in
[WHY-SEGMENTATION-IS-NOT-A-BASELINE.md](../stray/WHY-SEGMENTATION-IS-NOT-A-BASELINE.md);
it is no longer untested for grid models, and it was decisive.

## Two things that will mislead you if you skip them

**The volleyball scores say nothing.** Every setting lands between 0.471 and
0.509 on the leave-one-recording-out folds — a spread of 0.038, well inside
the noise. On the arena the same settings range from every model working to
every model blank. The setting that scores *worst* at home (`bev-stray`,
0.471) and the one that scores best on the arena (`bev-capped-relative`,
0.501) are 0.03 apart at home and are the difference between 6-of-6 blank and
0-of-6 blank away from it. A leave-one-out table cannot see this, by
construction.

**The random seed alone can produce the whole range.** Same setting, same
held-out recording, only the seed differs:

| recording | seed 42 | seed 43 |
|---|---|---|
| VolleyBallTest2 | 80 rocks, gap +0.990 | 2 rocks, gap +0.000 |
| VolleyBallTest9 | 74 rocks, gap +0.928 | 8 rocks, gap +0.000 |
| VolleyBallTest11 | 3 rocks, gap +0.023 | 70 rocks, gap +0.893 |

So a single trained model proves nothing about a setting. Across both seeds
the uncapped control goes blank on 7 of 12 checkpoints; against that rate,
getting 0 blank in 6 — which both capped settings did — is a 1-in-190 result
(p = 0.005). That is why the capped finding is believable and why every
single-checkpoint reading taken during the night was not.

## Clutter training does the opposite here

Turning 5% of each training frame into returns that sit on no surface is worth
+0.093 on the arena for the classifier and removes all of its blank runs. For
the grid it took the control from 3-of-6 blank to **6 of 6**, and capping the
loss weight did not rescue it (2 of 2 blank). A plausible reason: in a
top-down picture a single stray return **owns its cell** and sets that cell's
height outright, where in a point model it is one point among dozens inside a
half-metre ball. The same augmentation that teaches one model to ignore bad
returns teaches the other to distrust height everywhere.

## What this does not settle

- **One arena.** Everything rests on a single competition recording, already
  named as the weakest part of the segmentation write-up's evidence. A second
  arena is worth more than this entire sweep.
- **Six recordings a setting, not eleven.** Enough to see a family-wide
  pattern; not enough for a significance test on any single pair.
- **The cap was tried at one value.** Ten, against a raw 93. Nothing says ten
  is right, and nothing was tried between.
- **`bev-capped` stopped at two folds.** It had already answered its question
  (clutter training is not rescued by capping) and the time was better spent.

## Reproducing it

```
rocklabel-train ablate --suite bev          # trains everything above
```

The arena scoring is not a command yet — it lives in the sweep's scratch
script and reuses the recipe from `training/reports/stray/`. `lance_eval.txt`
next to this file is its raw output. Volleyball numbers, which are not the
point, are in `summary.json`.
