# `training/experiments/` — every model ever trained here

One folder per experiment. Inside, one folder per trained fold, holding the
model itself and the record of how it did.

```
experiments/<experiment>/<setting>/loro_<held-out recording>/
  config.json        exactly the settings this fold was trained with
  history.csv        one row per pass over the data (loss, score, learning rate)
  best.pt            the model — this is the file you load
  last.pt            resume point only; it will not load as a model
  test_metrics.json  how it scored on the recording it never saw
  predictions.npz    its raw per-sample scores, for re-scoring later
```

`loro` stands for **leave-one-run-out**: each fold trains on every recording
except one, then is graded on that one. Splitting samples randomly instead
would put near-identical frames on both sides and produce meaningless scores.

## Read this before drawing a conclusion from any folder here

Three traps have each cost this project real GPU time. All are cheap to avoid.

**1. A leave-one-recording-out score cannot see whether a model survives a new
arena.** In the `bev/` sweep every setting landed between 0.471 and 0.509 on
the volleyball folds — a spread well inside the noise — while the same
settings ranged from *every* checkpoint working to *every* checkpoint going
blank on the competition recording. The setting that scored worst at home was
the one that failed hardest away from it. If a result is meant to say
something about deployment, it has to be measured on the competition
recording. The recipe is in
[../reports/stray/WHY-SEGMENTATION-IS-NOT-A-BASELINE.md](../reports/stray/WHY-SEGMENTATION-IS-NOT-A-BASELINE.md)
and the script is `../reports/bev/bev_lance.py`.

**2. How much a rock outweighs bare ground in the loss decides whether a
per-point model works at all.** Training weights a rock example by the raw
class imbalance: about **93–99** for anything labelling every point, about
**4.3** for the sliding-window classifier, and the whole gap comes from the
dataset builder discarding 95% of clear candidates for one format and none for
the other. Measured on the `bev/` sweep, leaving it raw sent **10 of 18**
checkpoints blank on the arena — ranking rocks correctly while emitting almost
no confidence — and capping it at 10 sent **0 of 12**. Set it with
`--pos-weight-cap` (Advanced, on the Train / Compare / Ablation cards) for a
`bev_cnn` run. **It does not carry to the whole-frame segmenter** — tried there
on three folds it sent 2 of 3 checkpoints blank where none had been before. It
does not apply to the classifier either, whose weight is already small. Why the
same knob helps one per-point family and hurts the other is open.

**3. One trained model tells you almost nothing.** Same setting, same held-out
recording, different random seed: 80 rocks found at +0.990 confidence gap
versus 2 rocks at +0.000. Score several checkpoints of a setting before
believing anything about it.

## What's here now

| experiment | folds | what it was asking |
|---|---|---|
| `fullsweep/` | 88, done | Does building frames from whole sensor rotations beat single raw bursts, and can a per-point segmenter compete with the sliding-window classifiers? **Yes to the first.** The second was read as "yes" for a year and is now known to be wrong away from home — see the warning above. |
| `reflectivity/` | 121, done | Does the LiDAR brightness channel earn its place beside shape? **No.** |
| `segdense/` | 48 planned (12 folds x 4 settings) | The whole-frame segmenter fed properly: every sensor rotation kept instead of every fourth, 60 epochs instead of 30, and a setting that looks at 10 cm scales instead of 25 cm. The one experiment with twelve folds — VolleyBallTest13 joins as a fold here. |
| `seedstudy/` | 14 | How far apart do two runs of the *identical* setting land? This is the yardstick that says whether any other difference is real. |
| `compare/` | flat | Output of `rocklabel-train compare` — plain per-model folds, named `<model>_loro_<run>` instead of the setting/fold nesting a sweep uses. |
| `compare-fused/` | flat | The same, for the old Comforter recordings. |
| `stray/` | 26, done | Does training against returns that sit on no surface survive a change of arena? **Yes for the classifier** (+0.093 on the competition recording, ahead on 11 of 11 folds). Also holds `seg-capped`, which asks whether the segmenter's loss weight is what breaks it. |
| `bev/` | 44, done | A convolutional network reading the frame as a top-down picture instead of a set of points. **Does not beat the deployed classifier** on rocks found (83% against 97%), though it scores higher on the shared grid. Its real result is the loss-weight finding above. |

## Two shapes of folder, on purpose

A **sweep** (`ablate`) gives every setting its own folder, so two settings that
differ only in an augmentation value cannot land on the same directory and
overwrite each other. A **compare** run predates that and keeps flat
`<model>_loro_<run>` names. Both are read by the dashboard's model picker,
which groups them the same way regardless.

## Folders ending in `.superseded-<timestamp>`

A fold that was retrained with different settings gets its old directory moved
aside rather than deleted, so the previous result is not silently lost. The
dashboard marks these **archived** and hides them behind a tick-box. They are
safe to delete once you no longer care what the old settings scored.

## Safe to delete

The whole folder is rebuildable, which is why it is not in git — but rebuilding
it means re-running the training, which for the two big sweeps was an overnight
job each. Delete individual folds freely; think before deleting an experiment.
