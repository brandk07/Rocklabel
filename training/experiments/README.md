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

## What's here now

| experiment | folds | what it was asking |
|---|---|---|
| `fullsweep/` | 88, done | Does building frames from whole sensor rotations beat single raw bursts, and can a per-point segmenter compete with the sliding-window classifiers? **Yes to both.** |
| `reflectivity/` | 121, done | Does the LiDAR brightness channel earn its place beside shape? **No.** |
| `seedstudy/` | 14 | How far apart do two runs of the *identical* setting land? This is the yardstick that says whether any other difference is real. |
| `compare/` | flat | Output of `rocklabel-train compare` — plain per-model folds, named `<model>_loro_<run>` instead of the setting/fold nesting a sweep uses. |
| `compare-fused/` | flat | The same, for the old Comforter recordings. |

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
