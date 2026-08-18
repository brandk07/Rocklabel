# `training/reports/` — what each experiment concluded

One folder per experiment. This is the readable end of the pipeline: no models
here, just the figures and tables. **Start with `summary.md`.**

```
reports/<experiment>/
  summary.md      the write-up — read this one
  summary.json    the same numbers, for anything that wants to read them back
  arm_ranking.png     every setting, ranked
  paired_deltas.png   setting vs setting, recording by recording
  per_fold.png        how each recording scored
```

## What's here now

| folder | from |
|---|---|
| `fullsweep/` | the full-sweep + segmentation sweep (88 folds) |
| `fullsweep/matched/` | the segmenter and the classifier re-scored on one shared set of spots, so their numbers can honestly be compared |
| `reflectivity/` | the brightness-channel sweep (121 folds) |
| `reflect/` | the quick brightness check that reads the cache directly and trains nothing |
| `compare/` | `rocklabel-train compare` output — comparison bars, ROC/PR curves, confusion matrices |
| `archive-compare-fused/` | the same for the old Comforter recordings |

## Regenerating

Cheap and safe, and it trains nothing — it only re-reads the folds already on
disk. Safe to run while a sweep is still going:

```bash
rocklabel-train ablate --suite fullsweep --report-only
```

Or tick **"Report only"** on the Ablation sweep card in the dashboard.

## Why this folder is in git when the others are not

It is about 7 MB, and it is the only durable record of what was found. Someone
cloning this repository gets the conclusions without needing 6 GB of
checkpoints or an overnight retrain.

## Reading a paired comparison

Every table here compares two settings **recording by recording**, never as two
overall averages. Which recording gets held out swings the score far more than
any setting does — the folds range from about 0.42 to 0.93 — so an averaged
comparison drowns the thing being measured. `summary.md` gives you the
per-recording difference, a win/loss count, and a significance test.

And check the **noise floor** quoted at the top: that is the gap between two
runs of the *same* setting with only the random seed changed. Anything smaller
than that is noise, not a finding.
