# Agent 1 — Clean up and build the foundation

**Read [README.md](README.md) first.** You are the most important of the three agents:
the other two work inside whatever structure you leave behind.

Your job is not cosmetic. Right now three separate agents have each guessed at their own
naming, and the result is that Brandon cannot tell what is what — and neither can the
agents. Two of them independently misread the same data. Fix the structure so that
confusion is not possible.

## The mess, measured

| what | state |
|---|---|
| Root-level files | `test.py`, `tempCodeRunnerFile.py`, `vb_run.py`, `vb_watch.sh` and **four** config files (`config.yaml`, `config.example.yaml`, `config.fused.yaml`, `config.vbsweep.yaml`) |
| `recordings/` | 52 mcap files, 32 GB, four unrelated projects mixed together (VolleyBall, Comforter, myroom, garage/lance/lidar_*), plus a stray `.labels.json` |
| `datasets/` | 29 directories including `DATASET_DIR`, `DATASET1_DIR`, `DATASET2`, `DATASET3` |
| `training/` | 15 sibling roots — `cache`, `cache_fused`, `cache_vb_fullsweep`, `runs`, `runs_fused`, `ablate`, `ablate_vb`, `results`, `results_ablate`, `results_fused`, `results_reflect`, `results_v1_baseline`, `results_vb_fullsweep`, `results_vb_matched`, `exported` — plus 5 loose `.log` files |
| Model picker | **354 entries, flat and unsorted**, including `.superseded-*` archives. 289 real checkpoints across 4 run roots. |

## Hard constraints — read before deleting anything

1. **Delete no recording and no label file.** `recordings/` is 32 GB and the raw
   `VolleyBallTest*.mcap` (non-`.reslam`) files are **the SLAM agent's input** — it must
   re-solve them. Labels represent hours of Brandon's manual work. Archive and reorganise;
   do not remove. Disk is fine (344 GB free).
2. **Move, don't rewrite, existing results.** The finished sweeps
   (`training/ablate/` = 121 folds, `training/ablate_vb/` = 88 folds) are the baseline the
   next suite is measured against. They must remain readable and comparable after the move.
   Migrate them into the new layout and verify the reports still render.
3. **Ask before destructive consolidation.** If you believe something should actually be
   deleted (e.g. `.superseded-*` archives, `results_v1_baseline`), list it and ask. Do not
   assume.
4. **`altslam/` is self-contained and was deliberately kept isolated.** Coordinate with
   [02-slam.md](02-slam.md): it now needs a dashboard home, but do not restructure its
   internals.

## What to build

### 1. Generation profiles — a first-class, named concept

This is the highest-value single change, and Brandon asked for it directly. There are now
genuinely different ways to build a dataset, and they are currently expressed as loose
config files nobody can tell apart:

- **raw bursts** — `frame_window_s: 0.0`. One ~4 ms sensor batch per frame, ~110 points in
  the crop box. What `config.yaml` does. **Known to be a bad default** — it starves the
  models and makes segmentation impossible.
- **full sweep** — `frame_window_s: 0.05`. One whole 20 Hz rotation, ~1,250 points.
  What `config.vbsweep.yaml` does. **Should become the default.**
- **fused** — `config.fused.yaml`, an older variant used for the Comforter runs.
- future ones — e.g. multi-rotation windows, which [03-training.md](03-training.md) wants
  to test.

Build a **profile registry** modelled on the existing `SUITES` dict in
`rocklabel/train/ablate.py` — that pattern already works and the dashboard already reads
it. Each profile carries a name, a plain-English description of what it does and when to
use it, and its generator overrides. Then:

- `rocklabel generate --profile full-sweep` replaces hand-managed config files.
- The dashboard's Generate card gets a **profile dropdown** with the descriptions shown,
  so Brandon can pick "old way" vs "full sweep" vs future variants without editing YAML.
- The profile name goes **into the dataset manifest and into the directory name**, so a
  dataset says on its face how it was built.
- Keep the raw config-file path working for one-offs, but make profiles the front door.

Collapse the four root config files into this. Keep `config.example.yaml` as documentation.

### 2. A data layout that encodes provenance

Adopt something like the following — the exact spelling is yours, but these properties are
required: **project separated from project, generation profile visible in the path, and
every derived artifact traceable to what produced it.**

```
recordings/volleyball/{raw,reslam}/     # raw = SLAM agent's input, never delete
recordings/archive/{comforter,myroom,misc}/
labels/volleyball/
datasets/<profile>/<run_id>/            # e.g. datasets/full-sweep/VolleyBallTest4.reslam/
training/caches/<profile>/
training/experiments/<experiment>/<arm>/<fold>/
training/reports/<experiment>/
```

A cache must record which profile and which dataset it came from, and training must refuse
to mix profiles — `build_cache` already refuses mixed config hashes; extend that to a clear
message naming the profile.

### 3. Fix the model picker — 354 flat entries is unusable

Brandon named this explicitly. In `rocklabel/dashboard/inventory.py`, `checkpoints()`
currently returns a flat list including `.superseded-*` directories. Make it return
**grouped, sorted, annotated** entries:

- Group by experiment → arm → fold, so the run that produced a model is visible.
- Hide `.superseded-*` by default behind a "show archived" toggle.
- Annotate each entry with its held-out recording, its test PR-AUC and its date, read from
  the `test_metrics.json` that already sits beside every checkpoint.
- Sort best-first within a group, and surface a "best of this experiment" shortcut — a
  cheap win, since Brandon usually wants the good one.
- Update `sourceOptions()` in `rocklabel/dashboard/static/app.js`, which currently flattens
  everything into one `<option>` list; it needs `<optgroup>` support.

### 4. A live training view

Brandon asked to see what is training and how far along it is. Everything needed is already
on disk and nothing needs to be added to the training loop:

- `history.csv` is written every epoch (epoch, train/val loss, val PR-AUC, learning rate).
- `test_metrics.json` appears when a fold finishes.
- A sweep prints `=== [n/total] arm / fold ===` to its log.

Build a dashboard panel that reads those and shows: which experiment is running, fold *n*
of *N*, a per-epoch validation curve for the fold in flight, elapsed and projected time,
and which folds are done/pending/failed. `inventory.ablations()` already does something
like this for progress — extend rather than duplicate. Make it work for a sweep launched
outside the dashboard too, since that is how both long sweeps were actually run.

### 5. Housekeeping

- Delete `tempCodeRunnerFile.py`; fold `test.py` into `tests/` or remove it.
- `vb_run.py` caps this process's GPU share so a second sweep cannot push a running one
  into an out-of-memory crash — that is genuinely useful with multiple agents on one
  machine. **Promote it into the CLI as a `--gpu-fraction` flag** rather than leaving a
  script in the root. `vb_watch.sh` has served its purpose; drop it or fold it into the
  training view above.
- Move the loose `training/*.log` files under the experiment they belong to.
- `rocklabel/train/engine.py:362` references a `report.py` that never existed; the function
  it describes now lives in `rocklabel/train/matched.py`. Fix the comment.

## Definition of done

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q` passes (currently **467 tests**),
  including `tests/test_dashboard.py`, which fails if the dashboard catalog drifts from the
  real argument parsers.
- Both existing sweeps still report correctly from their new locations: regenerate the
  reflectivity report and the full-sweep report and confirm the numbers are unchanged.
- A fresh agent can open the dashboard and answer, without reading code: what recordings
  exist, how each dataset was generated, what is training right now, and which model is the
  best one for a given recording.
- Write `handoff/01-cleanup-RESULT.md` describing the final layout and anything you moved,
  so agents 2 and 3 are not working from this document's guesses.
