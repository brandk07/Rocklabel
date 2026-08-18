# Agent 1 result — the layout agents 2 and 3 work inside

Everything in [01-cleanup.md](01-cleanup.md) is done. **Read this file, not that
one, for where things actually are** — that document guessed at names, this one
records them.

Nothing was regenerated and nothing was retrained. Both finished sweeps were
re-reported from their new homes and the numbers came back byte-identical
(checked on `summary.json` for both suites and `matched.json`).

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q` → **492 passed** (was 467;
25 new tests cover profiles, the training view and the model picker).

---

## 1. Where everything lives now

```
recordings/volleyball/raw/      13 VolleyBallTest*.mcap      ← agent 2's INPUT
recordings/volleyball/reslam/   13 VolleyBallTest*.reslam.mcap ← agent 2's OUTPUT
recordings/archive/comforter/   8 ComforterTest*.mcap
recordings/archive/myroom/      myroom1-6, myrun5
recordings/archive/misc/        garage, lance, lidar_*, backroom, run_0520, …

labels/volleyball/              16 label files (raw + .reslam)
labels/archive/{comforter,myroom,misc}/

datasets/full-sweep/volleyball/         11 runs, config hash 3ccba26a
datasets/raw-burst/VolleyBallTest*.reslam/   11 dirs, config hash a81b9c29
datasets/archive/{comforter-fused,comforter-raw,myroom}/

training/caches/full-sweep/     11 runs  ← the default cache
training/caches/raw-burst/      11 runs
training/caches/archive-comforter-fused/

training/experiments/fullsweep/<arm>/loro_<run>/       88 folds, complete
training/experiments/reflectivity/<arm>/loro_<run>/    121 folds, complete
training/experiments/seedstudy/<arm>/loro_<run>/       14 folds
training/experiments/compare/<model>_loro_<run>/       flat, from `compare`
training/experiments/compare-fused/<model>_loro_<run>/

training/reports/fullsweep/{summary.json,summary.md,*.png}
training/reports/fullsweep/matched/
training/reports/reflectivity/
training/reports/compare/
training/reports/reflect/
training/reports/archive-compare-fused/

training/exported/              deployable models — unchanged
```

**Old → new, for anything you have a path to:**

| was | is |
|---|---|
| `training/ablate/<suite>/` | `training/experiments/<suite>/` |
| `training/ablate_vb/fullsweep/` | `training/experiments/fullsweep/` |
| `training/runs/` | `training/experiments/compare/` |
| `training/runs_fused/` | `training/experiments/compare-fused/` |
| `training/cache/` | `training/caches/raw-burst/` |
| `training/cache_vb_fullsweep/` | `training/caches/full-sweep/` |
| `training/cache_fused/` | `training/caches/archive-comforter-fused/` |
| `training/results_ablate/<suite>/` | `training/reports/<suite>/` |
| `training/results_vb_fullsweep/` | `training/reports/fullsweep/` |
| `training/results_vb_matched/` | `training/reports/fullsweep/matched/` |
| `training/results/` | `training/reports/compare/` |
| `training/results_reflect/` | `training/reports/reflect/` |
| `datasets/vb_fullsweep/` | `datasets/full-sweep/volleyball/` |
| `datasets/VolleyBallTest*.reslam/` | `datasets/raw-burst/VolleyBallTest*.reslam/` |

The CLI defaults moved with them, so **plain `rocklabel-train ablate --suite X`
now writes to the right place with no flags**. Only pass `--ablate-root` /
`--cache-dir` if you want something other than the default.

## 2. Generation profiles — use `--profile`, not a config file

`config.yaml`, `config.vbsweep.yaml` and `config.fused.yaml` are **gone**. They
are replaced by a registry in `rocklabel/profiles.py`, shaped exactly like
`SUITES` in `rocklabel/train/ablate.py`.

```bash
rocklabel generate RUN.mcap --profile full-sweep      # --out defaults to
                                                      # datasets/full-sweep/RUN/
```

| profile | overrides | config hash |
|---|---|---|
| `full-sweep` **(default)** | `frame_window_s 0.05`, `frame_stride 4`, `segmentation_points 2048` | `3ccba26a` |
| `double-sweep` | `frame_window_s 0.1`, `frame_stride 2`, `segmentation_points 4096` | `92cc95cb` |
| `raw-burst` (legacy) | none — the built-in defaults | `a81b9c29` |
| `fused` (legacy) | `level.mode off`, `frame_window_s 0.05`, `frame_stride 4` | `26c0d01c` |

**`raw-burst` and `full-sweep` reproduce the exact fingerprints of the datasets
already on disk.** That is deliberate and it is tested
(`tests/test_profiles.py::test_a_profile_reproduces_the_datasets_already_on_disk`).
Changing those numbers orphans every existing dataset and cache. `double-sweep`
is the arm §4 of [03-training.md](03-training.md) asks for, pre-wired.

`config.example.yaml` stays as reference. A `--config` file still works, and a
`--profile` is applied *after* it, so the profile wins on the settings it owns.

The profile name is stored in the manifest **beside** the config, never inside
it — folding it into the hash would give two spellings of one setting two
different dataset directories. `rocklabel-train cache` refuses to pool two
profiles and names both in the error.

## 2b. The code moved too — `rocklabel/` is foldered now

22 loose modules became four folders named after the job the code does.
`cli.py` is still the only file that maps a command name to a function.

| was | is |
|---|---|
| `rocklabel/{mcap_io,lidarrig_io,pose,pipeline,inspect_cmd,trim}.py` | `rocklabel/recording/` |
| `rocklabel/{leveling,accumulate,relief}.py` | `rocklabel/geometry/` |
| `rocklabel/{generate,labeling,neighborhoods,bev}.py` | `rocklabel/dataset/` |
| `rocklabel/{viewer,camera,labeler,preview,driftcheck}.py` | `rocklabel/gui/` |
| `altslam/` | `rocklabel/slam/` — internals untouched |
| `altslam/tests/test_altslam.py` | `tests/test_slam.py` |

`cli.py`, `config.py`, `profiles.py` and `labels.py` stayed at the top level.

**Agent 2 — this affects you directly.** The solver is now
`python -m rocklabel.slam` (was `python -m altslam`), imports are
`rocklabel.slam.<module>`, and its 32 tests run with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_slam.py -q`. Nothing inside
it was restructured — only the folder it sits in and its import lines.
`rocklabel/slam/README.md` and `HANDOFF.md` both carry a note saying so.

Read [`rocklabel/README.md`](../rocklabel/README.md) before editing: it is the
map, and it states the four rules the layout depends on. Three of those rules
are now enforced by `tests/test_layout.py` — only `gui/` and `live/viz/` may
import Open3D at module level, torch stays inside `train/`, and the modules the
dashboard reads must import without torch. Break one and a test names the file.

Every folder under `training/` also has a README saying what is in it, starting
at [`training/README.md`](../training/README.md).

## 3. Adding a new profile or a new suite

Both are one entry in a dict, and both reach the dashboard automatically:

- **profile** → `rocklabel/profiles.py`, a `Profile(name, title, what, when, overrides)`.
- **suite** → `rocklabel/train/ablate.py`, an entry in `SUITES`.

`tests/test_profiles.py` validates every profile's overrides against the config
schema, so a typo fails in 0.05 s instead of overnight.

## 4. What the dashboard gained

Per `CLAUDE.md`, all of this is reachable from `rocklabel dash` — nothing was
left as a CLI-only feature.

- **Generate card**: a "How to cut frames" dropdown listing every profile with
  its full description. `--out` is now optional.
- **Model picker**: was 354 flat paths, now grouped `<optgroup>` by
  experiment → arm, sorted best-first, each entry annotated with the recording
  it held out and its PR-AUC. A "★ Best of each experiment" group sits at the
  top. `.superseded-*` runs are marked archived and hidden behind a
  "show archived runs" tick-box.
- **Training now panel** (Models view, above everything else): which experiment
  is running, fold *n* of *N*, a live per-epoch validation curve for the fold in
  flight, projected time left, and one chip per fold coloured
  done / running / pending / stalled. Reads only `history.csv`,
  `test_metrics.json` and `config.json`, so **it works for a sweep launched from
  a terminal** — which is how both long sweeps were actually run. Backed by
  `inventory.training_activity()`.
- **`--gpu-fraction`** on Train / Compare / Ablation sweep (replaces the old
  root-level `vb_run.py`). Set it when starting a job beside a running sweep.
- **A `slam` stage** ("Solve poses") exists in `spec.py`'s `STAGES`, between
  Triage and Label. **Agent 2: put the altslam commands there.**

## 5. Deleted (Brandon approved each)

- 16 `.superseded-*` run archives (284 MB)
- `datasets/DATASET_DIR`, `DATASET1_DIR`, `DATASET2`, `DATASET3` (1.3 GB)
- `training/results_v1_baseline` (3.2 MB)
- `training/runs/pointnet_loro_ConforterTest1` (the typo dir, one file)
- `tempCodeRunnerFile.py`, `test.py`, `vb_run.py`, `vb_watch.sh`
- the empty `out/` folder
- `config.yaml`, `config.vbsweep.yaml`, `config.fused.yaml`

**No recording and no label file was deleted.** All 13 raw
`VolleyBallTest*.mcap` are intact under `recordings/volleyball/raw/` for
agent 2.

## 6. Git

`training/experiments/` and `training/caches/` (~6 GB) are gitignored and the
1,607 previously-tracked training files were untracked with `git rm --cached` —
they are still on disk and still in Brandon's backup commits, they just stop
being committed. **Still tracked:** `training/exported/`, `training/reports/`
(~7 MB of conclusions), and every dataset `manifest.json`.

If you write a new experiment root, it is ignored automatically. If you write
report output somewhere other than `training/reports/`, it is **not** — put it
there.

## 7. Things worth knowing that bit me

- `pytest` needs `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` here (a ROS `launch_testing`
  plugin imports a missing `lark` and kills collection).
- `rocklabel/dashboard/spec.py` must stay torch-free; it imports from
  `rocklabel.train.cli` for the real path defaults, which is safe today.
  Check with `python3 -c "import sys, rocklabel.dashboard.spec; print('torch' in sys.modules)"`.
- `default_labels_path()` now *searches* `labels/` for an existing file before
  deciding where to write, then mirrors the recording's project folder. So
  labelling `recordings/volleyball/raw/X.mcap` reopens
  `labels/volleyball/X.labels.json` rather than starting an empty one.
- A folder is a dataset because it holds a `manifest.json`, not because of where
  it sits. `inventory.datasets()` walks for that.
- `datasets/archive/comforter-fused` has a config hash (`ac94194b`) that no
  profile reproduces — it predates the `level` config section entirely. That is
  expected; it is an archive, not something to rebuild.
