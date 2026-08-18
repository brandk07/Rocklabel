# `rocklabel/` — where the code lives

Folders are named after **the job the code does**, not the command that calls
it. `cli.py` is the only file that maps a command name to a function, so a
subcommand's implementation sits with the concern it belongs to rather than in
a commands bin.

Every folder's `__init__.py` explains that folder in more detail. This page is
the map.

```
rocklabel/
  cli.py            the `rocklabel` command: routes every subcommand
  config.py         built-in defaults <- YAML file <- CLI overrides
  profiles.py       named ways of cutting a recording into frames
  labels.py         the rock-label file format (load/save/versioning)

  recording/        getting points off disk
  geometry/         maths on a cloud — headless, knows nothing about rocks
  dataset/          labeled recording -> training data
  gui/              every Open3D window in the offline tool

  slam/             offline trajectory solver, run on its own
  live/             the live sensor rig: capture, view, record, score
  train/            the training stack (the only place torch is imported)
  dashboard/        the web UI that drives all of the above
```

## What goes where

| folder | holds | reach for it when |
|---|---|---|
| **`recording/`** | `mcap_io`, `lidarrig_io`, `pose`, `pipeline`, `inspect_cmd`, `trim` | you need points and poses out of an `.mcap`. Both recording formats are auto-detected here so nothing above ever has to ask which one it has. |
| **`geometry/`** | `leveling`, `accumulate`, `relief` | you need a cloud transformed, fused, or measured. No file reading, no windows, no labels. |
| **`dataset/`** | `generate`, `labeling`, `neighborhoods`, `bev` | you are turning hand-placed labels into training samples, or adding a dataset format. |
| **`gui/`** | `viewer`, `camera`, `labeler`, `preview`, `driftcheck` | you are touching a 3D window. Nothing outside this folder imports Open3D. |
| **`slam/`** | its own solver, registration and voxel map | you are improving where the sensor *was*. Self-contained; run it with `python -m rocklabel.slam`. |
| **`live/`** | sources, surfaces, viz, webui | you are working with the sensor in real time. |
| **`train/`** | data, engine, models, ablate, matched, plots, export | you are training or evaluating. **The only place that imports torch.** |
| **`dashboard/`** | `spec`, `inventory`, `server`, `jobs`, static files | you are changing what the web UI can do. |

## Four rules worth knowing before you edit

**1. `cli.py` routes, it does not implement.** Each subcommand's work lives with
its concern — `rocklabel trim` in `recording/`, `rocklabel generate` in
`dataset/`, `rocklabel label` in `gui/`. Adding a subcommand means adding a
parser entry in `cli.py` and a module in the right folder.

**2. Any CLI change must land in the dashboard in the same piece of work.**
`dashboard/spec.py` generates the run form, the help text, the validation and
the command preview from one entry per flag. `tests/test_dashboard.py` compares
that catalog against the real argument parsers and fails if they drift apart.
This is not optional — see `CLAUDE.md`.

**3. `train/` and `dashboard/` must stay importable without torch.** The
dashboard reads the real training defaults, the real ablation suites and the
real default paths straight out of `train/`, which is what keeps them from
going stale — and it can only do that because those modules are torch-free by
construction. Check it with:

```bash
python3 -c "import sys, rocklabel.dashboard.spec; print('torch' in sys.modules)"
```

**4. Only `gui/` and `live/viz/` import Open3D.** Everything else has to run on
a machine with no display: the generator, the training stack and the dashboard
all do. Where a headless path needs a window, the import is deferred inside the
function rather than done at the top of the file.

## Running the tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q
```

The environment variable is needed here: a ROS `launch_testing` plugin tries to
import a missing library and kills test collection before anything runs. It has
nothing to do with this project.

## Where the data lives

Code is here; everything the code produces is described in
[`../training/README.md`](../training/README.md) and
[`../DOCS.md`](../DOCS.md).
