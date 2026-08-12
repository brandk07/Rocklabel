# rocklabel

Offline LiDAR rock labeling and training-dataset generation for the Lunabotics
perception stack. Label rock positions **once per recording** on the fused
odom-frame point cloud, then automatically generate labeled training samples
from **every frame** of the recording.

Works on two recording formats, **auto-detected** — just point any command at
the file:

- **ROS 2 rosbag2 mcaps** (`sensor_msgs/msg/PointCloud2` + `/tf`), e.g. the
  competition robot logs. No ROS 2 installation required (messages are decoded
  from the schemas embedded in the mcap file).
- **Native lidarrig recordings** (`/lidar/frames`), from the handheld
  test rig. Each frame already carries the world pose its SLAM/IMU pipeline
  computed, plus per-point **reflectivity (RSSI)** — no TF, no `topics:`
  config needed at all.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # from this directory
pip install -e '.[dev]'     # + pytest, if you want to run the tests
pip install -e '.[dash]'    # + flask, for the web dashboard
```

Dependencies (installed automatically): `numpy`, `scipy`, `open3d`, `pyyaml`,
`tqdm`, `mcap`, `mcap-ros2-support`.

## `rocklabel dash` — the dashboard

```bash
rocklabel dash              # opens http://localhost:8765 in your browser
```

Everything below can also be driven from one page: a **Commands** view with a
generated form per command (with what/why explanations, tooltips per flag, and
the exact command line shown before it runs), a **Live LiDAR** view that says
whether the sensor is actually streaming, and **Data** / **Models** views over
every recording, label file, dataset and training run in the project.

It is a control surface, not a second implementation — each button shells out
to the same `rocklabel` / `rocklabel-train` command you would type and streams
its output back, so nothing here can do something the CLI cannot. GUI commands
(`label`, `live`, `preview`, …) open their normal Open3D window on your desktop
while their console output appears in the dashboard's job log. Logs also land
in `.dashboard/logs/`.

**Live view and Record bring their controls with them.** Launched from here they
get [`--web-ui`](#0-rocklabel-record--rocklabel-live--the-live-rig)
automatically — you are already in a browser, so that is where the knobs belong
— and the **Live LiDAR** view embeds the running job's control panel as soon as
it is serving. Threshold, scoring region, layers, levelling and the replay
transport are all right there, while the Open3D window gives the scene its whole
screen on your other monitor. *Pop out ↗* moves the panel to its own window if
you would rather place it yourself. Two live jobs at once each get their own
port, and the exact command line — `--web-ui --no-browser` and all — is still
what the drawer previews before you press Run.

The exceptions are **Rename** and **Delete** in the **Data** view, on every
recording, label file and dataset — housekeeping a file manager would do, with
no CLI to defer to. Both refuse while a running job has that path on its command
line, and neither can name anything outside `recordings/`, `labels/` and a direct
child of `datasets/`.

**Rename** treats a recording and its label file as one run, because the rest of
the pipeline does: `label` derives the label filename from the mcap stem and
rewrites the `run_id` and `mcap_file` inside it from that stem every session.
So renaming either one moves both and retags the JSON to match — exactly what
the next `label` session would have written. Renaming a dataset folder is a
plain `mv`; nothing identifies a dataset by folder name.

Already-generated work is untouched by a rename, which is the one thing to keep
in mind: a dataset generated before you renamed a run keeps the old `run_id` and
a `mcap_path` pointing at the old filename. Regenerating into that dataset
therefore *adds* a second run under the new id instead of replacing the old one,
and pooling it into the cache would count the same recording twice. Delete the
stale run (or the whole dataset) before regenerating. A cache that is already
built keeps working either way — only the `dataset_dir` provenance in
`training/cache/meta.json` goes stale.

**Delete** is narrower on purpose: it removes only what you clicked, never the
rest of the run, since labels cost an afternoon and an `.mcap` costs gigabytes.
The page asks first, spelling out what is about to go and what will be left
behind. There is no undo and no trash folder — the space comes back immediately.
A training cache built from a dataset you delete keeps its own copy of the
samples, so runs trained from it stay valid.

```bash
rocklabel dash --root ../other-project   # manage a different project directory
rocklabel dash --port 9000 --no-browser
```

It binds to `127.0.0.1` by design: the dashboard starts processes, so it has no
business listening on the network. `--host 0.0.0.0` works but warns.

### When the sensor is plugged straight into a laptop

There is no DHCP server on that cable, so NetworkManager keeps retrying the
port and tearing its addresses back down — the link looks up, but the UDP
stream lands nowhere. The **Live LiDAR** view detects this: when the wired NIC
is missing the addresses the sensor needs, it lists what is wrong and offers
the exact repair sequence to paste into a terminal (unmanage the port, flush
it, add the three static addresses, bring it back up). It never runs them for
you — every step needs root, and an `ip addr flush` on the wrong interface is
how you drop your ssh session.

The addresses it suggests come from `SourceConfig` in
[`rocklabel/live/config.py`](rocklabel/live/config.py) — `host_addr` (this host
on the sensor's subnet), `udp_dest_addr` (where the sensor streams Compact
data) and `gateway_addr` (the sensor's gateway, which must also resolve here).
Set `wired_iface` if the machine has more than one Ethernet port.

## Workflow

```
rocklabel record   ->  rocklabel label  ->  rocklabel generate  ->  rocklabel-train
   (capture from          (place sphere        (write both             (train PointNet /
    the live rig)          labels once)         dataset formats)        PointNet++)
                                                       -> rocklabel live --model best.pt
                                                            (run the model on the LIVE
                                                             sensor stream, predictions
                                                             as a point color mode)
```

For ROS 2 robot logs the front of the chain is `inspect` / `trim` instead of
`record`:

```
rocklabel inspect  ->  [rocklabel trim]  ->  rocklabel label  ->  rocklabel driftcheck
   (find topic/          (shrink huge /         (place sphere        (verify odometry
    frame names)          salvage broken         labels once)         didn't drift)
                          recordings)
                              -> rocklabel generate  ->  rocklabel preview
                                   (write both              (eyeball what was
                                    dataset formats)         actually written)
```

### Cheat sheet: one new recording, start to finish

By convention this repo keeps three top-level folders: `recordings/` (raw
`.mcap` files), `labels/` (label JSON), and `datasets/` (generated training
data) — commands below assume a recording already dropped into `recordings/`.

Native recording (handheld rig — no config needed):

```bash
rocklabel record recordings/RUN.mcap --source udp         # 0. capture live (S stops/restarts, Q quits)
rocklabel label recordings/RUN.mcap                       # 1. click rocks, Q to quit (auto-saves to labels/)
rocklabel generate recordings/RUN.mcap --out datasets/DATASET   # 2. write dataset
rocklabel preview datasets/DATASET                         # 3. skip through the written frames
```

ROS 2 robot log (needs topic/frame names in a config):

```bash
rocklabel inspect recordings/RUN.mcap                                     # 1. discover topics/frames
rocklabel trim recordings/RUN.mcap --out recordings/RUN.lidar.mcap \
    --start-s 120 --end-s 900                                             # 2. (optional) shrink + cut time
rocklabel label recordings/RUN.lidar.mcap --config config.yaml           # 3. click rocks, Q to quit
rocklabel driftcheck recordings/RUN.lidar.mcap \
    --labels labels/RUN.lidar.labels.json --rock-id 1 --config config.yaml   # 4. verify odometry
rocklabel generate recordings/RUN.lidar.mcap \
    --labels labels/RUN.lidar.labels.json --config config.yaml --out datasets/DATASET  # 5. write dataset
rocklabel preview datasets/DATASET                                        # 6. eyeball the result
```

(The old `--mcap`/`--out` flag spellings still work everywhere.)

### 0. `rocklabel record` / `rocklabel live` — the live rig

The live SICK multiScan pipeline (UDP ingest → IMU/SLAM pose → 2.5D Kalman
surface → Open3D viewer), ported from the old lidarGraphing project, now lives
in this repo under `rocklabel.live`. Recordings are written in the native
`/lidar/frames` format, so everything downstream (`label`, `generate`,
`rocklabel-train replay`, …) reads them directly — the full stack runs from
one codebase.

```bash
rocklabel record --source udp                      # record the live sensor (auto-named in recordings/)
rocklabel record recordings/RUN.mcap --source udp  # explicit output path
rocklabel record --headless --duration 60          # no GUI, e.g. over SSH
rocklabel live --source udp                        # view only; S starts/stops a recording
rocklabel live --play recordings/RUN.mcap          # replay any mcap through the live pipeline
rocklabel live --source udp --web-ui               # controls in a browser, 3D on its own screen
```

`record` starts writing immediately; `live` is the same app without
auto-record. The window has a settings panel (styled like the other rocklabel
viewers) where everything is adjustable while running: layer visibility,
color mode, point size, recording start/stop, and all model-scoring
parameters. Keys mirror the panel: `S` record on/off, `V` cycle point colors,
`P`/`M`/`C`/`B` toggle layers, `L` re-measure mount tilt, `space` pause, `R`
reset, `Q` quit. Live-rig tuning (grid/crop/SLAM/display) comes from
`--rig-config RIG.yaml` plus flags like `--floor-band`, `--no-slam`,
`--yaw-only` — this is a separate config from the offline rocklabel YAML.

**`--web-ui` — the controls on a second monitor.** Open3D's widget toolkit can
host the 3D scene or a good settings panel, not both: it has no wrapping, no
real typography, and labels that reflow at runtime and shove the sections below
them out of frame. So `--web-ui` moves every control to a browser page and
gives the whole Open3D window to the scene:

```bash
rocklabel live --play recordings/RUN.mcap --model best.pt --web-ui
# [rocklabel] control panel: http://localhost:8770/
```

Same knobs, same help text, plus the live status readouts and the replay
transport — laid out properly, in `dash`'s theme (it loads the same
stylesheet). It is a control panel, not a viewer: the points stay in the
Open3D window, which keeps all its keyboard shortcuts, and changes made there
show up in the browser within a moment. `--web-port` (default 8770, since
`dash` owns 8765), `--web-host` and `--no-browser` are there if you need them;
it binds `127.0.0.1` because this endpoint can re-aim the scoring region and
start writing files. `--headless --web-ui` drives the rig with no GUI at all,
in which case the page shows only the controls such a run really has.

The page also carries three views the 3D window has no room for:

* **Overhead** — the fused surface from above, with every detection on it, the
  sensor's position and heading, and the scoring radius. Height is a neutral
  ramp and rocks are the one colored thing on it, so what you are hunting is
  the only thing that stands out. Hovering a mark gives its probability and
  position; *Table* lists the strongest.
* **Prediction confidence** — the distribution of every scored center with the
  decision threshold drawn on it, above-threshold bars in the same orange the
  map paints rocks. This is what makes the threshold slider legible: you can
  see both lobes the model produces and exactly how many centers a given cut
  keeps. Counts default to log, because rocks are rare and a linear axis
  flattens them into the baseline.
* **Trends** — throughput, points in region, scoring-pass time and detection
  count over the last few minutes, one measure per plot on its own axis.

They cost a second endpoint (`/api/scene`) polled once a second, and only
appear when there is something to show — no `--model`, no detections and no
histogram, but the overhead map still works.

**Levelling (a tilt-mounted sensor).** The world frame is otherwise just the
sensor frame at startup, so a LiDAR bolted to a slanted mast tilts the entire
map with it — the crop band cuts a diagonal wedge out of the floor, the 2.5D
heightmap tries to represent a ramp as one height per column and folds into
spikes, and the model sees neighborhoods whose height relief is mostly mount
angle. On startup the rig now measures that tilt and rotates it out:

* the multiScan's IMU quaternion is **gravity-referenced**, so its startup
  orientation seeds the estimate from the first telegram;
* a **RANSAC ground-plane fit** over the next few seconds refines it (on our
  robot the IMU alone leaves a consistent ~2° offset between its axes and the
  optical frame) and measures how high the sensor rides above the floor.

Then it **locks** — the rotation must be constant, or the SLAM map and the
fused heightmap would shift underneath themselves. `L` (or the Levelling
panel) re-measures on demand. Control it with `--level auto|imu|ground|manual|off`
(default `auto`; `ground` needs no IMU, which is what replays use) or give the
angles yourself with `--mount-roll/--mount-pitch`, in the same convention the
IMU and the status line report.

Because levelling knows where the floor is, prefer **`--floor-band LOW HIGH`**
over `--z-min/--z-max`: it anchors the crop to the measured ground plane, so
the same setting works whether the sensor rides 0.4 m up on a robot or 1 m up
in your hand. If the configured band does not contain the floor, the rig says
so at startup instead of silently fusing nothing.

```bash
rocklabel live --source udp --floor-band -0.10 0.60 --max-range 8
rocklabel live --source udp --level manual --mount-pitch 31   # angle already known
```

**Live model predictions:** pass a trained checkpoint to either command and
the viewer gains a third color mode, cycled with `V` alongside height and
reflectivity:

```bash
rocklabel live --source udp --model training/runs/pointnet2_loro_myroom2/best.pt \
    --floor-band -0.10 0.60 --max-range 8        # band measured from the floor
rocklabel live --source udp --model training/runs/.../best.pt \
    --z-min -1.5 --z-max -0.5 --max-range 8      # or fixed, sensor ~1 m up
rocklabel record --source udp --model training/runs/.../best.pt   # record + predict at once
rocklabel live --play recordings/RUN.mcap --model training/runs/.../best.pt  # over a replay
```

Each scoring pass takes the **freshest raw scan** (not the accumulated map —
the model is trained on single-scan clouds, and feeding it the ever-densifying
fused map made predictions decay over time), restricts it to a **scoring
region** around the sensor — a z band and a horizontal radius, so walls and
ceiling never reach the model — intersects the *checkpoint's own* crop box,
and scores the candidate centers exactly the way dataset generation builds
them. Scored centers merge into a **persistent prediction map** (one slot per
5 cm voxel, newest probability wins), so coverage builds up as you sweep the
room and revisiting a spot refreshes it. Displayed points take the
probability of their nearest mapped center (turbo: blue = clear, red = rock;
there is also a binary detections view at the decision threshold); points
with no prediction keep dimmed height colors. A per-scan pass runs in tens of
milliseconds on a GPU — comfortably real time.

Everything about scoring is tunable live in the panel's **Model** section:
on/off, confidence vs. detections display, decision threshold, update
interval (default 0.5 s), scan window (0 = single scan, matching training),
the region's z band and max range, prediction-map on/off + clear, and the
max-centers-per-pass cap. "Crop view to region" in the View section also
hides the out-of-region points (walls/ceiling) from the display itself.
`--floor-band` (or `--z-min/--z-max`) and `--max-range` seed the region from
the CLI; `--score-interval` and `--device` set the cadence and torch device.
When a pass finds nothing in the region, the readout says what the band was
and where the scan's points actually were — the difference between "no rocks"
and "no data" is otherwise invisible. Requires the `[train]` extra.

The same crop flags also work on the offline replay — a big speedup on
wall/ceiling-heavy recordings:

```bash
rocklabel-train replay recordings/RUN.mcap --checkpoint training/runs/.../best.pt \
    --z-min -1.5 --z-max -0.5 --max-range 8
```

### 1. `rocklabel inspect <file.mcap>`

Prints everything you need to fill in the config: all topics with counts and
types, the PointCloud2 topics and their fields, whether an intensity field was
detected, the TF tree (parent → child, static/dynamic), and the time span.

```bash
rocklabel inspect run_2026_05_03.mcap [--config config.yaml]
```

For a **native lidarrig recording** it instead reports the frame rate,
points/frame, whether reflectivity and poses are present, and the lidarrig
config that was embedded when it was recorded — nothing to configure.

For a ROS 2 bag: copy `config.example.yaml`, set `topics.pointcloud_topic` and
the three frame IDs to match what `inspect` shows, and use that config for
every later step.

### 1b. `rocklabel trim` — shrink huge recordings, salvage broken ones

Competition recordings often carry dozens of topics (cameras, motor telemetry,
metrics) and can reach tens of GB; only the LiDAR topic and TF matter here.
`trim` copies just those into a new, properly indexed mcap — typically an
order of magnitude smaller — and can cut a time window (e.g. to drop the
minutes before the run started, when people were still walking in the arena):

```bash
rocklabel trim --mcap huge.mcap --out run1.mcap --start-s 120 --end-s 900
```

- `--start-s/--end-s` are **seconds relative to the start of the recording**
  (so "skip the first 2 minutes" is `--start-s 120`). TF topics are exempt
  from the window so pose lookups still work at its edges.
- `--topic /extra/topic` (repeatable) keeps additional topics;
  `--all-topics` keeps everything and trims only by time.
- The input is read sequentially, so `trim` doubles as a **recovery tool**: a
  truncated / never-finalized recording (recorder killed mid-write — the
  classic `RecordLengthLimitExceeded` or "no summary section" error) is
  salvaged up to the corruption point into a valid output file.

Label against the trimmed file and keep using the trimmed file for
`driftcheck`/`generate` — labels are tied to the recording they were made on.

### 2. `rocklabel label` — interactive labeling

Fuses all LiDAR scans into one voxel-accumulated world-frame cloud and opens
the labeling app for placing labels around rocks. Three label shapes are
available: **spheres** (Shift+click), **boxes** (drag a footprint, then set
width/depth/height), and **lasso polygons** (click an outline, then set base
and top z). Points inside the active shape light up cyan as you drag or
resize, so you can see exactly what a label captures.

```bash
rocklabel label RUN.mcap [--config config.yaml] \
    [--labels RUN.labels.json]   # resume an existing label file
    [--stride N]                 # accumulate every Nth scan (default 1)
    [--z-min A --z-max B]        # initial z clip planes (meters)
    [--dump-accumulated CLOUD.PLY]  # write fused cloud and exit (no window; SSH-friendly)
    [--fallback-viewer]          # legacy pick-then-terminal viewer
```

If `--labels` is omitted, labels are saved to `labels/<mcap basename>.labels.json`
(the `labels/` folder is created if missing). The file is auto-saved on every change.

The window has a side panel with everything mouse-driven — a **Tool** combo
(Navigate / Box / Lasso), a **Camera** combo (orbit / WASD fly), a **color-by
combo** (height / reflectivity), **point size** and **z-clip sliders**, grid
and axes toggles, a **rock list** (click to select, with Focus/Delete buttons
and per-shape sliders: radius for spheres, width/depth/height for boxes,
base/top z for lassos), and a Save button — plus keyboard shortcuts for the
fast path:

| input | action |
|---|---|
| drag / wheel | orbit / zoom (orbit spins around the pivot) |
| **double-click** | set the orbit pivot on the clicked point (CAD-style) |
| **Shift + click** | place a new rock sphere at the clicked point (any tool) |
| **Ctrl + click** | select the nearest existing rock |
| `N` / `B` / `L` | switch tool: navigate / box / lasso |
| box tool | left-drag the footprint on the ground, release, then set the sliders |
| lasso tool | left-click outline points; `Enter` or double-click closes, `Esc` cancels |
| `C` | cycle color mode: height (turbo) / reflectivity (inferno) |
| `+` / `-` | grow / shrink the selected shape by 0.02 m (radius / box / lasso top) |
| arrow keys | nudge the selected rock in x/y by 0.02 m |
| `PgUp` / `PgDn` | nudge the selected rock in z by 0.02 m |
| `X` / `Del` | delete the selected rock |
| `F` | fly the camera to the selected rock (and orbit around it) |
| `S` | save (labels are also auto-saved on every change) |
| `Q` / `Esc` | quit |

Clicks are forgiving: the picker searches a ~10 px patch around the cursor,
so you don't have to hit a 3-px point exactly. In **Fly** camera mode, WASD
moves and dragging looks around; `Esc` returns to orbit mode. Right/middle
mouse drags always navigate, even while the box or lasso tool is active.

Labeling tips: drag the **z-max slider** down near the floor level — rocks pop
out visually as everything above them disappears. Switch to **reflectivity**
coloring to find retroreflective markers / distinctly-reflective rocks
instantly (RSSI is percentile-normalized so bright targets pop without washing
out the floor).

### 3. `rocklabel driftcheck` — verify odometry per run

The label-once-project-everywhere strategy silently breaks if odometry drifts.
Before generating a dataset, check at least one rock per run:

```bash
rocklabel driftcheck --mcap RUN.mcap --labels RUN.labels.json --rock-id 1 [--config config.yaml]
```

Accumulates only the **first 10%** (blue) and **last 10%** (orange) of scans,
cropped to a 1 m box around the chosen rock, and overlays them. If the two
colors show the same rock surface, odometry held; if the rock appears doubled
or smeared between colors, don't trust this run's labels.

### 4. `rocklabel generate` — write the dataset

```bash
rocklabel generate --mcap RUN.mcap --labels RUN.labels.json --config config.yaml --out DATASET_DIR
```

Non-interactive. Replays the recording, keeps every `frame_stride`-th scan,
transforms it to odom, crops a robot-centered axis-aligned box, projects the
sphere labels into the frame, and writes **both** output formats.

**Native lidarrig recordings: set `generator.frame_window_s`** (e.g. `0.25`).
The rig records raw ~4 ms sensor batches of ~400 points each — far too sparse
to be one training frame. `frame_window_s` fuses all batches inside each time
window into one dense frame (using their recorded poses) before
`frame_stride` picks every Nth *window*. It defaults to 0 (one frame per
message, correct for ROS 2 bags, whose messages are full scans). Frames with
no pose or an empty crop are skipped and counted. Ends with a per-run summary
table (frames, samples per format, rock fraction) and a loud warning if a
labeled run produced zero rock samples (label/frame misalignment — run
`driftcheck`).

Multiple recordings accumulate into one `DATASET_DIR` as long as they were
generated with the **identical** config; re-running a run_id replaces its
files. A different config against an existing dataset directory is refused
(see manifest below) — mixed-config datasets are impossible by construction.

### 5. `rocklabel preview` — skip through what was actually written

```bash
rocklabel preview DATASET_DIR [--run RUN_ID] [--frame N] [--list]
```

Opens an interactive **frame browser** over the generated dataset,
reconstructed **from the written npz files** (no mcap needed) — so what you
see is literally what a training job would load. A transport bar along the
bottom skips through frames: prev/next buttons, **play/pause** (with an
adjustable frames-per-second slider), and a **seek slider**; the `Left`/
`Right` arrow keys, `space`, and `Home`/`End` do the same from the keyboard
(click the 3D view first so it has key focus). The side panel shows per-frame
stats (robot position, occupied/rock/clear cell counts, sample counts) and a
color legend.

The **Combine frames** dropdown accumulates trailing data into the view:
*Current frame*, *Last 0.25 s*, *Last 1 s*, *Last 5 s*, or the *Entire run* —
every combined point keeps its classification color, so "show me everything
we wrote and how it was labeled" is the Entire-run setting. This matters most
for native lidarrig datasets generated without `frame_window_s`, where a
single frame is one sparse 4 ms batch.

Colors: gray points are occupied BEV cells (drawn at their max-z surface
height), red cells are labeled rock, olive cells are the ignore shell, cyan
and yellow dots are the format-A sample centers (clear / rock), red
wireframes are the original label spheres, and the axis triad is the robot
base pose. Rock cells and yellow dots should sit inside the wireframes; if
they don't, something is misaligned. `--frame N` starts the browser at a
specific frame; `--list` prints the frame indices and exits.

## Label file schema

JSON, versioned, everything in odom-frame meters:

```json
{
  "schema_version": 2,
  "mcap_file": "run_2026_05_03.mcap",
  "run_id": "run_2026_05_03",
  "odom_frame": "odom",
  "created": "2026-07-06T12:00:00Z",
  "tool_version": "0.1.0",
  "intensity_available": true,
  "accumulator_voxel_m": 0.03,
  "rocks": [
    {"id": 1, "shape": "sphere", "center": [3.412, -0.881, 0.094], "radius": 0.18},
    {"id": 2, "shape": "box", "center": [5.020, 1.303, 0.121], "radius": 0.31,
     "size": [0.40, 0.35, 0.30]},
    {"id": 3, "shape": "polygon", "center": [7.100, 0.250, 0.200], "radius": 0.55,
     "vertices": [[6.8, 0.0], [7.4, 0.1], [7.3, 0.5]], "z_range": [0.05, 0.35]}
  ]
}
```

Schema v2 adds shaped labels: axis-aligned **boxes** (`center` + full-extent
`size`) and extruded **polygons** (`vertices` in odom xy + `z_range`). Every
rock, whatever its shape, also stores its bounding sphere (`center` +
`radius`) so shape-agnostic consumers (driftcheck, preview, sample placement)
keep working. v1 files (spheres only, no `shape` field) still load.

Rock ids are stable within a session (monotonic counter, never reused).

## Ground-truth definition

A point is **rock (1)** if inside any labeled shape, **ignore** if inside the
shape grown by `boundary_shell_m` on every side but outside the shape itself
(the fuzzy boundary — encoded as 255 in masks, skipped as samples), else
**clear (0)**.

The two output formats apply this definition to **different populations**, so
their rock/clear counts for one frame do *not* match — this is by design, not
a bug (see [The two formats count different things](#the-two-formats-count-different-things)).

## Output format A: point-neighborhood samples

`DATASET_DIR/points/<run_id>/frame_<idx:06d>.npz`, one per kept frame:

| key | shape / dtype | meaning |
|---|---|---|
| `neighborhoods` | `[S, 256, 4]` float32 | canonicalized neighbor points: x, y relative to the center; z relative to the lowest neighbor (local ground ≈ 0); intensity |
| `labels` | `[S]` int8 | 0 = clear, 1 = rock (ignore-labeled centers are skipped) |
| `true_counts` | `[S]` int16 | real neighbor count before subsample/pad |
| `centers_odom` | `[S, 3]` float32 | candidate center positions in odom |
| `frame_time` | scalar | sanitized scan time (s) |
| `robot_pose` | `[4, 4]` float64 | odom ← base_link at frame time |

One **sample** = one candidate center plus the fixed-size point neighborhood
around it. `S` is the number of samples kept from the frame; the `neighborhoods`
axis of 256 is the per-sample point budget (`neighborhood_points`), unrelated
to `S`.

Candidate centers come from a `centers_voxel_m` voxel grid over the cropped
cloud; centers with fewer than `min_neighbors` points within
`neighborhood_radius_m` are dropped. Neighborhoods larger than 256 points are
randomly subsampled, smaller ones padded by repeating random points. All rock
samples are kept; clear samples are kept with probability
`negative_keep_prob` (seeded per frame — regeneration is bit-reproducible).

**Format A is deliberately class-rebalanced.** Because `negative_keep_prob`
(default `0.05`) throws away ~95% of clear candidates while keeping every rock
candidate, the rock fraction of `labels` — e.g. 49 of 50 in a single frame —
is a property of *the sampler*, not of the scene. It is not telling you the
frame is 98% rock. For the true spatial rock/clear balance of a frame, read
format B's `label_mask` instead.

## Output format B: BEV rasters

`DATASET_DIR/bev/<run_id>/frame_<idx:06d>.npz`, one per kept frame:

- `channels` — float32 `[8, H, W]`; with default crop and `bev_cell_m` = 0.10:
  80 × 80. Row index i runs along odom x (from `base_x - crop_backward_m`),
  column index j along odom y (from `base_y - crop_right_m`). Empty cells are
  0 in every channel.

  | # | channel |
  |---|---|
  | 0 | valid (1 if the cell has ≥ 1 point) |
  | 1 | point count (log1p) |
  | 2 | max z − robot base z |
  | 3 | min z − robot base z |
  | 4 | z span (max − min) |
  | 5 | z standard deviation |
  | 6 | mean intensity |
  | 7 | max intensity |

- `label_mask` — uint8 `[H, W]`, per cell:
  - `1` = **rock** — cell contains at least one rock point.
  - `0` = **clear** — cell has points, none rock, none only-shell.
  - `255` = **ignore** — used for **both** *empty* cells (no points at all)
    and *occupied-but-shell-only* cells (points, but all in the boundary
    shell). The vast majority of `255`s are simply empty cells outside sensor
    range: with the default 80×80 crop most of the grid is `255`.

  Unlike format A, **every** occupied cell is labeled — no subsampling — so
  this mask is the honest spatial rock/clear split of the frame. Note that
  `channel 0` (valid/occupied) counts rock + clear + occupied-ignore cells, so
  `occupied ≠ (rock cells) + (clear cells)` whenever any occupied cell is
  shell-only.
- `frame_time`, `robot_pose` — as in format A.

The crop box is axis-aligned in odom and **not** rotated with robot heading;
heading invariance is a training-time augmentation, not a dataset property.

### The two formats count different things

For one frame the point samples (A) and BEV mask (B) will report different
rock/clear numbers, and the `preview` side panel shows both — e.g.
`samples: 50 (49 rock)` next to `cells: 38 occupied · rock 15, clear 22`. Both
are correct:

- **A (`samples: 50 (49 rock)`)** counts *class-rebalanced sample centers* —
  yellow/cyan dots in `preview`. Almost all clear centers were dropped by
  `negative_keep_prob`, so this is rock-heavy on purpose.
- **B (`cells: 38 occupied · rock 15, clear 22`)** counts *every occupied BEV
  cell* — the red/gray cells in `preview`. This is the true scene balance;
  the extra `38 − 15 − 22 = 1` occupied cell is an ignore (shell-only) cell.

If you `np.load` a BEV file and print `label_mask` or `channels` directly,
numpy elides the interior of the 80×80 grid with `...` and you mostly see the
empty `255` / `0` border — the occupied cells are hidden in the middle. Index
explicitly (`(label_mask == 1).sum()`, `channels[0] > 0`) rather than trusting
the truncated repr.

## Dataset manifest

Every `generate` run writes `DATASET_DIR/manifest.json`: tool version, the
fully resolved config, a SHA-256 hash of that config, generation timestamp,
and per-run entries (run_id, mcap/labels paths, frames kept/skipped, samples
per format, rock/clear/ignore counts). Generating into a directory whose
manifest has a **different** config hash is refused — choose a new `--out`.

## Configuration

One YAML file shared by all subcommands — see the fully commented
[config.example.yaml](config.example.yaml). Precedence: CLI flags > config
file > built-in defaults.

Notable robustness knobs:

- `topics.static_lidar_to_base` — fallback LiDAR mount transform if the
  recording lacks it on `/tf_static`.
- `topics.pose_tolerance_s` — max TF extrapolation before a frame is skipped.
- Missing intensity field → one prominent warning, zeros written, and
  `"intensity_available": false` recorded in all output metadata.
- Zero/garbage header stamps fall back to mcap log time (counted in the
  end-of-run summary).
- If your TF tree has a localization-corrected frame (e.g. `map -> odom`
  published by a SLAM/fiducial node), consider `topics.odom_frame: map` —
  rocks are stationary in the *corrected* frame, so labels projected from it
  suffer less drift. Verify either choice with `driftcheck`.

## Raw vs. filtered point cloud topics

Recordings often contain both the raw driver topic and filtered topics from
the robot's perception stack (crop box, self-hit removal, outlier rejection).
Both work — set `topics.pointcloud_topic` to either; each message's own
`header.frame_id` is used for the transform, so topics already published in
the odom frame need no extra configuration. Choosing:

- **Match inference**: train on the topic your classifier will consume on the
  robot. This outweighs everything else.
- **Raw** keeps intensity (a useful rock feature) but includes robot
  self-hits (points at ~0 range that smear along the trajectory in the fused
  labeling view) and reflection ghosts (e.g. mirrored points below the floor
  from windows). Per-frame datasets are less affected than the fused view:
  the crop box bounds them and moving objects appear at their true
  instantaneous position labeled "clear".
- **Filtered** topics give a much cleaner labeling view and dataset, but your
  filter chain may strip intensity — check with `rocklabel inspect`, which
  prints frame and fields for every PointCloud2 topic.

Labels transfer between topics of the same recording (they live in the odom
frame, independent of which cloud you look at), but switching topics changes
the config hash, so `generate` requires a fresh `--out` directory.

## Troubleshooting

- **`RecordLengthLimitExceeded` / "no summary section" when opening an
  mcap** — the file is truncated or was never finalized: the recorder was
  killed mid-write, or the file is *still being copied* from the robot (check
  its size twice a few seconds apart). If the copy is complete and it still
  fails, salvage it: `rocklabel trim --mcap broken.mcap --out fixed.mcap`.
- **Recording is tens of GB** — `trim` it first (see above); all later steps
  re-read the mcap on every invocation, so working from a small LiDAR-only
  file makes everything faster.
- **"no intensity/reflectivity field" warning** — that recording's driver
  didn't publish intensity. Everything still works; intensity is 0 everywhere
  and `intensity_available: false` is recorded so training code can drop that
  channel.
- **`[entity=...] missing required attributes` lines from the labeler
  window** — harmless Open3D/filament rendering chatter about the transparent
  spheres; ignore.
- **`error: unrecognized arguments`** — each subcommand takes only its own
  flags; run `rocklabel <subcommand> -h` to see them.
- **Rock ids** — ids are stable but *not* renumbered (deleting rock 1 leaves
  ids 2, 3, ...). `driftcheck` lists the available ids if you name a missing
  one; they're also visible in the labels JSON.

## Tests

```bash
python -m pytest tests/
```

The suite includes a synthetic end-to-end fixture
([tests/make_synthetic_mcap.py](tests/make_synthetic_mcap.py)) that writes a
valid ROS 2 mcap — 100 scans of a noisy flat floor with 3 hemispherical rocks
at known positions, a robot translating 3 m on `/tf`, and a static LiDAR mount
on `/tf_static`, with points expressed in the LiDAR frame so the full
transform chain is exercised. Run it directly to get a demo recording:

```bash
python tests/make_synthetic_mcap.py demo.mcap   # + demo.labels.json
rocklabel generate --mcap demo.mcap --labels demo.labels.json --out demo_dataset
```

All Open3D code is isolated in `rocklabel/viewer.py`; everything else runs
headless (that's also what `--dump-accumulated` is for).

## Training (optional): PointNet / PointNet++ rock classifiers

The `rocklabel.train` subpackage trains binary rock classifiers on the
format-A neighborhood samples. It is an optional extra so the core CLI never
needs torch:

```bash
pip install -e .[train]

rocklabel-train cache      # pool the four myroom runs into training/cache/
                           # (verifies config hashes + manifest counts)
rocklabel-train compare    # both models x 4 leave-one-run-out folds,
                           # then every figure into training/results/
rocklabel-train view datasets/myroomdataset2 \
    --checkpoint training/runs/pointnet2_loro_myroom2/best.pt   # 3D confidence replay
rocklabel-train replay recordings/anything.mcap \
    --checkpoint training/runs/pointnet_loro_myroom4/best.pt    # any mcap, no labels needed
rocklabel-train export training/runs/pointnet2_loro_myroom2/best.pt
```

`view` replays a *generated dataset* and can show ground truth and error
modes; `replay` runs the model on **any recording** (either mcap format,
labeled or not) - it rebuilds neighborhoods on the fly with the exact geometry
stored in the checkpoint, scores every candidate center, and browses the run
with confidence coloring and a threshold slider.

Key design points (see the module docstrings for the details):

- **Evaluation is leave-one-run-out, on purpose.** Candidate centers sit on a
  5 cm grid inside 50 cm neighborhoods and consecutive frames barely move, so
  a random sample split leaks near-duplicates and produces meaningless scores.
  Each fold trains on 3 runs and tests on the whole held-out run; the
  early-stopping val split uses contiguous tail frame blocks with a 25-frame
  gap. `data.py` refuses to pool datasets with different config hashes.
- **Padding is handled explicitly.** Neighborhoods are padded to 256 points by
  repeating real ones (most samples have only ~20-120 real points).
  PointNet's max-pool is duplicate-safe; PointNet++ masks padded points out of
  FPS and ball queries entirely (`models.py`).
- **Class imbalance** (~29% rock) is handled with class-weighted BCE, and the
  headline metrics are PR-AUC / ROC-AUC / F1 with the majority-class baseline
  printed next to them - never bare accuracy.
- **Runs resume**: each fold lives in `training/runs/<model>_loro_<run>/` with
  `config.json`, `history.csv`, `last.pt` / `best.pt`, `test_metrics.json`,
  and `predictions.npz`; `rocklabel-train report` regenerates all plots
  without retraining.
- **Export** writes TorchScript + ONNX plus `metadata.json` pinning the whole
  preprocessing contract (channel semantics, radius, voxel size, config hash,
  threshold) and a standalone `infer_example.py` that needs only torch.

Caveat: the four myroom runs share one small set of labeled rocks and are much
rock-richer (24-44%) than e.g. the lance arena runs (1-3%), so treat the
scores as "does the architecture learn this scene", not as competition-arena
performance.

## Non-goals

No ROS 2 nodes, no live operation, no loop closure or odometry correction.
This is a competition team's offline tool, kept small and readable. (Offline
model training lives in the optional `rocklabel.train` extra above; there is
still no live/ROS inference here.)


actual running 

run model with live predictions (robot rig — sensor on the tilted mast, ~0.44 m
up; levelling measures the mount angle and the floor height, --floor-band then
anchors the crop to the ground so the rig height does not matter):
rocklabel live --source udp --model training/runs/pointnet_loro_myroom4/best.pt \
    --floor-band -0.10 0.60 --max-range 8

run model with previously recorded mcap:
rocklabel-train replay recordings/myroom5.mcap --checkpoint training/runs/pointnet_loro_myroom4/best.pt     --z-min -1.5 --z-max -0.5 --max-range 8

