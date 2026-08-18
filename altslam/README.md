# altslam — a second SLAM for rocklabel recordings

Reads a recording, works out a better path for the sensor, writes a **new**
recording. The original file is never touched.

```bash
# one recording -> recordings/VolleyBallTest1.reslam.mcap
python -m altslam recordings/VolleyBallTest1.mcap

# all of them, in parallel
ls recordings/VolleyBallTest*.mcap | xargs -P 6 -I{} python -m altslam {} --force

# try settings without writing anything
python -m altslam recordings/VolleyBallTest1.mcap --score-only --passes 1
```

The output is the same file format with the same points, the same brightness
values and the same IMU samples — only the sensor pose attached to each batch
changes. So `rocklabel label`, `generate`, `train` and `live --play` all read it
with no changes at all; they just see a better-aligned world.

Nothing under `rocklabel/` imports this package. It only ever reads from it.

## Why the stock one struggles on the volleyball court

Measured on `VolleyBallTest1.mcap`:

| | |
|---|---|
| Stock tracker's own confidence | 99% of points matched, 0 windows dropped |
| Actual error | the same patch of sand, revisited, lands 116 mm away |

It is not losing lock. It is confidently wrong, and three separate things cause
that.

**1. A sand court is a plane, and a plane cannot tell you where you are.**
It tells you how high above it you are and nothing about where you are *along*
it. An aligner that scores itself by counting matched points is perfectly happy
to slide the whole scan sideways — the match count stays at 99% the whole time.

**2. The far scenery was being thrown away.** The stock tracker only aligns on
points within 12 m. On a court, everything within 12 m is sand. The grass edge,
the fence and the tree line at 15-25 m are the only things in the scene that
say "you have not moved sideways", and they were being discarded.

**3. The IMU's sense of "down" is wrong exactly when you are moving.** The
sensor works out which way is down by feeling gravity. Swing it by hand and it
feels your swing too, and cannot tell the two apart. The stock tracker treats
tilt as absolute truth from the IMU and never corrects it. On these hand-swept
recordings **this is the single biggest error** — worth about 2x on its own.

## What this one does instead

- **Point-to-plane matching.** Only the error *across* the surface is charged
  for, so sliding along the sand costs nothing (correctly — it is not an error).
  Each map cell keeps a covariance, which gives it a surface normal.
- **A degeneracy guard.** The alignment equations are eigen-decomposed and each
  of the six directions is checked for whether the geometry actually pins it
  down. Directions that are not observed are *not solved for* — the prediction
  is left standing instead of being overwritten with noise.
  (Zhang et al., *On Degeneracy of Optimization-based State Estimation Problems*.)
- **Equal say per surface direction.** Half the visible world is flat sand all
  facing the same way; the fence is a thin minority. Correspondences are
  bucketed by which way they face and each bucket gets the same total vote, so a
  handful of fence points counts for as much as an acre of sand.
  (Normal-space sampling, from Rusinkiewicz & Levoy's ICP survey.)
- **Tilt is free by default.** See point 3 above. `--lock-tilt` restores the
  stock behaviour — correct for a tripod or mast-mounted rig.
- **Points out to 30 m**, not 12.
- **A robust kernel** (Geman-McClure) so somebody walking through cannot drag
  the solution.
- **Several passes.** Pass 1 is causal, so its early poses are its worst and
  they get baked in. Later passes rebuild the map from the finished trajectory
  and re-align everything against it.
- **Smoothed output.** Each batch's pose is interpolated between window
  solutions rather than extrapolated on velocity, so there is no step in the
  trajectory every 0.1 s.

## Does it work

`VolleyBallTest1.mcap`, 41.5 s, hand-swept. The score is how thick the sand
surface comes out: every pass over the same 10 cm patch should land on top of
the last one. Sensor noise and real sand roughness put a floor at about 13 mm —
no pose solution can beat that.

| | surface thickness | worst 10% |
|---|---|---|
| stock SLAM | 32.7 mm | 100.3 mm |
| **altslam** | **17.9 mm** | **31.9 mm** |
| floor (sensor noise) | ~13 mm | — |

That matters because the rocks are only 5-10 cm proud of the sand. At 33 mm of
fuzz a rock is barely two noise-widths tall; at 18 mm it stands clear.

Cost: about 110 s per 40 s recording, versus under 2 s for the stock tracker.
That is the trade — it is not fast enough to run live, which is why it runs from
a file instead.

## Layout

| file | what it is |
|---|---|
| `config.py` | every tuning knob, with why-you-would-touch-it notes |
| `voxelmap.py` | the map: one surface patch (centre + normal) per cell |
| `register.py` | the aligner: robust point-to-plane, plus the degeneracy guard |
| `solver.py` | multi-pass whole-recording solve, and per-batch pose output |
| `evaluate.py` | the surface-thickness score |
| `reprocess.py` | read mcap → solve → write new mcap |
| `cli.py` | `python -m altslam` |
| `tests/` | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest altslam/tests -q` |

The env var is only needed because a ROS `launch_testing` plugin in this
machine's Python environment crashes pytest collection; it has nothing to do
with this package.

## Knobs worth knowing

| flag | default | when to touch it |
|---|---|---|
| `--passes` | 3 | 1 is ~3x faster and nearly as good; more rarely helps |
| `--lock-tilt` | off | turn on for a tripod/mast rig |
| `--window` | 0.10 s | 0.05 diverges on this data — too few points per window |
| `--range-max` | 30 m | lower only if distant scenery is moving (traffic, crowds) |
| `--degeneracy` | 0.03 | 0 disables the flat-ground guard entirely |
| `--voxel` | 0.20 m | registration only; does not limit final rock detail |

## Known limits

- **No loop closure.** Revisiting a spot after a long excursion will not snap
  the map back into place. The multi-pass refinement helps but is not the same
  thing.
- **The refinement passes find a local optimum.** They sharpen the trajectory
  they are given; they cannot rescue one that is badly wrong to begin with.
- **Moving objects get built into the map.** The robust kernel stops them
  dragging the solve, but they still land in the cloud.
- **Offline only.** At ~2.7x realtime it cannot run live as-is.
