# Agent 2 result — SLAM, and what is really holding the model back

**Short version: the SLAM is fine, and it is not what is limiting your
accuracy.** The one thing the brief asked me to check first — "does a better
sensor path make the model better?" — turned out to be false, and chasing it
would have been a waste. Along the way I found what *is* wrong with your two
worst recordings, and it is one bad label in each.

**Nothing was re-solved, because nothing needed re-solving.** All 13 volleyball
recordings were already re-solved with identical settings, and I proved the
solver is exactly repeatable, so **your labels are safe and you do not have to
relabel anything.**

---

## 1. The claim I was sent to test, and why it is wrong

The previous notes said: recordings that are harder to work out a sensor path
for also score worse, so improving the path would improve the model. That was
based on a −0.56 correlation.

I measured the sensor-path quality of **all 13** recordings — not the four that
were spot-checked — using three independent measures, and lined them up against
the fold scores corrected for how many rocks each recording actually has.

**The correlation is one data point in a trench coat.** Ten of the eleven
recordings come out within a hair of each other (16.7–17.9 mm of surface
fuzz — a 1.2 mm spread), and only VolleyBallTest4 sits outside at 23.1 mm.
Drop that single recording and the correlation goes from −0.61 to **+0.004**.
It is not a trend; it is one point acting as a lever.

The nail in it: the measure that captures *actual path error* — how far apart
the same patch of sand lands when the sensor looks at it twice — correlates
**+0.02** with fold score. That is nothing at all. If path error drove
accuracy, that is the number that would show it, and it doesn't.

| what I correlated against fold score | correlation | after dropping VB4 |
|---|---|---|
| surface fuzz (the headline number) | −0.61 | **+0.00** |
| path error specifically | +0.02 | +0.67 |
| how flat the reconstructed sand is | −0.70 | −0.39 |
| how much the sensor was waved about | −0.38 | −0.60 |

Bootstrapped confidence intervals on every one of these span zero
(e.g. −0.93 to +0.64). With eleven recordings there is simply not enough
evidence for any of it.

**Verdict: model accuracy is limited by something other than the sensor path.
Agent 3 should not wait on SLAM work, and no datasets need regenerating on
SLAM's account.** The remaining SLAM prize is real but small — about 17 mm of
surface fuzz down to about 12 mm — and it is worth having for its own sake, not
as a lever on detection.

---

## 2. The VolleyBallTest4 contradiction, resolved — and it is a label

Two earlier write-ups disagreed: one called VB4 the hardest run, the other
called it the cleanest. **Both were half right, because they were measuring
different things, and neither had found the actual problem.**

**On steadiness, VB4 is genuinely one of the calmest recordings.** Its sensor
tilt wanders 4.6° (second-lowest of thirteen). Whoever called it clean was
right about that.

**On reconstruction, VB4 has a wrecked first five seconds.** I sliced the run
into 5-second chunks:

| slice | surface fuzz |
|---|---|
| 0–5 s | **67.5 mm** |
| 5–55 s | 12.5–14.0 mm — better than any other run |

That is a start-up failure, and I know why. The first pass through a recording
walks forwards from an empty map, so the earliest scans have almost nothing to
line up against; their bad positions then get baked *into* the map, and every
later cleanup pass rebuilds the map from those same bad scans. It is
self-confirming. I ran it with five passes instead of three to check: 67.6 mm.
No improvement at all.

**But that is not why VB4 scores badly either.** I re-scored the fold with the
damaged frames thrown out — the model untouched, just its existing predictions
re-tallied. Throwing away the first 20 seconds, 35% of the data, moved the
score from 0.355 to 0.384. Against a noise floor of 0.013 that is barely
anything.

**Here is what is actually wrong.** VB4's precision is fine (0.72) and its
recall is dreadful (0.26): when the model fires it is usually right, it just
almost never fires. So I asked which *rocks* it misses:

| VB4 rock | training samples | recall | model's confidence |
|---|---|---|---|
| 2, 3, 4, 5, 6, 7 | 107–392 each | 0.33–0.55 | 0.44–0.63 |
| **8** | **1053** | **0.03** | **0.07** |

**One label — rock 8 — is 46% of all VB4's rock samples, and the model is
confident it is not a rock.** Look at its shape and you can see why: it covers
47 cm of ground but stands only 6.5 cm proud, on sand that is itself 5.1 cm
rough. Every other VB4 rock stands 7.8–13.5 cm proud. **It reads as a broad low
mound of sand, not a rock.**

**Take rock 8 out and VB4 goes from 0.355 to 0.583.** That is +0.23 — about
eighteen times the noise floor, and far more than every SLAM improvement ever
made on this project put together.

**The same thing is wrong with VolleyBallTest6**, your other collapsing fold.
Its rock 8 is the opposite extreme — the smallest label in the whole project at
10.6 cm across, standing 3.4 cm proud of sand that is 3.5 cm rough. It is
literally the same height as the sand's own bumpiness. Recall 0.00, confidence
0.04. **Take it out and VB6 goes from 0.513 to 0.655.**

So the old note that "the two collapsing folds are exactly the two hardest runs
to solve" had the right two folds for the wrong reason. VB6's path is among the
*best* of the thirteen — I checked rock 8's own patch of ground and it holds
still to 9.2 mm, which is pure sensor noise.

**What I'd ask you to do:** open VB4 rock 8 and VB6 rock 8 in the labeler and
look at them. If they are sand mounds rather than rocks, deleting those two
labels is worth more than anything else in this handoff. If they *are* rocks,
that is just as important to know — it means the model cannot see the flattest
ones, and that is a training problem worth naming.

---

## 3. Your labels are safe. Measured, not assumed.

The worry was that re-solving a recording moves the world underneath the labels,
because rock positions are stored as world coordinates.

**I re-solved VolleyBallTest4 from scratch and compared it against the stored
re-solved file: the two paths agree to 0.000 mm.** The solver is exactly
repeatable. Same recording plus same settings gives the same answer every time,
so re-running it never disturbs a label.

To be sure I also did it the expensive way — gathered every point inside each
labelled rock, remembered which scan each came from, pushed those same points
through the freshly-solved path, and asked where the rock ended up. **Median
movement across all seven rocks: 0.0 mm.**

**So no migration tool was needed and none was built.** If you ever *do* change
solver settings, the method above is the right one and it is written down in
this repo's scratch notes — but it would be dead code today, and I would rather
not leave you dead code.

I validated the labels independently with `driftcheck`, which overlays the
first and last tenth of a run around one rock:

| recording | early-vs-late disagreement | reading |
|---|---|---|
| VolleyBallTest3 | 8.6–12.9 mm | sensor noise; rock held still |
| VolleyBallTest6 | 9.2–17.0 mm | sensor noise; rock held still |
| VolleyBallTest4 | 24.9–39.0 mm | elevated, still far under a rock radius |

VB4 being the elevated one is exactly right — `driftcheck`'s "early" window
*is* its damaged first five seconds. Even there the labels sit well within the
rock, so they remain usable.

---

## 4. What changed in the code

### `rocklabel slam` — the solver is now a normal command

It used to be `python -m rocklabel.slam`, which meant it did not exist for you.
It is now **a card on the dashboard, in the "Solve poses" stage between Triage
and Label.** Recordings picker, every knob with prose explaining when to touch
it, a "Score only" tick-box that tries settings without writing a file, and the
command preview.

Both spellings share one definition, so a knob renamed in one moves in the
other — and the dashboard test that compares the form against the real command
now covers it.

**One bug fixed while wiring it up:** the solver used to write its output
*beside its input*, which after the folder reshuffle meant re-solved recordings
landed back in `recordings/volleyball/raw/` alongside the originals. A recording
read out of a `raw` folder now lands in the `reslam` folder next to it.

### `driftcheck` now prints a number, not just a picture

It used to accumulate the early and late scans, draw them in two colours, and
leave you to judge by eye whether they line up. It now **measures the gap and
tells you** — how far apart the two surfaces sit in millimetres, with a plain
reading of what that means (under 20 mm is sensor noise; near a rock radius
means the label has come off the rock).

**There is a new "Numbers only" tick-box on the Driftcheck card** that skips the
3D window entirely, so you can check a whole run's labels quickly, or over SSH.
That is how I validated the table in section 3.

**A second bug fixed here:** `driftcheck` was re-measuring the mount tilt
instead of replaying the angle stored in the label file, the way `generate`
already does. That fit is only repeatable to about half a degree, which at the
far edge of the box is centimetres — the same size as the drift the command
exists to detect. It now replays the recorded angle and refuses a mismatch.

---

## 5. Numbers you asked for

Surface fuzz per recording, stock tracker versus re-solved. Lower is better;
sensor noise puts a floor at about 13 mm.

| recording | stock SLAM | re-solved | gain | worst 10% | fold score |
|---|---|---|---|---|---|
| VB1 | 32.7 mm | 17.1 mm | 1.91x | 30.7 mm | not a fold — no rocks labelled |
| VB2 | 33.2 mm | 17.9 mm | 1.86x | 31.6 mm | 0.878 |
| VB3 | 40.0 mm | 17.6 mm | 2.28x | 28.7 mm | 0.904 |
| VB4 | 43.6 mm | 23.1 mm | 1.89x | **178.7 mm** | 0.355 |
| VB5 | 42.4 mm | 17.6 mm | 2.40x | 28.7 mm | 0.619 |
| VB6 | 72.8 mm | 17.9 mm | 4.06x | 28.6 mm | 0.513 |
| VB7 | 51.7 mm | 17.7 mm | 2.92x | 29.0 mm | 0.756 |
| VB8 | 36.2 mm | 16.7 mm | 2.17x | 26.4 mm | 0.895 |
| VB9 | 48.3 mm | 17.6 mm | 2.74x | 30.6 mm | 0.861 |
| VB10 | 45.3 mm | 17.9 mm | 2.53x | 29.6 mm | 0.794 |
| VB11 | 50.4 mm | 17.6 mm | 2.86x | 25.9 mm | 0.765 |
| VB12 | 36.0 mm | 16.9 mm | 2.13x | 24.6 mm | 0.486 |
| VB13 | 46.4 mm | 23.0 mm | 2.01x | 39.9 mm | **not a fold, but it could be** |

"Fold score" is corrected for how many rocks each recording has, so the folds
are actually comparable — the raw numbers are not, and comparing them is what
produced the wrong conclusion in the first place.

### The surface is sharper than 17 mm where it matters

The 17 mm headline is measured out to 8 m. **The generator only trains on
points within about 4 m** (its crop box is 6 m forward, 4 m sideways). Inside
that, the sand comes out at **14.7 mm**:

| scored within | 3 m | 4 m | 6 m | 8 m | 12 m |
|---|---|---|---|---|---|
| typical run | 14.5 mm | **14.7 mm** | 16.7 mm | 17.6 mm | 18.0 mm |

The extra 3 mm is far-away sensor noise (about 11 mm inside 6 m, about 27 mm
beyond), not path error. **That answers the brief's "crop the fusion range, not
the registration range" item: it is already done.** Alignment reaches out to
30 m for the fence and tree line, and the dataset is built from the near band
only. There was nothing to change, and now there is a number saying so.

It also shrinks the remaining prize: the model is already seeing ~14.7 mm, and a
perfect path would get it to about 12 mm.

---

## 6. Things I did NOT do, and why

- **No joint optimisation of all poses at once**, and no distribution-matching
  or continuous-time path. These were the ranked next steps, and section 1
  removed the reason for them. They remain the correct order *if* the sensor
  path ever becomes the bottleneck — and I can now point at a much smaller
  target than "everything": the failure is specifically **the first few seconds
  of a recording**, where the map is still empty. That is a far more tractable
  thing to fix than the whole trajectory.
- **No loop closure.** Already measured and ruled out, and nothing I found
  changes that.
- **No re-solving.** All 13 were already done, with identical settings — I
  checked the settings recorded inside each file. Re-running would have produced
  byte-identical output.
- **No label migration tool.** Nothing moved, so it would have been dead code.

---

## 7. Recommendations for agent 3

1. **Do not wait for SLAM, and do not regenerate datasets on SLAM's account.**
   The recordings are as good as they are going to get without a hardware
   change, and path quality does not predict accuracy.
2. **Settle VB4 rock 8 and VB6 rock 8 first.** Deleting two bad labels is worth
   more than every training tweak on the list. If they are legitimate rocks, the
   finding becomes "the model cannot see the flattest rocks", which is a
   different and equally useful thing to know.
3. **Report per-fold scores corrected for rock count.** Rock share runs 6.3% to
   31.6% across recordings — a 5x spread — so raw per-fold PR-AUC says as much
   about how many rocks a run has as about how hard it is. This has produced two
   wrong conclusions already.
4. **VolleyBallTest13 can be a twelfth fold.** It has 6 rocks, an arena, a
   height band and a level record — everything a dataset needs — and has simply
   never been generated. That is a 9% bigger evaluation set for one `generate`
   run. VolleyBallTest1 cannot: it has an arena but zero rocks labelled.
5. **Optionally drop VolleyBallTest4's first 5 seconds.** Its damage is confined
   there and the rest of the run is the sharpest data you have. Low priority —
   I measured that it is worth only about +0.03.

---

## 8. The thing that outranks everything above

Repeating this because it is still true and still the biggest lever available:

**The sensor is swung by hand, and that is the single largest source of error.**
The sensor works out which way is down by feeling gravity — swing it and it
feels your swing too, and cannot tell the two apart. A tripod, a monopod, or a
slow cart removes it at the source. That beats anything achievable in software.

Second: **the recordings only store the sensor's already-combined orientation,
not the raw gyroscope and accelerometer readings.** Combining them is exactly
the step the hand-swinging corrupts. Recording the raw values would let the
solver do that job properly. It needs a change to the recording format.

---

## Tests

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/` → **533 passed**, including
`tests/test_slam.py` (now 37, up from 32) and `tests/test_dashboard.py` (97).
The five new tests cover the re-solved-output folder fix, `rocklabel slam`
staying in step with `python -m rocklabel.slam`, and driftcheck's new measured
offset — including that it refuses to report a number when it has too little
overlap to mean anything.
