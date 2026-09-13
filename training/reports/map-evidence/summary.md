# Clearing predictions the sensor has looked through

**The short version: your diagnosis of the mechanism was right, the fix is
built, it is safe at the settings recommended below, and it only reaches
about 6% of the false ground on the competition recording.** The reason is not that the clearing is too timid.
It is that 81% of that recording's false detections are not floating in mid-air
at all — they sit on the ground, within 12 cm of it, which is where a rock sits
too. Nothing that reasons about free space can remove those, because there is
genuinely something there; the model is simply calling flat ground a rock. That
is a model problem, and the rest of tonight's work is aimed at it.

What the clearing did buy is worth keeping anyway, and one part of it was a
surprise: **coverage of the real rocks went up**, from 0.650 to 0.683, because a
false detection floating over a rock stops standing in for the real detection
underneath it. The weakest rock was untouched, and the one rock that lost
anything lost 2% of its cells.

## What was measured

The competition recording, start to finish: 2,599 scoring passes over 35
minutes, through the real live preparation — operational floor band (−0.10 to
+0.60 m about the measured floor), 8 m range, newest answer per 5 cm cell,
reduced to 10 cm ground cells. The model is the deployed classifier
(`deploy/cls-stray/trainall`) at its own stored threshold of 0.71.

Two maps are built from the same scores in the same pass, so they share every
input and every denominator: one exactly as the rig leaves it today, one in
which a remembered detection may be retracted.

| | as the rig leaves it | with clearing |
|---|---:|---:|
| ground claimed with no rock on it | 2,211 cells | 2,074 cells |
| rock coverage, averaged per rock | 0.650 | **0.683** |
| weakest rock | 0.294 | 0.294 |
| rocks never covered at all | 0 | 0 |
| median time to first cover a rock | 158 s | 158 s |
| remembered detections retracted | — | 13,023 |

Nothing got slower to acquire. Per-rock numbers are in
`deploy-stray-baseline/per-rock.csv`: rocks 3, 7, 11 and 13 improved, the last
from 0.405 to 0.548, and rock 4 lost 2% of its cells.

## Why it cannot do much more

Of the wrongly-claimed detections still standing at the end:

| | share |
|---|---:|
| have a surface established underneath them at all | 71% |
| stand 0.20 m or more clear of it (eligible for clearing) | **19%** |
| median height above the local ground | **+0.12 m** |

For comparison, the *correct* detections sit a median +0.10 m above the same
fitted ground. The false ones are not distinguishable from real rocks by height,
because most of them are the same height as rocks. Of the ones that *are*
detached, the beam test works well — 87% of them do accumulate enough
contradiction to be retracted — so the machinery is not the limit.

Pushing the height gate down trades directly against real rocks, measured over
the whole recording:

| height gate | false cells left | change | rock voxels destroyed | rock coverage | weakest rock |
|---:|---:|---:|---:|---:|---:|
| (no clearing) | 2,302 | — | 0 | 0.650 | 0.294 |
| 0.30 m | 2,233 | −3.0% | 207 | 0.687 | 0.294 |
| 0.25 m | 2,214 | −3.8% | 343 | 0.687 | 0.294 |
| **0.20 m** | 2,161 | **−6.1%** | 656 | 0.687 | 0.294 |
| 0.15 m | 2,089 | −9.3% | 1,197 | 0.676 | **0.235** |
| 0.10 m | 1,836 | −20% | 1,984 | 0.682 | 0.412 |
| 0.05 m | 1,166 | −49% | 2,747 | 0.629 | 0.294 |
| 0.00 m | 826 | −64% | 3,214 | **0.512** | **0.176** |

(This table applies the retraction once at the end rather than during
accumulation, which is why its control count is 2,302 where the side-by-side
above reads 2,211; the shape of the trade is the point, not the last digit.)

At 0.15 m the weakest rock starts losing coverage. Below that the layer is
removing rocks to remove fog. **0.20 m is the setting to run, and 0.25 m is the
setting to run if anything about the rocks changes.** Requiring two
contradicting sweeps instead of three changes almost nothing (−145 cells against
−141), which says the same thing from the other side: what limits this is the
geometry, not how much evidence is demanded.

## What it will not do

It never touches what the model says on the current pass — only the remembered
map. It never touches space no beam has crossed, so a rock the robot has driven
away from is safe however stale it gets. It leaves alone anything that took a
return in the same sweep. And a fresh return puts evidence back faster than a
contradiction takes it away, so an obstacle coming back into view is restored
rather than lost.

Deliberately absent: no decay with age, and no rule holding a detection back
until it has been confirmed several times. The earlier confirmation experiment
cost more than half the coverage on the weakest rocks, and age is not evidence —
the previous audit's finding that 2,077 of 2,162 wrong cells had not refreshed
in a minute turns out to mean the robot drove away, not that they were floating.

## How it decides

Three things, kept apart on purpose.

*Occupancy evidence* is geometric and knows nothing about the model. Each sweep
either puts a return in a 5 cm cell or sends a beam through it to something
further away, and those accumulate against each other the way they do in an
occupancy grid — [OctoMap](https://www.arminhornung.de/Research/pub/hornung13auro.pdf)
is the reference. Space no beam has crossed accumulates nothing.

Beams are traced from where they were actually measured, one viewpoint per raw
scan, not one per merged sweep. That matters more than it sounds: the sensor
covers up to **6.9 cm** during a single 0.05 s sweep on this recording, which is
larger than the 5 cm cell being judged. Folding that into a single viewpoint
would mean widening every cell's angular target, and a wider target means more
beams count as having gone through — the error would land squarely on the side
of deleting real things.

*Surface support* is a local ground model: a plane fitted per 10 cm cell from
the lowest returns seen in at least two separate sweeps around it, carrying a
slope and a residual, with the cell under test left out of its own fit.
Deliberately not a single minimum and not a point count — dense accumulated fog
is perfectly capable of being both.

*Suspicion* is the gate between them. Standing clear of that surface makes a
detection eligible and nothing more; beams still have to contradict it. A rock's
top is above its neighbours too.

## Where it is in the dashboard

Two new places.

**Map evaluation** is a new card under Deploy. It replays a labelled recording
and grades the map — per-rock coverage, acquisition timing, wrongly-claimed
ground — with a tick-box, *Clear looked-through predictions*, that adds the
cleaned map beside the untouched one. The replayed geometry and the model's
scores are both cached, so the first run on a recording takes a few minutes and
changing the clearing settings afterwards takes seconds. Finished evaluations
are listed on the Training tab under **Map evaluations**.

**Live** has the same tick-box, *Forget predictions the beams go through*, off
by default, on the run form and also in the viewer's own panel and the browser
control panel so it can be turned on mid-run. The two settings that matter —
height above the ground, and contradictions needed — sit beside it under
Advanced.

## What it cost

Training: nothing. This does not touch the models or the dataset, and no
checkpoint changes because of it.

Runtime: one beam test per remembered detection per pass, over the detections
only — nothing below threshold is touched. Measured over the same 2,599 passes
on a machine that was also training: maintaining the map alone costs 118 ms a
pass, and maintaining it *and* a cleared copy costs 324 ms, so the clearing
itself is roughly 90 ms on top of a 118 ms map. That is on the processor, not
the graphics card, and it is small beside the model pass it sits behind. Off by
default, so nothing changes unless it is asked for.

Both runs produced exactly the same control map — 2,211 wrongly-claimed cells
either way — which is the check that the clearing arm is not quietly perturbing
the thing it is being compared against.

## The bigger thing this uncovered

Reading each scoring pass on its own and reading the map it builds give very
different answers, and the gap is a factor of three:

| | claimed cells with a rock on them |
|---|---:|
| each pass read on its own, pooled | **0.416** |
| the accumulated map, same scores, same threshold | **0.136** |

So the map really does lose most of the model's precision — your instinct about
that was right, and it is a much bigger effect than the mid-air phantoms are.
But the cause is not staleness exactly. It is that **the map is an OR over
time.** Predictions are stored per 5 cm voxel and a 10 cm ground cell is
displayed as the maximum over its whole column, so the highest answer anything
in that column was *ever* given is the answer that shows — a fresh "clear" on
the floor cannot overwrite a stale "rock" 12 cm above it, because they are
different voxels.

That OR is also what buys the coverage. Replaying the same scores under
different storage rules, every 2nd pass, everything else identical:

| what the map stores | cells claimed | on a rock | precision | rock coverage | weakest rock |
|---|---:|---:|---:|---:|---:|
| newest per 5 cm voxel, column max — **what the rig does** | 2,436 | 358 | 0.147 | **0.689** | 0.438 |
| newest per voxel, but a rescored column drops what it did not rescore | 752 | 201 | 0.267 | 0.397 | 0.211 |
| newest per 10 cm ground column | 697 | 195 | 0.280 | 0.390 | 0.158 |
| newest per ground column, best candidate of that pass | 764 | 211 | 0.276 | 0.419 | 0.211 |

Any of the forgetting rules roughly doubles precision and removes about 70% of
the wrongly-claimed ground — **and halves the rock coverage**, taking the
weakest rock from 0.44 to about 0.2. That is the same trade the earlier
confirmation experiment fell into, arrived at from the other direction.

Evidence-based clearing is the only rule measured here that moves off that
curve rather than along it: wrongly-claimed cells down and coverage *up*,
because it removes only what the sensor positively contradicted rather than
everything it has not lately re-confirmed. It just cannot reach very much.

**So the map question is not "how do we forget faster".** It is that the rig is
currently showing the union of every detection it has ever made, which is a
deliberate-looking choice nobody made, and the coverage number everyone quotes
depends on it. Whether that is the right display is a navigation question, not
a perception one — but it should be a decision rather than a side effect of
storing predictions in 3D and reading them in 2D.

## The finding that should drive the next round

Those ground-level false positives are the thing to attack, and one measurement
from tonight's training says how *not* to attack them. The classifier was given
a new input — how high its candidate sits above the lowest point of its own
neighbourhood — on the theory that a floating return and a rock can currently
hand it the same tensor. The channel is real and the model does read it, but it
learned it backwards for this purpose: raising a candidate 30 cm lifts the
model's average confidence from 0.171 to 0.194. On a flat volleyball court
"higher" means "on a rock", and eleven of those courts is what it was trained
on. Adding the synthetic floating clumps on top did not reverse the sign.

So the missing negative is not a height cue. It is flat ground the model calls
rock, and nothing in the training set looks like the flat ground of this arena.

## It behaves the same on better models

The layer was run against three checkpoints over the same recording, and it
does the same thing to all of them: a few percent off the wrongly-claimed
ground, no rock lost, coverage flat or up.

| checkpoint | claimed, no rock | after clearing | rock coverage | after clearing |
|---|---:|---:|---:|---:|
| deployed — PointNet, stray only | 2,211 | 2,074 | 0.650 | 0.683 |
| PointNet, stray + phantom, all recordings | 2,122 | 2,041 | 0.694 | 0.691 |
| PointNet++, no clutter training | 1,690 | 1,548 | 0.675 | 0.680 |

In each of the three, exactly one rock loses cells to the clearing — 2 to 3% of
that rock's own cells — and it is never the weakest one.

The model change is the larger lever by some way: swapping the checkpoint
removes 521 wrongly-claimed cells where the clearing removes 137. They are not
alternatives — clearing costs nothing and stacks on whatever is deployed — but
if only one thing is going to be done, it is the model.

## Honest limits

One recording, one arena, one checkpoint. Lance has now informed model
selection several times over, so it is development data — these numbers say what
the layer does *there*, and a fresh competition-like recording is still the only
thing that would settle whether it generalises. The ground model interpolates up
to 30 cm past the returns that support it, which is what lets it cover small
gaps and is also the furthest it should ever be trusted to.
