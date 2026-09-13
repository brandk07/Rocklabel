# The clutter recipe, repeated and taken apart

**Deploy the combined recipe. Stop tuning it.** Adding the synthetic phantom
clumps to the deployed classifier's stray-jitter training is worth about
**+0.09 average precision on the competition recording**: 0.540 for what is
deployed today, against 0.634 averaged over the fifteen fits trained the same
way on all eleven recordings — 0.648 for the reference recipe alone. Every one
of thirty new fits beat the deployed checkpoint, the worst of them by 0.05.
That is far outside any noise here, and it is the one thing tonight produced
that you should act on.

**Nothing tried on top of it helped.** Three ideas, three seeds each, twice
over: half the stray jitter, phantom clumps that stop eating 8% of the rock
examples, and giving the model the candidate's own height. All three landed
inside the seed-to-seed spread. The most interesting of them failed in an
instructive way, below.

**And the 0.7112 checkpoint does not reproduce.** Thirty fits of that recipe
span 0.592 to 0.704. It was a good draw from a good recipe, not a setting to
chase.

## The numbers

Average precision on the labelled competition cache, three seeds per setting.

Trained on all eleven volleyball recordings — what you should be doing now:

| setting | seed 42 | 43 | 44 | mean ± sd |
|---|---:|---:|---:|---:|
| combined reference | 0.6215 | 0.6173 | 0.7044 | **0.648 ± 0.049** |
| half the stray jitter | 0.6050 | 0.6326 | 0.6321 | 0.623 ± 0.016 |
| phantoms that spare the rocks | 0.6157 | 0.6306 | 0.6438 | 0.630 ± 0.014 |
| + the candidate's height | 0.6256 | 0.6791 | 0.5924 | 0.632 ± 0.044 |
| the candidate's height alone | 0.6141 | 0.6433 | 0.6466 | 0.635 ± 0.018 |
| *deployed `cls-stray/trainall`* | 0.5399 | | | |

The same five settings with VolleyBallTest4 held out, which is the control that
reproduces the winning checkpoint's exact split:

| setting | seed 42 | 43 | 44 | mean ± sd |
|---|---:|---:|---:|---:|
| combined reference | 0.6611 | 0.6892 | 0.6509 | **0.667 ± 0.020** |
| half the stray jitter | 0.6653 | 0.6200 | 0.6884 | 0.658 ± 0.035 |
| phantoms that spare the rocks | 0.6596 | 0.6525 | 0.6536 | 0.655 ± 0.004 |
| + the candidate's height | 0.6833 | 0.6371 | 0.6745 | 0.665 ± 0.025 |
| the candidate's height alone | 0.6364 | 0.6283 | 0.6203 | 0.628 ± 0.008 |
| *the original winner, same split* | 0.7112 | | | |

**The noise floor is 0.014 to 0.049.** That is the spread between two runs of
one setting with only the seed changed. Every difference between settings above
is smaller than it. Read the table accordingly: there is one effect here
(phantom clumps, +0.11) and no others.

## Does holding a recording out still buy anything?

No. Pairing the combined recipe seed for seed, training on all eleven
recordings scores 0.019 *lower* on the arena than holding VolleyBallTest4 out —
inside the noise, in the unhelpful direction. Adding that recording back to
training changes nothing measurable.

What the two tables do show, put beside their volleyball scores, is that the
volleyball score is not worth selecting on. Over the fifteen held-out
checkpoints its ranking and the arena's agree at **Spearman 0.28**, and the best
checkpoint on the arena (0.689) was the sixth-best of fifteen on volleyball.

## PointNet++ gets to the same place by a different road — for free

Two more settings, three seeds each, trained the same way on all eleven
recordings: PointNet++ with no clutter augmentation at all, and PointNet++ with
the combined recipe.

| setting | seed 42 | 43 | 44 | mean ± sd |
|---|---:|---:|---:|---:|
| PointNet++, no clutter training | 0.6738 | 0.6746 | 0.6359 | **0.661 ± 0.022** |
| PointNet++, combined clutter training | 0.6391 | 0.6389 | 0.6582 | 0.645 ± 0.011 |
| *PointNet, combined clutter training* | 0.6215 | 0.6173 | 0.7044 | 0.648 ± 0.049 |

Two things fall out. Adding the clutter recipe to PointNet++ does **not** help
(−0.016, one seed of three up) — the architecture already gets where the
augmentation takes PointNet, so the two directions overlap rather than add. And
PointNet++ untouched lands at 0.661 against PointNet-with-clutter's 0.648, a
difference well inside the spread.

The old objection to PointNet++ was that it costs three times as much per
forward pass. **On this card it costs nothing extra:** a thousand candidate
balls take 39 ms through PointNet++ and 40 ms through PointNet, median of twelve
runs after warm-up. It is a bigger network (1.47M weights against 0.81M) that
does not fill the card any more fully than the smaller one does.

## Which checkpoint to actually run

Ranking scores do not settle this — the map does. All three replayed through
the operational pipeline over the whole 35-minute recording, graded at **equal
wrongly-claimed ground** so that no model is being flattered by a more
cautious threshold:

| at 2,200 wrongly-claimed cells | rock coverage | weakest rock |
|---|---:|---:|
| deployed — PointNet, stray only | 0.650 | 0.294 |
| **PointNet, stray + phantom, all recordings** | **0.694** | **0.364** |
| PointNet++, no clutter training | 0.693 | 0.294 |

| at 1,700 wrongly-claimed cells | rock coverage | weakest rock |
|---|---:|---:|
| deployed — PointNet, stray only | 0.632 | 0.294 |
| **PointNet, stray + phantom, all recordings** | **0.694** | **0.364** |
| PointNet++, no clutter training | 0.675 | 0.294 |

The deployed checkpoint cannot get below about 1,650 wrongly-claimed cells at
*any* threshold; both new ones can. And the weakest rock — the one that decides
whether the arena is safe to drive — goes from 0.29 to 0.36 with the combined
recipe and stays at 0.29 with PointNet++.

**Run `clutter/cls-both-s44/trainall`.** It beats the deployed checkpoint on
every measure at every matched operating point, it is the best of the three on
the weakest rock, and it is the smaller network.

Two cautions on that choice. It is the top checkpoint of fifteen on a recording
that has now driven selection many times over, so some of its 0.704 is selection
on development data; the *recipe* average of 0.648 is the number to expect from
a fresh fit. And it is an odd checkpoint: early-stopped at epoch 5 of a run that
gave up at 15, on the worst validation score of its three seeds.

Which leads to the sharpest result in this table, and the one worth acting on
next.

## Checkpoint selection is on the wrong signal too

Across all twenty-one no-holdout fits, the validation score that picks `best.pt`
inside every run correlates with the arena at **Spearman 0.12** — that is, not
at all. The epoch it stops at carries nothing either (0.05).

| | Spearman vs arena average precision |
|---|---:|
| volleyball validation score (picks `best.pt`) | **+0.12** |
| epoch the validation score peaked at | +0.05 |
| held-out volleyball test score, 15 fits | +0.28 |

So it is not only *which recording to hold out* that is being decided on a
signal that does not track the arena. It is also **which epoch's weights to
keep**, inside every run, on every fit this project has ever made. The best
arena checkpoint tonight was kept at epoch 5 with the worst validation score in
its arm; the two fits with the highest validation scores of all (the PointNet++
ones, 0.877 and 0.875) sit third and fourth on the arena.

That is a bigger lever than any augmentation tried tonight, and it is cheap to
test: keep periodic epoch checkpoints through a run and score them all on the
arena. If the arena's best epoch is routinely somewhere else, the selection rule
is costing more than the recipe ever gained.

## The candidate's height: a clean failure worth understanding

The idea was sound. The classifier's input measures dx and dy from the
candidate but dz from the lowest point of the ball, so the candidate's own
height is the one coordinate never written down — lift a candidate half a metre
without moving a neighbour and the stored sample is byte for byte identical. A
rock and a return floating over that rock are exactly that pair.

The channel was added, and the model does read it. It just read it backwards
for this purpose: **raising a candidate 30 cm lifts the model's average
confidence from 0.171 to 0.194.** On eleven flat volleyball courts "higher"
means "on a rock" — the channel alone separates rock from clear at ROC-AUC 0.80
there — and that is the lesson it learned. Training it alongside tall synthetic
phantom clumps did not reverse the sign.

The arm that had the height input and no clutter negatives at all
(`cls-qz-plain`) was the only setting to lose clearly on the control split:
−0.039, worse on all three seeds. So the coordinate is not worth anything on
its own, and with clutter negatives it is worth nothing measurable either.

This is a representation that is present and readable and still does not help.
Worth knowing before someone has the same idea again.

## Where this leaves the model

The arena's problem is not floating returns. Grading the deployed classifier on
the map it actually builds (see
[../map-evidence/summary.md](../map-evidence/summary.md)) says 81% of its false
detections sit within 12 cm of the ground — the same height as a rock — and the
accumulated map claims 73% of the arena floor. The missing negative is flat
arena ground that the model calls rock, and nothing in eleven recordings of a
volleyball court looks like it.

That is the next thing to attack, and none of tonight's augmentations touches
it.

## How it was run

`rocklabel-train ablate --suite clutter --folds VolleyBallTest4.reslam` for the
control and `--folds all` for the campaign; 1.9 hours on the graphics card for
thirty fits. Ordinary PointNet, geometry only, 30 epochs, batch 256, learning
rate 0.001, cosine decay, 15% tail-block validation with a two-second gap — the
winning checkpoint's settings unchanged except where an arm changes one of them.

Two things about reproduction. The phantom generator's height reference was
corrected after the original winner was trained, and a checkpoint carries no
source hash, so these fits are current-code replications rather than reruns.
And the dataset gained a fifth channel for the height experiment: it was
regenerated and verified bit-identical to the old one on all 2,428 frames
before anything was trained on it, and three existing checkpoints reproduce
their stored scores on the rebuilt cache to within 1.7e-5.
