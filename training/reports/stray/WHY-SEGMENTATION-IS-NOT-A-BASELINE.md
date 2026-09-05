# Do not benchmark against the whole-frame segmenter

*Written 3 September 2026, after a day of measuring it. If you are training a
new model and looking for a number to beat, read this first — the segmenter's
published scores are much better than the model actually is, and the reports
that produced those scores are still on disk.*

## The short version

The **whole-frame segmenter** — the model that labels every point of a scan in
one pass — finds **30 to 78 rocks out of 138** on the competition recording.
The **sliding-window classifier** trained against stray returns finds **135 to
137 of 138**, every time.

That is a threefold gap. It held across three different segmenter settings,
every fold tested, and every way of measuring it once the measurement was
honest. Beating the segmenter means almost nothing. Beat `cls-stray`:

```
training/experiments/deploy/cls-stray/trainall/best.pt
```

| what to compare against | rocks found (of 138) | contrast | lance PR-AUC |
|---|---|---|---|
| **cls-stray — use this** | **135–137** | +0.90 | 0.651 |
| seg-base (best segmenter) | 30–63 | +0.45 | 0.534 |
| seg-relief | 37–78 | +0.34 | 0.492 |
| the segmenter deployed before today | 46 | +0.28 | 0.223 |

"Contrast" is explained below. It is the number the old reports were missing.

## Why the old numbers said the opposite

Two reports say segmentation ties the classifier. Both are still on disk and
both are wrong in ways worth understanding, because the same traps are waiting
for any new model.

### Trap 1 — PR-AUC cannot see a model going blank

PR-AUC (the area under the precision-recall curve — a single number for "how
well does this model rank rocks above bare ground") is computed **only from the
ordering** of the scores. Multiply every score by a thousandth and PR-AUC does
not move.

That is not a hypothetical. One classifier checkpoint scored **0.403 PR-AUC on
the competition arena while its median confidence on an actual rock was
0.004**. It ranked rocks above ground perfectly respectably and emitted
essentially zero for everything on the screen. Every metric this project
tracked — PR-AUC, ROC-AUC, and the F1 score at a threshold chosen on a
different arena — is blind to that.

**So measure contrast too.** Contrast here means: the median confidence the
model puts on points that really are rock, minus the median confidence it puts
on bare ground, in the same 0-to-1 units the colour ramp spans. It is what a
person watching the live view actually sees. Measured on the competition
recording:

| model | ground | on a rock | contrast |
|---|---|---|---|
| classifier `cls-stray` | 0.055 | 0.981 | **+0.93** |
| segmenter `seg-base` | 0.000 | 0.956 | **+0.96** |
| segmenter deployed before today | 0.638 | 0.914 | +0.28 |
| segmenter `segdeploy-frame-18` | 0.000 | 0.139 | **+0.14** |
| segmenter `seg-stray` | 0.000 | 0.102 | **+0.10** |
| classifier `cls-base` | 0.000 | 0.004 | **+0.004** |

On the volleyball recordings all six of those sit between +0.82 and +0.95. They
all look fine at home. Some go flat away from home, and only contrast shows it.

### Trap 2 — `matched` grades on a population that is 20x too rich in rock

`rocklabel-train matched` re-scores a segmenter and a classifier on one shared
set of candidate spots so their numbers can be compared. The pooling is fine —
the tie it reports survives all three pooling methods (best-of, average, and
nearest-point). The problem is the population.

Those candidate spots come from the classifier's training format, and the
dataset generator **keeps every rock candidate but throws away 95% of the clear
ones** (`negative_keep_prob: 0.05` in the generator settings). The result is a
test set that is about **19% rock**, where the real arena is about **1–5%**. Both
models are let off the same hook, so the ranking between them is not rigged —
but the absolute scores (~0.78) describe a world with twenty times less bare
ground to reject than exists. They do not predict live behaviour and should
never be quoted as if they do.

## How the gap was actually measured

Four steps, each fixing a flaw in the one before. Reuse this recipe.

**1. Put both model families on the same grid.** A segmenter is scored per
point (~1% rock), a classifier per candidate ball (~19% rock). Those are
different populations and their scores are not the same measurement. The fix
that does not involve a rock-enriched dataset: reduce both outputs to a shared
**10 cm ground grid**, each cell taking the highest confidence that model placed
in it, then score cells. Same cells, same labels, same prevalence, and nothing
depends on how either model spells its answer.

**2. Count rocks, not points.** A model can score well per point by being
confidently right about a lot of ground and one big rock while missing every
other rock in the scene — which on screen reads as "it found nothing". So count
**a rock as found when at least 2 lit cells fall on it**, over rocks carrying at
least 15 real returns in that frame (a rock nobody's sensor hit is not a miss),
and count lit cells on bare ground as the cost.

**3. Separate a bad threshold from a bad model.** Sweep the threshold and ask
how many rocks each model finds when allowed a fixed budget of false cells per
frame. At 20 false cells per frame: `cls-stray` 54/54, `seg-base` 51/54,
`segdeploy-frame-18` 37/54, `seg-stray` 32/54, `cls-base` 25/54 — the last one
capped there at *any* threshold, so that one is genuinely short of capability
rather than badly calibrated.

**4. Score many checkpoints, not one.** Transfer to a new arena is a lottery.
Scoring all 34 leave-one-run-out checkpoints of a sweep on the competition
recording:

| setting | lance PR-AUC | contrast | runs that go flat |
|---|---|---|---|
| cls-base | 0.490 ± 0.144 | +0.27 ± 0.27 | **6 of 11** |
| cls-stray | **0.715 ± 0.079** | **+0.89 ± 0.06** | **0 of 11** |
| seg-base | 0.530 ± 0.070 | +0.53 ± 0.20 | 1 of 6 |
| seg-stray | 0.485 ± 0.104 | +0.39 ± 0.22 | 2 of 6 |

A single trained model tells you almost nothing about whether a *setting*
transfers. Half the un-augmented classifier's runs went flat.

**5. Use enough frames.** A difference of 0.05 PR-AUC was produced purely by
picking a different 25 frames of the same recording — it flipped the sign of a
result reported that morning. Use 100 frames or more, and treat anything under
about 0.05 as noise.

## What was tried to fix it, and why it failed

A real mechanism was found. A whole-frame segmenter reads the frame as one
object, so frame-level properties matter, and **every competition frame is
outside the training range on both of them**:

| | training frames | competition frames |
|---|---|---|
| vertical structure per frame | 0.10 – 0.26 m | **0.46 – 0.55 m** |
| points in the crop box | median 1,145 | **3,479** |

Training was a flat volleyball court. The arena has half a metre of berm and
wall in every frame, so "sticks up above the floor" — a cue that works
perfectly at home — is worthless there. The classifier is structurally immune:
its input is a 0.5 m ball with its own local ground subtracted and a fixed
point budget, so it never sees frame-level context at all.

That diagnosis is measured and it still stands. **The treatment did not work.**
An arm (`seg-relief`) trained directly against both gaps — the training floor
tipped up to 64 cm across the crop instead of 24 cm, and density thinning down
to a quarter of a frame instead of a half — was trained on four folds and
scored on 100 competition frames:

| fold | seg-base | seg-relief |
|---|---|---|
| Test10 | 0.564 / +0.84 / 63 rocks | 0.497 / +0.32 / 49 |
| Test11 | 0.601 / +0.44 / 38 | 0.385 / +0.02 / 37 |
| Test6 | 0.484 / +0.25 / 39 | 0.521 / +0.32 / 45 |
| Test9 | 0.488 / +0.29 / 30 | 0.567 / +0.73 / 78 |
| **mean** | **0.534 / +0.45** | **0.492 / +0.34** |

Two folds better, two worse, mean down on both. **So the frame-level domain gap
is real but is not the cause, or not the only one.** That is the honest state:
a sound diagnosis whose treatment failed, which means the explanation is
incomplete.

## Why it is unlikely to come good

- **The gap is large and stable.** Threefold in rocks found, across three
  settings and every fold. Effects that size do not usually come from one
  missing setting.
- **The obvious mechanism was measured, targeted, and did not pay.** The next
  hypothesis has to explain a gap that survives closing the one difference that
  could be measured.
- **It is erratic in a way the classifier is not.** Segmenter contrast swings
  from +0.02 to +0.96 across checkpoints of the same setting. A family where
  half the runs silently go blank on a new arena is not deployable whatever its
  average.
- **It costs 4x more to train** — about 26 minutes a fold against 6 — so every
  experiment on it buys less.
- **The classifier is already good enough**, and its remaining failure mode is
  understood and fixed (see below).

## What would change the verdict

Do not treat this as proof segmentation can never work. It is one arm, four
folds, two hours, on one competition recording. Specifically:

- **A second competition-like recording.** Everything here rests on one bag.
  A second arena is the single most valuable thing anyone could add.
- **Capping the class-imbalance correction is still untested** — but do not
  expect much, and do not chase the apparent gap between the two formats,
  because it is not real. See "The class imbalance is already handled" below.
- **Feeding it properly.** The live path randomly subsamples the cloud to the
  segmenter's point budget rather than thinning to training density. Thinning
  competition frames to training density was worth **+0.09 PR-AUC** for
  `seg-base` and doubled the rocks found for `seg-stray`. That is free and
  costs no training.

If someone picks this up, do those three before training anything else.

## The class imbalance is already handled — do not "fix" it

An obvious-looking idea, worth writing down because it does not survive the
arithmetic: the dataset generator builds the classifier's training set by
keeping every rock candidate and discarding 95% of the clear ones, giving a set
that is ~17.9% rock. The segmenter gets no such treatment — it labels every
point of the frame, ~0.98% of which are rock. Why not do the same for it?

Two reasons.

**You cannot.** A classifier sample is one 0.5 m ball, so dropping clear ones
just means fewer samples. For the segmenter the clear points *are* the input —
they are the scene geometry it reasons over. Deleting them deletes the ground
the rocks sit on.

**It is already done anyway, in the loss instead of the data.** Both pipelines
end up applying almost exactly the same total correction:

| | starting rock share | clear points weighted | rock points weighted | total ratio |
|---|---|---|---|---|
| classifier | ~1.08% | x0.05 (95% discarded) | x4.6 (in the loss) | **~92x** |
| segmenter | ~0.98% | x1 | x101 (in the loss) | **~101x** |

Both are "one over the true prevalence". The classifier splits the correction
across two steps; the segmenter does it in one. In expectation they are the
same reweighting, so giving the segmenter the classifier's treatment would
change essentially nothing. (The classifier's ~1.08% starting share is backed
out of the observed 17.9% and the 5% keep rate, and lands right next to the
segmenter's 0.98%, as it should — they are the same scenes.)

**And the prediction runs backwards from the symptom.** A x101 weight pushes
outputs *up*, not down: a point with a genuine 1-in-100 chance of being rock is
pushed toward an output of 0.505 for the segmenter against 0.044 for the
classifier. That is what the data shows — median confidence on bare ground in
the volleyball recordings is 0.087-0.154 for the segmenters and 0.038-0.056 for
the classifiers. **The segmenter over-calls at home.** Its near-zero outputs on
the competition arena are therefore not a starvation-of-rock-examples problem;
they are the features landing outside anything it saw in training.

The one real consequence of the split is about *metrics*, not training: the
classifier's reported 17.9% rock share is an artifact of the discarding, and
PR-AUC starts at the prevalence of whatever it is measured on. That alone makes
a classifier's 0.78 and a segmenter's 0.40 look like a chasm when the
underlying problems are equally hard. It is the same trap as Trap 2 above,
seen from the training side.

## What actually made the classifier transfer

One change, and it is the strongest transfer result this project has: training
against **stray returns** — 5% of every training sample turned into a return
sitting on no surface, slid along the line of sight it arrived on.

- On the volleyball recordings it is **free**: +0.0034 PR-AUC over 11 paired
  folds, p = 0.52, well inside the 0.017 noise floor.
- On the competition arena it is **decisive**: +0.093 PR-AUC, ahead on **11 of
  11** folds, p = 0.001, and contrast from +0.27 to +0.89 with the number of
  runs that go flat dropping from 6 of 11 to **0 of 11**.

The reason it cannot be seen at home is the reason it matters: every recording
this project owns was made over flat ground the sensor struck steeply, so
almost every return landed on a real surface. A leave-one-run-out table on that
data cannot measure a cue that is only false somewhere else. **If you are
training a new model, put clutter in the training data and check it on the
competition bag.** It did nothing for the segmenter, but for the classifier it
was the difference between a coin flip and a model that never fails.

## Reproducing any of this

Everything is in `training/reports/stray/`:

| file | what is in it |
|---|---|
| `summary.md` | the overnight sweep on volleyball data — where the augmentation looks like nothing |
| `lance_paired.txt` | every fold's with/without pair scored on the competition bag |
| `lance_confidence_contrast.txt` | the contrast table |
| `lance_reliability.txt` | all 34 checkpoints, PR-AUC and contrast, spread and flat-run counts |
| `lance_deploy_eval.txt` | the four deployment models plus two references |
| `seg_relief_lance.txt` | the arm that tried to close the frame-level gap |
| `matched/matched.md` | the tie that started this — read the population caveat above first |

The one-off scripts that produced the shared-grid, per-rock, threshold-sweep
and density measurements were written for this investigation and are not part
of the tool. If you need them again, the recipe in "How the gap was actually
measured" is enough to rebuild them; the fiddly parts are using the project's
own per-point ground truth with the fuzzy shell around each rock excluded, the
generator's robot-centred crop, the label file's own levelling angle, and the
0.05 second frame window the checkpoints were trained on. Get any of those
wrong and the numbers are quietly meaningless.
