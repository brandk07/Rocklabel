# The combined-clutter recipe, repeated and taken apart

One PointNet checkpoint - the stray-plus-phantom classifier held out on VolleyBallTest4 - scores better on the labelled competition cache than anything else this project has trained, and its own held-out volleyball score is mediocre, so no sweep picked by volleyball would ever have found it. This suite does not chase that number. It repeats the recipe under three seeds to find out whether it repeats at all, and changes one thing at a time around it: half the stray jitter, phantom clumps that stop eating 8% of the rock examples, and an extra input telling the model how high its candidate sits in its own neighborhood - the one coordinate the sample tensor has never carried. It also asks whether PointNet++, dismissed on volleyball scores, is worth another look now that there is an arena to judge on. Run it with --folds all, which fits every recording and leaves the competition cache as the only held-out thing; --folds VolleyBallTest4.reslam reproduces the original winner's exact split, which is the control for whether that checkpoint was luck.

Leave-one-run-out over 1 recordings: every setting is trained on all but one run and scored on the run it never saw. PR-AUC is the headline number — it is the one that stays honest when rocks are a small share of the samples.

**The column that decides anything here is the last one.** These folds are scored on a held-out volleyball recording, and the best classifier this project has trained scored 0.52 on its own held-out volleyball run while beating everything else on the competition arena. Where a setting's checkpoints have been scored on the labelled competition cache (`rocklabel-train lancebench`), that average is shown beside the volleyball one. Each row's competition score is over one training set: the no-holdout fits where a setting has them, its leave-one-out folds otherwise. A setting fitted without a holdout has no volleyball column at all, because there is no unseen recording left to score it on - which is rather the point. Each competition cell says which fits it covers; where a row's two columns describe different fits, they are different checkpoints and the competition one is the one to read.

## Every setting

**Normalized PR-AUC** is the same score with the fold's own rock share divided out: `(PR-AUC - rock share) / (1 - rock share)`. Guessing scores 0 and a perfect model scores 1, so recordings with very different numbers of rocks can be put on one line. Raw PR-AUC is kept beside it for continuity with the earlier sweeps. The two are never comparable *across* tasks — a segmenter graded per point and a classifier graded per candidate ball are still two different measurements after normalizing.

| setting | folds | PR-AUC | normalized | ROC-AUC | F1 | competition PR-AUC | what it is |
|---|---|---|---|---|---|---|---|
| Classifier · combined reference (seed 44) | 1 | **0.493** | 0.368 | 0.649 | 0.403 | **0.704** (1, no holdout) | The 'cls-both' setting again under seed 44, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| Classifier · clutter, plus the candidate's height (seed 43) | 1 | **0.543** | 0.430 | 0.743 | 0.438 | **0.679** (1, no holdout) | The 'cls-qz' setting again under seed 43, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| PointNet++ · no clutter training (seed 43) | 0 | **—** | — | — | — | **0.675** (1, no holdout) | The 'pp-plain' setting again under seed 43, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| PointNet++ · no clutter training | 0 | **—** | — | — | — | **0.674** (1, no holdout) | PointNet++ with no clutter augmentation at all - the control for the pair below. It is back in this project on the strength of the competition rescoring rather than on anything measured at the volleyball court: its geometry-only checkpoints average 0.6418 there against plain PointNet's 0.6169, which is the opposite of the ordering the volleyball folds gave, and is why the older dismissal of the architecture does not stand. The other half of that dismissal was its cost, and that does not stand either: measured on this card, a thousand candidate balls take 39 ms through PointNet++ and 40 ms through PointNet. Whatever it is three times as much of, it is not scoring latency here. |
| PointNet++ · combined clutter training (seed 44) | 0 | **—** | — | — | — | **0.658** (1, no holdout) | The 'pp-both' setting again under seed 44, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| Classifier · the candidate's height alone (seed 44) | 1 | **0.506** | 0.384 | 0.679 | 0.412 | **0.647** (1, no holdout) | The 'cls-qz-plain' setting again under seed 44, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| Classifier · phantoms that spare the rocks (seed 44) | 1 | **0.500** | 0.376 | 0.656 | 0.415 | **0.644** (1, no holdout) | The 'cls-keeppos' setting again under seed 44, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| Classifier · the candidate's height alone (seed 43) | 1 | **0.513** | 0.392 | 0.709 | 0.418 | **0.643** (1, no holdout) | The 'cls-qz-plain' setting again under seed 43, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| PointNet++ · combined clutter training | 0 | **—** | — | — | — | **0.639** (1, no holdout) | PointNet++ with the reference recipe's stray jitter and phantom clumps. The two directions that have shown anything on the competition arena are a better architecture and better clutter negatives, and neither has been tried with the other. This is whether they add up or overlap. |
| PointNet++ · combined clutter training (seed 43) | 0 | **—** | — | — | — | **0.639** (1, no holdout) | The 'pp-both' setting again under seed 43, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| PointNet++ · no clutter training (seed 44) | 0 | **—** | — | — | — | **0.636** (1, no holdout) | The 'pp-plain' setting again under seed 44, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| Classifier · half the stray jitter (seed 43) | 1 | **0.523** | 0.405 | 0.703 | 0.425 | **0.633** (1, no holdout) | The 'cls-lowstray' setting again under seed 43, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| Classifier · half the stray jitter (seed 44) | 1 | **0.523** | 0.405 | 0.705 | 0.417 | **0.632** (1, no holdout) | The 'cls-lowstray' setting again under seed 44, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| Classifier · phantoms that spare the rocks (seed 43) | 1 | **0.539** | 0.425 | 0.725 | 0.431 | **0.631** (1, no holdout) | The 'cls-keeppos' setting again under seed 43, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| Classifier · clutter, plus the candidate's height | 1 | **0.524** | 0.406 | 0.694 | 0.428 | **0.626** (1, no holdout) | The reference with one extra input: how high the candidate center sits above the lowest point of its own ball. The classifier is asked to label a point, and its tensor measures dx and dy from that point but dz from the ball's floor - so the query's own height is the one coordinate never written down. Lift a candidate half a metre without moving a neighbour and the stored sample does not change. A rock and a return floating over that rock are exactly that pair, and floating returns are the competition arena's failure mode. Measured on the training cache the channel alone separates rock from clear at ROC-AUC 0.80, so it is not a null input - but on this flat court 'high' means 'on a rock', and in the arena it has to mean 'suspect'. Which way it generalizes is the question. |
| Classifier · combined reference | 1 | **0.550** | 0.438 | 0.751 | 0.425 | **0.622** (1, no holdout) | The winning recipe, retrained on today's code: 5% of each ball's points slid along their line of sight, and 8% of samples replaced outright by a synthetic phantom clump labelled clear. The reference every other arm here is a single change away from. It is also the replication: the checkpoint being reproduced was trained before a height-referencing fix in the phantom generator, and carries no source hash, so nothing on disk can certify what executable produced it. Treat these three fits as the measurement and the old checkpoint as history. |
| Classifier · combined reference (seed 43) | 1 | **0.529** | 0.412 | 0.701 | 0.441 | **0.617** (1, no holdout) | The 'cls-both' setting again under seed 43, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |
| Classifier · phantoms that spare the rocks | 1 | **0.550** | 0.438 | 0.750 | 0.435 | **0.616** (1, no holdout) | The reference with one accounting change: a phantom clump may only replace a clear sample, and its draw probability is raised so the number of phantoms per batch is unchanged. As written, the augmentation replaces any sample, so 8% of the rock examples in every batch are thrown away and the loss weight is not adjusted for it. Nobody chose that; it fell out of how the augmentation was written. This is what the recipe scores when it is not also a 8% cut to the positives. |
| Classifier · the candidate's height alone | 1 | **0.510** | 0.389 | 0.679 | 0.418 | **0.614** (1, no holdout) | The height input with no clutter augmentation at all, against plain PointNet. The paired arm above adds the channel to a recipe whose synthetic phantoms are tall by construction, so a model there could reach a good score by reading the new channel as a phantom detector. This arm cannot: it never sees a phantom. It is what says whether the coordinate is worth anything by itself. |
| Classifier · half the stray jitter | 1 | **0.541** | 0.427 | 0.709 | 0.420 | **0.605** (1, no holdout) | The reference with the stray fraction halved, 5% to 2.5%. Stray jitter on its own is the one clutter setting measured to HURT on the competition cache - it loses on every fold - and it works by teaching tolerance to corrupted returns, which is also how a model learns to call clutter a rock. The phantom clump supplies the opposing signal. This asks whether the recipe still needs a full dose of the half that costs precision. |
| Classifier · clutter, plus the candidate's height (seed 44) | 1 | **0.508** | 0.386 | 0.685 | 0.386 | **0.592** (1, no holdout) | The 'cls-qz' setting again under seed 44, changing nothing else. Repeats like this are the only scale there is for reading a difference between two settings: on this data two seeds of one setting land as far apart as the effects being compared. |

## Head to head, paired fold by fold

Each row trains two settings on the exact same folds and compares them one fold at a time. `W/L` counts folds won and lost. The p-value is a Wilcoxon signed-rank test: below 0.05 means the pattern of wins is unlikely to be chance.

| comparison | folds | change in PR-AUC | same, normalized | W/L | p | verdict |
|---|---|---|---|---|---|---|
| Does the combined clutter recipe still help the hierarchical model? | 0 | — | — | — | — | not run yet |
| Is PointNet++ worth three times the forward pass once both are trained against clutter? | 0 | — | — | — | — | not run yet |
| Does halving the stray jitter cost anything on clean data? | 1 | -0.0088 ± 0.0000 | -0.0110 | 0/1 | 1.000 | no measurable difference (-0.0088, p=1.00) |
| Does keeping every rock example, at the same phantom dose, change the recipe's score? | 1 | -0.0000 ± 0.0000 | -0.0001 | 0/1 | 1.000 | no measurable difference (-0.0000, p=1.00) |
| Does telling the classifier how high its candidate sits help? | 1 | -0.0258 ± 0.0000 | -0.0322 | 0/1 | 1.000 | no measurable difference (-0.0258, p=1.00) |
| Is the height input worth more once clutter negatives are present? | 1 | +0.0139 ± 0.0000 | +0.0173 | 1/0 | 1.000 | no measurable difference (+0.0139, p=1.00) |

## Per-fold detail, raw PR-AUC

This is the table the two earlier sweeps report, and on its own it is misleading: a recording with more rocks starts higher for free.

| setting | Test4 |
|---|---|
| Classifier · combined reference | 0.550 |
| Classifier · combined reference (seed 43) | 0.529 |
| Classifier · combined reference (seed 44) | 0.493 |
| Classifier · half the stray jitter | 0.541 |
| Classifier · half the stray jitter (seed 43) | 0.523 |
| Classifier · half the stray jitter (seed 44) | 0.523 |
| Classifier · phantoms that spare the rocks | 0.550 |
| Classifier · phantoms that spare the rocks (seed 43) | 0.539 |
| Classifier · phantoms that spare the rocks (seed 44) | 0.500 |
| Classifier · clutter, plus the candidate's height | 0.524 |
| Classifier · clutter, plus the candidate's height (seed 43) | 0.543 |
| Classifier · clutter, plus the candidate's height (seed 44) | 0.508 |
| Classifier · the candidate's height alone | 0.510 |
| Classifier · the candidate's height alone (seed 43) | 0.513 |
| Classifier · the candidate's height alone (seed 44) | 0.506 |

## Per-fold detail, with rock share divided out

The same folds, scored so that guessing is 0 and perfect is 1. This is the table to read when asking *which recording is hard*.

| setting | Test4 |
|---|---|
| Classifier · combined reference | 0.438 |
| Classifier · combined reference (seed 43) | 0.412 |
| Classifier · combined reference (seed 44) | 0.368 |
| Classifier · half the stray jitter | 0.427 |
| Classifier · half the stray jitter (seed 43) | 0.405 |
| Classifier · half the stray jitter (seed 44) | 0.405 |
| Classifier · phantoms that spare the rocks | 0.438 |
| Classifier · phantoms that spare the rocks (seed 43) | 0.425 |
| Classifier · phantoms that spare the rocks (seed 44) | 0.376 |
| Classifier · clutter, plus the candidate's height | 0.406 |
| Classifier · clutter, plus the candidate's height (seed 43) | 0.430 |
| Classifier · clutter, plus the candidate's height (seed 44) | 0.386 |
| Classifier · the candidate's height alone | 0.389 |
| Classifier · the candidate's height alone (seed 43) | 0.392 |
| Classifier · the candidate's height alone (seed 44) | 0.384 |

How much of each recording is rock, in the units each model is graded in — this is exactly the amount the second table removes:

| graded per | Test4 |
|---|---|
| pointnet | 19.8% |
| pointnet_qz | 19.8% |

![](arm_ranking.png)

![](paired_deltas.png)

![](per_fold.png)
