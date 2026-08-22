# Full sensor sweeps, and per-point segmentation

Trained on frames built from whole 20 Hz sensor rotations (~1250 points in the crop box) instead of single ~4 ms sensor batches (~110 points). Adds the whole-frame segmenter, which the batch-sized frames were too sparse to train at all.

Leave-one-run-out over 11 recordings: every setting is trained on all but one run and scored on the run it never saw. PR-AUC is the headline number — it is the one that stays honest when rocks are a small share of the samples.

**A difference of nothing looks like 0.0168 PR-AUC on this data.** That is the average fold-level gap between two runs of the *same* setting with only the random seed changed. Any effect smaller than that is noise, whatever the average says.

## Every setting

**Normalized PR-AUC** is the same score with the fold's own rock share divided out: `(PR-AUC - rock share) / (1 - rock share)`. Guessing scores 0 and a perfect model scores 1, so recordings with very different numbers of rocks can be put on one line. Raw PR-AUC is kept beside it for continuity with the earlier sweeps. The two are never comparable *across* tasks — a segmenter graded per point and a classifier graded per candidate ball are still two different measurements after normalizing.

| setting | folds | PR-AUC | normalized | ROC-AUC | F1 | what it is |
|---|---|---|---|---|---|---|
| PointNet · shape + reflectivity | 11 | **0.781 ± 0.148** | 0.729 ± 0.187 | 0.878 ± 0.115 | 0.682 ± 0.151 | Plain PointNet with reflectivity included. |
| PointNet · shape only | 11 | **0.780 ± 0.151** | 0.730 ± 0.188 | 0.881 ± 0.109 | 0.676 ± 0.150 | Plain PointNet on single balls, shape only - the cheapest model, kept as the floor everything else has to beat. |
| PointNet++ · shape only (seed 43) | 11 | **0.768 ± 0.148** | 0.715 ± 0.181 | 0.882 ± 0.097 | 0.669 ± 0.153 | Same setting as the shape-only PointNet++ arm, different random seed. Exists only to measure how far two identical settings land apart, so a difference between two real arms can be called real or not. |
| PointNet++ · shape only | 11 | **0.765 ± 0.158** | 0.712 ± 0.189 | 0.879 ± 0.103 | 0.665 ± 0.155 | PointNet++ scoring one 0.5 m ball at a time, with the reflectivity channel removed. This is the arm to line up against the same-named arm of the reflectivity suite: same model, same settings, same folds - the only difference is that a frame here is a whole sensor rotation. |
| PointNet++ · shape + reflectivity | 11 | **0.754 ± 0.167** | 0.699 ± 0.202 | 0.869 ± 0.120 | 0.660 ± 0.165 | PointNet++ on single balls with reflectivity added back, using the standard brightness jitter. |
| Segmentation · shape only (80 epochs) | 11 | **0.414 ± 0.161** | 0.407 ± 0.163 | 0.878 ± 0.093 | 0.427 ± 0.130 | The shape-only segmenter given 80 epochs instead of 30, with the early-stopping patience raised to match. Nothing else changes - same frames, same model, same seed - so the gap between this and the 30-epoch arm is purely what stopping too early cost. Not one of the eleven 30-epoch folds early-stopped: every one ran to the cap, the median best epoch was 28 of 30, and 7 of 11 peaked in the last five epochs. That makes 0.767 a floor, not a ceiling. |
| Segmentation · 80 epochs, finer levels | 11 | **0.403 ± 0.164** | 0.397 ± 0.166 | 0.882 ± 0.069 | 0.429 ± 0.138 | The 80-epoch segmenter looking at finer scales: the first of its three levels now keeps 1,024 of the frame's points instead of 512 and pools over a 0.10 m radius instead of 0.25 m. The labelled rocks measure 21-68 cm across, so the stock 0.25 m radius is a half-metre ball that swallows a whole rock - the smallest scale the model looked at was bigger than the thing it was hunting. Everything else matches the 80-epoch arm exactly, so the gap between them is the level geometry and nothing else. This is the combination of the two changes that measured as worth having, without the one that did not: more epochs helped (+0.019, p=0.032), finer levels look like they help, and keeping four times the frames did nothing at four times the cost. |
| Segmentation · shape only | 11 | **0.395 ± 0.160** | 0.388 ± 0.162 | 0.884 ± 0.074 | 0.410 ± 0.119 | PointNet++ labelling every point of a whole frame in one pass, shape only. The headline arm: it answers a rock question in one forward pass per frame instead of one per candidate ball. |
| Segmentation · shape + reflectivity | 11 | **0.384 ± 0.155** | 0.377 ± 0.156 | 0.883 ± 0.083 | 0.406 ± 0.111 | The whole-frame segmenter with reflectivity added back. Denser frames give the segmenter far more brightness context than a single ball has, so this is where reflectivity has its best chance of paying off. |
| Segmentation · shape only (seed 43) | 11 | **0.382 ± 0.147** | 0.375 ± 0.148 | 0.882 ± 0.075 | 0.410 ± 0.108 | Seed repeat of the shape-only segmentation arm - the noise floor for the segmenter. |

## Head to head, paired fold by fold

Each row trains two settings on the exact same folds and compares them one fold at a time. `W/L` counts folds won and lost. The p-value is a Wilcoxon signed-rank test: below 0.05 means the pattern of wins is unlikely to be chance.

| comparison | folds | change in PR-AUC | same, normalized | W/L | p | verdict |
|---|---|---|---|---|---|---|
| Shape only: does whole-frame segmentation beat the sliding-window classifier? | 11 | -0.3700 ± 0.1386 | -0.3237 | 0/11 | 0.001 | hurts (-0.3700, p=0.001, 22.1x the noise floor) |
| Shape + reflectivity: does whole-frame segmentation beat the classifier? | 11 | -0.3700 ± 0.1381 | -0.3215 | 0/11 | 0.001 | hurts (-0.3700, p=0.001, 22.1x the noise floor) |
| Shape only: is PointNet++ better than plain PointNet? | 11 | -0.0158 ± 0.0307 | -0.0186 | 3/8 | 0.102 | no measurable difference (-0.0158, p=0.10) |
| PointNet++: does adding reflectivity beat shape alone? | 11 | -0.0103 ± 0.0119 | -0.0126 | 1/10 | 0.014 | consistently hurts, but by less than changing the random seed does (-0.0103 vs a 0.0168 noise floor) — real, not worth acting on |
| Segmentation: does adding reflectivity beat shape alone? | 11 | -0.0104 ± 0.0273 | -0.0104 | 6/5 | 0.320 | no measurable difference (-0.0104, p=0.32) |
| PointNet: does adding reflectivity beat shape alone? | 11 | +0.0003 ± 0.0163 | -0.0011 | 5/6 | 0.700 | no measurable difference (+0.0003, p=0.70) |
| Noise floor: the same PointNet++ setting, two different seeds. | 11 | +0.0035 ± 0.0195 | +0.0038 | 4/7 | 0.831 | yardstick — the same setting twice, so this spread (+0.0035) is what no difference looks like |
| Noise floor: the same segmentation setting, two different seeds. | 11 | -0.0123 ± 0.0282 | -0.0125 | 3/8 | 0.206 | yardstick — the same setting twice, so this spread (-0.0123) is what no difference looks like |
| Segmentation: does it just need longer than 30 epochs? | 11 | +0.0192 ± 0.0245 | +0.0195 | 9/2 | 0.032 | helps (+0.0192, p=0.032, 1.1x the noise floor) |
| Segmentation at 80 epochs: do finer levels beat the stock geometry? | 11 | -0.0104 ± 0.0328 | -0.0105 | 3/8 | 0.278 | no measurable difference (-0.0104, p=0.28) |
| Segmentation: longer training and finer levels together, against the original 30-epoch arm. | 11 | +0.0088 ± 0.0271 | +0.0090 | 8/3 | 0.206 | no measurable difference (+0.0088, p=0.21) |

## Per-fold detail, raw PR-AUC

This is the table the two earlier sweeps report, and on its own it is misleading: a recording with more rocks starts higher for free.

| setting | Test10 | Test11 | Test12 | Test2 | Test3 | Test4 | Test5 | Test6 | Test7 | Test8 | Test9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PointNet++ · shape only | 0.811 | 0.839 | 0.634 | 0.902 | 0.927 | 0.483 | 0.665 | 0.544 | 0.795 | 0.907 | 0.903 |
| Segmentation · shape only | 0.490 | 0.627 | 0.258 | 0.419 | 0.605 | 0.150 | 0.214 | 0.446 | 0.287 | 0.316 | 0.529 |
| PointNet · shape only | 0.827 | 0.873 | 0.604 | 0.922 | 0.928 | 0.515 | 0.657 | 0.633 | 0.790 | 0.913 | 0.921 |
| PointNet++ · shape + reflectivity | 0.809 | 0.837 | 0.602 | 0.899 | 0.931 | 0.458 | 0.654 | 0.521 | 0.794 | 0.892 | 0.901 |
| Segmentation · shape + reflectivity | 0.469 | 0.603 | 0.265 | 0.434 | 0.586 | 0.151 | 0.232 | 0.401 | 0.293 | 0.250 | 0.543 |
| PointNet · shape + reflectivity | 0.835 | 0.847 | 0.592 | 0.915 | 0.928 | 0.511 | 0.658 | 0.674 | 0.794 | 0.912 | 0.921 |
| PointNet++ · shape only (seed 43) | 0.811 | 0.848 | 0.623 | 0.902 | 0.917 | 0.500 | 0.659 | 0.598 | 0.783 | 0.896 | 0.912 |
| Segmentation · shape only (seed 43) | 0.488 | 0.556 | 0.284 | 0.413 | 0.577 | 0.148 | 0.231 | 0.431 | 0.238 | 0.320 | 0.522 |
| Segmentation · shape only (80 epochs) | 0.501 | 0.651 | 0.277 | 0.463 | 0.625 | 0.141 | 0.249 | 0.413 | 0.345 | 0.338 | 0.549 |
| Segmentation · 80 epochs, finer levels | 0.435 | 0.637 | 0.274 | 0.449 | 0.608 | 0.139 | 0.199 | 0.469 | 0.317 | 0.344 | 0.568 |

## Per-fold detail, with rock share divided out

The same folds, scored so that guessing is 0 and perfect is 1. This is the table to read when asking *which recording is hard*.

| setting | Test10 | Test11 | Test12 | Test2 | Test3 | Test4 | Test5 | Test6 | Test7 | Test8 | Test9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PointNet++ · shape only | 0.794 | 0.765 | 0.486 | 0.878 | 0.904 | 0.355 | 0.619 | 0.513 | 0.756 | 0.895 | 0.861 |
| Segmentation · shape only | 0.486 | 0.620 | 0.244 | 0.413 | 0.600 | 0.139 | 0.208 | 0.443 | 0.281 | 0.311 | 0.522 |
| PointNet · shape only | 0.812 | 0.814 | 0.444 | 0.902 | 0.905 | 0.395 | 0.611 | 0.608 | 0.749 | 0.903 | 0.888 |
| PointNet++ · shape + reflectivity | 0.792 | 0.762 | 0.441 | 0.874 | 0.908 | 0.324 | 0.607 | 0.488 | 0.754 | 0.879 | 0.858 |
| Segmentation · shape + reflectivity | 0.465 | 0.595 | 0.252 | 0.428 | 0.580 | 0.141 | 0.227 | 0.397 | 0.287 | 0.245 | 0.536 |
| PointNet · shape + reflectivity | 0.820 | 0.776 | 0.427 | 0.893 | 0.905 | 0.390 | 0.612 | 0.652 | 0.754 | 0.902 | 0.887 |
| PointNet++ · shape only (seed 43) | 0.794 | 0.778 | 0.470 | 0.877 | 0.891 | 0.377 | 0.613 | 0.571 | 0.741 | 0.883 | 0.875 |
| Segmentation · shape only (seed 43) | 0.485 | 0.547 | 0.270 | 0.407 | 0.572 | 0.137 | 0.225 | 0.428 | 0.231 | 0.315 | 0.514 |
| Segmentation · shape only (80 epochs) | 0.498 | 0.644 | 0.264 | 0.458 | 0.620 | 0.130 | 0.244 | 0.410 | 0.340 | 0.334 | 0.542 |
| Segmentation · 80 epochs, finer levels | 0.431 | 0.630 | 0.261 | 0.442 | 0.603 | 0.128 | 0.193 | 0.466 | 0.311 | 0.339 | 0.561 |

How much of each recording is rock, in the units each model is graded in — this is exactly the amount the second table removes:

| graded per | Test10 | Test11 | Test12 | Test2 | Test3 | Test4 | Test5 | Test6 | Test7 | Test8 | Test9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 8.2% | 31.6% | 28.8% | 20.1% | 24.4% | 19.8% | 12.0% | 6.3% | 16.2% | 10.8% | 29.9% |
| pointnet2 | 8.2% | 31.6% | 28.8% | 20.1% | 24.4% | 19.8% | 12.0% | 6.3% | 16.2% | 10.8% | 29.9% |
| pointnet2_seg | 0.7% | 1.9% | 1.8% | 1.1% | 1.2% | 1.3% | 0.7% | 0.5% | 0.8% | 0.7% | 1.6% |

![](arm_ranking.png)

![](paired_deltas.png)

![](per_fold.png)
