# Does reflectivity help?

PointNet and PointNet++, with and without the reflectivity channel, plus seed repeats that show how big a meaningless difference looks.

Leave-one-run-out over 11 recordings: every setting is trained on all but one run and scored on the run it never saw. PR-AUC is the headline number — it is the one that stays honest when rocks are a small share of the samples.

**A difference of nothing looks like 0.0078 PR-AUC on this data.** That is the average fold-level gap between two runs of the *same* setting with only the random seed changed. Any effect smaller than that is noise, whatever the average says.

## Every setting

**Normalized PR-AUC** is the same score with the fold's own rock share divided out: `(PR-AUC - rock share) / (1 - rock share)`. Guessing scores 0 and a perfect model scores 1, so recordings with very different numbers of rocks can be put on one line. Raw PR-AUC is kept beside it for continuity with the earlier sweeps. The two are never comparable *across* tasks — a segmenter graded per point and a classifier graded per candidate ball are still two different measurements after normalizing.

| setting | folds | PR-AUC | normalized | ROC-AUC | F1 | what it is |
|---|---|---|---|---|---|---|
| PointNet · shape only (seed 43) | 11 | **0.733 ± 0.174** | 0.678 ± 0.201 | 0.867 ± 0.104 | 0.613 ± 0.179 | Same setting as the shape-only arm, different random seed. Exists only to measure how far two identical settings land apart. |
| PointNet · shape only | 11 | **0.732 ± 0.175** | 0.677 ± 0.200 | 0.870 ± 0.098 | 0.625 ± 0.162 | PointNet with the reflectivity channel removed entirely. The control: whatever this scores is what pure geometry is worth. |
| PointNet · shape only (seed 44) | 11 | **0.731 ± 0.174** | 0.676 ± 0.202 | 0.868 ± 0.100 | 0.613 ± 0.173 | Second seed repeat of the shape-only arm. |
| PointNet · reflectivity unjittered | 11 | **0.728 ± 0.184** | 0.670 ± 0.221 | 0.860 ± 0.136 | 0.634 ± 0.177 | PointNet with reflectivity included and the reflectivity augmentation switched off, so the model may use the raw absolute brightness. If reflectivity helps anywhere, it helps most here — and the gap between this and the jittered arm is the size of the cue the augmentation deliberately destroys. |
| PointNet · shape + reflectivity (seed 43) | 11 | **0.728 ± 0.181** | 0.670 ± 0.215 | 0.863 ± 0.121 | 0.627 ± 0.179 | Seed repeat of the shape+reflectivity arm. |
| PointNet · shape + reflectivity (seed 44) | 11 | **0.726 ± 0.177** | 0.668 ± 0.210 | 0.860 ± 0.123 | 0.624 ± 0.171 | Second seed repeat of the shape+reflectivity arm. |
| PointNet · shape + reflectivity | 11 | **0.726 ± 0.180** | 0.668 ± 0.213 | 0.861 ± 0.120 | 0.621 ± 0.178 | PointNet with reflectivity included, using the standard augmentation (reflectivity randomly rescaled and shifted each sample). This is the setting the existing runs used. |
| PointNet++ · shape only | 11 | **0.723 ± 0.167** | 0.666 ± 0.190 | 0.866 ± 0.095 | 0.612 ± 0.163 | PointNet++ with no reflectivity. The shape-only control for the hierarchical model. |
| PointNet++ · shape + reflectivity | 11 | **0.714 ± 0.186** | 0.655 ± 0.213 | 0.855 ± 0.115 | 0.627 ± 0.167 | PointNet++ with reflectivity included and the standard augmentation. |
| PointNet++ · reflectivity unjittered | 11 | **0.711 ± 0.182** | 0.650 ± 0.217 | 0.847 ± 0.138 | 0.615 ± 0.174 | PointNet++ with reflectivity included and its augmentation off. |
| PointNet · reflectivity only | 11 | **0.224 ± 0.124** | 0.057 ± 0.081 | 0.546 ± 0.092 | 0.260 ± 0.117 | PointNet fed nothing but reflectivity — no coordinates at all. It cannot see shape, so its score is a direct read of how much the brightness numbers alone can separate rock from sand. |

## Head to head, paired fold by fold

Each row trains two settings on the exact same folds and compares them one fold at a time. `W/L` counts folds won and lost. The p-value is a Wilcoxon signed-rank test: below 0.05 means the pattern of wins is unlikely to be chance.

| comparison | folds | change in PR-AUC | same, normalized | W/L | p | verdict |
|---|---|---|---|---|---|---|
| PointNet: does adding reflectivity beat shape alone? | 11 | -0.0061 ± 0.0246 | -0.0091 | 6/5 | 0.966 | no measurable difference (-0.0061, p=0.97) |
| PointNet++: does adding reflectivity beat shape alone? | 11 | -0.0090 ± 0.0396 | -0.0114 | 5/6 | 0.638 | no measurable difference (-0.0090, p=0.64) |
| PointNet: does reflectivity help when its augmentation is switched off? | 11 | -0.0039 ± 0.0371 | -0.0070 | 5/6 | 0.898 | no measurable difference (-0.0039, p=0.90) |
| PointNet++: does reflectivity help when its augmentation is switched off? | 11 | -0.0119 ± 0.0406 | -0.0158 | 4/7 | 0.413 | no measurable difference (-0.0119, p=0.41) |
| PointNet: how much does the reflectivity augmentation cost? | 11 | +0.0022 ± 0.0166 | +0.0021 | 6/5 | 0.700 | no measurable difference (+0.0022, p=0.70) |
| Shape only: is PointNet++ better than PointNet? | 11 | -0.0090 ± 0.0184 | -0.0108 | 3/8 | 0.175 | no measurable difference (-0.0090, p=0.17) |
| Shape + reflectivity: is PointNet++ better than PointNet? | 11 | -0.0119 ± 0.0281 | -0.0131 | 5/6 | 0.240 | no measurable difference (-0.0119, p=0.24) |
| Noise floor: the same shape-only setting, two different seeds. | 11 | +0.0007 ± 0.0116 | +0.0009 | 5/6 | 0.898 | yardstick — the same setting twice, so this spread (+0.0007) is what no difference looks like |
| Noise floor: the same shape+reflectivity setting, two different seeds. | 11 | +0.0021 ± 0.0084 | +0.0026 | 8/3 | 0.413 | yardstick — the same setting twice, so this spread (+0.0021) is what no difference looks like |

## Per-fold detail, raw PR-AUC

This is the table the two earlier sweeps report, and on its own it is misleading: a recording with more rocks starts higher for free.

| setting | Test10 | Test11 | Test12 | Test2 | Test3 | Test4 | Test5 | Test6 | Test7 | Test8 | Test9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PointNet · shape only | 0.766 | 0.809 | 0.646 | 0.900 | 0.927 | 0.438 | 0.656 | 0.424 | 0.748 | 0.853 | 0.882 |
| PointNet · shape + reflectivity | 0.772 | 0.814 | 0.571 | 0.887 | 0.925 | 0.436 | 0.638 | 0.431 | 0.752 | 0.870 | 0.885 |
| PointNet++ · shape only | 0.764 | 0.801 | 0.644 | 0.886 | 0.918 | 0.444 | 0.631 | 0.436 | 0.752 | 0.798 | 0.878 |
| PointNet++ · shape + reflectivity | 0.758 | 0.818 | 0.578 | 0.889 | 0.897 | 0.456 | 0.650 | 0.348 | 0.733 | 0.849 | 0.876 |
| PointNet · reflectivity unjittered | 0.779 | 0.848 | 0.541 | 0.899 | 0.921 | 0.426 | 0.643 | 0.450 | 0.751 | 0.871 | 0.877 |
| PointNet++ · reflectivity unjittered | 0.762 | 0.845 | 0.529 | 0.870 | 0.907 | 0.448 | 0.637 | 0.390 | 0.745 | 0.815 | 0.872 |
| PointNet · reflectivity only | 0.115 | 0.483 | 0.278 | 0.218 | 0.286 | 0.211 | 0.183 | 0.048 | 0.187 | 0.098 | 0.358 |
| PointNet · shape only (seed 43) | 0.772 | 0.831 | 0.637 | 0.895 | 0.931 | 0.427 | 0.669 | 0.435 | 0.748 | 0.834 | 0.880 |
| PointNet · shape + reflectivity (seed 43) | 0.767 | 0.826 | 0.572 | 0.890 | 0.921 | 0.422 | 0.654 | 0.439 | 0.755 | 0.873 | 0.886 |
| PointNet · shape only (seed 44) | 0.769 | 0.814 | 0.636 | 0.897 | 0.924 | 0.410 | 0.668 | 0.456 | 0.738 | 0.856 | 0.877 |
| PointNet · shape + reflectivity (seed 44) | 0.771 | 0.825 | 0.565 | 0.891 | 0.920 | 0.450 | 0.651 | 0.433 | 0.737 | 0.862 | 0.884 |

## Per-fold detail, with rock share divided out

The same folds, scored so that guessing is 0 and perfect is 1. This is the table to read when asking *which recording is hard*.

| setting | Test10 | Test11 | Test12 | Test2 | Test3 | Test4 | Test5 | Test6 | Test7 | Test8 | Test9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PointNet · shape only | 0.753 | 0.727 | 0.484 | 0.876 | 0.902 | 0.320 | 0.611 | 0.396 | 0.702 | 0.835 | 0.839 |
| PointNet · shape + reflectivity | 0.760 | 0.734 | 0.375 | 0.861 | 0.901 | 0.317 | 0.591 | 0.404 | 0.707 | 0.854 | 0.843 |
| PointNet++ · shape only | 0.750 | 0.714 | 0.481 | 0.860 | 0.891 | 0.327 | 0.582 | 0.409 | 0.707 | 0.772 | 0.834 |
| PointNet++ · shape + reflectivity | 0.744 | 0.740 | 0.384 | 0.864 | 0.863 | 0.341 | 0.604 | 0.316 | 0.684 | 0.830 | 0.831 |
| PointNet · reflectivity unjittered | 0.767 | 0.782 | 0.331 | 0.875 | 0.896 | 0.305 | 0.596 | 0.424 | 0.705 | 0.855 | 0.833 |
| PointNet++ · reflectivity unjittered | 0.749 | 0.778 | 0.314 | 0.840 | 0.877 | 0.332 | 0.589 | 0.361 | 0.698 | 0.792 | 0.826 |
| PointNet · reflectivity only | 0.065 | 0.258 | -0.052 | 0.038 | 0.050 | 0.044 | 0.075 | 0.001 | 0.037 | -0.014 | 0.124 |
| PointNet · shape only (seed 43) | 0.760 | 0.757 | 0.472 | 0.871 | 0.908 | 0.306 | 0.625 | 0.408 | 0.701 | 0.814 | 0.836 |
| PointNet · shape + reflectivity (seed 43) | 0.754 | 0.751 | 0.377 | 0.865 | 0.895 | 0.300 | 0.609 | 0.412 | 0.710 | 0.858 | 0.845 |
| PointNet · shape only (seed 44) | 0.756 | 0.734 | 0.470 | 0.873 | 0.899 | 0.286 | 0.624 | 0.430 | 0.690 | 0.838 | 0.833 |
| PointNet · shape + reflectivity (seed 44) | 0.759 | 0.748 | 0.367 | 0.866 | 0.894 | 0.334 | 0.605 | 0.406 | 0.689 | 0.845 | 0.841 |

How much of each recording is rock, in the units each model is graded in — this is exactly the amount the second table removes:

| graded per | Test10 | Test11 | Test12 | Test2 | Test3 | Test4 | Test5 | Test6 | Test7 | Test8 | Test9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 5.3% | 30.3% | 31.4% | 18.7% | 24.8% | 17.4% | 11.7% | 4.6% | 15.6% | 11.0% | 26.8% |
| pointnet2 | 5.3% | 30.3% | 31.4% | 18.7% | 24.8% | 17.4% | 11.7% | 4.6% | 15.6% | 11.0% | 26.8% |

![](arm_ranking.png)

![](paired_deltas.png)

![](per_fold.png)
