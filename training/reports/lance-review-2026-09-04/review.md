# Lance obstacle detection review — 4 September 2026

**Campaign complete.** See [final-results.md](final-results.md) for the final outcome, checkpoint paths and deployment decision. The progress statements below document the earlier review stages.

**Update:** [The overnight continuation](continued-review.md) contains completed training comparisons, filtering ablations, the full-recording persistence audit, separate XY footprint metrics, and the confirmation-delay results. The campaign is still under evaluation; no model has been promoted.

The strongest next experiment is a controlled extension of the sliding-window classifier, with explicit tests for loss of rock coverage. A cleaner confidence map is insufficient: the existing phantom model already trades away substantial coverage on some real rocks. The campaign described below is now running, following the user's subsequent authorization to start immediately.

This review treats previous write-ups and code comments as claims to check, not as instructions or proof. Repository model-comparison guidance is in `AGENTS.md`. No claim of classifier/segmenter deployment parity is made here.

## Evidence inspected

The [run inventory](run-inventory.csv) captures 256 configurations present during inspection: 246 with held-out test metrics, 7 with validation-only deployment metrics, and 3 unfinished. The preceding run was still writing folds while the inventory was collected. These counts are a timestamped snapshot, not the final campaign state.

I inspected the dataset and cache builders, candidate canonicalization, PointNet/PointNet++/segmenter/BEV implementations, augmentation and loss, validation splits and threshold selection, checkpoint loading/export, live ingest/filtering/scoring/persistence, visual-audit bookkeeping, dashboard history, training logs, and the fullsweep, reflectivity, segdense, seed, stray and BEV reports. Some reports describe experiments whose checkpoint directories are no longer present; their numbers remain historical evidence rather than newly reproduced results. The latest dashboard replay recorded `stray/cls-phantom/loro_VolleyBallTest3.reslam/best.pt` on trimmed Lance, with an 8 m range and −0.10…+0.60 m floor band.

### What the earlier training establishes

| Evidence | Result | Interpretation |
|---|---|---|
| Full-sweep volleyball PointNet, geometry only | Mean held-out AP 0.7804 across 11 recordings | Useful classifier reference; not a Lance performance estimate |
| Raw-burst geometry-only PointNet | Mean AP 0.7319 in the reflectivity report | Supports keeping full sweeps; populations differ, so the score difference is not a paired deployment gain |
| Full-sweep PointNet++ geometry only | Mean AP 0.7646 | No reason from this measurement alone to pay for a larger classifier |
| Reflectivity added to raw-burst PointNet | Paired AP change −0.0061; reported p=0.966 | Geometry-only is a reasonable starting point |
| Stray augmentation on volleyball | 0.7804 → 0.7838 mean AP | Little in-domain difference |
| Historical paired Lance stray evaluation | +0.093 mean AP, ahead on 11/11 classifier checkpoints | Promising transfer evidence, but candidate ranking is not accumulated-map completeness |
| Phantom versus stock, completed volleyball folds | +0.000526 mean paired AP; 7 wins/4 losses | Essentially unchanged in-domain ranking |
| Both augmentations versus stray, 10 common completed folds at inventory time | −0.003155 mean paired AP; 5 wins/5 losses | No convincing in-domain improvement yet |

The old BEV/segmentation tables count 112 or 138 **rock sightings**, not that many distinct physical rocks. The Lance label file contains **12** rock IDs. Some historical scripts also define a hit as only two positive 10 cm cells. Those figures cannot establish collision avoidance, complete mapping, or competition readiness. The legacy `bev_lance.py` uses the label file's z band, which differs from the operational floor band. Its results are not interchangeable with the new audit.

## Fresh visual audit: the present noise/recall tradeoff

[Summary](audit/summary.md), [per-rock results](audit/per-rock.csv), and all nine selected images were inspected. This compares completed VB3-held-out classifiers: `cls-stray` at its stored threshold **0.64**, and `cls-phantom` at **0.79**. Both use their classifier inference path, newest prediction per native 5 cm voxel, then maximum confidence per shared **10 cm ground cell**.

The audit examined recording-relative **150–450 seconds**, with the full **−0.10…+0.60 m floor band**, 8 m range, 50 ms input windows and stride 10. It selected five visibility-rich five-second buckets; actual retained frame spans were 3.4–4.4 seconds. Nine distinct rocks had eligible observations. **Rocks 6, 7 and 8 were not audited** in this excerpt. Selecting one best interval per rock can still produce several observations of that rock because intervals selected for other rocks overlap its visibility; aggregation gives each physical rock equal weight.

| Measurement | Stray classifier | Phantom classifier |
|---|---:|---:|
| Median false-positive cells per selected interval | 133 | 69 |
| Same cells expressed as ground area, before any robot-footprint inflation | 1.33 m² | 0.69 m² |
| Rock 1 median coverage | 66.7% | 36.4% |
| Rock 13 median coverage | 52.9% | 0% |
| Rock 13 selected case | 8/13 cells covered | 0/13 cells covered |
| Rock 10 median coverage | 70.1% | 81.9% |

The images confirm reduced spurious confidence along scan arcs and arena edges, alongside missing coverage on rock 13 and a reduced patch on rock 1. Rock 10 improves; several other rocks change little. This is a tradeoff, not a uniform regression or a demonstrated replacement. Each model uses its own threshold, so this audit does **not** isolate augmentation from threshold effects.

Limits: this initial audit does not apply live floating-point filtering, does not reproduce a full-run persistent map, and favors visibility-rich intervals. Its 25% coverage “detected” flag is permissive and is not an obstacle-map acceptance criterion. No equal-FP sweep was produced for this initial audit; the campaign's evaluator now writes one separately. Do not turn the reduction from 133 to 69 false cells into a whole-run traversability improvement claim.

## Why phantoms can survive the current pipeline

1. **Training and Lance have very different neighborhood distributions.** Directly reading the existing caches gives 94,519 volleyball candidates and 89,612 Lance candidates. Median clear/rock counts are **66/55** on volleyball versus **377.5/192** on Lance. Only **0.73%/0.35%** of volleyball clear/rock neighborhoods exceed 256 returns; **63.1%/41.1%** do on Lance. These are descriptive, differently sampled caches, not a new deployment benchmark. Absolute density alone is therefore a dangerous rejection rule.

2. **The classifier does not retain original density above the point budget.** The generator saves counts, but PointNet consumes them only as a validity mask. Counts above 256 all produce a full mask. Max pooling summarizes feature extremes, not how broadly a shape is supported. The proposed support-statistics model tests whether regional means and spread help distinguish scattered returns from surfaces; it does not assume a raw-count threshold is transferable.

3. **The old phantom augmentation covers a narrow negative distribution.** Its synthetic clumps contain only 20–59 valid points. It replaces positives as well as negatives, lowering effective positive prevalence while retaining the original loss weight. It also previously subtracted a height minimum that might belong to a discarded point. The existing ray-jitter code uses ball-relative coordinates as a stand-in for the true sensor ray and can move points outside the original neighborhood. Its historical transfer gain is useful, but this is not a faithful sensor simulator.

4. **The local minimum is outlier-sensitive.** The full ball's lowest return determines every point's height, before subsampling. A low phantom can shift a whole neighborhood even when the phantom itself is not selected into the tensor. This is a plausible failure mechanism, not yet isolated experimentally. Changing the reference on an existing checkpoint would change its input contract; the campaign keeps that contract fixed.

5. **Filtering controls affect different outputs.** Live ingest applies the floating filter before storing the recent scoring cloud. Its default is a 0.5 m maximum above the local 5th-percentile height in 1 m columns, with sparse columns left alone. Lower floating returns survive by design. The statistical outlier filter is applied inside the Kalman heightmap, after the scorer's input has branched off. Adjusting that filter does not clean the classifier's red points. These are different products, not a malfunctioning slider.

6. **A transient prediction can become a persistent map entry.** The scorer replaces predictions only when the same native 3D voxel is scored again. There is no expiry, independent-observation count, or visibility-aware clearing. A phantom above the ground can survive subsequent ground observations in a different z voxel. This is a confirmed code mechanism; its contribution to this particular screenshot has not been measured. Adding unconditional decay would also erase occluded rocks, so it needs a separate replay experiment.

7. **“Confidence” is not calibrated collision probability.** Training retains only 5% of clear candidates and uses class-weighted BCE. F1 thresholds come from temporal validation tails on volleyball. Neither a red color nor a 0.99 sigmoid score establishes a 99% chance of a physical obstacle in Lance. The same physical rock can appear on both sides of a temporal split; the split blocks frame overlap, not identity overlap.

The scanner's raw ring/echo/ray provenance is not retained in the model's four-channel neighborhood input. A future beam-consistency filter would need upstream data plumbing and evaluation of retained rock returns. Statements about particular beam percentages in old comments were not independently reproduced in this inspection.

## Fixes and experimental additions

- Centralized model reconstruction from checkpoint config across training, live scoring, replay, dataset viewing, visual auditing and export. Live/replay/export previously omitted BEV geometry/channel/density options; a density-normalized model could load with the wrong preprocessing without a weight-shape error. This warrants revisiting affected visual BEV comparisons, not declaring BEV successful.
- Corrected the synthetic clump's height reference to use surviving points. Clarified that the legacy replacement changes class balance.
- Save the validation-selected threshold in **every best checkpoint**, not only after training finishes. Earlier interim checkpoints silently fell back to 0.5 in the viewer. Existing checkpoint files were not rewritten.
- Publish checkpoints atomically so a viewer cannot read a partially written file. Save/restore the random states for augmentation, sample order and dropout when resuming newly written runs. Legacy checkpoints remain loadable but cannot recover random states they never saved.
- Replace the coarse 0.01–0.99 F1-selection grid with an exact sweep of distinct validation scores. This fixes endpoint resolution; it does not solve cross-domain calibration or make F1 the final deployment objective.
- Add opt-in `pointnet_stats`: per-point features with LayerNorm, max/mean/spread pooling over the whole ball and central 15 cm disk, plus the central support fraction. It keeps the four-channel tensor and ignores intensity for this campaign. It does not consume absolute density.
- Add opt-in `--aug-phantom-mode matched`: diffuse ellipsoidal negatives contained within the candidate ball, retaining the clear sample's count and preserving every positive example. A 0.20 setting replaces 20% of clear samples, not 20% of all samples. This still needs real-world validation; it does not simulate all phantom mechanisms.
- Add separate operating-point tables to future visual audits, with equal rock weighting, persistent misses, worst-rock median coverage and false-positive area. State explicitly that equal-FP thresholds selected from Lance are evaluation-fitted diagnostics.
- Correct segmentation export's point count and metadata to describe whole frames and per-point outputs. A regression check exercises actual TorchScript export; ONNX runtime validation remains a separate deployment check.
- During verification, also fixed small configurable segmentation levels crashing when requesting more neighbors than exist, and a dashboard sensor-status crash when socket creation itself is denied.

The last two ancillary fixes and a historical config-matching fix were made after the campaign snapshot. They do not affect the running classifier training or its evaluations. The snapshot, source hash, manifest and logs preserve precisely what the campaign uses.

## Running campaign

Started **4 September, approximately 19:27 America/Chicago**, on the **RTX 2000 Ada Generation Laptop GPU (8 GB)**, after confirming the previous GPU job was gone. Supervisor PID and log location are in [campaign-process.json](campaign-process.json). The campaign has a maximum **12-hour** budget, with training capped at **8 hours** and the remainder reserved for audits. It may finish earlier; it records budget exhaustion rather than silently claiming every fit completed.

| Arm | Model | Stray fraction | Matched phantom probability on clear samples |
|---|---|---:|---:|
| pointnet-stray | Existing PointNet | 0.05 | 0 |
| pointnet-matched | Existing PointNet | 0.05 | 0.20 |
| stats-stray | Regional statistics classifier | 0.05 | 0 |
| stats-matched | Regional statistics classifier | 0.05 | 0.20 |

Each arm uses seeds **42, 43, 44**, held-out recordings **VB3, VB4, VB6**, geometry-only full-sweep inputs, up to **60 epochs**, patience 15, and batch 256: **36 planned fits**. VB3 connects to the user's recent model; VB4 and VB6 are difficult existing folds. All arms share the corrected training harness so an architecture effect is not confused with a threshold-code change. Other volleyball recordings participate in training with the existing temporal validation gap. Lance never enters the training cache.

This is an initial factorial campaign, not an assertion that all four arms deserve deployment. It trains the fixed matrix before evaluating rather than adapting hyperparameters to Lance results. Completed fits receive full-recording, sampled per-rock audits against `deploy/cls-stray/trainall/best.pt`. Evaluations preserve the operational floor band and write separate stored-threshold and operating-point results. The campaign does not automatically select, export, or deploy a winner.

The launch code is `rocklabel/train/lance_campaign.py`. It is dry by default, refuses an occupied GPU, refuses an existing output directory, freezes source for child processes, and bounds runtime. Outputs are under `training/experiments/lance-campaign-v1/`; [campaign.log](campaign.log) is the supervisor log. The first fit was verified advancing through epochs with roughly 1.4 GB GPU memory usage.

## Decision after the runs

A candidate should reduce false occupied area **without new persistent rock misses**, and its per-rock coverage, fragmentation and worst visible intervals must be examined before aggregate scores. Compare against both the stray deployment checkpoint and the recently tested phantom checkpoint, including their own stored thresholds and a separately identified equal-FP sweep. A two-cell or 25%-coverage hit does not meet the intended map requirement.

Before using a result as traversability input, additionally measure first reliable detection versus distance/time to collision, full-run stale false positives, and blocked area after the robot footprint and stopping margin are applied. Those limits require robot speed, footprint and controller behavior; they cannot be inferred from these labels. A rock classifier also does not label berms, walls, pits or unknown space as safe: absence of a rock detection cannot by itself authorize traversal.

The highest-priority follow-on is an ablation of temporal evidence and preprocessing with the retained **unfiltered** rock geometry as the denominator, so filtering away an obstacle cannot improve the score by removing it from evaluation. Another competition-like recording, held outside model development, would test whether improvements selected using Lance transfer further. Lance remains outside training here, but repeated use of it for development means it is not an untouched final test.

Architecture motivation is consistent with the original [PointNet paper](https://arxiv.org/abs/1612.00593), which describes critical points under max pooling, and [PointNet++](https://arxiv.org/abs/1706.02413), which addresses nonuniform point sampling. These motivate hypotheses; neither paper establishes that the new classifier beats the existing system on Lance.

## Verification

The affected training, model, live-scoring, visual-audit, matched-comparison and dashboard tests pass: **259 passed**. Python compilation and `git diff --check` also pass. ROS pytest plugin autoload was disabled because the environment lacks an unrelated ROS plugin dependency. TorchScript export emitted fixed-point-budget tracing warnings; the eager/serialized round trip passed. ONNX runtime was not tested.
