# Overnight continuation — 5 September 2026

**Campaign complete.** See [final-results.md](final-results.md) for the final outcome, checkpoint paths and deployment decision. The progress statements below document the earlier review stages.

The 36-fit matrix has finished training. Its full-recording, sampled visual audits are running. A paired follow-on is queued to fit PointNet with and without matched phantom augmentation on all volleyball recordings, using seeds 42 and 43. It waits for the original sweep, stops if that sweep fails, and shares the **original deadline of 07:27:49 America/Chicago on 5 September**. It does not add another 12 hours. No candidate has been promoted to deployment.

The detailed initial code review is in [review.md](review.md). The strongest new finding is that long-lived accumulated predictions materially increase false occupied area. However, simple confirmation rules can delay mapping real rocks by minutes, so a cleaner final map is not sufficient evidence for using one in navigation.

## Completed training and the initial transfer screen

The PointNet matched-phantom arm averages **0.7204 AP** across nine held-out volleyball evaluations (three seeds × three recordings), versus **0.7023** for the corresponding stray-only arm. These are nine paired fits on three distinct held-out recordings; they are not nine independent environments. Effects vary by seed and recording. VB4 improves for seeds 42 and 43 but declines for seed 44. VB6 declines for seeds 42 and 43 but improves for seed 44. This supports a paired final fit, not a universal augmentation-win claim.

The CPU [pilot screening table](pilot-screening.csv) compares completed seed-42 candidates on the same five visibility-rich Lance intervals used earlier, with nine eligible rocks. It preserves the −0.10…+0.60 m floor band and unfiltered inputs. The table's `oracle_at_reference_fp` columns choose thresholds using these evaluation data; those thresholds are not deployed calibrations.

The matched PointNet VB4 candidate is the most promising of the initial matched-PointNet pilot fits. At stored thresholds it has 76.1% mean per-rock median coverage and 103.8 mean false cells, against 72.9% and 89.8 for the old stray train-all reference. At a diagnostic threshold giving at most the reference's false-cell count, it has 74.9% coverage and 89.6 false cells. This small result is development evidence from this excerpt. It is not sufficient to choose a deployment model.

I read that candidate's summary, per-rock table, and all nine selected images. Rock 13 improves from 7/13 to 12/13 cells in its selected view; rock 12 improves from 7/15 to 10/15. Rock 4 remains poorly covered in one view (6/35 cells, three fragments). Rock 11 loses one cell in its selected view; rock 9 also loses one. The automatically selected case set therefore contains both improvements and remaining weaknesses. The new statistics classifiers have mixed volleyball results and generally worse false-detection tradeoffs in this initial pilot; they have not earned deployment preference. The full sweep is still being evaluated.

The same matched PointNet VB4 checkpoint's broader [12-rock audit](../../experiments/lance-campaign-v1/audit-05/summary.md) is now complete. I read its summary, per-rock CSV and all 12 selected images. Across the selected five-second intervals throughout the recording, mean per-rock median coverage is **70.7% versus 65.6%** for the reference, and mean false cells are **119.0 versus 127.3**, at their respective stored thresholds of 0.637 and 0.710. Ten rock medians improve, one is unchanged, and rock 12's median declines. Rock 4 remains weak at 15.7% median coverage. In the selected rock 13 view, coverage falls from 2/5 to 1/5 cells and false cells increase from 132 to 159. The candidate improves both aggregate measures in this development audit but does not dominate every rock or view. Its checkpoint was trained with VB4 held out; the queued all-volleyball fits must be evaluated independently.

The final volleyball AP means across nine fits per arm are 0.7023 (PointNet stray), 0.7204 (PointNet matched), 0.7189 (statistics stray), and 0.7157 (statistics matched). [Campaign tables](campaign-summary.json) distinguish completed training from completed audits and preserve each fold and seed. Statistics pooling has not produced the intended reliable improvement.

## Filtering and short-window accumulation

These CPU ablations use **unchanged unfiltered rock geometry as the denominator**. A filter cannot improve its score by deleting the obstacle from the ground truth. They run on cropped 50 ms scoring windows, rather than the exact live pre-ingest batches, and the temporal examples restart accumulation in each selected interval. They diagnose tradeoffs and do not change live defaults.

| Input filter | Mean per-rock median coverage | Mean false cells | Worst raw rock-cell retention in an interval |
|---|---:|---:|---:|
| None | 72.9% | 89.8 | 100% |
| Existing floating filter, 0.5 m | 74.4% | 90.0 | 100% |
| At least three other returns within 0.10 m | 64.2% | 77.2 | 75% |
| At least three other returns within 0.15 m | 71.4% | 85.4 | 80% |
| Return within 0.10 m of previous sampled frame | 65.0% | 69.2 | 36.4% |

The previous-frame filter eliminates the detected patch on rock 9 in one interval: coverage falls from 10/11 cells to zero. The 0.10 m radius filter reduces rock 10 from 8/9 to 5/9 cells in one view. These failures disqualify those particular settings as a general cleanup fix.

Requiring two positive observations in the exact same native voxel reduces mean per-rock median coverage to 38.4%. Allowing nearby support within 0.15 m recovers it to 67.5%, but rock 12 still falls from full coverage to 25% in one interval. These sparse-window rules have substantial observation costs.

Changing the frozen classifier's height reference to the sampled neighborhood's first or fifth percentile also fails to produce a clear improvement: the fifth-percentile clamped variant changes coverage from 72.9% to 71.5% and false cells from 89.8 to 83.8. Rock 1 loses 20 percentage points in one interval. These are explicit input-contract ablations; an existing checkpoint's production preprocessing was not changed.

Artifacts: [filter results](filter-ablation/summary.csv), [temporal results](temporal-ablation/summary.csv), [height-reference results](height-ablation/summary.csv). All per-rock measurements are retained. The short-window ablation image sets are generated; only the image sets explicitly recorded in [visual-inspection.json](visual-inspection.json) have received a complete manual visual review.

## Full-recording persistence and footprint measurements

The [full-map baseline audit](full-map/summary.json) replays **2,599 sampled scoring windows over 2,131.4 seconds**, using the old stray train-all checkpoint at threshold 0.71. All 12 physical rocks are represented. The input remains unfiltered, at an 8 m range and the operational floor band. It reproduces native classifier scoring followed by latest-value-per-native-voxel accumulation, rather than the GUI's complete filtering and rendering pipeline.

At the end, the map has 79,708 native voxels and 3,533 scored 10 cm ground cells inside the arena. Using the existing 3D label-attribution convention, 2,162 are false-positive cells. Of those, **2,077 were last refreshed more than 60 seconds earlier**; their mean age is about 908 seconds. [The persistence plot](full-map-persistence.png) shows false area growing through the recording. Age is not visibility evidence: it does not establish that a stale point is safe to clear, and automatic decay would also remove occluded rocks.

### A distinction that matters for traversability

The previous audit reduces each ground cell to its highest-probability 3D representative, then tests that representative against the 3D rock label. A floating point above a rock can win this reduction. Removing it can expose a lower rock point and improve the 3D attribution score even though the occupied ground cell never changed.

I added a **separate XY footprint attribution** helper and regression check. It projects the annotated rock shapes into XY, uses the same boundary shell, and measures coverage against the same unfiltered observed rock-cell sets. The older 3D-attribution results remain available; they are not silently relabelled as footprint coverage. The two metrics answer different questions.

For the full baseline map, the mean per-rock **3D-attributed coverage is 65.9%**, whereas **occupied-footprint coverage is 99.5%**. The latter can receive credit from a positive point above a rock; it does not demonstrate that the model correctly identified that rock's surface. Conversely, it correctly records that the ground cell would be blocked. False area outside annotated XY footprints is **21.44 m²**, close to the 21.62 m² from 3D attribution. Both omit robot-footprint inflation.

| Accumulation rule | Mean final footprint coverage | Worst rock | False footprint area |
|---|---:|---:|---:|
| Existing native latest-value map | 99.5% | 95.7% | 21.44 m² |
| Two positive observations per native voxel | 97.2% | 85.0% | 15.60 m² |
| Three positive observations per native voxel | 95.5% | 85.0% | 12.87 m² |
| Latest value per ground cell | 57.8% | 35.3% | 5.49 m² |
| Ground-cell hysteresis: one positive, three clear | 90.8% | 70.2% | 13.30 m² |
| Ground-cell hysteresis: three positive, three clear | 82.8% | 61.7% | 8.27 m² |

“Clear” in the experimental hysteresis rule means three observed cell scores below 0.20, not geometric ray-certified free space. Confirmation scores and hysteresis flags are not calibrated probabilities. [Full tables](full-map-ablation/summary.csv) preserve every per-rock result. [The four-panel comparison](full-map-comparison.png) was visually inspected: confirmation thins widespread false positives but leaves a large noisy region; ground-cell hysteresis removes more of the smaller and western rock footprints.

### Why confirmation is not ready to gate hazards

[The timing audit](confirmation-latency.csv) reconstructs every cached scoring step and independently reproduces the baseline's final footprint coverage. Its 50% and 80% milestones use each rock's full observed footprint, not an arbitrary two-cell “hit.” They are diagnostic milestones, not stopping-distance specifications.

Requiring three positive observations delays reaching 50% footprint coverage by **202 seconds for rock 9**, **196 seconds for rock 11**, and **136 seconds for rock 3** relative to the baseline. Other delays include 36 seconds for rock 8 and 11 seconds for rock 6. At its 50% milestone, rock 8's label center is only about 0.63 m from the recorded base position in XY; this is not a bumper clearance or a calculated stopping margin.

Therefore the final-map noise reduction cannot justify making this rule the sole immediate obstacle signal. A future system needs to preserve immediate hazards while separately evaluating persistence and visibility-aware clearing. None of these confirmation rules was installed as a live default.

## Further fixes and remaining execution

The continuation corrected a threshold sweep's hardcoded boundary-shell size, clarified that the audit's 25% detection flag is not map completeness, and removed live warning text that advised narrowing the floor crop until a segmenter warning cleared. The warning now identifies the model/band mismatch without recommending removal of required obstacle geometry. These changes do not alter the frozen classifier training snapshot.

The new footprint and audit regression checks pass (**19 tests**); the affected live checks pass (**21 tests**). These are targeted follow-ups to the earlier 259-test verification, not additional independent test counts to sum. The full-map audit records executed source and checkpoint hashes. The paired finalist fits use the original training source snapshot, preserving the comparison.

The queued volleyball-only final phase is described in [finalist-plan.json](finalist-plan.json), with execution state under `training/experiments/lance-finalists-v1/status.json`. It audits completed fits over the full Lance recording and leaves deployment unchanged. Full-recording classifier results, per-rock footprint coverage, false occupied area and detection timing still need to be reviewed before naming a replacement.

## Separate real-negative adaptation experiment

A paired control/adaptation trial is queued after the finalists, subject to the same original deadline. Unlike the volleyball-only experiments, this trial **uses Lance negative samples for training**. It cannot be presented as zero-shot transfer or an independent competition test. Its spatial and temporal holdouts come from the same recording.

Before launching it, I found that the historical Lance cache inherits the labeler's −12.4104…+0.511 m height crop. Matching the generator dictionary alone would miss this difference. I rebuilt the negative samples using `build_inference_samples`, the measured floor, the complete −0.10…+0.60 m operational floor band, an 8 m range, and 50 ms windows sampled every 240 windows. The labeler's height band is explicitly not applied. [The split manifest](lance-negative-split.json) records the contract and cache hash.

Every retained center lies outside all annotated rock footprints expanded by 0.6 m, exceeding the 0.5 m neighborhood radius. There are **5,735 training negatives**, **1,708 validation negatives**, and **24,957 reserved eastern negatives**. Training centers have x < −3.6 m; reserved test centers have x > −2.4 m. Validation is the final block of western observations, with two selected frames excluded before it. These counts replace the preliminary counts from the historical cache.

Both arms start from the existing stray train-all checkpoint, use the same seed, freeze BatchNorm running statistics, and receive the same volleyball and target minibatches. The control assigns zero loss to target negatives; adaptation weights their loss by 0.5. Half the target tensors are thinned to counts drawn from volleyball positives, reducing the opportunity to recognize negatives merely by density. No Lance rock neighborhood contributes a training loss.

Selection keeps the original 0.71 threshold. A new checkpoint is eligible only if volleyball validation AP stays within 0.01 of its initial value and rock recall stays at least 98% of its initial value. Among eligible checkpoints, selection minimizes the western validation false-positive fraction, then mean probability. The initial weights remain an eligible fallback. Eastern negatives are evaluated only after selection; they provide no rock-recall measurement. Full-map rock evaluation follows if time permits.

The CPU smoke check against the rebuilt operational cache completed a forward/backward pass, evaluation and checkpoint round trip. Its trial update reduced recall beyond the allowed limit and was correctly rejected. This checks execution and the guard; it does not establish adaptation quality. Execution state is under `training/experiments/lance-target-negatives-v1/status.json`.
