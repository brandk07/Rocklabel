# Lance campaign results — 5 September 2026

**Completed: 42 training fits and all 42 scheduled candidate audits. No checkpoint earned an unconditional deployment recommendation.** The best volleyball-only changes gave modest noise reduction with some rock-coverage and timing regressions. Training on real Lance negatives gave a much larger reduction in noise, but weakened obstacle coverage enough that I reject it as a replacement at its stored threshold.

The original campaign started at 00:27:49 UTC on 5 September, with a deadline of 12:27:49 UTC (07:27:49 America/Chicago). The initial 36-fit sweep and its audits finished at 09:06 UTC, the four all-volleyball finalists at 10:36 UTC, and the two adaptation arms and their audits at 11:02 UTC. All finished within the original budget. The three supervisors have exited; the GPU is idle. Report comparison and interpretation were completed afterward. No deployment checkpoint or live filtering default was replaced.

## What was trained

The initial matrix compared PointNet and a new classifier that pools masked maximum, mean and standard deviation features, including features around the candidate center. Each architecture was tested with stray augmentation alone and with additional count-matched phantom augmentation: three held-out volleyball recordings × three seeds × four arms = 36 fits. Matched phantoms replace negative samples only, retain their counts, and preserve all positive examples. The training source was frozen for reproducibility.

Mean held-out volleyball AP was 0.7023 for PointNet stray, 0.7204 for PointNet matched, 0.7189 for statistics stray, and 0.7157 for statistics matched. These nine fits per arm span only three held-out recordings, not nine independent environments. The corresponding Lance audits show the statistics model's higher stored-threshold coverage frequently comes with more false detections. It did not provide the intended improvement at a comparable false-positive burden.

The follow-on trained PointNet stray and matched variants on all volleyball recordings with seeds 42 and 43. A separate paired experiment fine-tuned the existing stray train-all model, with and without a loss on real Lance negative samples. The latter is explicitly **target-domain adaptation**, not volleyball-only transfer.

Artifacts: [training table](campaign-training.csv), [all 36 short-map audits](campaign-audits.csv), [campaign scope](campaign-summary.json), [initial code review](review.md), and [filtering, persistence and adaptation details](continued-review.md).

## Full-recording comparison

Every finalist was evaluated on identical **2,599 sampled 50 ms scoring windows across the 35.5-minute Lance recording**, with an 8 m range, the measured floor and the complete −0.10…+0.60 m floor band. Inference uses native classifier neighborhoods followed by latest-value-per-native-voxel accumulation and a 10 cm ground-cell reduction. Inputs are unfiltered. This is not a reproduction of every GUI filter, scheduling decision or navigation component.

The comparison independently verified identical frame indices, times, base poses and unfiltered observed rock geometry in all seven caches. All 12 physical rocks receive equal weight. False occupied area excludes annotated rock footprints and their boundary shell, and does not include robot-footprint inflation. Other scored cells are not certified free space.

| Checkpoint | Stored threshold | False footprint area | Mean final footprint coverage | Worst rock | Mean 3D-attributed coverage |
|---|---:|---:|---:|---:|---:|
| Existing stray train-all reference | 0.710 | 21.44 m² | 99.5% | 95.7% | 65.9% |
| New stray, seed 42 | 0.690 | 21.67 m² | 99.3% | 93.6% | 65.5% |
| New stray, seed 43 | 0.727 | 21.43 m² | 99.3% | 93.6% | 68.7% |
| Matched phantom, seed 42 | 0.773 | 20.41 m² | 98.3% | 92.9% | 70.2% |
| Matched phantom, seed 43 | 0.732 | 20.25 m² | 98.5% | 92.5% | 71.4% |
| Adaptation control | 0.710 | 22.41 m² | 99.1% | 95.0% | 67.8% |
| Lance-negative adaptation | 0.710 | 12.67 m² | 89.3% | 57.1% | 65.2% |

**Footprint occupancy and 3D rock attribution are different measurements.** A floating positive above a labelled rock can block its ground cell and receive footprint credit, without identifying its surface. Removing that representative can improve 3D attribution while leaving the ground cell blocked. Neither metric alone proves correct obstacle recognition or safe navigation.

The reference here is `training/experiments/deploy/cls-stray/trainall/best.pt`. The user's previously tested phantom checkpoint was separately compared in the initial excerpt audits; it did not receive this full-recording comparison. These results must not be described as a full-recording win over every historical model.

All six final comparison JSON summaries, all six 12-rock CSVs, and all twelve generated full-map/per-rock figures were inspected. [The inspection ledger](visual-inspection.json) distinguishes these from other generated image sets that have not received a complete manual review.

## Interpretation of the finalists

The matched seed-43 finalist reduces false footprint area by **5.6%** relative to the reference and raises mean 3D-attributed coverage from 65.9% to 71.4%. Its map still has extensive false positives. Rock 10's 3D coverage falls from 7/11 to 5/11 cells; rock 8's final footprint coverage falls from 40/40 to 37/40 cells. Rock 13's 3D coverage improves substantially, from 16/42 to 32/42 cells, but it reaches the 50% footprint milestone about 100 seconds later. Rock 12 reaches that milestone about 50 seconds later; rock 3 reaches 80% about 191 seconds later.

These timing milestones use the full observed footprint accumulated over the recording. They are retrospective diagnostics, not first-recognition timestamps, physical clearances or stopping-distance requirements. They can also receive credit from floating positives over the rock. Nevertheless, the regressions prevent a claim that the modestly cleaner map is an across-the-board improvement.

The seed-42 matched finalist has similar final-map tradeoffs and a roughly 313-second delay in reaching 80% footprint coverage on rock 8. The two stray refits do not materially reduce final noise. The more promising held-out VB4 matched checkpoint improved the broader five-second-map audit, but still has sparse-view failures and has not passed a full-recording timing comparison. The detailed review records all twelve selected cases for that checkpoint.

For manual replay research, the matched seed-43 checkpoint is retained at:

`training/experiments/lance-finalists-v1/pointnet-matched/seed-43/pointnet_trainall_dx-dy-dz/best.pt`

It is a candidate for inspection, **not a recommended navigation replacement**. Its [map](completed-map-comparison/pointnet-matched-43/map.png), [per-rock figure](completed-map-comparison/pointnet-matched-43/per-rock.png), and [coverage/timing table](completed-map-comparison/pointnet-matched-43/per-rock.csv) show both gains and regressions.

## Why the adaptation result is rejected

Adaptation used 5,735 western Lance negative samples, with 1,708 western validation negatives and 24,957 reserved eastern negatives. Every training neighborhood center was excluded from all rock footprints expanded by 0.6 m, exceeding the 0.5 m input radius. The cache was rebuilt through the operational inference builder after discovering that the historical Lance cache used a different height crop. No Lance rock neighborhood contributed a training loss. The holdouts are still within the same recording.

At the fixed 0.71 threshold, the selected adapted checkpoint reduced the reserved eastern negative false-positive fraction from **15.54% to 2.39%**. The control increased it to 16.06%. Thus the negative-sample gain is real in this reserved region, rather than merely a validation-set improvement. It is not a measure of rock recall.

The selected adaptation epoch passed the predeclared volleyball validation guard: AP 0.8438 versus 0.8500 initially, and recall 0.7137 versus 0.7277. That guard **did not preserve Lance rock behavior**. Final false area falls 40.9%, but rock 13 ends with only 24/42 footprint cells occupied. Rocks 8 and 11 fall to 85%. Reaching 80% footprint coverage is delayed approximately 648 seconds for rock 9 and 606 seconds for rock 13; coverage can later fall again because the map replaces old voxel scores.

The [adapted map](completed-map-comparison/target-adapt/map.png) is visibly cleaner, especially toward the western arena, while retaining a noisy eastern region. Its [per-rock figure](completed-map-comparison/target-adapt/per-rock.png) makes the coverage loss clear. I reject this checkpoint as a replacement at the stored threshold.

Separate threshold sweeps are retained for every final map. They are **oracle diagnostics fitted to these evaluation data**, not deployment calibrations. Even at an adapted threshold of 0.21 chosen to stay within the reference's false-area budget, worst-rock final footprint coverage reaches only 81.0%, versus the reference's 95.7%. Lowering the threshold therefore does not establish a recovered operating point in this sweep.

## Code fixes and verification

The review fixed phantom height normalization, added negative-only matched phantom augmentation, centralized checkpoint reconstruction to preserve BEV geometry and density settings, stored validation thresholds in intermediate best checkpoints, made checkpoint publication atomic, restored RNG state on resume, and replaced the coarse F1 threshold grid with an exact tied-score sweep. Export now preserves segmentation input/output metadata. Small-neighborhood grouping and dashboard socket error handling were corrected. Audit diagnostics now preserve boundary-shell settings and explicitly separate XY footprint attribution from 3D attribution. Live warning text no longer recommends narrowing the required floor band.

The affected suite passed **259 tests**. Later targeted checks passed 19 training/audit tests and 21 live tests; these overlap and should not be summed. The new operational-cache adaptation smoke check exercised a real CPU update and correctly rejected excessive recall loss. The parameterized timing tool exactly reproduced the original baseline CSV. All six final comparisons independently reproduced their stored map metrics and native final coverage. Python compilation and `git diff --check` pass. TorchScript round-trip validation passed with tracing warnings; ONNX runtime was not tested.

## What the evidence supports next

The persistence audit establishes that most final false occupied cells are old predictions. Simple temporal confirmation reduces noise but introduces substantial coverage delays; aggressive proximity filters delete real rock returns. Those particular cleanup rules should not become the sole immediate obstacle signal.

The next experiment should keep an immediate hazard output while separately testing map persistence with actual visibility/free-space evidence. Model selection should constrain per-rock coverage and detection timing on target-domain positive examples as well as false occupied area; a volleyball-only recall guard was inadequate here. Any use of Lance positives for development must be declared, with a separate recording or untouched physical-rock evaluation reserved for the final claim. These are follow-up directions supported by this campaign, not additional training started beyond its budget.
