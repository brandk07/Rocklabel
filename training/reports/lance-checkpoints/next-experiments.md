**Recommendation:** spend tonight on a small, controlled extension of the best combined-clutter PointNet recipe, plus an independent replay experiment that can retract unsupported map predictions. The most interesting new model input is the candidate's vertical position within its neighborhood. The strongest map change is clearing based on observed free space, with local surface support as supporting evidence.

This is an investigation and implementation handoff, not a training launch or a deployment recommendation. Existing checkpoints, training code and live defaults were left unchanged. Findings below come from the September 12 checkpoint results, checkpoint/config/history inspection, current inference/training code, and the earlier Lance campaign reports. Historical map measurements are attributed to those reports; I did not rerun or independently visually certify their audits in this investigation.

**What the rescoring actually says**

The highest classifier candidate AP is `stray/cls-both/loro_VolleyBallTest4.reslam/best.pt`: **0.7112**, with precision **0.5946**, recall **0.7180**, and stored threshold **0.75**. This is the combined stray-and-phantom recipe. `deploy/cls-stray/trainall/best.pt` is a different model and scores **0.5399**.

I recomputed these arm summaries directly from `results.json`. All rows below use the same labelled Lance classifier candidate population; means describe checkpoints evaluated on one arena, not independent arena trials.

| Recipe | Checkpoints | Mean Lance AP | Best Lance AP |
|---|---:|---:|---:|
| Plain PointNet (`stray/cls-base`) | 11 | 0.6169 | 0.6586 |
| Stray jitter alone (`stray/cls-stray`) | 11 | 0.5394 | 0.5960 |
| Phantom negatives alone (`stray/cls-phantom`) | 11 | 0.6208 | 0.6608 |
| Both (`stray/cls-both`) | 11 | 0.6416 | 0.7112 |
| PointNet, matched phantoms plus stray, previous campaign | 9 | 0.6460 | 0.6911 |
| PointNet++, geometry, original seed | 11 | 0.6418 | 0.6846 |
| PointNet++, reflectivity | 11 | 0.6458 | 0.7074 |

Adding phantom negatives to stray improves AP on **11/11 matched folds**, averaging **+0.1022**. Stray alone loses to plain PointNet on **11/11**, averaging **−0.0775**. Both beats phantom alone on **8/11**, averaging **+0.0208**. This supports the combination, but does not establish 5% as the optimal jitter dose. The PointNet++ results also justify revisiting that architecture for Lance; older volleyball-based dismissals are insufficient. The 0.0038 gap between the top two classifiers is too small to interpret as a decisive architecture result.

Some old reports say stray gained +0.093 on Lance on every fold. Their report used 60 frames; the current benchmark uses 109 classifier frames. That historical measurement is not the current measurement, and its conclusion should not be carried forward without reproducing both evaluation contracts. I could establish the conflict, not isolate its exact cause from the surviving text reports.

**What made the winning training run different**

The winner is ordinary PointNet, geometry only (`dx, dy, dz`), no T-Net, default dropout 0.3. It uses a 50 ms input window, 0.5 m radius neighborhoods, 256 sampled points, 5 cm candidate voxels and at least 20 neighbors. Training used ten volleyball recordings, excluding VB4, with seed 42. Settings: AdamW, learning rate 0.001, weight decay 0.0001, batch 256, 30 epochs, cosine decay, patience 10, 15% tail-block validation with frame/time gaps. The stored best checkpoint is epoch **25**, zero indexed; all 30 epochs ran. There is no evidence here that substantially longer training is the missing ingredient.

The two relevant augmentations have different jobs:

- Stray jitter moves about 5% of valid points in ordinary neighborhoods; classifier labels stay unchanged. It encourages recognition despite corrupted surrounding returns. The direction is based on canonicalized coordinates, not the actual sensor ray, so this is useful corruption augmentation rather than a faithful range-error simulator.
- Phantom augmentation replaces 8% of samples with negative synthetic clouds containing 20–59 valid points and roughly 0.27–0.81 m vertical span. It teaches rejection of an entire loose cloud. Historically, it replaces positive samples too: 8% of positive training exposures are lost, and the original positive loss weight is retained.

My interpretation is that jitter increases tolerance to clutter, while whole-cloud negatives supply a competing rejection signal. The paired results support that interpretation; they do not prove which internal feature the network learned. Sparse synthetic negatives could also create a density shortcut. PointNet uses counts to mask padding and max-pools point features; it does not explicitly receive the full neighborhood count as a scalar density feature.

The winner's held-out **VB4 AP was only 0.5237**, despite volleyball validation AP 0.8705. This explains why selecting by the old test score could overlook it. The all-volleyball `cls-both` checkpoint scores 0.6382 on Lance. Do not conclude that VB4 is harmful: adding that recording changes data composition, optimization and checkpoint selection together. First repeat the winning split across seeds. Only then compare a matched all-recording refit.

There is a reproduction wrinkle: the historical phantom code subtracted minimum height before selecting its sparse surviving points. Current code subtracts after selection. The old code is visible in commit `3e62e5c`; the campaign report documents the correction. The checkpoint does not embed a source hash, so configuration alone cannot certify its exact historical executable. Keep the original checkpoint as the reference, freeze/hash current source, and describe new fits as current-code replications. Do not silently restore the old bug to production.

**The four training arms I would queue first: 12 fits**

Use seeds **42, 43, 44**, the winner's exact ten-recording split, 30 epochs and all other settings above. Each arm changes one intended factor relative to A. Save outputs in a new experiment root. New augmentation randomness should use a separate reproducible generator where necessary, so adding a feature does not unnecessarily change all other random draws.

| Arm | Change from A | Question |
|---|---|---|
| A: combined reference | Current-code `cls-both`: stray 0.05, legacy phantom 0.08 | Is the winning recipe repeatable? |
| B: less jitter | Stray fraction 0.025 | Can we retain clutter tolerance with less false-positive pressure? |
| C: preserve positives | Sparse phantom negatives replace clear samples only | Can we retain rejection while avoiding loss of positive exposures? |
| D: candidate height | Add candidate height above the same neighborhood minimum to A | Is missing vertical query information causing avoidable ambiguity? |

For C, retain A's sparse-clump geometry/count distribution and height correction. If the training positive fraction is `p`, use replacement probability `0.08 / (1-p)` **among clear samples** to match A's expected total synthetic-negative dose. Keep the loss weight fixed for this experiment and record the resulting class exposure. This isolates a different question from the existing `matched` mode, which also changes point counts, shape and, in previous runs, dose and training schedule. Reuse previous matched checkpoints as additional references; do not rerun the 36-fit campaign wholesale.

For D, the current builders compute `dx = x - candidate_x`, `dy = y - candidate_y`, but `dz = z - neighborhood_min_z`. The model is asked to label the candidate, yet the candidate's own vertical coordinate is absent. Two vertically separated candidates with the same neighbor set can therefore have the same input geometry. Whether that ambiguity accounts for many actual errors remains unmeasured, but the representation issue is visible directly in both training and inference builders.

Add scalar `candidate_z - neighborhood_min_z`, concatenated to the pooled feature before the classifier head. Keep existing point coordinates and their reference unchanged for this first test. Generate/store the scalar before subsampling; update the dataset, cache, training, native inference, checkpoint versioning and export consistently. The old cache lacks the absolute neighborhood minimum, so the scalar cannot generally be recovered exactly from its stored tensors: rebuild from recordings. For synthetic clumps, give the synthetic query a plausible height derived from a retained synthetic return, without using its label to select a special constant. Do not let the new scalar trivially identify all synthetic negatives. Preserve headings/thinning semantics and explicitly handle the synthetic metadata.

D should include targeted checks that vertically moving the query changes this scalar, translating the whole scene vertically does not, and training/live preprocessing agree. Compare on identical candidate centers, labels and sampled neighborhoods between feature variants. Audit negatives above/beside rocks separately from isolated fog, because this feature should help most with the former. It is a research hypothesis, not an established gain.

A–C can train while D's cache/input work is being completed by the implementation agents. If D is not ready in time, spend remaining compute on its implementation and evaluation preparation rather than launching unrelated sweeps. Save periodic epoch checkpoints for later analysis; distinguish volleyball-selected `best.pt` from any checkpoint subsequently selected using Lance.

**Optional second batch: six PointNet++ fits**

If time remains, use the same split, schedule and three seeds to compare geometry-only PointNet++ without clutter augmentation against PointNet++ with A's combined augmentation. This directly tests whether the two promising directions compose. Use identical source and model settings within each pair; benchmark inference latency because it affects actual scoring frequency and obstacle acquisition. The high reflectivity checkpoint deserves replay as an existing reference, but the small average reflectivity advantage does not make a new reflectivity sweep my first choice.

Do not prioritize another statistics-pooling campaign, more stray-only all-recording fits, or negative-only Lance fine-tuning. Previous all-recording matched fits reduced reported false footprint area only about 5–6%. Previous Lance-negative adaptation cut it about 41%, but worst-rock footprint coverage fell from 95.7% to 57.1%; lowering its threshold did not recover a comparable operating point in that report. If target adaptation is revisited, it needs target positive preservation and a fresh or honestly separated physical-rock evaluation, not just a volleyball recall guard.

**Map cleanup is a separate, high-priority implementation ticket**

The current flow is recent scans → optional per-batch floating filter → native model inference → latest score per 3D voxel. The model does not reclassify the accumulated cloud. `LiveScorer._map` stores only position and probability. A positive floating voxel can remain indefinitely if that exact voxel is never scored again; a lower ground return does not replace it.

The existing floating filter already uses a low height percentile, minimum data counts and a 0.5 m height gate, but only within each incoming batch. It is enabled by default, although the actual replay setting can differ. The old ablation reported essentially no benefit from its tested 0.5 m setting. Deleting accumulated points alone also does not prevent future incoming phantoms from being scored again. Cleanup must update persistent predictions as well as the relevant displayed/obstacle geometry, and must have a defined reinsertion policy.

Implement an optional evidence map beside the scorer, initially evaluated in replay:

1. Retain timestamped observation support per voxel/component, with counts from distinct scan windows and preferably different viewpoints. Repeated rendering or rescoring the same buffered returns must not add evidence. Keep model confidence separate from geometric occupancy evidence.
2. Estimate local supporting surfaces from spatially distributed, repeatedly observed returns. Track uncertainty and coverage, allow slope, and exclude the suspect component from its own reference fit. A single minimum or a large raw point count is inadequate. Dense accumulated fog can support itself; retain time/height structure instead of fitting indiscriminately to the fused cloud.
3. Mark a component as suspect when it is detached above that supported surface and lacks stable nearby surface support. This is evidence for investigation, not a deletion rule by itself: rock tops are also above their neighbors, and real rock sides/bases can be occluded.
4. Accumulate contradictory observations when valid later measurement rays pass through the suspect volume and terminate beyond it. Respect sensor origin, pose uncertainty, endpoint margins and occlusion. A ray hitting ground underneath the point, a ray ending before it, or no return is not automatically evidence that its volume is free. Use sensor/LiDAR origins, not assumed robot-base origins; preserve per-observation poses rather than raycasting a fused cloud from the latest pose.
5. Retract persistent occupancy after repeated contradictory evidence; keep insufficiently observed space unknown. A new hit can restore provisional occupancy. Preserve an immediate fresh-obstacle output while this persistent layer is tested, and measure its own false alarms as well as map cleanliness.

Ray-based probabilistic updates separating occupied, free and unknown space are established in [OctoMap](https://www.arminhornung.de/Research/pub/hornung13auro.pdf). The proposed combination with local support and this scorer is an untested project-specific design. A sparse voxel sidecar is sufficient for a prototype; adopting an entirely new mapping library is not a prerequisite.

Prototype candidate settings can bracket two versus three independent contradictory windows, 10 versus 15 cm spatial support, and minimum height separation starting conservatively around 15–25 cm plus uncertainty. These are diagnostic sweep values, not validated defaults or rock-size cutoffs. Do not reinstate the earlier hard radius filter or blanket three-hit admission gate: prior reports show loss of real rock returns and substantial coverage delays from those particular rules. The old persistence audit found 2,077 of 2,162 false 3D-attributed ground cells had not refreshed for over 60 seconds. That motivates measuring contradictory visibility; age alone cannot justify clearing them.

**Evaluation ticket and acceptance criteria**

Before spending the entire night training, replay the existing VB4 `cls-both` winner, the deployed stray reference, the existing matched seed-44 VB6 checkpoint, and the strongest PointNet++ checkpoints through the operational pipeline. This gives the training agents useful baselines immediately. No classifier/segmenter deployment parity claim is made here.

Retain the new sampled benchmark as a diagnostic, but its manifest contains a labelled world-height band of **−12.4104…+0.511 m**, which the dataset generator substitutes for the generator's vertical crop. This is not the operational floor-relative −0.10…+0.60 m band. The benchmark's generator-dictionary compatibility flag does not capture that difference. Build a separate operational benchmark through the actual inference preprocessing and document both contracts; do not overwrite the historical cache or interpret its compatibility flag as full live equivalence.

For every finalist, evaluate chronological map accumulation across Lance, full operational band, 8 m range, native inference then latest-value 5 cm voxel storage and shared 10 cm ground-cell reduction. For the cleanup experiment retain the old map as a side-by-side control and preserve identical unfiltered observed rock geometry as the denominator. Include raw-ray processing at its actual cadence, model cadence/caps and replay filter settings; sparsely sampled scoring windows alone cannot establish real-time clearing latency.

Give each physical rock equal weight. Report per-rock final and interval coverage, persistent misses, fragmentation, first acquisition/coverage timing, erroneous removal and reacquisition, false area outside rock footprints, and false-area duration. Report both XY footprint occupancy and 3D rock attribution: a floating positive over a rock can otherwise get undeserved footprint credit. Inspect `summary.md`, `per-rock.csv` and every automatically selected case image before interpreting aggregates.

Show stored-threshold behavior separately from an equal-false-area threshold sweep. Calibrate any proposed threshold on a declared development split; thresholds optimized on the evaluation map are oracle diagnostics. Do not choose the experiment on AP alone: prioritize lower persistent false area without new persistent rock misses or materially worse weakest-rock coverage and acquisition timing. Set operational timing tolerances with the navigation requirements rather than inventing them from these offline plots.

Lance has now informed repeated model selection and augmentation design, so it is development data. Use the available recording honestly for improvement tonight; reserve a new competition-like recording for final confirmation. If only this recording is available, group by physical rock/region with neighborhood exclusion buffers and keep all sightings together. Random frame splits do not create independent rocks or a fresh arena.

Relevant implementation entry points: `rocklabel/train/engine.py`, `rocklabel/train/models.py`, `rocklabel/train/data.py`, `rocklabel/dataset/neighborhoods.py`, `rocklabel/dataset/generate.py`, `rocklabel/live/scoring.py`, `rocklabel/live/pipeline.py`, and `rocklabel/live/filters.py`. Historical evidence: `training/reports/lance-review-2026-09-04/final-results.md` and `continued-review.md`. The source analysis and numerical comparisons above support prioritizing these experiments; they do not substitute for running them.
