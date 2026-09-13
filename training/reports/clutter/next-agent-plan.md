**Assignment: determine whether epoch selection can improve on the existing 0.7112 Lance checkpoint, and repair the map evaluation/clearing issues before making deployment claims.** The last campaign improved over the deployed stray-only checkpoint, but did not beat the previous best. Do not substitute the weaker deployed baseline for the best available model.

This is a plan for the implementing agent. No new fits or runtime changes were made while preparing it. Read `judgment-review.md` beside this file for the reproduced problems. Preserve existing user work: the working tree contains substantial modifications and new files from the previous campaign. Reconcile current code before applying each item; a failure already fixed should be verified, not reimplemented.

**1. Establish the correct incumbents and record the comparison contracts**

Use these three exact checkpoints, identified by full path and content hash:

| Role | Checkpoint | Historical Lance candidate average precision |
|---|---|---:|
| Primary incumbent: best known sampled ranking | `training/experiments/stray/cls-both/loro_VolleyBallTest4.reslam/best.pt` | **0.7112027028739645** |
| New campaign reference | `training/experiments/clutter/cls-both-s44/trainall/best.pt` | 0.7043799953382626 |
| Secondary operational reference: currently deployed | `training/experiments/deploy/cls-stray/trainall/best.pt` | 0.539916144568302 |

The original incumbent still exists. Its stored threshold is 0.75; the new model's is 0.722051203250885. On the historical candidate cache their precision/recall are approximately 0.595/0.718 and 0.434/0.808 respectively. Those different operating points cannot be compared as though they used the same false-positive budget.

The new campaign's best score misses the original by 0.006823. Do not dismiss the original checkpoint as unusable because current-code refits did not reproduce it. It is a real available model, and the source changed between experiments. Repeatability of a recipe and utility of an existing checkpoint are separate questions.

Create a campaign manifest under a new output root, for example `training/reports/epoch-selection-v1/`, with checkpoint hashes, code snapshot hash, dataset/cache identities, seed list, configuration and evaluation policy. Keep the historical 0.7112 comparison intact. Establish two named evaluations:

- **Historical Lance candidates:** the same 89,612-candidate cache used for 0.7112. Re-score the three references through the present loader and verify the stored numbers within a small declared numerical tolerance. Record device and inference batch. If a discrepancy exceeds tolerance, diagnose preprocessing/cache/loader changes before proceeding; do not silently move the target.
- **Operational Lance evaluation:** the actual shared live preparation, measured floor, full −0.10…+0.60 m floor band, 8 m range, checkpoint-compatible 50 ms inputs, real candidate construction, recorded scoring schedule and explicit filter/cap settings. Use this for map behavior. Scores on this different candidate population are not directly comparable to historical 0.7112.

Build the missing full operational map comparison for the **original incumbent**. Existing map results for the two newer references do not establish that either beats it. Report all three at stored thresholds and at matched false-footprint-area budgets, with 3D-attribution counts as a separate diagnostic. Include PointNet++ only as a secondary existing candidate if resources allow; no new architecture training is required in this assignment.

Before the first expensive operation, time one representative benchmark and one short replay, estimate the remaining cost, and record a bounded execution schedule. Protect time for the incumbent audit and final comparisons. Do not spend the entire budget producing weights without evaluating them.

**2. Repair evaluation before using it to select another winner**

Main files: `rocklabel/train/map_eval.py`, `rocklabel/train/cli.py`, `rocklabel/train/lance_benchmark.py`, the shared inference preparation, and dashboard report readers.

**2a. Correct cache identity and invalidation.** Current `mapeval` reuses geometry merely when `settings.json` exists, and score caches are keyed only by checkpoint hash. Reusing the same directory with different recording/labels/stride can silently reuse the old data; rebuilding geometry can also leave another model's old scores behind.

Use immutable, versioned identities. Geometry identity must include recording and label content identities, level/extrinsics, preparation schema/source identity, time bounds, evidence-stream selection, scoring schedule, window construction, floor/range/region settings and relevant filters. Score identity must additionally include geometry identity, checkpoint bytes, input builder version, generator contract, sampling seed and caps. Evaluation identity includes metric version, labels, thresholds and evidence policy. Hash large source files once and record their identity; avoid rereading them on every frame.

Build into a fresh temporary directory and publish only after a completion manifest verifies frame count, acquisition identities and required arrays. An interrupted cache must not look complete. Reject a conflicting cache or create a new keyed directory automatically. A rebuild must not leave stale frame files in the published artifact. Demonstrate invalidation by changing one geometry parameter while holding checkpoint bytes fixed.

**2b. Give coverage two explicit definitions.** For each physical rock and each chronological observation, retain an unfiltered observed-footprint denominator shared by all variants. Report:

- Occupied footprint coverage: observed rock XY cells occupied by any eligible detection, regardless of the representative's height.
- 3D-attributed coverage: the current representative lies on that physical rock's annotated 3D geometry.
- Optionally, any surviving on-rock native voxel per ground cell, explicitly labelled as a third attribution diagnostic. Do not silently substitute it for the existing representative rule.

Report false area outside annotated XY footprints separately from false 3D representatives. Explain boundary-shell treatment and do not label uncertain shell points green as certified rock. The previous 65.0% → 68.3% result was attribution, while occupied observed rock cells changed 372 → 371. The corrected report must make such cases obvious.

**2c. Use the complete threshold range.** For uncleaned final maps, sweep all distinct stored scores with ties handled as a group, including the all-positive and all-negative endpoints. Use a sorted cumulative computation rather than a quadratic loop if the map grows. Retain thresholds near 1.0; rounding to two or three decimals before application can change the result substantially. Report exact applied thresholds and actual achieved budgets; do not interpolate imaginary operating points across ties.

For cleaned maps, separate a frozen-final-map score sweep from a full chronological policy replay. Cleanup currently updates only above-threshold voxels, so changing the threshold changes map history. Use the complete uncleaned frontier to select a small explicit set of thresholds, then rerun the cleaned policy from cached geometry/scores at each selected threshold. Do not run thousands of full replays merely to emulate an exact static sweep.

Show common false-footprint-cell budgets of 2,200, 1,700, 1,000 and 500 as diagnostics; include 200 as a stress case, not an acceptable operating requirement. Recompute the earlier 3D-budget comparisons for continuity, without calling them footprint budgets. Every threshold chosen from this recording is a development/oracle choice, not blind deployment calibration.

**2d. Measure timing that reflects coverage.** First one-cell contact is insufficient. For each rock, record first raw visibility, first occupied contact, times to 25/50/80% coverage against a fixed final observed-footprint denominator, coverage over time, intervals falling back below those levels, final fragmentation and reacquisition after retraction. Also show coverage against geometry observed so far where useful, naming that changing denominator explicitly. Neither metric is a physical stopping-distance requirement.

Measure false-area duration chronologically: state after event i occupies the interval from event i to i+1. The current implementation charges the previous interval to the new map state. Restrict outside-footprint area-duration to the same arena domain used for final false area; `_false_seconds` currently lacks that arena restriction. Verify with a synthetic sequence containing a short-lived false cell and a cell outside the arena.

Write per-rock, per-time measurements in machine-readable files. Inspect rocks 10 and 12 in particular: the new model's better global minimum masks rock 10 falling from 7/11 to 4/11 attributed cells. Generate automatic cases for largest per-rock losses, longest delayed coverage, erroneous deletion and worst persistent false patches, and inspect every selected case before naming a winner.

**Acceptance for step 2:** reference scores reproduce; stale/partial caches cannot be reused; a deletion cannot increase occupied-footprint coverage with frozen predictions; distinct attribution improvements remain measurable; exact sweep extends beyond 0.95; time integration and arena masks are correct; reports expose per-rock regressions.

**3. Correct the evidence layer with regression tests before tuning its settings**

Main files: `rocklabel/live/evidence.py`, `pipeline.py`, `scoring.py`, shared replay preparation, and `tests/test_live_evidence.py`. Retain the off-by-default behavior while repairing it. The three counterexamples in `judgment-review.md` are mandatory regression cases, not optional robustness work.

**3a. Make beam evidence match the stated geometry.** The angular nearest-neighbor search can efficiently shortlist rays. It must not by itself certify a crossing. Check the ray segment against the actual world-aligned voxel bounds and use its exit distance when checking that the endpoint lies beyond the voxel plus the endpoint margin. Handle zero direction components, behind-origin queries, tangent/boundary cases and hits in/near the voxel. Keep current-window hit protection independent of this calculation.

The default counterexample is query `(1.025, 0.025, 0.425)`, origin `(0, 0.025, 0.425)`, endpoint `(2.05, 0.145, 0.425)`, voxel size 0.05 m: the ray misses the voxel and must not add contradiction. Also test a clear central crossing, a ray ending before the query, one ending inside it, and two genuinely different sensor origins.

Do not keep the current uncertainty behavior, where increasing pose error widens the acceptance cone and invents stronger clearing evidence. Choose and document a conservative uncertainty policy; at minimum, uncertain/grazing crossings must abstain or receive less weight, and increasing assumed uncertainty must not increase contradiction for a fixed measurement. If a bounded-error formulation erodes the trustworthy crossing region to nothing at 3 cm uncertainty in a 5 cm voxel, report the resulting abstention. Do not hide it by reducing assumed uncertainty until the old 6% result reappears. Exact intersection is necessary to fix the current bug, but one thin ray crossing part of a voxel is still an occupancy observation, not proof its whole volume contains no object.

**3b. Count sensor observations, not scorer calls.** Carry acquisition identities and timestamps alongside points and sensor origins. Derive evidence-window identities from acquisition events, and retain member scan identities. Reusing the same snapshot must not add hits, surface support or contradictions. Two overlapping scorer snapshots must not vote twice with shared returns, and calling `retract()` twice after one observation must be idempotent for evidence.

Maintain the meaning of `free_windows` explicitly: one vote per independently acquired, non-overlapping evidence window, not one per point, raw sub-scan or GUI refresh. Repeated viewpoints with genuinely new measurements may count as new windows, but do not present them as statistically independent viewpoints. Test pause, stalled source, overlap, repeated timestamps with distinct sequence IDs, reset and resumed playback. Keep evidence in chronological acquisition order; do not let an old asynchronously finished score resurrect a detection as though it were a new sensor hit.

**3c. A current hit vetoes this cycle's deletion.** The final drop mask must protect voxels hit in the current window, even if old evidence is strongly negative. Parameterize tests over every supported `free_windows` setting, especially 2 and 3. Start at evidence −4, add a current hit, verify occupancy remains and the map can reinsert it. Separately verify that future independent contradictory observations can remove it again.

**3d. Verify the actual ray stream used offline and live.** Keep per-scan LiDAR origins through all transforms. Retain full valid ray endpoints separately from the cropped model cloud; a return beyond the model crop can establish contradiction inside it. Define one shared observation-preparation contract for live and replay, including floating-filter behavior. A raw endpoint should not lose its current-hit veto simply because a model-input filter rejected it. If filtered returns are used differently as hit evidence and as free rays, document and test that choice.

First support a replay that matches the live scorer's selected observation cadence and window membership. If processing all acquired evidence windows between scoring passes is added later, identify it as a separate policy: it changes both evidence strength and runtime and must not be conflated with a bug fix. Report actual timestamps and measured intervals rather than assuming `stride=10` is exactly 0.5 seconds.

Replay a short recorded sequence through both adapters and compare acquisition IDs, transformed origins, model inputs, hit sets, contradiction updates and retractions. Before reporting a live deletion guarantee or latency, exercise pauses and delayed scoring, not only offline arrays.

**Acceptance for step 3:** all three regressions pass, increased pose uncertainty cannot increase contradiction, unobserved space remains untouched, repeated processing adds no evidence, fresh hits survive allowed settings, and matching live/offline observations produce matching decisions. Then rerun full-recording cleanup comparisons for all three incumbents at their stored thresholds. Report any loss of the earlier cleanup benefit honestly. Keep immediate fresh detections available separately from persistent-map decisions and verify which viewer/obstacle outputs actually consume each layer.

**4. Run one focused training experiment: where should a run's checkpoint be selected?**

This is the main training work. Do not launch another broad clutter sweep, increase network size, change loss weights, or hold out a volleyball recording as a new default. Those changes would make the interpretation ambiguous.

Use a new root such as `training/experiments/epoch-selection-v1/`. Train ordinary geometry-only PointNet with all eleven volleyball recordings, the existing per-recording 15% tail validation blocks and gaps, and seeds **42, 43, 44**. Copy the combined reference config, changing only seed, output root, snapshot retention and stopping behavior:

| Parameter | Setting |
|---|---|
| Model/features | `pointnet`; `dx, dy, dz`; no T-Net; existing default dropout |
| Input cache | Verified current `training/caches/full-sweep` |
| Input geometry | 50 ms, 0.5 m neighborhoods, 256 points, 5 cm candidate grid |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Schedule | 30 epochs, existing cosine schedule with horizon 30 |
| Batch | 256 |
| Augmentation | Stray fraction 0.05, reach 1.0; legacy phantom fraction 0.08, extent 0.54; thin minimum 0.5 |
| Training changes | Save all 30 epochs; complete all 30 despite validation patience |
| Original stopping rule retained for comparison | Patience 10, same improvement comparison and validation metric as today |

That is **three fits and 90 epoch snapshots**, not 90 fits. Copy the actual config rather than reconstructing unstated defaults from this table. The old 0.7112 model remains an evaluation reference even though it used a different training split. We are testing a new full-data selection process, not claiming an exact retraining of that historical model.

**4a. Add reliable snapshot retention.** Main files: training defaults, `engine.py`, `cli.py`, ablation passthrough/config handling, dashboard spec/inventory. Proposed interfaces are `--save-every 1` and `--no-early-stop`; these are proposed additions, not commands that already work. Use equivalent existing controls if added by another agent. If a dedicated no-stop flag is unnecessary, patience greater than the run length can produce the experiment, but record original patience 10 separately and expose the intended behavior clearly in the dashboard.

Publish inference-ready `epochs/epoch-000.pt` through `epoch-029.pt` atomically with model state, complete training/generator contract, zero-indexed epoch, that epoch's volleyball-picked threshold, validation metrics, input/cache identity and source identity. Reuse validation predictions already computed in the loop to calculate thresholds. Preserve the full resume state in `last.pt`; do not multiply optimizer-state files unnecessarily. Snapshotting must not consume training random draws or modify model/BatchNorm state. Keep ordinary `best.pt` semantics as volleyball-selected, and identify any Lance-selected checkpoint separately.

On resume, preserve completed snapshot hashes and generate each remaining epoch exactly once. Verify uninterrupted versus interrupted/resumed training on a meaningful small deterministic fixture: histories, selected epochs and inference outputs must match within the declared determinism tolerance. A snapshot must load through native scoring and export paths. Old checkpoints must still load unchanged.

Freeze the training executable before launching these fits, including uncommitted changes and the effective config, so CPU-side evaluation work cannot change the running training module or augmentation implementation. Record GPU/device/library versions and peak memory. Do not let the evaluator alter the training process's random state. Launch the first fit after the snapshot tests pass; it need not wait for unrelated clearing implementation to finish.

**4b. Reconstruct the original early-stop baseline correctly.** Completing 30 epochs means the new run may contain weights the previous policy would never have reached. Therefore record these separately for each seed:

- `original_stop_epoch`: first epoch where the original patience-10 rule would have stopped.
- `original_selected_epoch`: best volleyball validation checkpoint available at that original stopping point.
- `val_best_30_epoch`: best volleyball validation epoch across the full completed schedule.
- `lance_best_before_stop_epoch`: highest historical Lance candidate average precision among epochs the original run would actually have reached.
- `lance_best_30_epoch`: highest historical Lance candidate average precision across all 30 epochs.
- `operational_best_30_epoch`: highest candidate average precision on the separate fixed operational cache.

This separates a change in **selection signal** from a gain requiring **additional training beyond the previous stopping point**. Comparing only against the best volleyball epoch across all 30 would misstate the baseline. Handle tied validation scores exactly as the original loop does, and record the no-stop case when the rule never triggers.

**4c. Evaluate the epochs on fixed inputs.** Extend benchmarking to accept an explicit checkpoint list/manifest or create a dedicated epoch-evaluation command using the existing loader and scoring code. `lancebench` currently finds only `best.pt`; do not copy epoch files into many fake run directories or flood the normal model picker with 90 independent runs.

Score all 90 snapshots on the historical Lance cache and on one frozen operational candidate cache. For the latter, select deterministic windows spread across the recording before inspecting epoch predictions; retain all eligible candidate centers and record actual windows, labels and prevalence. Reuse identical candidate centers, neighborhoods, point samples and labels across every epoch/model compatible with the contract. Store raw predictions keyed by sample/acquisition identity and checkpoint hash so metrics can be independently recomputed.

Keep the historical and operational columns separate. For each epoch record average precision, precision/recall at its own saved threshold, per-rock candidate recall and per-region false-positive diagnostics. Candidate recall is not map completeness. Do not silently drop rocks with no positive predictions or change negative prevalence between models.

Plot both Lance curves and volleyball validation against epoch for each seed, marking all selection rules above. Add within-run rank correlations as descriptive statistics, but lead with the actual paired changes in selected-checkpoint performance. Report individual seeds, not just a mean over 90 correlated snapshots. More searched epochs creates more opportunity to select noise; it does not create more independent experiments.

**4d. Decide which snapshots need expensive map replay.** For each seed, replay the original-policy checkpoint and the historical-Lance-best checkpoint. Also replay the operational-best checkpoint when different and budget permits; record any deferred candidate. Deduplicate identical checkpoint hashes. Alongside the three incumbents, this is normally nine and at most twelve unique model map evaluations. Cache scores once per checkpoint and share all geometry. Run cleanup policies afterward from these cached scores; do not rerun model inference for every cleanup setting.

The map table must explicitly compare every finalist with the original 0.7112 checkpoint. Include stored thresholds and exact uncleaned frontiers, followed by a small set of actual cleaned-policy replays at declared thresholds. Inspect all automatic cases, particularly any new loss on rock 10, rock 12 or the incumbent's weakest rock. Retain both footprint and 3D attribution, timing and false-area duration.

**4e. Define the result before seeing it.** A historical Lance score greater than **0.7112027028739645**, reproduced on the same cache and loader, is a new observed sampled ranking record. A tiny excess is not proof of generalization. Report its absolute difference, seed, epoch, selection method and map behavior. If all candidates remain below it, say plainly that the incumbent was not beaten and retain its weights as the best known ranking checkpoint.

A model can improve the operational map while remaining below 0.7112 on the historical cache. That is a separate result, requiring direct comparison against the original incumbent's operational map, not a comparison only with the deployed stray model. Do not turn a better global minimum into a claim that every rock was preserved. Until navigation timing tolerances exist, report the full per-rock tradeoff and keep candidates with regressions as review candidates rather than declaring unconditional replacement.

Lance-selected snapshots must carry explicit selection provenance in manifests and dashboard labels. The recording is development data. Cross-region or cross-rock validation within it can reduce some selection reuse, but cannot turn the already-used arena into a fresh test. Final confirmation requires a new competition-like recording when available; this does not block useful development comparisons now.

**5. Audit the false positives before designing real-negative training**

Use the original incumbent and the best new operational candidate, not only the deployed model. Find roughly **60 spatially distinct persistent false-positive patches**, sampled across arena regions, ranges, time intervals, model agreement/disagreement, known/unknown ground support and clearance bands. Collapse repeated sightings into one physical patch. Use a shared matched false-footprint budget for model comparisons, with stored-threshold diagnostics alongside it.

For every patch retain a 3D view, top-down view, several raw windows, sensor trajectory, local support estimate/uncertainty, prediction history and label boundaries. Assign a category supported by the images: verified floor, rough substrate/berm or other real geometry, detached returns, registration/pose artifact, label ambiguity, or unresolved. Height alone must not determine the category. Report distribution among the selected audit cases, not an unweighted estimate of all errors if sampling was stratified.

Recompute height distributions with explicit denominators: known versus unknown support, fractions below 5/12/20 cm among known points, signed clearance, support quality, representative versus native voxel populations, and physical patches rather than repeated points. Do not infer “81% within 12 cm” from “19% passed a detachment gate.”

The output is a labelled hard-case manifest and a recommendation for the next data experiment. **Do not automatically launch target adaptation in this campaign.** If verified ground negatives dominate, propose a small paired adaptation experiment using both reviewed target negatives and target positive preservation. Keep all sightings of a physical rock together and exclude overlapping neighborhoods from spatial splits. Compare to a control with the same initialization, training schedule and target-positive exposure but without the added hard-negative loss. Use a declared development split for loss/selection and a separately identified within-recording holdout; a new recording remains the independent test. Avoid a generic negative-only loss guarded solely by volleyball recall—the previous campaign already demonstrated its failure mode.

**6. Make the work usable through the dashboard and deliver a reviewable result**

All new CLI controls must appear through `rocklabel/dashboard/spec.py`; inventory/readers must expose the new artifact types. Training should offer snapshot frequency and full-schedule behavior under Advanced. Epochs should be grouped under their parent run, with a comparison plot and explicit volleyball-selected/Lance-selected labels. Provide a way to replay a chosen epoch without renaming it to `best.pt`.

Map evaluation should name footprint coverage versus 3D attribution, show the true threshold range, and distinguish frozen-map sweeps from policy replays. Update help for pose uncertainty and evidence windows to match corrected behavior; remove claims that a larger pose uncertainty simply widens a safe clearing target. Show pending/incomplete audits distinctly from completed ones.

Meaningful verification includes the new evidence regressions; live/replay preparation parity; cache invalidation and interrupted-build recovery; synthetic coverage/threshold/time accounting; epoch snapshot load/resume; and dashboard/parser agreement. Run `tests/test_live_evidence.py`, affected training/live/evaluation tests, and `tests/test_dashboard.py`; expand only where the changes justify it. Finish with `git diff --check`. Record tested behavior and any limitations rather than claiming safety from a test count.

Suggested deliverables:

- `training/reports/epoch-selection-v1/manifest.json`: identities, frozen source/configs, planned fits, evaluation contracts and selection policy.
- `epoch-scores.csv` and per-seed curves: every retained epoch, both Lance populations, original-policy selection, raw prediction references.
- `selection-comparison.csv`: each seed's old selected epoch, new selected epochs, score differences and map audit links.
- `incumbent-map-comparison/`: original 0.7112 model, new 0.7044 reference, deployed reference and finalists, with both coverage definitions and exact frontiers.
- `training/reports/map-evidence-v2/`: corrected evidence experiments, paired control maps, runtime, error cases and inspected-image ledger.
- `false-positive-audit/`: patch manifest, reviewed categories, images and an explicitly proposed next data experiment.
- A concise final report answering: **Was 0.7112 beaten? Was its operational map beaten? Did epoch selection help before the original stopping point? Did extra epochs help? What did corrected clearing remove, and which rocks did it hurt?**

Correct the unsupported standing claims in `CLAUDE.md` and summaries using these results, preserving a clear historical record rather than rewriting old numbers. In particular, preserve the original checkpoint as incumbent until directly superseded; replace the universal false-positive floor, universal seed-noise floor, “same recipe thirty times,” “every measure improved,” and unqualified clearing-safety claims with statements the measurements support.

Work order: establish hashes/contracts and snapshot support first; start the three frozen fits once snapshot verification passes; improve evaluation and evidence while those fits run; score epochs only on verified fixed caches; replay incumbent and finalists; inspect hard cases; then write the recommendation. If time is constrained, prioritize the original-incumbent audit and complete epoch comparisons over optional PointNet++ work or additional hyperparameters. Preserve checkpoints and resumable state rather than rushing an unevaluated promotion.
