**Judgment after reviewing the clutter campaign and map-evidence results:** the combined-clutter classifier is a strong replacement candidate, the tested recipe variants did not earn further tuning, and epoch selection is a worthwhile next experiment. However, the recommendation to deploy unconditionally, the claim that the clearing implementation is safe, and several explanations in the write-ups exceed the evidence. Correct the evaluation and clearing issues below before turning those conclusions into standing guidance.

**Incumbent clarification:** the replacement comparison in this review is against the deployed 0.5399 model, not proof of improvement over the existing best 0.7112 checkpoint. The new campaign tops out at 0.7044. The original `stray/cls-both/loro_VolleyBallTest4.reslam/best.pt` remains the primary ranking incumbent and needs a direct operational map comparison. The subsequent implementation plan is [next-agent-plan.md](next-agent-plan.md).

This review read both full write-ups, `CLAUDE.md`, all three map summaries and per-rock tables, all three map images, and the relevant training, inference and evidence code. It recomputed exact operating points from the saved control maps and original per-rock observed-cell sets. It also exercised small, direct counterexamples against the current clearing implementation. It did not rerun inference or train models, change runtime code, or update the dashboard.

**What I agree with**

- Combined phantom-and-stray training is a productive direction. Its mean competition average precision is substantially above the existing deployed checkpoint. The new evidence reinforces the previous multi-fold comparison. Keep all volleyball recordings in the main training campaign.
- My proposed candidate-height feature did not deliver a repeatable improvement under the tested training setup. Further tuning of that input is not a priority. The observed positive response to height is consistent with a source-domain shortcut; the counterfactual intervention alone does not establish the mechanism for every real error.
- Halving jitter and preserving positives did not show reliable gains. Three seeds do not prove zero effect, but they provide sufficient reason to spend the next experiment elsewhere.
- PointNet++ deserves to remain a candidate. Its measured speed on this card removes the old blanket performance objection. Its clutter combination did not show a gain in these fits; that does not prove the architecture and augmentation learn identical mechanisms.
- Evidence-based clearing addresses a real persistence mechanism and gives a modest improvement in the recorded maps. It should remain optional while the implementation issues below are resolved.

**The new model is better on aggregate, but not better on every rock**

The proposed `clutter/cls-both-s44/trainall` checkpoint raises final mean 3D-attributed rock coverage from 0.650 to 0.694 at stored thresholds. However, rock 10 falls from **7/11 cells (63.6%) to 4/11 (36.4%)**. The old minimum was rock 12 at 5/17 (29.4%); that rock improves while rock 10 becomes the new minimum. A rising minimum across rocks is not the same as preserving every rock.

The rock-10 regression persists at equal false-positive budgets of 2,200 and 1,700 cells. The combined model is a worthwhile candidate, but “beats every measure” and an unconditional deployment recommendation hide a material per-rock tradeoff. Inspect acquisition and loss intervals for rocks 10 and 12, not just final means. The first-covered timestamp records the first single covered cell, not useful obstacle completeness or a stopping margin.

**The threshold sweep was incomplete**

`map_eval.py` sweeps thresholds only from 0.05 through 0.95. The reported claim that the deployed model cannot get below about 1,650 false cells at any threshold is therefore incorrect. On its saved map, threshold 0.99 leaves **1,149** false 3D-attributed cells and 0.999 leaves **388**. Recall falls, as expected.

I evaluated every distinct score in the saved control maps. At each budget, the table selects the lowest threshold satisfying the budget, with tied scores handled together. These are retrospective oracle comparisons on this recording, not calibrated deployment thresholds. Coverage uses the existing evaluator's 3D-attributed, equally weighted physical-rock definition.

| False 3D-attributed cells allowed | Deployed mean / minimum coverage | Combined mean / minimum | PointNet++ mean / minimum |
|---|---:|---:|---:|
| 2,200 | 0.648 / 0.294 | **0.694 / 0.364** | 0.693 / 0.294 |
| 1,700 | 0.632 / 0.294 | **0.694 / 0.364** | 0.675 / 0.294 |
| 1,000 | 0.573 / 0.294 | **0.645 / 0.364** | 0.630 / 0.118 |
| 500 | 0.404 / 0.118 | **0.552 / 0.176** | 0.482 / 0.059 |
| 200 | 0.259 / **0.071** | **0.386** / 0.000 | 0.306 / 0.059 |

The combined model's mean-coverage advantage survives this correction at these budgets. Its minimum is not superior at every operating point: at 200 false cells it misses a rock entirely. This is a stronger and more useful conclusion than the incomplete sweep supported.

For a cleaned map, changing the threshold also changes which voxels receive contradiction updates and can be removed. Reading a finished cleaned map at other thresholds does not replay that policy at those thresholds. Label such curves as frozen-map diagnostics, or rebuild the evidence map from cached geometry and scores for each threshold.

**Separate representation accuracy from occupied rock footprint**

The reported 0.650 → 0.683 clearing gain measures whether the highest-scoring representative of each ground cell lies inside the 3D rock label. Removing a floating representative can expose a correctly labelled lower point in the same already occupied ground cell. That is better attribution, not added obstacle coverage.

The baseline summary actually reports occupied observed rock cells changing **372 → 371** with clearing. For the combined model they remain **371 → 371**, while 3D-attributed mean coverage slightly falls, 0.694 → 0.691. Report both quantities explicitly. The first table's 2,211 “ground claimed with no rock on it” is also the 3D-attribution false count; the actual outside-footprint count is 2,195, falling to 2,068 with clearing. These quantities are close here but answer different questions.

The figures confirm that substantial false occupied area remains with every tested combination. The cleanup is useful incremental work, not a resolution of the map problem.

**Three reproduced clearing implementation issues**

1. **Nearby rays can be counted as crossing a voxel they miss.** `evidence.py:300` uses an angular cone based on the voxel half-diagonal plus 3 cm of pose slack. With a query at `(1.025, 0.025, 0.425)`, origin `(0, 0.025, 0.425)`, and endpoint `(2.05, 0.145, 0.425)`, `passed_through` returns true using default settings. The actual query voxel is `[1,1.05] × [0,0.05] × [0.4,0.45]`; the ray's y coordinate is about 0.084–0.086 while crossing that x interval, entirely outside the voxel. Thus this is a proximity heuristic, not proof of free volume. Increasing uncertainty currently makes deletion easier. Use the angular search as a broad candidate search, then an appropriate voxel/beam intersection and uncertainty-aware evidence test. A single ray through part of a voxel still does not certify its entire volume empty.

2. **Distinct observation windows are not enforced.** `observe()` has no scan/window identity and increments evidence every call. The live scorer can repeatedly read the same buffered snapshot during pause, stalled input or overlapping scoring windows. Repeating the identical observation three times produces a retraction with `require_detached=False`; the default ground gate does not cure duplicated evidence once its surface is established. Pass acquisition identifiers through the buffer, count new sensor observations once, and make repeated processing idempotent. Repeated calls to `retract()` within one observed window should likewise not spend evidence again.

3. **A current return does not universally prevent deletion.** `retract()` excludes fresh hits from new contradiction updates, but its final `drop` mask omits `~fresh`. At the supported setting `free_windows=2`, prior evidence −4 becomes −2 after a new hit, and the voxel is still immediately retracted. This was reproduced directly with the ground gate disabled to isolate the evidence rule. The same final-mask logic applies when the ground gate marks a voxel detached. Make current-hit protection explicit in the final decision and test it after maximum negative evidence for every allowed setting.

The existing evidence suite passed all 12 tests during this review; those tests do not cover these counterexamples. Add regression cases and rerun the replay after fixing the rules; the reported 6% improvement may change.

There is also an evaluation/live mismatch: the offline evidence layer receives only cropped returns from every tenth 50 ms window, while live evidence receives the whole recent window after the floating filter. Offline geometry omits that filter. This is useful sampled replay evidence, but not an exact reproduction of the whole live evidence stream. In particular, rays ending beyond the evaluation crop cannot contribute offline. Use shared preparation and preserve the full ray stream separately from cropped model inputs before declaring a live clearing ceiling or latency guarantee.

**Statements to soften in CLAUDE.md and the write-ups**

- “81% are within 12 cm” is not established by the displayed diagnostics. They report 19% passing a detachment gate, 71% with known surface support, and a median clearance of 12 cm. Not passing a gate that includes support, residual and 20 cm separation is not equivalent to being within 12 cm. Unknown support is also not proof of ground contact. Ground-level semantic errors are plausible, but visually audit their geometry and report the actual height distribution before treating that percentage as fact.
- “Noise floor 0.014–0.049” uses standard deviations as though they were a universal minimum detectable effect. Effects require paired differences and their uncertainty. “Thirty fits of the same recipe” also includes five different arms and two training splits. The original source changed, so failure to exactly reproduce the old checkpoint is not evidence that it is intrinsically irreproducible.
- Correlation 0.12 across selected checkpoints supports questioning volleyball validation. It does not measure the within-run relationship between epoch scores and Lance performance. Saving and scoring other epochs is the correct next experiment, not an already demonstrated larger gain.
- The map stores the newest probability per 3D voxel, then maximizes over surviving voxels in a column. It behaves like persistent accumulation across different voxels, but is not literally the maximum score ever observed: revisits can lower a voxel's score.

**What I would do next**

First fix the clearing counterexamples and reporting distinctions, extend threshold sweeps through all unique scores, and recheck per-rock acquisition/coverage losses. Keep the combined classifier as the leading candidate rather than starting another broad architecture campaign.

For the next training experiment, save every epoch for three paired seeds of the combined recipe on all volleyball recordings. Complete the planned schedule for this diagnostic instead of allowing volleyball patience to discard all later epochs. Score the same operational Lance candidates at each epoch; report each seed's gain over its volleyball-selected checkpoint and evaluate the most promising epochs on full maps. Treat Lance selection as development and reserve a new recording for confirmation.

Separately inspect a modest set of spatially distinct, persistent false-positive patches, stratified by height/support, distance, proximity to rocks and time. Determine whether they are flat floor, rough arena substrate, boundaries, registration artifacts or missed labels. Train on verified real arena negatives only after identifying the dominant error, and include target-domain positive preservation: the previous negative-only adaptation sacrificed Lance rocks despite passing its volleyball recall guard. More synthetic airborne clouds are unlikely to address true ground-surface confusion.
