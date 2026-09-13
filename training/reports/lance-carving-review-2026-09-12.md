# Lance accumulation review and implementation handoff

Review only. No application code or tests were changed. The existing uncommitted experiment was inspected in place. The supplied handoff and its memory note were treated as evidence to evaluate, not instructions to adopt their conclusion.

**Recommendation:** continue experimenting with conservative temporal accumulation. Fix visibility and observation bookkeeping before tuning stronger removal. The current result demonstrates a tradeoff in this implementation; it does not establish that smarter accumulation is unsuitable for Lance. Keep the experiment opt-in until a matched, per-rock visual evaluation passes.

## Evidence checked

- `rocklabel/geometry/carve.py`, `rocklabel/live/carving.py`, all integration changes in the working diff, and the existing accumulation, replay, filtering, and label-membership paths.
- Original mapper: `/home/brandon/lance-ws/src/cardinal-perception/include/modules/impl/kfc_map_impl.hpp`.
- Original note: `/home/brandon/.claude/projects/-home-brandon-Documents-perception-2026-testing-rocklabel/memory/kfc-mapping-not-worth-porting.md`.
- Original experiment scripts and caches under `/tmp/claude-1000/-home-brandon-Documents-perception-2026-testing-rocklabel/82a4df22-1b8d-455b-961b-97612f225704/scratchpad/`, particularly `cache.py`, `final.py`, and `sweeps_moving.npz`.
- The supplied screenshot. It shows scattered returns and coherent bands of geometry, but does not establish which individual points are phantoms, which layer is displayed, or their observation histories.

Read-only Python probes exercised the existing classes. A subclass observed `_delete()` calls without changing map updates during replay of the cached 300-sweep driving window. The unchanged implementation reproduced **45,991 total voxels and 2,360 voxels inside the experiment's rock spheres**.

| Driving-window diagnostic | Count |
| --- | ---: |
| Total deletion events | 43,270 |
| Events with no triggering ray endpoint >5 cm farther than the mapped point | 24,832 |
| Events where every triggering ray endpoint was >5 cm nearer | 17,676 |
| Events involving an invalid normal | 110 |
| Deletion events inside the experiment's rock spheres | 5,678 |
| Of those, no triggering endpoint >5 cm farther | 2,969 |
| Of those, every triggering endpoint >5 cm nearer | 2,190 |

For this diagnostic, a triggering pair is one for which the current `on_surface` test is false. Range difference is `srng[ray_i] - sub_rng[map_i]`; 5 cm matches the configured half surface width. These are repeated deletion events, not distinct physical rocks, unique lost voxels, or a prediction of recoverable coverage. The cached poses and points were accepted as supplied; the recording was not decoded again for this reproduction.

**Interpretation:** missing normals explain very few deletion events under the current tuned configuration. This does not refute an earlier result with a smaller search radius. It does mean that further increasing that radius is a poor first response to the current failure. Pose error remains a hypothesis, not an established explanation.

## Findings, in priority order

**1. The deletion rule does not distinguish free space from occlusion.** In `geometry/carve.py:133–153`, a near-collinear ray can remove a map point whether its return is nearer or farther. The local-plane comparison uses absolute range disagreement. A point at `(2,0,0)` is deleted by a new return at `(1,0,0)` from the origin. The new return does not observe space behind itself. This behavior also exists in the original C++; copying it faithfully does not make it appropriate for this use.

**2. A single observation can erase accumulated support, including a matching endpoint.** A lone point at `(2,0,0)` followed by an identical return is deleted and recreated with `hits=1`. One contradictory pair wins over other supporting pairs because deletion takes the union of every failed surface test. There is no confidence, disagreement history, or hysteresis. Existing `hits` counts raw returns, not distinct sweeps; the additive accumulator has the same raw-count distinction. Raising `min_hits` is therefore not a substitute for temporal evidence.

**3. Batching corrupts the ray origins.** `live/carving.py:75–96` preserves endpoints but assigns the latest origin to the entire pool. Preserve each observation's origin while batching computation. A diagnostic grouping of eight consecutive cached sweeps, approximating 0.4 seconds, produced origin displacement relative to the last member of 1.7 cm median, 10.1 cm at the 95th percentile, and 16.6 cm maximum. These are translation displacements, not measured ray errors or a direct replay of `CarvingAccum`, but they undermine the claim that viewpoint movement is negligible compared with the 8 mm radial gate. `WindowedScanStream` also retains only the last member's pose, so future ray caches must retain source origins before that merge.

**4. Sparse and degenerate neighborhoods can also protect ghosts.** `_normals()` declares a plane valid solely from neighbor count. Three collinear points are accepted. A probe with `(1,0,0)`, `(1.1,0,0)`, `(1.2,0,0)` followed by a farther return at `(2,0,0)` retained all three old points: an arbitrary normal and the parallel-plane exception protected them. Larger neighborhoods can bridge separate surfaces; a valid neighborhood needs geometric quality checks.

**5. The live map receives only a subset of potential evidence.** `live/pipeline.py:650–677` applies the height filter and then `accum_subsample`, normally every fourth point, before carving. The display stride therefore changes deletion evidence. Missing samples must mean unknown, not free space. Filtered or absent returns cannot be converted into infinite rays. The existing 8 mm corridor also tests a voxel centroid, which need not coincide with any original return; both missed contradictions and edge damage need explicit examination before changing its width.

**6. Cleanup scope is limited.** Live carving changes `self.accum`, while `_recent`, `self.surface`, and `self.raw` receive the original filtered batches. It cannot directly remove stale model or heightmap output. Offline labeling does use the carved map, but its defaults differ: 3 cm voxels and 6 m carving ranges versus live's 5 cm and 8 m. The label accumulation path does not apply the live height filter. These are separate configurations and must be reported separately.

**7. There are smaller correctness and interface gaps.** `add_max_range` is enforced only on the final `_add()` branch, not initialization or early returns; a first return at 10 m is accepted with the 6 m default. A map point coincident with the origin can create NaNs and crash the KD-tree. `snapshot()` never flushes pending batches, so the recording tail may remain absent indefinitely. Intensity is discarded. The point-cap setter does not trim immediately, and the implementation silently deletes distant map geometry to meet a render cap. Carved-point counts are reported in the old frame-count field. There are no carving-specific tests; the only test change accepts a new fake argument.

## Why the original comparison needs repair

The original additive and height-filter baselines accept all cached ranges; KFC uses a nominal 6 m addition limit, inconsistently applied. Rebuilding just the baselines with a consistent per-observation 6 m limit gives:

| Cached driving baseline, 5 cm voxels | All ranges: total / rock spheres | Within 6 m: total / rock spheres |
| --- | --- | --- |
| Additive | 114,637 / 3,883 | 62,511 / 3,883 |
| Height filter | 67,944 / 3,769 | 40,426 / 3,769 |

Thus much of a total-count difference can come from range restriction. This particular restriction does not explain the lost rock-sphere voxels in this window.

The script evaluates label centers and bounding radii as spheres, although Lance labels include polygon footprints and height intervals. Use `dataset.labeling.points_in_rock()` for the actual shapes. Aggregate voxel totals also let large or frequently observed rocks dominate.

The fog metric recomputes ground and neighbor density separately from each candidate output. Removing part of a wall can turn its surviving points from “structure” into “fog”; removing ground can change the height test itself. This is a useful diagnostic, not fixed phantom ground truth. Points outside rock labels are not automatically phantoms.

## Implementation sequence for the next agent

### 1. Establish a reproducible Lance mapping audit

Preserve the existing KFC variant as a reference. Add a recording/config manifest with exact source time windows, label and recording identity, level transform, source topic/frame, voxel size, ranges, prefilters, source grouping, rendering stride, memory limits, and software revision. Decode a ray-preserving cache with endpoints, corresponding origins, timestamps, source scan IDs, intensity, and validity information. Retain return/beam metadata where the recording actually supplies it; explicitly record unavailable fields.

Use identical inputs and operational range/floor band for every primary arm. Compare additive, existing height filter, legacy KFC, corrected KFC geometry, and then temporal-evidence variants, with the height filter on/off as a separate ablation. Keep native live display comparisons separate from matched algorithm comparisons because a frame window and a voxel map retain different populations.

Select and freeze inspection regions before tuning: every labeled physical rock, clear isolated phantoms, coherent suspected ghosts, ground, walls, and obstacle edges. Inspect source sweeps to distinguish ghosts from real structure. Include parked, moving, turning, occluded, and late-revisit intervals where available. Reserve separate intervals/regions for evaluation after tuning; replay the full Lance run for the final check.

### 2. Correct visibility and observation handling first

Introduce a shared ray-batch contract for offline and live accumulation. Each endpoint must retain its source origin, time, and observation group. Batch scheduling may change when work runs, but must not invent one viewpoint for different observations. Confirm frame transforms and available scan timing before assuming finer deskew is possible.

For endpoint `q`, origin `o`, and map representative `p`, compute ray unit direction `u`, endpoint range `R`, projected distance `t=(p-o)·u`, and perpendicular distance `rho=||(p-o)-t*u||`. A miss requires positive distance along the ray, membership in a justified narrow corridor, and `t < R - endpoint_margin`. Space behind a return is occluded/unknown. Nearby endpoints provide positive support independently of normal availability. Ambiguous geometry provides no negative vote.

Set endpoint and lateral uncertainty rules from voxel representation, sensor residuals, and measured pose consistency. Poor pose quality should suppress negative evidence or enlarge the ambiguity region; it should not simply widen a destructive corridor. Reject grazing/edge cases whose evidence cannot resolve the occupied region. Where multiple echoes are available, do not interpret a farther echo as free space through a nearer return.

Aggregate support and contradictions before mutation. Supporting endpoints in the same observation group must prevent destructive delete/re-add churn. Enforce input validity and addition range on every branch. Preserve intensity and raw counts, adding distinct observation counts separately.

Add regression tests for near occlusion, repeated endpoints, real pass-through, competing support/miss rays, different origins in one scheduled fold, uncertain poses, surface edges, invalid/zero-length inputs, and range handling across early returns. Re-run the audit before introducing more heuristics, so the effect of this correction is measurable.

### 3. Add bounded temporal evidence for remaining phantoms

Track positive support, contradiction evidence, last-seen time, and last independent observation group per voxel. Cap evidence to one vote per voxel per source sweep/time group; hundreds of telegram points must not become hundreds of independent confirmations. Use bounded scores or a finite evidence history so neither ancient hits nor a single later miss controls a voxel forever.

Maintain tentative and confirmed states. Require repeated credible observations for confirmation and multiple independent free-space contradictions for removal of confirmed geometry. Repeated supported observations should restore confidence. Retain a short contradiction history after removal so one weak return cannot immediately resurrect a known ghost. All thresholds are experimental parameters, not values established by this review.

Show tentative observations distinctly in diagnostics so new small rocks remain visible. Confirmed rocks should not expire solely because they were occluded, out of range, or between beam rings. A tentative-point age policy may help never-reobserved fog, but evaluate its first-detection delay and sparse-rock loss separately. Persistent coherent ghosts can acquire support too; temporal hit count alone is insufficient.

Use normal estimation as additional geometric evidence. Require non-collinear spatial support, sensible eigenvalue ratios and plane residuals, and guarded near-parallel intersections. Missing normals mean insufficient surface information, not automatic deletion. Keep the batched neighbor query and stacked eigendecomposition; do not return to Python loops per map point. Attempt range-adaptive neighborhoods only after the corrected visibility and temporal variants have been evaluated.

### 4. Complete integration without changing the evidence

Keep rendering decimation downstream of evidence collection. If ray reduction is necessary for throughput, benchmark an angular/source-aware scheme that preserves nearest returns and document the evidence lost. Preserve source ordering within scheduled batches and ensure grouping parameters do not silently redefine temporal votes.

Flush pending data deterministically on replay completion/export and define pause/reset/seek behavior. Keep snapshots inexpensive and consistent. Separate render limits from explicit map-memory eviction, and report eviction separately from evidence-based removal. Fix intensity, statistics labels, configuration validation, and inactive frame-window controls.

Measure ingest backlog, dropped batches, fold latency including its tail, memory, and displayed-map delay on Lance at real time and accelerated replay. Completion of an unpaced replay alone does not prove live throughput. Cache/reuse spatial work where profiling supports it; preserve vectorization.

Treat downstream model/heightmap use as a separate integration milestone after map quality passes. Define deletion propagation or rebuilding for those consumers; changing the display cloud alone cannot retract existing detections. If scoring inputs change, evaluate the same checkpoint at its stored threshold before any threshold adjustment. Any classifier/segmenter comparison must follow repository `AGENTS.md` and the labeled visual audit requirements.

### 5. Acceptance and deliverables

Deliver `summary.md`, per-rock coverage tables, fixed-region phantom results, time-history/fragmentation measurements, top-down and side-view case images, resolved configuration, runtime results, and regression tests. Use a shared spatial grid and report each physical rock once in the main aggregate. Include both XY footprint completeness and 3D support so a few ground cells cannot stand in for a mapped rock. Distinguish never-observed cells from cells actually lost by accumulation.

Inspect every selected case and each rock before recommending the new mode. Require lower phantom occupancy/persistence in the fixed regions, no newly persistently missed rock, preserved useful rock shape and obstacle geometry, and sustainable ingestion. Report each rock's regression even if an aggregate improves. Choose any numeric coverage tolerance before the tuning run and state it in the manifest; this review does not establish a justified universal percentage. A narrow floor crop or whole-scene thinning cannot satisfy the primary acceptance test.

The existing screenshot and cached count reproduction are not a completed visual audit, and no revised mapper was implemented or measured here. The next agent's first deliverable should be corrected visibility plus trustworthy evidence, followed by temporal accumulation if the remaining cases justify it.
