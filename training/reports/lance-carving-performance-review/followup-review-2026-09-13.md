# Follow-up review: 2026-09-13

The performance improvement is reproducible, and the major earlier geometry/group fixes are present. Four issues remain worth fixing before this iteration is accepted. This review changed no application code or tests.

## Findings

### P1: Pause/resume can terminate the mapping worker

Locations: `rocklabel/live/pipeline.py:404`, `rocklabel/live/pipeline.py:638`, `rocklabel/live/carving.py:349`, `rocklabel/geometry/carve.py:401`.

Both live pause and replay pause call the same finalizing `flush()` used for EOF. That closes the current time bucket and records its identifier in `_closed_group_keys`. Resuming can deliver another observation in that same 50 ms bucket. `_stage_group()` rejects it as already finalized, and the exception terminates the carving worker.

Confirmed using `CarvingAccum` with default evidence grouping:

1. Add a point at timestamp `1.001` and flush, as pause does.
2. Resume with another point at timestamp `1.011` and flush.
3. Result: `RuntimeError('carving worker failed')`, caused by `ValueError('observation group was already finalized')`.

Both timestamps belong to group 20. This also matters for successive small forward seeks. The existing EOF test passes, but does not exercise resumed delivery within a finalized bucket.

**Request:** distinguish a resumable processing/publication barrier from terminal source-group finalization. Pause should drain already queued work while retaining the open group's evidence; EOF can finalize it. If the paused display must include the open observations, publish a clearly defined provisional view without making an irreversible evidence decision. Do not fix this by silently assigning a fresh group ID or clearing the closed-ID set: that would change vote counts with playback controls. Test pause/resume and forward-seek behavior within one bucket against uninterrupted playback, including prior contradictions and both support/miss orders.

### P2: Tentative surfaces lost plane protection as part of the optimization

Location: `rocklabel/geometry/carve.py:1094`, especially `map_index[self.confirmed[map_index]]`.

The optimized path computes normals only for confirmed voxels. Tentative points consequently bypass the grazing/same-surface protection even when they have a valid, well-conditioned local plane. This is a change in removal policy beyond the requested spatial/storage optimization.

A five-point planar neighborhood around `(1,0,0)` has a valid normal. After one observation, two later rays along the plane to `(2,0,0)` delete the central point and two neighbors. When the same geometry is confirmed first, with the same two-miss deletion threshold, the plane protects those points. Thus the difference is not just the configured tentative/confirmed contradiction thresholds.

An isolated in-memory diagnostic restored only plane evaluation for all candidate rows, leaving the optimized indexing, candidate search, storage, and grouping in place:

| Same 300-sweep Lance cache | Submitted code | Plane protection restored |
| --- | ---: | ---: |
| Final voxels | 76,019 | 76,201 |
| Total deletion events | 3,915 | 3,628 |
| Time for final 60 sweeps | 2.298 s | 2.323 s |
| Mean update | 38.29 ms | 38.72 ms |
| p95 update | 44.97 ms | 47.15 ms |

The timing difference is small enough that repeated trials would be needed to establish its precise size. These are voxel counts and repeated deletion events, not per-rock coverage measurements. The restored count matches the earlier review's count, but this alone does not establish complete state equivalence or quality acceptance.

**Request:** restore plane protection for tentative candidates while retaining the performance refactors. If confirmed-only protection is desirable, expose it as a separate experimental policy, document it, and run the actual per-rock and fixed-phantom visual evaluation before recommending it. The current measurement gives no compelling speed reason to bundle that policy change into the optimization.

### P2: The candidate-pair budget does not bound peak pair storage

Locations: `rocklabel/geometry/carve.py:1042` and `rocklabel/geometry/carve.py:1090`.

The overflow query first materializes all neighbor lists. Every broad-phase chunk is then retained in `broad_chunks`, and `exact_chunks = list(candidate_chunks())` retains every exact-pair array as well. Individual chunks respect the budget, but their aggregate storage still grows with all candidate pairs.

A diagnostic with 50 collinear mapped voxels, 1,000 identical farther rays, and `candidate_pair_budget=64` retained **50,000 broad pairs in 800 chunks and 50,000 exact pairs in 800 chunks simultaneously**. This contradicts the documented fixed allocation bound. It may not dominate the present Lance benchmark, whose new candidate counts are small, but dense/repeated rays or larger groups can still cause allocation spikes.

**Request:** bound retrieval before allocating neighbor lists and consume chunks incrementally. Use bounded map-query batches, with bounded ray batches for pathological overflow rows. For the two-pass normal scheme, retain only unique candidate row IDs in the first pass, then regenerate bounded exact chunks for classification, or cache normals incrementally while reducing row-level results. Preserve all valid evidence; do not silently cap the number of neighbors. Add a dense-overflow memory test in addition to the existing chunk-output-equivalence test.

### P2: Queued point arrays can still belong to the caller

Location: `rocklabel/live/carving.py:148`.

`np.ascontiguousarray(..., dtype=np.float64)` does not copy an already contiguous float64 input. The new asynchronous `add()` returns before that array is consumed, so a caller reusing its buffer can change previously queued observations. Intensity and origin are copied, but points are not.

Confirmed by blocking the worker with an event, submitting `[[1,0,0]]`, modifying the caller's array to `[[2,0,0]]` after `add()` returned, then releasing the worker and flushing. The map contained `[[2,0,0]]`.

**Request:** copy point data at the asynchronous ownership boundary, or introduce and enforce an explicit ownership-transfer interface. Also define ownership for staged `RayBatch` inputs: the map now retains them until group finalization. Add a regression that reuses the input buffer after submission. Published evidence arrays should receive read-only flags too if the snapshot contract promises immutability.

## Verified improvements and remaining performance budget

The focused suite passed **170 tests**, covering carving, live accumulation/configuration, labeling, replay recording/lifecycle, and web controls. The initial miss-before-support regression is fixed when the logical group remains open correctly. Unknown/nonfinite uncertainty, close-range candidate completeness, incremental deletion/reinsertion, and EOF publication have targeted tests. The reported 762-test full-suite result was read from the implementation handoff; it was not independently rerun here.

I independently ran `training/benchmarks/benchmark_carving.py` on the same existing driving cache, with 240 warm sweeps and 60 timed sweeps, 5 cm voxels, and 8 m ranges. The benchmark explicitly supplies zero assumed pose uncertainty, so destructive carving was enabled throughout the timed workload. It excludes decoding, rendering, height filtering, and model inference. Source timestamps in this cache are unavailable; speed uses the nominal 50 ms per sweep.

The submitted code achieved **1.31x nominal pure-mapper speed**, compared with approximately **0.29x** in the previous review: about **4.5 times faster**. Its output counts match the implementation report. Mean stage times were:

| Stage | Mean |
| --- | ---: |
| Support | 8.07 ms |
| Visibility and normals | 25.75 ms |
| Deletion | 0.29 ms |
| Addition | 3.22 ms |

The incremental deletion/storage changes have removed the earlier roughly 25 ms-per-group deletion bottleneck. The mapper still exceeds the proposed 30 ms mean budget that leaves room for inference and display. This test does not establish sustained end-to-end performance with the user's model and UI enabled, or across the whole Lance recording.

For subsequent performance work, profile visibility/local-normal work on the corrected version. Keep the tight per-point angular bounds and exact-origin processing. Test larger historical maps: some global-sized lookup arrays and tile scans remain, although the main work is now much more local. Avoid another broad rewrite before fixing the lifecycle/ownership issues and checking the full application workload.

## Operational behavior to make explicit

`--carve` alone now accumulates support but does not carve when the source supplies no uncertainty estimate. The implementation documents `--carve-assumed-pose-uncertainty 0` as an explicit exact-pose experiment. This is an assumption, not evidence that Lance poses are exact. The current benchmark used that assumption; faster support-only playback must not be presented as the carving benchmark.

The full labeled visual audit remains outstanding. Review actual rock polygons, distinct-rock coverage, persistent misses, fragmentation, obstacle geometry, and fixed phantom regions before drawing quality conclusions. This review did not compare classifier and segmenter performance.

## Reproduction

Focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -B -m pytest \
  tests/test_carving.py tests/test_live_accum.py tests/test_live.py \
  tests/test_labeling.py tests/test_live_recording.py tests/test_live_webui.py \
  -q -p no:cacheprovider
```

The independent benchmark is saved alongside this report as `followup-benchmark-2026-09-13.json`. Its manifest records the cache path/hash and mapper source hash. The plane-policy diagnostic changed one condition in a function loaded in memory, not repository code. Its temporary script is `/tmp/carving-third-review-surface-probe-cached.py`; it loads the compressed cache once before timing, matching the benchmark's exclusion of decoding. Initial exploratory diagnostic timings that repeatedly decoded the compressed arrays were discarded. Final diagnostic snapshots remain under `/tmp/carving-third-review-surface-reference.npz` and `/tmp/carving-third-review-snapshot.npz`.
