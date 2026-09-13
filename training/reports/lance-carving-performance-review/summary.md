# Lance carving: second review and performance handoff

The revised mapper fixes the major original visibility mistakes, but still has an observation-group correctness bug and performs too much work on the whole map. Prioritize group correctness, spatial-query cost, and map storage. Thread settings and display reductions alone will not recover real-time performance.

This was a review, not a feature implementation. Application code and tests were left unchanged. Temporary diagnostic scripts were written under `/tmp`; this directory contains only the review and measurement data. See [measurements.json](measurements.json) for source hashes, configuration, timings, and environment. The first review remains in `training/reports/lance-carving-review-2026-09-12.md`.

## What was verified

The new implementation correctly distinguishes nearer occluding returns from farther free-space observations, protects matching endpoints, preserves per-endpoint origins, limits temporal votes, checks normal degeneracy, keeps intensity, enforces addition range across paths, and adds resurrection evidence. The test selection covering carving, live accumulation, live configuration, and labeling passed: **68 tests**.

The first pytest invocation hit an unrelated, auto-loaded ROS plugin's missing `lark` dependency. Disabling third-party plugin auto-loading allowed the focused suite to run. The full repository suite was not repeated.

Additional probes exposed gaps below. No revised algorithm was implemented, and no complete per-rock visual audit was performed in this review. Passing the focused tests is not proof of Lance map-quality acceptance.

Recent dashboard jobs `j0126` and `j0127` replayed `trimmed_lance.mcap` with `--carve`, `--max-range 8`, the full `--floor-band -0.10 0.60`, model coloring, and the web UI. Their launch commands did not override carving parameters. They do not establish which controls the user subsequently changed.

## Performance evidence

Measurements used the repository virtual environment: NumPy 2.5.1, SciPy 1.18.0, 22 logical CPUs. The existing 300-sweep Lance driving cache supplied points and origins. The first 240 sweeps built a 58,612-voxel map. Each worker-count comparison started from a copy of that same map and processed the final 60 sweeps, ending at 76,201 voxels.

Settings were 5 cm voxels, 8 m addition/deletion range, and current default geometric/evidence parameters. Each cached sweep was one observation group. The benchmark excludes decoding, height filtering, rendering, inference, and the live wrapper. The cache does not retain original timestamps; the speed equivalents below assume the experiment's nominal 50 ms per sweep. They isolate a slowdown comparable in magnitude to the user's report, not an exact reproduction of every live setting.

| KD-tree workers | Time for final 60 sweeps | Nominal speed | Median update | 95th percentile update |
| --- | ---: | ---: | ---: | ---: |
| Current `-1` | 10.32 s | 0.29x | 168 ms | 207 ms |
| 1 | 13.26 s | 0.23x | 219 ms | 268 ms |
| 4 | 10.26 s | 0.29x | 172 ms | 201 ms |

The compared map coordinates, hits, evidence arrays, confirmation flags, and intensity accumulators matched exactly between worker variants. Each complete timing arm ran once; medians/percentiles are over its 60 updates, not repeated complete trials. Method-level instrumentation was active; a separate cProfile run was not included in these timings.

Approximate mean costs per update with current workers:

| Stage | Time |
| --- | ---: |
| Endpoint support search | 36.6 ms |
| Visibility search, candidate processing, and normals | 97.5 ms |
| Deletion and storage compaction | 24.9 ms |
| Addition | 5.6 ms |
| Other update/group work | 7.4 ms |

Within visibility, building the directional tree costs about 24.0 ms. Normal estimation costs 31.6 ms, of which 23.9 ms is another tree build. The implementation already limits eigendecomposition to candidate points; matrix decomposition itself is not the main normal-estimation bottleneck.

The direction and normal trees were rebuilt for every measured update. `_delete()` ran on all 60 measured updates, copying all aligned arrays and rebuilding the complete key-to-row dictionary each time. The map need not hit its point cap for this cost to grow.

### A measured small optimization

`KFCMap._supported_rows()` builds a neighbor list for every mapped point, then discards the neighbors and retains only whether each list is nonempty. On a frozen 58,612-voxel map and one incoming Lance sweep, prebuilt-tree queries gave:

| Query form, workers=-1 | Median of four measurements | Supported rows |
| --- | ---: | ---: |
| Existing lists plus Python boolean conversion | 22.4 ms | 8,304 |
| `query_ball_point(..., return_length=True) > 0` | 10.0 ms | 8,304 |
| Bounded nearest-neighbor query, `k=1` | 5.0 ms | 8,304 |

All three produced identical support masks for this sample. Tree construction and same-voxel support lookup are excluded here. This is a stage-level improvement, not a fourfold speedup of the mapper. Preserve inclusive endpoint-margin behavior and same-voxel support; the diagnostic nearest query used `nextafter(margin, +inf)` as its search bound and an explicit `distance <= margin` test.

### Larger moving groups can cause an explosion in candidate pairs

`_visibility_rows()` expands a single angular search radius using the largest origin spread and the nearest map point. This radius is then applied to all rays and map depths. A frozen-map diagnostic using consecutive cached Lance sweeps measured:

| Sweeps combined into one geometric query | Origin spread | Candidate pairs |
| --- | ---: | ---: |
| 1 | effectively zero | 51,490 |
| 2 | 9.6 mm | 643,613 |
| 4 | 14.7 mm | 2,294,495 |
| 8 | 25.4 mm | 11,723,917 |

The same eight sweeps queried separately from their source origins produced **326,350 pairs** in total: about 36 times fewer than the combined query. The counts-only diagnostic avoided materializing those millions of pairs. The real implementation materializes several arrays per pair, potentially exceeding a gigabyte of temporary storage at that count.

This applies when enlarging the **evidence group** or otherwise combining moving observations within one geometric query. The default 0.4 s **scheduling interval** does not itself combine all eight evidence groups into one query: `_fold()` loops over them. These two controls must remain distinct.

## Correctness findings requiring changes

**P1 — Logical groups are not atomic across scheduling boundaries.** `geometry/carve.py:419–428` deletes immediately, although `live/carving.py:125–165` can send parts of the same evidence group in different folds. Group IDs prevent duplicate votes but cannot undo a completed deletion.

Reproduction: confirm a voxel at `(1,0,0)` with two observations, then give it two independent misses with returns at `(2,0,0)`. In the next group, supply one more miss and one supporting endpoint. A single combined call preserves the voxel. Splitting that group into a miss call followed by a support call deletes it, and its tombstone suppresses the later support. Both cases use identical observations and group IDs. Existing tests cover support-first splitting, not this reverse order or accumulated prior contradiction.

Required behavior: finalize evidence and deletion once all members of a group are known. Keep the open time bucket pending until the next bucket or an explicit final flush. Preview points can be displayed separately if immediate feedback is needed. Geometric query chunks must accumulate support/miss/ambiguous sets against a consistent pre-group map and commit once. Merely deferring `_delete()` is insufficient if support/contradiction counters or surface-blocking decisions still depend on chunk order.

**P2 — End-of-replay flushing is incomplete.** A `flush()` method exists and `IngestEngine.stop()` calls it, but EOF and pause do not stop the engine. The replay source marks itself finished and returns `None`; `_loop()` immediately continues at `live/pipeline.py:614–615`. The last pending observations can remain absent while the finished recording is displayed. Add an explicit EOF/pause completion contract and test the engine/source integration, not only a direct call to `CarvingAccum.flush()`. Coordinate this with open-group finalization, seeking, and restart semantics.

**P2 — Unknown pose uncertainty can be treated as trustworthy.** `update_batch()` drops nonfinite uncertainty values before taking the maximum. A group containing `[NaN, 0]` allows both rays to vote negatively. A probe deleted a mapped point using repeated such groups. Reject invalid negative-evidence inputs per ray, or conservatively mark the whole group uncertain. Both real adapters currently supply zero uncertainty for every ray, so the guard is not connected to measured pose quality. Explicitly distinguish unavailable uncertainty from measured zero uncertainty; establish a documented fallback rather than inventing confidence values.

**P2 — The candidate query is not always a superset of its final radial test.** With one origin, the broad-phase radius is fixed at `fsr=0.015`. A map point at `(0.1, 0.005, 0)` lies inside the 8 mm radial corridor of a ray to `(1,0,0)`, and is clearly before its endpoint, but its direction is outside that angular radius. It gets no contradictions even after repeated observations. If the intended rule is the documented finite radial corridor, derive a range-aware conservative query bound and verify it against brute force. If an angular cap is intentional, make it part of the exact rule and documentation; it should not appear accidentally through candidate retrieval. This is a potential missed-cleanup mechanism, not evidence that this near-range case dominates Lance.

Other limitations to retain in the handoff:

- Endpoints beyond `delete_max_range` are discarded as carving rays even when they could establish free space inside that range. Separate the trusted sensor-return limit from the map's deletion radius before deciding whether to change this behavior.
- `timestamp` is discarded inside the map. Group ordering is the actual temporal contract. Noncontiguous group reuse across separate calls and backward timestamps need a defined reset/rejection policy.
- Evidence uses `uint8` and some `int16` intermediates, but constructor validation does not bound `max_evidence` to its representation. Validate limits or use an appropriate wider type.
- Tentative and confirmed points are both returned and rendered the same way. Confirmation thresholds affect removal thresholds; they do not by themselves hide one-off phantoms. No tentative expiry policy or distinct diagnostic coloring has been wired into the viewer.
- UI statistics still label evidence groups as frames, and the frame-count control remains enabled despite being a no-op. The point cap still evicts actual map geometry. Eviction is now counted separately, which is an improvement.
- Carving still affects the accumulated display cloud, not the scoring window or heightmap. Do not attribute changes in those outputs to this map without a separate integration change and evaluation.

## Implementation requests, in order

### 1. Pin down correctness and establish a performance regression harness

Implement group finalization and the missing lifecycle/uncertainty tests first. Add a small brute-force finite-ray reference to test candidate completeness, support dominance, close-range cases, mixed origins, range boundaries, and chunk invariance. Test prior contradictions and both hit/miss orders. Assert coordinates, raw counts, independent observations, evidence, intensity, tombstones, and deletion totals after each finalized group.

Create a repeatable warmed-map benchmark. Retain recording/cache identity, source timestamps and group IDs, resolved settings, code version, input rays, local/global voxel counts, candidates before/after each gate, support counts, deletion counts, normal queries/tree rebuilds, allocation/memory high-water marks, and stage latency. Separate decoding, accumulation, inference, and display costs. Preserve the current implementation as a timing/reference fixture rather than silently replacing the baseline.

### 2. Apply low-risk query and allocation improvements

- Replace support neighbor-list construction with an existence query. Benchmark nearest-neighbor and `return_length=True`; check boundary behavior and union with same-voxel support. This has a measured benefit.
- Exclude already-supported rows from destructive-candidate processing before normal estimation. In one sample, 80 of 254 normal-query rows were already positively supported. Preserve same-group support accumulated across all chunks.
- Flatten candidate lists with bulk operations rather than allocating a separate `np.full()` array per ray. Avoid sorted neighbor results when order is unused. Use bounded chunks for pair filtering so a moving batch cannot allocate an unbounded pair tensor. Never silently truncate valid candidates.
- Eliminate the per-endpoint Python `_same_group()` loop on already-known single-group input. Represent source observations/group offsets compactly; retain a validated general batch path. This is a secondary optimization, not the dominant fix.
- Choose query worker counts by workload. All-core workers helped the large support query but added overhead to small normal queries. Forcing every call to one worker made the full benchmark worse; changing everything to four workers was effectively neutral.

Reprofile after this step. These changes alone are unlikely to supply the roughly fourfold overall gain needed from a 0.25x baseline.

### 3. Make work depend on local changes instead of total accumulated history

**Storage:** use stable voxel IDs and capacity-managed arrays with a free list/active mask, or another representation with incremental insertion/removal. Remove deleted keys incrementally and compact only occasionally. Avoid copying twelve aligned arrays and rebuilding every key on each group. Keep public snapshots dense while internal rows can be sparse. Stable IDs must also prevent cached spatial results from referring to a different voxel after reuse.

**Spatial indexing:** stop rebuilding both full directional and normal-neighbor trees every group. Introduce a local spatial index, such as voxel blocks/tiles with reusable trees and a small update set. Endpoint-support queries should visit cells near new endpoints; normal queries should visit candidate neighborhoods with the required halo. A world map may grow while active work remains local to the sensor. Coarse candidate selection must be conservative and followed by the existing exact per-ray tests.

**Normals:** cache by voxel/neighborhood version and invalidate when nearby geometry changes, including centroid movement, insertions, and deletions. Do not leave a KD-tree referencing an array that is mutated in place, or reuse old normals as if they were exact. Keep the vectorized covariance/eigendecomposition. Benchmark the invalidation rate; caching that is invalidated almost everywhere may not be worthwhile compared with cheaper local queries.

**Mixed origins:** separate logical evidence groups from geometric query chunks. Query exact source-origin subgroups, or range bins with tighter proven bounds, then reduce all evidence once for the original group. The pair-count experiment makes this a high-priority safeguard. Do not collapse viewpoints, change temporal vote counts, or enlarge evidence-group duration as a purportedly equivalent speed fix.

Benchmark a prototype of the local index against the existing reference before committing to a larger storage rewrite. A Python loop over thousands of neighboring voxel keys can also be slow; use measured batched/native operations. Moving the current whole-map algorithm to a GPU would preserve its unnecessary work and add transfer/coordination costs. Native compilation may help a measured kernel later, after its work is bounded.

### 4. Keep the interface responsive independently of throughput

`CarvingAccum.add()` calls `_fold()` while holding `_lock` in the ingest thread. `snapshot()`, `stats()`, and cap changes acquire the same lock. A 0.4 s schedule commonly means eight expensive updates under that lock; at the measured mean that is roughly 1.4 seconds of work. Scheduling a larger interval can lengthen freezes without reducing the update count.

Have one owner for mutable map state and publish versioned immutable render/stat snapshots with a short lock. Do not rebuild or copy the entire map for every UI poll. Keep ingress and UI control paths responsive while mapping runs, with a bounded queue and explicit backlog measurements. This does not solve insufficient processing capacity by itself: an unbounded asynchronous queue would only hide growing latency. Preserve every source group in the reference mode and make any overload/decimation mode explicit and separately evaluated.

Fix EOF/pause/seek/reset/shutdown ownership at the same time. `stop()` currently drops its thread handle after a two-second join timeout even if the expensive fold is still running; avoid concurrent lifecycle actions against a surviving ingest owner.

### 5. Acceptance criteria

- Correctness tests pass, including group/fold/chunk invariance, conservative candidate retrieval, endpoint-boundary support, map-index invalidation, deletion/resurrection bookkeeping, and EOF lifecycle behavior.
- Pure performance refactors preserve finalized map/evidence output against the corrected reference. Any approximate optimization is a separately named ablation with its own quality review.
- Run the same Lance configuration through startup, a filled map, motion, turns, and revisits. The required minimum is sustained real-time end-to-end playback/ingestion with no growing backlog. A useful engineering target is at most 30 ms mean mapping work per 50 ms source group, leaving time for the rest of the application; this is a proposed budget, not an achieved result. Report tail latency and worst sustained windows too.
- Test larger global maps while keeping the active local region comparable. Runtime should track local occupancy and actual ray candidates, not the number of distant voxels retained from earlier travel. Report memory and UI latency under the default point cap.
- Retain the full operational floor band and useful obstacle geometry. Review actual polygon labels, per-rock completeness/fragmentation/persistent misses, fixed phantom regions, and every selected case image. Numeric count retention alone cannot establish map quality. Keep classifier/segmenter claims subject to `AGENTS.md`.

## Reproduction assets

Focused suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -B -m pytest \
  tests/test_carving.py tests/test_live_accum.py tests/test_live.py \
  tests/test_labeling.py -q -p no:cacheprovider
```

Temporary scripts are `/tmp/rocklabel_carving_profile.py` and `/tmp/rocklabel_carving_microbench.py`. Warm and filled map snapshots are `/tmp/rocklabel-carving-profile-warm.pkl` and `/tmp/rocklabel-carving-profile-filled.pkl`; a cProfile capture is `/tmp/rocklabel-carving-profile.prof`. These are review artifacts and may disappear with `/tmp` cleanup. Persist an appropriate benchmark harness and ray-preserving cache manifest when implementing the changes. Durable numerical results and source fingerprints are in [measurements.json](measurements.json).
