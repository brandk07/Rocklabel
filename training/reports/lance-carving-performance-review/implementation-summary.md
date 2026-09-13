# Lance carving implementation result

Implemented 2026-09-13 from the adjacent performance review.

## Result

- Evidence groups finalize atomically. Chunks with one group are evaluated
  against one pre-group map state, so miss/support ordering no longer changes
  deletion.
- Every observation retains its source origin. Geometry chunks and logical
  evidence groups are separate.
- Unknown or non-finite pose uncertainty disables destructive votes for the
  whole group. Replays can opt into an explicit fallback; `0` is no longer an
  implicit live/offline adapter assumption.
- Endpoint support uses a nearest-neighbor existence query. Visibility uses a
  conservative range-derived broad phase, bounded candidate chunks, and exact
  finite-ray filtering.
- The map uses capacity-managed arrays plus incrementally maintained voxel,
  block, and tile indexes. Deletion no longer copies all aligned arrays or
  rebuilds the full voxel lookup.
- Normal queries use local block unions and a versioned cache. The cache turns
  itself off when measured invalidation makes it counterproductive.
- The live accumulator has one background map owner and publishes immutable
  snapshots under a short UI lock. Its bounded queue, high-water mark,
  producer waits, and last fold latency are visible in browser, native, and
  headless status.
- Replay EOF and shutdown flush the final open group.
- Pause and completed forward seeks now use a resumable drain barrier rather
  than finalizing their time bucket. EOF, export, and shutdown remain terminal
  flushes, so resuming within one evidence group cannot kill the map worker or
  change its vote count.
- Tentative and confirmed surface candidates receive the same valid-plane
  ambiguity protection; their configured contradiction thresholds remain
  distinct.
- Candidate retrieval, exact filtering, and row-level reduction are streamed
  under the pair budget. Dense overflow no longer retains every broad and
  exact pair chunk simultaneously.
- Asynchronous submissions defensively copy caller-owned point buffers, staged
  ray batches own all aligned arrays, and every published evidence array is
  read-only.

## Measurements

The durable detailed result is in `implementation-benchmark.json`. On the
review's 300-sweep cache, after 240 warm sweeps, the final 60 groups took in
the post-follow-up implementation:

- 2.745 s total instead of the review's 10.3 s reproduction
- 45.75 ms mean, 45.33 ms median, 56.07 ms p95, 62.42 ms max
- 1.09x nominal 50 ms/group throughput
- 76,201 final voxels, 3,628 deletions, zero invalid inputs

This clears the pure-mapper real-time floor on total throughput, though its
p95 exceeds one nominal group period. It does not reach the review's optional
30 ms mean engineering budget. The earlier 2.198 s result used confirmed-only
plane protection and retained all pair chunks in aggregate; it is not the
runtime of the corrected policy now in the worktree.

A post-follow-up 45-second real-time replay of `trimmed_lance.mcap`, with 8 m
range and the full -0.10 to 0.60 m floor band, stayed caught up through motion:
the queue high-water mark reached seven source telegrams, producer waits
stayed at zero, and 0.4-second scheduled folds were about 263--388 ms on the
later, larger portion tested. Rendering and model inference were disabled for
this check.

## Verification

In the current environment, the full repository suite reports `759 passed, 1
skipped`; the focused carving/live/configuration/labeling/replay/web selection
reports `176 passed`. Regressions cover group and chunk invariance, both
hit/miss orders after prior contradictions, mixed origins, unknown uncertainty,
close-range and boundary candidates, brute-force candidate completeness,
incremental deletion/reinsertion, asynchronous UI access, EOF publication,
resumable same-bucket synchronization, tentative plane protection, dense
candidate overflow, caller-buffer ownership, and immutable evidence snapshots.

The implementation remains experimental. This work did not rerun and inspect
the complete labelled per-rock/phantom visual audit, so it does not by itself
establish Lance map-quality acceptance.

## Run

Command line replay:

```bash
rocklabel live \
  --play recordings/archive/misc/trimmed_lance.mcap \
  --carve --carve-assumed-pose-uncertainty 0 \
  --max-range 8 --floor-band -0.10 0.60 --web-ui
```

For conservative support-only behavior, omit
`--carve-assumed-pose-uncertainty 0`.

Dashboard:

```bash
rocklabel dash
```

Open **Live view**, choose the recording, enable **Ray carving
(experimental)** under Advanced, set max range and floor band as above, and
set **Assumed pose uncertainty** to `0` only for the explicit exact-replay
experiment. Launching from the dashboard automatically embeds the browser
control panel. The runtime Accumulation status reports groups, points, queue
high-water, latest mapping latency, and any producer waits.
