# `training/caches/` — the pooled point clouds training reads

One folder per **generation profile** — the named way a recording was cut into
frames. Written by `rocklabel-train cache`, read by everything that trains.

```
caches/<profile>/
  meta.json          which profile, which datasets, and the per-run counts
  <run_id>/*.npy     that recording's samples, ready to memory-map
```

## What's here now

| folder | built from | holds |
|---|---|---|
| `full-sweep/` | `datasets/full-sweep/volleyball` | 11 volleyball recordings, one whole sensor rotation per frame. **The one to train the sliding-window classifiers on.** |
| `full-sweep-dense/` | `datasets/full-sweep-dense/volleyball` | the same rotations with every frame kept instead of every fourth — 12 recordings (VolleyBallTest13 joins here), 10,822 frames and 410,954 samples against full-sweep's 2,700 and 94,519. **The one to train the whole-frame segmenter on**, which learns from frames and so was short of them. |
| `raw-burst/` | the 11 `datasets/raw-burst/*` folders | the same 11 recordings cut the old way, one raw ~4 ms sensor batch per frame. Kept only to reproduce old results. |
| `archive-comforter-fused/` | `datasets/archive/comforter-fused` | 8 Comforter recordings. Old project, kept for reference. |

## Why one cache per profile, and not one big one

A full-sweep frame holds about 1,250 points inside the crop box; a raw-burst
frame holds about 110. Samples cut from the two are not the same kind of thing,
so a model trained on a mixture would be learning from two different
populations and its score would describe neither. `rocklabel-train cache`
refuses to pool two profiles and names both in the error message.

## Why a cache exists at all

Training walks over roughly 75,000 samples every pass. Reading those from
thousands of separate compressed files each time would dominate the runtime, so
each recording gets concatenated once into plain `.npy` arrays that load
instantly.

## Safe to delete

Yes — entirely rebuildable from `datasets/` in a few minutes, which is why this
folder is not in git. Deleting it does not invalidate anything already trained:
a finished model carries its own settings.
