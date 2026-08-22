# Agent 3 result — the next training suite

**Read this, not [03-training.md](03-training.md), for what is actually true.**
That brief made several predictions; two of them turned out to be wrong, and
saying so is more useful than quietly working around them.

Written while the sweep is still running. The section marked **LIVE** is
updated as arms finish; everything else is final.

---

## 1. The two old data problems, re-measured

The brief asked for this before designing anything, and it was right to. Both
problems were measured on the **raw-burst** datasets and both are now fixed —
but only one of them was fixed by the full-sweep change alone.

### Rocks the model never sees

On raw bursts, **12 of 63 labelled rocks produced no training samples at all**.
A candidate needs 20 returns inside a half-metre ball to become a sample, and
on sparse frames some rocks never got there in any frame. They were labelled,
they looked fine in the manifest, and the model was never shown them.

| dataset | rocks with no samples |
|---|---|
| raw-burst (old) | 12 of 63 |
| full-sweep (the current default) | **2 of 63** — VolleyBallTest6 rocks 4 and 6 |
| full-sweep-dense (new, see §3) | **0 of 69** |

So the full-sweep change fixed ten of the twelve, and keeping every frame
instead of every fourth fixes the last two. Both of VB6's silent rocks *do*
appear in the whole-frame segmentation data even on full-sweep — 145 and 59
points — so the segmenter has always been able to see rocks the sliding-window
classifier gets no sample for at all. That is a point in the segmenter's favour
that nobody had noticed.

### How far away a rock can be and still be learned from

On raw bursts, 88% of rock samples sat within 2 m and the furthest was 2.76 m,
against a 6 m crop. On full-sweep data:

| | raw-burst (old) | full-sweep |
|---|---|---|
| median rock-sample range | — | **2.07 m** |
| 95% of rock samples within | — | **3.00 m** |
| furthest rock sample | 2.76 m | **5.18 m** |
| share within 2 m | 88% | **45%** |

**Fixed.** Rocks are now learned from out to about 5 m, which covers the useful
part of the crop box (6 m forward, 4 m sideways, so the far corners are past
where the sensor returns anything useful anyway). This is no longer a cap on
anything.

### This check is now a command, not a one-off script

`rocklabel coverage <dataset>` prints both tables in a few seconds, plus which
labels sit too close together to tell apart. **There is a new "Rock coverage
check" card on the dashboard's Dataset stage** — run it after every Generate.
It is the only thing that catches a labelled rock being silently absent from
the training data.

## 2. VolleyBallTest4 — what is actually wrong with it

Agent 2 found the big one: **rock 8 is 46% of VB4's rock samples and the model
is confident it is not a rock** — it covers 47 cm of ground but stands only
6.5 cm proud of sand that is itself 5.1 cm rough. Removing that one label takes
VB4 from 0.355 to 0.583.

I found the second thing, and it explains why VB4's per-rock numbers were hard
to read at all:

**VB4's rocks are touching each other.** Five pairs of its labels sit closer
together than their own radii — rocks 3, 4, 5 and 6 form a chain 21 to 29 cm
apart while each is about 47 cm across. No other recording has more than three
such pairs. The "which rock did the model miss" attribution assigns each sample
to its nearest label, which cannot separate two labels 21 cm apart, so VB4's
per-rock recall figures split one pile of samples between several rocks.

| run | rocks | overlapping label pairs |
|---|---|---|
| **VB4** | 7 | **5** — 2-3, 3-4, 4-5, 4-6, 5-6 |
| VB11 | 7 | 4 |
| VB5 | 5 | 3 |
| VB9 | 7 | 2 |
| VB8 | 4 | 2 |
| all others | — | none |

VB4's rocks are **not** unusually small — mean label size 46.6 cm, above the
project average of 44 cm. So the brief's guess ("check whether they are
unusually small, flat, or clustered") was right on clustered, wrong on small,
and agent 2 had already nailed flat for the one rock that matters.

`rocklabel coverage` reports these overlapping pairs now, so this stops being
something you have to go looking for.

## 3. What changed in the code

### The segmenter's three levels are settings now

They were written into the model class and unreachable without editing it. The
built-in geometry keeps 512, 128 and 32 points at its three zoom-out levels and
pools over balls of 0.25, 0.6 and 1.4 m. Two things look wrong with that for
this problem: it throws away three quarters of the frame at the very first
level, and **its finest ball is half a metre across (a 0.25 m radius) while the
labelled rocks measure 21-68 cm, mean 44 cm** — the smallest scale it ever
looked at already swallowed the whole object it was hunting. (The brief said
the rocks were 20-30 cm. Measured off the label file they are not; the
conclusion survives the correction, the number does not.)

`--seg-npoints` and `--seg-radii` set them, on Train, Compare and the Ablation
sweep, **with matching controls on all three dashboard cards** (under Advanced).
Left blank they keep exactly what every segmentation run so far used, so old
checkpoints still load and finished folds still resume rather than reading as a
settings change nobody made.

### A generation profile that stops starving the segmenter

`full-sweep-dense`: the same 0.05 s sensor rotations, but every one is kept
instead of every fourth, at a 1,280-point-per-frame budget instead of 2,048.
It is on the dashboard's Generate dropdown with the others.

|  | full-sweep | full-sweep-dense |
|---|---|---|
| recordings | 11 | **12** (VolleyBallTest13 joins) |
| frames | ~2,700 | **10,822** |
| classifier samples | 94,519 | **410,954** |
| points stored per frame | 2,048 (43% of it padding) | 1,280 |
| silent rocks | 2 of 63 | **0 of 69** |

The point budget drop costs nothing: the median frame held 1,145 real points
and only 9 of 2,381 frames ever hit the old 2,048 cap.

**VolleyBallTest13 is a twelfth fold now.** Agent 2 spotted it had everything a
dataset needs and had simply never been generated. It has 6 rocks and adds 9%
to the evaluation set. It also means the new suite's folds are not a
fold-for-fold match with the old one — compare the eleven shared folds when
putting the two side by side.

### Every suite is now tied to the cache it was defined against

Three suites now exist and they train on three different caches. Pointing one
at another's cache produces a table of numbers that looks perfectly fine and
answers nothing, so it is refused with a message naming both. Leaving
`--cache-dir` alone picks the right one automatically.

### Reports print raw and prevalence-normalized scores side by side

Every per-fold table appears twice: raw PR-AUC, then
`(PR-AUC - rock share) / (1 - rock share)` where guessing is 0 and perfect is 1.
The rock shares themselves are printed underneath so the gap between the two
tables is explicable. Both finished sweeps get the new column for free — it is
computed from two numbers every fold already stored.

**But the brief overstated what this changes, and that is worth correcting.**
It predicted the normalization would "change which recordings look hard". On
the current full-sweep data it does not, much:

| fold | rock share | raw | normalized | rank raw → normalized |
|---|---|---|---|---|
| VB4 | 19.8% | 0.483 | 0.355 | 1 → 1 |
| VB12 | 28.8% | 0.634 | 0.486 | 3 → 2 |
| VB6 | 6.3% | 0.544 | 0.513 | 2 → 3 |
| VB5 | 12.0% | 0.665 | 0.619 | 4 → 4 |
| VB11 | 31.6% | 0.839 | 0.765 | 7 → 6 |
| VB10 | 8.2% | 0.811 | 0.794 | 6 → 7 |
| VB3 | 24.4% | 0.927 | 0.904 | 11 → 11 |

Three adjacent pairs swap. VB4 is the hardest fold either way, and the hardest
and easiest ends of the table do not move at all. For the **segmenter** it
changes nothing whatsoever — every fold is about 1% rock, so there is no spread
to divide out. The reordered table quoted in the handoff README came from the
older raw-burst numbers.

The column is still the right thing to report — it is honest and it costs
nothing — but it is not the lever the brief thought it was.

## 4. Two things about running this hardware

**Never run two trainings on this GPU at once.** I tried it, expecting the card
to be under-used at 8 frames per step. Both jobs slowed by about **nine times**
— far worse than sharing. The sweep queue is strictly sequential now.

**The segmentation training was badly under-batched.** The batch setting is
divided by 32 for whole-frame training, so the default 256 meant 8 frames per
optimizer step. Raising it to 512 (16 frames) makes an epoch **1.5x faster** at
identical settings, and 1024 makes it about 2x. That speed-up is what pays for
running twice as many epochs. The new segmentation arms use 512.

## 5. LIVE — the sweep

Order is priority order, so an interrupted sweep still leaves the headline
finished. Finished folds are skipped and every fold resumes from its last
checkpoint, so this can be stopped and restarted freely:

```
/tmp/.../run_queue.sh          # or, one arm at a time:
rocklabel-train ablate --suite segdense  --arms seg-long
rocklabel-train ablate --suite fullsweep --arms pointnet2seg-geom-e80
rocklabel-train ablate --suite segdense  --arms seg-fine
rocklabel-train ablate --suite segdense  --arms pointnet-geom
rocklabel-train ablate --suite segdense  --arms seg-long-s43
```

| arm | asks | folds | measured cost |
|---|---|---|---|
| `segdense/seg-long` | 4x the frames + 60 epochs instead of 30 | 12 | 45 s/epoch → ~45 min/fold, **~9 h** |
| `fullsweep/pointnet2seg-geom-e80` | epochs *alone*, on the old data, so the two effects can be told apart | 11 | 17 s/epoch → ~22 min/fold, **~4 h** |
| `segdense/seg-fine` | a 0.10 m finest radius instead of 0.25 m | 12 | ~1.9x seg-long, **~17 h** |
| `segdense/pointnet-geom` | the classifier partner for the matched comparison | 12 | **~11 h** |
| `segdense/seg-long-s43` | the noise floor at the new settings | 12 | **~9 h** |

That is about fifty hours in total, so **expect this to run for days, not
overnight** — which is fine, because it resumes. The order matters: the first
two arms between them answer the headline question, and the third is the
expensive gamble.

**The headline does not need the classifier arm.** "Did more frames and more
epochs help the segmenter?" is answered by comparing `segdense/seg-long`
against `fullsweep/pointnet2seg-geom` — same task, same unit, same eleven
shared folds. Only "does segmentation now *beat* the classifier?" needs the
fourth arm.

### `seg-long` is finished, and the answer is no

Four times the frames plus 60 epochs instead of 30, paired fold by fold against
the old 30-epoch arm on the eleven shared recordings:

| fold | old 30ep, every 4th frame | new 60ep, every frame | change |
|---|---|---|---|
| VB10 | 0.4898 | 0.3903 | -0.0995 |
| VB11 | 0.6269 | 0.5484 | -0.0785 |
| VB12 | 0.2581 | 0.2526 | -0.0055 |
| VB2  | 0.4193 | 0.5106 | +0.0913 |
| VB3  | 0.6047 | 0.6236 | +0.0189 |
| VB4  | 0.1498 | 0.1461 | -0.0037 |
| VB5  | 0.2137 | 0.1781 | -0.0356 |
| VB6  | 0.4463 | 0.3091 | -0.1372 |
| VB7  | 0.2875 | 0.3683 | +0.0808 |
| VB8  | 0.3158 | 0.3745 | +0.0587 |
| VB9  | 0.5293 | 0.5497 | +0.0204 |
| VB13 | — | 0.1331 | new fold, no partner |

**Mean -0.0081, median -0.0037, the new arm wins 5 of 11, p = 0.83.** The
average change is less than half the 0.0207 noise floor. This is a clean
negative: **the segmenter was not starved of frames.**

Individual folds swing by ±0.09 to ±0.14 in both directions, which is the
scatter this data always has and the reason the brief insists on pairing. The
two-fold panic recorded in earlier drafts of this file was that scatter, not a
trend.

Two things worth keeping from it:

* **It still did not early-stop.** Best epoch 53 of 60 on the first fold. Even
  at 60 epochs on four times the frames the model was still improving on
  validation — while gaining nothing on a recording it had never seen. That is
  the distinction the next arm exists to pin down.
* **Validation went up while test went down.** On VB10, validation PR-AUC rose
  0.441 -> 0.601 while the held-out score fell 0.490 -> 0.390. Consecutive
  frames are 0.05 s apart and overlap almost completely, so keeping every one
  is closer to four times the repetition than four times the data — and the
  validation block is a tail slice of recordings the model has already seen.
  **Do not trust the validation curve to say whether a change helps here.**

**VB4 did not move** (0.1498 -> 0.1461, inside the noise). Four times the data
and twice the training does nothing for it, which is what you would expect if
its problem is a bad label rather than a shortage of data — agent 2's finding
survives this test.

**VolleyBallTest13, evaluated for the first time, is the hardest fold in the
set**: 0.1331 raw, 0.1209 normalized, against VB4's 0.1461. A recording nobody
had ever scored on is the one the segmenter handles worst.

### `pointnet2seg-geom-e80` is finished — training longer does help, a little

Identical to the 30-epoch segmentation arm in every way except the epoch cap
and a matching patience. Same cache, same frames, same seed. Eleven folds:

| fold | 30 epochs | 80 epochs | change |
|---|---|---|---|
| VB10 | 0.4898 | 0.5012 | +0.0114 |
| VB11 | 0.6269 | 0.6513 | +0.0244 |
| VB12 | 0.2581 | 0.2769 | +0.0188 |
| VB2  | 0.4193 | 0.4635 | +0.0441 |
| VB3  | 0.6047 | 0.6247 | +0.0201 |
| VB4  | 0.1498 | 0.1408 | -0.0090 |
| VB5  | 0.2137 | 0.2490 | +0.0353 |
| VB6  | 0.4463 | 0.4130 | -0.0333 |
| VB7  | 0.2875 | 0.3451 | +0.0576 |
| VB8  | 0.3158 | 0.3382 | +0.0224 |
| VB9  | 0.5293 | 0.5490 | +0.0197 |

**Mean +0.0192, median +0.0201, wins 9 of 11, p = 0.032.** Significant, and the
mean sits just under the 0.0207 noise floor — so the honest phrasing is **real
but small**: consistent enough to believe, not large enough to change what the
model can do. Nine of eleven folds land in a tight +0.011 to +0.058 band.

**It still has not converged at 80 epochs.** Best epochs ran 61 to 79 of 80, and
5 of 11 folds peaked in the final five. Doubling the cap again would probably
buy a little more, at the same shrinking rate.

### Putting the two arms together

The brief's section 1 and section 2 were one recommendation ("do them together,
they pull the same way"). Split apart, they pull differently:

| change | mean | wins | p | verdict |
|---|---|---|---|---|
| 30 -> 80 epochs, same frames | **+0.0192** | 9/11 | **0.032** | real but small |
| 4x the frames + 60 epochs | -0.0081 | 5/11 | 0.83 | nothing |

**Train longer. Do not keep more frames.** Frames 0.05 s apart overlap almost
completely, so four times the frames is four times the repetition — it costs
four times the GPU time and returns nothing. That is why the combined arm
scored *worse* than the epochs-only arm despite doing more of everything: the
extra frames spent the epoch budget on repeats.

**VB4 and VB6 are the two folds that lose in both arms.** VB4 is immune to every
training change tried today, which is what a bad label looks like rather than a
training problem. Both are the recordings agent 2 named.

### The queue was trimmed and re-pointed (Brandon's call, 2026-08-20)

`seg-long` and `pointnet2seg-geom-e80` between them answered the brief, and
they disagreed with each other, so the remaining queue was rebuilt around the
answer rather than the original plan.

**Stopped:**

* `segdense/seg-fine` after 4 of 12 folds. The finer-levels idea has nothing to
  do with how the frames were cut, and on the dense cache it costs about three
  times as much per fold for the same answer. **The four finished folds are
  kept on disk, not deleted** — they are what says the finer levels are worth
  testing at all.
* `segdense/pointnet-geom` (~11 h), never started. It only existed to give the
  matched comparison a partner on the dense cache; the `fullsweep` suite
  already has finished classifier arms that do the job.
* `segdense/seg-long-s43` (~9 h), never started. A seed repeat measuring the
  noise floor of a setting now known to do nothing.

**Started instead:** `fullsweep/pointnet2seg-geom-e80-fine` — 80 epochs *and*
the finer levels, on the ordinary full-sweep cache. It is identical to
`pointnet2seg-geom-e80` in every other respect, so the gap between them is the
level geometry and nothing else, and it pairs directly against both finished
segmentation arms on the same eleven folds.

That is the combination of the two changes that measured as worth having,
without the one that did not.

### What the four dense finer-levels folds showed before they were stopped

| fold | 30ep baseline | seg-long (stock levels) | seg-fine (finer levels) | vs stock | vs baseline |
|---|---|---|---|---|---|
| VB10 | 0.4898 | 0.3903 | 0.4337 | +0.0434 | -0.0561 |
| VB11 | 0.6269 | 0.5484 | 0.6302 | +0.0818 | +0.0033 |
| VB12 | 0.2581 | 0.2526 | 0.2998 | +0.0472 | +0.0417 |

**Finer levels beat the stock geometry on all three shared folds, mean +0.057**,
every fold in a tight +0.043 to +0.082 band — nearly three times the noise
floor. Three folds cannot be significant (the smallest p available is 0.25), but
the consistency is the same signature the epochs result had, and the opposite of
the dense-frames result's wild scatter.

Against the ordinary baseline the same arm is level (-0.004): the finer levels
are paying back the dense cache's penalty rather than adding on top of it. Which
is exactly why the question is worth re-asking without that penalty.

### `pointnet2seg-geom-e80-fine` is finished — finer levels do not help

80 epochs *and* the finer levels (1,024/256/64 centroids at 0.10/0.3/0.8 m), on
the ordinary full-sweep cache, against the 80-epoch arm that differs only in the
level geometry:

| fold | 30ep | 80ep | 80ep + finer | finer vs 80ep | finer vs 30ep |
|---|---|---|---|---|---|
| VB10 | 0.4898 | 0.5012 | 0.4352 | -0.0659 | -0.0545 |
| VB11 | 0.6269 | 0.6513 | 0.6369 | -0.0144 | +0.0100 |
| VB12 | 0.2581 | 0.2769 | 0.2740 | -0.0029 | +0.0159 |
| VB2  | 0.4193 | 0.4635 | 0.4486 | -0.0149 | +0.0292 |
| VB3  | 0.6047 | 0.6247 | 0.6083 | -0.0165 | +0.0036 |
| VB4  | 0.1498 | 0.1408 | 0.1392 | -0.0016 | -0.0105 |
| VB5  | 0.2137 | 0.2490 | 0.1986 | -0.0504 | -0.0151 |
| VB6  | 0.4463 | 0.4130 | 0.4688 | +0.0558 | +0.0225 |
| VB7  | 0.2875 | 0.3451 | 0.3167 | -0.0285 | +0.0292 |
| VB8  | 0.3158 | 0.3382 | 0.3439 | +0.0057 | +0.0281 |
| VB9  | 0.5293 | 0.5490 | 0.5682 | +0.0192 | +0.0389 |

**Finer levels vs stock, both at 80 epochs: mean -0.0104, wins 3 of 11,
p = 0.278.** Not a gain. If anything a small loss, though inside the noise floor.

**The +0.057 seen on the dense cache was the stock geometry doing badly there,
not the finer geometry doing well.** Both readings fit together once put on one
line: against the ordinary 30-epoch baseline, the finer geometry scores about
level on the dense cache (-0.004) and about level on the full-sweep cache
(+0.009). It is the *stock* geometry that fell apart on the dense frames. Three
consistent folds looked convincing and were not — which is the same lesson the
brief's own "two traps" section is about, arriving a third time.

One real difference: with the finer levels only **2 of 11 folds peaked in the
final five epochs**, against 5 of 11 for the stock geometry. The finer model
converges sooner. It just converges to the same place.

### The answer to the whole brief

| change | mean | wins | p | verdict |
|---|---|---|---|---|
| 30 -> 80 epochs | **+0.0192** | 9/11 | **0.032** | **real but small — keep it** |
| finer levels (at 80 epochs) | -0.0104 | 3/11 | 0.278 | no |
| 4x the frames (+60 epochs) | -0.0081 | 5/11 | 0.83 | no, at 4x the cost |
| 80 epochs + finer levels together | +0.0088 | 8/11 | 0.206 | worse than epochs alone |

**The best segmentation setting found is: the ordinary `full-sweep` data, stock
level geometry, 80 epochs.** That is `fullsweep/pointnet2seg-geom-e80`, and it
is the arm to export from. Everything else on the brief's list was tested and
came back empty.

Of the brief's four headroom items, one paid (§1, epochs), two did not (§2
frames, §3 geometry), and §5 and §6 were data questions that turned out to
matter more than any of them — see sections 1 and 2 of this document.

**Which puts the remaining headroom back where agent 2 left it: the labels.**
Two bad labels are worth +0.23 and +0.14 on the two worst folds. Nothing tried
here comes close, and VB4 was immune to every single training change.

**Results so far: see [training/reports/segdense/summary.md](../training/reports/segdense/summary.md)
and [training/reports/fullsweep/summary.md](../training/reports/fullsweep/summary.md).**
Both rebuild from whatever has finished with
`rocklabel-train ablate --suite <name> --report-only`, or the "Report only"
tick-box on the dashboard's Ablation card.

## 6. What I did not do

- **No matched classifier-vs-segmenter comparison yet.** It needs the classifier
  arm on the dense cache, which is fourth in the queue. Until that finishes the
  only honest classifier-vs-segmenter statement is the existing one from the
  full-sweep suite (they tie).
- **No `double-sweep` (longer window) arm.** It was fifth on the brief's list
  and last in value; the queue above is already about forty hours of GPU.
- **I did not touch VB4 rock 8 or VB6 rock 8.** Agent 2 asked Brandon to look at
  them, and deleting a label is his call, not mine. It remains the single
  highest-value thing available: worth more than every training change here put
  together.

## Tests

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/` → **556 passed** (was 533).
The 23 new tests cover the normalized score (including that a
fold can lead on raw and trail once its rock share is removed), the segmenter's
level geometry and its validation, old configs still resuming, the suite-to-cache
binding, and the three things the coverage check exists to catch.
