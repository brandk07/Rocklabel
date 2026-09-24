# Result: 05-pointnet-neighborhood-history

**History helps. The adaptive radius does not.** A fixed 0.75 m ball fed the
current sweep plus sweeps 2, 4, 6 and 8 seconds old (`fixed-r075__h5-8s`)
is the only setting that improved the Lance map on all three seeds at equal
false area. Against the 0.5 m single-sweep baseline trained with the same seed:

- +0.104 3D rock coverage at 500 false cells (+0.081 to +0.118 across seeds);
- 412 fewer false cells at its stored threshold, for the same coverage.

It is not deployable yet, for three reasons:

- **Scoring cost:** 2.6x (310 ms median and 493 ms at the 95th percentile,
  against 120 and 164 ms), all of it CPU ball cutting.
- **Weakest rock:** it does not reliably improve. Rock 12 loses 12-18 points
  on every seed.
- **Evidence:** it rests on Lance, which is development data.

The 0.5 m ball with 30 s of history gains as much at equal area and is cheaper
(182 ms median and 267 ms at the 95th percentile), but it barely reduces false
ground at its own threshold.

Did not help: adaptive radius with K = 256, 0.2/0.3 m balls, a 0.75 m ball
without history, and the query-height and point-age inputs. Volleyball
validation ranked the settings differently from Lance again. Startup behaviour
cannot be measured on Lance: one rock is visible in the first 46 s.

An independent review's six defects are fixed and tested, including a
project-wide bug in the equal-false-area table, which credited only rock 1.
All saved frontiers have been recomputed. That changes one CLAUDE.md claim:
the incumbent ties `cls-both-s44` at 1,000 cells and trails it at 500.

Next:

1. Retrain `fixed-r075__h5-8s` on all eleven recordings and `mapeval` it
   against the incumbent.
2. Speed up ball cutting.
3. Look at rock 12.
4. Record a fresh arena run.

Full write-up: [summary.md](summary.md). Tables: [results.md](results.md),
[comparison.csv](comparison.csv), [latency.csv](latency.csv),
[strata.csv](strata.csv), [per-rock.csv](per-rock.csv).
