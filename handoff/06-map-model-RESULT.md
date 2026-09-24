# Result: 06-map-model

**Reading the accumulated map, with arena labels, is the first change in this
project that moves the arena map a lot.** On a cross-validated map of the
competition arena (every cell graded by a model that never saw a label on that
ground), three seeds averaged:

- 0.58 mean rock coverage at 100 wrongly-claimed cells, against 0.42 for the
  best earlier model; 0.55 against 0.32 at 50; 0.60 against 0.50 at 200.
- 42 wrongly-claimed cells at its default threshold, against 1,800-2,400 for
  every earlier model at its own.
- 29 ms a whole-arena pass on the GPU; 0.17 ms to add a sweep to the map.

What limits it and what did not work:

- **Needs arena labels.** Trained on volleyball alone it is worse than the
  earlier models (0.26 at 100 false cells).
- **It is the map, not the labels.** The single-sweep baseline given the same
  arena labels stays at 0.21-0.30 at 100 false cells.
- **Three atypical rocks** (a 60 cm box-like block, a nearly flat rock and a
  rock against a wall) are missed, so above about 400 false cells it only ties.
  Averaged with the 0.5 m + 30 s history classifier it is the best at every
  budget (0.71 against 0.63 at 500).
- Rock transplants, walls-as-clear, taller training rocks and test-time
  averaging did not help.

One arena, one session: a new arena will do worse. Next:

1. Label the 12 and 14 May arena recordings to get a second arena.
2. Put the map model on the robot, masked by the arena outline.

Full write-up: [training/reports/mapnet-v1/summary.md](../training/reports/mapnet-v1/summary.md).
