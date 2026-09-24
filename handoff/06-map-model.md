# 06 — Read the map, not the sweep

September 2026. Written at the start of the work; results go in
`06-map-model-RESULT.md` and `training/reports/mapnet-v1/`.

## Why a new direction

Every model this project has trained answers one question: is this 0.5 m ball
of points, cut from the newest sweep, a rock? Twelve months of work on that
question (clutter, stray returns, grids, segmenters, synthetic worlds, history,
neighbourhood size) have moved the arena map by a few hundredths. The best
setting ever measured covers 0.63 of each rock at 500 wrongly-claimed cells,
0.51 at 200 and 0.42 at 100. At its stored threshold the typical checkpoint
claims about 2,000 false cells, against about 380 real rock cells.

Stacking the whole recording shows why. In the accumulated arena every
labelled rock is an unmistakable 15-30 cm bump with a crisp outline. The false
ground piles up where a single sweep looks bumpy but the map does not: the
rough, dug-over right half, the crater, and the base of the walls. The one
thing that helped last time, giving the ball older sweeps, was a small step in
this direction; it gained most at range, where one sweep is sparse.

So this phase takes the step all the way. The robot already keeps a map. Keep
a 2.5 cm height histogram of every return seen so far, turn it into image
channels (heights above a local ground estimate, how returns are spread with
height, density, brightness), and segment rocks with a small U-Net that sees
about two metres around each cell.

## What is measured, and how

Graded by `mapeval`'s own frontier and per-rock code, against the same
visible-cell denominators, so every number sits beside the old table:
3D-attributed coverage at 1,000 / 500 / 200 / 100 false footprint cells.

Two kinds of evaluation, never mixed:

1. **Volleyball only → arena.** The same training data every earlier model
   had. The fair comparison.
2. **Arena spatial hold-out.** The arena's twelve rocks are split into two
   groups of six, and each group owns the ground nearest to it (rocks that
   touch stay together; a 25 cm buffer at the line carries no labels).
   Train on volleyball plus one group, grade on the other group's ground only.
   Earlier models are re-graded on exactly the same ground. This measures what
   in-domain labels are worth. It is still one arena and one session, so it
   is optimistic about a new arena.

Also: the map graded as it grew (every 30 s, only returns already measured),
scoring time, and pictures of every result.

## Ideas in the queue

- Height-profile channels (share of returns per 2.5 cm slice above ground).
- Rock transplants: paste real rocks' returns onto other real floors.
- Rotation/mirror averaging at grading time; seed ensembles.
- Unlabelled arena recordings (12 and 14 May) as a sanity check.
