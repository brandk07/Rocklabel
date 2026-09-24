# How the map model works, and how it was measured

Companion to [summary.md](summary.md). Code: `rocklabel/train/mapnet/`.

## The input: the map the robot already keeps

Every return in the rig's operational band (10 cm below to 60 cm above the
measured floor, within 8 m of the robot) goes into a 2.5 cm grid in the
levelled world frame. Each cell keeps a histogram of heights in 1 cm slices
plus the sum and maximum of the returns' brightness (scaled per recording, so
two sensors' raw numbers do not matter). Adding a sweep costs about 0.2 ms.
Nothing else is stored.

From that grid, 27 image channels are computed:

- whether the cell was seen, and how many returns it holds (capped at 64, so
  a 35-minute map and a 45-second one sit on one scale);
- the 10th, 50th and 90th percentile and top of its heights, each measured
  from a **local ground** estimate (a low percentile of the median-height
  surface over 1.5 m), because a crater or a sloping court moves the floor by
  more than a rock is tall;
- the spread of heights, and the local ground itself;
- mean and maximum brightness, and whether the sensor reports brightness;
- the **height profile**: the share of the cell's returns in each 2.5 cm
  slice from 5 cm below to 35 cm above the local ground. A rock's faces spread
  returns over many slices; a lump of soil or a noisy patch of floor piles
  them into one or two. Without these 16 channels the model was far behind.

A U-Net (four downsamplings, 32 to 512 channels, 7.8 million weights) reads
the channels and gives every 2.5 cm cell a rock probability. It sees about two
metres around each cell.

## Training

Crops of 4.8 m are cut from the map as it stood at a random moment of a
random recording, then rotated, mirrored, stretched (0.9-1.5x across, 0.85-1.4x
in height), thinned, laid over synthetic tilt and broad mounds or dips, and
given extra height noise. All of that happens to the points before the grid is
built, so every channel is rebuilt honestly. Crops are built on the graphics
card: about 25 seconds per 2,000-crop epoch.

Labels are the rock outlines: 1 inside, a 5 cm ignore shell outside, 0
elsewhere, ignored where nothing was seen or outside the arena. The loss is
cross-entropy (rock weighted 3x) plus an overlap (Dice) term. The saved
weights are a slow running average (decay 0.998): single epochs moved the
arena score by 0.1-0.2, the average does not.

## Grading: exactly as `mapeval`

The 2.5 cm probabilities are reduced to 10 cm ground cells by taking the
sub-cell the model is surest of; its representative point is its own measured
surface (90th-percentile height). The frontier, the budgets and the per-rock
table are then computed by `map_eval.threshold_frontier` itself, against the
visible-cell denominators of the `mapeval` geometry cache. Earlier models'
saved maps are re-graded by the same function, which reproduces every
published number exactly.

Both coverage measures are reported:

- **3D attribution**: the claimed cell's representative lies on the rock's own
  3D shape. This is the number the project quotes.
- **Occupied footprint**: the cell is claimed at all, which is what a
  navigation stack drives around.

The gap between them is mostly rock edges: a 10 cm cell that only touches a
rock's outline counts toward that rock, and the sub-cell chosen to represent
it often lies just outside. Every model pays that cost.

## The arena hold-outs

The arena is development data and its twelve rocks are the only arena labels
there are. To measure what arena labels are worth without grading a model on
ground it was trained on, the rocks are split into groups and every piece of
ground belongs to the group of its nearest rock. Rocks that touch are kept in
one group, so no outline is cut. A model trains on volleyball plus some groups
(with a 25 cm buffer at the boundary carrying no labels) and is graded only on
the ground it did not train on.

- Two-way: A = rocks 5, 8, 9, 10, 11, 12 (the rough, dug-over east side and
  rock 8); B = rocks 1, 3, 4, 6, 7, 13 (the smoother west and centre).
- Four-way: g1 = 9, 10, 11; g2 = 1, 3, 4, 13; g3 = 5, 8, 12; g4 = 6, 7.

A **stitched** map takes every cell's probability from the model that never
saw a label on that ground, and is graded on all twelve rocks like any other
map. It is a cross-validated estimate for this arena and this session, which
is optimistic about a new arena.
