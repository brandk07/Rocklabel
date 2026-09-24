"""Rock detection on the accumulated map instead of on one sweep at a time.

Every earlier model in this project looks at a ball of points cut from the
newest sweep (or a few recent ones) and answers "is this a rock?". The robot,
though, is driving on a *map*: every return it has ever seen, registered by
SLAM. Stacked over a run, a rock is an unmistakable 15-30 cm bump with a
crisp outline, while the things single sweeps confuse with rocks - noisy
patches of floor, crater rims, the base of a wall - are either smooth or only
look bumpy in one sweep.

So this package keeps a 2.5 cm height histogram of everything seen so far
(:mod:`.grid`), turns it into a handful of image channels (:mod:`.features`),
and segments rocks with a small convolutional network (:mod:`.model`). It is
graded by the exact same frontier and per-rock tables as ``mapeval``
(:mod:`.grade`), against the same visible-cell denominators.
"""
