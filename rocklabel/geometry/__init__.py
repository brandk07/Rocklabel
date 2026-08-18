"""Maths on a point cloud. No windows, no files, no rocks.

Deliberately headless and deliberately free of any notion of a label — these
are the operations that have to happen the same way for the labeler, the
generator and the drift check, or the three disagree about where a rock is:

* :mod:`~rocklabel.geometry.leveling` — measure a tilted sensor mount and
  rotate it out. Labels are stored in world coordinates, so labelling levelled
  and generating unlevelled misplaces every rock; this is the module that keeps
  all three commands on one geometry.
* :mod:`~rocklabel.geometry.accumulate` — a streaming voxel grid that fuses a
  whole recording into one cloud in bounded memory, however long it runs.
* :mod:`~rocklabel.geometry.relief` — height above the *local* ground rather
  than raw height, so a small rock on a slope stops hiding inside the terrain.
"""
