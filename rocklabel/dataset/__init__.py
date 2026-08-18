"""Turning a labeled recording into something a model can train on.

:mod:`~rocklabel.dataset.generate` is the command (``rocklabel generate``). It
replays a recording, crops a box around the robot, projects the hand-placed
label shapes into every kept frame, and writes all three dataset formats plus a
manifest recording exactly which settings produced them.

The three formats, each in its own module:

* :mod:`~rocklabel.dataset.neighborhoods` — **format A**, one sample per
  candidate ball of points. What the sliding-window classifiers train on.
* :mod:`~rocklabel.dataset.bev` — **format B**, a bird's-eye-view raster with a
  label per cell.
* :mod:`~rocklabel.dataset.neighborhoods` also builds **format C**, whole-frame
  per-point segmentation samples (``build_segmentation_frame``).

:mod:`~rocklabel.dataset.labeling` is what all three stand on: it turns the
label shapes (spheres, boxes, extruded polygons) into a per-point verdict —
rock, clear, or ignore for the shell around a shape's edge.

How the frames are *cut* is not decided here: that is a generation profile
(:mod:`rocklabel.profiles`), and the choice is recorded in the manifest.
"""
