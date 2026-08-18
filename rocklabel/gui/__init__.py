"""Every Open3D window in the offline tool.

Kept together so that everything else stays headless and importable on a
machine with no display — the training stack, the dashboard and the generator
never touch this package.

* :mod:`~rocklabel.gui.viewer` — the labeling window itself: pick rocks on the
  fused cloud, size them, set the height band, draw the arena outline.
* :mod:`~rocklabel.gui.camera` — the CAD-style orbit/pan/zoom controls shared
  by every window here, so they all behave the same way.
* :mod:`~rocklabel.gui.labeler` — ``rocklabel label``: fuse the recording, then
  hand the cloud to the viewer. Also owns where a label file is written.
* :mod:`~rocklabel.gui.preview` — ``rocklabel preview``: step through the frames
  a dataset actually contains, rebuilt from the written files rather than the
  recording, so what you see is what training loads.
* :mod:`~rocklabel.gui.driftcheck` — ``rocklabel driftcheck``: overlay the start
  and end of a recording around one rock to see whether the odometry drifted.

Note this is *not* :mod:`rocklabel.live.viz`, which is the separate live-rig
viewer for the sensor running in real time.
"""
