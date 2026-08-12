"""rocklabel.live — the live LiDAR rig (ported from the lidarrig project).

Live ingest -> SLAM/IMU pose -> 2.5D fusion -> Open3D visualization, plus
MCAP recording of the raw stream. Recordings land in the native
``/lidar/frames`` format that every offline rocklabel command
(label / generate / train / replay) already reads.

Three decoupled layers:

* :mod:`rocklabel.live.sources`  — data sources yielding ``(N, 3)`` point batches.
* :mod:`rocklabel.live.surfaces` — incremental 2.5D surface builders (fusion).
* :mod:`rocklabel.live.viz`      — the Open3D real-time visualizer / app loop.

:mod:`rocklabel.live.scoring` adds live PointNet inference: a background
thread that scores the accumulated world-frame cloud with a trained
checkpoint so the viewer can color points by rock probability.
"""

__version__ = "0.1.0"
