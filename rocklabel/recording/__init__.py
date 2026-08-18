"""Getting points off disk: read a recording, decode it, put it in world frame.

Everything here answers "what did the sensor see, and where was it". Two
recording formats are supported and auto-detected, so nothing above this layer
ever has to ask which one it is holding:

* :mod:`~rocklabel.recording.mcap_io` — ROS 2 rosbag2 mcaps (PointCloud2 + /tf),
  decoded from the schemas embedded in the file. No ROS install needed.
* :mod:`~rocklabel.recording.lidarrig_io` — native lidarrig recordings
  (``/lidar/frames``), which carry their own poses.

:mod:`~rocklabel.recording.pose` buffers transforms and interpolates them to a
scan's timestamp. :mod:`~rocklabel.recording.pipeline` is the one everything
else actually imports: ``ScanStream`` glues decode → pose → odom-frame points
into an iterator, and ``WindowedScanStream`` merges scans into time windows
(what a generation profile's ``frame_window_s`` drives).

Two subcommands live here too, because both are recording surgery rather than
anything to do with rocks: ``rocklabel inspect`` (what topics and frames does
this file have) and ``rocklabel trim`` (cut it down, or salvage a truncated one).
"""
