"""Read a recording, re-solve its trajectory, write a new recording.

The input file is opened read-only and never written to. The output is a fresh
MCAP in exactly the same ``lidarrig/Frame`` format, carrying the same points,
the same brightness values and the same IMU samples — only the pose attached to
each batch is replaced. That means every existing tool (``rocklabel label``,
``generate``, ``train``, ``live --play``) reads the new file with no changes at
all; they simply see a better-aligned world.
"""

from __future__ import annotations

import datetime as _dt
import os

import numpy as np
import yaml

from rocklabel.slam.config import AltSlamConfig
from rocklabel.slam.evaluate import accumulate, surface_sharpness
from rocklabel.slam.solver import OfflineSolver
from rocklabel.live.recording import (
    SCHEMA_NAME,
    TOPIC,
    _METADATA_NAME,
    _SCHEMA_DOC,
    decode_frame,
    encode_frame,
)


def load_ros2_frames(path: str, stride: int = 1, max_frames: int | None = None,
                     progress=None) -> list:
    """Read a ROS 2 bag (PointCloud2 + /tf) as solver-ready frames.

    Competition logs are rosbag2 MCAPs, not lidarrig's own ``/lidar/frames``,
    so the solver could not see them at all. This produces the same
    :class:`RecordedFrame` objects the rest of the pipeline expects: points in
    the **sensor frame**, plus the bag's own tf pose as the starting guess the
    solver refines.

    ``stride`` keeps every Nth scan. These bags run at ~19 Hz for half an hour,
    which is far denser in time than the solver needs and more than fits in
    memory at once; every 4th scan still leaves ~5 Hz.
    """
    from rocklabel.live.ros2bag import (TfTree, decode_pointcloud2,
                                        decode_tfmessage, find_lidar_topics)
    from rocklabel.live.recording import RecordedFrame
    from mcap.reader import make_reader

    frames: list = []
    with open(path, "rb") as fh:
        reader = make_reader(fh)
        summary = reader.get_summary()
        if summary is None:
            raise ValueError(f"{path}: no summary; run 'rocklabel trim' first")
        found = find_lidar_topics(summary)
        if found is None:
            raise ValueError(f"{path}: no PointCloud2 topics")
        cloud_topic, tf_topics = found
        total = 0
        if summary.statistics is not None:
            total = summary.statistics.channel_message_counts.get(
                next((cid for cid, ch in summary.channels.items()
                      if ch.topic == cloud_topic), -1), 0)
        tf = TfTree()
        seen = 0
        for _s, ch, m in reader.iter_messages(
            topics=[cloud_topic, *tf_topics]
        ):
            if ch.topic != cloud_topic:
                for parent, child, pos, quat in decode_tfmessage(m.data):
                    tf.update(parent, child, pos, quat)
                continue
            seen += 1
            if (seen - 1) % stride:
                continue
            cloud = decode_pointcloud2(m.data)
            pts = cloud.points
            keep = np.isfinite(pts).all(axis=1)
            # Zero-range rows are the driver's "no return" placeholder, not a
            # point at the sensor origin; they would anchor ICP onto nothing.
            keep &= np.linalg.norm(pts, axis=1) > 1e-3
            pts = pts[keep]
            if pts.shape[0] == 0:
                continue
            inten = cloud.intensity
            pos, quat = tf.pose(cloud.frame_id)
            frames.append(RecordedFrame(
                points=pts,
                intensity=None if inten is None else inten[keep],
                timestamp=cloud.stamp,
                # The solver wants a gravity-referenced orientation prior and
                # normally gets it from the rig's IMU, which these bags do not
                # carry. The robot's own tf rotation is the same thing: it is
                # reported in a z-up odometry frame, so it fixes which way is
                # down and leaves the solver to correct the rest.
                orientation=quat,
                pose_position=pos,
                pose_quat=quat,
                log_time_ns=m.log_time,
            ))
            if progress is not None and len(frames) % 200 == 0:
                progress("read", seen, total or seen)
            if max_frames is not None and len(frames) >= max_frames:
                break
        if progress is not None:
            progress("read", total or seen, total or seen)
    return frames


def is_ros2_bag(path: str) -> bool:
    """True when a recording carries PointCloud2 clouds instead of /lidar/frames."""
    from mcap.reader import make_reader
    from rocklabel.live.ros2bag import find_lidar_topics

    try:
        with open(path, "rb") as fh:
            summary = make_reader(fh).get_summary()
            if summary is None:
                return False
            topics = {ch.topic for ch in summary.channels.values()}
            if TOPIC in topics:
                return False
            return find_lidar_topics(summary) is not None
    except Exception:
        return False


def load_frames(path: str) -> list:
    """Decode every ``/lidar/frames`` message, tolerating a truncated tail."""
    from mcap.exceptions import McapError
    from mcap.reader import NonSeekingReader, make_reader

    frames = []
    with open(path, "rb") as fh:
        try:
            reader = make_reader(fh)
            summary = reader.get_summary()
            ok = bool(summary and summary.statistics and summary.statistics.message_count)
        except Exception:
            ok = False
        if ok:
            it = reader.iter_messages(topics=[TOPIC])
        else:  # never finalized: salvage what is readable
            fh.seek(0)
            it = NonSeekingReader(fh).iter_messages(topics=[TOPIC], log_time_order=False)
        while True:
            try:
                _s, _c, msg = next(it)
            except StopIteration:
                break
            except (McapError, EOFError, ValueError):
                break
            frames.append(decode_frame(msg.data, msg.log_time))
    return frames


def read_metadata(path: str) -> dict:
    """Return the ``lidarrig`` metadata block of a recording, or an empty dict."""
    from mcap.reader import make_reader

    try:
        with open(path, "rb") as fh:
            for md in make_reader(fh).iter_metadata():
                if md.name == _METADATA_NAME:
                    return dict(md.metadata)
    except Exception:
        pass
    return {}


def write_frames(path: str, frames, positions, quats, metadata: dict) -> None:
    """Write a new recording with replaced poses, preserving everything else."""
    from mcap.writer import Writer

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start(profile="x-lidarrig", library="lidarrig")
        if metadata:
            w.add_metadata(_METADATA_NAME, metadata)
        sid = w.register_schema(
            name=SCHEMA_NAME, encoding="x-lidarrig", data=_SCHEMA_DOC.encode()
        )
        cid = w.register_channel(
            topic=TOPIC, message_encoding="x-lidarrig-frame", schema_id=sid
        )
        for i, fr in enumerate(frames):
            data = encode_frame(
                fr.points, fr.intensity, fr.timestamp, fr.orientation,
                positions[i], quats[i],
            )
            w.add_message(
                channel_id=cid, log_time=fr.log_time_ns or i,
                publish_time=fr.log_time_ns or i, data=data, sequence=i,
            )
        w.finish()


def reprocess(
    src: str,
    dst: str,
    cfg: AltSlamConfig | None = None,
    progress=None,
    score: bool = True,
    write: bool = True,
    ros2_stride: int = 4,
) -> dict:
    """Re-solve ``src`` and write the result to ``dst``. Returns a report dict.

    With ``write=False`` the solve and the scoring still run but no file is
    produced, which is how ``--score-only`` tries settings out cheaply.
    """
    cfg = cfg or AltSlamConfig()
    ros2 = is_ros2_bag(src)
    if ros2:
        # Competition logs are rosbag2, not /lidar/frames. They also run for
        # half an hour at ~19 Hz, which is denser in time than the solver needs
        # and more than fits in memory, so only every Nth scan is read.
        frames = load_ros2_frames(src, stride=ros2_stride, progress=progress)
        if not frames:
            raise ValueError(f"no PointCloud2 messages in {src}")
    else:
        frames = load_frames(src)
        if not frames:
            raise ValueError(f"no {TOPIC} messages in {src}")

    solver = OfflineSolver(cfg)
    solver.build_windows(frames)
    stats = solver.solve(progress=progress)
    positions, quats = solver.batch_poses(frames)

    report = {
        "source": src,
        "output": dst,
        "batches": len(frames),
        "duration_s": float(frames[-1].timestamp - frames[0].timestamp),
        "windows": stats.windows,
        "registered": stats.registered,
        "failed": stats.failed,
        "match_ratio_median": stats.ratio_median,
        "residual_rmse_mm": stats.rmse_median * 1000.0,
        "suppressed_dirs_mean": stats.suppressed_mean,
        "path_length_m": stats.path_length,
    }

    if score:
        up = solver.up
        stride = max(1, len(frames) // 4000)
        # The two trajectories are expressed in different frames, so each must
        # be measured against its own up-vector. The solver works in a
        # sensor-at-startup frame (``solver.up``); a rosbag's own poses are
        # already in the robot's z-up odometry frame. Scoring both with one
        # up-vector tilts the ground plane under whichever is not in that
        # frame and reports a thickness that is mostly the tilt.
        old_up = np.array([0.0, 0.0, 1.0]) if ros2 else up
        old = surface_sharpness(accumulate(frames, stride=stride), old_up)
        new = surface_sharpness(
            accumulate(frames, positions, quats, stride=stride), up
        )
        report["sharpness_before_mm"] = old["median_mm"]
        report["sharpness_after_mm"] = new["median_mm"]
        report["sharpness_before_p90_mm"] = old["p90_mm"]
        report["sharpness_after_p90_mm"] = new["p90_mm"]
        if old["median_mm"] and not np.isnan(old["median_mm"]):
            report["improvement_x"] = old["median_mm"] / max(new["median_mm"], 1e-9)

    if write:
        # A re-solved rosbag is written as a native recording: the whole point
        # is that the pose now lives on each frame rather than in a tf tree.
        md = read_metadata(src)
        md["reslam_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        md["reslam_source"] = os.path.basename(src)
        md["reslam_config_yaml"] = yaml.safe_dump(cfg.__dict__, sort_keys=False)
        write_frames(dst, frames, positions, quats, md)
    return report
