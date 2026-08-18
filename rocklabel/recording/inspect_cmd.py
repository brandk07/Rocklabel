"""`rocklabel inspect`: discover topic names, frames, fields, and time span of an mcap."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .mcap_io import POINTCLOUD2, decode_pointcloud2, iter_decoded, read_info, structured_dtype
from .pose import build_pose_buffer


def _fmt_time(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_inspect(mcap_path: str, cfg: dict) -> None:
    info = read_info(mcap_path)

    print(f"=== {mcap_path} ===")
    print(f"time span: {_fmt_time(info.start_time_ns)} -> {_fmt_time(info.end_time_ns)}"
          f"  ({info.duration_s:.1f} s)\n")

    print("topics:")
    width = max((len(t) for t in info.topics), default=10)
    for topic, (schema, count) in sorted(info.topics.items()):
        print(f"  {topic:<{width}}  {count:>9} msgs  {schema}")

    if info.is_lidarrig:
        _inspect_lidarrig(mcap_path, info)
        return

    candidates = info.pointcloud_topics()
    configured = cfg["topics"]["pointcloud_topic"]
    print("\nPointCloud2 topics (frame / fields from each topic's first message):")
    for topic in sorted(candidates):
        for _t, _lt, msg in iter_decoded(mcap_path, [topic]):
            _xyz, _inten, has_int = decode_pointcloud2(msg)
            fields = ",".join(f.name for f in msg.fields)
            mark = "  <- configured" if topic == configured else ""
            print(f"  {topic}{mark}")
            print(f"      frame={msg.header.frame_id!r}  {msg.width}x{msg.height}  "
                  f"intensity={'yes' if has_int else 'NO'}  fields: {fields}")
            break
    if configured in candidates:
        chosen = configured
        print(f"configured topic {configured!r}: OK")
    elif candidates:
        chosen = sorted(candidates)[0]
        print(f"configured topic {configured!r} NOT in this file - set topics.pointcloud_topic "
              f"to one of the above. Showing detail for {chosen!r}:")
    else:
        chosen = None

    if chosen:
        for _t, _lt, msg in iter_decoded(mcap_path, [chosen]):
            dt = structured_dtype(msg.fields, int(msg.point_step), bool(msg.is_bigendian))
            print(f"\nfield layout of {chosen} (point_step={msg.point_step}, "
                  f"bigendian={bool(msg.is_bigendian)}):")
            for name in dt.names:
                print(f"  {name:<14} {dt.fields[name][0]}  offset {dt.fields[name][1]}")
            break

    _inspect_tf(mcap_path, cfg)


def _inspect_lidarrig(mcap_path: str, info) -> None:
    from . import lidarrig_io

    topic = info.lidarrig_topics()[0]
    print(f"\nformat: native lidarrig recording ({topic}) - poses are embedded per "
          "frame, no TF needed and the topics: config section is ignored.")

    n_probe = pts_total = 0
    has_inten = has_pose = False
    peak = 0.0
    for frame in lidarrig_io.iter_frames(mcap_path, topic):
        pts_total += len(frame.points)
        has_pose = has_pose or frame.has_pose
        if frame.intensity is not None:
            has_inten = True
            finite = frame.intensity[np.isfinite(frame.intensity)] if len(frame.intensity) else frame.intensity
            if len(finite):
                peak = max(peak, float(finite.max()))
        n_probe += 1
        if n_probe >= 50:
            break
    print(f"first {n_probe} frames: ~{pts_total // max(n_probe, 1)} points/frame, "
          f"reflectivity={'yes (peak %.0f)' % peak if has_inten else 'NO'}, "
          f"pose={'yes' if has_pose else 'NO (identity - SLAM never locked?)'}")

    config_yaml = lidarrig_io.read_embedded_config(mcap_path)
    if config_yaml:
        print("\nembedded lidarrig config:")
        for line in config_yaml.rstrip().splitlines():
            print("  " + line)


def _inspect_tf(mcap_path: str, cfg: dict) -> None:
    print("\nTF tree (parent -> child):")
    pb = build_pose_buffer(mcap_path, cfg["topics"], check_lidar_mount=False)
    for (parent, child), kind in sorted(pb.pairs().items()):
        print(f"  {parent} -> {child}   [{kind}]")
    needed = (cfg["topics"]["odom_frame"], cfg["topics"]["base_frame"], cfg["topics"]["lidar_frame"])
    missing = [f for f in needed if f not in pb.frames()]
    if missing:
        print(f"\nWARNING: configured frames not seen in TF: {missing}")
    else:
        print(f"\nconfigured frames {needed}: all present")
