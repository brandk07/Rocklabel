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

from altslam.config import AltSlamConfig
from altslam.evaluate import accumulate, surface_sharpness
from altslam.solver import OfflineSolver
from rocklabel.live.recording import (
    SCHEMA_NAME,
    TOPIC,
    _METADATA_NAME,
    _SCHEMA_DOC,
    decode_frame,
    encode_frame,
)


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
) -> dict:
    """Re-solve ``src`` and write the result to ``dst``. Returns a report dict.

    With ``write=False`` the solve and the scoring still run but no file is
    produced, which is how ``--score-only`` tries settings out cheaply.
    """
    cfg = cfg or AltSlamConfig()
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
        old = surface_sharpness(accumulate(frames, stride=stride), up)
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
        md = read_metadata(src)
        md["reslam_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        md["reslam_source"] = os.path.basename(src)
        md["reslam_config_yaml"] = yaml.safe_dump(cfg.__dict__, sort_keys=False)
        write_frames(dst, frames, positions, quats, md)
    return report
