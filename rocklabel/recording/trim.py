"""`rocklabel trim`: cut a recording down to the topics and time range that matter.

Copies messages raw (no decoding) into a new, properly indexed mcap, reading
the input sequentially. This makes it double as a recovery tool: a truncated /
unfinalized recording (recorder killed mid-write, missing footer) is read up
to the first corrupt byte and everything before it is salvaged into a valid
output file.

By default only the pointcloud topic plus /tf and /tf_static are kept — for a
recording that also carries cameras etc. this typically shrinks the file by
an order of magnitude. --start-s/--end-s crop time as seconds relative to the
first message.

Poses need a little slack around the cut so lookups at the very edges of the
window still bracket a transform, so /tf is kept for ``TF_PAD_S`` seconds
either side of the window. It is only slack, though: keeping /tf for the whole
recording (as this used to) leaves the output file *advertising* the original
time span while holding clouds for a small slice of it. Players trust that
span, so a bag trimmed to start at 150 s would sit at 0:00 doing nothing until
you scrubbed forward to 150 s. The single latched /tf_static message is kept
whatever its timestamp and restamped to the start of the window, for the same
reason — a static transform is valid for all time, so moving it is safe.
"""

from __future__ import annotations

import os
from collections import Counter

from mcap.records import Channel, Message, Schema
from mcap.stream_reader import StreamReader
from mcap.writer import Writer as McapWriter
from tqdm import tqdm

from .. import __version__

#: Seconds of /tf kept either side of the time window, so a pose lookup at the
#: very edge of the window still has transforms bracketing it. At the ~38 Hz
#: these recordings publish /tf, this is ~75 messages of slack per side.
TF_PAD_S = 2.0


def run_trim(in_path: str, out_path: str, cfg: dict, extra_topics: list[str] | None = None,
             start_s: float | None = None, end_s: float | None = None,
             all_topics: bool = False) -> None:
    if os.path.abspath(in_path) == os.path.abspath(out_path):
        raise SystemExit("trim: --out must differ from the input file")
    topics_cfg = cfg["topics"]
    tf_topic = topics_cfg["tf_topic"]
    tf_static_topic = topics_cfg["tf_static_topic"]
    tf_topics = {tf_topic, tf_static_topic}
    keep: set[str] | None = None
    if not all_topics:
        from .lidarrig_io import TOPIC as LIDARRIG_TOPIC
        keep = {topics_cfg["pointcloud_topic"], LIDARRIG_TOPIC} | tf_topics | set(extra_topics or ())

    schemas: dict[int, Schema] = {}
    channels: dict[int, Channel] = {}
    new_schema_ids: dict[int, int] = {}
    new_channel_ids: dict[int, int] = {}
    counts: Counter = Counter()
    t0_ns: int | None = None
    kept_range: list[int] = []
    read_error: Exception | None = None

    total = os.path.getsize(in_path)
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        writer = McapWriter(fout)
        writer.start(profile="ros2", library=f"rocklabel {__version__}")
        records = StreamReader(fin, emit_chunks=False).records
        progress = tqdm(total=total, unit="B", unit_scale=True, desc="trim")
        pos = 0
        while True:
            try:
                rec = next(records)
            except StopIteration:
                break
            except Exception as e:  # truncated/corrupt input: salvage what we have
                read_error = e
                break
            here = fin.tell()
            if here > pos:
                progress.update(here - pos)
                pos = here
            if isinstance(rec, Schema):
                schemas[rec.id] = rec
            elif isinstance(rec, Channel):
                channels[rec.id] = rec
            elif isinstance(rec, Message):
                if t0_ns is None:
                    t0_ns = rec.log_time
                channel = channels.get(rec.channel_id)
                if channel is None:
                    continue
                if keep is not None and channel.topic not in keep:
                    continue
                # Time window. /tf gets TF_PAD_S of slack either side so edge
                # lookups still bracket; /tf_static is latched (one message,
                # valid for all time) so it is kept and restamped below.
                log_time = rec.log_time
                if channel.topic != tf_static_topic:
                    pad = TF_PAD_S if channel.topic == tf_topic else 0.0
                    t_rel = (log_time - t0_ns) / 1e9
                    if start_s is not None and t_rel < start_s - pad:
                        continue
                    if end_s is not None and t_rel > end_s + pad:
                        continue
                elif start_s is not None:
                    # Keep the latched transform, but do not let its original
                    # timestamp stretch the output's advertised time span back
                    # to the start of the untrimmed recording.
                    window_start = t0_ns + int(start_s * 1e9)
                    log_time = max(log_time, window_start)
                new_cid = new_channel_ids.get(rec.channel_id)
                if new_cid is None:
                    schema = schemas.get(channel.schema_id)
                    if schema is None:
                        continue
                    new_sid = new_schema_ids.get(channel.schema_id)
                    if new_sid is None:
                        new_sid = writer.register_schema(schema.name, schema.encoding, schema.data)
                        new_schema_ids[channel.schema_id] = new_sid
                    new_cid = writer.register_channel(
                        channel.topic, channel.message_encoding, new_sid, dict(channel.metadata)
                    )
                    new_channel_ids[rec.channel_id] = new_cid
                writer.add_message(new_cid, log_time, rec.data, rec.publish_time, rec.sequence)
                counts[channel.topic] += 1
                if not kept_range:
                    kept_range = [log_time, log_time]
                else:
                    kept_range[0] = min(kept_range[0], log_time)
                    kept_range[1] = max(kept_range[1], log_time)
        progress.close()
        writer.finish()

    print(f"\n=== trim summary: {out_path} ===")
    if not counts:
        print("  NO messages kept - check topic names with 'rocklabel inspect' "
              "(or --all-topics to keep everything).")
    for topic, n in sorted(counts.items()):
        print(f"  {topic:<40} {n:>9} msgs")
    if kept_range:
        span = (kept_range[1] - kept_range[0]) / 1e9
        print(f"  kept time span:  {span:.1f} s")
    print(f"  output size:     {os.path.getsize(out_path) / 1e6:.1f} MB "
          f"(input {total / 1e6:.1f} MB)")
    if read_error is not None:
        print(
            f"  NOTE: input ended early at byte {pos} of {total} "
            f"({100 * pos / max(total, 1):.0f}%): {type(read_error).__name__}. "
            "The recording was likely never finalized (recorder killed mid-write). "
            "Everything before that point was salvaged; the output file is valid."
        )
