"""`rocklabel trim`: cut a recording down to the topics and time range that matter.

Copies messages raw (no decoding) into a new, properly indexed mcap, reading
the input sequentially. This makes it double as a recovery tool: a truncated /
unfinalized recording (recorder killed mid-write, missing footer) is read up
to the first corrupt byte and everything before it is salvaged into a valid
output file.

By default only the pointcloud topic plus /tf and /tf_static are kept — for a
recording that also carries cameras etc. this typically shrinks the file by
an order of magnitude. --start-s/--end-s crop time as seconds relative to the
first message; TF topics are exempt from the time window so pose lookups near
the window edges keep working.
"""

from __future__ import annotations

import os
from collections import Counter

from mcap.records import Channel, Message, Schema
from mcap.stream_reader import StreamReader
from mcap.writer import Writer as McapWriter
from tqdm import tqdm

from .. import __version__


def run_trim(in_path: str, out_path: str, cfg: dict, extra_topics: list[str] | None = None,
             start_s: float | None = None, end_s: float | None = None,
             all_topics: bool = False) -> None:
    if os.path.abspath(in_path) == os.path.abspath(out_path):
        raise SystemExit("trim: --out must differ from the input file")
    topics_cfg = cfg["topics"]
    tf_topics = {topics_cfg["tf_topic"], topics_cfg["tf_static_topic"]}
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
                if channel.topic not in tf_topics:  # TF exempt from the time window
                    t_rel = (rec.log_time - t0_ns) / 1e9
                    if start_s is not None and t_rel < start_s:
                        continue
                    if end_s is not None and t_rel > end_s:
                        continue
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
                writer.add_message(new_cid, rec.log_time, rec.data, rec.publish_time, rec.sequence)
                counts[channel.topic] += 1
                if not kept_range:
                    kept_range = [rec.log_time, rec.log_time]
                else:
                    kept_range[0] = min(kept_range[0], rec.log_time)
                    kept_range[1] = max(kept_range[1], rec.log_time)
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
