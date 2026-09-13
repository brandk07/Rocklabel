"""`rocklabel-train mapeval`: replay a whole recording and grade the map it builds.

The sampled Lance benchmark ranks checkpoints by average precision over a fixed
set of candidate balls. That is a useful ranking and a poor description of what
the rig actually does, which is: score fresh scans, keep the newest probability
per 3D voxel, and drive off the result. A model can rank rocks above ground
correctly and still leave the arena covered in mid-air false positives that
never refresh, because nothing in that pipeline ever revisits a prediction.

So this grades the map. It replays the recording chronologically through the
real inference preprocessing, accumulates predictions exactly as
:class:`~rocklabel.live.scoring.LiveScorer` does, and measures what a navigation
stack would see: how much of each physical rock is covered and when it first
was, how much false ground is claimed and for how long, and how badly either
fragments.

It also answers the question the old audit could not. With ``--clean`` it runs
a second map alongside the first, identical in every respect except that
:class:`~rocklabel.live.evidence.EvidenceMap` may retract predictions the
sensor has since looked straight through. Both maps are built from one set of
scores in one pass, so the comparison shares its inputs, its rock geometry and
its denominators exactly; nothing about it depends on two runs agreeing.

Three caches, each skipped when it already exists:

``<frames-dir>/geometry``   the cropped returns, their per-point sensor
                            origins and the visible rock geometry. Built once
                            per (recording, band, range); independent of any
                            model.
``<frames-dir>/scores/<id>``  one checkpoint's native predictions on those
                            frames.
``<out>``                   the graded result.

Sweeping cleanup settings therefore costs neither a decode nor a forward pass.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, fields

import numpy as np

from ..dataset.labeling import LABEL_CLEAR, inside_arena, label_rocks, points_in_rock
from ..live.evidence import EvidenceMap, EvidenceSettings
from .visual_audit import (CellMap, cell_ids, cell_set, connected_components,
                           projected_rock_labels)

#: Operational vertical band, relative to the measured floor plane. This is the
#: band the rig runs with, and it is NOT the band the sampled benchmark cache
#: was generated against - that one carries the labeller's own world-frame z
#: clip. Numbers from the two are not interchangeable.
FLOOR_BAND = (-0.10, 0.60)
#: Horizontal range from the robot base, metres.
MAX_RANGE_M = 8.0
#: Ground-cell size every map measurement is reduced to.
CELL_M = 0.10


# --------------------------------------------------------------------------- #
# Stage 1: geometry
# --------------------------------------------------------------------------- #
#: Bumped whenever the cached frames change shape or meaning, so a cache
#: written by an older build is rejected rather than silently reread.
GEOMETRY_SCHEMA = 2


def _file_identity(path: str) -> dict:
    """Cheap content identity for a source file.

    The labels file is small enough to hash outright. The recording is not - a
    decode-length mcap would be read twice for every rebuild - so it is
    identified by its size and modification time, which is what every build
    system on the machine already trusts for the same reason.
    """
    st = os.stat(path)
    ident = {"path": os.path.abspath(path), "bytes": st.st_size,
             "mtime_ns": st.st_mtime_ns}
    if st.st_size <= (8 << 20):
        ident["sha256"] = _sha256(path)
    return ident


def _geometry_settings(recording, labels_path, stride, window_s) -> dict:
    """Everything a cached frame's contents depend on.

    This is the cache's identity, not a description of it: a rebuild with a
    different stride, band or labels file must not be able to land on top of an
    existing cache, and reusing one whose recording has been re-exported since
    is the same mistake with a slower fuse.
    """
    return {"schema": GEOMETRY_SCHEMA,
            "recording": os.path.abspath(recording),
            "labels": os.path.abspath(labels_path),
            "recording_identity": _file_identity(recording),
            "labels_identity": _file_identity(labels_path),
            "stride": int(stride), "window_s": float(window_s),
            "floor_band": list(FLOOR_BAND), "max_range_m": MAX_RANGE_M,
            "cell_m": CELL_M}


#: Keys of the geometry settings that define what is in the cache. The rest -
#: frame count, timings - are results of building it.
_IDENTITY_KEYS = ("schema", "recording", "labels", "recording_identity",
                  "labels_identity", "stride", "window_s", "floor_band",
                  "max_range_m", "cell_m")


def geometry_identity(settings: dict) -> dict:
    return {k: settings.get(k) for k in _IDENTITY_KEYS}


def check_geometry_cache(geo_dir: str, recording: str, labels_path: str,
                         stride: int, window_s: float) -> tuple[bool, str]:
    """``(reusable, why_not)`` for an existing geometry cache.

    A cache is reusable only if it is complete - its manifest was written, and
    every frame it claims is still on disk - and if it was built from the same
    sources under the same settings.
    """
    manifest = os.path.join(geo_dir, "settings.json")
    if not os.path.exists(manifest):
        return False, "no cached geometry"
    try:
        have = json.load(open(manifest))
    except (OSError, ValueError) as exc:
        return False, f"unreadable manifest ({exc})"
    want = _geometry_settings(recording, labels_path, stride, window_s)
    changed = {k: (have.get(k), want[k]) for k in _IDENTITY_KEYS
               if have.get(k) != want[k]}
    if changed:
        return False, "settings changed: " + ", ".join(sorted(changed))
    on_disk = sum(1 for f in os.listdir(geo_dir) if f.startswith("frame-"))
    if on_disk != int(have.get("frames", -1)):
        return False, (f"incomplete: manifest claims {have.get('frames')} "
                       f"windows, {on_disk} on disk")
    return True, ""


def _windows(stream, window_s: float):
    """Merge scans into time windows, keeping every member's sensor origin.

    :class:`~rocklabel.recording.pipeline.WindowedScanStream` keeps only the
    last member's pose, which is fine for cropping and wrong for raycasting: a
    beam has to be traced from where it was actually measured. The origins
    differ by centimetres inside one window, which is small but is exactly the
    size of the thing being decided, so it is carried rather than assumed away.
    """
    buf: list = []
    for scan in stream:
        if buf and scan.time_s - buf[0].time_s >= window_s:
            yield buf
            buf = []
        buf.append(scan)
    if buf:
        yield buf


def build_geometry(recording: str, labels_path: str, out_dir: str,
                   stride: int = 10, window_s: float = 0.05,
                   progress_every: int = 200) -> dict:
    """Cache the cropped returns and their sensor origins, window by window.

    Built into a staging directory and published by a rename, so an
    interrupted build leaves no cache at all rather than a short one that the
    next run would happily reuse as complete.
    """
    import shutil

    from ..config import load_config
    from ..geometry.leveling import check_level_match, level_record, pin_level_to_labels
    from ..labels import load_labels
    from ..recording.pipeline import ScanStream

    published, out_dir = out_dir, out_dir + ".partial"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    labels = load_labels(labels_path)
    cfg = pin_level_to_labels(load_config(None), labels.level)
    stream = ScanStream(recording, cfg, stride=1, progress=False)
    check_level_match(labels.level, level_record(stream), labels_path)
    floor = getattr(getattr(stream, "solution", None), "floor_z", None)
    if floor is None and labels.level:
        floor = labels.level.get("floor_z")
    if floor is None:
        raise SystemExit("a measured floor is required to apply the operational band")
    floor = float(floor)

    settings = _geometry_settings(recording, labels_path, stride, window_s)
    settings["floor_z"] = floor
    kept, origin_span, t0, origin_time = 0, 0.0, time.monotonic(), None
    for k, members in enumerate(_windows(stream, window_s)):
        if k % stride:
            continue
        last = members[-1]
        if origin_time is None:
            origin_time = float(last.time_s)
        base = last.T_odom_base[:3, 3].astype(np.float64)
        origins = np.stack([m.T_odom_lidar[:3, 3] for m in members]).astype(np.float32)
        owner = np.concatenate([np.full(len(m.xyz_odom), i, np.int16)
                                for i, m in enumerate(members)])
        xyz = np.concatenate([m.xyz_odom for m in members])
        inten = np.concatenate([m.intensity for m in members])

        lo = np.array([base[0] - MAX_RANGE_M, base[1] - MAX_RANGE_M,
                       floor + FLOOR_BAND[0]])
        hi = np.array([base[0] + MAX_RANGE_M, base[1] + MAX_RANGE_M,
                       floor + FLOOR_BAND[1]])
        keep = ((xyz >= lo) & (xyz <= hi)).all(axis=1)
        d = xyz[:, :2] - base[:2]
        keep &= (d * d).sum(axis=1) <= MAX_RANGE_M ** 2
        if not keep.any():
            continue
        xyz, inten, owner = xyz[keep], inten[keep], owner[keep]
        if len(origins) > 1:
            origin_span = max(origin_span,
                              float(np.linalg.norm(origins - origins[0], axis=1).max()))

        # The denominator every coverage number is measured against: ground
        # cells a real return actually landed on, per rock, unfiltered.
        rows = []
        for rock in labels.rocks:
            on = xyz[points_in_rock(xyz, rock)]
            if len(on):
                rows.extend((rock.id, *key) for key in cell_set(on[:, :2], CELL_M))
        np.savez_compressed(
            os.path.join(out_dir, f"frame-{kept:05d}.npz"),
            index=last.index, time_s=float(last.time_s) - origin_time,
            base=base, origins=origins, origin_of_point=owner,
            xyz=xyz.astype(np.float32), intensity=inten.astype(np.float32),
            raw_rock_cells=np.array(rows, dtype=np.int64).reshape(-1, 3))
        kept += 1
        if progress_every and kept % progress_every == 0:
            print(f"  cached {kept} windows ({time.monotonic() - t0:.0f}s)", flush=True)
    settings.update(frames=kept, max_origin_spread_m=round(origin_span, 4),
                    wall_seconds=round(time.monotonic() - t0, 1))
    _atomic_json(os.path.join(out_dir, "settings.json"), settings)
    shutil.rmtree(published, ignore_errors=True)
    os.replace(out_dir, published)
    print(f"geometry: {kept} windows, sensor moved at most "
          f"{origin_span * 100:.1f} cm within one window", flush=True)
    return settings


# --------------------------------------------------------------------------- #
# Stage 2: native predictions
# --------------------------------------------------------------------------- #
def build_scores(geometry_dir: str, checkpoint: str, out_dir: str,
                 device: str | None = None, batch: int = 512) -> dict:
    """Run one checkpoint over the cached windows, in its own native units.

    The result records which geometry cache it was produced from, because a
    score file says nothing about the frame it scored: rebuilding geometry
    under a new stride and leaving an old model's scores beside it would
    otherwise pair the two without a word.
    """
    import shutil

    import torch

    from .mcapview import _score_balls, _score_frame
    from .visual_audit import _load_checkpoint

    geometry = json.load(open(os.path.join(geometry_dir, "settings.json")))
    published, out_dir = out_dir, out_dir + ".partial"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loaded = _load_checkpoint(checkpoint, dev)
    g = loaded["generator"]
    frames = sorted(f for f in os.listdir(geometry_dir) if f.startswith("frame-"))
    t0 = time.monotonic()
    for i, name in enumerate(frames):
        with np.load(os.path.join(geometry_dir, name)) as z:
            xyz = z["xyz"].astype(np.float64)
            inten = z["intensity"].astype(np.float32)
            base = z["base"]
            index = int(z["index"])
        rng = np.random.default_rng([int(g["seed"]), index])
        if loaded["task"] == "classify":
            pos, prob = _score_balls(xyz, inten, g, loaded["model"], dev, rng, batch)
        else:
            pos, prob = _score_frame(xyz, inten, base, g, loaded["model"], dev, rng)
        np.savez_compressed(os.path.join(out_dir, name),
                            positions=np.asarray(pos, np.float32),
                            probabilities=np.asarray(prob, np.float32))
        if (i + 1) % 200 == 0:
            print(f"  scored {i + 1}/{len(frames)} ({time.monotonic() - t0:.0f}s)",
                  flush=True)
    meta = {"checkpoint": os.path.abspath(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "model": loaded["name"], "task": loaded["task"],
            "threshold": loaded["threshold"], "generator": g,
            "frames": len(frames), "device": str(dev),
            "geometry_identity": geometry_identity(geometry),
            "wall_seconds": round(time.monotonic() - t0, 1)}
    _atomic_json(os.path.join(out_dir, "meta.json"), meta)
    shutil.rmtree(published, ignore_errors=True)
    os.replace(out_dir, published)
    print(f"scores: {len(frames)} windows in {meta['wall_seconds']:.0f}s", flush=True)
    return meta


# --------------------------------------------------------------------------- #
# Stage 3: accumulate and grade
# --------------------------------------------------------------------------- #
class _Map:
    """One accumulated prediction map, optionally with the evidence layer.

    The control and the cleaned map are two of these fed the identical stream
    of scores, so every difference between their numbers is the retraction
    policy and nothing else.
    """

    def __init__(self, name: str, voxel: float, threshold: float,
                 evidence: EvidenceSettings | None):
        self.name = name
        self.voxel = float(voxel)
        self.threshold = float(threshold)
        self.map: dict[tuple[int, int, int], tuple[np.ndarray, float]] = {}
        self.ev = EvidenceMap(voxel, evidence) if evidence else None
        #: 10 cm ground cell -> seconds it has stood above threshold.
        self.hot_seconds: dict[tuple[int, int], float] = {}
        self.retracted = 0
        #: The hot set the last interval is owed to, and when that interval
        #: started. See :meth:`tick`.
        self._prev_hot: set[tuple[int, int]] = set()
        self._prev_t: float | None = None

    def update(self, positions, probabilities, origins, xyz, time_s: float,
               stamps=None) -> None:
        keys = np.floor(positions / self.voxel).astype(np.int64)
        for key, pos, prob in zip(map(tuple, keys), positions, probabilities):
            self.map[key] = (np.asarray(pos, float), float(prob))
        if self.ev is not None:
            self.ev.observe(origins, xyz, stamps)
            hot = [(k, v) for k, v in self.map.items() if v[1] >= self.threshold]
            if hot:
                drop = self.ev.retract(np.stack([v[0] for _, v in hot]))
                for (key, _), gone in zip(hot, drop):
                    if gone:
                        del self.map[key]
                        self.retracted += 1

    def cells(self):
        """Newest value per native voxel, reduced to 10 cm ground cells by max."""
        if not self.map:
            return CellMap(np.empty((0, 3)), np.empty(0), np.empty((0, 2), np.int64))
        positions = np.stack([v[0] for v in self.map.values()])
        probs = np.array([v[1] for v in self.map.values()], float)
        ids = cell_ids(positions[:, :2], CELL_M)
        best: dict[tuple[int, int], int] = {}
        for i, key in enumerate(map(tuple, ids)):
            old = best.get(key)
            if old is None or probs[i] > probs[old]:
                best[key] = i
        take = np.fromiter(best.values(), dtype=np.int64, count=len(best))
        return CellMap(positions[take], probs[take], ids[take])

    def tick(self, hot: set, time_s: float) -> None:
        """Bank how long each ground cell has stood above threshold.

        The interval between two windows belongs to the state the map was in
        *during* it - that is, to the state the earlier window left behind. So
        each call closes the previous interval against the previous hot set and
        opens a new one; charging the elapsed time to the state that has only
        just been written would credit every retraction with time it never
        served, and every new false cell with time before it existed.

        ``hot`` is already restricted to the arena, so a cell outside the
        measured region cannot accumulate false-area seconds it will never be
        charged for in the final count.
        """
        if self._prev_t is not None:
            dt = max(time_s - self._prev_t, 0.0)
            for key in self._prev_hot:
                self.hot_seconds[key] = self.hot_seconds.get(key, 0.0) + dt
        self._prev_hot, self._prev_t = hot, time_s


def evaluate(geometry_dir: str, scores_dir: str, out_dir: str,
             evidence: EvidenceSettings | None, measure_every: int = 20,
             threshold: float | None = None) -> dict:
    """Accumulate chronologically and grade the control and cleaned maps."""
    from ..labels import load_labels

    os.makedirs(out_dir, exist_ok=True)
    geo = json.load(open(os.path.join(geometry_dir, "settings.json")))
    meta = json.load(open(os.path.join(scores_dir, "meta.json")))
    labels = load_labels(geo["labels"])
    shell = float(meta["generator"].get("boundary_shell_m", 0.05))
    voxel = float(meta["generator"]["centers_voxel_m"])
    thr = float(threshold if threshold is not None else meta["threshold"])

    maps = [_Map("control", voxel, thr, None)]
    if evidence is not None:
        maps.append(_Map("cleaned", voxel, thr, evidence))

    visible: dict[int, set] = {r.id: set() for r in labels.rocks}
    #: When a real return first landed on any of a rock's ground cells. The
    #: clock every acquisition number is measured from: a model cannot cover a
    #: rock the sensor has not reached yet, and crediting it with the wait
    #: would measure the robot's path rather than the model.
    first_visible: dict[int, float] = {}
    # Per rock and per map: cells ever covered, and cells covered then lost.
    ever: dict[tuple[str, int], set] = {(m.name, r.id): set()
                                        for m in maps for r in labels.rocks}
    lost: dict[tuple[str, int], set] = {k: set() for k in ever}
    first_seen: dict[tuple[str, int], float] = {}
    first_occupied: dict[tuple[str, int], float] = {}
    #: (map, rock) -> [(time, occupied cells, attributed cells)] every window.
    #: Kept in full because every timing question below - when a rock reached
    #: half its final coverage, how often it fell back under it, whether a
    #: retraction was ever undone - is a question about the curve and not about
    #: its endpoint, and the denominator those fractions need is not known
    #: until the recording has finished playing.
    history: dict[tuple[str, int], list] = {k: [] for k in ever}

    frames = sorted(f for f in os.listdir(geometry_dir) if f.startswith("frame-"))
    timeline: list[dict] = []
    t0 = time.monotonic()
    rel = 0.0
    for n, name in enumerate(frames):
        with np.load(os.path.join(geometry_dir, name)) as z:
            rel = float(z["time_s"])
            xyz = z["xyz"].astype(np.float64)
            origins = z["origins"].astype(np.float64)[z["origin_of_point"]]
            rows = z["raw_rock_cells"]
            index = int(z["index"])
        with np.load(os.path.join(scores_dir, name)) as z:
            positions = z["positions"].astype(np.float64)
            probabilities = z["probabilities"].astype(np.float64)
        for rid, i, j in rows:
            visible[int(rid)].add((int(i), int(j)))
            first_visible.setdefault(int(rid), rel)

        for m in maps:
            # The window's acquisition index is its identity: replaying the
            # same window twice is not two looks at the arena.
            m.update(positions, probabilities, origins, xyz, rel, stamps=index)
            cells = m.cells()
            arena = inside_arena(cells.positions, labels.arena)
            hot_mask = (cells.probabilities >= thr) & arena
            hot = {tuple(k) for k in cells.cell_ids[hot_mask]}
            m.tick(hot, rel)
            for rock in labels.rocks:
                key = (m.name, rock.id)
                seen = visible[rock.id]
                # Occupied: the cell is claimed at all. Attributed: the
                # representative also lies on this rock's own 3D geometry.
                occupied = {tuple(k) for k in cells.cell_ids[hot_mask]} & seen
                on = hot_mask & points_in_rock(cells.positions, rock)
                now = {tuple(k) for k in cells.cell_ids[on]} & seen
                gone = ever[key] - now
                lost[key] |= gone
                if now and key not in first_seen:
                    first_seen[key] = rel
                if occupied and key not in first_occupied:
                    first_occupied[key] = rel
                ever[key] |= now
                history[key].append((rel, len(occupied), len(now)))
            if n % measure_every == 0 or n == len(frames) - 1:
                timeline.append(_measure(m, cells, labels, shell, thr, rel, n, visible))
        if (n + 1) % 200 == 0:
            print(f"  accumulated {n + 1}/{len(frames)} ({time.monotonic() - t0:.0f}s)",
                  flush=True)
    # Close the last interval so the final state is charged for nothing rather
    # than for the whole recording.
    for m in maps:
        m.tick(set(), rel)

    payload = {"geometry": geo, "scores": meta, "threshold": thr,
               "evidence": (None if evidence is None else asdict(evidence)),
               "wall_seconds": round(time.monotonic() - t0, 1),
               "maps": {}}
    per_rock_rows = []
    for m in maps:
        cells = m.cells()
        final = _measure(m, cells, labels, shell, thr, rel, len(frames) - 1, visible)
        arena = inside_arena(cells.positions, labels.arena)
        hot_mask = (cells.probabilities >= thr) & arena
        rocks = []
        for rock in labels.rocks:
            key = (m.name, rock.id)
            seen = visible[rock.id]
            occupied = {tuple(k) for k in cells.cell_ids[hot_mask]} & seen
            on = hot_mask & points_in_rock(cells.positions, rock)
            now = {tuple(k) for k in cells.cell_ids[on]} & seen
            rocks.append({
                "map": m.name, "rock_id": rock.id,
                "visible_cells": len(seen),
                # Two different questions, reported side by side. Occupied
                # footprint asks whether the map claims that piece of ground at
                # all - which is what a navigation stack drives on. 3D
                # attribution asks whether the thing it claims there is
                # actually this rock's surface, which is what says the map is
                # describing the arena rather than agreeing with it by
                # accident. Retracting a floating positive can raise the second
                # without touching the first.
                "occupied_cells": len(occupied),
                "occupied_coverage": len(occupied) / len(seen) if seen else None,
                "covered_cells": len(now),
                "coverage": len(now) / len(seen) if seen else None,
                "ever_covered_cells": len(ever[key]),
                "lost_cells": len(lost[key]),
                "still_lost_cells": len(lost[key] - now),
                "components": connected_components(now),
                "first_visible_s": first_visible.get(rock.id),
                "first_occupied_s": first_occupied.get(key),
                "first_covered_s": first_seen.get(key),
                **_coverage_timing(history[key], len(seen),
                                   first_visible.get(rock.id)),
            })
        per_rock_rows.extend(rocks)
        got = [r["coverage"] for r in rocks if r["coverage"] is not None]
        occ = [r["occupied_coverage"] for r in rocks
               if r["occupied_coverage"] is not None]
        acquired = [r["first_covered_s"] for r in rocks
                    if r["first_covered_s"] is not None]
        false_seconds = _false_seconds(m, labels, shell)
        payload["maps"][m.name] = {
            **final,
            "retracted_voxels": m.retracted,
            "audited_rocks": len(got),
            "macro_coverage": float(np.mean(got)) if got else None,
            "worst_rock_coverage": min(got) if got else None,
            "macro_coverage_footprint": float(np.mean(occ)) if occ else None,
            "worst_rock_coverage_footprint": min(occ) if occ else None,
            "occupied_rock_cells": int(sum(r["occupied_cells"] for r in rocks)),
            "rocks_never_covered": int(sum(1 for r in rocks if not r["ever_covered_cells"])),
            "rocks_with_lost_cells": int(sum(1 for r in rocks if r["lost_cells"])),
            "median_first_covered_s": float(np.median(acquired)) if acquired else None,
            "slowest_first_covered_s": max(acquired) if acquired else None,
            "false_cell_seconds": round(false_seconds, 1),
            "evidence": None if m.ev is None else m.ev.stats(),
        }

    curve = [row for m in maps for row in threshold_frontier(m, labels, shell, visible)]
    payload["budgets"] = {
        m.name: {str(b): at_budget([r for r in curve if r["map"] == m.name], b)
                 for b in (2200, 1700, 1000, 500, 200)}
        for m in maps}
    _atomic_json(os.path.join(out_dir, "summary.json"), payload)
    _write_csv(os.path.join(out_dir, "per-rock.csv"), per_rock_rows)
    _write_csv(os.path.join(out_dir, "timeline.csv"), timeline)
    _write_csv(os.path.join(out_dir, "threshold-frontier.csv"), curve)
    _plot_maps(os.path.join(out_dir, "map.png"), maps, labels, thr)
    for m in maps:
        cells = m.cells()
        np.savez_compressed(os.path.join(out_dir, f"map-{m.name}.npz"),
                            positions=cells.positions,
                            probabilities=cells.probabilities,
                            cell_ids=cells.cell_ids)
    _write_summary_md(os.path.join(out_dir, "summary.md"), payload, per_rock_rows)
    return payload


def check_scores_cache(score_dir: str, geo_dir: str, checkpoint: str) -> tuple[bool, str]:
    """``(reusable, why_not)`` for an existing per-checkpoint score cache."""
    manifest = os.path.join(score_dir, "meta.json")
    if not os.path.exists(manifest):
        return False, "no cached scores"
    try:
        have = json.load(open(manifest))
        geometry = json.load(open(os.path.join(geo_dir, "settings.json")))
    except (OSError, ValueError) as exc:
        return False, f"unreadable manifest ({exc})"
    want = geometry_identity(geometry)
    if have.get("geometry_identity") != want:
        return False, "scored against different geometry"
    if have.get("checkpoint_sha256") != _sha256(checkpoint):
        return False, "different checkpoint bytes"
    on_disk = sum(1 for f in os.listdir(score_dir) if f.startswith("frame-"))
    if on_disk != int(have.get("frames", -1)):
        return False, (f"incomplete: manifest claims {have.get('frames')} "
                       f"windows, {on_disk} on disk")
    return True, ""


def threshold_frontier(m, labels, shell, visible) -> list[dict]:
    """The finished map read at **every** distinct score it holds.

    Needed because two checkpoints compared at their own stored thresholds are
    not being compared on anything: one of them is simply more willing to say
    rock. The honest question is how much of each rock each one covers at the
    same amount of wrongly-claimed ground - which is what this frontier is for.

    Every distinct value, not a grid. A grid stopping at 0.95 was read as
    proving a model could not get below sixteen hundred false cells; the same
    map at 0.99 holds 1,149 and at 0.999 holds 388. Ties are one row, because
    a threshold between two equal scores is not an operating point. It is
    computed as running sums down the sorted scores rather than by re-reading
    the map per threshold, so the cost is one sort however many distinct scores
    there are.

    A threshold picked by reading this is an oracle diagnostic, not an
    operating point: the map it is read off is the map being graded. And for a
    *cleaned* map it is weaker still - the cleanup policy only ever judged the
    voxels that were above the threshold it actually ran at, so reading the
    finished map at another threshold is a frozen-map diagnostic and not what
    that policy would have produced.
    """
    cells = m.cells()
    keep = inside_arena(cells.positions, labels.arena)
    pos, prob, ids = cells.positions[keep], cells.probabilities[keep], cells.cell_ids[keep]
    if not len(pos):
        return []
    attributed = label_rocks(pos, labels.rocks, shell)
    footprint = projected_rock_labels(pos, labels.rocks, shell)

    order = np.argsort(-prob, kind="stable")
    prob = prob[order]
    false_3d = np.cumsum(attributed[order] == LABEL_CLEAR)
    false_fp = np.cumsum(footprint[order] == LABEL_CLEAR)
    # Per rock: this cell is one of that rock's observed ground cells.
    per_rock = {}
    for rock in labels.rocks:
        seen = visible[rock.id]
        if not seen:
            continue
        on_cell = np.array([tuple(k) in seen for k in ids[order]], bool)
        per_rock[rock.id] = {
            "seen": len(seen),
            # Occupied: the cell is claimed at all. The representative is the
            # highest-scoring voxel in its column, so "the representative is
            # above threshold" and "something in this column is" are the same
            # statement - this is occupied footprint coverage.
            "occupied": np.cumsum(on_cell),
            # Attributed: the representative also lies on the rock's own 3D
            # geometry. A floating positive over a rock occupies its cell and
            # is not attributed to it.
            "attributed": np.cumsum(on_cell & (attributed[order] == rock.id)),
        }

    # One row per distinct score: the last index holding that value.
    last = np.nonzero(np.r_[prob[1:] != prob[:-1], True])[0]
    rows = [{"map": m.name, "threshold": None, "ground_cells": 0,
             "false_cells_3d": 0, "false_cells_footprint": 0,
             "macro_coverage": 0.0, "macro_coverage_footprint": 0.0,
             "worst_rock_coverage": 0.0, "worst_rock_coverage_footprint": 0.0}]
    for i in last:
        attr = [per_rock[r]["attributed"][i] / per_rock[r]["seen"] for r in per_rock]
        occ = [per_rock[r]["occupied"][i] / per_rock[r]["seen"] for r in per_rock]
        rows.append({
            "map": m.name, "threshold": float(prob[i]),
            "ground_cells": int(i + 1),
            "false_cells_3d": int(false_3d[i]),
            "false_cells_footprint": int(false_fp[i]),
            "macro_coverage": float(np.mean(attr)) if attr else None,
            "macro_coverage_footprint": float(np.mean(occ)) if occ else None,
            "worst_rock_coverage": min(attr) if attr else None,
            "worst_rock_coverage_footprint": min(occ) if occ else None,
        })
    return rows


def at_budget(rows: list[dict], budget: int, column: str = "false_cells_footprint"):
    """The frontier row with the most coverage inside a false-area budget.

    The frontier is monotone in the budget, so this is the lowest threshold
    whose false area still fits - the point of reading it is to compare two
    models at equal wrongly-claimed area rather than at whichever threshold
    each happens to carry.
    """
    fit = [r for r in rows if r[column] <= budget]
    return max(fit, key=lambda r: r["ground_cells"]) if fit else None


def _coverage_timing(series, denominator: int, visible_from) -> dict:
    """When a rock reached a share of its final coverage, and what happened after.

    Everything here is measured against the coverage the map ends with, and
    from the moment a real return first landed on the rock rather than from the
    start of the recording: the alternative measures how long the robot took to
    drive past it. "First contact" is one cell, which is what the old summary
    reported and is not a useful statement about whether a rock is on the map -
    a single cell of a seventeen-cell rock is not something to drive around.

    ``fell_below_*`` counts how many times coverage dropped back under a level
    it had already reached. Nonzero means the map is losing ground it had, and
    the map cannot say so from its endpoint.
    """
    out: dict = {}
    if not series or not denominator:
        return {f"t{int(p * 100)}_cover_s": None for p in (0.25, 0.5, 0.8)} | {
            f"fell_below_{int(p * 100)}": None for p in (0.25, 0.5, 0.8)}
    times = np.array([r[0] for r in series])
    frac = np.array([r[2] for r in series], float) / denominator
    for p in (0.25, 0.5, 0.8):
        at = np.nonzero(frac >= p)[0]
        reach = float(times[at[0]]) if len(at) else None
        out[f"t{int(p * 100)}_cover_s"] = (
            None if reach is None or visible_from is None else round(reach - visible_from, 2))
        above = frac >= p
        # Transitions from "had it" to "lost it", counted only after the level
        # was first reached.
        out[f"fell_below_{int(p * 100)}"] = (
            int(np.count_nonzero(above[:-1] & ~above[1:])) if len(at) else 0)
    out["peak_coverage"] = float(frac.max())
    out["final_minus_peak"] = float(frac[-1] - frac.max())
    return out


def _measure(m, cells, labels, shell, thr, rel, n, visible) -> dict:
    """One row of the timeline for one map."""
    keep = inside_arena(cells.positions, labels.arena)
    pos, prob, ids = cells.positions[keep], cells.probabilities[keep], cells.cell_ids[keep]
    hot = prob >= thr
    attributed = label_rocks(pos, labels.rocks, shell)
    footprint = projected_rock_labels(pos, labels.rocks, shell)
    hot_ids = {tuple(k) for k in ids[hot]}
    covered = sum(len(hot_ids & v) for v in visible.values())
    return {
        "map": m.name, "time_s": round(rel, 2), "frames": n + 1,
        "native_voxels": len(m.map),
        "ground_cells": int(hot.sum()),
        # A floating positive over a rock still sits over that rock's footprint,
        # so footprint credit alone would reward it. Both are reported.
        "false_cells_3d": int((hot & (attributed == LABEL_CLEAR)).sum()),
        "false_cells_footprint": int((hot & (footprint == LABEL_CLEAR)).sum()),
        "covered_rock_cells": covered,
        "retracted": m.retracted,
    }


def _false_seconds(m, labels, shell) -> float:
    """Total cell-seconds spent hot on arena ground that is not a rock footprint.

    Restricted to the same arena polygon the final false-area count uses.
    Without that restriction the two numbers answer different questions and the
    duration one silently includes everything outside the measured region.
    """
    if not m.hot_seconds:
        return 0.0
    keys = np.array(list(m.hot_seconds), dtype=np.int64)
    centres = np.column_stack([(keys[:, 0] + 0.5) * CELL_M,
                               (keys[:, 1] + 0.5) * CELL_M,
                               np.zeros(len(keys))])
    lab = projected_rock_labels(centres, labels.rocks, shell)
    seconds = np.fromiter(m.hot_seconds.values(), float, len(m.hot_seconds))
    return float(seconds[(lab == LABEL_CLEAR)
                         & inside_arena(centres, labels.arena)].sum())


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: str, value: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, default=float)
    os.replace(tmp, path)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _plot_maps(path: str, maps, labels, thr: float) -> None:
    """Top-down picture of each finished map against the labelled rocks.

    Aggregates hide the thing that matters most here - whether the wrongly
    claimed ground is scattered everywhere or piled in two places - and a
    number cannot show that. One panel per map, same axes, rocks outlined.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, len(maps), figsize=(7 * len(maps), 6.5),
                             squeeze=False, sharex=True, sharey=True)
    for ax, m in zip(axes[0], maps):
        cells = m.cells()
        keep = inside_arena(cells.positions, labels.arena)
        pos, prob = cells.positions[keep], cells.probabilities[keep]
        hot = prob >= thr
        attributed = label_rocks(pos, labels.rocks, 0.05)
        if labels.arena is not None:
            poly = np.vstack([labels.arena, labels.arena[:1]])
            ax.plot(poly[:, 0], poly[:, 1], color="0.6", lw=1)
        ax.scatter(pos[~hot, 0], pos[~hot, 1], s=1, c="0.85", label="below threshold")
        wrong = hot & (attributed == LABEL_CLEAR)
        ax.scatter(pos[wrong, 0], pos[wrong, 1], s=4, c="#d1495b",
                   label=f"claimed, no rock ({int(wrong.sum())})")
        right = hot & (attributed != LABEL_CLEAR)
        ax.scatter(pos[right, 0], pos[right, 1], s=6, c="#2e8b57",
                   label=f"claimed, on a rock ({int(right.sum())})")
        for rock in labels.rocks:
            centre = (rock.center[:2] if getattr(rock, "shape", "") != "polygon"
                      else np.mean(np.asarray(rock.vertices)[:, :2], axis=0))
            ax.annotate(str(rock.id), centre, color="#12355b", fontsize=8,
                        ha="center", va="center")
        ax.set_title(f"{m.name} — {int(hot.sum())} cells at threshold {thr:.2f}")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8, markerscale=3)
    fig.suptitle("Accumulated map, looking down. Rock numbers mark the labels.")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fmt(value, places: int) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _write_summary_md(path: str, payload: dict, per_rock: list[dict]) -> None:
    names = list(payload["maps"])
    lines = [f"# Map evaluation: {os.path.basename(payload['scores']['checkpoint'])}", ""]
    lines += [f"Model `{payload['scores']['model']}`, threshold "
              f"{payload['threshold']:.2f}, {payload['geometry']['frames']} windows "
              f"of {os.path.basename(payload['geometry']['recording'])}.", ""]
    lines += [f"Operational band {payload['geometry']['floor_band']} m about the "
              f"measured floor, {payload['geometry']['max_range_m']} m range, "
              f"latest value per {payload['scores']['generator']['centers_voxel_m']} m "
              f"voxel, reduced to {CELL_M} m ground cells.", ""]
    headline = ["ground_cells", "occupied_rock_cells",
                "macro_coverage_footprint", "worst_rock_coverage_footprint",
                "false_cells_3d", "false_cells_footprint",
                "false_cell_seconds", "macro_coverage", "worst_rock_coverage",
                "rocks_never_covered", "rocks_with_lost_cells",
                "median_first_covered_s", "slowest_first_covered_s",
                "retracted_voxels"]
    lines += ["| measure | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    for key in headline:
        row = []
        for n in names:
            v = payload["maps"][n].get(key)
            row.append("-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        lines.append(f"| {key} | " + " | ".join(row) + " |")
    lines += ["", "## Per rock", "",
              "Occupied = the map claims that cell at all. Attributed = its "
              "representative lies on this rock's 3D surface. Times are "
              "seconds after a real return first landed on the rock, so they "
              "measure the model rather than the robot's route; t50 is when "
              "half the rock's final coverage was reached, and `fell` counts "
              "the times it dropped back under that level afterwards.", "",
              "| rock | cells | " + " | ".join(
                  f"{n} occupied | {n} attributed | {n} t50 (s) | {n} fell" for n in names)
              + " |",
              "|---|---|" + "---|" * (4 * len(names))]
    by = {(r["map"], r["rock_id"]): r for r in per_rock}
    for rid in sorted({r["rock_id"] for r in per_rock}):
        cells = [f"{by[(n, rid)]['visible_cells']}" for n in names][0]
        row = []
        for n in names:
            r = by[(n, rid)]
            row += [_fmt(r["occupied_coverage"], 3), _fmt(r["coverage"], 3),
                    _fmt(r.get("t50_cover_s"), 0), str(r.get("fell_below_50", "-"))]
        lines.append(f"| {rid} | {cells} | " + " | ".join(row) + " |")
    lines += ["", "Coverage denominators are ground cells a real return actually "
                  "landed on inside each rock, unfiltered and shared by both maps.",
              "",
              "`threshold-frontier.csv` is each map read at every distinct "
              "score it holds - not a grid, and not stopping at 0.95 - so two "
              "checkpoints can be compared at equal wrongly-claimed area "
              "rather than at whichever threshold each happens to carry. A "
              "threshold chosen by reading it is an oracle diagnostic on this "
              "recording, not an operating point. For the cleaned map it is "
              "weaker still: the cleanup policy only ever judged voxels above "
              "the threshold it actually ran at, so the finished map read at "
              "another threshold is a frozen-map diagnostic, not what that "
              "policy would have produced.\n",
              "Coverage is reported two ways. **Occupied footprint** is "
              "whether the map claims that piece of a rock's ground at all, "
              "which is what a navigation stack drives on. **3D attribution** "
              "is whether the thing it claims there is that rock's own "
              "surface. Retracting a floating positive over a rock can raise "
              "the second while leaving the first untouched, and reporting "
              "only one of them hides which happened.", "",
              "`map.png` is the same thing looking down: aggregates cannot "
              "show whether the wrongly-claimed ground is scattered or piled "
              "in two places."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def evidence_settings_from_args(args) -> EvidenceSettings | None:
    """Build the cleanup settings from parsed CLI arguments, or None for off."""
    if not getattr(args, "clean", False):
        return None
    known = {f.name for f in fields(EvidenceSettings)}
    kw = {k: v for k, v in vars(args).items() if k in known and v is not None}
    kw["enabled"] = True
    return EvidenceSettings(**kw)
