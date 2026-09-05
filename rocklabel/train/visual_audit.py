"""Deployment-first visual comparison of two checkpoints on labelled recordings.

The ordinary training reports answer whether a model ranks held-out volleyball
examples correctly.  This audit answers the operational question: after several
seconds of predictions have accumulated, which *physical rocks* have a complete,
usable map?  Every rock is weighted once, and the report includes deterministic
top-down evidence so a person (or an agent with image inspection) can verify the
numbers instead of reasoning from PR-AUC alone.
"""

from __future__ import annotations

import csv
import heapq
import json
import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..dataset.labeling import LABEL_CLEAR, inside_arena, label_rocks, points_in_rock
from ..geometry.leveling import check_level_match, level_record, pin_level_to_labels
from ..labels import LabelSet, Rock, load_labels
from ..recording.pipeline import ScanStream, WindowedScanStream


@dataclass
class AuditFrame:
    index: int
    time_s: float
    base: np.ndarray
    xyz: np.ndarray
    intensity: np.ndarray


@dataclass
class AuditInterval:
    sequence: int
    start_s: float
    end_s: float
    frames: list[AuditFrame]
    visible_points: dict[int, int] = field(default_factory=dict)
    visible_cells: dict[int, int] = field(default_factory=dict)


@dataclass
class CellMap:
    positions: np.ndarray
    probabilities: np.ndarray
    cell_ids: np.ndarray


@dataclass
class ModelResult:
    path: str
    name: str
    task: str
    threshold: float
    cells: CellMap
    hot_cells: set[tuple[int, int]]
    false_cells: int
    false_components: int


def cell_ids(xy: np.ndarray, cell_m: float) -> np.ndarray:
    """Stable integer ground-cell ids for ``xy`` positions."""
    return np.floor(np.asarray(xy, float) / float(cell_m)).astype(np.int64)


def cell_set(xy: np.ndarray, cell_m: float) -> set[tuple[int, int]]:
    return {tuple(v) for v in cell_ids(xy, cell_m)}


def connected_components(cells: Iterable[tuple[int, int]]) -> int:
    """Number of 8-connected components in a set of ground cells."""
    remaining = set(cells)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    n = (x + dx, y + dy)
                    if n in remaining:
                        remaining.remove(n)
                        stack.append(n)
    return count


class PredictionAccumulator:
    """The live map's newest-value-per-native-voxel accumulation contract."""

    def __init__(self, native_voxel_m: float):
        self.voxel = float(native_voxel_m)
        self._map: dict[tuple[int, int, int], tuple[np.ndarray, float]] = {}

    def update(self, positions: np.ndarray, probabilities: np.ndarray) -> None:
        if len(positions) == 0:
            return
        keys = np.floor(np.asarray(positions) / self.voxel).astype(np.int64)
        for key, pos, prob in zip(keys, positions, probabilities):
            self._map[tuple(key)] = (np.asarray(pos, float), float(prob))

    def reduce(self, cell_m: float) -> CellMap:
        """Reduce native voxels to shared ground cells, keeping max confidence."""
        if not self._map:
            return CellMap(np.empty((0, 3)), np.empty(0), np.empty((0, 2), np.int64))
        positions = np.stack([v[0] for v in self._map.values()])
        probabilities = np.array([v[1] for v in self._map.values()], float)
        ids = cell_ids(positions[:, :2], cell_m)
        chosen: dict[tuple[int, int], int] = {}
        for i, (key, prob) in enumerate(zip(map(tuple, ids), probabilities)):
            old = chosen.get(key)
            if old is None or prob > probabilities[old]:
                chosen[key] = i
        take = np.fromiter(chosen.values(), dtype=np.int64)
        return CellMap(positions[take], probabilities[take], ids[take])


def rock_metrics(raw_xyz: np.ndarray, rock: Rock, result: ModelResult,
                 cell_m: float, min_coverage: float) -> dict:
    """Completeness and fragmentation for one physical rock in one interval."""
    raw_on = raw_xyz[points_in_rock(raw_xyz, rock)]
    raw_cells = cell_set(raw_on[:, :2], cell_m) if len(raw_on) else set()
    if len(result.cells.positions):
        pred_on = points_in_rock(result.cells.positions, rock)
        pred_cells = {tuple(v) for v in result.cells.cell_ids[
            pred_on & (result.cells.probabilities >= result.threshold)]}
    else:
        pred_cells = set()
    covered = raw_cells & pred_cells
    coverage = len(covered) / max(len(raw_cells), 1)
    return {
        "visible_points": int(len(raw_on)),
        "visible_cells": int(len(raw_cells)),
        "positive_cells": int(len(pred_cells)),
        "covered_cells": int(len(covered)),
        "coverage": float(coverage),
        "detected": bool(coverage >= min_coverage),
        "fragments": int(connected_components(covered)),
    }


def _crop_limits(generators: list[dict]) -> dict[str, float]:
    """Intersection of checkpoint crop boxes: neither model gets extra context."""
    return {
        "forward": min(float(g["crop_forward_m"]) for g in generators),
        "backward": min(float(g["crop_backward_m"]) for g in generators),
        "left": min(float(g["crop_left_m"]) for g in generators),
        "right": min(float(g["crop_right_m"]) for g in generators),
        "down": min(float(g["crop_down_m"]) for g in generators),
        "up": min(float(g["crop_up_m"]) for g in generators),
    }


def _crop_frame(scan, limits: dict[str, float], floor_z: float,
                floor_band: tuple[float, float], max_range: float | None) -> AuditFrame | None:
    base = scan.T_odom_base[:3, 3].astype(float)
    xyz = scan.xyz_odom
    lo = np.array([base[0] - limits["backward"], base[1] - limits["right"],
                   max(base[2] - limits["down"], floor_z + floor_band[0])])
    hi = np.array([base[0] + limits["forward"], base[1] + limits["left"],
                   min(base[2] + limits["up"], floor_z + floor_band[1])])
    keep = ((xyz >= lo) & (xyz <= hi)).all(axis=1)
    if max_range is not None and max_range > 0:
        dxy = xyz[:, :2] - base[:2]
        keep &= (dxy * dxy).sum(axis=1) <= max_range * max_range
    if not keep.any():
        return None
    return AuditFrame(int(scan.index), float(scan.time_s), base,
                      xyz[keep].astype(np.float64),
                      scan.intensity[keep].astype(np.float32))


def _finish_interval(sequence: int, frames: list[AuditFrame], labels: LabelSet,
                     cell_m: float) -> AuditInterval | None:
    if not frames:
        return None
    cloud = np.concatenate([f.xyz for f in frames])
    points, cells = {}, {}
    for rock in labels.rocks:
        on = cloud[points_in_rock(cloud, rock)]
        if len(on):
            points[rock.id] = int(len(on))
            cells[rock.id] = int(len(cell_set(on[:, :2], cell_m)))
    return AuditInterval(sequence, frames[0].time_s, frames[-1].time_s,
                         list(frames), points, cells)


def collect_intervals(recording: str, labels: LabelSet, labels_path: str, cfg: dict,
                      generators: list[dict], floor_band: tuple[float, float],
                      max_range: float | None, stride: int, window_s: float,
                      accum_seconds: float, candidates_per_rock: int,
                      min_visible_points: int, start_s: float | None,
                      end_s: float | None, cell_m: float) -> tuple[list[AuditInterval], float]:
    """Decode once and retain visibility-rich accumulated intervals per rock."""
    stream = ScanStream(recording, cfg, stride=1, progress=True, desc="visual audit")
    check_level_match(labels.level, level_record(stream), labels_path)
    solution = getattr(stream, "solution", None)
    floor_z = getattr(solution, "floor_z", None)
    if floor_z is None and labels.level:
        floor_z = labels.level.get("floor_z")
    if floor_z is None:
        raise ValueError("the floor-relative audit needs a measured floor; use labelled "
                         "levelling or a config whose level mode measures ground")
    limits = _crop_limits(generators)
    scans = WindowedScanStream(stream, window_s) if window_s > 0 else stream

    # A small heap per physical rock prevents one frequently visible rock from
    # owning the entire audit. Intervals can be shared by several heaps.
    best: dict[int, list[tuple[int, int, AuditInterval]]] = {
        r.id: [] for r in labels.rocks
    }
    origin = None
    current_bucket = None
    current: list[AuditFrame] = []
    sequence = 0

    def offer(interval: AuditInterval | None) -> None:
        if interval is None:
            return
        for rock in labels.rocks:
            visible = interval.visible_points.get(rock.id, 0)
            if visible < min_visible_points:
                continue
            heap = best[rock.id]
            item = (interval.visible_cells.get(rock.id, 0), interval.sequence, interval)
            if len(heap) < candidates_per_rock:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)

    for k, scan in enumerate(scans):
        if k % max(int(stride), 1):
            continue
        if origin is None:
            origin = float(scan.time_s)
        rel = float(scan.time_s) - origin
        if start_s is not None and rel < start_s:
            continue
        if end_s is not None and rel > end_s:
            break
        bucket = int((rel - (start_s or 0.0)) // accum_seconds)
        if current_bucket is None:
            current_bucket = bucket
        if bucket != current_bucket:
            offer(_finish_interval(sequence, current, labels, cell_m))
            sequence += 1
            current, current_bucket = [], bucket
        frame = _crop_frame(scan, limits, float(floor_z), floor_band, max_range)
        if frame is not None:
            # Reports and case names use recording-relative time. Absolute ROS
            # stamps are useless to a person seeking the same replay position.
            frame.time_s = rel
            current.append(frame)
    offer(_finish_interval(sequence, current, labels, cell_m))

    selected: dict[int, AuditInterval] = {}
    for heap in best.values():
        for _count, _seq, interval in heap:
            selected[interval.sequence] = interval
    return [selected[k] for k in sorted(selected)], float(floor_z)


def _load_checkpoint(path: str, device):
    import torch
    from .models import build_model, model_task

    ck = torch.load(path, map_location="cpu", weights_only=False)
    c, g = ck["config"], ck["generator"]
    model = build_model(
        c["model"], tnet=c.get("tnet", False), dropout=c.get("dropout"),
        features=c.get("features"), seg_npoints=c.get("seg_npoints"),
        seg_radii=c.get("seg_radii"), seg_height_ref=c.get("seg_height_ref"),
        seg_coord_ref=c.get("seg_coord_ref"), bev_cell=c.get("bev_cell"),
        bev_grid=c.get("bev_grid"), bev_width=c.get("bev_width"),
        bev_depth=c.get("bev_depth"), bev_channels=c.get("bev_channels"),
        bev_density_norm=c.get("bev_density_norm"))
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    return {
        "path": os.path.abspath(path), "name": c["model"],
        "task": model_task(c["model"]), "threshold": float(ck.get("threshold", 0.5)),
        "generator": g, "model": model,
    }


def _score_interval(interval: AuditInterval, loaded: dict, device, batch: int,
                    cell_m: float, labels: LabelSet) -> ModelResult:
    from .mcapview import _score_balls, _score_frame

    g, model = loaded["generator"], loaded["model"]
    accum = PredictionAccumulator(float(g["centers_voxel_m"]))
    for frame in interval.frames:
        rng = np.random.default_rng([int(g["seed"]), frame.index])
        if loaded["task"] == "classify":
            pos, prob = _score_balls(frame.xyz, frame.intensity, g, model,
                                     device, rng, batch)
        else:
            pos, prob = _score_frame(frame.xyz, frame.intensity, frame.base, g,
                                     model, device, rng)
        accum.update(pos, prob)
    cells = accum.reduce(cell_m)
    if len(cells.positions) and labels.arena is not None:
        keep = inside_arena(cells.positions, labels.arena)
        cells = CellMap(cells.positions[keep], cells.probabilities[keep], cells.cell_ids[keep])
    hot = cells.probabilities >= loaded["threshold"]
    hot_cells = {tuple(v) for v in cells.cell_ids[hot]}
    if len(cells.positions):
        lab = label_rocks(cells.positions, labels.rocks,
                          float(g.get("boundary_shell_m", 0.05)))
        false = {tuple(v) for v in cells.cell_ids[hot & (lab == LABEL_CLEAR)]}
    else:
        false = set()
    return ModelResult(loaded["path"], loaded["name"], loaded["task"],
                       loaded["threshold"], cells, hot_cells, len(false),
                       connected_components(false))


def _rock_outline(ax, rock: Rock, color: str, linewidth: float = 1.5) -> None:
    from matplotlib.patches import Circle, Polygon, Rectangle

    if rock.shape == "polygon":
        patch = Polygon(np.asarray(rock.vertices), closed=True, fill=False,
                        edgecolor=color, linewidth=linewidth)
    elif rock.shape == "box":
        half = np.asarray(rock.size[:2], float) / 2.0
        patch = Rectangle(np.asarray(rock.center[:2]) - half, *(2.0 * half),
                          fill=False, edgecolor=color, linewidth=linewidth)
    else:
        patch = Circle(rock.center[:2], rock.radius, fill=False,
                       edgecolor=color, linewidth=linewidth)
    ax.add_patch(patch)


def _draw_panel(ax, raw: np.ndarray, labels: LabelSet, target: Rock,
                result: ModelResult | None, title: str, limits=None) -> None:
    # Deterministic display cap; metrics always use the complete arrays.
    if len(raw) > 60_000:
        raw = raw[np.linspace(0, len(raw) - 1, 60_000, dtype=int)]
    ax.scatter(raw[:, 0], raw[:, 1], s=0.5, c="#aaa9a3", alpha=0.25,
               linewidths=0, rasterized=True)
    if result is not None and len(result.cells.positions):
        p = result.cells.probabilities
        ax.scatter(result.cells.positions[:, 0], result.cells.positions[:, 1],
                   s=3.0, c=p, cmap="turbo", vmin=0.0, vmax=1.0,
                   linewidths=0, rasterized=True)
    for rock in labels.rocks:
        _rock_outline(ax, rock, "#00a6d6" if rock.id != target.id else "#ffe600",
                      1.0 if rock.id != target.id else 2.5)
        if rock.id == target.id:
            ax.text(rock.center[0], rock.center[1], f"  rock {rock.id}",
                    color="#8a6d00", fontsize=9, weight="bold")
    ax.set_title(title)
    ax.set_aspect("equal")
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    ax.grid(False)


def render_case(path: str, interval: AuditInterval, target: Rock, labels: LabelSet,
                results: list[ModelResult], metrics: list[dict], floor_band,
                cell_m: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = np.concatenate([f.xyz for f in interval.frames])
    if labels.arena is not None:
        v = np.asarray(labels.arena)
        full = (v[:, 0].min() - .3, v[:, 0].max() + .3,
                v[:, 1].min() - .3, v[:, 1].max() + .3)
    else:
        full = (raw[:, 0].min(), raw[:, 0].max(), raw[:, 1].min(), raw[:, 1].max())
    margin = max(float(target.radius) + .7, 1.0)
    zoom = (target.center[0] - margin, target.center[0] + margin,
            target.center[1] - margin, target.center[1] + margin)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    duration = interval.end_s - interval.start_s
    raw_title = (f"Ground truth · rock {target.id}\n{metrics[0]['visible_points']} visible "
                 f"returns · {metrics[0]['visible_cells']} occupied cells")
    _draw_panel(axes[0, 0], raw, labels, target, None, raw_title, full)
    _draw_panel(axes[1, 0], raw, labels, target, None, "Ground-truth zoom", zoom)
    for column, (result, metric) in enumerate(zip(results, metrics), 1):
        title = (f"{result.name} ({result.task}) · threshold {result.threshold:.3f}\n"
                 f"coverage {metric['coverage']:.0%} · {metric['fragments']} fragments · "
                 f"{result.false_cells} false cells")
        _draw_panel(axes[0, column], raw, labels, target, result, title, full)
        _draw_panel(axes[1, column], raw, labels, target, result,
                    f"Rock {target.id} zoom · {metric['covered_cells']}/"
                    f"{metric['visible_cells']} cells covered", zoom)
    fig.suptitle(f"Visual deployment audit · {duration:.1f}s accumulated · "
                 f"floor band {floor_band[0]:+.2f}…{floor_band[1]:+.2f} m · "
                 f"cell {cell_m:.2f} m", fontsize=13)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _summarize_rocks(observations: list[dict], model_keys: list[str],
                     min_coverage: float) -> list[dict]:
    rows = []
    for rock_id in sorted({o["rock_id"] for o in observations}):
        obs = [o for o in observations if o["rock_id"] == rock_id]
        row = {"rock_id": rock_id, "intervals": len(obs),
               "max_visible_points": max(o["visible_points"] for o in obs)}
        for key in model_keys:
            cov = np.array([o[key]["coverage"] for o in obs], float)
            row[key] = {
                "median_coverage": float(np.median(cov)),
                "best_coverage": float(cov.max()),
                "worst_coverage": float(cov.min()),
                "detected_intervals": int((cov >= min_coverage).sum()),
                "detected_any": bool((cov >= min_coverage).any()),
            }
        rows.append(row)
    return rows


def select_cases(observations: list[dict], model_keys: list[str],
                 max_cases: int) -> list[dict]:
    """One strongest disagreement per physical rock, then globally ranked."""
    chosen = []
    for rock_id in sorted({o["rock_id"] for o in observations}):
        rows = [o for o in observations if o["rock_id"] == rock_id]
        def rank(o):
            a, b = o[model_keys[0]], o[model_keys[1]]
            miss = int(a["detected"] != b["detected"])
            return (miss, abs(a["coverage"] - b["coverage"]),
                    o["visible_cells"], -o["interval_sequence"])
        chosen.append(max(rows, key=rank))
    chosen.sort(key=lambda o: (
        int(o[model_keys[0]]["detected"] != o[model_keys[1]]["detected"]),
        abs(o[model_keys[0]]["coverage"] - o[model_keys[1]]["coverage"]),
        o["visible_cells"]), reverse=True)
    return chosen[:max_cases]


def comparison_summary(observations: list[dict], per_rock: list[dict],
                       model_keys: list[str], material_gap: float) -> dict:
    """Transparent parity check based on distinct-rock median completeness.

    A binary contact test can call 26% and 100% coverage the same outcome.  The
    material-gap count deliberately retains that difference.  It rejects a
    claim of parity when even one labelled obstacle has a predeclared, large
    completeness gap; it does *not* turn that into a blanket architecture win.
    """
    a, b = model_keys
    wins = {a: [], b: []}
    similar = []
    detection_advantage = {a: [], b: []}
    for row in per_rock:
        rock_id = int(row["rock_id"])
        delta = row[a]["median_coverage"] - row[b]["median_coverage"]
        if delta >= material_gap:
            wins[a].append(rock_id)
        elif delta <= -material_gap:
            wins[b].append(rock_id)
        else:
            similar.append(rock_id)
        detected_a = row[a]["detected_any"]
        detected_b = row[b]["detected_any"]
        if detected_a and not detected_b:
            detection_advantage[a].append(rock_id)
        elif detected_b and not detected_a:
            detection_advantage[b].append(rock_id)

    # False-cell counts describe the whole accumulated interval, so deduplicate
    # intervals that happen to contain several labelled rocks before aggregating.
    by_interval: dict[int, dict] = {}
    for row in observations:
        by_interval.setdefault(int(row["interval_sequence"]), row)
    false_cells = {}
    for key in model_keys:
        values = np.array([row[key]["false_cells"] for row in by_interval.values()], float)
        false_cells[key] = {
            "median_per_interval": float(np.median(values)),
            "max_per_interval": int(values.max()),
        }

    mismatch = bool(wins[a] or wins[b] or
                    detection_advantage[a] or detection_advantage[b])
    return {
        "verdict": "does_not_support_parity" if mismatch else "no_material_gap_found",
        "material_gap": float(material_gap),
        "audited_rocks": len(per_rock),
        "coverage_leads": wins,
        "within_material_gap": similar,
        "persistent_detection_advantage": detection_advantage,
        "false_cells": false_cells,
    }


def _write_reports(out_dir: str, settings: dict, checkpoints: list[dict],
                   observations: list[dict], per_rock: list[dict], cases: list[dict],
                   comparison: dict, model_keys: list[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    payload = {"settings": settings, "checkpoints": checkpoints,
               "observations": observations, "per_rock": per_rock,
               "comparison": comparison, "cases": cases}
    with open(os.path.join(out_dir, "audit.json"), "w") as f:
        json.dump(payload, f, indent=2)

    with open(os.path.join(out_dir, "observations.csv"), "w", newline="") as f:
        fields = ["rock_id", "interval_sequence", "start_s", "end_s",
                  "visible_points", "visible_cells"]
        for key in model_keys:
            fields += [f"{key}_coverage", f"{key}_detected", f"{key}_fragments",
                       f"{key}_false_cells"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for o in observations:
            row = {k: o[k] for k in fields[:6]}
            for key in model_keys:
                row[f"{key}_coverage"] = o[key]["coverage"]
                row[f"{key}_detected"] = o[key]["detected"]
                row[f"{key}_fragments"] = o[key]["fragments"]
                row[f"{key}_false_cells"] = o[key]["false_cells"]
            writer.writerow(row)

    with open(os.path.join(out_dir, "per-rock.csv"), "w", newline="") as f:
        fields = ["rock_id", "intervals", "max_visible_points"]
        for key in model_keys:
            fields += [f"{key}_median_coverage", f"{key}_best_coverage",
                       f"{key}_worst_coverage", f"{key}_detected_intervals",
                       f"{key}_detected_any"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in per_rock:
            row = {k: r[k] for k in fields[:3]}
            for key in model_keys:
                for metric in ("median_coverage", "best_coverage", "worst_coverage",
                               "detected_intervals", "detected_any"):
                    row[f"{key}_{metric}"] = r[key][metric]
            writer.writerow(row)

    lines = ["# Visual deployment audit\n\n",
             "This report evaluates accumulated maps on **distinct physical rocks**. ",
             "Every rock contributes one row; repeated sightings do not increase its weight. ",
             f"Coverage is the fraction of the rock's occupied "
             f"{settings['cell_m'] * 100:g} cm ground cells receiving a ",
             "prediction above the checkpoint's stored threshold. PR-AUC is intentionally not ",
             "used as a substitute for map completeness.\n\n",
             f"Recording: `{settings['recording']}`  \n",
             f"Labels: `{settings['labels']}`  \n",
             f"Floor band: `{settings['floor_band'][0]:+.2f}…"
             f"{settings['floor_band'][1]:+.2f} m`  \n",
             f"Accumulation: `{settings['accum_seconds']:.1f} s`  \n",
             f"Complete detection threshold: `{settings['min_coverage']:.0%}` coverage\n\n",
             "| model | task | stored threshold | checkpoint |\n",
             "|---|---|---:|---|\n"]
    for i, ck in enumerate(checkpoints):
        lines.append(f"| model {i + 1} | {ck['task']} | {ck['threshold']:.3f} | "
                     f"`{ck['path']}` |\n")
    lines += ["\n",
             "| rock | intervals | visible returns | "]
    for i, ck in enumerate(checkpoints):
        lines.append(f"model {i + 1} median / best / detected | ")
    lines.append("\n|---:|---:|---:|" + "---:|" * len(checkpoints) + "\n")
    for row in per_rock:
        lines.append(f"| {row['rock_id']} | {row['intervals']} | "
                     f"{row['max_visible_points']} | ")
        for key in model_keys:
            m = row[key]
            lines.append(f"{m['median_coverage']:.0%} / {m['best_coverage']:.0%} / "
                         f"{m['detected_intervals']}/{row['intervals']} | ")
        lines.append("\n")
    a, b = model_keys
    verdict = comparison["verdict"]
    lines += ["\n## Automated parity check\n\n",
              ("**Result: the evidence does not support describing these deployed maps as "
               "similar.**\n\n" if verdict == "does_not_support_parity" else
               "**Result: no predeclared material completeness gap was found. This does "
               "not prove that the architectures are generally equivalent.**\n\n"),
              f"A material difference was fixed at "
              f"**{comparison['material_gap']:.0%} median rock coverage** before inspecting "
              "the selected images. ",
              f"Model 1 leads on {len(comparison['coverage_leads'][a])}/"
              f"{comparison['audited_rocks']} distinct rocks; model 2 leads on "
              f"{len(comparison['coverage_leads'][b])}/"
              f"{comparison['audited_rocks']}; "
              f"{len(comparison['within_material_gap'])}/"
              f"{comparison['audited_rocks']} are within the gap.\n\n",
              f"- Model 1 coverage leads: "
              f"`{comparison['coverage_leads'][a] or 'none'}`\n",
              f"- Model 2 coverage leads: "
              f"`{comparison['coverage_leads'][b] or 'none'}`\n",
              f"- Model 1 persistent detection-only rocks: "
              f"`{comparison['persistent_detection_advantage'][a] or 'none'}`\n",
              f"- Model 2 persistent detection-only rocks: "
              f"`{comparison['persistent_detection_advantage'][b] or 'none'}`\n",
              f"- Median clear-region positive cells per interval: model 1 "
              f"`{comparison['false_cells'][a]['median_per_interval']:.1f}`, model 2 "
              f"`{comparison['false_cells'][b]['median_per_interval']:.1f}`\n\n",
              "Coverage and clear-region positives are separate tradeoffs. A completeness "
              "lead is evidence that the live maps are not equivalent, not permission to "
              "ignore false detections or declare an architecture universally superior.\n\n",
              "## Automatically selected disagreements\n\n"]
    for case in cases:
        lines.append(f"- [Rock {case['rock_id']} · interval "
                     f"{case['interval_sequence']}](cases/{case['image']})\n")
    lines += ["\n## Interpretation guardrails\n\n",
              "- A narrow crop is a diagnostic ablation, not proof that the full operational "
              "map works.\n",
              "- Stored-threshold completeness and threshold-free ranking answer different "
              "questions.\n",
              "- Native outputs differ (candidate centers versus sampled points); both are "
              "reduced to the same ground-cell size only after their real inference path.\n",
              "- Inspect the PNGs before claiming that the models are operationally similar.\n"]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("".join(lines))


def run_visual_audit(recording: str, labels_path: str, model_a: str, model_b: str,
                     out_dir: str, config_path: str | None = None,
                     floor_band: tuple[float, float] = (-0.10, 0.60),
                     max_range: float | None = 8.0, stride: int | None = None,
                     window_s: float | None = None, accum_seconds: float = 5.0,
                     candidates_per_rock: int = 3, min_visible_points: int = 15,
                     min_coverage: float = 0.25, max_cases: int = 12,
                     material_gap: float = 0.20, cell_m: float = 0.10,
                     start_s: float | None = None,
                     end_s: float | None = None, device: str | None = None,
                     batch: int = 512) -> dict:
    """Run the complete audit and return the same payload written to JSON."""
    import torch
    from ..config import load_config

    if accum_seconds <= 0 or cell_m <= 0:
        raise ValueError("accumulation seconds and cell size must be positive")
    if candidates_per_rock <= 0 or max_cases <= 0 or min_visible_points <= 0:
        raise ValueError("candidate, case, and visibility counts must be positive")
    if not 0 < min_coverage <= 1:
        raise ValueError("min coverage must be in (0, 1]")
    if not 0 < material_gap <= 1:
        raise ValueError("material coverage gap must be in (0, 1]")
    floor_band = tuple(sorted(map(float, floor_band)))
    if np.isclose(floor_band[0], floor_band[1]):
        raise ValueError("floor band must have non-zero height")
    labels = load_labels(labels_path)
    if not labels.rocks:
        raise ValueError(f"{labels_path} contains no rock labels")
    cfg = pin_level_to_labels(load_config(config_path), labels.level)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loaded = [_load_checkpoint(model_a, dev), _load_checkpoint(model_b, dev)]
    generators = [m["generator"] for m in loaded]
    windows = [float(g.get("frame_window_s") or 0.0) for g in generators]
    if window_s is None:
        if not np.isclose(windows[0], windows[1]):
            raise ValueError(f"checkpoints trained with different frame windows {windows}; "
                             "pass --window-s explicitly and report the mismatch")
        window_s = windows[0]
    strides = [int(g.get("frame_stride") or 1) for g in generators]
    if stride is None:
        if strides[0] != strides[1]:
            raise ValueError(f"checkpoints trained with different strides {strides}; "
                             "pass --stride explicitly and report the mismatch")
        stride = strides[0]

    intervals, floor_z = collect_intervals(
        recording, labels, labels_path, cfg, generators, floor_band, max_range,
        stride, window_s, accum_seconds, candidates_per_rock,
        min_visible_points, start_s, end_s, cell_m)
    if not intervals:
        raise ValueError("no interval contained a sufficiently visible labeled rock")
    print(f"selected {len(intervals)} visibility-rich accumulated intervals", flush=True)

    model_keys = ["model_a", "model_b"]
    scored: dict[int, list[ModelResult]] = {}
    observations = []
    for number, interval in enumerate(intervals, 1):
        print(f"score interval {number}/{len(intervals)}", flush=True)
        results = [_score_interval(interval, model, dev, batch, cell_m, labels)
                   for model in loaded]
        scored[interval.sequence] = results
        raw = np.concatenate([f.xyz for f in interval.frames])
        for rock in labels.rocks:
            if interval.visible_points.get(rock.id, 0) < min_visible_points:
                continue
            metrics = [rock_metrics(raw, rock, result, cell_m, min_coverage)
                       for result in results]
            obs = {
                "rock_id": rock.id, "interval_sequence": interval.sequence,
                "start_s": interval.start_s, "end_s": interval.end_s,
                "visible_points": interval.visible_points[rock.id],
                "visible_cells": interval.visible_cells[rock.id],
            }
            for key, metric, result in zip(model_keys, metrics, results):
                obs[key] = metric | {"false_cells": result.false_cells,
                                     "false_components": result.false_components}
            observations.append(obs)

    per_rock = _summarize_rocks(observations, model_keys, min_coverage)
    comparison = comparison_summary(observations, per_rock, model_keys, material_gap)
    cases = select_cases(observations, model_keys, max_cases)
    case_dir = os.path.join(out_dir, "cases")
    interval_by_seq = {i.sequence: i for i in intervals}
    for case in cases:
        interval = interval_by_seq[case["interval_sequence"]]
        rock = labels.get(case["rock_id"])
        filename = f"rock-{rock.id:03d}-interval-{interval.sequence:04d}.png"
        case["image"] = filename
        results = scored[interval.sequence]
        render_case(os.path.join(case_dir, filename), interval, rock, labels, results,
                    [case[k] for k in model_keys], floor_band, cell_m)

    settings = {
        "recording": os.path.abspath(recording), "labels": os.path.abspath(labels_path),
        "out": os.path.abspath(out_dir), "floor_band": list(map(float, floor_band)),
        "floor_z": floor_z, "max_range": max_range, "stride": stride,
        "window_s": window_s, "accum_seconds": accum_seconds,
        "candidates_per_rock": candidates_per_rock,
        "min_visible_points": min_visible_points, "min_coverage": min_coverage,
        "material_gap": material_gap, "cell_m": cell_m,
        "start_s": start_s, "end_s": end_s,
        "device": str(dev),
    }
    checkpoints = [{k: m[k] for k in ("path", "name", "task", "threshold")}
                   for m in loaded]
    _write_reports(out_dir, settings, checkpoints, observations, per_rock, cases,
                   comparison, model_keys)
    print(f"wrote visual audit to {out_dir}")
    return {"settings": settings, "checkpoints": checkpoints,
            "observations": observations, "per_rock": per_rock,
            "comparison": comparison, "cases": cases}
