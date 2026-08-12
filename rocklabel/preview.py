"""`rocklabel preview`: interactively browse the frames of a generated dataset.

Reads only the dataset directory (manifest + npz files), no mcap needed:
reconstructs each frame's BEV surface from the stored channels, colors it by
the stored label mask, and overlays the format-A sample centers plus the
original label spheres (if the labels JSON still exists). What you see is
literally what was written to disk, so this is the ground-truth check for
`generate`.

The viewer is a frame browser: skip through frames with the transport bar
(prev/next/play/slider) or the arrow keys instead of relaunching per frame.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np

from .generate import MANIFEST_NAME

# Point colors (RGB in 0..1)
COLOR_CELL_ROCK = (0.95, 0.20, 0.15)     # red
COLOR_CELL_CLEAR = (0.55, 0.55, 0.58)    # gray
COLOR_CELL_IGNORE = (0.65, 0.58, 0.15)   # olive
COLOR_CENTER_ROCK = (1.00, 1.00, 0.20)   # yellow
COLOR_CENTER_CLEAR = (0.25, 0.75, 1.00)  # cyan

LEGEND_LINES = [
    ("BEV cell · clear", COLOR_CELL_CLEAR),
    ("BEV cell · ROCK", COLOR_CELL_ROCK),
    ("BEV cell · ignore (shell)", COLOR_CELL_IGNORE),
    ("sample center · clear", COLOR_CENTER_CLEAR),
    ("sample center · ROCK", COLOR_CENTER_ROCK),
    ("label sphere (wireframe)", COLOR_CELL_ROCK),
]

LEGEND = "legend:\n" + "\n".join(f"  {t}" for t, _ in LEGEND_LINES)


def _frame_indices(directory: str) -> list[int]:
    out = []
    if os.path.isdir(directory):
        for name in os.listdir(directory):
            m = re.fullmatch(r"frame_(\d{6})\.npz", name)
            if m:
                out.append(int(m.group(1)))
    return sorted(out)


@dataclass
class _Dataset:
    out_dir: str
    run_id: str
    frames: list
    gcfg: dict
    spheres: list
    labels_path: str

    @property
    def bev_dir(self) -> str:
        return os.path.join(self.out_dir, "bev", self.run_id)

    @property
    def points_dir(self) -> str:
        return os.path.join(self.out_dir, "points", self.run_id)


def open_dataset(out_dir: str, run_id: str | None = None) -> _Dataset:
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        raise SystemExit(f"{out_dir!r} has no {MANIFEST_NAME} - is this a dataset directory "
                         "produced by 'rocklabel generate'?")
    with open(manifest_path) as f:
        manifest = json.load(f)
    runs = sorted(manifest["runs"])
    if run_id is None:
        if len(runs) != 1:
            raise SystemExit(f"Dataset contains several runs; pick one with --run: {runs}")
        run_id = runs[0]
    if run_id not in manifest["runs"]:
        raise SystemExit(f"run {run_id!r} not in this dataset; available: {runs}")

    ds = _Dataset(out_dir, run_id, [], manifest["config"]["generator"], [],
                  manifest["runs"][run_id].get("labels_path", ""))
    ds.frames = _frame_indices(ds.bev_dir)
    if not ds.frames:
        raise SystemExit(f"No frames found under {ds.bev_dir}")
    if ds.labels_path and os.path.exists(ds.labels_path):
        from .labels import load_labels
        ds.spheres = [(rock.center, rock.radius) for rock in load_labels(ds.labels_path).rocks]
    return ds


def load_frame(out_dir: str, run_id: str | None = None, frame: int | None = None) -> dict:
    """Load one generated frame back from disk into plottable arrays.

    Compatibility entry point (used by tests/scripts): resolves the run,
    defaults to the middle frame, and includes the run's frame list and label
    spheres in the returned dict. The interactive browser uses
    :func:`open_dataset` + :func:`_load_frame_arrays` directly.
    """
    ds = open_dataset(out_dir, run_id)
    if frame is None:
        frame = ds.frames[len(ds.frames) // 2]  # middle of the run: usually plenty in view
    if frame not in ds.frames:
        raise SystemExit(f"frame {frame} not in run {ds.run_id!r}; available: "
                         f"{ds.frames[:10]}{'...' if len(ds.frames) > 10 else ''} (use --list)")
    data = _load_frame_arrays(ds, frame)
    data.update({
        "run_id": ds.run_id,
        "frames": ds.frames,
        "spheres": ds.spheres,
        "labels_path": ds.labels_path,
    })
    return data


def _load_frame_arrays(ds: _Dataset, frame: int) -> dict:
    """Load one frame's npz files into plottable arrays."""
    gcfg = ds.gcfg
    bev = np.load(os.path.join(ds.bev_dir, f"frame_{frame:06d}.npz"))
    channels, mask, pose = bev["channels"], bev["label_mask"], bev["robot_pose"]
    base = pose[:3, 3]
    cell = gcfg["bev_cell_m"]
    x0 = base[0] - gcfg["crop_backward_m"]
    y0 = base[1] - gcfg["crop_right_m"]

    ii, jj = np.nonzero(channels[0] > 0)  # valid cells
    cells_xyz = np.column_stack([
        x0 + (ii + 0.5) * cell,
        y0 + (jj + 0.5) * cell,
        base[2] + channels[2, ii, jj],  # top-of-surface height per cell
    ])
    cell_labels = mask[ii, jj]
    cells_rgb = np.empty((len(ii), 3))
    cells_rgb[:] = COLOR_CELL_CLEAR
    cells_rgb[cell_labels == 255] = COLOR_CELL_IGNORE
    cells_rgb[cell_labels == 1] = COLOR_CELL_ROCK

    centers_xyz = np.empty((0, 3))
    centers_rgb = np.empty((0, 3))
    n_samples = n_rock_samples = 0
    points_npz = os.path.join(ds.points_dir, f"frame_{frame:06d}.npz")
    if os.path.exists(points_npz):
        pts = np.load(points_npz)
        centers_xyz = pts["centers_odom"].astype(float)
        labels = pts["labels"]
        n_samples = len(labels)
        n_rock_samples = int((labels == 1).sum())
        centers_rgb = np.where((labels == 1)[:, None], COLOR_CENTER_ROCK, COLOR_CENTER_CLEAR)

    return {
        "frame": frame,
        "frame_time": float(bev["frame_time"]),
        "robot_pose": pose,
        "cells_xyz": cells_xyz,
        "cells_rgb": cells_rgb,
        "rock_cells": int((mask == 1).sum()),
        "clear_cells": int((mask == 0).sum()),
        "centers_xyz": centers_xyz,
        "centers_rgb": centers_rgb,
        "n_samples": n_samples,
        "n_rock_samples": n_rock_samples,
    }


# Accumulation windows selectable in the browser: seconds of trailing data to
# combine into the view (0 = just the current frame, None = the entire run).
WINDOW_OPTIONS: list[tuple[str, float | None]] = [
    ("Current frame", 0.0),
    ("Last 0.25 s", 0.25),
    ("Last 1 s", 1.0),
    ("Last 5 s", 5.0),
    ("Entire run", None),
]
DEFAULT_WINDOW = 2  # "Last 1 s"


class _FrameCache:
    """Lazily loaded, memoized per-frame arrays for one run."""

    def __init__(self, ds: _Dataset):
        self.ds = ds
        self._cache: dict[int, dict] = {}

    def get(self, frame_idx: int) -> dict:
        data = self._cache.get(frame_idx)
        if data is None:
            data = self._cache[frame_idx] = _load_frame_arrays(self.ds, frame_idx)
        return data

    def combined(self, pos: int, window_s: float | None) -> tuple[dict, list[dict]]:
        """(current frame's arrays, all member frames of the window ending there).

        window_s: 0 = current frame only, None = every frame in the run,
        otherwise all frames within the trailing window_s seconds.
        """
        current = self.get(self.ds.frames[pos])
        if window_s == 0.0:
            return current, [current]
        if window_s is None:
            return current, [self.get(f) for f in self.ds.frames]
        t_end = current["frame_time"]
        members = []
        for j in range(pos, -1, -1):
            data = self.get(self.ds.frames[j])
            if t_end - data["frame_time"] > window_s:
                break
            members.append(data)
        members.reverse()
        return current, members


def run_preview(out_dir: str, run_id: str | None, frame: int | None, list_only: bool) -> None:
    ds = open_dataset(out_dir, run_id)
    if list_only:
        print(f"run {ds.run_id}: {len(ds.frames)} frames")
        print(" ".join(str(i) for i in ds.frames))
        return

    start = 0
    if frame is not None:
        if frame not in ds.frames:
            raise SystemExit(f"frame {frame} not in run {ds.run_id!r}; available: "
                             f"{ds.frames[:10]}{'...' if len(ds.frames) > 10 else ''} (use --list)")
        start = ds.frames.index(frame)

    print(f"run {ds.run_id}: {len(ds.frames)} frames, "
          f"{len(ds.spheres)} label spheres ({ds.labels_path or 'labels JSON not found'})")
    print("browse with the transport bar, Left/Right arrows, or space to play\n")
    print(LEGEND)

    from . import viewer

    cache = _FrameCache(ds)

    def loader(i: int, window_idx: int) -> "viewer.FrameView":
        window_s = WINDOW_OPTIONS[window_idx][1]
        current, members = cache.combined(i, window_s)
        cells_xyz = np.concatenate([m["cells_xyz"] for m in members])
        cells_rgb = np.concatenate([m["cells_rgb"] for m in members])
        centers_xyz = np.concatenate([m["centers_xyz"] for m in members])
        centers_rgb = np.concatenate([m["centers_rgb"] for m in members])
        rock_cells = sum(m["rock_cells"] for m in members)
        clear_cells = sum(m["clear_cells"] for m in members)
        n_samples = sum(m["n_samples"] for m in members)
        n_rock = sum(m["n_rock_samples"] for m in members)

        base = current["robot_pose"][:3, 3]
        point_sets = [("cells", cells_xyz, cells_rgb, 5.0)]
        if len(centers_xyz):
            point_sets.append(("centers", centers_xyz, centers_rgb, 8.0))
        span = current["frame_time"] - members[0]["frame_time"]
        stats = [
            f"robot: ({base[0]:.2f}, {base[1]:.2f}, {base[2]:.2f})",
            f"combined: {len(members)} frame{'s' if len(members) != 1 else ''}"
            + (f" ({span:.2f} s)" if len(members) > 1 else ""),
            f"cells: {len(cells_xyz)} occupied",
            f"  rock {rock_cells}, clear {clear_cells}",
            f"samples: {n_samples} ({n_rock} rock)",
        ]
        if rock_cells == 0 and ds.spheres:
            stats.append("no rock cells in view")
        return viewer.FrameView(
            index=current["frame"],
            time_s=current["frame_time"],
            point_sets=point_sets,
            spheres=ds.spheres,
            pose=current["robot_pose"],
            stats_lines=stats,
        )

    app = viewer.FrameBrowserApp(
        loader, n_frames=len(ds.frames),
        title=f"rocklabel - preview - {ds.run_id}",
        start=start, legend_lines=LEGEND_LINES,
        window_labels=[label for label, _ in WINDOW_OPTIONS],
        window_default=DEFAULT_WINDOW,
    )
    app.run()
