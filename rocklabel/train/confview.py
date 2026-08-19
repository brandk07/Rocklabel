"""3D confidence viewer: replay a generated run with sample centers colored by
the trained model's rock probability (`rocklabel-train view`).

Reuses the preview machinery end to end - dataset reconstruction from npz,
frame cache, and FrameBrowserApp (transport bar, accumulation windows) - and
adds a mode toggle (predicted / ground truth / error) plus a threshold slider
via the browser's extras hook. All predictions for the run are computed once
up front (a run is ~20k samples, seconds on GPU), so scrubbing stays instant.
"""

from __future__ import annotations

import os
import re

import numpy as np
import torch

from ..gui.preview import (COLOR_CELL_IGNORE, COLOR_CENTER_CLEAR, COLOR_CENTER_ROCK,
                       DEFAULT_WINDOW, WINDOW_OPTIONS, _FrameCache, open_dataset)
from .models import build_model

CELL_DIM = (0.32, 0.32, 0.35)      # BEV context, deliberately muted
COLOR_TP = (0.20, 0.85, 0.35)      # green: correctly called rock
COLOR_TN = (0.42, 0.44, 0.50)      # gray: correctly called clear
COLOR_FP = (1.00, 0.55, 0.05)      # orange: false alarm
COLOR_FN = (1.00, 0.15, 0.60)      # magenta: missed rock

MODES = ["Predicted confidence", "Ground truth", "Errors @ threshold"]


def _turbo_legend():
    from ..gui.viewer import _turbo
    stops = []
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        stops.append((f"p(rock) = {p:.2f}", tuple(_turbo(np.array([p]))[0])))
    return stops


def _predict_run(ds, model, device, batch: int = 512) -> dict[int, np.ndarray]:
    """{frame_index: probs} for every points npz of the run."""
    from tqdm import tqdm
    per_frame: dict[int, np.ndarray] = {}
    files = sorted(f for f in os.listdir(ds.points_dir)
                   if re.fullmatch(r"frame_\d{6}\.npz", f))
    model.eval().to(device)
    with torch.no_grad():
        for name in tqdm(files, desc="scoring frames"):
            with np.load(os.path.join(ds.points_dir, name)) as z:
                pts = torch.from_numpy(z["neighborhoods"].astype(np.float32))
                cnt = torch.from_numpy(z["true_counts"].astype(np.int64))
            probs = []
            for i in range(0, len(pts), batch):
                logits = model(pts[i:i + batch].to(device), cnt[i:i + batch].to(device))
                probs.append(torch.sigmoid(logits).float().cpu().numpy())
            per_frame[int(name[6:12])] = np.concatenate(probs)
    return per_frame


def run_confview(out_dir: str, run_id: str | None, checkpoint: str,
                 device: str | None = None, frame: int | None = None) -> None:
    ds = open_dataset(out_dir, run_id)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if ck.get("config_hash") != _dataset_hash(out_dir):
        print(f"WARNING: checkpoint config_hash {str(ck.get('config_hash'))[:12]} != "
              f"dataset hash {_dataset_hash(out_dir)[:12]} - neighborhood geometry may differ")
    if cfg.get("test_run") == ds.run_id:
        print(f"note: {ds.run_id} was this checkpoint's held-out run (honest view)")
    elif ds.run_id in cfg.get("train_runs", []):
        print(f"note: {ds.run_id} was in this checkpoint's TRAINING set - expect "
              "optimistic-looking predictions")
    model = build_model(cfg["model"], tnet=cfg["tnet"], dropout=cfg.get("dropout"),
                        features=cfg.get("features"))
    model.load_state_dict(ck["model"])
    probs_by_frame = _predict_run(ds, model, dev)

    from ..gui import viewer

    state = {"mode": 0, "threshold": float(ck.get("threshold", 0.5))}
    cache = _FrameCache(ds)

    def center_colors(frame_idx: int, labels: np.ndarray) -> tuple[np.ndarray, dict]:
        probs = probs_by_frame.get(frame_idx)
        if probs is None or len(probs) != len(labels):  # points npz missing for frame
            return np.tile(COLOR_TN, (len(labels), 1)), {}
        mode, thr = state["mode"], state["threshold"]
        if mode == 0:
            return viewer._turbo(probs), {}
        if mode == 1:
            return np.where((labels == 1)[:, None], COLOR_CENTER_ROCK, COLOR_CENTER_CLEAR), {}
        pred = probs >= thr
        pos = labels == 1
        rgb = np.empty((len(labels), 3))
        rgb[pred & pos] = COLOR_TP
        rgb[~pred & ~pos] = COLOR_TN
        rgb[pred & ~pos] = COLOR_FP
        rgb[~pred & pos] = COLOR_FN
        return rgb, {"fp": int((pred & ~pos).sum()), "fn": int((~pred & pos).sum())}

    def loader(i: int, window_idx: int) -> "viewer.FrameView":
        window_s = WINDOW_OPTIONS[window_idx][1]
        current, members = cache.combined(i, window_s)
        cells_xyz = np.concatenate([m["cells_xyz"] for m in members])
        # BEV cells stay as muted context; the GT label mask only tints ignore
        # cells so shell regions remain recognizable.
        cells_rgb = np.concatenate([m["cells_rgb"] for m in members])
        dim = np.tile(CELL_DIM, (len(cells_rgb), 1))
        ignore = np.all(np.isclose(cells_rgb, COLOR_CELL_IGNORE), axis=1)
        dim[ignore] = np.multiply(COLOR_CELL_IGNORE, 0.6)
        point_sets = [("cells", cells_xyz, dim, 4.0)]

        n_samples = n_rock = fp = fn = 0
        centers_all, colors_all = [], []
        for m in members:
            labels = _frame_labels(ds, m["frame"])
            if labels is None:
                continue
            rgb, errs = center_colors(m["frame"], labels)
            centers_all.append(m["centers_xyz"])
            colors_all.append(rgb)
            n_samples += len(labels)
            n_rock += int((labels == 1).sum())
            fp += errs.get("fp", 0)
            fn += errs.get("fn", 0)
        if centers_all:
            point_sets.append(("centers", np.concatenate(centers_all),
                               np.concatenate(colors_all), 9.0))

        stats = [
            f"model: {cfg['model']}  (held out: {cfg.get('test_run', '?')})",
            f"mode: {MODES[state['mode']]}",
            f"threshold: {state['threshold']:.2f}",
            f"samples in view: {n_samples} ({n_rock} rock)",
        ]
        if state["mode"] == 2:
            stats.append(f"false alarms {fp} · missed rocks {fn}")
        return viewer.FrameView(
            index=current["frame"], time_s=current["frame_time"],
            point_sets=point_sets, spheres=ds.spheres,
            pose=current["robot_pose"], stats_lines=stats,
        )

    def extras(app, panel, em):
        import open3d.visualization.gui as gui
        sec = gui.CollapsableVert("Model confidence", 0.35 * em, gui.Margins(em, 0, 0, 0))
        sec.set_is_open(True)
        combo = gui.Combobox()
        for m in MODES:
            combo.add_item(m)
        combo.selected_index = state["mode"]

        def on_mode(_t, idx):
            state["mode"] = int(idx)
            app.refresh()

        combo.set_on_selection_changed(on_mode)
        sec.add_child(combo)
        sec.add_child(gui.Label("Decision threshold"))
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(0.0, 1.0)
        slider.double_value = state["threshold"]

        def on_thr(v):
            state["threshold"] = float(v)
            if state["mode"] == 2:  # only the error view depends on it
                app.refresh()

        slider.set_on_value_changed(on_thr)
        sec.add_child(slider)
        err_legend = gui.Label("errors:  green TP · gray TN\norange FP · magenta FN")
        err_legend.text_color = gui.Color(0.62, 0.64, 0.70)
        sec.add_child(err_legend)
        panel.add_child(sec)

    start = 0
    if frame is not None and frame in ds.frames:
        start = ds.frames.index(frame)
    print(f"run {ds.run_id}: {len(ds.frames)} frames scored; "
          f"threshold from checkpoint: {state['threshold']:.2f}")
    app = viewer.FrameBrowserApp(
        loader, n_frames=len(ds.frames),
        title=f"rocklabel - confidence - {ds.run_id} - {cfg['model']}",
        start=start, legend_lines=_turbo_legend(),
        window_labels=[label for label, _ in WINDOW_OPTIONS],
        window_default=DEFAULT_WINDOW,
        extras_builder=extras,
    )
    app.run()


def _frame_labels(ds, frame_idx: int) -> np.ndarray | None:
    path = os.path.join(ds.points_dir, f"frame_{frame_idx:06d}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return z["labels"].copy()


def _dataset_hash(out_dir: str) -> str:
    import json
    from ..dataset.generate import MANIFEST_NAME
    with open(os.path.join(out_dir, MANIFEST_NAME)) as f:
        return json.load(f)["config_hash"]
