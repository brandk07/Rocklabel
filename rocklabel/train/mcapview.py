"""`rocklabel-train replay`: run a trained model directly on a recording.

Unlike `view` (which replays a *generated dataset* and can show ground truth),
this takes any .mcap - labeled or not, either recording format - and shows the
model's rock confidence per candidate center, with the frame's cropped cloud
as context. The neighborhood geometry (crop box, voxel grid, radius, padding)
comes from the checkpoint itself, so preprocessing always matches what the
model was trained on; only the topics section of the user config matters here.

The whole recording is decoded and scored up front (a few thousand frames take
well under a minute on GPU), then browsed with the standard transport-bar app;
that keeps scrubbing instant and playback smooth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..dataset.neighborhoods import build_inference_samples
from ..recording.pipeline import ScanStream, WindowedScanStream
from .models import build_model

MAX_CLOUD_POINTS = 30_000   # per-frame context cloud cap (display only)
CLOUD_DIM = 0.40            # context cloud brightness relative to height colors


@dataclass
class FrameRec:
    """Everything kept per frame after the scoring pass."""
    index: int
    time_s: float
    pose: np.ndarray
    cloud: np.ndarray      # cropped, display-subsampled [M, 3]
    cloud_rgb: np.ndarray
    centers: np.ndarray    # [S, 3]
    probs: np.ndarray      # [S]


def _score_recording(mcap_path: str, cfg: dict, gcfg: dict, model, device,
                     stride: int | None, window_s: float | None,
                     z_min: float | None = None, z_max: float | None = None,
                     max_range: float | None = None,
                     batch: int = 512) -> list[FrameRec]:
    from .. import viewer  # height_colors, without importing open3d at module load

    eff_stride = stride if stride is not None else gcfg["frame_stride"]
    eff_window = window_s if window_s is not None else (gcfg.get("frame_window_s") or 0.0)
    if eff_window > 0.0:
        stream = WindowedScanStream(
            ScanStream(mcap_path, cfg, stride=1, progress=True, desc="replay"), eff_window)
        frames = (s for k, s in enumerate(stream) if k % eff_stride == 0)
    else:
        frames = ScanStream(mcap_path, cfg, stride=eff_stride, progress=True, desc="replay")

    recs: list[FrameRec] = []
    model.eval().to(device)
    for scan in frames:
        base = scan.T_odom_base[:3, 3]
        lo = np.array([base[0] - gcfg["crop_backward_m"], base[1] - gcfg["crop_right_m"],
                       base[2] - gcfg["crop_down_m"]], np.float32)
        hi = np.array([base[0] + gcfg["crop_forward_m"], base[1] + gcfg["crop_left_m"],
                       base[2] + gcfg["crop_up_m"]], np.float32)
        inside = ((scan.xyz_odom >= lo) & (scan.xyz_odom <= hi)).all(axis=1)
        # Optional user crop on top of the checkpoint box: a z band relative to
        # the sensor height and a horizontal radius, to skip walls/ceiling.
        if z_min is not None:
            inside &= scan.xyz_odom[:, 2] >= base[2] + z_min
        if z_max is not None:
            inside &= scan.xyz_odom[:, 2] <= base[2] + z_max
        if max_range is not None and max_range > 0:
            dxy = scan.xyz_odom[:, :2] - base[:2]
            inside &= (dxy * dxy).sum(axis=1) <= max_range * max_range
        if not inside.any():
            continue
        xyz, inten = scan.xyz_odom[inside], scan.intensity[inside]

        # Same per-frame seeding as generate: identical frames score identically.
        rng = np.random.default_rng([int(gcfg["seed"]), scan.index])
        samples = build_inference_samples(xyz, inten, gcfg, rng)

        centers = np.empty((0, 3), np.float32)
        probs = np.empty(0, np.float32)
        if samples is not None:
            pts = torch.from_numpy(samples["neighborhoods"])
            cnt = torch.from_numpy(samples["true_counts"].astype(np.int64))
            out = []
            with torch.no_grad():
                for i in range(0, len(pts), batch):
                    logits = model(pts[i:i + batch].to(device), cnt[i:i + batch].to(device))
                    out.append(torch.sigmoid(logits).float().cpu().numpy())
            centers, probs = samples["centers_odom"], np.concatenate(out)

        if len(xyz) > MAX_CLOUD_POINTS:
            keep = rng.choice(len(xyz), MAX_CLOUD_POINTS, replace=False)
            xyz = xyz[keep]
        recs.append(FrameRec(
            index=scan.index, time_s=scan.time_s, pose=scan.T_odom_base,
            cloud=xyz.astype(np.float64),
            cloud_rgb=viewer.height_colors(xyz[:, 2]) * CLOUD_DIM,
            centers=centers.astype(np.float64), probs=probs,
        ))
    return recs


def run_mcap_replay(mcap_path: str, checkpoint: str, cfg: dict,
                    device: str | None = None, stride: int | None = None,
                    window_s: float | None = None, dump: str | None = None,
                    z_min: float | None = None, z_max: float | None = None,
                    max_range: float | None = None) -> None:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tcfg, gcfg = ck["config"], ck["generator"]
    model = build_model(tcfg["model"], tnet=tcfg["tnet"], dropout=tcfg.get("dropout"),
                        features=tcfg.get("features"))
    model.load_state_dict(ck["model"])
    print(f"model: {tcfg['model']} (trained on {', '.join(tcfg['train_runs'])}; "
          f"held out {tcfg['test_run']}), threshold {ck.get('threshold', 0.5):.2f}")
    print(f"input channels: {', '.join(model.features)}")

    recs = _score_recording(mcap_path, cfg, gcfg, model, dev, stride, window_s,
                            z_min=z_min, z_max=z_max, max_range=max_range)
    if not recs:
        raise SystemExit("no scorable frames (no pose? empty crop? try rocklabel inspect)")
    n = sum(len(r.probs) for r in recs)
    print(f"scored {n} samples over {len(recs)} frames")

    if dump:
        np.savez_compressed(
            dump,
            frame=np.concatenate([np.full(len(r.probs), r.index, np.int32) for r in recs]),
            centers=np.concatenate([r.centers for r in recs]).astype(np.float32),
            probs=np.concatenate([r.probs for r in recs]),
            threshold=float(ck.get("threshold", 0.5)),
        )
        print(f"wrote {dump}")
        return

    from .. import viewer

    state = {"mode": 0, "threshold": float(ck.get("threshold", 0.5))}
    modes = ["Confidence (blue-red)", "Detections @ threshold"]
    det_rock = (1.00, 0.25, 0.15)
    det_clear = (0.30, 0.35, 0.42)

    # trailing accumulation windows, same choices as the dataset preview
    from ..gui.preview import DEFAULT_WINDOW, WINDOW_OPTIONS

    def members_for(pos: int, window_s_: float | None) -> list[FrameRec]:
        if window_s_ == 0.0:
            return [recs[pos]]
        if window_s_ is None:
            return recs
        t_end = recs[pos].time_s
        j = pos
        while j > 0 and t_end - recs[j - 1].time_s <= window_s_:
            j -= 1
        return recs[j:pos + 1]

    def loader(i: int, window_idx: int) -> "viewer.FrameView":
        members = members_for(i, WINDOW_OPTIONS[window_idx][1])
        cur = recs[i]
        cloud = np.concatenate([m.cloud for m in members])
        cloud_rgb = np.concatenate([m.cloud_rgb for m in members])
        centers = np.concatenate([m.centers for m in members])
        probs = np.concatenate([m.probs for m in members])
        thr = state["threshold"]
        if state["mode"] == 0:
            rgb = viewer._turbo(probs)
        else:
            rgb = np.where((probs >= thr)[:, None], det_rock, det_clear)
        point_sets = [("cloud", cloud, cloud_rgb, 3.5)]
        if len(centers):
            point_sets.append(("centers", centers, rgb, 9.0))
        stats = [
            f"model: {tcfg['model']}",
            f"threshold: {thr:.2f}",
            f"samples in view: {len(probs)} ({int((probs >= thr).sum())} >= thr)",
            f"cloud points: {len(cloud)}",
        ]
        return viewer.FrameView(index=cur.index, time_s=cur.time_s,
                                point_sets=point_sets, pose=cur.pose,
                                stats_lines=stats)

    def extras(app, panel, em):
        import open3d.visualization.gui as gui
        sec = gui.CollapsableVert("Model", 0.35 * em, gui.Margins(em, 0, 0, 0))
        sec.set_is_open(True)
        combo = gui.Combobox()
        for m in modes:
            combo.add_item(m)

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
            app.refresh()

        slider.set_on_value_changed(on_thr)
        sec.add_child(slider)
        panel.add_child(sec)

    legend = [(f"p(rock) = {p:.2f}", tuple(viewer._turbo(np.array([p]))[0]))
              for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    app = viewer.FrameBrowserApp(
        loader, n_frames=len(recs),
        title=f"rocklabel - mcap replay - {tcfg['model']}",
        legend_lines=legend,
        window_labels=[label for label, _ in WINDOW_OPTIONS],
        window_default=DEFAULT_WINDOW,
        extras_builder=extras,
    )
    app.run()
