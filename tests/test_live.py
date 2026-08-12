"""Live rig integration: `rocklabel record` produces native recordings the
offline pipeline reads, CLI config plumbing behaves, and the live scorer's
point-coloring contract holds."""

import argparse

import numpy as np
import pytest

from rocklabel.config import load_config
from rocklabel.lidarrig_io import iter_frames, read_embedded_config
from rocklabel.live.config import AppConfig
from rocklabel.live.pipeline import IngestEngine
from rocklabel.live.run import _build_config, add_live_args
from rocklabel.live.sources import make_source
from rocklabel.live.surfaces import make_surface_builder
from rocklabel.pipeline import ScanStream


def _fast_sim_config() -> AppConfig:
    cfg = AppConfig()
    cfg.source.kind = "sim"
    cfg.source.sim_points_per_sec = 100_000
    cfg.source.sim_batch_size = 2_000
    cfg.slam.enabled = False
    return cfg


def test_record_roundtrip_through_offline_pipeline(tmp_path):
    """A live-rig recording must be readable by every offline rocklabel stage."""
    import time

    cfg = _fast_sim_config()
    out = str(tmp_path / "live.mcap")
    engine = IngestEngine(make_source(cfg), make_surface_builder(cfg), cfg)
    engine.start()
    assert engine.start_recording(out) == out
    deadline = time.monotonic() + 5.0
    while engine.stats.batches_total < 10 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert engine.stop_recording() == out
    engine.stop()

    frames = list(iter_frames(out))
    assert len(frames) >= 10
    assert frames[0].points.shape[1] == 3
    assert frames[0].has_pose
    assert "kind: sim" in (read_embedded_config(out) or "")

    # The exact reader stack label/generate/train use.
    stream = ScanStream(out, load_config(None), stride=5, progress=False)
    scans = list(stream)
    assert scans and stream.format_name == "lidarrig"
    assert np.isfinite(scans[0].xyz_odom).all()
    assert scans[0].intensity.shape == (len(scans[0].xyz_odom),)


def _parse(record_cmd: bool, argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_live_args(p, record_cmd=record_cmd)
    return p.parse_args(argv)


def test_record_command_autostarts_recording():
    cfg = _build_config(_parse(True, ["out.mcap", "--no-slam"]), record_cmd=True)
    assert cfg.record.autostart
    assert cfg.record.path == "out.mcap"
    assert not cfg.slam.enabled


def test_live_command_defaults_to_view_only():
    cfg = _build_config(_parse(False, []), record_cmd=False)
    assert not cfg.record.autostart


def test_play_disables_motion_and_recording(tmp_path):
    # --play must never re-run SLAM/IMU or re-record the replayed stream.
    play = str(tmp_path / "x.mcap")
    open(play, "wb").close()
    cfg = _build_config(_parse(False, ["--play", play, "--record"]), record_cmd=False)
    assert not cfg.slam.enabled and not cfg.motion.use_imu
    assert not cfg.record.autostart


def _bare_scorer(centers, probs, settings=None):
    """A LiveScorer with just the state probs_for/status need (no torch)."""
    import threading

    from rocklabel.live.scoring import LiveScorer, ScoreSettings, _Result

    scorer = LiveScorer.__new__(LiveScorer)  # bypass torch/checkpoint loading
    scorer._lock = threading.Lock()
    scorer.threshold = 0.5
    scorer.model_name = "test"
    scorer.settings = settings or ScoreSettings()
    scorer.version = 1
    scorer._last_ms = 12.0
    scorer._last_centers_capped = False
    scorer._last_pass_centers = len(probs)
    scorer._last_in_region = 100
    scorer._last_miss = None
    scorer._map = {}
    scorer._clear_requested = False
    scorer._result = _Result(centers, probs, match_radius=0.2)
    return scorer


def test_scorer_probs_map_nearest_center():
    """probs_for: nearest center's probability inside the match radius,
    unmatched flagged so the viewer can dim those points."""
    pytest.importorskip("scipy")

    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    probs = np.array([0.9, 0.1], np.float32)
    scorer = _bare_scorer(centers, probs)

    pts = np.array([[0.05, 0.0, 0.0],   # near center 0
                    [1.0, 0.05, 0.0],   # near center 1
                    [5.0, 5.0, 5.0]])   # no prediction
    p, matched = scorer.probs_for(pts)
    assert matched.tolist() == [True, True, False]
    assert p[0] == pytest.approx(0.9) and p[1] == pytest.approx(0.1)

    dets = scorer.detections()
    assert len(dets[0]) == 1 and dets[1][0] == pytest.approx(0.9)
    assert "1 >= thr" in scorer.status()


def test_scorer_map_persists_latest_prob_per_voxel():
    """_update_map: revisiting a spot replaces its probability; new spots
    accumulate — the rolling prediction map behind 'Remember predictions'."""
    pytest.importorskip("scipy")

    scorer = _bare_scorer(np.zeros((1, 3)), np.zeros(1, np.float32))
    scorer._gcfg = {"centers_voxel_m": 0.05}

    res1 = scorer._update_map(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                              np.array([0.2, 0.8], np.float32))
    assert len(res1.probs) == 2
    # Same two voxels again with new probs + one new voxel.
    res2 = scorer._update_map(
        np.array([[0.001, 0.0, 0.0], [1.001, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([0.9, 0.1, 0.5], np.float32))
    assert len(res2.probs) == 3
    by_x = {round(c[0]): p for c, p in zip(res2.centers, res2.probs)}
    assert by_x[0] == pytest.approx(0.9) and by_x[1] == pytest.approx(0.1)


def test_engine_recent_snapshot_windows(tmp_path):
    """recent_snapshot: window 0 = single latest batch (the training-frame
    distribution); larger windows concatenate recent batches."""
    import time

    cfg = _fast_sim_config()
    engine = IngestEngine(make_source(cfg), make_surface_builder(cfg), cfg)
    engine.start()
    deadline = time.monotonic() + 5.0
    while engine.stats.batches_total < 5 and time.monotonic() < deadline:
        time.sleep(0.05)
    engine.stop()

    last_pts, last_inten = engine.recent_snapshot(0.0)
    assert 0 < len(last_pts) <= cfg.source.sim_batch_size
    assert last_inten.shape == (len(last_pts),)
    win_pts, _ = engine.recent_snapshot(10.0)
    assert len(win_pts) > len(last_pts)


def test_scorer_crop_mask_region():
    """crop_mask: z band + horizontal radius relative to the sensor pose,
    intersected with the checkpoint's own crop box."""
    pytest.importorskip("scipy")
    from rocklabel.live.scoring import ScoreSettings

    scorer = _bare_scorer(np.zeros((1, 3)), np.zeros(1, np.float32),
                          ScoreSettings(z_min=-1.5, z_max=-0.5, range_max=8.0))
    scorer._gcfg = {"crop_backward_m": 6.0, "crop_forward_m": 6.0,
                    "crop_left_m": 6.0, "crop_right_m": 6.0,
                    "crop_down_m": 3.0, "crop_up_m": 3.0}
    base = np.array([0.0, 0.0, 1.0])  # sensor 1 m above the floor
    pts = np.array([
        [1.0, 0.0, 0.0],    # floor band, close -> kept
        [1.0, 0.0, 2.5],    # ceiling -> cut by z band
        [12.0, 0.0, 0.0],   # floor band but 12 m away -> cut by range
        [1.0, 0.0, -1.0],   # below the band -> cut
    ])
    assert scorer.crop_mask(pts, base).tolist() == [True, False, False, False]


def test_inference_samples_max_centers_cap():
    """max_centers bounds the number of scored candidates (live memory cap)."""
    pytest.importorskip("scipy")
    from rocklabel.neighborhoods import build_inference_samples

    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 2.0, (4000, 3)).astype(np.float32)
    inten = rng.uniform(0, 1, 4000).astype(np.float32)
    gcfg = {"neighborhood_points": 32, "centers_voxel_m": 0.1,
            "neighborhood_radius_m": 0.5, "min_neighbors": 5}
    full = build_inference_samples(xyz, inten, gcfg, np.random.default_rng(1))
    capped = build_inference_samples(xyz, inten, gcfg, np.random.default_rng(1),
                                     max_centers=50)
    assert len(full["centers_odom"]) > 50
    assert len(capped["centers_odom"]) <= 50
    assert capped["neighborhoods"].shape[1:] == (32, 4)
