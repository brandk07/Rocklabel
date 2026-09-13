"""Live rig integration: `rocklabel record` produces native recordings the
offline pipeline reads, CLI config plumbing behaves, and the live scorer's
point-coloring contract holds."""

import argparse
import json

import numpy as np
import pytest

from rocklabel.config import load_config
from rocklabel.recording.lidarrig_io import iter_frames, read_embedded_config
from rocklabel.live.config import AppConfig
from rocklabel.live.pipeline import IngestEngine
from rocklabel.live.run import (_build_config, _label_level_for_replay,
                                add_live_args)
from rocklabel.live.sources import make_source
from rocklabel.live.surfaces import make_surface_builder
from rocklabel.recording.pipeline import ScanStream


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


def test_live_carving_evidence_thresholds_are_configurable():
    cfg = _build_config(_parse(False, [
        "--carve",
        "--carve-evidence-group", "0.08",
        "--carve-assumed-pose-uncertainty", "0.01",
        "--carve-confirm-observations", "4",
        "--carve-tentative-contradictions", "3",
        "--carve-confirmed-contradictions", "7",
    ]), record_cmd=False)
    assert cfg.display.carve
    assert cfg.display.carve_evidence_group == 0.08
    assert cfg.display.carve_assumed_pose_uncertainty == 0.01
    assert cfg.display.carve_confirm_observations == 4
    assert cfg.display.carve_tentative_contradictions == 3
    assert cfg.display.carve_confirmed_contradictions == 7


def test_play_disables_motion_and_recording(tmp_path):
    # --play must never re-run SLAM/IMU or re-record the replayed stream.
    play = str(tmp_path / "x.mcap")
    open(play, "wb").close()
    cfg = _build_config(_parse(False, ["--play", play, "--record"]), record_cmd=False)
    assert not cfg.slam.enabled and not cfg.motion.use_imu
    assert not cfg.record.autostart


def test_a_labelled_replay_reuses_the_exact_frame_training_used(tmp_path):
    labels = tmp_path / "labels" / "court"
    labels.mkdir(parents=True)
    (labels / "run.labels.json").write_text(json.dumps({
        "mcap_file": "run.mcap",
        "level": {"mode": "auto", "roll_deg": -7.3597,
                  "pitch_deg": 31.202, "floor_z": -0.7813},
    }))
    level = _label_level_for_replay(str(tmp_path / "run.mcap"), str(tmp_path / "labels"))
    assert level["roll_deg"] == pytest.approx(-7.3597)
    assert level["pitch_deg"] == pytest.approx(31.202)


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
    scorer._last_error = None
    scorer.task = "classify"
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
    from rocklabel.dataset.neighborhoods import (SAMPLE_CHANNELS,
                                                 build_inference_samples)

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
    assert capped["neighborhoods"].shape[1:] == (32, SAMPLE_CHANNELS)


# --------------------------------------------------------------------------- #
# live scoring with a per-point segmenter
# --------------------------------------------------------------------------- #
def test_inference_frame_matches_what_the_segmenter_was_trained_on():
    """Live has to hand the model the exact tensor shape training did, and it
    has to be able to put each prediction back on the point it came from."""
    from rocklabel.config import DEFAULTS
    from rocklabel.dataset.neighborhoods import (build_inference_frame,
                                                 build_segmentation_frame)
    from rocklabel.profiles import apply_profile

    g = apply_profile(DEFAULTS, "full-sweep")["generator"]
    n = int(g["segmentation_points"])
    xyz = np.random.default_rng(0).uniform(-3, 3, (900, 3)).astype(np.float32)
    inten = np.zeros(900, np.float32)
    base = np.array([1.0, 2.0, -0.5], np.float32)

    live = build_inference_frame(xyz, inten, base, g, np.random.default_rng(1))
    trained = build_segmentation_frame(xyz, inten, np.zeros(900, np.int8), base,
                                       g, np.random.default_rng(1))
    assert live["points"].shape == trained["points"].shape == (n, 4)
    assert int(live["true_count"]) == int(trained["true_count"]) == 900
    # same selection under the same seed, so live sees training's canonicalization
    assert np.allclose(live["points"], trained["points"])
    # the extra piece live needs: which original point each row came from
    idx = live["index"][:900]
    assert np.allclose(live["points"][:900, :3], xyz[idx] - base)


def test_a_frame_too_sparse_to_score_is_skipped_not_guessed():
    from rocklabel.config import DEFAULTS
    from rocklabel.dataset.neighborhoods import build_inference_frame
    from rocklabel.profiles import apply_profile

    g = apply_profile(DEFAULTS, "full-sweep")["generator"]
    thin = np.zeros((int(g["segmentation_min_points"]) - 1, 3), np.float32)
    out = build_inference_frame(thin, np.zeros(len(thin), np.float32),
                                np.zeros(3, np.float32), g,
                                np.random.default_rng(0))
    assert out is None


def test_the_segmenter_returns_one_probability_per_point_not_per_ball():
    """The bug this guards: the live scorer fed a segmenter format-A balls and
    got back [balls, points] where it expected [balls], so every pass raised
    and the panel sat on 'warming up' forever."""
    torch = pytest.importorskip("torch")
    from rocklabel.train.models import build_model

    balls = torch.randn(4, 256, 4)
    counts = torch.full((4,), 256, dtype=torch.long)
    clf = build_model("pointnet", features=["dx", "dy", "dz"]).eval()
    seg = build_model("pointnet2_seg", features=["dx", "dy", "dz"]).eval()
    with torch.no_grad():
        assert clf(balls, counts).shape == (4,)          # one per ball
        assert seg(balls, counts).shape == (4, 256)      # one per point


def test_live_loader_preserves_the_checkpoint_height_reference(tmp_path):
    """The floor reference changes no parameter shapes, so the weights load
    cleanly even if serving forgets the setting.  That made the retrained model
    silently run with the old preprocessing and look just like the old model."""
    torch = pytest.importorskip("torch")
    from rocklabel.live.scoring import LiveScorer, ScoreSettings
    from rocklabel.train.models import build_model

    config = {
        "model": "pointnet2_seg", "tnet": False, "dropout": None,
        "features": ["dx", "dy", "dz"],
        "seg_npoints": [32, 16, 8], "seg_radii": [0.1, 0.3, 0.8],
        "seg_height_ref": "floor",
        "seg_coord_ref": "local",
    }
    generator = {"seed": 42, "frame_window_s": 0.0}
    trained = build_model(
        config["model"], features=config["features"],
        seg_npoints=config["seg_npoints"], seg_radii=config["seg_radii"],
        seg_height_ref=config["seg_height_ref"],
        seg_coord_ref=config["seg_coord_ref"],
    )
    checkpoint = tmp_path / "floor.pt"
    torch.save({"config": config, "generator": generator,
                "model": trained.state_dict(),
                "floor_band": (-1.05, -0.78)}, checkpoint)

    scorer = LiveScorer(str(checkpoint), object(), device="cpu",
                        settings=ScoreSettings())
    assert scorer._model.height_ref == "floor"
    assert scorer._model.coord_ref == "local"
    # The saved sensor-relative band is useful checkpoint provenance, but a
    # floor-referenced model subtracts the live floor and is invariant to this
    # offset.  The old warning falsely blamed Lance's mounting height even when
    # the new preprocessing was active.
    scorer.floor_warning = "stale warning"
    points = torch.zeros((1, 32, 4), dtype=torch.float32)
    points[0, :, 2] = -0.57
    scorer._check_floor(points, torch.tensor([32]))
    assert scorer.floor_warning is None


def test_a_failing_scorer_says_so_instead_of_warming_up_forever():
    """The panel used to show 'warming up…' indefinitely while every pass
    raised, with the reason only in a terminal log."""
    scorer = _bare_scorer(np.zeros((0, 3)), np.zeros(0, np.float32))
    scorer._result = None
    scorer._last_error = "TypeError: only 0-dimensional arrays can be converted"
    st = scorer.status_dict()
    assert st["ready"] is False
    assert "scoring is failing" in st["warning"]
    assert "every pass is failing" in scorer.status()
    # and with no error it still reads as warming up
    scorer._last_error = None
    assert "warming up" in scorer.status()


def test_a_tall_z_band_warns_a_segmenter_instead_of_going_quietly_dead():
    """Measured on the competition recording: the floor-band preset lets 0.70 m
    of berm and equipment into a model trained on 0.2 m of flat ground, and it
    then reports no rocks anywhere - which looks exactly like a clean arena."""
    torch = pytest.importorskip("torch")
    scorer = _bare_scorer(np.zeros((0, 3)), np.zeros(0, np.float32))
    scorer.frame_band = (0.15, 0.25)
    scorer.height_warning = None

    tall = torch.zeros((1, 64, 4), dtype=torch.float32)
    tall[0, :, 2] = torch.linspace(0.0, 0.70, 64)
    scorer._check_frame_height(tall, torch.tensor([64]))
    assert scorer.height_warning is not None
    # floor (10th pct) to ceiling (99th pct) of the ramp, not its raw extremes
    assert "0.62 m" in scorer.height_warning

    thin = torch.zeros((1, 64, 4), dtype=torch.float32)
    thin[0, :, 2] = torch.linspace(0.0, 0.25, 64)
    scorer._check_frame_height(thin, torch.tensor([64]))
    assert scorer.height_warning is None


def test_a_region_too_sparse_for_a_whole_frame_pass_says_so():
    """A segmenter refuses a region under segmentation_min_points. Silently,
    the panel just kept showing the previous pass's numbers."""
    scorer = _bare_scorer(np.zeros((0, 3)), np.zeros(0, np.float32))
    scorer._last_thin = (140, 512)
    st = scorer.status_dict()
    assert "140 points" in st["warning"] and "512" in st["warning"]


# --------------------------------------------------------------------------- #
# Phantom-point rejection
# --------------------------------------------------------------------------- #
def _sweep_with_strays(n_ground=800, n_stray=20, seed=0):
    """Flat ground over a 6 m square, plus a few returns stranded in mid-air."""
    rng = np.random.default_rng(seed)
    ground = np.column_stack([
        rng.uniform(-3.0, 3.0, n_ground),
        rng.uniform(-3.0, 3.0, n_ground),
        rng.normal(0.0, 0.01, n_ground),
    ])
    strays = np.column_stack([
        rng.uniform(-3.0, 3.0, n_stray),
        rng.uniform(-3.0, 3.0, n_stray),
        rng.uniform(1.0, 2.5, n_stray),
    ])
    return ground, strays


def test_floating_filter_drops_strays_and_keeps_ground_and_rocks():
    from rocklabel.live.filters import floating_keep_mask

    ground, strays = _sweep_with_strays()
    # A 0.3 m rock sitting on the ground must survive; it is well under the cut.
    rock = np.column_stack([
        np.full(60, 1.0), np.full(60, 1.0), np.linspace(0.0, 0.3, 60),
    ])
    pts = np.vstack([ground, rock, strays])
    keep = floating_keep_mask(pts, cell_size=1.0, max_height=0.5)

    n_g, n_r = len(ground), len(rock)
    assert keep[:n_g].all()                      # ground untouched
    assert keep[n_g:n_g + n_r].all()             # rock untouched
    assert not keep[n_g + n_r:].any()            # every stray dropped


def test_floating_filter_beats_the_mean_based_outlier_test():
    """The spikes inflate their own column's std, so the mean test misses them.

    This is why the phantom points survived: a return 1-2.5 m up widens its
    column's spread enough that a +-2.5 sigma gate no longer reaches it, while
    the genuinely raised points nearby (rock tops) get clipped instead.
    """
    from rocklabel.live.filters import floating_keep_mask, statistical_outlier_mask

    ground, strays = _sweep_with_strays()
    pts = np.vstack([ground, strays])
    n_g = len(ground)

    mean_keep = statistical_outlier_mask(pts, cell_size=0.5, std_ratio=2.5)
    robust_keep = floating_keep_mask(pts, cell_size=1.0, max_height=0.5)

    mean_caught = (~mean_keep[n_g:]).mean()
    robust_caught = (~robust_keep[n_g:]).mean()
    assert robust_caught == 1.0
    assert mean_caught < robust_caught


def test_floating_filter_leaves_sparse_columns_alone():
    """Too few points in a column to say where the ground is -> keep them."""
    from rocklabel.live.filters import floating_keep_mask

    pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 3.0]])
    assert floating_keep_mask(pts, cell_size=1.0, max_height=0.5,
                              min_cell_points=4).all()


def test_floating_filter_mask_passthrough_and_disable():
    from rocklabel.live.config import FloatingConfig
    from rocklabel.live.filters import floating_filter_mask

    ground, strays = _sweep_with_strays()
    pts = np.vstack([ground, strays])
    assert floating_filter_mask(pts, FloatingConfig(enabled=False)) is None
    # Batch below min_points: no statistics worth trusting, pass through.
    assert floating_filter_mask(pts[:8], FloatingConfig()) is None
    # Nothing to drop -> None, so callers can skip the copy.
    assert floating_filter_mask(ground, FloatingConfig()) is None
    assert floating_filter_mask(pts, FloatingConfig()) is not None
