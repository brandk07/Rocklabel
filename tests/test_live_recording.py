"""Tests for MCAP recording/replay: the frame codec, writer/reader roundtrip,
transport controls (seek, rewind), config metadata, and an end-to-end
record→replay reconstruction equivalence check through the real engine."""

from __future__ import annotations

import time

import numpy as np
import pytest

from rocklabel.live.config import AppConfig
from rocklabel.live.motion import matrix_to_quat, quat_to_matrix
from rocklabel.live.pipeline import IngestEngine
from rocklabel.live.recording import (
    McapRecorder,
    McapReplaySource,
    decode_frame,
    encode_frame,
    normalize_recording_path,
    read_recording_config,
)
from rocklabel.live.sources.simulated import SimulatedSource
from rocklabel.live.surfaces.kalman_heightmap import KalmanHeightmap


def _rand_frame(rng, n=257, intensity=True, orientation=True):
    pts = rng.uniform(-5, 5, size=(n, 3))
    inten = rng.uniform(0, 255, size=n).astype(np.float32) if intensity else None
    quat = None
    if orientation:
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
    return pts, inten, quat


# --------------------------------------------------------------------------- #
# Codec
# --------------------------------------------------------------------------- #
def test_frame_roundtrip_full():
    rng = np.random.default_rng(0)
    pts, inten, quat = _rand_frame(rng)
    pos = np.array([1.0, -2.0, 0.5])
    pquat = np.array([0.9238795, 0.0, 0.0, 0.3826834])  # 45° yaw
    data = encode_frame(pts, inten, 123.456, quat, pos, pquat)
    f = decode_frame(data, log_time_ns=42)

    assert f.points.shape == (257, 3)
    np.testing.assert_allclose(f.points, pts.astype(np.float32))
    np.testing.assert_allclose(f.intensity, inten)
    assert f.timestamp == pytest.approx(123.456)
    np.testing.assert_allclose(f.orientation, quat)
    np.testing.assert_allclose(f.pose_position, pos)
    np.testing.assert_allclose(f.pose_quat, pquat)
    assert f.log_time_ns == 42


def test_frame_roundtrip_minimal():
    rng = np.random.default_rng(1)
    pts, _, _ = _rand_frame(rng, n=10, intensity=False, orientation=False)
    f = decode_frame(encode_frame(pts, None, 0.0, None, None, None))
    assert f.intensity is None
    assert f.orientation is None
    np.testing.assert_allclose(f.pose_position, np.zeros(3))
    np.testing.assert_allclose(f.pose_quat, [1, 0, 0, 0])


def test_world_points_applies_pose():
    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    pos = np.array([10.0, 0.0, 1.0])
    yaw90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # +90° about z
    f = decode_frame(encode_frame(pts, None, 0.0, None, pos, yaw90))
    np.testing.assert_allclose(f.world_points(), [[10.0, 1.0, 1.0]], atol=1e-6)


def test_matrix_quat_roundtrip():
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q2 = matrix_to_quat(quat_to_matrix(q))
        # q and -q encode the same rotation.
        assert np.allclose(q2, q, atol=1e-9) or np.allclose(q2, -q, atol=1e-9)


# --------------------------------------------------------------------------- #
# Recorder -> replay roundtrip
# --------------------------------------------------------------------------- #
def _write_recording(path, n_frames=8, n_pts=100):
    rng = np.random.default_rng(3)
    cfg = AppConfig()
    cfg.grid.cell_size = 0.25
    rec = McapRecorder(str(path), config=cfg)
    frames = []
    for i in range(n_frames):
        pts, inten, quat = _rand_frame(rng, n=n_pts)
        pos = np.array([0.1 * i, 0.0, 0.0])
        rec.write_frame(pts, inten, float(i), quat, pos, np.array([1.0, 0, 0, 0]))
        frames.append((pts.astype(np.float32), inten, pos))
    rec.close()
    return frames


def _drain(src, limit=1000):
    """Read every batch the source will deliver until it reports finished."""
    out = []
    for _ in range(limit):
        b = src.read(timeout=0.01)
        if b is not None:
            out.append(b)
        if src.finished:
            break
    return out


def test_record_replay_roundtrip(tmp_path):
    path = tmp_path / "r.mcap"
    frames = _write_recording(path)

    src = McapReplaySource(str(path), autoplay=False)
    src.start()
    assert src.duration_sec >= 0.0
    src.seek(src.duration_sec)  # fast-forward: deliver everything unpaced
    batches = _drain(src)
    src.stop()

    assert len(batches) == len(frames)
    for batch, (pts, inten, pos) in zip(batches, frames):
        # Emitted points are world-frame: identity rotation + recorded offset.
        np.testing.assert_allclose(batch.points, pts.astype(np.float64) + pos, atol=1e-6)
        np.testing.assert_allclose(batch.intensity, inten)
        assert batch.orientation is None  # never re-rotated downstream


def test_backward_seek_rewinds_and_resets(tmp_path):
    path = tmp_path / "r.mcap"
    frames = _write_recording(path)
    resets = []

    src = McapReplaySource(str(path), on_rewind=lambda: resets.append(1), autoplay=False)
    src.start()
    src.seek(src.duration_sec)
    first_pass = _drain(src)
    assert len(first_pass) == len(frames)
    assert src.finished

    src.seek(0.0)  # backward: must rewind the file and reset downstream
    b = src.read(timeout=0.5)
    assert len(resets) == 1
    assert b is not None
    np.testing.assert_allclose(b.points, frames[0][0] + frames[0][2], atol=1e-6)
    assert not src.finished
    src.stop()


def test_play_pause_and_pacing(tmp_path):
    path = tmp_path / "r.mcap"
    _write_recording(path, n_frames=3)

    src = McapReplaySource(str(path), autoplay=False)
    src.start()
    assert src.read(timeout=0.05) is None  # paused: nothing flows
    src.play()
    b = src.read(timeout=1.0)  # first frame due immediately at its anchor
    assert b is not None
    src.pause()
    assert src.read(timeout=0.05) is None
    src.stop()


def test_truncated_recording_still_replays(tmp_path):
    """A crash/kill leaves no summary index and a partial tail record; replay
    must fall back to a linear scan and deliver every complete frame."""
    path = tmp_path / "crash.mcap"
    rng = np.random.default_rng(5)
    rec = McapRecorder(str(path), config=AppConfig())
    for i in range(40):  # big frames so several chunks flush before the "crash"
        pts = rng.uniform(-5, 5, size=(4000, 3))
        rec.write_frame(pts, None, float(i), None, np.zeros(3), np.array([1.0, 0, 0, 0]))
    rec.close()

    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * 0.6)])  # chop index + tail mid-record

    assert read_recording_config(str(path)) is not None  # metadata is up front

    src = McapReplaySource(str(path), autoplay=False)
    src.start()
    assert src.duration_sec > 0.0
    src.seek(src.duration_sec)
    batches = _drain(src)
    src.stop()
    assert 1 <= len(batches) < 40  # partial but non-empty recovery
    assert all(b.points.shape == (4000, 3) for b in batches)


def test_config_metadata_roundtrip(tmp_path):
    path = tmp_path / "r.mcap"
    _write_recording(path)
    cfg = read_recording_config(str(path))
    assert cfg is not None
    assert cfg.grid.cell_size == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Reflectivity fusion on the heightmap
# --------------------------------------------------------------------------- #
def _flat_cfg():
    cfg = AppConfig()
    cfg.grid.origin = (-2.0, -2.0)
    cfg.grid.extent = (4.0, 4.0)
    cfg.grid.cell_size = 0.5
    cfg.outlier.enabled = False
    return cfg


def test_mesh_reflectivity_coloring():
    cfg = _flat_cfg()
    # Gray ramp: mean RGB grows monotonically with RSSI, which the assertions
    # below rely on (turbo, the default, is not brightness-monotonic).
    cfg.display.reflectivity_colormap = "gray"
    hm = KalmanHeightmap(cfg)
    rng = np.random.default_rng(4)
    pts = np.column_stack(
        [rng.uniform(-2, 2, 4000), rng.uniform(-2, 2, 4000), np.zeros(4000)]
    )
    # Realistic u16 RSSI values (colors use fixed 0-65535 scaling).
    inten = np.where(pts[:, 0] > 0, 45000.0, 8000.0).astype(np.float32)
    for _ in range(3):
        hm.add_points(pts, inten)

    hm.set_color_mode("reflectivity")
    mesh = hm.get_mesh_arrays()
    assert not mesh.is_empty() and mesh.vertex_colors is not None
    left = mesh.vertices[:, 0] < -0.5
    right = mesh.vertices[:, 0] > 0.5
    assert mesh.vertex_colors[right].mean() > mesh.vertex_colors[left].mean() + 0.3

    # Toggling back re-colors by height (flat surface -> uniform mid colormap).
    hm.set_color_mode("height")
    mesh_h = hm.get_mesh_arrays()
    assert mesh_h is not mesh  # cache invalidated by the mode switch


def test_add_points_without_intensity_still_works():
    cfg = _flat_cfg()
    hm = KalmanHeightmap(cfg)
    pts = np.column_stack([np.linspace(-1, 1, 500), np.zeros(500), np.ones(500)])
    hm.add_points(pts)  # legacy call signature
    hm.add_points(pts, None)
    assert hm.cells_occupied() > 0


# --------------------------------------------------------------------------- #
# End-to-end: record a sim session, replay it, get the same surface back
# --------------------------------------------------------------------------- #
def test_engine_record_then_replay_reconstructs(tmp_path):
    path = str(tmp_path / "session.mcap")
    cfg = AppConfig()
    cfg.source.sim_points_per_sec = 60_000
    cfg.source.sim_batch_size = 3_000
    cfg.source.sim_seed = 7
    cfg.slam.enabled = False

    live = IngestEngine(SimulatedSource(cfg), KalmanHeightmap(cfg), cfg)
    assert live.start_recording(path) == path  # arm before start: capture all
    live.start()
    time.sleep(1.0)
    live.stop()
    assert live.stats.batches_total > 5

    replay_cfg = read_recording_config(path)
    assert replay_cfg is not None
    replay_cfg.slam.enabled = False
    replay_cfg.motion.use_imu = False

    src = McapReplaySource(path, autoplay=False)
    surface = KalmanHeightmap(replay_cfg)
    engine = IngestEngine(src, surface, replay_cfg)
    src.on_rewind = engine.reset_surface
    assert engine.start_recording() is None  # replays are never re-recorded
    engine.start()
    src.seek(src.duration_sec + 1.0)
    deadline = time.monotonic() + 15.0
    while not src.finished and time.monotonic() < deadline:
        time.sleep(0.05)
    engine.stop()
    assert src.finished

    h_live, hits_live = live.surface._height, live.surface._hits
    h_rep, hits_rep = surface._height, surface._hits
    both = (hits_live > 0) & (hits_rep > 0)
    assert both.sum() > 100
    # float32 storage in the file introduces tiny rounding; heights must agree.
    np.testing.assert_allclose(h_rep[both], h_live[both], atol=5e-3)
    # Occupancy should be essentially identical (boundary binning may wiggle).
    assert abs(int(hits_live.astype(bool).sum()) - int(hits_rep.astype(bool).sum())) <= 5
    # The sim emits intensity, so the replayed surface has reflectivity data.
    assert np.isfinite(surface._inten[hits_rep > 0]).any()


def test_normalize_recording_path(tmp_path):
    """A typed-in name is a name, not a path: it lands in the recordings dir
    with a .mcap extension, so the dashboard inventory can see it."""
    d = str(tmp_path / "recordings")
    import os

    assert normalize_recording_path("myrun", d) == os.path.join(d, "myrun.mcap")
    # An explicit path is left pointing where it points.
    explicit = str(tmp_path / "out" / "run.mcap")
    assert normalize_recording_path(explicit, d) == explicit
    # A missing extension is added even on an explicit path.
    assert normalize_recording_path(str(tmp_path / "out" / "run"), d) == explicit
    # An already-correct path is unchanged (and not double-suffixed).
    good = os.path.join(d, "myrun.mcap")
    assert normalize_recording_path(good, d) == good
    # The parent directory is created so the writer cannot fail on open.
    assert os.path.isdir(os.path.dirname(explicit))
