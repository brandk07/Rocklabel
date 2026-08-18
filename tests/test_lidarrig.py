"""Native lidarrig recording support: codec round-trip, format auto-detection,
world-frame geometry, and the full generate pipeline."""

import glob
import os

import numpy as np
import pytest

import make_synthetic_mcap as synth
from rocklabel.config import load_config
from rocklabel.dataset.generate import run_generate
from rocklabel.gui.labeler import accumulate_cloud
from rocklabel.recording.lidarrig_io import decode_frame, encode_frame, iter_frames
from rocklabel.recording.mcap_io import read_info
from rocklabel.recording.pipeline import LidarrigScanStream, ScanStream


@pytest.fixture(scope="module")
def lidarrig_recording(tmp_path_factory):
    root = tmp_path_factory.mktemp("lidarrig")
    mcap_path = root / "handheld.mcap"
    labels_path = root / "handheld.labels.json"
    synth.write_synthetic_lidarrig_mcap(str(mcap_path))
    synth.write_matching_labels(str(labels_path), mcap_name="handheld.mcap")
    return str(mcap_path), str(labels_path)


def test_codec_round_trip():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(50, 3)).astype(np.float32)
    inten = rng.uniform(0, 65535, 50).astype(np.float32)
    pos = np.array([1.0, -2.0, 0.5])
    quat = np.array([np.cos(0.3), 0.0, 0.0, np.sin(0.3)])  # wxyz

    frame = decode_frame(encode_frame(pts, inten, 123.5, pos, quat))
    np.testing.assert_array_equal(frame.points, pts)
    np.testing.assert_array_equal(frame.intensity, inten)
    assert frame.timestamp == 123.5
    assert frame.has_pose
    np.testing.assert_allclose(frame.pose_position, pos)
    np.testing.assert_allclose(frame.pose_quat, quat)

    # No intensity, no pose.
    bare = decode_frame(encode_frame(pts, None, 1.0, None, None))
    assert bare.intensity is None
    assert not bare.has_pose
    np.testing.assert_allclose(bare.pose_matrix(), np.eye(4))


def test_autodetect_and_iter(lidarrig_recording):
    mcap_path, _ = lidarrig_recording
    info = read_info(mcap_path)
    assert info.is_lidarrig
    stream = ScanStream(mcap_path, load_config(None), progress=False)
    assert isinstance(stream, LidarrigScanStream)
    assert stream.scan_count == synth.N_SCANS

    frames = list(iter_frames(mcap_path))
    assert len(frames) == synth.N_SCANS
    assert all(f.has_pose for f in frames)
    assert all(f.intensity is not None for f in frames)


def test_world_frame_geometry_and_intensity(lidarrig_recording):
    """Recorded poses must place the floor at z~0 and rocks where labeled,
    and u16 RSSI must be normalized back to the synthetic 0.2/0.8 levels."""
    mcap_path, _ = lidarrig_recording
    cfg = load_config(None)
    stream = ScanStream(mcap_path, cfg, progress=False)
    scans = [s for _, s in zip(range(10), iter(stream))]
    assert stream.counters.intensity_available is True

    xyz = np.concatenate([s.xyz_odom for s in scans])
    inten = np.concatenate([s.intensity for s in scans])
    # Floor points sit near z=0 in the world frame despite the moving pose.
    floor = inten < 0.5
    assert abs(float(np.median(xyz[floor, 2]))) < 0.02
    # Intensity was normalized from u16 back to ~0.2 (floor) / ~0.8 (rocks).
    assert np.isclose(float(np.median(inten[floor])), 0.2, atol=0.01)
    assert np.isclose(float(np.median(inten[~floor])), 0.8, atol=0.01)
    # Rock-level points exist near the first labeled rock.
    center = np.asarray(synth.ROCKS[0]["center"])
    near = np.linalg.norm(xyz - center, axis=1) < synth.ROCKS[0]["radius"] + 0.05
    assert near.sum() > 50


def test_generate_with_frame_window(lidarrig_recording, tmp_path):
    """frame_window_s merges scans into dense frames; frame_stride then applies
    to the merged frames, not the raw messages."""
    mcap_path, labels_path = lidarrig_recording
    cfg = load_config(None)
    cfg["generator"]["frame_window_s"] = 0.5  # 10 Hz scans -> 5 scans per window

    entry = run_generate(mcap_path, labels_path, str(tmp_path / "ds"), cfg)
    # 100 scans / 5-per-window = 20 windows; stride 5 keeps windows 0,5,10,15.
    assert entry["frames_kept"] == 4
    assert entry["sample_labels"]["rock"] > 0

    files = sorted(glob.glob(os.path.join(str(tmp_path / "ds"), "bev", "handheld", "*.npz")))
    assert len(files) == 4
    # A merged frame carries ~5x the points of one message: its BEV occupancy
    # must beat a single-scan frame's from the unwindowed dataset.
    merged = np.load(files[0])
    assert (merged["channels"][0] > 0).sum() > 500


def test_accumulate_and_generate(lidarrig_recording, tmp_path):
    mcap_path, labels_path = lidarrig_recording
    cfg = load_config(None)

    xyz, _inten, _counts, stream = accumulate_cloud(mcap_path, cfg, stride=2)
    assert stream.counters.scans_used == synth.N_SCANS // 2
    assert len(xyz) > 1000

    out_dir = str(tmp_path / "ds")
    entry = run_generate(mcap_path, labels_path, out_dir, cfg)
    assert entry["frames_kept"] == synth.N_SCANS // cfg["generator"]["frame_stride"]
    assert entry["frames_skipped_pose"] == 0
    assert entry["sample_labels"]["rock"] > 0, "labels misaligned with native poses"
    assert entry["intensity_available"] is True
    assert len(glob.glob(os.path.join(out_dir, "bev", "handheld", "*.npz"))) == entry["bev_frames"]
