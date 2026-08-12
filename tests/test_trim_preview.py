"""`trim` (time window, topic filter, truncated-file salvage) and `preview` loading."""

import numpy as np
import pytest

import make_synthetic_mcap as synth
from rocklabel.config import load_config
from rocklabel.generate import run_generate
from rocklabel.labeler import accumulate_cloud
from rocklabel.mcap_io import McapFormatError, read_info
from rocklabel.preview import load_frame
from rocklabel.trim import run_trim


def test_trim_time_window_keeps_tf_in_full(synthetic_recording, tmp_path):
    mcap_path, labels_path = synthetic_recording
    out = str(tmp_path / "trimmed.mcap")
    cfg = load_config(None)
    # Scans run t_rel ~0.025..9.925 s; keep [2, 5] -> scan indices 20..49.
    run_trim(mcap_path, out, cfg, start_s=2.0, end_s=5.0)
    info = read_info(out)
    assert info.message_count("/multiscan/lidar_scan") == 30
    # TF is exempt from the window so edge scans still get poses.
    assert info.message_count("/tf") == 203
    assert info.message_count("/tf_static") == 1

    # The trimmed file goes through the full pipeline without pose skips.
    entry = run_generate(out, labels_path, str(tmp_path / "ds"), cfg)
    assert entry["frames_kept"] == 6  # ceil(30 / frame_stride 5)
    assert entry["frames_skipped_pose"] == 0


def test_trim_extra_and_all_topics(synthetic_recording, tmp_path):
    mcap_path, _ = synthetic_recording
    cfg = load_config(None)
    out = str(tmp_path / "all.mcap")
    run_trim(mcap_path, out, cfg, all_topics=True)
    info = read_info(out)
    assert info.message_count("/multiscan/lidar_scan") == 100
    assert info.message_count("/tf") == 203


def test_truncated_mcap_salvage(synthetic_recording, tmp_path):
    mcap_path, _ = synthetic_recording
    cfg = load_config(None)
    # Simulate a recorder killed mid-write: cut the file at 60%.
    raw = open(mcap_path, "rb").read()
    broken = tmp_path / "broken.mcap"
    broken.write_bytes(raw[: int(len(raw) * 0.6)])

    with pytest.raises(McapFormatError, match="rocklabel trim"):
        read_info(str(broken))

    fixed = str(tmp_path / "fixed.mcap")
    run_trim(str(broken), fixed, cfg)
    info = read_info(fixed)  # output is a valid, indexed mcap
    n_scans = info.message_count("/multiscan/lidar_scan")
    assert 0 < n_scans < 100  # salvaged a prefix of the recording
    xyz, _, _, stream = accumulate_cloud(fixed, cfg, stride=1)
    assert stream.counters.scans_used > 0
    assert len(xyz) > 100


def test_preview_load_frame(synthetic_recording, tmp_path):
    mcap_path, labels_path = synthetic_recording
    cfg = load_config(None)
    out_dir = str(tmp_path / "ds")
    run_generate(mcap_path, labels_path, out_dir, cfg)

    data = load_frame(out_dir)  # single run resolved automatically, middle frame
    assert data["run_id"] == "synthetic"
    assert len(data["frames"]) == 20
    assert data["frame"] in data["frames"]
    assert len(data["cells_xyz"]) > 100
    assert data["rock_cells"] > 0
    assert (data["cells_rgb"] == np.asarray([0.95, 0.20, 0.15])).all(axis=1).sum() == data["rock_cells"]
    assert data["n_samples"] > 0
    assert len(data["spheres"]) == len(synth.ROCKS)

    first = load_frame(out_dir, frame=data["frames"][0])
    assert first["frame"] == data["frames"][0]

    with pytest.raises(SystemExit, match="not in run"):
        load_frame(out_dir, frame=999999)
    with pytest.raises(SystemExit, match="no manifest.json"):
        load_frame(str(tmp_path))
