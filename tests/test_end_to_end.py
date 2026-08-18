"""End-to-end: synthetic mcap -> generate -> assert dataset contents and manifest guard."""

import copy
import glob
import os

import numpy as np
import pytest

import make_synthetic_mcap as synth
from rocklabel.config import load_config
from rocklabel.generate import ManifestConflict, run_generate
from rocklabel.labeler import accumulate_cloud


@pytest.fixture(scope="module")
def generated(synthetic_recording, tmp_path_factory):
    mcap_path, labels_path = synthetic_recording
    out_dir = str(tmp_path_factory.mktemp("dataset"))
    cfg = load_config(None)
    entry = run_generate(mcap_path, labels_path, out_dir, cfg)
    return mcap_path, labels_path, out_dir, cfg, entry


def test_frames_kept_and_no_skips(generated):
    _, _, out_dir, _, entry = generated
    # 100 scans / frame_stride 5 = 20 frames, and synthetic TF covers every scan.
    assert entry["frames_kept"] == 20
    assert entry["frames_skipped_pose"] == 0
    assert entry["stamp_fallbacks"] == 0
    assert len(glob.glob(os.path.join(out_dir, "bev", "synthetic", "*.npz"))) == 20


def test_rock_fraction_in_expected_bounds(generated):
    _, _, _, _, entry = generated
    rock = entry["sample_labels"]["rock"]
    clear = entry["sample_labels"]["clear"]
    assert rock > 0, "no rock samples generated - label/frame misalignment"
    frac = rock / (rock + clear)
    # All rock candidates are kept, clear candidates at 5%: rock fraction is
    # substantial but the floor still dominates candidate counts.
    assert 0.02 < frac < 0.95


def test_point_sample_shapes_and_canonicalization(generated):
    _, _, out_dir, cfg, _ = generated
    files = sorted(glob.glob(os.path.join(out_dir, "points", "synthetic", "*.npz")))
    assert files
    d = np.load(files[0])
    n_points = cfg["generator"]["neighborhood_points"]
    S = len(d["labels"])
    assert d["neighborhoods"].shape == (S, n_points, 4)
    assert d["neighborhoods"].dtype == np.float32
    assert d["labels"].dtype == np.int8
    assert set(np.unique(d["labels"])) <= {0, 1}
    assert d["true_counts"].dtype == np.int16
    assert d["centers_odom"].shape == (S, 3)
    assert d["robot_pose"].shape == (4, 4)
    # Canonicalization: local z is shifted so the lowest neighbor sits at 0.
    assert d["neighborhoods"][..., 2].min() >= 0.0
    # xy is center-relative, so bounded by the neighborhood radius.
    r = cfg["generator"]["neighborhood_radius_m"]
    assert np.abs(d["neighborhoods"][..., :2]).max() <= r + 1e-5


def test_bev_rock_cells_at_expected_positions(generated):
    _, _, out_dir, cfg, _ = generated
    gcfg = cfg["generator"]
    files = sorted(glob.glob(os.path.join(out_dir, "bev", "synthetic", "*.npz")))
    d = np.load(files[0])  # first frame: robot near x=0, all 3 rocks in crop
    mask = d["label_mask"]
    base = d["robot_pose"][:3, 3]
    cell = gcfg["bev_cell_m"]
    x0 = base[0] - gcfg["crop_backward_m"]
    y0 = base[1] - gcfg["crop_right_m"]
    for rock in synth.ROCKS:
        cx, cy, _ = rock["center"]
        ix = int(np.floor((cx - x0) / cell))
        iy = int(np.floor((cy - y0) / cell))
        window = mask[max(ix - 1, 0):ix + 2, max(iy - 1, 0):iy + 2]
        assert (window == 1).any(), f"no rock cell within one cell of {rock['center']}"
    # Rock cells only appear near labeled rocks: total rock cells stay small.
    assert 0 < (mask == 1).sum() < 200


def test_z_band_from_the_labeler_cuts_points_out_of_the_dataset(synthetic_recording, tmp_path):
    """The labeler's z clip is saved as a height band, and generate has to
    honour it instead of the crop box's sensor-relative up/down limits."""
    from rocklabel.labels import load_labels

    mcap_path, labels_path = synthetic_recording
    cfg = load_config(None)

    banded_labels = str(tmp_path / "banded.labels.json")
    ls = load_labels(labels_path)
    # A slab far above every synthetic return: nothing should survive it.
    ls.set_z_band(50.0, 60.0)
    ls.save(banded_labels)

    wide = run_generate(mcap_path, labels_path, str(tmp_path / "wide"), cfg)
    narrow = run_generate(mcap_path, banded_labels, str(tmp_path / "narrow"), cfg)

    assert wide["frames_kept"] > 0
    assert narrow["frames_kept"] == 0
    assert narrow["frames_skipped_empty"] == wide["frames_kept"] + wide["frames_skipped_empty"]
    # The manifest records which band a dataset was built under, so two
    # differently-bounded runs can never be pooled by accident unnoticed.
    assert wide["z_band"] is None
    assert narrow["z_band"] == [50.0, 60.0]


def test_z_band_keeps_the_horizontal_crop_limits(synthetic_recording, tmp_path):
    """The band replaces only the vertical crop; forward/back/left/right still
    come from the config, so a generous band cannot pull in the whole room."""
    from rocklabel.labels import load_labels

    mcap_path, labels_path = synthetic_recording
    cfg = load_config(None)
    narrow_cfg = copy.deepcopy(cfg)
    narrow_cfg["generator"]["crop_forward_m"] = 0.5
    narrow_cfg["generator"]["crop_left_m"] = 0.5
    narrow_cfg["generator"]["crop_right_m"] = 0.5

    banded_labels = str(tmp_path / "tall.labels.json")
    ls = load_labels(labels_path)
    ls.set_z_band(-100.0, 100.0)   # vertically unbounded
    ls.save(banded_labels)

    tight = run_generate(mcap_path, banded_labels, str(tmp_path / "tight"), narrow_cfg)
    loose = run_generate(mcap_path, banded_labels, str(tmp_path / "loose"), cfg)
    assert tight["point_labels"]["clear"] < loose["point_labels"]["clear"]


def test_manifest_guard_rejects_changed_config(generated):
    mcap_path, labels_path, out_dir, cfg, _ = generated
    changed = copy.deepcopy(cfg)
    changed["generator"]["bev_cell_m"] = 0.2
    with pytest.raises(ManifestConflict):
        run_generate(mcap_path, labels_path, out_dir, changed)


def test_regenerate_same_run_id_overwrites(generated):
    mcap_path, labels_path, out_dir, cfg, entry = generated
    entry2 = run_generate(mcap_path, labels_path, out_dir, cfg)
    assert entry2["frames_kept"] == entry["frames_kept"]
    assert entry2["point_samples"] == entry["point_samples"]  # seeded RNG: reproducible


def test_missing_lidar_mount_errors_without_fallback(synthetic_recording):
    from rocklabel.pose import PoseUnavailable, build_pose_buffer
    mcap_path, _ = synthetic_recording
    cfg = load_config(None)
    topics = dict(cfg["topics"], lidar_frame="some_other_lidar")
    with pytest.raises(PoseUnavailable, match="static_lidar_to_base"):
        build_pose_buffer(mcap_path, topics)
    # The config fallback satisfies the chain.
    topics["static_lidar_to_base"] = {"translation": [0, 0, 0.4], "quaternion": [0, 0, 0, 1]}
    pb = build_pose_buffer(mcap_path, topics)
    m = pb.lookup("base_link", "some_other_lidar", synth.T0_S + 1.0)
    np.testing.assert_allclose(m[:3, 3], [0, 0, 0.4])


def test_unreachable_odom_frame_fails_loudly(synthetic_recording):
    """A frame misconfiguration must raise a clear error, not skip every scan."""
    from rocklabel.pipeline import ScanStream
    mcap_path, _ = synthetic_recording
    cfg = load_config(None)
    cfg["topics"]["odom_frame"] = "map"  # not present in the synthetic TF tree
    # Levelling makes its own pass first, so the failure surfaces while the
    # stream is being opened rather than on the first scan out of it.
    with pytest.raises(SystemExit, match="no transform chain"):
        next(iter(ScanStream(mcap_path, cfg, progress=False)))


def test_scan_frame_taken_from_message_header(synthetic_recording):
    """Clouds are transformed from their own header.frame_id, so a wrong
    lidar_frame in the config doesn't matter when messages declare theirs."""
    from rocklabel.pipeline import ScanStream
    mcap_path, _ = synthetic_recording
    cfg = load_config(None)
    cfg["topics"]["lidar_frame"] = "wrong_frame_name"
    stream = ScanStream(mcap_path, cfg, progress=False)
    scan = next(iter(stream))
    # Floor points land near z=0 in odom only if the real chain was used.
    assert abs(float(np.median(scan.xyz_odom[:, 2]))) < 0.05


def test_min_range_filter_drops_self_hits(synthetic_recording):
    from rocklabel.pipeline import ScanStream
    mcap_path, _ = synthetic_recording
    cfg = load_config(None)
    cfg["topics"]["min_range_m"] = 1.0
    stream = ScanStream(mcap_path, cfg, progress=False)
    scan = next(iter(stream))
    base_xy = scan.T_odom_base[:2, 3]
    dist = np.linalg.norm(scan.xyz_odom[:, :2] - base_xy, axis=1)
    assert dist.min() >= 1.0
    assert len(scan.xyz_odom) > 1000  # only the near ring removed, not the scene


def test_pipeline_normalizes_raw_counts_in_a_float_intensity_field(tmp_path):
    """Regression: the SICK multiScan publishes RSSI as 0-65535 *floats*, which
    decode_pointcloud2 cannot normalize (no dtype max to divide by). Left
    unscaled, generating from such a bag would write intensity ~40000 while
    every lidarrig recording writes ~0.6, silently poisoning any pooled cache.
    """
    from tests import make_synthetic_mcap as synth

    from rocklabel.pipeline import ScanStream

    raw = str(tmp_path / "raw_counts.mcap")
    synth.write_synthetic_mcap(raw, n_scans=6, intensity_scale=65535.0)
    plain = str(tmp_path / "normalized.mcap")
    synth.write_synthetic_mcap(plain, n_scans=6)

    cfg = load_config(None)
    scan = next(iter(ScanStream(raw, cfg, progress=False)))
    assert scan.intensity.max() <= 1.0
    # ...and the same bag written already-normalized must be left untouched,
    # so the probe never double-scales.
    ref = next(iter(ScanStream(plain, cfg, progress=False)))
    np.testing.assert_allclose(scan.intensity, ref.intensity, atol=1e-4)


def test_accumulate_and_ply_dump(synthetic_recording, tmp_path):
    mcap_path, _ = synthetic_recording
    cfg = load_config(None)
    xyz, inten, counts, stream = accumulate_cloud(mcap_path, cfg, stride=5)
    assert stream.counters.intensity_available
    assert len(xyz) > 1000
    # Cloud spans the floor patch and sits near z=0.
    assert xyz[:, 2].min() > -0.5 and xyz[:, 2].max() < 0.5
    from rocklabel.accumulate import write_ply
    ply = tmp_path / "cloud.ply"
    write_ply(str(ply), xyz, inten)
    assert ply.stat().st_size > 16 * len(xyz)
