"""Offline levelling: ground fit, manual angles, frame pinning, and the
label/generate frame guard."""

import json

import numpy as np
import pytest

from rocklabel.config import apply_overrides, config_hash, load_config
from rocklabel.labels import SCHEMA_VERSION, LabelSet, load_labels
from rocklabel.geometry.leveling import (
    ALREADY_LEVEL_DEG,
    LevelError,
    MODES,
    LevelSolution,
    check_level_match,
    level_record,
    mode_of,
    pin_level_to_labels,
    solve_level,
)
from rocklabel.live.leveling import mount_rotation, roll_pitch_deg, tilt_deg
from rocklabel.recording.pipeline import LevelledScanStream, OdomScan, ScanStream

ROLL_DEG, PITCH_DEG = -3.0, 25.0


def _floor_scans(level_rot, n_scans=12, sensor_height=0.8, seed=0, step=0.4):
    """Scans of a flat floor seen from a rig that walks, tipped by a mount.

    ``step=0`` parks the rig: its path then says nothing about the tilt, which
    is what forces the ground fit to work unaided.

    Ground truth is a floor at ``z = -sensor_height`` with the sensor at
    ``z = 0``. Everything - points and poses alike - is then tipped by the
    *inverse* of ``level_rot``, so ``level_rot`` is exactly the correction
    ``solve_level`` is expected to recover. (``mount_rotation`` returns the
    correction for a given mount angle, not the tilt itself.)
    """
    rot = np.asarray(level_rot).T
    rng = np.random.default_rng(seed)
    scans = []
    for i in range(n_scans):
        origin = np.array([step * i, 0.0, 0.0])
        # An annulus of floor around the sensor, inside the default 0.6-6.0 m band.
        r = rng.uniform(1.0, 5.0, 4000)
        th = rng.uniform(0, 2 * np.pi, 4000)
        pts = np.column_stack([
            origin[0] + r * np.cos(th),
            origin[1] + r * np.sin(th),
            np.full(4000, -sensor_height) + rng.normal(0, 0.005, 4000),
        ])
        pose = np.eye(4)
        pose[:3, 3] = origin
        tilted = np.eye(4)
        tilted[:3, :3] = rot
        scans.append(OdomScan(
            index=i,
            time_s=float(i) * 0.1,
            xyz_odom=(pts @ rot.T).astype(np.float32),
            intensity=np.zeros(len(pts), np.float32),
            T_odom_base=tilted @ pose,
            T_odom_lidar=tilted @ pose,
        ))
    return scans


def _factory(scans):
    return lambda: iter(scans)


def _cfg(**level):
    cfg = load_config(None)
    cfg["level"].update(level)
    return cfg


# --------------------------------------------------------------------------- #
# solve_level
# --------------------------------------------------------------------------- #
def test_off_returns_no_solution():
    assert solve_level(_cfg(mode="off"), _factory(_floor_scans(np.eye(3)))) is None


def test_ground_fit_recovers_the_mount_angle():
    correction = mount_rotation(ROLL_DEG, PITCH_DEG)
    solution = solve_level(_cfg(mode="ground"), _factory(_floor_scans(correction)))

    assert solution.source == "ground"
    roll, pitch = roll_pitch_deg(solution.rotation)
    assert roll == pytest.approx(ROLL_DEG, abs=0.3)
    assert pitch == pytest.approx(PITCH_DEG, abs=0.3)
    # The fit undoes the tilt rather than merely measuring it.
    assert tilt_deg(solution.rotation @ correction.T) < 0.5
    assert solution.floor_z == pytest.approx(-0.8, abs=0.05)
    assert solution.inlier_frac > 0.9


def test_ground_fit_levels_an_already_level_recording():
    solution = solve_level(_cfg(mode="ground"), _factory(_floor_scans(np.eye(3))))
    assert tilt_deg(solution.rotation) < 0.5


def test_manual_uses_given_angles_and_still_measures_the_floor():
    correction = mount_rotation(ROLL_DEG, PITCH_DEG)
    solution = solve_level(
        _cfg(mode="manual", mount_roll_deg=ROLL_DEG, mount_pitch_deg=PITCH_DEG),
        _factory(_floor_scans(correction)),
    )
    assert solution.source == "manual"
    np.testing.assert_allclose(solution.rotation, mount_rotation(ROLL_DEG, PITCH_DEG))
    # No tape measure of mount angle tells you where the floor is; the fit does.
    assert solution.floor_z == pytest.approx(-0.8, abs=0.05)


def test_unknown_mode_is_rejected():
    with pytest.raises(LevelError, match="unknown level.mode"):
        solve_level(_cfg(mode="imu"), _factory(_floor_scans(np.eye(3))))


def test_yaml_boolean_off_is_treated_as_off_not_as_a_mode():
    """An unquoted `mode: off` in YAML parses as False, not the string."""
    assert mode_of(_cfg(mode=False)) == "off"
    assert solve_level(_cfg(mode=False), _factory(_floor_scans(np.eye(3)))) is None
    # `mode: on` is a boolean too, but it is not a mode - fail loudly.
    with pytest.raises(LevelError, match="parse as booleans"):
        solve_level(_cfg(mode=True), _factory(_floor_scans(np.eye(3))))


def test_shipped_configs_keep_the_mode_a_string():
    """Regression for the same footgun in the checked-in YAML files."""
    import glob

    for path in sorted(glob.glob("config*.yaml")):
        mode = load_config(path)["level"]["mode"]
        assert isinstance(mode, str), path
        assert mode in MODES, path


def test_ceiling_is_rejected_rather_than_levelled_to():
    """A ceiling is just as planar and just as level as a floor."""
    scans = _floor_scans(np.eye(3), sensor_height=-2.5)  # plane *above* the sensor
    with pytest.raises(LevelError, match="ceiling, not the floor"):
        solve_level(_cfg(mode="ground"), _factory(scans))


def test_a_plane_tilted_past_the_gate_is_not_accepted():
    """With nothing to seed from, the tilt gate is what keeps the fit off walls."""
    steep = mount_rotation(0.0, 75.0)
    parked = _floor_scans(steep, step=0.0)  # no travel, so no path to read
    with pytest.raises(LevelError, match="no plane within"):
        solve_level(_cfg(mode="ground", max_tilt_deg=30.0), _factory(parked))


def test_a_readable_path_recovers_a_tilt_past_the_gate():
    """A rig that walks reveals the tilt regardless of how steep it is.

    The gate exists to stop an unseeded fit locking onto a wall. Once the
    sensor's own path has said which way is up, that job is done and a mount
    steeper than the gate is recovered anyway.
    """
    steep = mount_rotation(0.0, 75.0)
    solution = solve_level(_cfg(mode="ground", max_tilt_deg=30.0),
                           _factory(_floor_scans(steep)))
    assert tilt_deg(solution.rotation) == pytest.approx(75.0, abs=1.0)


def test_range_band_that_excludes_every_point_fails_loudly():
    scans = _floor_scans(np.eye(3))
    with pytest.raises(LevelError, match="nothing to fit"):
        solve_level(_cfg(mode="ground", range_min_m=50.0, range_max_m=60.0), _factory(scans))


def test_fit_scans_bounds_the_pass():
    """fit_scans caps how much of the recording is pooled."""
    correction = mount_rotation(ROLL_DEG, PITCH_DEG)
    scans = _floor_scans(correction, n_scans=12)
    few = solve_level(_cfg(mode="ground", fit_scans=2), _factory(scans))
    every = solve_level(_cfg(mode="ground"), _factory(scans))
    assert few.pooled_points < every.pooled_points
    assert tilt_deg(few.rotation @ correction.T) < 0.5


# --------------------------------------------------------------------------- #
# LevelledScanStream
# --------------------------------------------------------------------------- #
class _FakeStream:
    format_name = "fake"

    def __init__(self, scans):
        self.scans = scans
        self.counters = "counters-sentinel"
        self.info = "info-sentinel"

    @property
    def scan_count(self):
        return len(self.scans)

    def __iter__(self):
        return iter(self.scans)


def test_levelled_stream_rotates_points_and_both_poses():
    scans = _floor_scans(mount_rotation(ROLL_DEG, PITCH_DEG), n_scans=3)
    solution = solve_level(_cfg(mode="ground"), _factory(scans))
    stream = LevelledScanStream(_FakeStream(scans), solution)

    for out in stream:
        # The floor lands flat...
        assert out.xyz_odom[:, 2].std() < 0.02
        assert out.xyz_odom[:, 2].mean() == pytest.approx(-0.8, abs=0.05)
        # ...and the poses come with it, so a crop box built from T_odom_base
        # is aligned with the levelled floor rather than with the mast.
        assert out.T_odom_base[2, 3] == pytest.approx(0.0, abs=0.05)
        np.testing.assert_allclose(out.T_odom_base, out.T_odom_lidar)


def test_levelled_stream_proxies_the_wrapped_stream():
    scans = _floor_scans(np.eye(3), n_scans=3)
    inner = _FakeStream(scans)
    stream = LevelledScanStream(inner, LevelSolution(np.eye(3), "manual"))
    assert stream.counters == "counters-sentinel"
    assert stream.info == "info-sentinel"
    assert stream.scan_count == 3
    assert stream.format_name == "fake"


# --------------------------------------------------------------------------- #
# level_record / pinning / the frame guard
# --------------------------------------------------------------------------- #
def test_level_record_is_none_for_an_unlevelled_stream():
    assert level_record(_FakeStream([])) is None


def test_level_record_round_trips_through_a_label_file(tmp_path):
    solution = LevelSolution(mount_rotation(ROLL_DEG, PITCH_DEG), "ground",
                             floor_z=-0.8, inlier_frac=0.9)
    record = level_record(LevelledScanStream(_FakeStream([]), solution))
    assert record["mode"] == "ground"
    assert record["roll_deg"] == pytest.approx(ROLL_DEG, abs=1e-3)
    assert record["pitch_deg"] == pytest.approx(PITCH_DEG, abs=1e-3)

    path = tmp_path / "x.labels.json"
    labelset = LabelSet(level=record)
    labelset.add([1.0, 2.0, 3.0], 0.2)
    labelset.save(str(path))
    assert load_labels(str(path)).level == record


def test_pre_v4_label_files_load_as_unlevelled(tmp_path):
    """Absent 'level' means unlevelled, so old work keeps loading and matching."""
    path = tmp_path / "v3.labels.json"
    path.write_text(json.dumps({
        "schema_version": 3, "mcap_file": "a.mcap", "run_id": "a",
        "rocks": [{"id": 1, "shape": "sphere", "center": [0, 0, 0], "radius": 0.1}],
    }))
    labelset = load_labels(str(path))
    assert labelset.level is None
    check_level_match(labelset.level, None, str(path))  # does not raise


def test_saved_files_declare_the_current_schema(tmp_path):
    path = tmp_path / "x.labels.json"
    LabelSet().save(str(path))
    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION


def test_matching_frames_pass_and_mismatched_ones_raise():
    ground = {"mode": "ground", "roll_deg": -0.64, "pitch_deg": 39.25}

    check_level_match(ground, dict(ground), "x.json")
    check_level_match(None, None, "x.json")
    # A ground fit lands a fraction of a degree away on a different stride.
    check_level_match(ground, {"mode": "ground", "roll_deg": -0.70, "pitch_deg": 39.62},
                      "x.json")

    with pytest.raises(LevelError, match="different frame"):
        check_level_match(ground, None, "x.json")
    with pytest.raises(LevelError, match="different frame"):
        check_level_match(None, ground, "x.json")
    with pytest.raises(LevelError, match="different frame"):
        check_level_match(ground, {"mode": "ground", "roll_deg": -0.64, "pitch_deg": 35.0},
                          "x.json")


def test_mismatch_message_names_the_flags_that_fix_it():
    ground = {"mode": "ground", "roll_deg": -0.64, "pitch_deg": 39.25}
    with pytest.raises(LevelError, match=r"--mount-roll -0.64 --mount-pitch 39.25"):
        check_level_match(ground, None, "x.json")
    # The other direction is fixed by turning levelling off, not by a 0/0 angle.
    with pytest.raises(LevelError, match=r"--level off"):
        check_level_match(None, ground, "x.json")


def test_pinning_replays_the_labelled_angle_exactly():
    cfg = _cfg(mode="ground")
    record = {"mode": "ground", "roll_deg": -0.6356, "pitch_deg": 39.2459}
    pinned = pin_level_to_labels(cfg, record)

    assert pinned["level"]["mode"] == "manual"
    assert pinned["level"]["mount_roll_deg"] == -0.6356
    assert pinned["level"]["mount_pitch_deg"] == 39.2459
    # The caller's cfg is what gets hashed into the manifest; pinning is
    # per-recording and must not touch it.
    assert cfg["level"]["mode"] == "ground"
    assert cfg["level"]["mount_roll_deg"] == 0.0


def test_pinning_leaves_explicit_modes_alone():
    """'manual' is the user overriding the measurement; 'off' has nothing to pin."""
    record = {"mode": "ground", "roll_deg": 1.0, "pitch_deg": 2.0}
    for mode in ("off", "manual"):
        cfg = _cfg(mode=mode)
        assert pin_level_to_labels(cfg, record) is cfg
    assert pin_level_to_labels(_cfg(mode="ground"), None) is not None


def test_pinned_solution_reproduces_the_labelled_frame_bit_for_bit():
    scans = _floor_scans(mount_rotation(ROLL_DEG, PITCH_DEG))
    labelled = level_record(LevelledScanStream(
        _FakeStream(scans), solve_level(_cfg(mode="ground"), _factory(scans))))

    # A different stride pools different scans, so a fresh fit would land a
    # fraction of a degree away. Pinned, it reproduces the angle exactly.
    generating = pin_level_to_labels(_cfg(mode="ground"), labelled)
    solution = solve_level(generating, _factory(scans[::3]))
    roll, pitch = roll_pitch_deg(solution.rotation)
    assert (round(roll, 4), round(pitch, 4)) == (labelled["roll_deg"], labelled["pitch_deg"])
    check_level_match(labelled, level_record(
        LevelledScanStream(_FakeStream([]), solution)), "x.json")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_levelling_is_automatic_by_default():
    """Off-by-default is what let thirteen tilted runs be labelled as-is."""
    assert load_config(None)["level"]["mode"] == "auto"


def test_levelling_changes_the_config_hash():
    """A levelled dataset must never be mixed into an unlevelled one."""
    base = load_config(None)
    assert config_hash(apply_overrides(base, {"level.mode": "ground"})) != config_hash(base)
    assert config_hash(apply_overrides(base, {"level.mount_pitch_deg": 30.0})) != config_hash(base)


# --------------------------------------------------------------------------- #
# through the real pipeline
# --------------------------------------------------------------------------- #
def test_scanstream_is_untouched_when_levelling_is_off(synthetic_recording):
    mcap_path, _ = synthetic_recording
    stream = ScanStream(mcap_path, load_config(None), stride=10, progress=False)
    assert not isinstance(stream, LevelledScanStream)
    assert level_record(stream) is None


def test_scanstream_applies_a_manual_mount_angle(synthetic_recording):
    """The synthetic floor is level, so a manual angle tips it by exactly that."""
    mcap_path, _ = synthetic_recording
    cfg = load_config(None)
    cfg["level"].update(mode="manual", mount_roll_deg=0.0, mount_pitch_deg=PITCH_DEG)
    stream = ScanStream(mcap_path, cfg, stride=10, progress=False)
    assert isinstance(stream, LevelledScanStream)

    flat = next(iter(ScanStream(mcap_path, load_config(None), stride=10, progress=False)))
    tipped = next(iter(stream))
    expected = flat.xyz_odom @ mount_rotation(0.0, PITCH_DEG).T.astype(np.float32)
    np.testing.assert_allclose(tipped.xyz_odom, expected, atol=1e-4)


def test_scanstream_ground_fit_levels_the_synthetic_recording(synthetic_recording):
    mcap_path, _ = synthetic_recording
    cfg = load_config(None)
    cfg["level"]["mode"] = "ground"
    stream = ScanStream(mcap_path, cfg, stride=10, progress=False)
    # The synthetic floor is already flat: the fit must find it and barely move.
    assert tilt_deg(stream.solution.rotation) < 1.0
    assert stream.solution.floor_z == pytest.approx(0.0, abs=0.1)


# --------------------------------------------------------------------------- #
# Catching a recording whose world frame was never levelled at capture time
# --------------------------------------------------------------------------- #
def _walk(tilt_deg_: float, n: int = 400, wobble: float = 0.0, seed: int = 0,
          span: float = 3.0):
    """A rig walking a straight line over flat ground, in a frame tilted by
    ``tilt_deg_`` — so its recorded height climbs as it goes."""
    import math as _math

    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, span, n)
    y = 0.3 * np.sin(np.linspace(0.0, 4.0, n))
    z = _math.tan(_math.radians(tilt_deg_)) * x + rng.normal(0.0, wobble, n)
    return np.column_stack([x, y, z])


def test_path_tilt_measures_an_unlevelled_frame():
    from rocklabel.recording.pipeline import path_tilt_deg

    measured = path_tilt_deg(_walk(31.0))
    assert measured is not None
    tilt, slope = measured
    assert tilt == pytest.approx(31.0, abs=0.5)
    assert slope == pytest.approx(0.60, abs=0.02)


def test_path_tilt_stays_quiet_on_a_level_recording():
    from rocklabel.recording.pipeline import path_tilt_deg

    assert path_tilt_deg(_walk(0.0, wobble=0.02)) is None
    assert path_tilt_deg(_walk(2.0, wobble=0.01)) is None  # under the gate


def test_path_tilt_will_not_call_it_on_a_rig_that_stayed_put():
    """Bobbing up and down on the spot is not evidence of anything."""
    from rocklabel.recording.pipeline import path_tilt_deg

    assert path_tilt_deg(_walk(31.0, span=0.2)) is None          # no travel
    assert path_tilt_deg(_walk(31.0, wobble=2.0, seed=3)) is None  # all noise
    assert path_tilt_deg(np.zeros((2, 3))) is None                # nothing to fit


# --------------------------------------------------------------------------- #
# auto: measure it without being told, and never fail
# --------------------------------------------------------------------------- #
def test_auto_recovers_the_mount_angle():
    correction = mount_rotation(ROLL_DEG, PITCH_DEG)
    solution = solve_level(_cfg(mode="auto"), _factory(_floor_scans(correction)))

    assert solution.source == "auto"
    roll, pitch = roll_pitch_deg(solution.rotation)
    assert roll == pytest.approx(ROLL_DEG, abs=0.5)
    assert pitch == pytest.approx(PITCH_DEG, abs=0.5)
    assert solution.floor_z == pytest.approx(-0.8, abs=0.02)


def test_auto_leaves_an_already_level_recording_alone():
    """A run the live rig levelled at capture time must not be re-levelled.

    This is what makes 'auto' safe as a default: on the runs that were already
    right it does nothing at all, rather than erroring or adding a fresh
    fraction of a degree of its own.
    """
    assert solve_level(_cfg(mode="auto"), _factory(_floor_scans(np.eye(3)))) is None


def test_auto_never_raises_where_ground_would():
    """'auto' is the default, so a recording it cannot read must still open."""
    empty_band = _cfg(mode="auto", range_min_m=50.0, range_max_m=60.0)
    scans = _floor_scans(mount_rotation(0.0, 30.0))
    with pytest.raises(LevelError):
        solve_level(_cfg(mode="ground", range_min_m=50.0, range_max_m=60.0),
                    _factory(scans))
    # Same recording, same impossible band: auto falls back to the path seed.
    solution = solve_level(empty_band, _factory(scans))
    assert solution is not None
    assert tilt_deg(solution.rotation) == pytest.approx(30.0, abs=2.0)
    assert "seed" in solution.note


def test_auto_ignores_a_ceiling():
    """Indoors the ceiling is as planar and as level as the floor.

    Seeding from the path puts the frame near level, which is what makes
    'below the sensor' a meaningful test again.
    """
    scans = _floor_scans(mount_rotation(0.0, 25.0), sensor_height=-2.5)
    solution = solve_level(_cfg(mode="auto"), _factory(scans))
    # Either it declines to use the ceiling and keeps the path seed, or it
    # returns nothing — but it must never level *to* the plane overhead.
    assert solution is None or solution.floor_z is None or solution.floor_z < 0.0


def test_path_level_rotation_reads_the_tilt_off_the_poses():
    from rocklabel.geometry.leveling import path_level_rotation

    walk = np.column_stack([
        np.linspace(0.0, 4.0, 60),
        np.zeros(60),
        np.tan(np.radians(30.0)) * np.linspace(0.0, 4.0, 60),
    ])
    rot, r2 = path_level_rotation(walk)
    assert tilt_deg(rot) == pytest.approx(30.0, abs=0.2)
    assert r2 > 0.99
    # Flat path, parked rig, and a two-sample path all say "I cannot tell".
    flat = np.column_stack([np.linspace(0, 4, 60), np.zeros(60), np.zeros(60)])
    assert path_level_rotation(flat) is None
    assert path_level_rotation(np.zeros((60, 3))) is None
    assert path_level_rotation(walk[:2]) is None


def test_auto_is_pinned_to_the_label_file_like_ground_is():
    """label and generate must not each measure their own angle."""
    from rocklabel.geometry.leveling import pin_level_to_labels

    record = {"mode": "auto", "roll_deg": 1.25, "pitch_deg": 28.5}
    pinned = pin_level_to_labels(_cfg(mode="auto"), record)
    assert pinned["level"]["mode"] == "manual"
    assert pinned["level"]["mount_pitch_deg"] == pytest.approx(28.5)


def test_labels_made_before_levelling_existed_keep_their_frame():
    """Turning levelling on by default must not move anyone's existing rocks.

    A label file with no frame recorded predates levelling, so its centers are
    world coordinates in the recording's own tilted frame. Re-measuring an
    angle now would slide every one of them off the rock it marks.
    """
    from rocklabel.geometry.leveling import pin_level_to_labels

    assert pin_level_to_labels(_cfg(mode="auto"), None)["level"]["mode"] == "off"
    assert pin_level_to_labels(_cfg(mode="ground"), None)["level"]["mode"] == "off"
    # An explicit choice is the user's, and is left alone.
    assert pin_level_to_labels(_cfg(mode="off"), None)["level"]["mode"] == "off"
    assert pin_level_to_labels(_cfg(mode="manual"), None)["level"]["mode"] == "manual"
