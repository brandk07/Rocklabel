"""Tests for the alternative SLAM.

Run with:  pytest tests/test_slam.py -q
"""

from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import pytest

from rocklabel.slam.config import AltSlamConfig
from rocklabel.slam.evaluate import accumulate, detilt_rotation, surface_sharpness
from rocklabel.slam.register import _normal_balance_weights, register, so3_exp
from rocklabel.slam.solver import OfflineSolver, _slerp
from rocklabel.slam.voxelmap import NormalVoxelMap, voxel_downsample
from rocklabel.live.motion import matrix_to_quat, quat_to_matrix
from rocklabel.live.recording import RecordedFrame

RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------- #
# Synthetic scene
# --------------------------------------------------------------------------- #
def scene_points(n_ground=120000, n_walls=40000):
    """A flat ground plane plus four distant walls — a mini volleyball court."""
    g = np.column_stack([
        RNG.uniform(-8, 8, n_ground),
        RNG.uniform(-8, 8, n_ground),
        RNG.normal(0.0, 0.004, n_ground),  # a little surface roughness
    ])
    per = n_walls // 4
    walls = []
    for sign, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        p = np.zeros((per, 3))
        p[:, axis] = sign * 12.0
        p[:, 1 - axis] = RNG.uniform(-12, 12, per)
        p[:, 2] = RNG.uniform(0.0, 3.0, per)
        walls.append(p)
    return np.vstack([g] + walls)


def make_frames(traj, world=None, n_per=1500, dt=0.01):
    """Turn a list of (position, rotation) into RecordedFrames of a fixed scene."""
    if world is None:
        world = scene_points()
    frames = []
    for i, (pos, rot) in enumerate(traj):
        idx = RNG.choice(len(world), n_per, replace=False)
        # World -> sensor frame is the inverse of the sensor's pose.
        pts = (world[idx] - pos) @ rot
        frames.append(RecordedFrame(
            points=pts.astype(np.float32),
            intensity=RNG.uniform(0, 100, n_per).astype(np.float32),
            timestamp=i * dt,
            orientation=matrix_to_quat(rot),
            pose_position=np.zeros(3),
            pose_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            log_time_ns=int(i * dt * 1e9),
        ))
    return frames, world


def straight_line_traj(n=260, dt=0.01, speed=0.35):
    """Sensor drifting along +x with a gentle yaw wobble."""
    traj = []
    for i in range(n):
        t = i * dt
        pos = np.array([speed * t, 0.0, 1.2])
        yaw = 0.25 * math.sin(2.0 * t)
        traj.append((pos, so3_exp(np.array([0.0, 0.0, yaw]))))
    return traj


# --------------------------------------------------------------------------- #
# Voxel map
# --------------------------------------------------------------------------- #
def test_voxel_downsample_reduces_and_stays_in_bounds():
    p = RNG.uniform(-1, 1, (5000, 3))
    d = voxel_downsample(p, 0.25)
    assert d.shape[0] < p.shape[0]
    assert d.min() >= p.min() - 1e-9 and d.max() <= p.max() + 1e-9


def test_map_recovers_a_known_plane_normal():
    """Points on the z=0 plane must yield a +/-z normal and high planarity."""
    pts = np.column_stack([
        RNG.uniform(-2, 2, 4000), RNG.uniform(-2, 2, 4000), np.zeros(4000)
    ])
    m = NormalVoxelMap(0.5, min_points_normal=8)
    m.insert(pts)
    m.refresh_normals()
    usable = m._count >= 8
    assert usable.sum() > 10
    nz = np.abs(m._normal[usable][:, 2])
    assert np.median(nz) > 0.99
    assert np.median(m._planarity[usable]) > 0.3


def test_map_query_finds_the_nearest_patch():
    pts = np.column_stack([
        RNG.uniform(-2, 2, 4000), RNG.uniform(-2, 2, 4000), np.zeros(4000)
    ])
    m = NormalVoxelMap(0.5, min_points_normal=8)
    m.insert(pts)
    probe = np.array([[0.1, 0.1, 0.05], [50.0, 50.0, 50.0]])
    valid, cen, nrm, pla = m.query(probe, 1.0)
    assert valid[0] and not valid[1]
    assert abs(cen[0][2]) < 0.05


def test_map_freezes_a_voxel_after_max_points():
    m = NormalVoxelMap(1.0, max_points=50, min_points_normal=4)
    m.insert(np.zeros((50, 3)))
    before = m._count.copy()
    m.insert(np.zeros((50, 3)) + 0.4)
    assert m._count[0] == before[0]  # frozen, not re-averaged


def test_empty_inputs_are_safe():
    m = NormalVoxelMap(0.3)
    m.insert(np.zeros((0, 3)))
    assert m.size == 0
    valid, _, _, _ = m.query(np.zeros((0, 3)), 1.0)
    assert valid.shape == (0,)
    valid, _, _, _ = m.query(np.ones((5, 3)), 1.0)
    assert not valid.any()


# --------------------------------------------------------------------------- #
# Weighting
# --------------------------------------------------------------------------- #
def test_normal_balance_lifts_the_rare_direction():
    """1000 ground normals + 10 wall normals: each wall point must end up
    weighing far more than each ground point."""
    n = np.zeros((1010, 3))
    n[:1000, 2] = 1.0   # ground
    n[1000:, 0] = 1.0   # wall
    w = _normal_balance_weights(n, 6)
    assert w[1000:].mean() > 50 * w[:1000].mean()
    # and the two groups end up with comparable total say
    assert 0.5 < w[1000:].sum() / w[:1000].sum() < 2.0


def test_so3_exp_matches_a_known_rotation():
    R = so3_exp(np.array([0.0, 0.0, math.pi / 2]))
    assert np.allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-9)
    assert np.allclose(so3_exp(np.zeros(3)), np.eye(3))


def test_slerp_endpoints_and_midpoint():
    q0 = np.array([1.0, 0, 0, 0])
    q1 = matrix_to_quat(so3_exp(np.array([0.0, 0.0, 1.0])))
    assert np.allclose(_slerp(q0, q1, 0.0), q0, atol=1e-9)
    assert np.allclose(abs(_slerp(q0, q1, 1.0) @ q1), 1.0, atol=1e-9)
    assert abs(np.linalg.norm(_slerp(q0, q1, 0.5)) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def build_map(points, cfg):
    m = NormalVoxelMap(cfg.voxel_size, cfg.map_max_points,
                       cfg.min_points_normal, cfg.map_capacity)
    m.insert(points)
    m.refresh_normals()
    return m


def test_register_recovers_a_known_translation():
    cfg = AltSlamConfig()
    cfg.min_matches = 20
    world = scene_points()
    vmap = build_map(world, cfg)
    shift = np.array([0.12, -0.08, 0.05])
    src = voxel_downsample(world, cfg.voxel_size) + shift
    res = register(src, np.zeros(3), vmap, cfg, up=np.array([0.0, 0.0, 1.0]))
    assert res.ok
    # The correction must undo the shift.
    assert np.linalg.norm(res.translation + shift) < 0.03


def test_register_recovers_a_known_yaw():
    cfg = AltSlamConfig()
    cfg.min_matches = 20
    world = scene_points()
    vmap = build_map(world, cfg)
    yaw = math.radians(1.5)
    R = so3_exp(np.array([0.0, 0.0, yaw]))
    src = voxel_downsample(world, cfg.voxel_size) @ R.T
    res = register(src, np.zeros(3), vmap, cfg, up=np.array([0.0, 0.0, 1.0]))
    assert res.ok
    recovered = math.atan2(res.rotation[1, 0], res.rotation[0, 0])
    assert abs(recovered + yaw) < math.radians(0.4)


def test_flat_ground_alone_is_reported_as_degenerate():
    """The whole point of the exercise: a bare plane must NOT be solved for
    sideways motion. Sliding along it has to come back as unobserved."""
    cfg = AltSlamConfig()
    cfg.min_matches = 20
    ground = np.column_stack([
        RNG.uniform(-8, 8, 120000), RNG.uniform(-8, 8, 120000),
        RNG.normal(0, 0.004, 120000),
    ])
    vmap = build_map(ground, cfg)
    src = voxel_downsample(ground, cfg.voxel_size) + np.array([0.15, 0.10, 0.0])
    res = register(src, np.zeros(3), vmap, cfg, up=np.array([0.0, 0.0, 1.0]))
    # At least the two in-plane translation directions are unobservable.
    assert res.suppressed >= 2
    # And it must not have invented a large sideways correction.
    assert np.linalg.norm(res.translation[:2]) < 0.05


def test_walls_make_the_same_scene_observable():
    """Add structure and the very same sideways shift becomes solvable."""
    cfg = AltSlamConfig()
    cfg.min_matches = 20
    world = scene_points()
    vmap = build_map(world, cfg)
    shift = np.array([0.15, 0.10, 0.0])
    src = voxel_downsample(world, cfg.voxel_size) + shift
    res = register(src, np.zeros(3), vmap, cfg, up=np.array([0.0, 0.0, 1.0]))
    assert res.ok
    assert np.linalg.norm(res.translation[:2] + shift[:2]) < 0.04


def test_register_refuses_an_empty_map():
    cfg = AltSlamConfig()
    vmap = NormalVoxelMap(cfg.voxel_size)
    res = register(RNG.uniform(-1, 1, (500, 3)), np.zeros(3), vmap, cfg)
    assert not res.ok
    assert np.allclose(res.rotation, np.eye(3))
    assert np.allclose(res.translation, 0.0)


def test_outliers_do_not_drag_the_solution():
    """A slab of bogus points (a person walking through) must be shrugged off."""
    cfg = AltSlamConfig()
    cfg.min_matches = 20
    world = scene_points()
    vmap = build_map(world, cfg)
    src = voxel_downsample(world, cfg.voxel_size)
    junk = np.column_stack([
        RNG.uniform(-1, 1, 400), RNG.uniform(-1, 1, 400),
        RNG.uniform(1.0, 1.6, 400),
    ])
    res = register(np.vstack([src, junk]), np.zeros(3), vmap, cfg,
                   up=np.array([0.0, 0.0, 1.0]))
    assert res.ok
    assert np.linalg.norm(res.translation) < 0.05


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def test_solver_tracks_a_straight_line():
    traj = straight_line_traj()
    frames, _ = make_frames(traj)
    cfg = AltSlamConfig()
    cfg.passes = 2
    cfg.min_matches = 40
    cfg.window_sec = 0.05
    s = OfflineSolver(cfg)
    s.build_windows(frames)
    s.solve()
    pos, quats = s.batch_poses(frames)
    truth = np.array([p for p, _ in traj])
    # The world frame is anchored at the sensor's starting pose, so compare
    # motion relative to the start rather than absolute coordinates.
    err = np.linalg.norm((pos - pos[0]) - (truth - truth[0]), axis=1)
    assert np.median(err) < 0.12, f"median tracking error {np.median(err):.3f} m"
    assert pos.shape == (len(frames), 3)
    assert quats.shape == (len(frames), 4)
    assert np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-6)


def test_solver_needs_imu():
    frames, _ = make_frames(straight_line_traj(20))
    for f in frames:
        f.orientation = None
    with pytest.raises(ValueError):
        OfflineSolver(AltSlamConfig()).build_windows(frames)


def test_solver_rejects_solve_before_build():
    with pytest.raises(ValueError):
        OfflineSolver(AltSlamConfig()).solve()


def test_batch_poses_are_continuous():
    """Smoothed output must not step at window boundaries."""
    frames, _ = make_frames(straight_line_traj(200))
    cfg = AltSlamConfig()
    cfg.passes = 1
    cfg.min_matches = 40
    cfg.window_sec = 0.05
    s = OfflineSolver(cfg)
    s.build_windows(frames)
    s.solve()
    pos, _ = s.batch_poses(frames)
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    # No single batch-to-batch jump should dwarf the typical one.
    assert steps.max() < 10.0 * np.median(steps) + 1e-3


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def test_detilt_rotation_puts_up_on_z():
    up = np.array([-0.636, 0.01, 0.771])
    R = detilt_rotation(up)
    assert np.allclose(R @ (up / np.linalg.norm(up)), [0, 0, 1], atol=1e-9)
    assert np.allclose(detilt_rotation(np.array([0.0, 0, 1])), np.eye(3))


def test_sharpness_detects_a_smeared_surface():
    n = 250000
    flat = np.column_stack([
        RNG.uniform(-4, 4, n), RNG.uniform(-4, 4, n), RNG.normal(0, 0.005, n),
    ])
    smeared = flat + np.column_stack([
        np.zeros(n), np.zeros(n), RNG.normal(0, 0.05, n)
    ])
    up = np.array([0.0, 0.0, 1.0])
    a = surface_sharpness(flat, up)["median_mm"]
    b = surface_sharpness(smeared, up)["median_mm"]
    assert a < 12.0 and b > 40.0


def test_sharpness_handles_empty_input():
    r = surface_sharpness(np.zeros((0, 3)), np.array([0.0, 0, 1]))
    assert r["cells"] == 0 and math.isnan(r["median_mm"])


# --------------------------------------------------------------------------- #
# Round-trip through MCAP
# --------------------------------------------------------------------------- #
def test_reprocess_writes_a_readable_copy_and_leaves_the_source_alone(tmp_path):
    from rocklabel.slam.reprocess import load_frames, reprocess, write_frames

    frames, _ = make_frames(straight_line_traj(160))
    src = str(tmp_path / "in.mcap")
    dst = str(tmp_path / "out.mcap")
    write_frames(src, frames,
                 np.zeros((len(frames), 3)),
                 np.tile([1.0, 0, 0, 0], (len(frames), 1)),
                 {"config_yaml": "slam:\n  enabled: true\n"})
    before = hashlib.sha256(open(src, "rb").read()).hexdigest()

    cfg = AltSlamConfig()
    cfg.passes = 1
    cfg.min_matches = 40
    cfg.window_sec = 0.05
    report = reprocess(src, dst, cfg, score=False)

    # Source untouched, byte for byte.
    assert hashlib.sha256(open(src, "rb").read()).hexdigest() == before
    assert os.path.exists(dst)

    a = load_frames(src)
    b = load_frames(dst)
    assert len(a) == len(b) == len(frames)
    for x, y in zip(a, b):
        assert np.array_equal(x.points, y.points)          # geometry preserved
        assert np.array_equal(x.intensity, y.intensity)    # brightness preserved
        assert np.allclose(x.orientation, y.orientation)   # IMU preserved
        assert x.timestamp == y.timestamp
    # ...and the poses are the one thing that changed.
    moved = sum(not np.allclose(x.pose_position, y.pose_position) for x, y in zip(a, b))
    assert moved > len(a) // 2
    assert report["batches"] == len(frames)


def test_reprocess_rejects_an_empty_recording(tmp_path):
    from rocklabel.slam.reprocess import reprocess, write_frames

    src = str(tmp_path / "empty.mcap")
    write_frames(src, [], np.zeros((0, 3)), np.zeros((0, 4)), {})
    with pytest.raises(ValueError):
        reprocess(src, str(tmp_path / "o.mcap"))


def test_score_only_writes_nothing(tmp_path):
    from rocklabel.slam.reprocess import reprocess, write_frames

    frames, _ = make_frames(straight_line_traj(120))
    src = str(tmp_path / "in.mcap")
    dst = str(tmp_path / "out.mcap")
    write_frames(src, frames, np.zeros((len(frames), 3)),
                 np.tile([1.0, 0, 0, 0], (len(frames), 1)), {})
    cfg = AltSlamConfig()
    cfg.passes = 1
    cfg.min_matches = 40
    cfg.window_sec = 0.05
    reprocess(src, dst, cfg, score=False, write=False)
    assert not os.path.exists(dst)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_output_naming():
    from rocklabel.slam.cli import build_parser, config_from_args, output_path

    a = build_parser().parse_args(["recordings/Run1.mcap"])
    assert output_path("recordings/Run1.mcap", a) == "recordings/Run1.reslam.mcap"
    a = build_parser().parse_args(["r/Run1.mcap", "--out-dir", "/tmp/x",
                                   "--suffix", ".v2"])
    assert output_path("r/Run1.mcap", a) == "/tmp/x/Run1.v2.mcap"


def test_cli_flags_reach_the_config():
    from rocklabel.slam.cli import build_parser, config_from_args

    a = build_parser().parse_args(["x.mcap", "--passes", "4", "--voxel", "0.3",
                                   "--lock-tilt", "--degeneracy", "0.1"])
    cfg = config_from_args(a)
    assert cfg.passes == 4
    assert cfg.voxel_size == 0.3
    assert cfg.lock_roll_pitch is True
    assert cfg.degeneracy_threshold == 0.1


def test_cli_refuses_output_with_many_inputs():
    from rocklabel.slam.cli import main

    assert main(["a.mcap", "b.mcap", "-o", "out.mcap"]) == 2


# --------------------------------------------------------------------------- #
# Revisit diagnostics
# --------------------------------------------------------------------------- #
def _flat_visits(offsets, noise=0.005, n=40000, dt=1.0):
    """One frame per 'visit' of a flat patch, each nudged up/down by offsets[i].

    Sensor-frame geometry is identical every visit, so anything the diagnostics
    report as *between*-visit error is exactly the offset we injected.
    """
    frames = []
    for i, off in enumerate(offsets):
        g = np.column_stack([
            RNG.uniform(-2, 2, n), RNG.uniform(-2, 2, n), RNG.normal(0, noise, n),
        ])
        frames.append(RecordedFrame(
            points=g.astype(np.float32),
            intensity=np.ones(n, dtype=np.float32),
            timestamp=i * dt,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            pose_position=np.array([0.0, 0.0, off]),
            pose_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            log_time_ns=int(i * dt * 1e9),
        ))
    return frames


def test_revisit_error_separates_sensor_noise_from_pose_error():
    from rocklabel.slam.evaluate import revisit_error

    offsets = [0.0, 0.02, -0.02, 0.01, -0.01]
    frames = _flat_visits(offsets, noise=0.005)
    r = revisit_error(frames, up=np.array([0.0, 0.0, 1.0]), bin_sec=1.0,
                      stride=1, radius=8.0)
    # The floor should read back the roughness we put in (5 mm).
    assert 3.0 < r["within_mm"] < 8.0, r
    # ...and the pose term should read back the spread of the offsets.
    expected = np.std(offsets) * 1000
    assert abs(r["between_mm"] - expected) < 5.0, (r, expected)
    assert r["cells"] > 100


def test_revisit_error_reports_a_clean_trajectory_as_clean():
    from rocklabel.slam.evaluate import revisit_error

    r = revisit_error(_flat_visits([0.0] * 5, noise=0.005),
                      up=np.array([0.0, 0.0, 1.0]), bin_sec=1.0, stride=1)
    assert r["between_mm"] < 3.0, r


def test_error_vs_gap_detects_accumulating_drift():
    """A trajectory that slides steadily must show error growing with the gap."""
    from rocklabel.slam.evaluate import error_vs_gap

    drift = np.linspace(0.0, 0.12, 14)
    rows = error_vs_gap(_flat_visits(drift, noise=0.003), up=np.array([0.0, 0.0, 1.0]),
                        bin_sec=1.0, stride=1, min_pairs=50)
    assert len(rows) >= 3, rows
    assert rows[-1][2] > 3.0 * rows[0][2], rows


def test_error_vs_gap_is_flat_without_drift():
    """Random per-visit wobble (no accumulation) must NOT grow with the gap —
    this is the signature that says loop closure would not help."""
    from rocklabel.slam.evaluate import error_vs_gap

    wobble = RNG.normal(0, 0.01, 14)
    rows = error_vs_gap(_flat_visits(wobble, noise=0.003), up=np.array([0.0, 0.0, 1.0]),
                        bin_sec=1.0, stride=1, min_pairs=50)
    assert len(rows) >= 3, rows
    assert rows[-1][2] < 2.0 * rows[0][2], rows


def test_revisit_diagnostics_need_an_up_vector():
    from rocklabel.slam.evaluate import error_vs_gap, revisit_error

    with pytest.raises(ValueError):
        revisit_error(_flat_visits([0.0, 0.01]))
    with pytest.raises(ValueError):
        error_vs_gap(_flat_visits([0.0, 0.01]))
