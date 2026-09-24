"""Adaptive radii, causal scan history, and the shared version-2 input contract."""

import copy
import glob
import json
import os

import numpy as np
import pytest

from rocklabel.config import ConfigError, DEFAULTS, config_hash, load_config, validate_generator
from rocklabel.dataset.history import (HistoryPolicy, RadiusPolicy, Support, Sweep,
                                       SweepAssembler, SweepHistory, assemble_support,
                                       build_neighborhoods, candidate_centers, input_contract,
                                       is_legacy, sample_rng, select_slots,
                                       single_sweep_support)
from rocklabel.dataset.labeling import LABEL_CLEAR, LABEL_ROCK, label_rocks
from rocklabel.dataset.neighborhoods import (QUERY_HEIGHT_CHANNEL,
                                             build_neighborhood_samples)


def _gcfg(**over):
    g = copy.deepcopy(DEFAULTS["generator"])
    g.update(over)
    return g


def _v2(**over):
    return _gcfg(preprocessing_version=2, formats=["points"], **over)


def _sweep(i, t, xyz=None, origin=(0.0, 0.0, 0.0)):
    xyz = np.zeros((1, 3), np.float32) if xyz is None else xyz
    return Sweep(i, float(t), xyz, np.zeros(len(xyz), np.float32),
                 np.asarray(origin, float))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_new_keys_leave_legacy_config_hashes_alone():
    cfg = load_config(None)
    legacy = copy.deepcopy(cfg)
    for key in ("preprocessing_version", "neighborhood_mode", "adaptive_radius_min_m",
                "adaptive_radius_max_m", "adaptive_k", "history_ages_s",
                "history_tolerance_s", "formats"):
        del legacy["generator"][key]
    assert config_hash(cfg) == config_hash(legacy)
    changed = copy.deepcopy(cfg)
    changed["generator"].update(preprocessing_version=2, formats=["points"],
                                history_ages_s=[0.0, 2.0])
    assert config_hash(changed) != config_hash(cfg)


def test_arms_get_distinct_hashes():
    seen = set()
    for mode, ages in (("fixed", [0.0]), ("adaptive", [0.0]), ("fixed", [0.0, 2.0])):
        cfg = load_config(None)
        cfg["generator"].update(preprocessing_version=2, formats=["points"],
                                neighborhood_mode=mode, history_ages_s=ages)
        seen.add(config_hash(cfg))
    assert len(seen) == 3


@pytest.mark.parametrize("bad", [
    {"neighborhood_mode": "adaptive"},                          # needs version 2
    {"history_ages_s": [0.0, 2.0]},                             # needs version 2
    {"preprocessing_version": 2, "formats": ["points"], "history_ages_s": [2.0]},
    {"preprocessing_version": 2, "formats": ["points"], "history_ages_s": [0.0, 4.0, 2.0]},
    {"preprocessing_version": 2, "formats": ["points", "bev"]},
    {"preprocessing_version": 2, "formats": ["points"], "adaptive_radius_min_m": 0.6,
     "adaptive_radius_max_m": 0.5},
    {"preprocessing_version": 3},
])
def test_validation_rejects_unsupported_settings(bad):
    with pytest.raises(ConfigError):
        validate_generator(_gcfg(**bad))


def test_yaml_cannot_enable_an_unknown_field(tmp_path):
    p = tmp_path / "arm.yaml"
    p.write_text("generator:\n  history_seconds: 30\n")
    with pytest.raises(ConfigError):
        load_config(str(p))


def test_old_checkpoints_resolve_to_the_legacy_contract():
    old = {k: v for k, v in DEFAULTS["generator"].items()
           if k not in ("preprocessing_version", "neighborhood_mode", "history_ages_s")}
    assert is_legacy(old)
    c = input_contract(old)
    assert c["preprocessing_version"] == 1 and c["history"] == "current sweep only"
    r = RadiusPolicy.from_generator(old)
    assert r.mode == "fixed" and r.radius_m == 0.5


# --------------------------------------------------------------------------- #
# Adaptive radius
# --------------------------------------------------------------------------- #
def _cluster(n, spread, rng, center=(0.0, 0.0, 0.0)):
    return (np.asarray(center) + rng.normal(0.0, spread, (n, 3))).astype(np.float32)


def test_adaptive_radius_clips_to_its_bounds_and_reports_shortfalls():
    rng = np.random.default_rng(0)
    pol = RadiusPolicy.from_generator(_v2(neighborhood_mode="adaptive",
                                          adaptive_radius_min_m=0.2,
                                          adaptive_radius_max_m=0.5, adaptive_k=64))
    dense = _cluster(2000, 0.02, rng)                          # K-th point ~3 cm away
    sparse = _cluster(40, 0.05, rng, center=(5.0, 0.0, 0.0))   # never K points
    xyz = np.concatenate([dense, sparse])
    sup = single_sweep_support(xyz, np.zeros(len(xyz), np.float32))
    out = build_neighborhoods(np.array([[0, 0, 0], [5, 0, 0]], float), sup, pol,
                              np.random.default_rng(1))
    assert np.allclose(out["radius"], [0.2, 0.5])
    assert out["ball_count"][1] == 40 and out["ball_count"][1] < pol.k   # shortfall kept


def test_adaptive_radius_equals_kth_distance_and_includes_the_tie():
    # Points on a line at 1 cm spacing: the K-th nearest is exactly at a known
    # distance, which must become the radius and must itself be in the ball.
    k = 30
    xs = np.arange(1, 101) * 0.01
    xyz = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1).astype(np.float32)
    pol = RadiusPolicy.from_generator(_v2(neighborhood_mode="adaptive",
                                          adaptive_radius_min_m=0.05,
                                          adaptive_radius_max_m=0.9, adaptive_k=k))
    sup = single_sweep_support(xyz, np.zeros(len(xyz), np.float32))
    out = build_neighborhoods(np.zeros((1, 3)), sup, pol, np.random.default_rng(0))
    assert out["radius"][0] == pytest.approx(0.30, abs=1e-6)
    assert out["ball_count"][0] == k


def test_large_balls_sample_the_whole_ball_not_the_nearest_points():
    rng = np.random.default_rng(3)
    # 600 points uniformly in a 0.5 m disc: nearest-256 would stop near 0.33 m.
    r = 0.5 * np.sqrt(rng.random(600))
    a = rng.random(600) * 2 * np.pi
    xyz = np.stack([r * np.cos(a), r * np.sin(a), np.zeros(600)], 1).astype(np.float32)
    pol = RadiusPolicy.from_generator(_v2())
    sup = single_sweep_support(xyz, np.zeros(600, np.float32))
    out = build_neighborhoods(np.zeros((1, 3)), sup, pol, np.random.default_rng(0))
    d = np.linalg.norm(out["neighborhoods"][0, :, :2], axis=1)
    assert out["true_counts"][0] == 600
    assert d.max() > 0.45                              # reaches the rim
    assert len(np.unique(out["neighborhoods"][0, :, :2], axis=0)) == 256  # no repeats


def test_small_balls_keep_every_point_then_pad_and_small_ones_are_unscorable():
    rng = np.random.default_rng(4)
    xyz = np.concatenate([_cluster(50, 0.03, rng),
                          _cluster(10, 0.03, rng, center=(3.0, 0.0, 0.0))])
    pol = RadiusPolicy.from_generator(_v2())
    sup = single_sweep_support(xyz, np.zeros(len(xyz), np.float32))
    out = build_neighborhoods(np.array([[0, 0, 0], [3, 0, 0]], float), sup, pol,
                              np.random.default_rng(0))
    assert out["scorable"].tolist() == [True, False]
    rows = out["neighborhoods"][0]
    assert out["true_counts"][0] == 50
    real = {tuple(r) for r in rows[:50, :3].round(6)}
    assert len(real) == 50
    assert {tuple(r) for r in rows[50:, :3].round(6)} <= real   # padding repeats reals
    # Physical metres, not normalised to the ball radius.
    assert np.abs(rows[:, :2]).max() < 0.2


def test_z_reference_and_query_height_come_from_the_full_ball():
    xyz = np.array([[0, 0, 0.0]] * 300 + [[0.3, 0, -0.2]], np.float32)
    xyz[:300, :2] += np.random.default_rng(0).normal(0, 0.05, (300, 2)).astype(np.float32)
    pol = RadiusPolicy.from_generator(_v2())
    sup = single_sweep_support(xyz, np.zeros(len(xyz), np.float32))
    out = build_neighborhoods(np.array([[0, 0, 0.0]]), sup, pol, np.random.default_rng(0))
    # The single low point may be dropped by sampling, but it still sets z = 0.
    assert out["neighborhoods"][0, :, QUERY_HEIGHT_CHANNEL][0] == pytest.approx(0.2, abs=1e-6)
    assert out["neighborhoods"][0, :, 2].min() >= 0.2 - 1e-6 or \
        out["neighborhoods"][0, :, 2].min() == pytest.approx(0.0)


def test_sampling_is_deterministic_for_a_seed():
    rng = np.random.default_rng(5)
    xyz = _cluster(1000, 0.1, rng)
    sup = single_sweep_support(xyz, np.zeros(len(xyz), np.float32))
    cand = candidate_centers(xyz, _v2())[:20]
    pol = RadiusPolicy.from_generator(_v2(neighborhood_mode="adaptive"))
    a = build_neighborhoods(cand, sup, pol, sample_rng(42, 7))
    b = build_neighborhoods(cand, sup, pol, sample_rng(42, 7))
    assert np.array_equal(a["neighborhoods"], b["neighborhoods"])


# --------------------------------------------------------------------------- #
# Version-1 parity
# --------------------------------------------------------------------------- #
def test_fixed_single_sweep_matches_the_version_1_builder_exactly():
    rng = np.random.default_rng(6)
    floor = np.column_stack([rng.uniform(-2, 2, 4000), rng.uniform(-2, 2, 4000),
                             rng.normal(0, 0.01, 4000)]).astype(np.float32)
    inten = rng.random(len(floor)).astype(np.float32)
    from rocklabel.labels import Rock
    rocks = [Rock(id=1, center=np.array([0.5, 0.5, 0.0]), radius=0.2)]
    g = _gcfg(negative_keep_prob=0.3)

    old = build_neighborhood_samples(floor, inten, rocks, g, np.random.default_rng([42, 3]))

    shared = np.random.default_rng([42, 3])
    cand = candidate_centers(floor, g)
    lab = label_rocks(cand, rocks, g["boundary_shell_m"])
    keep = (lab == LABEL_ROCK) | ((lab == LABEL_CLEAR) & (shared.random(len(cand))
                                                          < g["negative_keep_prob"]))
    new = build_neighborhoods(cand[keep], single_sweep_support(floor, inten),
                              RadiusPolicy.from_generator(g), shared)
    assert np.array_equal(old["neighborhoods"], new["neighborhoods"])
    assert np.array_equal(old["true_counts"], new["true_counts"])
    assert np.array_equal(old["centers_odom"], new["centers_odom"])
    assert np.array_equal(old["labels"], lab[keep][new["scorable"]])


# --------------------------------------------------------------------------- #
# History selection
# --------------------------------------------------------------------------- #
def test_selection_is_causal_unique_and_leaves_gaps_empty():
    sweeps = [_sweep(i, 0.05 * i) for i in range(200)]    # 0 .. 9.95 s
    slots = select_slots(sweeps, (0.0, 2.0, 4.0, 30.0), 0.10)
    assert [s.status for s in slots] == ["ok", "ok", "ok", "before_start"]
    assert slots[0].sweep is sweeps[-1]
    for s, want in zip(slots[1:3], (2.0, 4.0)):
        assert s.sweep.time_s <= sweeps[-1].time_s - want + 1e-9      # at or before
        assert want <= s.age_s <= want + 0.10
    # A 1 s hole: the 2 s slot's nearest older sweep is beyond tolerance.
    gap = [s for s in sweeps if not (7.0 < s.time_s < 8.2)]
    slots = select_slots(gap, (0.0, 2.0), 0.10)
    assert slots[1].status == "gap" and slots[1].sweep is None


def test_one_sweep_never_fills_two_slots():
    sweeps = [_sweep(0, 0.0), _sweep(1, 5.0)]
    slots = select_slots(sweeps, (0.0, 4.9, 4.95), 1.0)
    assert [s.status for s in slots] == ["ok", "ok", "duplicate"]


def test_buffer_resets_on_backwards_time_and_pose_jumps_and_evicts_old_sweeps():
    h = SweepHistory(retention_s=2.0)
    for i in range(100):
        h.push(_sweep(i, 0.05 * i))
    assert all(s.time_s >= 4.95 - 2.0 for s in h._sweeps)
    assert h.push(_sweep(100, 1.0)) and len(h) == 1              # replay seek
    assert h.push(_sweep(101, 1.05, origin=(5.0, 0, 0))) and len(h) == 1   # relocalised
    assert h.resets == 2


def test_live_assembly_matches_the_offline_windowed_stream():
    from rocklabel.recording.pipeline import OdomScan, WindowedScanStream

    rng = np.random.default_rng(8)
    t = np.cumsum(rng.uniform(0.003, 0.006, 600))
    batches = [(tt, rng.normal(0, 1, (20, 3)).astype(np.float32),
                rng.random(20).astype(np.float32)) for tt in t]

    class Stream:
        counters = info = None

        def __iter__(self):
            for i, (tt, x, v) in enumerate(batches):
                yield OdomScan(i, tt, x, v, np.eye(4), np.eye(4))

    offline = list(WindowedScanStream(Stream(), 0.05))
    asm, live = SweepAssembler(0.05), []
    for tt, x, v in batches:
        done = asm.add(tt, x, v, np.zeros(3))
        if done is not None:
            live.append(done)
    assert len(live) == len(offline) - 1          # the last window is still open
    for a, b in zip(live, offline):
        assert a.time_s == b.time_s
        assert np.array_equal(a.xyz, b.xyz_odom)


def test_support_crops_every_sweep_with_the_current_box_and_records_ages():
    s0 = _sweep(0, 0.0, np.array([[0, 0, 0], [9, 9, 9]], np.float32))
    s1 = _sweep(1, 2.0, np.array([[0.1, 0, 0]], np.float32))
    slots = select_slots([s0, s1], (0.0, 2.0), 0.1)
    sup = assemble_support(slots, lambda xyz: np.abs(xyz).max(axis=1) < 1.0)
    assert len(sup) == 2 and sup.slot.tolist() == [0, 1]
    assert sup.age.tolist() == [0.0, 2.0]
    assert sup.slot_points == [1, 1]


def test_history_adds_support_without_changing_candidates():
    rng = np.random.default_rng(9)
    now = _cluster(300, 0.1, rng)
    old = _cluster(300, 0.1, rng)
    sweeps = [_sweep(0, 0.0, old), _sweep(1, 2.0, now)]
    g = _v2(history_ages_s=[0.0, 2.0])
    h = HistoryPolicy.from_generator(g)
    sup = assemble_support(select_slots(sweeps, h.ages_s, h.tolerance_s),
                           lambda xyz: np.ones(len(xyz), bool))
    cand = candidate_centers(sup.xyz[sup.current], g)
    single = candidate_centers(now, g)
    assert np.array_equal(cand, single)
    out = build_neighborhoods(cand, sup, RadiusPolicy.from_generator(g), sample_rng(1, 1))
    assert out["ball_sweeps"].max() == 2
    assert set(np.unique(out["point_age"])) <= {0.0, 2.0}


# --------------------------------------------------------------------------- #
# Generation, cache, training round trip on the synthetic recording
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def v1_and_v2(synthetic_recording, tmp_path_factory):
    from rocklabel.dataset.generate import run_generate

    mcap, labels = synthetic_recording
    root = tmp_path_factory.mktemp("nh")
    cfg1 = load_config(None)
    cfg1["generator"]["frame_stride"] = 2
    run_generate(mcap, labels, str(root / "v1"), cfg1)
    cfg2 = copy.deepcopy(cfg1)
    cfg2["generator"].update(preprocessing_version=2, formats=["points"])
    run_generate(mcap, labels, str(root / "v2"), cfg2)
    cfg3 = copy.deepcopy(cfg2)
    cfg3["generator"].update(neighborhood_mode="adaptive", history_ages_s=[0.0, 0.5, 1.0])
    run_generate(mcap, labels, str(root / "v3"), cfg3)
    return mcap, labels, root, cfg3


def _frames(d):
    return sorted(glob.glob(os.path.join(d, "points", "synthetic", "*.npz")))


def test_version_2_fixed_single_sweep_keeps_version_1_candidates_and_balls(v1_and_v2):
    _, _, root, _ = v1_and_v2
    a_files, b_files = _frames(str(root / "v1")), _frames(str(root / "v2"))
    assert [os.path.basename(f) for f in a_files] == [os.path.basename(f) for f in b_files]
    for fa, fb in zip(a_files, b_files):
        a, b = np.load(fa), np.load(fb)
        assert np.array_equal(a["labels"], b["labels"])
        assert np.array_equal(a["centers_odom"], b["centers_odom"])
        assert np.array_equal(a["true_counts"], b["true_counts"])
        # Only the sampling stream differs: small balls hold identical real rows.
        for i in np.flatnonzero(a["true_counts"] <= 256)[:20]:
            n = int(a["true_counts"][i])
            ra = np.sort(a["neighborhoods"][i, :n].view("f4,f4,f4,f4,f4"), axis=0)
            rb = np.sort(b["neighborhoods"][i, :n].view("f4,f4,f4,f4,f4"), axis=0)
            assert np.array_equal(ra, rb)


def test_history_arm_keeps_the_candidate_set_and_records_its_sweeps(v1_and_v2):
    _, _, root, _ = v1_and_v2
    man = json.load(open(root / "v3" / "manifest.json"))
    run = man["runs"]["synthetic"]
    base = json.load(open(root / "v2" / "manifest.json"))["runs"]["synthetic"]
    assert run["candidates_selected"] == base["candidates_selected"]
    slots = run["diagnostics"]["slots"]
    assert [s["target_age_s"] for s in slots] == [0.0, 0.5, 1.0]
    assert slots[0]["fill_rate"] == 1.0 and 0 < slots[2]["fill_rate"] < 1.0   # startup
    z = np.load(_frames(str(root / "v3"))[-1])
    assert (z["history_sweep_ids"] >= 0).all()
    assert z["history_sweep_ids"][0] > z["history_sweep_ids"][1] > z["history_sweep_ids"][2]
    assert z["point_age"].dtype == np.float16 and z["radius"].min() >= 0.2 - 1e-6


def test_sweep_cache_generation_is_identical_and_refuses_other_inputs(v1_and_v2, tmp_path):
    from rocklabel.dataset.generate import run_generate
    from rocklabel.dataset.sweep_cache import build_cache
    from rocklabel.geometry.leveling import pin_level_to_labels
    from rocklabel.labels import load_labels

    mcap, labels, root, cfg3 = v1_and_v2
    stream_cfg = pin_level_to_labels(cfg3, load_labels(labels).level)
    build_cache(mcap, labels, stream_cfg, str(tmp_path / "sweeps"))
    run_generate(mcap, labels, str(tmp_path / "v3c"), cfg3,
                 sweep_cache=str(tmp_path / "sweeps"))
    for fa, fb in zip(_frames(str(root / "v3")), _frames(str(tmp_path / "v3c"))):
        a, b = np.load(fa), np.load(fb)
        assert np.array_equal(a["neighborhoods"], b["neighborhoods"])
    other = copy.deepcopy(cfg3)
    other["generator"]["frame_window_s"] = 0.2
    with pytest.raises(SystemExit):
        run_generate(mcap, labels, str(tmp_path / "bad"), other,
                     sweep_cache=str(tmp_path / "sweeps"))


def test_cache_carries_diagnostics_and_refuses_a_different_population(v1_and_v2, tmp_path):
    from rocklabel.train.data import DataError, RunData, build_cache

    _, _, root, _ = v1_and_v2
    meta = build_cache([str(root / "v3")], str(tmp_path / "c"))
    assert meta["input_contract"]["neighborhood_mode"] == "adaptive"
    run = RunData(str(tmp_path / "c"), "synthetic")
    assert len(run.extra("radius")) == len(run) and run.extra("point_age").shape[1] == 256
    assert os.path.exists(tmp_path / "c" / "synthetic" / "history.json")
    with pytest.raises(DataError):
        build_cache([str(root / "v2")], str(tmp_path / "c"))


def test_split_checks_refuse_shared_recordings():
    from rocklabel.train.data import DataError, check_split_runs

    check_split_runs(["a", "b"], ["c"])
    with pytest.raises(DataError):
        check_split_runs(["a", "b"], ["b"])


def test_whole_run_validation_trains_and_rejects_a_changed_cache(v1_and_v2, tmp_path):
    torch = pytest.importorskip("torch")
    from rocklabel.train.data import build_cache
    from rocklabel.train.engine import default_config, train_fold

    _, _, root, _ = v1_and_v2
    import shutil
    ds_a, ds_b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(root / "v3", ds_a)
    # A second "recording": same frames under another run id.
    shutil.copytree(root / "v3", ds_b)
    os.rename(ds_b / "points" / "synthetic", ds_b / "points" / "synthetic2")
    m = json.load(open(ds_b / "manifest.json"))
    m["runs"] = {"synthetic2": m["runs"]["synthetic"]}
    json.dump(m, open(ds_b / "manifest.json", "w"))
    cache = str(tmp_path / "cache")
    build_cache([str(ds_a), str(ds_b)], cache)
    cfg = default_config(model="pointnet", features=["dx", "dy", "dz"], cache_dir=cache,
                         train_runs=["synthetic"], val_runs=["synthetic2"], test_run="",
                         epochs=1, batch=16, device="cpu")
    summary = train_fold(cfg, str(tmp_path / "run"))
    assert summary["val_runs"] == ["synthetic2"]
    ck = torch.load(str(tmp_path / "run" / "best.pt"), weights_only=False)
    assert ck["generator"]["neighborhood_mode"] == "adaptive"
    assert ck["input_contract"]["history_ages_s"] == [0.0, 0.5, 1.0]
    with open(tmp_path / "run" / "cache.json", "w") as f:
        json.dump({"config_hash": "something-else"}, f)
    with pytest.raises(SystemExit):
        train_fold(cfg, str(tmp_path / "run"))
    bad = dict(cfg, val_runs=["synthetic"])
    with pytest.raises(SystemExit):
        train_fold(bad, str(tmp_path / "run2"))


# --------------------------------------------------------------------------- #
# Inference dispatch
# --------------------------------------------------------------------------- #
class _Recorder:
    """A stand-in model that records exactly what it was given."""

    def __init__(self):
        self.seen = []

    def __call__(self, pts, cnt):
        import torch
        self.seen.append((pts.clone(), cnt.clone()))
        return torch.zeros(len(pts))


def test_inference_uses_the_generation_builder_on_the_same_support():
    pytest.importorskip("torch")
    from rocklabel.train.policy_scoring import score_classifier

    rng = np.random.default_rng(10)
    now, old = _cluster(800, 0.2, rng), _cluster(800, 0.2, rng)
    g = _v2(neighborhood_mode="adaptive", history_ages_s=[0.0, 2.0])
    h = HistoryPolicy.from_generator(g)
    sup = assemble_support(select_slots([_sweep(0, 0.0, old), _sweep(9, 2.0, now)],
                                        h.ages_s, h.tolerance_s),
                           lambda xyz: np.ones(len(xyz), bool))
    model = _Recorder()
    centers, probs, diag = score_classifier(now, np.zeros(len(now), np.float32), g,
                                            model, "cpu", index=9, support=sup)
    want = build_neighborhoods(candidate_centers(now, g), sup,
                               RadiusPolicy.from_generator(g), sample_rng(g["seed"], 9),
                               diagnostics=False)
    got = np.concatenate([p.numpy() for p, _ in model.seen])
    assert np.array_equal(got, want["neighborhoods"])
    assert diag["candidates"] == diag["scored"] + diag["unscorable"]
    assert diag["slot_ages"] == [0.0, 2.0]


def test_legacy_checkpoints_keep_the_historical_inference_path():
    pytest.importorskip("torch")
    from rocklabel.dataset.neighborhoods import build_inference_samples
    from rocklabel.train.policy_scoring import score_classifier

    rng = np.random.default_rng(11)
    xyz = _cluster(900, 0.3, rng)
    inten = rng.random(900).astype(np.float32)
    g = {k: v for k, v in DEFAULTS["generator"].items() if k != "preprocessing_version"}
    model = _Recorder()
    score_classifier(xyz, inten, g, model, "cpu", index=4)
    want = build_inference_samples(xyz, inten, g, np.random.default_rng([g["seed"], 4]))
    got = np.concatenate([p.numpy() for p, _ in model.seen])
    assert np.array_equal(got, want["neighborhoods"])


def test_audit_warms_history_before_its_window_and_dispatches_per_checkpoint(
        synthetic_recording):
    from rocklabel.labels import load_labels
    from rocklabel.train.visual_audit import collect_intervals

    mcap, labels_path = synthetic_recording
    labels = load_labels(labels_path)
    cfg = load_config(None)
    cfg["level"]["mode"] = "off"
    labels.level = {"mode": "off", "floor_z": 0.0}
    h = HistoryPolicy((0.0, 2.0), 0.1)
    try:
        intervals, _ = collect_intervals(
            mcap, labels, labels_path, cfg, [DEFAULTS["generator"]], (-0.5, 1.0), 8.0,
            stride=2, window_s=0.0, accum_seconds=2.0, candidates_per_rock=5,
            min_visible_points=1, start_s=3.0, end_s=6.0, cell_m=0.1, histories=[h])
    except ValueError as exc:            # no measured floor in this environment
        pytest.skip(str(exc))
    first = intervals[0].frames[0]
    sup = first.support[h.key()]
    # 3 s into the recording, before the audit window's own first frame, the
    # 2 s slot is already filled: history came from sweeps the window skipped.
    assert sup.slots[1].status == "ok" and sup.slot_points[0] > 0


def test_a_history_checkpoint_refuses_to_score_without_its_support():
    pytest.importorskip("torch")
    from rocklabel.labels import LabelSet
    from rocklabel.train.visual_audit import AuditFrame, AuditInterval, _score_interval

    g = _v2(history_ages_s=[0.0, 2.0])
    frame = AuditFrame(0, 0.0, np.zeros(3), np.zeros((50, 3)), np.zeros(50, np.float32))
    loaded = {"generator": g, "model": _Recorder(), "task": "classify", "path": "x",
              "name": "pointnet", "threshold": 0.5}
    with pytest.raises(RuntimeError):
        _score_interval(AuditInterval(0, 0.0, 0.0, [frame]), loaded, "cpu", 64, 0.1,
                        LabelSet(rocks=[]))


# --------------------------------------------------------------------------- #
# Threshold frontier: attribution by physical rock, not by class label
# --------------------------------------------------------------------------- #
def _frontier_for(rock_ids):
    from rocklabel.labels import LabelSet, Rock
    from rocklabel.train.map_eval import threshold_frontier
    from rocklabel.train.visual_audit import CellMap, cell_ids

    rocks = [Rock(id=rid, center=np.array([2.0 * k, 0.0, 0.1]), radius=0.3)
             for k, rid in enumerate(rock_ids)]
    labels = LabelSet(rocks=rocks)
    # A perfect map: one cell on each rock at probability 0.9, one clear cell.
    pos = np.array([[2.0 * k, 0.0, 0.1] for k in range(len(rocks))] + [[10.0, 5.0, 0.0]])
    probs = np.array([0.9] * len(rocks) + [0.2])
    ids = cell_ids(pos[:, :2], 0.10)

    class M:
        name = "control"

        def cells(self):
            return CellMap(pos, probs, ids)

    visible = {r.id: {tuple(ids[k])} for k, r in enumerate(rocks)}
    return threshold_frontier(M(), labels, 0.05, visible)


def test_frontier_coverage_is_invariant_to_rock_ids():
    for ids in ([1, 2, 3], [3, 7, 12], [12, 1, 5]):
        rows = _frontier_for(ids)
        at_09 = next(r for r in rows if r["threshold"] == 0.9)
        assert at_09["macro_coverage"] == 1.0, ids
        assert at_09["worst_rock_coverage"] == 1.0, ids
        assert at_09["false_cells_3d"] == 0
        last = rows[-1]
        assert last["false_cells_3d"] == 1


# --------------------------------------------------------------------------- #
# The live scorer's own entry point
# --------------------------------------------------------------------------- #
class _HistoryEngine:
    """A live engine stand-in: fixed sweeps, a recent window, a pose."""

    def __init__(self, sweeps, recent):
        self.sweeps, self.recent = sweeps, recent

    def recent_snapshot(self, window_s=0.0, with_origins=False, with_stamps=False):
        out = (self.recent, np.zeros(len(self.recent), np.float32))
        if with_origins:
            out += (np.zeros((len(self.recent), 3)),)
        if with_stamps:
            out += (np.zeros(len(self.recent)),)
        return out

    def current_pose(self):
        return np.array([0.0, 0.0, 0.5]), np.array([1.0, 0, 0, 0])

    def history_slots(self, policy):
        return select_slots(self.sweeps, policy.ages_s, policy.tolerance_s)


def _live_scorer(tmp_path, engine, name, **gen):
    torch = pytest.importorskip("torch")
    from rocklabel.live.scoring import LiveScorer, ScoreSettings
    from rocklabel.train.models import build_model

    g = _v2(**gen)
    config = {"model": "pointnet", "tnet": False, "dropout": None,
              "features": ["dx", "dy", "dz"]}
    ck = tmp_path / f"{name}.pt"
    torch.save({"config": config, "generator": g, "threshold": 0.0,
                "model": build_model("pointnet", features=config["features"]).state_dict()},
               ck)
    scorer = LiveScorer(str(ck), engine, device="cpu", settings=ScoreSettings(
        z_min=-5.0, z_max=5.0, range_max=0.0, max_centers=40))
    return scorer, g


def test_live_history_scores_a_sparse_current_sweep_on_combined_support(tmp_path):
    from rocklabel.train.policy_scoring import score_classifier

    rng = np.random.default_rng(3)
    now = _cluster(5, 0.05, rng).astype(np.float32)
    old = _cluster(60, 0.08, rng).astype(np.float32)
    engine = _HistoryEngine([_sweep(0, 0.0, old), _sweep(9, 2.0, now)], now)
    scorer, g = _live_scorer(tmp_path, engine, "sparse", history_ages_s=[0.0, 2.0],
                             min_neighbors=20)
    assert len(now) < g["min_neighbors"]
    scorer._score_once()
    h = HistoryPolicy.from_generator(g)
    sup = assemble_support(engine.history_slots(h), lambda xyz: np.ones(len(xyz), bool))
    _, want, _ = score_classifier(now, np.zeros(len(now), np.float32), g, _Recorder(),
                                  "cpu", 0, support=sup)
    assert len(want) > 0
    assert len(scorer._map) > 0
    assert scorer._last_history["unscorable"] + len(want) == scorer._last_history["candidates"]


def test_live_candidate_draws_do_not_depend_on_the_radius_or_history(tmp_path):
    rng = np.random.default_rng(4)
    # A uniformly dense slab: even a corner candidate's 0.3 m ball is full.
    now = (rng.random((20_000, 3)) * [2.0, 2.0, 0.3]).astype(np.float32)
    old = (rng.random((20_000, 3)) * [2.0, 2.0, 0.3]).astype(np.float32)
    engine = _HistoryEngine([_sweep(0, 0.0, old), _sweep(9, 2.0, now)], now)
    small, _ = _live_scorer(tmp_path, engine, "small", neighborhood_radius_m=0.3)
    wide, _ = _live_scorer(tmp_path, engine, "wide", neighborhood_radius_m=0.75,
                           history_ages_s=[0.0, 2.0])
    for _ in range(3):
        small._map.clear()
        wide._map.clear()
        small._score_once()
        wide._score_once()
        # Dense cloud, so every capped candidate is scorable in both: the
        # published voxels are the candidate sets themselves.
        assert small._last_centers_capped and wide._last_centers_capped
        assert set(small._map) == set(wide._map)


def test_the_recording_viewer_scores_under_the_checkpoint_contract(v1_and_v2, monkeypatch):
    pytest.importorskip("torch")
    import rocklabel.train.policy_scoring as ps
    from rocklabel.train.mcapview import _score_recording

    mcap, _, _, cfg3 = v1_and_v2
    g = cfg3["generator"]
    seen = []
    real = ps.score_classifier

    def spy(xyz, inten, gcfg, model, device, index, batch=512, support=None, **kw):
        seen.append((gcfg, support))
        return real(xyz, inten, gcfg, model, device, index, batch, support=support, **kw)

    monkeypatch.setattr(ps, "score_classifier", spy)
    recs = _score_recording(mcap, cfg3, g, _RecorderModule(), "cpu", stride=2,
                            window_s=None)
    assert recs and seen
    assert all(s[0]["neighborhood_mode"] == "adaptive" for s in seen)
    # Past the first second the older slots are filled from sweeps the stride
    # skipped - the viewer keeps its own causal history.
    assert any(sum(n > 0 for n in sup.slot_points) == 3 for _, sup in seen)


class _RecorderModule(_Recorder):
    def eval(self):
        return self

    def to(self, device):
        return self


def test_history_stages_are_exact_and_straddling_intervals_are_set_aside():
    from rocklabel.train.visual_audit import history_stage, stage_rows

    assert history_stage(0.0, 2.0) == "startup 0-4 s"
    assert history_stage(2.0, 4.0) == "startup 0-4 s"
    assert history_stage(0.0, 5.0).startswith("straddles")
    assert history_stage(28.0, 30.0) == "partial 4-30 s"
    assert history_stage(40.0, 45.0) == "mature >30 s"
    obs = [{"rock_id": 1, "history_stage": history_stage(a, b),
            "m": {"coverage": c}} for a, b, c in ((0, 2, 0.1), (0, 5, 0.9), (40, 45, 0.5))]
    rows = {r["stage"]: r for r in stage_rows(obs, ["m"], 0.3)}
    assert rows["startup 0-4 s"]["m_macro_median_coverage"] == 0.1
    assert rows["straddling (excluded)"]["observations"] == 1


# --------------------------------------------------------------------------- #
# Point-age feature follow-up
# --------------------------------------------------------------------------- #
def test_the_age_model_reads_the_age_channel_and_refuses_without_it():
    torch = pytest.importorskip("torch")
    from rocklabel.dataset.neighborhoods import AGE_CHANNEL
    from rocklabel.train.models import build_model

    torch.manual_seed(0)
    m = build_model("pointnet_age", features=["dx", "dy", "dz"]).eval()
    pts = torch.randn(4, 64, AGE_CHANNEL + 1)
    cnt = torch.full((4,), 64)
    a = m(pts, cnt)
    older = pts.clone()
    older[..., AGE_CHANNEL] += 5.0
    assert not torch.allclose(a, m(older, cnt))
    with pytest.raises(ValueError):
        m(pts[..., :AGE_CHANNEL], cnt)


def test_age_is_appended_identically_in_training_and_scoring(v1_and_v2):
    pytest.importorskip("torch")
    from rocklabel.train.data import DataError, RunData, build_cache

    _, _, root, _ = v1_and_v2
    cache = str(root / "cache-age")
    if not os.path.exists(cache):
        build_cache([str(root / "v3")], cache)
    run = RunData(cache, "synthetic")
    age = run.extra("point_age")
    run.append_point_age()
    assert run.points.shape[-1] == 6
    assert np.array_equal(run.points[..., 5], age.astype(np.float32))
    legacy = str(root / "cache-v1")
    if not os.path.exists(legacy):
        build_cache([str(root / "v1")], legacy)
    with pytest.raises(DataError):
        RunData(legacy, "synthetic").append_point_age()


def test_the_scorer_appends_point_age_for_the_age_model():
    torch = pytest.importorskip("torch")
    from rocklabel.train.policy_scoring import score_classifier

    rng = np.random.default_rng(12)
    now, old = _cluster(600, 0.2, rng), _cluster(600, 0.2, rng)
    g = _v2(history_ages_s=[0.0, 2.0])
    h = HistoryPolicy.from_generator(g)
    sup = assemble_support(select_slots([_sweep(0, 0.0, old), _sweep(9, 2.0, now)],
                                        h.ages_s, h.tolerance_s),
                           lambda xyz: np.ones(len(xyz), bool))
    model = _Recorder()
    model.reads_point_age = True
    score_classifier(now, np.zeros(len(now), np.float32), g, model, "cpu", 9, support=sup)
    got = np.concatenate([p.numpy() for p, _ in model.seen])
    assert got.shape[-1] == 6
    assert set(np.unique(got[..., 5]).round(3)) <= {0.0, 2.0}
    with pytest.raises(ValueError):
        score_classifier(now, np.zeros(len(now), np.float32), DEFAULTS["generator"], model,
                         "cpu", 9)
