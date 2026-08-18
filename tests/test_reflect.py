"""Tests for the reflectivity probe.

The probe's whole value is that its numbers are trustworthy enough to act on —
"the channel is empty, stop spending GPU hours on it" is a conclusion, not a
figure. So the measurements are checked against hand-built neighborhoods whose
answers are known by construction, and the padding rule is checked explicitly:
padded rows repeat real points, so counting them would quietly weight dense
samples' own points twice and skew every average in the report.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from rocklabel.train import reflect as R

N = 32  # points per stored neighborhood in these fixtures


def _write_run(cache_dir: str, run_id: str, points: np.ndarray,
               labels: np.ndarray, counts: np.ndarray) -> None:
    d = os.path.join(cache_dir, run_id)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "points.npy"), points.astype(np.float32))
    np.save(os.path.join(d, "labels.npy"), labels.astype(np.int8))
    np.save(os.path.join(d, "counts.npy"), counts.astype(np.int16))
    np.save(os.path.join(d, "centers.npy"), np.zeros((len(labels), 3), np.float32))
    np.save(os.path.join(d, "frame.npy"), np.arange(len(labels), dtype=np.int32))


def _cache_meta(cache_dir: str, runs: list[str]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump({"config_hash": "x", "generator": {},
                   "runs": {r: {"n": 0} for r in runs}}, f)


# --------------------------------------------------------------------------- #
# the measurements themselves
# --------------------------------------------------------------------------- #
def test_measurements_match_hand_built_neighborhoods(tmp_path):
    cache = str(tmp_path / "cache")
    pts = np.zeros((2, N, 4), np.float32)
    # Sample 0: middle points sit at radius 0.05 and are bright (0.9); ring
    # points sit at radius 0.4 and are dark (0.1). 8 of each, 16 real points.
    pts[0, :8, 0] = 0.05
    pts[0, :8, 3] = 0.9
    pts[0, 8:16, 0] = 0.4
    pts[0, 8:16, 3] = 0.1
    # Sample 1: uniform brightness 0.5 everywhere, same layout.
    pts[1, :8, 0] = 0.05
    pts[1, 8:16, 0] = 0.4
    pts[1, :16, 3] = 0.5
    counts = np.array([16, 16])
    _write_run(cache, "r1", pts, np.array([1, 0]), counts)

    _, s = R.measure_run(cache, "r1")
    assert s["i_mean"][0] == pytest.approx(0.5, abs=1e-6)   # half 0.9, half 0.1
    assert s["i_core"][0] == pytest.approx(0.9, abs=1e-6)
    assert s["i_ring"][0] == pytest.approx(0.1, abs=1e-6)
    assert s["i_core_minus_ring"][0] == pytest.approx(0.8, abs=1e-6)
    assert s["i_max"][0] == pytest.approx(0.9) and s["i_min"][0] == pytest.approx(0.1)
    assert s["n_real"][0] == pytest.approx(16)
    # The flat sample: every brightness measurement collapses to zero contrast.
    assert s["i_std"][1] == pytest.approx(0.0, abs=1e-6)
    assert s["i_core_minus_ring"][1] == pytest.approx(0.0, abs=1e-6)
    assert s["i_p90_p10"][1] == pytest.approx(0.0, abs=1e-6)


def test_padding_is_excluded_from_every_average(tmp_path):
    """Padding repeats real points; counting it would double-weight them."""
    cache = str(tmp_path / "cache")
    pts = np.zeros((1, N, 4), np.float32)
    pts[0, :4, 3] = 1.0            # four real points, all bright
    pts[0, 4:, 3] = 0.0            # padded tail, deliberately dark
    _write_run(cache, "r1", pts, np.array([1]), np.array([4]))

    _, s = R.measure_run(cache, "r1")
    assert s["i_mean"][0] == pytest.approx(1.0)     # not 4/32
    assert s["i_min"][0] == pytest.approx(1.0)      # the dark tail is not a minimum
    assert s["i_std"][0] == pytest.approx(0.0, abs=1e-6)
    assert s["n_real"][0] == pytest.approx(4)


def test_counts_above_the_stored_width_are_clamped(tmp_path):
    """A neighborhood with more real points than stored rows is capped, not
    read past the end of the tensor."""
    cache = str(tmp_path / "cache")
    pts = np.zeros((1, N, 4), np.float32)
    pts[0, :, 3] = 0.5
    _write_run(cache, "r1", pts, np.array([0]), np.array([9999]))
    _, s = R.measure_run(cache, "r1")
    assert s["n_real"][0] == pytest.approx(N)
    assert s["i_mean"][0] == pytest.approx(0.5)


def test_brightness_tracking_height_is_a_correlation(tmp_path):
    cache = str(tmp_path / "cache")
    pts = np.zeros((2, N, 4), np.float32)
    z = np.linspace(0.0, 1.0, 16)
    pts[0, :16, 2] = z
    pts[0, :16, 3] = z             # brightness rises exactly with height
    pts[1, :16, 2] = z
    pts[1, :16, 3] = 1.0 - z       # and falls exactly with height
    _write_run(cache, "r1", pts, np.array([1, 0]), np.array([16, 16]))

    _, s = R.measure_run(cache, "r1")
    assert s["i_corr_z"][0] == pytest.approx(1.0, abs=1e-4)
    assert s["i_corr_z"][1] == pytest.approx(-1.0, abs=1e-4)
    assert s["i_high_minus_low"][0] > 0 and s["i_high_minus_low"][1] < 0


def test_a_neighborhood_with_no_ring_points_gives_nan_not_zero(tmp_path):
    """An empty ring is 'unknown', not 'zero contrast'. Zero would be counted
    as a real measurement and drag the run's average toward no-difference."""
    cache = str(tmp_path / "cache")
    pts = np.zeros((1, N, 4), np.float32)
    pts[0, :6, 0] = 0.05           # everything in the middle, nothing past 0.30 m
    pts[0, :6, 3] = 0.7
    _write_run(cache, "r1", pts, np.array([1]), np.array([6]))
    _, s = R.measure_run(cache, "r1")
    assert np.isnan(s["i_ring"][0]) and np.isnan(s["i_core_minus_ring"][0])
    assert s["i_core"][0] == pytest.approx(0.7)


def test_auc_ignores_undefined_measurements():
    y = np.array([1, 1, 0, 0], np.int8)
    v = np.array([np.nan, 1.0, 0.0, np.nan])
    assert R._auc(y, v) == pytest.approx(1.0)
    # Nothing left to score with -> nan rather than a made-up 0.5.
    assert np.isnan(R._auc(y, np.full(4, np.nan)))


# --------------------------------------------------------------------------- #
# the leave-one-run-out formula probe
# --------------------------------------------------------------------------- #
def test_logistic_probe_finds_a_real_signal_and_not_a_fake_one():
    from rocklabel.train.metrics import roc_auc

    rng = np.random.default_rng(0)
    per_run = {}
    for r in ("a", "b", "c"):
        y = np.r_[np.ones(200), np.zeros(200)].astype(np.int8)
        signal = np.r_[rng.normal(2.0, 1.0, 200), rng.normal(0.0, 1.0, 200)]
        noise = rng.normal(0.0, 1.0, 400)
        per_run[r] = (y, {"real": signal, "fake": noise})

    good = R.leave_one_run_out_probe(per_run, ["real"])
    bad = R.leave_one_run_out_probe(per_run, ["fake"])
    assert all(v["roc_auc"] > 0.85 for v in good.values())
    assert all(0.35 < v["roc_auc"] < 0.65 for v in bad.values())


def test_logistic_fit_separates_a_linearly_separable_set():
    rng = np.random.default_rng(1)
    X = np.r_[rng.normal(3.0, 0.5, (150, 2)), rng.normal(-3.0, 0.5, (150, 2))]
    y = np.r_[np.ones(150), np.zeros(150)]
    w = R.logistic_fit(X, y)
    p = R.logistic_predict(X, w, X)
    assert ((p > 0.5) == (y == 1)).mean() > 0.98


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_render_writes_a_report_and_scores_a_planted_signal(tmp_path):
    """Plant brightness that really does mark rocks and check the report says so."""
    pytest.importorskip("matplotlib")
    cache = str(tmp_path / "cache")
    rng = np.random.default_rng(3)
    runs = ["run1", "run2", "run3"]
    for r in runs:
        n = 120
        y = (np.arange(n) % 3 == 0).astype(np.int8)
        pts = np.zeros((n, N, 4), np.float32)
        pts[:, :20, 0] = rng.uniform(0.0, 0.45, (n, 20))
        pts[:, :20, 2] = rng.uniform(0.0, 0.2, (n, 20))
        # Rocks are plainly brighter here, well beyond the noise.
        base = np.where(y == 1, 0.80, 0.30)[:, None]
        pts[:, :20, 3] = base + rng.normal(0, 0.02, (n, 20))
        _write_run(cache, r, pts, y, np.full(n, 20))
    _cache_meta(cache, runs)

    out = str(tmp_path / "out")
    s = R.render_reflectivity(cache, out)

    assert s["auc_mean"]["i_mean"] > 0.95, "a planted brightness signal must show up"
    assert s["probe_mean"]["brightness only"]["roc_auc"] > 0.9
    for name in ("summary.json", "summary.md", "measurement_power.png",
                 "brightness_histogram.png", "brightness_drift.png",
                 "formula_probe.png"):
        assert os.path.exists(os.path.join(out, name)), name
    text = open(os.path.join(out, "summary.md")).read()
    assert "coin flip" in text and "brightness · average" in text


def test_every_measurement_is_labeled_in_plain_english():
    for key, label, kind in R.MEASUREMENTS:
        assert kind in ("intensity", "geometry")
        assert "·" in label and len(label) > 10
        # No bare variable names leaking into the report the user reads.
        assert key not in label
