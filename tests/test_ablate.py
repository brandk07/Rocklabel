"""Tests for the ablation sweep and the reflectivity probe.

Neither needs torch to be exercised end to end: the sweep's *reporting* half
reads run artifacts off disk, and the reflectivity probe reads the cache. Only
the arm-to-config translation touches the training engine, so that one test
skips without torch and the rest always run.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from rocklabel.train import metrics as M
from rocklabel.train.ablate import (SUITES, arms_of, collect,
                                    wilcoxon_signed_rank)


# --------------------------------------------------------------------------- #
# the paired significance test
# --------------------------------------------------------------------------- #
def test_wilcoxon_matches_the_exact_distribution():
    # Every pair positive: the only assignment at least this extreme is the
    # all-positive one and its mirror, so p = 2 / 2**n.
    for n in (5, 6, 8):
        stat, p = wilcoxon_signed_rank(np.arange(1, n + 1) / 100.0)
        assert stat == 0.0
        assert p == pytest.approx(2.0 / 2 ** n)


def test_wilcoxon_is_symmetric_and_ignores_direction():
    d = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
    assert wilcoxon_signed_rank(d)[1] == pytest.approx(wilcoxon_signed_rank(-d)[1])


def test_wilcoxon_reports_no_evidence_for_zero_deltas():
    assert wilcoxon_signed_rank(np.zeros(9)) == (0.0, 1.0)
    # Zeros are dropped, not counted as evidence: three real pairs remain.
    stat, p = wilcoxon_signed_rank(np.array([0.0, 0.0, 0.1, 0.2, 0.3]))
    assert p == pytest.approx(2.0 / 8)


def test_wilcoxon_p_never_exceeds_one():
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = rng.normal(size=rng.integers(2, 15))
        stat, p = wilcoxon_signed_rank(d)
        assert 0.0 <= p <= 1.0 and stat >= 0.0


# --------------------------------------------------------------------------- #
# collecting a suite off disk
# --------------------------------------------------------------------------- #
def _fake_fold(root: str, suite: str, arm: str, test_run: str, pr_auc: float,
               prevalence: float = 0.2) -> None:
    d = os.path.join(root, suite, arm, f"loro_{test_run}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "test_metrics.json"), "w") as f:
        json.dump({"test_run": test_run, "pr_auc": pr_auc, "roc_auc": 0.9,
                   "f1": 0.5, "precision": 0.5, "recall": 0.5,
                   "baseline_pr_auc": prevalence}, f)


def test_collect_pairs_arms_fold_by_fold(tmp_path):
    root = str(tmp_path)
    folds = ["r1", "r2", "r3"]
    # The variant wins every fold by exactly 0.02, but the folds themselves
    # differ by 0.30 - the case an unpaired comparison cannot see.
    for i, f in enumerate(folds):
        _fake_fold(root, "reflectivity", "pointnet-geom", f, 0.50 + 0.15 * i)
        _fake_fold(root, "reflectivity", "pointnet-refl", f, 0.52 + 0.15 * i)

    data = collect(root, "reflectivity")
    by_name = {c["baseline"] + "|" + c["variant"]: c for c in data["contrasts"]}
    c = by_name["pointnet-geom|pointnet-refl"]
    assert c["n_folds"] == 3
    assert c["pr_auc"]["mean_delta"] == pytest.approx(0.02)
    assert (c["pr_auc"]["wins"], c["pr_auc"]["losses"]) == (3, 0)
    assert c["pr_auc"]["std_delta"] == pytest.approx(0.0, abs=1e-9)


def test_collect_only_uses_folds_both_arms_finished(tmp_path):
    root = str(tmp_path)
    for f in ["r1", "r2", "r3"]:
        _fake_fold(root, "reflectivity", "pointnet-geom", f, 0.5)
    _fake_fold(root, "reflectivity", "pointnet-refl", "r1", 0.6)

    data = collect(root, "reflectivity")
    c = next(c for c in data["contrasts"]
             if (c["baseline"], c["variant"]) == ("pointnet-geom", "pointnet-refl"))
    assert c["n_folds"] == 1 and c["folds"] == ["r1"]
    arms = {a["arm"]: a for a in data["arms"]}
    assert arms["pointnet-geom"]["folds_done"] == 3
    assert arms["pointnet-refl"]["folds_done"] == 1


def test_collect_survives_a_suite_with_nothing_trained(tmp_path):
    data = collect(str(tmp_path), "reflectivity")
    assert data["folds"] == []
    assert all(a["folds_done"] == 0 for a in data["arms"])
    assert all(c["n_folds"] == 0 for c in data["contrasts"])


def test_every_contrast_names_arms_that_exist():
    for name, suite in SUITES.items():
        known = {a.name for a in suite["arms"]}
        for base, var, question in suite["contrasts"]:
            assert base in known, f"{name}: unknown baseline {base}"
            assert var in known, f"{name}: unknown variant {var}"
            assert len(question) > 20, f"{name}: {base} vs {var} needs a real question"


def test_arm_names_are_unique_and_every_arm_is_described():
    for name, suite in SUITES.items():
        names = [a.name for a in suite["arms"]]
        assert len(names) == len(set(names)), f"{name} has a duplicate arm name"
        for a in suite["arms"]:
            assert len(a.what) > 40, f"{a.name} needs a real description"
            assert a.label and "·" in a.label


def test_the_suite_contains_a_seed_repeat_of_the_headline_arms():
    """Without a same-setting repeat there is no scale for reading a delta."""
    arms = {a.name: a for a in arms_of("reflectivity")}
    for base in ("pointnet-geom", "pointnet-refl"):
        repeats = [a for a in arms.values()
                   if a.model == arms[base].model
                   and a.features == arms[base].features
                   and a.overrides == arms[base].overrides
                   and a.seed is not None]
        assert repeats, f"{base} has no differing-seed repeat"


# --------------------------------------------------------------------------- #
# arm -> training config
# --------------------------------------------------------------------------- #
def test_arm_overrides_beat_the_sweep_wide_settings():
    pytest.importorskip("torch")
    from rocklabel.train.engine import default_config

    arms = {a.name: a for a in arms_of("reflectivity")}
    # The sweep asks for jitter; the unjittered arm must still get none, or the
    # arm testing "reflectivity with its augmentation off" silently tests
    # nothing at all.
    extra = {"aug_intensity_gain": 0.25, "aug_intensity_shift": 0.10, "epochs": 3}
    cfg = arms["pointnet-refl-raw"].config(
        lambda **kw: default_config(cache_dir="c", **{**extra, **kw}), ["a"], "b")
    assert cfg["aug_intensity_gain"] == 0.0 and cfg["aug_intensity_shift"] == 0.0
    assert cfg["epochs"] == 3          # sweep-wide settings still get through
    assert cfg["features"] == ["dx", "dy", "dz", "intensity"]

    geom = arms["pointnet-geom"].config(
        lambda **kw: default_config(cache_dir="c", **{**extra, **kw}), ["a"], "b")
    assert geom["features"] == ["dx", "dy", "dz"]


def test_seed_repeat_arms_differ_only_in_the_seed():
    pytest.importorskip("torch")
    from rocklabel.train.engine import default_config

    arms = {a.name: a for a in arms_of("reflectivity")}
    make = lambda a: a.config(  # noqa: E731
        lambda **kw: default_config(cache_dir="c", **kw), ["a"], "b")
    base, repeat = make(arms["pointnet-refl"]), make(arms["pointnet-refl-s43"])
    assert base["seed"] != repeat["seed"]
    assert {k: v for k, v in base.items() if k != "seed"} == \
           {k: v for k, v in repeat.items() if k != "seed"}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_verdict_keeps_consistent_and_worth_acting_on_apart():
    """A tiny effect repeated on every fold is significant and still useless.
    Collapsing the two into one word is how a 0.002 'win' gets acted on."""
    from rocklabel.train.ablate_report import _verdict

    tiny = {"question": "does x help?",
            "pr_auc": {"mean_delta": 0.002, "p_value": 0.001}}
    assert "not worth acting on" in _verdict(tiny, floor=0.01)

    big = {"question": "does x help?",
           "pr_auc": {"mean_delta": 0.08, "p_value": 0.001}}
    v = _verdict(big, floor=0.01)
    assert "helps" in v and "not worth acting on" not in v and "8.0x" in v

    null = {"question": "does x help?",
            "pr_auc": {"mean_delta": 0.001, "p_value": 0.6}}
    assert "no measurable difference" in _verdict(null, floor=0.01)


def test_seed_repeat_rows_are_labeled_as_the_yardstick_not_a_result():
    from rocklabel.train.ablate_report import _verdict

    seed = {"question": "Noise floor: the same setting, two seeds.",
            "pr_auc": {"mean_delta": 0.03, "p_value": 0.001}}
    v = _verdict(seed, floor=0.02)
    assert "yardstick" in v
    assert "helps" not in v, "a same-setting repeat must never read as an effect"


def test_noise_floor_comes_only_from_the_seed_repeat_contrasts(tmp_path):
    from rocklabel.train.ablate_report import _noise_floor

    root = str(tmp_path)
    for i, f in enumerate(["r1", "r2", "r3"]):
        # A big difference between two genuinely different settings...
        _fake_fold(root, "reflectivity", "pointnet-geom", f, 0.50)
        _fake_fold(root, "reflectivity", "pointnet-refl", f, 0.90)
        # ...and a small one between two runs of the same setting.
        _fake_fold(root, "reflectivity", "pointnet-geom-s43", f, 0.51)
    data = collect(root, "reflectivity")
    assert _noise_floor(data) == pytest.approx(0.01)


def test_report_renders_and_survives_a_half_finished_sweep(tmp_path):
    pytest.importorskip("matplotlib")
    from rocklabel.train.ablate_report import render_ablation

    root, out = str(tmp_path / "ablate"), str(tmp_path / "out")
    for f in ["r1", "r2"]:
        _fake_fold(root, "reflectivity", "pointnet-geom", f, 0.7)
    data = render_ablation(root, "reflectivity", out)
    assert os.path.exists(os.path.join(out, "summary.md"))
    assert os.path.exists(os.path.join(out, "arm_ranking.png"))
    # No contrast has data on both sides yet, so no paired figure is drawn -
    # and nothing crashes trying.
    assert not os.path.exists(os.path.join(out, "paired_deltas.png"))
    assert all(c["n_folds"] == 0 for c in data["contrasts"])


def test_report_on_an_empty_sweep_writes_nothing_and_does_not_raise(tmp_path):
    from rocklabel.train.ablate_report import render_ablation

    out = str(tmp_path / "out")
    render_ablation(str(tmp_path / "ablate"), "reflectivity", out)
    assert not os.path.isdir(out)


def test_arm_run_directories_never_collide(tmp_path):
    """Two arms differing only in augmentation must not share a directory.

    This is the failure `compare` has by construction: it names a run after
    model + fold + channels, so the unjittered arm would land on the jittered
    arm's directory and archive it as stale.
    """
    from rocklabel.train.ablate import arm_dir

    arms = arms_of("reflectivity")
    dirs = [arm_dir(str(tmp_path), "reflectivity", a, "r1") for a in arms]
    assert len(set(dirs)) == len(dirs)


# --------------------------------------------------------------------------- #
# prevalence-normalized scores
# --------------------------------------------------------------------------- #
def test_normalizing_puts_guessing_at_zero_and_perfect_at_one():
    """The whole point: a fold with many rocks must not look better for free."""
    # A no-skill model's PR-AUC *is* the rock share, whatever that share is.
    assert M.normalized_pr_auc(0.063, 0.063) == pytest.approx(0.0)
    assert M.normalized_pr_auc(0.316, 0.316) == pytest.approx(0.0)
    assert M.normalized_pr_auc(1.0, 0.2) == pytest.approx(1.0)
    # Two folds five times apart in rock share, both a fifth of the way from
    # guessing to perfect, now read as equally hard.
    low = M.normalized_pr_auc(0.063 + 0.2 * (1 - 0.063), 0.063)
    high = M.normalized_pr_auc(0.316 + 0.2 * (1 - 0.316), 0.316)
    assert low == pytest.approx(high) == pytest.approx(0.2)


def test_summarize_carries_the_normalized_score_beside_the_raw_one():
    labels = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], np.int8)
    probs = np.linspace(0.9, 0.1, 10)
    out = M.summarize(labels, probs, 0.5)
    assert out["norm_pr_auc"] == pytest.approx(
        M.normalized_pr_auc(out["pr_auc"], out["baseline_pr_auc"]))


def test_reordering_bites_a_high_prevalence_fold_that_looked_fine(tmp_path):
    """A fold can lead on raw PR-AUC and trail once its rock share is removed.

    This is the trap the normalized column exists for, so it is asserted
    rather than trusted: 'which recording is hard' must be answered by the
    normalized number.
    """
    root = str(tmp_path)
    _fake_fold(root, "fullsweep", "pointnet2-geom", "rocky", 0.70, prevalence=0.32)
    _fake_fold(root, "fullsweep", "pointnet2-geom", "sparse", 0.60, prevalence=0.06)
    arm = collect(root, "fullsweep")["arms"][0]
    assert arm["pr_auc"]["per_fold"]["rocky"] > arm["pr_auc"]["per_fold"]["sparse"]
    assert arm["norm_pr_auc"]["per_fold"]["rocky"] < arm["norm_pr_auc"]["per_fold"]["sparse"]
    assert arm["prevalence"] == {"rocky": 0.32, "sparse": 0.06}


def test_folds_finished_before_the_normalized_score_existed_are_backfilled(tmp_path):
    """The two completed sweeps predate the column and must still fill it in."""
    root = str(tmp_path)
    d = os.path.join(root, "fullsweep", "pointnet2-geom", "loro_old")
    os.makedirs(d)
    with open(os.path.join(d, "test_metrics.json"), "w") as f:
        json.dump({"test_run": "old", "pr_auc": 0.5, "roc_auc": 0.8, "f1": 0.4,
                   "precision": 0.4, "recall": 0.4, "baseline_pr_auc": 0.2}, f)
    arm = collect(root, "fullsweep")["arms"][0]
    assert arm["norm_pr_auc"]["per_fold"]["old"] == pytest.approx(0.375)


def test_the_report_prints_both_a_raw_and_a_normalized_per_fold_table(tmp_path):
    from rocklabel.train.ablate_report import render_ablation

    root, out = str(tmp_path / "runs"), str(tmp_path / "rep")
    for fold, prev in (("a", 0.30), ("b", 0.08)):
        _fake_fold(root, "fullsweep", "pointnet2-geom", fold, 0.6, prevalence=prev)
    render_ablation(root, "fullsweep", out)
    text = open(os.path.join(out, "summary.md")).read()
    assert "Per-fold detail, raw PR-AUC" in text
    assert "Per-fold detail, with rock share divided out" in text
    # and the shares themselves, so the gap between the two tables is explicable
    assert "30.0%" in text and "8.0%" in text


# --------------------------------------------------------------------------- #
# a suite is bound to the cache it was defined against
# --------------------------------------------------------------------------- #
def test_every_suite_names_a_cache_that_a_profile_can_build():
    from rocklabel.profiles import PROFILES
    from rocklabel.train.ablate import cache_profile, default_cache_dir

    for suite in SUITES:
        assert cache_profile(suite) in PROFILES, suite
        assert default_cache_dir(suite).endswith(cache_profile(suite))


def test_a_suite_refuses_a_cache_cut_a_different_way(tmp_path):
    """Arms are only comparable if every one of them saw the same frames, so
    pointing a suite at another suite's cache has to fail loudly rather than
    produce a table that looks fine and answers nothing."""
    from rocklabel.profiles import apply_profile
    from rocklabel.config import DEFAULTS, config_hash
    from rocklabel.train.ablate import check_cache_matches

    cache = tmp_path / "cache"
    cache.mkdir()
    wrong = config_hash(apply_profile(DEFAULTS, "raw-burst"))
    (cache / "meta.json").write_text(json.dumps(
        {"config_hash": wrong, "profile": "raw-burst", "runs": {}}))
    check_cache_matches("reflectivity", str(cache))          # its own cache
    with pytest.raises(SystemExit) as e:
        check_cache_matches("segdense", str(cache))
    assert "full-sweep-dense" in str(e.value) and "raw-burst" in str(e.value)
