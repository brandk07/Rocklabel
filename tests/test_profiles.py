"""Generation profiles: the named ways of cutting a recording into frames.

The point of naming them is that a dataset should say on its face how it was
built. Two agents have already drawn the wrong conclusion from four look-alike
YAML files at the repo root, so these tests guard the two properties that make
the names trustworthy: a profile is exactly a set of config overrides (so it
reproduces what is already on disk), and one profile can never be pooled with
another.
"""

from __future__ import annotations

import json
import os

import pytest

from rocklabel import profiles
from rocklabel.config import DEFAULTS, config_hash, load_config
from rocklabel.dataset.generate import ManifestConflict, check_manifest

#: Config fingerprints of the datasets already on disk when profiles were
#: introduced. A profile that stops reproducing these has orphaned real data:
#: every dataset and every cache built under it would have to be regenerated.
#: If one of these has to change, the datasets have to be rebuilt with it.
FROZEN_HASHES = {
    "raw-burst": "a81b9c29",     # the eleven original volleyball datasets
    "full-sweep": "3ccba26a",    # datasets/full-sweep/volleyball
}


@pytest.mark.parametrize("name,prefix", sorted(FROZEN_HASHES.items()))
def test_a_profile_reproduces_the_datasets_already_on_disk(name, prefix):
    cfg = profiles.apply_profile(load_config(), name)
    assert config_hash(cfg).startswith(prefix), (
        f"profile {name!r} no longer produces the config that built the existing "
        "datasets — they would all have to be regenerated"
    )


def test_every_profile_has_prose_and_valid_overrides():
    from rocklabel.config import apply_overrides

    assert profiles.DEFAULT_PROFILE in profiles.PROFILES
    for name, p in profiles.PROFILES.items():
        assert p.name == name
        assert len(p.what) > 40, f"{name} needs a real 'what it does'"
        assert len(p.when) > 40, f"{name} needs a real 'when to reach for it'"
        # apply_overrides raises on an unknown key, so this is the guard that a
        # typo in a profile is caught here and not on an overnight sweep.
        apply_overrides(load_config(), dict(p.overrides))


def test_the_default_profile_is_the_one_measurement_favours():
    """Full sweep beat raw bursts on every model and nearly every fold."""
    assert profiles.DEFAULT_PROFILE == "full-sweep"
    assert profiles.PROFILES["raw-burst"].legacy, (
        "raw bursts starve the models and cannot train the segmenter at all — "
        "they must not be offered as a fresh choice"
    )


def test_profiles_are_identified_from_a_config_hash_alone():
    """A dataset generated before profiles existed still gets named."""
    for name in FROZEN_HASHES:
        cfg = profiles.apply_profile(load_config(), name)
        assert profiles.identify(cfg) == name
        assert profiles.identify(config_hash(cfg)) == name
    assert profiles.identify("deadbeef" * 8) is None


def test_two_profiles_never_share_a_fingerprint():
    seen = {}
    for name in profiles.PROFILES:
        h = config_hash(profiles.apply_profile(dict(DEFAULTS), name))
        assert h not in seen, f"{name} and {seen[h]} would collide on one dataset"
        seen[h] = name


def test_an_unknown_profile_says_which_ones_exist():
    with pytest.raises(profiles.ProfileError, match="full-sweep"):
        profiles.get("no-such-profile")


def test_a_dataset_built_one_way_refuses_data_built_another(tmp_path):
    """The guard that keeps two ways of cutting frames out of one folder.

    Pooling them would mix populations whose scores are not comparable, and the
    refusal has to name both ways round — a pair of hex fingerprints tells
    nobody what went wrong.
    """
    out = tmp_path / "ds"
    out.mkdir()
    full = profiles.apply_profile(load_config(), "full-sweep")
    manifest = check_manifest(str(out), full, "full-sweep")
    (out / "manifest.json").write_text(json.dumps(manifest))

    raw = profiles.apply_profile(load_config(), "raw-burst")
    with pytest.raises(ManifestConflict) as e:
        check_manifest(str(out), raw, "raw-burst")
    assert "full-sweep" in str(e.value) and "raw-burst" in str(e.value)


def test_a_generated_dataset_records_the_profile_that_built_it(tmp_path):
    out = tmp_path / "ds"
    out.mkdir()
    cfg = profiles.apply_profile(load_config(), "full-sweep")
    manifest = check_manifest(str(out), cfg, "full-sweep")
    assert manifest["profile"] == "full-sweep"
    # Recorded beside the config, never inside it: folding the name into the
    # hash would give two spellings of one setting two dataset directories.
    assert "profile" not in manifest["config"]


def test_the_default_output_folder_carries_the_profile():
    from rocklabel.cli import default_dataset_dir

    assert default_dataset_dir(
        "full-sweep", "recordings/volleyball/reslam/VolleyBallTest4.reslam.mcap"
    ) == os.path.join("datasets", "full-sweep", "VolleyBallTest4.reslam")


def test_cache_refuses_to_pool_two_profiles_and_names_them(tmp_path):
    from rocklabel.train.data import DataError, build_cache

    dirs = []
    for name in ("full-sweep", "raw-burst"):
        d = tmp_path / name
        d.mkdir()
        cfg = profiles.apply_profile(load_config(), name)
        (d / "manifest.json").write_text(json.dumps({
            "profile": name, "config": cfg, "config_hash": config_hash(cfg),
            "runs": {f"run_{name}": {"sample_labels": {"rock": 1, "clear": 1}}},
        }))
        dirs.append(str(d))

    with pytest.raises(DataError) as e:
        build_cache(dirs, str(tmp_path / "cache"))
    assert "full-sweep" in str(e.value) and "raw-burst" in str(e.value)


def test_cache_with_nothing_to_pool_says_what_to_do(tmp_path):
    from rocklabel.train.data import DataError, build_cache, default_datasets

    assert default_datasets(root=str(tmp_path / "nope")) == []
    with pytest.raises(DataError, match="generate"):
        build_cache([], str(tmp_path / "cache"))


def test_default_datasets_reads_the_profile_folder(tmp_path):
    from rocklabel.train.data import default_datasets

    base = tmp_path / "datasets" / "full-sweep"
    for name in ("b", "a"):
        (base / name).mkdir(parents=True)
        (base / name / "manifest.json").write_text("{}")
    (base / "not-a-dataset").mkdir()
    found = default_datasets("full-sweep", root=str(tmp_path / "datasets"))
    assert [os.path.basename(p) for p in found] == ["a", "b"]


# --------------------------------------------------------------------------- #
# the rock-coverage check
# --------------------------------------------------------------------------- #
def _tiny_dataset(tmp_path, frames, rocks, radius=0.2):
    """A dataset directory with just enough on disk for coverage to read."""
    import json
    import numpy as np

    d = tmp_path / "ds"
    (d / "points" / "run1").mkdir(parents=True)
    labels = d / "run1.labels.json"
    labels.write_text(json.dumps({"rocks": [
        {"id": i + 1, "center": list(c), "radius": radius}
        for i, c in enumerate(rocks)]}))
    for i, (centers, lab, origin) in enumerate(frames):
        np.savez(d / "points" / "run1" / f"frame_{i:06d}.npz",
                 centers_odom=np.array(centers, np.float32).reshape(-1, 3),
                 labels=np.array(lab, np.int8),
                 robot_pose=np.eye(4) * 1.0 + np.pad(
                     np.array(origin, float).reshape(3, 1), ((0, 1), (3, 0))))
    (d / "manifest.json").write_text(json.dumps({
        "profile": "full-sweep",
        "config": {"generator": {"crop_forward_m": 6.0, "min_neighbors": 20}},
        "runs": {"run1": {"labels_path": str(labels)}}}))
    return str(d)


def test_coverage_names_a_labelled_rock_that_produced_no_samples(tmp_path):
    """The silent failure this command exists for: the rock is labelled, the
    dataset builds cleanly, and the model never sees it."""
    from rocklabel.dataset.coverage import measure_dataset

    ds = _tiny_dataset(
        tmp_path,
        # samples land on rock 1 only; rock 2 is never sampled
        frames=[([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], [1, 0], [0.0, 0.0, 0.0])],
        rocks=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    out = measure_dataset(ds)
    assert out["silent_rocks"] == 1
    assert out["runs"][0]["silent_rocks"] == [2]
    assert out["runs"][0]["samples_per_rock"] == {1: 1, 2: 0}


def test_coverage_measures_range_from_the_sensor_not_the_world_origin(tmp_path):
    """The rig walks, so distance from the origin is a different question."""
    from rocklabel.dataset.coverage import measure_dataset

    ds = _tiny_dataset(
        tmp_path,
        frames=[([[10.0, 0.0, 0.0]], [1], [8.0, 0.0, 0.0])],
        rocks=[(10.0, 0.0, 0.0)])
    run = measure_dataset(ds)["runs"][0]
    assert run["rock_range_max_m"] == pytest.approx(2.0)


def test_coverage_flags_labels_too_close_to_tell_apart(tmp_path):
    """Two labels closer than their own radii split one pile of samples, so a
    low count on one of them is not evidence that rock was missed."""
    from rocklabel.dataset.coverage import measure_dataset

    ds = _tiny_dataset(
        tmp_path,
        frames=[([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], [1, 1], [0.0, 0.0, 0.0])],
        rocks=[(0.0, 0.0, 0.0), (0.25, 0.0, 0.0)], radius=0.2)
    run = measure_dataset(ds)["runs"][0]
    assert run["overlapping_labels"] == [{"rocks": [1, 2], "gap_m": 0.25}]
