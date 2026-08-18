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
