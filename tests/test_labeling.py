"""Shape membership labeling: rock / ignore shell / clear."""

import json

import numpy as np
import pytest

from rocklabel.labeling import (LABEL_CLEAR, LABEL_IGNORE, LABEL_ROCK,
                                inside_arena, label_points, label_rocks,
                                points_in_rock)
from rocklabel.neighborhoods import build_neighborhood_samples
from rocklabel.labels import LabelSet, load_labels
from rocklabel.config import load_config


def test_single_sphere_membership():
    pts = np.array([
        [0.3, 0.0, 0.0],   # inside sphere -> rock
        [0.5, 0.0, 0.0],   # exactly on boundary -> rock
        [0.55, 0.0, 0.0],  # in shell -> ignore
        [0.7, 0.0, 0.0],   # outside shell -> clear
    ])
    labels = label_points(pts, np.array([[0.0, 0.0, 0.0]]), np.array([0.5]), shell_m=0.1)
    np.testing.assert_array_equal(labels, [LABEL_ROCK, LABEL_ROCK, LABEL_IGNORE, LABEL_CLEAR])


def test_rock_wins_over_other_spheres_shell():
    # Point inside sphere A but within sphere B's shell must stay rock.
    pts = np.array([[0.45, 0.0, 0.0]])
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii = np.array([0.5, 0.5])
    labels = label_points(pts, centers, radii, shell_m=0.1)
    np.testing.assert_array_equal(labels, [LABEL_ROCK])


def test_no_spheres_all_clear():
    pts = np.random.default_rng(0).normal(size=(50, 3))
    labels = label_points(pts, np.empty((0, 3)), np.empty(0), shell_m=0.05)
    assert (labels == LABEL_CLEAR).all()


def test_box_membership():
    ls = LabelSet()
    ls.add_box(center=[0.0, 0.0, 0.5], size=[1.0, 2.0, 1.0])  # x/y/z in ±0.5/±1/[0,1]
    pts = np.array([
        [0.0, 0.0, 0.5],    # dead center -> rock
        [0.49, 0.99, 0.99],  # just inside a corner -> rock
        [0.55, 0.0, 0.5],    # 5 cm past +x face -> shell
        [0.0, 0.0, 1.2],     # 20 cm above the top -> clear
        [3.0, 0.0, 0.5],     # far away -> clear
    ])
    labels = label_rocks(pts, ls.rocks, shell_m=0.1)
    np.testing.assert_array_equal(
        labels, [LABEL_ROCK, LABEL_ROCK, LABEL_IGNORE, LABEL_CLEAR, LABEL_CLEAR])


def test_polygon_membership():
    ls = LabelSet()
    # unit square footprint, extruded z in [0, 1]
    ls.add_polygon([[0, 0], [1, 0], [1, 1], [0, 1]], z_min=0.0, z_max=1.0)
    pts = np.array([
        [0.5, 0.5, 0.5],   # inside -> rock
        [0.5, 0.5, 1.5],   # above the top -> clear (past shell)
        [1.05, 0.5, 0.5],  # 5 cm outside the edge -> shell
        [0.5, 0.5, 1.05],  # 5 cm above the top -> shell
        [2.0, 2.0, 0.5],   # far away -> clear
    ])
    labels = label_rocks(pts, ls.rocks, shell_m=0.1)
    np.testing.assert_array_equal(
        labels, [LABEL_ROCK, LABEL_CLEAR, LABEL_IGNORE, LABEL_IGNORE, LABEL_CLEAR])


def test_concave_polygon():
    # L-shape: the notch (upper right) is outside the polygon.
    verts = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
    ls = LabelSet()
    rock = ls.add_polygon(verts, z_min=0.0, z_max=1.0)
    pts = np.array([[0.5, 1.5, 0.5],   # in the vertical arm
                    [1.5, 0.5, 0.5],   # in the horizontal arm
                    [1.5, 1.5, 0.5]])  # in the notch -> outside
    np.testing.assert_array_equal(points_in_rock(pts, rock), [True, True, False])


def test_schema_v2_roundtrip(tmp_path):
    ls = LabelSet()
    ls.add([1.0, 2.0, 3.0], 0.25)
    ls.add_box([0.0, 0.0, 0.5], [1.0, 2.0, 1.0])
    ls.add_polygon([[0, 0], [1, 0], [1, 1]], z_min=0.1, z_max=0.9)
    path = str(tmp_path / "labels.json")
    ls.save(path)

    back = load_labels(path)
    shapes = [r.shape for r in back.rocks]
    assert shapes == ["sphere", "box", "polygon"]
    np.testing.assert_allclose(back.rocks[1].size, [1.0, 2.0, 1.0])
    assert back.rocks[2].z_range == (0.1, 0.9)
    # bounding sphere is kept for every shape
    assert all(r.radius > 0 for r in back.rocks)
    # labeling through the reloaded set matches the original
    pts = np.random.default_rng(1).uniform(-1, 3, size=(200, 3))
    np.testing.assert_array_equal(label_rocks(pts, ls.rocks, 0.1),
                                  label_rocks(pts, back.rocks, 0.1))


def test_schema_v1_still_loads(tmp_path):
    import json
    v1 = {"schema_version": 1, "run_id": "old",
          "rocks": [{"id": 3, "center": [1.0, 2.0, 3.0], "radius": 0.4}]}
    path = tmp_path / "old.labels.json"
    path.write_text(json.dumps(v1))
    ls = load_labels(str(path))
    assert ls.rocks[0].shape == "sphere"
    assert ls.rocks[0].radius == 0.4
    assert ls._next_id == 4


# -- arena boundary -----------------------------------------------------------

def test_inside_arena_accepts_everything_when_unset():
    pts = np.random.default_rng(0).uniform(-50, 50, size=(100, 3))
    assert inside_arena(pts, None).all()
    assert len(inside_arena(np.empty((0, 3)), None)) == 0


def test_inside_arena_is_an_xy_test():
    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    pts = np.array([
        [2.0, 2.0, 0.0],     # middle
        [2.0, 2.0, 99.0],    # same xy, absurd height: still inside
        [-1.0, 2.0, 0.0],    # outside in x
        [2.0, 9.0, 0.0],     # outside in y
    ])
    np.testing.assert_array_equal(inside_arena(pts, square),
                                  [True, True, False, False])


def test_arena_survives_a_save_load_round_trip(tmp_path):
    ls = LabelSet()
    ls.add([1.0, 1.0, 0.0], 0.2)
    ls.set_arena([[0, 0], [5, 0], [5, 5], [0, 5]])
    path = str(tmp_path / "labels.json")
    ls.save(path)
    back = load_labels(path)
    np.testing.assert_allclose(back.arena, [[0, 0], [5, 0], [5, 5], [0, 5]])
    assert len(back.rocks) == 1


def test_labels_without_an_arena_stay_arena_free(tmp_path):
    ls = LabelSet()
    ls.add([0.0, 0.0, 0.0], 0.3)
    path = str(tmp_path / "labels.json")
    ls.save(path)
    assert "arena" not in json.loads(open(path).read())
    assert load_labels(path).arena is None


# -- training height band ------------------------------------------------------

def test_z_band_survives_a_save_load_round_trip(tmp_path):
    ls = LabelSet()
    ls.add([1.0, 1.0, 0.0], 0.2)
    ls.set_z_band(-0.4, 0.9)
    path = str(tmp_path / "labels.json")
    ls.save(path)
    assert load_labels(path).z_band == (-0.4, 0.9)


def test_labels_without_a_z_band_stay_band_free(tmp_path):
    ls = LabelSet()
    ls.add([0.0, 0.0, 0.0], 0.3)
    path = str(tmp_path / "labels.json")
    ls.save(path)
    assert "z_band" not in json.loads(open(path).read())
    assert load_labels(path).z_band is None


def test_set_z_band_sorts_its_ends():
    """The two ends come straight off slider callbacks, which can arrive in
    either order while a drag crosses over."""
    ls = LabelSet()
    assert ls.set_z_band(1.5, -0.5) == (-0.5, 1.5)
    ls.clear_z_band()
    assert ls.z_band is None


def test_set_arena_rejects_a_degenerate_polygon():
    ls = LabelSet()
    for bad in ([], [[0, 0]], [[0, 0], [1, 1]]):
        with pytest.raises(ValueError, match="at least 3"):
            ls.set_arena(bad)


def test_arena_restricts_sample_centers_but_not_neighborhood_context():
    """A center outside the arena is dropped; a center just inside keeps the
    full ball of context, including the points beyond the boundary."""
    rng = np.random.default_rng(0)
    # dense ground slab spanning x = -2..2
    g = rng.uniform([-2, -2, 0], [2, 2, 0.02], size=(20000, 3))
    inten = np.full(len(g), 0.5)
    gcfg = {"centers_voxel_m": 0.1, "neighborhood_radius_m": 0.5, "min_neighbors": 5,
            "neighborhood_points": 64, "negative_keep_prob": 1.0,
            "boundary_shell_m": 0.05}
    half = np.array([[-2.0, -2.0], [0.0, -2.0], [0.0, 2.0], [-2.0, 2.0]])  # keep x <= 0

    full = build_neighborhood_samples(g, inten, [], gcfg, np.random.default_rng(1))
    bounded = build_neighborhood_samples(g, inten, [], gcfg, np.random.default_rng(1),
                                         arena=half)
    assert full is not None and bounded is not None
    assert bounded["centers_odom"][:, 0].max() <= 1e-9      # no center past the line
    assert len(bounded["labels"]) < len(full["labels"])     # and fewer of them

    # A center within a ball's reach of the boundary still sees points from the
    # far side: its neighbor count matches what it would have had unbounded.
    edge = np.argmax(bounded["centers_odom"][:, 0])
    assert bounded["true_counts"][edge] >= gcfg["min_neighbors"]
    near = np.argmin(np.linalg.norm(
        full["centers_odom"] - bounded["centers_odom"][edge], axis=1))
    assert abs(int(full["true_counts"][near]) - int(bounded["true_counts"][edge])) <= 2


def test_resuming_labels_pins_the_frame_before_accumulating(tmp_path, monkeypatch):
    """`label` must read the label file first, so the cloud it fuses is in the
    same frame the existing rocks were picked in."""
    import json

    import rocklabel.labeler as labeler

    labels_path = tmp_path / "run.labels.json"
    labels_path.write_text(json.dumps({
        "schema_version": 4, "run_id": "run", "mcap_file": "run.mcap",
        "rocks": [{"id": 1, "shape": "sphere", "center": [1.0, 0.0, 0.0],
                   "radius": 0.15}],
    }))

    seen = {}

    def fake_accumulate(mcap_path, cfg, stride):
        seen["mode"] = cfg["level"]["mode"]
        raise SystemExit("stop after the config is settled")

    monkeypatch.setattr(labeler, "accumulate_cloud", fake_accumulate)
    cfg = load_config(None)
    assert cfg["level"]["mode"] == "auto"
    with pytest.raises(SystemExit):
        labeler.run_label(str(tmp_path / "run.mcap"), cfg, str(labels_path),
                          stride=1, z_min=None, z_max=None)
    # The label file carries no frame, so levelling is pinned off for it.
    assert seen["mode"] == "off"
    # ...and the caller's config was not mutated on the way through.
    assert cfg["level"]["mode"] == "auto"
