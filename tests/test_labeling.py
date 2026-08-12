"""Shape membership labeling: rock / ignore shell / clear."""

import numpy as np

from rocklabel.labeling import (LABEL_CLEAR, LABEL_IGNORE, LABEL_ROCK,
                                label_points, label_rocks, points_in_rock)
from rocklabel.labels import LabelSet, load_labels


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
