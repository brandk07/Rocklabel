"""The map model: accumulated height grid, its channels, labels and grading."""

from __future__ import annotations

import numpy as np
import pytest

from rocklabel.labels import LabelSet, Rock
from rocklabel.train.mapnet import grade as G
from rocklabel.train.mapnet.grid import (CELL_M, CHANNELS, NBINS, HeightGrid, channels, features,
                                   normalise_brightness)


def _scene(seed=0, n=40_000):
    """A flat floor at z=0 with one 20 cm square block standing 15 cm tall."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, (n, 2))
    z = rng.normal(0.0, 0.003, n)
    on = (np.abs(xy[:, 0] - 0.3) < 0.1) & (np.abs(xy[:, 1] + 0.2) < 0.1)
    z[on] = rng.uniform(0.0, 0.15, on.sum())
    return np.c_[xy, z], rng.normal(0, 1, n).astype(np.float32)


def _block_labels():
    v = [[0.2, -0.3], [0.4, -0.3], [0.4, -0.1], [0.2, -0.1]]
    ls = LabelSet(mcap_file="x", run_id="x", odom_frame="odom",
                  intensity_available=True, accumulator_voxel_m=0.03, level=None)
    ls.rocks = [Rock(id=1, center=[0.3, -0.2, 0.07], radius=0.15, shape="polygon",
                     vertices=v, z_range=(-0.02, 0.2))]
    ls.arena = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    return ls


def test_one_sweep_at_a_time_equals_everything_at_once():
    xyz, b = _scene()
    whole = HeightGrid(-1, -1, 80, 80, 0.0)
    whole.add(xyz, b)
    swept = HeightGrid(-1, -1, 80, 80, 0.0)
    for a in range(0, len(xyz), 500):          # small adds take the sparse path
        swept.add(xyz[a:a + 500], b[a:a + 500])
    assert np.array_equal(whole.hist, swept.hist)
    assert np.allclose(whole.bsum, swept.bsum, atol=1e-3)
    assert np.array_equal(whole.bmax, swept.bmax)


def test_the_block_stands_out_of_the_local_ground():
    xyz, b = _scene()
    g = HeightGrid(-1, -1, 80, 80, 0.0)
    g.add(xyz, b)
    x, st = features(g, "profile")
    assert x.shape == (len(channels("profile")), 80, 80)
    rel = x[CHANNELS.index("rel_q90")]
    col = lambda v: int((v + 1) / CELL_M)    # noqa: E731
    on_block = rel[col(-0.2), col(0.3)]
    on_floor = rel[col(0.5), col(-0.6)]
    # Channels are in tenths of a metre.
    assert on_block > 1.0 and abs(on_floor) < 0.2


def test_heights_outside_the_band_are_dropped():
    g = HeightGrid(0, 0, 4, 4, 0.0)
    g.add(np.array([[0.01, 0.01, -0.5], [0.01, 0.01, 0.9], [0.01, 0.01, 0.05]]), None)
    assert int(g.hist.sum()) == 1 and g.hist.shape == (16, NBINS)
    assert g.has_bright is False


def test_brightness_is_scaled_per_recording():
    v = normalise_brightness(np.array([10, 20, 30, 40, 50], np.float32))
    assert abs(float(np.median(v))) < 1e-6


def test_a_perfect_map_covers_every_rock_with_no_false_ground():
    xyz, b = _scene()
    labels = _block_labels()
    g = HeightGrid(-1, -1, 80, 80, 0.0)
    g.add(xyz, b)
    _, st = features(g)
    xs, ys = g.cell_centers()
    gx, gy = np.meshgrid(xs, ys)
    truth = ((np.abs(gx - 0.3) < 0.1) & (np.abs(gy + 0.2) < 0.1)).astype(float)
    cells = G.to_cells(truth, st, g)
    on = xyz[(np.abs(xyz[:, 0] - 0.3) < 0.1) & (np.abs(xyz[:, 1] + 0.2) < 0.1)]
    visible = {1: {tuple(k) for k in np.floor(on[:, :2] / 0.1).astype(int)}}
    res = G.grade(cells, labels, visible, 0.5)
    assert res["false_cells_footprint"] == 0
    assert res["per_rock"][0]["occupied"] == 1.0
    assert res["budgets"]["100"]["macro_coverage"] == res["macro_coverage"]


def test_labels_follow_the_crop_transform():
    from rocklabel.train.mapnet.data import Xform, rasterize_labels
    labels = _block_labels()
    g = HeightGrid(-0.5, -0.5, 40, 40, 0.0)
    observed = np.ones((40, 40), bool)
    # Centre the crop on the block and turn it a quarter: the block is still
    # at the crop's centre and still 20 cm square.
    y = rasterize_labels(labels, Xform([0.3, -0.2], np.pi / 2, False, 1.0), g, observed)
    assert y[20, 20] == 1
    assert 30 <= (y == 1).sum() <= 90       # ~64 cells of 2.5 cm, give or take the edge


def test_the_arena_folds_split_the_ground_and_keep_touching_rocks_together():
    from rocklabel.train.mapnet.train import LANCE_FOLDS, LANCE_LABELS, fold_region
    from rocklabel.labels import load_labels
    try:
        labels = load_labels(LANCE_LABELS)
    except FileNotFoundError:
        pytest.skip("competition labels not present")
    a, b = set(LANCE_FOLDS["A"]), set(LANCE_FOLDS["B"])
    assert not (a & b) and (a | b) == {r.id for r in labels.rocks}
    xy = np.random.default_rng(0).uniform([-6, -4], [0.7, 1], (5000, 2))
    in_a = fold_region("A", labels, True)(xy)
    in_b = fold_region("A", labels, False)(xy)
    assert not (in_a & in_b).any()
    # Every rock's whole outline lies on its own group's ground.
    for rock in labels.rocks:
        v = np.asarray(rock.vertices, float)
        mine = fold_region("A", labels, rock.id in a)(v)
        assert mine.all(), rock.id


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs a GPU")
def test_the_gpu_channels_match_the_cpu_reference():
    import torch
    from rocklabel.train.mapnet.gpu import features_from_grid
    xyz, b = _scene(n=80_000)
    g = HeightGrid(-1, -1, 80, 80, 0.0)
    g.add(xyz, b)
    cpu, st = features(g, "profile")
    gpu, valid, q90 = features_from_grid(g, torch.device("cuda"), "profile")
    gpu = gpu[0].cpu().numpy()
    assert np.array_equal(valid, st["valid"])
    assert np.allclose(q90, st["q90_abs"])
    # Everything but the ground-relative channels is identical; those differ
    # only through how holes in the ground are filled, by millimetres here.
    for i, name in enumerate(CHANNELS):
        tol = 0.05 if name.startswith("rel_") or name == "ground" else 1e-5
        assert np.abs(cpu[i] - gpu[i])[st["valid"]].max() <= tol, name
