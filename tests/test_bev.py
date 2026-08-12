"""BEV channel values for a hand-constructed 4-point cloud on a 4x4 grid."""

import numpy as np

from rocklabel.bev import bev_grid_shape, rasterize_bev
from rocklabel.labeling import MASK_IGNORE

GCFG = {
    "crop_forward_m": 0.2, "crop_backward_m": 0.2,
    "crop_left_m": 0.2, "crop_right_m": 0.2,
    "bev_cell_m": 0.1,
}


def test_hand_constructed_cloud():
    assert bev_grid_shape(GCFG) == (4, 4)
    base = np.array([0.0, 0.0, 0.0])
    xyz = np.array([
        [0.05, 0.05, 0.2],    # cell (2,2), clear
        [0.05, 0.05, 0.4],    # cell (2,2), rock
        [-0.15, -0.15, 0.1],  # cell (0,0), ignore only
        [0.15, -0.05, 0.0],   # cell (3,1), clear
    ])
    inten = np.array([0.5, 1.0, 0.3, 0.7], np.float32)
    labels = np.array([0, 1, -1, 0], np.int8)

    channels, mask = rasterize_bev(xyz, inten, labels, base, GCFG)
    assert channels.shape == (8, 4, 4)
    assert mask.shape == (4, 4)

    # cell (2,2): two points
    np.testing.assert_allclose(channels[0, 2, 2], 1.0)                 # valid
    np.testing.assert_allclose(channels[1, 2, 2], np.log1p(2))         # count
    np.testing.assert_allclose(channels[2, 2, 2], 0.4)                 # max z - base z
    np.testing.assert_allclose(channels[3, 2, 2], 0.2)                 # min z - base z
    np.testing.assert_allclose(channels[4, 2, 2], 0.2)                 # z span
    np.testing.assert_allclose(channels[5, 2, 2], 0.1)                 # z std (population)
    np.testing.assert_allclose(channels[6, 2, 2], 0.75)                # mean intensity
    np.testing.assert_allclose(channels[7, 2, 2], 1.0)                 # max intensity
    assert mask[2, 2] == 1                                             # contains a rock point

    # cell (0,0): single ignore point -> valid but masked out
    np.testing.assert_allclose(channels[0, 0, 0], 1.0)
    np.testing.assert_allclose(channels[4, 0, 0], 0.0)  # one point: zero span
    np.testing.assert_allclose(channels[5, 0, 0], 0.0)  # one point: zero std
    assert mask[0, 0] == MASK_IGNORE

    # cell (3,1): single clear point
    assert mask[3, 1] == 0
    np.testing.assert_allclose(channels[2, 3, 1], 0.0)   # z == base z
    np.testing.assert_allclose(channels[6, 3, 1], 0.7)

    # every empty cell: all channels 0, mask 255
    empty = np.ones((4, 4), bool)
    empty[2, 2] = empty[0, 0] = empty[3, 1] = False
    assert (channels[:, empty] == 0).all()
    assert (mask[empty] == MASK_IGNORE).all()


def test_out_of_grid_points_ignored():
    xyz = np.array([[5.0, 5.0, 1.0]])
    channels, mask = rasterize_bev(
        xyz, np.array([1.0]), np.array([0], np.int8), np.zeros(3), GCFG
    )
    assert (channels == 0).all()
    assert (mask == MASK_IGNORE).all()
