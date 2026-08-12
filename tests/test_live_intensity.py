"""Tests for intensity (reflectivity) threading: crop-mask alignment and the
ring buffer's parallel intensity channel."""

from __future__ import annotations

import numpy as np

from rocklabel.live.config import CropConfig
from rocklabel.live.filters import crop_batch, crop_mask
from rocklabel.live.pipeline import RawPointBuffer


def test_crop_mask_none_when_all_kept():
    cfg = CropConfig(enabled=True, z_min=-10, z_max=10)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert crop_mask(pts, cfg) is None
    assert crop_batch(pts, cfg) is pts


def test_crop_mask_aligns_with_intensity():
    cfg = CropConfig(enabled=True, z_min=-1.5, z_max=-0.5)
    pts = np.array(
        [[1, 0, -1.0], [2, 0, 0.0], [3, 0, -1.2], [4, 0, 2.0]], dtype=np.float64
    )
    inten = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    keep = crop_mask(pts, cfg)
    assert keep is not None
    kept_pts = pts[keep]
    kept_int = inten[keep]
    assert np.allclose(kept_pts[:, 2], [-1.0, -1.2])
    assert np.allclose(kept_int, [10.0, 30.0])  # stayed aligned with points


def test_buffer_stores_and_returns_intensity():
    buf = RawPointBuffer(100)
    pts = np.arange(30, dtype=np.float64).reshape(10, 3)
    inten = np.arange(10, dtype=np.float32)
    buf.add(pts, inten)
    out_pts, out_int = buf.snapshot()
    assert np.allclose(out_pts, pts)
    assert np.allclose(out_int, inten)


def test_buffer_nan_intensity_when_absent():
    buf = RawPointBuffer(10)
    buf.add(np.zeros((4, 3)))
    _, inten = buf.snapshot()
    assert np.all(np.isnan(inten))


def test_buffer_intensity_survives_wraparound():
    """Intensity must wrap with its points, staying pairwise aligned."""
    buf = RawPointBuffer(8)
    # Two adds totaling 12 rows: the ring wraps; x coordinate == intensity tag.
    for start in (0, 6):
        n = 6
        tags = np.arange(start, start + n, dtype=np.float32)
        pts = np.column_stack([tags, np.zeros(n), np.zeros(n)]).astype(np.float64)
        buf.add(pts, tags)
    out_pts, out_int = buf.snapshot()
    assert out_pts.shape == (8, 3)
    # Pairing invariant: intensity always equals its point's x tag.
    assert np.allclose(out_pts[:, 0], out_int)
    # The 8 survivors are the newest 8 of the 12 tags (4..11), any order.
    assert set(out_int.astype(int)) == set(range(4, 12))


def test_buffer_oversized_add_keeps_tail():
    buf = RawPointBuffer(5)
    tags = np.arange(12, dtype=np.float32)
    pts = np.column_stack([tags, np.zeros(12), np.zeros(12)]).astype(np.float64)
    buf.add(pts, tags)
    out_pts, out_int = buf.snapshot()
    assert np.allclose(out_int, [7, 8, 9, 10, 11])
    assert np.allclose(out_pts[:, 0], out_int)
