"""Tests for the accumulated display cloud's frame-based retention.

The viewer exposes "Accum frames"; these pin down that the buffer really keeps
that many frames (and no more), that the point ceiling still guards memory, and
that intensity stays paired with its points through eviction.
"""

from __future__ import annotations

import numpy as np

from rocklabel.live.pipeline import FrameAccumBuffer


def _frame(tag: float, n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """A frame whose every point (and intensity) carries the same tag."""
    tags = np.full(n, tag, dtype=np.float32)
    pts = np.column_stack([tags, np.zeros(n), np.zeros(n)]).astype(np.float64)
    return pts, tags


def test_keeps_exactly_max_frames():
    buf = FrameAccumBuffer(max_frames=3, max_points=10_000)
    for i in range(10):
        buf.add(*_frame(i), timestamp=float(i))
    frames, points, span = buf.stats()
    assert (frames, points) == (3, 12)
    assert span == 2.0  # timestamps 7..9
    _, inten = buf.snapshot()
    assert set(inten.astype(int)) == {7, 8, 9}


def test_snapshot_pairs_points_with_intensity_after_eviction():
    buf = FrameAccumBuffer(max_frames=2, max_points=10_000)
    for i in range(5):
        buf.add(*_frame(i), timestamp=float(i))
    pts, inten = buf.snapshot()
    assert pts.shape == (8, 3)
    assert np.allclose(pts[:, 0], inten)      # pairing invariant
    assert np.allclose(inten[:4], 3.0)        # oldest retained frame first
    assert np.allclose(inten[4:], 4.0)


def test_point_ceiling_drops_frames_before_the_frame_limit():
    """The cap is a memory guard: it bites even when frames are under budget."""
    buf = FrameAccumBuffer(max_frames=1000, max_points=10)
    for i in range(5):
        buf.add(*_frame(i), timestamp=float(i))
    frames, points, _ = buf.stats()
    assert frames == 2 and points == 8  # a 3rd frame would exceed 10 points


def test_raising_max_frames_does_not_resurrect_dropped_frames():
    buf = FrameAccumBuffer(max_frames=2, max_points=10_000)
    for i in range(5):
        buf.add(*_frame(i), timestamp=float(i))
    buf.set_max_frames(10)
    assert buf.stats()[0] == 2
    buf.add(*_frame(5), timestamp=5.0)
    assert buf.stats()[0] == 3


def test_lowering_max_frames_trims_immediately():
    buf = FrameAccumBuffer(max_frames=10, max_points=10_000)
    for i in range(10):
        buf.add(*_frame(i), timestamp=float(i))
    buf.set_max_frames(2)
    frames, points, _ = buf.stats()
    assert (frames, points) == (2, 8)
    _, inten = buf.snapshot()  # the cache was invalidated by the trim
    assert set(inten.astype(int)) == {8, 9}


def test_missing_intensity_is_nan():
    buf = FrameAccumBuffer(max_frames=4, max_points=10_000)
    buf.add(np.zeros((3, 3)), None, timestamp=0.0)
    _, inten = buf.snapshot()
    assert inten.shape == (3,) and np.all(np.isnan(inten))


def test_empty_and_cleared_snapshots_are_well_shaped():
    buf = FrameAccumBuffer(max_frames=4, max_points=10_000)
    pts, inten = buf.snapshot()
    assert pts.shape == (0, 3) and inten.shape == (0,)
    buf.add(*_frame(1.0), timestamp=0.0)
    buf.clear()
    pts, inten = buf.snapshot()
    assert pts.shape == (0, 3) and inten.shape == (0,)
    assert buf.stats() == (0, 0, 0.0)


def test_raising_the_point_ceiling_lets_the_frame_limit_rule():
    """The knob the operator turns is frames; the cap must not silently win."""
    buf = FrameAccumBuffer(max_frames=5, max_points=10)
    for i in range(5):
        buf.add(*_frame(i), timestamp=float(i))
    assert buf.stats()[0] == 2  # ceiling is the binding limit
    buf.set_max_points(1_000)
    for i in range(5, 10):
        buf.add(*_frame(i), timestamp=float(i))
    frames, points, _ = buf.stats()
    assert (frames, points) == (5, 20)


def test_lowering_the_point_ceiling_trims_immediately():
    buf = FrameAccumBuffer(max_frames=100, max_points=1_000)
    for i in range(10):
        buf.add(*_frame(i), timestamp=float(i))
    buf.set_max_points(9)
    frames, points, _ = buf.stats()
    assert (frames, points) == (2, 8)
    assert buf.snapshot()[0].shape == (8, 3)
