"""PointCloud2 decode round-trips: endianness, intensity variants, NaN drop, row padding."""

from types import SimpleNamespace

import numpy as np
import pytest

from rocklabel.mcap_io import decode_pointcloud2


def _field(name, offset, datatype, count=1):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=count)


def _msg(data, fields, width, height=1, point_step=None, row_step=None, is_bigendian=False):
    return SimpleNamespace(
        data=data, fields=fields, width=width, height=height,
        point_step=point_step, row_step=row_step if row_step is not None else point_step * width,
        is_bigendian=is_bigendian, is_dense=True,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0), frame_id="lidar_link"),
    )


def test_float32_with_uint16_intensity_roundtrip():
    xyz = np.array([[1.0, 2.0, 3.0], [-4.5, 0.25, 9.0]], np.float32)
    inten = np.array([0, 65535], np.uint16)
    rows = np.zeros(2, np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<u2")]))
    rows["x"], rows["y"], rows["z"] = xyz.T
    rows["intensity"] = inten
    msg = _msg(rows.tobytes(), [
        _field("x", 0, 7), _field("y", 4, 7), _field("z", 8, 7), _field("intensity", 12, 4),
    ], width=2, point_step=14)
    out_xyz, out_inten, available = decode_pointcloud2(msg)
    np.testing.assert_allclose(out_xyz, xyz)
    np.testing.assert_allclose(out_inten, [0.0, 1.0])  # uint16 normalized by dtype max
    assert available


def test_float32_intensity_is_passed_through_for_the_stream_to_scale():
    """A float field carries no dtype max to divide by, so decode leaves it
    alone and the scan stream probes instead - see
    test_pipeline_normalizes_raw_counts_in_a_float_intensity_field."""
    rows = np.zeros(2, np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")]))
    rows["intensity"] = [0.0, 65535.0]
    msg = _msg(rows.tobytes(), [
        _field("x", 0, 7), _field("y", 4, 7), _field("z", 8, 7), _field("intensity", 12, 7),
    ], width=2, point_step=16)
    _xyz, out_inten, available = decode_pointcloud2(msg)
    np.testing.assert_allclose(out_inten, [0.0, 65535.0])
    assert available


@pytest.mark.parametrize("peak, expect", [
    (0.0, 1.0), (1.0, 1.0), (1.5, 1.0),           # already normalized
    (200.0, 1 / 255.0), (255.0, 1 / 255.0),       # 8-bit
    (4000.0, 1 / 65535.0), (65535.0, 1 / 65535.0),  # raw u16 counts
])
def test_intensity_scale_ladder(peak, expect):
    from rocklabel.mcap_io import intensity_scale_for_peak

    assert intensity_scale_for_peak(peak) == pytest.approx(expect)


def test_bigendian_decode():
    xyz = np.array([[1.5, -2.5, 3.25]], np.float32)
    rows = np.zeros(1, np.dtype([("x", ">f4"), ("y", ">f4"), ("z", ">f4")]))
    rows["x"], rows["y"], rows["z"] = xyz.T
    msg = _msg(rows.tobytes(), [_field("x", 0, 7), _field("y", 4, 7), _field("z", 8, 7)],
               width=1, point_step=12, is_bigendian=True)
    out_xyz, out_inten, available = decode_pointcloud2(msg)
    np.testing.assert_allclose(out_xyz, xyz)
    assert not available
    np.testing.assert_array_equal(out_inten, [0.0])


def test_nonfinite_points_dropped():
    rows = np.zeros(3, np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]))
    rows["x"] = [1.0, np.nan, 3.0]
    rows["y"] = [0.0, 0.0, np.inf]
    rows["z"] = [0.0, 0.0, 0.0]
    msg = _msg(rows.tobytes(), [_field("x", 0, 7), _field("y", 4, 7), _field("z", 8, 7)],
               width=3, point_step=12)
    out_xyz, _, _ = decode_pointcloud2(msg)
    np.testing.assert_allclose(out_xyz, [[1.0, 0.0, 0.0]])


def test_organized_cloud_with_row_padding():
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    rows = np.zeros((2, 2), dt)
    rows["x"] = [[1, 2], [3, 4]]
    pad = b"\xff" * 8
    data = rows[0].tobytes() + pad + rows[1].tobytes() + pad
    msg = _msg(data, [_field("x", 0, 7), _field("y", 4, 7), _field("z", 8, 7)],
               width=2, height=2, point_step=12, row_step=32)
    out_xyz, _, _ = decode_pointcloud2(msg)
    np.testing.assert_allclose(out_xyz[:, 0], [1, 2, 3, 4])


def test_uint8_reflectivity_detected():
    rows = np.zeros(1, np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("reflectivity", "u1")]))
    rows["reflectivity"] = 255
    msg = _msg(rows.tobytes(), [
        _field("x", 0, 7), _field("y", 4, 7), _field("z", 8, 7), _field("reflectivity", 12, 2),
    ], width=1, point_step=13)
    _, out_inten, available = decode_pointcloud2(msg)
    assert available
    np.testing.assert_allclose(out_inten, [1.0])


# -- reflectivity color scales ------------------------------------------------

def test_fixed_reflectivity_scale_is_absolute():
    """Identical materials must keep identical colors between frames, so the
    fixed scale is a plain divide by full scale and ignores the frame."""
    from rocklabel.live.colormap import reflectivity_values

    v = reflectivity_values(np.array([0.0, 32767.5, 65535.0]))
    np.testing.assert_allclose(v, [0.0, 0.5, 1.0])
    # a retroreflector in view must not move anything else
    with_retro = reflectivity_values(np.array([100.0, 200.0, 65535.0]))
    without = reflectivity_values(np.array([100.0, 200.0]))
    np.testing.assert_allclose(with_retro[:2], without)


def test_stretched_reflectivity_expands_a_narrow_band():
    """Real arena returns occupy a narrow slice of full scale; the stretch mode
    exists to spread that slice over the whole colormap."""
    from rocklabel.live.colormap import reflectivity_values

    band = np.linspace(0.26, 0.82, 500) * 65535.0
    fixed = reflectivity_values(band)
    stretched = reflectivity_values(band, stretch=True, pct=(5.0, 95.0))
    assert fixed.max() - fixed.min() < 0.60          # squeezed into part of the ramp
    assert stretched.max() - stretched.min() > 0.99  # uses all of it


def test_reflectivity_marks_missing_data_mid_gray():
    from rocklabel.live.colormap import reflectivity_values

    v = reflectivity_values(np.array([np.nan, 65535.0, np.inf]))
    assert v[0] == 0.5 and v[2] == 0.5 and v[1] == 1.0
    assert np.all(reflectivity_values(np.full(4, np.nan)) == 0.5)


# -- the manual contrast window ----------------------------------------------

def test_window_saturates_outside_and_spreads_inside():
    """The point of the window: returns past either end take the end color
    instead of compressing everything between them into one flat shade."""
    from rocklabel.live.colormap import reflectivity_values

    inten = np.array([0.0, 0.40, 0.475, 0.50, 0.525, 0.60, 1.0]) * 65535.0
    v = reflectivity_values(inten, limits=(0.45, 0.55))
    np.testing.assert_allclose(v, [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    # the same band on the full scale is the flat blob the window fixes
    flat = reflectivity_values(inten[2:5])
    assert flat.max() - flat.min() < 0.06


def test_window_is_absolute_so_frames_stay_comparable():
    """Unlike the stretch mode, the window does not move when the scene does:
    a rock leaving the view must not repaint the ground behind it."""
    from rocklabel.live.colormap import reflectivity_values

    window = (0.3, 0.7)
    ground = np.array([0.45, 0.50, 0.55]) * 65535.0
    with_rock = np.concatenate([ground, [0.95 * 65535.0]])
    np.testing.assert_allclose(
        reflectivity_values(ground, limits=window),
        reflectivity_values(with_rock, limits=window)[:3])
    # the stretch mode is exactly what does not hold this property
    assert not np.allclose(
        reflectivity_values(ground, stretch=True),
        reflectivity_values(with_rock, stretch=True)[:3])


def test_autofit_reproduces_the_stretch_it_replaces():
    """Auto-fit hands back the window the stretch mode picks per frame, so the
    first click changes nothing on screen — it only stops it moving."""
    from rocklabel.live.colormap import percentile_range, reflectivity_values

    band = np.linspace(0.26, 0.82, 500) * 65535.0
    window = percentile_range(band, (5.0, 95.0))
    np.testing.assert_allclose(
        reflectivity_values(band, limits=window),
        reflectivity_values(band, stretch=True, pct=(5.0, 95.0)), atol=1e-6)
    assert percentile_range(np.full(4, np.nan)) is None


def test_window_ends_cannot_cross_and_the_dragged_one_wins():
    """Two sliders, one window: the end being moved lands where it was put and
    the other yields, rather than the handle snapping back under the cursor."""
    from rocklabel.live.colormap import (MIN_RANGE_SPAN, clamp_range,
                                         move_range_end)

    lo, hi = move_range_end((0.20, 0.60), "lo", 0.90)
    assert lo == pytest.approx(0.90) and hi > lo
    lo, hi = move_range_end((0.20, 0.60), "hi", 0.10)
    assert hi == pytest.approx(0.10) and lo < hi
    # and a window can never invert or collapse, however it is asked to
    for asked in [(0.8, 0.2), (0.5, 0.5), (-3.0, 7.0), (1.0, 1.0)]:
        lo, hi = clamp_range(*asked)
        assert 0.0 <= lo < hi <= 1.0 and hi - lo >= MIN_RANGE_SPAN - 1e-12
