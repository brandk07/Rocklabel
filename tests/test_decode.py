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
