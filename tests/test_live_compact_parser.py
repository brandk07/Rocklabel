"""Unit tests for the SICK Compact-format parser against synthetic packets.

Uses :func:`encode_compact_segment` (the module's own encoder) to build small,
fully-known telegrams and asserts the parser recovers the header, geometry, and
Cartesian points for both telegram versions and both azimuth modes.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from rocklabel.live.sources import compact_parser as cp


def _make_geometry(num_beams: int, num_layers: int):
    phi = np.linspace(-0.35, 0.35, num_layers).astype(np.float32)
    theta_start = np.full(num_layers, -0.5, np.float32)
    theta_stop = np.full(num_layers, 0.5, np.float32)
    rng = np.random.default_rng(7)
    ranges = rng.uniform(1.0, 20.0, size=(num_beams, num_layers))
    return phi, theta_start, theta_stop, ranges


def test_header_roundtrip():
    phi, ts, te, ranges = _make_geometry(10, 4)
    data = cp.encode_compact_segment(ranges, phi, ts, te, telegram_version=4, telegram_counter=99)
    hdr = cp.parse_header(data)
    assert hdr.command_id == 1
    assert hdr.telegram_version == 4
    assert hdr.telegram_counter == 99
    assert hdr.size_module0 > 0


def test_bad_start_of_frame_raises():
    phi, ts, te, ranges = _make_geometry(4, 2)
    data = bytearray(cp.encode_compact_segment(ranges, phi, ts, te))
    data[0] = 0x00  # corrupt the StartOfFrame magic
    with pytest.raises(cp.CompactParseError):
        cp.parse_header(bytes(data))


def test_truncated_buffer_raises():
    phi, ts, te, ranges = _make_geometry(4, 2)
    data = cp.encode_compact_segment(ranges, phi, ts, te)
    with pytest.raises(cp.CompactParseError):
        cp.parse_header(data[:16])


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("per_beam_azimuth", [False, True])
def test_point_count_and_ranges(version: int, per_beam_azimuth: bool):
    B, L = 64, 8
    phi, ts, te, ranges = _make_geometry(B, L)
    data = cp.encode_compact_segment(
        ranges, phi, ts, te, telegram_version=version, per_beam_azimuth=per_beam_azimuth
    )
    seg = cp.parse_segment(data)

    assert seg.header.telegram_version == version
    assert len(seg.modules) == 1
    assert seg.modules[0].num_beams == B
    assert seg.modules[0].num_layers == L
    assert seg.points.shape == (B * L, 3)

    # Range is recoverable from the Cartesian norm (within u16 quantization).
    recovered = np.linalg.norm(seg.points, axis=1)
    assert np.allclose(np.sort(recovered), np.sort(ranges.ravel()), atol=2e-3)


def test_cartesian_decoding_matches_spherical():
    """Verify the exact x/y/z formula, including the negated elevation."""
    B, L = 1, 3
    phi = np.array([0.0, 0.2, -0.2], np.float32)  # elevation per layer
    theta_start = np.array([0.1, 0.1, 0.1], np.float32)
    theta_stop = np.array([0.1, 0.1, 0.1], np.float32)  # single beam -> azimuth=start
    ranges = np.array([[5.0, 8.0, 3.0]])  # (B=1, L=3)
    data = cp.encode_compact_segment(ranges, phi, theta_start, theta_stop, telegram_version=4)
    seg = cp.parse_segment(data)

    # Expected per sick_scan_xd: elevation = -phi, az = theta_start.
    for layer in range(L):
        r = ranges[0, layer]
        az = theta_start[layer]
        el = -phi[layer]
        exp = np.array(
            [r * np.cos(az) * np.cos(el), r * np.sin(az) * np.cos(el), r * np.sin(el)]
        )
        # Points are ordered beam-major, layer-minor; B=1 so index == layer.
        got = seg.points[layer]
        assert np.allclose(got, exp, atol=3e-3), (layer, got, exp)


def test_rssi_present_and_absent():
    B, L = 16, 4
    phi, ts, te, ranges = _make_geometry(B, L)
    rssi = np.random.default_rng(3).uniform(10, 500, size=(B, L))

    seg_with = cp.parse_segment(cp.encode_compact_segment(ranges, phi, ts, te, rssi=rssi))
    assert seg_with.intensity is not None
    assert seg_with.intensity.shape == (B * L,)

    seg_without = cp.parse_segment(cp.encode_compact_segment(ranges, phi, ts, te))
    assert seg_without.intensity is None


def test_zero_range_points_are_dropped():
    """Beams with zero range (no return) must not become spurious origin points."""
    B, L = 4, 2
    phi, ts, te, ranges = _make_geometry(B, L)
    ranges[0, :] = 0.0  # first beam: no return on any layer
    data = cp.encode_compact_segment(ranges, phi, ts, te)
    seg = cp.parse_segment(data)
    assert seg.points.shape[0] == (B * L) - L  # dropped one beam across all layers
    assert np.all(np.linalg.norm(seg.points, axis=1) > 1e-6)


def test_imu_telegram_roundtrip():
    """A commandId==2 (IMU) telegram decodes accel/gyro/orientation and carries
    no points."""
    accel = (0.1, -0.2, 9.81)
    gyro = (0.01, 0.02, -0.5)
    quat = (0.7071068, 0.0, 0.0, 0.7071068)  # 90 deg yaw
    data = cp.encode_compact_imu(accel, gyro, quat, timestamp_us=123456)
    assert len(data) == cp.IMU_TELEGRAM_SIZE  # fixed 64-byte telegram

    seg = cp.parse_segment(data)
    assert seg.header.command_id == 2
    assert seg.points.shape == (0, 3)
    assert seg.imu is not None
    assert np.allclose(seg.imu.acceleration, accel, atol=1e-6)
    assert np.allclose(seg.imu.angular_velocity, gyro, atol=1e-6)
    assert np.allclose(seg.imu.orientation, quat, atol=1e-6)  # (w, x, y, z)
    assert seg.imu.timestamp_us == 123456


def test_imu_telegram_truncated_raises():
    data = cp.encode_compact_imu()[:30]
    with pytest.raises(cp.CompactParseError):
        cp.parse_segment(data)


def test_segment_to_xyz_convenience():
    B, L = 8, 4
    phi, ts, te, ranges = _make_geometry(B, L)
    pts, inten = cp.segment_to_xyz(cp.encode_compact_segment(ranges, phi, ts, te))
    assert pts.shape == (B * L, 3)
    assert inten is None
