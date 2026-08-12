"""Parser for the SICK **Compact** scan-data format (multiScan / picoScan).

This is a clean-room, pure-NumPy reimplementation of the byte layout used by
``sick_scan_xd`` (``compact_parser.cpp`` / ``compact_parser.h``).  It has no
dependency on ROS, Open3D, or sockets, so it can be reused or swapped for
``sick_scan_xd``'s own parser later.

Wire layout (all fields little-endian)
--------------------------------------
Telegram::

    [0 : 4)    StartOfFrame magic  = 0x02 0x02 0x02 0x02
    [4 : 32)   Compact data header (28 bytes):
                 commandId         u32   (1 = scan data, 2 = imu)
                 telegramCounter   u64
                 timeStampTransmit u64   (microseconds since epoch)
                 telegramVersion   u32   (3 or 4; 4 is the current default)
                 sizeModule0       u32   (size of the first module in bytes)
    [32 : ...) one or more modules, each `module_size` bytes:
                 <module metadata> <module measurement data>
                 the metadata's NextModuleSize gives the size of the following
                 module (0 => this is the last module).
    [end-4 : end)  u32 CRC (not validated here).

Module metadata (L = NumberOfLinesInModule)::

    SegmentCounter        u64
    FrameNumber           u64
    SenderId              u32
    NumberOfLinesInModule u32   (L, number of layers)
    NumberOfBeamsPerScan  u32   (B, beams per layer)
    NumberOfEchosPerBeam  u32   (E)
    TimeStampStart[L]     u64
    TimeStampStop[L]      u64
    Phi[L]                f32   (elevation per layer, radians)
    ThetaStart[L]         f32   (azimuth of first beam per layer, radians)
    ThetaStop[L]          f32   (azimuth of last beam per layer, radians)
    DistanceScalingFactor f32   (present only when telegramVersion == 4)
    NextModuleSize        u32
    Availability          u8
    DataContentEchos      u8    (bit0 = distance present, bit1 = RSSI present)
    DataContentBeams      u8    (bit0 = beam props present, bit1 = azimuth present)
    reserved              u8

Module measurement data — iterated **point-major, layer-minor**
(outer loop over B beams, inner loop over L layers), one record per (beam,
layer)::

    for echo in range(E):
        distance u16   (if DataContentEchos bit0)
        rssi     u16   (if DataContentEchos bit1)
    # telegramVersion 4: property then azimuth; version 3: azimuth then property
    property u8        (if DataContentBeams bit0)
    azimuth  u16       (if DataContentBeams bit1)

Decoding a point (echo *e*, beam *b*, layer *l*)::

    range_m   = DistanceScalingFactor * distance_u16 / 1000.0
    azimuth   = (azimuth_u16 - 16384.0) / 5215.0 + azimuth_offset   # if per-beam
                else ThetaStart[l] + b * (ThetaStop[l]-ThetaStart[l])/(B-1)
    elevation = -Phi[l]                                             # note the sign
    x = range_m * cos(azimuth) * cos(elevation)
    y = range_m * sin(azimuth) * cos(elevation)
    z = range_m * sin(elevation)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

START_OF_FRAME = b"\x02\x02\x02\x02"
HEADER_SIZE = 32  # SOF (4) + header fields (28)

# Azimuth quantization constants from sick_scan_xd compact_parser.cpp:
#   azimuth_rad = (azimuth_u16 - AZIMUTH_OFFSET_TICKS) / AZIMUTH_TICKS_PER_RAD
AZIMUTH_OFFSET_TICKS = 16384.0
AZIMUTH_TICKS_PER_RAD = 5215.0


class CompactParseError(ValueError):
    """Raised when a buffer is not a well-formed Compact telegram."""


@dataclass
class CompactHeader:
    """Decoded 32-byte Compact telegram header."""

    command_id: int
    telegram_counter: int
    timestamp_transmit_us: int
    telegram_version: int
    size_module0: int


@dataclass
class CompactImuData:
    """Decoded IMU telegram (commandId == 2).

    The multiScan streams IMU data over the *same* Compact UDP stream as scan
    data. Layout (after the 4-byte StartOfFrame, all little-endian)::

        commandId        u32  (= 2)
        telegramVersion  u32  (= 1)
        acceleration     3 x f32   (m/s^2, includes gravity)
        angular_velocity 3 x f32   (rad/s)
        orientation      4 x f32   (unit quaternion w, x, y, z; sensor->world)
        timeStampTransmit u64      (microseconds since epoch)

    i.e. a fixed 64-byte telegram (60 bytes + 4-byte CRC).
    """

    acceleration: np.ndarray  # (3,) float64
    angular_velocity: np.ndarray  # (3,) float64
    orientation: np.ndarray  # (4,) float64, (w, x, y, z)
    timestamp_us: int


@dataclass
class ModuleMeta:
    """Decoded metadata for a single Compact module."""

    segment_counter: int
    frame_number: int
    sender_id: int
    num_layers: int
    num_beams: int
    num_echos: int
    timestamp_start_us: np.ndarray  # (L,) uint64
    timestamp_stop_us: np.ndarray  # (L,) uint64
    phi: np.ndarray  # (L,) float32 elevation per layer (radians)
    theta_start: np.ndarray  # (L,) float32
    theta_stop: np.ndarray  # (L,) float32
    distance_scale: float
    next_module_size: int
    availability: int
    data_content_echos: int
    data_content_beams: int
    metadata_size: int  # number of bytes consumed by this metadata block

    @property
    def has_distance(self) -> bool:
        return bool(self.data_content_echos & 0x01)

    @property
    def has_rssi(self) -> bool:
        return bool(self.data_content_echos & 0x02)

    @property
    def has_beam_props(self) -> bool:
        return bool(self.data_content_beams & 0x01)

    @property
    def has_beam_azimuth(self) -> bool:
        return bool(self.data_content_beams & 0x02)


@dataclass
class CompactSegment:
    """A fully decoded Compact segment: header, per-module metadata, and the
    concatenated Cartesian points from every module (first echo only)."""

    header: CompactHeader
    modules: list[ModuleMeta] = field(default_factory=list)
    #: (N, 3) float64 Cartesian points (x, y, z) in meters, sensor frame.
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float64))
    #: (N,) float32 intensity/RSSI, or None if the sensor did not send RSSI.
    intensity: np.ndarray | None = None
    #: (N,) int32 layer index each point came from (useful for debugging).
    layer_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), np.int32))
    #: Decoded IMU data when the telegram was an IMU telegram (commandId == 2).
    imu: CompactImuData | None = None


# --------------------------------------------------------------------------- #
# Header + metadata parsing
# --------------------------------------------------------------------------- #
_HEADER_STRUCT = struct.Struct("<IQQII")  # commandId, counter, ts, version, sizeMod0
_IMU_STRUCT = struct.Struct("<10fQ")  # accel(3f) gyro(3f) quat wxyz(4f) timestamp
IMU_TELEGRAM_SIZE = 4 + 4 + 4 + _IMU_STRUCT.size + 4  # SOF+cmd+ver+payload+CRC = 64


def parse_header(data: bytes) -> CompactHeader:
    """Parse the telegram header (including the 4-byte StartOfFrame).

    Scan telegrams (commandId == 1) carry the full 28-byte header; IMU
    telegrams (commandId == 2) have a different fixed layout, so only
    ``command_id`` and ``telegram_version`` are meaningful for them here — the
    IMU payload itself is decoded by :func:`parse_segment`.
    """
    if len(data) < 12:
        raise CompactParseError(f"buffer too short for header: {len(data)} < 12")
    if data[:4] != START_OF_FRAME:
        raise CompactParseError(
            f"bad StartOfFrame: {data[:4].hex()} != {START_OF_FRAME.hex()}"
        )
    command_id = struct.unpack_from("<I", data, 4)[0]
    if command_id != 1:  # IMU (2) or unknown: only commandId + version are valid
        version = struct.unpack_from("<I", data, 8)[0]
        return CompactHeader(
            command_id=command_id,
            telegram_counter=0,
            timestamp_transmit_us=0,
            telegram_version=version,
            size_module0=0,
        )
    if len(data) < HEADER_SIZE:
        raise CompactParseError(
            f"buffer too short for scan header: {len(data)} < {HEADER_SIZE}"
        )
    command_id, counter, ts, version, size_mod0 = _HEADER_STRUCT.unpack_from(data, 4)
    return CompactHeader(
        command_id=command_id,
        telegram_counter=counter,
        timestamp_transmit_us=ts,
        telegram_version=version,
        size_module0=size_mod0,
    )


def _parse_imu(data: bytes) -> CompactImuData:
    """Decode the fixed-size IMU payload following the commandId/version words."""
    if len(data) < 12 + _IMU_STRUCT.size:
        raise CompactParseError(
            f"IMU telegram too short: {len(data)} < {12 + _IMU_STRUCT.size}"
        )
    vals = _IMU_STRUCT.unpack_from(data, 12)
    return CompactImuData(
        acceleration=np.array(vals[0:3], dtype=np.float64),
        angular_velocity=np.array(vals[3:6], dtype=np.float64),
        orientation=np.array(vals[6:10], dtype=np.float64),  # (w, x, y, z)
        timestamp_us=vals[10],
    )


def _parse_module_meta(buf: memoryview, version: int) -> ModuleMeta:
    """Parse one module's metadata starting at the beginning of ``buf``."""
    off = 0

    def u(fmt: str, size: int):
        nonlocal off
        val = struct.unpack_from(fmt, buf, off)[0]
        off += size
        return val

    segment_counter = u("<Q", 8)
    frame_number = u("<Q", 8)
    sender_id = u("<I", 4)
    num_layers = u("<I", 4)
    num_beams = u("<I", 4)
    num_echos = u("<I", 4)

    if num_layers < 1 or num_layers > 64:
        raise CompactParseError(f"implausible NumberOfLinesInModule={num_layers}")

    def read_array(fmt_char: str, size: int, count: int) -> np.ndarray:
        nonlocal off
        arr = np.frombuffer(buf, dtype=f"<{fmt_char}", count=count, offset=off).copy()
        off += size * count
        return arr

    ts_start = read_array("u8", 8, num_layers)
    ts_stop = read_array("u8", 8, num_layers)
    phi = read_array("f4", 4, num_layers)
    theta_start = read_array("f4", 4, num_layers)
    theta_stop = read_array("f4", 4, num_layers)

    if version == 4:
        distance_scale = float(u("<f", 4))
    elif version == 3:
        distance_scale = 1.0
    else:
        raise CompactParseError(f"unsupported telegramVersion={version}")

    next_module_size = u("<I", 4)
    availability = u("<B", 1)
    data_content_echos = u("<B", 1)
    data_content_beams = u("<B", 1)
    _reserved = u("<B", 1)

    return ModuleMeta(
        segment_counter=segment_counter,
        frame_number=frame_number,
        sender_id=sender_id,
        num_layers=num_layers,
        num_beams=num_beams,
        num_echos=num_echos,
        timestamp_start_us=ts_start,
        timestamp_stop_us=ts_stop,
        phi=phi,
        theta_start=theta_start,
        theta_stop=theta_stop,
        distance_scale=distance_scale,
        next_module_size=next_module_size,
        availability=availability,
        data_content_echos=data_content_echos,
        data_content_beams=data_content_beams,
        metadata_size=off,
    )


def _measurement_dtype(meta: ModuleMeta, version: int) -> np.dtype:
    """Build the structured dtype for one (beam, layer) measurement record.

    Fields are laid out exactly in the on-wire order and tightly packed
    (``align=False``), so ``np.frombuffer`` reads the whole block in one shot.
    """
    fields: list[tuple[str, str]] = []
    for e in range(meta.num_echos):
        if meta.has_distance:
            fields.append((f"dist{e}", "<u2"))
        if meta.has_rssi:
            fields.append((f"rssi{e}", "<u2"))
    azim_field = ("azimuth", "<u2")
    prop_field = ("prop", "u1")
    if version == 4:  # property then azimuth
        if meta.has_beam_props:
            fields.append(prop_field)
        if meta.has_beam_azimuth:
            fields.append(azim_field)
    else:  # version 3: azimuth then property
        if meta.has_beam_azimuth:
            fields.append(azim_field)
        if meta.has_beam_props:
            fields.append(prop_field)
    return np.dtype(fields)  # unaligned / packed by default


def _decode_module_points(
    meta: ModuleMeta,
    payload: memoryview,
    version: int,
    azimuth_offset: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Decode a module's measurement block into Cartesian points (first echo).

    Returns ``(points (M,3), intensity (M,) | None, layer_ids (M,))`` where
    ``M`` is the number of beams * layers with a positive range.
    """
    B, L, E = meta.num_beams, meta.num_layers, meta.num_echos
    if B < 1 or L < 1 or E < 1:
        empty = np.empty((0, 3), np.float64)
        return empty, None, np.empty((0,), np.int32)

    rec = _measurement_dtype(meta, version)
    need = rec.itemsize * B * L
    if len(payload) < need:
        raise CompactParseError(
            f"measurement block too short: have {len(payload)}, need {need}"
        )

    # (B, L) structured array: outer beam, inner layer (matches wire order).
    table = np.frombuffer(payload, dtype=rec, count=B * L).reshape(B, L)

    # Range from the first echo. Missing distance field => all-zero range.
    if meta.has_distance:
        dist_u16 = table["dist0"].astype(np.float64)
    else:
        dist_u16 = np.zeros((B, L), np.float64)
    range_m = dist_u16 * (meta.distance_scale / 1000.0)

    # Azimuth: per-beam if provided, else linearly interpolated per layer.
    if meta.has_beam_azimuth:
        azimuth = (table["azimuth"].astype(np.float64) - AZIMUTH_OFFSET_TICKS) / (
            AZIMUTH_TICKS_PER_RAD
        ) + azimuth_offset
    else:
        beam_idx = np.arange(B, dtype=np.float64)[:, None]  # (B,1)
        denom = max(1, B - 1)
        delta = (meta.theta_stop - meta.theta_start) / denom  # (L,)
        azimuth = meta.theta_start[None, :] + beam_idx * delta[None, :] + azimuth_offset

    # Elevation per layer (negated, matching sick_scan_xd), broadcast over beams.
    elevation = -meta.phi.astype(np.float64)[None, :]  # (1, L)
    cos_el = np.cos(elevation)
    sin_el = np.sin(elevation)
    cos_az = np.cos(azimuth)
    sin_az = np.sin(azimuth)

    x = range_m * cos_az * cos_el
    y = range_m * sin_az * cos_el
    z = range_m * sin_el

    layer_ids = np.broadcast_to(np.arange(L, dtype=np.int32), (B, L))

    # Keep only real returns (positive range).
    valid = range_m > 1e-6
    pts = np.column_stack((x[valid], y[valid], z[valid])).astype(np.float64)
    layers = layer_ids[valid].astype(np.int32)

    intensity: np.ndarray | None = None
    if meta.has_rssi:
        intensity = table["rssi0"].astype(np.float32)[valid]

    return pts, intensity, layers


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse_segment(data: bytes, azimuth_offset: float = 0.0) -> CompactSegment:
    """Parse a full Compact telegram into a :class:`CompactSegment`.

    Args:
        data: raw bytes of one UDP telegram (starting at the StartOfFrame).
        azimuth_offset: optional constant azimuth offset (radians) applied to
            every point, mirroring sick_scan_xd's ``azimuth_offset``.

    Returns:
        The decoded segment. Only scan-data telegrams (commandId == 1) carry
        points; IMU telegrams (commandId == 2) return an empty point set.

    Raises:
        CompactParseError: if the buffer is malformed / truncated.
    """
    header = parse_header(data)
    seg = CompactSegment(header=header)

    if header.command_id == 2:  # IMU telegram: decode orientation, no points
        seg.imu = _parse_imu(data)
        seg.header.timestamp_transmit_us = seg.imu.timestamp_us
        return seg
    if header.command_id != 1:  # unknown telegram type -> nothing to decode
        return seg

    buf = memoryview(data)
    module_offset = HEADER_SIZE
    module_size = header.size_module0
    all_points: list[np.ndarray] = []
    all_intensity: list[np.ndarray] = []
    all_layers: list[np.ndarray] = []
    any_missing_rssi = False

    while module_size > 0:
        if module_offset + module_size > len(data):
            raise CompactParseError(
                f"module at offset {module_offset} claims {module_size} bytes but "
                f"only {len(data) - module_offset} remain"
            )
        module_buf = buf[module_offset : module_offset + module_size]
        meta = _parse_module_meta(module_buf, header.telegram_version)
        seg.modules.append(meta)

        meas = module_buf[meta.metadata_size :]
        pts, inten, layers = _decode_module_points(
            meta, meas, header.telegram_version, azimuth_offset
        )
        all_points.append(pts)
        all_layers.append(layers)
        if inten is not None:
            all_intensity.append(inten)
        else:
            any_missing_rssi = True

        module_offset += module_size
        module_size = meta.next_module_size

    if all_points:
        seg.points = np.concatenate(all_points, axis=0)
        seg.layer_ids = np.concatenate(all_layers, axis=0)
        if all_intensity and not any_missing_rssi:
            seg.intensity = np.concatenate(all_intensity, axis=0)
    return seg


def segment_to_xyz(
    data: bytes, azimuth_offset: float = 0.0
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convenience: parse and return just ``(points (N,3), intensity | None)``."""
    seg = parse_segment(data, azimuth_offset=azimuth_offset)
    return seg.points, seg.intensity


# --------------------------------------------------------------------------- #
# Encoder — for tests and local UDP loopback simulation ONLY.
# --------------------------------------------------------------------------- #
def encode_compact_segment(
    ranges_m: np.ndarray,
    phi: np.ndarray,
    theta_start: np.ndarray,
    theta_stop: np.ndarray,
    *,
    rssi: np.ndarray | None = None,
    telegram_version: int = 4,
    distance_scale: float = 1.0,
    telegram_counter: int = 1,
    per_beam_azimuth: bool = False,
) -> bytes:
    """Encode a single-module, single-echo Compact telegram.

    This is the inverse of :func:`parse_segment` for the common multiScan case
    (one module, one echo). It exists so tests can round-trip a known point set
    and so the UDP path can be exercised over loopback without hardware.

    Args:
        ranges_m: ``(B, L)`` array of ranges in meters (beam-major, layer-minor).
        phi: ``(L,)`` elevation per layer (radians).
        theta_start / theta_stop: ``(L,)`` azimuth bounds per layer (radians).
        rssi: optional ``(B, L)`` intensity; if given, RSSI is encoded.
        telegram_version: 3 or 4.
        distance_scale: DistanceScalingFactor (mm per distance tick / scale).
        per_beam_azimuth: if True, encode a per-beam azimuth field.

    Returns:
        Bytes of a complete telegram (SOF + header + module + CRC placeholder).
    """
    ranges_m = np.asarray(ranges_m, dtype=np.float64)
    B, L = ranges_m.shape
    phi = np.asarray(phi, np.float32)
    theta_start = np.asarray(theta_start, np.float32)
    theta_stop = np.asarray(theta_stop, np.float32)

    has_rssi = rssi is not None
    # --- module metadata ---
    meta = bytearray()
    meta += struct.pack("<Q", 0)  # SegmentCounter
    meta += struct.pack("<Q", 0)  # FrameNumber
    meta += struct.pack("<I", 12345)  # SenderId
    meta += struct.pack("<I", L)  # NumberOfLinesInModule
    meta += struct.pack("<I", B)  # NumberOfBeamsPerScan
    meta += struct.pack("<I", 1)  # NumberOfEchosPerBeam
    meta += np.zeros(L, "<u8").tobytes()  # TimeStampStart
    meta += (np.ones(L, "<u8") * 1000).tobytes()  # TimeStampStop
    meta += phi.astype("<f4").tobytes()
    meta += theta_start.astype("<f4").tobytes()
    meta += theta_stop.astype("<f4").tobytes()
    if telegram_version == 4:
        meta += struct.pack("<f", distance_scale)
    meta += struct.pack("<I", 0)  # NextModuleSize (0 => last module)
    meta += struct.pack("<B", 0)  # Availability
    meta += struct.pack("<B", 0x03 if has_rssi else 0x01)  # DataContentEchos
    meta += struct.pack("<B", 0x02 if per_beam_azimuth else 0x00)  # DataContentBeams
    meta += struct.pack("<B", 0)  # reserved

    # --- module measurement data (beam-major, layer-minor) ---
    dist_ticks = np.rint(ranges_m * 1000.0 / distance_scale).astype("<u2")
    fields: list[tuple[str, str]] = [("dist0", "<u2")]
    if has_rssi:
        fields.append(("rssi0", "<u2"))
    if per_beam_azimuth:
        fields.append(("azimuth", "<u2"))
    rec = np.dtype(fields)
    table = np.zeros((B, L), dtype=rec)
    table["dist0"] = dist_ticks
    if has_rssi:
        table["rssi0"] = np.asarray(rssi, np.float64).astype("<u2")
    if per_beam_azimuth:
        beam_idx = np.arange(B, dtype=np.float64)[:, None]
        delta = (theta_stop - theta_start) / max(1, B - 1)
        az = theta_start[None, :].astype(np.float64) + beam_idx * delta[None, :]
        table["azimuth"] = np.rint(az * AZIMUTH_TICKS_PER_RAD + AZIMUTH_OFFSET_TICKS).astype("<u2")
    measurement = table.tobytes()

    module = bytes(meta) + measurement

    # --- header ---
    header = bytearray()
    header += START_OF_FRAME
    header += struct.pack("<I", 1)  # commandId = scan data
    header += struct.pack("<Q", telegram_counter)
    header += struct.pack("<Q", 0)  # timeStampTransmit
    header += struct.pack("<I", telegram_version)
    header += struct.pack("<I", len(module))  # sizeModule0

    return bytes(header) + module + struct.pack("<I", 0)  # trailing CRC placeholder


def encode_compact_imu(
    acceleration: tuple[float, float, float] = (0.0, 0.0, 9.81),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    timestamp_us: int = 0,
) -> bytes:
    """Encode a 64-byte Compact IMU telegram (commandId == 2), for tests and
    local loopback simulation."""
    payload = struct.pack("<I", 2) + struct.pack("<I", 1)
    payload += _IMU_STRUCT.pack(*acceleration, *angular_velocity, *orientation_wxyz, timestamp_us)
    return START_OF_FRAME + payload + struct.pack("<I", 0)  # CRC placeholder
