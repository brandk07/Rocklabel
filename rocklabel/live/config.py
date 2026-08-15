"""Configuration for the lidarrig test rig.

A single set of dataclasses describes *everything* that is tunable: the data
source, the sensor network parameters, the heightmap grid geometry, the Kalman
fusion parameters, outlier rejection, and the display.

Config can be built from defaults, loaded from a YAML file, or overridden by
CLI flags (see :mod:`lidarrig.__main__`).  This module has **no** dependency on
Open3D or any other visualization library so it can be imported by a headless
ROS 2 node later on.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, get_type_hints

try:  # PyYAML is a hard runtime dep, but keep the import defensive for tooling.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@dataclass
class SourceConfig:
    """Which data source to run and how fast the simulator should push data."""

    #: "sim" (default, no hardware) or "udp" (live SICK multiScan136).
    kind: str = "sim"

    # --- SimulatedSource parameters ---
    #: Target throughput in points/second (the prompt asks for ~100k-300k).
    sim_points_per_sec: int = 200_000
    #: Number of points emitted per batch.
    sim_batch_size: int = 4_000
    #: Std-dev of Gaussian range noise applied to the true surface height (m).
    sim_noise_std: float = 0.02
    #: Fraction of points that become gross outliers each batch (0..1).
    sim_outlier_fraction: float = 0.01
    #: Vertical magnitude (m) of a gross outlier spike.
    sim_outlier_magnitude: float = 2.0
    #: Random seed for reproducible terrain / noise (None = nondeterministic).
    sim_seed: int | None = 42

    # --- MultiScanUDPSource parameters ---
    #: Default multiScan136 sensor IP (verify against your SOPAS config).
    sensor_ip: str = "10.11.10.3"
    #: UDP port the sensor streams Compact data to. sick_scan_xd default: 2115.
    udp_port: int = 2115
    #: Local interface/host to bind the receiving socket to ("" = all interfaces).
    udp_bind_host: str = ""
    #: Max decoded batches buffered before the oldest is dropped (bounded latency).
    udp_queue_size: int = 8

    # --- wired-link setup (used by the dashboard's repair suggestion) --------
    # Plugged straight into a laptop there is no DHCP on the cable, so these
    # addresses have to be set by hand. See :mod:`rocklabel.dashboard.netfix`.
    #: Wired NIC the sensor is plugged into ("" = autodetect the only candidate).
    wired_iface: str = ""
    #: This host on the sensor's subnet, CIDR — reaches SOPAS / the web UI.
    host_addr: str = "10.11.10.5/24"
    #: The address the sensor was configured to stream Compact data to, CIDR.
    #: On a different /24 than the sensor, so it must be added explicitly.
    udp_dest_addr: str = "10.11.11.8/24"
    #: The sensor's configured gateway, CIDR — it routes its UDP packets there,
    #: so this host has to answer to that address as well.
    gateway_addr: str = "10.11.10.1/24"


@dataclass
class GridConfig:
    """2.5D heightmap grid geometry (a regular lattice in the sensor x/y plane)."""

    #: World-space origin (min corner) of the grid in meters: (x0, y0).
    origin: tuple[float, float] = (-10.0, -10.0)
    #: Physical size of the grid in meters: (size_x, size_y).
    extent: tuple[float, float] = (20.0, 20.0)
    #: Edge length of a square cell in meters. Smaller = finer surface.
    cell_size: float = 0.05


@dataclass
class SlamConfig:
    """Scan-to-map ICP odometry so the sensor can *move* through the space.

    Uses the IMU as the rotation prior and registers short windows of points
    against a sparse voxel map to solve translation (+ yaw drift). Engages only
    when IMU orientation is available (i.e. the live sensor); the simulator is
    unaffected. See :mod:`lidarrig.slam`.
    """

    #: Master toggle. When False, only IMU rotation compensation runs.
    enabled: bool = True
    #: Voxel edge length (m) of the registration map and downsampling.
    voxel_size: float = 0.15
    #: Seconds of data pooled per registration window (~1-2 sensor revolutions).
    window_sec: float = 0.10
    #: ICP iterations per window (correspondence radius shrinks each iteration).
    icp_iterations: int = 5
    #: Starting correspondence radius (m) — also bounds how far the sensor may
    #: move between windows and still re-lock. Keep moves at walking pace.
    max_corr_dist: float = 0.45
    #: Minimum matched correspondences for a window to update the pose.
    min_matches: int = 100
    #: ICP rotation freedom: "yaw" (default; IMU pins roll/pitch via gravity),
    #: "full" (SE(3)), or "none" (translation only).
    rotation_mode: str = "yaw"
    #: 3D range band (m, sensor frame) of points used for registration.
    #: Walls/ceiling ARE wanted here — they constrain x/y translation.
    reg_range_min: float = 0.5
    reg_range_max: float = 12.0
    #: Freeze a map voxel's centroid after this many hits (map stability).
    map_max_hits: int = 30
    #: Hard cap on map voxels (safety; ~1M voxels ≈ a large building).
    map_capacity: int = 1_000_000


@dataclass
class KalmanConfig:
    """Per-cell 1D Kalman filter parameters for height fusion."""

    #: Sensor measurement variance R (m^2). Larger => trust each point less.
    sensor_variance: float = 0.0009  # ~ (0.03 m)^2
    #: Process noise Q (m^2) added to cell variance each update so cells stay
    #: responsive when the true surface changes over time.
    process_noise: float = 1e-5
    #: Initial per-cell variance P0 (m^2) for a never-before-seen cell.
    initial_variance: float = 1.0
    #: Cells whose variance stays above this are hidden in the mesh (m^2).
    max_render_variance: float = 0.05
    #: Skip triangulating any quad whose corner heights span more than this (m).
    #: Prevents walls / column-mixing artifacts from rendering as huge spikes.
    #: 0 disables the guard.
    max_edge_dz: float = 1.0


@dataclass
class LevelConfig:
    """Gravity levelling of the world frame (see :mod:`rocklabel.live.leveling`).

    The world frame is otherwise just the sensor frame at startup, so a LiDAR
    bolted to a slanted mast tilts the entire map, the crop band, the
    heightmap, and the model's neighborhoods with it. Levelling measures that
    mount angle once — from the gravity-referenced IMU quaternion and/or a
    RANSAC fit of the floor — and rotates it out of every batch.
    """

    #: Estimation strategy:
    #: ``"auto"``   — IMU seed (instant) refined by a ground-plane fit (default);
    #: ``"imu"``    — IMU gravity reference only, no ground fit;
    #: ``"ground"`` — ground-plane fit only (works without an IMU, e.g. replays);
    #: ``"manual"`` — use ``mount_roll_deg``/``mount_pitch_deg`` verbatim;
    #: ``"off"``    — legacy behaviour: world frame = sensor frame at startup.
    mode: str = "auto"
    #: Mount angles (deg) for ``mode="manual"``, in the IMU's roll/pitch
    #: convention: a sensor pitched nose-up by θ about its +y axis is
    #: ``mount_pitch_deg = θ``.
    mount_roll_deg: float = 0.0
    mount_pitch_deg: float = 0.0
    #: Seconds of data pooled before the ground fit runs.
    calib_sec: float = 3.0
    #: Stop waiting for a usable fit after this long and keep the IMU seed.
    calib_timeout_sec: float = 15.0
    #: 3D range band (m) of points fed to the ground fit. The inner radius
    #: skips the robot's own frame; the outer keeps the floor dense and flat.
    range_min: float = 0.6
    range_max: float = 6.0
    #: Ignore points above this height (m) while fitting — the floor is below
    #: the sensor, and the margin absorbs an unseeded (still tilted) frame.
    ceiling_margin: float = 0.2
    #: RANSAC inlier distance (m) and hypothesis count.
    plane_thresh: float = 0.04
    ransac_iters: int = 128
    #: Reject planes tilted more than this from +z (keeps the fit off walls).
    max_tilt_deg: float = 50.0
    #: Acceptance gates for a fit. The fraction is deliberately loose: with an
    #: unseeded (still tilted) frame the pooled set is mostly walls, so the
    #: floor is a minority of it — the tilt gate and the below-the-sensor check
    #: are what actually keep the fit honest. The absolute count is what rules
    #: out fitting noise.
    min_inlier_frac: float = 0.15
    min_inliers: int = 1_000
    #: Minimum pooled points before fitting, and the cap the pool is trimmed
    #: to (oldest first) while a fit keeps failing.
    min_points: int = 4_000
    max_points: int = 400_000
    #: Keep every Nth point per batch while pooling.
    sample_stride: int = 2


@dataclass
class CropConfig:
    """Region-of-interest crop applied to every batch *before* fusion.

    Essential with a real 360° sensor indoors: the multiScan sees walls,
    ceiling, and fixtures in every direction, but a 2.5D heightmap can only
    represent one surface height per (x, y) column.  Crop to the z-band that
    contains the surface you care about (e.g. the floor) and, optionally, a
    max range, so wall/ceiling returns never reach the Kalman grid.

    Heights are in the levelled world frame, measured from the sensor's
    startup height: a sensor 1 m above the floor puts the floor at z ~ -1.0.
    With ``floor_relative`` the band is anchored to the *measured* floor plane
    instead, which makes one setting work across rigs of different heights.
    """

    #: Master toggle.
    enabled: bool = True
    #: Keep only points with z >= z_min (m).
    z_min: float = -3.0
    #: Keep only points with z <= z_max (m).
    z_max: float = 3.0
    #: Interpret z_min/z_max relative to the levelled floor plane (CLI:
    #: --floor-band). Falls back to absolute until levelling has measured a
    #: floor height.
    floor_relative: bool = False
    #: Keep only points closer than this 3D range from the sensor (m).
    #: 0 disables the gate.
    range_max: float = 0.0
    #: Drop points closer than this 3D range (m) — kills self-hits/near dust.
    #: 0 disables.
    range_min: float = 0.0


@dataclass
class OutlierConfig:
    """Statistical outlier rejection applied to each batch before fusion.

    Uses a fast O(N) per-column z-deviation test (see :mod:`lidarrig.filters`):
    points are binned into ``cell_size`` columns and rejected when their height
    deviates from the column mean by more than ``std_ratio`` standard deviations.
    """

    #: Master toggle. When False, points are integrated as-is.
    enabled: bool = True
    #: Side length (m) of the binning column for computing local z statistics.
    #: Make it a few times the grid cell size so each column has enough points.
    cell_size: float = 0.5
    #: A point is rejected if |z - column_mean| > std_ratio * column_std.
    std_ratio: float = 2.5
    #: Floor on per-column std so flat, low-noise columns don't over-reject (m).
    min_std: float = 0.01
    #: Minimum points in a column before the deviation test is applied.
    min_cell_points: int = 4
    #: Skip filtering entirely when a batch has fewer than this many points.
    min_points: int = 32


@dataclass
class MotionConfig:
    """IMU-based motion compensation (multiScan onboard IMU over Compact UDP).

    When the sensor itself rotates (e.g. mounted on a chair you spin to sweep
    the room), incoming points must be rotated into a fixed world frame using
    the sensor's orientation quaternion, or the scene smears into rings.
    """

    #: Use IMU orientation telegrams to de-rotate incoming points. Ignored for
    #: sources that provide no orientation (e.g. the simulator).
    use_imu: bool = True
    #: Apply only the yaw component of the rotation (pivot-about-z rigs like a
    #: chair). Filters out small tilt/wobble noise; leave False for a full 3D
    #: correction.
    yaw_only: bool = False


@dataclass
class RecordConfig:
    """MCAP recording of the raw ingest stream (toggle live with the S key).

    Each recorded frame holds the sensor-frame points, per-point reflectivity,
    the IMU quaternion, and the live SLAM/IMU world pose; the full app config
    is embedded as file metadata. Replay with ``python -m lidarrig --play
    <file.mcap>`` reconstructs the session exactly as it looked live.
    """

    #: Directory where auto-named recordings (lidar_YYYYmmdd_HHMMSS.mcap) land.
    dir: str = "recordings"
    #: Start recording immediately on launch (CLI: --record).
    autostart: bool = False
    #: Explicit output path (empty = auto-named inside ``dir``).
    path: str = ""


@dataclass
class DisplayConfig:
    """Visualization / render-loop options (only read by the viz layer)."""

    #: Target render frame rate (integration loop is decoupled from this).
    fps: float = 20.0
    #: Point size for the raw point cloud.
    point_size: float = 3.0
    #: How many recent points to keep in the rolling raw-point buffer.
    raw_buffer_size: int = 60_000
    #: Show the raw point cloud on startup.
    show_points: bool = True
    #: Show the reconstructed mesh on startup.
    show_mesh: bool = True
    #: How many recent frames the accumulated world-frame cloud keeps (toggle:
    #: C, adjustable live in the View panel). Retention is counted in frames so
    #: it means the same thing whatever a source calls a frame: the SICK
    #: streams ~225 small telegrams/s, a ros2 bag one full cloud per rev.
    accum_frames: int = 5_000
    #: Hard point ceiling for that cloud — a memory / render-cost guard, not
    #: the retention knob. Whichever limit is hit first drops the oldest frame;
    #: raising it above ~1M makes each rebuild visibly expensive.
    accum_buffer_size: int = 800_000
    #: Keep every Nth point in the accumulated cloud (memory / render cost).
    accum_subsample: int = 4
    #: Show the accumulated cloud on startup.
    show_accum: bool = True
    #: Matplotlib-style colormap name for height coloring.
    colormap: str = "viridis"
    #: Point-cloud color source: "height" (z colormap), "reflectivity"
    #: (multiScan RSSI on a fixed 0-65535 scale) or "reflectivity_stretch"
    #: (the same RSSI, contrast-stretched across the frame's own percentile
    #: range). Toggle live with the V key. Both reflectivity modes need the
    #: sensor's RSSI output enabled (DataContentEchos bit 1).
    color_mode: str = "height"
    #: Colormap used in both reflectivity modes.
    reflectivity_colormap: str = "turbo"
    #: Percentile window for "reflectivity_stretch". Real surfaces occupy a
    #: narrow slice of the sensor's full scale - measured on arena data, the
    #: middle 98% of returns spans roughly 0.26-0.82 of full scale - so on the
    #: fixed scale an entire arena renders as one flat color. Stretching a
    #: percentile window across the colormap makes rock/ground contrast
    #: visible; the cost is that colors no longer compare between frames.
    reflectivity_percentiles: tuple[float, float] = (5.0, 95.0)
    #: Show the oriented box marking the sensor's live pose (toggle: B).
    show_lidar_box: bool = True
    #: Lidar box dimensions (x, y, z) in meters (multiScan136 is ~0.08 ⌀ x 0.09).
    lidar_box_size: tuple[float, float, float] = (0.10, 0.10, 0.09)
    window_width: int = 1280
    window_height: int = 800


@dataclass
class AppConfig:
    """Top-level configuration aggregating every subsystem's settings."""

    source: SourceConfig = field(default_factory=SourceConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    level: LevelConfig = field(default_factory=LevelConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    slam: SlamConfig = field(default_factory=SlamConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    record: RecordConfig = field(default_factory=RecordConfig)

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load configuration from a YAML file, filling in defaults for anything
        the file omits."""
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load YAML config files.")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Recursively build an :class:`AppConfig` from a plain dict, coercing
        list values into tuples where the dataclass field expects one."""
        return _build_dataclass(cls, data)  # type: ignore[return-value]


def _build_dataclass(dc_type: type, data: dict[str, Any]) -> Any:
    """Construct a (possibly nested) dataclass from a dict, ignoring unknown
    keys and converting lists to tuples for tuple-typed fields."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {dc_type.__name__}, got {type(data)}")
    # `from __future__ import annotations` turns f.type into a string, so resolve
    # the real annotation objects to detect nested dataclasses reliably.
    hints = get_type_hints(dc_type)
    kwargs: dict[str, Any] = {}
    for f in fields(dc_type):
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = hints.get(f.name, f.type)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = _build_dataclass(field_type, value)
        elif isinstance(value, list):
            # tuple-typed geometry fields (origin, extent) arrive as YAML lists.
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return dc_type(**kwargs)
