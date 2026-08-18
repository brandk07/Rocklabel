"""Configuration: built-in defaults <- YAML file <- CLI overrides.

The fully resolved config dict is what gets hashed into the dataset manifest,
so any change to any parameter produces a different dataset directory.
"""

from __future__ import annotations

import copy
import hashlib
import json

import yaml

DEFAULTS: dict = {
    "topics": {
        "pointcloud_topic": "/multiscan/lidar_scan",
        "odom_frame": "odom",
        "base_frame": "base_link",
        "lidar_frame": "lidar_link",
        "tf_topic": "/tf",
        "tf_static_topic": "/tf_static",
        # Max extrapolation (seconds) allowed when a scan stamp falls outside
        # the recorded TF range for a transform.
        "pose_tolerance_s": 0.15,
        # Fallback if the recording has no static base->lidar transform:
        # {"translation": [x, y, z], "quaternion": [x, y, z, w]}
        "static_lidar_to_base": None,
        # Drop points within this xy-distance (m) of the robot base in every
        # scan - removes LiDAR self-hits on the robot body (raw driver topics
        # often include them). 0 disables.
        "min_range_m": 0.0,
    },
    "level": {
        # Undo a tilted sensor mount. A recording's odom/world frame is only as
        # level as the rig that made it - a native lidarrig frame is the sensor
        # frame at startup - so a LiDAR on a slanted mast tilts the entire
        # cloud. A z clip then slices a diagonal wedge out of the floor instead
        # of a horizontal slab, and the model's neighborhoods see the mount
        # angle as relief. Levelling measures the angle once and rotates it out
        # of every scan, before the labeler, the generator, or driftcheck sees
        # a point, so all three share one geometry.
        #   "off"    - no rotation (what every pre-levelling dataset used)
        #   "ground" - RANSAC ground-plane fit over the recording
        #   "manual" - use mount_roll_deg / mount_pitch_deg verbatim
        "mode": "auto",
        # Mount angles (deg) for mode "manual", in the same convention the live
        # rig's IMU and `rocklabel label` report: a sensor pitched nose-up by
        # theta about its own +y axis is mount_pitch_deg = theta.
        "mount_roll_deg": 0.0,
        "mount_pitch_deg": 0.0,
        # Pool floor candidates from the first N scans (0 = whole recording).
        "fit_scans": 0,
        # 3D range band (m) around the sensor for pooled points: the inner
        # radius skips the rig's own frame, the outer keeps the floor dense
        # and flat. Measured per scan, so it follows a rig that walks.
        "range_min_m": 0.6,
        "range_max_m": 6.0,
        # Voxel size the pooled cloud is downsampled to before fitting; bounds
        # memory over a long recording and evens out dwell-time density bias.
        "fit_voxel_m": 0.05,
        # RANSAC inlier distance (m) and hypothesis count.
        "plane_thresh_m": 0.05,
        "ransac_iters": 512,
        # Reject planes tilted more than this from +z. This is what keeps the
        # fit off walls: indoors a wall often has more coplanar points than the
        # floor, so an unconstrained "largest plane" fit locks onto one.
        "max_tilt_deg": 50.0,
        # Acceptance gate. Deliberately loose: in a still-tilted frame the
        # pooled set is mostly walls, so the floor is a minority of it - the
        # tilt gate and the below-the-sensor check are the honest guards.
        "min_inlier_frac": 0.15,
        # A fitted plane this far above the sensor is the ceiling, not the
        # floor: a ceiling is just as planar and just as level.
        "ceiling_margin_m": 0.2,
    },
    "labeler": {
        "accumulator_voxel_m": 0.03,
        "default_rock_radius_m": 0.15,
        "stride": 1,
        "z_min": None,
        "z_max": None,
    },
    "generator": {
        "frame_stride": 5,
        # Merge all scans within this time window (seconds) into one dataset
        # frame before frame_stride is applied. 0 = one frame per message.
        # Essential for native lidarrig recordings, whose messages are raw
        # ~4 ms sensor batches (~400 points each): 0.25 fuses ~60 batches
        # into a dense frame. Harmless for ROS 2 bags (already full scans).
        "frame_window_s": 0.0,
        "crop_forward_m": 6.0,
        "crop_backward_m": 2.0,
        "crop_left_m": 4.0,
        "crop_right_m": 4.0,
        "crop_up_m": 1.5,
        "crop_down_m": 1.0,
        "boundary_shell_m": 0.05,
        # Format A: point-neighborhood samples
        "centers_voxel_m": 0.05,
        "neighborhood_radius_m": 0.5,
        "min_neighbors": 20,
        "neighborhood_points": 256,
        "negative_keep_prob": 0.05,
        # Format C: whole-frame per-point segmentation
        "segmentation_points": 4096,
        "segmentation_min_points": 512,
        # Format B: BEV rasters
        "bev_cell_m": 0.10,
        "seed": 42,
    },
}


class ConfigError(Exception):
    pass


def _merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        here = f"{path}.{key}" if path else key
        if key not in base:
            raise ConfigError(f"Unknown config key: {here!r}")
        if isinstance(base[key], dict) and isinstance(val, dict):
            out[key] = _merge(base[key], val, here)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | None = None) -> dict:
    """Return the fully resolved config: DEFAULTS overlaid with the YAML file."""
    if path is None:
        return copy.deepcopy(DEFAULTS)
    with open(path, "r") as f:
        user = yaml.safe_load(f) or {}
    if not isinstance(user, dict):
        raise ConfigError(f"Config file {path} must contain a YAML mapping")
    return _merge(DEFAULTS, user)


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply CLI overrides given as {"section.key": value}; None values are skipped."""
    cfg = copy.deepcopy(cfg)
    for dotted, val in overrides.items():
        if val is None:
            continue
        section, key = dotted.split(".", 1)
        if section not in cfg or key not in cfg[section]:
            raise ConfigError(f"Unknown config key: {dotted!r}")
        cfg[section][key] = val
    return cfg


def config_hash(cfg: dict) -> str:
    """SHA-256 over the canonical JSON encoding of the resolved config."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
