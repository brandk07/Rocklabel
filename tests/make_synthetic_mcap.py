"""Generate a small synthetic ROS 2 mcap for testing rocklabel end to end.

Scene: a flat floor at z ~ 0 (Gaussian noise sigma = 0.01) with three
hemispherical rocks of known centers/radii. The robot translates 3 m in +x
over the run; odom->base_link is published on /tf at 20 Hz, and the lidar
mount (base_link->lidar_link, offset + yaw) on /tf_static. Scan points are
expressed in the LIDAR frame so the full TF chain is exercised. Scan header
stamps fall between TF samples, exercising interpolation.

Run directly:  python tests/make_synthetic_mcap.py out.mcap
"""

from __future__ import annotations

import json
import sys

import numpy as np
from mcap_ros2.writer import Writer
from scipy.spatial.transform import Rotation

# Known ground truth (odom frame). Hemisphere centers sit on the floor.
ROCKS = [
    {"center": [1.5, 0.5, 0.0], "radius": 0.15},
    {"center": [3.0, -1.0, 0.0], "radius": 0.20},
    {"center": [4.5, 1.2, 0.0], "radius": 0.12},
]

N_SCANS = 100
SCAN_HZ = 10.0
TF_HZ = 20.0
DURATION_S = N_SCANS / SCAN_HZ          # 10 s
TRAVEL_M = 3.0                          # base_link moves +x at 0.3 m/s
T0_S = 1_000_000.0                      # arbitrary epoch offset

# base_link -> lidar_link: mounted forward+up, yawed 30 degrees
LIDAR_TRANSLATION = [0.2, 0.0, 0.5]
LIDAR_QUAT_XYZW = Rotation.from_euler("z", 30, degrees=True).as_quat().tolist()

_SEP = "=" * 80

_TIME_DEF = """MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

_HEADER_DEF = """MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id"""

POINTCLOUD2_MSGDEF = f"""std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense
{_SEP}
{_HEADER_DEF}
{_SEP}
{_TIME_DEF}
{_SEP}
MSG: sensor_msgs/PointField
uint8 INT8=1
uint8 UINT8=2
uint8 INT16=3
uint8 UINT16=4
uint8 INT32=5
uint8 UINT32=6
uint8 FLOAT32=7
uint8 FLOAT64=8
string name
uint32 offset
uint8 datatype
uint32 count"""

TFMESSAGE_MSGDEF = f"""geometry_msgs/TransformStamped[] transforms
{_SEP}
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform
{_SEP}
{_HEADER_DEF}
{_SEP}
{_TIME_DEF}
{_SEP}
MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation
{_SEP}
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
{_SEP}
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w"""


def base_position(t_rel: float) -> np.ndarray:
    return np.array([TRAVEL_M * t_rel / DURATION_S, 0.0, 0.0])


def T_odom_base(t_rel: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = base_position(t_rel)
    return m


def T_base_lidar() -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(LIDAR_QUAT_XYZW).as_matrix()
    m[:3, 3] = LIDAR_TRANSLATION
    return m


def sample_scene(rng: np.random.Generator, robot_x: float) -> tuple[np.ndarray, np.ndarray]:
    """World-frame points visible from the robot: floor patch + rock hemispheres."""
    n_floor = 4000
    floor = np.column_stack([
        rng.uniform(robot_x - 3.0, robot_x + 7.0, n_floor),
        rng.uniform(-4.0, 4.0, n_floor),
        rng.normal(0.0, 0.01, n_floor),
    ])
    inten = [np.full(n_floor, 0.2, np.float32)]
    parts = [floor]
    for rock in ROCKS:
        n_rock = 250
        v = rng.normal(size=(n_rock, 3))
        v[:, 2] = np.abs(v[:, 2])  # upper hemisphere
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        parts.append(np.asarray(rock["center"]) + v * rock["radius"])
        inten.append(np.full(n_rock, 0.8, np.float32))
    return np.concatenate(parts), np.concatenate(inten)


def make_pointcloud2_msg(xyz_lidar: np.ndarray, inten: np.ndarray, stamp_s: float,
                         frame_id: str = "lidar_link") -> dict:
    n = len(xyz_lidar)
    rows = np.zeros(n, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")]))
    rows["x"], rows["y"], rows["z"] = xyz_lidar.T.astype(np.float32)
    rows["intensity"] = inten
    sec = int(stamp_s)
    return {
        "header": {"stamp": {"sec": sec, "nanosec": int(round((stamp_s - sec) * 1e9))},
                   "frame_id": frame_id},
        "height": 1,
        "width": n,
        "fields": [
            {"name": "x", "offset": 0, "datatype": 7, "count": 1},
            {"name": "y", "offset": 4, "datatype": 7, "count": 1},
            {"name": "z", "offset": 8, "datatype": 7, "count": 1},
            {"name": "intensity", "offset": 12, "datatype": 7, "count": 1},
        ],
        "is_bigendian": False,
        "point_step": 16,
        "row_step": 16 * n,
        "data": rows.tobytes(),
        "is_dense": True,
    }


def make_tf_msg(parent: str, child: str, stamp_s: float, translation, quat_xyzw) -> dict:
    sec = int(stamp_s)
    return {"transforms": [{
        "header": {"stamp": {"sec": sec, "nanosec": int(round((stamp_s - sec) * 1e9))},
                   "frame_id": parent},
        "child_frame_id": child,
        "transform": {
            "translation": {"x": float(translation[0]), "y": float(translation[1]), "z": float(translation[2])},
            "rotation": {"x": float(quat_xyzw[0]), "y": float(quat_xyzw[1]),
                         "z": float(quat_xyzw[2]), "w": float(quat_xyzw[3])},
        },
    }]}


def write_synthetic_mcap(path: str, seed: int = 0, n_scans: int = N_SCANS,
                         intensity_scale: float = 1.0) -> None:
    """``intensity_scale`` mimics a driver that publishes raw RSSI counts in a
    float32 field (the SICK multiScan writes 0-65535 that way): pass 65535.0 to
    get an unnormalized bag."""
    rng = np.random.default_rng(seed)
    T_bl = T_base_lidar()
    with open(path, "wb") as f:
        writer = Writer(f)
        pc2_schema = writer.register_msgdef("sensor_msgs/msg/PointCloud2", POINTCLOUD2_MSGDEF)
        tf_schema = writer.register_msgdef("tf2_msgs/msg/TFMessage", TFMESSAGE_MSGDEF)

        static_stamp = T0_S
        writer.write_message(
            "/tf_static", tf_schema,
            make_tf_msg("base_link", "lidar_link", static_stamp, LIDAR_TRANSLATION, LIDAR_QUAT_XYZW),
            log_time=int(static_stamp * 1e9), publish_time=int(static_stamp * 1e9),
        )

        # /tf at 20 Hz, extended slightly past both ends so scans never need
        # extrapolation.
        n_tf = int(DURATION_S * TF_HZ) + 3
        for i in range(n_tf):
            t_rel = (i - 1) / TF_HZ
            stamp = T0_S + t_rel
            writer.write_message(
                "/tf", tf_schema,
                make_tf_msg("odom", "base_link", stamp,
                            base_position(np.clip(t_rel, 0.0, DURATION_S)), [0.0, 0.0, 0.0, 1.0]),
                log_time=int(stamp * 1e9), publish_time=int(stamp * 1e9),
            )

        # Scans at 10 Hz, offset by half a TF period to force interpolation.
        for i in range(n_scans):
            t_rel = i / SCAN_HZ + 0.5 / TF_HZ
            stamp = T0_S + t_rel
            world, inten = sample_scene(rng, base_position(t_rel)[0])
            T_ol = T_odom_base(t_rel) @ T_bl
            T_lo = np.linalg.inv(T_ol)
            lidar_pts = world @ T_lo[:3, :3].T + T_lo[:3, 3]
            writer.write_message(
                "/multiscan/lidar_scan", pc2_schema,
                make_pointcloud2_msg(lidar_pts, inten * intensity_scale, stamp),
                log_time=int(stamp * 1e9), publish_time=int(stamp * 1e9),
            )
        writer.finish()


def write_matching_labels(path: str, mcap_name: str = "synthetic.mcap", margin: float = 0.02) -> None:
    """Labels JSON matching the known synthetic rocks (radius padded by margin)."""
    data = {
        "schema_version": 1,
        "mcap_file": mcap_name,
        "run_id": mcap_name.rsplit(".", 1)[0],
        "odom_frame": "odom",
        "created": "2026-01-01T00:00:00Z",
        "tool_version": "test",
        "intensity_available": True,
        "accumulator_voxel_m": 0.03,
        "rocks": [
            {"id": i + 1, "center": r["center"], "radius": r["radius"] + margin}
            for i, r in enumerate(ROCKS)
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_synthetic_lidarrig_mcap(path: str, seed: int = 0, n_frames: int = N_SCANS,
                                  with_intensity: bool = True) -> None:
    """Native lidarrig recording of the same scene: sensor-frame points +
    embedded world pose per frame (no TF), RSSI-style u16-range intensity."""
    from mcap.writer import Writer as McapWriter

    from rocklabel.recording.lidarrig_io import SCHEMA_NAME, TOPIC, encode_frame

    rng = np.random.default_rng(seed)
    with open(path, "wb") as f:
        writer = McapWriter(f)
        writer.start(profile="x-lidarrig", library="lidarrig")
        schema_id = writer.register_schema(name=SCHEMA_NAME, encoding="x-lidarrig",
                                           data=b"synthetic")
        channel_id = writer.register_channel(topic=TOPIC, message_encoding="x-lidarrig-frame",
                                             schema_id=schema_id)
        for i in range(n_frames):
            t_rel = i / SCAN_HZ
            stamp = T0_S + t_rel
            # The handheld sensor translates +x and yaws slowly while scanning.
            pos = base_position(t_rel) + np.array([0.0, 0.0, 0.8])
            yaw = np.deg2rad(20.0) * np.sin(2 * np.pi * t_rel / DURATION_S)
            half, s, c = yaw / 2.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)
            quat_wxyz = np.array([c, 0.0, 0.0, s])
            rot = Rotation.from_quat([0.0, 0.0, s, c]).as_matrix()  # xyzw

            world, inten = sample_scene(rng, base_position(t_rel)[0])
            sensor_pts = (world - pos) @ rot  # rot^T applied from the right
            intensity = (inten * 65535.0).astype(np.float32) if with_intensity else None
            data = encode_frame(sensor_pts.astype(np.float32), intensity, stamp,
                                pos, quat_wxyz)
            log_ns = int(stamp * 1e9)
            writer.add_message(channel_id=channel_id, log_time=log_ns,
                               publish_time=log_ns, data=data, sequence=i)
        writer.finish()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic.mcap"
    write_synthetic_mcap(out)
    write_matching_labels(out.rsplit(".", 1)[0] + ".labels.json", mcap_name=out)
    print(f"wrote {out} ({N_SCANS} scans, {len(ROCKS)} rocks) and matching labels JSON")
