"""Tests for ROS 2 bag replay: the hand-rolled CDR decoders (validated against
rclpy's serializer, so they need a sourced ROS environment and are skipped
without one), TF pose composition, and end-to-end rosbag replay through
:class:`McapReplaySource` — auto-detection, world-frame transform, intensity
rescaling, and transport controls."""

from __future__ import annotations

import numpy as np
import pytest

from rocklabel.live.motion import quat_to_matrix
from rocklabel.live.recording import McapReplaySource
from rocklabel.live.ros2bag import TfTree, decode_pointcloud2, decode_tfmessage

pytest.importorskip("rclpy.serialization", reason="ROS 2 (rclpy) not available")
pytest.importorskip("sensor_msgs.msg")
pytest.importorskip("tf2_msgs.msg")

from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.serialization import serialize_message  # noqa: E402
from sensor_msgs.msg import PointCloud2, PointField  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402


def _cloud_msg(pts, inten=None, frame_id="lidar_link", stamp=1.5):
    """CDR bytes of a PointCloud2 with float32 x/y/z (+ optional reflective)."""
    pts = np.asarray(pts, dtype=np.float32)
    cols = [pts]
    fields = [
        PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, n in enumerate("xyz")
    ]
    if inten is not None:
        cols.append(np.asarray(inten, dtype=np.float32).reshape(-1, 1))
        fields.append(
            PointField(name="reflective", offset=12, datatype=PointField.FLOAT32, count=1)
        )
    data = np.ascontiguousarray(np.hstack(cols))
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int((stamp % 1.0) * 1e9)
    msg.height = 1
    msg.width = pts.shape[0]
    msg.fields = fields
    msg.point_step = data.shape[1] * 4
    msg.row_step = msg.point_step * msg.width
    msg.data = data.tobytes()
    msg.is_dense = True
    return serialize_message(msg)


def _tf_msg(entries):
    """CDR bytes of a TFMessage from (parent, child, pos, quat wxyz) tuples."""
    msg = TFMessage()
    for parent, child, pos, quat in entries:
        t = TransformStamped()
        t.header.frame_id = parent
        t.child_frame_id = child
        tr, q = t.transform.translation, t.transform.rotation
        tr.x, tr.y, tr.z = (float(v) for v in pos)
        q.w, q.x, q.y, q.z = (float(v) for v in quat)
        msg.transforms.append(t)
    return serialize_message(msg)


_YAW90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])


# --------------------------------------------------------------------------- #
# CDR decoders (ground truth: rclpy's own serializer)
# --------------------------------------------------------------------------- #
def test_decode_pointcloud2():
    pts = np.array([[1.0, 2.0, 3.0], [-4.5, 0.0, 9.25], [np.nan, 1.0, 2.0]])
    inten = np.array([0.0, 1.0, 0.5])
    cloud = decode_pointcloud2(_cloud_msg(pts, inten, frame_id="sensor", stamp=7.25))
    assert cloud.frame_id == "sensor"
    assert cloud.stamp == pytest.approx(7.25)
    np.testing.assert_array_equal(cloud.points, pts.astype(np.float32))
    np.testing.assert_array_equal(cloud.intensity, inten.astype(np.float32))


def test_decode_pointcloud2_without_intensity():
    cloud = decode_pointcloud2(_cloud_msg(np.zeros((4, 3))))
    assert cloud.intensity is None
    assert cloud.points.shape == (4, 3)


def test_decode_tfmessage():
    entries = [
        ("odom", "base_link", np.array([1.0, -2.0, 0.5]), _YAW90),
        ("base_link", "lidar_link", np.array([0.0, 0.0, 0.6]), np.array([1.0, 0, 0, 0])),
    ]
    decoded = decode_tfmessage(_tf_msg(entries))
    assert len(decoded) == 2
    for (parent, child, pos, quat), exp in zip(decoded, entries):
        assert (parent, child) == exp[:2]
        np.testing.assert_allclose(pos, exp[2])
        np.testing.assert_allclose(quat, exp[3])


def test_tf_tree_composes_chain():
    tree = TfTree()
    tree.update("odom", "base_link", np.array([1.0, 0.0, 0.0]), _YAW90)
    tree.update("base_link", "lidar_link", np.array([0.0, 0.0, 0.5]), np.array([1.0, 0, 0, 0]))
    pos, quat = tree.pose("lidar_link")
    np.testing.assert_allclose(pos, [1.0, 0.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(quat_to_matrix(quat), quat_to_matrix(_YAW90), atol=1e-12)
    # Unknown frames report identity (edges not seen yet early in a bag).
    pos, quat = tree.pose("nonexistent")
    np.testing.assert_allclose(pos, np.zeros(3))


# --------------------------------------------------------------------------- #
# End-to-end rosbag replay
# --------------------------------------------------------------------------- #
def _write_bag(path, messages):
    """Write a rosbag2-style MCAP: [(topic, schema_name, log_time_ns, data)]."""
    from mcap.writer import Writer

    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start(profile="ros2", library="test")
        channels = {}
        for topic, schema_name, _t, _d in messages:
            if topic not in channels:
                sid = w.register_schema(name=schema_name, encoding="ros2msg", data=b"")
                channels[topic] = w.register_channel(
                    topic=topic, message_encoding="cdr", schema_id=sid
                )
        for topic, _s, t_ns, data in messages:
            w.add_message(
                channel_id=channels[topic], log_time=t_ns, publish_time=t_ns, data=data
            )
        w.finish()


def _drain(src, limit=1000):
    out = []
    for _ in range(limit):
        b = src.read(timeout=0.01)
        if b is not None:
            out.append(b)
        if src.finished:
            break
    return out


def test_ros2_bag_replay_end_to_end(tmp_path):
    path = str(tmp_path / "rosbag.mcap")
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.5], [np.nan, 0.0, 0.0]])
    inten = np.array([1.0, 0.0, 1.0])  # binary reflector flag, as real bags have
    tf1 = ("odom", "lidar_link", np.array([1.0, 2.0, 0.0]), np.array([1.0, 0, 0, 0]))
    tf2 = ("odom", "lidar_link", np.array([3.0, 2.0, 0.0]), np.array([1.0, 0, 0, 0]))
    _write_bag(
        path,
        [
            ("/tf", "tf2_msgs/msg/TFMessage", 0, _tf_msg([tf1])),
            ("/pc", "sensor_msgs/msg/PointCloud2", 100_000_000, _cloud_msg(pts, inten)),
            ("/tf", "tf2_msgs/msg/TFMessage", 150_000_000, _tf_msg([tf2])),
            ("/pc", "sensor_msgs/msg/PointCloud2", 200_000_000, _cloud_msg(pts, inten)),
        ],
    )

    src = McapReplaySource(path, autoplay=False)
    src.start()
    assert src.duration_sec == pytest.approx(0.2)
    src.seek(src.duration_sec)
    batches = _drain(src)

    assert len(batches) == 2
    for batch, offset in zip(batches, (tf1[2], tf2[2])):
        # NaN point dropped; the rest world-transformed by the latest tf pose.
        np.testing.assert_allclose(batch.points, pts[:2] + offset, atol=1e-6)
        # 0-1 reflector flag rescaled onto the u16 RSSI display range.
        np.testing.assert_allclose(batch.intensity, [65535.0, 0.0])
        assert batch.orientation is None
    np.testing.assert_allclose(src.current_pose()[0], tf2[2])
    assert src.finished

    # Backward seek: rewind, reset downstream, fast-forward to the target
    # (0.11s lies past the first cloud, so exactly that one is redelivered).
    resets = []
    src.on_rewind = lambda: resets.append(1)
    src.seek(0.11)
    b = src.read(timeout=0.5)
    assert len(resets) == 1 and b is not None
    np.testing.assert_allclose(b.points, pts[:2] + tf1[2], atol=1e-6)
    src.stop()


def test_ros2_bag_without_clouds_raises(tmp_path):
    path = str(tmp_path / "empty.mcap")
    _write_bag(path, [("/tf", "tf2_msgs/msg/TFMessage", 0, _tf_msg([]))])
    src = McapReplaySource(path, autoplay=False)
    with pytest.raises(ValueError, match="nothing to replay"):
        src.start()


def test_intensity_scale_probe_u8_range(tmp_path):
    path = str(tmp_path / "u8.mcap")
    _write_bag(
        path,
        [
            (
                "/pc",
                "sensor_msgs/msg/PointCloud2",
                0,
                _cloud_msg(np.zeros((3, 3)), np.array([10.0, 200.0, 0.0])),
            )
        ],
    )
    src = McapReplaySource(path, autoplay=False)
    src.start()
    src.seek(1.0)
    (batch,) = _drain(src)
    np.testing.assert_allclose(batch.intensity, [2570.0, 51400.0, 0.0])  # x257
    src.stop()
