"""`selfhits`: removing the points a robot-mounted sensor sees of its own body."""

import numpy as np
import pytest

from rocklabel.config import load_config
from rocklabel.live.recording import decode_frame, encode_frame
from rocklabel.live.ros2bag import decode_pointcloud2
from rocklabel.recording.mcap_io import read_info
from rocklabel.recording.selfhits import (
    DEFAULT_RADIUS,
    SelfHitError,
    SelfHitMask,
    filter_lidarrig_frame,
    filter_pointcloud2,
    measure,
    run_selfhits,
)


# ------------------------------------------------------------------ the mask
def test_mask_sphere_selects_only_close_points():
    mask = SelfHitMask(radius=0.5)
    pts = np.array([
        [0.1, 0.0, 0.0],    # inside
        [0.0, 0.0, 0.49],   # inside
        [0.0, 0.0, 0.51],   # outside
        [3.0, 0.0, 0.0],    # far
    ])
    assert mask.hits(pts).tolist() == [True, True, False, False]


def test_mask_box_catches_what_the_sphere_misses():
    mask = SelfHitMask(radius=0.1, boxes=[(0.5, 1.0, -0.2, 0.2, -0.1, 0.1)])
    pts = np.array([
        [0.7, 0.0, 0.0],   # in the box, well outside the sphere
        [0.7, 0.5, 0.0],   # beside the box
        [0.05, 0.0, 0.0],  # in the sphere
    ])
    assert mask.hits(pts).tolist() == [True, False, True]


def test_mask_never_flags_non_finite_points():
    mask = SelfHitMask(radius=0.5)
    pts = np.array([[np.nan, 0.0, 0.0], [0.0, np.inf, 0.0], [0.1, 0.0, 0.0]])
    assert mask.hits(pts).tolist() == [False, False, True]


def test_mask_rejects_nonsense():
    with pytest.raises(SelfHitError):
        SelfHitMask(radius=0.0)                       # would remove nothing
    with pytest.raises(SelfHitError):
        SelfHitMask(radius=0.1, boxes=[(1.0, 0.0, 0, 1, 0, 1)])   # reversed edge
    with pytest.raises(SelfHitError):
        SelfHitMask(radius=0.1, boxes=[(0.0, 1.0, 0.0)])          # wrong length


# ------------------------------------------------------- per-message surgery
def test_filter_pointcloud2_preserves_every_other_point_exactly(synthetic_recording):
    """Survivors must come back bit-identical, with all fields intact."""
    from mcap.reader import make_reader

    mcap_path, _ = synthetic_recording
    mask = SelfHitMask(radius=1.0)
    checked = 0
    with open(mcap_path, "rb") as fh:
        for _s, _c, m in make_reader(fh).iter_messages(
            topics=["/multiscan/lidar_scan"]
        ):
            orig = decode_pointcloud2(m.data)
            data, kept, removed = filter_pointcloud2(m.data, mask)
            new = decode_pointcloud2(data)
            keep = ~mask.hits(orig.points)
            assert kept == int(keep.sum()) == new.points.shape[0]
            assert removed == int((~keep).sum())
            assert np.array_equal(new.points, orig.points[keep], equal_nan=True)
            assert np.array_equal(
                new.intensity, orig.intensity[keep], equal_nan=True
            )
            assert new.frame_id == orig.frame_id
            assert new.stamp == pytest.approx(orig.stamp)
            # nothing inside the mask may survive
            assert not mask.hits(new.points).any()
            checked += 1
            if checked >= 5:
                break
    assert checked == 5


def test_filter_pointcloud2_is_a_no_op_when_nothing_is_close(synthetic_recording):
    from mcap.reader import make_reader

    mcap_path, _ = synthetic_recording
    with open(mcap_path, "rb") as fh:
        _s, _c, m = next(
            iter(make_reader(fh).iter_messages(topics=["/multiscan/lidar_scan"]))
        )
    data, _kept, removed = filter_pointcloud2(m.data, SelfHitMask(radius=1e-6))
    assert removed == 0
    assert data == m.data  # byte-for-byte the original message


def test_filter_lidarrig_frame_keeps_pose_and_reflectivity():
    pts = np.array([[0.1, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
                   dtype=np.float32)
    inten = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    pos = np.array([1.0, 2.0, 3.0])
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    payload = encode_frame(pts, inten, 12.5, np.array([1.0, 0, 0, 0]), pos, quat)

    data, kept, removed = filter_lidarrig_frame(payload, SelfHitMask(radius=0.5))
    assert (kept, removed) == (2, 1)
    fr = decode_frame(data)
    assert np.array_equal(fr.points, pts[1:])
    assert np.array_equal(fr.intensity, inten[1:])
    assert fr.timestamp == pytest.approx(12.5)
    assert np.allclose(fr.pose_position, pos)      # pose must survive untouched
    assert np.allclose(fr.pose_quat, quat)


# -------------------------------------------------------------- the command
def _add_self_hits(src: str, dst: str, n_hits: int = 40) -> None:
    """Copy a recording, injecting n_hits fixed points 0.2 m from the sensor."""
    import struct

    from mcap.records import Channel, Message, Metadata, Schema
    from mcap.stream_reader import StreamReader
    from mcap.writer import Writer

    from rocklabel.recording.selfhits import (_cloud_records, _parse_cloud)

    rng = np.random.default_rng(0)
    unit = rng.normal(size=(n_hits, 3))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    body = (unit * 0.2).astype(np.float32)     # a shell at r = 0.2 m

    schemas, channels, sids, cids = {}, {}, {}, {}
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        w = Writer(fout)
        w.start(profile="ros2", library="test")
        for rec in StreamReader(fin, emit_chunks=False).records:
            if isinstance(rec, Schema):
                schemas[rec.id] = rec
            elif isinstance(rec, Channel):
                channels[rec.id] = rec
            elif isinstance(rec, Metadata):
                w.add_metadata(rec.name, dict(rec.metadata))
            elif isinstance(rec, Message):
                ch = channels[rec.channel_id]
                data = rec.data
                if ch.topic == "/multiscan/lidar_scan":
                    lay = _parse_cloud(data)
                    recs = _cloud_records(data, lay)
                    extra = np.zeros((n_hits, lay.point_step), np.uint8)
                    for j, name in enumerate("xyz"):
                        off = lay.fields[name][0]
                        extra[:, off:off + 4] = np.ascontiguousarray(
                            body[:, j]
                        ).view(np.uint8).reshape(n_hits, 4)
                    allr = np.vstack([recs, extra])
                    head = bytearray(data[: lay.data_start])
                    n = allr.shape[0]
                    struct.pack_into("<I", head, lay.off_height, 1)
                    struct.pack_into("<I", head, lay.off_width, n)
                    struct.pack_into("<I", head, lay.off_row_step,
                                     n * lay.point_step)
                    struct.pack_into("<I", head, lay.off_data_len,
                                     n * lay.point_step)
                    data = (bytes(head) + allr.tobytes() + data[lay.data_end:])
                if rec.channel_id not in cids:
                    sc = schemas[ch.schema_id]
                    if ch.schema_id not in sids:
                        sids[ch.schema_id] = w.register_schema(
                            sc.name, sc.encoding, sc.data)
                    cids[rec.channel_id] = w.register_channel(
                        ch.topic, ch.message_encoding, sids[ch.schema_id],
                        dict(ch.metadata))
                w.add_message(cids[rec.channel_id], rec.log_time, data,
                              rec.publish_time, rec.sequence)
        w.finish()


@pytest.fixture(scope="module")
def recording_with_self_hits(synthetic_recording, tmp_path_factory):
    """The synthetic recording plus 40 fake bodywork points per scan."""
    src, labels = synthetic_recording
    dirty = str(tmp_path_factory.mktemp("selfhits") / "dirty.mcap")
    _add_self_hits(src, dirty)
    return dirty, labels


def test_run_selfhits_removes_the_body_and_leaves_the_rest(
    recording_with_self_hits, tmp_path
):
    from mcap.reader import make_reader

    dirty, _ = recording_with_self_hits
    out = str(tmp_path / "clean.mcap")
    run_selfhits(dirty, out, load_config(None), radius=DEFAULT_RADIUS)

    before, after = read_info(dirty), read_info(out)
    # every topic and message survives - only points inside a scan are dropped
    assert after.topics.keys() == before.topics.keys()
    for topic in before.topics:
        assert after.message_count(topic) == before.message_count(topic)

    mask = SelfHitMask(radius=DEFAULT_RADIUS)
    with open(out, "rb") as fh:
        for i, (_s, _c, m) in enumerate(
            make_reader(fh).iter_messages(topics=["/multiscan/lidar_scan"])
        ):
            cloud = decode_pointcloud2(m.data)
            assert not mask.hits(cloud.points).any()   # body gone
            assert cloud.points.shape[0] > 0           # scene kept
            if i >= 4:
                break


def test_run_selfhits_never_touches_the_input(recording_with_self_hits, tmp_path):
    dirty, _ = recording_with_self_hits
    before = open(dirty, "rb").read()
    run_selfhits(dirty, str(tmp_path / "out.mcap"), load_config(None))
    assert open(dirty, "rb").read() == before


def test_run_selfhits_refuses_to_overwrite_the_input(recording_with_self_hits):
    dirty, _ = recording_with_self_hits
    with pytest.raises(SelfHitError, match="must differ"):
        run_selfhits(dirty, dirty, load_config(None))


def test_dry_run_writes_nothing(recording_with_self_hits, tmp_path):
    dirty, _ = recording_with_self_hits
    out = tmp_path / "nope.mcap"
    run_selfhits(dirty, str(out), load_config(None), dry_run=True)
    assert not out.exists()


def test_cleaned_recording_still_generates_a_dataset(
    recording_with_self_hits, tmp_path
):
    """The output must stay a fully usable recording downstream."""
    from rocklabel.dataset.generate import run_generate

    dirty, labels = recording_with_self_hits
    out = str(tmp_path / "clean.mcap")
    cfg = load_config(None)
    run_selfhits(dirty, out, cfg)
    entry = run_generate(out, labels, str(tmp_path / "ds"), cfg)
    assert entry["frames_kept"] > 0
    assert entry["frames_skipped_pose"] == 0


def test_measure_finds_the_injected_body(recording_with_self_hits):
    """--measure should spot a fixed 0.2 m shell and suggest a radius past it."""
    dirty, _ = recording_with_self_hits
    result = measure(dirty, load_config(None), sample=30)
    assert result["rigid_directions"] > 0
    assert result["max_rigid_range_m"] == pytest.approx(0.2, abs=0.05)
    assert result["suggested_radius_m"] > result["max_rigid_range_m"]


def test_dashboard_box_field_survives_its_own_commas():
    """A --box is six comma-separated numbers, so the UI must not comma-split it."""
    from rocklabel.dashboard.spec import COMMANDS, build_argv

    cmd = next(c for c in COMMANDS if c.id == "selfhits")
    argv = build_argv(cmd, {
        "mcap": "r.mcap", "out": "o.mcap",
        "box": "0.5,1.0,-0.3,0.3,-0.2,0.2; 1.0,1.5,0,1,0,1",
    })
    assert argv.count("--box") == 2
    assert "0.5,1.0,-0.3,0.3,-0.2,0.2" in argv
    assert "1.0,1.5,0,1,0,1" in argv

    # and the CLI parses back exactly what the dashboard previewed
    from rocklabel.cli import _parse_box, build_parser

    parser = build_parser()
    args = parser.parse_args(argv[1:])
    assert [_parse_box(b, parser) for b in args.box] == [
        (0.5, 1.0, -0.3, 0.3, -0.2, 0.2),
        (1.0, 1.5, 0.0, 1.0, 0.0, 1.0),
    ]


# ------------------------------------------------- solving a ROS 2 competition bag
def test_slam_can_read_a_ros2_bag(synthetic_recording):
    """The pose solver only spoke /lidar/frames; competition logs are rosbags."""
    from rocklabel.slam.reprocess import (is_ros2_bag, load_frames,
                                          load_ros2_frames)

    mcap_path, _ = synthetic_recording
    assert is_ros2_bag(mcap_path)
    assert load_frames(mcap_path) == []          # the old reader sees nothing

    frames = load_ros2_frames(mcap_path, stride=1)
    assert len(frames) == 100
    fr = frames[0]
    assert fr.points.shape[1] == 3
    assert fr.points.shape[0] > 0
    # a gravity-referenced orientation prior must be present or the solver
    # refuses to run at all
    assert fr.orientation is not None
    # zero-range "no return" rows must not survive as points at the origin
    assert (np.linalg.norm(fr.points, axis=1) > 1e-3).all()


def test_ros2_stride_thins_the_scans(synthetic_recording):
    from rocklabel.slam.reprocess import load_ros2_frames

    mcap_path, _ = synthetic_recording
    assert len(load_ros2_frames(mcap_path, stride=4)) == 25
    assert len(load_ros2_frames(mcap_path, stride=1, max_frames=10)) == 10


def test_solver_accepts_bag_frames(synthetic_recording):
    """End to end: a rosbag's frames must actually drive the solver."""
    from rocklabel.slam.config import AltSlamConfig
    from rocklabel.slam.reprocess import load_ros2_frames
    from rocklabel.slam.solver import OfflineSolver

    mcap_path, _ = synthetic_recording
    frames = load_ros2_frames(mcap_path, stride=2)
    solver = OfflineSolver(AltSlamConfig())
    solver.build_windows(frames)        # used to raise "no IMU orientation"
    assert len(solver.windows) > 0
    stats = solver.solve()
    assert stats.windows > 0
    positions, quats = solver.batch_poses(frames)
    assert positions.shape == (len(frames), 3)
    assert quats.shape == (len(frames), 4)
