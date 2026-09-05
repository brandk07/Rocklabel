"""Tests for `rocklabel live --web-ui`: the control catalog, the controller,
and the API the page polls.

The contract with the browser is "every control the schema advertises can
actually be set, and every readout the page renders is in the state payload",
so these drive the real Flask app over a real (replayed) engine rather than
mocking one. Flask is an optional extra, so the whole module skips without it.

There is no Open3D window here — the controller is built with ``viz=None``,
which is also the ``--headless --web-ui`` path — so the display-only controls
are expected to drop out of the schema. That gating *is* the thing under test
in :func:`test_headless_schema_drops_display_only_controls`.
"""

from __future__ import annotations

import base64
import time

import numpy as np
import pytest

flask = pytest.importorskip("flask", reason="the web UI needs the [dash] extra")

from rocklabel.live.config import AppConfig  # noqa: E402
from rocklabel.live.pipeline import IngestEngine  # noqa: E402
from rocklabel.live.recording import McapReplaySource  # noqa: E402
from rocklabel.live.sources.simulated import SimulatedSource  # noqa: E402
from rocklabel.live.surfaces.kalman_heightmap import KalmanHeightmap  # noqa: E402
from rocklabel.live.webui import spec  # noqa: E402
from rocklabel.live.webui.control import LiveController  # noqa: E402
from rocklabel.live.webui.server import create_app  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _live_engine() -> tuple[IngestEngine, AppConfig]:
    cfg = AppConfig()
    cfg.source.sim_points_per_sec = 20_000
    cfg.source.sim_batch_size = 2_000
    cfg.source.sim_seed = 3
    cfg.slam.enabled = False
    return IngestEngine(SimulatedSource(cfg), KalmanHeightmap(cfg), cfg), cfg


@pytest.fixture(scope="module")
def replay_recording(tmp_path_factory) -> str:
    """A short real recording, so the replay controller has a real transport."""
    path = str(tmp_path_factory.mktemp("webui") / "session.mcap")
    engine, _cfg = _live_engine()
    assert engine.start_recording(path) == path
    engine.start()
    time.sleep(0.6)
    engine.stop()
    engine.stop_recording()
    return path


def _replay_controller(path: str, fuse_sec: float = 0.0,
                       with_model: bool = False) -> LiveController:
    """A controller over a replayed recording.

    ``fuse_sec`` runs the pipeline for that long first, so the surface has
    something in it — needed by anything that looks at the overhead view, and
    pointless for the tests that only poke at controls.

    ``with_model`` adds the stand-in scorer and window. They are passed to the
    constructor rather than bolted on afterwards because the controller builds
    its model slots from them — a comparison assembled in the wrong order would
    be holding a scorer nothing else references.
    """
    cfg = AppConfig()
    cfg.slam.enabled = False
    cfg.motion.use_imu = False
    src = McapReplaySource(path, autoplay=False)
    engine = IngestEngine(src, KalmanHeightmap(cfg), cfg)
    src.on_rewind = engine.reset_surface
    if fuse_sec > 0:
        engine.start()          # starts the source as part of starting up
        src.play()
        time.sleep(fuse_sec)
        engine.stop()
        src.start()             # re-open: engine.stop() closed the file
    else:
        src.start()             # duration_sec needs the file open
    scorer = FakeScorer() if with_model else None
    viz = FakeViz(scorer, cfg) if with_model else None
    return LiveController(cfg, engine, scorer=scorer, viz=viz)


@pytest.fixture
def replay_ctl(replay_recording):
    ctl = _replay_controller(replay_recording)
    yield ctl
    ctl._engine.source.stop()


@pytest.fixture
def live_ctl():
    engine, cfg = _live_engine()
    return LiveController(cfg, engine)


@pytest.fixture
def client(replay_ctl):
    app = create_app(replay_ctl)
    app.config.update(TESTING=True)
    return app.test_client()


# --------------------------------------------------------------------------- #
# Stand-ins for the two things a test cannot have: an Open3D window and a GPU.
#
# They are not mocks of convenience — the controller is *only* allowed to touch
# the surface below, and test_controller_only_uses_the_real_viz_api pins that
# against the real VizApp so this cannot quietly drift into fiction.
# --------------------------------------------------------------------------- #
class FakeViz:
    """The VizApp surface LiveController drives."""

    def __init__(self, scorer=None, cfg=None) -> None:
        # The real VizApp shares the controller's AppConfig, which is how a
        # crop write reaches the engine: both hold the same CropConfig object.
        self._cfg = cfg
        self.color_mode = "height"
        self.reflectivity_range = (0.0, 1.0)
        self.nav_mode = "orbit"
        self.point_size = 3.0
        self._show_points = True
        self._show_mesh = True
        self._show_accum = True
        self._show_box = True
        self._crop_view = True
        self._model_display = 0
        self._scorer = scorer
        self.calls: list[tuple] = []
        # Comparison bookkeeping: the second window this one opened, and the
        # callback it reports its own closing through.
        self.peer = None
        self.title = "rocklabel - live"
        self.on_closed = None
        self.closed = False

    # Runs inline and reports "queued", which is what a real window does once
    # it exists. Returning False is the not-yet-built case the controller must
    # not block on — see test_a_write_before_the_window_exists_does_not_hang.
    def post(self, fn):
        fn()
        return True

    def set_color_mode(self, mode):
        self.color_mode = mode
        self.calls.append(("color_mode", mode))

    def set_reflectivity_range(self, lo, hi):
        from rocklabel.live.colormap import clamp_range

        self.reflectivity_range = clamp_range(lo, hi)

    def autofit_reflectivity(self):
        self.calls.append(("autofit_reflectivity",))

    def set_nav_mode(self, mode):
        self.nav_mode = mode

    def set_point_size(self, value):
        self.point_size = float(value)

    def set_layer(self, attr, value):
        setattr(self, attr, bool(value))

    def set_crop_view(self, value):
        self._crop_view = bool(value)

    def set_model_display(self, index):
        self._model_display = int(index)

    # -- the comparison window's surface ------------------------------- #
    def set_scorer(self, scorer):
        self._scorer = scorer

    def open_comparison_window(self, scorer, title, on_closed=None):
        peer = FakeViz(scorer, self._cfg)
        peer.title = title
        peer.on_closed = on_closed
        self.peer = peer
        return peer

    def close_window(self):
        self.closed = True
        if callable(self.on_closed):
            self.on_closed()

    def set_threshold(self, value):
        self._scorer.threshold = float(value)

    def set_score_setting(self, name, value):
        setattr(self._scorer.settings, name, value)

    def set_crop_setting(self, name, value):
        setattr(self._cfg.crop, name, value)

    def reset_camera(self):
        self.calls.append(("reset_camera",))

    def recalibrate_level(self):
        self.calls.append(("recalibrate_level",))

    def refresh_accum(self):
        self.calls.append(("refresh_accum",))

    def toggle_recording(self):
        self.calls.append(("toggle_recording",))

    def clear_predictions(self):
        self._scorer.clear_map()


class FakeScorer:
    """A LiveScorer without torch: settings, a threshold and a status."""

    model_name = "pointnet"
    #: Candidate voxel edge, as LiveScorer reports it — the outline builder
    #: inflates its polygons by half a cell.
    center_spacing_m = 0.1

    def __init__(self, checkpoint: str = "training/fake/best.pt",
                 settings=None, model_name: str = "", window_s: float = 0.0
                 ) -> None:
        from rocklabel.live.scoring import ScoreSettings

        # A second model is handed the FIRST one's settings object — that
        # sharing is the contract the comparison rests on, so the stand-in has
        # to honour it rather than quietly making its own.
        self.settings = settings if settings is not None \
            else ScoreSettings(window_sec=0.0)
        self.threshold = 0.89
        self.tuned_threshold = 0.89
        self.checkpoint = checkpoint
        self.frame_window_s = float(window_s)
        if model_name:
            self.model_name = model_name
        self.cleared = 0
        self.version = 1
        self.started = 0
        self.stopped = 0
        self._centers = np.empty((0, 3))
        self._probs = np.empty((0,), np.float32)

    def attach_result(self, centers, probs) -> None:
        """Stand in for a completed scoring pass."""
        self._centers = np.asarray(centers, float)
        self._probs = np.asarray(probs, np.float32)
        self.version += 1

    def detections(self):
        keep = self._probs >= self.threshold
        return self._centers[keep], self._probs[keep]

    def all_probs(self):
        return self._probs

    def clear_map(self) -> None:
        self.cleared += 1

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def status_dict(self) -> dict:
        return {
            "enabled": bool(self.settings.enabled), "ready": True,
            "map_centers": 4210, "detections": 128, "pass_centers": 900,
            "pass_ms": 41.0, "in_region": 18_400, "capped": False,
            "warning": "", "threshold": self.threshold,
            "model_name": self.model_name,
        }


def full_controller(path: str, fuse_sec: float = 0.0) -> LiveController:
    """A replay controller with every capability turned on."""
    ctl = _replay_controller(path, fuse_sec=fuse_sec, with_model=True)
    # Levelling defaults to mode="auto", which is what makes it active.
    assert ctl._engine.leveler.active
    return ctl


# --------------------------------------------------------------------------- #
# The catalog
# --------------------------------------------------------------------------- #
def test_every_control_has_prose_and_a_usable_kind():
    kinds = {"bool", "float", "int", "enum", "action", "readout", "transport"}
    assert spec.SECTIONS
    for sec in spec.SECTIONS:
        assert sec.requires in ("",) + spec.CAPABILITIES
        for c in sec.controls:
            assert c.kind in kinds, f"{c.id}: unknown kind {c.kind}"
            assert c.requires in ("",) + spec.CAPABILITIES, f"{c.id}: bad requires"
            # The help text is the whole reason these moved out of Open3D
            # tooltips; a control without it is a number with no meaning.
            assert len(c.help) > 25, f"{c.id} needs a real explanation"
            assert c.label, f"{c.id} needs a label"


def test_numeric_controls_are_bounded_and_enums_have_choices():
    for c in spec.CONTROLS_BY_ID.values():
        if c.kind in ("int", "float"):
            assert c.min is not None and c.max is not None, f"{c.id} is unbounded"
            assert c.min < c.max, f"{c.id} has an inverted range"
            assert c.step and c.step > 0, f"{c.id} needs a step"
        if c.kind == "enum":
            # A runtime-discovered enum (the checkpoint pickers) has no static
            # options — the controller fills them in — but it must say where
            # they come from, or the page renders an empty dropdown forever.
            assert c.choices or c.choices_from, \
                f"{c.id} is an enum with no choices and no source"


def test_control_ids_are_unique():
    flat = [c.id for s in spec.SECTIONS for c in s.controls]
    assert len(flat) == len(set(flat))


# --------------------------------------------------------------------------- #
# Capability gating
# --------------------------------------------------------------------------- #
def test_replay_run_has_a_transport_and_no_recording(replay_ctl):
    caps = replay_ctl.capabilities
    assert "replay" in caps and "live" not in caps
    ids = {c["id"] for s in replay_ctl.schema()["sections"] for c in s["controls"]}
    assert "replay.play_pause" in ids and "replay.position" in ids
    # Replays are never re-recorded, so offering the button would be a lie.
    assert not any(i.startswith("record.") for i in ids)


def test_live_run_has_recording_and_no_transport(live_ctl):
    caps = live_ctl.capabilities
    assert "live" in caps and "replay" not in caps
    ids = {c["id"] for s in live_ctl.schema()["sections"] for c in s["controls"]}
    assert "record.toggle" in ids and "status.pause" in ids
    assert not any(i.startswith("replay.") for i in ids)


def test_headless_schema_drops_display_only_controls(replay_ctl):
    """With no Open3D window there is nothing to color or hide."""
    assert "viewer" not in replay_ctl.capabilities
    ids = {c["id"] for s in replay_ctl.schema()["sections"] for c in s["controls"]}
    for gone in ("view.color_mode", "view.point_size", "view.show_points",
                 "view.crop_view", "view.reset_camera"):
        assert gone not in ids
    # ...but the engine-level knobs in the same section stay.
    assert "view.accum_frames" in ids and "view.accum_max_points" in ids


def test_no_model_means_no_model_or_region_sections(replay_ctl):
    sections = {s["id"] for s in replay_ctl.schema()["sections"]}
    assert "model" not in sections and "region" not in sections


def test_the_crop_band_is_offered_without_a_model(replay_ctl):
    """The z band a replay run is watching is not the scorer's to own.

    Without --model there is no Scoring region card, and before this the crop
    was a startup-only flag: the operator could see the wrong band but not move
    it. The Crop card is the same knobs on the CropConfig the engine reads.
    """
    sections = {s["id"] for s in replay_ctl.schema()["sections"]}
    assert "crop" in sections and "region" not in sections
    ids = {c["id"] for s in replay_ctl.schema()["sections"] for c in s["controls"]}
    assert {"crop.z_min", "crop.z_max", "crop.range_max"} <= ids


def test_crop_writes_reach_the_config_the_engine_reads(replay_ctl):
    """Same object the pipeline holds, so the next batch is cropped by it."""
    assert replay_ctl._engine._crop is replay_ctl._cfg.crop
    replay_ctl.set("crop.z_min", -1.5)
    replay_ctl.set("crop.z_max", -0.5)
    replay_ctl.set("crop.range_max", 8.0)
    replay_ctl.set("crop.enabled", True)
    crop = replay_ctl._engine._crop
    assert (crop.z_min, crop.z_max, crop.range_max) == (-1.5, -0.5, 8.0)
    values = replay_ctl.snapshot()["values"]
    assert values["crop.z_min"] == pytest.approx(-1.5)
    assert values["crop.z_max"] == pytest.approx(-0.5)


def test_crop_band_draws_the_ring_on_the_overhead_map(replay_ctl):
    """With no scorer the map's region ring follows the crop instead."""
    replay_ctl.set("crop.enabled", True)
    replay_ctl.set("crop.range_max", 6.0)
    assert replay_ctl.scene()["region"]["range_max"] == pytest.approx(6.0)
    replay_ctl.set("crop.enabled", False)
    assert replay_ctl.scene().get("region") is None


def test_model_choice_appears_only_with_a_scorer():
    with_scorer = spec.to_json({"viewer", "scorer", "live"})
    without = spec.to_json({"viewer", "live"})

    def modes(payload):
        for s in payload["sections"]:
            for c in s["controls"]:
                if c["id"] == "view.color_mode":
                    return [ch["value"] for ch in c["choices"]]
        return []

    assert "model" in modes(with_scorer)
    assert "model" not in modes(without)


# --------------------------------------------------------------------------- #
# Reading state
# --------------------------------------------------------------------------- #
def test_snapshot_covers_every_readout_the_schema_advertises(replay_ctl):
    snap = replay_ctl.snapshot()
    for sec in replay_ctl.schema()["sections"]:
        for c in sec["controls"]:
            if c["kind"] == "readout":
                assert c["id"] in snap["status"], f"{c['id']} has no value"
            elif c["kind"] not in ("action",):
                assert c["id"] in snap["values"], f"{c['id']} has no value"


def test_snapshot_reports_the_transport(replay_ctl):
    t = replay_ctl.snapshot()["transport"]
    assert t["duration_sec"] > 0
    assert t["playing"] is False  # autoplay=False
    assert t["position_sec"] >= 0
    assert t["speed"] == 1.0


def test_playback_speed_is_settable_and_reported(replay_ctl):
    replay_ctl.set("replay.speed", 2.0)
    assert replay_ctl._engine.source.speed == 2.0
    snap = replay_ctl.snapshot()
    assert snap["values"]["replay.speed"] == 2.0
    assert snap["transport"]["speed"] == 2.0

    replay_ctl.set("replay.speed", 0.0)  # "max": unpaced
    assert replay_ctl._engine.source.speed == 0.0

    # Off-menu rates are refused rather than silently clamped, so a stale page
    # cannot leave the replay running at a rate nothing on screen names.
    with pytest.raises(ValueError):
        replay_ctl.set("replay.speed", 7.5)


def test_status_line_names_the_playback_rate(replay_ctl):
    replay_ctl.set("replay.speed", 3.0)
    replay_ctl._engine.source.play()
    try:
        assert "3x" in replay_ctl.status()[0]["status.state"]
    finally:
        replay_ctl._engine.source.pause()
    replay_ctl.set("replay.speed", 1.0)
    replay_ctl._engine.source.play()
    try:
        state = replay_ctl.status()[0]["status.state"]
        assert "playing" in state and "1x" not in state  # real time is unmarked
    finally:
        replay_ctl._engine.source.pause()


def test_status_flags_track_the_pipeline(live_ctl):
    snap = live_ctl.snapshot()
    assert snap["flags"]["paused"] is False
    assert snap["flags"]["recording"] is False
    live_ctl.action("status.pause")
    assert live_ctl.snapshot()["flags"]["paused"] is True


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def test_setting_accum_retention_reaches_the_engine(replay_ctl):
    replay_ctl.set("view.accum_frames", 1234)
    assert replay_ctl._engine.accum_frames == 1234
    assert replay_ctl.snapshot()["values"]["view.accum_frames"] == 1234


def test_values_are_clamped_to_the_declared_range(replay_ctl):
    replay_ctl.set("view.accum_frames", 10**9)
    assert replay_ctl._engine.accum_frames == spec.CONTROLS_BY_ID[
        "view.accum_frames"].max
    replay_ctl.set("view.accum_frames", -5)
    assert replay_ctl._engine.accum_frames == 1


def test_seeking_moves_the_replay(replay_ctl):
    src = replay_ctl._engine.source
    target = src.duration_sec / 2.0
    replay_ctl.set("replay.position", target)
    assert src._seek_target_ns is not None


def test_play_pause_toggles(replay_ctl):
    src = replay_ctl._engine.source
    assert not src.playing
    replay_ctl.action("replay.play_pause")
    assert src.playing
    replay_ctl.action("replay.play_pause")
    assert not src.playing


def test_unknown_control_and_action_are_rejected(replay_ctl):
    with pytest.raises(KeyError):
        replay_ctl.set("view.nonsense", 1)
    with pytest.raises(KeyError):
        replay_ctl.action("view.nonsense")


def test_readouts_and_actions_are_not_settable(replay_ctl):
    with pytest.raises(ValueError):
        replay_ctl.set("status.rate", 5)
    with pytest.raises(ValueError):
        replay_ctl.set("replay.play_pause", True)


def test_controls_outside_this_run_are_refused(replay_ctl):
    """A replay has no recorder and no viewer; asking anyway must not 500."""
    with pytest.raises(ValueError):
        replay_ctl.action("record.toggle")
    with pytest.raises(ValueError):
        replay_ctl.set("view.point_size", 4.0)


def test_nan_is_not_written_into_the_pipeline(replay_ctl):
    with pytest.raises(ValueError):
        replay_ctl.set("view.accum_frames", float("nan"))
    with pytest.raises(ValueError):
        replay_ctl.set("view.accum_frames", "not a number")


# --------------------------------------------------------------------------- #
# The full surface: a viewer and a scorer attached
# --------------------------------------------------------------------------- #
@pytest.fixture
def full_ctl(replay_recording):
    ctl = full_controller(replay_recording)
    yield ctl
    ctl._engine.source.stop()


def test_controller_only_uses_the_real_viz_api():
    """FakeViz is allowed to stand in for VizApp only while it tells the truth.

    Every attribute the controller reaches for has to exist on the real class,
    or these tests would happily pass against an API that no longer exists.
    """
    gui = pytest.importorskip("open3d.visualization.gui")  # noqa: F841
    from rocklabel.live.viz.app import VizApp

    for name in vars(FakeViz):
        if name.startswith("__") or name == "calls":
            continue
        assert hasattr(VizApp, name), f"VizApp has no {name}() any more"


def test_every_settable_control_round_trips(full_ctl):
    """Set each control to a legal value; read it back through the snapshot."""
    probe = {
        "bool": lambda c, cur: not cur,
        "enum": lambda c, cur: c.choices[-1].value,
        "int": lambda c, cur: int(c.min + (c.max - c.min) // 3),
        "float": lambda c, cur: round(c.min + (c.max - c.min) / 3.0, 3),
    }
    values = full_ctl.snapshot()["values"]
    checked = 0
    for sec in full_ctl.schema()["sections"]:
        for c in sec["controls"]:
            control = spec.CONTROLS_BY_ID[c["id"]]
            if control.kind not in probe or control.id == "replay.position":
                continue
            # The model pickers are swept in the comparison tests instead:
            # writing one loads a checkpoint off disk, which is not something a
            # blind "set every control" pass should be doing.
            if control.choices_from:
                continue
            want = probe[control.kind](control, values.get(control.id))
            full_ctl.set(control.id, want)
            got = full_ctl.snapshot()["values"][control.id]
            assert got == pytest.approx(want) if isinstance(want, float) \
                else got == want, f"{control.id} did not stick"
            checked += 1
    assert checked >= 15, "the sweep is not covering the panel"


def test_writes_reach_the_objects_they_claim_to(full_ctl):
    viz, scorer = full_ctl._viz, full_ctl._scorer
    full_ctl.set("view.color_mode", "model")
    assert viz.color_mode == "model"
    full_ctl.set("view.show_mesh", False)
    assert viz._show_mesh is False
    full_ctl.set("model.threshold", 0.42)
    assert scorer.threshold == pytest.approx(0.42)
    full_ctl.set("region.z_min", -1.5)
    assert scorer.settings.z_min == pytest.approx(-1.5)
    full_ctl.set("crop.z_max", -0.25)   # the ingest crop, not the scoring band
    assert full_ctl._engine._crop.z_max == pytest.approx(-0.25)
    assert scorer.settings.z_max != pytest.approx(-0.25)
    full_ctl.set("model.enabled", False)
    assert scorer.settings.enabled is False

    full_ctl.action("view.reset_camera")
    assert ("reset_camera",) in viz.calls
    full_ctl.action("level.recalibrate")
    assert ("recalibrate_level",) in viz.calls
    full_ctl.action("model.clear")
    assert scorer.cleared == 1


def test_reflectivity_window_is_two_sliders_over_one_window(full_ctl):
    """The page sends one end at a time; the controller has to fold that into
    the pair the viewer holds without the ends crossing."""
    full_ctl.set("view.refl_min", 0.30)
    full_ctl.set("view.refl_max", 0.60)
    assert full_ctl._viz.reflectivity_range == pytest.approx((0.30, 0.60))
    values = full_ctl.snapshot()["values"]
    assert values["view.refl_min"] == pytest.approx(0.30)
    assert values["view.refl_max"] == pytest.approx(0.60)

    # dragging max below min moves max where it was asked and min out of its way
    full_ctl.set("view.refl_max", 0.10)
    lo, hi = full_ctl._viz.reflectivity_range
    assert hi == pytest.approx(0.10) and lo < hi

    full_ctl.action("view.refl_autofit")
    assert ("autofit_reflectivity",) in full_ctl._viz.calls


def test_a_write_waits_for_the_gui_thread_before_answering(full_ctl):
    """The HTTP response carries a snapshot, so it has to be taken *after* the
    change landed — otherwise the page is told the old value and flickers back.

    Modelled on the real thing: the queued callable runs on another thread a
    beat later, exactly as the Open3D event loop does.
    """
    import threading

    viz = full_ctl._viz
    delivered = []

    def deferred_post(fn):
        t = threading.Timer(0.05, lambda: (fn(), delivered.append(1)))
        t.daemon = True
        t.start()
        return True

    viz.post = deferred_post
    full_ctl.set("view.point_size", 6.0)
    assert delivered, "set() answered before the GUI thread had run the change"
    assert full_ctl.snapshot()["values"]["view.point_size"] == 6.0


def test_a_write_before_the_window_exists_does_not_hang(full_ctl):
    """VizApp.post refuses work until run() builds the window. A write then has
    nothing to wait for, and must say so immediately rather than time out."""
    full_ctl._viz.post = lambda fn: False
    start = time.perf_counter()
    full_ctl.set("view.point_size", 5.0)
    assert time.perf_counter() - start < 0.2


def test_region_bounds_are_exact_not_rounded(full_ctl):
    """The floor band is a number you arrive with; -1.5 must stay -1.5."""
    full_ctl.set("region.z_min", -1.5)
    full_ctl.set("region.z_max", -0.5)
    v = full_ctl.snapshot()["values"]
    assert v["region.z_min"] == -1.5 and v["region.z_max"] == -0.5


def test_scoring_region_can_be_reanchored_to_the_floor_live(full_ctl):
    """The ingest Crop card already had this switch, but it did not change the
    separate band the model is fed. A robot changing height could therefore
    leave model scoring behind even while its displayed crop followed ground."""
    full_ctl.set("region.floor_relative", True)
    full_ctl.set("region.z_min", -0.10)
    full_ctl.set("region.z_max", 0.60)
    values = full_ctl.snapshot()["values"]
    assert values["region.floor_relative"] is True
    assert full_ctl._scorer.settings.floor_relative is True
    assert values["region.z_min"] == pytest.approx(-0.10)
    assert values["region.z_max"] == pytest.approx(0.60)


def test_model_readouts_render_the_scorer_numbers(full_ctl):
    status = full_ctl.snapshot()["status"]
    assert "4,210" in status["model.map"] and "128" in status["model.map"]
    assert "41 ms" in status["model.pass"]
    assert status["model.region"].startswith("18.4k")


def test_scoring_off_is_said_plainly(full_ctl):
    full_ctl.set("model.enabled", False)
    assert full_ctl.snapshot()["status"]["model.map"] == "scoring off"


# --------------------------------------------------------------------------- #
# The extra views: overhead map, confidence histogram, trends
# --------------------------------------------------------------------------- #
def _fused_engine(seconds: float = 1.2):
    """A live engine that has actually fused some surface to look down on."""
    engine, cfg = _live_engine()
    engine.start()
    time.sleep(seconds)
    engine.stop()
    return engine, cfg


@pytest.fixture(scope="module")
def fused():
    return _fused_engine()


def test_height_raster_crops_to_what_was_measured(fused):
    engine, cfg = fused
    raster = engine.surface.height_raster(256)
    assert raster is not None
    h, w = raster.shape
    # The grid is 20 m square at 5 cm — 400x400 cells — and a scan fills a
    # fraction of it. Sending the whole lattice would be mostly empty bytes.
    assert h < 400 and w < 400
    assert np.isfinite(raster.heights).any()
    assert raster.cell >= cfg.grid.cell_size


def test_height_raster_downsamples_to_the_requested_side(fused):
    engine, _cfg = fused
    small = engine.surface.height_raster(48)
    assert small is not None
    assert max(small.shape) <= 48
    # Downsampling coarsens the cell; the world extent it covers must not shrink
    # to match, or the overhead view would silently crop as you zoom out.
    full = engine.surface.height_raster(256)
    assert small.cell > full.cell
    span_small = small.shape[1] * small.cell
    span_full = full.shape[1] * full.cell
    assert span_small >= span_full - full.cell


def test_height_raster_keeps_peaks_rather_than_averaging_them_away():
    """Block reduction takes the max: a rock averaged with the floor around it
    is a rock that vanishes when the view zooms out."""
    from rocklabel.live.config import AppConfig as _Cfg
    from rocklabel.live.surfaces.kalman_heightmap import KalmanHeightmap as _KH

    cfg = _Cfg()
    surface = _KH(cfg)
    # A flat floor at z=0 with one tall, narrow spike.
    xs, ys = np.meshgrid(np.linspace(-1, 1, 120), np.linspace(-1, 1, 120))
    flat = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    spike = np.tile(np.array([[0.0, 0.0, 1.0]]), (60, 1))
    for _ in range(6):                       # settle the Kalman variance
        surface.add_points(flat)
        surface.add_points(spike)

    coarse = surface.height_raster(8)
    assert coarse is not None
    assert np.nanmax(coarse.heights) > 0.9, "the spike was averaged away"


def test_empty_surface_has_no_raster():
    from rocklabel.live.config import AppConfig as _Cfg
    from rocklabel.live.surfaces.kalman_heightmap import KalmanHeightmap as _KH

    assert _KH(_Cfg()).height_raster() is None


def test_encoded_raster_keeps_unmeasured_ground_distinct(fused):
    """Level 0 is reserved for 'never measured'. Without that reservation the
    page paints a phantom floor everywhere the sensor did not look."""
    from rocklabel.live.webui import scene as scene_mod

    engine, _cfg = fused
    raster = engine.surface.height_raster(64)
    payload = scene_mod.encode_raster(raster)
    levels = np.frombuffer(base64.b64decode(payload["data"]), dtype=np.uint8)
    assert levels.size == payload["w"] * payload["h"]

    known = np.isfinite(raster.heights).ravel()
    assert np.all(levels[~known] == 0)
    assert np.all(levels[known] >= 1)
    assert payload["z_min"] <= payload["z_max"]


def test_encoded_raster_of_a_flat_surface_does_not_divide_by_zero():
    from rocklabel.live.surfaces.base import HeightRaster
    from rocklabel.live.webui import scene as scene_mod

    flat = HeightRaster(heights=np.full((4, 4), 0.25, np.float32),
                        x0=0.0, y0=0.0, cell=0.1)
    payload = scene_mod.encode_raster(flat)
    levels = np.frombuffer(base64.b64decode(payload["data"]), dtype=np.uint8)
    assert set(levels.tolist()) == {128}
    assert payload["z_min"] == payload["z_max"] == pytest.approx(0.25)


def test_scene_payload_has_what_the_page_draws(fused):
    engine, cfg = fused
    ctl = LiveController(cfg, engine)
    sc = ctl.scene()
    assert set(sc) >= {"bev", "detections", "sensor", "history", "histogram"}
    assert sc["bev"]["w"] > 0 and sc["bev"]["h"] > 0
    assert {"x", "y", "z", "yaw_deg"} <= set(sc["sensor"])
    # No scorer: nothing to detect, nothing to plot a distribution of.
    assert sc["detections"] == {"rows": [], "total": 0, "shown": 0}
    assert sc["histogram"] is None
    # No scorer, but the ingest crop is still a region, and it is the one the
    # ring on the map has to describe. Off entirely, there is nothing to draw.
    assert sc["region"]["z_min"] == pytest.approx(cfg.crop.z_min)
    cfg.crop.enabled = False
    assert "region" not in ctl.scene()
    cfg.crop.enabled = True


def test_detections_and_histogram_come_from_the_scorer(fused):
    """With a scorer the map gets marks and the histogram gets a distribution
    covering every center, not just the ones above the threshold."""
    from rocklabel.live.webui import scene as scene_mod

    engine, cfg = fused
    scorer = FakeScorer()
    rng = np.random.default_rng(0)
    centers = rng.uniform(-2, 2, size=(400, 3))
    # Two lobes, both clear of the 0.89 threshold, so the expected counts do not
    # depend on where exactly the draws land.
    probs = np.concatenate([rng.uniform(0.0, 0.3, 300),
                            rng.uniform(0.9, 1.0, 100)]).astype(np.float32)
    scorer.attach_result(centers, probs)

    ctl = LiveController(cfg, engine, scorer=scorer)
    sc = ctl.scene()
    assert sc["detections"]["total"] == 100        # only >= threshold 0.89...
    hist = sc["histogram"]
    assert hist["total"] == 400, "the histogram must cover every scored center"
    assert hist["above"] == sc["detections"]["total"]
    assert sum(hist["counts"]) == 400
    assert len(hist["edges"]) == len(hist["counts"]) + 1
    assert hist["threshold"] == pytest.approx(scorer.threshold)
    assert sc["region"]["range_max"] == pytest.approx(scorer.settings.range_max)
    _ = scene_mod  # imported for the constants asserted on below


def test_detections_are_thinned_by_dropping_the_weakest(fused):
    from rocklabel.live.webui import scene as scene_mod

    engine, cfg = fused
    scorer = FakeScorer()
    n = scene_mod.MAX_DETECTIONS + 500
    centers = np.zeros((n, 3))
    probs = np.linspace(0.9, 1.0, n).astype(np.float32)
    scorer.attach_result(centers, probs)

    payload = scene_mod.detections_payload(scorer)
    assert payload["total"] == n
    assert payload["shown"] == scene_mod.MAX_DETECTIONS
    kept = sorted(row[3] for row in payload["rows"])
    # The survivors are exactly the top MAX_DETECTIONS — if marks must be
    # dropped, the marginal ones are the ones to lose.
    cutoff = float(np.sort(probs)[n - scene_mod.MAX_DETECTIONS])
    assert min(kept) == pytest.approx(cutoff, abs=1e-3)


# --------------------------------------------------------------------------- #
# Comparing two models in two windows
#
# The loader is stubbed out: what is under test is the wiring — which settings
# the two models share, what the panel says, and what happens when a window
# closes — none of which needs torch, a GPU, or a real checkpoint on disk.
# --------------------------------------------------------------------------- #
def _fake_checkpoint(tmp_path, name: str = "second/best.pt") -> str:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really weights")
    return str(path)


def _stub_loader(ctl, **kw):
    """Make the comparison build stand-in scorers instead of loading torch.

    Keeps the real sharing rule: the second scorer is handed the first one's
    settings object, exactly as :meth:`Comparison._build` does.
    """
    built = []

    def build(checkpoint):
        anchor = ctl._compare.scorer_a
        scorer = FakeScorer(checkpoint=checkpoint,
                            settings=anchor.settings if anchor else None, **kw)
        built.append(scorer)
        return scorer

    ctl._compare._build = build
    return built


def _settled(ctl, timeout: float = 3.0) -> str:
    """Wait for a load to finish — they run on a worker thread by design."""
    end = time.time() + timeout
    while time.time() < end and ctl._compare.state == "loading":
        time.sleep(0.01)
    return ctl._compare.state


def test_the_compare_card_is_offered_whenever_a_model_is_loaded(full_ctl):
    ids = {c["id"] for s in full_ctl.schema()["sections"] for c in s["controls"]}
    assert {"compare.model_a", "compare.model_b", "compare.open",
            "compare.close"} <= ids


def test_opening_a_comparison_runs_a_second_model_in_a_second_window(
        full_ctl, tmp_path):
    built = _stub_loader(full_ctl)
    ck = _fake_checkpoint(tmp_path)
    full_ctl.set("compare.model_b", ck)
    assert not full_ctl._compare.open, "picking a model must not open by itself"

    full_ctl.action("compare.open")
    assert _settled(full_ctl) == "open"
    assert full_ctl._compare.scorer_b is built[0]
    assert built[0].started == 1, "the second model has to actually score"
    peer = full_ctl._viz.peer
    assert peer is not None and "compare" in peer.title
    assert peer._scorer is built[0]
    status = full_ctl.snapshot()["status"]
    assert "window 2" in status["compare.state"]
    assert "4,210" in status["compare.map"]      # the second model's own numbers


def test_the_two_windows_cannot_disagree_about_the_settings(full_ctl, tmp_path):
    """Not 'kept in sync' — the same object. A comparison where the two halves
    can drift apart is not evidence of anything."""
    _stub_loader(full_ctl)
    full_ctl.set("compare.model_b", _fake_checkpoint(tmp_path))
    full_ctl.action("compare.open")
    assert _settled(full_ctl) == "open"

    a, b = full_ctl._compare.scorer_a, full_ctl._compare.scorer_b
    assert b.settings is a.settings
    full_ctl.set("region.z_min", -1.25)
    full_ctl.set("outline.min_points", 9)
    assert b.settings.z_min == pytest.approx(-1.25)
    assert b.settings.cluster_min_points == 9


def test_one_threshold_drives_both_models(full_ctl, tmp_path):
    _stub_loader(full_ctl)
    full_ctl.set("compare.model_b", _fake_checkpoint(tmp_path))
    full_ctl.action("compare.open")
    assert _settled(full_ctl) == "open"
    full_ctl.set("model.threshold", 0.42)
    assert full_ctl._compare.scorer_a.threshold == pytest.approx(0.42)
    assert full_ctl._compare.scorer_b.threshold == pytest.approx(0.42)


def test_clearing_predictions_clears_both(full_ctl, tmp_path):
    _stub_loader(full_ctl)
    full_ctl.set("compare.model_b", _fake_checkpoint(tmp_path))
    full_ctl.action("compare.open")
    assert _settled(full_ctl) == "open"
    full_ctl.action("model.clear")
    assert full_ctl._compare.scorer_a.cleared == 1
    assert full_ctl._compare.scorer_b.cleared == 1


def test_closing_the_comparison_stops_the_second_model(full_ctl, tmp_path):
    _stub_loader(full_ctl)
    full_ctl.set("compare.model_b", _fake_checkpoint(tmp_path))
    full_ctl.action("compare.open")
    assert _settled(full_ctl) == "open"
    second = full_ctl._compare.scorer_b

    full_ctl.action("compare.close")
    assert full_ctl._compare.open is False
    assert second.stopped == 1
    assert full_ctl._viz.peer.closed is True
    status = full_ctl.snapshot()["status"]
    assert status["compare.state"] == "closed"
    assert status["compare.map"] == "—"
    # The first model is untouched — closing a comparison is not a teardown.
    assert full_ctl._scorer.stopped == 0


def test_the_second_window_closing_itself_reaches_the_panel(full_ctl, tmp_path):
    """Someone hits the X on window 2. The panel has to notice, or it goes on
    reporting a model that is not running."""
    _stub_loader(full_ctl)
    full_ctl.set("compare.model_b", _fake_checkpoint(tmp_path))
    full_ctl.action("compare.open")
    assert _settled(full_ctl) == "open"

    full_ctl._viz.peer.close_window()          # the window's own close handler
    assert full_ctl._compare.open is False
    assert full_ctl.snapshot()["status"]["compare.state"] == "closed"


def test_picking_a_new_model_swaps_the_window_that_shows_it(full_ctl, tmp_path):
    built = _stub_loader(full_ctl)
    full_ctl.set("compare.model_b", _fake_checkpoint(tmp_path, "first/best.pt"))
    full_ctl.action("compare.open")
    assert _settled(full_ctl) == "open"

    full_ctl.set("compare.model_b", _fake_checkpoint(tmp_path, "other/best.pt"))
    assert _settled(full_ctl) == "open"
    assert len(built) == 2
    assert full_ctl._compare.scorer_b is built[1]
    assert built[0].stopped == 1, "the model it replaced must stop scoring"
    assert full_ctl._viz.peer._scorer is built[1]


def test_swapping_window_1_rebinds_everything_that_held_the_old_model(
        full_ctl, tmp_path):
    built = _stub_loader(full_ctl)
    first = full_ctl._scorer
    full_ctl.set("compare.model_a", _fake_checkpoint(tmp_path, "swapped/best.pt"))
    assert _settled(full_ctl) == "closed"      # no comparison was open
    assert full_ctl._scorer is built[0] is not first
    assert first.stopped == 1
    assert full_ctl._viz._scorer is built[0]
    assert full_ctl.snapshot()["values"]["compare.model_a"].endswith(
        "swapped/best.pt")


def test_a_checkpoint_that_is_not_there_is_refused(full_ctl):
    _stub_loader(full_ctl)
    with pytest.raises(ValueError):
        full_ctl.set("compare.model_b", "/nope/does/not/exist.pt")
    with pytest.raises(ValueError):
        full_ctl.action("compare.open")        # nothing selected yet


def test_a_load_that_fails_is_reported_rather_than_swallowed(full_ctl, tmp_path):
    """A checkpoint the training code cannot read used to be a traceback in a
    terminal nobody is watching. It has to land on the page."""
    def explode(checkpoint):
        raise KeyError("generator")

    full_ctl._compare._build = explode
    full_ctl.set("compare.model_b", _fake_checkpoint(tmp_path))
    full_ctl.action("compare.open")
    assert _settled(full_ctl) == "error"
    snap = full_ctl.snapshot()
    assert "KeyError" in snap["status"]["compare.state"]
    assert snap["flags"]["compare_error"] is True
    assert full_ctl._compare.open is False


def test_checkpoints_that_cannot_be_compared_say_so(full_ctl, tmp_path):
    """Two models trained on different scan windows are fed one shared window,
    so one of them is seeing a density it never trained on."""
    _stub_loader(full_ctl, window_s=0.05)
    full_ctl.set("compare.model_b", _fake_checkpoint(tmp_path))
    full_ctl.action("compare.open")
    assert _settled(full_ctl) == "open"
    snap = full_ctl.snapshot()
    assert "scan window" in snap["status"]["compare.caveat"]
    assert snap["flags"]["compare_caveat"] is True


def test_the_model_pickers_offer_what_is_loaded_now(full_ctl, tmp_path):
    _stub_loader(full_ctl)
    ck = _fake_checkpoint(tmp_path, "picked/best.pt")
    full_ctl.set("compare.model_b", ck)
    full_ctl.action("compare.open")
    assert _settled(full_ctl) == "open"

    choices = full_ctl.checkpoint_choices()
    values = [c["value"] for c in choices]
    assert "" in values, "there has to be a way to pick no second model"
    assert ck in values, "a picker must show what is already running"
    # The cap, plus the "none" row and the two slots' own checkpoints.
    assert len(choices) <= 3 + _control_max_choices(), "the picker is unbounded"
    assert all(c["label"] for c in choices)


def _control_max_choices() -> int:
    from rocklabel.live.webui.control import MAX_CHECKPOINT_CHOICES

    return MAX_CHECKPOINT_CHOICES


def test_a_run_without_a_model_has_no_compare_card(replay_ctl):
    sections = {s["id"] for s in replay_ctl.schema()["sections"]}
    assert "compare" not in sections


# --------------------------------------------------------------------------- #
# Rock outlines: detections grouped into objects
# --------------------------------------------------------------------------- #
def _two_rocks_and_a_speck(rng=None):
    """Two tight clumps of detections, plus one lone point off on its own."""
    rng = rng or np.random.default_rng(4)
    a = rng.normal(0.0, 0.05, size=(20, 3)) + np.array([1.0, 1.0, 0.0])
    b = rng.normal(0.0, 0.05, size=(20, 3)) + np.array([-2.0, 0.5, 0.0])
    speck = np.array([[5.0, 5.0, 0.0]])
    centers = np.vstack([a, b, speck])
    return centers, np.full(len(centers), 0.97, np.float32)


def _outline_ctl(fused):
    engine, cfg = fused
    scorer = FakeScorer()
    scorer.attach_result(*_two_rocks_and_a_speck())
    ctl = LiveController(cfg, engine, scorer=scorer)
    ctl.set("outline.link_m", 0.3)
    ctl.set("outline.min_points", 5)
    return ctl, scorer


def test_outline_settings_reach_the_scorer(fused):
    ctl, scorer = _outline_ctl(fused)
    ctl.set("outline.grouping", "robust")
    ctl.set("outline.core_points", 6)
    ctl.set("outline.max_diameter_m", 0.7)
    ctl.set("outline.max_height_m", 0.4)
    ctl.set("outline.min_mean_prob", 0.92)
    ctl.set("outline.contour_m", 0.18)
    ctl.set("outline.padding_m", 0.035)
    st = scorer.settings
    assert st.cluster_grouping == "robust"
    assert scorer.settings.cluster_link_m == pytest.approx(0.3)
    assert st.cluster_core_points == 6
    assert st.cluster_min_points == 5
    assert st.cluster_max_diameter_m == pytest.approx(0.7)
    assert st.cluster_max_height_m == pytest.approx(0.4)
    assert st.cluster_min_mean_prob == pytest.approx(0.92)
    assert st.cluster_contour_m == pytest.approx(0.18)
    assert st.cluster_padding_m == pytest.approx(0.035)
    values = ctl.snapshot()["values"]
    assert values["outline.grouping"] == "robust"
    assert values["outline.link_m"] == pytest.approx(0.3)
    assert values["outline.core_points"] == 6
    assert values["outline.min_points"] == 5
    assert values["outline.max_diameter_m"] == pytest.approx(0.7)
    assert values["outline.max_height_m"] == pytest.approx(0.4)
    assert values["outline.min_mean_prob"] == pytest.approx(0.92)
    assert values["outline.contour_m"] == pytest.approx(0.18)
    assert values["outline.padding_m"] == pytest.approx(0.035)


def test_the_map_draws_polygons_when_the_display_asks_for_them(fused):
    """Mode 2 is the whole feature: the same detections, delivered as shapes."""
    ctl, _scorer = _outline_ctl(fused)
    ctl.set("model.display", 2)
    sc = ctl.scene()
    assert sc["display"] == 2
    rocks = sc["rocks"]
    assert rocks["total"] == 2, "two clumps should make two rocks"
    assert rocks["noise_points"] == 1, "the lone point is not a rock"
    for row in rocks["rows"]:
        assert len(row["poly"]) >= 3          # a real, drawable ring
        assert row["n"] >= 5 and row["area"] > 0
        assert 0.0 <= row["prob"] <= 1.0
    # Biggest first, so a truncated table keeps the rocks that matter.
    areas = [r["area"] for r in rocks["rows"]]
    assert areas == sorted(areas, reverse=True)


def test_the_noise_gate_is_the_knob_that_decides(fused):
    """Raising 'Min points' past a clump's size makes that outline disappear —
    and the dropped count says so, so the setting can be walked back."""
    ctl, _scorer = _outline_ctl(fused)
    assert ctl.scene()["rocks"]["total"] == 2
    ctl.set("outline.min_points", 100)
    rocks = ctl.scene()["rocks"]
    assert rocks["total"] == 0
    assert rocks["noise_points"] == 41 and rocks["noise_groups"] == 3


def test_outline_readouts_report_what_was_drawn_and_dropped(fused):
    ctl, _scorer = _outline_ctl(fused)
    status = ctl.snapshot()["status"]
    assert "2 rocks" in status["outline.rocks"]
    assert "1 of 41 detections" in status["outline.noise"]


def test_the_display_mode_works_without_an_open3d_window(fused):
    """--headless --web-ui has no viewer to hold the mode, but the overhead map
    is still a viewer — so the control has to exist and stick."""
    ctl, _scorer = _outline_ctl(fused)
    ids = {c["id"] for s in ctl.schema()["sections"] for c in s["controls"]}
    assert "model.display" in ids and "viewer" not in ctl.capabilities
    ctl.set("model.display", 2)
    assert ctl.snapshot()["values"]["model.display"] == 2
    ctl.set("model.display", 0)
    assert ctl.scene()["display"] == 0


def test_outlines_are_computed_once_per_pass_not_once_per_poll(fused):
    """The page asks four times a second; clustering a full map is real work."""
    from rocklabel.live import clusters

    ctl, scorer = _outline_ctl(fused)
    calls = []
    real = clusters.find_rocks

    def counted(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    clusters.find_rocks = counted
    try:
        for _ in range(5):
            ctl.snapshot()
            ctl.scene()
        assert len(calls) == 1, "the cached outlines were recomputed"
        scorer.attach_result(*_two_rocks_and_a_speck())   # a new pass landed
        ctl.scene()
        assert len(calls) == 2, "a new scoring pass must re-outline"
    finally:
        clusters.find_rocks = real


def test_history_is_rate_limited_and_bounded():
    from rocklabel.live.webui.scene import History

    hist = History(maxlen=5)
    for i in range(20):
        hist._last = 0.0            # pretend a second passed between samples
        hist.sample({"detections": i, "in_region": i, "pass_ms": i, "rate": i})
    payload = hist.payload()
    assert len(payload["t"]) == 5
    assert [s["id"] for s in payload["series"]] == list(History.SERIES)
    assert payload["series"][0]["values"] == [15, 16, 17, 18, 19]

    # Two samples in the same instant is one sample: polling faster does not
    # buy more history, only a shorter one.
    hist.clear()
    hist.sample({"detections": 1})
    hist.sample({"detections": 2})
    assert len(hist.payload()["t"]) == 1


def test_history_is_cleared_when_the_map_it_described_is(fused):
    engine, cfg = fused
    ctl = LiveController(cfg, engine)
    for _ in range(3):
        ctl._history._last = 0.0
        ctl.snapshot()
    assert len(ctl.scene()["history"]["t"]) >= 2
    ctl.action("view.reset_surface")
    assert ctl.scene()["history"]["t"] == []


def test_scene_endpoint_is_served(client):
    body = client.get("/api/scene").get_json()
    assert set(body) >= {"bev", "detections", "sensor", "history", "histogram"}


def test_charts_js_is_borrowed_not_copied(client):
    """The trend plots are the dashboard's lineChart, served from its own file."""
    assert client.get("/theme/charts.js").status_code == 200
    assert client.get("/theme/app.js").status_code == 404
    assert client.get("/theme/../server.py").status_code in (400, 404)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def test_page_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "live.js" in body and "live.css" in body
    # Styling is self-contained now: this page must not reach into the
    # dashboard's stylesheet, which the other surface owns.
    assert "/theme/app.css" not in body


def test_theme_route_serves_only_the_shared_chart_code(client):
    """The page borrows the dashboard's chart primitives, not its stylesheet."""
    assert client.get("/theme/charts.js").status_code == 200
    # live.css carries this panel's own tokens; app.css belongs to `rocklabel dash`.
    assert client.get("/theme/app.css").status_code == 404
    # Not a general-purpose file server for the package.
    assert client.get("/theme/app.js").status_code == 404


def test_schema_and_state_endpoints(client):
    schema = client.get("/api/schema").get_json()
    assert schema["mode"] == "replay"
    assert schema["duration_sec"] > 0
    assert [s["id"] for s in schema["sections"]][0] == "status"

    state = client.get("/api/state").get_json()
    assert set(state) >= {"values", "status", "flags", "transport"}


def test_set_endpoint_applies_and_echoes_state(client):
    res = client.post("/api/set", json={"key": "view.accum_frames", "value": 640})
    assert res.status_code == 200
    assert res.get_json()["values"]["view.accum_frames"] == 640


def test_set_endpoint_errors_are_json_not_500(client):
    assert client.post("/api/set", json={"key": "nope", "value": 1}).status_code == 404
    bad = client.post("/api/set", json={"key": "view.accum_frames"})
    assert bad.status_code == 400 and "error" in bad.get_json()
    refused = client.post("/api/set", json={"key": "view.point_size", "value": 3})
    assert refused.status_code == 400 and "error" in refused.get_json()


def test_action_endpoint(client):
    res = client.post("/api/action", json={"name": "replay.play_pause"})
    assert res.status_code == 200
    assert res.get_json()["transport"]["playing"] is True
    assert client.post("/api/action", json={"name": "nope"}).status_code == 404
    bad = client.post("/api/action", json={"name": "replay.restart", "args": 5})
    assert bad.status_code == 400
