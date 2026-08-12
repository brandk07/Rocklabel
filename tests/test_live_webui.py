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


def _replay_controller(path: str, fuse_sec: float = 0.0) -> LiveController:
    """A controller over a replayed recording.

    ``fuse_sec`` runs the pipeline for that long first, so the surface has
    something in it — needed by anything that looks at the overhead view, and
    pointless for the tests that only poke at controls.
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
    return LiveController(cfg, engine)


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

    def __init__(self, scorer=None) -> None:
        self.color_mode = "height"
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

    # Runs inline and reports "queued", which is what a real window does once
    # it exists. Returning False is the not-yet-built case the controller must
    # not block on — see test_a_write_before_the_window_exists_does_not_hang.
    def post(self, fn):
        fn()
        return True

    def set_color_mode(self, mode):
        self.color_mode = mode
        self.calls.append(("color_mode", mode))

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

    def set_threshold(self, value):
        self._scorer.threshold = float(value)

    def set_score_setting(self, name, value):
        setattr(self._scorer.settings, name, value)

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

    def __init__(self) -> None:
        from rocklabel.live.scoring import ScoreSettings

        self.settings = ScoreSettings(window_sec=0.0)
        self.threshold = 0.89
        self.cleared = 0
        self._centers = np.empty((0, 3))
        self._probs = np.empty((0,), np.float32)

    def attach_result(self, centers, probs) -> None:
        """Stand in for a completed scoring pass."""
        self._centers = np.asarray(centers, float)
        self._probs = np.asarray(probs, np.float32)

    def detections(self):
        keep = self._probs >= self.threshold
        return self._centers[keep], self._probs[keep]

    def all_probs(self):
        return self._probs

    def clear_map(self) -> None:
        self.cleared += 1

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
    ctl = _replay_controller(path, fuse_sec=fuse_sec)
    ctl._scorer = FakeScorer()
    ctl._viz = FakeViz(ctl._scorer)
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
            assert c.choices, f"{c.id} is an enum with no choices"


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
    full_ctl.set("model.enabled", False)
    assert scorer.settings.enabled is False

    full_ctl.action("view.reset_camera")
    assert ("reset_camera",) in viz.calls
    full_ctl.action("level.recalibrate")
    assert ("recalibrate_level",) in viz.calls
    full_ctl.action("model.clear")
    assert scorer.cleared == 1


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
    assert "region" not in sc


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
    assert "live.js" in body and "/theme/app.css" in body


def test_theme_route_serves_the_dashboard_stylesheet(client):
    assert client.get("/theme/app.css").status_code == 200
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
