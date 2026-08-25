"""Tests for running two models side by side (:mod:`rocklabel.live.compare`).

Two things are pinned here, and they are the two that make a comparison mean
anything:

* **the windows cannot drift apart** — every public setting written to one
  viewer is applied to the other, whichever route it came in by (a keypress, a
  panel widget, the browser page);
* **the models cannot drift apart** — both scorers read one settings object and
  one threshold, so the only difference left between them is the checkpoint.

No window is ever created: :class:`VizApp` builds its state in ``__init__`` and
only touches the renderer in :meth:`run`, so the mirroring can be driven and
checked headlessly. The loading side is stubbed — a real checkpoint would drag
in torch, a GPU and a training run.
"""

from __future__ import annotations

import time

import pytest

from rocklabel.live.compare import Comparison, short_name
from rocklabel.live.config import AppConfig
from rocklabel.live.pipeline import IngestEngine
from rocklabel.live.scoring import ScoreSettings
from rocklabel.live.sources.simulated import SimulatedSource
from rocklabel.live.surfaces.kalman_heightmap import KalmanHeightmap


@pytest.fixture
def engine():
    cfg = AppConfig()
    cfg.slam.enabled = False
    cfg.source.sim_points_per_sec = 5_000
    return IngestEngine(SimulatedSource(cfg), KalmanHeightmap(cfg), cfg), cfg


class StubScorer:
    """A scorer with no model behind it: settings, a threshold, a lifecycle."""

    model_name = "stub"
    center_spacing_m = 0.1

    def __init__(self, checkpoint="a/best.pt", settings=None, window_s=0.0):
        self.checkpoint = checkpoint
        self.settings = settings or ScoreSettings(window_sec=0.0)
        self.threshold = 0.5
        self.tuned_threshold = 0.5
        self.frame_window_s = window_s
        self.version = 0
        self.started = self.stopped = self.cleared = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def clear_map(self):
        self.cleared += 1

    def detections(self):
        return None

    def status_dict(self):
        return {"enabled": True, "ready": False, "error": None}


def _settle(comparison, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end and comparison.state == "loading":
        time.sleep(0.01)
    return comparison.state


def _stub_loads(comparison, **kw):
    made = []

    def build(checkpoint):
        s = StubScorer(checkpoint, settings=comparison.scorer_a.settings, **kw)
        made.append(s)
        return s

    comparison._build = build
    return made


# --------------------------------------------------------------------------- #
# The two windows
# --------------------------------------------------------------------------- #
@pytest.fixture
def pair(engine):
    """Two viewers over one engine, paired the way a comparison pairs them."""
    o3d_gui = pytest.importorskip("open3d.visualization.gui")  # noqa: F841
    from rocklabel.live.viz.app import VizApp

    eng, cfg = engine
    a = VizApp(cfg, eng, scorer=StubScorer(), web_ui=True)
    b = VizApp(cfg, eng, scorer=StubScorer("b/best.pt", settings=a._scorer.settings),
               web_ui=True, secondary=True)
    a.set_peer(b)
    return a, b


def test_display_settings_reach_the_other_window(pair):
    a, b = pair
    a.set_layer("_show_mesh", False)
    a.set_layer("_show_accum", False)
    a.set_point_size(6.0)
    a.set_color_mode("reflectivity")
    a.set_crop_view(False)
    a.set_model_display(2)
    assert (b._show_mesh, b._show_accum) == (False, False)
    assert b.point_size == pytest.approx(6.0)
    assert b.color_mode == "reflectivity"
    assert b._crop_view is False
    assert b._model_display == 2


def test_mirroring_works_in_both_directions(pair):
    """The second window has a keyboard too."""
    a, b = pair
    b.set_layer("_show_points", False)
    assert a._show_points is False
    b.set_reflectivity_range(0.25, 0.75)
    assert a.reflectivity_range == pytest.approx((0.25, 0.75))


def test_a_mirrored_write_does_not_bounce_back_forever(pair):
    """a -> b -> a -> … is the obvious way to write this wrong."""
    a, b = pair
    a.set_color_mode("height")           # completes at all = no recursion
    assert (a.color_mode, b.color_mode) == ("height", "height")


def test_the_threshold_is_shared_across_the_pair(pair):
    a, b = pair
    a.set_threshold(0.77)
    assert b._scorer.threshold == pytest.approx(0.77)


def test_clearing_predictions_clears_both_windows(pair):
    a, b = pair
    a.clear_predictions()
    assert a._scorer.cleared == 1 and b._scorer.cleared == 1


def test_the_second_window_copies_the_first_ones_view(pair):
    a, b = pair
    a.set_point_size(7.0)
    a.set_layer("_show_box", False)
    a.set_model_display(1)
    fresh = type(a)(a._cfg, a._engine, scorer=b._scorer, web_ui=True,
                    secondary=True)
    fresh.copy_view_from(a)
    assert fresh.point_size == pytest.approx(7.0)
    assert fresh._show_box is False
    assert fresh._model_display == 1
    assert fresh._color_mode == a._color_mode


def test_swapping_the_model_drops_what_the_old_one_drew(pair):
    a, _b = pair
    a._accum_model_key = ("stale",)
    a._rocks_key = ("stale",)
    a.set_scorer(StubScorer("swapped/best.pt"))
    assert a._accum_model_key is None and a._rocks_key is None
    assert a._scorer.checkpoint == "swapped/best.pt"


# --------------------------------------------------------------------------- #
# The two models
# --------------------------------------------------------------------------- #
def test_the_second_model_shares_the_first_ones_settings(engine, tmp_path):
    eng, cfg = engine
    first = StubScorer()
    comparison = Comparison(cfg, eng, scorer=first)
    _stub_loads(comparison)
    ck = tmp_path / "b.pt"
    ck.write_bytes(b"x")

    comparison.select("b", str(ck))
    comparison.open_comparison()
    assert _settle(comparison) == "open"
    assert comparison.scorer_b.settings is first.settings

    # A region write lands on one object, so there is nothing to keep in sync.
    first.settings.z_min = -2.25
    assert comparison.scorer_b.settings.z_min == pytest.approx(-2.25)


def test_a_second_model_starts_at_the_first_ones_threshold(engine, tmp_path):
    eng, cfg = engine
    first = StubScorer()
    first.threshold = 0.91
    comparison = Comparison(cfg, eng, scorer=first)
    _stub_loads(comparison)
    ck = tmp_path / "b.pt"
    ck.write_bytes(b"x")
    comparison.select("b", str(ck))
    comparison.open_comparison()
    assert _settle(comparison) == "open"
    assert comparison.scorer_b.threshold == pytest.approx(0.91)


def test_loading_the_same_checkpoint_twice_is_not_a_reload(engine, tmp_path):
    eng, cfg = engine
    comparison = Comparison(cfg, eng, scorer=StubScorer())
    made = _stub_loads(comparison)
    ck = tmp_path / "b.pt"
    ck.write_bytes(b"x")
    comparison.select("b", str(ck))
    comparison.open_comparison()
    assert _settle(comparison) == "open"
    comparison.load("b", str(ck))
    assert _settle(comparison) == "open"
    assert len(made) == 1, "the same weights were read off disk twice"


def test_without_a_window_the_second_model_still_runs(engine, tmp_path):
    """--headless --web-ui: nothing to draw, but the numbers are the point."""
    eng, cfg = engine
    comparison = Comparison(cfg, eng, scorer=StubScorer())
    _stub_loads(comparison)
    ck = tmp_path / "b.pt"
    ck.write_bytes(b"x")
    comparison.select("b", str(ck))
    comparison.open_comparison()
    assert _settle(comparison) == "open"
    assert comparison.scorer_b.started == 1
    assert comparison.status()["windowed"] is False


def test_a_comparison_needs_a_first_model(engine):
    eng, cfg = engine
    comparison = Comparison(cfg, eng, scorer=None)
    comparison.selected_b = "whatever.pt"
    with pytest.raises(ValueError):
        comparison.open_comparison()


def test_window_one_cannot_be_emptied(engine):
    eng, cfg = engine
    comparison = Comparison(cfg, eng, scorer=StubScorer())
    with pytest.raises(ValueError):
        comparison.select("a", "")


def test_short_name_keeps_the_run_not_just_the_file():
    """Every checkpoint in this project is called best.pt."""
    assert short_name("training/experiments/sweep/arm/loro_VB3/best.pt") \
        == "loro_VB3/best.pt"
    assert short_name("") == ""
