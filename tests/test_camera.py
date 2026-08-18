"""Shared 3D camera controls (rocklabel.gui.camera).

The labeler, the frame browsers and the live viewer all mix in
:class:`PivotCamera`, so the double-click pivot and the fly-mode key gating are
tested once here against fakes rather than three times against real windows
(which need a GL context).
"""

from __future__ import annotations

import numpy as np
import open3d.visualization.gui as gui
import pytest

from rocklabel.gui.camera import DOUBLE_CLICK_PX, DOUBLE_CLICK_SEC, PivotCamera


class _FakeScene:
    """Just enough gui.SceneWidget for the mixin: controls + pivot."""

    def __init__(self):
        self.controls = None
        self.center_of_rotation = None

    def set_view_controls(self, controls):
        self.controls = controls

    def set_on_mouse(self, handler):
        self.on_mouse = handler


class _FakeWindow:
    def __init__(self):
        self.redraws = 0

    def post_redraw(self):
        self.redraws += 1


class _Host(PivotCamera):
    def __init__(self):
        self.status = []
        self.scene = _FakeScene()
        self.window = _FakeWindow()
        self.install_camera(self.scene, self.window)

    def _camera_status(self, text):
        self.status.append(text)


class _Key:
    def __init__(self, key):
        self.key = key


@pytest.fixture()
def host():
    return _Host()


def test_installs_orbit_controls_quietly(host):
    """Startup arms the orbit controls but says nothing: the host's panel (and
    its status line) does not exist yet when install_camera runs."""
    assert host.nav_mode == "orbit"
    assert host.scene.controls == gui.SceneWidget.Controls.ROTATE_CAMERA
    assert host.status == []


def test_nav_mode_switches_controls_and_reports(host):
    host.set_nav_mode("fly")
    assert host.scene.controls == gui.SceneWidget.Controls.FLY
    host.set_nav_mode("orbit")
    assert host.scene.controls == gui.SceneWidget.Controls.ROTATE_CAMERA
    assert host.status == ["camera: fly", "camera: orbit"]


def test_unknown_nav_mode_falls_back_to_orbit(host):
    host.set_nav_mode("wobble")
    assert host.nav_mode == "orbit"


def test_double_click_needs_two_close_clicks(host):
    assert host._double_click(100, 100) is False       # first click of a pair
    assert host._double_click(100, 100) is True
    # The pair is consumed, so a third click starts over rather than chaining.
    assert host._double_click(100, 100) is False


def test_double_click_rejects_a_moved_second_click(host):
    host._double_click(100, 100)
    assert host._double_click(100 + DOUBLE_CLICK_PX + 1, 100) is False


def test_double_click_rejects_a_slow_second_click(host, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("rocklabel.gui.camera.time.monotonic", lambda: now[0])
    host._double_click(100, 100)
    now[0] += DOUBLE_CLICK_SEC + 0.01
    assert host._double_click(100, 100) is False


def test_set_pivot_moves_the_center_of_rotation(host):
    host._set_pivot(np.array([1.5, -2.0, 0.25]))
    assert host.scene.center_of_rotation == pytest.approx([1.5, -2.0, 0.25])
    assert host.window.redraws == 1
    assert host.status == ["orbit pivot set"]


def test_set_pivot_ignores_a_miss(host):
    """A double-click on empty background must not move the pivot to nowhere."""
    host._set_pivot(None)
    assert host.scene.center_of_rotation is None
    assert host.window.redraws == 0


def test_recenter_pivot_runs_the_pick_through_the_snap_hook(host, monkeypatch):
    """The labeler overrides _snap_pivot to land the pivot on a real point."""
    picked = np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr(_Host, "_pick_depth",
                        lambda self, x, y, cb: cb(picked))
    monkeypatch.setattr(_Host, "_snap_pivot",
                        lambda self, pos: np.asarray(pos) + 1.0)
    host._recenter_pivot(10, 20)
    assert host.scene.center_of_rotation == pytest.approx([1.1, 1.2, 1.3])


def test_fly_mode_takes_the_keyboard_and_esc_gives_it_back(host):
    assert host.camera_key_result(_Key(gui.KeyName.S)) is None  # orbit: host's

    host.set_nav_mode("fly")
    # WASD (and the viewers' own letter shortcuts) belong to the controller.
    assert (host.camera_key_result(_Key(gui.KeyName.S))
            == gui.Widget.EventCallbackResult.IGNORED)
    assert (host.camera_key_result(_Key(gui.KeyName.ESCAPE))
            == gui.Widget.EventCallbackResult.HANDLED)
    assert host.nav_mode == "orbit"
