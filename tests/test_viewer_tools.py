"""Labeler tool dispatch (rocklabel.gui.viewer._LabelerApp).

The real app needs a GL context, so the mouse/key handlers are exercised as
unbound functions against a fake host that stands in for the widgets.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import open3d.visualization.gui as gui
import pytest

from rocklabel.gui.viewer import _LabelerApp


class _FakeApp:
    """Only the pieces _on_mouse / _on_key actually touch."""

    def __init__(self, tool: str):
        self.tool = tool
        self._lasso: list[np.ndarray] = []
        self._lasso_gen = 0
        self._shift_down = False
        self._box_active = False
        self._box_anchor = self._box_corner = None
        self.status: list[str] = []
        self.arena_cleared = 0
        self.tools_set: list[str] = []
        self.previews = 0
        self.pivots = 0

    # -- stand-ins for the camera mixin / scene ------------------------------
    def _scene_xy(self, event):
        return int(event.x), int(event.y)

    def _double_click(self, x, y):
        return False

    def _pick_depth(self, x, y, callback):
        callback(np.array([float(x), float(y), 0.0]))  # picks always hit

    def _recenter_pivot(self, x, y):
        self.pivots += 1

    def _ground_z(self):
        return 0.0

    def _update_lasso_preview(self):
        self.previews += 1

    def _update_status(self, flash=""):
        self.status.append(flash)

    def _clear_arena(self):
        self.arena_cleared += 1

    def _set_tool(self, tool):
        self.tools_set.append(tool)
        self.tool = tool

    def camera_key_result(self, event):
        return None  # orbit mode: keys belong to the app


class _ShadingApp:
    """Only the pieces the arena shading helpers touch."""

    def __init__(self, arena=None, tool="navigate", lasso=()):
        self.tool = tool
        self._lasso = [np.asarray(p, float) for p in lasso]
        self.labelset = SimpleNamespace(arena=arena)
        self._shaded_arena = None
        self._inside_count = None
        self.rebuilds = 0

    _shading_arena = _LabelerApp._shading_arena

    def _rebuild_cloud(self):
        self.rebuilds += 1
        self._shaded_arena = self._shading_arena()


_SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])


def test_points_outside_the_arena_lose_their_color():
    app = _ShadingApp(arena=_SQUARE)
    pts = np.array([[1.0, 1.0, 0.0], [9.0, 9.0, 0.0]])
    colors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = _LabelerApp._gray_outside_arena(app, pts, colors)
    assert np.allclose(out[0], [1.0, 0.0, 0.0])       # inside: untouched
    assert out[1][0] == out[1][1] == out[1][2]        # outside: gray
    assert 0.0 < out[1][0] < 1.0
    assert app._inside_count == 1


def test_no_arena_leaves_every_color_alone():
    app = _ShadingApp(arena=None)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]])
    out = _LabelerApp._gray_outside_arena(app, np.zeros((2, 3)), colors)
    assert out is colors and app._inside_count is None


def test_an_outline_in_progress_shades_before_it_is_committed():
    """The draft ring wins over the committed arena - that live feedback is
    the only thing on screen that says which points are being roped in."""
    draft = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 5.0, 0.0), (0.0, 5.0, 0.0)]
    app = _ShadingApp(arena=_SQUARE, tool="arena", lasso=draft)
    assert np.allclose(_LabelerApp._shading_arena(app), np.asarray(draft)[:, :2])
    # a point outside the small committed arena but inside the draft stays lit
    colors = np.array([[1.0, 0.0, 0.0]])
    out = _LabelerApp._gray_outside_arena(app, np.array([[4.0, 4.0, 0.0]]), colors)
    assert np.allclose(out[0], [1.0, 0.0, 0.0])


def test_two_clicks_are_not_a_footprint_yet():
    app = _ShadingApp(arena=None, tool="arena", lasso=[(0, 0, 0), (1, 0, 0)])
    assert _LabelerApp._shading_arena(app) is None


def test_shading_refresh_skips_the_rebuild_when_nothing_changed():
    """Every outline click routes through here; recoloring the whole cloud for
    an unchanged polygon is a stall the user feels."""
    app = _ShadingApp(arena=_SQUARE)
    _LabelerApp._refresh_arena_shading(app)
    assert app.rebuilds == 1              # first pass: shading not applied yet
    _LabelerApp._refresh_arena_shading(app)
    assert app.rebuilds == 1              # same polygon: no second rebuild
    app._lasso = [np.array([0.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]),
                  np.array([3.0, 3.0, 0.0])]
    app.tool = "arena"
    _LabelerApp._refresh_arena_shading(app)
    assert app.rebuilds == 2              # a draft ring is a different polygon


def test_committing_the_draft_does_not_change_the_shading():
    """Enter turns the draft into the arena; the colors were already right, so
    they must not flicker back and forth as it is saved."""
    draft = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 3.0, 0.0)]
    app = _ShadingApp(tool="arena", lasso=draft)
    _LabelerApp._refresh_arena_shading(app)
    app.labelset.arena = np.asarray(draft)[:, :2]
    app._lasso = []
    _LabelerApp._refresh_arena_shading(app)
    assert app.rebuilds == 1


class _Mouse:
    def __init__(self, x, y, type_=gui.MouseEvent.Type.BUTTON_DOWN,
                 shift=False, ctrl=False):
        self.x, self.y, self.type = x, y, type_
        self._mods = {gui.KeyModifier.SHIFT: shift, gui.KeyModifier.CTRL: ctrl}

    def is_modifier_down(self, mod):
        return self._mods.get(mod, False)

    def is_button_down(self, button):
        return button == gui.MouseButton.LEFT


class _Key:
    def __init__(self, key, type_=gui.KeyEvent.Type.DOWN):
        self.key, self.type = key, type_


@pytest.mark.parametrize("tool", ["lasso", "arena"])
def test_outline_tools_collect_clicked_vertices(tool):
    """The arena shares the lasso's click-to-outline flow; before it did, its
    clicks fell through to the camera and nothing appeared on screen."""
    app = _FakeApp(tool)
    result = _LabelerApp._on_mouse(app, _Mouse(10, 20))
    assert result == gui.Widget.EventCallbackResult.CONSUMED
    assert len(app._lasso) == 1
    _LabelerApp._on_mouse(app, _Mouse(30, 40))
    assert np.allclose(np.stack(app._lasso), [[10, 20, 0], [30, 40, 0]])
    assert app.previews == 2
    assert app.status[-1].startswith(f"{tool}: 2 points")


@pytest.mark.parametrize("tool", ["lasso", "arena"])
def test_double_click_closes_an_open_outline(tool, monkeypatch):
    app = _FakeApp(tool)
    app._lasso = [np.zeros(3), np.ones(3), np.array([1.0, 0.0, 0.0])]
    monkeypatch.setattr(_FakeApp, "_double_click", lambda self, x, y: True)
    closed = []
    monkeypatch.setattr(_FakeApp, "_finish_lasso", lambda self: closed.append(True),
                        raising=False)
    _LabelerApp._on_mouse(app, _Mouse(10, 20))
    assert closed == [True] and app.pivots == 0


def test_double_click_without_an_outline_moves_the_pivot():
    app = _FakeApp("arena")
    app._double_click = lambda x, y: True
    _LabelerApp._on_mouse(app, _Mouse(10, 20))
    assert app.pivots == 1


def test_navigate_tool_leaves_clicks_to_the_camera():
    app = _FakeApp("navigate")
    assert (_LabelerApp._on_mouse(app, _Mouse(10, 20))
            == gui.Widget.EventCallbackResult.IGNORED)
    assert app._lasso == []


def test_shift_is_tracked_from_its_own_key_events():
    """gui.KeyEvent carries no modifier state, so A vs Shift+A depends on the
    shift press/release being recorded."""
    app = _FakeApp("navigate")
    _LabelerApp._on_key(app, _Key(gui.KeyName.A))
    assert app.tools_set == ["arena"] and app.arena_cleared == 0

    app.tools_set.clear()
    _LabelerApp._on_key(app, _Key(gui.KeyName.LEFT_SHIFT))
    assert app._shift_down
    _LabelerApp._on_key(app, _Key(gui.KeyName.A))
    assert app.arena_cleared == 1 and app.tools_set == []

    _LabelerApp._on_key(app, _Key(gui.KeyName.LEFT_SHIFT, gui.KeyEvent.Type.UP))
    assert not app._shift_down
    _LabelerApp._on_key(app, _Key(gui.KeyName.A))
    assert app.tools_set == ["arena"] and app.arena_cleared == 1
