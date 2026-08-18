"""Shared 3D camera controls for every rocklabel Open3D window.

The labeler grew a CAD-style camera: left-drag orbits around a *pivot*, and a
double-click drops that pivot onto whatever point you clicked, so the scene
turns around the thing you are looking at instead of around a fixed point
somewhere else. The replay/preview browsers and the live viewer used Open3D's
plain defaults, where the pivot is whatever ``setup_camera`` chose at startup
and can never be moved — the same drag feels completely different.

:class:`PivotCamera` is that behaviour lifted out of the labeler so all three
share one implementation:

  drag / wheel     orbit / zoom (orbit spins around the pivot)
  double-click     set the orbit pivot on the clicked point (CAD-style)
  fly mode         WASD + drag to look; Esc returns to orbit

Hosts mix it in, call :meth:`install_camera` once the SceneWidget exists, and
optionally place :meth:`make_nav_combo` in their settings panel.
"""

from __future__ import annotations

import time

import numpy as np
import open3d.visualization.gui as gui

#: Click tolerance for the depth pick: the nearest rendered geometry within
#: this radius of the cursor, so a double-click doesn't have to land exactly
#: on a 3-px point to register.
PICK_PATCH_PX = 10
#: Two clicks within this long and this close count as a double-click.
DOUBLE_CLICK_SEC = 0.40
DOUBLE_CLICK_PX = 8

#: The two mouse lines every viewer's shortcut list now shares.
CAMERA_HELP = """\
mouse
  drag / wheel     orbit / zoom (orbit spins around the pivot)
  double-click     set orbit pivot on the clicked point (CAD-style)"""

#: Panel hint that goes next to the nav-mode combobox.
CAMERA_HINT = "double-click a point to orbit around it"


class PivotCamera:
    """Labeler-style camera controls for a ``gui.SceneWidget``.

    Mix into a viewer class and call :meth:`install_camera` after the widget
    is built. Hosts with no other mouse tools get the double-click pivot for
    free; hosts that own the mouse themselves (the labeler's box/lasso tools)
    pass ``handle_mouse=False`` and call :meth:`_double_click` /
    :meth:`_recenter_pivot` from their own handler.
    """

    NAV_MODES = ("orbit", "fly")

    # -- setup ---------------------------------------------------------------

    def install_camera(self, scene: gui.SceneWidget, window: gui.Window,
                       handle_mouse: bool = True) -> None:
        self._cam_scene = scene
        self._cam_window = window
        self.nav_mode = "orbit"
        self.nav_combo: gui.Combobox | None = None
        self._last_click = (0.0, -100, -100)   # (t, x, y)
        # Quiet: install_camera runs before the host has built its panel, so
        # there is no status line (or combobox) to report into yet.
        self.set_nav_mode("orbit", quiet=True)
        if handle_mouse:
            scene.set_on_mouse(self._on_camera_mouse)

    def make_nav_combo(self) -> gui.Combobox:
        """The Orbit/Fly selector; the host places it in its own panel."""
        combo = gui.Combobox()
        combo.add_item("Orbit around pivot")
        combo.add_item("Fly (WASD + drag)")
        combo.selected_index = self.NAV_MODES.index(self.nav_mode)
        combo.set_on_selection_changed(
            lambda _text, index: self.set_nav_mode(self.NAV_MODES[int(index)]))
        self.nav_combo = combo
        return combo

    # -- nav mode ------------------------------------------------------------

    def set_nav_mode(self, mode: str, quiet: bool = False) -> None:
        self.nav_mode = mode if mode in self.NAV_MODES else "orbit"
        self._cam_scene.set_view_controls(
            gui.SceneWidget.Controls.ROTATE_CAMERA if self.nav_mode == "orbit"
            else gui.SceneWidget.Controls.FLY)
        if self.nav_combo is not None:
            index = self.NAV_MODES.index(self.nav_mode)
            if self.nav_combo.selected_index != index:
                self.nav_combo.selected_index = index
        if not quiet:
            self._camera_status(f"camera: {self.nav_mode}")

    def camera_key_result(self, event):
        """Key gating for fly mode, for hosts to call first in ``set_on_key``.

        While flying, WASD and friends belong to the fly controller, so the
        viewer's own single-letter shortcuts have to stand down; only Esc is
        ours, and it returns to orbit. Returns an ``EventCallbackResult`` to
        return immediately, or None when the host should handle the key.
        """
        if self.nav_mode != "fly":
            return None
        if event.key == gui.KeyName.ESCAPE:
            self.set_nav_mode("orbit")
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    # -- pivot ---------------------------------------------------------------

    def _set_pivot(self, pos, quiet: bool = False) -> None:
        if pos is None:
            return
        self._cam_scene.center_of_rotation = np.asarray(pos, np.float32)
        if not quiet:
            self._camera_status("orbit pivot set")
        self._cam_window.post_redraw()

    def _recenter_pivot(self, x: int, y: int) -> None:
        """Move the orbit pivot onto whatever is rendered at pixel (x, y)."""
        self._pick_depth(x, y, lambda p: self._set_pivot(
            None if p is None else self._snap_pivot(p)))

    def _snap_pivot(self, pos: np.ndarray) -> np.ndarray:
        """Hook: refine a picked position (the labeler snaps to a cloud point)."""
        return pos

    def _camera_status(self, text: str) -> None:
        """Hook: report a camera change in the host's status line."""

    # -- picking -------------------------------------------------------------

    def _pick_depth(self, x: int, y: int, callback) -> None:
        """Find rendered geometry near pixel (x, y) and pass its world position
        to ``callback`` on the main thread (None if nothing is close enough).

        Searches a patch around the cursor instead of the single pixel, so a
        click doesn't have to land exactly on a 3-px point to register."""
        scene = self._cam_scene
        frame_w, frame_h = scene.frame.width, scene.frame.height

        def depth_cb(depth_image):
            depth = np.asarray(depth_image)
            h, w = depth.shape[:2]
            pos = None
            if 0 <= x < w and 0 <= y < h:
                x0, x1 = max(x - PICK_PATCH_PX, 0), min(x + PICK_PATCH_PX + 1, w)
                y0, y1 = max(y - PICK_PATCH_PX, 0), min(y + PICK_PATCH_PX + 1, h)
                sub = depth[y0:y1, x0:x1]
                yy, xx = np.nonzero(sub < 1.0)
                if len(xx):
                    j = int(np.argmin((yy + y0 - y) ** 2 + (xx + x0 - x) ** 2))
                    world = scene.scene.camera.unproject(
                        int(xx[j] + x0), int(yy[j] + y0), float(sub[yy[j], xx[j]]),
                        frame_w, frame_h)
                    pos = np.array([world[0], world[1], world[2]], float)
            gui.Application.instance.post_to_main_thread(
                self._cam_window, lambda: callback(pos))

        scene.scene.scene.render_to_depth_image(depth_cb)

    # -- mouse ---------------------------------------------------------------

    def _scene_xy(self, event) -> tuple[int, int]:
        """Mouse position relative to the scene widget, not the window."""
        return (int(event.x - self._cam_scene.frame.x),
                int(event.y - self._cam_scene.frame.y))

    def _double_click(self, x: int, y: int) -> bool:
        """Record a click and report whether it completed a double-click."""
        now = time.monotonic()
        t0, px, py = self._last_click
        self._last_click = (now, x, y)
        if (now - t0 < DOUBLE_CLICK_SEC
                and abs(x - px) < DOUBLE_CLICK_PX and abs(y - py) < DOUBLE_CLICK_PX):
            self._last_click = (0.0, -100, -100)  # don't chain into a triple
            return True
        return False

    def _on_camera_mouse(self, event):
        """Default handler: double-click recenters, everything else navigates."""
        Result = gui.Widget.EventCallbackResult
        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                and event.is_button_down(gui.MouseButton.LEFT)):
            x, y = self._scene_xy(event)
            if self._double_click(x, y):
                self._recenter_pivot(x, y)
                return Result.CONSUMED
        return Result.IGNORED
