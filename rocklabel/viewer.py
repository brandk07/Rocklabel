"""All Open3D code lives in this module (see §9.4: keep everything else headless).

Entry points:
  run_labeler_gui       modern labeling app: side panel, buttons, sliders,
                        rock list, turbo/inferno colormaps
  run_labeler_fallback  VisualizerWithVertexSelection + terminal prompts
  FrameBrowserApp       interactive dataset-preview browser with a transport
                        bar (prev/next/play/slider) used by `rocklabel preview`
  show_driftcheck       overlay early/late accumulations around one rock
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

from types import SimpleNamespace

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial import cKDTree

from .camera import CAMERA_HELP, CAMERA_HINT, PivotCamera
from .labeling import points_in_rock
from .labels import sync_bounds, translate_rock

RADIUS_STEP = 0.02
NUDGE_STEP = 0.02
ZCLIP_STEP = 0.05
HIGHLIGHT_COLOR = [0.10, 1.00, 0.85]
PREVIEW_COLOR = [0.30, 0.85, 1.00]
WIRE_SELECTED = [1.00, 0.85, 0.10]
WIRE_NORMAL = [0.95, 0.20, 0.15]

BG_COLOR = [0.055, 0.06, 0.075, 1.0]
ACCENT = gui.Color(0.30, 0.65, 1.00)
DIM = gui.Color(0.62, 0.64, 0.70)
OK_GREEN = gui.Color(0.35, 0.85, 0.45)
WARN = gui.Color(1.00, 0.75, 0.25)


# -- colormaps ---------------------------------------------------------------

def _turbo(t: np.ndarray) -> np.ndarray:
    """Google's Turbo colormap (polynomial approximation), t in [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))
    g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))
    b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


_INFERNO_STOPS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
_INFERNO_COLORS = np.array([
    [0.0015, 0.0005, 0.0139],
    [0.2582, 0.0386, 0.4065],
    [0.5783, 0.1480, 0.4044],
    [0.8650, 0.3168, 0.2261],
    [0.9876, 0.6453, 0.0399],
    [0.9884, 0.9984, 0.6449],
])


def _inferno(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return np.stack(
        [np.interp(t, _INFERNO_STOPS, _INFERNO_COLORS[:, c]) for c in range(3)], axis=-1
    )


def colorize(values: np.ndarray, cmap: str = "turbo",
             pct: tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
    """Map scalars to RGB with robust percentile normalization."""
    if len(values) == 0:
        return np.empty((0, 3))
    lo, hi = np.percentile(values, list(pct))
    t = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return _inferno(t) if cmap == "inferno" else _turbo(t)


def height_colors(z: np.ndarray) -> np.ndarray:
    return colorize(z, "turbo")


def reflectivity_colors(inten: np.ndarray) -> np.ndarray:
    """Reflectivity (RSSI) coloring: 5-95 percentile so retro tape pops."""
    if len(inten) and float(inten.max()) <= 0.0:
        return np.full((len(inten), 3), 0.35)  # no RSSI data: flat gray
    return colorize(inten, "inferno", pct=(5.0, 95.0))


# -- small helpers -----------------------------------------------------------

def _ensure_app() -> gui.Application:
    # Open3D's Filament renderer needs X11 window handles, but its windowing
    # layer picks the native Wayland backend when XDG_SESSION_TYPE=wayland,
    # which segfaults in XSendEvent during create_window. Claim an x11
    # session so it falls back to XWayland.
    if sys.platform.startswith("linux") and os.environ.get("XDG_SESSION_TYPE") == "wayland":
        os.environ["XDG_SESSION_TYPE"] = "x11"
    app = gui.Application.instance
    app.initialize()
    return app


def _sphere_mesh(center: np.ndarray, radius: float) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=24)
    mesh.compute_vertex_normals()
    mesh.translate(np.asarray(center, float))
    return mesh


def _point_cloud(xyz: np.ndarray, rgb: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(xyz, np.float64)))
    rgb = np.asarray(rgb, np.float64)
    if rgb.ndim == 1:
        pcd.paint_uniform_color(rgb)
    else:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def _unlit_material(point_size: float = 3.0) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(point_size)
    return mat


def _line_material(width: float = 2.0) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = float(width)
    return mat


def _box_lineset(lo: np.ndarray, hi: np.ndarray) -> o3d.geometry.LineSet:
    hi = np.maximum(np.asarray(hi, float), np.asarray(lo, float) + 1e-4)
    aabb = o3d.geometry.AxisAlignedBoundingBox(np.asarray(lo, float), hi)
    return o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)


def _prism_lineset(verts_xy: np.ndarray, z0: float, z1: float) -> o3d.geometry.LineSet:
    """Wireframe of a polygon extruded from z0 to z1."""
    v = np.asarray(verts_xy, float)
    n = len(v)
    pts = np.vstack([np.c_[v, np.full(n, float(z0))], np.c_[v, np.full(n, float(z1))]])
    ring = [[i, (i + 1) % n] for i in range(n)]
    lines = ring + [[a + n, b + n] for a, b in ring] + [[i, i + n] for i in range(n)]
    return o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts),
                                o3d.utility.Vector2iVector(np.asarray(lines, np.int32)))


def _polyline_lineset(pts: np.ndarray, close: bool) -> o3d.geometry.LineSet:
    pts = np.asarray(pts, float)
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    if close and len(pts) >= 3:
        lines.append([len(pts) - 1, 0])
    return o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts),
                                o3d.utility.Vector2iVector(np.asarray(lines, np.int32)))


def _setup_scene(scene_widget: gui.SceneWidget) -> None:
    scene_widget.scene.set_background(BG_COLOR)
    scene_widget.scene.show_skybox(False)
    scene_widget.scene.set_lighting(
        rendering.Open3DScene.LightingProfile.MED_SHADOWS, (0.577, -0.577, -0.577)
    )


def _header(panel: gui.Vert, em: float, title: str, subtitle: str) -> None:
    t = gui.Label(title)
    t.text_color = ACCENT
    panel.add_child(t)
    s = gui.Label(subtitle)
    s.text_color = DIM
    panel.add_child(s)


def _help_block(text: str) -> gui.Widget:
    lbl = gui.Label(text)
    lbl.text_color = DIM
    return lbl


# ===========================================================================
# Labeler
# ===========================================================================

HELP_TEXT = CAMERA_HELP + """
  Shift+click      place sphere rock (works in every tool)
  Ctrl+click       select nearest rock
tools  (keys N / B / L or the Tool dropdown)
  N  navigate      plain drags move the camera
  B  box           drag the footprint on the ground, release,
                   then edit width/depth/height sliders
  L  lasso         click the outline point by point;
                   Enter or double-click closes it, Esc cancels;
                   then edit base/top z sliders
  points inside the active shape light up in cyan
keys
  C          cycle color: height / reflectivity
  + / -      grow/shrink selected (radius, box, or lasso top)
  arrows     nudge selected in x/y
  PgUp/PgDn  nudge selected in z
  X / Del    delete selected
  F          fly camera to selected (also sets the orbit pivot)
  S          save      Q/Esc  quit (auto-saves)
fly mode: WASD + drag to look; panel keys are disabled while flying"""


class _LabelerApp(PivotCamera):
    """Interactive rock-labeling window (open3d.visualization.gui)."""

    def __init__(self, xyz, inten, labelset, save_path, cfg, z_min, z_max):
        self.xyz = xyz
        self.inten = inten
        self.labelset = labelset
        self.save_path = save_path
        self.default_radius = cfg["labeler"]["default_rock_radius_m"]
        self.color_mode = "height"
        self.point_size = 2.5
        zlo, zhi = float(xyz[:, 2].min()), float(xyz[:, 2].max())
        self.z_bounds = (zlo, zhi)
        self.z_min = zlo if z_min is None else float(z_min)
        self.z_max = zhi if z_max is None else float(z_max)
        self.selected_id: int | None = None
        self._tree: cKDTree | None = None
        self._status_flash = ""

        # tools / interaction state
        self.tool = "navigate"                 # navigate | box | lasso
        self._box_height = max(0.30, 2.0 * self.default_radius)
        self._box_active = False               # a box drag is in progress
        self._box_anchor: np.ndarray | None = None
        self._box_corner: np.ndarray | None = None
        self._lasso: list[np.ndarray] = []
        self._lasso_gen = 0                    # invalidates in-flight vertex picks
        self._highlight_rock = None            # shape whose points glow cyan

        app = _ensure_app()
        self.window = app.create_window("rocklabel - labeler", 1500, 940)
        em = self.window.theme.font_size

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        _setup_scene(self.scene)
        # The box/lasso tools own the mouse, so _on_mouse drives the shared
        # camera helpers itself instead of letting it install its own handler.
        self.install_camera(self.scene, self.window, handle_mouse=False)
        self.scene.set_on_mouse(self._on_mouse)
        self.scene.set_on_key(self._on_key)

        self.cloud_mat = _unlit_material(self.point_size)
        self.sphere_mat = rendering.MaterialRecord()
        self.sphere_mat.shader = "defaultLitTransparency"
        self.sphere_mat.base_color = [0.95, 0.20, 0.15, 0.45]
        self.selected_mat = rendering.MaterialRecord()
        self.selected_mat.shader = "defaultLitTransparency"
        self.selected_mat.base_color = [1.0, 0.85, 0.1, 0.60]

        self._build_panel(em)
        self.window.add_child(self.scene)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        self._rebuild_cloud(first=True)
        for rock in self.labelset.rocks:
            self._draw_rock(rock)
        self._refresh_rock_list()
        self._update_status()

    # -- panel construction -------------------------------------------------

    def _build_panel(self, em: float) -> None:
        # scrollable: the panel outgrows short windows now that it has
        # tool/camera sections and per-shape editors
        self.panel = gui.ScrollableVert(
            0.5 * em, gui.Margins(0.8 * em, 0.8 * em, 0.8 * em, 0.8 * em))
        _header(self.panel, em, "ROCK LABELER",
                f"{self.labelset.run_id or 'unnamed run'} · N/B/L tools · dbl-click sets pivot")

        # --- tool section ---
        tools = gui.CollapsableVert("Tool", 0.35 * em, gui.Margins(em, 0, 0, 0))
        tools.set_is_open(True)
        self.tool_combo = gui.Combobox()
        self.tool_combo.add_item("Navigate (N)")
        self.tool_combo.add_item("Box: drag footprint (B)")
        self.tool_combo.add_item("Lasso: click outline (L)")
        self.tool_combo.set_on_selection_changed(self._on_tool_combo)
        tools.add_child(self.tool_combo)
        self.tool_hint = gui.Label("Shift+click always places a sphere")
        self.tool_hint.text_color = DIM
        tools.add_child(self.tool_hint)
        self.panel.add_child(tools)

        # --- camera section ---
        cam = gui.CollapsableVert("Camera", 0.35 * em, gui.Margins(em, 0, 0, 0))
        cam.set_is_open(True)
        cam.add_child(self.make_nav_combo())
        hint = gui.Label(CAMERA_HINT + ";\nF flies to selection")
        hint.text_color = DIM
        cam.add_child(hint)
        self.panel.add_child(cam)

        # --- display section ---
        disp = gui.CollapsableVert("Display", 0.35 * em, gui.Margins(em, 0, 0, 0))
        disp.set_is_open(True)

        row = gui.Horiz(0.5 * em)
        row.add_child(gui.Label("Color by"))
        self.color_combo = gui.Combobox()
        self.color_combo.add_item("Height")
        self.color_combo.add_item("Reflectivity")
        self.color_combo.set_on_selection_changed(self._on_color_combo)
        row.add_child(self.color_combo)
        disp.add_child(row)

        disp.add_child(gui.Label("Point size"))
        self.size_slider = gui.Slider(gui.Slider.DOUBLE)
        self.size_slider.set_limits(1.0, 8.0)
        self.size_slider.double_value = self.point_size
        self.size_slider.set_on_value_changed(self._on_point_size)
        disp.add_child(self.size_slider)

        zlo, zhi = self.z_bounds
        pad = max((zhi - zlo) * 0.02, 0.01)
        disp.add_child(gui.Label("Z clip · min / max (m)"))
        self.zmin_slider = gui.Slider(gui.Slider.DOUBLE)
        self.zmin_slider.set_limits(zlo - pad, zhi + pad)
        self.zmin_slider.double_value = self.z_min
        self.zmin_slider.set_on_value_changed(self._on_zmin)
        disp.add_child(self.zmin_slider)
        self.zmax_slider = gui.Slider(gui.Slider.DOUBLE)
        self.zmax_slider.set_limits(zlo - pad, zhi + pad)
        self.zmax_slider.double_value = self.z_max
        self.zmax_slider.set_on_value_changed(self._on_zmax)
        disp.add_child(self.zmax_slider)

        toggles = gui.Horiz(0.8 * em)
        self.grid_check = gui.Checkbox("Grid")
        self.grid_check.set_on_checked(self._on_grid)
        toggles.add_child(self.grid_check)
        self.axes_check = gui.Checkbox("Axes")
        self.axes_check.set_on_checked(self._on_axes)
        toggles.add_child(self.axes_check)
        disp.add_child(toggles)
        self.panel.add_child(disp)

        # --- rocks section ---
        self.rocks_section = gui.CollapsableVert("Rocks", 0.35 * em, gui.Margins(em, 0, 0, 0))
        self.rocks_section.set_is_open(True)
        self.rock_list = gui.ListView()
        self.rock_list.set_max_visible_items(8)
        self.rock_list.set_on_selection_changed(self._on_list_select)
        self.rocks_section.add_child(self.rock_list)

        # per-shape editors; visibility follows the selected rock's shape
        self.sphere_edit = gui.Vert(0.25 * em)
        self.sphere_edit.add_child(gui.Label("Radius of selected (m)"))
        self.radius_slider = gui.Slider(gui.Slider.DOUBLE)
        self.radius_slider.set_limits(0.03, 1.0)
        self.radius_slider.double_value = self.default_radius
        self.radius_slider.set_on_value_changed(self._on_radius)
        self.sphere_edit.add_child(self.radius_slider)
        self.rocks_section.add_child(self.sphere_edit)

        self.box_edit = gui.Vert(0.25 * em)
        self.box_sliders = []
        for axis, caption in enumerate(("Width X (m)", "Depth Y (m)", "Height Z (m)")):
            self.box_edit.add_child(gui.Label(caption))
            s = gui.Slider(gui.Slider.DOUBLE)
            s.set_limits(0.03, 4.0)
            s.set_on_value_changed(lambda v, a=axis: self._on_box_size(a, v))
            self.box_edit.add_child(s)
            self.box_sliders.append(s)
        self.rocks_section.add_child(self.box_edit)

        zlo, zhi = self.z_bounds
        zpad = max((zhi - zlo) * 0.05, 0.2)
        self.poly_edit = gui.Vert(0.25 * em)
        self.poly_edit.add_child(gui.Label("Lasso base z (m)"))
        self.poly_z0_slider = gui.Slider(gui.Slider.DOUBLE)
        self.poly_z0_slider.set_limits(zlo - zpad, zhi + zpad)
        self.poly_z0_slider.set_on_value_changed(lambda v: self._on_poly_z(0, v))
        self.poly_edit.add_child(self.poly_z0_slider)
        self.poly_edit.add_child(gui.Label("Lasso top z (m)"))
        self.poly_z1_slider = gui.Slider(gui.Slider.DOUBLE)
        self.poly_z1_slider.set_limits(zlo - zpad, zhi + zpad)
        self.poly_z1_slider.set_on_value_changed(lambda v: self._on_poly_z(1, v))
        self.poly_edit.add_child(self.poly_z1_slider)
        self.rocks_section.add_child(self.poly_edit)
        self.box_edit.visible = False
        self.poly_edit.visible = False

        btns = gui.Horiz(0.5 * em)
        focus_btn = gui.Button("Focus")
        focus_btn.set_on_clicked(self._focus_selected)
        btns.add_child(focus_btn)
        del_btn = gui.Button("Delete")
        del_btn.set_on_clicked(self._delete_selected)
        btns.add_child(del_btn)
        self.rocks_section.add_child(btns)
        self.panel.add_child(self.rocks_section)

        # --- shortcuts ---
        help_section = gui.CollapsableVert("Shortcuts", 0.35 * em, gui.Margins(em, 0, 0, 0))
        help_section.set_is_open(False)
        help_section.add_child(_help_block(HELP_TEXT))
        self.panel.add_child(help_section)

        self.panel.add_stretch()
        save_btn = gui.Button("Save labels")
        save_btn.set_on_clicked(self._save_clicked)
        self.panel.add_child(save_btn)
        self.status = gui.Label("")
        self.status.text_color = DIM
        self.panel.add_child(self.status)

    def _on_layout(self, ctx) -> None:
        rect = self.window.content_rect
        panel_w = min(int(20 * ctx.theme.font_size), rect.width // 3)
        self.scene.frame = gui.Rect(rect.x, rect.y, rect.width - panel_w, rect.height)
        self.panel.frame = gui.Rect(rect.get_right() - panel_w, rect.y, panel_w, rect.height)

    # -- rendering ----------------------------------------------------------

    def _visible_mask(self) -> np.ndarray:
        return (self.xyz[:, 2] >= self.z_min) & (self.xyz[:, 2] <= self.z_max)

    def _rebuild_cloud(self, first: bool = False) -> None:
        mask = self._visible_mask()
        pts = self.xyz[mask]
        if len(pts) == 0:  # never blank the screen entirely
            pts = self.xyz[:1]
            mask = np.zeros(len(self.xyz), bool)
            mask[0] = True
        self._tree = cKDTree(pts)
        self._visible_pts = pts
        colors = (height_colors(pts[:, 2]) if self.color_mode == "height"
                  else reflectivity_colors(self.inten[mask]))
        pcd = _point_cloud(pts, colors)
        if self.scene.scene.has_geometry("cloud"):
            self.scene.scene.remove_geometry("cloud")
        self.cloud_mat.point_size = self.point_size
        self.scene.scene.add_geometry("cloud", pcd, self.cloud_mat)
        self._refresh_highlight()
        if first:
            bbox = pcd.get_axis_aligned_bounding_box()
            self.scene.setup_camera(60.0, bbox, bbox.get_center())

    def _remove(self, name: str) -> None:
        if self.scene.scene.has_geometry(name):
            self.scene.scene.remove_geometry(name)

    def _draw_rock(self, rock) -> None:
        self._erase_rock(rock.id)
        selected = rock.id == self.selected_id
        mat = self.selected_mat if selected else self.sphere_mat
        wire_color = WIRE_SELECTED if selected else WIRE_NORMAL
        name, wname = f"rock_{rock.id}", f"rockw_{rock.id}"
        if rock.shape == "box":
            size = np.maximum(np.asarray(rock.size, float), 1e-3)
            lo = np.asarray(rock.center, float) - size / 2.0
            mesh = o3d.geometry.TriangleMesh.create_box(*size)
            mesh.compute_vertex_normals()
            mesh.translate(lo)
            self.scene.scene.add_geometry(name, mesh, mat)
            wire = _box_lineset(lo, lo + size)
            wire.paint_uniform_color(wire_color)
            self.scene.scene.add_geometry(wname, wire, _line_material(2.5))
        elif rock.shape == "polygon":
            wire = _prism_lineset(rock.vertices, *rock.z_range)
            wire.paint_uniform_color(wire_color)
            self.scene.scene.add_geometry(wname, wire, _line_material(2.5))
        else:
            self.scene.scene.add_geometry(name, _sphere_mesh(rock.center, rock.radius), mat)

    def _erase_rock(self, rock_id: int) -> None:
        self._remove(f"rock_{rock_id}")
        self._remove(f"rockw_{rock_id}")

    def _refresh_rock_list(self) -> None:
        def fmt(r):
            if r.shape == "box":
                return f"#{r.id}  box  {r.size[0]:.2f} × {r.size[1]:.2f} × {r.size[2]:.2f} m"
            if r.shape == "polygon":
                return f"#{r.id}  lasso  {len(r.vertices)} pts  h={r.z_range[1] - r.z_range[0]:.2f} m"
            return f"#{r.id}  sphere  r={r.radius:.2f} m  ({r.center[0]:.2f}, {r.center[1]:.2f}, {r.center[2]:.2f})"

        self.rock_list.set_items([fmt(r) for r in self.labelset.rocks])
        ids = [r.id for r in self.labelset.rocks]
        if self.selected_id in ids:
            self.rock_list.selected_index = ids.index(self.selected_id)

    def _update_status(self, flash: str = "") -> None:
        n = len(self.labelset.rocks)
        parts = [f"{n} rock{'s' if n != 1 else ''}", f"color: {self.color_mode}"]
        if flash:
            parts.append(flash)
        self.status.text = "  ·  ".join(parts) + f"\n{self.save_path}"
        self.window.set_needs_layout()
        self.window.post_redraw()

    # -- widget callbacks ---------------------------------------------------

    def _on_color_combo(self, _text, index) -> None:
        self.color_mode = "height" if index == 0 else "reflectivity"
        self._rebuild_cloud()
        self._update_status()

    def _on_point_size(self, value) -> None:
        self.point_size = float(value)
        self.cloud_mat.point_size = self.point_size
        self.scene.scene.modify_geometry_material("cloud", self.cloud_mat)
        if self.scene.scene.has_geometry("highlight"):
            self.scene.scene.modify_geometry_material(
                "highlight", _unlit_material(self.point_size + 2.5))
        self.window.post_redraw()

    def _on_zmin(self, value) -> None:
        self.z_min = min(float(value), self.z_max)
        self._rebuild_cloud()

    def _on_zmax(self, value) -> None:
        self.z_max = max(float(value), self.z_min)
        self._rebuild_cloud()

    def _on_grid(self, checked) -> None:
        # The GroundPlane enum moved between Open3D versions; find it wherever
        # this build put it.
        ground = (getattr(rendering.Scene, "GroundPlane", None)
                  or getattr(rendering.Open3DScene, "GroundPlane", None))
        if ground is None:
            self._update_status("grid not supported by this Open3D build")
            return
        self.scene.scene.show_ground_plane(checked, ground.XY)
        self.window.post_redraw()

    def _on_tool_combo(self, _text, index) -> None:
        self._set_tool(("navigate", "box", "lasso")[int(index)])

    def _set_tool(self, tool: str) -> None:
        if tool != "lasso":
            self._cancel_lasso(quiet=True)
        self._box_active = False
        self._clear_preview()
        self.tool = tool
        self.tool_combo.selected_index = ("navigate", "box", "lasso").index(tool)
        hints = {
            "navigate": "Shift+click always places a sphere",
            "box": "drag the footprint on the ground,\nrelease, then set the sliders",
            "lasso": "click outline points; Enter or\ndouble-click closes, Esc cancels",
        }
        self.tool_hint.text = hints[tool]
        self._update_status(f"tool: {tool}")

    def _camera_status(self, text: str) -> None:
        self._update_status(text)

    def _on_axes(self, checked) -> None:
        self.scene.scene.show_axes(checked)
        self.window.post_redraw()

    def _on_list_select(self, *_args) -> None:
        i = self.rock_list.selected_index
        if 0 <= i < len(self.labelset.rocks):
            self._set_selected(self.labelset.rocks[i].id)
            self._update_status()

    def _selected_rock(self):
        return self.labelset.get(self.selected_id) if self.selected_id else None

    def _after_edit(self, rock) -> None:
        self._draw_rock(rock)
        self._save()
        self._refresh_rock_list()
        self._set_highlight(rock)

    def _on_radius(self, value) -> None:
        rock = self._selected_rock()
        if rock is None or rock.shape != "sphere":
            return
        rock.radius = float(value)
        self._after_edit(rock)

    def _on_box_size(self, axis: int, value) -> None:
        rock = self._selected_rock()
        if rock is None or rock.shape != "box":
            return
        size = np.asarray(rock.size, float).copy()
        if axis == 2:  # height grows upward from the base, not from the center
            base = rock.center[2] - size[2] / 2.0
            size[2] = float(value)
            rock.center = np.array([rock.center[0], rock.center[1], base + size[2] / 2.0])
        else:
            size[axis] = float(value)
        rock.size = size
        sync_bounds(rock)
        self._after_edit(rock)

    def _on_poly_z(self, which: int, value) -> None:
        rock = self._selected_rock()
        if rock is None or rock.shape != "polygon":
            return
        z0, z1 = rock.z_range
        if which == 0:
            z0 = min(float(value), z1 - 0.01)
        else:
            z1 = max(float(value), z0 + 0.01)
        rock.z_range = (z0, z1)
        sync_bounds(rock)
        self._after_edit(rock)

    def _sync_edit_widgets(self) -> None:
        rock = self._selected_rock()
        shape = rock.shape if rock is not None else None
        self.sphere_edit.visible = shape in (None, "sphere")
        self.box_edit.visible = shape == "box"
        self.poly_edit.visible = shape == "polygon"
        if shape == "sphere":
            self.radius_slider.double_value = rock.radius
        elif shape == "box":
            for s, v in zip(self.box_sliders, np.asarray(rock.size, float)):
                s.double_value = float(v)
        elif shape == "polygon":
            self.poly_z0_slider.double_value = rock.z_range[0]
            self.poly_z1_slider.double_value = rock.z_range[1]
        self.window.set_needs_layout()

    def _focus_selected(self) -> None:
        rock = self._selected_rock()
        if rock is None:
            return
        c = np.asarray(rock.center, float)
        # Keep the current viewing direction; just move closer and re-aim so
        # the jump doesn't disorient. The rock becomes the orbit pivot.
        view = np.asarray(self.scene.scene.camera.get_view_matrix(), float)
        rot = view[:3, :3]
        eye_now = -rot.T @ view[:3, 3]
        direction = c - eye_now
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction, norm = np.array([1.0, 1.0, -0.75]), np.linalg.norm([1.0, 1.0, -0.75])
        distance = max(6.0 * rock.radius, 1.5)
        eye = c - direction / norm * distance
        self.scene.look_at(c, eye, [0.0, 0.0, 1.0])
        self._set_pivot(c, quiet=True)
        self.window.post_redraw()

    def _delete_selected(self) -> None:
        if self.selected_id is None:
            return
        self._erase_rock(self.selected_id)
        self.labelset.remove(self.selected_id)
        self.selected_id = None
        self._set_highlight(None)
        self._sync_edit_widgets()
        self._save()
        self._refresh_rock_list()
        self._update_status("deleted")

    def _save_clicked(self) -> None:
        self._save()
        self._update_status("saved")

    # -- interaction --------------------------------------------------------

    def _save(self) -> None:
        self.labelset.save(self.save_path)

    # -- picking helpers ----------------------------------------------------
    # (_pick_depth / _set_pivot live in PivotCamera, shared with the browsers)

    def _ray_hit_z(self, x: int, y: int, z0: float) -> np.ndarray | None:
        """Where the ray through pixel (x, y) crosses the horizontal plane z=z0."""
        cam = self.scene.scene.camera
        w, h = self.scene.frame.width, self.scene.frame.height
        a = np.asarray(cam.unproject(x, y, 0.1, w, h), float)
        b = np.asarray(cam.unproject(x, y, 0.9, w, h), float)
        d = b - a
        if abs(d[2]) < 1e-9:
            return None
        return a + (z0 - a[2]) / d[2] * d

    def _ground_z(self) -> float:
        return float(np.median(self._visible_pts[:, 2]))

    # -- live highlight -------------------------------------------------------

    def _set_highlight(self, rock_like) -> None:
        """Paint the visible points inside ``rock_like`` cyan (None clears)."""
        self._highlight_rock = rock_like
        self._refresh_highlight()
        self.window.post_redraw()

    def _refresh_highlight(self) -> None:
        self._remove("highlight")
        if self._highlight_rock is None:
            return
        mask = points_in_rock(self._visible_pts, self._highlight_rock)
        if mask.any():
            pcd = _point_cloud(self._visible_pts[mask], HIGHLIGHT_COLOR)
            self.scene.scene.add_geometry("highlight", pcd,
                                          _unlit_material(self.point_size + 2.5))

    def _clear_preview(self) -> None:
        self._remove("_preview")
        self.window.post_redraw()

    # -- box tool -------------------------------------------------------------

    def _box_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        a, c = self._box_anchor, self._box_corner
        lo = np.array([min(a[0], c[0]), min(a[1], c[1]), a[2] - 0.02])
        hi = np.array([max(a[0], c[0]), max(a[1], c[1]), a[2] - 0.02 + self._box_height])
        return lo, hi

    def _update_box_preview(self) -> None:
        lo, hi = self._box_bounds()
        wire = _box_lineset(lo, hi)
        wire.paint_uniform_color(PREVIEW_COLOR)
        self._remove("_preview")
        self.scene.scene.add_geometry("_preview", wire, _line_material(3.0))
        center, size = (lo + hi) / 2.0, hi - lo
        self._set_highlight(SimpleNamespace(shape="box", center=center, size=size,
                                            radius=float(np.linalg.norm(size)) / 2.0))
        self._update_status(f"box {size[0]:.2f} × {size[1]:.2f} m")

    def _finish_box(self) -> None:
        self._box_active = False
        self._clear_preview()
        if self._box_anchor is None or self._box_corner is None:
            return
        lo, hi = self._box_bounds()
        self._box_anchor = self._box_corner = None
        if (hi[0] - lo[0]) < 0.03 or (hi[1] - lo[1]) < 0.03:
            self._set_highlight(None)
            self._update_status("box too small - drag the footprint out first")
            return
        rock = self.labelset.add_box((lo + hi) / 2.0, hi - lo)
        self._save()
        self._set_selected(rock.id)
        self._refresh_rock_list()
        self._update_status(f"placed box #{rock.id} - set height with the slider")

    # -- lasso tool -----------------------------------------------------------

    def _update_lasso_preview(self) -> None:
        self._remove("_preview")
        if len(self._lasso) >= 2:
            line = _polyline_lineset(np.stack(self._lasso), close=len(self._lasso) >= 3)
            line.paint_uniform_color(PREVIEW_COLOR)
            self.scene.scene.add_geometry("_preview", line, _line_material(3.0))
        self.window.post_redraw()

    def _finish_lasso(self) -> None:
        if len(self._lasso) < 3:
            self._update_status("lasso needs at least 3 points")
            return
        pts = np.stack(self._lasso)
        self._lasso = []
        self._lasso_gen += 1
        self._clear_preview()
        base = float(pts[:, 2].min()) - 0.03
        # Default the top to hug whatever points the outline actually contains.
        probe = SimpleNamespace(shape="polygon", vertices=pts[:, :2],
                                z_range=(base, base + 2.5))
        inside = points_in_rock(self._visible_pts, probe)
        if inside.any():
            top = float(np.percentile(self._visible_pts[inside, 2], 98.0)) + 0.03
            top = max(top, base + 0.05)
        else:
            top = base + self._box_height
        rock = self.labelset.add_polygon(pts[:, :2], base, top)
        self._save()
        self._set_selected(rock.id)
        self._refresh_rock_list()
        self._update_status(f"placed lasso #{rock.id} - adjust base/top z sliders")

    def _cancel_lasso(self, quiet: bool = False) -> None:
        had = bool(self._lasso)
        self._lasso = []
        self._lasso_gen += 1
        self._clear_preview()
        if had and not quiet:
            self._update_status("lasso cancelled")

    # -- mouse dispatch -------------------------------------------------------

    def _on_mouse(self, event):
        Type = gui.MouseEvent.Type
        Result = gui.Widget.EventCallbackResult
        x, y = self._scene_xy(event)

        if event.type == Type.BUTTON_DOWN:
            place = event.is_modifier_down(gui.KeyModifier.SHIFT)
            select = event.is_modifier_down(gui.KeyModifier.CTRL)
            if place or select:
                self._pick_depth(x, y, lambda p: self._handle_pick(p, place))
                return Result.CONSUMED
            if not event.is_button_down(gui.MouseButton.LEFT):
                return Result.IGNORED  # right/middle drags always navigate

            if self._double_click(x, y):
                if self.tool == "lasso" and self._lasso:
                    self._finish_lasso()
                else:
                    self._recenter_pivot(x, y)
                return Result.CONSUMED

            if self.tool == "box":
                self._box_anchor = self._box_corner = None
                self._box_active = True

                def got_anchor(p, x=x, y=y):
                    if not self._box_active:
                        return
                    if p is None:
                        p = self._ray_hit_z(x, y, self._ground_z())
                    if p is None:
                        self._box_active = False
                        return
                    self._box_anchor = np.asarray(p, float)
                    self._box_corner = self._box_anchor.copy()
                    self._update_box_preview()

                self._pick_depth(x, y, got_anchor)
                return Result.CONSUMED

            if self.tool == "lasso":
                def got_vertex(p, x=x, y=y, gen=self._lasso_gen):
                    if gen != self._lasso_gen:
                        return  # the lasso was closed/cancelled while this pick ran
                    if p is None:
                        z = float(self._lasso[-1][2]) if self._lasso else self._ground_z()
                        p = self._ray_hit_z(x, y, z)
                    if p is None:
                        return
                    self._lasso.append(np.asarray(p, float))
                    self._update_lasso_preview()
                    self._update_status(
                        f"lasso: {len(self._lasso)} points - Enter or double-click closes")

                self._pick_depth(x, y, got_vertex)
                return Result.CONSUMED
            return Result.IGNORED

        if event.type == Type.DRAG and self._box_active:
            if self._box_anchor is not None:
                p = self._ray_hit_z(x, y, float(self._box_anchor[2]))
                if p is not None:
                    self._box_corner = p
                    self._update_box_preview()
            return Result.CONSUMED

        if event.type == Type.BUTTON_UP and self._box_active:
            self._finish_box()
            return Result.CONSUMED

        return Result.IGNORED

    def _snap(self, pos: np.ndarray) -> np.ndarray:
        """Snap a picked position to the nearest visible cloud point."""
        _d, i = self._tree.query(np.asarray(pos, float))
        return self._visible_pts[i].astype(float)

    #: The labeler keeps a KD-tree of the visible cloud anyway, so its orbit
    #: pivot lands exactly on a point rather than on the depth buffer's guess.
    _snap_pivot = _snap

    def _handle_pick(self, pos: np.ndarray | None, place: bool) -> None:
        if pos is None:
            self._update_status("nothing under the cursor")
            return
        snapped = self._snap(pos)
        if place:
            rock = self.labelset.add(snapped, self.default_radius)
            self._save()
            self._set_selected(rock.id)
            self._refresh_rock_list()
            self._update_status(f"placed #{rock.id}")
        else:
            if not self.labelset.rocks:
                return
            centers, _ = self.labelset.centers_radii()
            j = int(np.argmin(np.linalg.norm(centers - snapped, axis=1)))
            self._set_selected(self.labelset.rocks[j].id)
            self._refresh_rock_list()
            self._update_status()

    def _set_selected(self, rock_id: int | None) -> None:
        old = self.selected_id
        self.selected_id = rock_id
        for rid in (old, rock_id):
            rock = self.labelset.get(rid) if rid else None
            if rock is not None:
                self._draw_rock(rock)
        self._sync_edit_widgets()
        self._set_highlight(self._selected_rock())

    def _mutate_selected(self, d_size: float = 0.0, d_center=(0, 0, 0)) -> None:
        rock = self._selected_rock()
        if rock is None:
            return
        if d_size:
            if rock.shape == "box":
                base = rock.center[2] - rock.size[2] / 2.0
                rock.size = np.maximum(np.asarray(rock.size, float) + d_size, RADIUS_STEP)
                rock.center = np.array([rock.center[0], rock.center[1],
                                        base + rock.size[2] / 2.0])
                sync_bounds(rock)
            elif rock.shape == "polygon":
                z0, z1 = rock.z_range
                rock.z_range = (z0, max(z1 + d_size, z0 + 0.01))
                sync_bounds(rock)
            else:
                rock.radius = max(rock.radius + d_size, RADIUS_STEP)
        if any(d_center):
            translate_rock(rock, d_center)
        self._sync_edit_widgets()
        self._after_edit(rock)
        self._update_status()

    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        k = event.key
        # WASD etc. belong to the fly controller; only Esc leaves fly mode.
        flying = self.camera_key_result(event)
        if flying is not None:
            return flying
        if k == gui.KeyName.ENTER:
            if self.tool == "lasso" and self._lasso:
                self._finish_lasso()
        elif k == gui.KeyName.N:
            self._set_tool("navigate")
        elif k == gui.KeyName.B:
            self._set_tool("box")
        elif k == gui.KeyName.L:
            self._set_tool("lasso")
        elif k == gui.KeyName.C:
            self.color_mode = "reflectivity" if self.color_mode == "height" else "height"
            self.color_combo.selected_index = 0 if self.color_mode == "height" else 1
            self._rebuild_cloud()
        elif k == gui.KeyName.EQUALS:  # '+' is Shift+'=' on US layouts
            self._mutate_selected(d_size=RADIUS_STEP)
        elif k == gui.KeyName.MINUS:
            self._mutate_selected(d_size=-RADIUS_STEP)
        elif k == gui.KeyName.LEFT:
            self._mutate_selected(d_center=(-NUDGE_STEP, 0, 0))
        elif k == gui.KeyName.RIGHT:
            self._mutate_selected(d_center=(NUDGE_STEP, 0, 0))
        elif k == gui.KeyName.UP:
            self._mutate_selected(d_center=(0, NUDGE_STEP, 0))
        elif k == gui.KeyName.DOWN:
            self._mutate_selected(d_center=(0, -NUDGE_STEP, 0))
        elif k == gui.KeyName.PAGE_UP:
            self._mutate_selected(d_center=(0, 0, NUDGE_STEP))
        elif k == gui.KeyName.PAGE_DOWN:
            self._mutate_selected(d_center=(0, 0, -NUDGE_STEP))
        elif k in (gui.KeyName.X, gui.KeyName.DELETE):
            self._delete_selected()
        elif k == gui.KeyName.F:
            self._focus_selected()
        elif k == gui.KeyName.S:
            self._save_clicked()
        elif k == gui.KeyName.ESCAPE and (self._lasso or self._box_active):
            self._box_active = False
            self._box_anchor = self._box_corner = None
            self._cancel_lasso()
            self._clear_preview()
        elif k in (gui.KeyName.Q, gui.KeyName.ESCAPE):
            self._save()
            self.window.close()
            return gui.Widget.EventCallbackResult.HANDLED
        else:
            return gui.Widget.EventCallbackResult.IGNORED
        self._update_status()
        return gui.Widget.EventCallbackResult.HANDLED

    def run(self) -> None:
        gui.Application.instance.run()


def run_labeler_gui(xyz, inten, labelset, save_path, cfg, z_min=None, z_max=None) -> None:
    print("\n" + HELP_TEXT + "\n")
    try:
        app = _LabelerApp(xyz, inten, labelset, save_path, cfg, z_min, z_max)
    except Exception as e:  # e.g. no display / GL context
        print(f"GUI viewer failed to start ({e}); falling back to legacy picking viewer.")
        run_labeler_fallback(xyz, inten, labelset, save_path, cfg)
        return
    app.run()
    labelset.save(save_path)
    print(f"saved {len(labelset.rocks)} labels to {save_path}")


# ===========================================================================
# Dataset frame browser (rocklabel preview)
# ===========================================================================

@dataclass
class FrameView:
    """Everything the browser needs to draw one dataset frame."""

    index: int                       # the frame's on-disk index (frame_XXXXXX)
    time_s: float
    point_sets: list = field(default_factory=list)   # [(name, xyz [N,3], rgb [N,3] or (3,), size)]
    spheres: list = field(default_factory=list)      # [(center, radius)]
    pose: np.ndarray | None = None                   # 4x4 robot pose (drawn as triad)
    stats_lines: list = field(default_factory=list)  # shown in the side panel


BROWSER_HELP = CAMERA_HELP + """
keys  (click the 3D view first so it has focus)
  Left / Right   previous / next frame
  space          play / pause
  Home / End     first / last frame
  R              reset view
  Q / Esc        quit
fly mode: WASD + drag to look; Esc returns to orbit"""


class FrameBrowserApp(PivotCamera):
    """Skip through generated dataset frames with a transport bar.

    ``loader(i, window_idx)`` returns a :class:`FrameView` for frame position
    ``i`` with the accumulation window ``window_labels[window_idx]`` applied;
    frames are loaded lazily as you scrub, so startup is instant even for
    long runs.
    """

    def __init__(self, loader, n_frames: int, title: str = "rocklabel - preview",
                 start: int = 0, legend_lines: list[tuple[str, tuple]] | None = None,
                 fps: float = 5.0, window_labels: list[str] | None = None,
                 window_default: int = 0, extras_builder=None):
        self.loader = loader
        self.n_frames = int(n_frames)
        self.pos = int(np.clip(start, 0, self.n_frames - 1))
        self.playing = False
        self.fps = float(fps)
        self.point_size = 5.0
        self.window_labels = window_labels or ["Current frame"]
        self.window_idx = int(np.clip(window_default, 0, len(self.window_labels) - 1))
        self._last_advance = 0.0
        self._camera_ready = False
        self._geom_names: list[str] = []
        self._bbox = None

        app = _ensure_app()
        self.window = app.create_window(title, 1500, 940)
        em = self.window.theme.font_size

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        _setup_scene(self.scene)
        self.install_camera(self.scene, self.window)
        self.scene.set_on_key(self._on_key)

        # --- side panel ---
        self.panel = gui.Vert(0.5 * em, gui.Margins(0.8 * em, 0.8 * em, 0.8 * em, 0.8 * em))
        _header(self.panel, em, "DATASET PREVIEW", "what was written to disk, frame by frame")

        stats_section = gui.CollapsableVert("Frame", 0.35 * em, gui.Margins(em, 0, 0, 0))
        stats_section.set_is_open(True)
        self.stats_label = gui.Label("")
        stats_section.add_child(self.stats_label)
        self.panel.add_child(stats_section)

        combine = gui.CollapsableVert("Combine frames", 0.35 * em, gui.Margins(em, 0, 0, 0))
        combine.set_is_open(True)
        hint = gui.Label("accumulate trailing data\ninto the view:")
        hint.text_color = DIM
        combine.add_child(hint)
        self.window_combo = gui.Combobox()
        for label in self.window_labels:
            self.window_combo.add_item(label)
        self.window_combo.selected_index = self.window_idx
        self.window_combo.set_on_selection_changed(self._on_window_combo)
        combine.add_child(self.window_combo)
        self.panel.add_child(combine)

        if legend_lines:
            legend = gui.CollapsableVert("Legend", 0.35 * em, gui.Margins(em, 0, 0, 0))
            legend.set_is_open(True)
            for text, rgb in legend_lines:
                lbl = gui.Label("[#] " + text)
                lbl.text_color = gui.Color(*rgb)
                legend.add_child(lbl)
            self.panel.add_child(legend)

        # Callers (e.g. the confidence viewer) can inject extra panel sections
        # here; call app.refresh() from their callbacks to recolor the frame.
        if extras_builder is not None:
            extras_builder(self, self.panel, em)

        view = gui.CollapsableVert("View", 0.35 * em, gui.Margins(em, 0, 0, 0))
        view.set_is_open(True)
        view.add_child(self.make_nav_combo())
        cam_hint = gui.Label(CAMERA_HINT)
        cam_hint.text_color = DIM
        view.add_child(cam_hint)
        view.add_child(gui.Label("Point size"))
        size_slider = gui.Slider(gui.Slider.DOUBLE)
        size_slider.set_limits(1.0, 12.0)
        size_slider.double_value = self.point_size
        size_slider.set_on_value_changed(self._on_point_size)
        view.add_child(size_slider)
        view.add_child(gui.Label("Playback speed (frames/s)"))
        fps_slider = gui.Slider(gui.Slider.DOUBLE)
        fps_slider.set_limits(1.0, 30.0)
        fps_slider.double_value = self.fps
        fps_slider.set_on_value_changed(self._on_fps)
        view.add_child(fps_slider)
        reset_btn = gui.Button("Reset view")
        reset_btn.set_on_clicked(self._reset_view)
        view.add_child(reset_btn)
        self.panel.add_child(view)

        help_section = gui.CollapsableVert("Shortcuts", 0.35 * em, gui.Margins(em, 0, 0, 0))
        help_section.set_is_open(False)
        help_section.add_child(_help_block(BROWSER_HELP))
        self.panel.add_child(help_section)
        self.panel.add_stretch()

        # --- transport bar ---
        self.transport = gui.Horiz(0.5 * em, gui.Margins(0.8 * em, 0.4 * em, 0.8 * em, 0.4 * em))

        def tbutton(text, cb, tooltip=""):
            b = gui.Button(text)
            b.horizontal_padding_em = 0.8
            b.vertical_padding_em = 0.2
            if tooltip:
                b.tooltip = tooltip
            b.set_on_clicked(cb)
            self.transport.add_child(b)
            return b

        tbutton("|<", lambda: self._go(0), "first frame (Home)")
        tbutton("<", lambda: self._go(self.pos - 1), "previous frame (Left)")
        self.play_btn = tbutton("Play", self._toggle_play, "play / pause (space)")
        tbutton(">", lambda: self._go(self.pos + 1), "next frame (Right)")
        tbutton(">|", lambda: self._go(self.n_frames - 1), "last frame (End)")

        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(0, max(self.n_frames - 1, 1))
        self.slider.int_value = self.pos
        self.slider.set_on_value_changed(self._on_slider)
        self.transport.add_child(self.slider)
        self.frame_label = gui.Label("")
        self.frame_label.text_color = DIM
        self.transport.add_child(self.frame_label)

        self.window.add_child(self.scene)
        self.window.add_child(self.panel)
        self.window.add_child(self.transport)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_tick_event(self._on_tick)

        self._show_frame(self.pos)

    # -- layout -------------------------------------------------------------

    def _on_layout(self, ctx) -> None:
        rect = self.window.content_rect
        em = ctx.theme.font_size
        panel_w = min(int(19 * em), rect.width // 3)
        bar_h = int(2.6 * em)
        self.scene.frame = gui.Rect(rect.x, rect.y, rect.width - panel_w, rect.height - bar_h)
        self.panel.frame = gui.Rect(rect.get_right() - panel_w, rect.y, panel_w, rect.height - bar_h)
        self.transport.frame = gui.Rect(rect.x, rect.get_bottom() - bar_h, rect.width, bar_h)

    # -- frame display ------------------------------------------------------

    def _show_frame(self, i: int) -> None:
        i = int(np.clip(i, 0, self.n_frames - 1))
        self.pos = i
        view: FrameView = self.loader(i, self.window_idx)

        for name in self._geom_names:
            if self.scene.scene.has_geometry(name):
                self.scene.scene.remove_geometry(name)
        self._geom_names.clear()

        combined_min, combined_max = None, None
        for name, xyz, rgb, size in view.point_sets:
            if len(xyz) == 0:
                continue
            pcd = _point_cloud(xyz, rgb)
            self.scene.scene.add_geometry(name, pcd, _unlit_material(size * self.point_size / 5.0))
            self._geom_names.append(name)
            lo, hi = np.min(xyz, axis=0), np.max(xyz, axis=0)
            combined_min = lo if combined_min is None else np.minimum(combined_min, lo)
            combined_max = hi if combined_max is None else np.maximum(combined_max, hi)

        for j, (center, radius) in enumerate(view.spheres):
            wire = o3d.geometry.LineSet.create_from_triangle_mesh(_sphere_mesh(center, radius))
            wire.paint_uniform_color([0.95, 0.20, 0.15])
            name = f"sphere_{j}"
            self.scene.scene.add_geometry(name, wire, _line_material())
            self._geom_names.append(name)

        if view.pose is not None:
            triad = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            triad.transform(np.asarray(view.pose, np.float64))
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            self.scene.scene.add_geometry("pose", triad, mat)
            self._geom_names.append("pose")

        if combined_min is not None:
            # Track the current content's bounds so "Reset view" fits what's
            # on screen; only aim the camera automatically on the first frame.
            self._bbox = o3d.geometry.AxisAlignedBoundingBox(combined_min.astype(float),
                                                             combined_max.astype(float))
            if not self._camera_ready:
                self.scene.setup_camera(60.0, self._bbox, self._bbox.get_center())
                self._camera_ready = True

        self.slider.int_value = self.pos
        self.frame_label.text = f"frame {view.index:06d}  ·  {self.pos + 1}/{self.n_frames}"
        self.stats_label.text = "\n".join(
            [f"frame {view.index:06d}  ({self.pos + 1} of {self.n_frames})",
             f"t = {view.time_s:.3f} s"] + list(view.stats_lines)
        )
        self.window.set_needs_layout()
        self.window.post_redraw()

    def refresh(self) -> None:
        """Redraw the current frame (for extras whose state affects colors)."""
        self._show_frame(self.pos)

    def _go(self, i: int) -> None:
        self.playing = False
        self._sync_play_btn()
        self._show_frame(i)

    def _toggle_play(self) -> None:
        if self.pos >= self.n_frames - 1:
            self.pos = -1  # play from the top when at the end
        self.playing = not self.playing
        self._last_advance = time.monotonic()
        self._sync_play_btn()

    def _sync_play_btn(self) -> None:
        self.play_btn.text = "Pause" if self.playing else "Play"
        self.window.set_needs_layout()
        self.window.post_redraw()

    def _on_tick(self) -> bool:
        if not self.playing:
            return False
        now = time.monotonic()
        if now - self._last_advance < 1.0 / max(self.fps, 1e-3):
            return False
        self._last_advance = now
        if self.pos + 1 >= self.n_frames:
            self.playing = False
            self._sync_play_btn()
            return True
        self._show_frame(self.pos + 1)
        return True

    # -- callbacks ----------------------------------------------------------

    def _on_slider(self, value) -> None:
        self.playing = False
        self._sync_play_btn()
        self._show_frame(int(value))

    def _on_window_combo(self, _text, index) -> None:
        self.window_idx = int(index)
        self._show_frame(self.pos)

    def _on_point_size(self, value) -> None:
        self.point_size = float(value)
        self._show_frame(self.pos)

    def _on_fps(self, value) -> None:
        self.fps = float(value)

    def _reset_view(self) -> None:
        if self._bbox is not None:
            self.scene.setup_camera(60.0, self._bbox, self._bbox.get_center())
            self.window.post_redraw()

    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        # WASD etc. belong to the fly controller; only Esc leaves fly mode.
        flying = self.camera_key_result(event)
        if flying is not None:
            return flying
        k = event.key
        if k == gui.KeyName.LEFT:
            self._go(self.pos - 1)
        elif k == gui.KeyName.RIGHT:
            self._go(self.pos + 1)
        elif k == gui.KeyName.SPACE:
            self._toggle_play()
        elif k == gui.KeyName.HOME:
            self._go(0)
        elif k == gui.KeyName.END:
            self._go(self.n_frames - 1)
        elif k == gui.KeyName.R:
            self._reset_view()
        elif k in (gui.KeyName.Q, gui.KeyName.ESCAPE):
            self.window.close()
            return gui.Widget.EventCallbackResult.HANDLED
        else:
            return gui.Widget.EventCallbackResult.IGNORED
        return gui.Widget.EventCallbackResult.HANDLED

    def run(self) -> None:
        gui.Application.instance.run()


# -- legacy fallback ---------------------------------------------------------

_FALLBACK_HELP = """\
Fallback labeling loop:
  1. A window opens. Shift+click points on rocks (shift+click again to unpick).
  2. Close the window (q). Every picked point becomes a new sphere label.
  3. Edit at the terminal prompt:
       list                     show all labels
       del <id>                 delete a label
       radius <id> <meters>     set a label's radius
       move <id> <dx> <dy> <dz> nudge a label center (meters)
       pick                     reopen the window to pick more points
       done                     save and exit"""


def run_labeler_fallback(xyz, inten, labelset, save_path, cfg) -> None:
    print("\n" + _FALLBACK_HELP + "\n")
    default_radius = cfg["labeler"]["default_rock_radius_m"]

    def pick_round():
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
        pcd.colors = o3d.utility.Vector3dVector(height_colors(xyz[:, 2]))
        vis = o3d.visualization.VisualizerWithVertexSelection()
        vis.create_window("rocklabel fallback - shift+click rocks, then close window")
        vis.add_geometry(pcd)
        vis.run()
        picked = vis.get_picked_points()
        vis.destroy_window()
        for p in picked:
            rock = labelset.add(np.asarray(p.coord, float), default_radius)
            print(f"added rock #{rock.id} at {np.round(rock.center, 3).tolist()}")
        labelset.save(save_path)

    pick_round()
    while True:
        try:
            line = input("rocklabel> ").strip()
        except EOFError:
            break
        if not line:
            continue
        cmd, *rest = line.split()
        try:
            if cmd == "list":
                for r in labelset.rocks:
                    print(f"  #{r.id}  center {np.round(r.center, 3).tolist()}  radius {r.radius:.2f}")
            elif cmd == "del":
                labelset.remove(int(rest[0]))
            elif cmd == "radius":
                labelset.get(int(rest[0])).radius = float(rest[1])
            elif cmd == "move":
                rock = labelset.get(int(rest[0]))
                rock.center = rock.center + np.asarray([float(v) for v in rest[1:4]])
            elif cmd == "pick":
                pick_round()
            elif cmd in ("done", "quit", "q"):
                break
            else:
                print(_FALLBACK_HELP)
                continue
            labelset.save(save_path)
        except (IndexError, ValueError, AttributeError):
            print(f"bad command: {line!r}")
    labelset.save(save_path)
    print(f"saved {len(labelset.rocks)} labels to {save_path}")


# -- shared simple renderer (driftcheck) -------------------------------------

def show_point_sets(sets, spheres=(), pose=None, window="rocklabel", point_size=5.0) -> None:
    """Render [(xyz, rgb-or-color), ...] plus label-sphere wireframes on a dark
    background with fat points. rgb may be one color (3,) or per-point [N, 3]."""
    geoms = []
    for xyz, rgb in sets:
        geoms.append(_point_cloud(xyz, rgb))
    for center, radius in spheres:
        wire = o3d.geometry.LineSet.create_from_triangle_mesh(_sphere_mesh(center, radius))
        wire.paint_uniform_color([0.9, 0.1, 0.1])
        geoms.append(wire)
    if pose is not None:
        triad = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        triad.transform(np.asarray(pose, np.float64))
        geoms.append(triad)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array(BG_COLOR[:3])
    opt.point_size = float(point_size)
    vis.run()
    vis.destroy_window()


# -- driftcheck --------------------------------------------------------------

def show_driftcheck(early_xyz, late_xyz, center, radius) -> None:
    show_point_sets(
        [(early_xyz, [0.2, 0.45, 1.0]),   # blue = early
         (late_xyz, [1.0, 0.55, 0.1])],   # orange = late
        spheres=[(center, radius)],
        window="rocklabel driftcheck (blue=early 10%, orange=late 10%)",
        point_size=6.0,
    )
