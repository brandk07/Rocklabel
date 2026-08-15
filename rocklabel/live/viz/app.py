"""Real-time Open3D visualizer for the live rocklabel pipeline.

Uses Open3D's ``gui`` / ``rendering`` application API (``gui.SceneWidget`` +
``rendering.Open3DScene``) with a settings panel on the right, styled like the
offline rocklabel viewers: every runtime knob — layer visibility, color mode,
point size, recording, and all live-model scoring parameters (region, update
interval, threshold, display mode) — is adjustable while running.

Under ``web_ui=True`` (``rocklabel live --web-ui``) that panel is not built at
all and the scene gets the whole window: the controls live in a browser page
instead (:mod:`rocklabel.live.webui`), which drives this class through the
public control API below — ``set_color_mode``, ``set_layer``, ``set_threshold``
and friends, each posted onto the GUI thread by :meth:`VizApp.post`. Those
methods are the single implementation of each knob; the panel's widget
callbacks call exactly the same ones.

The scene shows, simultaneously:

* the most recent raw points (a rolling buffer, small point size),
* the accumulated world-frame cloud — the last N frames, N set live by the
  View panel's "Accum frames" box, and
* the reconstructed 2.5D surface mesh,

refreshing at a display rate fully decoupled from ingestion (and from model
scoring): a background "tick" thread wakes at the target FPS and posts a scene
update onto the GUI (main) thread via ``Application.post_to_main_thread``.

Color modes (``V`` key or the panel combobox):

* **height** — z colormap,
* **reflectivity** — multiScan RSSI per point,
* **model** — with ``--model``: each point takes its nearest scored center's
  rock probability (turbo blue→red), or a binary detections view at the
  decision threshold; points the model has no prediction for (outside the
  scoring region, or not yet scored) keep dimmed height colors.

Camera controls are the labeler's (:mod:`rocklabel.camera`): left-drag orbits
around a pivot, and a double-click moves that pivot onto the clicked point.

Keyboard shortcuts: P points · M mesh · C accumulated cloud · B sensor box ·
V cycle colors · S record on/off · L re-measure mount tilt · R reset (replay:
restart) · space pause/play · ←/→ seek ±5 s (replay) · Q/esc quit.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from rocklabel.camera import CAMERA_HELP, CAMERA_HINT, PivotCamera
from rocklabel.live.colormap import apply_colormap, normalize, reflectivity_values
from rocklabel.live.config import AppConfig
from rocklabel.live.motion import quat_to_matrix
from rocklabel.live.pipeline import IngestEngine

_POINTS_NAME = "raw_points"
_ACCUM_NAME = "accum_points"
_MESH_NAME = "surface_mesh"
_BOX_NAME = "lidar_box"

#: Settle time before a slider drag is applied as a seek (coalesces scrubbing).
_SEEK_DEBOUNCE_SEC = 0.25
#: Tolerance for recognising a slider callback as an echo of our own mirroring.
#: Open3D round-trips the value through float32 (see _synced_values), so the
#: echo comes back off by up to half a float32 ulp — 1.9e-6 s at a 32 s
#: position, and it doubles at every power of two. 1 ms is comfortably above
#: that for any recording length we replay, and far below _SEEK_MIN_DELTA_SEC,
#: so it can never swallow a scrub that would actually have moved playback.
_SLIDER_ECHO_EPS_SEC = 1e-3
#: Seeks smaller than this are dropped: a tiny backward step costs a full
#: rewind + re-fuse, which is never what such a nudge meant.
_SEEK_MIN_DELTA_SEC = 0.1

# --------------------------------------------------------------------------- #
# Panel palette — the same tokens as the `rocklabel dash` dark theme, so the two
# surfaces of this project read as one tool. Open3D wants gui.Color(r, g, b).
# --------------------------------------------------------------------------- #
BG_COLOR = [0.051, 0.051, 0.049, 1.0]           # #0d0d0d — scene plane
ACCENT = gui.Color(0.224, 0.529, 0.898)         # #3987e5
DIM = gui.Color(0.537, 0.529, 0.506)            # #898781 — muted ink
_SECOND = gui.Color(0.765, 0.761, 0.718)        # #c3c2b7 — secondary ink
_GOOD = gui.Color(0.047, 0.639, 0.047)          # #0ca30c
_WARN = gui.Color(0.980, 0.698, 0.098)          # #fab219
_CRIT = gui.Color(0.816, 0.231, 0.231)          # #d03b3b

#: Detection-view colors (same as the offline mcap replay viewer).
_DET_ROCK = (1.00, 0.25, 0.15)
_DET_CLEAR = (0.30, 0.35, 0.42)
#: Brightness of height colors on points the model has no prediction for.
_UNSCORED_DIM = 0.35

#: (key, action) pairs — laid out as an aligned two-column grid, not one blob.
SHORTCUTS = [
    ("dbl-click", "set orbit pivot"),
    ("P", "raw points"),
    ("M", "surface mesh"),
    ("C", "accumulated cloud"),
    ("B", "sensor pose box"),
    ("V", "cycle color mode"),
    ("S", "start / stop recording"),
    ("L", "re-measure mount tilt"),
    ("R", "reset surface (replay: restart)"),
    ("space", "pause / resume"),
    ("← →", "replay: seek ∓5 s"),
    ("Q esc", "quit"),
]
HELP_TEXT = CAMERA_HELP + "\n" + "\n".join(  # kept for the docstring
    f"{k}  {v}" for k, v in SHORTCUTS)


def _compact(n: int) -> str:
    """Counts that fit a one-line readout cell: 812, 4.9k, 498k, 1.2M."""
    if n < 1_000:
        return str(int(n))
    if n < 10_000:
        return f"{n / 1e3:.1f}k"
    if n < 1_000_000:
        return f"{n / 1e3:.0f}k"
    return f"{n / 1e6:.1f}M"


class VizApp(PivotCamera):
    """Owns the GUI window and drives live updates from an :class:`IngestEngine`.

    ``scorer`` (a :class:`rocklabel.live.scoring.LiveScorer`, optional) adds
    the "model" color mode plus a Model panel section whose controls edit the
    scorer's :class:`~rocklabel.live.scoring.ScoreSettings` live.
    """

    def __init__(self, config: AppConfig, engine: IngestEngine, scorer=None,
                 web_ui: bool = False) -> None:
        self._cfg = config
        self._engine = engine
        self._scorer = scorer
        #: The browser control panel is driving this session: drop the docked
        #: settings panel so the scene gets the whole window. The keyboard
        #: shortcuts and the replay transport bar stay.
        self._web_ui = bool(web_ui)

        self._show_points = config.display.show_points
        self._show_mesh = config.display.show_mesh
        self._show_accum = config.display.show_accum
        self._show_box = config.display.show_lidar_box
        self._color_modes = ["height", "reflectivity", "reflectivity_stretch"] + (
            ["model"] if scorer is not None else []
        )
        self._color_mode = (
            config.display.color_mode
            if config.display.color_mode in self._color_modes
            else "height"
        )
        self._model_display = 0  # 0 = confidence colormap, 1 = detections @ thr
        #: Display-only crop: hide points outside the scoring region (walls /
        #: ceiling) from the raw + accumulated clouds. Defaults on with a
        #: model; without one it uses the fusion crop's z band when enabled.
        self._crop_view = scorer is not None
        #: Cache key of the last accum rebuild in model mode: recoloring the
        #: full accumulated cloud (KD-tree lookup on up to 500k points) is only
        #: worth doing when a new scoring pass landed, not every render tick.
        self._accum_model_key: tuple | None = None
        self._fps = max(1.0, float(config.display.fps))
        self._tick_count = 0
        self._last_mesh_data = None  # identity cache: skip o3d rebuild if unchanged
        #: Replay mode: the source is an McapReplaySource with transport controls.
        self._replay = bool(getattr(engine.source, "is_replay", False))

        self._window: gui.Window | None = None
        self._widget: gui.SceneWidget | None = None
        self._scene: rendering.Open3DScene | None = None
        # Panel widgets. All stay None under --web-ui, where no panel is built:
        # the control API and _refresh_stats both have to tolerate their absence.
        self._panel: gui.ScrollableVert | None = None
        self._color_combo: gui.Combobox | None = None
        self._layer_checks: dict[str, gui.Checkbox] = {}
        self._rec_btn: gui.Button | None = None
        self._level_label: gui.Label | None = None
        # Readout cells, filled by _build_panel and written by _refresh_stats.
        # Each holds exactly one line — see _readout for why that matters.
        self._stat_rate = self._stat_cells = self._stat_pose = self._stat_state = None
        self._stat_accum: gui.Label | None = None
        self._rec_state = self._rec_stats = self._rec_file = None
        self._rec_label: gui.Label | None = None          # alias of _rec_state
        self._model_map = self._model_pass = self._model_region = None
        self._model_warn: gui.Label | None = None
        self._model_label: gui.Label | None = None        # alias of _model_map

        # Replay transport bar (created only in replay mode).
        self._play_btn: gui.Button | None = None
        self._time_label: gui.Label | None = None
        self._slider: gui.Slider | None = None
        self._pending_seek: float | None = None
        self._pending_seek_time = 0.0
        #: Playback position when the pending scrub landed. The settled seek is
        #: measured against this, not the live position, which has advanced by a
        #: debounce window by the time the seek fires.
        self._pending_seek_from = 0.0
        self._last_user_seek = 0.0
        # Open3D fires on_value_changed for programmatic slider sets too — and
        # delivers it asynchronously on a later event-loop tick, so a plain
        # "am I syncing" flag can't filter it. Remember the values we set and
        # drop callback echoes that match one; anything else is a real scrub.
        self._synced_values: deque[float] = deque(maxlen=8)

        self._tick_stop = threading.Event()
        self._tick_thread: threading.Thread | None = None
        self._last_console_stat = 0.0

        # Materials (created once the Application is initialized).
        self._point_mat = rendering.MaterialRecord()
        self._point_mat.shader = "defaultUnlit"
        self._point_mat.point_size = float(config.display.point_size)

        self._accum_mat = rendering.MaterialRecord()
        self._accum_mat.shader = "defaultUnlit"
        self._accum_mat.point_size = max(1.0, float(config.display.point_size) - 1.0)

        self._mesh_mat = rendering.MaterialRecord()
        self._mesh_mat.shader = "defaultLit"
        self._mesh_mat.base_color = (1.0, 1.0, 1.0, 1.0)  # let vertex colors show

        self._box_mat = rendering.MaterialRecord()
        self._box_mat.shader = "defaultLit"
        self._box_mat.base_color = (1.0, 1.0, 1.0, 1.0)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Build the window and run the GUI event loop (blocks until closed)."""
        # Open3D's Filament renderer needs X11 window handles, but its windowing
        # layer picks the native Wayland backend when XDG_SESSION_TYPE=wayland,
        # which segfaults in XSendEvent during create_window. Claim an x11
        # session so it falls back to XWayland.
        if sys.platform.startswith("linux") and os.environ.get("XDG_SESSION_TYPE") == "wayland":
            os.environ["XDG_SESSION_TYPE"] = "x11"
        app = gui.Application.instance
        app.initialize()

        d = self._cfg.display
        title = "rocklabel - live" if not self._replay else "rocklabel - live replay"
        self._window = app.create_window(title, d.window_width, d.window_height)
        self._window.set_on_close(self._on_close)
        self._window.set_on_layout(self._on_layout)
        em = self._window.theme.font_size

        self._widget = gui.SceneWidget()
        self._widget.scene = rendering.Open3DScene(self._window.renderer)
        self._widget.scene.set_background(BG_COLOR)
        self._widget.scene.show_axes(True)
        self.install_camera(self._widget, self._window)
        self._widget.set_on_key(self._on_key)
        self._window.add_child(self._widget)
        self._scene = self._widget.scene

        if not self._web_ui:
            self._build_panel(em)
            self._window.add_child(self._panel)

        self._add_lidar_box()
        self.reset_camera()

        # Start ingestion + the display-rate tick that posts updates.
        self._engine.start()
        if self._scorer is not None:
            self._scorer.start()
        if self._replay:
            # After engine/source start: the slider limits need the recording's
            # duration, which the source only knows once it has been started.
            self._build_transport_bar()
        if self._cfg.record.autostart and not self._replay:
            path = self._engine.start_recording()
            if path:
                print(f"[rocklabel] recording -> {path}", flush=True)
        self._tick_thread = threading.Thread(
            target=self._tick_loop, name="viz-tick", daemon=True
        )
        self._tick_thread.start()

        app.run()  # blocks on the main thread until the window closes

    def _on_close(self) -> bool:
        self._tick_stop.set()
        if self._scorer is not None:
            self._scorer.stop()
        path = self._engine.stop_recording()
        if path:
            print(f"[rocklabel] recording saved: {path}", flush=True)
        self._engine.stop()
        return True  # allow the window to close

    # ------------------------------------------------------------------ #
    # Panel
    # ------------------------------------------------------------------ #
    # -- small builders shared by every section ------------------------------ #
    @staticmethod
    def _heading(text: str, color=None) -> gui.Label:
        lbl = gui.Label(text)
        lbl.text_color = color or DIM
        return lbl

    def _section(self, em: float, title: str, open_: bool = True) -> gui.CollapsableVert:
        """A collapsible card. Left margin only — the panel supplies the rest."""
        sec = gui.CollapsableVert(title, 0.3 * em, gui.Margins(0.9 * em, 0, 0, 0.5 * em))
        sec.set_is_open(open_)
        return sec

    @staticmethod
    def _grid(em: float, cols: int = 2) -> gui.VGrid:
        return gui.VGrid(cols, 0.4 * em, gui.Margins(0, 0.15 * em, 0, 0.15 * em))

    def _pair(self, grid: gui.VGrid, caption: str, widget: gui.Widget,
              tip: str = "") -> gui.Widget:
        """caption | control on ONE row.

        The old panel put every caption on its own line above its slider, which
        is what made the Model section a two-screen scroll. A two-column grid
        halves it and lines the controls up.
        """
        lbl = gui.Label(caption)
        lbl.text_color = _SECOND
        if tip:
            lbl.tooltip = tip
            widget.tooltip = tip
        grid.add_child(lbl)
        grid.add_child(widget)
        return widget

    def _readout(self, grid: gui.VGrid, caption: str, tip: str = "") -> gui.Label:
        """A live value cell. Single-line by construction: a Label whose line
        count changes at runtime silently corrupts the layout of everything
        below it (that was the old status/legend overlap)."""
        name = gui.Label(caption)
        name.text_color = DIM
        value = gui.Label("-")
        value.text_color = _SECOND
        if tip:
            name.tooltip = tip
        value.tooltip = tip
        grid.add_child(name)
        grid.add_child(value)
        return value

    #: Characters that fit the value column before Open3D wraps the label onto a
    #: second line — and a wrapped cell shifts every section below it.
    _VALUE_CHARS = 26

    @classmethod
    def _put(cls, cell: gui.Label, text: str) -> None:
        """Write one line into a readout cell; the full text lives in the tooltip.

        Sources like ``GroundLeveler.status()`` are written for a terminal and
        can run long or span lines, so they are flattened and elided here rather
        than being allowed to reflow the panel.
        """
        if cell is None:
            return
        flat = " ".join(str(text).split()) or "-"
        cell.tooltip = flat
        cell.text = (flat if len(flat) <= cls._VALUE_CHARS
                     else flat[: cls._VALUE_CHARS - 3] + "...")

    @staticmethod
    def _make_slider(kind, lo, hi, value, on_change) -> gui.Slider:
        """For values you explore by feel — threshold, point size, cadence."""
        s = gui.Slider(kind)
        s.set_limits(lo, hi)
        if kind == gui.Slider.INT:
            s.int_value = int(value)
        else:
            s.double_value = float(value)
        s.set_on_value_changed(on_change)
        return s

    @staticmethod
    def _make_number(kind, lo, hi, value, on_change, precision: int = 2) -> gui.NumberEdit:
        """For values you already know — the region bounds and the centers cap.

        A slider cannot land on exactly -1.50 without a fight, and those are
        numbers the operator arrives with (the handheld rig's floor band is
        -1.5 .. -0.5). A typed box is the right control for a known value; the
        sliders above it stay for the ones you tune by watching the scene.
        """
        n = gui.NumberEdit(kind)
        n.set_limits(lo, hi)
        if kind == gui.NumberEdit.INT:
            n.int_value = int(value)
        else:
            n.decimal_precision = precision
            n.double_value = float(value)
        n.set_on_value_changed(on_change)
        return n

    # ----------------------------------------------------------------------- #
    def _build_panel(self, em: float) -> None:
        self._panel = gui.ScrollableVert(
            0.45 * em, gui.Margins(0.8 * em, 0.7 * em, 0.8 * em, 0.8 * em))

        self._build_header(em)
        self._build_status_section(em)
        if not self._replay:
            self._build_recording_section(em)
        self._build_view_section(em)
        if self._engine.leveler.active:
            self._build_level_section(em)
        if self._scorer is not None:
            self._build_model_section(em)
        self._build_help_section(em)

    # -- header --------------------------------------------------------------- #
    def _build_header(self, em: float) -> None:
        row = gui.Horiz(0.4 * em)
        title = gui.Label("ROCKLABEL")
        title.text_color = ACCENT
        row.add_child(title)
        row.add_stretch()
        self._mode_label = gui.Label("REPLAY" if self._replay else "LIVE")
        self._mode_label.text_color = _SECOND
        row.add_child(self._mode_label)
        self._panel.add_child(row)

        if self._replay:
            sub = os.path.basename(getattr(self._engine.source, "path", "") or "recording")
        else:
            sub = f"source · {self._cfg.source.kind}"
        subtitle = gui.Label(sub)
        subtitle.text_color = DIM
        subtitle.tooltip = ("Where the points are coming from. 'udp' is the live "
                            "SICK multiScan; 'sim' is synthetic terrain.")
        self._panel.add_child(subtitle)

    # -- status --------------------------------------------------------------- #
    def _build_status_section(self, em: float) -> None:
        sec = self._section(em, "Status")
        grid = self._grid(em)
        self._stat_rate = self._readout(
            grid, "Throughput", "Points per second arriving from the source.")
        self._stat_cells = self._readout(
            grid, "Cells", "Occupied cells in the fused 2.5D height map.")
        self._stat_accum = self._readout(
            grid, "Accum", "Frames, points and time span currently held by the "
                           "accumulated cloud. 'cap' means the point ceiling is "
                           "dropping frames before the frame limit does.")
        self._stat_pose = self._readout(
            grid, "Pose", "SLAM / IMU state. 'locked' means scan-to-map odometry "
                         "is tracking; without it the sensor must stay put.")
        self._stat_state = self._readout(
            grid, "State", "Whether ingestion is running, paused, or replaying.")
        sec.add_child(grid)
        self._panel.add_child(sec)

    # -- recording ------------------------------------------------------------ #
    def _build_recording_section(self, em: float) -> None:
        sec = self._section(em, "Recording")
        self._rec_btn = gui.Button("Start recording   (S)")
        self._rec_btn.tooltip = ("Write the incoming stream to a native .mcap. "
                                 "Everything downstream reads that format directly.")
        self._rec_btn.set_on_clicked(self.toggle_recording)
        sec.add_child(self._rec_btn)

        grid = self._grid(em)
        self._rec_state = self._readout(grid, "State", "Recording on or off.")
        self._rec_stats = self._readout(
            grid, "Captured", "Elapsed time and total points written so far.")
        self._rec_file = self._readout(grid, "File", "Output path being written.")
        sec.add_child(grid)
        self._panel.add_child(sec)
        # Legacy alias: _refresh_stats used to write one blob into this label.
        self._rec_label = self._rec_state

    # -- view ----------------------------------------------------------------- #
    def _build_view_section(self, em: float) -> None:
        sec = self._section(em, "View")

        grid = self._grid(em)
        self._color_combo = gui.Combobox()
        for m in self._color_modes:
            self._color_combo.add_item(
                {"height": "Height", "reflectivity": "Reflectivity",
                 "reflectivity_stretch": "Reflectivity (contrast)",
                 "model": "Model prediction"}[m])
        self._color_combo.selected_index = self._color_modes.index(self._color_mode)
        self._color_combo.set_on_selection_changed(self._on_color_combo)
        self._pair(grid, "Color by", self._color_combo,
                   "What the points are colored by. V cycles it. 'Model "
                   "prediction' needs a checkpoint (--model).")

        size_slider = self._make_slider(gui.Slider.DOUBLE, 1.0, 8.0,
                                   self._cfg.display.point_size, self.set_point_size)
        self._pair(grid, "Point size", size_slider,
                   "Screen size of each drawn point. Larger reads better on a "
                   "sparse cloud; smaller shows fine surface detail.")
        sec.add_child(grid)

        sec.add_child(self._heading("Layers"))
        self._layer_checks = {}
        for attr, caption, tip in (
            ("_show_points", "Raw points  (P)",
             "The most recent scans, a rolling buffer. This is what the model scores."),
            ("_show_mesh", "Surface mesh  (M)",
             "The reconstructed 2.5D Kalman-fused surface."),
            ("_show_accum", "Accumulated cloud  (C)",
             "The world-frame cloud built from the last N frames — how many is "
             "the 'Accum frames' box below."),
            ("_show_box", "Sensor box  (B)",
             "An orange box marking the sensor's current world pose."),
        ):
            cb = gui.Checkbox(caption)
            cb.checked = getattr(self, attr)
            cb.tooltip = tip
            cb.set_on_checked(lambda v, a=attr: self.set_layer(a, v))
            self._layer_checks[attr] = cb
            sec.add_child(cb)

        agrid = self._grid(em)
        frames = self._make_number(gui.NumberEdit.INT, 1, 200_000,
                                   self._engine.accum_frames,
                                   self._on_accum_frames)
        self._pair(agrid, "Accum frames", frames,
                   "How many recent frames the accumulated cloud keeps before the "
                   "oldest drop out. Raise it to hold the map on screen longer — "
                   "the Status/Accum readout shows the time span and point count "
                   "that buys. A frame is one source batch: the SICK sends ~225 "
                   "small ones a second, a ros2 bag one full cloud per revolution.")
        cap = self._make_number(gui.NumberEdit.INT, 10_000, 5_000_000,
                                self._engine.accum_max_points,
                                self._on_accum_max_points)
        self._pair(agrid, "Point cap", cap,
                   "Memory / render guard on that cloud: frames also drop once "
                   "their points exceed this. Status shows 'cap' while it is the "
                   "limit that bites — raise it to let the frame count rule, but "
                   "past ~1M every rebuild costs visible frame time.")
        sec.add_child(agrid)

        crop_cb = gui.Checkbox("Crop view to region")
        crop_cb.checked = self._crop_view
        crop_cb.tooltip = ("Hide points outside the scoring region (walls, ceiling) "
                           "from the display. Affects the view only, never the data.")
        crop_cb.set_on_checked(self.set_crop_view)
        sec.add_child(crop_cb)

        cgrid = self._grid(em)
        self._pair(cgrid, "Camera", self.make_nav_combo(),
                   "Orbit turns the scene around the pivot; " + CAMERA_HINT
                   + ". Fly is WASD + drag, Esc returns to orbit.")
        sec.add_child(cgrid)

        reset_btn = gui.Button("Reset view")
        reset_btn.tooltip = ("Return the camera to the default overview angle "
                             "and put the pivot back at the middle of the grid.")
        reset_btn.set_on_clicked(self.reset_camera)
        sec.add_child(reset_btn)
        self._panel.add_child(sec)

    # -- levelling ------------------------------------------------------------ #
    def _build_level_section(self, em: float) -> None:
        sec = self._section(em, "Levelling")
        grid = self._grid(em)
        self._level_label = self._readout(
            grid, "Mount tilt",
            "Gravity levelling of the world frame: the measured tilt of the "
            "sensor mount, solved from the floor plane.")
        sec.add_child(grid)
        lvl_btn = gui.Button("Re-measure tilt   (L)")
        lvl_btn.tooltip = ("Re-solve the mount tilt from the floor in view. Do this "
                           "after the rig is repositioned or re-mounted.")
        lvl_btn.set_on_clicked(self.recalibrate_level)
        sec.add_child(lvl_btn)
        self._panel.add_child(sec)

    # -- model ---------------------------------------------------------------- #
    def _build_model_section(self, em: float) -> None:
        s = self._scorer.settings
        sec = self._section(em, f"Model · {self._scorer.model_name}")

        enabled = gui.Checkbox("Score live cloud")
        enabled.checked = s.enabled
        enabled.tooltip = ("Run the classifier on the freshest scan every update "
                           "interval. Turn it off to free the GPU.")
        enabled.set_on_checked(self._on_score_enabled)
        sec.add_child(enabled)

        persist = gui.Checkbox("Remember predictions (map)")
        persist.checked = s.persist
        persist.tooltip = ("Merge each pass into a persistent per-voxel prediction "
                           "map so coverage builds up as you sweep the room. Off "
                           "shows only the latest pass.")
        persist.set_on_checked(lambda v: self.set_score_setting("persist", bool(v)))
        sec.add_child(persist)

        grid = self._grid(em)
        disp = gui.Combobox()
        disp.add_item("Confidence")
        disp.add_item("Detections @ thr")
        disp.set_on_selection_changed(self._on_model_display)
        self._pair(grid, "Display", disp,
                   "Confidence paints the turbo ramp (blue = clear, red = rock). "
                   "Detections is the binary view at the decision threshold.")

        thr = self._make_slider(gui.Slider.DOUBLE, 0.0, 1.0,
                           self._scorer.threshold, self.set_threshold)
        self._pair(grid, "Threshold", thr,
                   "Probability above which a center counts as rock. The "
                   "checkpoint's own tuned value is the starting point.")

        ival = self._make_slider(gui.Slider.DOUBLE, 0.25, 5.0, s.interval_sec,
                            lambda v: self.set_score_setting("interval_sec", v))
        self._pair(grid, "Interval (s)", ival,
                   "Seconds between scoring passes. A pass costs tens of "
                   "milliseconds, so 0.5 leaves the GPU nearly idle.")

        win = self._make_number(gui.NumberEdit.DOUBLE, 0.0, 2.0, float(s.window_sec or 0.0),
                                lambda v: self.set_score_setting("window_sec", v))
        self._pair(grid, "Scan window (s)", win,
                   "0 scores a single scan, which is what the models were trained "
                   "on. Raising it densifies the input and drifts out of "
                   "distribution — predictions decay.")
        sec.add_child(grid)

        sec.add_child(self._heading("Scoring region  (relative to the sensor)"))
        rgrid = self._grid(em)
        zmin = self._make_number(gui.NumberEdit.DOUBLE, -5.0, 2.0, s.z_min,
                                 lambda v: self.set_score_setting("z_min", v))
        self._pair(rgrid, "z min (m)", zmin,
                   "Lower edge of the band kept around the sensor. Handheld rig "
                   "~1 m above the floor: -1.5.")
        zmax = self._make_number(gui.NumberEdit.DOUBLE, -3.0, 4.0, s.z_max,
                                 lambda v: self.set_score_setting("z_max", v))
        self._pair(rgrid, "z max (m)", zmax,
                   "Upper edge. Keeping it below the sensor (-0.5) throws away "
                   "walls and ceiling before they ever reach the model.")
        rng = self._make_number(gui.NumberEdit.DOUBLE, 0.0, 15.0, s.range_max,
                                lambda v: self.set_score_setting("range_max", v))
        self._pair(rgrid, "Max range (m)", rng,
                   "Horizontal radius to keep. 0 disables it. 8 m covers a room.")
        cap = self._make_number(gui.NumberEdit.INT, 500, 20000, int(s.max_centers),
                                lambda v: self.set_score_setting("max_centers", int(v)))
        self._pair(rgrid, "Max centers", cap,
                   "Hard cap on candidate centers per pass — the guard rail that "
                   "keeps a dense room from allocating gigabytes.")
        sec.add_child(rgrid)

        clear_btn = gui.Button("Clear predictions")
        clear_btn.tooltip = "Empty the prediction map and start accumulating again."
        clear_btn.set_on_clicked(self.clear_predictions)
        sec.add_child(clear_btn)

        sec.add_child(self._heading("Last pass"))
        sgrid = self._grid(em)
        self._model_map = self._readout(
            sgrid, "Map", "Centers held in the prediction map, and how many are "
                          "above the threshold.")
        self._model_pass = self._readout(
            sgrid, "Pass", "Centers scored in the most recent pass, and how long "
                           "it took.")
        self._model_region = self._readout(
            sgrid, "In region", "Points from the fresh scan that fell inside the "
                                "scoring region. Zero means the region is wrong.")
        sec.add_child(sgrid)

        #: Shown only when the scorer reports an empty region — a variable-height
        #: widget, so every toggle re-lays the panel out explicitly.
        self._model_warn = gui.Label("")
        self._model_warn.text_color = _WARN
        self._model_warn.visible = False
        sec.add_child(self._model_warn)

        #: Legacy alias — the old single blob label.
        self._model_label = self._model_map
        self._panel.add_child(sec)
        self._build_legend_section(em)

    def _build_legend_section(self, em: float) -> None:
        """A real colorbar beats five colored text rows."""
        sec = self._section(em, "Legend")
        sec.add_child(self._colorbar_widget(em))

        row = gui.Horiz()
        for text in ("0.0  clear", "0.5", "1.0  rock"):
            lbl = gui.Label(text)
            lbl.text_color = DIM
            row.add_child(lbl)
            if text != "1.0  rock":
                row.add_stretch()
        sec.add_child(row)

        note = gui.Label("dimmed = no prediction yet")
        note.text_color = DIM
        note.tooltip = ("Points outside the scoring region, or not yet scored, keep "
                        "dimmed height colors so the room stays readable.")
        sec.add_child(note)
        self._panel.add_child(sec)

    @staticmethod
    def _colorbar_widget(em: float) -> gui.ImageWidget:
        width, height = 256, max(10, int(0.9 * em))
        ramp = apply_colormap(np.linspace(0.0, 1.0, width), "turbo")
        img = np.repeat(ramp[None, :, :], height, axis=0)
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        return gui.ImageWidget(o3d.geometry.Image(np.ascontiguousarray(img)))

    # -- shortcuts ------------------------------------------------------------ #
    def _build_help_section(self, em: float) -> None:
        sec = self._section(em, "Keyboard", open_=False)
        grid = self._grid(em)
        for key, action in SHORTCUTS:
            k = gui.Label(key)
            k.text_color = _SECOND
            a = gui.Label(action)
            a.text_color = DIM
            grid.add_child(k)
            grid.add_child(a)
        sec.add_child(grid)
        self._panel.add_child(sec)

    # ------------------------------------------------------------------ #
    # Control API
    #
    # One implementation per knob, called by both surfaces: the docked panel's
    # widget callbacks and — through LiveController — the browser page. Each
    # owns the cache invalidation its change needs, which is the part that is
    # easy to get wrong twice.
    #
    # All of these must run on the GUI thread. Panel callbacks already do;
    # the controller gets there via post().
    # ------------------------------------------------------------------ #
    def post(self, fn) -> bool:
        """Queue ``fn`` to run on the GUI (main) thread. Safe from any thread.

        Returns whether it was queued: before :meth:`run` builds the window
        there is no event loop to queue onto, and the caller needs to know that
        rather than waiting for work that will never run.
        """
        if self._window is None:
            return False
        gui.Application.instance.post_to_main_thread(self._window, fn)
        return True

    def _post_update(self) -> None:
        self.post(self._update_scene)

    @property
    def color_mode(self) -> str:
        return self._color_mode

    @property
    def point_size(self) -> float:
        return float(self._point_mat.point_size)

    def _on_color_combo(self, _text: str, idx: int) -> None:
        self.set_color_mode(self._color_modes[int(idx)])

    def set_color_mode(self, mode: str) -> None:
        if mode not in self._color_modes:
            return
        self._color_mode = mode
        self._accum_model_key = None  # force accum recolor
        self._tick_count = -1
        # The surface mesh recolors too (per-cell fused reflectivity); unknown
        # modes ("model") fall back to height coloring mesh-side.
        set_mode = getattr(self._engine.surface, "set_color_mode", None)
        if callable(set_mode):
            set_mode(mode)
        if self._color_combo is not None:
            idx = self._color_modes.index(mode)
            if self._color_combo.selected_index != idx:
                self._color_combo.selected_index = idx
        self._post_update()

    def set_layer(self, attr: str, value: bool) -> None:
        """Show/hide one scene layer, keeping the panel checkbox in step."""
        setattr(self, attr, bool(value))
        cb = self._layer_checks.get(attr)
        if cb is not None:
            cb.checked = bool(value)
        self._post_update()

    def set_point_size(self, value: float) -> None:
        self._point_mat.point_size = float(value)
        self._accum_mat.point_size = max(1.0, float(value) - 1.0)
        self._accum_model_key = None  # rebuild so the new size applies
        self._post_update()

    def _on_accum_frames(self, value: float) -> None:
        """Retention of the accumulated cloud, in frames.

        Shrinking trims immediately, so force a rebuild rather than waiting for
        the throttled tick — otherwise the dropped frames linger on screen.
        """
        self._engine.set_accum_frames(int(value))
        self.refresh_accum()

    def _on_accum_max_points(self, value: float) -> None:
        """Point ceiling on the accumulated cloud (see _on_accum_frames)."""
        self._engine.set_accum_max_points(int(value))
        self.refresh_accum()

    def refresh_accum(self) -> None:
        """Force the accumulated cloud to be rebuilt on the next tick."""
        self._accum_model_key = None
        self._tick_count = -1
        self._post_update()

    def _on_score_enabled(self, value: bool) -> None:
        self._scorer.settings.enabled = bool(value)

    def _on_model_display(self, _text: str, idx: int) -> None:
        self.set_model_display(int(idx))

    def set_model_display(self, index: int) -> None:
        self._model_display = int(index)
        self._accum_model_key = None
        self._post_update()

    def set_threshold(self, value: float) -> None:
        self._scorer.threshold = float(value)
        if self._model_display == 1:
            self._accum_model_key = None
        self._post_update()

    def set_score_setting(self, name: str, value) -> None:
        setattr(self._scorer.settings, name, value)
        if name in ("z_min", "z_max", "range_max"):
            self._accum_model_key = None  # region moved: recolor/crop the cloud
            self._post_update()

    def recalibrate_level(self) -> None:
        """Re-measure the mount tilt and re-fuse from scratch."""
        self._engine.recalibrate_level()
        self._engine.reset_surface(reset_slam=False)
        self._accum_model_key = None
        self._tick_count = -1
        print("[rocklabel] levelling: re-measuring mount tilt…", flush=True)
        self._post_update()

    def set_crop_view(self, value: bool) -> None:
        self._crop_view = bool(value)
        self._accum_model_key = None
        self._tick_count = -1  # force an accum rebuild next tick
        self._post_update()

    def clear_predictions(self) -> None:
        self._scorer.clear_map()
        self._accum_model_key = None
        self._post_update()

    def _view_region_mask(self, pts: np.ndarray) -> np.ndarray | None:
        """Display-only mask hiding points outside the scoring region; None
        when the crop-view toggle is off (or no region is defined)."""
        if not self._crop_view:
            return None
        if self._scorer is not None:
            s = self._scorer.settings
            z_min, z_max, range_max = s.z_min, s.z_max, s.range_max
            floor_relative = s.floor_relative
        elif self._cfg.crop.enabled:
            c = self._cfg.crop
            z_min, z_max, range_max = c.z_min, c.z_max, c.range_max
            floor_relative = c.floor_relative
        else:
            return None
        base = self._engine.current_pose()[0]
        anchor = base[2]
        if floor_relative:
            floor = self._engine.floor_z()
            if floor is not None:
                anchor = floor
        keep = (pts[:, 2] >= anchor + z_min) & (pts[:, 2] <= anchor + z_max)
        if range_max > 0.0:
            dx = pts[:, 0] - base[0]
            dy = pts[:, 1] - base[1]
            keep &= dx * dx + dy * dy <= range_max * range_max
        return keep

    def toggle_recording(self) -> None:
        if self._replay:
            return
        if self._engine.recorder is None:
            path = self._engine.start_recording()
            if path:
                print(f"[rocklabel] recording -> {path}", flush=True)
        else:
            path = self._engine.stop_recording()
            print(f"[rocklabel] recording saved: {path}", flush=True)
        self._post_update()

    # ------------------------------------------------------------------ #
    # Lidar pose box + replay transport bar
    # ------------------------------------------------------------------ #
    def _add_lidar_box(self) -> None:
        """A small oriented box marking the sensor's world pose (toggle: B)."""
        assert self._scene is not None
        bx, by, bz = (float(v) for v in self._cfg.display.lidar_box_size)
        box = o3d.geometry.TriangleMesh.create_box(bx, by, bz)
        box.translate((-bx / 2.0, -by / 2.0, -bz / 2.0))  # center on the origin
        box.paint_uniform_color([1.0, 0.55, 0.05])  # SICK orange
        box.compute_vertex_normals()
        self._scene.add_geometry(_BOX_NAME, box, self._box_mat)
        self._scene.show_geometry(_BOX_NAME, self._show_box)

    def _build_transport_bar(self) -> None:
        """Play/pause + time + seek slider, laid out along the bottom edge."""
        assert self._window is not None
        src = self._engine.source
        self._play_btn = gui.Button("Pause" if src.playing else "Play")
        self._play_btn.set_on_clicked(self._on_play_clicked)
        self._time_label = gui.Label(f"{0.0:6.1f} / {src.duration_sec:6.1f} s")
        self._slider = gui.Slider(gui.Slider.DOUBLE)
        self._slider_max = max(0.001, src.duration_sec)
        self._slider.set_limits(0.0, self._slider_max)
        self._slider.set_on_value_changed(self._on_slider_changed)
        for w in (self._play_btn, self._time_label, self._slider):
            self._window.add_child(w)
        self._window.set_needs_layout()

    def _on_play_clicked(self) -> None:
        self._engine.source.toggle_play()
        self._refresh_transport()

    def _on_slider_changed(self, value: float) -> None:
        """Debounced: the seek fires once the scrub settles (see tick loop)."""
        if any(abs(value - v) < _SLIDER_ECHO_EPS_SEC for v in self._synced_values):
            return  # async echo of our own position mirroring, not a user scrub
        # Second line of defence: whatever its provenance, a callback that
        # already sits where playback is cannot be asking us to move. Judge it
        # against the position *now*, not at fire time — by then playback has
        # drifted past the debounce window and an echo looks like a real jump
        # backwards, which would rewind and re-fuse the whole map.
        src = self._engine.source
        if abs(value - src.position_sec) < _SEEK_MIN_DELTA_SEC and not src.finished:
            return  # ...unless we're parked at the end, where any scrub replays
        self._pending_seek = float(value)
        self._pending_seek_from = src.position_sec
        self._pending_seek_time = time.perf_counter()
        self._last_user_seek = self._pending_seek_time

    # ------------------------------------------------------------------ #
    # Layout: scene left, panel right, transport bar along the scene bottom
    # ------------------------------------------------------------------ #
    def _on_layout(self, ctx: gui.LayoutContext) -> None:
        assert self._window is not None and self._widget is not None
        r = self._window.content_rect
        if self._panel is None:
            scene_w = r.width  # web UI: the scene gets the whole window
        else:
            # 24em: the two-column readout/control grid needs a real value
            # column, otherwise every slider track collapses to a stub.
            panel_w = min(int(24 * ctx.theme.font_size), r.width // 3)
            scene_w = r.width - panel_w
            self._panel.frame = gui.Rect(r.x + scene_w, r.y, panel_w, r.height)
        self._widget.frame = gui.Rect(r.x, r.y, scene_w, r.height)
        if self._slider is not None:
            pad = 8
            bpref = self._play_btn.calc_preferred_size(ctx, gui.Widget.Constraints())
            tpref = self._time_label.calc_preferred_size(ctx, gui.Widget.Constraints())
            h = max(bpref.height, tpref.height)
            y = r.y + r.height - h - pad
            x = r.x + pad
            self._play_btn.frame = gui.Rect(x, y, bpref.width, h)
            x += bpref.width + pad
            self._time_label.frame = gui.Rect(x, y + (h - tpref.height) // 2, tpref.width, tpref.height)
            x += tpref.width + pad
            self._slider.frame = gui.Rect(x, y, max(50, r.x + scene_w - pad - x), h)

    # ------------------------------------------------------------------ #
    # Camera
    # ------------------------------------------------------------------ #
    def reset_camera(self) -> None:
        g = self._cfg.grid
        x0, y0 = g.origin
        sx, sy = g.extent
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            (x0, y0, -2.0), (x0 + sx, y0 + sy, 4.0)
        )
        assert self._widget is not None
        self._widget.setup_camera(60.0, bbox, bbox.get_center())
        # Look down at an angle for a good view of the 2.5D surface.
        center = bbox.get_center()
        eye = center + np.array([0.0, -0.6 * sy, 0.9 * max(sx, sy)])
        self._widget.look_at(center, eye, [0.0, 0.0, 1.0])

    # ------------------------------------------------------------------ #
    # Display-rate tick -> posts a scene update to the main thread
    # ------------------------------------------------------------------ #
    def _tick_loop(self) -> None:
        period = 1.0 / self._fps
        app = gui.Application.instance
        while not self._tick_stop.is_set():
            time.sleep(period)
            if self._tick_stop.is_set():
                break
            if self._window is not None:
                app.post_to_main_thread(self._window, self._update_scene)

    # -- coloring ------------------------------------------------------------ #
    def _model_colors(self, pts: np.ndarray) -> np.ndarray:
        """Model mode: nearest scored center's probability per point; points
        without a prediction keep dimmed height colors for context."""
        rgb = apply_colormap(normalize(pts[:, 2]), self._cfg.display.colormap)
        rgb *= _UNSCORED_DIM
        res = self._scorer.probs_for(pts)
        if res is None:
            return rgb
        probs, matched = res
        if not matched.any():
            return rgb
        if self._model_display == 0:
            scored = apply_colormap(probs[matched], "turbo")
        else:
            thr = self._scorer.threshold
            scored = np.where((probs[matched] >= thr)[:, None], _DET_ROCK, _DET_CLEAR)
        rgb[matched] = scored
        return rgb

    def _point_colors(self, pts: np.ndarray, inten: np.ndarray) -> np.ndarray | None:
        """Per-point RGB for the active color mode; None = uniform color."""
        if self._color_mode == "model" and self._scorer is not None:
            return self._model_colors(pts)
        if self._color_mode in ("reflectivity", "reflectivity_stretch"):
            finite = np.isfinite(inten)
            if not np.any(finite):
                return None  # no RSSI available: fall back to uniform
            v = reflectivity_values(
                inten, finite, stretch=self._color_mode == "reflectivity_stretch",
                pct=self._cfg.display.reflectivity_percentiles)
            return apply_colormap(v, self._cfg.display.reflectivity_colormap)
        return apply_colormap(normalize(pts[:, 2]), self._cfg.display.colormap)

    def _set_point_geometry(
        self,
        name: str,
        pts: np.ndarray,
        inten: np.ndarray,
        mat: rendering.MaterialRecord,
        uniform: tuple[float, float, float] | None,
    ) -> None:
        """Replace a point-cloud geometry, colored per the active mode."""
        assert self._scene is not None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        colors = self._point_colors(pts, inten)
        if colors is not None and (uniform is None or self._color_mode != "height"):
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif uniform is not None:
            pcd.paint_uniform_color(list(uniform))
        else:  # reflectivity requested but no RSSI in the stream
            pcd.paint_uniform_color([0.6, 0.6, 0.6])
        if self._scene.has_geometry(name):
            self._scene.remove_geometry(name)
        self._scene.add_geometry(name, pcd, mat)

    def _update_scene(self) -> None:
        """Runs on the GUI (main) thread: refresh geometry + stats."""
        if self._scene is None:
            return
        self._tick_count += 1

        # --- raw points (uniform gray in height mode, colored otherwise) ---
        if self._show_points:
            pts, inten = self._engine.raw_snapshot()
            mask = self._view_region_mask(pts) if pts.shape[0] else None
            if mask is not None:
                pts, inten = pts[mask], inten[mask]
            if pts.shape[0] > 0:
                self._set_point_geometry(
                    _POINTS_NAME, pts, inten, self._point_mat, (0.75, 0.78, 0.85)
                )
            elif self._scene.has_geometry(_POINTS_NAME):
                self._scene.remove_geometry(_POINTS_NAME)
        elif self._scene.has_geometry(_POINTS_NAME):
            self._scene.remove_geometry(_POINTS_NAME)

        # --- accumulated world-frame cloud (throttled: it can hold 500k pts).
        #     In model mode the rebuild is keyed on the scoring pass + display
        #     settings, so the expensive recolor runs once per model update
        #     instead of every few ticks. ---
        if self._show_accum:
            if self._color_mode == "model" and self._scorer is not None:
                key = (self._scorer.version, self._model_display,
                       round(self._scorer.threshold, 3))
                needs = key != self._accum_model_key or not self._scene.has_geometry(_ACCUM_NAME)
                if needs:
                    self._accum_model_key = key
            else:
                needs = not self._scene.has_geometry(_ACCUM_NAME) or self._tick_count % 4 == 0
            if needs:
                apts, ainten = self._engine.accum_snapshot()
                mask = self._view_region_mask(apts) if apts.shape[0] else None
                if mask is not None:
                    apts, ainten = apts[mask], ainten[mask]
                if apts.shape[0] > 0:
                    self._set_point_geometry(
                        _ACCUM_NAME, apts, ainten, self._accum_mat, None
                    )
                elif self._scene.has_geometry(_ACCUM_NAME):
                    self._scene.remove_geometry(_ACCUM_NAME)
        elif self._scene.has_geometry(_ACCUM_NAME):
            self._scene.remove_geometry(_ACCUM_NAME)

        # --- surface mesh (rebuild the Open3D object only when the builder
        #     actually produced a new MeshData — identity check on the cache) ---
        if self._show_mesh:
            mesh_data = self._engine.surface.get_mesh_arrays()
            if not mesh_data.is_empty() and (
                mesh_data is not self._last_mesh_data
                or not self._scene.has_geometry(_MESH_NAME)
            ):
                mesh = mesh_data.to_open3d()
                if self._scene.has_geometry(_MESH_NAME):
                    self._scene.remove_geometry(_MESH_NAME)
                self._scene.add_geometry(_MESH_NAME, mesh, self._mesh_mat)
                self._last_mesh_data = mesh_data
        elif self._scene.has_geometry(_MESH_NAME):
            self._scene.remove_geometry(_MESH_NAME)

        # --- lidar pose box: cheap rigid transform, no geometry rebuild ---
        self._scene.show_geometry(_BOX_NAME, self._show_box)
        if self._show_box:
            pos, quat = self._engine.current_pose()
            t = np.eye(4)
            t[:3, :3] = quat_to_matrix(quat)
            t[:3, 3] = pos
            self._scene.set_geometry_transform(_BOX_NAME, t)

        # --- replay transport: fire settled seeks, mirror position ---
        if self._replay and self._slider is not None:
            src = self._engine.source
            now = time.perf_counter()
            if (
                self._pending_seek is not None
                and now - self._pending_seek_time >= _SEEK_DEBOUNCE_SEC
            ):
                # Suppress micro-seeks at the current position (slider echo /
                # jitter): a tiny backward step would rebuild the whole map.
                # Measured from where playback was when the scrub landed —
                # against the position *now*, playback having advanced through
                # the debounce would make a stationary scrub look like a jump.
                if (
                    abs(self._pending_seek - self._pending_seek_from)
                    >= _SEEK_MIN_DELTA_SEC
                    or src.finished
                ):
                    src.seek(self._pending_seek)
                self._pending_seek = None
            self._refresh_transport()

        self._refresh_stats()

    def _refresh_transport(self) -> None:
        src = self._engine.source
        dur = max(0.001, src.duration_sec)
        if self._slider is not None and abs(dur - self._slider_max) > 1e-9:
            self._slider_max = dur
            self._slider.set_limits(0.0, dur)
        if self._play_btn is not None:
            self._play_btn.text = "Pause" if src.playing else "Play"
        if self._time_label is not None:
            self._time_label.text = f"{src.position_sec:6.1f} / {src.duration_sec:6.1f} s"
        # Don't fight the user's drag: only mirror playback position onto the
        # slider once seeks have settled.
        if (
            self._slider is not None
            and self._pending_seek is None
            and not src.seeking
            and time.perf_counter() - self._last_user_seek > 0.75
        ):
            self._slider.double_value = src.position_sec
            # Read back post-clamp so the echo filter sees the exact value.
            self._synced_values.append(float(self._slider.double_value))

    # ------------------------------------------------------------------ #
    # Stats readout (panel + occasional console line)
    # ------------------------------------------------------------------ #
    def _refresh_stats(self) -> None:
        """Refresh the readouts, then echo a line to the console.

        Split from the panel writes because under ``--web-ui`` there is no
        panel: every cell below is None, and writing ``text_color`` on one
        would fail. The console echo runs either way — it is the only progress
        report a terminal-only run gets.
        """
        if self._panel is not None:
            self._refresh_panel_stats()

        # Throttled console echo (~1 Hz) so terminal logs still show progress.
        s = self._engine.stats
        now = time.perf_counter()
        if now - self._last_console_stat > 1.0:
            self._last_console_stat = now
            print(
                f"[rocklabel] {s.points_per_sec()/1e3:6.1f}k pts/s | "
                f"cells occupied: {s.cells_occupied:6d} | "
                f"pose: {self._engine.pose_status()} | "
                f"{'PAUSED' if self._engine.paused else 'running'}",
                flush=True,
            )

    def _refresh_panel_stats(self) -> None:
        """Write into the docked panel's readout cells.

        Every cell is one line by construction: a Label that grows a line
        mid-run pushes everything below it out of its measured frame, which is
        what used to make the model status overlap the Legend header.
        """
        s = self._engine.stats
        if self._replay:
            src = self._engine.source
            state = "seeking…" if src.seeking else ("playing" if src.playing else "paused")
            tail = f"{src.position_sec:.1f}/{src.duration_sec:.1f} s · {state}"
        else:
            tail = "paused" if self._engine.paused else "running"

        rate = s.points_per_sec() / 1e3
        self._put(self._stat_rate, f"{rate:.1f}k pts/s")
        self._stat_rate.text_color = _SECOND if rate > 0.5 else _WARN
        self._put(self._stat_cells, f"{s.cells_occupied:,}")
        frames, pts, span = self._engine.accum_stats()
        capped = pts >= self._engine.accum_max_points
        secs = f"{span:.1f}s" if span < 100 else f"{span:.0f}s"
        self._put(self._stat_accum,
                  f"{_compact(frames)}f · {_compact(pts)}p · {secs}"
                  + (" cap" if capped else ""))
        self._stat_accum.text_color = _WARN if capped else _SECOND
        self._put(self._stat_pose, self._engine.pose_status())
        self._put(self._stat_state, tail)
        self._stat_state.text_color = (
            _WARN if (self._engine.paused and not self._replay) else _SECOND)

        if self._level_label is not None:
            self._put(self._level_label,
                      self._engine.leveler.status().replace("level: ", ""))

        if not self._replay and self._rec_state is not None:
            rec = self._engine.recorder
            if rec is None:
                self._rec_btn.text = "Start recording   (S)"
                self._put(self._rec_state, "off")
                self._rec_state.text_color = DIM
                self._put(self._rec_stats, "-")
                self._put(self._rec_file, "-")
            else:
                self._rec_btn.text = "Stop recording   (S)"
                self._put(self._rec_state, "REC")
                self._rec_state.text_color = _CRIT
                self._put(self._rec_stats, f"{rec.elapsed_sec:.0f}s · "
                                           f"{rec.points_total / 1e6:.1f}M pts")
                self._put(self._rec_file, os.path.basename(rec.path))

        if self._scorer is not None:
            self._refresh_model_stats()

    def _refresh_model_stats(self) -> None:
        """Fan the scorer's numbers out into the readout cells.

        Reads :meth:`LiveScorer.status_dict` rather than re-parsing the
        terminal string ``status()`` produces — that parse recovered these
        values by splitting on ": " in a fixed line order, so any rewording of
        the console output silently emptied the panel.
        """
        m = self._scorer.status_dict()
        warn = m["warning"]
        if not m["enabled"]:
            rows = ("scoring off", "-", "-")
        elif not m["ready"]:
            rows = ("warming up…", "no pass finished yet", "-")
        else:
            cap = " (capped)" if m["capped"] else ""
            rows = (
                f"{m['map_centers']} centers, {m['detections']} >= thr",
                f"{m['pass_centers']} scored{cap} in {m['pass_ms']:.0f} ms",
                f"{m['in_region'] / 1e3:.1f}k",
            )
        for cell, value in zip(
            (self._model_map, self._model_pass, self._model_region), rows
        ):
            self._put(cell, value)

        # The warning is the one widget here allowed to wrap, so both its
        # visibility AND its length change the panel's height. Open3D does not
        # re-measure on its own: without this the Legend below keeps its stale
        # frame and the two overlap, which is the bug this panel started with.
        if warn != self._model_warn.text or bool(warn) != bool(self._model_warn.visible):
            self._model_warn.text = warn
            self._model_warn.visible = bool(warn)
            if self._window is not None:
                self._window.set_needs_layout()

    # ------------------------------------------------------------------ #
    # Keyboard handling
    # ------------------------------------------------------------------ #
    def _on_key(self, event: gui.KeyEvent) -> int:
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED

        # While flying, WASD (and S, which is also "stop recording" here)
        # belong to the fly controller; only Esc is ours, and it leaves fly.
        flying = self.camera_key_result(event)
        if flying is not None:
            return flying

        # set_layer keeps the panel checkbox in step when there is one, and is
        # a no-op on that front under --web-ui where there isn't.
        key = event.key
        if key == gui.KeyName.P:
            self.set_layer("_show_points", not self._show_points)
        elif key == gui.KeyName.M:
            self.set_layer("_show_mesh", not self._show_mesh)
        elif key == gui.KeyName.C:
            self.set_layer("_show_accum", not self._show_accum)
        elif key == gui.KeyName.B:
            self.set_layer("_show_box", not self._show_box)
        elif key == gui.KeyName.V:
            i = self._color_modes.index(self._color_mode)
            self.set_color_mode(self._color_modes[(i + 1) % len(self._color_modes)])
            return gui.Widget.EventCallbackResult.HANDLED
        elif key == gui.KeyName.S and not self._replay:
            self.toggle_recording()
            return gui.Widget.EventCallbackResult.HANDLED
        elif key == gui.KeyName.L and self._engine.leveler.active:
            self.recalibrate_level()
            return gui.Widget.EventCallbackResult.HANDLED
        elif key == gui.KeyName.R:
            if self._replay:
                self._engine.source.seek(0.0)  # rewind + re-fuse from the top
            else:
                self._engine.reset_surface()
            if self._scorer is not None:
                self._scorer.clear_map()
                self._accum_model_key = None
        elif key == gui.KeyName.SPACE:
            if self._replay:
                self._engine.source.toggle_play()
            else:
                self._engine.toggle_pause()
        elif key == gui.KeyName.LEFT and self._replay:
            src = self._engine.source
            src.seek(max(0.0, src.position_sec - 5.0))
        elif key == gui.KeyName.RIGHT and self._replay:
            src = self._engine.source
            src.seek(src.position_sec + 5.0)
        elif key in (gui.KeyName.Q, gui.KeyName.ESCAPE):
            if self._window is not None:
                gui.Application.instance.post_to_main_thread(
                    self._window, self._window.close
                )
        else:
            return gui.Widget.EventCallbackResult.IGNORED

        # Reflect the change immediately.
        self._post_update()
        return gui.Widget.EventCallbackResult.HANDLED
