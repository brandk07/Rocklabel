"""The control surface behind the live web panel.

:class:`LiveController` is the one place that knows how a control id maps onto
the running objects — the :class:`~rocklabel.live.pipeline.IngestEngine`, the
:class:`~rocklabel.live.scoring.LiveScorer`, and the Open3D
:class:`~rocklabel.live.viz.app.VizApp`. The Flask layer above it only moves
JSON; everything about *what a knob does* lives here.

Threading
---------
Flask serves on its own thread, so every write lands off the GUI thread:

* engine and scorer state is safe to touch directly — ``set_accum_frames``,
  ``source.seek`` and friends are the same "called from the UI thread" controls
  the Open3D panel already used, and each is lock-guarded internally;
* anything that touches the Open3D **scene** must not be. Those go through
  :meth:`post`, which hands the callable to the GUI thread via
  ``Application.post_to_main_thread`` — the identical path a panel click takes.

With ``viz=None`` (``--headless --web-ui``) :meth:`post` simply calls through,
and the display-only controls drop out of the schema, so the page shows only
what such a run actually has.
"""

from __future__ import annotations

import os
import threading

import numpy as np

from ..colormap import move_range_end
from . import scene, spec

#: How long a write waits for the GUI thread to apply it before answering
#: anyway. The GUI drains its queue every frame (60 ms at the default 16 fps),
#: so this is ~10 frames of headroom — long enough that it never trips in
#: practice, short enough that a wedged renderer cannot hang the web server.
_POST_TIMEOUT_SEC = 0.6


def _compact(n: float) -> str:
    """Counts that fit a readout: 812, 4.9k, 498k, 1.2M."""
    n = float(n)
    if n < 1_000:
        return str(int(n))
    if n < 10_000:
        return f"{n / 1e3:.1f}k"
    if n < 1_000_000:
        return f"{n / 1e3:.0f}k"
    return f"{n / 1e6:.1f}M"


class LiveController:
    """Reads and writes one running live session on behalf of the web page."""

    def __init__(self, config, engine, scorer=None, viz=None) -> None:
        self._cfg = config
        self._engine = engine
        self._scorer = scorer
        self._viz = viz
        self._replay = bool(getattr(engine.source, "is_replay", False))
        self._history = scene.History()

    # ------------------------------------------------------------------ #
    # Capabilities
    # ------------------------------------------------------------------ #
    @property
    def capabilities(self) -> set[str]:
        caps = {"replay" if self._replay else "live"}
        if self._scorer is not None:
            caps.add("scorer")
        if self._viz is not None:
            caps.add("viewer")
        if getattr(self._engine.leveler, "active", False):
            caps.add("leveler")
        return caps

    def post(self, fn) -> None:
        """Run ``fn`` on the Open3D GUI thread and wait for it to land.

        Waiting matters because the caller is an HTTP request that answers with
        a fresh snapshot: return before the GUI thread has run ``fn`` and the
        response reports the *old* value, so the page briefly contradicts the
        change the user just made. The GUI thread drains its queue every frame,
        so this costs one frame — and :data:`_POST_TIMEOUT_SEC` bounds it, so a
        stalled renderer degrades to a stale echo instead of a hung request.
        """
        if self._viz is None:
            fn()
            return
        done = threading.Event()

        def _run():
            try:
                fn()
            finally:
                done.set()

        if self._viz.post(_run):
            done.wait(_POST_TIMEOUT_SEC)

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def schema(self) -> dict:
        src = self._engine.source
        if self._replay:
            title = os.path.basename(getattr(src, "path", "") or "recording")
        else:
            title = f"source · {self._cfg.source.kind}"
        return spec.to_json(self.capabilities) | {
            "mode": "replay" if self._replay else "live",
            "subtitle": title,
            "model": self._scorer.model_name if self._scorer else "",
            "duration_sec": (float(src.duration_sec) if self._replay else 0.0),
        }

    # ------------------------------------------------------------------ #
    # Values — current setting of every writable control
    # ------------------------------------------------------------------ #
    def values(self) -> dict:
        v: dict = {}
        e, z, s = self._engine, self._viz, self._scorer
        v["view.accum_frames"] = int(e.accum_frames)
        v["view.accum_max_points"] = int(e.accum_max_points)
        crop = self._cfg.crop
        v["crop.enabled"] = bool(crop.enabled)
        v["crop.z_min"] = float(crop.z_min)
        v["crop.z_max"] = float(crop.z_max)
        v["crop.floor_relative"] = bool(crop.floor_relative)
        v["crop.range_max"] = float(crop.range_max)
        v["crop.range_min"] = float(crop.range_min)
        if z is not None:
            v["view.color_mode"] = z.color_mode
            lo, hi = z.reflectivity_range
            v["view.refl_min"] = float(lo)
            v["view.refl_max"] = float(hi)
            v["view.nav_mode"] = getattr(z, "nav_mode", "orbit")
            v["view.point_size"] = float(z.point_size)
            v["view.show_points"] = bool(z._show_points)
            v["view.show_mesh"] = bool(z._show_mesh)
            v["view.show_accum"] = bool(z._show_accum)
            v["view.show_box"] = bool(z._show_box)
            v["view.crop_view"] = bool(z._crop_view)
            if s is not None:
                v["model.display"] = int(z._model_display)
        if s is not None:
            st = s.settings
            v["model.enabled"] = bool(st.enabled)
            v["model.persist"] = bool(st.persist)
            v["model.threshold"] = float(s.threshold)
            v["model.interval_sec"] = float(st.interval_sec)
            v["model.window_sec"] = float(st.window_sec or 0.0)
            v["region.z_min"] = float(st.z_min)
            v["region.z_max"] = float(st.z_max)
            v["region.range_max"] = float(st.range_max)
            v["region.max_centers"] = int(st.max_centers)
        if self._replay:
            v["replay.position"] = float(self._engine.source.position_sec)
        return v

    # ------------------------------------------------------------------ #
    # Status — the readouts, already formatted for display
    # ------------------------------------------------------------------ #
    def status(self) -> tuple[dict, dict]:
        """``(text_by_control_id, flags)``. Flags drive the warning coloring."""
        e = self._engine
        st = e.stats
        text: dict[str, str] = {}
        flags: dict[str, bool] = {}

        rate = st.points_per_sec() / 1e3
        text["status.rate"] = f"{rate:.1f}k pts/s"
        flags["rate_low"] = rate <= 0.5
        text["status.cells"] = f"{st.cells_occupied:,}"

        frames, points, span = e.accum_stats()
        capped = points >= e.accum_max_points
        secs = f"{span:.1f}s" if span < 100 else f"{span:.0f}s"
        text["status.accum"] = (f"{_compact(frames)} frames · "
                                f"{_compact(points)} pts · {secs}"
                                + (" · cap" if capped else ""))
        flags["accum_capped"] = bool(capped)
        text["status.pose"] = e.pose_status()

        if self._replay:
            src = e.source
            state = "seeking…" if src.seeking else (
                "playing" if src.playing else "paused")
            text["status.state"] = (f"{src.position_sec:.1f} / "
                                    f"{src.duration_sec:.1f} s · {state}")
            flags["paused"] = not src.playing
        else:
            text["status.state"] = "paused" if e.paused else "running"
            flags["paused"] = bool(e.paused)

        if "leveler" in self.capabilities:
            text["level.tilt"] = e.level_status().replace("level: ", "")

        if not self._replay:
            rec = e.recorder
            if rec is None:
                text["record.state"] = "off"
                text["record.captured"] = "—"
                text["record.file"] = "—"
                flags["recording"] = False
            else:
                text["record.state"] = "REC"
                text["record.captured"] = (f"{rec.elapsed_sec:.0f}s · "
                                           f"{rec.points_total / 1e6:.1f}M pts")
                text["record.file"] = os.path.basename(rec.path)
                flags["recording"] = True

        if self._scorer is not None:
            m = self._scorer.status_dict()
            if not m["enabled"]:
                text["model.map"] = text["model.pass"] = "scoring off"
                text["model.region"] = "—"
            elif not m["ready"]:
                text["model.map"] = "warming up…"
                text["model.pass"] = "no pass finished yet"
                text["model.region"] = "—"
            else:
                text["model.map"] = (f"{m['map_centers']:,} centers · "
                                     f"{m['detections']:,} ≥ thr")
                cap = " (capped)" if m["capped"] else ""
                text["model.pass"] = (f"{m['pass_centers']:,} scored{cap} in "
                                      f"{m['pass_ms']:.0f} ms")
                text["model.region"] = f"{m['in_region'] / 1e3:.1f}k pts"
            flags["region_empty"] = bool(m["warning"])
            if m["warning"]:
                text["model.warning"] = m["warning"]
        return text, flags

    def snapshot(self) -> dict:
        text, flags = self.status()
        self._sample_history()
        out = {"values": self.values(), "status": text, "flags": flags}
        if self._replay:
            src = self._engine.source
            out["transport"] = {
                "position_sec": float(src.position_sec),
                "duration_sec": float(src.duration_sec),
                "playing": bool(src.playing),
                "seeking": bool(src.seeking),
                "finished": bool(src.finished),
            }
        return out

    # ------------------------------------------------------------------ #
    # The extra views: overhead map, confidence histogram, trends
    # ------------------------------------------------------------------ #
    def _sample_history(self) -> None:
        """Record one trend sample. Rate-limited inside :class:`scene.History`,
        so calling it from every status poll costs nothing extra."""
        m = self._scorer.status_dict() if self._scorer is not None else {}
        self._history.sample({
            "detections": m.get("detections", 0),
            "in_region": m.get("in_region", 0),
            "pass_ms": m.get("pass_ms", 0.0),
            "rate": self._engine.stats.points_per_sec(),
        })

    def scene(self) -> dict:
        """The overhead map, the detections on it, and the charts' data.

        Separate from :meth:`snapshot` because it is a different order of
        magnitude — tens of kilobytes against a few hundred bytes — and the page
        polls it more slowly to match.
        """
        raster = None
        get_raster = getattr(self._engine.surface, "height_raster", None)
        if callable(get_raster):
            raster = get_raster(scene.BEV_MAX_SIDE)

        pos, quat = self._engine.current_pose()
        yaw = float(np.degrees(np.arctan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )))
        out = {
            "bev": scene.encode_raster(raster),
            "detections": scene.detections_payload(self._scorer),
            "sensor": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                       "yaw_deg": yaw},
            "history": self._history.payload(),
            "histogram": scene.confidence_histogram(self._scorer),
        }
        # The ring on the overhead map is whichever region is actually in
        # force: the scoring band with a model, the ingest crop without one.
        band = self._scorer.settings if self._scorer is not None else (
            self._cfg.crop if self._cfg.crop.enabled else None)
        if band is not None:
            out["region"] = {"range_max": float(band.range_max),
                             "z_min": float(band.z_min), "z_max": float(band.z_max),
                             "floor_relative": bool(band.floor_relative)}
        return out

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    #: Layer checkbox ids -> the VizApp flag they own.
    _LAYERS = {
        "view.show_points": "_show_points",
        "view.show_mesh": "_show_mesh",
        "view.show_accum": "_show_accum",
        "view.show_box": "_show_box",
    }
    #: Ingest-crop ids -> the CropConfig field they own. The engine holds the
    #: same CropConfig object, and reads it per batch, so a write here lands on
    #: the next scan without restarting anything.
    _CROP = {
        "crop.enabled": "enabled",
        "crop.z_min": "z_min",
        "crop.z_max": "z_max",
        "crop.floor_relative": "floor_relative",
        "crop.range_max": "range_max",
        "crop.range_min": "range_min",
    }
    #: Scoring-region ids -> the ScoreSettings field they own.
    _REGION = {
        "region.z_min": "z_min",
        "region.z_max": "z_max",
        "region.range_max": "range_max",
        "region.max_centers": "max_centers",
    }

    def set(self, key: str, value) -> None:
        """Apply one control. Raises ``KeyError``/``ValueError`` on bad input."""
        control = spec.CONTROLS_BY_ID.get(key)
        if control is None:
            raise KeyError(key)
        if control.requires and control.requires not in self.capabilities:
            raise ValueError(f"{key} is not available in this run")
        if control.kind in ("action", "readout"):
            raise ValueError(f"{key} is not a setting")
        value = self._coerce(control, value)

        if key in self._LAYERS:
            self.post(lambda: self._viz.set_layer(self._LAYERS[key], value))
        elif key in self._CROP:
            # Same story as the region below: the crop band is what the display
            # crop follows when there is no model, so the viewer has to hear
            # about it rather than being left with a stale accumulated cloud.
            self._crop_setting(self._CROP[key], value)
        elif key in self._REGION:
            # Region bounds also drive the display crop, so the viewer has to
            # invalidate its recolor cache — _on_setting does both.
            self._scorer_setting(self._REGION[key], value)
        elif key == "view.color_mode":
            self.post(lambda: self._viz.set_color_mode(value))
        elif key in ("view.refl_min", "view.refl_max"):
            # One window, two sliders: resolve them into a pair here, with the
            # end the page just moved winning and the other yielding.
            end = "lo" if key == "view.refl_min" else "hi"
            lo, hi = move_range_end(self._viz.reflectivity_range, end, value)
            self.post(lambda: self._viz.set_reflectivity_range(lo, hi))
        elif key == "view.nav_mode":
            self.post(lambda: self._viz.set_nav_mode(value))
        elif key == "view.point_size":
            self.post(lambda: self._viz.set_point_size(value))
        elif key == "view.crop_view":
            self.post(lambda: self._viz.set_crop_view(value))
        elif key == "view.accum_frames":
            self._engine.set_accum_frames(value)
            self._refresh_accum()
        elif key == "view.accum_max_points":
            self._engine.set_accum_max_points(value)
            self._refresh_accum()
        elif key == "model.display":
            self.post(lambda: self._viz.set_model_display(value))
        elif key == "model.threshold":
            if self._viz is not None:
                # The detections view is drawn *at* the threshold, so moving it
                # has to invalidate the viewer's recolor cache.
                self.post(lambda: self._viz.set_threshold(value))
            else:
                self._scorer.threshold = value
        elif key == "model.enabled":
            self._scorer.settings.enabled = value
        elif key == "model.persist":
            self._scorer.settings.persist = value
        elif key in ("model.interval_sec", "model.window_sec"):
            self._scorer_setting(key.split(".", 1)[1], value)
        elif key == "replay.position":
            self._engine.source.seek(value)
        else:  # pragma: no cover - the table above covers every settable id
            raise KeyError(key)

    def action(self, name: str, **kw) -> None:
        control = spec.CONTROLS_BY_ID.get(name)
        if control is None or control.kind != "action":
            raise KeyError(name)
        if control.requires and control.requires not in self.capabilities:
            raise ValueError(f"{name} is not available in this run")

        if name == "view.reset_camera":
            self.post(self._viz.reset_camera)
        elif name == "view.refl_autofit":
            self.post(self._viz.autofit_reflectivity)
        elif name == "view.reset_surface":
            self._engine.reset_surface()
            self._history.clear()   # the trend described a map that is now gone
            self._refresh_accum()
        elif name == "status.pause":
            self._engine.toggle_pause()
        elif name == "level.recalibrate":
            if self._viz is not None:
                self.post(self._viz.recalibrate_level)
            else:
                self._engine.recalibrate_level()
                self._engine.reset_surface(reset_slam=False)
        elif name == "record.toggle":
            self._toggle_recording()
        elif name == "model.clear":
            self._scorer.clear_map()
            self._refresh_accum()
        elif name == "replay.play_pause":
            self._engine.source.toggle_play()
        elif name == "replay.restart":
            self._engine.source.seek(0.0)
            if self._scorer is not None:
                self._scorer.clear_map()
            self._history.clear()
            self._refresh_accum()
        else:  # pragma: no cover
            raise KeyError(name)

    # -- helpers -------------------------------------------------------- #
    @staticmethod
    def _coerce(control, value):
        """JSON in, the type the underlying field wants out."""
        if control.kind == "bool":
            if isinstance(value, str):
                return value.lower() in ("1", "true", "yes", "on")
            return bool(value)
        if control.kind == "enum":
            allowed = [c.value for c in control.choices]
            if isinstance(allowed[0], int):
                value = int(value)
            if value not in allowed:
                raise ValueError(f"{control.id}: {value!r} is not one of {allowed}")
            return value
        if control.kind in ("int", "float", "transport"):
            try:
                num = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{control.id}: {value!r} is not a number")
            if num != num:  # NaN would poison whatever it is written into
                raise ValueError(f"{control.id}: not a number")
            lo = control.min if control.min is not None else num
            hi = control.max if control.max is not None else num
            num = min(max(num, lo), hi)
            return int(round(num)) if control.kind == "int" else num
        raise ValueError(f"{control.id}: unsettable kind {control.kind}")

    def _crop_setting(self, field: str, value) -> None:
        """Write a CropConfig field, via the viewer when there is one so the
        display crop and the accumulated cloud pick the new band up at once."""
        if self._viz is not None:
            self.post(lambda: self._viz.set_crop_setting(field, value))
        else:
            setattr(self._cfg.crop, field, value)

    def _scorer_setting(self, field: str, value) -> None:
        """Write a ScoreSettings field, via the viewer when there is one so its
        recolor cache is invalidated the same way a panel edit would."""
        if self._viz is not None:
            self.post(lambda: self._viz.set_score_setting(field, value))
        else:
            setattr(self._scorer.settings, field, value)

    def _refresh_accum(self) -> None:
        if self._viz is not None:
            self.post(self._viz.refresh_accum)

    def _toggle_recording(self) -> None:
        if self._replay:
            raise ValueError("replay runs cannot record")
        if self._viz is not None:
            self.post(self._viz.toggle_recording)
            return
        if self._engine.recorder is None:
            path = self._engine.start_recording()
            if path:
                print(f"[rocklabel] recording -> {path}", flush=True)
        else:
            path = self._engine.stop_recording()
            print(f"[rocklabel] recording saved: {path}", flush=True)


__all__ = ["LiveController"]
