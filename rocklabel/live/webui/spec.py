"""Declarative catalog of every control the live web panel offers.

One :class:`Control` per knob, grouped into :class:`Section`\\ s, each carrying
the help text the page shows. The server hands this to the browser, so the page
renders whatever the backend supports instead of hardcoding a form — the same
arrangement :mod:`rocklabel.dashboard.spec` uses for CLI commands.

Sections and controls declare what they ``require`` of the run: a replay-only
transport bar, a Model section that only exists with ``--model``, a Levelling
section that only exists when levelling is active.
:func:`sections_for` filters the table down to one run's capabilities.

The help strings are the ones the Open3D panel carried as tooltips. They are the
accumulated answer to "what does this number actually do", and are worth more
than the widgets they were attached to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Capability flags a run can have. Everything in the table gates on these.
CAPABILITIES = ("replay", "live", "scorer", "leveler", "viewer")


@dataclass
class Choice:
    value: object
    label: str
    requires: str = ""


@dataclass
class Control:
    """One control. ``id`` is what :meth:`LiveController.set` dispatches on."""

    id: str
    kind: str                       # bool|float|int|enum|action|readout|transport
    label: str
    help: str = ""
    requires: str = ""              # capability needed, "" = always
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str = ""
    choices: list[Choice] = field(default_factory=list)
    key: str = ""                   # keyboard shortcut shown beside the label
    style: str = ""                 # "" | "primary" | "danger" — actions only
    #: Readouts only: render the value in the warning color when this is true
    #: in the snapshot's ``flags``.
    warn_flag: str = ""


@dataclass
class Section:
    id: str
    title: str
    blurb: str = ""
    requires: str = ""
    controls: list[Control] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
SECTIONS: list[Section] = [
    Section(
        "status", "Status",
        "What the pipeline is doing right now.",
        controls=[
            Control("status.rate", "readout", "Throughput",
                    "Points per second arriving from the source.",
                    warn_flag="rate_low"),
            Control("status.cells", "readout", "Cells",
                    "Occupied cells in the fused 2.5D height map."),
            Control("status.accum", "readout", "Accumulated",
                    "Frames, points and time span currently held by the "
                    "accumulated cloud. 'cap' means the point ceiling is "
                    "dropping frames before the frame limit does.",
                    warn_flag="accum_capped"),
            Control("status.pose", "readout", "Pose",
                    "SLAM / IMU state. 'locked' means scan-to-map odometry is "
                    "tracking; without it the sensor must stay put."),
            Control("status.state", "readout", "State",
                    "Whether ingestion is running, paused, or replaying.",
                    warn_flag="paused"),
            Control("status.pause", "action", "Pause / resume ingest", key="space",
                    help="Stop consuming from the source without tearing the "
                         "pipeline down. The map and the model keep whatever "
                         "they already have.",
                    requires="live"),
        ],
    ),

    Section(
        "replay", "Replay", "Transport for the recording being played back.",
        requires="replay",
        controls=[
            Control("replay.play_pause", "action", "Play / pause", key="space",
                    help="Start or stop playback. The pipeline keeps fusing "
                         "whatever has already been delivered.",
                    style="primary"),
            Control("replay.position", "transport", "Position", unit="s",
                    help="Scrub through the recording. Seeking backwards rewinds "
                         "the file and re-fuses the map from the top, so it costs "
                         "more than seeking forward."),
            Control("replay.restart", "action", "Restart", key="R",
                    help="Jump back to 0 s and rebuild the map from scratch."),
        ],
    ),

    Section(
        "view", "View", "How the scene is drawn. Display only — never the data.",
        controls=[
            Control("view.color_mode", "enum", "Color by",
                    "What the points are colored by. 'Model prediction' needs a "
                    "checkpoint (--model).",
                    key="V", requires="viewer",
                    choices=[
                        Choice("height", "Height"),
                        Choice("reflectivity", "Reflectivity"),
                        Choice("reflectivity_stretch", "Reflectivity (contrast)"),
                        Choice("model", "Model prediction", requires="scorer"),
                    ]),
            Control("view.refl_min", "float", "Refl. min",
                    "Low end of the reflectivity color ramp, as a fraction of "
                    "the sensor's full scale. Everything at or below it takes "
                    "the bottom color instead of being squeezed into the ramp. "
                    "Absolute, so unlike 'Reflectivity (contrast)' the colors "
                    "still mean the same thing frame to frame.",
                    min=0.0, max=1.0, step=0.005, requires="viewer"),
            Control("view.refl_max", "float", "Refl. max",
                    "High end of the ramp. Everything at or above it takes the "
                    "top color. Bring the two ends in around the band the arena "
                    "actually returns (try auto-fit, then nudge) and "
                    "ground/rock separate instead of rendering as one flat "
                    "blob.",
                    min=0.0, max=1.0, step=0.005, requires="viewer"),
            Control("view.refl_autofit", "action", "Auto-fit reflectivity range",
                    "Set both ends from what is on screen now — the 5th and "
                    "95th percentile of the current returns. That is the same "
                    "window 'Reflectivity (contrast)' picks per frame, except "
                    "it stops moving, so from here you can nudge it by hand and "
                    "compare frames.",
                    key="A", requires="viewer"),
            Control("view.nav_mode", "enum", "Navigation",
                    "How the mouse drives the camera. Orbit spins around the "
                    "pivot — double-click a point in the 3D window to move it "
                    "there. Fly is WASD + drag, for getting inside the cloud.",
                    requires="viewer",
                    choices=[Choice("orbit", "Orbit around pivot"),
                             Choice("fly", "Fly (WASD + drag)")]),
            Control("view.point_size", "float", "Point size",
                    "Screen size of each drawn point. Larger reads better on a "
                    "sparse cloud; smaller shows fine surface detail.",
                    min=1.0, max=8.0, step=0.1, requires="viewer"),
            Control("view.show_points", "bool", "Raw points", key="P",
                    requires="viewer",
                    help="The most recent scans, a rolling buffer. This is what "
                         "the model scores."),
            Control("view.show_mesh", "bool", "Surface mesh", key="M",
                    requires="viewer",
                    help="The reconstructed 2.5D Kalman-fused surface."),
            Control("view.show_accum", "bool", "Accumulated cloud", key="C",
                    requires="viewer",
                    help="The world-frame cloud built from the last N frames — "
                         "how many is the 'Accum frames' box below."),
            Control("view.show_box", "bool", "Sensor box", key="B",
                    requires="viewer",
                    help="An orange box marking the sensor's current world pose."),
            Control("view.accum_frames", "int", "Accum frames",
                    "How many recent frames the accumulated cloud keeps before "
                    "the oldest drop out. Raise it to hold the map on screen "
                    "longer — the Accumulated readout shows the time span and "
                    "point count that buys. A frame is one source batch: the "
                    "SICK sends ~225 small ones a second, a ros2 bag one full "
                    "cloud per revolution.",
                    min=1, max=200_000, step=1),
            Control("view.accum_max_points", "int", "Point cap",
                    "Memory / render guard on that cloud: frames also drop once "
                    "their points exceed this. The Accumulated readout shows "
                    "'cap' while it is the limit that bites — raise it to let "
                    "the frame count rule, but past ~1M every rebuild costs "
                    "visible frame time.",
                    min=10_000, max=5_000_000, step=10_000, unit="pts"),
            Control("view.crop_view", "bool", "Crop view to region",
                    requires="viewer",
                    help="Hide points outside the scoring region (walls, "
                         "ceiling) from the display. Affects the view only, "
                         "never the data."),
            Control("view.reset_camera", "action", "Reset view",
                    "Return the camera to the default overview angle.",
                    requires="viewer"),
            Control("view.reset_surface", "action", "Reset surface", key="R",
                    help="Throw away the fused height map and start rebuilding "
                         "it from the incoming scans.",
                    requires="live"),
        ],
    ),

    Section(
        "crop", "Crop",
        "The band of points the pipeline keeps. Applied to every batch before "
        "fusion, so unlike the View card this one changes the data. Without a "
        "model loaded it is also the region 'Crop view to region' follows.",
        controls=[
            Control("crop.enabled", "bool", "Apply crop",
                    help="Off feeds every return to the heightmap — walls and "
                         "ceiling included, which a 2.5D map cannot represent."),
            Control("crop.z_min", "float", "z min",
                    "Lower edge of the band. Handheld rig ~1 m above the "
                    "floor: -1.5.",
                    min=-10.0, max=10.0, step=0.05, unit="m"),
            Control("crop.z_max", "float", "z max",
                    "Upper edge. Keeping it below the sensor (-0.5) throws "
                    "away walls and ceiling before they ever reach the map.",
                    min=-10.0, max=10.0, step=0.05, unit="m"),
            Control("crop.floor_relative", "bool", "Anchor to floor",
                    help="Measure the band from the levelled floor plane "
                         "instead of the sensor, so one setting works across "
                         "rigs of different heights (CLI: --floor-band). "
                         "Stays absolute until levelling has found a floor."),
            Control("crop.range_max", "float", "Max range",
                    "Keep only returns closer than this to the sensor. 0 "
                    "disables the gate; 8 m covers a room.",
                    min=0.0, max=50.0, step=0.1, unit="m"),
            Control("crop.range_min", "float", "Min range",
                    "Drop returns closer than this — self-hits off the rig, "
                    "dust at the lens. 0 disables it.",
                    min=0.0, max=10.0, step=0.05, unit="m"),
        ],
    ),

    Section(
        "level", "Levelling",
        "Gravity levelling of the world frame, so a tilt-mounted sensor does "
        "not tilt the whole map.",
        requires="leveler",
        controls=[
            Control("level.tilt", "readout", "Mount tilt",
                    "The measured tilt of the sensor mount, solved from the "
                    "floor plane."),
            Control("level.recalibrate", "action", "Re-measure tilt", key="L",
                    help="Re-solve the mount tilt from the floor in view. Do "
                         "this after the rig is repositioned or re-mounted."),
        ],
    ),

    Section(
        "record", "Recording", "Write the incoming stream to disk.",
        requires="live",
        controls=[
            Control("record.state", "readout", "State",
                    "Whether a recording is being written right now. Start and "
                    "stop it with the button below, or S in the Open3D window.",
                    warn_flag="recording"),
            Control("record.captured", "readout", "Captured",
                    "Elapsed time and total points written so far — the size of "
                    "the take you would have if you stopped now."),
            Control("record.file", "readout", "File",
                    "Name of the .mcap being written. It lands in recordings/ "
                    "unless a path was passed on the command line."),
            Control("record.toggle", "action", "Start / stop recording", key="S",
                    help="Write the incoming stream to a native .mcap. "
                         "Everything downstream reads that format directly.",
                    style="primary"),
        ],
    ),

    Section(
        "model", "Model", "Live scoring with the loaded checkpoint.",
        requires="scorer",
        controls=[
            Control("model.enabled", "bool", "Score live cloud",
                    help="Run the classifier on the freshest scan every update "
                         "interval. Turn it off to free the GPU."),
            Control("model.persist", "bool", "Remember predictions (map)",
                    help="Merge each pass into a persistent per-voxel prediction "
                         "map so coverage builds up as you sweep the room. Off "
                         "shows only the latest pass."),
            Control("model.display", "enum", "Display",
                    "Confidence paints the turbo ramp (blue = clear, red = "
                    "rock). Detections is the binary view at the decision "
                    "threshold.",
                    requires="viewer",
                    choices=[Choice(0, "Confidence"),
                             Choice(1, "Detections @ threshold")]),
            Control("model.threshold", "float", "Threshold",
                    "Probability above which a center counts as rock. The "
                    "checkpoint's own tuned value is the starting point.",
                    min=0.0, max=1.0, step=0.001),
            Control("model.interval_sec", "float", "Interval",
                    "Seconds between scoring passes. A pass costs tens of "
                    "milliseconds, so 0.5 leaves the GPU nearly idle.",
                    min=0.25, max=5.0, step=0.05, unit="s"),
            Control("model.window_sec", "float", "Scan window",
                    "0 scores a single scan, which is what the models were "
                    "trained on. Raising it densifies the input and drifts out "
                    "of distribution — predictions decay.",
                    min=0.0, max=2.0, step=0.05, unit="s"),
            Control("model.clear", "action", "Clear predictions",
                    "Empty the prediction map and start accumulating again.",
                    style="danger"),
            Control("model.map", "readout", "Map",
                    "Centers held in the prediction map, and how many are above "
                    "the threshold."),
            Control("model.pass", "readout", "Last pass",
                    "Centers scored in the most recent pass, and how long it "
                    "took."),
            Control("model.region", "readout", "In region",
                    "Points from the fresh scan that fell inside the scoring "
                    "region. Zero means the region is wrong.",
                    warn_flag="region_empty"),
        ],
    ),

    Section(
        "region", "Scoring region",
        "The band around the sensor the model is fed, and the crop the view "
        "follows. Anchored to the measured floor when --floor-band was used, "
        "otherwise to the sensor.",
        requires="scorer",
        controls=[
            Control("region.z_min", "float", "z min",
                    "Lower edge of the band. Handheld rig ~1 m above the floor: "
                    "-1.5.",
                    min=-5.0, max=2.0, step=0.05, unit="m"),
            Control("region.z_max", "float", "z max",
                    "Upper edge. Keeping it below the sensor (-0.5) throws away "
                    "walls and ceiling before they ever reach the model.",
                    min=-3.0, max=4.0, step=0.05, unit="m"),
            Control("region.range_max", "float", "Max range",
                    "Horizontal radius to keep. 0 disables it. 8 m covers a room.",
                    min=0.0, max=15.0, step=0.1, unit="m"),
            Control("region.max_centers", "int", "Max centers",
                    "Hard cap on candidate centers per pass — the guard rail "
                    "that keeps a dense room from allocating gigabytes.",
                    min=500, max=20_000, step=100),
        ],
    ),
]

#: Every control, flat, keyed by id — the server's dispatch table.
CONTROLS_BY_ID: dict[str, Control] = {
    c.id: c for s in SECTIONS for c in s.controls
}


# --------------------------------------------------------------------------- #
# Filtering + serialization
# --------------------------------------------------------------------------- #
def sections_for(caps: set[str]) -> list[Section]:
    """The sections and controls that apply to a run with ``caps``.

    A section whose controls all drop out is dropped too, so a headless run
    does not show an empty View card.
    """
    out = []
    for sec in SECTIONS:
        if sec.requires and sec.requires not in caps:
            continue
        keep = [c for c in sec.controls if not c.requires or c.requires in caps]
        if keep:
            out.append(Section(sec.id, sec.title, sec.blurb, sec.requires, keep))
    return out


def _control_json(c: Control, caps: set[str]) -> dict:
    d = {"id": c.id, "kind": c.kind, "label": c.label, "help": c.help}
    for name in ("min", "max", "step"):
        if getattr(c, name) is not None:
            d[name] = getattr(c, name)
    for name in ("unit", "key", "style", "warn_flag"):
        if getattr(c, name):
            d[name] = getattr(c, name)
    if c.choices:
        d["choices"] = [{"value": ch.value, "label": ch.label}
                        for ch in c.choices
                        if not ch.requires or ch.requires in caps]
    return d


def to_json(caps: set[str]) -> dict:
    """The schema payload the page renders from."""
    return {
        "capabilities": sorted(caps),
        "sections": [
            {"id": s.id, "title": s.title, "blurb": s.blurb,
             "controls": [_control_json(c, caps) for c in s.controls]}
            for s in sections_for(caps)
        ],
    }


__all__ = ["CAPABILITIES", "Choice", "Control", "Section", "SECTIONS",
           "CONTROLS_BY_ID", "sections_for", "to_json"]
