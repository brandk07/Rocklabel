"""Declarative catalog of every rocklabel command the dashboard can run.

One :class:`Command` per CLI subcommand, each carrying the prose the dashboard
shows ("what is this / when do I reach for it") and a :class:`Param` per flag so
the run form can be generated instead of hand-written. :func:`build_argv` turns
a dict of form values back into the exact argv the CLI expects, which is also
what the UI previews — so the command you see is the command that runs.

Keeping this a data structure (rather than HTML in a template) means adding a
flag to the CLI is a one-line addition here, and the form, the help popover,
the validation, and the command preview all update together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Both are torch-free by construction (see rocklabel/train/__init__.py), which
# is what lets the dashboard quote the real training defaults and the real list
# of ablation suites instead of copies that go stale.
from ..profiles import DEFAULT_PROFILE, PROFILES
from ..train import TRAIN_DEFAULTS
from ..train.ablate import (DEFAULT_REPORT_ROOT as REPORT_ROOT,
                            DEFAULT_ROOT as EXPERIMENTS_ROOT,
                            SUITES as ABLATION_SUITES)
from ..train.cli import DEFAULT_CACHE, DEFAULT_RUNS_ROOT
from ..train.models_meta import BEV_CHANNELS
from ..train.models_meta import MODELS as ARCHITECTURES

# Quoted, not copied: the Solve-poses form offers the solver's real defaults, so
# a knob retuned in rocklabel/slam/config.py moves the form with it. The module
# is plain dataclass fields with no heavy imports, which keeps spec.py torch-free.
from ..slam.config import AltSlamConfig as _SlamConfig
from ..recording.selfhits import DEFAULT_RADIUS as _SELFHIT_RADIUS

_SLAM_DEFAULTS = _SlamConfig()

#: Early-stop patience, read from the training defaults rather than repeated.
#: It was repeated once, and the form went on offering 6 for months after the
#: real default moved to 10.
TRAIN_PATIENCE = TRAIN_DEFAULTS["patience"]

#: Pipeline stages, in workflow order. Drives the Overview flow diagram and the
#: grouping of the command list.
STAGES = [
    {"id": "capture", "title": "Capture", "blurb": "Get LiDAR data onto disk."},
    {"id": "triage", "title": "Triage", "blurb": "Understand and clean a recording."},
    {"id": "slam", "title": "Solve poses", "blurb": "Work out where the sensor was, "
                                                    "scan by scan."},
    {"id": "label", "title": "Label", "blurb": "Mark where the rocks are."},
    {"id": "dataset", "title": "Dataset", "blurb": "Turn labels into training data."},
    {"id": "train", "title": "Train", "blurb": "Fit and evaluate the classifiers."},
    {"id": "deploy", "title": "Deploy", "blurb": "Run a trained model on real data."},
]


@dataclass
class Param:
    """One CLI argument, rendered as one form control.

    ``arg`` is the flag spelling (``--z-min``); ``None`` means positional.
    ``source`` names an inventory list the UI turns into a picker dropdown.
    """

    name: str
    kind: str                      # path|dir|outpath|outdir|enum|multi|int|float|bool|text
    label: str
    help: str = ""
    arg: str | None = None         # None => positional
    default: object = None
    choices: list[str] = field(default_factory=list)
    source: str | None = None      # recordings|labels|datasets|checkpoints|configs|runs
    min: float | None = None
    max: float | None = None
    step: float | None = None
    required: bool = False
    placeholder: str = ""
    advanced: bool = False         # tucked under "Advanced" in the form
    repeat: bool = False           # repeatable flag (comma-split in the UI)
    #: What separates the values of a repeatable field in the UI. Defaults to a
    #: comma; a value that *contains* commas (a --box is six of them) sets this
    #: to ";" so the parts survive the split.
    sep: str = ","
    #: With repeat: emit one flag followed by every value (argparse nargs="+")
    #: instead of repeating the flag per value.
    nargs: bool = False
    unit: str = ""


@dataclass
class Preset:
    """A named bundle of parameter values — the settings that actually work."""

    name: str
    help: str
    values: dict


@dataclass
class Command:
    id: str
    bin: str                       # "rocklabel" | "rocklabel-train"
    sub: str
    title: str
    tagline: str                   # one line, shown on the card
    stage: str
    what: str                      # the "what does this do" paragraph
    why: str                       # when you reach for it
    notes: list[str] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)
    presets: list[Preset] = field(default_factory=list)
    gui: bool = False              # opens an Open3D window
    #: Serves the live browser control panel (`--web-ui`). The dashboard adds
    #: that flag itself for these commands and embeds the result — you are
    #: already in a browser, so the docked Open3D panel is the wrong surface
    #: here and the 3D window is better off giving the scene the whole screen.
    panel: bool = False
    long_running: bool = False     # progress-bar style job, not instant
    icon: str = "▸"
    #: How prominently the UI shows this command. Presentational only.
    #: "core" = the handful of commands run on nearly every pass (record,
    #: label, generate, cache, train, live) — these are the cards the Pipeline
    #: view leads with; "pipeline" = the rest of the main loop, folded into
    #: the Pipeline view's "More functions" list; "tool" = specialists, which
    #: live in that same list and in the Tools view.
    tier: str = "pipeline"

    @property
    def cli(self) -> str:
        return f"{self.bin} {self.sub}"


# --------------------------------------------------------------------------- #
# Shared parameter builders
# --------------------------------------------------------------------------- #
def _recording(required: bool = True, name: str = "mcap") -> Param:
    return Param(
        name, "path", "Recording", source="recordings", required=required,
        help="The .mcap file to work on. Both formats are auto-detected: native "
             "lidarrig recordings (/lidar/frames, poses embedded) and ROS 2 bags "
             "(PointCloud2 + /tf).",
    )


def _labels(required: bool = False) -> Param:
    return Param(
        "labels", "path", "Label file", arg="--labels", source="labels",
        required=required,
        help="Label JSON written by 'label'. Leave empty to use the default "
             "labels/<recording>.labels.json.",
    )


def _config(advanced: bool = True) -> Param:
    return Param(
        "config", "path", "Config file", arg="--config", source="configs",
        advanced=advanced,
        help="Offline rocklabel YAML (topics, crop box, sample geometry). Only "
             "ROS 2 bags need the topics: section — native recordings carry "
             "their own poses. Any change here forces a fresh dataset directory.",
    )


def _level_params() -> list[Param]:
    """Undo a sensor mount tilt baked into a recording.

    Shared by `label`, `driftcheck` and `generate` because all three must be
    given the *same* answer: rock centers are stored in world coordinates, so
    labelling levelled and generating unlevelled misplaces every one of them.
    The CLI refuses that combination outright, but the form is where a user
    would otherwise never think to look twice.
    """
    return [
        Param("level", "enum", "Undo mount tilt", arg="--level",
              choices=["", "auto", "off", "ground", "manual"],
              help="A sensor bolted on at an angle tilts every point in the "
                   "recording: the floor climbs steadily as you pan, the z-clip "
                   "range spans tens of metres, and a z clip cuts a diagonal "
                   "wedge instead of a horizontal slab. 'auto' is the default "
                   "and normally the right answer — it measures the angle from "
                   "the sensor's own path and the floor, and leaves a recording "
                   "that was already levelled at capture time untouched. "
                   "'ground' insists on a floor fit and fails loudly if it "
                   "cannot get one; 'manual' uses the roll/pitch below; 'off' "
                   "keeps the recording exactly as captured. Leave empty for "
                   "the config's setting. Pass the SAME choice to Label and "
                   "Generate — labels are world coordinates, and a mismatch is "
                   "refused rather than silently misplacing every rock."),
        Param("mount_roll", "float", "Mount roll", arg="--mount-roll", unit="°",
              min=-90.0, max=90.0, step=0.5, advanced=True,
              help="Known mount roll, in the same convention the live rig's IMU "
                   "and Label's own levelling readout report. Setting this "
                   "implies --level manual."),
        Param("mount_pitch", "float", "Mount pitch", arg="--mount-pitch", unit="°",
              min=-90.0, max=90.0, step=0.5, advanced=True,
              help="Known mount pitch (a sensor tipped nose-up by 40° is +40). "
                   "Use this when the ground fit picks the wrong plane, or to "
                   "reproduce an earlier session's angle exactly. Setting this "
                   "implies --level manual."),
    ]


def _profile() -> Param:
    """Which named way of cutting frames a dataset is built with.

    The single biggest lever on a dataset, and until now it lived in four
    look-alike YAML files at the repo root that nobody could tell apart. The
    descriptions ride along so the form says what each one does and when it is
    the wrong choice.
    """
    return Param(
        "profile", "enum", "How to cut frames", arg="--profile",
        choices=list(PROFILES), default=DEFAULT_PROFILE, required=True,
        help="  ".join(
            f"• {p.title}: {p.what} {p.when}" for p in PROFILES.values()),
    )


def _gpu_fraction() -> Param:
    return Param(
        "gpu_fraction", "float", "Cap GPU memory share", arg="--gpu-fraction",
        min=0.05, max=1.0, step=0.05, advanced=True,
        help="Most of the graphics card's memory this job may ever take, as a "
             "fraction — 0.45 means 45%. Only worth setting when something else "
             "is already training on the same card: whichever job asks for "
             "memory second is the one that crashes, and that is usually not "
             "the one at fault. Capping the new job means it fails on its own "
             "account instead of knocking over a sweep that has been going for "
             "hours. Leave empty for no cap.",
    )


def _seg_geometry() -> list[Param]:
    """The whole-frame segmenter's three levels, which used to be unreachable.

    Only the per-point segmenter reads these; the two sliding-window
    classifiers ignore them. Left blank they keep the geometry every
    segmentation run so far was trained with.
    """
    return [
        Param("seg_npoints", "text", "Segmenter level sizes", arg="--seg-npoints",
              repeat=True, nargs=True, advanced=True, placeholder="1024, 256, 64",
              help="Segmenter only. How many points each of the three "
                   "zoom-out levels keeps, biggest first. Blank means the "
                   "built-in 512 128 32, which throws away three quarters of "
                   "a frame at the very first level. Three numbers, "
                   "descending, comma separated."),
        Param("seg_radii", "text", "Segmenter level widths (m)", arg="--seg-radii",
              repeat=True, nargs=True, advanced=True, placeholder="0.1, 0.3, 0.8",
              help="Segmenter only. How wide a ball, in metres, each level "
                   "looks at — finest first. Blank means the built-in 0.25 0.6 "
                   "1.4. The labelled rocks measure 21-68 cm across (mean "
                   "44 cm), so a 0.25 m radius is a half-metre ball that "
                   "swallows a whole rock; smaller values "
                   "let the first level see the surface of a rock rather than "
                   "the rock as one blob. Three numbers, ascending, comma separated."),
        Param("seg_height_ref", "enum", "Segmenter height reference",
              arg="--seg-height-ref", choices=["floor", "base"],
              default=TRAIN_DEFAULTS["seg_height_ref"], advanced=True,
              help="Segmenter only. What counts as height zero for every point. "
                   "\"floor\" measures from the ground the frame itself shows, so "
                   "it stops mattering how high the robot rides above the floor — "
                   "leave it here. \"base\" measures from the robot instead, which "
                   "is what every segmenter before August 2026 was trained with "
                   "and why they read an empty arena on the competition "
                   "recording: the robot sat 0.87-0.97 m above the floor in all "
                   "twelve volleyball recordings and about 0.37 m above it at "
                   "the competition, and a gap that size takes the model's best "
                   "confidence anywhere to 0.003. Pick \"base\" only to "
                   "reproduce an old run."),
    ]


def _bev_geometry() -> list[Param]:
    """The BEV CNN's grid and network size.

    Only the grid model reads these; the point-based models ignore them. The
    defaults are what the 'bev' sweep was defined against, so leaving them
    alone reproduces it.
    """
    return [
        Param("bev_cell", "float", "Grid cell size (m)", arg="--bev-cell",
              default=TRAIN_DEFAULTS["bev_cell"], min=0.02, max=0.5, step=0.01,
              advanced=True,
              help="BEV CNN only. How much ground one cell of the picture "
                   "covers. 0.10 m matches the stored BEV rasters and puts a "
                   "44 cm rock across about four cells. Smaller sees finer "
                   "detail and costs the square of the change in speed."),
        Param("bev_grid", "int", "Grid size (cells)", arg="--bev-grid",
              default=TRAIN_DEFAULTS["bev_grid"], min=32, max=512, advanced=True,
              help="BEV CNN only. How many cells across the square grid is. It "
                   "has to hold a whole frame after training's random heading "
                   "rotation: the furthest point from a frame's centre "
                   "anywhere in the cache is 6.90 m, so 144 cells of 0.10 m "
                   "(reaching 7.2 m either way) clips nothing, while a grid "
                   "sized to the 8x8 m crop box would cut a rotated corner "
                   "off. Must divide evenly by 2 as many times as there are "
                   "levels."),
        Param("bev_width", "int", "Network width", arg="--bev-width",
              default=TRAIN_DEFAULTS["bev_width"], min=8, max=128, advanced=True,
              help="BEV CNN only. How many channels the first level of the "
                   "network carries; each level below doubles it. 32 is about "
                   "1.9 million weights, 16 about half a million. With rock "
                   "only 1.2% of labelled cells, wider is not obviously "
                   "better."),
        Param("bev_depth", "int", "Network levels", arg="--bev-depth",
              default=TRAIN_DEFAULTS["bev_depth"], min=1, max=5, advanced=True,
              help="BEV CNN only. How many times the network halves the grid "
                   "before building it back up. More levels means each cell's "
                   "answer is informed by more of the scene around it."),
        Param("bev_channels", "multi", "Cell measurements", arg="--bev-channels",
              repeat=True, nargs=True, choices=list(BEV_CHANNELS),
              default=list(BEV_CHANNELS), advanced=True,
              help="BEV CNN only. Which measurements of each cell the network "
                   "reads. \"occupied\" and \"count\" are the density pair — "
                   "whether anything came back from this patch of ground, and "
                   "how much. They are the one thing a grid can see that a "
                   "point model cannot, because the point tensor is padded by "
                   "repeating real points and the true count is only used to "
                   "mask them off. A stray return that hit nothing lands in a "
                   "cell of its own with a count of one. Untick both to test "
                   "whether that is really what the grid wins on. The two "
                   "intensity rows are ignored unless reflectivity is also "
                   "ticked as an input channel."),
    ]


def _cache_dir(advanced: bool = True) -> Param:
    return Param(
        "cache_dir", "dir", "Cache folder", arg="--cache-dir",
        default=DEFAULT_CACHE, source="caches", advanced=advanced,
        help="The pooled cache to train from. There is one per way of cutting "
             f"frames (default: {DEFAULT_CACHE}). Pointing this at the wrong "
             "one trains on frames built a different way, and the score will "
             "not compare to anything.",
    )


def _device() -> Param:
    return Param(
        "device", "enum", "Torch device", arg="--device", choices=["", "cuda", "cpu"],
        advanced=True,
        help="Leave on auto unless you want to force CPU (e.g. the GPU is busy "
             "with a training run).",
    )


#: Every architecture the trainer accepts, in the CLI's own sorted order.
#: Quoted from the registry rather than retyped so a fourth model shows up
#: here automatically.
ARCHITECTURE_CHOICES = sorted(ARCHITECTURES)

#: The two sliding-window classifiers. Deliberately separate from the full
#: list: Compare and Report exist to put raw scores side by side, and those are
#: only comparable between models graded on the same candidate spots. The
#: segmenter is graded per point and gets its fair comparison from
#: 'Segmenter vs classifier' instead.
CLASSIFIER_CHOICES = [m for m in ARCHITECTURE_CHOICES
                      if ARCHITECTURES[m][0] == "classify"]

# Back-compat alias used by older call sites.
MODEL_CHOICES = CLASSIFIER_CHOICES


def _models(help: str) -> Param:
    """Which architectures a sweep or report covers.

    A checkbox pair rather than free text: the vocabulary is closed, and a
    typo here costs a whole training sweep before anything complaints.
    Deliberately the two sliding-window classifiers only — see
    :data:`CLASSIFIER_CHOICES` for why the segmenter stays out of these forms.
    """
    return Param(
        "models", "multi", "Models", arg="--models", repeat=True, nargs=True,
        choices=CLASSIFIER_CHOICES, default=list(CLASSIFIER_CHOICES), help=help,
    )


def _features() -> Param:
    """Which channels of the stored sample tensor reach the model.

    Shared by both training commands. The vocabulary comes from
    :data:`rocklabel.dataset.neighborhoods.FEATURES` so the form can never drift from
    what the generator actually writes.
    """
    from ..dataset.neighborhoods import FEATURES

    return Param(
        "features", "multi", "Input channels", arg="--features", repeat=True,
        nargs=True, choices=list(FEATURES), default=list(FEATURES),
        help="The per-point values fed to the network. dx/dy are horizontal "
             "offsets from the sample center, dz is height above the "
             "neighborhood's lowest point, intensity is LiDAR reflectivity. "
             "Untick intensity to train on shape alone — reflectivity is the "
             "channel least likely to survive a change of surface, and a model "
             "that leans on it learns your test surface rather than what a rock "
             "is. Selection happens inside the model, so no dataset needs "
             "regenerating and any two selections stay directly comparable on "
             "the same cache. PointNet++ groups points by position and so needs "
             "dx, dy and dz; PointNet takes any subset.",
    )


#: The three live-rig region flags, shared by every command that scores points.
def _region_params(advanced: bool = False) -> list[Param]:
    return [
        Param("z_min", "float", "Region z min", arg="--z-min", unit="m",
              min=-5.0, max=2.0, step=0.1, advanced=advanced,
              help="Lower edge of the band kept around the sensor, in meters "
                   "relative to the sensor. The handheld rig sits ~1 m above the "
                   "floor, so -1.5 puts the floor comfortably inside the band."),
        Param("z_max", "float", "Region z max", arg="--z-max", unit="m",
              min=-3.0, max=4.0, step=0.1, advanced=advanced,
              help="Upper edge of the band. Keeping this below the sensor "
                   "(-0.5) throws away walls and ceiling before they ever reach "
                   "the model — the single biggest speedup available."),
        Param("max_range", "float", "Max range", arg="--max-range", unit="m",
              min=0.0, max=30.0, step=0.5, advanced=advanced,
              help="Horizontal radius around the sensor to keep. 0 disables it. "
                   "8 m covers a room; larger values pull in far walls that cost "
                   "time and teach the model nothing."),
    ]


def _floating_params(advanced: bool = False) -> list[Param]:
    """Phantom-point filter, shared by the record and live/replay cards."""
    return [
        Param("max_height_above_ground", "float", "Max height above ground",
              arg="--max-height-above-ground", unit="m",
              min=0.1, max=5.0, step=0.1, advanced=advanced,
              help="Throws away a point sitting more than this far above the "
                   "ground directly beneath it. The sensor's shallowest beam "
                   "rings skim the floor several metres out and every so often "
                   "report a distance shorter than the real one, which drops "
                   "that return into mid-air; stacked over a run they build the "
                   "fog of floating specks over the arena. Measured on the "
                   "competition recording the 0.5 m default clears 86% of them "
                   "and does not lose a single labelled rock return (the tallest "
                   "rock in any recording stands 0.44 m off its own base). It "
                   "also strips walls and other tall structure out of the "
                   "accumulated view — raise it, or untick it below, if you want "
                   "that scenery back."),
        Param("floating_cell", "float", "Ground-estimate column",
              arg="--floating-cell", unit="m",
              min=0.2, max=5.0, step=0.1, advanced=advanced,
              help="Width of the square column used to work out where the "
                   "ground is under each point. Wider still finds the ground "
                   "beside a big obstacle; narrower follows steep terrain more "
                   "closely but has fewer points to judge from. 1 m is the "
                   "measured sweet spot."),
        Param("keep_floating", "bool", "Keep floating points",
              arg="--keep-floating", advanced=advanced,
              help="Turns the phantom-point filter off entirely and keeps every "
                   "return no matter how high it hangs. Use it to see the raw "
                   "problem, or when you actually want walls and ceiling in the "
                   "accumulated cloud."),
    ]


_HANDHELD = Preset(
    "Floor-anchored rock band",
    "Keep 10 cm below through 60 cm above the measured floor. This works "
    "whether the sensor is handheld or mounted on the robot. Right for the "
    "sliding-window classifiers (PointNet, PointNet++), which look at one "
    "half-metre ball at a time and do not care how tall the band is.",
    {"floor_band": "-0.10, 0.60", "max_range": 8.0},
)

_SEG_SLAB = Preset(
    "Thin slab (segmenter)",
    "Keep 5 cm below through 25 cm above the measured floor. The per-point "
    "segmenter reads the whole band in one go, and it has only ever been "
    "trained on the arena recordings, whose frames hold about 20 cm of "
    "anything. Measured on the competition recording, the wider band above "
    "hands it 0.70 m of structure and it finds nothing at all; this one gets "
    "it back to hundreds of detections a pass. Use it whenever the model name "
    "ends in _seg.",
    {"floor_band": "-0.05, 0.25", "max_range": 8.0},
)


# --------------------------------------------------------------------------- #
# The catalog
# --------------------------------------------------------------------------- #
COMMANDS: list[Command] = [
    # ---------------------------------------------------------------- capture
    Command(
        id="record", bin="rocklabel", sub="record", stage="capture", gui=True,
        tier="core",
        panel=True,
        icon="●",
        title="Record",
        tagline="Capture the live sensor straight to an .mcap.",
        what="Runs the full live rig — UDP ingest from the SICK multiScan, IMU "
             "de-rotation, scan-to-map SLAM, 2.5D Kalman surface — and starts "
             "writing to disk immediately. The 3D window opens so you can watch "
             "what you are capturing.",
        why="The front of the pipeline for your own runs. Everything downstream "
            "(label, generate, train, replay) reads the native format it writes, "
            "so no conversion step is ever needed.",
        notes=[
            "Recording starts the moment the window opens. S stops it and "
            "finalizes the file; pressing S again begins a NEW recording — if "
            "you named the output explicitly, give each press a fresh name or "
            "the take is overwritten. There is no pause.",
            "Use 'Live view' instead if you want to look before you commit to "
            "writing a file.",
            "--headless works over SSH: no window, prints throughput once a second.",
        ],
        params=[
            Param("out_pos", "outpath", "Output file", placeholder="recordings/myrun.mcap",
                  help="Where to write. A bare name ('myrun') lands in "
                       "recordings/ as myrun.mcap. Leave empty and the rig "
                       "auto-names it recordings/lidar_<timestamp>.mcap."),
            Param("source", "enum", "Source", arg="--source", choices=["sim", "udp"],
                  default="udp",
                  help="'udp' is the real SICK multiScan; 'sim' generates synthetic "
                       "terrain so you can exercise the pipeline with no hardware. "
                       "The command line itself falls back to 'sim' when this flag "
                       "is omitted — this form pre-selects 'udp' because the real "
                       "sensor is what Record is for."),
            Param("model", "path", "Live model", arg="--model", source="checkpoints",
                  help="Optional. Score the live cloud with a trained checkpoint "
                       "while recording, so you can see predictions on the rocks "
                       "as you capture them."),
            *_region_params(),
            Param("floor_band", "text", "Floor band (m)", arg="--floor-band",
                  repeat=True, nargs=True, placeholder="-0.05, 0.6",
                  help="Crop band measured from the DETECTED FLOOR instead of "
                       "the sensor: two numbers, metres below and above ground — "
                       "'-0.05, 0.6' keeps 5 cm below to 60 cm above the floor. "
                       "It does not care how high you hold the rig, so it beats "
                       "the z band whenever the sensor height wanders during a "
                       "take. Needs levelling, which is on by default."),
            *_floating_params(),
            Param("sensor_ip", "text", "Sensor IP", arg="--sensor-ip", advanced=True,
                  placeholder="10.11.10.3",
                  help="Only if the sensor is not at the configured default."),
            Param("udp_port", "int", "UDP port", arg="--udp-port", advanced=True,
                  min=1, max=65535,
                  help="Port the sensor streams Compact data to. sick_scan_xd "
                       "defaults to 2115."),
            Param("cell_size", "float", "Surface cell size", arg="--cell-size", unit="m",
                  min=0.01, max=0.5, step=0.01, advanced=True,
                  help="Edge length of one heightmap cell. Smaller = finer surface, "
                       "more memory."),
            Param("no_slam", "bool", "Disable SLAM", arg="--no-slam", advanced=True,
                  help="Turn off scan-to-map odometry. Only correct if the sensor "
                       "never moves — otherwise the map smears."),
            Param("no_imu", "bool", "Disable IMU", arg="--no-imu", advanced=True,
                  help="Skip IMU de-rotation."),
            Param("yaw_only", "bool", "Yaw-only IMU", arg="--yaw-only", advanced=True,
                  help="Apply only the yaw component of the IMU rotation."),
            Param("headless", "bool", "Headless", arg="--headless", advanced=True,
                  help="No 3D window — prints stats instead. For SSH sessions."),
            Param("duration", "float", "Duration", arg="--duration", unit="s",
                  min=0, step=10, advanced=True,
                  help="Headless only: stop after this many seconds. 0 = until "
                       "you stop the job."),
        ],
        presets=[_HANDHELD, _SEG_SLAB],
    ),
    Command(
        id="live", bin="rocklabel", sub="live", stage="deploy", gui=True,
        tier="core",
        panel=True,
        icon="◉",
        title="Live view",
        tagline="Watch the sensor — or replay a file — with optional live predictions.",
        what="The same live pipeline as Record, minus the auto-recording. Point it "
             "at the sensor to look around, or at an .mcap with --play to replay a "
             "recording through the identical pipeline with a transport bar. With "
             "--model it scores the freshest scan continuously and colors points by "
             "rock probability.",
        why="This is where a trained model actually gets used, and the fastest way "
            "to sanity-check a checkpoint against reality before you trust it.",
        notes=[
            "Each pass scores the newest raw scan, not the accumulated map — the "
            "models are trained on single ~20 ms scans, and feeding them the "
            "ever-densifying fused map made predictions decay over time.",
            "Scored centers merge into a persistent per-voxel prediction map, so "
            "coverage builds up as you sweep the room.",
            "The Model card's Display dropdown has three views: raw confidence, "
            "detections at the threshold, and rock outlines — which groups the "
            "detections into clumps and draws a polygon around each one, "
            "ignoring clumps too small to be a rock. Link distance and the "
            "minimum clump size are live knobs in the panel, on the 3D window "
            "and in the browser alike.",
            "When watching the live sensor, S starts a recording at any moment "
            "(it does nothing while a replay is running). V cycles height → "
            "reflectivity → stretched reflectivity → model coloring, the last "
            "only when a checkpoint is loaded.",
        ],
        params=[
            Param("play", "path", "Replay a recording", arg="--play", source="recordings",
                  help="Replay this file instead of reading the sensor. Gives you "
                       "play/pause and a seek bar."),
            Param("speed", "float", "Playback speed", arg="--speed", unit="x",
                  default=1.0, min=0.0, max=8.0, step=0.05,
                  help="Replay only: how fast the file plays, as a multiple of "
                       "real time. 1 is real time, 2 is twice as fast, 0.25 is "
                       "quarter-speed slow motion, and 0 means no pacing at "
                       "all — run as fast as the machine can fuse the frames. "
                       "Nothing is skipped or repeated at any speed. You can "
                       "also change it mid-run from the Speed dropdown in the "
                       "replay transport bar of the live control panel."),
            Param("source", "enum", "Source", arg="--source", choices=["sim", "udp"],
                  default="udp",
                  help="Ignored when replaying. 'udp' is the real sensor; 'sim' is "
                       "synthetic terrain for testing with no hardware. The command "
                       "line itself falls back to 'sim' when this flag is omitted — "
                       "this form pre-selects 'udp' because the real sensor is what "
                       "Live view is for."),
            Param("model", "path", "Model checkpoint", arg="--model", source="checkpoints",
                  help="A best.pt from training. Adds the 'model' color mode and "
                       "the whole Model panel in the viewer."),
            Param("compare_model", "path", "Compare against", arg="--compare-model",
                  source="checkpoints",
                  help="Optional SECOND checkpoint. Opens a second window on the "
                       "same scene scored by this model, so you can watch two "
                       "checkpoints disagree about the same rock in real time. "
                       "Both windows share one set of settings — region, "
                       "threshold, display mode, rock outlines — so the model is "
                       "the only thing that differs between them. With 'Browser "
                       "control panel' on you can also open, close and re-pick "
                       "both models while it is running."),
            *_region_params(),
            Param("floor_band", "text", "Floor band (m)", arg="--floor-band",
                  repeat=True, nargs=True, placeholder="-0.05, 0.6",
                  help="Crop band measured from the DETECTED FLOOR instead of "
                       "the sensor: two numbers, metres below and above ground. "
                       "It does not care how high you hold the rig, so it beats "
                       "the z band whenever the sensor height wanders. Needs "
                       "levelling, which is on by default."),
            *_floating_params(),
            Param("carve", "bool", "Ray carving (experimental)", arg="--carve",
                  advanced=True,
                  help="Changes what the accumulated cloud *is*. Normally it "
                       "keeps the last N frames and drops the oldest, so a "
                       "phantom fades out only once it ages off the end. With "
                       "this on it keeps one map of the whole space. Repeated "
                       "credible free-space observations can retract a voxel; "
                       "nearby returns support it. Every observation keeps its "
                       "source viewpoint even when computation is batched. Keep "
                       "this opt-in until the labelled Lance per-rock audit "
                       "passes. The viewer's 'Accum frames' knob has no effect."),
            Param("carve_voxel", "float", "Carving resolution", arg="--carve-voxel",
                  unit="m", min=0.01, max=0.5, step=0.01, advanced=True,
                  help="Map cell size for ray carving. Finer keeps more detail "
                       "and costs more time on every fold. 0.05 m is the default "
                       "and what the measurements above used."),
            Param("carve_interval", "float", "Carving interval", arg="--carve-interval",
                  unit="s", min=0.05, max=5.0, step=0.05, advanced=True,
                  help="How often pooled scans are folded into the carved map. "
                       "The sensor sends about 225 telegrams a second and carving "
                       "is scheduled in batches. Each telegram retains its own "
                       "viewpoint and evidence group. Raise it if the display "
                       "stutters; lower it for a map that reacts faster."),
            Param("carve_evidence_group", "float", "Carve evidence group",
                  arg="--carve-evidence-group", unit="s", min=0.01, max=1.0,
                  step=0.01, advanced=True,
                  help="Time span counted as one independent evidence vote. "
                       "Separate from scheduling so fold latency cannot silently "
                       "change support/removal thresholds."),
            Param("carve_assumed_pose_uncertainty", "float",
                  "Assumed pose uncertainty",
                  arg="--carve-assumed-pose-uncertainty", unit="m", min=0.0,
                  max=1.0, step=0.005, advanced=True,
                  help="Fallback only when the source has no measured pose "
                       "uncertainty. Leave blank to disable destructive carving "
                       "for unknown-quality poses. Zero treats replay poses as "
                       "exact and should be an explicit experiment choice."),
            Param("carve_confirm_observations", "int", "Carve confirmations",
                  arg="--carve-confirm-observations", min=1, max=20, advanced=True,
                  help="Independent supporting observations needed to confirm a voxel."),
            Param("carve_tentative_contradictions", "int", "Tentative contradictions",
                  arg="--carve-tentative-contradictions", min=1, max=20, advanced=True,
                  help="Independent free-space observations needed to remove a tentative voxel."),
            Param("carve_confirmed_contradictions", "int", "Confirmed contradictions",
                  arg="--carve-confirmed-contradictions", min=1, max=20, advanced=True,
                  help="Independent free-space observations needed to remove confirmed geometry."),
            Param("color_mode", "enum", "Initial coloring", arg="--color-mode",
                  choices=["", "height", "reflectivity", "reflectivity_stretch",
                           "model"],
                  help="What points are colored by on open. 'reflectivity' is the "
                       "sensor's calibrated full scale, so colors compare across "
                       "frames; 'reflectivity_stretch' spreads each frame's own "
                       "brightness range across the ramp — the best rock/ground "
                       "contrast indoors. 'model' needs a checkpoint. V cycles "
                       "through all of them at runtime."),
            Param("score_interval", "float", "Scoring interval", arg="--score-interval",
                  unit="s", min=0.1, max=5.0, step=0.1, advanced=True,
                  help="Seconds between scoring passes. A pass costs 9-20 ms on "
                       "the RTX 2000, so 0.5 leaves the GPU almost idle."),
            *_level_params(),
            _device(),
            Param("record", "outpath", "Record from launch", arg="--record", advanced=True,
                  placeholder="recordings/myrun.mcap",
                  help="Also start recording immediately, as if you had pressed S."),
            Param("sensor_ip", "text", "Sensor IP", arg="--sensor-ip", advanced=True),
            Param("udp_port", "int", "UDP port", arg="--udp-port", advanced=True,
                  min=1, max=65535),
            Param("no_slam", "bool", "Disable SLAM", arg="--no-slam", advanced=True,
                  help="Only correct if the sensor never moves."),
            Param("no_crop", "bool", "Disable crop", arg="--no-crop", advanced=True,
                  help="Keep every point that arrives, including walls and ceiling."),
            Param("headless", "bool", "Headless", arg="--headless", advanced=True),
            Param("duration", "float", "Duration", arg="--duration", unit="s",
                  min=0, step=10, advanced=True,
                  help="Headless only: stop after this many seconds. 0 = until "
                       "you stop the job."),
        ],
        presets=[_HANDHELD, _SEG_SLAB],
    ),
    # ---------------------------------------------------------------- triage
    Command(
        id="inspect", bin="rocklabel", sub="inspect", stage="triage", tier="tool",
        icon="🔍",
        title="Inspect",
        tagline="Print topics, fields, TF frames and time span of a recording.",
        what="Reads a recording's index and first messages and prints everything "
             "you need to configure the rest of the pipeline: every topic with "
             "message counts and schemas, each PointCloud2 topic's frame and "
             "fields, whether intensity is present, the TF tree, and the time span.",
        why="Always the first thing to run on a recording you did not make "
            "yourself. For a ROS 2 bag its output is what you copy into "
            "config.yaml; for a native recording it confirms poses and "
            "reflectivity survived.",
        notes=[
            "Read-only and fast — it never decodes the whole file.",
            "For native lidarrig recordings it also dumps the rig config that was "
            "embedded at record time.",
        ],
        params=[_recording(), _config(advanced=False)],
    ),
    Command(
        id="trim", bin="rocklabel", sub="trim", stage="triage", tier="tool",
        icon="✂",
        title="Trim",
        tagline="Shrink a huge recording, cut a time window, or salvage a broken one.",
        what="Copies only the LiDAR and TF topics — plus any extra topic you name "
             "— into a new, properly indexed .mcap, optionally keeping just a time "
             "window. Because it reads sequentially it also recovers truncated "
             "files that no other tool will open.",
        why="Competition bags carry cameras and telemetry and reach tens of GB "
            "when only the LiDAR topic matters. Trim typically takes an order of "
            "magnitude off. It is also the fix for a recorder killed mid-write "
            "('no summary section' / RecordLengthLimitExceeded).",
        notes=[
            "--start-s / --end-s are seconds from the start of the recording, so "
            "'skip the first two minutes' is --start-s 120.",
            "TF topics are exempt from the time window, so pose lookups still "
            "work at its edges.",
            "Label against the trimmed file and keep using it downstream — labels "
            "are tied to the recording they were made on.",
        ],
        params=[
            _recording(),
            Param("out", "outpath", "Output file", arg="--out", required=True,
                  placeholder="recordings/run.lidar.mcap",
                  help="The new .mcap to write. Never the same path as the input."),
            Param("start_s", "float", "Start at", arg="--start-s", unit="s", min=0, step=1,
                  help="Drop everything before this many seconds into the recording."),
            Param("end_s", "float", "End at", arg="--end-s", unit="s", min=0, step=1,
                  help="Drop everything after this many seconds into the recording."),
            Param("topic", "text", "Extra topics", arg="--topic", repeat=True, advanced=True,
                  placeholder="/imu/data, /odom",
                  help="Comma-separated extra topics to keep. LiDAR and TF are "
                       "always kept."),
            Param("all_topics", "bool", "Keep every topic", arg="--all-topics", advanced=True,
                  help="Trim by time only, keeping all topics. Use when you are "
                       "salvaging a broken file rather than shrinking a fat one."),
            _config(),
        ],
        long_running=True,
    ),
    Command(
        id="selfhits", bin="rocklabel", sub="selfhits", stage="triage", tier="tool",
        icon="🤖",
        title="Remove robot self-hits",
        tagline="Delete the points the LiDAR sees of the robot carrying it.",
        what="On a competition run the sensor is bolted to the robot, so part of "
             "every sweep lands on the machine itself. Those returns sit at a "
             "fixed spot relative to the sensor, but once each sweep is placed "
             "into the world they smear into a long trail following wherever the "
             "robot drove. This writes a new recording with them gone, copying "
             "everything else through untouched.",
        why="Reach for it on the raw competition recordings, where the robot's "
            "own body sits on top of the terrain you are trying to look at and "
            "label. The original file is never modified - you get a clean copy "
            "to work from.",
        notes=[
            "Start with 'Just measure' ticked. It reports how far the robot's own "
            "structure actually reaches and tells you the radius to use, instead "
            "of you having to guess.",
            "The radius works because the sensor is mounted above the ground: real "
            "terrain physically cannot come closer than the mount height, so "
            "anything inside the sphere can only be the robot.",
            "Expect a big number - on the Lunabotics runs about 40% of all points "
            "are the robot, because it fills a large part of the sensor's view.",
            "Only points are removed. Every topic, message, timestamp and pose is "
            "copied through, so the cleaned file drops straight into labelling and "
            "dataset generation.",
        ],
        params=[
            _recording(),
            Param("out", "outpath", "Output file", arg="--out",
                  placeholder="recordings/run.clean.mcap",
                  help="The new .mcap to write. Never the same path as the input. "
                       "Not needed when you are only measuring or doing a dry run."),
            Param("radius", "float", "Robot radius", arg="--radius", unit="m",
                  default=_SELFHIT_RADIUS, min=0.0, max=5.0, step=0.05,
                  help="Throw away every point closer than this to the sensor. "
                       "Anything within it is the machine the sensor is bolted "
                       "to, because real ground is at least a mount height away."),
            Param("measure", "bool", "Just measure", arg="--measure",
                  help="Write nothing. Reports how far the robot's own structure "
                       "reaches and what radius to use. Run this first."),
            Param("dry_run", "bool", "Dry run", arg="--dry-run",
                  help="Write nothing, but report exactly how many points the "
                       "current radius would remove."),
            Param("box", "text", "Extra boxes", arg="--box", repeat=True,
                  sep=";", advanced=True,
                  placeholder="0.5,1.0,-0.3,0.3,-0.2,0.2",
                  help="Also remove points inside a box in sensor coordinates, "
                       "given as x0,x1,y0,y1,z0,z1 in meters. For a part that "
                       "sticks out past the radius, like a mast or a raised "
                       "blade. Separate several boxes with a semicolon."),
            _config(),
        ],
        long_running=True,
    ),
    # ---------------------------------------------------------------- slam
    Command(
        id="slam", bin="rocklabel", sub="slam", stage="slam",
        icon="⌖",
        title="Solve poses",
        tagline="Work out a better sensor path and write a re-solved recording.",
        what="Reads a recording, works out where the sensor really was scan by "
             "scan, and writes a NEW recording with the same points, the same "
             "reflectivity and the same IMU samples — only the pose attached to "
             "each batch changes. The original file is never touched. A "
             "recording read out of a 'raw' folder is written to the 'reslam' "
             "folder beside it.",
        why="The sensor's own live tracker is confidently wrong on a flat sand "
            "court: a plane tells you your height and nothing about where you "
            "are along it, so the whole scan slides sideways while the tracker "
            "still reports 99% of points matched. This solver aims with the far "
            "scenery (fence, tree line), lets the alignment correct tilt instead "
            "of trusting the IMU, and makes several passes over the recording. "
            "On the volleyball runs it takes the sand surface from about 33 mm "
            "thick to about 18 mm — which matters because the rocks are only "
            "5-10 cm proud of the sand.",
        notes=[
            "Everything downstream reads the output with no changes: label, "
            "generate, train and replay just see a better-aligned world.",
            "Labels are stored in world coordinates, so a recording you re-solve "
            "with DIFFERENT settings needs its labels checked. Re-running with "
            "the same settings reproduces the same trajectory exactly, and "
            "existing labels stay valid.",
            "About 110 s per 40 s recording at the default 3 passes. Far too "
            "slow to run live, which is why it runs from a file.",
            "Tick 'Score only' to try settings without writing anything — it "
            "prints the surface thickness before and after.",
            "The parser also takes --output to name a single result file, but it "
            "is only valid with exactly one input — batch jobs here always use "
            "the suffix/folder scheme, so the form leaves it out on purpose.",
            "Measured and rejected, so do not reach for them: 0.05 s windows "
            "(diverges badly), voxels above 0.20 m, more than about 2 extra "
            "passes, and locking tilt to the IMU (2x worse on a hand-swept rig).",
        ],
        params=[
            Param("inputs", "path", "Recordings", source="recordings", required=True,
                  repeat=True,
                  help="The recording(s) to re-solve. Comma-separate several to "
                       "do a batch — each is solved and written independently."),
            Param("out_dir", "outdir", "Output folder", arg="--out-dir",
                  placeholder="recordings/volleyball/reslam",
                  help="Where to write. Leave empty and a recording from a 'raw' "
                       "folder lands in the 'reslam' folder beside it; anything "
                       "else lands next to its input."),
            Param("score_only", "bool", "Score only", arg="--score-only",
                  help="Solve and print the quality numbers but write no file. "
                       "The cheap way to try a setting before committing to it."),
            Param("force", "bool", "Overwrite existing", arg="--force",
                  help="Replace an output file that already exists. Without this "
                       "an existing output is left alone and reported."),
            Param("passes", "int", "Solver passes", arg="--passes", min=1, max=6,
                  default=_SLAM_DEFAULTS.passes,
                  help="Pass 1 walks the recording forwards, so its earliest "
                       "poses are its worst and get baked in. Every later pass "
                       "rebuilds the map from the finished path and re-aligns "
                       "everything against it. 1 is about 3x faster and nearly "
                       "as good; past 3 the gain is not measurable."),
            Param("window", "float", "Seconds per window", arg="--window", unit="s",
                  min=0.05, max=0.5, step=0.05, default=_SLAM_DEFAULTS.window_sec,
                  advanced=True,
                  help="How much of the recording is pooled into one pose "
                       "solution. 0.10 and 0.20 both work. 0.05 DIVERGES on this "
                       "data — too few points per window to align with."),
            Param("range_max", "float", "Furthest points used", arg="--range-max",
                  unit="m", min=5.0, max=60.0, step=1.0,
                  default=_SLAM_DEFAULTS.reg_range_max, advanced=True,
                  help="Points past this distance take no part in alignment. "
                       "Everything within 12 m of a sand court is sand, and sand "
                       "cannot tell you where you are along it — the fence and "
                       "tree line at 15-25 m are what pin down sliding "
                       "sideways. Lower it only if the distant scenery is itself "
                       "moving (traffic, a crowd)."),
            Param("voxel", "float", "Map voxel size", arg="--voxel", unit="m",
                  min=0.05, max=0.5, step=0.05, default=_SLAM_DEFAULTS.voxel_size,
                  advanced=True,
                  help="Cell size of the map used for alignment. Bigger is "
                       "consistently worse here. This does not limit rock detail "
                       "— the written point cloud is never voxelized."),
            Param("iterations", "int", "Alignment iterations", arg="--iterations",
                  min=1, max=100, default=_SLAM_DEFAULTS.iterations, advanced=True,
                  help="How many times each window is nudged towards the map. "
                       "Offline there is time to spare, so this sits well above "
                       "what a live tracker could afford."),
            Param("degeneracy", "float", "Flat-ground guard", arg="--degeneracy",
                  min=0.0, max=0.5, step=0.01,
                  default=_SLAM_DEFAULTS.degeneracy_threshold, advanced=True,
                  help="Directions the visible geometry does not actually pin "
                       "down are left alone rather than solved for from noise. "
                       "This is the setting that stops flat sand letting the "
                       "pose slide. 0 turns the guard off entirely."),
            Param("robust_sigma", "float", "Outlier scale", arg="--robust-sigma",
                  unit="m", min=0.01, max=1.0, step=0.01,
                  default=_SLAM_DEFAULTS.robust_sigma, advanced=True,
                  help="Points disagreeing with the map by more than roughly "
                       "this are treated as outliers and stop pulling, so "
                       "somebody walking through the shot cannot drag the "
                       "answer."),
            Param("lock_tilt", "bool", "Trust the IMU for tilt", arg="--lock-tilt",
                  advanced=True,
                  help="Only let the alignment correct heading, taking roll and "
                       "pitch from the IMU. Right for a tripod or mast. WRONG "
                       "for a hand-swept sensor: swinging it accelerates it, and "
                       "an accelerometer cannot tell that from gravity — it "
                       "costs about 2x on the volleyball recordings."),
            Param("suffix", "text", "Output name suffix", arg="--suffix",
                  default=".reslam", advanced=True,
                  help="Added to the recording's name for the output, so "
                       "Run1.mcap becomes Run1.reslam.mcap. The rest of the "
                       "project expects '.reslam'."),
            Param("quiet", "bool", "No progress bar", arg="--quiet", advanced=True),
            Param("ros2_stride", "int", "Bag scan stride", arg="--ros2-stride",
                  default=4, min=1, max=50, advanced=True,
                  help="ROS 2 competition bags only: solve using every Nth "
                       "scan. Those run half an hour at ~19 Hz, which is far "
                       "denser than the solver needs and more than fits in "
                       "memory at once. Ignored for native rig recordings."),
        ],
        long_running=True,
    ),
    # ---------------------------------------------------------------- label
    Command(
        id="label", bin="rocklabel", sub="label", stage="label", gui=True,
        tier="core",
        icon="◆",
        title="Label",
        tagline="Fuse every scan into one cloud and mark the rocks by hand.",
        what="Accumulates all scans into a single voxel-fused world-frame cloud "
             "and opens the labeling app. Three shapes are available: spheres "
             "(Shift+click), boxes (drag a footprint, then set the dimensions), "
             "and lasso polygons (click an outline, then set base and top z). "
             "Points inside the active shape light up cyan as you drag.",
        why="The one manual step in the pipeline. Because everything is labeled "
            "once in the world frame and then projected into every frame, a few "
            "minutes here produces tens of thousands of training samples.",
        notes=[
            "Labels auto-save on every change to labels/<recording>.labels.json — "
            "there is no way to lose work by forgetting to save.",
            "Can't see the rocks at all? Use the Relief controls in the Display "
            "panel, not the z clip. Relief is how far each point stands above "
            "the ground directly beneath it, so it ignores whatever the floor "
            "is doing — sag, slope, ruts, footprints. Press C to colour by it, "
            "then drag 'hide below' up to about 8-10 cm: the sand vanishes and "
            "only things standing proud of the ground are left on screen. That "
            "works on outdoor ground, where the z clip does not, because a "
            "single flat z plane cannot follow a floor that is not flat.",
            "On flat indoor floors the z clip is still the quickest thing: drag "
            "z-max down near floor level and the rocks pop out as everything "
            "above them disappears.",
            "The z clip is also the run's training height band. Dragging it only "
            "hides points — nothing is deleted, and it stays as responsive as it "
            "always was — but wherever you leave it is written to the label file "
            "on save, and Generate then builds training data from that slab of "
            "height only. It is the vertical partner of the arena ring: the "
            "arena bounds the floor plan, the z clip bounds the height. Use it "
            "to throw away everything above the sensor — ceiling, lights, the "
            "person holding the rig — which otherwise trains as clear ground. "
            "The panel shows the saved band live, and a 'Reset to full height' "
            "button clears the limit.",
            "Switch to reflectivity coloring to find retroreflective markers "
            "instantly.",
            "Cloud looks like a tilted ramp and the z-clip range spans tens of "
            "metres? The sensor's mount angle is baked into the recording. Set "
            "'Undo mount tilt' to ground, and give Generate the same setting.",
            "--dump-accumulated writes the fused cloud to a PLY and exits without "
            "a window, which is the SSH-friendly way to check fusion.",
        ],
        params=[
            _recording(),
            _labels(),
            Param("stride", "int", "Scan stride", arg="--stride", min=1, max=50,
                  help="Accumulate every Nth scan. 1 uses everything (densest, "
                       "slowest to open); raise it on very long recordings."),
            Param("min_hits", "int", "Drop voxels seen fewer than N times",
                  arg="--min-hits", min=1, max=200, default=1,
                  help="Throws away every fused voxel that fewer than N scans "
                       "ever hit. Ground and rocks come back in the same voxel "
                       "every time the sensor looks at them; a stray return "
                       "lands somewhere new each scan and claims a voxel of its "
                       "own. Because the display draws one dot per voxel, a "
                       "couple of percent of bad returns can be a third of what "
                       "you see — on the lance arena 30%% of voxels were hit "
                       "exactly once out of 10,102 scans, and those sat a median "
                       "of 2.3 m up in mid-air. Reach for this when the cloud "
                       "looks like fog or the ground looks too thick to label. "
                       "1 keeps everything; 3-10 is a good range on a long "
                       "recording. Rocks do lose some surface too, so lower it "
                       "if small rocks start vanishing. Note this only cleans "
                       "the cloud you label on — Generate still builds training "
                       "frames from every scan."),
            Param("carve", "bool", "Ray carving (experimental)", arg="--carve",
                  advanced=True,
                  help="Builds the cloud with a map that can delete as well as "
                       "add. A voxel is removed only after repeated later "
                       "observations put it in visible free space; nearer returns "
                       "cannot carve occluded geometry and matching endpoints add "
                       "support. Keep this opt-in until the labelled Lance "
                       "per-rock visual audit passes. Pair it with 'Dump fused "
                       "cloud to PLY' for side-by-side inspection."),
            Param("carve_assumed_pose_uncertainty", "float",
                  "Assumed pose uncertainty",
                  arg="--carve-assumed-pose-uncertainty", unit="m", min=0.0,
                  max=1.0, step=0.005, advanced=True,
                  help="Recordings do not currently carry measured pose "
                       "uncertainty. Leave blank to collect support without "
                       "destructive carving. Zero treats the replay poses as "
                       "exact and must be an explicit experiment choice."),
            Param("z_min", "float", "Initial z min", arg="--z-min", unit="m", step=0.1,
                  help="Starting lower clip plane in odom meters. The clip is "
                       "also the saved training height band, so this seeds that "
                       "band too — but you normally set it by eye in the window "
                       "instead. Leave empty to reopen on whatever band the run "
                       "was last saved with."),
            Param("z_max", "float", "Initial z max", arg="--z-max", unit="m", step=0.1,
                  help="Starting upper clip plane. Dropping this near floor level "
                       "is how you find rocks fast — and, because the clip is the "
                       "saved height band, it is also how you stop the ceiling, "
                       "the lights and your own head from training as clear "
                       "ground. Leave empty to reopen on the run's saved band."),
            *_level_params(),
            _config(),
            Param("dump_accumulated", "outpath", "Dump fused cloud to PLY",
                  arg="--dump-accumulated", advanced=True, placeholder="cloud.ply",
                  help="Write the fused cloud and exit without opening a window."),
            Param("fallback_viewer", "bool", "Legacy viewer", arg="--fallback-viewer",
                  advanced=True,
                  help="The old pick-then-terminal viewer. Only if the GUI misbehaves."),
        ],
    ),
    Command(
        id="driftcheck", bin="rocklabel", sub="driftcheck", stage="label",
        tier="tool", gui=True,
        icon="⊕",
        title="Drift check",
        tagline="Overlay the start and end of a run around one rock to catch odometry drift.",
        what="Accumulates only the first 10% (blue) and last 10% (orange) of "
             "scans, crops both to a box around the rock you name — at least "
             "1 m across, widened for big rocks so they keep context — and "
             "overlays them.",
        why="Label-once-project-everywhere breaks silently if odometry drifts — "
            "you get a dataset where the labels no longer sit on the rocks and no "
            "error tells you. Check at least one rock per run before generating.",
        notes=[
            "Same rock surface in both colors: odometry held, the run is good.",
            "Rock doubled or smeared between the colors: do not trust this run's "
            "labels. Trim to the part that held, or re-record.",
        ],
        params=[
            _recording(),
            _labels(required=True),
            Param("rock_id", "int", "Rock id", arg="--rock-id", required=True, min=1,
                  help="Which labeled rock to inspect. Ids are listed in the label "
                       "file and in the Labels table."),
            Param("report_only", "bool", "Numbers only", arg="--report-only",
                  help="Skip the 3D window and just print how far apart the "
                       "early and late surfaces sit, in millimetres. Under "
                       "about 20 mm is sensor noise and the labels are fine; "
                       "anything approaching a rock radius (150-260 mm) means "
                       "the label no longer sits on the rock. Use this to check "
                       "a whole run's labels quickly, or after re-solving a "
                       "trajectory."),
            *_level_params(),
            _config(),
        ],
    ),
    # ---------------------------------------------------------------- dataset
    Command(
        id="generate", bin="rocklabel", sub="generate", stage="dataset",
        tier="core",
        icon="⚙",
        title="Generate dataset",
        tagline="Turn a labeled recording into training data in all three dataset formats.",
        what="Non-interactive. Replays the recording, transforms it to the odom "
             "frame, crops a robot-centered box, projects the labels into each "
             "frame, and writes all three outputs: point-neighborhood samples "
             "(format A — what the sliding-window classifiers train on), BEV "
             "rasters (format B), and whole-frame segmentation frames (format C "
             "— what the segmenter trains on). With the default profile a frame "
             "is one merged sensor rotation, and every Nth of those frames is kept.",
        why="The bridge from 'a labeled recording' to 'something a model can "
            "train on'. Several recordings accumulate into one dataset directory "
            "as long as they share an identical config.",
        notes=[
            "'How to cut frames' is the setting that matters most here, and "
            "'Full sweep' is the right answer unless you are reproducing an old "
            "result. It merges each whole sensor rotation into one frame — "
            "about 1,250 points instead of 110 — and measurably beat the old "
            "way on every model and nearly every recording.",
            "A dataset built one way cannot be added to with another — that is "
            "refused outright, because pooling them would mix populations whose "
            "scores mean different things. Pick a different output folder.",
            "A loud warning at the end about zero rock samples means label/frame "
            "misalignment — go run Drift check.",
            "'Undo mount tilt' must match what Label used. With 'ground' the fit "
            "is not re-run: the label file's own angle is replayed, so the two "
            "frames match exactly. A mismatch is an error, not a warning.",
            "The height band comes from the label file, not from here. Wherever "
            "you left the z clip in Label is the slab of height this trains on, "
            "and it replaces the crop box's up/down limits entirely (the "
            "forward/back/left/right limits still apply). The run's first lines "
            "print which band was used, so check there if the output looks thin. "
            "No band in the label file means the crop box's crop_up_m/"
            "crop_down_m are the only vertical bound, as before — and those are "
            "measured from the sensor, so crop_up_m lets in everything overhead.",
        ],
        params=[
            _recording(),
            _profile(),
            Param("out", "outdir", "Dataset directory", arg="--out",
                  source="datasets", placeholder="datasets/<profile>/<recording>",
                  help="Where to write. Leave empty and it goes to "
                       "datasets/<profile>/<recording name>, which is what puts "
                       "the way the frames were cut into the path itself. An "
                       "existing directory is appended to, as long as it was "
                       "built the same way."),
            _labels(),
            *_level_params(),
            _config(advanced=False),
        ],
        long_running=True,
    ),
    Command(
        id="preview", bin="rocklabel", sub="preview", stage="dataset",
        tier="tool", gui=True,
        icon="▦",
        title="Preview dataset",
        tagline="Browse the frames that were actually written, reconstructed from the npz files.",
        what="An interactive frame browser over a generated dataset, rebuilt from "
             "the written npz files rather than the mcap — so what you see is "
             "literally what a training job loads. Transport bar with prev/next, "
             "play/pause, an fps slider and a seek bar.",
        why="The check between generating and training. Rock cells and yellow "
            "sample dots should sit inside the red label wireframes; if they do "
            "not, something is misaligned and training on it would be wasted time.",
        notes=[
            "Gray = occupied BEV cells, red = labeled rock, olive = the ignore "
            "shell, cyan/yellow dots = format-A sample centers (clear/rock), red "
            "wireframes = the original labels.",
            "The Combine frames dropdown accumulates trailing data — set it to "
            "'Entire run' for a whole-dataset overview. Essential for native "
            "datasets generated without frame_window_s, where one frame is one "
            "sparse 4 ms batch.",
        ],
        params=[
            Param("out_pos", "dir", "Dataset", source="datasets", required=True,
                  help="A directory written by Generate."),
            Param("run", "text", "Run id", arg="--run",
                  help="Only needed if the dataset holds several runs."),
            Param("frame", "int", "Start frame", arg="--frame", min=0,
                  help="Open at a specific frame index instead of the first."),
            Param("list", "bool", "List frames and exit", arg="--list", advanced=True,
                  help="Print the available frame indices instead of opening a window."),
        ],
    ),
    Command(
        id="coverage", bin="rocklabel", sub="coverage", stage="dataset",
        icon="◎",
        title="Rock coverage check",
        tagline="Does the dataset actually contain every rock you labelled, and how far out?",
        what="Reads a generated dataset and the labels it was built from, then "
             "reports, per recording: how many of the labelled rocks produced "
             "any training samples at all, how far from the sensor those "
             "samples sit, and which labels are close enough together that "
             "their samples cannot be told apart. Prints a table in seconds — "
             "no model, no GPU.",
        why="Two things go wrong silently here, and the dataset looks perfectly "
            "healthy while they do. A correctly labelled rock can be too "
            "sparse in every single frame to clear the 20-neighbour floor, so "
            "it contributes nothing and nothing says so — twelve of the "
            "sixty-three volleyball rocks did that on the old raw-burst "
            "datasets. And the effective range can be far shorter than the "
            "crop box: those same datasets cropped to 6 m but put 88% of their "
            "rock samples inside 2 m, so the model was never shown a rock at "
            "distance. Run this after every Generate.",
        notes=[
            "Rocks reported as producing no samples are labelled and invisible. "
            "The usual fix is a denser generation profile, not a new label.",
            "Range is measured from the sensor's own position in the frame the "
            "sample came from, not from the world origin — the rig walks.",
            "Labels sitting closer together than their own radii are flagged: "
            "the per-rock counts split one pile between them, so a low count "
            "on one of a touching pair does not mean that rock was missed.",
        ],
        params=[
            Param("dataset_dir", "dir", "Dataset", source="datasets", required=True,
                  help="A directory written by Generate."),
            Param("labels_dir", "dir", "Labels folder", arg="--labels-dir",
                  advanced=True,
                  help="Only needed if the label files have moved since the "
                       "dataset was generated — the manifest records where they "
                       "were, and that is used first."),
            Param("as_json", "outpath", "Also write JSON to", arg="--json",
                  advanced=True,
                  help="Writes the full per-rock sample counts to a file, which "
                       "the printed table summarizes."),
        ],
    ),
    # ---------------------------------------------------------------- train
    Command(
        id="train-cache", bin="rocklabel-train", sub="cache", stage="train",
        tier="core",
        icon="▤",
        title="Build cache",
        tagline="Pool dataset runs into a flat .npy cache for training.",
        what="Reads the named dataset directories, validates that their config "
             "fingerprints and manifest counts agree, and writes one flat .npy "
             f"cache under {DEFAULT_CACHE}/.",
        why="Training reads the cache, not the datasets. This step is also the "
            "guard rail: it refuses to pool datasets generated with different "
            "configs, which is exactly the mistake that would silently poison a "
            "training run.",
        notes=[
            "Re-run it whenever you generate a new dataset you want to train on.",
            "One cache per way of cutting frames. Pooling datasets built two "
            "different ways is refused, and the error names both.",
        ],
        params=[
            Param("datasets", "text", "Datasets", arg="--datasets", repeat=True,
                  nargs=True, source="datasets",
                  help="Comma-separated dataset directories. Leave empty to pool "
                       f"every dataset under datasets/{DEFAULT_PROFILE}/."),
            Param("cache_dir", "outdir", "Cache folder", arg="--cache-dir",
                  default=DEFAULT_CACHE, advanced=True,
                  help="Where to write the pooled cache. Keep one folder per "
                       "way of cutting frames."),
        ],
        long_running=True,
    ),
    Command(
        id="train-train", bin="rocklabel-train", sub="train", stage="train",
        tier="core",
        icon="◈",
        title="Train one fold",
        tagline="Fit one model on one leave-one-run-out fold.",
        what="Trains one model on one fold — either of the two sliding-window "
             "classifiers or the whole-frame per-point segmenter — holding out "
             "one whole run for testing and using contiguous tail frame blocks "
             "for early stopping. Writes config.json, history.csv, "
             "last.pt/best.pt, test_metrics.json and predictions.npz into "
             f"{DEFAULT_RUNS_ROOT}/<model>_loro_<run>/.",
        why="The quick loop: one fold to see whether a change helps, before "
            "spending the time on the full comparison.",
        notes=[
            "Evaluation is leave-one-run-out on purpose. Candidate centers sit on "
            "a 5 cm grid inside 50 cm neighborhoods and consecutive frames barely "
            "move, so a random sample split leaks near-duplicates and produces "
            "meaningless scores.",
            "Runs resume from last.pt unless you tick Fresh.",
            "A non-default Input channels selection is tagged into the run "
            "directory name (pointnet_loro_run3_dx-dy-dz), so training the same "
            "fold with and without reflectivity gives you two runs to compare "
            "rather than a collision on one directory.",
            "Class imbalance (~19% rock overall, but anywhere from ~5% to ~31% "
            "in a single run) is handled with class-weighted BCE — read PR-AUC "
            "and F1, never bare accuracy, and never compare raw PR-AUC across "
            "runs whose rock share differs.",
        ],
        params=[
            Param("model", "enum", "Architecture", arg="--model",
                  choices=ARCHITECTURE_CHOICES, default="pointnet", required=True,
                  help="PointNet is smaller and faster; PointNet++ masks padded "
                       "points out of FPS and ball queries entirely; "
                       "pointnet2_seg is the whole-frame segmenter — it labels "
                       "every point in one pass and reads the cache's "
                       "segmentation arrays (every cache except the old "
                       "raw-burst one has them). Its score is per point, so "
                       "compare it to the classifiers only through 'Segmenter "
                       "vs classifier'."),
            Param("test_run", "text", "Held-out run", arg="--test-run", required=True,
                  source="cache_runs",
                  help="The run kept out of training and used as the test set. "
                       "Must be a run present in the cache — or the word 'all', "
                       "which holds nothing out and fits every recording. Use "
                       "'all' only to build a model you intend to run: it has "
                       "no honest score, because there is no unseen recording "
                       "left to score it on, so it reports validation numbers "
                       "off the tail of its own training data instead."),
            _features(),
            Param("epochs", "int", "Epochs", arg="--epochs", default=30, min=1, max=500),
            Param("batch", "int", "Batch size", arg="--batch", default=256, min=8, max=4096),
            Param("lr", "float", "Learning rate", arg="--lr", default=0.001,
                  min=0.00001, max=0.1, step=0.0001),
            Param("patience", "int", "Early-stop patience", arg="--patience",
                  default=TRAIN_PATIENCE, min=1, max=100,
                  help="Epochs without validation improvement before stopping. "
                       "Keep it long enough for the learning-rate schedule to "
                       "finish annealing, or the fold stops before it ever sees "
                       "its fine-tuning phase."),
            Param("weight_decay", "float", "Weight decay", arg="--weight-decay",
                  default=0.0001, advanced=True, step=0.0001),
            Param("val_frac", "float", "Validation fraction", arg="--val-frac",
                  default=0.15, min=0.01, max=0.5, step=0.01, advanced=True),
            Param("gap_frames", "int", "Gap frames", arg="--gap-frames", default=25,
                  min=0, advanced=True,
                  help="MINIMUM frames dropped between the train and validation "
                       "blocks so near-duplicate neighborhoods cannot straddle "
                       "the split. A seconds-sized buffer (Gap seconds, below) "
                       "widens this whenever the wall-clock gap is bigger, which "
                       "at the defaults it always is."),
            Param("gap_seconds", "float", "Gap seconds",
                  arg="--gap-seconds", default=TRAIN_DEFAULTS["gap_seconds"],
                  min=0.0, step=0.5, advanced=True,
                  help="Wall-clock no-man's-land between the train and "
                       "validation blocks, in seconds. At ~20 kept frames per "
                       "second this usually sets the real gap, and Gap frames "
                       "is only the floor."),
            Param("dropout", "float", "Dropout", arg="--dropout", min=0.0, max=0.9,
                  step=0.05, advanced=True),
            Param("tnet", "bool", "Enable T-Nets", arg="--tnet", advanced=True,
                  help="PointNet input + feature transforms. Off by default "
                       "because the data is already canonicalized."),
            Param("aug_intensity_gain", "float", "Augment: brightness gain jitter",
                  arg="--aug-intensity-gain",
                  default=TRAIN_DEFAULTS["aug_intensity_gain"], min=0.0, max=2.0,
                  step=0.05, advanced=True,
                  help="How hard training randomly rescales each neighborhood's "
                       "brightness spread, so the model cannot memorize exact "
                       "reflectivity values. Only bites when the intensity "
                       "channel is ticked."),
            Param("aug_intensity_shift", "float", "Augment: brightness offset jitter",
                  arg="--aug-intensity-shift",
                  default=TRAIN_DEFAULTS["aug_intensity_shift"], min=0.0, max=1.0,
                  step=0.05, advanced=True,
                  help="Random whole-neighborhood brightness nudge, on top of "
                       "the gain jitter — simulates the sensor's exposure-style "
                       "drift between runs. Only bites with the intensity "
                       "channel ticked."),
            Param("aug_thin_min", "float", "Augment: thinning floor",
                  arg="--aug-thin-min",
                  default=TRAIN_DEFAULTS["aug_thin_min"], min=0.1, max=1.0,
                  step=0.05, advanced=True,
                  help="Training randomly drops points to mimic sparser scans; "
                       "this is the fraction of points always kept. 0.5 means a "
                       "neighborhood never loses more than half its points."),
            Param("aug_stray_frac", "float", "Augment: stray returns",
                  arg="--aug-stray-frac", step=0.01,
                  default=TRAIN_DEFAULTS["aug_stray_frac"], min=0.0, max=0.3,
                  help="Fraction of each training frame turned into returns "
                       "that sit on no surface, pushed along their own line of "
                       "sight. Every recording these models learned from was "
                       "made over flat ground the sensor struck steeply, so "
                       "almost every return landed where it should and nothing "
                       "ever taught them a return can simply be wrong. A "
                       "competition arena is 7 m across with the sensor half a "
                       "metre up, so most of it is seen at a grazing angle and "
                       "is full of them. Measured on the lance recording, "
                       "1-2%% of returns inside 1.5 m and 5-7%% further out land "
                       "more than 25 cm off the real surface, so 0.05 is a "
                       "realistic setting. 0 turns it off, which is what every "
                       "existing checkpoint was trained with."),
            Param("aug_phantom_frac", "float", "Augment: phantom clumps",
                  arg="--aug-phantom-frac", step=0.01,
                  default=TRAIN_DEFAULTS["aug_phantom_frac"], min=0.0, max=0.5,
                  help="Sliding-window classifier only. Fraction of training "
                       "samples swapped out for a fake 'phantom clump' marked "
                       "as clear: a loose 3D scatter of a few returns with no "
                       "surface underneath. This is aimed at a different miss "
                       "than the stray-returns setting above. That one nudges a "
                       "few points of an otherwise normal ball; this one builds "
                       "the whole ball, because on the competition arena the "
                       "bad returns clump together hard enough that the "
                       "candidate finder centres a ball on them and asks the "
                       "model whether it is a rock. Inside that ball the clump "
                       "is not obviously wrong - measured, it stands 0.54 m "
                       "tall against a real rock's 0.46 m - and the one thing "
                       "that would give it away, having 7 times fewer returns, "
                       "is thrown away when every ball is cut to 256 points. "
                       "Meanwhile every ball in the eleven training recordings "
                       "is a nearly flat patch under 12 cm tall, so nothing has "
                       "ever shown the model a tall loose scatter and told it "
                       "that is not a rock. 0.08 is the measured starting "
                       "point; 0 turns it off."),
            Param("aug_phantom_extent", "float", "Augment: phantom clump height",
                  arg="--aug-phantom-extent", unit="m", advanced=True,
                  step=0.05, default=TRAIN_DEFAULTS["aug_phantom_extent"],
                  min=0.1, max=2.0,
                  help="How tall a fake phantom clump is, varied half to one "
                       "and a half times this each time. The default matches "
                       "what a real clump measured on the competition arena. "
                       "Only does anything when the setting above is above 0."),
            Param("pos_weight_cap", "float", "Loss: rock-vs-clear weight cap",
                  arg="--pos-weight-cap", advanced=True, min=1.0, max=200.0,
                  step=1.0, default=TRAIN_DEFAULTS["pos_weight_cap"],
                  placeholder="uncapped",
                  help="How much one rock example is allowed to outweigh one "
                       "clear one while training. Left blank the model uses the "
                       "raw imbalance in the data, which is about 94 for a "
                       "model that labels every point and about 4.3 for the "
                       "sliding-window classifier - and that 22x gap exists "
                       "only because the dataset builder throws away 95% of the "
                       "clear spots for one format and none for the other. A "
                       "weight that large buys recall by pushing the model "
                       "until almost everything looks like rock, which is the "
                       "leading suspect for why per-point models put rocks in "
                       "the right order on a new arena but give everything "
                       "near-zero confidence, so nothing lights up on screen. "
                       "Try 10 if a model ranks well but looks blank."),
            Param("aug_stray_reach", "float", "Augment: stray reach",
                  arg="--aug-stray-reach", unit="m", step=0.1,
                  default=TRAIN_DEFAULTS["aug_stray_reach"], min=0.1, max=5.0,
                  advanced=True,
                  help="How far a stray return is thrown. Heavy-tailed, so the "
                       "median lands at about a third of this and a few go "
                       "several times further - which is how the real ones "
                       "behave: a beam skimming the ground reads short or long "
                       "by anything from centimetres to metres. Only matters "
                       "when the stray fraction is above 0."),
            Param("aug_ground_tilt", "float", "Augment: ground tilt",
                  arg="--aug-ground-tilt",
                  default=TRAIN_DEFAULTS["aug_ground_tilt"], min=0.0, max=0.15,
                  step=0.005, advanced=True,
                  help="Segmenter only. Training tips the whole floor at a "
                       "random angle by up to this much rise per metre, so the "
                       "model cannot assume a level arena. 0.03 is about 24 cm "
                       "of rise across the 8 m crop. The floor was flat in "
                       "every recording we trained on, and a lumpy regolith bin "
                       "is what breaks that assumption — on the old model, "
                       "±20 cm of unevenness cost a third of its confidence on "
                       "real rocks. 0 turns it off."),
            Param("no_augment", "bool", "Disable augmentation", arg="--no-augment",
                  advanced=True,
                  help="Turn off every augmentation knob above."),
            *_seg_geometry(),
            *_bev_geometry(),
            Param("seed", "int", "Seed", arg="--seed", default=42, advanced=True),
            _cache_dir(),
            _device(),
            _gpu_fraction(),
            Param("fresh", "bool", "Fresh start", arg="--fresh", advanced=True,
                  help="Ignore an existing last.pt and train from scratch."),
        ],
        long_running=True,
    ),
    Command(
        id="train-compare", bin="rocklabel-train", sub="compare", stage="train",
        tier="tool",
        icon="⊞",
        title="Compare models",
        tagline="Train both architectures on every fold, then render all figures.",
        what="Loops both sliding-window classifiers over every leave-one-run-out "
             "fold, skipping folds that already have test_metrics.json, then "
             f"renders the full figure set into {REPORT_ROOT}/compare/ — "
             "comparison bars, per-fold ROC/PR curves, confusion matrices, "
             "threshold sweeps and summary.json. A non-default channel selection "
             "adds a suffix to that folder (compare_dx-dy-dz), so selections "
             "never overwrite each other's reports.",
        why="The real evaluation. One command produces every number and figure "
            "you would put in front of the team. For 'does this change actually "
            "help' questions, an Ablation sweep is the stronger tool — this one "
            "is for establishing the two-architecture baseline.",
        notes=[
            "This is the long one — it is a full training sweep, not a report.",
            "Already-evaluated folds are skipped, so re-running after adding a "
            "run only trains what is missing (unless you tick Fresh).",
        ],
        params=[
            _models("Which architectures to sweep. Both is the point of the "
                    "command — untick one only to finish a half-done sweep. "
                    "Deliberately no segmenter here: raw scores are only "
                    "comparable between models graded on the same candidate "
                    "spots, and the segmenter gets its fair comparison from "
                    "'Segmenter vs classifier'."),
            _features(),
            Param("epochs", "int", "Epochs", arg="--epochs", default=30, min=1, max=500),
            Param("batch", "int", "Batch size", arg="--batch", default=256, min=8, max=4096),
            Param("lr", "float", "Learning rate", arg="--lr", default=0.001, step=0.0001),
            Param("patience", "int", "Early-stop patience", arg="--patience",
                  default=TRAIN_PATIENCE, min=1),
            *_seg_geometry(),
            *_bev_geometry(),
            Param("weight_decay", "float", "Weight decay", arg="--weight-decay",
                  default=TRAIN_DEFAULTS["weight_decay"], step=0.0001, advanced=True),
            Param("val_frac", "float", "Validation fraction", arg="--val-frac",
                  default=TRAIN_DEFAULTS["val_frac"], min=0.01, max=0.5, step=0.01,
                  advanced=True),
            Param("dropout", "float", "Dropout", arg="--dropout", min=0.0, max=0.9,
                  step=0.05, advanced=True),
            Param("tnet", "bool", "Enable T-Nets", arg="--tnet", advanced=True),
            Param("no_augment", "bool", "Disable augmentation", arg="--no-augment",
                  advanced=True,
                  help="Also see the three augmentation knobs on 'Train one "
                       "fold' — the parser accepts them here too; they are kept "
                       "off this form to keep it readable."),
            Param("seed", "int", "Seed", arg="--seed",
                  default=TRAIN_DEFAULTS["seed"], advanced=True),
            _cache_dir(),
            _device(),
            _gpu_fraction(),
            Param("fresh", "bool", "Retrain everything", arg="--fresh", advanced=True,
                  help="Ignore existing results and redo every fold from scratch."),
        ],
        long_running=True,
    ),
    Command(
        id="train-ablate", bin="rocklabel-train", sub="ablate", stage="train",
        icon="⚖",
        title="Ablation sweep",
        tagline="Settle whether one thing — a channel, a model — actually changes the score.",
        what="Trains a whole set of settings on every leave-one-run-out fold, then "
             "compares them fold by fold rather than as two averages. The built-in "
             "'reflectivity' set covers PointNet and PointNet++ with and without the "
             "reflectivity channel, a run with the reflectivity augmentation switched "
             "off, a reflectivity-only model, and repeats of two settings under "
             "different random seeds. Those repeats are the point of the whole thing: "
             "they show how far apart two runs of the *same* setting land, which is "
             "the only way to know whether a difference between two *different* "
             "settings means anything.",
        why="Reach for this instead of eyeballing two Compare runs. Which recording "
            "gets held out swings the score far more than any channel does, so an "
            "unpaired comparison cannot see an effect this small. The report gives "
            "you a per-fold difference, a win/loss count and a significance test.",
        notes=[
            "One training per setting per held-out run — 121 for 'reflectivity', "
            "110 for 'fullsweep', 48 for 'segdense' — so this is the longest job "
            "in the tool whichever question you pick. Finished folds are skipped, "
            "so a stopped sweep picks up where it left off.",
            f"Every setting gets its own folder under {EXPERIMENTS_ROOT}/<question>/, so two "
            "settings that differ only in, say, an augmentation value never "
            "collide on one folder. Compare's flat <model>_<fold> names would; "
            "it moves an older different-settings run aside rather than share.",
            "Tick 'Report only' to rebuild the tables and figures from whatever "
            "has already finished — safe to do while the sweep is still running.",
        ],
        params=[
            Param("suite", "enum", "Question to settle", arg="--suite",
                  choices=sorted(ABLATION_SUITES), default="reflectivity",
                  help="Which set of settings to run. "
                       + " ".join(f"'{k}': {v['title']} (trains on the "
                                  f"{v['cache']} cache)."
                                  for k, v in ABLATION_SUITES.items())
                       + " Leave the Cache folder blank and each question picks "
                         "its own cache — the settings inside one question are "
                         "only comparable if every one of them saw the same "
                         "frames."),
            Param("arms", "text", "Only these settings", arg="--arms", repeat=True,
                  nargs=True, advanced=True,
                  placeholder="pointnet-geom, pointnet-refl",
                  help="Run only the named settings instead of the whole set. "
                       "Leave empty for all of them, which is the normal case. "
                       "Useful for finishing a sweep that was stopped partway."),
            Param("report_only", "bool", "Report only (no training)",
                  arg="--report-only",
                  help="Skip straight to the figures and tables, built from the "
                       "folds already on disk. Costs seconds and touches nothing."),
            Param("epochs", "int", "Epochs", arg="--epochs", default=30, min=1, max=500,
                  advanced=True),
            Param("batch", "int", "Batch size", arg="--batch", default=256, min=8,
                  max=4096, advanced=True),
            Param("patience", "int", "Early-stop patience", arg="--patience",
                  default=TRAIN_PATIENCE, min=1, advanced=True,
                  help="Stop a fold after this many epochs with no validation "
                       "gain. Keep it long enough for the learning-rate schedule "
                       "to finish, or no fold ever sees its fine-tuning phase."),
            *_seg_geometry(),
            *_bev_geometry(),
            # The rest of the shared training hyperparameters: every one of
            # these passes through to each arm unless the arm's own definition
            # overrides it, which is why they sit behind Advanced.
            Param("lr", "float", "Learning rate", arg="--lr",
                  default=TRAIN_DEFAULTS["lr"], step=0.0001, advanced=True,
                  help="Applies to every arm unless the arm's own definition "
                       "sets one — arm settings win over form values."),
            Param("weight_decay", "float", "Weight decay", arg="--weight-decay",
                  default=TRAIN_DEFAULTS["weight_decay"], step=0.0001, advanced=True),
            Param("val_frac", "float", "Validation fraction", arg="--val-frac",
                  default=TRAIN_DEFAULTS["val_frac"], min=0.01, max=0.5,
                  step=0.01, advanced=True),
            Param("dropout", "float", "Dropout", arg="--dropout", min=0.0,
                  max=0.9, step=0.05, advanced=True),
            Param("tnet", "bool", "Enable T-Nets", arg="--tnet", advanced=True),
            Param("no_augment", "bool", "Disable augmentation", arg="--no-augment",
                  advanced=True),
            Param("seed", "int", "Seed", arg="--seed",
                  default=TRAIN_DEFAULTS["seed"], advanced=True),
            Param("ablate_root", "outdir", "Runs folder", arg="--ablate-root",
                  default=EXPERIMENTS_ROOT, advanced=True,
                  help="Where each setting's trained folds are written, as "
                       "<this folder>/<question>/<setting>/<held-out run>/."),
            _cache_dir(),
            _device(),
            _gpu_fraction(),
            Param("fresh", "bool", "Retrain everything", arg="--fresh", advanced=True,
                  help="Ignore finished folds and redo the whole matrix."),
        ],
        long_running=True,
    ),
    Command(
        id="train-matched", bin="rocklabel-train", sub="matched", stage="train",
        tier="tool",
        icon="⇔",
        title="Segmenter vs classifier",
        tagline="Compare a per-point model against a sliding-window one, fairly.",
        what="Takes a finished sweep that trained both kinds of model and re-scores "
             "them on one shared set of candidate spots. The sliding-window model "
             "already gives each spot a score. The per-point model is asked for one "
             "by taking the strongest rock score it gave to any point sitting within "
             "a few centimetres of that spot. Both then get graded on the same spots "
             "with the same right answers, and you get a per-run table, a win/loss "
             "count and a significance test.",
        why="Because the two kinds of model are normally graded on different things "
            "and their scores cannot be read off one table. A sliding-window model "
            "is graded on candidate spots, about a fifth of which are rock; a "
            "per-point model is graded on individual points, only about one in a "
            "hundred of which are rock. The main score used everywhere in this tool "
            "moves with that proportion, so the per-point model looks far worse than "
            "it is. This puts them on equal footing.",
        notes=[
            "Needs a sweep that has both kinds of model in it - the 'fullsweep' "
            "set does. Runs in seconds and trains nothing, so it is safe to run "
            "while a sweep is still going.",
            "Point --cache-dir at the same cache the sweep was trained on; the "
            "point positions are read from it.",
        ],
        params=[
            Param("suite", "enum", "Which sweep", arg="--suite",
                  choices=sorted(ABLATION_SUITES), default="fullsweep",
                  help="Only a set holding both a per-point model and a "
                       "sliding-window model has anything to compare."),
            Param("cache_dir", "dir", "Cache folder", arg="--cache-dir",
                  default=DEFAULT_CACHE, source="caches",
                  help="The cache the sweep was trained on."),
            Param("ablate_root", "dir", "Runs folder", arg="--ablate-root",
                  default=EXPERIMENTS_ROOT,
                  help="Where that sweep's trained folds live."),
            Param("out", "outdir", "Output directory", arg="--out",
                  help="Where the table and figures land. Defaults to "
                       f"{REPORT_ROOT}/<sweep>/matched."),
            Param("radius", "float", "Match radius", arg="--radius", default=0.15,
                  min=0.01, max=1.0, step=0.01, unit="m", advanced=True,
                  help="How far from a candidate spot a scored point may sit and "
                       "still count as describing it. Candidate spots are the "
                       "centre of a 5 cm cell, so their own points are within about "
                       "4 cm; the default keeps those and the immediate surround."),
            Param("aggregation", "enum", "Combine nearby points by", arg="--aggregation",
                  choices=["max", "mean", "nearest"], default="max", advanced=True,
                  help="How the scores of the points near a spot become one number. "
                       "'max' asks whether the model thinks anything there is rock, "
                       "which is the question the label actually poses."),
        ],
    ),
    Command(
        id="train-reflect", bin="rocklabel-train", sub="reflect", stage="train",
        tier="tool",
        icon="✸",
        title="Reflectivity check",
        tagline="Measure what the brightness channel carries — in seconds, without training.",
        what="Goes at the cached samples directly and scores each labeled "
             "neighborhood several ways: plain average brightness, its spread, the "
             "brightness of the middle versus the outer ring, the brightness of the "
             "tall points versus the low ones, and how well brightness tracks "
             "height. Each is rated on how well it alone tells a rock from clear "
             "ground, run by run, against the same measurements made on height as a "
             "reference. It also reports how far the overall brightness level drifts "
             "between recordings, which decides whether any fixed brightness "
             "threshold could ever transfer.",
        why="Run this before spending hours on an ablation sweep. If the channel is "
            "empty at this level, no model is going to find something in it, and "
            "you have the answer in under a minute instead of overnight.",
        notes=[
            "Needs a built cache (Build cache), nothing else. No GPU, no model.",
            f"Writes {REPORT_ROOT}/reflect/ — four figures plus summary.md.",
        ],
        params=[
            Param("out", "outdir", "Output directory", arg="--out",
                  default=f"{REPORT_ROOT}/reflect",
                  help="Where the figures and tables land."),
            _cache_dir(),
        ],
    ),
    Command(
        id="train-report", bin="rocklabel-train", sub="report", stage="train",
        tier="tool",
        icon="▥",
        title="Regenerate report",
        tagline="Rebuild every figure and table from existing runs — no retraining.",
        what=f"Re-renders {REPORT_ROOT}/compare/ from whatever is already in "
             f"{DEFAULT_RUNS_ROOT}/: comparison figure, ROC/PR curves, confusion "
             "matrices, threshold sweeps, summary.json and summary.md.",
        why="Cheap and safe. Reach for it after deleting a bad fold, or whenever "
            "the figures look stale.",
        params=[
            _models("Which architectures to report on. Every ticked model needs "
                    "a finished run for every fold, or the render fails."),
        ],
    ),
    Command(
        id="train-export", bin="rocklabel-train", sub="export", stage="deploy",
        icon="⇪",
        title="Export checkpoint",
        tagline="Freeze a checkpoint to TorchScript + ONNX with its full preprocessing contract.",
        what="Writes TorchScript and ONNX next to a metadata.json that pins the "
             "entire preprocessing contract — channel semantics, neighborhood "
             "radius, voxel size, config hash, decision threshold — plus a "
             "standalone infer_example.py that needs only torch.",
        why="What you hand to the robot's perception stack. The metadata is the "
            "point: without it the numbers a model was trained on cannot be "
            "reproduced at inference time.",
        params=[
            Param("checkpoint", "path", "Checkpoint", source="checkpoints", required=True,
                  help="A best.pt from a training run."),
            Param("out", "outdir", "Output directory", arg="--out",
                  placeholder="training/exported/<run name>",
                  help="Leave empty for training/exported/<run name>."),
        ],
    ),
    Command(
        id="train-replay", bin="rocklabel-train", sub="replay", stage="deploy",
        tier="tool", gui=True,
        icon="▶",
        title="Model replay",
        tagline="Run a checkpoint over any recording — no labels or dataset needed.",
        what="Rebuilds neighborhoods on the fly with the exact geometry stored in "
             "the checkpoint, scores every candidate center, and browses the run "
             "with confidence coloring and a live threshold slider. Works on "
             "either mcap format, labeled or not.",
        why="The honest test of a checkpoint: a recording it has never seen, with "
            "no dataset preparation in between. This is how myroom5 was checked "
            "as a true holdout.",
        notes=[
            "The region flags are a big speedup on wall- and ceiling-heavy "
            "recordings — the same z band you use live.",
            "Works with a per-point segmenter as well: it scores whole frames "
            "and colors every point instead of one center per ball. Give it "
            "the same thin z band you would use live, or it sees far more "
            "vertical structure than it was trained on and finds nothing.",
            "--dump writes frames, centers and probabilities to an .npz and exits "
            "without a window.",
        ],
        params=[
            _recording(),
            Param("checkpoint", "path", "Checkpoint", arg="--checkpoint",
                  source="checkpoints", required=True),
            *_region_params(),
            Param("stride", "int", "Frame stride", arg="--stride", min=1, advanced=True,
                  help="Keep every Nth frame. Defaults to the training config's."),
            Param("window_s", "float", "Frame window", arg="--window-s", unit="s",
                  min=0, max=2, step=0.05, advanced=True,
                  help="Merge scans into time-window frames first, for native "
                       "recordings. Defaults to the training config's."),
            _config(),
            _device(),
            Param("dump", "outpath", "Dump to npz", arg="--dump", advanced=True,
                  placeholder="probs.npz",
                  help="Write frame/centers/probs and exit without a window."),
        ],
    ),
    Command(
        id="train-view", bin="rocklabel-train", sub="view", stage="train",
        tier="tool", gui=True,
        icon="◫",
        title="Confidence view",
        tagline="Replay a generated dataset colored by model confidence, against ground truth.",
        what="Steps through a dataset run with the model's confidence painted on, "
             "next to the labels it was scored against — so false positives and "
             "false negatives are visible as geometry, not just as a number.",
        why="When a fold's F1 is disappointing, this shows you *which* rocks it "
            "misses and whether the misses share a shape.",
        notes=[
            "Sliding-window classifiers only. This view shows one confidence "
            "per sample center, which is not what a per-point segmenter "
            "produces — pick a _seg checkpoint and it says so and sends you to "
            "Model replay, which does handle them.",
        ],
        params=[
            Param("dataset_dir", "dir", "Dataset", source="datasets", required=True),
            Param("checkpoint", "path", "Checkpoint", arg="--checkpoint",
                  source="checkpoints", required=True),
            Param("run", "text", "Run id", arg="--run",
                  help="Only needed if the dataset holds several runs."),
            Param("frame", "int", "Start frame", arg="--frame", min=0),
            _device(),
        ],
    ),
]

COMMANDS_BY_ID = {c.id: c for c in COMMANDS}


# --------------------------------------------------------------------------- #
# argv construction
# --------------------------------------------------------------------------- #
def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


#: Port the live control panel binds by default (rocklabel.live.webui). Kept
#: here as a literal so the preview does not have to import the live stack —
#: test_panel_port_matches_the_cli guards the two against drifting apart.
PANEL_DEFAULT_PORT = 8770


def build_argv(cmd: Command, values: dict, panel_port: int | None = None) -> list[str]:
    """Turn form values into the argv the CLI expects.

    Positionals keep their catalog order and come first (matching every
    rocklabel subparser); flags follow. Blank values are simply omitted, which
    is what makes "leave empty for the default" work end to end.

    For a :attr:`Command.panel` command this also appends ``--web-ui
    --no-browser``: launched from the dashboard you are already in a browser,
    so the controls belong on this page and the Open3D window is better off
    giving the scene its whole screen. ``--no-browser`` because the dashboard
    embeds the panel itself rather than letting the child open a stray tab.

    ``panel_port`` names the port the job will be given; it is only spelled out
    when it differs from the CLI's own default, so the drawer's preview matches
    what actually runs in the ordinary case of one live job at a time.
    """
    argv = [cmd.bin, cmd.sub]
    positional: list[str] = []
    flags: list[str] = []

    for p in cmd.params:
        raw = values.get(p.name)
        if p.kind == "bool":
            if raw:
                flags.append(p.arg)
            continue
        if _is_blank(raw):
            if p.required:
                raise ValueError(f"{p.label} is required")
            continue
        if p.repeat:
            parts = [s.strip() for s in str(raw).split(p.sep) if s.strip()]
            if p.arg is None:
                positional.extend(parts)
            elif p.nargs:
                # nargs="+" style: one flag, many values
                flags.append(p.arg)
                flags.extend(parts)
            else:
                for part in parts:  # repeatable flag: --topic A --topic B
                    flags.extend([p.arg, part])
            continue
        text = str(raw).strip()
        if p.arg is None:
            positional.append(text)
        else:
            flags.extend([p.arg, text])

    argv.extend(positional)
    argv.extend(flags)
    if cmd.panel:
        argv.append("--web-ui")
        if panel_port is not None and panel_port != PANEL_DEFAULT_PORT:
            argv.extend(["--web-port", str(panel_port)])
        argv.append("--no-browser")
    return argv


def quote_argv(argv: list[str]) -> str:
    """Shell-ish rendering for the command preview."""
    import shlex

    return " ".join(shlex.quote(a) for a in argv)


def to_json() -> dict:
    """Serialize the catalog for the browser."""
    from dataclasses import asdict

    return {
        "stages": STAGES,
        "commands": [asdict(c) | {"cli": c.cli} for c in COMMANDS],
    }
