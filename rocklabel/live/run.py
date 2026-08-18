"""Runner for `rocklabel record` / `rocklabel live` (adapted from lidarrig's
__main__).

Both commands drive the same live pipeline (source -> SLAM/IMU -> 2.5D fuse ->
Open3D viewer); they differ only in defaults:

* ``rocklabel record [OUT.mcap]`` — recording starts immediately (S toggles).
* ``rocklabel live`` — view only; S starts a recording, ``--play`` replays an
  existing mcap through the same pipeline, and ``--model best.pt`` colors the
  live points by the trained model's rock probability (V cycles height →
  reflectivity → model).

``--web-ui`` adds a browser control panel (:mod:`rocklabel.live.webui`) served
from this process and drops the Open3D window's docked settings panel, so the
scene fills one monitor while the controls sit on another. It also works with
``--headless``, where there is no window to control at all.
"""

from __future__ import annotations

import argparse
import time

from rocklabel.live.colormap import clamp_range
from rocklabel.live.config import AppConfig
from rocklabel.live.pipeline import IngestEngine
from rocklabel.live.sources import make_source
from rocklabel.live.surfaces import make_surface_builder


def add_live_args(p: argparse.ArgumentParser, record_cmd: bool) -> None:
    """Attach the shared live-pipeline flags to a rocklabel subparser."""
    if record_cmd:
        p.add_argument("out_pos", nargs="?", metavar="OUT.mcap",
                       help="recording path (default: recordings/lidar_<timestamp>.mcap)")
    else:
        p.add_argument("--record", nargs="?", const="", metavar="PATH",
                       help="also record from launch (S key toggles it live anyway)")
        p.add_argument("--play", metavar="FILE.mcap",
                       help="replay a recording through the live pipeline (transport "
                            "bar with play/pause + seek) instead of a live source")
    p.add_argument("--source", choices=["sim", "udp"],
                   help="data source: 'sim' (synthetic terrain, default) or "
                        "'udp' (live SICK multiScan)")
    p.add_argument("--rig-config", metavar="RIG.yaml",
                   help="live-rig YAML config (grid/crop/slam/display...); this is "
                        "NOT the offline rocklabel config - omit for defaults")
    p.add_argument("--sensor-ip", help="sensor IP for the UDP source")
    p.add_argument("--udp-port", type=int, help="UDP port for Compact data (default 2115)")
    p.add_argument("--cell-size", type=float, help="heightmap cell size in meters")
    p.add_argument("--z-min", type=float,
                   help="crop: keep only z >= this (m, sensor frame); with --model "
                        "this also bounds the scoring region (e.g. sensor 1 m above "
                        "the floor: --z-min -1.5 --z-max -0.5)")
    p.add_argument("--z-max", type=float, help="crop: keep only z <= this (m)")
    p.add_argument("--floor-band", nargs=2, type=float, metavar=("LOW", "HIGH"),
                   help="crop band relative to the MEASURED floor plane instead of the "
                        "sensor (e.g. --floor-band -0.05 0.6 keeps 5 cm below to 60 cm "
                        "above the ground). Rig-height independent - prefer this over "
                        "--z-min/--z-max. Needs levelling (on by default)")
    p.add_argument("--max-range", type=float,
                   help="crop: drop points beyond this range (m) from the sensor; with "
                        "--model this also bounds the scoring region")
    p.add_argument("--no-crop", action="store_true", help="disable the region-of-interest crop")
    p.add_argument("--level", choices=["auto", "imu", "ground", "manual", "off"],
                   help="gravity-level the world frame so a tilt-mounted sensor does not "
                        "tilt the whole map: 'auto' (default) seeds from the IMU and "
                        "refines with a ground-plane fit, 'imu'/'ground' use one source "
                        "only, 'manual' uses --mount-roll/--mount-pitch, 'off' keeps the "
                        "legacy world-frame = sensor-frame-at-startup behaviour")
    p.add_argument("--mount-roll", type=float, metavar="DEG",
                   help="known sensor mount roll (deg, IMU convention); implies --level manual")
    p.add_argument("--mount-pitch", type=float, metavar="DEG",
                   help="known sensor mount pitch, nose-up positive (deg, IMU convention); "
                        "implies --level manual")
    p.add_argument("--no-imu", action="store_true", help="disable IMU de-rotation")
    p.add_argument("--yaw-only", action="store_true",
                   help="apply only the yaw component of the IMU rotation")
    p.add_argument("--no-slam", action="store_true",
                   help="disable scan-to-map odometry (sensor must stay put)")
    p.add_argument("--color-mode",
                   choices=["height", "reflectivity", "reflectivity_stretch", "model"],
                   help="initial point coloring (V key cycles; 'model' needs --model). "
                        "'reflectivity' is the sensor's calibrated full scale, so colors "
                        "compare across frames; 'reflectivity_stretch' spreads each "
                        "frame's own percentile range across the ramp, which is what "
                        "makes rock/ground contrast visible inside an arena")
    p.add_argument("--refl-range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                   help="contrast window for 'reflectivity', as fractions of the "
                        "sensor's full scale (default 0 1 = the whole scale). "
                        "Returns at or above HIGH take the top color, at or below "
                        "LOW the bottom, hard-clamped — narrow it (arena data lives "
                        "around 0.3-0.8) to separate ground from rock while keeping "
                        "colors comparable across frames. Adjustable live with the "
                        "View panel sliders, or A to fit it to what is on screen")
    p.add_argument("--model", metavar="CHECKPOINT.pt",
                   help="trained rocklabel-train checkpoint (best.pt): score the live "
                        "cloud continuously and add the 'model' color mode")
    p.add_argument("--device", help="torch device for --model (default: auto)")
    p.add_argument("--score-interval", type=float, default=0.5,
                   help="seconds between live model scoring passes (default 0.5; "
                        "also adjustable in the GUI)")
    p.add_argument("--web-ui", action="store_true",
                   help="serve a browser control panel for every runtime knob "
                        "(put it on a second monitor) and drop the Open3D "
                        "window's settings panel so the scene fills it; the "
                        "keyboard shortcuts and the replay transport bar stay")
    p.add_argument("--web-port", type=int, default=8770,
                   help="port for --web-ui (default 8770; `dash` uses 8765)")
    p.add_argument("--web-host", default="127.0.0.1",
                   help="interface to bind --web-ui to (default 127.0.0.1; "
                        "anything else exposes the rig's controls)")
    p.add_argument("--no-browser", action="store_true",
                   help="with --web-ui, do not open a browser automatically")
    p.add_argument("--headless", action="store_true",
                   help="run without the Open3D window (prints stats); combine "
                        "with --web-ui to drive the rig entirely from a browser")
    p.add_argument("--duration", type=float, default=0.0,
                   help="in --headless mode, seconds to run (0 = until Ctrl-C)")


def _build_config(args: argparse.Namespace, record_cmd: bool) -> AppConfig:
    """Base config (replay-embedded > --rig-config YAML > defaults) + CLI overrides."""
    play = getattr(args, "play", None)
    cfg = None
    if play:
        from rocklabel.live.recording import read_recording_config

        cfg = read_recording_config(play)
        if cfg is None:
            print(f"[rocklabel] note: {play} has no embedded config; using defaults",
                  flush=True)
    if cfg is None:
        cfg = AppConfig.from_yaml(args.rig_config) if args.rig_config else AppConfig()

    if args.source:
        cfg.source.kind = args.source
    if args.sensor_ip:
        cfg.source.sensor_ip = args.sensor_ip
    if args.udp_port:
        cfg.source.udp_port = args.udp_port
    if args.cell_size:
        cfg.grid.cell_size = args.cell_size
    if args.z_min is not None:
        cfg.crop.z_min = args.z_min
        cfg.crop.enabled = True
    if args.z_max is not None:
        cfg.crop.z_max = args.z_max
        cfg.crop.enabled = True
    if args.floor_band is not None:
        lo, hi = sorted(args.floor_band)
        cfg.crop.z_min, cfg.crop.z_max = lo, hi
        cfg.crop.floor_relative = True
        cfg.crop.enabled = True
    if args.max_range is not None:
        cfg.crop.range_max = args.max_range
        cfg.crop.enabled = True
    if args.no_crop:
        cfg.crop.enabled = False
    if args.mount_roll is not None or args.mount_pitch is not None:
        cfg.level.mode = "manual"
        cfg.level.mount_roll_deg = args.mount_roll or 0.0
        cfg.level.mount_pitch_deg = args.mount_pitch or 0.0
    if args.level:
        cfg.level.mode = args.level
    if args.no_imu:
        cfg.motion.use_imu = False
    if args.yaw_only:
        cfg.motion.yaw_only = True
    if args.no_slam:
        cfg.slam.enabled = False
    if args.color_mode:
        cfg.display.color_mode = args.color_mode
    if getattr(args, "refl_range", None):
        cfg.display.reflectivity_range = clamp_range(*args.refl_range)

    if record_cmd:
        cfg.record.autostart = True
        if args.out_pos:
            cfg.record.path = args.out_pos
    else:
        if args.record is not None:
            cfg.record.autostart = True
            if args.record:
                cfg.record.path = args.record
    if play:
        # Replaying: batches are already world-frame at the recorded pose, so
        # no motion compensation may run again, and re-recording is disabled.
        cfg.slam.enabled = False
        cfg.motion.use_imu = False
        cfg.record.autostart = False
    return cfg


def _build_engine(cfg: AppConfig, play_path: str | None) -> IngestEngine:
    if play_path:
        from rocklabel.live.recording import McapReplaySource

        source = McapReplaySource(play_path)
    else:
        source = make_source(cfg)
    engine = IngestEngine(source, make_surface_builder(cfg), cfg)
    if play_path:
        # Backward seeks rewind the file and re-fuse from the start; the reset
        # runs on the ingest thread, safely between batches.
        source.on_rewind = engine.reset_surface
    return engine


def _build_scorer(args: argparse.Namespace, engine: IngestEngine):
    if not args.model:
        if args.color_mode == "model":
            raise SystemExit("--color-mode model requires --model CHECKPOINT.pt")
        return None
    try:
        from rocklabel.live.scoring import LiveScorer, ScoreSettings
    except ImportError as e:
        raise SystemExit(
            f"--model needs the training extra (pip install -e '.[train]'): {e}"
        )
    # The crop flags double as the scoring-region defaults: the z band /
    # radius that matters for fusion is the same band the rocks live in.
    if args.floor_band is not None:
        lo, hi = sorted(args.floor_band)
        z_min, z_max, floor_rel = lo, hi, True
    else:
        z_min = args.z_min if args.z_min is not None else -3.0
        z_max = args.z_max if args.z_max is not None else 1.0
        floor_rel = False
    settings = ScoreSettings(
        interval_sec=args.score_interval,
        z_min=z_min,
        z_max=z_max,
        floor_relative=floor_rel,
        range_max=args.max_range if args.max_range is not None else 8.0,
    )
    scorer = LiveScorer(args.model, engine, device=args.device, settings=settings)
    win = float(settings.window_sec or 0.0)
    anchor = "floor" if floor_rel else "sensor"
    print(f"[rocklabel] live model: {scorer.model_name} "
          f"(threshold {scorer.threshold:.2f}; scoring z "
          f"{settings.z_min:+.1f}..{settings.z_max:+.1f} m rel. {anchor}, "
          f"range {settings.range_max:g} m, every {settings.interval_sec:g} s, "
          f"scan window {win:g} s)", flush=True)
    return scorer


def _start_web_ui(args: argparse.Namespace, cfg: AppConfig, engine: IngestEngine,
                  scorer, viz=None):
    """Serve the browser control panel; returns the controller (or None).

    Started before the viewer's event loop so the URL is on screen while the
    window is still coming up.
    """
    if not getattr(args, "web_ui", False):
        return None
    try:
        from rocklabel.live.webui import LiveController, start_server
    except ImportError as e:
        raise SystemExit(
            f"--web-ui needs Flask (pip install -e '.[dash]'): {e}"
        )
    controller = LiveController(cfg, engine, scorer=scorer, viz=viz)
    url = start_server(controller, host=args.web_host, port=args.web_port,
                       open_browser=not args.no_browser)
    print(f"[rocklabel] control panel: {url}", flush=True)
    return controller


def _run_headless(cfg: AppConfig, args: argparse.Namespace, play_path: str | None) -> None:
    engine = _build_engine(cfg, play_path)
    scorer = _build_scorer(args, engine)
    _start_web_ui(args, cfg, engine, scorer)
    engine.start()
    if scorer is not None:
        scorer.start()
    if cfg.record.autostart:
        print(f"[rocklabel] recording -> {engine.start_recording()}", flush=True)
    what = play_path if play_path else f"source={cfg.source.kind}"
    print(f"[rocklabel] headless: {what}  (Ctrl-C to stop)", flush=True)
    t0 = time.perf_counter()
    try:
        while True:
            time.sleep(1.0)
            s = engine.stats
            extra = f" | {scorer.status()}" if scorer is not None else ""
            print(
                f"[rocklabel] {s.points_per_sec()/1e3:6.1f}k pts/s | "
                f"cells occupied: {s.cells_occupied:6d} | "
                f"batches: {s.batches_total:6d} | "
                f"pose: {engine.pose_status()}{extra}",
                flush=True,
            )
            if args.duration > 0 and (time.perf_counter() - t0) >= args.duration:
                break
            if play_path and engine.source.finished:
                print("[rocklabel] replay finished.", flush=True)
                break
    except KeyboardInterrupt:
        print("\n[rocklabel] stopping…", flush=True)
    finally:
        if scorer is not None:
            scorer.stop()
        path = engine.stop_recording()
        if path:
            print(f"[rocklabel] recording saved: {path}", flush=True)
        engine.stop()


def run_live(args: argparse.Namespace, record_cmd: bool) -> None:
    """Entry point for both `rocklabel record` and `rocklabel live`."""
    play = getattr(args, "play", None)
    cfg = _build_config(args, record_cmd)
    if args.headless:
        _run_headless(cfg, args, play)
        return
    from rocklabel.live.viz import VizApp  # deferred so --headless needs no GUI

    engine = _build_engine(cfg, play)
    scorer = _build_scorer(args, engine)
    viz = VizApp(cfg, engine, scorer=scorer, web_ui=bool(args.web_ui))
    _start_web_ui(args, cfg, engine, scorer, viz=viz)
    viz.run()
