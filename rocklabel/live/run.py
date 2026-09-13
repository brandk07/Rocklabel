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
import json
import time
from pathlib import Path

from rocklabel.live.colormap import clamp_range
from rocklabel.live.config import AppConfig
from rocklabel.live.pipeline import IngestEngine
from rocklabel.live.sources import make_source
from rocklabel.live.surfaces import make_surface_builder


def _frame_key(mcap_name: str) -> str:
    """Recording name reduced to the recordings that share one world frame.

    ``rocklabel selfhits`` writes a cleaned copy that drops points and copies
    every pose, timestamp and topic through untouched, so ``RUN.mcap`` and
    ``RUN.noselfhits.mcap`` are the same scene in the same frame and may share
    a label file's levelling angle. Anything else — a re-solved ``.reslam``
    copy above all, which deliberately changes the poses — must not.
    """
    stem = mcap_name[:-len(".mcap")] if mcap_name.endswith(".mcap") else mcap_name
    if stem.endswith(".noselfhits"):
        stem = stem[: -len(".noselfhits")]
    return stem


def _label_level_for_replay(play_path: str, labels_root: str = "labels") -> dict | None:
    """Return the labelled coordinate frame for an exact recording match.

    Dataset generation deliberately pins every recording to the angle stored in
    its label file. Re-fitting that same recording in the live viewer can differ
    by several degrees; unlike a constant height offset, that rotates the whole
    scene and can make a segmenter miss every rock. An exact ``mcap_file`` match
    is safe to reuse, while no match (the ordinary deployment case) leaves live
    levelling unchanged.
    """
    root = Path(labels_root)
    if not root.is_dir():
        return None
    wanted = _frame_key(Path(play_path).name)
    found: list[dict] = []
    for path in root.rglob("*.labels.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        level = data.get("level")
        name = data.get("mcap_file")
        if isinstance(name, str) and _frame_key(name) == wanted and isinstance(level, dict):
            found.append(level)
    if not found:
        return None
    angles = {(round(float(x.get("roll_deg", 0.0)), 4),
               round(float(x.get("pitch_deg", 0.0)), 4)) for x in found}
    return found[0] if len(angles) == 1 else None


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
        p.add_argument("--speed", type=float, default=1.0, metavar="X",
                       help="--play only: initial playback rate as a multiple of "
                            "real time (default 1.0; below 1 is slow motion, 0 = "
                            "as fast as the engine can fuse). Changeable at any "
                            "time from the transport bar")
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
    p.add_argument("--max-height-above-ground", type=float, metavar="M",
                   help="phantom-point filter: drop a return sitting more than this far "
                        "(m) above the ground directly beneath it (default 0.5). The "
                        "shallowest beam rings graze the floor at 3-6 m and sometimes "
                        "report short, leaving returns hanging in mid-air that pile up "
                        "into fog as sweeps accumulate. On the competition recording the "
                        "default clears 86%% of them and keeps every labelled rock return. "
                        "Raise it if your rocks are taller than 0.5 m, or to keep walls "
                        "and other tall structure in the accumulated view")
    p.add_argument("--floating-cell", type=float, metavar="M",
                   help="phantom-point filter: side length (m) of the column used to "
                        "estimate the local ground (default 1.0). Larger sees the ground "
                        "past a wide obstacle; smaller follows steep terrain more closely")
    p.add_argument("--carve", action="store_true",
                   help="EXPERIMENTAL. Build the accumulated cloud as a persistent "
                        "voxel map where repeated credible free-space observations "
                        "can retract geometry. Each observation keeps its original "
                        "viewpoint and supporting returns protect occupied voxels. "
                        "Turns 'Accum frames' into a no-op; keep opt-in until the "
                        "labelled Lance per-rock audit passes")
    p.add_argument("--carve-voxel", type=float, metavar="M",
                   help="map resolution for --carve in metres (default 0.05). "
                        "Finer keeps more detail and costs more time per fold")
    p.add_argument("--carve-interval", type=float, metavar="S",
                   help="how often --carve folds pooled scans into the map "
                        "(default 0.4 s). The sensor sends ~225 telegrams a "
                        "second and carving each one would stall the view; raise "
                        "this if the display stutters, lower it for a map that "
                        "reacts faster")
    p.add_argument("--carve-evidence-group", type=float, metavar="S",
                   help="seconds of source observations counted as one independent "
                        "support/contradiction group (default 0.05). Separate from "
                        "--carve-interval so scheduling does not redefine evidence")
    p.add_argument("--carve-assumed-pose-uncertainty", type=float, metavar="M",
                   help="explicit positional-uncertainty fallback for carving sources "
                        "that do not report one. Omit to suppress destructive evidence "
                        "from unknown-quality poses; use 0 only when treating recorded "
                        "poses as exact is justified")
    p.add_argument("--carve-confirm-observations", type=int, metavar="N",
                   help="independent supporting observations required to confirm "
                        "a carved-map voxel (default 2)")
    p.add_argument("--carve-tentative-contradictions", type=int, metavar="N",
                   help="independent free-space observations required to remove a "
                        "tentative carved-map voxel (default 2)")
    p.add_argument("--carve-confirmed-contradictions", type=int, metavar="N",
                   help="independent free-space observations required to remove a "
                        "confirmed carved-map voxel (default 3)")
    p.add_argument("--keep-floating", action="store_true",
                   help="turn the phantom-point filter off and keep every return, "
                        "however high above the ground it floats")
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
    p.add_argument("--compare-model", metavar="CHECKPOINT.pt",
                   help="open a SECOND window scoring the same scans with a "
                        "second checkpoint, for an A/B comparison. Both windows "
                        "share one set of settings (region, threshold, display, "
                        "outlines), so the only difference between them is the "
                        "model. Needs --model; with --web-ui you can also open, "
                        "close and re-pick both models while it runs")
    p.add_argument("--device", help="torch device for --model (default: auto)")
    p.add_argument("--score-interval", type=float, default=0.5,
                   help="seconds between live model scoring passes (default 0.5; "
                        "also adjustable in the GUI)")
    p.add_argument("--clear-looked-through", action="store_true",
                   help="forget a remembered detection once later beams have "
                        "passed straight through where it sits and come back "
                        "from further away. Remembered predictions are "
                        "otherwise only ever replaced by scoring that exact "
                        "spot again, so a false one hanging in mid-air stays on "
                        "the map for the rest of the run and seeing the floor "
                        "underneath it does not touch it. Never changes what "
                        "the model says this pass, and never touches space "
                        "nothing has looked through. Measured on the "
                        "competition recording it is safe but small - a few "
                        "percent of the wrongly-claimed ground, never the weakest "
                        "rock, and at most a few percent of one rock's cells "
                        "- because most of the false detections there sit on "
                        "the ground rather than above it. Also a tick-box in "
                        "the GUI and the browser panel")
    p.add_argument("--clear-separation", type=float, default=None, metavar="M",
                   help="with --clear-looked-through, how far above the local "
                        "ground a detection has to stand before clearing will "
                        "consider it at all (default 0.20 m). The rocks "
                        "measured here are 0.10-0.15 m tall, so below about "
                        "0.15 this starts taking the tops off real ones")
    p.add_argument("--clear-free-windows", type=float, default=None, metavar="N",
                   help="with --clear-looked-through, how many separate sweeps "
                        "have to send a beam through a spot and get something "
                        "back from beyond it before the detection there is "
                        "dropped (default 3). A fresh return puts evidence "
                        "back, so an obstacle coming into view again is "
                        "restored rather than lost")
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

    # A labelled replay has one authoritative coordinate frame: the frame its
    # labels and generated training tensors used. CLI level flags below still
    # win when the operator explicitly asks for something else.
    if play:
        labelled = _label_level_for_replay(play)
        if labelled is not None:
            cfg.level.mode = "manual"
            cfg.level.mount_roll_deg = float(labelled.get("roll_deg", 0.0))
            cfg.level.mount_pitch_deg = float(labelled.get("pitch_deg", 0.0))
            print(
                f"[rocklabel] replay frame pinned to labels: "
                f"roll{cfg.level.mount_roll_deg:+.2f}° "
                f"pitch{cfg.level.mount_pitch_deg:+.2f}°",
                flush=True,
            )
        elif (cfg.level.mode or "auto").lower() not in ("off", "manual"):
            # Unpinned, the world angle comes from a few seconds of ground fit
            # at startup. In a big arena that is not enough to see the floor
            # properly — measured on the lance recording it landed anywhere
            # from 0.7° to 10.4° depending only on where playback began — and
            # a frame that disagrees with the one the model trained in tilts
            # every height the model looks at. Say so rather than guessing
            # quietly.
            print(
                f"[rocklabel] WARNING: no label file matches {Path(play).name}, so the "
                "world angle will be guessed from the first few seconds of ground.\n"
                "[rocklabel]          On a large or cluttered site that guess can be "
                "several degrees off, which tilts the whole map.\n"
                "[rocklabel]          Label the recording first, or pass the known "
                "angle with --mount-roll/--mount-pitch.",
                flush=True,
            )

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
    if args.max_height_above_ground is not None:
        cfg.floating.max_height = args.max_height_above_ground
        cfg.floating.enabled = True
    if args.floating_cell is not None:
        cfg.floating.cell_size = args.floating_cell
        cfg.floating.enabled = True
    if args.keep_floating:
        cfg.floating.enabled = False
    if args.carve:
        cfg.display.carve = True
    if args.carve_voxel is not None:
        cfg.display.carve_voxel = args.carve_voxel
    if args.carve_interval is not None:
        cfg.display.carve_interval = args.carve_interval
    if args.carve_evidence_group is not None:
        cfg.display.carve_evidence_group = args.carve_evidence_group
    if args.carve_assumed_pose_uncertainty is not None:
        cfg.display.carve_assumed_pose_uncertainty = (
            args.carve_assumed_pose_uncertainty
        )
    if args.carve_confirm_observations is not None:
        cfg.display.carve_confirm_observations = args.carve_confirm_observations
    if args.carve_tentative_contradictions is not None:
        cfg.display.carve_tentative_contradictions = args.carve_tentative_contradictions
    if args.carve_confirmed_contradictions is not None:
        cfg.display.carve_confirmed_contradictions = args.carve_confirmed_contradictions
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


def _build_engine(cfg: AppConfig, play_path: str | None,
                  speed: float = 1.0) -> IngestEngine:
    if play_path:
        from rocklabel.live.recording import McapReplaySource

        source = McapReplaySource(play_path, speed=speed)
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
        if getattr(args, "compare_model", None):
            raise SystemExit(
                "--compare-model is the SECOND model: give the first with "
                "--model CHECKPOINT.pt too")
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
        clear_looked_through=bool(args.clear_looked_through),
    )
    for flag, field in (("clear_separation", "clear_separation_m"),
                        ("clear_free_windows", "clear_free_windows")):
        value = getattr(args, flag, None)
        if value is not None:
            setattr(settings, field, float(value))
    scorer = LiveScorer(args.model, engine, device=args.device, settings=settings)
    win = float(settings.window_sec or 0.0)
    anchor = "floor" if floor_rel else "sensor"
    print(f"[rocklabel] live model: {scorer.model_name} "
          f"(threshold {scorer.threshold:.2f}; scoring z "
          f"{settings.z_min:+.1f}..{settings.z_max:+.1f} m rel. {anchor}, "
          f"range {settings.range_max:g} m, every {settings.interval_sec:g} s, "
          f"scan window {win:g} s)", flush=True)
    return scorer


def _build_comparison(args: argparse.Namespace, cfg: AppConfig,
                      engine: IngestEngine, scorer, viz=None):
    """The session's model slots, with slot b pre-selected from the CLI.

    Always built when there is a model, even without ``--compare-model``: the
    browser panel can open a comparison at any point, and the slots are where
    "which checkpoint is window 1 running" lives either way.
    """
    from rocklabel.live.compare import Comparison

    if scorer is None:
        return None
    comparison = Comparison(cfg, engine, scorer=scorer, viz=viz,
                            device=args.device)
    wanted = getattr(args, "compare_model", None)
    if wanted:
        comparison.selected_b = wanted
    return comparison


def _start_web_ui(args: argparse.Namespace, cfg: AppConfig, engine: IngestEngine,
                  scorer, viz=None, comparison=None):
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
    controller = LiveController(cfg, engine, scorer=scorer, viz=viz,
                                comparison=comparison)
    url = start_server(controller, host=args.web_host, port=args.web_port,
                       open_browser=not args.no_browser)
    print(f"[rocklabel] control panel: {url}", flush=True)
    return controller


def _run_headless(cfg: AppConfig, args: argparse.Namespace, play_path: str | None) -> None:
    engine = _build_engine(cfg, play_path, getattr(args, "speed", 1.0))
    scorer = _build_scorer(args, engine)
    comparison = _build_comparison(args, cfg, engine, scorer)
    _start_web_ui(args, cfg, engine, scorer, comparison=comparison)
    engine.start()
    if scorer is not None:
        scorer.start()
    if comparison is not None and comparison.selected_b:
        # No window to draw it in, but the second model still scores and its
        # numbers still reach the panel — which is the whole comparison when
        # you are on the other end of an SSH session.
        comparison.open_comparison()
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
            if comparison is not None and comparison.open:
                # The second model's numbers are the whole comparison when
                # there is no window to look at.
                extra += f"\n[rocklabel] compare | {comparison.scorer_b.status()}"
            if engine.carving:
                backlog = engine.carving_backlog()
                if backlog is not None:
                    queued, high, waits, work_ms = backlog
                    extra += (f" | carve queue {queued}/{high}"
                              f" | map {work_ms:.0f} ms"
                              + (f" | waits {waits}" if waits else ""))
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
        if comparison is not None:
            comparison.close()
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

    engine = _build_engine(cfg, play, getattr(args, "speed", 1.0))
    scorer = _build_scorer(args, engine)
    viz = VizApp(cfg, engine, scorer=scorer, web_ui=bool(args.web_ui))
    comparison = _build_comparison(args, cfg, engine, scorer, viz=viz)
    _start_web_ui(args, cfg, engine, scorer, viz=viz, comparison=comparison)
    if comparison is not None and comparison.selected_b:
        # Deferred until the window exists: the second window is created on the
        # GUI thread, and there is no GUI thread until run() starts the loop.
        viz.on_ready = comparison.open_comparison
    viz.run()
