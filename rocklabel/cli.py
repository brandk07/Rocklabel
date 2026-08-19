"""Console entry point: rocklabel {inspect, slam, label, driftcheck, generate,
trim, preview, record, live, dash}.

Recording paths are positional (``rocklabel label run.mcap``); the old
``--mcap``/``--out`` flag spellings still work.

``rocklabel dash`` opens a local web dashboard over everything above - it runs
these same commands as subprocesses and shows the inventory they produce.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import ConfigError, apply_overrides, load_config
from .profiles import DEFAULT_PROFILE, PROFILES, ProfileError, apply_profile


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config file (see config.example.yaml); "
                                    "omit to use built-in defaults")


def _add_mcap_arg(p: argparse.ArgumentParser) -> None:
    """Accept the recording either positionally or as --mcap (legacy)."""
    p.add_argument("mcap_pos", nargs="?", metavar="RECORDING.mcap",
                   help="path to the .mcap recording (ROS 2 bag or native lidarrig)")
    p.add_argument("--mcap", dest="mcap_flag", help=argparse.SUPPRESS)


def _add_level_args(p: argparse.ArgumentParser) -> None:
    """Levelling flags, shared by every command that replays a recording.

    All three must agree on the frame: labels are stored in world coordinates,
    so labelling levelled and generating unlevelled misplaces every rock.
    """
    p.add_argument("--level", choices=["auto", "off", "ground", "manual"],
                   help="undo a tilted sensor mount before anything sees the points: "
                        "'auto' (default) measures it and leaves already-level "
                        "recordings alone, 'ground' insists on a floor fit, "
                        "'manual' uses --mount-roll/--mount-pitch "
                        "(default from config: off)")
    p.add_argument("--mount-roll", type=float, metavar="DEG",
                   help="known sensor mount roll (deg); implies --level manual")
    p.add_argument("--mount-pitch", type=float, metavar="DEG",
                   help="known sensor mount pitch (deg); implies --level manual")


def _apply_level_args(cfg: dict, args: argparse.Namespace) -> dict:
    mode = args.level
    if mode is None and (args.mount_roll is not None or args.mount_pitch is not None):
        mode = "manual"
    return apply_overrides(cfg, {
        "level.mode": mode,
        "level.mount_roll_deg": args.mount_roll,
        "level.mount_pitch_deg": args.mount_pitch,
    })


def _resolve_mcap(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    path = getattr(args, "mcap_pos", None) or getattr(args, "mcap_flag", None)
    if not path:
        parser.error("missing recording path (rocklabel <command> RECORDING.mcap)")
    return path


def default_dataset_dir(profile: str, mcap_path: str) -> str:
    """Where a dataset lands when --out is left off: datasets/<profile>/<run>.

    Putting the profile in the path is the point. A directory called
    ``datasets/VolleyBallTest4.reslam`` says nothing about how its frames were
    cut, and two agents already drew wrong conclusions from exactly that.
    """
    import os.path

    stem = os.path.basename(mcap_path)
    if stem.endswith(".mcap"):
        stem = stem[: -len(".mcap")]
    return os.path.join("datasets", profile, stem)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rocklabel",
        description="Offline LiDAR rock labeling and training-dataset generation. "
                    "Works on ROS 2 bags (PointCloud2 + /tf) and native lidarrig "
                    "recordings (/lidar/frames) alike - the format is auto-detected.",
    )
    parser.add_argument("--version", action="version", version=f"rocklabel {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="print topics, fields, TF frames, and time span of an mcap")
    p.add_argument("mcap", help="path to the .mcap recording")
    _add_config_arg(p)

    from .slam.cli import add_slam_args

    p = sub.add_parser(
        "slam",
        help="re-solve a recording's sensor trajectory offline and write a new "
             "recording; the input file is never modified",
    )
    add_slam_args(p)

    p = sub.add_parser("label", help="interactively label rocks on the fused world-frame cloud")
    _add_mcap_arg(p)
    _add_config_arg(p)
    p.add_argument("--labels", help="label JSON to resume editing (default: <mcap>.labels.json)")
    p.add_argument("--stride", type=int, help="accumulate every Nth scan (default from config: 1)")
    p.add_argument("--z-min", type=float, help="initial lower z clip (meters)")
    p.add_argument("--z-max", type=float, help="initial upper z clip (meters)")
    p.add_argument("--dump-accumulated", metavar="CLOUD.PLY",
                   help="write the fused cloud to a PLY file and exit without opening a window")
    p.add_argument("--fallback-viewer", action="store_true",
                   help="use the legacy pick-then-terminal viewer instead of the GUI")
    _add_level_args(p)

    p = sub.add_parser("driftcheck", help="overlay first/last 10%% of scans around one rock to spot odometry drift")
    _add_mcap_arg(p)
    p.add_argument("--labels", required=True, help="label JSON produced by 'rocklabel label'")
    p.add_argument("--rock-id", type=int, required=True, help="id of the rock to inspect")
    p.add_argument("--report-only", action="store_true",
                   help="print the measured early-vs-late offset and exit without "
                        "opening the 3D window (works over SSH and in scripts)")
    _add_config_arg(p)
    _add_level_args(p)

    p = sub.add_parser("generate", help="write both training-dataset formats from a labeled recording")
    _add_mcap_arg(p)
    p.add_argument("--labels", help="label JSON (default: <mcap>.labels.json)")
    p.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES),
                   help="named way of cutting frames, which is the single "
                        "biggest lever on the resulting dataset; "
                        + "; ".join(f"{k} = {v.title}" for k, v in PROFILES.items())
                        + f" (default: {DEFAULT_PROFILE})")
    _add_config_arg(p)
    p.add_argument("--out", help="dataset output directory (default: "
                                 "datasets/<profile>/<recording name>)")
    _add_level_args(p)

    p = sub.add_parser(
        "trim",
        help="shrink/salvage a recording: keep only LiDAR+TF topics and an optional "
             "time window; also recovers truncated (never-finalized) mcaps",
    )
    _add_mcap_arg(p)
    p.add_argument("--out", required=True, help="output .mcap to write")
    _add_config_arg(p)
    p.add_argument("--start-s", type=float,
                   help="keep messages from this many seconds after the recording starts")
    p.add_argument("--end-s", type=float,
                   help="keep messages up to this many seconds after the recording starts")
    p.add_argument("--topic", action="append", default=[], metavar="TOPIC",
                   help="extra topic to keep (repeatable); LiDAR+TF topics are always kept")
    p.add_argument("--all-topics", action="store_true",
                   help="keep every topic (trim only by time)")

    from .live.run import add_live_args

    p = sub.add_parser(
        "record",
        help="run the live rig (SICK multiScan over UDP, or the simulator) and "
             "record the stream to a native .mcap - recording starts immediately",
    )
    add_live_args(p, record_cmd=True)

    p = sub.add_parser(
        "live",
        help="live rig viewer without auto-recording: S toggles recording, --play "
             "replays an mcap, --model colors points by a trained model's predictions",
    )
    add_live_args(p, record_cmd=False)

    p = sub.add_parser("preview", help="browse the frames of a generated dataset (BEV cells, "
                                       "sample centers, label spheres) with a transport bar")
    p.add_argument("out_pos", nargs="?", metavar="DATASET_DIR",
                   help="dataset directory written by 'rocklabel generate'")
    p.add_argument("--out", dest="out_flag", help=argparse.SUPPRESS)
    p.add_argument("--run", help="run_id (needed only if the dataset has several runs)")
    p.add_argument("--frame", type=int, help="frame index to start at (default: first frame)")
    p.add_argument("--list", action="store_true", help="list available frame indices and exit")

    p = sub.add_parser(
        "dash",
        help="open the web dashboard: run every command from a browser, with "
             "live sensor status and an inventory of recordings/datasets/models",
    )
    p.add_argument("--root", default=".",
                   help="project directory to manage (default: the current one)")
    p.add_argument("--port", type=int, default=8765, help="port to serve on (default 8765)")
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to bind (default 127.0.0.1; the dashboard can "
                        "start processes, so only widen this on a trusted link)")
    p.add_argument("--no-browser", action="store_true",
                   help="do not open a browser window on startup")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "dash":
        # Before load_config: the dashboard manages a whole project directory
        # and has no single offline config of its own.
        try:
            from .dashboard import run_dashboard
        except ImportError as e:
            print(f"error: the dashboard needs Flask (pip install -e '.[dash]'): {e}",
                  file=sys.stderr)
            return 2
        try:
            run_dashboard(root=args.root, host=args.host, port=args.port,
                          open_browser=not args.no_browser)
        except KeyboardInterrupt:
            print("\n[rocklabel] dashboard stopped.")
        return 0

    if args.command == "slam":
        # Before load_config: the solver works on a recording's own frames and
        # poses and never touches the offline topics/crop/sample settings.
        from .slam.cli import run as run_slam
        return run_slam(args)

    cfg_path = getattr(args, "config", None)
    try:
        cfg = load_config(cfg_path)
    except (ConfigError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    from .dataset.generate import ManifestConflict
    from .geometry.leveling import LevelError
    from .recording.mcap_io import McapFormatError
    from .recording.pose import PoseUnavailable

    try:
        if args.command == "inspect":
            from .recording.inspect_cmd import run_inspect
            run_inspect(args.mcap, cfg)
        elif args.command == "label":
            mcap = _resolve_mcap(args, parser)
            cfg = apply_overrides(cfg, {
                "labeler.stride": args.stride,
                "labeler.z_min": args.z_min,
                "labeler.z_max": args.z_max,
            })
            cfg = _apply_level_args(cfg, args)
            from .gui.labeler import run_label
            run_label(mcap, cfg, args.labels, args.stride, args.z_min, args.z_max,
                      dump_accumulated=args.dump_accumulated,
                      fallback_viewer=args.fallback_viewer)
        elif args.command == "driftcheck":
            cfg = _apply_level_args(cfg, args)
            from .gui.driftcheck import run_driftcheck
            run_driftcheck(_resolve_mcap(args, parser), args.labels, args.rock_id, cfg,
                           report_only=args.report_only)
        elif args.command == "generate":
            mcap = _resolve_mcap(args, parser)
            labels = args.labels
            if not labels:
                from .gui.labeler import default_labels_path
                labels = default_labels_path(mcap)
            # Profile last, so it is the final word on the settings it owns
            # even when a --config file also named them.
            cfg = apply_profile(cfg, args.profile)
            cfg = _apply_level_args(cfg, args)
            out = args.out or default_dataset_dir(args.profile, mcap)
            from .dataset.generate import run_generate
            run_generate(mcap, labels, out, cfg, profile=args.profile)
        elif args.command == "trim":
            from .recording.trim import run_trim
            run_trim(_resolve_mcap(args, parser), args.out, cfg, extra_topics=args.topic,
                     start_s=args.start_s, end_s=args.end_s, all_topics=args.all_topics)
        elif args.command in ("record", "live"):
            from .live.run import run_live
            run_live(args, record_cmd=(args.command == "record"))
        elif args.command == "preview":
            out_dir = args.out_pos or args.out_flag
            if not out_dir:
                parser.error("missing dataset directory (rocklabel preview DATASET_DIR)")
            from .gui.preview import run_preview
            run_preview(out_dir, args.run, args.frame, args.list)
    except (ConfigError, ProfileError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (McapFormatError, PoseUnavailable, ManifestConflict, LevelError,
            FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
