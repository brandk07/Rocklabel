"""Command line for the alternative SLAM.

    python -m rocklabel.slam recordings/VolleyBallTest1.mcap
    python -m rocklabel.slam recordings/VolleyBallTest*.mcap --suffix .reslam
    python -m rocklabel.slam recordings/VolleyBallTest1.mcap --score-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from rocklabel.slam.config import AltSlamConfig
from rocklabel.slam.reprocess import reprocess


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rocklabel.slam",
        description="Re-solve a recording's trajectory offline and write a new "
                    "recording. The input file is never modified.",
    )
    p.add_argument("inputs", nargs="+", help="recording(s) to re-solve (.mcap)")
    p.add_argument("-o", "--output",
                   help="output path (only valid with a single input)")
    p.add_argument("--suffix", default=".reslam",
                   help="appended to the stem for auto-named outputs "
                        "(default: .reslam -> Run1.reslam.mcap)")
    p.add_argument("--out-dir", help="write outputs here instead of alongside "
                                     "the inputs")
    p.add_argument("--score-only", action="store_true",
                   help="solve and report the quality numbers, write nothing")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing output file")
    p.add_argument("--quiet", action="store_true", help="no progress bar")

    g = p.add_argument_group("solver")
    d = AltSlamConfig()
    g.add_argument("--passes", type=int, default=d.passes,
                   help=f"solver passes; pass 1 is causal, later passes re-align "
                        f"against the finished map (default: {d.passes})")
    g.add_argument("--voxel", type=float, default=d.voxel_size,
                   help=f"registration voxel size in m (default: {d.voxel_size})")
    g.add_argument("--window", type=float, default=d.window_sec,
                   help=f"seconds pooled per window (default: {d.window_sec})")
    g.add_argument("--iterations", type=int, default=d.iterations,
                   help=f"ICP iterations per window (default: {d.iterations})")
    g.add_argument("--range-max", type=float, default=d.reg_range_max,
                   help=f"furthest points used for alignment, m. Distant "
                        f"structure is what pins down sliding on flat ground "
                        f"(default: {d.reg_range_max})")
    g.add_argument("--degeneracy", type=float, default=d.degeneracy_threshold,
                   help=f"refuse to solve directions weaker than this; 0 "
                        f"disables the flat-ground guard "
                        f"(default: {d.degeneracy_threshold})")
    g.add_argument("--robust-sigma", type=float, default=d.robust_sigma,
                   help=f"outlier scale in m (default: {d.robust_sigma})")
    g.add_argument("--lock-tilt", action="store_true",
                   help="trust the IMU for roll/pitch and only let alignment "
                        "correct heading. Right for a tripod or mast; wrong for "
                        "a hand-swept sensor, where swinging it corrupts the "
                        "IMU's sense of down (that costs about 2x on the "
                        "volleyball recordings)")
    return p


def config_from_args(a) -> AltSlamConfig:
    cfg = AltSlamConfig()
    cfg.passes = a.passes
    cfg.voxel_size = a.voxel
    cfg.window_sec = a.window
    cfg.iterations = a.iterations
    cfg.reg_range_max = a.range_max
    cfg.degeneracy_threshold = a.degeneracy
    cfg.robust_sigma = a.robust_sigma
    cfg.lock_roll_pitch = a.lock_tilt
    return cfg


def output_path(src: str, a) -> str:
    if a.output:
        return a.output
    stem, ext = os.path.splitext(os.path.basename(src))
    directory = a.out_dir if a.out_dir else os.path.dirname(src)
    return os.path.join(directory, f"{stem}{a.suffix}{ext}")


def _progress(quiet: bool):
    if quiet:
        return None
    state = {"last": 0.0}

    def cb(label: str, i: int, n: int):
        now = time.time()
        if i < n and now - state["last"] < 0.2:
            return
        state["last"] = now
        bar = int(30 * i / max(1, n))
        sys.stderr.write(f"\r  {label}: [{'#' * bar}{'.' * (30 - bar)}] {i}/{n}")
        sys.stderr.flush()
        if i >= n:
            sys.stderr.write("\n")

    return cb


def _report(r: dict) -> None:
    print(f"  batches {r['batches']}  duration {r['duration_s']:.1f}s  "
          f"windows {r['windows']}")
    print(f"  registered {r['registered']}  failed {r['failed']}  "
          f"median match {r['match_ratio_median'] * 100:.0f}%")
    print(f"  along-normal residual {r['residual_rmse_mm']:.1f} mm   "
          f"unobserved directions/window {r['suppressed_dirs_mean']:.2f}")
    print(f"  solved path length {r['path_length_m']:.2f} m")
    if "sharpness_after_mm" in r:
        print(f"  surface thickness: {r['sharpness_before_mm']:.1f} mm "
              f"-> {r['sharpness_after_mm']:.1f} mm "
              f"({r.get('improvement_x', float('nan')):.2f}x sharper)")


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.output and len(a.inputs) > 1:
        print("--output takes a single input; use --out-dir/--suffix for many",
              file=sys.stderr)
        return 2
    cfg = config_from_args(a)

    rc = 0
    for src in a.inputs:
        if not os.path.exists(src):
            print(f"{src}: not found", file=sys.stderr)
            rc = 1
            continue
        dst = output_path(src, a)
        if not a.score_only and os.path.exists(dst) and not a.force:
            print(f"{dst}: exists (use --force to overwrite)", file=sys.stderr)
            rc = 1
            continue
        print(f"{os.path.basename(src)}")
        t0 = time.time()
        try:
            r = reprocess(src, dst, cfg, progress=_progress(a.quiet),
                          score=True, write=not a.score_only)
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            rc = 1
            continue
        _report(r)
        if a.score_only:
            print(f"  (score only, nothing written)  {time.time() - t0:.1f}s")
        else:
            print(f"  wrote {dst}  ({time.time() - t0:.1f}s)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
