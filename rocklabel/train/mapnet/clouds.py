"""Every in-band return of a recording, with its time and sensor origin.

Decoding a recording is the slow part, so it happens once per recording and the
result is kept as a flat ``.npz``: points in the levelled world frame (pinned to
the label file's own levelling, exactly as ``mapeval`` does), the time each was
measured, and the index of the scan it came from. Training cuts time prefixes
and crops out of this; grading replays it in order.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .grid import Z_HI, Z_LO, normalise_brightness

#: Horizontal range from the robot base kept per scan, metres. The rig's own.
MAX_RANGE_M = 8.0
#: Side of the square tiles a cloud is bucketed into for fast cropping.
TILE_M = 1.0


def _estimate_floor(recording: str, cfg: dict) -> float:
    """Floor height for a recording levelling could not measure: the median
    height of returns 1-3 m from the robot, over a sample of scans."""
    from ...recording.pipeline import ScanStream
    zs = []
    for k, s in enumerate(ScanStream(recording, cfg, stride=20, progress=False)):
        d = np.linalg.norm(s.xyz_odom[:, :2] - s.T_odom_base[:2, 3], axis=1)
        near = (d > 1.0) & (d < 3.0) & (s.xyz_odom[:, 2] < s.T_odom_base[2, 3])
        if near.any():
            zs.append(s.xyz_odom[near, 2])
        if k >= 300:
            break
    if not zs:
        raise SystemExit(f"{recording}: no returns near the robot to estimate a floor from")
    z = np.concatenate(zs)
    lo = np.percentile(z, 5)
    # The floor is the densest height in the lower part of what the robot sees.
    hist, edges = np.histogram(z[z < lo + 0.5], bins=100)
    return float((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2)


def dump_cloud(recording: str, labels_path: str | None, out: str,
               floor_z: float | None = None) -> dict:
    """Decode ``recording`` and write its in-band returns to ``out``."""
    from ...config import load_config
    from ...geometry.leveling import check_level_match, level_record, pin_level_to_labels
    from ...labels import load_labels
    from ...recording.pipeline import ScanStream

    labels = load_labels(labels_path) if labels_path else None
    cfg = load_config(None)
    if labels is not None:
        cfg = pin_level_to_labels(cfg, labels.level)
    stream = ScanStream(recording, cfg, stride=1, progress=False)
    if labels is not None:
        check_level_match(labels.level, level_record(stream), labels_path)
    floor = floor_z if floor_z is not None else getattr(
        getattr(stream, "solution", None), "floor_z", None)
    if floor is None and labels is not None and labels.level:
        floor = labels.level.get("floor_z")
    if floor is None:
        floor = _estimate_floor(recording, cfg)
        print(f"no measured floor; estimated {floor:.3f} m from returns near the robot",
              flush=True)
    floor = float(floor)
    xs, bs, ts, owner = [], [], [], []
    origins, bases, times = [], [], []
    t0, first = time.monotonic(), None
    has_intensity = True
    for s in stream:
        if first is None:
            first = float(s.time_s)
        base = s.T_odom_base[:3, 3]
        z = s.xyz_odom[:, 2]
        d = s.xyz_odom[:, :2] - base[:2]
        keep = ((z >= floor + Z_LO) & (z <= floor + Z_HI)
                & ((d * d).sum(1) <= MAX_RANGE_M ** 2))
        origins.append(s.T_odom_lidar[:3, 3])
        bases.append(base)
        times.append(float(s.time_s) - first)
        if keep.any():
            n = int(keep.sum())
            xs.append(s.xyz_odom[keep].astype(np.float32))
            bs.append(s.intensity[keep].astype(np.float32))
            ts.append(np.full(n, times[-1], np.float32))
            owner.append(np.full(n, len(origins) - 1, np.int32))
    counters = getattr(stream, "counters", None)
    if counters is not None and counters.intensity_available is False:
        has_intensity = False
    inten = np.concatenate(bs)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, xyz=np.concatenate(xs), intensity=inten.astype(np.float16),
             t=np.concatenate(ts), scan=np.concatenate(owner),
             origins=np.asarray(origins, np.float32), bases=np.asarray(bases, np.float32),
             scan_t=np.asarray(times, np.float64), floor_z=floor,
             has_intensity=has_intensity,
             recording=os.path.abspath(recording),
             labels=os.path.abspath(labels_path) if labels_path else "")
    info = {"recording": os.path.basename(recording), "scans": len(origins),
            "points": int(len(inten)), "duration_s": round(times[-1], 1),
            "wall_s": round(time.monotonic() - t0, 1)}
    print(f"cloud: {info}", flush=True)
    return info


class Cloud:
    """A dumped recording, loaded, with brightness normalised and tile-indexed."""

    def __init__(self, path: str, labels_path: str | None = None):
        with np.load(path, allow_pickle=False) as z:
            self.xyz = z["xyz"]
            inten = z["intensity"].astype(np.float32)
            self.t = z["t"]
            self.floor_z = float(z["floor_z"])
            self.bases = z["bases"]
            self.scan_t = z["scan_t"]
            self.has_bright = bool(z["has_intensity"]) if "has_intensity" in z.files else True
            stored = str(z["labels"]) if "labels" in z.files else ""
        self.name = os.path.basename(path).rsplit(".", 1)[0]
        self.brightness = normalise_brightness(inten) if self.has_bright else None
        self.duration = float(self.scan_t[-1])
        self.labels_path = labels_path or stored or None
        self.labels = None
        if self.labels_path:
            from ...labels import load_labels
            self.labels = load_labels(self.labels_path)
        # Points are in time order; a tile index keeps that order inside every
        # tile, so a time prefix of a tile is a slice.
        tx = np.floor(self.xyz[:, 0] / TILE_M).astype(np.int64)
        ty = np.floor(self.xyz[:, 1] / TILE_M).astype(np.int64)
        self.tile_lo = np.array([tx.min(), ty.min()])
        tw = int(tx.max() - tx.min() + 1)
        key = (ty - ty.min()) * tw + (tx - tx.min())
        order = np.argsort(key, kind="stable")
        self._order = order
        self._tw = tw
        counts = np.bincount(key, minlength=tw * int(ty.max() - ty.min() + 1))
        self._starts = np.r_[0, np.cumsum(counts)]

    def select(self, center, half_m: float, t_max: float | None = None,
               ) -> np.ndarray:
        """Indices of points within a square around ``center`` (and, with
        ``t_max``, measured no later than it)."""
        cx, cy = center
        lo = np.floor((np.array([cx, cy]) - half_m) / TILE_M).astype(int) - self.tile_lo
        hi = np.floor((np.array([cx, cy]) + half_m) / TILE_M).astype(int) - self.tile_lo
        th = (len(self._starts) - 1) // self._tw
        parts = []
        for ty in range(max(lo[1], 0), min(hi[1], th - 1) + 1):
            a = ty * self._tw + max(lo[0], 0)
            b = ty * self._tw + min(hi[0], self._tw - 1)
            if b < a:
                continue
            parts.append(self._order[self._starts[a]:self._starts[b + 1]])
        if not parts:
            return np.zeros(0, np.int64)
        idx = np.concatenate(parts)
        p = self.xyz[idx]
        keep = ((np.abs(p[:, 0] - cx) <= half_m) & (np.abs(p[:, 1] - cy) <= half_m))
        if t_max is not None:
            keep &= self.t[idx] <= t_max
        return idx[keep]


def add_arguments(ap) -> None:
    ap.add_argument("--recording", required=True, help="the .mcap to decode")
    ap.add_argument("--labels", default=None,
                    help="its label JSON; pins the levelling to the one the labels "
                         "were drawn in, and is remembered for training and grading")
    ap.add_argument("--out", required=True,
                    help="where to write the cloud (.npz); training refers to it by "
                         "file name without the extension")
    ap.add_argument("--floor-z", type=float, default=None,
                    help="floor height in metres, when levelling cannot measure one "
                         "(default: estimated from returns near the robot)")


def run(args) -> int:
    dump_cloud(args.recording, args.labels, args.out, args.floor_z)
    return 0
