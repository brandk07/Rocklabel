"""Live PointNet inference on the streaming LiDAR data.

**What gets scored — and why it is NOT the accumulated map.** The training
datasets are generated per *scan*: with ``frame_window_s: 0`` each sample
cloud is one raw sensor batch (~20 ms, a few thousand points), and the
model's ``true_counts`` input encodes that density. Feeding the fused
accumulated map (tens of overlapping scans, 10-50x denser, thickened by SLAM
drift) pushes every sample far out of the training distribution — live
predictions started fine and then decayed as the map filled in. So each pass
scores :meth:`IngestEngine.recent_snapshot`: the freshest scan (or a short
window of them), full resolution, exactly the distribution the checkpoint
was trained on. Quality no longer depends on how long the app has been
running.

**Region.** Before sampling, the scan is restricted to a z band + horizontal
radius around the current sensor pose (GUI-tunable; walls/ceiling never
reach the model), intersected with the checkpoint's own crop box, and two
hard caps (``max_cloud_points``, ``max_centers``) bound memory/latency.

**Persistence.** A single scan only covers the stripe the sensor is
currently sweeping, so scored centers are merged into a rolling prediction
map keyed by the candidate voxel grid (latest probability wins). The viewer
colors points from this map — coverage builds up as you sweep the room, and
re-visiting an area refreshes its predictions. Toggle/clear it from the GUI.

Torch is imported lazily so running without ``--model`` never requires the
training extra.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

#: Points farther than this many candidate-voxel edges from every scored
#: center get no prediction (the viewer dims them instead of coloring).
_MATCH_VOXELS = 3.0


@dataclass
class ScoreSettings:
    """Live-tunable scoring parameters (read fresh by every pass).

    ``z_min``/``z_max`` are relative to the sensor height (like the CLI crop
    flags: sensor 1 m above the floor -> floor band is roughly -1.5..-0.5), or
    to the levelled floor plane when ``floor_relative`` is set — which is the
    robust choice, since it does not silently break when the sensor is
    remounted at a different height.
    ``range_max`` is horizontal distance from the sensor; 0 disables it.
    """

    enabled: bool = True
    interval_sec: float = 0.5
    #: Seconds of recent scans scored per pass. 0 = the single latest batch,
    #: which matches how the training data was generated (frame_window_s: 0).
    #: None = resolve from the checkpoint's own frame_window_s on load.
    window_sec: float | None = None
    z_min: float = -3.0
    z_max: float = 1.0
    #: Anchor the z band to the measured floor plane instead of the sensor.
    floor_relative: bool = False
    range_max: float = 8.0
    #: Merge passes into a persistent per-voxel prediction map (latest prob
    #: wins) instead of showing only the current scan's stripe.
    persist: bool = True
    #: Cloud points fed to sampling per pass (random subsample above this).
    max_cloud_points: int = 60_000
    #: Candidate centers scored per pass (random subsample above this).
    max_centers: int = 5_000


class _Result:
    """One immutable scored-center set (swapped atomically under the lock)."""

    def __init__(self, centers: np.ndarray, probs: np.ndarray, match_radius: float) -> None:
        from scipy.spatial import cKDTree

        self.centers = centers
        self.probs = probs
        self.tree = cKDTree(centers) if len(centers) else None
        self.match_radius = match_radius


class LiveScorer:
    """Scores fresh scans with a trained checkpoint and keeps a prediction map.

    Call :meth:`start` after the engine started, :meth:`stop` on shutdown.
    """

    def __init__(
        self,
        checkpoint: str,
        engine,
        device: str | None = None,
        settings: ScoreSettings | None = None,
        batch: int = 512,
    ) -> None:
        import torch  # lazy: only --model runs need the training extra

        from rocklabel.train.models import build_model

        self._torch = torch
        self._engine = engine
        self.settings = settings or ScoreSettings()
        self._batch = int(batch)

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self._tcfg = ck["config"]
        self._gcfg = ck["generator"]
        self.threshold = float(ck.get("threshold", 0.5))
        self.model_name = self._tcfg["model"]
        if self.settings.window_sec is None:
            self.settings.window_sec = float(self._gcfg.get("frame_window_s") or 0.0)
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = build_model(
            self._tcfg["model"],
            tnet=self._tcfg["tnet"],
            dropout=self._tcfg.get("dropout"),
        )
        self._model.load_state_dict(ck["model"])
        self._model.eval().to(self._device)

        self._rng = np.random.default_rng(int(self._gcfg["seed"]))
        self._inten_scale: float | None = None

        #: Persistent prediction map: voxel key -> (center xyz, prob).
        #: Touched ONLY on the scorer thread; the GUI requests a clear via
        #: the flag so no lock is needed around the (possibly large) merge.
        self._map: dict[tuple[int, int, int], tuple[np.ndarray, float]] = {}
        self._clear_requested = False

        self._lock = threading.Lock()
        self._result: _Result | None = None
        self.version = 0  # bumps on every finished pass (viewer cache key)
        self._last_ms = 0.0
        self._last_pass_centers = 0
        self._last_in_region = 0
        self._last_centers_capped = False
        #: ``(scan_z_lo, scan_z_hi, band_lo, band_hi)`` of the last pass that
        #: found nothing in the region, else None.
        self._last_miss: tuple[float, float, float, float] | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="live-scorer", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def clear_map(self) -> None:
        """Drop every remembered prediction (GUI 'Clear predictions')."""
        self._clear_requested = True
        with self._lock:
            self._result = None
            self.version += 1

    # -- scoring loop ------------------------------------------------------- #
    def _probe_intensity_scale(self, inten: np.ndarray) -> float:
        """Bring live intensity into [0, 1] — the range the model trained on
        (mirrors LidarrigScanStream._probe_intensity_scale)."""
        finite = inten[np.isfinite(inten)]
        peak = float(finite.max()) if finite.size else 0.0
        if peak <= 1.5:
            return 1.0
        if peak <= 255.0:
            return 1.0 / 255.0
        return 1.0 / 65535.0  # raw u16 RSSI

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            if self.settings.enabled:
                try:
                    self._score_once()
                except Exception as e:  # never kill the app from the scorer
                    print(f"[rocklabel] live scorer error: {e}", flush=True)
            elapsed = time.perf_counter() - t0
            self._stop.wait(max(0.05, float(self.settings.interval_sec) - elapsed))

    def region_band(self, base: np.ndarray) -> tuple[float, float]:
        """Absolute z band of the scoring region for the current pose."""
        s = self.settings
        anchor = base[2]
        if s.floor_relative:
            floor = getattr(self._engine, "floor_z", lambda: None)()
            if floor is not None:
                anchor = floor
        return anchor + s.z_min, anchor + s.z_max

    def crop_mask(self, pts: np.ndarray, base: np.ndarray) -> np.ndarray:
        """The user region (z band + horizontal radius) ∩ the checkpoint box."""
        s, g = self.settings, self._gcfg
        z_lo, z_hi = self.region_band(base)
        keep = (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
        if s.range_max > 0.0:
            dx = pts[:, 0] - base[0]
            dy = pts[:, 1] - base[1]
            keep &= dx * dx + dy * dy <= s.range_max * s.range_max
        lo = np.array([base[0] - g["crop_backward_m"], base[1] - g["crop_right_m"],
                       base[2] - g["crop_down_m"]])
        hi = np.array([base[0] + g["crop_forward_m"], base[1] + g["crop_left_m"],
                       base[2] + g["crop_up_m"]])
        keep &= ((pts >= lo) & (pts <= hi)).all(axis=1)
        return keep

    def _update_map(self, centers: np.ndarray, probs: np.ndarray) -> _Result:
        """Merge one pass into the per-voxel map; return the merged result.

        Candidate centers snap to the same ``centers_voxel_m`` grid every
        pass, so keying by voxel index dedupes revisits: the newest
        probability for a spot replaces the old one.
        """
        voxel = float(self._gcfg["centers_voxel_m"])
        keys = np.floor(centers / voxel).astype(np.int64)
        for key, c, p in zip(map(tuple, keys), centers, probs):
            self._map[key] = (c, float(p))
        all_centers = np.array([v[0] for v in self._map.values()])
        all_probs = np.array([v[1] for v in self._map.values()], np.float32)
        match_radius = max(_MATCH_VOXELS * voxel, 0.15)
        return _Result(all_centers, all_probs, match_radius)

    def _score_once(self) -> None:
        from rocklabel.neighborhoods import build_inference_samples

        t0 = time.perf_counter()
        s, g = self.settings, self._gcfg
        if self._clear_requested:
            self._clear_requested = False
            self._map.clear()
        pts, inten = self._engine.recent_snapshot(float(s.window_sec or 0.0))
        if pts.shape[0] < int(g["min_neighbors"]):
            return

        base = self._engine.current_pose()[0]
        keep = self.crop_mask(pts, base)
        n_in = int(keep.sum())
        self._last_in_region = n_in
        if n_in < int(g["min_neighbors"]):
            # An empty region is the single most common live-scoring failure
            # and looks identical to "the model found nothing", so record what
            # the band actually was against where the points actually were.
            z_lo, z_hi = self.region_band(base)
            self._last_miss = (
                float(np.percentile(pts[:, 2], 5)), float(np.percentile(pts[:, 2], 95)),
                z_lo, z_hi,
            )
            return
        self._last_miss = None
        xyz = pts[keep].astype(np.float32)
        vals = inten[keep].astype(np.float32)
        if n_in > int(s.max_cloud_points):
            sub = self._rng.choice(n_in, int(s.max_cloud_points), replace=False)
            xyz, vals = xyz[sub], vals[sub]

        vals[~np.isfinite(vals)] = 0.0
        if self._inten_scale is None and np.any(vals > 0):
            self._inten_scale = self._probe_intensity_scale(vals)
        vals = vals * (self._inten_scale or 1.0)

        samples = build_inference_samples(xyz, vals, g, self._rng,
                                          max_centers=int(s.max_centers))
        if samples is None:
            return
        self._last_centers_capped = len(samples["centers_odom"]) >= int(s.max_centers)

        torch = self._torch
        neigh = torch.from_numpy(samples["neighborhoods"])
        cnt = torch.from_numpy(samples["true_counts"].astype(np.int64))
        out = []
        with torch.no_grad():
            for i in range(0, len(neigh), self._batch):
                logits = self._model(
                    neigh[i:i + self._batch].to(self._device),
                    cnt[i:i + self._batch].to(self._device),
                )
                out.append(torch.sigmoid(logits).float().cpu().numpy())
        probs = np.concatenate(out)
        centers = samples["centers_odom"].astype(np.float64)

        # Merge outside the lock (the map can be large); only swap under it.
        if s.persist:
            result = self._update_map(centers, probs)
        else:
            self._map.clear()
            result = _Result(
                centers, probs,
                max(_MATCH_VOXELS * float(g["centers_voxel_m"]), 0.15),
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._result = result
            self._last_ms = elapsed_ms
            self._last_pass_centers = len(probs)
            self.version += 1

    # -- viewer-facing API -------------------------------------------------- #
    @property
    def ready(self) -> bool:
        with self._lock:
            return self._result is not None

    def probs_for(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """``(probs (N,), matched (N,) bool)``: each point's nearest scored
        center's probability; unmatched points have prob 0. None before the
        first completed pass."""
        with self._lock:
            res = self._result
        if res is None or res.tree is None:
            return None
        dist, idx = res.tree.query(
            pts, k=1, distance_upper_bound=res.match_radius, workers=-1
        )
        matched = np.isfinite(dist)
        probs = np.zeros(pts.shape[0], np.float32)
        if matched.any():
            probs[matched] = res.probs[idx[matched]]
        return probs, matched

    def detections(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Centers and probs of the current map (probs >= threshold only)."""
        with self._lock:
            res = self._result
        if res is None:
            return None
        keep = res.probs >= self.threshold
        return res.centers[keep], res.probs[keep]

    def all_probs(self) -> np.ndarray | None:
        """Every scored center's probability, threshold or not.

        :meth:`detections` answers "what counts as rock right now"; this answers
        "what did the model actually say", which is what a distribution over the
        whole map is drawn from — including the centers a higher threshold would
        discard. None before the first completed pass.
        """
        with self._lock:
            res = self._result
        return None if res is None else res.probs

    def status_dict(self) -> dict:
        """The numbers behind :meth:`status`, for callers that render them
        themselves (the web control panel, the viewer's readout cells).

        This is the source of truth; ``status()`` is one rendering of it. The
        viewer used to recover these by parsing the terminal string back apart,
        which broke the moment a line was reworded.
        """
        with self._lock:
            res = self._result
            ms = self._last_ms
            n_pass = self._last_pass_centers
            n_in = self._last_in_region
            capped = self._last_centers_capped
            miss = self._last_miss
        warning = ""
        if miss is not None:
            s_lo, s_hi, b_lo, b_hi = miss
            warning = (f"region empty: scan z {s_lo:+.2f}..{s_hi:+.2f} m, "
                       f"band {b_lo:+.2f}..{b_hi:+.2f} m")
        return {
            "enabled": bool(self.settings.enabled),
            "ready": res is not None,
            "map_centers": int(len(res.probs)) if res is not None else 0,
            "detections": (int((res.probs >= self.threshold).sum())
                           if res is not None else 0),
            "pass_centers": int(n_pass),
            "pass_ms": float(ms),
            "in_region": int(n_in),
            "capped": bool(capped),
            "warning": warning,
            "threshold": float(self.threshold),
            "model_name": self.model_name,
        }

    def status(self) -> str:
        s = self.status_dict()
        if not s["enabled"]:
            return "scoring off"
        hint = f"\n! {s['warning']}" if s["warning"] else ""
        if not s["ready"]:
            return f"warming up… (no pass finished yet){hint}"
        cap_note = " (capped)" if s["capped"] else ""
        return (
            f"map: {s['map_centers']} centers, {s['detections']} >= thr\n"
            f"pass: {s['pass_centers']} scored{cap_note} in {s['pass_ms']:.0f} ms\n"
            f"scan pts in region: {s['in_region']/1e3:.1f}k{hint}"
        )
