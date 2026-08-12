"""The extra views: an overhead map, a confidence histogram, and trend history.

`/api/state` answers "what are the knobs set to". This module answers "what is
out there" — the payload behind the overhead view and the charts, which is a
different shape, a different size, and wants a different poll rate, so it is a
different endpoint.

Everything here is display data. The overhead terrain goes over the wire as a
quantized 8-bit image rather than a list of points: a 20 m grid at 5 cm is
160,000 cells, and the honest way to send a picture is to send a picture. The
detection list stays as coordinates because it is small, and because the page
draws each one as a mark you can hover.
"""

from __future__ import annotations

import base64
import threading
import time
from collections import deque

import numpy as np

#: Longest side of the overhead raster handed to the browser. 256 keeps a room
#: at better than 5 cm/px on any screen it will be drawn at, and the payload
#: under 90 KB before base64.
BEV_MAX_SIDE = 256
#: Detections drawn on the overhead map. Past this the marks merge into a blob
#: and the page is doing scatter-plot work for pixels nobody can hit; the count
#: is reported alongside so a truncated view says so.
MAX_DETECTIONS = 1500
#: Confidence-histogram resolution. 25 bins over [0, 1] puts a bin edge on every
#: 0.04 of probability — fine enough to see the shape, coarse enough that each
#: bar is a real mark rather than a hairline.
HIST_BINS = 25
#: Trend samples retained, and the floor on the gap between them. At 1 Hz this
#: is five minutes of history, which outlives any single sweep of a room.
HISTORY_LEN = 300
HISTORY_MIN_GAP_SEC = 1.0


def encode_raster(raster) -> dict | None:
    """Quantize a :class:`HeightRaster` to 8-bit and base64 it for the page.

    Level 0 is reserved for "nothing measured here", so the browser can tell
    unmeasured ground from ground at the low end of the height range — with a
    single ramp over 0..255 those two are the same pixel, and the room grows a
    phantom floor everywhere the sensor never looked.
    """
    if raster is None:
        return None
    z = raster.heights
    known = np.isfinite(z)
    if not known.any():
        return None
    lo = float(np.min(z[known]))
    hi = float(np.max(z[known]))
    span = hi - lo
    if span < 1e-6:
        # A perfectly flat surface: put it mid-ramp rather than dividing by ~0.
        levels = np.where(known, 128, 0).astype(np.uint8)
    else:
        scaled = 1.0 + (z - lo) / span * 254.0
        levels = np.where(known, scaled, 0.0)
        levels = np.clip(levels, 0.0, 255.0).astype(np.uint8)
    return {
        "w": int(z.shape[1]),
        "h": int(z.shape[0]),
        "x0": float(raster.x0),
        "y0": float(raster.y0),
        "cell": float(raster.cell),
        "z_min": lo,
        "z_max": hi,
        "data": base64.b64encode(levels.tobytes()).decode("ascii"),
    }


def detections_payload(scorer, limit: int = MAX_DETECTIONS) -> dict:
    """Scored centers above the threshold, as ``[x, y, z, prob]`` rows.

    Thinned by *lowest probability first* when there are too many: if the view
    has to drop marks, the ones worth keeping are the confident ones.
    """
    empty = {"rows": [], "total": 0, "shown": 0}
    if scorer is None:
        return empty
    got = scorer.detections()
    if got is None:
        return empty
    centers, probs = got
    total = int(len(probs))
    if total == 0:
        return empty
    if total > limit:
        keep = np.argpartition(probs, total - limit)[total - limit:]
        centers, probs = centers[keep], probs[keep]
    rows = [[round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2]), 3),
             round(float(p), 4)] for c, p in zip(centers, probs)]
    return {"rows": rows, "total": total, "shown": len(rows)}


def confidence_histogram(scorer, bins: int = HIST_BINS) -> dict | None:
    """Distribution of every scored center's probability, threshold included.

    This is the chart that makes the threshold slider legible: you can see the
    two lobes the model produces and exactly how many centers a given cut keeps,
    instead of nudging the number and counting dots.
    """
    if scorer is None:
        return None
    got = scorer.all_probs()
    if got is None or len(got) == 0:
        return None
    probs = np.asarray(got, dtype=np.float64)
    counts, edges = np.histogram(probs, bins=bins, range=(0.0, 1.0))
    return {
        "counts": [int(c) for c in counts],
        "edges": [round(float(e), 4) for e in edges],
        "threshold": float(scorer.threshold),
        "total": int(probs.size),
        "above": int((probs >= scorer.threshold).sum()),
    }


class History:
    """A rolling record of the numbers worth watching as a trend.

    Sampled from the same snapshot the status readouts are built from, so the
    charts and the readouts can never disagree. Rate-limited to one sample a
    second regardless of how fast the page polls — four samples a second of a
    number that moves once a second is not a longer history, just a wider one.
    """

    #: series id -> (label, unit). The page renders one small chart per entry,
    #: each on its own axis: these have wildly different scales and putting two
    #: of them on one plot would invent a correlation that is not there.
    SERIES = {
        "detections": ("Detections", "centers ≥ threshold"),
        "in_region": ("Points in region", "points per pass"),
        "pass_ms": ("Scoring pass", "ms"),
        "rate": ("Throughput", "points/s"),
    }

    def __init__(self, maxlen: int = HISTORY_LEN) -> None:
        self._lock = threading.Lock()
        self._t: deque[float] = deque(maxlen=maxlen)
        self._series: dict[str, deque[float]] = {
            name: deque(maxlen=maxlen) for name in self.SERIES
        }
        self._last = 0.0
        self._t0 = time.time()

    def sample(self, values: dict[str, float]) -> None:
        now = time.time()
        if now - self._last < HISTORY_MIN_GAP_SEC:
            return
        with self._lock:
            self._last = now
            self._t.append(round(now - self._t0, 2))
            for name, series in self._series.items():
                series.append(float(values.get(name, 0.0)))

    def clear(self) -> None:
        with self._lock:
            self._t.clear()
            for series in self._series.values():
                series.clear()
            # Also drop the rate limit, so an emptied chart starts refilling on
            # the very next poll instead of sitting blank for another second.
            self._last = 0.0

    def payload(self) -> dict:
        with self._lock:
            return {
                "t": list(self._t),
                "series": [
                    {"id": name, "label": label, "unit": unit,
                     "values": list(self._series[name])}
                    for name, (label, unit) in self.SERIES.items()
                ],
            }


__all__ = ["BEV_MAX_SIDE", "HIST_BINS", "MAX_DETECTIONS", "History",
           "confidence_histogram", "detections_payload", "encode_raster"]
