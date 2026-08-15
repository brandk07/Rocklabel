"""Tiny dependency-free colormaps (no matplotlib) for height coloring.

Kept as pure NumPy so the surface layer can color vertices without importing a
plotting/visualization library.  Colormaps are defined by a handful of RGB
anchor colors and linearly interpolated — accurate enough for a live readout.
"""

from __future__ import annotations

import numpy as np

# RGB anchor stops (evenly spaced in [0, 1]) approximating common colormaps.
_ANCHORS: dict[str, np.ndarray] = {
    "viridis": np.array(
        [
            [0.267, 0.005, 0.329],
            [0.229, 0.322, 0.545],
            [0.128, 0.567, 0.551],
            [0.369, 0.789, 0.383],
            [0.993, 0.906, 0.144],
        ]
    ),
    "turbo": np.array(
        [
            [0.190, 0.072, 0.232],
            [0.275, 0.635, 0.987],
            [0.150, 0.917, 0.545],
            [0.831, 0.900, 0.170],
            [0.980, 0.331, 0.098],
            [0.480, 0.016, 0.011],
        ]
    ),
    "gray": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
}


def apply_colormap(values: np.ndarray, name: str = "viridis") -> np.ndarray:
    """Map ``values`` in ``[0, 1]`` to ``(..., 3)`` RGB in ``[0, 1]``.

    Values are clamped to ``[0, 1]``. Unknown colormap names fall back to gray.
    """
    anchors = _ANCHORS.get(name, _ANCHORS["gray"])
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    stops = np.linspace(0.0, 1.0, len(anchors))
    r = np.interp(v, stops, anchors[:, 0])
    g = np.interp(v, stops, anchors[:, 1])
    b = np.interp(v, stops, anchors[:, 2])
    return np.stack([r, g, b], axis=-1)


#: Full scale of the multiScan's RSSI field (u16 counts).
RSSI_FULL_SCALE = 65535.0


def reflectivity_values(inten: np.ndarray, finite: np.ndarray | None = None,
                        stretch: bool = False,
                        pct: tuple[float, float] = (5.0, 95.0)) -> np.ndarray:
    """RSSI counts -> ``[0, 1]`` for coloring. Non-finite entries become 0.5.

    Two scales, because they answer different questions and neither one wins:

    *Fixed* (``stretch=False``) divides by the sensor's full scale. The RSSI is
    calibrated, so identical materials keep identical colors between frames and
    retroreflectors stay pinned at the top of the ramp. This is the right
    default and the reason per-frame scaling was dropped once already.

    *Stretched* spreads a percentile window of the frame's own returns across
    the whole ramp. Real surfaces sit in a narrow slice of full scale - on
    arena data the middle 98% of returns spans roughly 0.26-0.82 - so on the
    fixed scale a whole arena renders as one flat color and any rock/ground
    reflectivity difference is invisible. Stretching exposes that difference at
    the cost of frame-to-frame comparability, so it is a separate mode rather
    than a change to the default.
    """
    v = np.asarray(inten, dtype=np.float64)
    if finite is None:
        finite = np.isfinite(v)
    out = np.full(v.shape, 0.5)
    if not np.any(finite):
        return out
    good = v[finite]
    if stretch:
        lo, hi = np.percentile(good, list(pct))
        out[finite] = np.clip((good - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    else:
        out[finite] = np.clip(good / RSSI_FULL_SCALE, 0.0, 1.0)
    return out


def normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to ``[0, 1]``; returns 0.5 everywhere if degenerate."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v
    lo = float(np.min(v))
    hi = float(np.max(v))
    if hi - lo < 1e-9:
        return np.full_like(v, 0.5)
    return (v - lo) / (hi - lo)
