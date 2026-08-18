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

#: The whole scale — the window a fixed-scale reflectivity view starts with.
FULL_RANGE = (0.0, 1.0)
#: Narrowest window the contrast control may collapse to. Below this the ramp
#: is a step function and the readout stops meaning anything.
MIN_RANGE_SPAN = 0.005


def clamp_range(lo: float, hi: float) -> tuple[float, float]:
    """Sanitize a ``(lo, hi)`` reflectivity window in fraction-of-full-scale.

    Both ends are pinned into ``[0, 1]`` and forced at least
    :data:`MIN_RANGE_SPAN` apart, so a slider dragged past its partner pushes
    it rather than inverting the ramp or dividing by zero.
    """
    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, 0.0, 1.0))
    if hi - lo < MIN_RANGE_SPAN:
        if hi >= 1.0 - MIN_RANGE_SPAN:  # parked at the top: push lo down
            lo = 1.0 - MIN_RANGE_SPAN
            hi = 1.0
        else:
            hi = lo + MIN_RANGE_SPAN
    return lo, hi


def move_range_end(window: tuple[float, float], end: str,
                   value: float) -> tuple[float, float]:
    """Move one end of a ``(lo, hi)`` window, pushing the other out of the way.

    Two sliders edit one window, so they have to agree on what happens when
    they meet. The end being dragged is the one the operator is looking at, so
    it lands where they put it and the *other* end yields — the opposite
    (snapping the dragged handle back) reads as the control fighting you.
    ``end`` is ``"lo"`` or ``"hi"``.
    """
    lo, hi = window
    if end == "lo":
        lo = float(np.clip(value, 0.0, 1.0))
        hi = max(hi, lo + MIN_RANGE_SPAN)
    else:
        hi = float(np.clip(value, 0.0, 1.0))
        lo = min(lo, hi - MIN_RANGE_SPAN)
    return clamp_range(lo, hi)


def percentile_range(inten: np.ndarray, pct: tuple[float, float] = (5.0, 95.0),
                     full_scale: float = RSSI_FULL_SCALE
                     ) -> tuple[float, float] | None:
    """The window ``pct`` of these returns occupy, in fraction of full scale.

    This is what the stretch mode computes per frame, handed back as two
    numbers instead — the "auto-fit" of the manual window, after which the
    operator nudges the ends by hand and the colors stop moving underfoot.
    ``None`` when there is nothing finite to measure.
    """
    v = np.asarray(inten, dtype=np.float64)
    good = v[np.isfinite(v)]
    if good.size == 0:
        return None
    lo, hi = np.percentile(good, list(pct))
    return clamp_range(lo / full_scale, hi / full_scale)


def reflectivity_values(inten: np.ndarray, finite: np.ndarray | None = None,
                        stretch: bool = False,
                        pct: tuple[float, float] = (5.0, 95.0),
                        limits: tuple[float, float] | None = None,
                        full_scale: float = RSSI_FULL_SCALE) -> np.ndarray:
    """RSSI counts -> ``[0, 1]`` for coloring. Non-finite entries become 0.5.

    Two scales, because they answer different questions and neither one wins:

    *Fixed* (``stretch=False``) divides by the sensor's full scale, then maps
    the ``limits`` window — ``(lo, hi)`` as fractions of that scale — onto the
    whole ramp with a **hard clamp** at both ends: everything at or above
    ``hi`` takes the top color, everything at or below ``lo`` the bottom. The
    window defaults to the entire scale, which is the plain calibrated view.
    Because the ends are absolute counts rather than anything the frame
    contributes, identical materials keep identical colors between frames and
    retroreflectors stay pinned at the top — the reason per-frame scaling was
    dropped once already. Narrowing the window is how you get contrast without
    giving that up: an arena whose returns all sit near 0.5 of full scale is
    one flat color across the entire scale and a readable rock/ground split
    across ``(0.45, 0.55)``.

    *Stretched* does the same thing but re-derives the window from each
    frame's own percentiles, so it needs no tuning and costs comparability:
    the colors mean something different every frame, and a rock that leaves
    the view repaints everything left behind. Prefer it to *find* a window
    (see :func:`percentile_range`), not to live in.

    ``full_scale`` is what the input counts as "1.0": the sensor's u16 scale
    for live RSSI, or 1.0 for the offline path, which normalizes intensity at
    decode. Either way ``limits`` are in the same fraction-of-full-scale units,
    so one slider position means one thing in every viewer.
    """
    v = np.asarray(inten, dtype=np.float64)
    if finite is None:
        finite = np.isfinite(v)
    out = np.full(v.shape, 0.5)
    if not np.any(finite):
        return out
    good = v[finite] / max(float(full_scale), 1e-9)
    if stretch:
        lo, hi = np.percentile(good, list(pct))
        hi = max(hi, lo + 1e-9)
    else:
        lo, hi = clamp_range(*(limits if limits is not None else FULL_RANGE))
    out[finite] = np.clip((good - lo) / (hi - lo), 0.0, 1.0)
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
