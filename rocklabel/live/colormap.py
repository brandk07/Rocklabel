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
