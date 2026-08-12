"""A hardware-free synthetic point source.

:class:`SimulatedSource` sweeps a "scan strip" back and forth across a fake
terrain made of a ground plane, two Gaussian hills, a sinusoidal ripple, and a
rectangular step.  Each batch adds Gaussian range noise and a configurable
fraction of gross outliers.  Because the strip sweeps in x over time, you can
literally watch the reconstructed surface fill in and converge — which is what
the end-to-end sanity check relies on.
"""

from __future__ import annotations

import time

import numpy as np

from rocklabel.live.config import AppConfig
from rocklabel.live.sources.base import PointBatch, PointSource


def terrain_height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ground-truth surface height z = f(x, y), fully vectorized.

    The terrain is deterministic (no noise) so tests and the convergence check
    can compare the fused heightmap against it directly.

    Args:
        x: array of x coordinates (m), any shape.
        y: array of y coordinates (m), broadcastable to ``x``.

    Returns:
        Array of true heights (m), same broadcast shape as ``x``/``y``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    z = np.zeros(np.broadcast(x, y).shape, dtype=np.float64)

    # Two Gaussian hills.
    z += 1.5 * np.exp(-((x - 3.0) ** 2 + (y - 2.0) ** 2) / (2.0 * 2.0**2))
    z += 1.0 * np.exp(-((x + 4.0) ** 2 + (y + 3.0) ** 2) / (2.0 * 1.5**2))

    # A gentle sinusoidal ripple.
    z += 0.15 * np.sin(0.8 * x) * np.cos(0.8 * y)

    # A raised rectangular step (a "curb"/platform) in one quadrant.
    step = (x > 5.0) & (x < 9.0) & (y > -8.0) & (y < -2.0)
    z = np.where(step, z + 0.6, z)

    return z


class SimulatedSource(PointSource):
    """Emits point batches sampling :func:`terrain_height` at a realistic rate."""

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        sc = config.source
        self._batch_size = int(sc.sim_batch_size)
        self._noise_std = float(sc.sim_noise_std)
        self._outlier_fraction = float(sc.sim_outlier_fraction)
        self._outlier_magnitude = float(sc.sim_outlier_magnitude)
        self._rng = np.random.default_rng(sc.sim_seed)

        # Target pacing: seconds of data one batch represents.
        pts_per_sec = max(1, int(sc.sim_points_per_sec))
        self._batch_period = self._batch_size / pts_per_sec

        # Sweep bookkeeping: the strip's x position advances each batch.
        gx0, gy0 = config.grid.origin
        gsx, gsy = config.grid.extent
        self._x_min, self._x_max = gx0, gx0 + gsx
        self._y_min, self._y_max = gy0, gy0 + gsy
        self._sweep_x = self._x_min
        # Advance the strip by a fraction of the extent per batch so a full
        # left-to-right pass takes ~150 batches (a couple of seconds of sweep).
        self._sweep_step = gsx / 150.0
        self._strip_width = max(gsx / 40.0, config.grid.cell_size * 3.0)

        self._started = False
        self._last_emit = 0.0

    # -- PointSource contract ------------------------------------------------ #
    def start(self) -> None:
        self._started = True
        self._last_emit = time.perf_counter()

    def read(self, timeout: float | None = None) -> PointBatch | None:
        """Generate one batch, sleeping as needed to honor the target rate."""
        if not self._started:
            raise RuntimeError("SimulatedSource.read() called before start().")

        # Pace generation to approximate the configured points/second.
        now = time.perf_counter()
        wait = self._batch_period - (now - self._last_emit)
        if wait > 0:
            if timeout is not None and wait > timeout:
                time.sleep(timeout)
                return None
            time.sleep(wait)
        self._last_emit = time.perf_counter()

        return self._make_batch()

    def stop(self) -> None:
        self._started = False

    # -- internals ----------------------------------------------------------- #
    def _make_batch(self) -> PointBatch:
        n = self._batch_size

        # Points live in a vertical strip at the current sweep x, spanning y.
        x = self._sweep_x + self._rng.uniform(0.0, self._strip_width, size=n)
        x = np.clip(x, self._x_min, self._x_max)
        y = self._rng.uniform(self._y_min, self._y_max, size=n)

        z = terrain_height(x, y)

        # Gaussian range/height noise.
        z = z + self._rng.normal(0.0, self._noise_std, size=n)
        # Small planar jitter so points don't fall on a perfect line.
        x = x + self._rng.normal(0.0, self._noise_std, size=n)
        y = y + self._rng.normal(0.0, self._noise_std, size=n)

        # Occasional gross outliers (sensor glitches / flying pixels).
        if self._outlier_fraction > 0.0:
            mask = self._rng.random(n) < self._outlier_fraction
            n_out = int(mask.sum())
            if n_out:
                z[mask] += self._rng.uniform(
                    -self._outlier_magnitude, self._outlier_magnitude, size=n_out
                )

        intensity = self._rng.uniform(0.2, 1.0, size=n).astype(np.float32)

        # Advance and bounce the sweep strip.
        self._sweep_x += self._sweep_step
        if self._sweep_x > self._x_max or self._sweep_x < self._x_min:
            self._sweep_step = -self._sweep_step
            self._sweep_x = float(np.clip(self._sweep_x, self._x_min, self._x_max))

        points = np.column_stack((x, y, z)).astype(np.float64)
        return PointBatch(points=points, intensity=intensity, timestamp=time.time())
