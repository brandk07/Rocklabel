"""The common :class:`PointSource` interface and the :class:`PointBatch` value type.

Both the simulated source and the live UDP source implement the same small
contract so the app loop is agnostic to where points come from:

    source.start()
    while running:
        batch = source.read(timeout=...)   # -> PointBatch | None
        if batch is not None:
            surface.add_points(batch.points)
    source.stop()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PointBatch:
    """A batch of points in the sensor frame.

    Attributes:
        points: ``(N, 3)`` float array of (x, y, z) in meters.
        intensity: optional ``(N,)`` float array of per-point intensity/RSSI.
        timestamp: source timestamp in seconds (sensor time if available, else
            host wall-clock at receipt). Used only for stats/decay, never fusion.
        orientation: optional ``(4,)`` sensor orientation quaternion
            ``(w, x, y, z)`` (sensor→world, from the onboard IMU) valid at the
            time these points were captured. ``None`` when the source has no
            pose information (e.g. the simulator, or IMU output disabled).
    """

    points: np.ndarray
    intensity: np.ndarray | None = None
    timestamp: float = 0.0
    orientation: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"PointBatch.points must be (N, 3); got shape {self.points.shape!r}"
            )

    def __len__(self) -> int:
        return int(self.points.shape[0])


class PointSource(ABC):
    """Abstract streaming source of point batches.

    Implementations must be safe to :meth:`start` once, :meth:`read` repeatedly
    from a single consumer thread, and :meth:`stop` once.
    """

    #: True for sources that replay a recording (batches arrive already in the
    #: world frame, and recording from them is disabled).
    is_replay: bool = False

    @abstractmethod
    def start(self) -> None:
        """Begin producing data (open sockets, spin up threads, seed RNG)."""

    @abstractmethod
    def read(self, timeout: float | None = None) -> PointBatch | None:
        """Return the next available :class:`PointBatch`.

        Args:
            timeout: max seconds to wait for data. ``None`` blocks indefinitely.

        Returns:
            The next batch, or ``None`` if no batch was available within
            ``timeout`` (the caller should simply try again).
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop producing data and release all resources. Idempotent."""

    # Convenience so sources can be used as context managers in scripts/tests.
    def __enter__(self) -> "PointSource":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
