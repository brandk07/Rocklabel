"""Data-source layer.

A common :class:`~lidarrig.sources.base.PointSource` interface yields batches of
points as ``(N, 3)`` float NumPy arrays (x, y, z in meters, sensor frame) plus
optional intensity.  Nothing in this package imports a visualization library.
"""

from rocklabel.live.sources.base import PointBatch, PointSource
from rocklabel.live.sources.simulated import SimulatedSource

__all__ = ["PointBatch", "PointSource", "SimulatedSource", "make_source"]


def make_source(config) -> PointSource:  # type: ignore[no-untyped-def]
    """Factory: build the configured :class:`PointSource` from an ``AppConfig``.

    Importing :class:`MultiScanUDPSource` is deferred so the simulated path has
    zero chance of pulling in socket/threading state it does not need.
    """
    kind = config.source.kind.lower()
    if kind == "sim":
        return SimulatedSource(config)
    if kind == "udp":
        from rocklabel.live.sources.udp_source import MultiScanUDPSource

        return MultiScanUDPSource(config)
    raise ValueError(f"Unknown source kind {config.source.kind!r} (expected 'sim' or 'udp').")
