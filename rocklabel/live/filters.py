"""Fast, viz-free outlier rejection for incoming point batches.

For a 2.5D heightmap the effective and *fast* statistical outlier test is
per-column: bin points into coarse x/y cells and reject any point whose height
deviates from its cell's mean by more than ``std_ratio`` standard deviations.
This is O(N) and fully vectorized (via ``np.bincount``), so it keeps up with a
200k+ points/second stream — unlike a brute-force k-NN, which is O(N^2).

It targets exactly the failure mode the simulator injects (gross z-spikes /
flying pixels) and the ones a real multiScan produces (mixed pixels at depth
discontinuities, sky returns).  Kept dependency-light (NumPy only) so the fusion
path stays free of any visualization import and is reusable from a ROS 2 node.
"""

from __future__ import annotations

import numpy as np

from rocklabel.live.config import CropConfig, OutlierConfig


def crop_band(cfg: CropConfig, floor_z: float | None) -> tuple[float, float]:
    """Absolute ``(z_min, z_max)`` of the crop band for this batch.

    With ``floor_relative`` the configured band is offset by the measured floor
    height, so the same ``--floor-band`` works whether the sensor rides 0.4 m
    up on a robot or 1 m up in your hand. Before levelling has found a floor
    there is nothing to anchor to, so the band stays absolute.
    """
    if cfg.floor_relative and floor_z is not None:
        return cfg.z_min + floor_z, cfg.z_max + floor_z
    return cfg.z_min, cfg.z_max


def crop_mask(
    points: np.ndarray,
    cfg: CropConfig,
    origin: np.ndarray | None = None,
    floor_z: float | None = None,
) -> np.ndarray | None:
    """Keep-mask for the region-of-interest crop (z-band + range gate).

    Args:
        points: ``(N, 3)`` world-frame points.
        cfg: crop settings.
        origin: sensor position the range gate is measured from. Defaults to
            the world origin — correct only while the sensor stays put, which
            is why the mobile rig must pass its SLAM pose here: otherwise
            ``range_max`` gates on distance from the *starting* point and
            silently clips everything once the robot has driven that far.
        floor_z: measured floor height, for a ``floor_relative`` band.

    Returns ``None`` when nothing would be cropped (disabled / empty / all
    kept), so callers can skip the fancy-indexing copy entirely — and apply the
    same mask to parallel per-point arrays (e.g. intensity).
    """
    if not cfg.enabled or points.shape[0] == 0:
        return None
    z_min, z_max = crop_band(cfg, floor_z)
    z = points[:, 2]
    keep = (z >= z_min) & (z <= z_max)
    if cfg.range_min > 0.0 or cfg.range_max > 0.0:
        rel = points if origin is None else points - np.asarray(origin, dtype=points.dtype)
        r2 = np.einsum("ij,ij->i", rel, rel)  # squared 3D range from the sensor
        if cfg.range_min > 0.0:
            keep &= r2 >= cfg.range_min**2
        if cfg.range_max > 0.0:
            keep &= r2 <= cfg.range_max**2
    if keep.all():
        return None
    return keep


def crop_batch(points: np.ndarray, cfg: CropConfig) -> np.ndarray:
    """Crop a batch to the configured region of interest (z-band + range gate).

    With a 360° sensor indoors this is what keeps wall/ceiling returns out of
    the 2.5D grid: a heightmap can only hold one surface per (x, y) column, so
    anything outside the z-band of the surface you're mapping must be discarded
    *before* fusion.

    Args:
        points: ``(N, 3)`` array in the sensor frame.
        cfg: crop settings (z-band, optional min/max 3D range from the sensor).

    Returns:
        The kept points (possibly the input array unchanged if no crop applies).
    """
    keep = crop_mask(points, cfg)
    return points if keep is None else points[keep]


def statistical_outlier_mask(
    points: np.ndarray,
    cell_size: float,
    std_ratio: float = 2.0,
    min_std: float = 0.01,
    min_cell_points: int = 4,
) -> np.ndarray:
    """Return a boolean *keep* mask (True = inlier) using per-cell z statistics.

    Points are binned into square ``cell_size`` columns. Within each column with
    at least ``min_cell_points`` points, a point is rejected when
    ``|z - mean_z| > std_ratio * max(std_z, min_std)``. Sparse columns (too few
    points to estimate a spread) are kept as-is.

    Args:
        points: ``(N, 3)`` array.
        cell_size: side length (m) of the binning column. Should be a few times
            the heightmap cell size so each column holds enough points.
        std_ratio: rejection threshold in standard deviations.
        min_std: floor on the per-cell std so flat, low-noise columns don't
            reject their own tiny fluctuations.
        min_cell_points: minimum points in a column before the test applies.

    Returns:
        ``(N,)`` boolean keep mask.
    """
    n = points.shape[0]
    if n == 0:
        return np.ones(0, dtype=bool)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Integer column indices, made non-negative and compacted to a dense range.
    ix = np.floor((x - x.min()) / cell_size).astype(np.int64)
    iy = np.floor((y - y.min()) / cell_size).astype(np.int64)
    key = ix * (iy.max() + 1) + iy
    uniq, inv = np.unique(key, return_inverse=True)
    ncells = uniq.shape[0]

    count = np.bincount(inv, minlength=ncells).astype(np.float64)
    sum_z = np.bincount(inv, weights=z, minlength=ncells)
    sum_z2 = np.bincount(inv, weights=z * z, minlength=ncells)
    mean = sum_z / count
    var = np.maximum(sum_z2 / count - mean * mean, 0.0)
    std = np.sqrt(var)

    cell_mean = mean[inv]
    cell_std = np.maximum(std[inv], min_std)
    cell_count = count[inv]

    within = np.abs(z - cell_mean) <= std_ratio * cell_std
    sparse = cell_count < min_cell_points  # keep points in under-populated cells
    return within | sparse


def filter_keep_mask(points: np.ndarray, cfg: OutlierConfig) -> np.ndarray | None:
    """Configured statistical outlier keep-mask, or ``None`` for pass-through.

    Returning ``None`` (disabled / batch too small) lets callers skip the
    fancy-indexing copy and apply the same mask to parallel per-point arrays
    (e.g. intensity).
    """
    if not cfg.enabled or points.shape[0] < cfg.min_points:
        return None
    return statistical_outlier_mask(
        points,
        cell_size=cfg.cell_size,
        std_ratio=cfg.std_ratio,
        min_std=cfg.min_std,
        min_cell_points=cfg.min_cell_points,
    )


def filter_batch(points: np.ndarray, cfg: OutlierConfig) -> np.ndarray:
    """Apply configured statistical outlier rejection, returning kept points.

    Respects ``cfg.enabled`` and ``cfg.min_points`` (batches too small to
    estimate statistics are passed through unchanged).
    """
    keep = filter_keep_mask(points, cfg)
    return points if keep is None else points[keep]
