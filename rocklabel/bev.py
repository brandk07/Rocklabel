"""Dataset format B: bird's-eye-view multi-channel rasters with per-cell label masks.

Grid convention: channels[c, i, j] where i indexes x (rows, from
base_x - crop_backward toward +x) and j indexes y (columns, from
base_y - crop_right toward +y). All channels float32; cells with no points
are 0 in every channel, with channel 0 (valid) marking occupancy.

Channels:
  0 valid (1 if >=1 point)      4 z span (max - min)
  1 log1p(point count)          5 z standard deviation
  2 max z - robot base z        6 mean intensity
  3 min z - robot base z        7 max intensity
"""

from __future__ import annotations

import numpy as np

from .labeling import LABEL_CLEAR, LABEL_ROCK, MASK_IGNORE

N_CHANNELS = 8


def bev_grid_shape(gcfg: dict) -> tuple[int, int]:
    cell = gcfg["bev_cell_m"]
    nx = int(round((gcfg["crop_forward_m"] + gcfg["crop_backward_m"]) / cell))
    ny = int(round((gcfg["crop_left_m"] + gcfg["crop_right_m"]) / cell))
    return nx, ny


def rasterize_bev(
    xyz: np.ndarray,
    intensity: np.ndarray,
    point_labels: np.ndarray,
    base_xyz: np.ndarray,
    gcfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one cropped frame cloud. Returns (channels [8, H, W] f32, mask [H, W] u8)."""
    nx, ny = bev_grid_shape(gcfg)
    cell = gcfg["bev_cell_m"]
    x0 = base_xyz[0] - gcfg["crop_backward_m"]
    y0 = base_xyz[1] - gcfg["crop_right_m"]

    ix = np.floor((xyz[:, 0] - x0) / cell).astype(np.int64)
    iy = np.floor((xyz[:, 1] - y0) / cell).astype(np.int64)
    inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = ix[inb], iy[inb]
    z = xyz[inb, 2].astype(np.float64)
    inten = intensity[inb].astype(np.float64)
    labels = point_labels[inb]
    flat = ix * ny + iy
    ncell = nx * ny

    counts = np.bincount(flat, minlength=ncell).astype(np.float64)
    occupied = counts > 0
    safe = np.maximum(counts, 1)

    z_max = np.full(ncell, -np.inf)
    z_min = np.full(ncell, np.inf)
    np.maximum.at(z_max, flat, z)
    np.minimum.at(z_min, flat, z)
    z_max[~occupied] = 0.0
    z_min[~occupied] = 0.0
    z_sum = np.bincount(flat, weights=z, minlength=ncell)
    z_sq_sum = np.bincount(flat, weights=z * z, minlength=ncell)
    z_mean = z_sum / safe
    z_var = np.maximum(z_sq_sum / safe - z_mean**2, 0.0)

    i_sum = np.bincount(flat, weights=inten, minlength=ncell)
    i_max = np.full(ncell, -np.inf)
    np.maximum.at(i_max, flat, inten)
    i_max[~occupied] = 0.0

    base_z = float(base_xyz[2])
    channels = np.zeros((N_CHANNELS, ncell), np.float32)
    channels[0] = occupied
    channels[1] = np.log1p(counts)
    channels[2] = np.where(occupied, z_max - base_z, 0.0)
    channels[3] = np.where(occupied, z_min - base_z, 0.0)
    channels[4] = np.where(occupied, z_max - z_min, 0.0)
    channels[5] = np.where(occupied, np.sqrt(z_var), 0.0)
    channels[6] = i_sum / safe
    channels[7] = i_max

    rock_any = np.bincount(flat[labels == LABEL_ROCK], minlength=ncell) > 0
    clear_any = np.bincount(flat[labels == LABEL_CLEAR], minlength=ncell) > 0
    mask = np.full(ncell, MASK_IGNORE, np.uint8)  # empty or ignore-only cells
    mask[clear_any] = 0
    mask[rock_any] = 1

    return channels.reshape(N_CHANNELS, nx, ny), mask.reshape(nx, ny)
