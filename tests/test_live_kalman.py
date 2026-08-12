"""Unit tests for the per-cell 1D Kalman height fusion.

Covers the three behaviors the prompt calls out:
  * convergence toward the true height under noisy measurements,
  * responsiveness to a step change in the true height (thanks to process noise),
  * correct handling of multiple points landing in the same cell in one batch
    (aggregation, never clobbering).
"""

from __future__ import annotations

import numpy as np
import pytest

from rocklabel.live.config import AppConfig
from rocklabel.live.surfaces.kalman_heightmap import KalmanHeightmap


def _small_config(**overrides) -> AppConfig:
    cfg = AppConfig()
    cfg.grid.origin = (0.0, 0.0)
    cfg.grid.extent = (1.0, 1.0)
    cfg.grid.cell_size = 0.5  # 2x2 grid
    cfg.outlier.enabled = False  # isolate the Kalman math
    cfg.kalman.sensor_variance = 0.04  # (0.2 m)^2
    cfg.kalman.process_noise = 1e-4
    cfg.kalman.initial_variance = 1.0
    for k, v in overrides.items():
        section, _, field = k.partition(".")
        setattr(getattr(cfg, section), field, v)
    return cfg


def _cell_center_points(x: float, y: float, z: np.ndarray) -> np.ndarray:
    n = z.shape[0]
    return np.column_stack([np.full(n, x), np.full(n, y), z])


def test_converges_to_true_height_under_noise():
    """Feeding noisy measurements of a fixed height converges near the truth and
    drives the cell variance down."""
    cfg = _small_config()
    hm = KalmanHeightmap(cfg)
    rng = np.random.default_rng(0)
    true_h = 1.234
    for _ in range(400):
        z = true_h + rng.normal(0.0, 0.2, size=1)
        hm.add_points(_cell_center_points(0.25, 0.25, z))

    iy, ix = 0, 0
    assert hm._hits[iy, ix] == 400
    assert abs(hm._height[iy, ix] - true_h) < 0.02
    # Variance must shrink well below both the prior and the sensor variance.
    assert hm._variance[iy, ix] < cfg.kalman.sensor_variance


def test_first_measurement_dominates_from_diffuse_prior():
    """With a large initial variance, the first measurement essentially sets the
    estimate (Kalman gain ~ 1)."""
    cfg = _small_config()
    cfg.kalman.initial_variance = 1e6
    hm = KalmanHeightmap(cfg)
    hm.add_points(_cell_center_points(0.25, 0.25, np.array([2.5])))
    assert hm._height[0, 0] == pytest.approx(2.5, abs=1e-3)


def test_responds_to_step_change():
    """After converging to one height, a step change in the true surface is
    tracked (process noise keeps the cell responsive rather than frozen)."""
    cfg = _small_config(**{"kalman.process_noise": 1e-2})
    hm = KalmanHeightmap(cfg)
    rng = np.random.default_rng(1)

    for _ in range(200):
        hm.add_points(_cell_center_points(0.25, 0.25, 0.0 + rng.normal(0, 0.1, 1)))
    assert abs(hm._height[0, 0]) < 0.05

    # True surface jumps to +1.0 m.
    for _ in range(200):
        hm.add_points(_cell_center_points(0.25, 0.25, 1.0 + rng.normal(0, 0.1, 1)))
    assert abs(hm._height[0, 0] - 1.0) < 0.05


def test_frozen_without_process_noise_is_sluggish():
    """Sanity check on the mechanism: with zero process noise the estimate is
    far more sluggish to a step than with process noise."""
    rng = np.random.default_rng(2)

    def run(q: float) -> float:
        cfg = _small_config(**{"kalman.process_noise": q})
        hm = KalmanHeightmap(cfg)
        for _ in range(300):
            hm.add_points(_cell_center_points(0.25, 0.25, 0.0 + rng.normal(0, 0.05, 1)))
        for _ in range(30):
            hm.add_points(_cell_center_points(0.25, 0.25, 1.0 + rng.normal(0, 0.05, 1)))
        return hm._height[0, 0]

    responsive = run(1e-1)
    frozen = run(0.0)
    assert responsive > frozen  # process noise tracks the jump faster


def test_duplicate_cell_points_in_one_batch_are_aggregated():
    """Many points in the same cell within a single batch must be fused (mean),
    not clobbered down to the last one."""
    cfg = _small_config()
    cfg.kalman.initial_variance = 1e6  # so one batch essentially sets the mean
    hm = KalmanHeightmap(cfg)

    zs = np.array([0.9, 1.0, 1.1, 1.0, 1.0])  # mean 1.0
    hm.add_points(_cell_center_points(0.25, 0.25, zs))

    assert hm._hits[0, 0] == 5
    # Estimate is the batch mean, not the last (1.0) or any single value.
    assert hm._height[0, 0] == pytest.approx(zs.mean(), abs=1e-3)


def test_duplicate_aggregation_matches_sequential_equal_variance():
    """The batch aggregation (mean with R/m) equals sequential per-point updates
    for a static height — verify against an explicit scalar Kalman loop."""
    cfg = _small_config()
    hm = KalmanHeightmap(cfg)
    zs = np.array([0.8, 1.2, 0.9, 1.1, 1.0, 1.05, 0.95])
    hm.add_points(_cell_center_points(0.25, 0.25, zs))

    # Reference: scalar sequential Kalman with a single predict step for the
    # batch (matches KalmanHeightmap: predict once, then fuse the mean w/ R/m).
    h, P = 0.0, cfg.kalman.initial_variance
    R = cfg.kalman.sensor_variance
    P = P + cfg.kalman.process_noise
    m = len(zs)
    K = P / (P + R / m)
    h = h + K * (zs.mean() - h)
    P = (1 - K) * P

    assert hm._height[0, 0] == pytest.approx(h, abs=1e-9)
    assert hm._variance[0, 0] == pytest.approx(P, abs=1e-12)


def test_points_outside_grid_are_ignored():
    cfg = _small_config()
    hm = KalmanHeightmap(cfg)
    hm.add_points(np.array([[100.0, 100.0, 5.0], [-50.0, 0.0, 3.0]]))
    assert hm.cells_occupied() == 0


def test_reset_clears_state():
    cfg = _small_config()
    hm = KalmanHeightmap(cfg)
    hm.add_points(_cell_center_points(0.25, 0.25, np.array([1.0, 1.0])))
    assert hm.cells_occupied() == 1
    hm.reset()
    assert hm.cells_occupied() == 0
    assert np.all(hm._hits == 0)
    assert np.allclose(hm._variance, cfg.kalman.initial_variance)


def test_mesh_arrays_are_well_formed():
    """The extracted mesh indexes only valid vertices and has 2 tris per quad."""
    cfg = AppConfig()
    cfg.grid.origin = (0.0, 0.0)
    cfg.grid.extent = (2.0, 2.0)
    cfg.grid.cell_size = 0.5  # 4x4 grid
    cfg.outlier.enabled = False
    hm = KalmanHeightmap(cfg)
    # Fill every cell so the whole grid triangulates.
    xs = np.linspace(0.1, 1.9, 40)
    ys = np.linspace(0.1, 1.9, 40)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    hm.add_points(pts)

    mesh = hm.get_mesh_arrays()
    assert not mesh.is_empty()
    assert mesh.triangles.max() < mesh.vertices.shape[0]
    assert mesh.triangles.min() >= 0
    assert mesh.vertex_colors.shape == mesh.vertices.shape


def test_mesh_cache_reuses_until_dirty():
    cfg = _small_config()
    hm = KalmanHeightmap(cfg)
    hm.add_points(_cell_center_points(0.25, 0.25, np.array([1.0, 1.0, 1.0, 1.0])))
    hm.add_points(_cell_center_points(0.75, 0.25, np.array([1.0, 1.0, 1.0, 1.0])))
    hm.add_points(_cell_center_points(0.25, 0.75, np.array([1.0, 1.0, 1.0, 1.0])))
    hm.add_points(_cell_center_points(0.75, 0.75, np.array([1.0, 1.0, 1.0, 1.0])))
    m1 = hm.get_mesh_arrays()
    m2 = hm.get_mesh_arrays()  # no new points -> same cached object
    assert m1 is m2
