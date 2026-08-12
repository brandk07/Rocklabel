"""Tests for the region-of-interest crop and the mesh edge guard — the two
defenses against 360° indoor scans (walls/ceiling) polluting the 2.5D surface."""

from __future__ import annotations

import numpy as np

from rocklabel.live.config import AppConfig, CropConfig
from rocklabel.live.filters import crop_batch
from rocklabel.live.surfaces.kalman_heightmap import KalmanHeightmap


def test_crop_z_band():
    cfg = CropConfig(enabled=True, z_min=-1.5, z_max=-0.5)
    pts = np.array(
        [
            [1.0, 0.0, -1.0],  # floor: keep
            [2.0, 0.0, 0.0],  # sensor height (wall hit): drop
            [3.0, 0.0, 2.0],  # ceiling: drop
            [4.0, 0.0, -1.4],  # floor: keep
        ]
    )
    out = crop_batch(pts, cfg)
    assert out.shape == (2, 3)
    assert np.allclose(out[:, 2], [-1.0, -1.4])


def test_crop_range_gate():
    cfg = CropConfig(enabled=True, z_min=-10, z_max=10, range_min=1.0, range_max=5.0)
    pts = np.array(
        [
            [0.2, 0.0, 0.0],  # too near: drop
            [3.0, 0.0, 0.0],  # keep
            [40.0, 0.0, 0.0],  # too far: drop
        ]
    )
    out = crop_batch(pts, cfg)
    assert out.shape == (1, 3)
    assert out[0, 0] == 3.0


def test_crop_disabled_passthrough():
    cfg = CropConfig(enabled=False, z_min=0.0, z_max=0.0)
    pts = np.array([[0.0, 0.0, 99.0]])
    assert crop_batch(pts, cfg) is pts


def test_mesh_edge_guard_skips_wall_spikes():
    """Two adjacent cell rows with a huge height jump must not be bridged by
    triangles when max_edge_dz is set."""
    cfg = AppConfig()
    cfg.grid.origin = (0.0, 0.0)
    cfg.grid.extent = (2.0, 1.0)
    cfg.grid.cell_size = 0.5  # 4 x 2 grid
    cfg.outlier.enabled = False
    cfg.kalman.max_edge_dz = 1.0
    hm = KalmanHeightmap(cfg)

    # Left half at z=0, right half at z=5 (a "wall" column artifact).
    pts = []
    for _ in range(20):
        for cx in (0.25, 0.75):
            for cy in (0.25, 0.75):
                pts.append([cx, cy, 0.0])
        for cx in (1.25, 1.75):
            for cy in (0.25, 0.75):
                pts.append([cx, cy, 5.0])
    hm.add_points(np.array(pts))

    mesh = hm.get_mesh_arrays()
    # Quads exist within each half but never across the 5 m jump.
    assert not mesh.is_empty()
    z = mesh.vertices[:, 2]
    for tri in mesh.triangles:
        span = z[tri].max() - z[tri].min()
        assert span <= 1.0 + 1e-6, f"triangle spans {span} m"


def test_mesh_edge_guard_disabled_bridges():
    """Sanity: with the guard off, the jump IS bridged (old behavior)."""
    cfg = AppConfig()
    cfg.grid.origin = (0.0, 0.0)
    cfg.grid.extent = (2.0, 1.0)
    cfg.grid.cell_size = 0.5
    cfg.outlier.enabled = False
    cfg.kalman.max_edge_dz = 0.0  # disabled
    hm = KalmanHeightmap(cfg)
    pts = []
    for _ in range(20):
        for cx in (0.25, 0.75, 1.25, 1.75):
            for cy in (0.25, 0.75):
                pts.append([cx, cy, 0.0 if cx < 1.0 else 5.0])
    hm.add_points(np.array(pts))
    mesh = hm.get_mesh_arrays()
    z = mesh.vertices[:, 2]
    spans = [z[t].max() - z[t].min() for t in mesh.triangles]
    assert max(spans) > 4.0  # the wall gets bridged when the guard is off
