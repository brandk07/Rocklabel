"""Streaming voxel-grid accumulator: bounded memory regardless of recording length.

Each scan is reduced to per-voxel (sum xyz, sum intensity, count) chunks;
chunks are periodically compacted by merging equal voxel keys. Memory is
bounded by the number of occupied voxels (arena size / voxel size), not by
the number of scans.
"""

from __future__ import annotations

import numpy as np

# 21 bits per axis, signed: voxel indices in [-2^20, 2^20). At the default
# 0.03 m resolution that is +/- ~31 km, far beyond any arena.
_BITS = 21
_OFFSET = 1 << (_BITS - 1)
_MASK_RANGE = 1 << _BITS


class VoxelAccumulator:
    def __init__(self, voxel_m: float, compact_rows: int = 4_000_000):
        self.voxel_m = float(voxel_m)
        self._chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._pending = 0
        self._compact_rows = compact_rows

    def add(self, xyz: np.ndarray, intensity: np.ndarray) -> None:
        if len(xyz) == 0:
            return
        ijk = np.floor(xyz / self.voxel_m).astype(np.int64) + _OFFSET
        if (ijk < 0).any() or (ijk >= _MASK_RANGE).any():
            keep = ((ijk >= 0) & (ijk < _MASK_RANGE)).all(axis=1)
            ijk, xyz, intensity = ijk[keep], xyz[keep], intensity[keep]
            if len(ijk) == 0:
                return
        keys = (ijk[:, 0] << (2 * _BITS)) | (ijk[:, 1] << _BITS) | ijk[:, 2]
        uniq, inverse = np.unique(keys, return_inverse=True)
        sums = np.zeros((len(uniq), 4), np.float64)
        np.add.at(sums, inverse, np.column_stack([xyz.astype(np.float64), intensity.astype(np.float64)]))
        counts = np.bincount(inverse, minlength=len(uniq)).astype(np.int64)
        self._chunks.append((uniq, sums, counts))
        self._pending += len(uniq)
        if self._pending > self._compact_rows:
            self._compact()

    def _compact(self) -> None:
        if len(self._chunks) <= 1:
            return
        keys = np.concatenate([c[0] for c in self._chunks])
        sums = np.concatenate([c[1] for c in self._chunks])
        counts = np.concatenate([c[2] for c in self._chunks])
        uniq, inverse = np.unique(keys, return_inverse=True)
        merged_sums = np.zeros((len(uniq), 4), np.float64)
        np.add.at(merged_sums, inverse, sums)
        merged_counts = np.zeros(len(uniq), np.int64)
        np.add.at(merged_counts, inverse, counts)
        self._chunks = [(uniq, merged_sums, merged_counts)]
        self._pending = len(uniq)

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (centroids float32 [V,3], mean_intensity float32 [V], counts int64 [V])."""
        self._compact()
        if not self._chunks:
            return np.empty((0, 3), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)
        _, sums, counts = self._chunks[0]
        centroids = (sums[:, :3] / counts[:, None]).astype(np.float32)
        mean_int = (sums[:, 3] / counts).astype(np.float32)
        return centroids, mean_int, counts

    @property
    def voxel_count(self) -> int:
        self._compact()
        return len(self._chunks[0][0]) if self._chunks else 0


def voxel_downsample_centroids(xyz: np.ndarray, voxel_m: float) -> np.ndarray:
    """One-shot voxel downsample returning per-voxel centroids (float64 [V,3])."""
    if len(xyz) == 0:
        return np.empty((0, 3))
    ijk = np.floor(xyz / voxel_m).astype(np.int64)
    # Shift to non-negative so the packed key is collision-free.
    ijk -= ijk.min(axis=0)
    keys = (ijk[:, 0] << (2 * _BITS)) | (ijk[:, 1] << _BITS) | ijk[:, 2]
    uniq, inverse = np.unique(keys, return_inverse=True)
    sums = np.zeros((len(uniq), 3), np.float64)
    np.add.at(sums, inverse, xyz.astype(np.float64))
    counts = np.bincount(inverse, minlength=len(uniq))
    return sums / counts[:, None]


def write_ply(path: str, xyz: np.ndarray, intensity: np.ndarray) -> None:
    """Write a binary little-endian PLY with x, y, z, intensity (no Open3D needed)."""
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float intensity\nend_header\n"
    )
    rows = np.column_stack([xyz.astype(np.float32), intensity.astype(np.float32)])
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(rows, dtype="<f4").tobytes())
