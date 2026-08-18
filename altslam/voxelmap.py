"""Sparse voxel map that also carries a surface normal per voxel.

The stock map (:mod:`rocklabel.live.slam`) stores one centroid per voxel and
matches points to those centroids. On a rough sand surface that is a weak
target: the centroid of a 20 cm patch of sand is a noisy point, and pulling a
scan point *onto* it fights the sand's own roughness.

Here each voxel additionally accumulates the sum of outer products of its
points, which gives a covariance, which gives a **local surface normal** plus a
measure of how plane-like the patch is. Matching then only penalizes motion
*along the normal* (point-to-plane), so scan points are free to slide within
the surface — which is what you want, because sliding within the surface is not
an error, it is just a different part of the same sand.

Layout mirrors the stock map: keys are packed into sorted int64s so lookup and
insert stay fully vectorized, with no Python-level per-point work.
"""

from __future__ import annotations

import numpy as np

# 21 bits per axis packed into an int64; supports |coord| < 2^20 voxels.
_BITS = 21
_OFF = 1 << 20
_LIM = (1 << _BITS) - 1

#: 3x3x3 neighbourhood as direct int64 key deltas. The stock map searches only
#: the 6 face neighbours; the corners matter here because the correspondence
#: radius starts at 3x the voxel size.
_NEIGHBOR_DELTAS = tuple(
    int(dx * (1 << (2 * _BITS)) + dy * (1 << _BITS) + dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
)


def encode(ijk: np.ndarray) -> np.ndarray:
    """Pack integer voxel coords ``(N, 3)`` into sortable int64 keys."""
    c = np.clip(ijk + _OFF, 0, _LIM)
    return (c[:, 0] << (2 * _BITS)) | (c[:, 1] << _BITS) | c[:, 2]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Reduce ``(N, 3)`` points to one centroid per ``voxel``-sized cell."""
    if points.shape[0] == 0:
        return points
    keys = encode(np.floor(points / voxel).astype(np.int64))
    uniq, inv = np.unique(keys, return_inverse=True)
    counts = np.bincount(inv, minlength=uniq.shape[0])
    sums = np.empty((uniq.shape[0], 3), dtype=np.float64)
    for k in range(3):
        sums[:, k] = np.bincount(inv, weights=points[:, k], minlength=uniq.shape[0])
    return sums / counts[:, None]


class NormalVoxelMap:
    """Voxel map holding centroid, covariance, normal and planarity per cell."""

    def __init__(
        self,
        voxel_size: float,
        max_points: int = 60,
        min_points_normal: int = 8,
        capacity: int = 4_000_000,
    ) -> None:
        self._vs = float(voxel_size)
        self._max_pts = int(max_points)
        self._min_n = int(min_points_normal)
        self._capacity = int(capacity)
        self._keys = np.empty(0, dtype=np.int64)          # sorted
        self._sum = np.empty((0, 3), dtype=np.float64)    # sum of points
        self._sumsq = np.empty((0, 6), dtype=np.float64)  # xx xy xz yy yz zz
        self._count = np.empty(0, dtype=np.int64)
        # Cached per-voxel plane fit, recomputed lazily for changed voxels only.
        self._normal = np.empty((0, 3), dtype=np.float64)
        self._planarity = np.empty(0, dtype=np.float64)
        self._fitted_at = np.empty(0, dtype=np.int64)     # count when last fitted

    @property
    def size(self) -> int:
        """Number of occupied voxels."""
        return int(self._keys.shape[0])

    @property
    def voxel_size(self) -> float:
        return self._vs

    # -- building ----------------------------------------------------------- #
    def insert(self, points: np.ndarray) -> None:
        """Merge ``(N, 3)`` world-frame points into the map."""
        if points.shape[0] == 0 or self.size >= self._capacity:
            return
        keys = encode(np.floor(points / self._vs).astype(np.int64))
        uniq, inv = np.unique(keys, return_inverse=True)
        n_g = uniq.shape[0]

        cnt = np.bincount(inv, minlength=n_g)
        s = np.empty((n_g, 3), dtype=np.float64)
        for k in range(3):
            s[:, k] = np.bincount(inv, weights=points[:, k], minlength=n_g)
        # Outer-product sums, upper triangle only (xx xy xz yy yz zz).
        ss = np.empty((n_g, 6), dtype=np.float64)
        for j, (a, b) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
            ss[:, j] = np.bincount(
                inv, weights=points[:, a] * points[:, b], minlength=n_g
            )

        if self.size:
            pos = np.searchsorted(self._keys, uniq)
            pos_c = np.minimum(pos, self.size - 1)
            exists = self._keys[pos_c] == uniq
        else:
            pos_c = np.zeros(n_g, dtype=np.int64)
            exists = np.zeros(n_g, dtype=bool)

        # Existing voxels keep accumulating until they hit max_points, then
        # freeze. A voxel seen from many angles has already converged; letting a
        # late, badly-posed window keep editing it just smears the map.
        if np.any(exists):
            idx = pos_c[exists]
            open_ = self._count[idx] < self._max_pts
            tgt = idx[open_]
            self._sum[tgt] += s[exists][open_]
            self._sumsq[tgt] += ss[exists][open_]
            self._count[tgt] += cnt[exists][open_]

        new = ~exists
        if np.any(new):
            nk = uniq[new]  # already sorted (np.unique)
            ins = np.searchsorted(self._keys, nk)
            self._keys = np.insert(self._keys, ins, nk)
            self._sum = np.insert(self._sum, ins, s[new], axis=0)
            self._sumsq = np.insert(self._sumsq, ins, ss[new], axis=0)
            self._count = np.insert(self._count, ins, cnt[new])
            self._normal = np.insert(self._normal, ins, 0.0, axis=0)
            self._planarity = np.insert(self._planarity, ins, 0.0)
            self._fitted_at = np.insert(self._fitted_at, ins, -1)

    # -- plane fitting ------------------------------------------------------ #
    def refresh_normals(self) -> None:
        """Recompute normals for voxels whose contents changed since last fit."""
        stale = np.nonzero(
            (self._fitted_at != self._count) & (self._count >= self._min_n)
        )[0]
        if stale.size == 0:
            return
        n = self._count[stale].astype(np.float64)[:, None]
        mu = self._sum[stale] / n
        ss = self._sumsq[stale] / n
        # Rebuild the symmetric 3x3 covariance from the packed upper triangle.
        cov = np.empty((stale.size, 3, 3), dtype=np.float64)
        cov[:, 0, 0] = ss[:, 0] - mu[:, 0] * mu[:, 0]
        cov[:, 0, 1] = cov[:, 1, 0] = ss[:, 1] - mu[:, 0] * mu[:, 1]
        cov[:, 0, 2] = cov[:, 2, 0] = ss[:, 2] - mu[:, 0] * mu[:, 2]
        cov[:, 1, 1] = ss[:, 3] - mu[:, 1] * mu[:, 1]
        cov[:, 1, 2] = cov[:, 2, 1] = ss[:, 4] - mu[:, 1] * mu[:, 2]
        cov[:, 2, 2] = ss[:, 5] - mu[:, 2] * mu[:, 2]

        # eigh returns ascending eigenvalues; the smallest one's vector is the
        # surface normal, and how much smaller it is than the next says how
        # plane-like (rather than blobby) the patch is.
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, 0.0)
        total = evals.sum(axis=1) + 1e-12
        self._normal[stale] = evecs[:, :, 0]
        self._planarity[stale] = (evals[:, 1] - evals[:, 0]) / total
        self._fitted_at[stale] = self._count[stale]

    # -- lookup ------------------------------------------------------------- #
    def query(self, points: np.ndarray, max_dist: float):
        """Nearest usable map voxel per point, searching the 3x3x3 neighbourhood.

        Returns:
            ``(valid, centroid, normal, planarity)`` — ``valid`` is an ``(N,)``
            bool mask; the other three are only meaningful where it is True.
        """
        n = points.shape[0]
        cen = np.zeros((n, 3))
        nrm = np.zeros((n, 3))
        pla = np.zeros(n)
        if n == 0 or self.size == 0:
            return np.zeros(n, dtype=bool), cen, nrm, pla

        self.refresh_normals()
        usable = self._count >= self._min_n
        centroids = self._sum / np.maximum(self._count, 1)[:, None]

        base_keys = encode(np.floor(points / self._vs).astype(np.int64))
        best_d2 = np.full(n, np.inf)
        best_idx = np.full(n, -1, dtype=np.int64)
        last = self.size - 1
        for delta in _NEIGHBOR_DELTAS:
            keys = base_keys if delta == 0 else base_keys + delta
            pos = np.minimum(np.searchsorted(self._keys, keys), last)
            hit = np.nonzero((self._keys[pos] == keys) & usable[pos])[0]
            if hit.size == 0:
                continue
            cand = pos[hit]
            diff = points[hit] - centroids[cand]
            d2 = np.einsum("ij,ij->i", diff, diff)
            better = d2 < best_d2[hit]
            upd = hit[better]
            best_d2[upd] = d2[better]
            best_idx[upd] = cand[better]

        valid = (best_idx >= 0) & (best_d2 <= max_dist * max_dist)
        sel = best_idx[valid]
        cen[valid] = centroids[sel]
        nrm[valid] = self._normal[sel]
        pla[valid] = self._planarity[sel]
        return valid, cen, nrm, pla
