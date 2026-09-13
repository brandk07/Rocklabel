"""Conservative ray-carving voxel map.

Unlike an additive voxel accumulator, this map can retract an occupied voxel
after *independent* observations show free space in front of a later return.
Every update is one observation group (normally one source scan/telegram): a
group contributes at most one support or contradiction vote to a voxel.

The implementation keeps visibility evidence separate from surface evidence.
A return only contradicts a mapped point when the point is in front of the
return, lies inside the narrow ray corridor, and is outside the endpoint
ambiguity margin. A nearby endpoint is positive support. A well-conditioned
local plane may make a candidate surface-compatible or ambiguous, but a
missing/degenerate plane is never by itself a deletion reason.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

_BITS = 21
_OFF = 1 << (_BITS - 1)
_LIMIT = 1 << _BITS
_MAP_ARRAYS = (
    "pts", "tag", "hits", "intensity_sum", "intensity_hits",
    "observations", "support", "contradictions", "confirmed",
    "_last_support_group", "_last_contradiction_group", "_keys_by_row",
    "_block_keys", "_tile_keys",
)


@dataclass(frozen=True)
class CarvingEvidence:
    """Copies of the temporal evidence aligned with :attr:`KFCMap.pts`."""

    observations: np.ndarray
    support: np.ndarray
    contradictions: np.ndarray
    confirmed: np.ndarray


@dataclass(frozen=True)
class RayBatch:
    """Aligned ray observations shared by offline and live accumulation.

    Arrays are per endpoint. Consecutive rows with the same ``groups`` value
    are one independent observation and cast at most one vote per voxel.
    """

    endpoints: np.ndarray
    origins: np.ndarray
    timestamps: np.ndarray
    groups: np.ndarray
    intensity: np.ndarray
    floater: np.ndarray
    pose_uncertainty_m: np.ndarray

    def __post_init__(self) -> None:
        endpoints = np.asarray(self.endpoints, np.float64)
        if endpoints.ndim != 2 or endpoints.shape[1] != 3:
            raise ValueError("RayBatch.endpoints must have shape (N, 3)")
        n = len(endpoints)
        origins = np.asarray(self.origins, np.float64)
        if origins.shape != (n, 3):
            raise ValueError("RayBatch.origins must align one (x,y,z) with each endpoint")
        timestamps = np.asarray(self.timestamps, np.float64)
        groups = np.asarray(self.groups)
        intensity = np.asarray(self.intensity, np.float64)
        floater = np.asarray(self.floater, bool)
        uncertainty = np.asarray(self.pose_uncertainty_m, np.float64)
        for name, value in (
            ("timestamps", timestamps),
            ("groups", groups),
            ("intensity", intensity),
            ("floater", floater),
            ("pose_uncertainty_m", uncertainty),
        ):
            if value.shape != (n,):
                raise ValueError(f"RayBatch.{name} must have shape ({n},)")
        for name, value in (
            ("endpoints", endpoints), ("origins", origins),
            ("timestamps", timestamps), ("groups", groups),
            ("intensity", intensity), ("floater", floater),
            ("pose_uncertainty_m", uncertainty),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def one_group(
        cls,
        endpoints: np.ndarray,
        origin: np.ndarray,
        *,
        timestamp: float = 0.0,
        group: object = 0,
        intensity: np.ndarray | None = None,
        floater: np.ndarray | None = None,
        pose_uncertainty_m: float = 0.0,
    ) -> "RayBatch":
        """Build a batch for one group while preserving per-endpoint origins."""
        endpoints = np.asarray(endpoints, np.float64)
        n = len(endpoints)
        origins = np.asarray(origin, np.float64)
        if origins.shape == (3,):
            origins = np.broadcast_to(origins, (n, 3))
        return cls(
            endpoints=endpoints,
            origins=origins,
            timestamps=np.full(n, timestamp, np.float64),
            groups=np.full(n, group),
            intensity=(np.full(n, np.nan, np.float64) if intensity is None
                       else np.asarray(intensity, np.float64)),
            floater=(np.zeros(n, bool) if floater is None
                     else np.asarray(floater, bool)),
            pose_uncertainty_m=np.full(n, pose_uncertainty_m, np.float64),
        )


class KFCMap:
    """Voxel map with ordered visibility and bounded temporal evidence."""

    def __init__(
        self,
        voxel: float = 0.05,
        frustum_search_radius: float = 0.015,
        radial_dist_thresh: float = 0.008,
        surface_width: float = 0.10,
        delete_max_range: float = 6.0,
        add_max_range: float = 6.0,
        ray_max_range: float | None = None,
        normal_k_res: float = 8.0,
        normal_max_k: int = 32,
        normal_min_pts: int = 3,
        normal_min_second_ratio: float = 0.05,
        normal_max_residual: float | None = None,
        normal_min_incidence_cos: float = 0.10,
        endpoint_margin: float | None = None,
        confirm_observations: int = 2,
        tentative_contradictions: int = 2,
        confirmed_contradictions: int = 3,
        max_evidence: int = 8,
        resurrection_observations: int = 2,
        tombstone_groups: int = 64,
        max_pose_uncertainty: float = 0.03,
        candidate_pair_budget: int = 250_000,
        candidate_nearest_k: int = 1,
    ) -> None:
        positive = {
            "voxel": voxel,
            "frustum_search_radius": frustum_search_radius,
            "radial_dist_thresh": radial_dist_thresh,
            "surface_width": surface_width,
            "delete_max_range": delete_max_range,
            "add_max_range": add_max_range,
            "ray_max_range": (
                max(delete_max_range, add_max_range)
                if ray_max_range is None else ray_max_range
            ),
            "normal_k_res": normal_k_res,
        }
        bad = [name for name, value in positive.items()
               if not np.isfinite(value) or value <= 0]
        if bad:
            raise ValueError(
                f"carving parameters must be finite and positive: {', '.join(bad)}"
            )
        integers = {
            "normal_max_k": normal_max_k,
            "normal_min_pts": normal_min_pts,
            "confirm_observations": confirm_observations,
            "tentative_contradictions": tentative_contradictions,
            "confirmed_contradictions": confirmed_contradictions,
            "max_evidence": max_evidence,
            "resurrection_observations": resurrection_observations,
            "tombstone_groups": tombstone_groups,
            "candidate_pair_budget": candidate_pair_budget,
            "candidate_nearest_k": candidate_nearest_k,
        }
        for name, value in integers.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be at least 1")
        if int(max_evidence) < max(
            int(confirm_observations),
            int(tentative_contradictions),
            int(confirmed_contradictions),
        ):
            raise ValueError("max_evidence must cover every evidence threshold")
        if int(max_evidence) > np.iinfo(np.uint8).max:
            raise ValueError("max_evidence must fit in uint8 (at most 255)")
        if not 0.0 < normal_min_second_ratio <= 1.0:
            raise ValueError("normal_min_second_ratio must be in (0, 1]")
        if not 0.0 <= normal_min_incidence_cos <= 1.0:
            raise ValueError("normal_min_incidence_cos must be in [0, 1]")
        if not np.isfinite(max_pose_uncertainty) or max_pose_uncertainty < 0.0:
            raise ValueError("max_pose_uncertainty must be finite and non-negative")

        self.voxel = float(voxel)
        self.voxel_m = self.voxel
        self.fsr = float(frustum_search_radius)
        self.radial = float(radial_dist_thresh)
        self.half_w = float(surface_width) / 2.0
        self.endpoint_margin = float(
            endpoint_margin if endpoint_margin is not None else self.half_w
        )
        if not np.isfinite(self.endpoint_margin) or self.endpoint_margin <= 0.0:
            raise ValueError("endpoint_margin must be finite and positive")
        self.del_max = float(delete_max_range)
        self.add_max = float(add_max_range)
        self.ray_max = float(
            max(delete_max_range, add_max_range)
            if ray_max_range is None else ray_max_range
        )
        self.normal_r = float(normal_k_res) * self.voxel
        self.normal_max_k = int(normal_max_k)
        self.normal_min_pts = int(normal_min_pts)
        self.normal_min_second_ratio = float(normal_min_second_ratio)
        self.normal_max_residual = float(
            normal_max_residual if normal_max_residual is not None else self.voxel
        )
        if not np.isfinite(self.normal_max_residual) or self.normal_max_residual <= 0.0:
            raise ValueError("normal_max_residual must be finite and positive")
        self.normal_min_incidence_cos = float(normal_min_incidence_cos)
        self.confirm_observations = int(confirm_observations)
        self.tentative_contradictions = int(tentative_contradictions)
        self.confirmed_contradictions = int(confirmed_contradictions)
        self.max_evidence = int(max_evidence)
        self.resurrection_observations = int(resurrection_observations)
        self.tombstone_groups = int(tombstone_groups)
        self.max_pose_uncertainty = float(max_pose_uncertainty)
        self.candidate_pair_budget = int(candidate_pair_budget)
        self.candidate_nearest_k = int(candidate_nearest_k)

        self.pts = np.empty((0, 3), np.float64)
        self.tag = np.empty(0, np.uint8)
        self.hits = np.empty(0, np.int64)  # raw returns
        self.intensity_sum = np.empty(0, np.float64)
        self.intensity_hits = np.empty(0, np.int64)
        self.observations = np.empty(0, np.int64)
        self.support = np.empty(0, np.uint8)
        self.contradictions = np.empty(0, np.uint8)
        self.confirmed = np.empty(0, bool)
        self._last_support_group = np.empty(0, np.int64)
        self._last_contradiction_group = np.empty(0, np.int64)
        self._keys_by_row = np.empty(0, np.int64)
        self._key2row: dict[int, int] = {}
        self._block_voxels = max(1, int(np.ceil(self.normal_r / self.voxel)))
        self._block_size = self._block_voxels * self.voxel
        self._block_keys = np.empty((0, 3), np.int64)
        self._tile_voxels = max(1, int(np.ceil(2.0 / self.voxel)))
        self._tile_size = self._tile_voxels * self.voxel
        self._tile_keys = np.empty((0, 3), np.int64)
        self._tile2rows: dict[tuple[int, int, int], set[int]] = {}
        self._tile_index_keys_cache: np.ndarray | None = None
        self._storage: dict[str, np.ndarray] = {}
        self._capacity = 0
        self._size = 0
        self._block2rows: dict[tuple[int, int, int], set[int]] = {}
        self._block_versions: dict[tuple[int, int, int], int] = {}
        # voxel key -> (neighbourhood version signature, normal, valid)
        self._normal_cache: dict[
            int, tuple[tuple[int, ...], np.ndarray, bool]
        ] = {}
        self._normal_cache_enabled = True
        self._normal_cache_checks = 0
        self._normal_cache_hits = 0
        self._group = 0
        self._open_external_group: object | None = None
        self._open_group_key: object | None = None
        self._open_batches: list[RayBatch] = []
        self._closed_group_keys: set[object] = set()
        self._open_last_timestamp = -np.inf
        self._last_finalized_timestamp = -np.inf
        # voxel key -> (last group with resurrection support, support count)
        self._tombstones: dict[int, tuple[int, int]] = {}

        self.n_deleted = 0
        self.n_deleted_floater = 0
        self.n_evicted = 0
        self.n_invalid = 0
        self.n_uncertain_groups = 0
        self.last_metrics: dict[str, float | int | bool] = {}
        self._last_visibility_metrics: dict[str, int] = {}
        self._last_normal_metrics: dict[str, int] = {}

    def __getstate__(self) -> dict:
        """Serialize dense public views; capacity is rebuilt on load/copy."""
        state = self.__dict__.copy()
        state.pop("_storage", None)
        state["_capacity"] = 0
        state["_size"] = len(self.pts)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "_open_last_timestamp"):
            self._open_last_timestamp = -np.inf
            self._last_finalized_timestamp = -np.inf
        if not hasattr(self, "candidate_nearest_k"):
            self.candidate_nearest_k = 1
        if not hasattr(self, "_normal_cache_enabled"):
            self._normal_cache_enabled = True
            self._normal_cache_checks = 0
            self._normal_cache_hits = 0
        if not hasattr(self, "_tile_size"):
            self._tile_voxels = max(1, int(np.ceil(2.0 / self.voxel)))
            self._tile_size = self._tile_voxels * self.voxel
            self._tile_keys = np.floor(
                self.pts / self._tile_size
            ).astype(np.int64)
            self._tile2rows = {}
            for row, value in enumerate(self._tile_keys):
                key = tuple(int(v) for v in value)
                self._tile2rows.setdefault(key, set()).add(row)
            self._tile_index_keys_cache = None
        elif not hasattr(self, "_tile_index_keys_cache"):
            self._tile_index_keys_cache = None
        self._storage = {}
        self._capacity = 0
        self._size = len(self.pts)
        self._ensure_capacity(self._size)
        self._publish_views()

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._capacity:
            return
        capacity = max(1_024, required, max(1, self._capacity) * 2)
        storage: dict[str, np.ndarray] = {}
        for name in _MAP_ARRAYS:
            current = getattr(self, name)
            value = np.empty((capacity,) + current.shape[1:], current.dtype)
            if self._size:
                value[:self._size] = current[:self._size]
            storage[name] = value
        self._storage = storage
        self._capacity = capacity

    def _publish_views(self) -> None:
        if not self._storage:
            return
        for name in _MAP_ARRAYS:
            setattr(self, name, self._storage[name][:self._size])

    def update_batch(self, batch: RayBatch) -> None:
        """Stage ray-preserving observations in source-group order.

        The final group stays open because another scheduler fold may contain
        more observations from that same logical group.  A transition to the
        next group, or :meth:`flush`, evaluates and commits it atomically.
        """
        n = len(batch.endpoints)
        if n == 0:
            return
        try:
            one_group = bool(np.all(batch.groups == batch.groups[0]))
        except (TypeError, ValueError):
            one_group = False
        if one_group:
            self._stage_group(batch.groups[0], batch)
            return
        boundaries = [0]
        seen_keys = {self._group_key(batch.groups[0])}
        for index in range(1, n):
            if not self._same_group(batch.groups[index], batch.groups[index - 1]):
                key = self._group_key(batch.groups[index])
                if key in seen_keys:
                    raise ValueError(
                        "RayBatch rows for one observation group must be contiguous"
                    )
                seen_keys.add(key)
                boundaries.append(index)
        boundaries.append(n)
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            self._stage_group(
                batch.groups[start],
                RayBatch(
                    endpoints=batch.endpoints[start:stop],
                    origins=batch.origins[start:stop],
                    timestamps=batch.timestamps[start:stop],
                    groups=batch.groups[start:stop],
                    intensity=batch.intensity[start:stop],
                    floater=batch.floater[start:stop],
                    pose_uncertainty_m=batch.pose_uncertainty_m[start:stop],
                ),
            )

    @staticmethod
    def _group_key(value: object) -> object:
        """Return a stable hashable key for a scalar source group."""
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and np.isnan(value):
            return (float, "nan")
        try:
            hash(value)
        except TypeError:
            return (type(value), repr(value))
        return (type(value), value)

    def _stage_group(self, external_group: object, batch: RayBatch) -> None:
        key = self._group_key(external_group)
        if self._open_group_key is None:
            if key in self._closed_group_keys:
                raise ValueError("observation group was already finalized")
            self._open_external_group = external_group
            self._open_group_key = key
            self._open_last_timestamp = self._last_finalized_timestamp
        elif key != self._open_group_key:
            self.flush()
            if key in self._closed_group_keys:
                raise ValueError("observation groups cannot be reused noncontiguously")
            self._open_external_group = external_group
            self._open_group_key = key
            self._open_last_timestamp = self._last_finalized_timestamp
        finite_time = batch.timestamps[np.isfinite(batch.timestamps)]
        if len(finite_time):
            if np.any(np.diff(finite_time) < 0.0) or finite_time[0] < self._open_last_timestamp:
                raise ValueError("observation timestamps must be nondecreasing")
            self._open_last_timestamp = float(finite_time[-1])
        # Staging outlives ``update_batch``.  Own every aligned array so a
        # decoder or caller can safely reuse its input buffers while this
        # logical group remains open across scheduler folds.
        self._open_batches.append(RayBatch(
            endpoints=batch.endpoints.copy(),
            origins=batch.origins.copy(),
            timestamps=batch.timestamps.copy(),
            groups=batch.groups.copy(),
            intensity=batch.intensity.copy(),
            floater=batch.floater.copy(),
            pose_uncertainty_m=batch.pose_uncertainty_m.copy(),
        ))

    def flush(self) -> None:
        """Finalize the currently open logical group, if any."""
        if not self._open_batches:
            return
        batches = self._open_batches
        key = self._open_group_key
        self._open_batches = []
        self._open_external_group = None
        self._open_group_key = None
        finalized_timestamp = self._open_last_timestamp
        self._open_last_timestamp = self._last_finalized_timestamp
        self._group += 1
        group = self._group
        self._expire_tombstones(group)
        if len(batches) == 1:
            merged = batches[0]
        else:
            merged = RayBatch(
                endpoints=np.concatenate([b.endpoints for b in batches]),
                origins=np.concatenate([b.origins for b in batches]),
                timestamps=np.concatenate([b.timestamps for b in batches]),
                groups=np.concatenate([b.groups for b in batches]),
                intensity=np.concatenate([b.intensity for b in batches]),
                floater=np.concatenate([b.floater for b in batches]),
                pose_uncertainty_m=np.concatenate([
                    b.pose_uncertainty_m for b in batches
                ]),
            )
        self._process_group(merged, group)
        self._last_finalized_timestamp = finalized_timestamp
        if key is not None:
            self._closed_group_keys.add(key)

    def _valid_keys(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ijk = np.floor(xyz / self.voxel).astype(np.int64) + _OFF
        valid = ((ijk >= 0) & (ijk < _LIMIT)).all(axis=1)
        ijk = ijk[valid]
        keys = ((ijk[:, 0] << (2 * _BITS))
                | (ijk[:, 1] << _BITS) | ijk[:, 2])
        return keys, valid

    def _normals(
        self,
        sub_idx: np.ndarray,
        query_local: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return quality-checked normals for selected submap points.

        The neighbour tree contains the whole submap, but eigendecomposition is
        performed only for points that actually intersected a ray corridor.
        ``query_local=None`` retains the all-points form used by diagnostics.
        """
        p = self.pts[sub_idx]
        query = (np.arange(len(p), dtype=np.intp) if query_local is None
                 else np.asarray(query_local, np.intp))
        if len(p) < self.normal_min_pts or len(query) == 0:
            return np.zeros((len(query), 3)), np.zeros(len(query), bool)
        tree = cKDTree(p, compact_nodes=False, balanced_tree=False)
        k = min(self.normal_max_k, len(p))
        dist, idx = tree.query(
            p[query],
            k=k,
            distance_upper_bound=self.normal_r,
            workers=self._query_workers(len(query), len(p)),
        )
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        good = np.isfinite(dist)
        idx = np.where(good, idx, 0)
        count = good.sum(axis=1)
        neighbours = p[idx]
        weight = good[:, :, None]
        mean = ((neighbours * weight).sum(axis=1)
                / np.maximum(count, 1)[:, None])
        centered = (neighbours - mean[:, None, :]) * weight
        covariance = (np.einsum("nki,nkj->nij", centered, centered)
                      / np.maximum(count, 1)[:, None, None])
        values, vectors = np.linalg.eigh(covariance)
        largest = np.maximum(values[:, 2], np.finfo(float).eps)
        residual = np.sqrt(np.maximum(values[:, 0], 0.0))
        valid = (
            (count >= self.normal_min_pts)
            & np.isfinite(values).all(axis=1)
            & (values[:, 2] > np.finfo(float).eps)
            & (values[:, 1] / largest >= self.normal_min_second_ratio)
            & (residual <= self.normal_max_residual)
        )
        normals = vectors[:, :, 0]
        normals[~valid] = 0.0
        return normals, valid

    def _normal_neighbour_blocks(
        self, block: tuple[int, int, int]
    ) -> list[tuple[int, int, int]]:
        radius = int(np.ceil(self.normal_r / self._block_size))
        bx, by, bz = block
        return [
            (bx + dx, by + dy, bz + dz)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
        ]

    def _local_normals(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quality-checked normals from reusable local spatial blocks.

        Cache entries are keyed by stable voxel key. A signature over every
        block in the normal-radius halo invalidates them after centroid motion,
        insertion, deletion, or row swapping in that neighbourhood.
        """
        rows = np.asarray(rows, np.intp)
        if not self._normal_cache_enabled:
            return self._uncached_local_normals(rows)
        normals = np.zeros((len(rows), 3), np.float64)
        valid = np.zeros(len(rows), bool)
        stale_positions: list[int] = []
        stale_neighbours: list[list[tuple[int, int, int]]] = []
        stale_signatures: list[tuple[int, ...]] = []
        for position, row in enumerate(rows):
            block = tuple(int(v) for v in self._block_keys[row])
            neighbours = self._normal_neighbour_blocks(block)
            signature = tuple(self._block_versions.get(key, 0) for key in neighbours)
            cached = self._normal_cache.get(int(self._keys_by_row[row]))
            if cached is not None and cached[0] == signature:
                normals[position] = cached[1]
                valid[position] = cached[2]
            else:
                stale_positions.append(position)
                stale_neighbours.append(neighbours)
                stale_signatures.append(signature)
        if not stale_positions:
            self._track_normal_cache(len(rows), len(rows))
            self._last_normal_metrics = {
                "normal_queries": len(rows),
                "normal_cache_hits": len(rows),
                "normal_local_voxels": 0,
                "normal_tree_rebuilds": 0,
            }
            return normals, valid

        local_blocks = set()
        for neighbours in stale_neighbours:
            local_blocks.update(neighbours)
        local_rows = sorted({
            row
            for block in local_blocks
            for row in self._block2rows.get(block, ())
        })
        self._last_normal_metrics = {
            "normal_queries": len(rows),
            "normal_cache_hits": len(rows) - len(stale_positions),
            "normal_local_voxels": len(local_rows),
            "normal_tree_rebuilds": int(len(local_rows) >= self.normal_min_pts),
        }
        stale_rows = rows[np.asarray(stale_positions, np.intp)]
        if len(local_rows) >= self.normal_min_pts:
            local_rows_array = np.asarray(local_rows, np.intp)
            p = self.pts[local_rows_array]
            k = min(self.normal_max_k, len(p))
            dist, idx = cKDTree(
                p, compact_nodes=False, balanced_tree=False
            ).query(
                self.pts[stale_rows],
                k=k,
                distance_upper_bound=self.normal_r,
                workers=self._query_workers(len(stale_rows), len(p)),
            )
            if k == 1:
                dist, idx = dist[:, None], idx[:, None]
            good = np.isfinite(dist)
            idx = np.where(good, idx, 0)
            count = good.sum(axis=1)
            neighbours = p[idx]
            weight = good[:, :, None]
            mean = (
                (neighbours * weight).sum(axis=1)
                / np.maximum(count, 1)[:, None]
            )
            centered = (neighbours - mean[:, None, :]) * weight
            covariance = (
                np.einsum("nki,nkj->nij", centered, centered)
                / np.maximum(count, 1)[:, None, None]
            )
            values, vectors = np.linalg.eigh(covariance)
            largest = np.maximum(values[:, 2], np.finfo(float).eps)
            residual = np.sqrt(np.maximum(values[:, 0], 0.0))
            stale_valid = (
                (count >= self.normal_min_pts)
                & np.isfinite(values).all(axis=1)
                & (values[:, 2] > np.finfo(float).eps)
                & (values[:, 1] / largest >= self.normal_min_second_ratio)
                & (residual <= self.normal_max_residual)
            )
            stale_normals = vectors[:, :, 0]
            stale_normals[~stale_valid] = 0.0
        else:
            stale_normals = np.zeros((len(stale_rows), 3), np.float64)
            stale_valid = np.zeros(len(stale_rows), bool)

        for local, position in enumerate(stale_positions):
            normals[position] = stale_normals[local]
            valid[position] = stale_valid[local]
            key = int(self._keys_by_row[rows[position]])
            self._normal_cache[key] = (
                stale_signatures[local], stale_normals[local].copy(),
                bool(stale_valid[local]),
            )
        self._track_normal_cache(
            len(rows), len(rows) - len(stale_positions)
        )
        return normals, valid

    def _track_normal_cache(self, checks: int, hits: int) -> None:
        self._normal_cache_checks += checks
        self._normal_cache_hits += hits
        if (
            self._normal_cache_checks >= 500
            and self._normal_cache_hits / self._normal_cache_checks < 0.02
        ):
            # On moving Lance data nearly every candidate neighbourhood changes
            # every group. Signature construction then costs more than it saves.
            self._normal_cache_enabled = False
            self._normal_cache.clear()

    def _uncached_local_normals(
        self, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(rows) == 0:
            self._last_normal_metrics = {
                "normal_queries": 0, "normal_cache_hits": 0,
                "normal_local_voxels": 0, "normal_tree_rebuilds": 0,
            }
            return np.empty((0, 3)), np.empty(0, bool)
        query_blocks = np.unique(self._block_keys[rows], axis=0)
        local_blocks: set[tuple[int, int, int]] = set()
        for value in query_blocks:
            local_blocks.update(self._normal_neighbour_blocks(
                tuple(int(v) for v in value)
            ))
        local_rows = np.asarray(sorted({
            row for block in local_blocks
            for row in self._block2rows.get(block, ())
        }), np.intp)
        self._last_normal_metrics = {
            "normal_queries": len(rows), "normal_cache_hits": 0,
            "normal_local_voxels": len(local_rows),
            "normal_tree_rebuilds": int(len(local_rows) >= self.normal_min_pts),
        }
        if len(local_rows) < self.normal_min_pts:
            return np.zeros((len(rows), 3)), np.zeros(len(rows), bool)
        lookup = np.full(len(self.pts), -1, np.intp)
        lookup[local_rows] = np.arange(len(local_rows), dtype=np.intp)
        return self._normals(local_rows, lookup[rows])

    def _touch_block(self, block: tuple[int, int, int]) -> None:
        if not self._normal_cache_enabled:
            return
        self._block_versions[block] = self._block_versions.get(block, 0) + 1

    def _rows_in_box(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Return rows in coarse spatial tiles intersecting an axis-aligned box."""
        low = np.floor(np.asarray(lower) / self._tile_size).astype(np.int64)
        high = np.floor(np.asarray(upper) / self._tile_size).astype(np.int64)
        if self._tile_index_keys_cache is None:
            self._tile_index_keys_cache = np.asarray(
                list(self._tile2rows), dtype=np.int64
            ).reshape(-1, 3)
        keys = self._tile_index_keys_cache
        inside = ((keys >= low) & (keys <= high)).all(axis=1)
        if len(keys) and inside.all():
            return np.arange(len(self.pts), dtype=np.intp)
        selected = keys[inside]
        count = sum(
            len(self._tile2rows[tuple(int(v) for v in key)]) for key in selected
        )
        if count >= len(self.pts) // 2:
            return np.arange(len(self.pts), dtype=np.intp)
        return np.fromiter(
            (
                row for key in selected
                for row in self._tile2rows[tuple(int(v) for v in key)]
            ),
            np.intp,
            count=count,
        )

    def update(
        self,
        origin: np.ndarray,
        xyz: np.ndarray,
        floater: np.ndarray | None = None,
        *,
        intensity: np.ndarray | None = None,
        observation_group: object | None = None,
        timestamp: float | None = None,
        pose_uncertainty_m: float = 0.0,
    ) -> None:
        """Fold or stage one independent observation group into the map.

        A call contributes at most one vote per voxel. ``origin`` may be one
        viewpoint or an ``(N, 3)`` array aligned with endpoints, so deskewed or
        scheduled batches do not invent a shared viewpoint. ``observation_group``
        keeps the group open across calls; the next identifier or :meth:`flush`
        commits it. Calls without an identifier are complete groups and commit
        immediately. Excess or unknown pose uncertainty suppresses destructive
        votes for the whole group.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            raise ValueError(f"xyz must have shape (N, 3+), got {xyz.shape!r}")
        xyz = xyz[:, :3]
        n = len(xyz)
        origins = np.asarray(origin, dtype=np.float64)
        if origins.shape == (3,):
            origins = np.broadcast_to(origins, (n, 3))
        elif origins.shape != (n, 3):
            raise ValueError(
                f"origin must have shape (3,) or ({n}, 3), got {origins.shape!r}"
            )
        if floater is None:
            floater = np.zeros(n, bool)
        else:
            floater = np.asarray(floater, bool)
            if floater.shape != (n,):
                raise ValueError("floater must have one value per point")
        if intensity is None:
            intensity = np.full(n, np.nan, np.float64)
        else:
            intensity = np.asarray(intensity, np.float64)
            if intensity.shape != (n,):
                raise ValueError("intensity must have one value per point")

        stamp = np.nan if timestamp is None else float(timestamp)
        batch = RayBatch(
            endpoints=xyz,
            origins=np.asarray(origins, np.float64),
            timestamps=np.full(n, stamp, np.float64),
            groups=np.full(n, 0 if observation_group is None else observation_group),
            intensity=intensity,
            floater=floater,
            pose_uncertainty_m=np.full(n, pose_uncertainty_m, np.float64),
        )
        if observation_group is not None:
            self._stage_group(observation_group, batch)
            return

        # A group without an external identifier is known to be complete.
        # Close any explicitly staged predecessor, then process this call now.
        self.flush()
        self._group += 1
        group = self._group
        self._expire_tombstones(group)
        self._process_group(batch, group)

    def _process_group(self, batch: RayBatch, group: int) -> None:
        """Evaluate one complete group against one pre-group map and commit."""
        group_started = time.perf_counter()
        voxels_before = len(self.pts)
        xyz = batch.endpoints
        origins = batch.origins
        floater = batch.floater
        intensity = batch.intensity

        delta = xyz - origins
        ranges = np.linalg.norm(delta, axis=1)
        valid = (np.isfinite(xyz).all(axis=1)
                 & np.isfinite(origins).all(axis=1)
                 & np.isfinite(ranges) & (ranges > 1e-6))
        self.n_invalid += int((~valid).sum())
        xyz, floater, intensity = xyz[valid], floater[valid], intensity[valid]
        origins, delta, ranges = origins[valid], delta[valid], ranges[valid]
        if len(xyz) == 0:
            self.last_metrics = {
                "group": group, "input_rays": len(batch.endpoints),
                "valid_rays": 0, "global_voxels_before": voxels_before,
                "global_voxels_after": len(self.pts),
                "total_ms": (time.perf_counter() - group_started) * 1000.0,
            }
            return

        add_mask = ranges <= self.add_max
        ray_mask = ranges <= self.ray_max
        add_xyz, add_floater = xyz[add_mask], floater[add_mask]
        add_intensity, add_ranges = intensity[add_mask], ranges[add_mask]
        if len(self.pts) == 0:
            add_started = time.perf_counter()
            self._add(
                add_xyz, add_floater, add_intensity, add_ranges, group
            )
            self.last_metrics = {
                "group": group, "input_rays": len(batch.endpoints),
                "valid_rays": len(xyz), "global_voxels_before": voxels_before,
                "global_voxels_after": len(self.pts), "supported": 0,
                "contradicted": 0, "deleted": 0,
                "add_ms": (time.perf_counter() - add_started) * 1000.0,
                "total_ms": (time.perf_counter() - group_started) * 1000.0,
            }
            return

        support_started = time.perf_counter()
        supported = self._supported_rows(add_xyz)
        self._record_support(supported, group)
        support_ms = (time.perf_counter() - support_started) * 1000.0
        uncertainty_values = batch.pose_uncertainty_m[valid]
        uncertainty = (
            float(uncertainty_values.max())
            if len(uncertainty_values) and np.isfinite(uncertainty_values).all()
            else np.inf
        )
        negatives_allowed = (
            np.isfinite(uncertainty)
            and 0.0 <= uncertainty <= self.max_pose_uncertainty
        )
        if not negatives_allowed:
            self.n_uncertain_groups += 1
        contradicted = np.empty(0, np.intp)
        visibility_metrics = {
            "local_voxels": 0, "candidate_pairs": 0, "exact_pairs": 0,
            "normal_queries": 0, "normal_cache_hits": 0,
            "normal_local_voxels": 0, "normal_tree_rebuilds": 0,
        }
        visibility_started = time.perf_counter()
        if negatives_allowed and ray_mask.any():
            ray_origins = origins[ray_mask]
            ray_delta, ray_ranges = delta[ray_mask], ranges[ray_mask]
            miss_parts: list[np.ndarray] = []
            blocked_parts: list[np.ndarray] = []
            # A logical evidence group may contain many source viewpoints.
            # Query exact-origin subgroups, then reduce the evidence once.
            _unique_origins, origin_group = np.unique(
                ray_origins, axis=0, return_inverse=True
            )
            for origin_id in range(int(origin_group.max()) + 1):
                take = origin_group == origin_id
                miss, blocked = self._visibility_rows(
                    ray_origins[take], ray_delta[take], ray_ranges[take],
                    excluded_rows=supported,
                )
                for name in visibility_metrics:
                    visibility_metrics[name] += self._last_visibility_metrics.get(
                        name, 0
                    )
                if len(miss):
                    miss_parts.append(miss)
                if len(blocked):
                    blocked_parts.append(blocked)
            miss_rows = (
                np.unique(np.concatenate(miss_parts)) if miss_parts
                else np.empty(0, np.intp)
            )
            blocked_rows = (
                np.unique(np.concatenate(blocked_parts)) if blocked_parts
                else np.empty(0, np.intp)
            )
            contradicted = np.setdiff1d(
                miss_rows, np.union1d(blocked_rows, supported), assume_unique=True
            )
        self._record_contradictions(contradicted, group)
        visibility_ms = (time.perf_counter() - visibility_started) * 1000.0

        threshold = np.where(
            self.confirmed,
            self.confirmed_contradictions,
            self.tentative_contradictions,
        )
        doomed = np.flatnonzero(self.contradictions >= threshold)
        delete_started = time.perf_counter()
        deleted = len(doomed)
        if len(doomed):
            self.n_deleted += len(doomed)
            self.n_deleted_floater += int(self.tag[doomed].sum())
            self._delete(doomed, remember=True, group=group)
        delete_ms = (time.perf_counter() - delete_started) * 1000.0
        add_started = time.perf_counter()
        self._add(add_xyz, add_floater, add_intensity, add_ranges, group)
        add_ms = (time.perf_counter() - add_started) * 1000.0
        self.last_metrics = {
            "group": group,
            "input_rays": len(batch.endpoints),
            "valid_rays": len(xyz),
            "global_voxels_before": voxels_before,
            "global_voxels_after": len(self.pts),
            "supported": len(supported),
            "contradicted": len(contradicted),
            "deleted": deleted,
            "negative_evidence_allowed": negatives_allowed,
            "support_ms": support_ms,
            "visibility_ms": visibility_ms,
            "delete_ms": delete_ms,
            "add_ms": add_ms,
            "total_ms": (time.perf_counter() - group_started) * 1000.0,
            **visibility_metrics,
        }

    def _supported_rows(self, xyz: np.ndarray) -> np.ndarray:
        if len(xyz) == 0 or len(self.pts) == 0:
            return np.empty(0, np.intp)
        rows: list[np.ndarray] = []
        keys, valid = self._valid_keys(xyz)
        if valid.any():
            same = np.fromiter(
                (self._key2row.get(int(key), -1) for key in keys),
                dtype=np.int64,
                count=len(keys),
            )
            rows.append(same[same >= 0])
        # The observation tree is much cheaper to rebuild for each source
        # group than the continually growing map tree.
        candidate_rows = self._rows_in_box(
            np.nanmin(xyz, axis=0) - self.endpoint_margin,
            np.nanmax(xyz, axis=0) + self.endpoint_margin,
        )
        if len(candidate_rows) == 0:
            return np.unique(np.concatenate(rows))
        limit = np.nextafter(self.endpoint_margin, np.inf)
        distance, _nearest = cKDTree(
            xyz, compact_nodes=False, balanced_tree=False
        ).query(
            self.pts[candidate_rows],
            k=1,
            distance_upper_bound=limit,
            workers=self._query_workers(len(candidate_rows), len(xyz)),
        )
        rows.append(candidate_rows[distance <= self.endpoint_margin])
        return np.unique(np.concatenate(rows))

    def _visibility_rows(
        self,
        origins: np.ndarray,
        delta: np.ndarray,
        ranges: np.ndarray,
        *,
        excluded_rows: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(miss rows, ambiguous/surface rows)`` for exact rays.

        Calls contain one exact source viewpoint. Each mapped point receives a
        range-derived angular radius that is a conservative superset of the
        final finite radial corridor. Candidate tensors are processed under a
        fixed pair budget; no geometrically valid pair is truncated.
        """
        self._last_visibility_metrics = {
            "local_voxels": 0, "candidate_pairs": 0, "exact_pairs": 0,
            "normal_queries": 0, "normal_cache_hits": 0,
            "normal_local_voxels": 0, "normal_tree_rebuilds": 0,
        }
        if len(ranges) == 0:
            empty = np.empty(0, np.intp)
            return empty, empty
        reference = origins[0]
        if not np.all(origins == reference):
            raise ValueError("visibility query must contain one exact origin")
        ray_direction = delta / ranges[:, None]
        pair_budget = self.candidate_pair_budget
        local_rows = self._rows_in_box(
            reference - self.del_max, reference + self.del_max
        )
        map_delta = self.pts[local_rows] - reference
        map_range = np.linalg.norm(map_delta, axis=1)
        eligible = (
            np.isfinite(map_range)
            & (map_range > 1e-6)
            & (map_range <= self.del_max)
        )
        if excluded_rows is not None and len(excluded_rows):
            eligible &= ~np.isin(
                local_rows, np.asarray(excluded_rows, np.intp), assume_unique=False
            )
        sub = local_rows[np.flatnonzero(eligible)]
        self._last_visibility_metrics["local_voxels"] = len(sub)
        if len(sub) == 0:
            empty = np.empty(0, np.intp)
            return empty, empty

        # Invert the former map-tree query. The incoming scan is much smaller
        # than the accumulated map, and each mapped point can now use its exact
        # range-derived chord radius rather than a global worst-case radius.
        local_take = np.flatnonzero(eligible)
        map_direction = map_delta[local_take] / map_range[local_take, None]
        ratio = np.minimum(
            1.0, (self.radial + 1e-12) / map_range[local_take]
        )
        radial_chord = np.sqrt(
            np.maximum(
                0.0,
                2.0 - 2.0 * np.sqrt(np.maximum(0.0, 1.0 - ratio * ratio)),
            )
        )
        # ``fsr`` was the legacy fixed broad-phase radius. It is not part of
        # the documented exact rule and, when used as a minimum, produces tens
        # of thousands of false candidates at long range. The derived chord is
        # already the complete bound; nudge it outward for inclusive rho.
        candidate_radius = np.nextafter(radial_chord, np.inf)
        tree = cKDTree(
            ray_direction, compact_nodes=False, balanced_tree=False
        )
        workers = self._query_workers(len(sub), len(ray_direction))
        k = min(self.candidate_nearest_k, len(ray_direction))
        # Query map rows in bounded batches.  The nearest-k result proves that
        # rows whose kth neighbour misses are complete.  For overflow rows,
        # first ask only for counts, then materialize neighbour lists whose
        # aggregate size fits the pair budget.  A single pathological row is
        # regenerated from bounded ray slices instead of allocating its full
        # list at once.
        query_batch = max(1, pair_budget // k)

        def broad_candidate_chunks():
            for map_start in range(0, len(sub), query_batch):
                map_stop = min(len(sub), map_start + query_batch)
                batch_direction = map_direction[map_start:map_stop]
                batch_radius = candidate_radius[map_start:map_stop]
                batch_rows = sub[map_start:map_stop]
                distance, nearest = tree.query(
                    batch_direction, k=k, workers=workers
                )
                if k == 1:
                    distance, nearest = distance[:, None], nearest[:, None]
                within = distance <= batch_radius[:, None]
                overflow = (
                    within[:, -1]
                    if k < len(ray_direction)
                    else np.zeros(len(batch_rows), bool)
                )
                regular = ~overflow
                regular_within = within[regular]
                if regular_within.any():
                    yield (
                        nearest[regular][regular_within].astype(
                            np.intp, copy=False
                        ),
                        np.repeat(batch_rows[regular], k)[
                            regular_within.ravel()
                        ],
                    )
                overflow_rows = np.flatnonzero(overflow)
                if len(overflow_rows) == 0:
                    continue
                overflow_direction = batch_direction[overflow_rows]
                overflow_radius = batch_radius[overflow_rows]
                counts = np.asarray(tree.query_ball_point(
                    overflow_direction,
                    overflow_radius,
                    workers=self._query_workers(
                        len(overflow_rows), len(ray_direction)
                    ),
                    return_length=True,
                ), dtype=np.intp)
                start = 0
                while start < len(overflow_rows):
                    count = int(counts[start])
                    if count > pair_budget:
                        direction = overflow_direction[start]
                        radius = overflow_radius[start]
                        for ray_start in range(0, len(ray_direction), pair_budget):
                            ray_stop = min(
                                len(ray_direction), ray_start + pair_budget
                            )
                            distance = np.linalg.norm(
                                ray_direction[ray_start:ray_stop] - direction,
                                axis=1,
                            )
                            rays = np.flatnonzero(distance <= radius) + ray_start
                            if len(rays):
                                yield (
                                    rays.astype(np.intp, copy=False),
                                    np.full(
                                        len(rays),
                                        batch_rows[overflow_rows[start]],
                                        np.intp,
                                    ),
                                )
                        start += 1
                        continue
                    total = 0
                    stop = start
                    while (
                        stop < len(overflow_rows)
                        and int(counts[stop]) <= pair_budget
                        and total + int(counts[stop]) <= pair_budget
                    ):
                        total += int(counts[stop])
                        stop += 1
                    neighbours = tree.query_ball_point(
                        overflow_direction[start:stop],
                        overflow_radius[start:stop],
                        workers=self._query_workers(
                            stop - start, len(ray_direction)
                        ),
                        return_sorted=False,
                    )
                    lengths = counts[start:stop]
                    if total:
                        yield (
                            np.concatenate(neighbours).astype(
                                np.intp, copy=False
                            ),
                            np.repeat(
                                batch_rows[overflow_rows[start:stop]], lengths
                            ),
                        )
                    start = stop

        def exact_candidate_chunks():
            for ray_index, map_index in broad_candidate_chunks():
                exact = self._exact_candidate_pairs(
                    ray_index, map_index, origins, ray_direction, ranges
                )
                if len(exact[0]):
                    yield exact

        # Pass one retains only one bit per candidate map row.  Both tentative
        # and confirmed geometry receive the same valid-plane ambiguity
        # protection; their configured deletion thresholds remain distinct.
        normal_mask = np.zeros(len(self.pts), bool)
        candidate_pairs = 0
        exact_pairs = 0
        for ray_index, map_index in broad_candidate_chunks():
            candidate_pairs += len(ray_index)
            exact = self._exact_candidate_pairs(
                ray_index, map_index, origins, ray_direction, ranges
            )
            exact_pairs += len(exact[0])
            normal_mask[exact[1]] = True
        self._last_visibility_metrics["candidate_pairs"] = candidate_pairs
        self._last_visibility_metrics["exact_pairs"] = exact_pairs
        normal_rows = np.flatnonzero(normal_mask)
        if len(normal_rows):
            normals, normal_valid = self._local_normals(normal_rows)
            self._last_visibility_metrics.update(self._last_normal_metrics)
        else:
            normals = np.empty((0, 3), np.float64)
            normal_valid = np.empty(0, bool)
        if exact_pairs == 0:
            empty = np.empty(0, np.intp)
            return empty, empty
        normal_lookup = np.full(len(self.pts), -1, np.intp)
        normal_lookup[normal_rows] = np.arange(len(normal_rows), dtype=np.intp)

        # Pass two regenerates bounded candidates and reduces pair decisions to
        # row masks immediately; no list grows with total pair count.
        miss_mask = np.zeros(len(self.pts), bool)
        blocked_mask = np.zeros(len(self.pts), bool)
        for ray_index, map_index, candidate_delta in exact_candidate_chunks():
            miss, blocked = self._classify_visible_pairs(
                ray_index,
                map_index,
                candidate_delta,
                ray_direction,
                ranges,
                normals,
                normal_valid,
                normal_lookup,
            )
            miss_mask[miss] = True
            blocked_mask[blocked] = True

        return np.flatnonzero(miss_mask), np.flatnonzero(blocked_mask)

    def _exact_candidate_pairs(
        self,
        ray_index: np.ndarray,
        map_index: np.ndarray,
        origins: np.ndarray,
        ray_direction: np.ndarray,
        ranges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter one bounded broad-phase pair set by the exact ray corridor."""
        candidate_delta = self.pts[map_index] - origins[ray_index]
        candidate_range = np.linalg.norm(candidate_delta, axis=1)
        along = np.einsum(
            "ij,ij->i", candidate_delta, ray_direction[ray_index]
        )
        perpendicular = candidate_delta - along[:, None] * ray_direction[ray_index]
        visible = (
            (along > 1e-6)
            & (candidate_range <= self.del_max)
            & (np.linalg.norm(perpendicular, axis=1) <= self.radial)
            & (along < ranges[ray_index] - self.endpoint_margin)
        )
        if not visible.any():
            empty = np.empty(0, np.intp)
            return empty, empty, np.empty((0, 3), np.float64)
        return ray_index[visible], map_index[visible], candidate_delta[visible]

    def _classify_visible_pairs(
        self,
        ray_index: np.ndarray,
        map_index: np.ndarray,
        candidate_delta: np.ndarray,
        ray_direction: np.ndarray,
        ranges: np.ndarray,
        normals: np.ndarray,
        normal_valid: np.ndarray,
        normal_lookup: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply cached local-plane decisions to exact finite-ray pairs."""
        normal_index = normal_lookup[map_index]
        pair_normals = np.zeros((len(map_index), 3), np.float64)
        pair_normal_valid = np.zeros(len(map_index), bool)
        have_normal = normal_index >= 0
        pair_normals[have_normal] = normals[normal_index[have_normal]]
        pair_normal_valid[have_normal] = normal_valid[normal_index[have_normal]]
        denominator = np.einsum(
            "ij,ij->i", ray_direction[ray_index], pair_normals
        )
        grazing = pair_normal_valid & (
            np.abs(denominator) < self.normal_min_incidence_cos
        )
        plane_range = np.full(len(map_index), np.nan)
        intersecting = pair_normal_valid & ~grazing
        if intersecting.any():
            numerator = np.einsum(
                "ij,ij->i",
                candidate_delta[intersecting],
                pair_normals[intersecting],
            )
            plane_range[intersecting] = numerator / denominator[intersecting]
        same_surface = (
            intersecting
            & np.isfinite(plane_range)
            & (plane_range > 0.0)
            & (np.abs(ranges[ray_index] - plane_range) <= self.half_w)
        )
        blocked = np.unique(map_index[grazing | same_surface])
        misses = np.unique(map_index[~(grazing | same_surface)])
        return misses.astype(np.intp, copy=False), blocked.astype(np.intp, copy=False)

    @staticmethod
    def _query_workers(n_query: int, n_data: int) -> int:
        """Avoid thread-pool overhead on small spatial queries."""
        return 16 if n_query >= 4096 and n_data >= 512 else 1

    @staticmethod
    def _same_group(left: object, right: object) -> bool:
        """Safe scalar equality for contiguous external group identifiers."""
        try:
            equal = left == right
            return bool(equal) if np.ndim(equal) == 0 else False
        except (TypeError, ValueError):
            return False

    def _record_contradictions(self, rows: np.ndarray, group: int) -> None:
        if len(rows) == 0:
            return
        rows = rows[
            (self._last_contradiction_group[rows] != group)
            & (self._last_support_group[rows] != group)
        ]
        if len(rows) == 0:
            return
        self.contradictions[rows] = np.minimum(
            self.max_evidence,
            self.contradictions[rows].astype(np.int16) + 1,
        ).astype(np.uint8)
        self._last_contradiction_group[rows] = group

    def _record_support(self, rows: np.ndarray, group: int) -> None:
        """Apply one positive vote, including endpoints across voxel edges."""
        if len(rows) == 0:
            return
        rows = rows[self._last_support_group[rows] != group]
        if len(rows) == 0:
            return
        self.observations[rows] += 1
        self.support[rows] = np.minimum(
            self.max_evidence,
            self.support[rows].astype(np.int16) + 1,
        ).astype(np.uint8)
        self.contradictions[rows] = np.maximum(
            0, self.contradictions[rows].astype(np.int16) - 1,
        ).astype(np.uint8)
        self.confirmed[rows] |= self.support[rows] >= self.confirm_observations
        self._last_support_group[rows] = group

    def _delete(
        self,
        rows: np.ndarray,
        *,
        remember: bool = False,
        group: int = 0,
    ) -> None:
        rows = np.unique(np.asarray(rows, np.intp))
        if len(rows) == 0:
            return
        if remember:
            for key in self._keys_by_row[rows]:
                self._tombstones[int(key)] = (group, 0)
        # Descending swap-removal leaves all aligned arrays dense without
        # copying them or rebuilding the complete voxel dictionary.
        for row in np.sort(rows)[::-1]:
            row = int(row)
            last = len(self.pts) - 1
            deleted_key = int(self._keys_by_row[row])
            deleted_block = tuple(int(v) for v in self._block_keys[row])
            deleted_tile = tuple(int(v) for v in self._tile_keys[row])
            self._key2row.pop(deleted_key, None)
            self._normal_cache.pop(deleted_key, None)
            members = self._block2rows.get(deleted_block)
            if members is not None:
                members.discard(row)
                if not members:
                    del self._block2rows[deleted_block]
            self._touch_block(deleted_block)
            tile_members = self._tile2rows.get(deleted_tile)
            if tile_members is not None:
                tile_members.discard(row)
                if not tile_members:
                    del self._tile2rows[deleted_tile]
                    self._tile_index_keys_cache = None
            if row != last:
                moved_key = int(self._keys_by_row[last])
                moved_block = tuple(int(v) for v in self._block_keys[last])
                moved_tile = tuple(int(v) for v in self._tile_keys[last])
                moved_members = self._block2rows[moved_block]
                moved_members.discard(last)
                moved_members.add(row)
                moved_tile_members = self._tile2rows[moved_tile]
                moved_tile_members.discard(last)
                moved_tile_members.add(row)
                for name in _MAP_ARRAYS:
                    value = getattr(self, name)
                    value[row] = value[last]
                self._key2row[moved_key] = row
            self._size -= 1
            self._publish_views()

    def evict_farthest(self, max_points: int, origin: np.ndarray) -> int:
        """Enforce a display/memory cap without counting it as ray deletion."""
        max_points = max(1, int(max_points))
        if len(self.pts) <= max_points:
            return 0
        origin = np.asarray(origin, float).reshape(3)
        keep = np.argpartition(
            np.linalg.norm(self.pts - origin, axis=1), max_points - 1
        )[:max_points]
        remove = np.setdiff1d(np.arange(len(self.pts)), keep)
        self._delete(remove)
        self.n_evicted += len(remove)
        return len(remove)

    def _add(
        self,
        xyz: np.ndarray,
        floater: np.ndarray,
        intensity: np.ndarray,
        ranges: np.ndarray,
        group: int,
    ) -> None:
        if len(xyz) == 0:
            return
        near = np.isfinite(ranges) & (ranges <= self.add_max)
        xyz, floater, intensity = xyz[near], floater[near], intensity[near]
        if len(xyz) == 0:
            return
        keys, valid = self._valid_keys(xyz)
        self.n_invalid += int((~valid).sum())
        xyz, floater, intensity = xyz[valid], floater[valid], intensity[valid]
        if len(xyz) == 0:
            return

        unique, inverse = np.unique(keys, return_inverse=True)
        sums = np.zeros((len(unique), 3), np.float64)
        np.add.at(sums, inverse, xyz)
        counts = np.bincount(inverse, minlength=len(unique)).astype(np.int64)
        centroids = sums / counts[:, None]
        all_floater = np.ones(len(unique), np.uint8)
        np.minimum.at(all_floater, inverse, floater.astype(np.uint8))
        finite_intensity = np.isfinite(intensity)
        intensity_sums = np.zeros(len(unique), np.float64)
        np.add.at(
            intensity_sums,
            inverse[finite_intensity],
            intensity[finite_intensity],
        )
        intensity_counts = np.bincount(
            inverse[finite_intensity], minlength=len(unique)
        ).astype(np.int64)

        existing = np.fromiter(
            (self._key2row.get(int(key), -1) for key in unique),
            dtype=np.int64,
            count=len(unique),
        )
        hit = existing >= 0
        if hit.any():
            rows = existing[hit]
            old_counts = self.hits[rows]
            totals = old_counts + counts[hit]
            self.pts[rows] = (
                self.pts[rows] * old_counts[:, None]
                + centroids[hit] * counts[hit, None]
            ) / totals[:, None]
            self.hits[rows] = totals
            self.intensity_sum[rows] += intensity_sums[hit]
            self.intensity_hits[rows] += intensity_counts[hit]
            self.tag[rows] = np.minimum(self.tag[rows], all_floater[hit])
            if self._normal_cache_enabled:
                for value in np.unique(self._block_keys[rows], axis=0):
                    self._touch_block(tuple(int(v) for v in value))
            independent = self._last_support_group[rows] != group
            supported_rows = rows[independent]
            if len(supported_rows):
                self.observations[supported_rows] += 1
                self.support[supported_rows] = np.minimum(
                    self.max_evidence,
                    self.support[supported_rows].astype(np.int16) + 1,
                ).astype(np.uint8)
                self.contradictions[supported_rows] = np.maximum(
                    0,
                    self.contradictions[supported_rows].astype(np.int16) - 1,
                ).astype(np.uint8)
                self.confirmed[supported_rows] |= (
                    self.support[supported_rows] >= self.confirm_observations
                )
                self._last_support_group[supported_rows] = group

        new_indices = np.flatnonzero(~hit)
        accepted: list[int] = []
        initial_support: list[int] = []
        for index in new_indices:
            key = int(unique[index])
            tombstone = self._tombstones.get(key)
            if tombstone is None:
                accepted.append(int(index))
                initial_support.append(1)
                continue
            last_group, prior = tombstone
            if last_group == group:
                continue
            restored = prior + 1
            if restored < self.resurrection_observations:
                self._tombstones[key] = (group, restored)
                continue
            del self._tombstones[key]
            accepted.append(int(index))
            initial_support.append(min(self.max_evidence, restored))

        if accepted:
            take = np.asarray(accepted, np.intp)
            support = np.asarray(initial_support, np.uint8)
            base = len(self.pts)
            new_blocks = np.floor(
                centroids[take] / self._block_size
            ).astype(np.int64)
            new_tiles = np.floor(
                centroids[take] / self._tile_size
            ).astype(np.int64)
            values = {
                "pts": centroids[take],
                "tag": all_floater[take],
                "hits": counts[take],
                "intensity_sum": intensity_sums[take],
                "intensity_hits": intensity_counts[take],
                "observations": np.ones(len(take), np.int64),
                "support": support,
                "contradictions": np.zeros(len(take), np.uint8),
                "confirmed": support >= self.confirm_observations,
                "_last_support_group": np.full(len(take), group, np.int64),
                "_last_contradiction_group": np.full(len(take), -1, np.int64),
                "_keys_by_row": unique[take],
                "_block_keys": new_blocks,
                "_tile_keys": new_tiles,
            }
            end = base + len(take)
            self._ensure_capacity(end)
            for name, value in values.items():
                self._storage[name][base:end] = value
            self._size = end
            self._publish_views()
            for offset, key in enumerate(unique[take]):
                row = base + offset
                self._key2row[int(key)] = row
                block = tuple(int(v) for v in new_blocks[offset])
                self._block2rows.setdefault(block, set()).add(row)
                if self._normal_cache_enabled:
                    self._touch_block(block)
                tile = tuple(int(v) for v in new_tiles[offset])
                if tile not in self._tile2rows:
                    self._tile_index_keys_cache = None
                self._tile2rows.setdefault(tile, set()).add(row)

    def _expire_tombstones(self, group: int) -> None:
        expired = [
            key
            for key, (last_group, _support) in self._tombstones.items()
            if group - last_group > self.tombstone_groups
        ]
        for key in expired:
            del self._tombstones[key]

    def evidence(self) -> CarvingEvidence:
        """Return diagnostic evidence/state aligned with current points."""
        return CarvingEvidence(
            observations=self.observations.copy(),
            support=self.support.copy(),
            contradictions=self.contradictions.copy(),
            confirmed=self.confirmed.copy(),
        )

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(centroids, mean intensity, raw hit counts)``."""
        mean_intensity = np.full(len(self.pts), np.nan, np.float32)
        have_intensity = self.intensity_hits > 0
        mean_intensity[have_intensity] = (
            self.intensity_sum[have_intensity]
            / self.intensity_hits[have_intensity]
        ).astype(np.float32)
        return (
            self.pts.astype(np.float32),
            mean_intensity,
            self.hits.copy(),
        )
