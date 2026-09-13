"""Regression tests for conservative ray carving and its live scheduler."""
from __future__ import annotations

import threading
import time

import numpy as np

from rocklabel.geometry.carve import KFCMap, RayBatch
from rocklabel.live.carving import CarvingAccum


ORIGIN = np.zeros(3)


def test_offline_label_carving_requires_explicit_pose_assumption():
    from rocklabel.cli import build_parser

    parser = build_parser()
    conservative = parser.parse_args(["label", "run.mcap", "--carve"])
    assert conservative.carve_assumed_pose_uncertainty is None
    exact = parser.parse_args([
        "label", "run.mcap", "--carve",
        "--carve-assumed-pose-uncertainty", "0",
    ])
    assert exact.carve_assumed_pose_uncertainty == 0.0


def _has_point(map_: KFCMap, point: list[float], atol: float = 1e-6) -> bool:
    return bool(np.any(np.linalg.norm(map_.pts - point, axis=1) <= atol))


def test_nearer_return_cannot_carve_geometry_behind_it():
    map_ = KFCMap()
    map_.update(ORIGIN, np.array([[2.0, 0.0, 0.0]]))
    for _ in range(4):
        map_.update(ORIGIN, np.array([[1.0, 0.0, 0.0]]))
    assert _has_point(map_, [2.0, 0.0, 0.0])
    assert map_.n_deleted == 0


def test_matching_observation_preserves_and_confirms_accumulated_support():
    map_ = KFCMap()
    endpoint = np.array([[2.0, 0.0, 0.0]])
    map_.update(ORIGIN, endpoint, intensity=np.array([2.0]))
    map_.update(ORIGIN, endpoint, intensity=np.array([4.0]))
    xyz, intensity, hits = map_.result()
    assert xyz.shape == (1, 3)
    assert hits.tolist() == [2]
    assert intensity.tolist() == [3.0]
    assert map_.observations.tolist() == [2]
    assert map_.confirmed.tolist() == [True]
    assert map_.n_deleted == 0


def test_nearby_endpoint_across_voxel_boundary_is_positive_support():
    map_ = KFCMap(voxel=0.05, endpoint_margin=0.02)
    map_.update(ORIGIN, np.array([[1.049, 0.0, 0.0]]))
    map_.update(ORIGIN, np.array([[1.051, 0.0, 0.0]]))
    row = np.argmin(np.linalg.norm(map_.pts - [1.049, 0.0, 0.0], axis=1))
    assert map_.support[row] == 2
    assert map_.confirmed[row]


def test_confirmed_geometry_requires_repeated_real_pass_throughs():
    map_ = KFCMap()
    occupied = np.array([[1.0, 0.0, 0.0]])
    miss = np.array([[2.0, 0.0, 0.0]])
    map_.update(ORIGIN, occupied)
    map_.update(ORIGIN, occupied)
    for expected in (1, 2):
        map_.update(ORIGIN, miss)
        assert _has_point(map_, [1.0, 0.0, 0.0])
        row = np.argmin(np.linalg.norm(map_.pts - occupied[0], axis=1))
        assert map_.contradictions[row] == expected
    map_.update(ORIGIN, miss)
    assert not _has_point(map_, [1.0, 0.0, 0.0])
    assert map_.n_deleted == 1


def test_support_in_group_wins_over_competing_miss_ray():
    map_ = KFCMap()
    occupied = np.array([[1.0, 0.0, 0.0]])
    map_.update(ORIGIN, occupied)
    for _ in range(4):
        map_.update(ORIGIN, np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    assert _has_point(map_, [1.0, 0.0, 0.0])
    row = np.argmin(np.linalg.norm(map_.pts - occupied[0], axis=1))
    assert map_.contradictions[row] == 0


def test_one_group_can_retain_per_endpoint_origins():
    map_ = KFCMap()
    occupied = np.array([[1.0, 0.0, 0.0]])
    map_.update(ORIGIN, occupied)
    # The first ray sees through the old point from x=0; the second supports
    # it from x=2. Support in the same observation group must win.
    map_.update(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert _has_point(map_, [1.0, 0.0, 0.0])
    row = np.argmin(np.linalg.norm(map_.pts - occupied[0], axis=1))
    assert map_.contradictions[row] == 0


def test_split_calls_with_same_group_cast_only_one_vote():
    map_ = KFCMap(tentative_contradictions=2)
    map_.update(ORIGIN, np.array([[1.0, 0.0, 0.0]]), observation_group="scan-1")
    miss = np.array([[2.0, 0.0, 0.0]])
    map_.update(ORIGIN, miss, observation_group="scan-2")
    map_.update(ORIGIN, miss, observation_group="scan-2")
    assert _has_point(map_, [1.0, 0.0, 0.0])
    # A new group closes scan-2; duplicate chunks still cast only one vote.
    map_.update(ORIGIN, miss, observation_group="scan-3")
    row = np.argmin(np.linalg.norm(map_.pts - [1.0, 0.0, 0.0], axis=1))
    assert map_.contradictions[row] == 1
    map_.flush()
    assert not _has_point(map_, [1.0, 0.0, 0.0])


def test_ray_batch_keeps_aligned_origins_times_groups_and_intensity():
    batch = RayBatch(
        endpoints=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        origins=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        timestamps=np.array([1.0, 1.1]),
        groups=np.array([10, 11]),
        intensity=np.array([2.0, 4.0]),
        floater=np.array([False, False]),
        pose_uncertainty_m=np.array([0.0, 0.0]),
    )
    map_ = KFCMap()
    map_.update_batch(batch)
    map_.flush()
    assert map_.hits.tolist() == [2]
    assert map_.observations.tolist() == [2]
    assert map_.result()[1].tolist() == [3.0]


def test_ray_batch_rejects_noncontiguous_repeated_group():
    batch = RayBatch(
        endpoints=np.ones((3, 3)),
        origins=np.zeros((3, 3)),
        timestamps=np.arange(3.0),
        groups=np.array([1, 2, 1]),
        intensity=np.zeros(3),
        floater=np.zeros(3, bool),
        pose_uncertainty_m=np.zeros(3),
    )
    with np.testing.assert_raises_regex(ValueError, "contiguous"):
        KFCMap().update_batch(batch)


def test_pose_uncertainty_suppresses_only_negative_evidence():
    map_ = KFCMap(max_pose_uncertainty=0.02)
    occupied = np.array([[1.0, 0.0, 0.0]])
    map_.update(ORIGIN, occupied)
    for _ in range(4):
        map_.update(
            ORIGIN,
            np.array([[2.0, 0.0, 0.0]]),
            pose_uncertainty_m=0.10,
        )
    assert _has_point(map_, [1.0, 0.0, 0.0])
    assert _has_point(map_, [2.0, 0.0, 0.0])


def test_collinear_neighbourhood_is_not_a_valid_plane():
    map_ = KFCMap(normal_k_res=10.0)
    map_.update(
        ORIGIN,
        np.array([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]]),
    )
    _normals, valid = map_._normals(np.arange(3))
    assert not valid.any()


def test_collinear_points_do_not_gain_indefinite_plane_protection():
    map_ = KFCMap(normal_k_res=10.0)
    collinear = np.array([
        [1.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [1.2, 0.0, 0.0],
    ])
    map_.update(ORIGIN, collinear)
    farther = np.array([[2.0, 0.0, 0.0]])
    map_.update(ORIGIN, farther)
    map_.update(ORIGIN, farther)
    assert not any(_has_point(map_, point.tolist()) for point in collinear)


def test_tentative_planar_geometry_keeps_plane_protection():
    map_ = KFCMap(normal_k_res=10.0, tentative_contradictions=2)
    plane = np.array([
        [1.00, 0.00, 0.0],
        [0.95, -0.05, 0.0], [0.95, 0.05, 0.0],
        [1.05, -0.05, 0.0], [1.05, 0.05, 0.0],
    ])
    map_.update(ORIGIN, plane)
    assert not map_.confirmed.any()
    # The ray lies in the well-conditioned z=0 plane.  Its grazing geometry
    # is ambiguous for tentative and confirmed voxels alike.
    for _ in range(2):
        map_.update(ORIGIN, np.array([[2.0, 0.0, 0.0]]))
    assert _has_point(map_, [1.0, 0.0, 0.0])


def test_invalid_zero_length_and_out_of_range_inputs_are_not_added():
    map_ = KFCMap(add_max_range=1.0)
    map_.update(
        ORIGIN,
        np.array([
            [0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]),
    )
    assert len(map_.pts) == 0
    assert map_.n_invalid == 2
    map_.update(np.array([np.nan, 0.0, 0.0]), np.array([[0.5, 0.0, 0.0]]))
    assert len(map_.pts) == 0


def test_recently_deleted_voxel_needs_repeated_support_to_resurrect():
    map_ = KFCMap(tentative_contradictions=2, resurrection_observations=2)
    occupied = np.array([[1.0, 0.0, 0.0]])
    miss = np.array([[2.0, 0.0, 0.0]])
    map_.update(ORIGIN, occupied)
    map_.update(ORIGIN, miss)
    map_.update(ORIGIN, miss)
    assert not _has_point(map_, [1.0, 0.0, 0.0])
    map_.update(ORIGIN, occupied)
    assert not _has_point(map_, [1.0, 0.0, 0.0])
    map_.update(ORIGIN, occupied)
    assert _has_point(map_, [1.0, 0.0, 0.0])


def test_live_scheduler_retains_each_observations_origin_and_order(monkeypatch):
    accum = CarvingAccum(max_points=100, interval_s=10.0)
    calls: list[tuple[np.ndarray, np.ndarray, float, object]] = []
    original = accum._map._process_group

    def record(batch, group):
        calls.append((batch.origins.copy(), batch.endpoints.copy(),
                      float(batch.timestamps[-1]), batch.groups[0]))
        return original(batch, group)

    monkeypatch.setattr(accum._map, "_process_group", record)
    accum.add(np.array([[1.0, 0.0, 0.0]]), timestamp=0.0,
              origin=np.array([0.0, 0.0, 0.0]))
    accum.add(np.array([[2.0, 0.0, 0.0]]), timestamp=0.1,
              origin=np.array([0.1, 0.0, 0.0]))
    accum.add(np.array([[3.0, 0.0, 0.0]]), timestamp=0.2,
              origin=np.array([0.2, 0.0, 0.0]))
    accum.flush()
    assert [call[0].tolist() for call in calls] == [
        [[0.0, 0.0, 0.0]],
        [[0.1, 0.0, 0.0]],
        [[0.2, 0.0, 0.0]],
    ]
    assert [call[2] for call in calls] == [0.0, 0.1, 0.2]
    assert [call[3] for call in calls] == [0, 2, 4]


def test_live_evidence_group_batches_work_without_losing_origins(monkeypatch):
    accum = CarvingAccum(max_points=100, interval_s=10.0, evidence_group_s=0.05)
    calls: list[RayBatch] = []
    original = accum._map.update_batch

    def record(batch):
        calls.append(batch)
        return original(batch)

    monkeypatch.setattr(accum._map, "update_batch", record)
    # First input folds immediately; these two stay pending in one 50 ms group.
    accum.add(np.array([[9.0, 0.0, 0.0]]), timestamp=0.0, origin=ORIGIN)
    accum.flush()
    calls.clear()
    accum.add(np.array([[1.0, 0.0, 0.0]]), timestamp=0.10,
              origin=np.array([0.1, 0.0, 0.0]))
    accum.add(np.array([[2.0, 0.0, 0.0]]), timestamp=0.11,
              origin=np.array([0.2, 0.0, 0.0]))
    accum.flush()
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0].origins, [
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
    ])
    np.testing.assert_allclose(calls[0].endpoints, [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    np.testing.assert_allclose(calls[0].timestamps, [0.10, 0.11])
    assert calls[0].groups.tolist() == [2, 2]


def test_scheduling_boundary_does_not_split_an_evidence_vote():
    accum = CarvingAccum(
        max_points=100,
        interval_s=0.005,
        evidence_group_s=0.05,
        tentative_contradictions=2,
    )
    accum.add(np.array([[1.0, 0.0, 0.0]]), timestamp=0.0, origin=ORIGIN)
    # Both folds still belong to absolute 50 ms group zero. Its earlier support
    # blocks a later competing miss, and repeated misses cannot vote twice.
    accum.add(np.array([[2.0, 0.0, 0.0]]), timestamp=0.01, origin=ORIGIN)
    accum.add(np.array([[2.0, 0.0, 0.0]]), timestamp=0.02, origin=ORIGIN)
    accum.flush()
    assert _has_point(accum._map, [1.0, 0.0, 0.0])
    row = np.argmin(np.linalg.norm(accum._map.pts - [1.0, 0.0, 0.0], axis=1))
    assert accum._map.contradictions[row] == 0
    assert accum.stats()[0] == 1


def test_resumable_sync_keeps_same_time_bucket_open():
    accum = CarvingAccum(
        max_points=100, interval_s=10.0, evidence_group_s=0.05,
    )
    accum.add(np.array([[1.0, 0.0, 0.0]]), timestamp=1.001, origin=ORIGIN)
    accum.sync()
    # A pause/resume or completed forward seek can resume inside bucket 20.
    accum.add(np.array([[1.0, 0.0, 0.0]]), timestamp=1.011, origin=ORIGIN)
    accum.flush()
    assert accum._map.hits.tolist() == [2]
    assert accum._map.observations.tolist() == [1]
    accum.close()


def test_async_add_copies_the_callers_point_buffer(monkeypatch):
    accum = CarvingAccum(max_points=100, interval_s=10.0)
    entered = threading.Event()
    release = threading.Event()
    original = accum._owner_add

    def blocked_add(observation):
        entered.set()
        assert release.wait(timeout=2.0)
        return original(observation)

    monkeypatch.setattr(accum, "_owner_add", blocked_add)
    points = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    accum.add(points, timestamp=0.0, origin=ORIGIN)
    assert entered.wait(timeout=2.0)
    points[0, 0] = 2.0
    release.set()
    accum.flush()
    np.testing.assert_allclose(accum.snapshot()[0], [[1.0, 0.0, 0.0]])
    accum.close()


def test_staged_ray_batch_copies_caller_owned_arrays():
    endpoints = np.array([[1.0, 0.0, 0.0]])
    batch = RayBatch.one_group(endpoints, ORIGIN, group=4)
    map_ = KFCMap()
    map_.update_batch(batch)
    endpoints[0, 0] = 2.0
    batch.endpoints[0, 0] = 3.0
    map_.flush()
    np.testing.assert_allclose(map_.pts, [[1.0, 0.0, 0.0]])


def test_published_evidence_snapshot_is_immutable():
    accum = CarvingAccum(max_points=100)
    accum.add(np.array([[1.0, 0.0, 0.0]]), origin=ORIGIN)
    accum.flush()
    points, evidence = accum.evidence_snapshot()
    arrays = (
        points, evidence.observations, evidence.support,
        evidence.contradictions, evidence.confirmed,
    )
    assert all(not value.flags.writeable for value in arrays)
    accum.close()


def test_live_flush_includes_tail_and_preserves_intensity():
    accum = CarvingAccum(max_points=100, interval_s=10.0)
    accum.add(np.array([[1.0, 0.0, 0.0]]), np.array([2.0]), 0.0, ORIGIN)
    accum.add(np.array([[2.0, 0.0, 0.0]]), np.array([4.0]), 0.1, ORIGIN)
    # The only staged group is deliberately not published until it closes.
    assert len(accum.snapshot()[0]) == 0
    accum.flush()
    points, intensity = accum.snapshot()
    assert len(points) == 2
    assert sorted(intensity.tolist()) == [2.0, 4.0]
    assert accum.stats() == (2, 2, 0.1)


def test_live_point_cap_trims_immediately_and_reports_eviction_separately():
    accum = CarvingAccum(max_points=10, interval_s=10.0)
    points = np.column_stack([np.arange(1.0, 6.0), np.zeros(5), np.zeros(5)])
    accum.add(points, timestamp=0.0, origin=ORIGIN)
    accum.flush()
    accum.set_max_points(2)
    accum.flush()
    assert len(accum.snapshot()[0]) == 2
    assert accum.evicted == 3
    assert accum.deleted == 0


def test_live_snapshot_and_controls_do_not_wait_for_mapping_work(monkeypatch):
    accum = CarvingAccum(
        max_points=100, interval_s=0.001, evidence_group_s=0.05
    )
    entered = threading.Event()
    release = threading.Event()
    original = accum._map._process_group

    def slow_group(batch, group):
        entered.set()
        assert release.wait(timeout=2.0)
        return original(batch, group)

    monkeypatch.setattr(accum._map, "_process_group", slow_group)
    accum.add(np.array([[1.0, 0.0, 0.0]]), timestamp=0.0, origin=ORIGIN)
    accum.add(np.array([[2.0, 0.0, 0.0]]), timestamp=0.1, origin=ORIGIN)
    assert entered.wait(timeout=2.0)

    started = time.perf_counter()
    accum.snapshot()
    assert time.perf_counter() - started < 0.05
    started = time.perf_counter()
    accum.set_max_points(50)
    assert time.perf_counter() - started < 0.05
    assert accum.backlog_stats()[0] >= 1

    release.set()
    accum.flush()
    accum.close()


def _finalized_state(map_: KFCMap):
    map_.flush()
    order = np.lexsort(map_.pts.T[::-1]) if len(map_.pts) else np.empty(0, int)
    return (
        map_.pts[order], map_.hits[order], map_.observations[order],
        map_.support[order], map_.contradictions[order], map_.confirmed[order],
        map_.intensity_sum[order], map_.intensity_hits[order],
        map_.n_deleted, dict(map_._tombstones),
    )


def _prime_confirmed_with_two_misses(map_: KFCMap) -> None:
    occupied = np.array([[1.0, 0.0, 0.0]])
    miss = np.array([[2.0, 0.0, 0.0]])
    for group in ("support-1", "support-2"):
        map_.update(ORIGIN, occupied, observation_group=group)
    for group in ("prior-miss-1", "prior-miss-2"):
        map_.update(ORIGIN, miss, observation_group=group)
    map_.flush()


def test_miss_then_support_is_atomic_after_prior_contradictions():
    combined = KFCMap()
    split = KFCMap()
    for map_ in (combined, split):
        _prime_confirmed_with_two_misses(map_)

    combined.update(
        ORIGIN,
        np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        observation_group="deciding-group",
    )
    split.update(
        ORIGIN, np.array([[2.0, 0.0, 0.0]]),
        observation_group="deciding-group",
    )
    split.update(
        ORIGIN, np.array([[1.0, 0.0, 0.0]]),
        observation_group="deciding-group",
    )

    left = _finalized_state(combined)
    right = _finalized_state(split)
    for a, b in zip(left[:-2], right[:-2]):
        np.testing.assert_equal(a, b)
    assert left[-2:] == right[-2:]
    assert _has_point(split, [1.0, 0.0, 0.0])
    row = np.argmin(np.linalg.norm(split.pts - [1.0, 0.0, 0.0], axis=1))
    assert split.contradictions[row] == 1


def test_unknown_uncertainty_makes_the_whole_group_non_destructive():
    map_ = KFCMap()
    _prime_confirmed_with_two_misses(map_)
    batch = RayBatch(
        endpoints=np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        origins=np.zeros((2, 3)),
        timestamps=np.array([1.0, 1.0]),
        groups=np.array([99, 99]),
        intensity=np.full(2, np.nan),
        floater=np.zeros(2, bool),
        pose_uncertainty_m=np.array([np.nan, 0.0]),
    )
    map_.update_batch(batch)
    map_.flush()
    assert _has_point(map_, [1.0, 0.0, 0.0])
    row = np.argmin(np.linalg.norm(map_.pts - [1.0, 0.0, 0.0], axis=1))
    assert map_.contradictions[row] == 2
    assert map_.n_uncertain_groups == 1


def test_close_range_candidate_query_contains_the_exact_radial_corridor():
    map_ = KFCMap(tentative_contradictions=2)
    occupied = np.array([[0.1, 0.005, 0.0]])
    map_.update(ORIGIN, occupied)
    map_.update(ORIGIN, np.array([[1.0, 0.0, 0.0]]))
    map_.update(ORIGIN, np.array([[1.0, 0.0, 0.0]]))
    assert not _has_point(map_, occupied[0].tolist())


def test_candidate_chunk_budget_does_not_change_finalized_state():
    points = np.array([
        [0.10, 0.005, 0.0], [0.20, 0.006, 0.0],
        [0.50, 0.007, 0.0], [1.00, 0.007, 0.0],
    ])
    large = KFCMap(candidate_pair_budget=100_000)
    tiny = KFCMap(candidate_pair_budget=1)
    for map_ in (large, tiny):
        map_.update(ORIGIN, points)
        for _ in range(2):
            map_.update(ORIGIN, np.array([[2.0, 0.0, 0.0]]))
    left, right = _finalized_state(large), _finalized_state(tiny)
    for a, b in zip(left[:-2], right[:-2]):
        np.testing.assert_equal(a, b)
    assert left[-2:] == right[-2:]


def test_dense_candidate_overflow_never_materializes_an_oversized_chunk(
    monkeypatch,
):
    budget = 64
    map_ = KFCMap(candidate_pair_budget=budget, delete_max_range=4.0)
    mapped = np.column_stack([
        np.arange(1, 51) * 0.05,
        np.zeros(50),
        np.zeros(50),
    ])
    map_.update(ORIGIN, mapped)
    rays = np.repeat([[3.0, 0.0, 0.0]], 1_000, axis=0)
    ranges = np.linalg.norm(rays, axis=1)
    sizes = []
    original = map_._exact_candidate_pairs

    def record_size(ray_index, map_index, origins, direction, ray_ranges):
        sizes.append(len(ray_index))
        assert len(ray_index) <= budget
        return original(ray_index, map_index, origins, direction, ray_ranges)

    monkeypatch.setattr(map_, "_exact_candidate_pairs", record_size)
    map_._visibility_rows(np.zeros_like(rays), rays, ranges)
    assert sizes and max(sizes) <= budget
    assert map_._last_visibility_metrics["candidate_pairs"] == len(map_.pts) * 1_000


def test_max_evidence_must_fit_its_storage_type():
    with np.testing.assert_raises_regex(ValueError, "uint8"):
        KFCMap(max_evidence=256)


def test_candidate_query_matches_brute_force_finite_ray_reference(monkeypatch):
    rng = np.random.default_rng(12)
    points = rng.uniform([-1.5, -1.5, -0.2], [1.5, 1.5, 0.2], size=(250, 3))
    points = points[np.linalg.norm(points, axis=1) > 0.06]
    points = np.vstack([points, [0.1, 0.005, 0.0], [2.0, 0.008, 0.0]])
    map_ = KFCMap(delete_max_range=3.0, candidate_pair_budget=7)
    map_.update(ORIGIN, points)
    directions = rng.normal(size=(90, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    ray_ranges = rng.uniform(0.2, 3.5, size=len(directions))
    endpoints = directions * ray_ranges[:, None]
    # Pin the two hand-picked radial-boundary/close-range directions too.
    endpoints = np.vstack([endpoints, [1.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    ray_ranges = np.linalg.norm(endpoints, axis=1)
    ray_delta = endpoints.copy()
    origins = np.zeros_like(endpoints)

    monkeypatch.setattr(
        map_, "_local_normals",
        lambda rows: (np.zeros((len(rows), 3)), np.zeros(len(rows), bool)),
    )
    misses, blocked = map_._visibility_rows(origins, ray_delta, ray_ranges)

    expected = []
    ray_direction = ray_delta / ray_ranges[:, None]
    for row, point in enumerate(map_.pts):
        point_range = np.linalg.norm(point)
        along = ray_direction @ point
        rho = np.linalg.norm(
            point - along[:, None] * ray_direction, axis=1
        )
        visible = (
            (along > 1e-6)
            & (point_range <= map_.del_max)
            & (rho <= map_.radial)
            & (along < ray_ranges - map_.endpoint_margin)
        )
        if visible.any():
            expected.append(row)
    np.testing.assert_array_equal(misses, np.asarray(expected, np.intp))
    assert len(blocked) == 0


def test_incremental_delete_and_reinsert_keep_indexes_aligned():
    map_ = KFCMap(tentative_contradictions=1)
    points = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0],
    ])
    map_.update(ORIGIN, points)
    capacity_id = id(map_._storage["pts"])
    doomed = np.array([map_._key2row[int(map_._valid_keys(points[:1])[0][0])]])
    map_._delete(doomed)
    map_.update(ORIGIN, np.array([[0.0, -1.0, 0.0]]))

    assert id(map_._storage["pts"]) == capacity_id
    assert len(map_._key2row) == len(map_.pts) == 3
    for row, key in enumerate(map_._keys_by_row):
        assert map_._key2row[int(key)] == row
        block = tuple(int(v) for v in map_._block_keys[row])
        tile = tuple(int(v) for v in map_._tile_keys[row])
        assert row in map_._block2rows[block]
        assert row in map_._tile2rows[tile]


def test_noncontiguous_group_reuse_and_backward_time_are_rejected():
    map_ = KFCMap()
    map_.update(
        ORIGIN, np.array([[1.0, 0.0, 0.0]]),
        observation_group="a", timestamp=1.0,
    )
    map_.update(
        ORIGIN, np.array([[2.0, 0.0, 0.0]]),
        observation_group="b", timestamp=2.0,
    )
    with np.testing.assert_raises_regex(ValueError, "reused"):
        map_.update(
            ORIGIN, np.array([[1.0, 0.0, 0.0]]),
            observation_group="a", timestamp=3.0,
        )
    with np.testing.assert_raises_regex(ValueError, "nondecreasing"):
        map_.update(
            ORIGIN, np.array([[3.0, 0.0, 0.0]]),
            observation_group="c", timestamp=1.5,
        )
