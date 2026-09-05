"""Pure bookkeeping tests for the deployment visual audit."""

import numpy as np

from rocklabel.labels import Rock
from rocklabel.train.visual_audit import (
    CellMap,
    ModelResult,
    PredictionAccumulator,
    cell_ids,
    comparison_summary,
    connected_components,
    rock_metrics,
    select_cases,
)


def test_prediction_accumulator_matches_live_newest_value_then_shared_max():
    acc = PredictionAccumulator(0.05)
    acc.update(np.array([[0.01, 0.01, 0.0], [0.06, 0.01, 0.0]]),
               np.array([0.9, 0.6]))
    # Revisit the first native voxel: live behavior replaces its old 0.9.
    acc.update(np.array([[0.02, 0.01, 0.0]]), np.array([0.2]))
    cells = acc.reduce(0.10)
    # Both native voxels share one 10 cm audit cell, whose max is now 0.6.
    assert len(cells.probabilities) == 1
    assert cells.probabilities[0] == 0.6
    np.testing.assert_array_equal(cells.cell_ids, [[0, 0]])


def test_rock_coverage_counts_complete_cells_and_fragmentation():
    raw = np.array([[0.02, 0.02, 0.0], [0.12, 0.02, 0.0],
                    [0.22, 0.02, 0.0], [0.32, 0.02, 0.0]])
    rock = Rock(7, np.array([0.17, 0.02, 0.0]), 1.0)
    pos = raw.copy()
    probs = np.array([0.9, 0.1, 0.9, 0.1])
    cells = CellMap(pos, probs, cell_ids(pos[:, :2], 0.10))
    result = ModelResult("m.pt", "pointnet", "classify", 0.5, cells,
                         {(0, 0), (2, 0)}, 0, 0)
    metric = rock_metrics(raw, rock, result, 0.10, min_coverage=0.60)
    assert metric["covered_cells"] == 2
    assert metric["coverage"] == 0.5
    assert metric["fragments"] == 2
    assert metric["detected"] is False
    assert connected_components({(0, 0), (1, 0), (5, 5)}) == 2


def test_case_selection_uses_one_strongest_disagreement_per_rock():
    def observation(rock, seq, a, b, visible=10):
        return {
            "rock_id": rock, "interval_sequence": seq,
            "visible_cells": visible,
            "model_a": {"coverage": a, "detected": a >= .25},
            "model_b": {"coverage": b, "detected": b >= .25},
        }

    rows = [observation(1, 0, .9, .8), observation(1, 1, .9, .0),
            observation(2, 2, .6, .2), observation(3, 3, .7, .65)]
    chosen = select_cases(rows, ["model_a", "model_b"], max_cases=2)
    assert [(r["rock_id"], r["interval_sequence"]) for r in chosen] == [(1, 1), (2, 2)]


def test_comparison_rejects_binary_hit_parity_when_coverage_is_materially_different():
    observations = [{
        "interval_sequence": 4,
        "model_a": {"false_cells": 22},
        "model_b": {"false_cells": 0},
    }]
    per_rock = [{
        "rock_id": 12,
        "model_a": {"median_coverage": .86, "detected_any": True},
        "model_b": {"median_coverage": .43, "detected_any": True},
    }]
    summary = comparison_summary(
        observations, per_rock, ["model_a", "model_b"], material_gap=.20)
    assert summary["verdict"] == "does_not_support_parity"
    assert summary["coverage_leads"]["model_a"] == [12]
    assert summary["persistent_detection_advantage"]["model_a"] == []
    assert summary["false_cells"]["model_a"]["median_per_interval"] == 22


def test_visual_audit_cli_preserves_operational_defaults():
    from rocklabel.train.cli import build_parser

    args = build_parser().parse_args([
        "visual-audit", "arena.mcap", "--labels", "arena.labels.json",
        "--model-a", "classifier.pt", "--model-b", "segmenter.pt",
    ])
    assert args.floor_band == (-0.10, 0.60)
    assert args.accum_seconds == 5.0
    assert args.min_coverage == 0.25
    assert args.material_gap == 0.20
