"""Regression checks for the Lance training review; no experiment is trained."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rocklabel.train.engine import (
    _phantom_clumps, _restore_rng_state, _rng_state, _save_checkpoint,
)
from rocklabel.train.metrics import best_f1_threshold, confusion
from rocklabel.train.models import build_model_from_config


@pytest.mark.parametrize("scores", [[.001, .002, .003], [.995, .996, .997]])
def test_threshold_resolves_scores_outside_old_grid(scores):
    labels = np.array([0, 1, 1])
    scores = np.array(scores)
    threshold = best_f1_threshold(labels, scores)
    assert confusion(labels, scores, threshold)["f1"] == 1


def test_threshold_keeps_tied_predictions_together():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([.4, .4, .9, .8, .8])
    threshold = best_f1_threshold(y, p)
    assert confusion(y, p, threshold)["f1"] == max(
        confusion(y, p, t)["f1"] for t in np.unique(p))


@pytest.mark.parametrize("mode", ["legacy", "matched"])
def test_phantom_reference_uses_real_points(mode):
    pts = torch.randn(64, 256, 4)
    counts = torch.arange(20, 84)
    y = torch.zeros(64)
    out, cnt, labels = _phantom_clumps(
        pts, counts, y, 1.0, torch.Generator().manual_seed(3), mode=mode)
    for i, count in enumerate(cnt):
        assert out[i, :count, 2].min() == 0
        assert torch.all(out[i, :, 2] >= 0)
    assert torch.all(labels == 0)


def test_matched_phantoms_preserve_rocks_counts_and_ball_support():
    pts = torch.randn(4, 256, 4)
    cnt = torch.tensor([20, 60, 256, 700])
    y = torch.tensor([1., 0., 0., 1.])
    out, count, labels = _phantom_clumps(
        pts, cnt, y, 1.0, torch.Generator().manual_seed(9), mode="matched")
    torch.testing.assert_close(out[y == 1], pts[y == 1])
    torch.testing.assert_close(count, cnt)
    torch.testing.assert_close(labels, y)
    assert torch.all(out[y == 0, :, :2].norm(dim=-1) <= .5)
    assert torch.all(torch.cdist(out[1:3, :, :3], out[1:3, :, :3]) <= 1.00001)


def test_stats_model_padding_and_point_order_invariance_in_training():
    torch.manual_seed(5)
    model = build_model_from_config({"model": "pointnet_stats", "dropout": 0.0})
    pts = torch.randn(3, 256, 4) * .2
    counts = torch.tensor([20, 90, 256])
    alt = pts.clone()
    for i, count in enumerate(counts):
        alt[i, :count] = pts[i, torch.randperm(int(count))]
        alt[i, count:] = float("nan")
    torch.testing.assert_close(model(pts, counts), model(alt, counts), atol=1e-6, rtol=1e-5)
    model(alt, counts).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    # Central support can be empty for off-center candidates.
    pts[..., :2] = .4
    assert torch.isfinite(model(pts, counts)).all()


def test_live_bev_loads_nondefault_geometry_and_density(tmp_path):
    from rocklabel.live.scoring import LiveScorer

    cfg = {"model": "bev_cnn", "features": ["dx", "dy", "dz"],
           "bev_cell": .2, "bev_grid": 16, "bev_width": 8,
           "bev_depth": 2, "bev_channels": ["count", "z_span"],
           "bev_density_norm": True, "seg_height_ref": "floor"}
    trained = build_model_from_config(cfg).eval()
    path = tmp_path / "model.pt"
    torch.save({"config": cfg, "generator": {"seed": 42},
                "model": trained.state_dict()}, path)
    loaded = LiveScorer(str(path), object(), device="cpu")._model
    assert loaded.density_norm is True
    assert loaded.cell == .2 and loaded.grid == 16
    points = torch.randn(2, 64, 4) * .2
    counts = torch.tensor([30, 64])
    with torch.no_grad():
        torch.testing.assert_close(loaded(points, counts), trained(points, counts))


def test_rng_restore_reproduces_batch_dropout_and_augmentation_streams():
    device = torch.device("cpu")
    gen = torch.Generator().manual_seed(3)
    state = copy.deepcopy(_rng_state(gen, device))
    def draw():
        return torch.randperm(30), torch.rand(4, generator=gen), np.random.rand(3)
    expected = draw()
    draw()
    _restore_rng_state(state, gen, device)
    actual = draw()
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_failed_checkpoint_write_preserves_published_model(tmp_path, monkeypatch):
    target = tmp_path / "best.pt"
    _save_checkpoint({"epoch": 3}, str(target))
    def fail(payload, path):
        with open(path, "wb") as stream:
            stream.write(b"incomplete")
        raise OSError("disk error")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        _save_checkpoint({"epoch": 4}, str(target))
    assert torch.load(target, weights_only=False)["epoch"] == 3
    assert list(tmp_path.iterdir()) == [target]


def test_campaign_is_dry_by_default_and_every_training_command_parses(tmp_path, capsys):
    from rocklabel.train.lance_campaign import main, plan
    from rocklabel.train.cli import build_parser
    root = tmp_path / "campaign"
    assert main(["--run-root", str(root)]) == 0
    assert not root.exists()
    assert "DRY RUN" in capsys.readouterr().out
    tasks = plan(root)
    assert len(tasks) == 36
    for task in tasks:
        args = build_parser().parse_args(task["command"][3:])
        assert args.cache_dir == "training/caches/full-sweep"
        assert args.test_run.startswith("VolleyBallTest")
        assert args.aug_phantom_mode == "matched"


def test_sweep_reports_coverage_loss_and_false_area_separately():
    from rocklabel.labels import LabelSet, Rock
    from rocklabel.train.visual_audit import (
        AuditFrame, AuditInterval, CellMap, ModelResult, cell_ids, operating_point_rows,
    )
    from types import SimpleNamespace
    rock = Rock(1, np.zeros(3), .25)
    pos = np.array([[.01, .01, 0], [.11, .01, 0], [1.01, .01, 0]])
    cells = CellMap(pos, np.array([.9, .6, .7]), cell_ids(pos[:, :2], .1))
    result = ModelResult("a", "pointnet", "classify", .5, cells, set(), 1, 1)
    frame = AuditFrame(0, 0, np.zeros(3), pos, np.zeros(3))
    interval = AuditInterval(0, 0, 1, [frame], {1: 2}, {1: 2})
    rows = operating_point_rows([interval], {0: [result, result]},
                                 SimpleNamespace(rocks=[rock]), .1, .75, 1, [.5, .8])
    assert rows[0]["macro_median_coverage"] == 1
    assert rows[0]["mean_false_area_m2"] == pytest.approx(.01)
    assert rows[1]["macro_median_coverage"] == .5
    assert rows[1]["persistent_misses"] == 1
    assert rows[1]["mean_false_area_m2"] == 0

    # A checkpoint's boundary shell must also be honored by threshold sweeps.
    cells.positions[-1] = [.4, .01, 0]
    cells.cell_ids[:] = cell_ids(cells.positions[:, :2], .1)
    result.boundary_shell_m = .2
    rows = operating_point_rows([interval], {0: [result, result]},
                                SimpleNamespace(rocks=[rock]), .1, .75, 1, [.5])
    assert rows[0]["mean_false_cells"] == 0


def test_ground_footprint_attribution_does_not_depend_on_max_score_height():
    from rocklabel.labels import Rock
    from rocklabel.train.visual_audit import projected_rock_labels
    rock = Rock(1, np.zeros(3), .25)
    points = np.array([[.1, .1, 0], [.1, .1, .6], [.28, 0, .6], [.4, 0, 0]])
    np.testing.assert_array_equal(projected_rock_labels(points, [rock], .05), [1, 1, -1, 0])


def test_segmenter_export_uses_whole_frame_contract(tmp_path, monkeypatch):
    import json
    from rocklabel.train.engine import default_config
    from rocklabel.train.export import export_model
    from rocklabel.config import load_config

    cfg = default_config(model="pointnet2_seg", features=["dx", "dy", "dz"],
                         seg_npoints=[8, 4, 2], seg_radii=[.1, .3, .8])
    generator = dict(load_config(None)["generator"], segmentation_points=64,
                     segmentation_min_points=8)
    model = build_model_from_config(cfg)
    checkpoint = tmp_path / "segmenter.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "generator": generator,
                "config_hash": "test", "epoch": 0}, checkpoint)
    shapes = []
    # This verifies the exporter contract and real TorchScript round trip;
    # ONNX runtime compatibility remains a separate deployment check.
    monkeypatch.setattr(torch.onnx, "export", lambda model, inputs, *a, **kw: shapes.append(inputs[0].shape))
    output = tmp_path / "export"
    export_model(str(checkpoint), str(output))
    meta = json.loads((output / "metadata.json").read_text())
    assert shapes == [torch.Size([2, 64, 4])]
    assert meta["input"]["points_per_sample"] == 64
    assert "neighborhood_radius_m" not in meta["input"]
    assert "segmentation" in meta["task"]
    assert "robot-base" in meta["preprocessing_contract"]


def test_qz_export_declares_the_extra_channel(tmp_path):
    """A consumer outside this repo has to be told the sample is five wide and
    what the fifth number means, or it will pass four and get nothing."""
    import json

    import torch

    from rocklabel.train.export import export_model
    from rocklabel.train.models import build_model

    model = build_model("pointnet_qz", features=["dx", "dy", "dz"])
    ck = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(),
                "config": {"model": "pointnet_qz", "tnet": False, "dropout": None,
                           "features": ["dx", "dy", "dz"], "train_runs": ["r1"],
                           "test_run": "r2", "seed": 42, "augment": True},
                "config_hash": "deadbeef", "epoch": 0, "threshold": 0.5,
                "generator": {"neighborhood_points": 64, "neighborhood_radius_m": 0.5,
                              "centers_voxel_m": 0.05, "min_neighbors": 20,
                              "segmentation_points": 2048,
                              "segmentation_min_points": 512}}, ck)
    out = tmp_path / "exported"
    export_model(str(ck), str(out))          # raises if the round-trip diverges
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["input"]["input_channels"] == 5
    assert "query_height" in meta["input"]["points"]
    assert "query_height" in meta["preprocessing_contract"]
