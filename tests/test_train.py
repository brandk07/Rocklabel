"""Tests for the optional training stack (skipped wholesale without torch)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rocklabel.train import metrics as M
from rocklabel.train.data import DataError, block_val_mask, check_no_frame_overlap, loro_folds
from rocklabel.train.models import FEATURES, build_model, resolve_features


# -- metrics -----------------------------------------------------------------

def test_roc_auc_matches_hand_computed():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    # classic sklearn doc example: AUC = 0.75
    assert M.roc_auc(labels, scores) == pytest.approx(0.75)


def test_perfect_and_random_separation():
    labels = np.r_[np.zeros(50), np.ones(50)].astype(int)
    scores = np.r_[np.linspace(0, 0.4, 50), np.linspace(0.6, 1.0, 50)]
    assert M.roc_auc(labels, scores) == pytest.approx(1.0)
    assert M.average_precision(labels, scores) == pytest.approx(1.0)
    same = np.full(100, 0.5)
    # constant scores: PR-AUC equals prevalence, ROC-AUC is 0.5
    assert M.average_precision(labels, same) == pytest.approx(0.5)
    assert M.roc_auc(labels, same) == pytest.approx(0.5)


def test_confusion_and_baseline():
    labels = np.array([1, 1, 0, 0, 0])
    probs = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    c = M.confusion(labels, probs, 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 2)
    s = M.summarize(labels, probs, 0.5)
    assert s["baseline_accuracy"] == pytest.approx(0.6)
    assert s["baseline_pr_auc"] == pytest.approx(0.4)


# -- splits ------------------------------------------------------------------

def test_loro_folds_cover_each_run_once():
    folds = loro_folds(["a", "b", "c", "d"])
    assert [f["test"] for f in folds] == ["a", "b", "c", "d"]
    for f in folds:
        assert f["test"] not in f["train"] and len(f["train"]) == 3


def test_block_val_mask_has_gap():
    frame = np.repeat(np.arange(100), 3)  # 100 frames, 3 samples each
    train, val = block_val_mask(frame, val_frac=0.2, gap_frames=10)
    assert not np.any(train & val)
    # the gap really is empty: no sample between the two blocks is used
    tr_max, va_min = frame[train].max(), frame[val].min()
    assert va_min - tr_max >= 10
    assert va_min == 80 and tr_max == 69


def test_frame_overlap_check_raises():
    with pytest.raises(DataError):
        check_no_frame_overlap({"r": np.array([1, 2, 3])}, {"r": np.array([3, 4])})
    check_no_frame_overlap({"r": np.array([1, 2])}, {"r": np.array([3, 4])})  # ok
    check_no_frame_overlap({"a": np.array([1])}, {"b": np.array([1])})        # ok


# -- models ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["pointnet", "pointnet2"])
def test_model_padding_invariance(name):
    """Replacing the padded tail with different duplicates of real points must
    not change the output - the mask (or duplicate-safe max) has to hide it."""
    torch.manual_seed(0)
    model = build_model(name).eval()
    pts = torch.randn(3, 256, 4) * 0.2
    counts = torch.tensor([25, 256, 90])
    alt = pts.clone()
    for i, c in enumerate(counts):
        c = int(c)
        if c < 256:
            alt[i, c:] = pts[i, torch.randint(0, c, (256 - c,))]
    with torch.no_grad():
        assert torch.allclose(model(pts, counts), model(alt, counts), atol=1e-5)


def test_pointnet_tnet_regularizer():
    model = build_model("pointnet", tnet=True)
    out = model(torch.randn(2, 256, 4), torch.tensor([256, 100]))
    assert out.shape == (2,)
    assert float(model.pop_regularizer()) > 0.0


def test_pointnet2_shapes_with_tiny_counts():
    # fewer valid points than SA1's centroid budget: FPS must degrade gracefully
    model = build_model("pointnet2").eval()
    with torch.no_grad():
        out = model(torch.randn(2, 256, 4) * 0.2, torch.tensor([20, 21]))
    assert out.shape == (2,) and torch.isfinite(out).all()


def test_set_abstraction_sees_absolute_position():
    """A set-abstraction level must not be translation invariant.

    The reference SSG stack feeds its MLPs only centroid-relative offsets, so
    every level's output is *exactly* unchanged by translating the input - and
    this task asks "is the query *center* standing on a rock". Measured on the
    model trained that way, sliding a rock-centred neighborhood a full 2 m
    moved its rock probability from 0.391 to 0.382.
    """
    from rocklabel.train.models import SetAbstraction

    torch.manual_seed(0)
    sa = SetAbstraction(8, 0.3, 8, in_channel=1, mlp=[16, 16]).eval()
    xyz, feats = torch.randn(2, 32, 3) * 0.2, torch.randn(2, 32, 1)
    mask = torch.ones(2, 32, dtype=torch.bool)
    with torch.no_grad():
        _, here = sa(xyz, feats, mask)
        _, moved = sa(xyz + 2.0, feats, mask)
    assert not torch.allclose(here, moved, atol=1e-4)


# -- feature selection -------------------------------------------------------

def test_resolve_features_canonicalizes_and_defaults():
    assert resolve_features(None) == list(FEATURES)
    # Order is normalized, so a config comparison cannot see two spellings of
    # the same selection as two different runs.
    assert resolve_features(["intensity", "dz", "dx"]) == ["dx", "dz", "intensity"]
    for bad in ([], ["dx", "dx"], ["rgb"]):
        with pytest.raises(ValueError):
            resolve_features(bad)


@pytest.mark.parametrize("name", ["pointnet", "pointnet2"])
def test_deselected_intensity_cannot_reach_the_model(name):
    """The whole point of the switch: with intensity off, whatever sits in
    channel 3 must not move the output by even a float."""
    torch.manual_seed(0)
    model = build_model(name, features=["dx", "dy", "dz"]).eval()
    pts = torch.randn(3, 256, 4) * 0.2
    counts = torch.tensor([25, 256, 90])
    poisoned = pts.clone()
    poisoned[..., 3] = 1234.5
    with torch.no_grad():
        assert torch.equal(model(pts, counts), model(poisoned, counts))


def test_selection_keeps_the_stored_tensor_shape():
    """Channels are picked inside the model, so the [B, N, 4] contract (and
    therefore the dataset, the cache and the export signature) never changes."""
    model = build_model("pointnet", features=["dz"]).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 256, 4), torch.tensor([256, 100]))
    assert out.shape == (2,)
    assert model.mlp1[0].in_channels == 1


def test_pointnet2_and_tnets_refuse_to_lose_a_geometry_channel():
    with pytest.raises(ValueError, match="dx"):
        build_model("pointnet2", features=["dz", "intensity"])
    with pytest.raises(ValueError, match="T-Net"):
        build_model("pointnet", tnet=True, features=["dx", "dy"])


def test_run_dir_name_tags_only_a_non_default_selection():
    """The full channel set keeps the historical name so existing run
    directories stay resumable; anything else is tagged so two channel
    selections on the same fold cannot collide on one directory."""
    from rocklabel.train.data import run_dir_name

    assert run_dir_name("pointnet", "loro_run3") == "pointnet_loro_run3"
    assert run_dir_name("pointnet", "loro_run3", list(FEATURES)) == "pointnet_loro_run3"
    assert run_dir_name("pointnet", "loro_run3",
                        ["dx", "dy", "dz"]) == "pointnet_loro_run3_dx-dy-dz"
    # Order of typing must not spawn a second directory for one selection.
    assert (run_dir_name("pointnet2", "loro_a", ["dz", "dx", "dy"])
            == run_dir_name("pointnet2", "loro_a", ["dx", "dy", "dz"]))


def test_a_run_predating_the_feature_setting_still_resumes(tmp_path):
    """Configs written before --features existed have no such key; they were
    trained on every channel and must not read as a settings change."""
    from rocklabel.train.engine import default_config, train_fold

    cfg = default_config(train_runs=["a"], test_run="b", cache_dir=str(tmp_path))
    old = {k: v for k, v in cfg.items() if k != "features"}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(old))

    # Gets past the settings guard and fails later, on the absent cache.
    with pytest.raises(DataError):
        train_fold(cfg, str(run_dir))


def test_default_config_stores_a_canonical_feature_list():
    from rocklabel.train.engine import default_config

    assert default_config()["features"] == list(FEATURES)
    assert default_config(features=["intensity", "dx"])["features"] == ["dx", "intensity"]


def test_cli_defaults_cannot_shadow_the_training_defaults():
    """An unset CLI flag must leave TRAIN_DEFAULTS alone.

    argparse defaults used to be written down a second time in cli.py, so they
    silently won over engine's for every overlapping key: --patience stayed at
    6 after the default moved to 10 and an entire sweep trained with the
    setting it was meant to change. Parsing with no flags must reproduce
    TRAIN_DEFAULTS exactly.
    """
    from rocklabel.train import TRAIN_DEFAULTS
    from rocklabel.train.cli import _train_cfg, build_parser

    args = build_parser().parse_args(["train", "--model", "pointnet", "--test-run", "r1"])
    cfg = _train_cfg(args, "pointnet", ["r2"], "r1")
    drifted = {k: (TRAIN_DEFAULTS[k], cfg[k]) for k in TRAIN_DEFAULTS
               if k not in ("model", "train_runs", "test_run", "cache_dir")
               and cfg[k] != TRAIN_DEFAULTS[k]}
    assert not drifted, f"CLI defaults shadowed TRAIN_DEFAULTS: {drifted}"


# -- segmentation -------------------------------------------------------------

def test_model_registry_names_the_task_for_every_model():
    from rocklabel.train.models import MODELS, model_task

    assert model_task("pointnet") == "classify"
    assert model_task("pointnet2") == "classify"
    assert model_task("pointnet2_seg") == "segment"
    for name, (task, label) in MODELS.items():
        assert task in ("classify", "segment")
        # the label has to say which approach it is, not just the backbone
        assert "classifier" in label or "segmentation" in label
    with pytest.raises(ValueError):
        model_task("nope")


def test_segmenter_emits_one_logit_per_point():
    torch.manual_seed(0)
    model = build_model("pointnet2_seg").eval()
    pts = torch.randn(2, 512, 4) * 0.5
    counts = torch.tensor([512, 300])
    with torch.no_grad():
        out = model(pts, counts)
    assert out.shape == (2, 512) and torch.isfinite(out).all()


def test_segmenter_is_not_translation_invariant():
    """Same requirement as the classifier: moving the scene must move the
    answer. A segmenter that ignored absolute position would label a rock's
    points identically wherever the rock sat in the crop box."""
    torch.manual_seed(0)
    model = build_model("pointnet2_seg").eval()
    pts = torch.randn(2, 512, 4) * 0.5
    counts = torch.tensor([512, 512])
    shifted = pts.clone()
    shifted[..., 0] += 2.0
    with torch.no_grad():
        assert not torch.allclose(model(pts, counts), model(shifted, counts), atol=1e-4)


def test_seg_masks_exclude_padding_and_the_boundary_shell():
    from rocklabel.train.engine import seg_flatten, seg_valid_mask

    labels = np.array([[1, 0, -1, 0, 1], [0, 1, 0, 0, 0]], np.int8)
    counts = np.array([4, 2])          # row 0: 4 real rows, row 1: 2
    keep = seg_valid_mask(labels, counts)
    # row 0 drops index 2 (shell) and index 4 (padding); row 1 keeps only 0,1
    np.testing.assert_array_equal(keep, [[1, 1, 0, 1, 0], [1, 1, 0, 0, 0]])
    probs = np.arange(10, dtype=float).reshape(2, 5) / 10.0
    y, p = seg_flatten(labels, probs, counts)
    np.testing.assert_array_equal(y, [1, 0, 0, 0, 1])
    np.testing.assert_allclose(p, [0.0, 0.1, 0.3, 0.5, 0.6])


def test_thinning_keeps_per_point_labels_aligned():
    """The density augmentation reorders points; segmentation labels must be
    carried through the same permutation or the supervision is scrambled."""
    from rocklabel.train.engine import _thin

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    # tag each point with its own index in channel 0 so we can follow it
    pts = torch.zeros(4, 64, 4)
    pts[..., 0] = torch.arange(64).float()[None, :].expand(4, -1)
    labels = torch.arange(64).float()[None, :].expand(4, -1).contiguous()
    counts = torch.tensor([64, 40, 25, 12])
    out, new_counts, new_labels = _thin(pts, counts, 0.4, gen, extra=labels)
    assert new_labels is not None
    # every surviving point still carries its own label
    torch.testing.assert_close(out[..., 0], new_labels)
    assert (new_counts <= counts).all()


@pytest.mark.parametrize("model,task", [("pointnet", "classify"),
                                        ("pointnet2_seg", "segment")])
def test_one_training_step_runs_for_both_tasks(tmp_path, model, task):
    """Exercise the real train loop body for one step of each task.

    Guards a class of bug the unit tests missed: the augmentation returns a
    labels tensor only for segmentation, and threading that through the shared
    loop once clobbered the classifier's labels with None. It blew up on the
    first batch of a 24-fold sweep.
    """
    from rocklabel.train.engine import _augment, default_config, seg_valid_mask

    cfg = default_config(model=model)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    n = 256 if task == "classify" else 512
    pts = torch.randn(4, n, 4).abs().clamp(max=1.0)
    cnt = torch.tensor([n, n // 2, n // 3, n // 4])
    y = (torch.rand(4) < 0.3).float() if task == "classify" else \
        (torch.rand(4, n) < 0.1).float()

    pts_a, cnt_a, y_aug = _augment(pts, cnt, cfg, gen,
                                   labels=y if task == "segment" else None)
    if y_aug is not None:
        y = y_aug
    assert y is not None and torch.is_tensor(y)

    net = build_model(model)
    logits = net(pts_a, cnt_a)
    if task == "segment":
        keep = (torch.arange(n)[None, :] < cnt_a[:, None]) & (y >= 0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[keep], y[keep])
        assert seg_valid_mask(y.numpy().astype(np.int8), cnt_a.numpy()).sum() > 0
    else:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
