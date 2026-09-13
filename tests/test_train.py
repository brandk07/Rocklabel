"""Tests for the optional training stack (skipped wholesale without torch)."""

import os
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


# --------------------------------------------------------------------------- #
# the segmenter's level geometry, now a setting rather than a hardcoded number
# --------------------------------------------------------------------------- #
def test_segmenter_levels_can_be_set_and_are_what_the_model_uses():
    model = build_model("pointnet2_seg", seg_npoints=[1024, 256, 64],
                        seg_radii=[0.1, 0.3, 0.8]).eval()
    assert model.npoints == (1024, 256, 64) and model.radii == (0.1, 0.3, 0.8)
    assert (model.sa1.npoint, model.sa2.npoint, model.sa3.npoint) == (1024, 256, 64)
    assert (model.sa1.radius, model.sa2.radius, model.sa3.radius) == (0.1, 0.3, 0.8)
    pts = torch.randn(2, 1280, 4) * 0.5
    with torch.no_grad():
        out = model(pts, torch.tensor([1280, 900]))
    assert out.shape == (2, 1280) and torch.isfinite(out).all()


def test_leaving_the_segmenter_levels_unset_keeps_what_every_run_so_far_used():
    """Checkpoints trained before the setting existed have to load into the
    same shape they were saved from."""
    assert build_model("pointnet2_seg").npoints == (512, 128, 32)
    assert build_model("pointnet2_seg").radii == (0.25, 0.6, 1.4)


@pytest.mark.parametrize("npoints,radii", [
    ([128, 512, 32], [0.1, 0.3, 0.8]),      # levels must shrink, not grow
    ([1024, 256, 64], [0.8, 0.3, 0.1]),     # and the balls must widen
    ([1024, 256], [0.1, 0.3, 0.8]),         # three levels, not two
])
def test_segmenter_rejects_levels_that_are_not_a_coarsening(npoints, radii):
    with pytest.raises(ValueError):
        build_model("pointnet2_seg", seg_npoints=npoints, seg_radii=radii)


def test_a_run_started_before_the_level_setting_existed_still_resumes(tmp_path):
    """train_fold refuses to write into a directory whose config disagrees with
    it. Both finished sweeps have configs with no level geometry in them, so
    filling in the default has to count as agreement, not as a change."""
    from rocklabel.train import TRAIN_DEFAULTS
    from rocklabel.train.engine import default_config, train_fold

    run_dir = tmp_path / "loro_x"
    run_dir.mkdir()
    cfg = default_config(model="pointnet2_seg", cache_dir=str(tmp_path),
                         train_runs=["a"], test_run="b")
    old = {k: v for k, v in cfg.items() if k not in ("seg_npoints", "seg_radii")}
    (run_dir / "config.json").write_text(json.dumps(old))
    # Fails later for want of a cache, but never on "different settings".
    # (DataError subclasses SystemExit, so the catch has to be BaseException.)
    with pytest.raises(BaseException) as e:
        train_fold(cfg, str(run_dir))
    assert "different settings" not in str(e.value)
    assert TRAIN_DEFAULTS["seg_npoints"] == [512, 128, 32]


def test_floor_reference_makes_the_segmenter_immune_to_a_height_shift():
    """The bug this setting exists for: the stored height channel is measured
    from the robot base, the base rode 0.87-0.97 m above the floor in every
    training recording and ~0.37 m above it in the competition bag, and the
    trained segmenter's best confidence anywhere fell to 0.003 on that gap.
    With the floor reference the shift has to cancel exactly."""
    torch.manual_seed(0)
    model = build_model("pointnet2_seg", seg_height_ref="floor").eval()
    pts = torch.randn(2, 512, 4) * 0.5
    counts = torch.tensor([512, 400])
    raised = pts.clone()
    raised[..., 2] += 0.51
    with torch.no_grad():
        torch.testing.assert_close(model(pts, counts), model(raised, counts),
                                   rtol=0, atol=1e-5)


def test_floor_reference_still_moves_the_answer_sideways():
    """Cancelling a height shift must not cancel position: the model still has
    to answer 'is the point *here* a rock', not 'is a rock somewhere about'."""
    torch.manual_seed(0)
    model = build_model("pointnet2_seg", seg_height_ref="floor").eval()
    pts = torch.randn(2, 512, 4) * 0.5
    counts = torch.tensor([512, 512])
    moved = pts.clone()
    moved[..., 0] += 2.0
    with torch.no_grad():
        assert not torch.allclose(model(pts, counts), model(moved, counts), atol=1e-4)


def test_local_coordinate_segmenter_is_exactly_translation_invariant():
    """The deployment arm must read shape, not where that shape happened to
    sit in the volleyball arena.  Distances, grouping and interpolation all
    survive a uniform xyz translation; no grouping MLP receives absolute xyz."""
    torch.manual_seed(0)
    model = build_model("pointnet2_seg", seg_height_ref="floor",
                        seg_coord_ref="local").eval()
    pts = torch.randn(2, 512, 4) * 0.5
    counts = torch.tensor([512, 400])
    moved = pts.clone()
    moved[..., :3] += torch.tensor([2.3, -1.7, 0.51])
    with torch.no_grad():
        assert torch.allclose(model(pts, counts), model(moved, counts), atol=2e-5)


def test_frame_centered_segmenter_is_exactly_translation_invariant():
    """Frame centering keeps coordinates within the scene for semantic
    decoding, but the robot/arena origin cannot change the answer."""
    torch.manual_seed(0)
    model = build_model("pointnet2_seg", seg_height_ref="floor",
                        seg_coord_ref="frame").eval()
    pts = torch.randn(2, 512, 4) * 0.5
    counts = torch.tensor([512, 400])
    moved = pts.clone()
    moved[..., :3] += torch.tensor([2.3, -1.7, 0.51])
    with torch.no_grad():
        assert torch.allclose(model(pts, counts), model(moved, counts), atol=2e-5)


def test_a_segmenter_predating_the_height_setting_loads_as_base_relative():
    """Old checkpoints carry no seg_height_ref. They were trained base-relative,
    so silently upgrading them to the floor reference would change what their
    weights mean."""
    assert build_model("pointnet2_seg").height_ref == "base"
    assert build_model("pointnet2_seg", seg_height_ref=None).height_ref == "base"


def test_frame_floor_offset_ignores_padding():
    """Padding repeats real points, so quantiling the raw tensor would weight
    duplicates twice and drag the floor toward whatever got repeated."""
    from rocklabel.train.models import frame_floor_offset

    pts = torch.zeros(1, 10, 4)
    pts[0, :4, 2] = torch.tensor([-1.0, -0.9, -0.8, 2.0])   # 4 real points
    pts[0, 4:, 2] = 99.0                                     # padding, must not count
    out = frame_floor_offset(pts, torch.tensor([4]))
    assert -1.0 <= float(out[0]) <= -0.8


def test_ground_tilt_tilts_segmentation_frames_and_leaves_classifiers_alone():
    """The classifier's height channel is already relative to each ball's own
    lowest point, so tilting it would corrupt that reference rather than harden
    anything."""
    from rocklabel.train.engine import _augment

    cfg = {"aug_intensity_gain": 0.0, "aug_intensity_shift": 0.0,
           "aug_thin_min": 1.0, "aug_ground_tilt": 0.05}
    pts = torch.zeros(4, 64, 4)
    pts[..., 0] = torch.linspace(-4, 4, 64)[None, :].expand(4, -1)
    counts = torch.tensor([64, 64, 64, 64])
    gen = torch.Generator().manual_seed(0)
    seg, _, _ = _augment(pts, counts, cfg, gen, task="segment")
    gen = torch.Generator().manual_seed(0)
    clf, _, _ = _augment(pts, counts, cfg, gen, task="classify")
    assert seg[..., 2].abs().max() > 0.01, "segmentation frames must be tilted"
    assert clf[..., 2].abs().max() == 0.0, "classifier samples must not be"


def test_ground_tilt_survives_the_floor_reference():
    """A uniform height shift is cancelled by the floor reference, so jittering
    one would train against nothing. A tilt has to survive it - only its average
    is subtracted, and the slope is the part that matters."""
    from rocklabel.train.engine import _augment

    cfg = {"aug_intensity_gain": 0.0, "aug_intensity_shift": 0.0,
           "aug_thin_min": 1.0, "aug_ground_tilt": 0.05}
    torch.manual_seed(0)
    model = build_model("pointnet2_seg", seg_height_ref="floor").eval()
    pts = torch.randn(2, 512, 4) * 0.5
    counts = torch.tensor([512, 512])
    tilted, _, _ = _augment(pts.clone(), counts, cfg,
                            torch.Generator().manual_seed(3), task="segment")
    with torch.no_grad():
        assert not torch.allclose(model(pts, counts), model(tilted, counts), atol=1e-4)


def test_frame_height_span_measures_structure_not_outliers():
    """How tall a frame is decides whether a whole-frame segmenter recognises
    it at all, so the measurement must not swing on one stray ceiling return."""
    from rocklabel.train.models import frame_height_span

    pts = torch.zeros(1, 200, 4)
    pts[0, :100, 2] = torch.linspace(0.0, 0.2, 100)   # 20 cm of ground and rocks
    pts[0, 100:, 2] = 99.0                            # padding, must not count
    span = float(frame_height_span(pts, torch.tensor([100])))
    assert 0.15 <= span <= 0.25

    # one point on the ceiling must not turn a flat slab into a 3 m frame
    pts[0, 99, 2] = 3.0
    assert float(frame_height_span(pts, torch.tensor([100]))) < 0.3


def test_stray_augmentation_moves_points_along_their_own_ray():
    """Strays have to look like real bad returns, not like random litter.

    A mixed pixel or a grazing-angle range error slides a return along the beam
    it came in on, which is why real strays hang in mid-air above and below the
    surface instead of scattering evenly. The augmentation has to do the same,
    move roughly the requested share, leave the rest untouched, and demote a
    stray that came off a rock to clear.
    """
    import torch

    from rocklabel.train.engine import _stray

    torch.manual_seed(0)
    b, n = 64, 256
    pts = torch.randn(b, n, 4)
    pts[..., :3] += 3.0                      # keep directions well away from 0
    counts = torch.full((b,), n, dtype=torch.long)
    labels = torch.ones(b, n)
    gen = torch.Generator().manual_seed(1)

    out, out_labels = _stray(pts, counts, 0.05, 1.0, gen, labels=labels, task="segment")

    moved = (out[..., :3] - pts[..., :3]).norm(dim=-1) > 1e-6
    assert 0.02 < moved.float().mean() < 0.09, "about 5% of points should move"

    # Every moved point stayed on the line through the origin and its original
    # position: the cross product of old and new directions is zero.
    old = pts[..., :3][moved]
    new = out[..., :3][moved]
    old_u = old / old.norm(dim=-1, keepdim=True)
    new_u = new / new.norm(dim=-1, keepdim=True)
    parallel = torch.cross(old_u, new_u, dim=-1).norm(dim=-1)
    assert float(parallel.max()) < 1e-4, "strays must slide along their own ray"

    # Points that did not move are untouched, including their intensity.
    assert torch.allclose(out[~moved], pts[~moved])
    # A stray off a rock is no longer on the rock.
    assert torch.all(out_labels[moved] == 0.0)
    assert torch.all(out_labels[~moved] == 1.0)


def test_stray_augmentation_leaves_the_boundary_shell_unscored():
    """Label -1 marks the fuzzy edge of a rock, which is excluded from scoring.

    Turning one of those into a supervised 'clear' would invent ground truth
    the labeller explicitly declined to give.
    """
    import torch

    from rocklabel.train.engine import _stray

    b, n = 32, 128
    pts = torch.randn(b, n, 4) + 3.0
    counts = torch.full((b,), n, dtype=torch.long)
    labels = torch.full((b, n), -1.0)
    gen = torch.Generator().manual_seed(2)

    _out, out_labels = _stray(pts, counts, 0.5, 1.0, gen, labels=labels, task="segment")
    assert torch.all(out_labels == -1.0)


def test_stray_augmentation_is_off_by_default():
    """Every checkpoint on disk was trained without it, so the default must
    leave training exactly as it was."""
    from rocklabel.train import TRAIN_DEFAULTS

    assert TRAIN_DEFAULTS["aug_stray_frac"] == 0.0


# --------------------------------------------------------------------------- #
# training on every recording, with nothing held out
# --------------------------------------------------------------------------- #
def _fake_cache(tmp_path, runs):
    """The smallest thing `train` reads before it starts training."""
    import json
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "meta.json").write_text(json.dumps(
        {"config_hash": "x", "generator": {},
         "runs": {r: {"n": 1, "frames": 1} for r in runs}}))
    return str(tmp_path)


def test_test_run_all_trains_on_every_recording_and_holds_nothing_out(tmp_path, monkeypatch):
    """`--test-run all` is a deployment fit: every run trains, none is held out.

    The scores such a model reports are validation scores, not held-out ones,
    so it must be impossible to mistake it for a leave-one-run-out fold - hence
    the separate 'trainall' directory name, checked here alongside the split.
    """
    from rocklabel.train import cli

    cache = _fake_cache(tmp_path / "cache", ["runA", "runB", "runC"])
    seen = {}
    monkeypatch.setattr(cli, "DEFAULT_CACHE", cache)
    monkeypatch.setattr("rocklabel.train.engine.train_fold",
                        lambda cfg, run_dir, resume=True: seen.update(
                            cfg=cfg, run_dir=run_dir) or {})

    cli.main(["train", "--model", "pointnet", "--test-run", "all",
              "--cache-dir", cache, "--runs-root", str(tmp_path / "runs")])

    assert seen["cfg"]["train_runs"] == ["runA", "runB", "runC"]
    assert seen["cfg"]["test_run"] == ""
    assert os.path.basename(seen["run_dir"]) == "pointnet_trainall"


def test_a_named_test_run_still_holds_that_one_out(tmp_path, monkeypatch):
    """The sentinel must not have changed what an ordinary fold does."""
    from rocklabel.train import cli

    cache = _fake_cache(tmp_path / "cache", ["runA", "runB", "runC"])
    seen = {}
    monkeypatch.setattr("rocklabel.train.engine.train_fold",
                        lambda cfg, run_dir, resume=True: seen.update(
                            cfg=cfg, run_dir=run_dir) or {})

    cli.main(["train", "--model", "pointnet", "--test-run", "runB",
              "--cache-dir", cache, "--runs-root", str(tmp_path / "runs")])

    assert seen["cfg"]["train_runs"] == ["runA", "runC"]
    assert seen["cfg"]["test_run"] == "runB"
    assert os.path.basename(seen["run_dir"]) == "pointnet_loro_runB"


# --------------------------------------------------------------------------- #
# BEV CNN
# --------------------------------------------------------------------------- #
def _bev_frame(n=400, pad_to=600, seed=0):
    """One frame plus padding that repeats real rows, as the dataset writes it."""
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(1, pad_to, 4, generator=g)
    pts[..., :2] *= 2.0
    pts[..., 2] *= 0.3
    pts[..., 3] = pts[..., 3].sigmoid()
    pts[0, n:] = pts[0, :pad_to - n].clone()
    return pts, torch.tensor([n])


def test_bev_cnn_emits_one_logit_per_point():
    model = build_model("bev_cnn").eval()
    pts, counts = _bev_frame()
    with torch.no_grad():
        out = model(pts, counts)
    assert out.shape == (1, 600) and torch.isfinite(out).all()


def test_bev_padding_cannot_reach_a_cell():
    """The whole point of this model is the count channel, and padding repeats
    real points — so a padded frame and the same real points unpadded have to
    produce identical answers, or the density signal is a lie."""
    model = build_model("bev_cnn").eval()
    pts, counts = _bev_frame(n=400, pad_to=600)
    with torch.no_grad():
        padded = model(pts, counts)[0, :400]
        bare = model(pts[:, :400].clone(), torch.tensor([400]))[0]
    assert torch.equal(padded, bare)


def test_bev_count_channel_counts_real_returns_only():
    model = build_model("bev_cnn", features=list(FEATURES)).eval()
    pts, counts = _bev_frame(n=400, pad_to=600)
    mask = torch.arange(600)[None, :] < counts[:, None]
    with torch.no_grad():
        chans, _ = model._rasterize(pts[..., :3], pts[..., 3], mask)
    counted = torch.expm1(chans[0, model.bev_channels.index("count")]).sum()
    assert counted.round().item() == 400


def test_bev_points_in_one_cell_get_one_answer():
    model = build_model("bev_cnn").eval()
    pts, counts = _bev_frame(n=600, pad_to=600)
    mask = torch.ones(1, 600, dtype=torch.bool)
    with torch.no_grad():
        out = model(pts, counts)
        _, flat = model._rasterize(pts[..., :3], pts[..., 3], mask)
    for cell in flat[0].unique():
        same = out[0][flat[0] == cell]
        assert torch.allclose(same, same[0].expand_as(same), atol=1e-5)


def test_bev_density_channels_can_be_dropped():
    from rocklabel.train.models_meta import BEV_CHANNELS, BEV_DENSITY
    keep = [c for c in BEV_CHANNELS if c not in BEV_DENSITY]
    model = build_model("bev_cnn", bev_channels=keep)
    assert "count" not in model.bev_channels and "occupied" not in model.bev_channels
    assert model.enc[0][0].in_channels == len(model.bev_channels)


def test_bev_drops_intensity_cells_when_reflectivity_is_not_an_input():
    """Feeding zeros for a deselected channel would make an ablation unreadable."""
    model = build_model("bev_cnn", features=["dx", "dy", "dz"])
    assert not any(c.startswith("intensity") for c in model.bev_channels)


def test_bev_floor_reference_survives_a_height_shift():
    """Same guarantee the segmenter has: how high the robot rides must not
    matter, because it varied by 9.5 cm across the training recordings and by
    half a metre at the competition."""
    model = build_model("bev_cnn", seg_height_ref="floor").eval()
    pts, counts = _bev_frame(n=600, pad_to=600)
    lifted = pts.clone()
    lifted[..., 2] += 0.4
    with torch.no_grad():
        assert torch.allclose(model(pts, counts), model(lifted, counts), atol=1e-4)


def test_bev_grid_must_divide_by_its_levels():
    with pytest.raises(ValueError, match="divide"):
        build_model("bev_cnn", bev_grid=100, bev_depth=3)


def test_bev_rejects_an_unknown_cell_channel():
    with pytest.raises(ValueError, match="unknown BEV channel"):
        build_model("bev_cnn", bev_channels=["occupied", "elevation"])


# -- the candidate's own height ---------------------------------------------

def test_query_height_is_the_candidate_above_its_balls_floor():
    """The builder must measure it against the same z_min the point heights use,
    over the whole ball, before any subsampling - or training and live
    inference silently disagree about what the number means."""
    pytest.importorskip("scipy")
    from rocklabel.dataset.neighborhoods import (QUERY_HEIGHT_CHANNEL,
                                                 build_inference_samples)

    rng = np.random.default_rng(0)
    # A dish: a floor at z=0 with one column of points rising out of it, so the
    # candidate voxel on top of the column sits a known height above the min.
    floor = np.column_stack([rng.uniform(-1, 1, (2000, 2)), np.zeros(2000)])
    column = np.column_stack([np.full(60, 0.4), np.full(60, 0.4),
                              np.linspace(0.0, 0.30, 60)])
    xyz = np.vstack([floor, column]).astype(np.float32)
    inten = np.full(len(xyz), 0.5, np.float32)
    gcfg = {"neighborhood_points": 64, "centers_voxel_m": 0.05,
            "neighborhood_radius_m": 0.5, "min_neighbors": 5}
    out = build_inference_samples(xyz, inten, gcfg, np.random.default_rng(1))
    qz = out["neighborhoods"][:, 0, QUERY_HEIGHT_CHANNEL]
    centers = out["centers_odom"]
    # Every ball here contains floor, so its minimum is 0 and the channel is
    # just the candidate's own height.
    assert qz == pytest.approx(centers[:, 2], abs=1e-4)
    assert qz.max() > 0.25          # the top of the column was reached
    assert qz.min() == pytest.approx(0.0, abs=1e-4)


def test_query_height_is_constant_down_a_sample():
    """It describes the query, not the points, so every row carries the same
    value - which is what lets the model read row 0 and be done."""
    pytest.importorskip("scipy")
    from rocklabel.dataset.neighborhoods import (QUERY_HEIGHT_CHANNEL,
                                                 build_inference_samples)

    rng = np.random.default_rng(2)
    xyz = rng.uniform(0, 1.5, (3000, 3)).astype(np.float32)
    out = build_inference_samples(
        xyz, np.full(3000, 0.5, np.float32),
        {"neighborhood_points": 48, "centers_voxel_m": 0.1,
         "neighborhood_radius_m": 0.5, "min_neighbors": 5},
        np.random.default_rng(3))
    col = out["neighborhoods"][:, :, QUERY_HEIGHT_CHANNEL]
    assert np.all(col == col[:, :1])


def test_query_height_moves_with_the_query_not_the_scene():
    """Lifting the candidate has to change it; lifting everything must not."""
    pytest.importorskip("scipy")
    from rocklabel.dataset.neighborhoods import (QUERY_HEIGHT_CHANNEL,
                                                 build_inference_samples)

    rng = np.random.default_rng(4)
    xyz = np.column_stack([rng.uniform(-1, 1, (3000, 2)),
                           rng.uniform(0, 0.4, 3000)]).astype(np.float32)
    gcfg = {"neighborhood_points": 48, "centers_voxel_m": 0.05,
            "neighborhood_radius_m": 0.5, "min_neighbors": 5}
    here = build_inference_samples(xyz, np.full(3000, .5, np.float32), gcfg,
                                   np.random.default_rng(5))
    lifted = build_inference_samples(xyz + np.float32([0, 0, 7.0]),
                                     np.full(3000, .5, np.float32), gcfg,
                                     np.random.default_rng(5))
    # Translating the whole scene vertically cancels: the ball's own minimum
    # moves with the candidate.
    assert (here["neighborhoods"][:, 0, QUERY_HEIGHT_CHANNEL]
            == pytest.approx(lifted["neighborhoods"][:, 0, QUERY_HEIGHT_CHANNEL],
                             abs=1e-3))
    # Two candidates over the same ground at different heights differ.
    qz = here["neighborhoods"][:, 0, QUERY_HEIGHT_CHANNEL]
    assert qz.max() - qz.min() > 0.1


def test_qz_model_reads_the_height_and_nothing_else_changes():
    """The channel has to reach the head without disturbing the point MLP, or
    the comparison against plain PointNet measures two changes at once."""
    from rocklabel.dataset.neighborhoods import QUERY_HEIGHT_CHANNEL

    torch.manual_seed(0)
    model = build_model("pointnet_qz", features=["dx", "dy", "dz"]).eval()
    pts = torch.randn(4, 64, 5)
    counts = torch.full((4,), 64, dtype=torch.long)
    base = model(pts, counts)
    moved = pts.clone()
    moved[..., QUERY_HEIGHT_CHANNEL] += 0.5
    assert not torch.allclose(base, model(moved, counts))
    # ... and the per-point features it pools are exactly plain PointNet's.
    plain = build_model("pointnet", features=["dx", "dy", "dz"]).eval()
    plain.mlp1.load_state_dict(model.mlp1.state_dict())
    plain.mlp2.load_state_dict(model.mlp2.state_dict())
    with torch.no_grad():
        x = pts.index_select(-1, model.feature_idx).transpose(1, 2)
        pooled = plain.mlp2(plain.mlp1(x)).max(dim=2).values
        y = pts.index_select(-1, plain.feature_idx).transpose(1, 2)
        assert torch.allclose(pooled, plain.mlp2(plain.mlp1(y)).max(dim=2).values)


def test_qz_model_refuses_a_cache_without_the_channel():
    """Loudly, rather than reading intensity as a height."""
    model = build_model("pointnet_qz", features=["dx", "dy", "dz"]).eval()
    with pytest.raises(ValueError, match="candidate-height channel"):
        model(torch.randn(2, 32, 4), torch.full((2,), 32, dtype=torch.long))


# -- phantom clump draw policy ----------------------------------------------

def test_clear_only_phantoms_keep_the_dose_and_every_positive():
    from rocklabel.train.engine import _phantom_clumps, phantom_frac_for_mode

    gen = torch.Generator().manual_seed(0)
    n, rock_frac = 20_000, 0.18
    y = (torch.rand(n, generator=gen) < rock_frac).float()
    pts = torch.rand(n, 64, 5)
    counts = torch.full((n,), 64, dtype=torch.long)
    frac = phantom_frac_for_mode(0.08, "clear-only", float(y.mean()))
    out, new_counts, new_y = _phantom_clumps(pts, counts, y, frac, gen,
                                             mode="clear-only")
    replaced = new_counts != counts
    # No rock example was spent on a phantom...
    assert float(y[replaced].sum()) == 0.0
    assert torch.equal(new_y, y)
    # ...and the batch still got the 8% dose it was asked for.
    assert float(replaced.float().mean()) == pytest.approx(0.08, abs=0.006)


def test_legacy_phantoms_still_eat_positives():
    """The historical behaviour, unchanged - the checkpoints on disk were
    trained with it and 'clear-only' exists to be compared against it."""
    from rocklabel.train.engine import _phantom_clumps, phantom_frac_for_mode

    gen = torch.Generator().manual_seed(1)
    n = 20_000
    y = (torch.rand(n, generator=gen) < 0.18).float()
    pts = torch.rand(n, 64, 5)
    counts = torch.full((n,), 64, dtype=torch.long)
    assert phantom_frac_for_mode(0.08, "legacy", 0.18) == 0.08
    _, new_counts, new_y = _phantom_clumps(pts, counts, y, 0.08, gen, mode="legacy")
    replaced = new_counts != counts
    assert float(y[replaced].sum()) > 0.0
    assert float(new_y.sum()) < float(y.sum())


def test_a_synthetic_clump_gets_its_own_query_height():
    """Not the one from the real sample it replaced, and not a constant the
    model could read off as 'this is a fake'."""
    from rocklabel.dataset.neighborhoods import QUERY_HEIGHT_CHANNEL
    from rocklabel.train.engine import _phantom_clumps

    gen = torch.Generator().manual_seed(2)
    n = 4000
    pts = torch.zeros(n, 64, 5)
    pts[..., QUERY_HEIGHT_CHANNEL] = 9.0          # an impossible marker value
    counts = torch.full((n,), 64, dtype=torch.long)
    y = torch.zeros(n)
    out, new_counts, _ = _phantom_clumps(pts, counts, y, 1.0, gen, mode="legacy")
    qz = out[:, 0, QUERY_HEIGHT_CHANNEL]
    assert float(qz.max()) < 9.0                   # overwritten, not inherited
    assert float(qz.std()) > 0.05                  # a spread, not a constant
    # It lies inside the clump it was drawn from.
    assert float(qz.min()) >= 0.0
    assert float(qz.max()) <= float(out[..., 2].max()) + 1e-5


# -- sweeping without a holdout ---------------------------------------------

def test_folds_all_fits_every_recording():
    from rocklabel.train.ablate import TRAIN_ALL, arm_dir, arms_of

    arm = arms_of("clutter")[0]
    assert arm_dir("root", "clutter", arm, TRAIN_ALL).endswith("trainall")
    assert arm_dir("root", "clutter", arm, "VB4").endswith("loro_VB4")


def test_clutter_suite_repeats_every_setting_under_three_seeds():
    """Its whole point: the gap between two seeds of one setting on this data
    is larger than any of the effects it is comparing."""
    from rocklabel.train.ablate import arms_of

    arms = arms_of("clutter")
    seeds = {}
    for arm in arms:
        base = arm.name.replace("-s43", "").replace("-s44", "")
        seeds.setdefault(base, set()).add(arm.seed or 42)
    assert seeds and all(s == {42, 43, 44} for s in seeds.values())
