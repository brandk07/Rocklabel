"""Tests for the optional training stack (skipped wholesale without torch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rocklabel.train import metrics as M
from rocklabel.train.data import DataError, block_val_mask, check_no_frame_overlap, loro_folds
from rocklabel.train.models import build_model


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
    # fewer valid points than SA1's 64 centroids: FPS must degrade gracefully
    model = build_model("pointnet2").eval()
    with torch.no_grad():
        out = model(torch.randn(2, 256, 4) * 0.2, torch.tensor([20, 21]))
    assert out.shape == (2,) and torch.isfinite(out).all()
