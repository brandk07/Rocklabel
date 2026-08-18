"""Training/eval harness shared by both models.

One fold = one resumable run directory:
    training/runs/<name>/
        config.json      exact settings (re-running with different ones errors)
        history.csv      per-epoch train/val metrics (plots regenerate from this)
        last.pt          model+optimizer+epoch (resume point)
        best.pt          best val PR-AUC weights
        test_metrics.json, predictions.npz   written by evaluate()

Batching is hand-rolled over in-RAM tensors instead of a DataLoader: the whole
pooled training set is ~300 MB, so worker processes would only add overhead.
Heading invariance is a training-time augmentation by design (the odom-frame
crop is deliberately axis-aligned, see config.example.yaml), hence the random
z-rotation + mirror applied to dx/dy on the GPU each batch.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from . import TRAIN_DEFAULTS
from . import data as D
from . import metrics as M
from .models import FEATURES, build_model, model_task, resolve_features

VAL_METRIC = "val_pr_auc"


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Split:
    """Tensors for one side of a split, kept on CPU; batches move to device."""

    def __init__(self, runs: list[D.RunData], masks: list[np.ndarray] | None = None):
        def cat(key):
            return np.concatenate([getattr(r, key)[m] for r, m in
                                   zip(runs, masks or [slice(None)] * len(runs))])
        self.points = torch.from_numpy(cat("points"))
        self.labels = torch.from_numpy(cat("labels").astype(np.float32))
        self.counts = torch.from_numpy(cat("counts").astype(np.int64))
        self.frame = cat("frame")
        self.centers = cat("centers")
        self.run_id = np.concatenate([
            np.full(int(np.sum(m) if not isinstance(m, slice) else len(r)), r.run_id, dtype=object)
            for r, m in zip(runs, masks or [slice(None)] * len(runs))])

    def __len__(self) -> int:
        return len(self.labels)


#: Never thin a neighborhood below this many real points, whatever the
#: sampled fraction — the generator's own min_neighbors floor is 20, and a
#: handful of returns is not a sample any sensor would hand us.
MIN_KEEP = 8


def _thin(pts: torch.Tensor, counts: torch.Tensor, min_frac: float,
          gen: torch.Generator,
          extra: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor,
                                                      torch.Tensor | None]:
    """Randomly drop real points, keeping the 'real points first' contract.

    Neither model reads the padded tail — PointNet masks it and PointNet++
    exiles it to the sentinel — so the tail is refilled by cycling the
    survivors purely to preserve the stored tensor's shape and its
    duplicate-padding convention.

    ``extra`` (per-point labels, for segmentation) is reordered by the exact
    same indices. Thinning the points without carrying the labels along would
    silently scramble the supervision, which is the one way this augmentation
    could quietly poison a run rather than fail loudly.
    """
    b, n, _ = pts.shape
    dev = pts.device
    real = torch.clamp(counts, max=n)
    frac = min_frac + (1.0 - min_frac) * torch.rand(b, 1, generator=gen, device=dev)
    keep = torch.clamp((real[:, None].float() * frac).round().long(), min=MIN_KEEP)
    keep = torch.minimum(keep, real[:, None])                       # [B, 1]
    # Shuffle the real rows to the front (invalid rows sort last), then take
    # the first `keep` of them and cycle those into the remaining slots.
    order = torch.rand(b, n, generator=gen, device=dev).masked_fill(
        torch.arange(n, device=dev)[None, :] >= real[:, None], 2.0).argsort(dim=1)
    pos = torch.arange(n, device=dev)[None, :].expand(b, n)
    idx = order.gather(1, torch.where(pos < keep, pos, pos % keep))
    out = pts.gather(1, idx[..., None].expand(-1, -1, pts.shape[-1]))
    return out, keep.squeeze(1), (None if extra is None else extra.gather(1, idx))


def _augment(pts: torch.Tensor, counts: torch.Tensor, cfg: dict,
             gen: torch.Generator,
             labels: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Training-time augmentation, all on-device.

    *Heading*: random z-rotation + mirror of the center-relative dx/dy
    channels (the odom-frame crop is deliberately axis-aligned).

    *Reflectivity*: per-sample gain and offset on the intensity channel. This
    is the one that matters for leaving the room. Rock and clear separate at
    ROC-AUC 0.79-0.87 on mean intensity alone in this data, but the whole
    separation lives in a ~0.045 gap between two absolute levels (~0.714 vs
    ~0.669) that belong to one arena's surfaces, not to rocks. Jittering wider
    than that gap denies the model the absolute cue and leaves only the
    within-neighborhood contrast, which is the part that might transfer.

    *Density*: random thinning with counts updated to match, so the model
    cannot assume this sensor's return density. The measured cost of thinning
    to 25% was 0.986 -> 0.957 PR-AUC untrained-for; the point is to make that
    curve flatter still, and to widen it to sensors we do not own.
    """
    b = pts.shape[0]
    dev = pts.device
    theta = torch.rand(b, generator=gen, device=dev) * (2 * torch.pi)
    c, s = torch.cos(theta), torch.sin(theta)
    flip = torch.where(torch.rand(b, generator=gen, device=dev) < 0.5, -1.0, 1.0)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s * flip, c * flip], -1)], 1)
    out = pts.clone()
    out[..., :2] = torch.bmm(pts[..., :2], rot.transpose(1, 2))

    gain_amp, shift_amp = cfg["aug_intensity_gain"], cfg["aug_intensity_shift"]
    if gain_amp or shift_amp:
        gain = 1.0 + (torch.rand(b, 1, generator=gen, device=dev) * 2 - 1) * gain_amp
        shift = (torch.rand(b, 1, generator=gen, device=dev) * 2 - 1) * shift_amp
        out[..., 3] = (out[..., 3] * gain + shift).clamp(0.0, 1.0)

    if cfg["aug_thin_min"] < 1.0:
        out, counts, labels = _thin(out, counts, cfg["aug_thin_min"], gen, extra=labels)
    return out, counts, labels


@torch.no_grad()
def predict(model: torch.nn.Module, split: Split, device: torch.device,
            batch: int = 512, progress: bool = False) -> np.ndarray:
    """Probabilities for a whole split: [S] for a classifier, [F, N] for a
    segmenter (one row per frame, one column per point)."""
    model.eval()
    probs = []
    rng = range(0, len(split), batch)
    for i in tqdm(rng, desc="predict", disable=not progress, leave=False):
        pts = split.points[i:i + batch].to(device, non_blocking=True)
        cnt = split.counts[i:i + batch].to(device, non_blocking=True)
        probs.append(torch.sigmoid(model(pts, cnt)).float().cpu().numpy())
    return np.concatenate(probs)


def seg_valid_mask(labels: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """[F, N] bool: real (unpadded) points carrying a real label.

    Two things are excluded and both matter. Padded rows are duplicates of real
    points and would double-count. Label -1 is the boundary shell - the fuzzy
    centimeters at a rock's edge that the labeler refuses to call either way -
    and scoring against it would punish the model for the one thing the ground
    truth admits it does not know.
    """
    n = labels.shape[1]
    return (np.arange(n)[None, :] < np.asarray(counts)[:, None]) & (labels >= 0)


def seg_flatten(labels: np.ndarray, probs: np.ndarray,
                counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame [F, N] arrays -> flat per-point arrays over scorable points."""
    keep = seg_valid_mask(labels, counts)
    return labels[keep].astype(np.int8), probs[keep]


def _epoch_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    return {"pr_auc": M.average_precision(labels, probs),
            "roc_auc": M.roc_auc(labels, probs),
            "f1_at_0.5": M.confusion(labels, probs, 0.5)["f1"]}


def train_fold(cfg: dict, run_dir: str, resume: bool = True) -> dict:
    """Train one model on one split; returns the final test summary dict."""
    os.makedirs(run_dir, exist_ok=True)
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old = json.load(f)
        # Runs predating the input-channel setting were trained on every
        # channel. Filling the default in keeps them resumable instead of
        # reading as a settings change nobody made.
        old.setdefault("features", list(FEATURES))
        if old != cfg:
            raise SystemExit(f"{run_dir} was created with different settings; "
                             "pick a new --run-dir or delete it")
    else:
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    _seed_all(cfg["seed"])
    device = _device(cfg.get("device"))
    meta = D.load_cache_meta(cfg["cache_dir"])

    task = model_task(cfg["model"])
    train_runs = [D.RunData(cfg["cache_dir"], r, task=task) for r in cfg["train_runs"]]
    # Early-stopping val: tail frame block of each training run, with a
    # temporal gap so no neighborhood pair straddles the boundary.
    tr_masks, va_masks = zip(*(
        D.block_val_mask(r.frame, cfg["val_frac"], cfg["gap_frames"],
                         times=meta["runs"][r.run_id].get("frame_times"),
                         gap_seconds=cfg.get("gap_seconds"))
        for r in train_runs))
    D.check_no_frame_overlap(
        {r.run_id: r.frame[m] for r, m in zip(train_runs, tr_masks)},
        {r.run_id: r.frame[m] for r, m in zip(train_runs, va_masks)})
    tr = Split(train_runs, list(tr_masks))
    va = Split(train_runs, list(va_masks))

    # Class balance is counted over whatever the loss actually sees: one label
    # per sample for a classifier, one per scorable point for a segmenter
    # (padding and boundary-shell points excluded, exactly as in the loss).
    if task == "segment":
        tr_keep = seg_valid_mask(tr.labels.numpy(), tr.counts.numpy())
        n_pos = float((tr.labels.numpy()[tr_keep] == 1).sum())
        n_scored = float(tr_keep.sum())
        unit = "points"
    else:
        n_pos, n_scored, unit = float((tr.labels == 1).sum()), float(len(tr)), "samples"
    n_neg = n_scored - n_pos
    print(f"[{os.path.basename(run_dir)}] task {task}, train {len(tr)} "
          f"({n_scored:.0f} scored {unit}, {n_pos / max(n_scored, 1):.2%} rock), "
          f"val {len(va)}, test run {cfg['test_run']}, device {device}")

    model = build_model(cfg["model"], tnet=cfg["tnet"], dropout=cfg.get("dropout"),
                        features=cfg.get("features")).to(device)
    print(f"  input channels: {', '.join(model.features)}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(n_pos, 1.0),
                                                                 device=device))

    start_epoch, best_metric, bad_epochs = 0, -1.0, 0
    history: list[dict] = []
    last_path, best_path = os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "best.pt")
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best_metric, bad_epochs = ck["epoch"] + 1, ck["best_metric"], ck["bad_epochs"]
        history = ck["history"]
        print(f"  resumed at epoch {start_epoch}")

    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    labels_va = va.labels.numpy()
    counts_va = va.counts.numpy()
    # A segmenter's batch is whole frames (4096 points each), so the classifier's
    # batch size would be ~16x the memory. Scale it down rather than making the
    # user remember two different meanings for --batch.
    step_batch = cfg["batch"] if task != "segment" else max(cfg["batch"] // 32, 2)
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(tr))
        losses = []
        steps = range(0, len(tr) - step_batch + 1, step_batch)  # drop last (BatchNorm)
        for i in tqdm(steps, desc=f"epoch {epoch}", leave=False):
            idx = perm[i:i + step_batch]
            pts = tr.points[idx].to(device, non_blocking=True)
            cnt = tr.counts[idx].to(device, non_blocking=True)
            y = tr.labels[idx].to(device, non_blocking=True)
            if cfg["augment"]:
                # Only a segmenter has per-point labels to carry through the
                # thinning permutation; for a classifier _augment returns None
                # here and must not be allowed to overwrite y.
                pts, cnt, y_aug = _augment(pts, cnt, cfg, gen,
                                           labels=y if task == "segment" else None)
                if y_aug is not None:
                    y = y_aug
            logits = model(pts, cnt)
            if task == "segment":
                # Score only real, non-shell points. Doing this with a boolean
                # select rather than a weight keeps the mean over exactly the
                # points that count, so batches with more padding are not
                # quietly down-weighted.
                keep = (torch.arange(pts.shape[1], device=device)[None, :]
                        < cnt[:, None]) & (y >= 0)
                loss = loss_fn(logits[keep], y[keep])
            else:
                loss = loss_fn(logits, y)
            loss = loss + cfg["tnet_reg"] * model.pop_regularizer()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()

        probs_va = predict(model, va, device, step_batch)
        if task == "segment":
            y_va, p_va = seg_flatten(labels_va, probs_va, counts_va)
        else:
            y_va, p_va = labels_va, probs_va
        with torch.no_grad():
            val_loss = float(torch.nn.functional.binary_cross_entropy(
                torch.from_numpy(p_va).clamp(1e-6, 1 - 1e-6),
                torch.from_numpy(y_va.astype(np.float32))))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               **{f"val_{k}": v for k, v in _epoch_metrics(y_va, p_va).items()},
               "lr": sched.get_last_lr()[0]}
        history.append(row)
        print(f"  epoch {epoch}: train_loss {row['train_loss']:.4f}  "
              f"val_loss {val_loss:.4f}  val_pr_auc {row['val_pr_auc']:.4f}  "
              f"val_roc_auc {row['val_roc_auc']:.4f}")

        improved = row[VAL_METRIC] > best_metric
        if improved:
            best_metric, bad_epochs = row[VAL_METRIC], 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                        "config_hash": meta["config_hash"], "generator": meta["generator"]},
                       best_path)
        else:
            bad_epochs += 1
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch, "history": history,
                    "best_metric": best_metric, "bad_epochs": bad_epochs}, last_path)
        _write_history(run_dir, history)
        if bad_epochs >= cfg["patience"]:
            print(f"  early stop: no {VAL_METRIC} gain in {cfg['patience']} epochs")
            break

    # Final: best weights, threshold picked on val, evaluated on the held-out run.
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    probs_va = predict(model, va, device, step_batch)
    if task == "segment":
        y_va, p_va = seg_flatten(labels_va, probs_va, counts_va)
    else:
        y_va, p_va = labels_va, probs_va
    threshold = M.best_f1_threshold(y_va, p_va)
    ck["threshold"] = threshold
    torch.save(ck, best_path)
    return evaluate(model, cfg, run_dir, threshold, device, batch=step_batch)


def evaluate(model: torch.nn.Module, cfg: dict, run_dir: str, threshold: float,
             device: torch.device, batch: int | None = None) -> dict:
    task = model_task(cfg["model"])
    te = Split([D.RunData(cfg["cache_dir"], cfg["test_run"], task=task)])
    probs = predict(model, te, device, batch or cfg["batch"], progress=True)
    labels = te.labels.numpy().astype(np.int8)
    counts = te.counts.numpy()
    if task == "segment":
        # Headline metrics are per scorable point, which is the segmenter's own
        # unit of prediction. The point-level numbers are NOT comparable to a
        # classifier's sample-level ones (different populations, different
        # prevalence) - `rocklabel-train matched` (rocklabel/train/matched.py)
        # re-scores both at shared candidate centers for that.
        flat_labels, flat_probs = seg_flatten(labels, probs, counts)
        summary = M.summarize(flat_labels, flat_probs, threshold)
        summary["scored_points"] = int(len(flat_labels))
        summary["frames"] = int(len(labels))
    else:
        summary = M.summarize(labels, probs, threshold)
    summary.update({"test_run": cfg["test_run"], "model": cfg["model"],
                    "task": task, "val_threshold": threshold})
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(run_dir, "predictions.npz"),
                        probs=probs, labels=labels, frame=te.frame,
                        counts=counts, centers=te.centers,
                        run_id=str(cfg["test_run"]), task=task,
                        threshold=threshold)
    print(f"  test [{cfg['test_run']}] pr_auc {summary['pr_auc']:.4f}  "
          f"roc_auc {summary['roc_auc']:.4f}  f1@{threshold:.2f} {summary['f1']:.4f}  "
          f"(baseline acc {summary['baseline_accuracy']:.3f})")
    return summary


def _write_history(run_dir: str, history: list[dict]) -> None:
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def default_config(**overrides) -> dict:
    """Training config, defaults from :data:`rocklabel.train.TRAIN_DEFAULTS`.

    ``None`` overrides are dropped rather than applied, which is what lets the
    CLI pass every unset flag straight through without shadowing a default.
    """
    cfg = dict(TRAIN_DEFAULTS)
    cfg.update({k: v for k, v in overrides.items() if v is not None or k in ("device", "dropout")})
    # Canonicalize here, not at build time: config.json is compared verbatim on
    # resume, so "dz,dx,dy" and "dx,dy,dz" must not look like different runs.
    cfg["features"] = resolve_features(cfg["features"])
    return cfg
