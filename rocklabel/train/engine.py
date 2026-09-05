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
from .models import (FEATURES, build_model, frame_floor_offset, frame_height_span,
                     model_task, resolve_features)

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
             labels: torch.Tensor | None = None,
             task: str = "classify",
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

    *Ground relief* (segmentation only): a random plane tilted across the frame.
    The same argument as reflectivity, applied to the axis that actually broke.
    The floor was flat and level in all twelve training recordings, so nothing
    ever stopped the segmenter treating "height below the robot" as a landmark;
    an unlevelled bin is the case it then cannot handle. Measured on the trained
    seg-fine checkpoint, a floor sloped 30 cm across the 8 m crop cost little
    (0.698 -> 0.619 mean confidence on real rocks) but rolling +/-20 cm
    unevenness cost a third of it (0.698 -> 0.467), so there is real headroom
    here. Deliberately NOT a uniform height shift: ``height_ref="floor"``
    already cancels those exactly, so jittering them would train against
    nothing. A tilt survives the re-referencing because only its average is
    subtracted, and the gradient - the part that matters - is what is left.
    Classifier samples are skipped: their dz is already relative to each ball's
    own lowest point, and tilting a 0.5 m ball would only corrupt that floor.
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

    tilt_amp = float(cfg.get("aug_ground_tilt") or 0.0)
    if tilt_amp and task == "segment":
        # Random direction, random signed magnitude, in metres of rise per metre
        # of ground. Applied after the heading rotation so the tilt is expressed
        # in the same frame the model reads.
        ang = torch.rand(b, generator=gen, device=dev) * (2 * torch.pi)
        amp = (torch.rand(b, generator=gen, device=dev) * 2 - 1) * tilt_amp
        gx, gy = (amp * torch.cos(ang))[:, None], (amp * torch.sin(ang))[:, None]
        out[..., 2] = out[..., 2] + gx * out[..., 0] + gy * out[..., 1]

    stray_frac = float(cfg.get("aug_stray_frac") or 0.0)
    if stray_frac > 0.0:
        out, labels = _stray(out, counts, stray_frac,
                             float(cfg.get("aug_stray_reach") or 1.0),
                             gen, labels=labels, task=task)

    if cfg["aug_thin_min"] < 1.0:
        out, counts, labels = _thin(out, counts, cfg["aug_thin_min"], gen, extra=labels)
    return out, counts, labels


def _stray(pts: torch.Tensor, counts: torch.Tensor, frac: float, reach: float,
           gen: torch.Generator, labels: torch.Tensor | None = None,
           task: str = "classify") -> tuple[torch.Tensor, torch.Tensor | None]:
    """Turn a few returns per sample into strays that sit on no surface.

    The training recordings contain almost none of these, because the rig
    stood over flat ground it struck steeply. An arena the sensor has to look
    across at a grazing angle is full of them, and a model that has never seen
    one has no reason not to treat a clump of them as an object. This is the
    same argument as the reflectivity and ground-tilt jitter: deny a cue that
    happens to hold in every recording we own but will not hold at competition.

    A stray is an existing point pushed *outward along its own line of sight*,
    not a point invented somewhere random — that is what a mixed pixel or a
    grazing-angle range error physically is, and it is why strays land in
    mid-air above and below the surface rather than scattered evenly. The
    sensor sits near the origin of both stored layouts (a segmentation frame is
    robot-base-relative; a classifier ball is centred on itself, where the
    outward direction is the best available stand-in), so the point's own
    direction is the ray.

    Displacement is heavy-tailed: most strays land close to the surface and a
    few go metres, matching the per-beam range wander measured on a parked
    robot. Strays are labelled **clear**, because that is what they are — a
    return off nothing is not part of a rock.
    """
    b, n, _ = pts.shape
    dev = pts.device
    valid = torch.arange(n, device=dev)[None, :] < counts[:, None]
    pick = (torch.rand(b, n, generator=gen, device=dev) < frac) & valid
    if not bool(pick.any()):
        return pts, labels

    xyz = pts[..., :3]
    direction = xyz / xyz.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    # Heavy tail: an exponential draw, so the median is ~0.35 of `reach` and a
    # small share go several times further. Signed, because a range error can
    # read short as well as long.
    u = torch.rand(b, n, generator=gen, device=dev).clamp(1e-6, 1.0)
    mag = -torch.log(u) * (reach * 0.5)
    sign = torch.where(torch.rand(b, n, generator=gen, device=dev) < 0.5, -1.0, 1.0)
    shift = (mag * sign)[..., None] * direction

    out = pts.clone()
    out[..., :3] = torch.where(pick[..., None], xyz + shift, xyz)
    if labels is not None and task == "segment":
        # A moved point is no longer on whatever it was on, so a stray that
        # came off a rock becomes clear. Label -1 is the boundary shell, which
        # seg_valid_mask drops from scoring entirely; leave those alone rather
        # than promoting an unscored point into supervised ground truth.
        labels = torch.where(pick & (labels > 0), torch.zeros_like(labels), labels)
    return out, labels


def _phantom_clumps(pts: torch.Tensor, counts: torch.Tensor, y: torch.Tensor,
                    frac: float, gen: torch.Generator,
                    extent: float = 0.54, spread: float = 0.35,
                    min_pts: int = 20, max_pts: int = 60,
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace a share of classifier samples with synthetic phantom clumps.

    Measured on the competition recording (4 September 2026), the returns that
    hang over that arena are not a nudged version of a real surface: 96% of
    them arrive on beams pointing *up* from the sensor, 85% report a range
    shorter than the two neighbouring beams in their own ring, and they land
    where the beam would otherwise have returned nothing at all. Enough of them
    fall together that the candidate generator centres a 0.5 m ball on the
    clump, and the classifier is then asked whether that ball is a rock.

    Inside such a ball the clump is not obviously wrong. Against the same
    accumulated window, a phantom-centred ball measured 0.54 m of vertical
    extent and 0.090 m of thickness off its best-fit plane, where a
    rock-centred ball measured 0.46 m and 0.112 m. What separates them is
    density - 680 returns against 4,964 - and the fixed 256-point sample
    throws that away before the model ever sees it.

    So the gap this closes is a missing *sample*, not a missing jitter. Every
    ball in all eleven training recordings is a near-flat patch: referenced to
    its own lowest point, the median rock spans 0.115 m vertically and the
    median clear ball 0.094 m. Nothing in the set has ever been a loose 3D
    scatter with no surface under it, so nothing has ever taught the model that
    such a thing is not a rock. This builds that sample directly: points spread
    through a tall box with no flat base, labelled clear.

    Deliberately a *replacement* rather than an addition, so batch size, class
    balance bookkeeping and the loss weight all stay where the control put
    them and the arm remains single-variable against it.

    Args:
        pts: ``(B, N, C)`` samples; channels 0-2 are dx, dy, dz with dz already
            measured from each ball's own lowest point.
        counts: ``(B,)`` real point count per sample; the tail is pad-by-repeat.
        y: ``(B,)`` per-sample labels.
        frac: probability that a sample is replaced.
        extent: vertical extent of a clump (m), the measured 0.54 m.
        spread: horizontal half-width (m) the points scatter over.
        min_pts, max_pts: real-point count drawn per clump. The floor is the
            generator's own ``min_neighbors``, below which no candidate exists.
    """
    b, n, c = pts.shape
    dev = pts.device
    pick = torch.rand(b, generator=gen, device=dev) < frac
    if not bool(pick.any()):
        return pts, counts, y

    # A loose scatter: uniform across the ball horizontally, spread through the
    # full height rather than piled on a floor. No flat base is the whole point.
    dx = (torch.rand(b, n, generator=gen, device=dev) * 2 - 1) * spread
    dy = (torch.rand(b, n, generator=gen, device=dev) * 2 - 1) * spread
    tall = extent * (0.5 + torch.rand(b, 1, generator=gen, device=dev))
    dz = torch.rand(b, n, generator=gen, device=dev) * tall
    # dz is measured from the ball's lowest point, so re-reference the clump the
    # same way the generator would have.
    dz = dz - dz.min(dim=1, keepdim=True).values

    clump = pts.clone()
    clump[..., 0], clump[..., 1], clump[..., 2] = dx, dy, dz
    if c > 3:  # leave brightness in the range the cache actually holds
        clump[..., 3] = 0.6 + torch.rand(b, n, generator=gen, device=dev) * 0.4

    # Sparse, like the real thing: a real count well below a surface patch's,
    # with the tail padded by repeating a real point (the cache's convention).
    span = max_pts - min_pts
    new_counts = min_pts + (torch.rand(b, generator=gen, device=dev) * span).long()
    new_counts = torch.minimum(new_counts, torch.full_like(new_counts, n))
    ar = torch.arange(n, device=dev)[None, :]
    src = torch.remainder(ar, new_counts[:, None].clamp_min(1))
    clump = torch.gather(clump, 1, src[..., None].expand(-1, -1, c))

    out = torch.where(pick[:, None, None], clump, pts)
    counts = torch.where(pick, new_counts.to(counts.dtype), counts)
    y = torch.where(pick, torch.zeros_like(y), y)
    return out, counts, y


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
        # Same for the segmenter's level geometry: runs predating the setting
        # used what was hardcoded in the model, which is what the default still
        # is, so filling it in keeps them resumable rather than reading as a
        # settings change nobody made.
        for key in ("seg_npoints", "seg_radii"):
            old.setdefault(key, TRAIN_DEFAULTS[key])
        # Same for the BEV settings: folds trained before each of them existed
        # must keep resuming rather than reading as a settings change nobody made.
        for key in ("bev_cell", "bev_grid", "bev_width", "bev_depth",
                    "bev_channels", "bev_density_norm", "pos_weight_cap"):
            old.setdefault(key, TRAIN_DEFAULTS[key])
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
          f"val {len(va)}, "
          f"{('test run ' + cfg['test_run']) if cfg['test_run'] else 'no held-out run'}"
          f", device {device}")

    model = build_model(cfg["model"], tnet=cfg["tnet"], dropout=cfg.get("dropout"),
                        features=cfg.get("features"),
                        seg_npoints=cfg.get("seg_npoints"),
                        seg_radii=cfg.get("seg_radii"),
                        seg_height_ref=cfg.get("seg_height_ref"),
                        seg_coord_ref=cfg.get("seg_coord_ref"),
                        bev_cell=cfg.get("bev_cell"),
                        bev_grid=cfg.get("bev_grid"),
                        bev_width=cfg.get("bev_width"),
                        bev_depth=cfg.get("bev_depth"),
                        bev_channels=cfg.get("bev_channels"),
                        bev_density_norm=cfg.get("bev_density_norm")).to(device)
    print(f"  input channels: {', '.join(model.features)}")
    floor_band: tuple[float, float] | None = None
    frame_band: tuple[float, float] | None = None
    if task == "segment":
        # Both segmentation models describe their own geometry: the point-based
        # one by its sampling levels, the grid-based one by its raster.
        if hasattr(model, "npoints"):
            print(f"  levels: {model.npoints} centroids at {model.radii} m")
            print(f"  coordinate reference: {model.coord_ref}")
        else:
            print(f"  grid: {model.grid}x{model.grid} cells of {model.cell:.2f} m "
                  f"(+/-{model.grid * model.cell / 2:.1f} m), centred per frame")
            print(f"  cell channels: {', '.join(model.bev_channels)}")
        # Record where this run's ground actually sat, so live scoring can say
        # "the floor in front of you is nowhere near what this model was trained
        # on" instead of silently returning zeros for a whole arena.
        off = frame_floor_offset(tr.points, tr.counts).numpy()
        floor_band = (float(np.percentile(off, 1)), float(np.percentile(off, 99)))
        print(f"  height reference: {model.height_ref} "
              f"(training floor sat {floor_band[0]:+.2f}..{floor_band[1]:+.2f} m "
              f"below the robot base)")
        # And how tall those frames were. Nothing in the model cancels this the
        # way height_ref cancels a floor offset: a frame carrying half a metre
        # of berm wall is a different picture from the flat slab of ground the
        # arena recordings are, and the model reads the difference as "none of
        # this is rock". Recorded so live scoring can say so out loud.
        span = frame_height_span(tr.points, tr.counts).numpy()
        frame_band = (float(np.percentile(span, 1)), float(np.percentile(span, 99)))
        print(f"  frame height: training frames held "
              f"{frame_band[0]:.2f}..{frame_band[1]:.2f} m of vertical structure")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    # How much a rock example counts against a clear one. Uncapped this is the
    # raw imbalance, which is about 94 for a per-point model and about 4.3 for
    # the sliding-window classifier - a 22x difference arising purely from the
    # generator discarding 95% of clear candidates for one format and none for
    # the other. A weight that large buys recall by pushing the decision
    # boundary until almost everything reads as rock at training time, and is
    # the standing suspect for why per-point models rank rocks correctly on a
    # new arena while their confidence collapses to near zero. Capping it is
    # the one-line experiment nobody had run.
    pos_weight = n_neg / max(n_pos, 1.0)
    cap = cfg.get("pos_weight_cap")
    if cap:
        pos_weight = min(pos_weight, float(cap))
    print(f"  positive-example weight: {pos_weight:.1f}"
          + (f" (capped from {n_neg / max(n_pos, 1.0):.1f})" if cap and n_neg / max(n_pos, 1.0) > cap else ""))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

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
                                           labels=y if task == "segment" else None,
                                           task=task)
                if y_aug is not None:
                    y = y_aug
                # Whole-sample phantom clumps, classifier only: the arena's bad
                # returns arrive as a candidate centred on nothing, which is a
                # sample the training set has never held (see _phantom_clumps).
                ph = float(cfg.get("aug_phantom_frac") or 0.0)
                if ph > 0.0 and task != "segment":
                    pts, cnt, y = _phantom_clumps(
                        pts, cnt, y, ph, gen,
                        extent=float(cfg.get("aug_phantom_extent") or 0.54),
                    )
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
                        "config_hash": meta["config_hash"], "generator": meta["generator"],
                        "floor_band": floor_band, "frame_band": frame_band},
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
    if not cfg["test_run"]:
        # No held-out run: this is a deployment fit over every recording in the
        # cache, so there is nothing honest left to score it on. Write the
        # validation numbers instead of a test_metrics.json, so nothing
        # downstream can mistake them for a held-out result.
        summary = M.summarize(y_va, p_va, threshold)
        summary.update({"test_run": "", "model": cfg["model"], "task": task,
                        "val_threshold": threshold, "held_out": False})
        with open(os.path.join(run_dir, "val_metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  no held-out run (trained on all {len(cfg['train_runs'])} "
              f"recordings); val pr_auc {summary['pr_auc']:.4f} at "
              f"threshold {threshold:.2f}")
        return summary
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
