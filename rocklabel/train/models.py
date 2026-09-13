"""PointNet and PointNet++ (SSG) binary rock classifiers, plain PyTorch.

Input contract (matches neighborhoods.py): points [B, 256, 4] with channels
[dx, dy, dz, intensity] already canonicalized (dx/dy center-relative, dz
ground-relative), counts [B] = number of REAL points. Padding repeats real
points and is appended AFTER them, so the validity mask is arange(N) < counts.

Feature selection happens INSIDE the model: every model always takes the full
[B, N, 4] tensor and selects its own channels, so the dataset, the cache, the
augmentation and the export signature are identical whatever ``features``
holds. Training on geometry alone is therefore just a model setting, not a
regenerate — which matters because reflectivity is the channel least likely to
survive a change of arena (a white comforter and lunar regolith do not share
an RSSI distribution).

Padding policy: PointNet's max-pool is duplicate-safe, so masking there is
belt-and-braces. PointNet++ is not: duplicates would waste FPS centroids and
skew nothing else only because every pool here is a max. We therefore mask
explicitly - FPS picks farthest among real points only, and padded points are
moved to a far sentinel so ball queries never gather them. Group-all pooling
masks invalid columns. With counts=N (no padding info) both models degrade
gracefully to the classic unmasked behavior.

Position policy: the label is "is the *center* of this neighborhood standing
on a rock", which is a localization question, not a shape-classification one.
The reference PointNet++ feeds each set-abstraction MLP only centroid-relative
offsets, which makes every level locally translation invariant - correct for
ModelNet, wrong here. Measured on the trained SSG model, sliding a rock-centred
neighborhood 2 m sideways moved its output from 0.391 to 0.382: it had learned
"a rock is somewhere in this ball" and could not say where. SetAbstraction
therefore passes the absolute neighborhood-frame coordinate alongside the
relative offset, which costs 3 input channels per level. PointNet never had the
problem - it reads raw per-point coordinates straight into its MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dataset.neighborhoods import (FEATURES, GEOMETRY,  # noqa: F401  (re-exported)
                                     QUERY_HEIGHT_CHANNEL, SAMPLE_CHANNELS,
                                     has_query_height, resolve_features)
from .models_meta import (BEV_CHANNELS, BEV_DENSITY, MODELS,  # noqa: F401  (re-exported)
                          model_task)

SENTINEL = 1.0e3  # farther than any real neighborhood coordinate (meters)

#: Which height in a frame counts as "the floor" when the segmenter re-references
#: its z channel. Not the minimum: a single low outlier would drag the whole
#: frame. Rocks are ~2.6% of points, so the 10th percentile is solidly ground
#: while still sitting below the arena's own unevenness. It is a module constant
#: rather than a setting because train and inference MUST compute it identically
#: - a reference that differs between the two is the bug this whole mechanism
#: exists to prevent.
FLOOR_QUANTILE = 0.10


def _feature_buffer(names: list[str]) -> torch.Tensor:
    return torch.tensor([FEATURES.index(n) for n in names], dtype=torch.long)


def valid_mask(counts: torch.Tensor, n: int) -> torch.Tensor:
    """[B, N] bool; real points come first (see neighborhoods.build_...)."""
    return torch.arange(n, device=counts.device)[None, :] < counts[:, None]


def _masked_low_quantile(z: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    """[B] nearest-rank q-quantile of ``z`` over valid points only.

    torch.quantile has no mask, and padded rows repeat real points, so quantiling
    the raw tensor would weight duplicated points twice. Sorting with invalids
    pushed to +inf puts every real value in the low block, where a per-sample
    rank picked from that sample's own count lands on a real point.
    """
    filled = z.masked_fill(~mask, torch.finfo(z.dtype).max)
    srt = filled.sort(dim=1).values
    counts = mask.sum(dim=1).clamp_min(1)
    k = (q * (counts - 1).to(z.dtype)).round().long()
    return srt.gather(1, k[:, None]).squeeze(1)


def _mlp1d(channels: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(channels, channels[1:]):
        layers += [nn.Conv1d(a, b, 1), nn.BatchNorm1d(b), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def _head(in_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, 1),
    )


# ===========================================================================
# PointNet
# ===========================================================================

class TNet(nn.Module):
    """Spatial/feature transform regressor (predicts a k x k matrix)."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.mlp = _mlp1d([k, 64, 128, 1024])
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, k * k),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        g = self.mlp(x).masked_fill(~mask[:, None, :], -torch.inf).max(dim=2).values
        m = self.fc(g).view(-1, self.k, self.k)
        return m + torch.eye(self.k, device=x.device)[None]


class PointNet(nn.Module):
    """Vanilla PointNet classifier; T-Nets optional (the data is already
    canonicalized, so they default off)."""

    #: Channels of the stored sample this model needs handed to it. Not the
    #: channels it *reads* - that is ``features``, selected inside forward.
    #: Only the export tracer, which has to build a stand-in tensor without a
    #: dataset, needs to know the difference.
    input_channels = len(FEATURES)

    def __init__(self, tnet: bool = False, dropout: float = 0.3,
                 features: list[str] | None = None):
        super().__init__()
        self.features = resolve_features(features)
        if tnet and self.features[:3] != list(GEOMETRY):
            raise ValueError("T-Nets regress a 3x3 spatial transform, so they need "
                             f"all of {list(GEOMETRY)} selected; got {self.features}")
        self.use_tnet = tnet
        self.input_tnet = TNet(3) if tnet else None
        self.feature_tnet = TNet(64) if tnet else None
        self.mlp1 = _mlp1d([len(self.features), 64, 64])
        self.mlp2 = _mlp1d([64, 64, 128, 1024])
        self.head = _head(1024, dropout)
        self._reg = torch.zeros(())
        # Not persistent: the selection lives in the training config, and a
        # checkpoint that disagreed with it would be a silent contract break.
        self.register_buffer("feature_idx", _feature_buffer(self.features),
                             persistent=False)

    def forward(self, points: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        mask = valid_mask(counts, points.shape[1])
        x = points.index_select(-1, self.feature_idx).transpose(1, 2)  # [B, C, N]
        reg = x.new_zeros(())
        if self.input_tnet is not None:
            t = self.input_tnet(x[:, :3], mask)
            x = torch.cat([torch.bmm(t, x[:, :3]), x[:, 3:]], dim=1)
            reg = reg + _ortho_penalty(t)
        x = self.mlp1(x)
        if self.feature_tnet is not None:
            t = self.feature_tnet(x, mask)
            x = torch.bmm(t, x)
            reg = reg + _ortho_penalty(t)
        self._reg = reg
        x = self.mlp2(x)
        g = x.masked_fill(~mask[:, None, :], -torch.inf).max(dim=2).values
        return self.head(g).squeeze(-1)

    def pop_regularizer(self) -> torch.Tensor:
        return self._reg


class PointNetQZ(PointNet):
    """PointNet, plus one number: how high the candidate sits in its own ball.

    The classifier's input is canonicalized twice over and the two halves use
    different origins. dx and dy are measured from the candidate center, so the
    model always knows where the query is horizontally. dz is measured from the
    lowest point *of the ball*, which is a good height reference for describing
    a surface but says nothing about the query - the candidate's own z is never
    written down anywhere in the tensor. Slide a candidate straight up half a
    metre without moving a single neighbor and the stored sample is byte for
    byte identical. A rock, and a floating return hanging over that same rock,
    are exactly that pair.

    So this feeds ``candidate_z - ball_min_z`` (channel
    :data:`~rocklabel.dataset.neighborhoods.QUERY_HEIGHT_CHANNEL` of the stored
    sample, constant across its rows) to the classifier head, concatenated to
    the pooled global feature. Deliberately NOT as a per-point input channel:
    it is one fact about the query, and pushing a constant through the
    per-point MLP would let it modulate every point feature instead of just
    informing the decision.

    Whether the ambiguity it removes accounts for real errors is the open
    question this model exists to answer - see the ``clutter`` suite's
    ``cls-qz`` arm.
    """

    def __init__(self, tnet: bool = False, dropout: float = 0.3,
                 features: list[str] | None = None):
        super().__init__(tnet=tnet, dropout=dropout, features=features)
        self.head = _head(1024 + 1, dropout)

    #: Width of the sample tensor this model needs, for the export tracer.
    input_channels = SAMPLE_CHANNELS

    def forward(self, points: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        # Skipped while tracing: the export tracer builds a stand-in tensor of
        # the right width anyway, and reading a shape there bakes it into the
        # graph as a constant and warns about it.
        if not torch.jit.is_tracing() and not has_query_height(points):
            raise ValueError(
                f"{type(self).__name__} needs the candidate-height channel, but this "
                f"sample tensor is {points.shape[-1]} wide. It was built by a "
                "dataset generated before that channel existed - regenerate the "
                "dataset and rebuild the cache.")
        mask = valid_mask(counts, points.shape[1])
        x = points.index_select(-1, self.feature_idx).transpose(1, 2)  # [B, C, N]
        reg = x.new_zeros(())
        if self.input_tnet is not None:
            t = self.input_tnet(x[:, :3], mask)
            x = torch.cat([torch.bmm(t, x[:, :3]), x[:, 3:]], dim=1)
            reg = reg + _ortho_penalty(t)
        x = self.mlp1(x)
        if self.feature_tnet is not None:
            t = self.feature_tnet(x, mask)
            x = torch.bmm(t, x)
            reg = reg + _ortho_penalty(t)
        self._reg = reg
        x = self.mlp2(x)
        g = x.masked_fill(~mask[:, None, :], -torch.inf).max(dim=2).values
        # Constant down the sample, so row 0 is the whole of it. Read after
        # pooling, never before: the point MLP must see exactly what plain
        # PointNet sees, or the comparison between the two measures two changes.
        qz = points[:, 0, QUERY_HEIGHT_CHANNEL, None]
        return self.head(torch.cat([g, qz], dim=1)).squeeze(-1)


def _ortho_penalty(t: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(t.shape[1], device=t.device)[None]
    return ((torch.bmm(t, t.transpose(1, 2)) - eye) ** 2).sum(dim=(1, 2)).mean()


class PointNetStats(nn.Module):
    """Experimental classifier retaining support beyond feature maxima.

    Pool max, mean and spread in the whole ball and its central 15 cm disk.
    Fractions describe relative spatial support, never raw sensor density.
    Per-point LayerNorm keeps duplicate padding out of training statistics.
    The input and height reference match existing classifier checkpoints.
    """

    def __init__(self, dropout: float = 0.3, features=None):
        super().__init__()
        self.features = resolve_features(features)
        if self.features[:3] != list(GEOMETRY):
            raise ValueError("pointnet_stats requires dx, dy, dz geometry")
        self.register_buffer("feature_idx", _feature_buffer(self.features), persistent=False)
        layers = []
        for a, b in zip([len(self.features), 64, 128], [64, 128, 256]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(256 * 6 + 1, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

    @staticmethod
    def _pool(x, mask):
        weight = mask[..., None].to(x.dtype)
        count = weight.sum(dim=1).clamp_min(1)
        mean = (x * weight).sum(dim=1) / count
        var = ((x - mean[:, None]).square() * weight).sum(dim=1) / count
        maximum = x.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
        maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, 0.0)
        return torch.cat([maximum, mean, (var + 1e-6).sqrt()], dim=1)

    def forward(self, points, counts):
        mask = valid_mask(counts, points.shape[1])
        # Sanitize padding before the MLP as well as masking its pooled values.
        selected = points.index_select(-1, self.feature_idx)
        selected = selected.masked_fill(~mask[..., None], 0.0)
        x = self.mlp(selected)
        center = mask & (points[..., :2].square().sum(dim=-1) <= 0.15 ** 2)
        fraction = center.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = torch.cat([self._pool(x, mask), self._pool(x, center), fraction], dim=1)
        return self.head(pooled).squeeze(-1)

    def pop_regularizer(self):
        return next(self.parameters()).new_zeros(())


# ===========================================================================
# PointNet++ (single-scale grouping)
# ===========================================================================

def _fps(xyz: torch.Tensor, mask: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling restricted to valid points. [B, npoint] indices.

    If a sample has fewer valid points than npoint the leftovers repeat
    already-picked points, which downstream max-pools ignore.
    """
    b, n, _ = xyz.shape
    picked = torch.zeros(b, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((b, n), torch.inf, device=xyz.device)
    farthest = torch.zeros(b, dtype=torch.long, device=xyz.device)  # index 0 is always real
    batch = torch.arange(b, device=xyz.device)
    for i in range(npoint):
        picked[:, i] = farthest
        d = ((xyz - xyz[batch, farthest][:, None]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = dist.masked_fill(~mask, -1.0).argmax(dim=1)
    return picked


def _ball_group(xyz: torch.Tensor, centroids: torch.Tensor, radius: float,
                nsample: int) -> torch.Tensor:
    """Indices [B, S, nsample] of points within radius of each centroid;
    groups short of nsample repeat their first member (max-pool safe).
    Padded points sit at the SENTINEL so they never land inside a ball."""
    d2 = torch.cdist(centroids, xyz)  # [B, S, N]
    n = xyz.shape[1]
    idx = torch.arange(n, device=xyz.device)[None, None].expand_as(d2).clone()
    idx[d2 > radius] = n
    idx = idx.sort(dim=-1).values[:, :, :nsample]
    first = idx[:, :, :1]
    return torch.where(idx == n, first, idx)


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B, N, C] gathered by idx [B, ...] -> [B, ..., C]."""
    b = x.shape[0]
    flat = idx.reshape(b, -1)
    out = x.gather(1, flat[..., None].expand(-1, -1, x.shape[-1]))
    return out.reshape(*idx.shape, x.shape[-1])


class SetAbstraction(nn.Module):
    """One SSG level: sample centroids, group a ball around each, pool.

    Each grouped point is described to the MLP twice — once relative to its
    centroid and once in neighborhood coordinates. The relative copy is the
    classic formulation and is what makes the level a *local* shape detector;
    the absolute copy is the addition this task needs. See the module
    docstring for why: with offsets alone the whole stack is translation
    invariant, and "is the query center standing on a rock" is not a
    translation-invariant question.
    """

    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: list[int]):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers: list[nn.Module] = []
        last = in_channel + 6  # +3 centroid-relative offset, +3 absolute position
        for out in mlp:
            layers += [nn.Conv2d(last, out, 1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """xyz [B,N,3], feats [B,N,C], mask [B,N] -> (new_xyz [B,S,3], new_feats [B,S,C'])."""
        ctr_idx = _fps(xyz, mask, self.npoint)
        new_xyz = _gather(xyz, ctr_idx)
        grp_idx = _ball_group(xyz, new_xyz, self.radius, self.nsample)
        grouped_xyz = _gather(xyz, grp_idx)                          # [B,S,K,3] absolute
        local = grouped_xyz - new_xyz[:, :, None]                    # [B,S,K,3] centroid-relative
        grouped = torch.cat([local, grouped_xyz, _gather(feats, grp_idx)], -1)
        x = self.mlp(grouped.permute(0, 3, 1, 2))                    # [B,C',S,K]
        return new_xyz, x.max(dim=3).values.transpose(1, 2)


class PointNetPP(nn.Module):
    """PointNet++ SSG sized for these neighborhoods.

    The measured point budget drives the sizing: the median neighborhood holds
    ~57 real points inside the 0.5 m radius and the 10th percentile only 24, so
    SA1 asks for 32 centroids rather than the 64 it used to. At 64 more than
    half of all samples had fewer real points than centroids requested and FPS
    spent its budget duplicating picks.
    """

    def __init__(self, dropout: float = 0.4, features: list[str] | None = None):
        super().__init__()
        self.features = resolve_features(features)
        if self.features[:3] != list(GEOMETRY):
            raise ValueError("pointnet2 samples and groups by position, so it needs "
                             f"all of {list(GEOMETRY)} selected; got {self.features}. "
                             "Deselect only the non-geometry channels here, or use "
                             "model=pointnet for arbitrary subsets.")
        extra = self.features[3:]  # everything past the xyz block becomes features
        self.register_buffer("extra_idx", _feature_buffer(extra), persistent=False)
        self.sa1 = SetAbstraction(32, 0.15, 16, in_channel=len(extra), mlp=[64, 64, 128])
        self.sa2 = SetAbstraction(8, 0.30, 16, in_channel=128, mlp=[128, 128, 256])
        self.global_mlp = _mlp1d([256 + 3, 256, 512, 1024])
        self.head = _head(1024, dropout)

    def forward(self, points: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        mask = valid_mask(counts, points.shape[1])
        # Exile padded points so no ball query or FPS pick can reach them.
        xyz = torch.where(mask[..., None], points[..., :3], torch.full_like(points[..., :3], SENTINEL))
        # [B, N, 0] when only geometry is selected — every downstream cat and
        # Conv2d handles the zero-width case, so no special branch is needed.
        feats = points.index_select(-1, self.extra_idx)
        xyz, feats = self.sa1(xyz, feats, mask)
        # After SA1 every centroid is a real point (FPS was masked), so all
        # levels below are fully valid.
        full = torch.ones(xyz.shape[:2], dtype=torch.bool, device=xyz.device)
        xyz, feats = self.sa2(xyz, feats, full)
        x = self.global_mlp(torch.cat([xyz, feats], -1).transpose(1, 2))
        return self.head(x.max(dim=2).values).squeeze(-1)

    def pop_regularizer(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)


# ===========================================================================
# PointNet++ semantic segmentation (whole frame in, a label per point out)
# ===========================================================================

def _knn_group(xyz: torch.Tensor, centroids: torch.Tensor, radius: float,
               nsample: int) -> torch.Tensor:
    """Indices [B, S, nsample] of the nsample NEAREST points to each centroid,
    with anything beyond ``radius`` replaced by the centroid's nearest neighbor.

    The classifier's :func:`_ball_group` sorts the full [B, S, N] index tensor,
    which is fine at N=256 and ruinous at N=4096. topk is O(N) per centroid
    instead of O(N log N) and returns nearest-first rather than
    lowest-index-first, which is also the better grouping - but it is a
    different rule, so it lives here rather than replacing the classifier's and
    silently changing those results.
    """
    d = torch.cdist(centroids, xyz)                       # [B, S, N]
    # Configurable small abstraction levels may contain fewer than nsample
    # centroids. Pool the available neighbors instead of failing in topk.
    val, idx = torch.topk(d, min(nsample, xyz.shape[1]), dim=-1, largest=False)
    return torch.where(val > radius, idx[..., :1], idx)


class SegSetAbstraction(nn.Module):
    """Downsampling level: FPS centroids, kNN-in-ball grouping, max pool."""

    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: list[int], *, absolute_xyz: bool = True):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.absolute_xyz = bool(absolute_xyz)
        layers: list[nn.Module] = []
        last = in_channel + (6 if self.absolute_xyz else 3)
        for out in mlp:
            layers += [nn.Conv2d(last, out, 1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctr_idx = _fps(xyz, mask, self.npoint)
        new_xyz = _gather(xyz, ctr_idx)
        grp_idx = _knn_group(xyz, new_xyz, self.radius, self.nsample)
        grouped_xyz = _gather(xyz, grp_idx)
        local = grouped_xyz - new_xyz[:, :, None]
        geometry = [local, grouped_xyz] if self.absolute_xyz else [local]
        grouped = torch.cat([*geometry, _gather(feats, grp_idx)], -1)
        x = self.mlp(grouped.permute(0, 3, 1, 2))
        return new_xyz, x.max(dim=3).values.transpose(1, 2)


class FeaturePropagation(nn.Module):
    """Upsampling level: interpolate coarse features onto the finer point set,
    concatenate the skip connection, then a shared per-point MLP.

    This is the half a classifier does not have, and the reason segmentation
    needs one pass instead of one per candidate: features computed once on a
    coarse set are carried back out to every original point by inverse-distance
    weighting of its three nearest coarse neighbors.
    """

    def __init__(self, in_channel: int, mlp: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_channel
        for out in mlp:
            layers += [nn.Conv1d(last, out, 1), nn.BatchNorm1d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz_fine: torch.Tensor, xyz_coarse: torch.Tensor,
                feats_fine: torch.Tensor | None,
                feats_coarse: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(xyz_fine, xyz_coarse)                       # [B, N, S]
        k = min(3, xyz_coarse.shape[1])
        dist, idx = torch.topk(d, k, dim=-1, largest=False)
        w = 1.0 / dist.clamp_min(1e-8)
        w = w / w.sum(dim=-1, keepdim=True)                         # [B, N, k]
        gathered = _gather(feats_coarse, idx)                       # [B, N, k, C]
        interp = (gathered * w[..., None]).sum(dim=2)               # [B, N, C]
        if feats_fine is not None:
            interp = torch.cat([interp, feats_fine], dim=-1)
        return self.mlp(interp.transpose(1, 2)).transpose(1, 2)


def frame_floor_offset(points: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """[B] where each frame's own ground sits in the stored dz channel.

    Public because three places need this number and they must agree: the model
    subtracts it, training records the range it saw, and live scoring compares
    the frame in front of the robot against that range. Two implementations of
    "the floor" that drift apart would reintroduce the failure exactly.
    """
    mask = valid_mask(counts, points.shape[1])
    return _masked_low_quantile(points[..., 2], mask, FLOOR_QUANTILE)


#: Where the top of a frame is measured, for :func:`frame_height_span`. The 99th
#: percentile rather than the maximum for the same reason the floor uses the
#: 10th: one stray return off a ceiling or a passing head would otherwise decide
#: how tall the whole frame is called.
CEILING_QUANTILE = 0.99


def frame_height_span(points: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """[B] how much vertical structure each frame holds, floor to 99th pct.

    A whole-frame segmenter reads the shape of everything it is given, so the
    height of the slab it is handed is part of its input contract - and unlike
    the floor offset, nothing in the model cancels it. Training records the
    range it saw and live scoring compares the region in front of the robot
    against that range, from this one function so the two cannot drift.
    """
    mask = valid_mask(counts, points.shape[1])
    z = points[..., 2]
    return (_masked_low_quantile(z, mask, CEILING_QUANTILE)
            - _masked_low_quantile(z, mask, FLOOR_QUANTILE))


class PointNetPPSeg(nn.Module):
    """Per-point rock/clear segmentation over a whole cropped frame.

    Input is the same [B, N, 4] contract as the classifiers, but N is a whole
    frame (4096 points spanning the crop box) instead of one 0.5 m ball, and
    the output is [B, N] logits instead of [B].

    The radii are scene-scale rather than neighborhood-scale: the classifier
    sees a 1 m sphere and works in centimeters, this sees an 8 x 8 m crop and
    has to find 20-30 cm rocks in it, which is the hard part. Rocks are ~2.6%
    of points, so the loss is prevalence-weighted.

    Height reference: the stored dz channel is measured from the robot base,
    which is not a stable reference across recordings - the base rode 0.87-0.97 m
    above the floor in all twelve volleyball recordings (9.5 cm of variation in
    total) and ~0.37 m above it in the competition bag. Measured on the trained
    seg-fine checkpoint, shifting every height by a constant collapsed it
    completely: 270 points over threshold at +0.0 m, 162 at +0.2 m, zero at
    +0.3 m, and a highest-confidence-anywhere of 0.0026 at +0.51 m. With
    ``height_ref="floor"`` the frame's own ground is subtracted first, which
    makes that shift cancel exactly. The classifier never had the problem
    because its builder re-levels every ball to that ball's lowest point.

    ``height_ref`` defaults to "base" so checkpoints trained before this existed
    load and behave as they were trained; new runs default to "floor" via
    TRAIN_DEFAULTS.
    """

    def __init__(self, dropout: float = 0.3, features: list[str] | None = None,
                 npoints: tuple[int, int, int] = (512, 128, 32),
                 radii: tuple[float, float, float] = (0.25, 0.6, 1.4),
                 height_ref: str = "base", coord_ref: str = "scene"):
        super().__init__()
        if height_ref not in ("base", "floor"):
            raise ValueError(f"height_ref must be 'base' or 'floor'; got {height_ref!r}")
        self.height_ref = height_ref
        if coord_ref not in ("scene", "frame", "local"):
            raise ValueError(
                f"coord_ref must be 'scene', 'frame' or 'local'; got {coord_ref!r}")
        self.coord_ref = coord_ref
        self.features = resolve_features(features)
        if self.features[:3] != list(GEOMETRY):
            raise ValueError("segmentation samples and groups by position, so it needs "
                             f"all of {list(GEOMETRY)} selected; got {self.features}")
        if len(npoints) != 3 or len(radii) != 3:
            raise ValueError("segmentation has exactly three levels, so it needs "
                             f"three npoints and three radii; got {npoints} / {radii}")
        if list(npoints) != sorted(npoints, reverse=True):
            raise ValueError("each segmentation level samples fewer centroids than "
                             f"the one above it, so npoints must descend; got {npoints}")
        if list(radii) != sorted(radii):
            raise ValueError("each segmentation level pools over a wider ball than "
                             f"the one above it, so radii must ascend; got {radii}")
        self.npoints, self.radii = tuple(npoints), tuple(radii)
        extra = self.features[3:]
        self.register_buffer("extra_idx", _feature_buffer(extra), persistent=False)
        c0 = len(extra)
        absolute_xyz = coord_ref != "local"
        self.sa1 = SegSetAbstraction(npoints[0], radii[0], 32, c0, [32, 32, 64],
                                     absolute_xyz=absolute_xyz)
        self.sa2 = SegSetAbstraction(npoints[1], radii[1], 32, 64, [64, 64, 128],
                                     absolute_xyz=absolute_xyz)
        self.sa3 = SegSetAbstraction(npoints[2], radii[2], 32, 128, [128, 128, 256],
                                     absolute_xyz=absolute_xyz)
        self.fp3 = FeaturePropagation(256 + 128, [128, 128])
        self.fp2 = FeaturePropagation(128 + 64, [128, 64])
        self.fp1 = FeaturePropagation(64 + c0, [64, 64])
        self.head = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, 1, 1),
        )

    def forward(self, points: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        mask = valid_mask(counts, points.shape[1])
        xyz = points[..., :3]
        if self.height_ref == "floor":
            # Re-reference height to the frame's own ground instead of the robot
            # base. Subtracting a per-frame constant makes the model exactly
            # invariant to how high the base rides above the floor, which is the
            # one thing the stored dz channel cannot be trusted to hold steady
            # across recordings (see frame_floor_offset).
            floor = _masked_low_quantile(points[..., 2], mask, FLOOR_QUANTILE)
            xyz = torch.cat([xyz[..., :2], (xyz[..., 2] - floor[:, None])[..., None]], -1)
        if self.coord_ref == "frame":
            # Preserve position *within* the observed scene, which the feature
            # propagation decoder needs to assign local evidence back to the
            # right points, while denying it the robot/arena origin as a
            # shortcut. Only real rows contribute; repeated padding must not
            # drag the center toward whichever points happened to be copied.
            weight = mask[..., None].to(xyz.dtype)
            center_xy = (xyz[..., :2] * weight).sum(1) / weight.sum(1).clamp_min(1.0)
            xyz = torch.cat([xyz[..., :2] - center_xy[:, None], xyz[..., 2:]], -1)
        xyz0 = torch.where(mask[..., None], xyz, torch.full_like(xyz, SENTINEL))
        f0 = points.index_select(-1, self.extra_idx)
        xyz1, f1 = self.sa1(xyz0, f0, mask)
        full1 = torch.ones(xyz1.shape[:2], dtype=torch.bool, device=xyz1.device)
        xyz2, f2 = self.sa2(xyz1, f1, full1)
        full2 = torch.ones(xyz2.shape[:2], dtype=torch.bool, device=xyz2.device)
        xyz3, f3 = self.sa3(xyz2, f2, full2)
        f2 = self.fp3(xyz2, xyz3, f2, f3)
        f1 = self.fp2(xyz1, xyz2, f1, f2)
        # Padded rows sit at the sentinel; their interpolated features are
        # meaningless but the training loss and every metric mask them out.
        f0 = self.fp1(xyz0, xyz1, f0 if f0.shape[-1] else None, f1)
        return self.head(f0.transpose(1, 2)).squeeze(1)

    def pop_regularizer(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)


# ===========================================================================
# Bird's-eye-view CNN (whole frame in, a label per point out)
# ===========================================================================

def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class BEVCNN(nn.Module):
    """Rock/clear segmentation by rasterizing the frame and convolving it.

    Input and output contract are exactly the segmenter's - points [B, N, 4]
    plus counts [B] in, one logit per point out - so this trains, scores and
    reports through the existing segmentation path with nothing changed. The
    raster is built inside ``forward`` rather than read from disk on purpose:
    every augmentation the engine applies (heading rotation, ground tilt,
    thinning, and above all the stray-return jitter) happens to the points
    before they reach a model, and a model handed a pre-built raster would see
    none of it. Building the grid here also means this model and the PointNet++
    segmenter are fed byte-identical frames, so a difference between them is a
    difference in the models.

    Grid geometry. Cells are ``cell`` metres square and the grid is centred on
    the frame's own valid centroid, not on the robot base: the engine rotates
    every training frame by a random angle about the base, which swings a
    corner of the 8 x 8 m crop box well outside it. Measured over every cached
    full-sweep frame, the furthest point from a frame's centroid is 6.90 m, so
    a 144-cell grid at 0.10 m (+/- 7.2 m) contains every frame under any
    rotation; 128 contains 97.7% of them and clamps a handful of corner points
    on the rest. Points are clamped rather than dropped so that the scored
    population stays identical to the segmenter's, whatever the grid size.

    Height reference. Same argument and same default as PointNetPPSeg: the
    stored dz is measured from the robot base, which rode 0.87-0.97 m above the
    floor across the training recordings and ~0.37 m above it in the
    competition bag. ``height_ref="floor"`` subtracts the frame's own ground
    first, which makes that difference cancel exactly.
    """

    def __init__(self, cell: float = 0.10, grid: int = 144, width: int = 32,
                 depth: int = 3, dropout: float = 0.3,
                 features: list[str] | None = None,
                 height_ref: str = "floor",
                 bev_channels: list[str] | None = None,
                 density_norm: bool = False):
        super().__init__()
        self.density_norm = bool(density_norm)
        if height_ref not in ("base", "floor"):
            raise ValueError(f"height_ref must be 'base' or 'floor'; got {height_ref!r}")
        self.height_ref = height_ref
        self.cell, self.grid = float(cell), int(grid)
        if self.grid % (2 ** depth):
            raise ValueError(f"grid {self.grid} must divide by 2^depth ({2 ** depth}) "
                             "so the U-net's halvings land on whole cells")
        self.features = resolve_features(features)
        if self.features[:3] != list(GEOMETRY):
            raise ValueError("the BEV grid is built from position, so it needs all of "
                             f"{list(GEOMETRY)} selected; got {self.features}")
        self.has_intensity = "intensity" in self.features

        chosen = list(BEV_CHANNELS if bev_channels is None else bev_channels)
        unknown = [c for c in chosen if c not in BEV_CHANNELS]
        if unknown:
            raise ValueError(f"unknown BEV channel(s) {unknown}; pick from {list(BEV_CHANNELS)}")
        # Intensity channels are meaningless without the intensity input, and
        # silently feeding zeros would make a channel ablation unreadable.
        if not self.has_intensity:
            chosen = [c for c in chosen if not c.startswith("intensity")]
        if not chosen:
            raise ValueError("at least one BEV channel must be selected")
        self.bev_channels = [c for c in BEV_CHANNELS if c in chosen]  # storage order

        cin = len(self.bev_channels)
        chs = [width * (2 ** i) for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        prev = cin
        for c in chs[:-1]:
            self.enc.append(_conv_block(prev, c))
            prev = c
        self.bottleneck = _conv_block(prev, chs[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.up.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, 2))
            self.dec.append(_conv_block(chs[i] * 2, chs[i]))
        self.head = nn.Sequential(
            nn.Conv2d(chs[0], chs[0], 1, bias=False),
            nn.BatchNorm2d(chs[0]), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(chs[0], 1, 1),
        )
        self.pool = nn.MaxPool2d(2)

    # -- rasterize -------------------------------------------------------
    def _rasterize(self, xyz: torch.Tensor, inten: torch.Tensor,
                   mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(channels [B, C, G, G], flat cell index [B, N]) for a batch of frames.

        Invalid (padded) rows are routed to a scratch cell that is dropped
        before the grid is reshaped, so repeated padding can never inflate a
        cell's count - which would defeat the one channel this model exists to
        test.
        """
        b, n, _ = xyz.shape
        g, ncell = self.grid, self.grid * self.grid
        weight = mask.to(xyz.dtype)

        # Centre on the frame's own valid points; padded rows repeat real ones,
        # so they must not drag the centre toward whatever happened to be copied.
        denom = weight.sum(1).clamp_min(1.0)
        centre = (xyz[..., :2] * weight[..., None]).sum(1) / denom[:, None]
        xy = xyz[..., :2] - centre[:, None, :]

        idx = torch.floor(xy / self.cell).long() + g // 2
        idx = idx.clamp(0, g - 1)
        flat = idx[..., 0] * g + idx[..., 1]
        # Scratch cell for padded rows.
        flat_w = torch.where(mask, flat, torch.full_like(flat, ncell))

        z = xyz[..., 2]
        zero = torch.zeros(b, ncell + 1, device=xyz.device, dtype=xyz.dtype)
        ones = weight

        count = zero.clone().scatter_add_(1, flat_w, ones)
        z_sum = zero.clone().scatter_add_(1, flat_w, z * ones)
        z_sq = zero.clone().scatter_add_(1, flat_w, z * z * ones)
        big = torch.finfo(xyz.dtype).max
        z_max = zero.clone().fill_(-big).scatter_reduce_(
            1, flat_w, torch.where(mask, z, torch.full_like(z, -big)), "amax")
        z_min = zero.clone().fill_(big).scatter_reduce_(
            1, flat_w, torch.where(mask, z, torch.full_like(z, big)), "amin")

        occ = (count > 0).to(xyz.dtype)
        safe = count.clamp_min(1.0)
        z_mean = z_sum / safe
        z_var = (z_sq / safe - z_mean * z_mean).clamp_min(0.0)
        z_max = torch.where(occ > 0, z_max, torch.zeros_like(z_max))
        z_min = torch.where(occ > 0, z_min, torch.zeros_like(z_min))

        if self.density_norm:
            # Cell count divided by this frame's own average occupied cell, so
            # the channel says "denser or sparser than the rest of this frame"
            # instead of an absolute number of returns. The absolute number is
            # not portable: a frame is subsampled or padded to a fixed budget
            # before a model sees it, so a competition frame holding 3,479
            # points in the crop box arrives with 2,048 real rows where a
            # volleyball frame holding 1,145 arrives with 1,145 - a 1.8x shift
            # in every cell of the raster, in the one channel this model leans
            # on. A stray return is still the cell far below its frame's
            # average, which is what the channel is for.
            occupied_mean = (count.sum(1) / occ.sum(1).clamp_min(1.0))[:, None]
            density = count / occupied_mean.clamp_min(1e-6)
        else:
            density = count
        built = {
            "occupied": occ,
            "count": torch.log1p(density),
            "z_max": z_max,
            "z_min": z_min,
            "z_span": z_max - z_min,
            "z_std": z_var.sqrt(),
        }
        if self.has_intensity:
            i_sum = zero.clone().scatter_add_(1, flat_w, inten * ones)
            i_max = zero.clone().fill_(-big).scatter_reduce_(
                1, flat_w, torch.where(mask, inten, torch.full_like(inten, -big)), "amax")
            built["intensity_mean"] = i_sum / safe
            built["intensity_max"] = torch.where(occ > 0, i_max, torch.zeros_like(i_max))

        # Drop the scratch cell, then lay the flat cells out as a picture.
        chans = torch.stack([built[c][:, :ncell] for c in self.bev_channels], 1)
        return chans.reshape(b, len(self.bev_channels), g, g), flat

    def forward(self, points: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        mask = valid_mask(counts, points.shape[1])
        xyz = points[..., :3]
        if self.height_ref == "floor":
            floor = _masked_low_quantile(points[..., 2], mask, FLOOR_QUANTILE)
            xyz = torch.cat([xyz[..., :2],
                             (xyz[..., 2] - floor[:, None])[..., None]], -1)
        inten = points[..., 3]
        x, flat = self._rasterize(xyz, inten, mask)

        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], 1))
        cells = self.head(x).flatten(1)              # [B, G*G] one logit per cell
        # Read each point's answer out of the cell it landed in. Padded rows get
        # whatever their duplicated original got, which the loss masks anyway.
        return cells.gather(1, flat)

    def pop_regularizer(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)


def build_model_from_config(config: dict) -> nn.Module:
    """Rebuild the complete input contract, including settings without weights.

    Keep every checkpoint consumer on this path: density normalization and
    coordinate references can change predictions without changing weight shapes.
    Missing keys retain build_model's historical checkpoint defaults.
    """
    keys = ("tnet", "dropout", "features", "seg_npoints", "seg_radii",
            "seg_height_ref", "seg_coord_ref", "bev_cell", "bev_grid",
            "bev_width", "bev_depth", "bev_channels", "bev_density_norm")
    return build_model(config["model"], **{k: config[k] for k in keys if k in config})


def build_model(name: str, tnet: bool = False, dropout: float | None = None,
                features: list[str] | None = None,
                seg_npoints=None, seg_radii=None,
                seg_height_ref: str | None = None,
                seg_coord_ref: str | None = None,
                bev_cell: float | None = None, bev_grid: int | None = None,
                bev_width: int | None = None, bev_depth: int | None = None,
                bev_channels: list[str] | None = None,
                bev_density_norm: bool | None = None) -> nn.Module:
    """``features=None`` means all of :data:`FEATURES` — the historical
    behavior, so checkpoints trained before the setting existed still load.

    ``seg_npoints``/``seg_radii`` size the segmenter's three levels; ``None``
    keeps the geometry every segmentation run before them was trained with, so
    those checkpoints still load into the shape they were saved from.

    ``seg_height_ref=None`` likewise means "base", the reference every
    segmentation run before this setting was trained against.
    """
    if name == "pointnet":
        return PointNet(tnet=tnet, dropout=0.3 if dropout is None else dropout,
                        features=features)
    if name == "pointnet_qz":
        return PointNetQZ(tnet=tnet, dropout=0.3 if dropout is None else dropout,
                          features=features)
    if name == "pointnet_stats":
        if tnet:
            raise ValueError("pointnet_stats does not use T-Nets")
        return PointNetStats(dropout=0.3 if dropout is None else dropout, features=features)
    if name == "pointnet2":
        return PointNetPP(dropout=0.4 if dropout is None else dropout,
                          features=features)
    if name == "bev_cnn":
        kw = {}
        if bev_cell is not None:
            kw["cell"] = float(bev_cell)
        if bev_grid is not None:
            kw["grid"] = int(bev_grid)
        if bev_width is not None:
            kw["width"] = int(bev_width)
        if bev_depth is not None:
            kw["depth"] = int(bev_depth)
        if bev_channels is not None:
            kw["bev_channels"] = list(bev_channels)
        if bev_density_norm is not None:
            kw["density_norm"] = bool(bev_density_norm)
        if seg_height_ref is not None:
            kw["height_ref"] = str(seg_height_ref)
        return BEVCNN(dropout=0.3 if dropout is None else dropout,
                      features=features, **kw)
    if name == "pointnet2_seg":
        kw = {}
        if seg_npoints is not None:
            kw["npoints"] = tuple(int(n) for n in seg_npoints)
        if seg_radii is not None:
            kw["radii"] = tuple(float(r) for r in seg_radii)
        if seg_height_ref is not None:
            kw["height_ref"] = str(seg_height_ref)
        if seg_coord_ref is not None:
            kw["coord_ref"] = str(seg_coord_ref)
        return PointNetPPSeg(dropout=0.3 if dropout is None else dropout,
                             features=features, **kw)
    raise ValueError(f"unknown model {name!r} (pick from {sorted(MODELS)})")
