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

from ..dataset.neighborhoods import FEATURES, GEOMETRY, resolve_features  # noqa: F401  (re-exported)
from .models_meta import MODELS, model_task  # noqa: F401  (re-exported)

SENTINEL = 1.0e3  # farther than any real neighborhood coordinate (meters)


def _feature_buffer(names: list[str]) -> torch.Tensor:
    return torch.tensor([FEATURES.index(n) for n in names], dtype=torch.long)


def valid_mask(counts: torch.Tensor, n: int) -> torch.Tensor:
    """[B, N] bool; real points come first (see neighborhoods.build_...)."""
    return torch.arange(n, device=counts.device)[None, :] < counts[:, None]


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


def _ortho_penalty(t: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(t.shape[1], device=t.device)[None]
    return ((torch.bmm(t, t.transpose(1, 2)) - eye) ** 2).sum(dim=(1, 2)).mean()


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
    val, idx = torch.topk(d, nsample, dim=-1, largest=False)
    return torch.where(val > radius, idx[..., :1], idx)


class SegSetAbstraction(nn.Module):
    """Downsampling level: FPS centroids, kNN-in-ball grouping, max pool."""

    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: list[int]):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers: list[nn.Module] = []
        last = in_channel + 6  # relative offset + absolute position, as above
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
        grouped = torch.cat([local, grouped_xyz, _gather(feats, grp_idx)], -1)
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


class PointNetPPSeg(nn.Module):
    """Per-point rock/clear segmentation over a whole cropped frame.

    Input is the same [B, N, 4] contract as the classifiers, but N is a whole
    frame (4096 points spanning the crop box) instead of one 0.5 m ball, and
    the output is [B, N] logits instead of [B].

    The radii are scene-scale rather than neighborhood-scale: the classifier
    sees a 1 m sphere and works in centimeters, this sees an 8 x 8 m crop and
    has to find 20-30 cm rocks in it, which is the hard part. Rocks are ~2.6%
    of points, so the loss is prevalence-weighted.
    """

    def __init__(self, dropout: float = 0.3, features: list[str] | None = None,
                 npoints: tuple[int, int, int] = (512, 128, 32),
                 radii: tuple[float, float, float] = (0.25, 0.6, 1.4)):
        super().__init__()
        self.features = resolve_features(features)
        if self.features[:3] != list(GEOMETRY):
            raise ValueError("segmentation samples and groups by position, so it needs "
                             f"all of {list(GEOMETRY)} selected; got {self.features}")
        extra = self.features[3:]
        self.register_buffer("extra_idx", _feature_buffer(extra), persistent=False)
        c0 = len(extra)
        self.sa1 = SegSetAbstraction(npoints[0], radii[0], 32, c0, [32, 32, 64])
        self.sa2 = SegSetAbstraction(npoints[1], radii[1], 32, 64, [64, 64, 128])
        self.sa3 = SegSetAbstraction(npoints[2], radii[2], 32, 128, [128, 128, 256])
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
        xyz0 = torch.where(mask[..., None], points[..., :3],
                           torch.full_like(points[..., :3], SENTINEL))
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


def build_model(name: str, tnet: bool = False, dropout: float | None = None,
                features: list[str] | None = None) -> nn.Module:
    """``features=None`` means all of :data:`FEATURES` — the historical
    behavior, so checkpoints trained before the setting existed still load."""
    if name == "pointnet":
        return PointNet(tnet=tnet, dropout=0.3 if dropout is None else dropout,
                        features=features)
    if name == "pointnet2":
        return PointNetPP(dropout=0.4 if dropout is None else dropout,
                          features=features)
    if name == "pointnet2_seg":
        return PointNetPPSeg(dropout=0.3 if dropout is None else dropout,
                             features=features)
    raise ValueError(f"unknown model {name!r} (pick from {sorted(MODELS)})")
