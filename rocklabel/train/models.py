"""PointNet and PointNet++ (SSG) binary rock classifiers, plain PyTorch.

Input contract (matches neighborhoods.py): points [B, 256, 4] with channels
[dx, dy, dz, intensity] already canonicalized (dx/dy center-relative, dz
ground-relative), counts [B] = number of REAL points. Padding repeats real
points and is appended AFTER them, so the validity mask is arange(N) < counts.

Padding policy: PointNet's max-pool is duplicate-safe, so masking there is
belt-and-braces. PointNet++ is not: duplicates would waste FPS centroids and
skew nothing else only because every pool here is a max. We therefore mask
explicitly - FPS picks farthest among real points only, and padded points are
moved to a far sentinel so ball queries never gather them. Group-all pooling
masks invalid columns. With counts=N (no padding info) both models degrade
gracefully to the classic unmasked behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SENTINEL = 1.0e3  # farther than any real neighborhood coordinate (meters)


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

    def __init__(self, tnet: bool = False, dropout: float = 0.3):
        super().__init__()
        self.use_tnet = tnet
        self.input_tnet = TNet(3) if tnet else None
        self.feature_tnet = TNet(64) if tnet else None
        self.mlp1 = _mlp1d([4, 64, 64])
        self.mlp2 = _mlp1d([64, 64, 128, 1024])
        self.head = _head(1024, dropout)
        self._reg = torch.zeros(())

    def forward(self, points: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        mask = valid_mask(counts, points.shape[1])
        x = points.transpose(1, 2)  # [B, 4, N]
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
    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: list[int]):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers: list[nn.Module] = []
        last = in_channel + 3  # +3: local offsets are appended to features
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
        grouped = _gather(xyz, grp_idx) - new_xyz[:, :, None]        # [B,S,K,3]
        grouped = torch.cat([grouped, _gather(feats, grp_idx)], -1)  # [B,S,K,3+C]
        x = self.mlp(grouped.permute(0, 3, 1, 2))                    # [B,C',S,K]
        return new_xyz, x.max(dim=3).values.transpose(1, 2)


class PointNetPP(nn.Module):
    """PointNet++ SSG sized for these neighborhoods: typically only ~20-120
    real points inside the 0.5 m radius, so the hierarchy is shallow and the
    ball radii are coarse relative to the classic 1k-point configs."""

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.sa1 = SetAbstraction(64, 0.15, 16, in_channel=1, mlp=[64, 64, 128])
        self.sa2 = SetAbstraction(16, 0.30, 16, in_channel=128, mlp=[128, 128, 256])
        self.global_mlp = _mlp1d([256 + 3, 256, 512, 1024])
        self.head = _head(1024, dropout)

    def forward(self, points: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        mask = valid_mask(counts, points.shape[1])
        # Exile padded points so no ball query or FPS pick can reach them.
        xyz = torch.where(mask[..., None], points[..., :3], torch.full_like(points[..., :3], SENTINEL))
        feats = points[..., 3:]  # intensity
        xyz, feats = self.sa1(xyz, feats, mask)
        # After SA1 every centroid is a real point (FPS was masked), so all
        # levels below are fully valid.
        full = torch.ones(xyz.shape[:2], dtype=torch.bool, device=xyz.device)
        xyz, feats = self.sa2(xyz, feats, full)
        x = self.global_mlp(torch.cat([xyz, feats], -1).transpose(1, 2))
        return self.head(x.max(dim=2).values).squeeze(-1)

    def pop_regularizer(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)


def build_model(name: str, tnet: bool = False, dropout: float | None = None) -> nn.Module:
    if name == "pointnet":
        return PointNet(tnet=tnet, dropout=0.3 if dropout is None else dropout)
    if name == "pointnet2":
        return PointNetPP(dropout=0.4 if dropout is None else dropout)
    raise ValueError(f"unknown model {name!r} (pointnet | pointnet2)")
