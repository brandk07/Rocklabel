"""The map pipeline on the graphics card: histogram, channels and training crops.

The CPU version in :mod:`.grid` and :mod:`.data` is the readable reference; this
is the one training and grading use, because building a crop on the CPU costs
about half a second and the card sits idle. Both compute the same channels (see
:data:`.grid.CHANNELS`); the one place they differ is how the local ground is
filled across unobserved cells before it is filtered (nearest observed cell on
the CPU, repeated neighbour averaging here), which moves the ground by a few
millimetres next to large holes.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ...dataset.labeling import distance_to_polygon_xy, points_in_polygon_xy
from .grid import (CELL_M, DENSITY_CAP, DZ_M, GROUND_PERCENTILE, GROUND_WINDOW_M, NBINS,
                   PROFILE_CHANNELS, PROFILE_LO, PROFILE_STEP, Z_LO)

IGNORE = -1


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #
def _quantile(cum: torch.Tensor, total: torch.Tensor, q: float) -> torch.Tensor:
    target = torch.clamp(torch.ceil(total * q), min=1).unsqueeze(-1)
    return (cum < target).sum(-1)


def _fill(a: torch.Tensor, valid: torch.Tensor, max_iter: int = 200) -> torch.Tensor:
    """Fill invalid cells of ``a`` [B,h,w] by repeated 3x3 averaging of the
    filled neighbourhood - a cheap stand-in for nearest-cell filling."""
    a = torch.where(valid, a, torch.zeros_like(a))
    m = valid.float()
    x, w = a.unsqueeze(1), m.unsqueeze(1)
    for _ in range(max_iter):
        if bool(w.min() > 0):
            break
        s = F.avg_pool2d(F.pad(x * w, (1, 1, 1, 1), mode="replicate"), 3, 1)
        c = F.avg_pool2d(F.pad(w, (1, 1, 1, 1), mode="replicate"), 3, 1)
        new = (c > 0) & (w == 0)
        x = torch.where(new, s / c.clamp(min=1e-6), x)
        w = torch.where(new, torch.ones_like(w), w)
    return x[:, 0]


def local_ground(q50: torch.Tensor, valid: torch.Tensor, cell_m: float = CELL_M) -> torch.Tensor:
    """Smooth ground surface [B,H,W] (metres above floor); see ``grid.local_ground``."""
    k = max(int(round(0.10 / cell_m)), 1)
    b, h, w = q50.shape
    ph, pw = (-h) % k, (-w) % k
    z = F.pad(torch.where(valid, q50, torch.full_like(q50, float("nan"))), (0, pw, 0, ph),
              value=float("nan"))
    hh, ww = z.shape[1] // k, z.shape[2] // k
    blocks = z.reshape(b, hh, k, ww, k).permute(0, 1, 3, 2, 4).reshape(b, hh, ww, k * k)
    cvalid = torch.isfinite(blocks).any(-1)
    coarse = torch.nanmedian(torch.where(cvalid.unsqueeze(-1), blocks,
                                         torch.zeros_like(blocks)), dim=-1).values
    coarse = _fill(coarse, cvalid)
    size = max(int(round(GROUND_WINDOW_M / (k * cell_m))) | 1, 3)
    r = size // 2
    pad = F.pad(coarse.unsqueeze(1), (r, r, r, r), mode="replicate")
    cols = F.unfold(pad, size)                                  # [B, size*size, hh*ww]
    kth = max(int(math.ceil(GROUND_PERCENTILE / 100 * size * size)), 1)
    g = cols.kthvalue(kth, dim=1).values.reshape(b, 1, hh, ww)
    g = F.avg_pool2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), 3, 1)
    up = F.interpolate(g, scale_factor=k, mode="bilinear", align_corners=False)
    return up[:, 0, :h, :w]


def features_t(hist: torch.Tensor, bsum: torch.Tensor, bmax: torch.Tensor,
               has_bright: torch.Tensor, feature_set: str = "base"):
    """Channels ``[B,C,H,W]`` from a histogram batch ``[B,H,W,NBINS]``.

    Returns ``(x, valid, q90)`` with ``q90`` in metres above the floor.
    """
    hist = hist.float()
    cum = hist.cumsum(-1)
    total = cum[..., -1]
    valid = total > 0
    top = NBINS - 1 - torch.argmax((hist.flip(-1) > 0).to(torch.uint8), dim=-1)

    def z(i):
        return Z_LO + (i.float() + 0.5) * DZ_M

    q10 = z(_quantile(cum, total, 0.10))
    q50 = z(_quantile(cum, total, 0.50))
    q90 = z(_quantile(cum, total, 0.90))
    zmax = z(top)
    g = local_ground(q50, valid)
    v = valid.float()
    hb = has_bright.float().view(-1, 1, 1)
    bmean = torch.where(valid, bsum / total.clamp(min=1), torch.zeros_like(bsum)) * hb
    bmx = torch.where(valid & torch.isfinite(bmax), bmax, torch.zeros_like(bmax)) * hb
    chans = [v, torch.log1p(total.clamp(max=DENSITY_CAP)) / 4.0,
             (q10 - g) * 10 * v, (q50 - g) * 10 * v, (q90 - g) * 10 * v,
             (zmax - g) * 10 * v, (q90 - q10) * 10 * v, g * 10,
             bmean.clamp(-3, 3) * v, bmx.clamp(-3, 3) * v, hb.expand_as(v)]
    if feature_set == "profile":
        per = int(round(PROFILE_STEP / DZ_M))
        kf = len(PROFILE_CHANNELS) * per
        off = torch.round((g + PROFILE_LO - Z_LO) / DZ_M).long()
        src = off.unsqueeze(-1) + torch.arange(kf, device=hist.device)
        inside = (src >= 0) & (src < NBINS)
        fine = torch.gather(hist, -1, src.clamp(0, NBINS - 1)) * inside
        coarse = fine.reshape(*fine.shape[:-1], -1, per).sum(-1)
        prof = coarse / total.clamp(min=1).unsqueeze(-1) * 4.0
        chans.extend(prof.permute(3, 0, 1, 2))
    x = torch.stack(chans, 1)
    return x, valid, q90


def features_from_grid(grid, device, feature_set: str = "base"):
    """GPU channels for a CPU :class:`~.grid.HeightGrid` (whole-map grading)."""
    hist = torch.from_numpy(grid.hist.reshape(grid.h, grid.w, NBINS).astype(np.int32)).to(device)
    bsum = torch.from_numpy(grid.bsum.reshape(grid.h, grid.w)).to(device)
    bmax = torch.from_numpy(grid.bmax.reshape(grid.h, grid.w)).to(device)
    hb = torch.tensor([1.0 if grid.has_bright else 0.0], device=device)
    x, valid, q90 = features_t(hist[None], bsum[None], bmax[None], hb, feature_set)
    return x, valid[0].cpu().numpy(), (q90[0] + grid.floor_z).cpu().numpy()


# --------------------------------------------------------------------------- #
# Training crops
# --------------------------------------------------------------------------- #
LABEL_RES_M = 0.01


class LabelImage:
    """A recording's targets painted once at 1 cm in the world frame.

    1 on a rock footprint, 0 clear, ``IGNORE`` in the 5 cm shell around each
    rock, outside the arena and outside ``region``. Crops read it through their
    own transform, so no polygon test runs during training.
    """

    def __init__(self, labels, lo, hi, region=None, shell_m: float = 0.05,
                 outside_clear_m: float = 0.0):
        self.x0, self.y0 = float(lo[0]), float(lo[1])
        self.w = int(math.ceil((hi[0] - lo[0]) / LABEL_RES_M))
        self.h = int(math.ceil((hi[1] - lo[1]) / LABEL_RES_M))
        xs = self.x0 + (np.arange(self.w) + 0.5) * LABEL_RES_M
        ys = self.y0 + (np.arange(self.h) + 0.5) * LABEL_RES_M
        gx, gy = np.meshgrid(xs, ys)
        cxy = np.c_[gx.ravel(), gy.ravel()]
        y = np.zeros(len(cxy), np.int8)
        for rock in labels.rocks:
            if rock.shape == "polygon":
                v = np.asarray(rock.vertices, float)
            else:
                a = np.linspace(0, 2 * np.pi, 32, endpoint=False)
                v = np.asarray(rock.center[:2]) + float(rock.radius) * np.c_[np.cos(a), np.sin(a)]
            lo_, hi_ = v.min(0) - shell_m, v.max(0) + shell_m
            near = np.nonzero((cxy[:, 0] >= lo_[0]) & (cxy[:, 0] <= hi_[0])
                              & (cxy[:, 1] >= lo_[1]) & (cxy[:, 1] <= hi_[1]))[0]
            if not len(near):
                continue
            sub = cxy[near]
            inside = points_in_polygon_xy(sub, v)
            lab = np.where(inside, 1, 0).astype(np.int8)
            out = ~inside
            if out.any():
                d = distance_to_polygon_xy(sub[out], v)
                lab[np.nonzero(out)[0][d <= shell_m]] = IGNORE
            cur = y[near]
            y[near] = np.where(lab == 1, 1, np.where((lab == IGNORE) & (cur != 1), IGNORE, cur))
        keep = np.ones(len(cxy), bool)
        if labels.arena is not None:
            inside = points_in_polygon_xy(cxy, labels.arena)
            keep &= inside
            if outside_clear_m > 0:
                # Walls and whatever stands beyond them are not rocks: label a
                # band outside the arena clear instead of leaving it unknown,
                # or the model never learns that a wall is not a rock.
                band = ~inside & (distance_to_polygon_xy(cxy, labels.arena) <= outside_clear_m)
                keep |= band & (y == 0)
        if region is not None:
            keep &= region(cxy)
        y[~keep] = IGNORE
        self.img = y.reshape(self.h, self.w)
        self.t = None

    def to(self, device):
        self.t = torch.from_numpy(self.img).to(device)
        return self

    def sample(self, xy: torch.Tensor) -> torch.Tensor:
        c = torch.floor((xy[..., 0] - self.x0) / LABEL_RES_M).long()
        r = torch.floor((xy[..., 1] - self.y0) / LABEL_RES_M).long()
        ok = (c >= 0) & (c < self.w) & (r >= 0) & (r < self.h)
        out = torch.full(c.shape, IGNORE, dtype=torch.int8, device=xy.device)
        out[ok] = self.t[r[ok], c[ok]]
        return out



# --------------------------------------------------------------------------- #
# Rock transplants
# --------------------------------------------------------------------------- #
def _pip(xy: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Even-odd point-in-polygon for ``xy`` [N,2] against vertices ``v`` [M,2]."""
    x, y = xy[:, 0], xy[:, 1]
    inside = torch.zeros(len(xy), dtype=torch.bool, device=xy.device)
    v1 = torch.roll(v, -1, 0)
    for (x0, y0), (x1, y1) in zip(v.tolist(), v1.tolist()):
        if y0 == y1:
            continue
        cross = (y0 > y) != (y1 > y)
        xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= cross & (x < xi)
    return inside


def _poly_dist(xy: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    a = v.unsqueeze(0)
    b = torch.roll(v, -1, 0).unsqueeze(0)
    p = xy.unsqueeze(1)
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-12)).clamp(0, 1)
    d = p - (a + t.unsqueeze(-1) * ab)
    return d.norm(dim=-1).min(1).values


class RockBank:
    """Every labelled rock's own returns, lifted off its ground, ready to be
    set down somewhere else.

    Pasting real rocks into other real maps is the lidar-detection trick of
    ground-truth sampling: it shows the network a rock against floors, slopes
    and rubble it was never actually photographed on - most usefully the rough
    dug-over ground that the arena's false detections pile up on.
    """

    def __init__(self, sources, device, max_points: int = 60_000, grow_m: float = 0.08):
        self.rocks = []
        for src in sources:
            cloud = src.cloud
            for rock in cloud.labels.rocks:
                if rock.shape != "polygon":
                    continue
                c = np.asarray(rock.vertices, float).mean(0)
                if src.region is not None and not src.region(c[None])[0]:
                    continue
                v = np.asarray(rock.vertices, float)
                r = np.abs(v - c).max() + 0.6
                near = cloud.select(c, r)
                p = cloud.xyz[near].astype(np.float64)
                d = distance_to_polygon_xy(p[:, :2], v)
                ins = points_in_polygon_xy(p[:, :2], v)
                on = ins | (d <= grow_m)
                ring = (~ins) & (d > 0.15) & (d < 0.5)
                if on.sum() < 50:
                    continue
                zr = p[:, 2] - cloud.floor_z
                ground = (np.median(zr[ring]) if ring.sum() >= 30
                          else np.percentile(zr[on], 10))
                idx = np.nonzero(on)[0]
                if len(idx) > max_points:
                    idx = np.random.default_rng(rock.id).choice(idx, max_points, replace=False)
                pts = np.c_[p[idx, :2] - c, zr[idx] - ground]
                br = (np.zeros(len(idx), np.float32) if cloud.brightness is None
                      else cloud.brightness[near][idx])
                self.rocks.append({
                    "name": f"{cloud.name}:{rock.id}",
                    "pts": torch.tensor(pts, dtype=torch.float32, device=device),
                    "bright": torch.tensor(br, dtype=torch.float32, device=device),
                    "poly": torch.tensor(v - c, dtype=torch.float32, device=device),
                    "has_bright": cloud.brightness is not None})
        print(f"rock bank: {len(self.rocks)} rocks", flush=True)


class GpuSource:
    """A cloud resident on the card, with its label image.

    A long recording is thinned at random to ``max_points`` (time order kept):
    35 minutes of arena is about fifty times a volleyball run's density, far
    past where any channel still changes (the density channel saturates at
    :data:`.grid.DENSITY_CAP` returns per cell), and would not fit on the card.
    """

    def __init__(self, cloud, weight: float, region, device, max_points: int = 12_000_000,
                 outside_clear_m: float = 0.0):
        self.cloud = cloud
        self.weight = float(weight)
        self.region = region
        n = len(cloud.xyz)
        pick = slice(None)
        if n > max_points:
            pick = np.sort(np.random.default_rng(0).choice(n, max_points, replace=False))
        self.xyz = torch.from_numpy(np.ascontiguousarray(cloud.xyz[pick])).to(device)
        self.t = torch.from_numpy(np.ascontiguousarray(cloud.t[pick])).to(device)
        self.bright = (None if cloud.brightness is None else
                       torch.from_numpy(cloud.brightness[pick].astype(np.float16)).to(device))
        lo = cloud.xyz[:, :2].min(0) - 0.5
        hi = cloud.xyz[:, :2].max(0) + 0.5
        self.labels = LabelImage(cloud.labels, lo, hi, region,
                                 outside_clear_m=outside_clear_m).to(device)
        self.rocks = np.array([r.center[:2] for r in cloud.labels.rocks], float)
        # A pool of observed positions to centre crops on, in the region.
        rng = np.random.default_rng(0)
        pick = rng.choice(len(cloud.xyz), size=min(200_000, len(cloud.xyz)), replace=False)
        pick.sort()
        pos = cloud.xyz[pick, :2].astype(float)
        keep = np.ones(len(pos), bool) if region is None else region(pos)
        self.pool_xy, self.pool_t = pos[keep], cloud.t[pick][keep]


class GpuCrops:
    """Batches of augmented map crops, built entirely on the card."""

    def __init__(self, sources: list[GpuSource], size: int, aug, device,
                 feature_set: str = "base", rock_frac: float = 0.5, bank: RockBank | None = None):
        self.sources = sources
        self.bank = bank
        w = np.array([s.weight for s in sources], float)
        self.p = w / w.sum()
        self.size = int(size)
        self.aug = aug
        self.dev = device
        self.feature_set = feature_set
        self.rock_frac = rock_frac

    def _plan(self, rng):
        a = self.aug
        si = int(rng.choice(len(self.sources), p=self.p))
        src = self.sources[si]
        dur = src.cloud.duration
        t_max = dur * rng.uniform(*a.prefix) if (not a.off and rng.random() < a.prefix_p) else dur
        c = None
        for _ in range(50):
            if len(src.rocks) and rng.random() < self.rock_frac:
                c = src.rocks[rng.integers(len(src.rocks))] + rng.uniform(-1.2, 1.2, 2)
            else:
                ok = np.nonzero(src.pool_t <= t_max)[0]
                if not len(ok):
                    ok = np.arange(len(src.pool_t))
                c = src.pool_xy[ok[rng.integers(len(ok))]]
            if src.region is None or src.region(np.asarray(c)[None])[0]:
                break
        return src, float(t_max), np.asarray(c, float)


    def _paste(self, xy, zz, bright, y_local, rng, half):
        """Set 1-3 banked rocks down on this crop. Returns new points and the
        pasted outlines (crop frame) to be painted into the labels."""
        a = self.aug
        polys = []
        dev = xy.device
        for _ in range(int(rng.integers(1, a.paste_max + 1))):
            rock = self.bank.rocks[int(rng.integers(len(self.bank.rocks)))]
            if (bright is not None) and not rock["has_bright"]:
                continue
            for _try in range(10):
                pc = rng.uniform(-half + 0.5, half - 0.5, 2)
                pct = torch.tensor(pc, dtype=torch.float32, device=dev)
                dist = (xy - pct).norm(dim=1)
                ring = (dist > 0.3) & (dist < 0.7)
                if int(ring.sum()) < 200:
                    continue
                ang = rng.uniform(0, 2 * np.pi)
                sc = rng.uniform(*a.paste_scale)
                ca, sa = math.cos(ang), math.sin(ang)
                rot = torch.tensor([[ca, sa], [-sa, ca]], dtype=torch.float32, device=dev) * sc
                poly = rock["poly"] @ rot + pct
                # Only on labelled clear ground, never on top of a real rock
                # (or another transplant).
                under_cells = _pip(y_local[0], poly)
                if bool((under_cells & (y_local[1] != 0)).any()):
                    continue
                if any(bool(_pip(poly, q).any()) for q in polys):
                    continue
                ground = zz[ring].median()
                under = _pip(xy, poly)
                keep = ~under
                n = len(rock["pts"])
                take = torch.rand(n, device=dev) < rng.uniform(0.3, 1.0)
                rp = rock["pts"][take]
                new_xy = rp[:, :2] @ rot + pct
                new_z = rp[:, 2] * sc + ground
                xy = torch.cat([xy[keep], new_xy])
                zz = torch.cat([zz[keep], new_z])
                if bright is not None:
                    bright = torch.cat([bright[keep], rock["bright"][take]])
                polys.append(poly)
                break
        return xy, zz, bright, polys

    def batch(self, n: int, rng):
        a = self.aug
        S = self.size
        half = S * CELL_M / 2
        dev = self.dev
        flat_all, bidx_all, bright_all, cell_all = [], [], [], []
        has_b = torch.ones(n, device=dev)
        ys = []
        for b in range(n):
            src, t_max, c = self._plan(rng)
            scale = 1.0 if a.off else rng.uniform(*a.scale_xy)
            ang = 0.0 if (a.off or not a.rotate) else rng.uniform(0, 2 * np.pi)
            flip = (not a.off) and rng.random() < 0.5
            ca, sa = math.cos(ang), math.sin(ang)
            m = np.array([[ca, -sa], [sa, ca]]) * scale
            if flip:
                m = m @ np.diag([1.0, -1.0])
            mt = torch.tensor(m.T, dtype=torch.float32, device=dev)
            ct = torch.tensor(c, dtype=torch.float32, device=dev)
            r = half * 1.5 / scale
            p = src.xyz
            sel = ((p[:, 0] - ct[0]).abs() <= r) & ((p[:, 1] - ct[1]).abs() <= r) & (src.t <= t_max)
            idx = torch.nonzero(sel, as_tuple=True)[0]
            if not a.off and rng.random() < a.thin_p and len(idx):
                keep = torch.rand(len(idx), device=dev) < rng.uniform(*a.thin)
                idx = idx[keep]
            pts = p[idx]
            xy = (pts[:, :2] - ct) @ mt
            zz = pts[:, 2] - float(src.cloud.floor_z)
            bright = None if src.bright is None else src.bright[idx].float()
            g = (torch.arange(S, device=dev, dtype=torch.float32) + 0.5) * CELL_M - half
            gy, gx = torch.meshgrid(g, g, indexing="ij")
            local = torch.stack([gx, gy], -1).reshape(-1, 2)
            world = local @ torch.tensor(np.linalg.inv(m).T, dtype=torch.float32, device=dev) + ct
            ylab = src.labels.sample(world)
            polys = []
            if (self.bank is not None and not a.off and self.bank.rocks
                    and rng.random() < a.paste_p):
                xy, zz, bright, polys = self._paste(xy, zz, bright, (local, ylab), rng, half)
            if not a.off:
                zz = zz * rng.uniform(*a.scale_z)
                if rng.random() < a.terrain_p:
                    tilt = torch.tensor(rng.uniform(-a.tilt, a.tilt, 2), dtype=torch.float32, device=dev)
                    dz = xy @ tilt
                    for _ in range(int(rng.integers(1, 4))):
                        cc = torch.tensor(rng.uniform(-half, half, 2), dtype=torch.float32, device=dev)
                        s = rng.uniform(0.4, 1.5)
                        amp = rng.uniform(-a.bump_amp_m, a.bump_amp_m)
                        dz = dz + amp * torch.exp(-((xy - cc) ** 2).sum(1) / (2 * s * s))
                    zz = zz + dz
                zz = zz + torch.randn_like(zz) * rng.uniform(*a.jitter_m)
            if bright is not None and not a.off:
                if rng.random() < a.bright_drop_p:
                    bright = None
                else:
                    bright = bright * rng.uniform(*a.bright_gain) + rng.normal(0, 0.1)
            if bright is None:
                has_b[b] = 0.0
            col = torch.floor((xy[:, 0] + half) / CELL_M).long()
            row = torch.floor((xy[:, 1] + half) / CELL_M).long()
            zb = torch.floor((zz - Z_LO) / DZ_M).long()
            ok = (col >= 0) & (col < S) & (row >= 0) & (row < S) & (zb >= 0) & (zb < NBINS)
            cell = (b * S * S + row * S + col)[ok]
            flat_all.append(cell * NBINS + zb[ok])
            cell_all.append(cell)
            bright_all.append(torch.zeros(int(ok.sum()), device=dev) if bright is None else bright[ok])
            # Labels: crop cell centres read off the world label image, then
            # every transplant painted in (5 cm ignore shell around it).
            for poly in polys:
                inside = _pip(local, poly)
                shell = (~inside) & (_poly_dist(local, poly) <= 0.05)
                ylab = torch.where(inside, torch.ones_like(ylab),
                                   torch.where(shell & (ylab != 1),
                                               torch.full_like(ylab, IGNORE), ylab))
            ys.append(ylab.reshape(S, S))
        # Histogram and channels a few crops at a time: a whole batch's
        # histogram is ~330 MB of int64 before the channels are even built.
        xs = []
        for c0 in range(0, n, 4):
            c1 = min(c0 + 4, n)
            base = c0 * S * S
            fl = torch.cat(flat_all[c0:c1]) - base * NBINS
            ce = torch.cat(cell_all[c0:c1]) - base
            bv = torch.cat(bright_all[c0:c1])
            k = c1 - c0
            hist = torch.bincount(fl, minlength=k * S * S * NBINS).to(torch.int32)
            hist = hist.reshape(k, S, S, NBINS)
            bsum = torch.bincount(ce, weights=bv, minlength=k * S * S).reshape(k, S, S).float()
            bmax = torch.full((k * S * S,), float("-inf"), device=dev)
            bmax = bmax.scatter_reduce(0, ce, bv, reduce="amax").reshape(k, S, S)
            x, valid, _ = features_t(hist, bsum, bmax, has_b[c0:c1], self.feature_set)
            xs.append((x, valid))
            del hist
        x = torch.cat([a for a, _ in xs])
        valid = torch.cat([v for _, v in xs])
        y = torch.stack(ys).long()
        y[~valid] = IGNORE
        return x, y
