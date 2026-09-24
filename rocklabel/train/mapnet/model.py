"""A small U-Net over the map channels.

Four downsamplings give each output cell a view of roughly two metres of map
around it - enough to see a whole rock with the ground on every side of it,
which is exactly what a single sweep's 0.5 m ball could not.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int, drop: float = 0.0) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout),
              nn.SiLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout),
              nn.SiLU(inplace=True)]
    if drop:
        layers.append(nn.Dropout2d(drop))
    return nn.Sequential(*layers)


class MapUNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 32, depth: int = 4, drop: float = 0.1):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.stem = _block(in_ch, chs[0])
        self.down = nn.ModuleList(_block(chs[i], chs[i + 1], drop if i >= depth - 2 else 0.0)
                                  for i in range(depth))
        self.up = nn.ModuleList(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                for i in reversed(range(depth)))
        self.dec = nn.ModuleList(_block(chs[i] * 2, chs[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chs[0], 1, 1)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        skips = [self.stem(x)]
        for d in self.down:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        y = skips.pop()
        for up, dec in zip(self.up, self.dec):
            y = up(y)
            y = dec(torch.cat([y, skips.pop()], 1))
        return self.head(y)[..., :h, :w]


def build(in_ch: int, arch: str = "unet32", **kw) -> nn.Module:
    base = {"unet16": 16, "unet32": 32, "unet48": 48, "unet64": 64}[arch]
    return MapUNet(in_ch, base=base, **kw)
