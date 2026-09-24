"""Draw where a map model thinks the rocks are, on any stacked recording.

Labels are optional: this is how a fresh recording - a new arena, a practice
run - gets looked at before anyone has labelled it. The picture has two panels:
the map's height above the local ground, and the same map with every cell the
model calls rock at or above the threshold painted over it. With labels, the
rock outlines are drawn too. The probabilities are also written to an .npz.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from . import grade as G
from .clouds import Cloud
from .evaluate import load
from .gpu import features_from_grid
from .grid import CELL_M, HeightGrid


def cloud_grid(cloud: Cloud, margin_m: float = 0.5) -> HeightGrid:
    """The whole map, over the arena if labelled, else over where returns are dense."""
    if cloud.labels is not None and cloud.labels.arena is not None:
        lo, hi = cloud.labels.arena.min(0), cloud.labels.arena.max(0)
        margin_m = max(margin_m, 1.5)
    else:
        xy = cloud.xyz[::50, :2]
        lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
    g = HeightGrid.around(lo, hi, cloud.floor_z, CELL_M, margin_m)
    for a in range(0, len(cloud.xyz), 4_000_000):
        g.add(cloud.xyz[a:a + 4_000_000].astype(np.float64),
              None if cloud.brightness is None else cloud.brightness[a:a + 4_000_000])
    if cloud.brightness is None:
        g.has_bright = False
    return g


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("cloud", help="a stacked recording from mapnet-cloud (.npz)")
    ap.add_argument("--checkpoint", nargs="+", required=True,
                    help="map-model checkpoint(s); several are averaged")
    ap.add_argument("--out", required=True, help="picture to write (.png)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--labels", default=None,
                    help="draw these rock outlines too (default: the cloud's own, if any)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    return run(ap.parse_args(argv))


def run(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fs, _ = load(args.checkpoint, dev)
    cloud = Cloud(args.cloud, args.labels)
    g = cloud_grid(cloud)
    prob, st = G.predict(model, g, dev, fs)
    x, _, _ = features_from_grid(g, dev, fs)
    rel = np.where(st["valid"], x[0, 4].cpu().numpy() / 10, np.nan)
    ext = [g.x0, g.x0 + g.w * g.cell, g.y0, g.y0 + g.h * g.cell]
    fig, ax = plt.subplots(1, 2, figsize=(18, 7.5), sharex=True, sharey=True)
    for a in ax:
        a.imshow(rel, origin="lower", extent=ext, cmap="gray", vmin=-0.1, vmax=0.4)
        if cloud.labels is not None:
            for rock in cloud.labels.rocks:
                if rock.shape == "polygon":
                    v = np.vstack([rock.vertices, np.asarray(rock.vertices)[:1]])
                    a.plot(v[:, 0], v[:, 1], color="#4cc9f0", lw=1)
            if cloud.labels.arena is not None:
                v = np.vstack([cloud.labels.arena, cloud.labels.arena[:1]])
                a.plot(v[:, 0], v[:, 1], color="y", lw=1)
        a.set_aspect("equal")
    hot = np.ma.masked_where(~st["valid"] | (prob < args.threshold), prob)
    ax[1].imshow(hot, origin="lower", extent=ext, cmap="autumn_r", vmin=args.threshold,
                 vmax=1, alpha=0.85)
    ax[1].plot(cloud.bases[:, 0], cloud.bases[:, 1], color="c", lw=0.4)
    ax[0].set_title(f"{cloud.name}: height above local ground (0-40 cm)")
    ax[1].set_title(f"rock probability >= {args.threshold:g}; robot path in cyan")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=90)
    plt.close(fig)
    np.savez_compressed(os.path.splitext(args.out)[0] + ".npz", prob=prob.astype(np.float16),
                        valid=st["valid"], x0=g.x0, y0=g.y0, cell=g.cell)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    main()
