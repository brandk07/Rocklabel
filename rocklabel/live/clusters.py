"""Group above-threshold predictions into rock outlines.

The model answers one point at a time: "is this a rock". That is enough to
paint the cloud, but not enough to say *how many* rocks are in front of the
sensor, or where one ends and the next begins — and a single stray center over
the threshold looks exactly like a real detection when all you draw is dots.

A rock is a *clump*. This module turns the detection cloud into objects:

1. link every pair of detections closer than ``link_m`` and take the connected
   groups (single-link clustering, the same idea as DBSCAN without the density
   term — the prediction map is already on a regular voxel grid, so a plain
   distance link is all the structure that is available);
2. throw away any group with fewer than ``min_points`` detections — that is the
   noise gate, and it is the knob worth turning first;
3. wrap each survivor in a convex outline, inflated by ``pad_m`` so the polygon
   covers the ground the detections *represent* rather than the exact grid
   points they landed on.

Nothing here touches the model or the prediction map: outlines are a reading of
what the model already said, so the settings behind them are display settings
and changing one never costs a scoring pass.

Both viewers draw from this one implementation — the Open3D window builds line
loops from :func:`outline_wireframe`, the web panel's overhead map draws the
same polygons in 2D — so the two surfaces can never disagree about what counts
as a rock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Defaults for the two knobs, shared by the scorer's settings and the panels.
#: 0.30 m links neighbouring cells of any candidate grid this project uses
#: without bridging two rocks a hand's width apart; 5 detections is roughly the
#: smallest clump a real rock produces at typical voxel spacing.
DEFAULT_LINK_M = 0.30
DEFAULT_MIN_POINTS = 5

#: Detections fed to the clustering, highest probability first. The pair search
#: is the expensive step and it grows with the square of the local density, so
#: this is the guard rail that keeps a wide-open threshold from stalling the
#: panel. The result says when it bit.
MAX_INPUT_POINTS = 20_000

#: Smallest outline inflation, in metres. Three detections in a straight line
#: have no area at all, and a hull of exactly zero width cannot be drawn — a
#: 2 cm skin makes every outline a real polygon.
MIN_PAD_M = 0.02

#: Unit octagon. Each detection is replaced by its own little disc of this
#: shape before the hull is taken, which is what turns the hull of a handful of
#: grid points into an outline with real area around them.
_RING = np.array([
    (1.0, 0.0), (0.7071, 0.7071), (0.0, 1.0), (-0.7071, 0.7071),
    (-1.0, 0.0), (-0.7071, -0.7071), (0.0, -1.0), (0.7071, -0.7071),
])


@dataclass
class Rock:
    """One outlined clump of detections."""

    #: Outline as an open ring of (x, y) world coordinates, counter-clockwise.
    polygon: np.ndarray
    #: Detections that formed it — the number the noise gate compares against.
    points: int
    prob_mean: float
    prob_max: float
    #: Centroid of the detections (x, y, z), and the height band they span.
    center: np.ndarray
    z_min: float
    z_max: float
    area_m2: float


@dataclass
class Outlines:
    """Every rock found in one prediction map, plus what was thrown away."""

    rocks: list[Rock] = field(default_factory=list)
    #: Detections that landed in a group too small to be a rock, and how many
    #: such groups there were. This is the noise gate's bill, and the reason
    #: the panels print it: it is the only way to see that ``min_points`` is
    #: eating real rocks.
    noise_points: int = 0
    noise_groups: int = 0
    #: Detections considered at all (after the cap below).
    input_points: int = 0
    #: True when :data:`MAX_INPUT_POINTS` dropped the weakest detections.
    capped: bool = False

    def __len__(self) -> int:
        return len(self.rocks)

    def describe(self) -> str:
        """One line for a readout cell: how many rocks, and how big."""
        if not self.rocks:
            return "no rocks" if self.input_points else "no detections yet"
        biggest = max(r.area_m2 for r in self.rocks)
        held = sum(r.points for r in self.rocks)
        return (f"{len(self.rocks)} rock{'' if len(self.rocks) == 1 else 's'} · "
                f"{held:,} pts · largest {biggest:.2f} m²")

    def noise_note(self) -> str:
        """One line for a readout cell: what the noise gate threw away."""
        if not self.input_points:
            return "—"
        if not self.noise_points:
            return f"none · all {self.input_points:,} detections used"
        cap = " · capped" if self.capped else ""
        return (f"{self.noise_points:,} of {self.input_points:,} detections in "
                f"{self.noise_groups} clump{'' if self.noise_groups == 1 else 's'}"
                f"{cap}")


def find_rocks(
    centers: np.ndarray,
    probs: np.ndarray,
    link_m: float = DEFAULT_LINK_M,
    min_points: int = DEFAULT_MIN_POINTS,
    pad_m: float = 0.0,
    max_points: int = MAX_INPUT_POINTS,
) -> Outlines:
    """Cluster detections and outline the clumps big enough to be rocks.

    ``centers``/``probs`` are what :meth:`LiveScorer.detections` returns — the
    above-threshold part of the prediction map. Linking happens in 3D so a rock
    is not merged with the ceiling above it; the outline is the 2D footprint,
    which is the shape you steer a robot around.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    centers = np.asarray(centers, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if centers.ndim != 2 or centers.shape[0] == 0:
        return Outlines()

    capped = False
    if centers.shape[0] > int(max_points):
        # Keep the confident ones: if detections have to be dropped, the
        # marginal ones are the ones a rock can most afford to lose.
        keep = np.argpartition(probs, -int(max_points))[-int(max_points):]
        centers, probs = centers[keep], probs[keep]
        capped = True

    n = centers.shape[0]
    link = max(float(link_m), 1e-3)
    tree = cKDTree(centers)
    pairs = tree.query_pairs(link, output_type="ndarray")
    if len(pairs):
        adj = coo_matrix(
            (np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])),
            shape=(n, n),
        )
        n_groups, labels = connected_components(adj, directed=False)
    else:                      # every detection alone: nothing links anything
        n_groups, labels = n, np.arange(n)

    out = Outlines(input_points=n, capped=capped)
    order = np.argsort(labels, kind="stable")
    sizes = np.bincount(labels, minlength=n_groups)
    start = 0
    floor = max(1, int(min_points))
    for size in sizes:
        idx = order[start:start + size]
        start += size
        if size < floor:
            out.noise_points += int(size)
            out.noise_groups += 1
            continue
        pts = centers[idx]
        p = probs[idx]
        ring = _outline(pts[:, :2], pad_m)
        out.rocks.append(Rock(
            polygon=ring,
            points=int(size),
            prob_mean=float(p.mean()),
            prob_max=float(p.max()),
            center=pts.mean(axis=0),
            z_min=float(pts[:, 2].min()),
            z_max=float(pts[:, 2].max()),
            area_m2=float(polygon_area(ring)),
        ))
    # Biggest first: the list is read top-down, and a table or a legend that
    # truncates should keep the rocks that matter.
    out.rocks.sort(key=lambda r: r.area_m2, reverse=True)
    return out


def _outline(xy: np.ndarray, pad_m: float) -> np.ndarray:
    """Convex outline of a clump's footprint, inflated by ``pad_m``.

    Inflating first also removes every degenerate case for free: one detection,
    two, or a row of them along a wall all become a hull of little octagons
    rather than a shape with no area that no renderer can draw.
    """
    from scipy.spatial import ConvexHull

    pad = max(float(pad_m), MIN_PAD_M)
    blown = (xy[:, None, :] + _RING[None, :, :] * pad).reshape(-1, 2)
    try:
        hull = ConvexHull(blown)
        return blown[hull.vertices]
    except Exception:
        # Qhull can still refuse a pathological input; a box around the clump
        # is a worse outline than a hull but an honest one.
        lo = xy.min(axis=0) - pad
        hi = xy.max(axis=0) + pad
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                         [hi[0], hi[1]], [lo[0], hi[1]]])


def polygon_area(ring: np.ndarray) -> float:
    """Area of a closed ring given as open (x, y) vertices — the shoelace sum."""
    if len(ring) < 3:
        return 0.0
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def outline_wireframe(rocks: list[Rock], lift: float = 0.02
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Every outline as one 3D line soup: ``(points (N, 3), lines (M, 2))``.

    Each rock becomes a low prism — its footprint drawn at the bottom and the
    top of the detections' own height band, with verticals joining the two — so
    the outline stays readable from a low camera angle instead of disappearing
    into the ground the way a single flat ring does.
    """
    pts: list[np.ndarray] = []
    lines: list[tuple[int, int]] = []
    base = 0
    for rock in rocks:
        ring = rock.polygon
        m = len(ring)
        if m < 3:
            continue
        z_lo = rock.z_min - lift
        z_hi = rock.z_max + lift
        pts.append(np.column_stack([ring[:, 0], ring[:, 1], np.full(m, z_lo)]))
        pts.append(np.column_stack([ring[:, 0], ring[:, 1], np.full(m, z_hi)]))
        for i in range(m):
            j = (i + 1) % m
            lines.append((base + i, base + j))                  # bottom ring
            lines.append((base + m + i, base + m + j))          # top ring
            lines.append((base + i, base + m + i))              # riser
        base += 2 * m
    if not pts:
        return np.empty((0, 3)), np.empty((0, 2), np.int32)
    return np.vstack(pts), np.array(lines, np.int32)


__all__ = ["DEFAULT_LINK_M", "DEFAULT_MIN_POINTS", "MAX_INPUT_POINTS",
           "MIN_PAD_M", "Outlines", "Rock", "find_rocks", "outline_wireframe",
           "polygon_area"]
