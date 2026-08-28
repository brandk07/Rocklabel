"""Group above-threshold predictions into rock outlines.

The model answers one point at a time: "is this a rock". That is enough to
paint the cloud, but not enough to say *how many* rocks are in front of the
sensor, or where one ends and the next begins — and a single stray center over
the threshold looks exactly like a real detection when all you draw is dots.

A rock is a *clump*. This module turns the detection cloud into objects.  The
default ``robust`` path:

1. finds DBSCAN-style *core* detections with enough local neighbours, then lets
   only one layer of fringe detections attach to them.  Unlike single-link
   clustering, a sparse chain cannot walk an outline across the room;
2. rejects clumps that are too small, too large, too tall, or weak on average;
3. wraps each survivor in a concave Delaunay contour, inflated by ``pad_m`` so
   it covers the ground the detections represent without filling every empty
   corner of a convex hull.

The old single-link + convex-hull implementation remains available as
``grouping="legacy"``.  That is deliberately a whole-pipeline fallback: all
new shape and object-prior settings are ignored in legacy mode, making an A/B
comparison honest and making the change reversible from the live panel.

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

#: Defaults shared by the scorer settings, both viewers, and the web panel.
#: Candidate centers normally sit on a 5 cm grid.  Fifteen centimetres reaches
#: the next few cells without the six-cell jumps the old 0.30 m default allowed.
DEFAULT_GROUPING = "robust"
DEFAULT_LINK_M = 0.15
DEFAULT_CORE_POINTS = 4
DEFAULT_MIN_POINTS = 6
#: Object priors.  Zero disables each filter; the defaults are intentionally
#: broad enough for competition rocks while removing furniture-scale clumps.
DEFAULT_MAX_DIAMETER_M = 0.80
DEFAULT_MAX_HEIGHT_M = 0.50
DEFAULT_MIN_MEAN_PROB = 0.0
#: Maximum Delaunay edge retained by the tight outline.  Longer triangles are
#: exactly the ones that bridge empty bays and turn a contour into a convex hull.
DEFAULT_CONTOUR_M = 0.20

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
    #: Why robust mode rejected detections.  These still roll up into the two
    #: fields above, preserving the wire/browser payload and its old contract.
    sparse_points: int = 0
    small_points: int = 0
    oversize_points: int = 0
    overheight_points: int = 0
    weak_points: int = 0
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
        reasons = []
        for name, count in (("sparse", self.sparse_points),
                            ("small", self.small_points),
                            ("too wide", self.oversize_points),
                            ("too tall", self.overheight_points),
                            ("weak", self.weak_points)):
            if count:
                reasons.append(f"{name} {count:,}")
        why = f" · {', '.join(reasons)}" if reasons else ""
        return (f"{self.noise_points:,} of {self.input_points:,} detections in "
                f"{self.noise_groups} clump{'' if self.noise_groups == 1 else 's'}"
                f"{why}{cap}")

    def reject(self, size: int, reason: str) -> None:
        """Account for one rejected clump without duplicating readout logic."""
        size = int(size)
        self.noise_points += size
        self.noise_groups += 1
        attr = {
            "sparse": "sparse_points",
            "small": "small_points",
            "oversize": "oversize_points",
            "overheight": "overheight_points",
            "weak": "weak_points",
        }.get(reason)
        if attr is not None:
            setattr(self, attr, int(getattr(self, attr)) + size)


def find_rocks(
    centers: np.ndarray,
    probs: np.ndarray,
    link_m: float = DEFAULT_LINK_M,
    min_points: int = DEFAULT_MIN_POINTS,
    pad_m: float = 0.0,
    max_points: int = MAX_INPUT_POINTS,
    grouping: str = DEFAULT_GROUPING,
    core_points: int = DEFAULT_CORE_POINTS,
    max_diameter_m: float = DEFAULT_MAX_DIAMETER_M,
    max_height_m: float = DEFAULT_MAX_HEIGHT_M,
    min_mean_prob: float = DEFAULT_MIN_MEAN_PROB,
    contour_m: float = DEFAULT_CONTOUR_M,
) -> Outlines:
    """Cluster detections and outline the clumps big enough to be rocks.

    ``centers``/``probs`` are what :meth:`LiveScorer.detections` returns — the
    above-threshold part of the prediction map. Linking happens in 3D so a rock
    is not merged with the ceiling above it; the outline is the 2D footprint,
    which is the shape you steer a robot around.
    """
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
    out = Outlines(input_points=n, capped=capped)
    floor = max(1, int(min_points))
    robust = str(grouping).lower() != "legacy"
    if robust:
        groups, sparse = _density_groups(
            centers, link, max(1, min(int(core_points), floor)))
        for idx in sparse:
            out.reject(len(idx), "sparse")
    else:
        groups = _linked_groups(centers, link)

    for idx in groups:
        size = len(idx)
        if size < floor:
            out.reject(size, "small")
            continue
        pts = centers[idx]
        p = probs[idx]
        if robust and float(max_diameter_m) > 0.0:
            diameter = float(np.linalg.norm(np.ptp(pts[:, :2], axis=0)))
            if diameter > float(max_diameter_m):
                out.reject(size, "oversize")
                continue
        if robust and float(max_height_m) > 0.0:
            if float(np.ptp(pts[:, 2])) > float(max_height_m):
                out.reject(size, "overheight")
                continue
        if robust and float(min_mean_prob) > 0.0:
            if float(p.mean()) < float(min_mean_prob):
                out.reject(size, "weak")
                continue
        ring = (_tight_outline(pts[:, :2], pad_m, contour_m)
                if robust else _outline(pts[:, :2], pad_m))
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


def outline_options(settings, center_spacing_m: float = 0.0) -> dict:
    """Translate shared live settings into one :func:`find_rocks` call.

    Both renderers use this adapter so adding a knob cannot accidentally make
    the Open3D prism and browser footprint disagree.
    """
    grouping = str(getattr(settings, "cluster_grouping", DEFAULT_GROUPING))
    padding = float(getattr(settings, "cluster_padding_m", 0.0))
    # Legacy means legacy as a whole: its old automatic half-cell skin is not
    # quietly affected by a robust contour knob left at another value.
    if grouping == "legacy" or padding <= 0.0:
        padding = 0.5 * float(center_spacing_m)
    return {
        "grouping": grouping,
        "link_m": float(getattr(settings, "cluster_link_m", DEFAULT_LINK_M)),
        "core_points": int(getattr(settings, "cluster_core_points",
                                   DEFAULT_CORE_POINTS)),
        "min_points": int(getattr(settings, "cluster_min_points",
                                  DEFAULT_MIN_POINTS)),
        "max_diameter_m": float(getattr(settings, "cluster_max_diameter_m",
                                        DEFAULT_MAX_DIAMETER_M)),
        "max_height_m": float(getattr(settings, "cluster_max_height_m",
                                      DEFAULT_MAX_HEIGHT_M)),
        "min_mean_prob": float(getattr(settings, "cluster_min_mean_prob",
                                       DEFAULT_MIN_MEAN_PROB)),
        "contour_m": float(getattr(settings, "cluster_contour_m",
                                   DEFAULT_CONTOUR_M)),
        "pad_m": padding,
    }


def outline_settings_key(settings) -> tuple:
    """Hashable, precision-stable part of the viewers' outline cache keys."""
    o = outline_options(settings)
    if o["grouping"] == "legacy":
        return ("legacy", round(o["link_m"], 4), int(o["min_points"]))
    return (
        o["grouping"], round(o["link_m"], 4), int(o["core_points"]),
        int(o["min_points"]), round(o["max_diameter_m"], 4),
        round(o["max_height_m"], 4), round(o["min_mean_prob"], 4),
        round(o["contour_m"], 4),
        round(float(getattr(settings, "cluster_padding_m", 0.0)), 4),
    )


def _linked_labels(points: np.ndarray, link_m: float) -> tuple[int, np.ndarray]:
    """Connected-component labels for a radius graph, including isolates."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(points)
    if not n:
        return 0, np.empty((0,), np.int32)
    pairs = cKDTree(points).query_pairs(float(link_m), output_type="ndarray")
    if not len(pairs):
        return n, np.arange(n, dtype=np.int32)
    # connected_components treats an absent diagonal as an isolated vertex,
    # and directed=False supplies the reverse half of this upper triangle.
    adj = coo_matrix(
        (np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(n, n),
    )
    return connected_components(adj, directed=False)


def _groups_from_labels(labels: np.ndarray, n_groups: int,
                        source: np.ndarray | None = None) -> list[np.ndarray]:
    """Turn component labels into stable index arrays."""
    if n_groups <= 0:
        return []
    source = np.arange(len(labels)) if source is None else np.asarray(source)
    order = np.argsort(labels, kind="stable")
    sizes = np.bincount(labels, minlength=n_groups)
    groups: list[np.ndarray] = []
    start = 0
    for size in sizes:
        groups.append(source[order[start:start + size]])
        start += int(size)
    return groups


def _linked_groups(points: np.ndarray, link_m: float,
                   source: np.ndarray | None = None) -> list[np.ndarray]:
    n_groups, labels = _linked_labels(points, link_m)
    return _groups_from_labels(labels, n_groups, source)


def _density_groups(points: np.ndarray, link_m: float, core_points: int
                    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """DBSCAN groups and sparse/no-core groups, as original point indices.

    Only core points connect components.  A non-core point can border one core
    component, but cannot relay the connection to another fringe point.  That
    one distinction removes the single-link "string of pearls" failure.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    # Ask only for counts: materializing every neighbour list can become the
    # dominant allocation at the 20k-point guard rail.
    counts = tree.query_ball_point(
        points, float(link_m), workers=-1, return_length=True)
    is_core = np.asarray(counts) >= int(core_points)
    core_idx = np.flatnonzero(is_core)
    if not len(core_idx):
        return [], _linked_groups(points, link_m)

    n_groups, core_labels = _linked_labels(points[core_idx], link_m)
    labels = np.full(len(points), -1, np.int32)
    labels[core_idx] = core_labels

    border_idx = np.flatnonzero(~is_core)
    if len(border_idx):
        core_tree = cKDTree(points[core_idx])
        dist, near = core_tree.query(
            points[border_idx], k=1, distance_upper_bound=float(link_m))
        attached = np.isfinite(dist)
        labels[border_idx[attached]] = core_labels[near[attached]]

    groups = _groups_from_labels(labels[labels >= 0], n_groups,
                                  np.flatnonzero(labels >= 0))
    noise_idx = np.flatnonzero(labels < 0)
    sparse = (_linked_groups(points[noise_idx], link_m, noise_idx)
              if len(noise_idx) else [])
    return groups, sparse


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


def _tight_outline(xy: np.ndarray, pad_m: float, contour_m: float) -> np.ndarray:
    """Concave Delaunay boundary, falling back to the legacy convex hull.

    Triangles with an edge longer than ``contour_m`` are bridges over empty
    ground, so they do not take part in the boundary.  The largest closed ring
    wins when fringe points make a separate island; those points still count in
    the rock statistics but no longer stretch its polygon.
    """
    from collections import defaultdict
    from scipy.spatial import Delaunay

    unique = np.unique(np.asarray(xy, dtype=np.float64), axis=0)
    gap = max(float(contour_m), MIN_PAD_M * 2.0)
    if len(unique) < 4:
        return _outline(unique, pad_m)
    try:
        triangles = Delaunay(unique).simplices
    except Exception:
        return _outline(unique, pad_m)

    # Boundary edges are the triangle edges that occur exactly once.  Retain
    # their orientation so each manifold ring can be followed without an
    # angle-sort that might jump across a concavity.
    boundary: dict[tuple[int, int], tuple[int, int]] = {}
    kept = 0
    for tri in triangles:
        p = unique[tri]
        edges_m = np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)
        if float(edges_m.max()) > gap:
            continue
        a, b, c = map(int, tri)
        ab = unique[b] - unique[a]
        ac = unique[c] - unique[a]
        if ab[0] * ac[1] - ab[1] * ac[0] < 0:
            b, c = c, b
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            if key in boundary:
                del boundary[key]
            else:
                boundary[key] = edge
        kept += 1
    if not kept or len(boundary) < 3:
        return _outline(unique, pad_m)

    outgoing: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary.values():
        outgoing[a].append(b)
    unused = set(boundary.values())
    rings: list[np.ndarray] = []
    while unused:
        start_edge = next(iter(unused))
        start, current = start_edge
        loop = [start]
        unused.remove(start_edge)
        for _ in range(len(boundary) + 1):
            loop.append(current)
            if current == start:
                break
            candidates = [(current, nxt) for nxt in outgoing.get(current, [])
                          if (current, nxt) in unused]
            if len(candidates) != 1:
                break
            edge = candidates[0]
            unused.remove(edge)
            current = edge[1]
        if len(loop) >= 4 and loop[-1] == start:
            ring = unique[np.asarray(loop[:-1], dtype=int)]
            if polygon_area(ring) > 0.0:
                rings.append(ring)
    if not rings:
        return _outline(unique, pad_m)
    ring = max(rings, key=polygon_area)
    return _offset_ring(ring, max(float(pad_m), MIN_PAD_M))


def _offset_ring(ring: np.ndarray, pad_m: float) -> np.ndarray:
    """Inflate a simple polygon with bounded corner mitres."""
    ring = np.asarray(ring, dtype=np.float64)
    if _signed_area(ring) < 0.0:
        ring = ring[::-1]
    prev = ring - np.roll(ring, 1, axis=0)
    nxt = np.roll(ring, -1, axis=0) - ring
    prev /= np.maximum(np.linalg.norm(prev, axis=1, keepdims=True), 1e-12)
    nxt /= np.maximum(np.linalg.norm(nxt, axis=1, keepdims=True), 1e-12)
    n_prev = np.column_stack([prev[:, 1], -prev[:, 0]])
    n_next = np.column_stack([nxt[:, 1], -nxt[:, 0]])
    bisector = n_prev + n_next
    length = np.linalg.norm(bisector, axis=1, keepdims=True)
    use = length[:, 0] > 1e-6
    bisector[use] /= length[use]
    bisector[~use] = n_next[~use]
    denom = np.einsum("ij,ij->i", bisector, n_next)
    scale = np.divide(float(pad_m), denom,
                      out=np.full(len(ring), float(pad_m)),
                      where=np.abs(denom) > 0.2)
    # Acute Delaunay corners can otherwise create metre-long spikes.
    scale = np.clip(scale, -3.0 * pad_m, 3.0 * pad_m)
    return ring + bisector * scale[:, None]


def _signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return float((np.dot(x, np.roll(y, -1)) -
                  np.dot(y, np.roll(x, -1))) / 2.0)


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


__all__ = ["DEFAULT_GROUPING", "DEFAULT_LINK_M", "DEFAULT_CORE_POINTS",
           "DEFAULT_MIN_POINTS", "DEFAULT_MAX_DIAMETER_M",
           "DEFAULT_MAX_HEIGHT_M", "DEFAULT_MIN_MEAN_PROB",
           "DEFAULT_CONTOUR_M", "MAX_INPUT_POINTS", "MIN_PAD_M", "Outlines",
           "Rock", "find_rocks", "outline_options", "outline_settings_key",
           "outline_wireframe", "polygon_area"]
