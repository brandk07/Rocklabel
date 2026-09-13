"""Retract persistent predictions the sensor has since looked straight through.

**The problem this exists for.** :class:`~rocklabel.live.scoring.LiveScorer`
stores one probability per 3D candidate voxel, latest write wins. A voxel is
only ever rewritten by scoring that same voxel again, so a positive sitting in
mid-air over the arena - a phantom return the classifier called a rock - stays
there for as long as the app runs. Seeing the ground *underneath* it later does
not touch it: that is a different voxel. Measured on the competition recording,
2,077 of 2,162 false ground cells had not been refreshed in over a minute.

Age is not evidence, though, and that is the trap. A cell nothing has looked at
for ten minutes is not thereby wrong, and a rock the robot has driven away from
looks exactly like one. What makes a prediction *wrong* is a later measurement
that could not have happened if something were there: a beam that passed
through the spot and came back from further away. That is the free-space half
of an occupancy grid, and it is what this module accumulates.

**Three separate things, kept separate.**

*Occupancy evidence* is geometric and has nothing to do with the model. Each
observation window either puts a return in a voxel (evidence for) or sends a
beam through it to something beyond (evidence against), and the two accumulate
the way they do in an occupancy grid - OctoMap (Hornung et al., 2013) is the
reference - with the units here chosen to be windows rather than log-odds so
the settings can be read. Space no beam has crossed accumulates nothing and
stays unknown: never cleared, never trusted.

*Surface support* is a local ground model, fitted per 10 cm cell from the
lowest repeatedly-observed returns around it, carrying a slope and a residual.
It is deliberately not a single minimum and not a raw point count: dense
accumulated fog is perfectly capable of being both, and a model fitted
indiscriminately to the fused cloud would happily declare the fog to be the
floor.

*Suspicion* is the gate between them. A voxel standing well clear of the
supported surface beneath it is suspect - but only suspect. A rock's top is
also above its neighbours, and a real rock's base is often occluded, so being
detached is never on its own a reason to delete anything. It only earns a voxel
the right to be retracted once beams have actually contradicted it.

**What it deliberately does not do.** It does not decay with time, it does not
hold a detection back until it has been confirmed N times (an earlier
experiment did, and it cost more than half the coverage on the weakest rocks),
and it never touches the current pass's output. It answers one question - has
this particular persistent positive been looked through - and abstains
everywhere else.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

#: Nearest beams tested per query voxel. A voxel is contradicted if ANY beam
#: inside its angular footprint came back from beyond it, so testing only the
#: closest few can under-count evidence but can never invent it: the error is
#: always "failed to clear", never "cleared something real".
MAX_BEAMS_PER_QUERY = 32

#: Closer than this to the sensor a voxel's angular footprint is most of the
#: sky, and "a beam went past it" stops meaning anything. Nothing this close is
#: judged.
MIN_QUERY_RANGE_M = 0.30


@dataclass
class EvidenceSettings:
    """What it takes to call a persistent prediction contradicted.

    These are diagnostic starting points swept in the map-evidence experiment,
    not validated deployment numbers, and none of them is a rock-size cutoff.
    """

    #: Off unless asked for. The current pass's output is unaffected either
    #: way - this only ever removes entries from the persistent map.
    enabled: bool = False

    #: Ground-cell size for the local surface model.
    cell_m: float = 0.10
    #: Distinct observation windows a voxel needs before its returns are
    #: allowed to define the ground. One window is one sweep, so this is what
    #: stops a single bad sweep becoming the floor.
    support_windows: int = 2
    #: Half-width, in cells, of the neighbourhood the ground plane is fitted
    #: over. 3 is a 70 cm square: wide enough to carry a slope, wider than any
    #: rock here, so a rock cannot define the ground it sits on.
    surface_radius_cells: int = 3
    #: Cells that must contribute before a fitted plane is trusted. Fewer than
    #: this and the surface under that spot stays unknown, which means nothing
    #: above it is ever called detached.
    surface_min_cells: int = 8
    #: Largest RMS residual (m) a fitted plane may have and still count as a
    #: surface. Above it the neighbourhood is too rough to say what is detached
    #: from what.
    surface_max_residual_m: float = 0.08

    #: How far above the fitted surface a voxel must stand to be suspect, on
    #: top of that plane's own residual. Starts at the conservative end: a rock
    #: here is 0.10-0.15 m tall, and clearing rock tops is the failure this
    #: guards against.
    min_separation_m: float = 0.20

    #: Windows of "a beam went through here and came back from further away"
    #: needed to retract a voxel, counted from no evidence either way. A voxel
    #: in the prediction map always has returns in it, so the real cost is this
    #: plus whatever those returns already banked - see ``hit_windows`` and
    #: ``max_windows_either_way``.
    free_windows: float = 3.0
    #: What one window that saw a return in the voxel banks in its favour.
    #: Above 1 a fresh return restores occupancy faster than it was lost, which
    #: is the asymmetry a mapping layer wants: appearing is cheap, disappearing
    #: is expensive.
    hit_windows: float = 2.0
    #: Ceiling on accumulated evidence in either direction, so a long stare can
    #: neither make a voxel unretractable nor an absence unrecoverable.
    max_windows_either_way: float = 4.0

    #: Slack added to a voxel's angular footprint for SLAM pose error. The
    #: sensor's own movement within one merged window is measured and added on
    #: top of this rather than assumed.
    pose_sigma_m: float = 0.03
    #: How much further than the voxel a beam must reach before it counts as
    #: having passed through rather than landed on it.
    endpoint_margin_m: float = 0.12

    #: Retract only voxels detached from a known supported surface. Off makes
    #: free-space evidence sufficient on its own, which clears more and is the
    #: arm that says what the gate is worth.
    require_detached: bool = True


class _Surface:
    """Per-cell least-squares ground plane, with a residual and a validity flag."""

    __slots__ = ("cell", "i0", "j0", "a", "b", "c", "residual", "valid")

    def __init__(self, cell, i0, j0, a, b, c, residual, valid):
        self.cell, self.i0, self.j0 = cell, i0, j0
        self.a, self.b, self.c = a, b, c
        self.residual, self.valid = residual, valid

    @classmethod
    def empty(cls, cell: float) -> "_Surface":
        z = np.zeros((0, 0))
        return cls(cell, 0, 0, z, z, z, z, np.zeros((0, 0), bool))

    def at(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(ground_z, residual, known)`` under each xy position."""
        n = len(xy)
        if not self.valid.size:
            return np.zeros(n), np.zeros(n), np.zeros(n, bool)
        i = np.floor(xy[:, 0] / self.cell).astype(np.int64) - self.i0
        j = np.floor(xy[:, 1] / self.cell).astype(np.int64) - self.j0
        inside = ((i >= 0) & (i < self.valid.shape[0])
                  & (j >= 0) & (j < self.valid.shape[1]))
        ground = np.zeros(n)
        residual = np.zeros(n)
        known = np.zeros(n, bool)
        if not inside.any():
            return ground, residual, known
        ii, jj = i[inside], j[inside]
        known[inside] = self.valid[ii, jj]
        # Evaluate the plane where the point actually is, not at the cell
        # centre: over a 70 cm fit a real slope moves the answer by centimetres.
        cx = (ii + self.i0 + 0.5) * self.cell
        cy = (jj + self.j0 + 0.5) * self.cell
        ground[inside] = (self.c[ii, jj]
                          + self.a[ii, jj] * (xy[inside, 0] - cx)
                          + self.b[ii, jj] * (xy[inside, 1] - cy))
        residual[inside] = self.residual[ii, jj]
        return ground, residual, known


def fit_surface(low_z: dict[tuple[int, int], float], s: EvidenceSettings) -> _Surface:
    """Fit one ground plane per cell from the lowest well-observed returns.

    Every quantity the fit needs - the counts and the first and second moments
    of the neighbourhood - is a shift-invariant weighted sum over a fixed
    window, so the whole grid is fitted in a handful of correlations rather
    than a loop over cells.

    The centre cell is left out of its own fit. A voxel is going to be compared
    against this surface, and a surface that had already been pulled up by that
    same voxel's own returns would quietly hide exactly the thing being looked
    for.
    """
    from scipy import ndimage

    if not low_z:
        return _Surface.empty(s.cell_m)
    keys = np.array(list(low_z), dtype=np.int64)
    vals = np.fromiter(low_z.values(), dtype=np.float64, count=len(low_z))
    i0, j0 = keys[:, 0].min(), keys[:, 1].min()
    shape = (int(keys[:, 0].max() - i0) + 1, int(keys[:, 1].max() - j0) + 1)
    m = np.zeros(shape)
    z = np.zeros(shape)
    m[keys[:, 0] - i0, keys[:, 1] - j0] = 1.0
    z[keys[:, 0] - i0, keys[:, 1] - j0] = vals

    r = int(s.surface_radius_cells)
    du = np.arange(-r, r + 1)[:, None] * s.cell_m
    dv = np.arange(-r, r + 1)[None, :] * s.cell_m
    du, dv = np.broadcast_arrays(du, dv)
    ones = np.ones_like(du)

    def corr(field, kernel):
        return ndimage.correlate(field, kernel, mode="constant", cval=0.0)

    mz = m * z
    n = corr(m, ones)
    su, sv = corr(m, du), corr(m, dv)
    suu, suv, svv = corr(m, du * du), corr(m, du * dv), corr(m, dv * dv)
    sz, suz, svz = corr(mz, ones), corr(mz, du), corr(mz, dv)
    szz = corr(m * z * z, ones)
    # Leave the centre cell out. At the centre du = dv = 0, so it contributes
    # to the count and to the plain z sums and to nothing else.
    n = n - m
    sz = sz - mz
    szz = szz - m * z * z

    # Solve the 3x3 normal equations everywhere at once.
    a_mat = np.stack([np.stack([suu, suv, su], -1),
                      np.stack([suv, svv, sv], -1),
                      np.stack([su, sv, n], -1)], -2)
    rhs = np.stack([suz, svz, sz], -1)
    det = np.linalg.det(a_mat)
    ok = (n >= s.surface_min_cells) & (np.abs(det) > 1e-12)
    coef = np.zeros(a_mat.shape[:-1])
    if ok.any():
        # numpy 2 reads a trailing (K, 3) right-hand side as a stack of
        # matrices, not of vectors, so say which is meant.
        coef[ok] = np.linalg.solve(a_mat[ok], rhs[ok][..., None])[..., 0]
    a, b, c = coef[..., 0], coef[..., 1], coef[..., 2]
    # Residual sum of squares of a least-squares plane fit, without
    # reconstructing the fitted values: sum(z^2) - coefficients . moments.
    rss = np.maximum(szz - (a * suz + b * svz + c * sz), 0.0)
    residual = np.sqrt(rss / np.maximum(n, 1))
    valid = ok & (residual <= s.surface_max_residual_m)
    return _Surface(s.cell_m, int(i0), int(j0), a, b, c, residual, valid)


def _box_clearance(origin: np.ndarray, ends: np.ndarray, lo: np.ndarray,
                   hi: np.ndarray) -> np.ndarray:
    """Signed clearance between each beam segment and its own voxel box.

    Negative means the beam goes *into* the box, and the magnitude is how deep:
    the distance from the deepest point on the beam to the nearest face. Zero
    or positive means it stays outside. Both are the same expression - the
    largest per-axis overshoot of the box - which is a maximum of linear
    functions of the position along the beam, hence convex, hence has a single
    minimum a ternary search finds in a fixed number of vectorised steps.
    Forty halves the bracket to about a ten-millionth, far below the
    centimetres being judged.
    """
    d = ends - origin

    def clearance(t):
        p = origin + t[:, None] * d
        return np.maximum(lo - p, p - hi).max(axis=1)

    t0 = np.zeros(len(ends))
    t1 = np.ones(len(ends))
    for _ in range(40):
        span = (t1 - t0) / 3.0
        m0, m1 = t0 + span, t1 - span
        closer = clearance(m0) < clearance(m1)
        t1 = np.where(closer, m1, t1)
        t0 = np.where(closer, t0, m0)
    return clearance(t0)


def passed_through(positions: np.ndarray, origins: np.ndarray, xyz: np.ndarray,
                   voxel: float, s: EvidenceSettings) -> np.ndarray:
    """``[N] float`` in [0, 1]: how far this window's beams contradict each voxel.

    Worked in two stages. The **broad** stage is angular and cheap: beams are
    put in a KD-tree on their unit directions, and a voxel's candidates are the
    beams whose direction falls inside its angular footprint and that came back
    from beyond it. A beam that stops short, a beam that ends inside the voxel,
    and a direction nothing was measured along are all *not* evidence of free
    space, and all three fall out of this without a special case.

    An angular footprint is a cone, though, and a cone is not the voxel. At a
    metre's range the default footprint is 7 cm wide, so a beam passing 3.5 cm
    to the side of a 5 cm voxel sits inside it while missing the voxel
    entirely. That is why there is a **narrow** stage: each candidate is
    measured against the actual box, and a beam that misses it contributes
    nothing.

    Pose error is spent on the *weight*, not on the geometry, and only ever
    downward. A beam that misses the box contributes nothing at any pose error.
    A beam that enters it contributes in proportion to how deep it goes against
    the pose slack - all the way in relative to ``pose_sigma_m`` is a full
    window of free space, barely clipping a face is close to nothing - so a
    beam whose crossing would survive being wrong about the pose counts, and
    one that would not, mostly does not.

    That is the direction this has to push. The obvious reading of "allow for
    pose error" is to widen the acceptance region by it, and that makes a
    larger assumed error *invent* stronger evidence: uncertainty starts
    deleting things. Here more assumed error can only ever lower a weight, and
    the honest consequence is visible rather than hidden - at 3 cm of slack in
    a 5 cm voxel even a dead-centre crossing is worth 0.83 of a window, so
    clearing simply takes longer.

    A voxel's weight is the best single beam's, since one beam through is one
    look through - though one thin ray crossing part of a voxel is an occupancy
    observation, not proof that its whole volume is empty.

    Beams are grouped by the origin they were actually measured from, and each
    group traced from its own. Merging a window into one viewpoint would be
    tempting - the origins inside one are centimetres apart - but on this
    recording the sensor covers up to 6.9 cm during a window, which is larger
    than the 5 cm voxel being judged.
    """
    from scipy.spatial import cKDTree

    out = np.zeros(len(positions))
    if len(xyz) == 0 or len(positions) == 0:
        return out
    origins = np.atleast_2d(origins)
    if len(origins) == 1:
        origins = np.broadcast_to(origins, xyz.shape)
    viewpoints, group = np.unique(origins, axis=0, return_inverse=True)
    # The box each position is being judged inside, from its own voxel key.
    lo_all = np.floor(positions / voxel) * voxel

    pending = np.arange(len(positions))
    for v in range(len(viewpoints)):
        if not len(pending):
            break
        origin = viewpoints[v]
        ends = xyz[group == v]
        d = ends - origin
        beam_range = np.linalg.norm(d, axis=1)
        real = beam_range > 1e-3
        if not real.any():
            continue
        tree = cKDTree(d[real] / beam_range[real, None])
        beam_range, ends = beam_range[real], ends[real]

        q = positions[pending] - origin
        query_range = np.linalg.norm(q, axis=1)
        judged = query_range >= MIN_QUERY_RANGE_M
        if not judged.any():
            continue
        qd = q[judged] / query_range[judged, None]
        qr = query_range[judged]

        # Broad stage. Half-diagonal of the voxel plus the pose slack, as an
        # angle at that range - deliberately generous, because anything it
        # misses here the narrow stage never gets to see. Chord and angle agree
        # to well under a percent at these sizes.
        reach = voxel * np.sqrt(3) / 2 + s.pose_sigma_m
        footprint = np.minimum(reach / qr, 0.5)

        k = int(min(MAX_BEAMS_PER_QUERY, len(beam_range)))
        dist, idx = tree.query(qd, k=k, workers=-1)
        dist = dist.reshape(len(qd), -1)
        idx = idx.reshape(len(qd), -1).clip(0, len(beam_range) - 1)
        candidate = (dist <= footprint[:, None]) & (
            beam_range[idx] > (qr[:, None] + s.endpoint_margin_m))
        if not candidate.any():
            continue

        # Narrow stage, on the candidate (voxel, beam) pairs only.
        rows, cols = np.nonzero(candidate)
        lo = lo_all[pending[judged][rows]]
        clearance = _box_clearance(origin, ends[idx[rows, cols]], lo, lo + voxel)
        if s.pose_sigma_m > 0:
            weight = np.clip(-clearance / s.pose_sigma_m, 0.0, 1.0)
        else:
            weight = (clearance < 0.0).astype(float)
        best = np.zeros(len(qd))
        np.maximum.at(best, rows, weight)

        here = pending[judged]
        out[here] = np.maximum(out[here], best)
        # Only a full window of evidence settles a voxel; a partial one can
        # still be improved on by a later viewpoint in the same window.
        settled = here[best >= 1.0]
        if len(settled):
            pending = np.setdiff1d(pending, settled, assume_unique=True)
    return out


class EvidenceMap:
    """Geometric occupancy evidence kept beside a prediction map.

    Per processed frame, in this order::

        ev.observe(origins, xyz)        # one observation window
        drop = ev.retract(positions)    # which persistent voxels to drop now

    ``observe`` folds the window in; ``retract`` judges only the positions it
    is handed, so a caller that passes just its above-threshold detections
    spends nothing on the rest of the map.
    """

    def __init__(self, voxel_m: float, settings: EvidenceSettings | None = None):
        self.voxel = float(voxel_m)
        self.s = settings or EvidenceSettings()
        #: Identity of the window currently folded in, so the same one arriving
        #: twice is not two measurements. See :meth:`observe`.
        self._stamp: object = None
        #: Voxels this window has already spent contradiction on.
        self._spent: set[tuple[int, int, int]] = set()
        #: Acquisition time of the freshest return folded in, when the caller
        #: times its returns. Everything at or before it has been counted.
        self._newest = -np.inf
        #: Windows handed in that were a repeat of the one before.
        self.repeats = 0
        #: voxel key -> distinct windows that put a return in it.
        self._hits: dict[tuple[int, int, int], int] = {}
        #: voxel key -> occupancy evidence in windows (positive = seen,
        #: negative = looked through).
        self._evidence: dict[tuple[int, int, int], float] = {}
        #: ground cell -> lowest z of a voxel with enough distinct windows.
        self._low_z: dict[tuple[int, int], float] = {}
        self._surface: _Surface | None = None
        self._surface_dirty = True
        self._frame: tuple[np.ndarray, np.ndarray] | None = None
        self._hit_now: set[tuple[int, int, int]] = set()
        self.windows = 0
        #: Largest distance between the sensor origins merged into one window.
        #: Measured, not assumed: it is the whole error in treating a window as
        #: a single viewpoint.
        self.max_origin_spread_m = 0.0
        self.retracted = 0

    # -- accumulating evidence --------------------------------------------- #
    def observe(self, origins: np.ndarray, xyz: np.ndarray,
                stamps: np.ndarray | object = None) -> None:
        """Fold one observation window's returns in.

        ``origins`` is the sensor position each return was measured from - not
        the robot base, and not one pose for a fused cloud. Pass one row per
        point, or a single origin to broadcast.

        ``stamps`` says *when each return was acquired* - one time per point,
        or a single identifier for the whole window. It matters because the
        callers do not hand over scans, they hand over whatever is in a buffer,
        and the same return arrives repeatedly in three ordinary situations: a
        paused or stalled feed re-delivers an identical buffer, a scoring
        window longer than the interval between passes re-delivers most of its
        scans, and a pass that is retried re-delivers all of them. Counting
        those again is how a voxel banks three windows of evidence from one
        look. Only returns newer than the freshest already folded in are
        counted, and only those are traced for free space.

        A window with nothing new in it leaves everything as it was, so
        :meth:`retract` can still act on evidence already held but cannot spend
        any more of it. With no stamps at all the window's contents are
        fingerprinted, which still recognises an identical buffer.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        origins = np.asarray(origins, dtype=np.float64)
        if origins.ndim == 1:
            origins = np.broadcast_to(origins, xyz.shape)

        per_point = (stamps is not None and np.ndim(stamps) == 1
                     and len(stamps) == len(xyz) and len(xyz) > 0)
        if per_point:
            times = np.asarray(stamps, dtype=np.float64)
            keep = times > self._newest
            if not keep.any():
                self.repeats += 1
                return
            self._newest = float(times.max())
            origins, xyz = origins[keep], xyz[keep]
            stamp = self._newest
        else:
            stamp = stamps
            if stamp is None:
                stamp = (
                    hashlib.blake2b(np.ascontiguousarray(xyz), digest_size=16).digest(),
                    hashlib.blake2b(np.ascontiguousarray(origins), digest_size=16).digest())
            if len(xyz) and stamp == self._stamp:
                self.repeats += 1
                self._frame = (origins, xyz)
                return
        self._stamp = stamp
        self._spent = set()
        self._frame = (origins, xyz)
        self._hit_now = set()
        if len(xyz) == 0:
            return
        self.windows += 1
        self.max_origin_spread_m = max(
            self.max_origin_spread_m,
            float(np.linalg.norm(origins - origins[0], axis=1).max()))

        # One vote per voxel per window: a sweep that happens to drop forty
        # returns in one voxel is still one look at it.
        keys = np.unique(np.floor(xyz / self.voxel).astype(np.int64), axis=0)
        cap, gain = self.s.max_windows_either_way, self.s.hit_windows
        ratio = self.voxel / self.s.cell_m
        for key in map(tuple, keys):
            self._hit_now.add(key)
            n = self._hits.get(key, 0) + 1
            self._hits[key] = n
            self._evidence[key] = min(self._evidence.get(key, 0.0) + gain, cap)
            if n == self.s.support_windows:
                # Only now do this voxel's returns get to define the ground.
                z = (key[2] + 0.5) * self.voxel
                ckey = (int(np.floor(key[0] * ratio)), int(np.floor(key[1] * ratio)))
                if z < self._low_z.get(ckey, np.inf):
                    self._low_z[ckey] = z
                    self._surface_dirty = True

    # -- the local ground model -------------------------------------------- #
    def surface(self) -> _Surface:
        """The fitted ground, refitted only when a supporting return moved it."""
        if self._surface is None or self._surface_dirty:
            self._surface = fit_surface(self._low_z, self.s)
            self._surface_dirty = False
        return self._surface

    def detached(self, positions: np.ndarray) -> np.ndarray:
        """``[N] bool``: standing clear of an established surface beneath it.

        False wherever no surface has been established, which is the whole
        point: unobserved space stays unknown and nothing above it is suspect.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if len(positions) == 0:
            return np.zeros(0, bool)
        ground, residual, known = self.surface().at(positions[:, :2])
        clearance = positions[:, 2] - ground
        return known & (clearance >= self.s.min_separation_m + residual)

    # -- the decision ------------------------------------------------------- #
    def retract(self, positions: np.ndarray) -> np.ndarray:
        """``[N] bool``: which of these persistent predictions to drop now.

        A position is dropped once beams have passed through its voxel enough
        times to outweigh the returns that put it there and to carry it
        ``free_windows`` past neutral - and, unless the detachment gate is
        switched off, only while it also stands clear of a surface the map has
        actually established underneath it.

        A voxel that took a return in this same window is left alone entirely -
        neither contradicted nor dropped. Whatever else went past it, something
        is there now, and a voxel can hold the maximum negative evidence and
        still be somewhere a rock has just appeared.

        Evidence is spent at most once per voxel per observation window, so
        calling this twice inside one window - which the live scorer does
        whenever a pass is retried - is idempotent rather than a second
        measurement.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if len(positions) == 0 or self._frame is None:
            return np.zeros(len(positions), bool)
        origins, xyz = self._frame

        suspect = (self.detached(positions) if self.s.require_detached
                   else np.ones(len(positions), bool))
        keys = np.floor(positions / self.voxel).astype(np.int64)
        tuples = [tuple(k) for k in keys]
        fresh = np.array([t in self._hit_now for t in tuples], bool)

        spent = np.array([t in self._spent for t in tuples], bool)
        through = np.zeros(len(positions))
        ask = suspect & ~fresh & ~spent
        if ask.any():
            through[ask] = passed_through(positions[ask], origins, xyz,
                                          self.voxel, self.s)

        cap = self.s.max_windows_either_way
        for i in np.nonzero(through > 0.0)[0]:
            key = tuples[i]
            self._evidence[key] = max(self._evidence.get(key, 0.0) - through[i], -cap)
            self._spent.add(key)

        drop = suspect & ~fresh & np.array(
            [self._evidence.get(t, 0.0) <= -self.s.free_windows for t in tuples], bool)
        self.retracted += int(drop.sum())
        return drop

    # -- reporting ---------------------------------------------------------- #
    def stats(self) -> dict:
        surf = self.surface()
        return {
            "windows": self.windows,
            "voxels_observed": len(self._hits),
            "voxels_supporting_ground": sum(
                1 for n in self._hits.values() if n >= self.s.support_windows),
            "ground_cells": len(self._low_z),
            "ground_cells_with_plane": int(surf.valid.sum()),
            "retracted": self.retracted,
            "repeated_windows": self.repeats,
            "max_origin_spread_m": round(self.max_origin_spread_m, 4),
        }
