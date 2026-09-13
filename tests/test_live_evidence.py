"""Tests for the free-space layer that retracts persistent false positives.

The thing under test is a deletion rule, so most of these pin down what it must
*not* do: never clear a real rock, never clear anything it has no beams for,
never act on the strength of age or height alone. The one positive test is the
case the layer exists for - a prediction hanging in mid-air that later beams go
straight through.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocklabel.live.evidence import EvidenceMap, EvidenceSettings, fit_surface

VOXEL = 0.05


def _floor(rng, half=6.0, n=20_000, tilt=(0.0, 0.0)):
    """A flat (or tilted) ground patch of returns."""
    xy = rng.uniform(-half, half, (n, 2))
    z = tilt[0] * xy[:, 0] + tilt[1] * xy[:, 1]
    return np.column_stack([xy, z])


def _rock(rng, at=(1.0, 0.0), height=0.12, n=80):
    xy = rng.uniform(-0.1, 0.1, (n, 2)) + np.asarray(at)
    return np.column_stack([xy, rng.uniform(0.0, height, n)])


def _origins(points, w, n_sub=5, height=0.60):
    """One viewpoint per sub-scan, 2 cm apart, as a real merged window has."""
    subs = np.array([[0.02 * w + 0.02 * i, 0.0, height] for i in range(n_sub)])
    return subs[np.arange(len(points)) % n_sub]


def _sweep(ev, rng, windows, extra=None, extra_until=0, **kw):
    """Run ``windows`` observation windows over a floor plus an optional blob."""
    for w in range(windows):
        pts = np.vstack([_floor(rng, **kw), _rock(rng)])
        if extra is not None and w < extra_until:
            pts = np.vstack([pts, extra + rng.normal(0, 0.005, (8, 3))])
        ev.observe(_origins(pts, w), pts)
        yield w, pts


def test_retracts_a_floating_positive_the_beams_go_through():
    rng = np.random.default_rng(0)
    ev = EvidenceMap(VOXEL, EvidenceSettings(enabled=True))
    phantom = np.array([[-1.0, 0.5, 0.45]])
    dropped = []
    for w, _ in _sweep(ev, rng, 20, extra=phantom, extra_until=3):
        dropped.append(bool(ev.retract(phantom)[0]))
    # Not on the strength of being high up: it takes the banked returns to be
    # spent first, and only then free_windows more.
    assert not any(dropped[:5])
    assert dropped[-1]


def test_never_retracts_a_real_rocks_top():
    """The failure this whole layer is one bad threshold away from."""
    rng = np.random.default_rng(1)
    ev = EvidenceMap(VOXEL, EvidenceSettings(enabled=True))
    top = np.array([[1.0, 0.0, 0.11]])
    for w, _ in _sweep(ev, rng, 20):
        assert not ev.retract(top)[0]
    assert not ev.detached(top)[0]


def test_unobserved_space_is_never_cleared():
    """No beams anywhere near it: unknown stays unknown, whatever its height."""
    rng = np.random.default_rng(2)
    ev = EvidenceMap(VOXEL, EvidenceSettings(enabled=True))
    far = np.array([[30.0, 30.0, 1.5]])
    for w, _ in _sweep(ev, rng, 10):
        assert not ev.retract(far)[0]
    assert not ev.detached(far)[0]


def test_a_fresh_return_restores_a_voxel():
    """Appearing is cheap, disappearing is expensive - by design."""
    rng = np.random.default_rng(3)
    ev = EvidenceMap(VOXEL, EvidenceSettings(enabled=True))
    phantom = np.array([[-1.0, 0.5, 0.45]])
    for w, _ in _sweep(ev, rng, 8, extra=phantom, extra_until=1):
        ev.retract(phantom)
    key = tuple(np.floor(phantom[0] / VOXEL).astype(int))
    spent = ev._evidence[key]
    # One window that sees something there again buys back hit_windows of it.
    pts = np.vstack([_floor(rng), phantom + rng.normal(0, 0.005, (8, 3))])
    ev.observe(_origins(pts, 9), pts)
    assert ev._evidence[key] == pytest.approx(spent + ev.s.hit_windows)


def test_a_return_in_the_same_window_blocks_that_windows_contradiction():
    rng = np.random.default_rng(4)
    ev = EvidenceMap(VOXEL, EvidenceSettings(enabled=True))
    # On a voxel centre, so the jittered returns all land in the same voxel
    # the query is asking about.
    phantom = np.array([[-0.975, 0.525, 0.475]])
    key = tuple(np.floor(phantom[0] / VOXEL).astype(int))
    for w in range(6):
        pts = np.vstack([_floor(rng), phantom + rng.normal(0, 0.005, (8, 3))])
        ev.observe(_origins(pts, w), pts)
        before = ev._evidence[key]
        ev.retract(phantom)
        assert ev._evidence[key] == before  # never lost ground while being hit


def test_detachment_gate_can_be_switched_off():
    """The arm that measures what the gate costs. It must clear more, not less."""
    rng = np.random.default_rng(5)
    phantom = np.array([[-1.0, 0.5, 0.12]])   # too low to count as detached
    gated = EvidenceMap(VOXEL, EvidenceSettings(enabled=True))
    open_ = EvidenceMap(VOXEL, EvidenceSettings(enabled=True, require_detached=False))
    for ev in (gated, open_):
        r = np.random.default_rng(5)
        for w, _ in _sweep(ev, r, 22, extra=phantom, extra_until=2):
            ev.retract(phantom)
    assert not gated.retract(phantom)[0]
    assert open_.retract(phantom)[0]


def test_surface_follows_a_slope():
    """A tilted floor must not read as half the arena floating above itself."""
    rng = np.random.default_rng(6)
    ev = EvidenceMap(VOXEL, EvidenceSettings(enabled=True))
    for w in range(4):
        pts = _floor(rng, half=3.0, tilt=(0.08, 0.0))
        ev.observe(_origins(pts, w), pts)
    probe = np.column_stack([np.linspace(-2, 2, 40), np.zeros(40), np.zeros(40)])
    probe[:, 2] = 0.08 * probe[:, 0] + 0.02      # 2 cm above a 8 cm/m slope
    assert not ev.detached(probe).any()
    probe[:, 2] += 0.40                          # now half a metre above it
    assert ev.detached(probe).all()


def test_surface_leaves_the_cell_out_of_its_own_fit():
    """A cell's own returns must not be able to lift the ground under it."""
    s = EvidenceSettings(surface_min_cells=8, surface_radius_cells=3)
    low = {(i, j): 0.0 for i in range(-4, 5) for j in range(-4, 5)}
    flat = fit_surface(low, s)
    low[(0, 0)] = 1.0                     # one cell claims to be a metre up
    lifted = fit_surface(low, s)
    xy = np.array([[0.05, 0.05]])
    assert flat.at(xy)[0][0] == pytest.approx(lifted.at(xy)[0][0], abs=1e-9)


def test_surface_is_unknown_where_too_little_was_seen():
    s = EvidenceSettings(surface_min_cells=8)
    sparse = fit_surface({(0, 0): 0.0, (1, 0): 0.0, (0, 1): 0.0}, s)
    assert not sparse.at(np.array([[0.05, 0.05]]))[2][0]


def test_off_by_default():
    assert not EvidenceSettings().enabled


# --------------------------------------------------------------------------- #
# wired into the live scorer
# --------------------------------------------------------------------------- #
def test_the_scorer_clears_a_floating_detection_and_keeps_a_real_one(tmp_path):
    """End to end through LiveScorer: a real classifier, a fake engine feeding
    it a floor with a rock on it and a false positive hanging over the floor,
    and the clearing switched on. The one the beams go through goes; the one on
    the rock stays."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from rocklabel.live.scoring import LiveScorer, ScoreSettings
    from rocklabel.train.models import build_model

    generator = {"seed": 42, "frame_window_s": 0.0, "centers_voxel_m": 0.05,
                 "neighborhood_radius_m": 0.5, "min_neighbors": 20,
                 "neighborhood_points": 64, "crop_forward_m": 8.0,
                 "crop_backward_m": 8.0, "crop_left_m": 8.0, "crop_right_m": 8.0,
                 "crop_up_m": 2.0, "crop_down_m": 2.0}
    config = {"model": "pointnet", "tnet": False, "dropout": None,
              "features": ["dx", "dy", "dz"]}
    ck = tmp_path / "m.pt"
    torch.save({"config": config, "generator": generator, "threshold": 0.0,
                "model": build_model("pointnet", features=config["features"]).state_dict()},
               ck)

    rng = np.random.default_rng(7)
    # The sensor rides high enough that a beam grazing past the phantom still
    # comes down on floor inside the patch; otherwise there is no beam through
    # it and nothing to clear with.
    SENSOR_Z = 1.2
    phantom = np.array([-1.0, 0.5, 0.45])

    class _Engine:
        """Floor + rock every window; the phantom only in the first three."""

        def __init__(self):
            self.window = 0

        def _cloud(self):
            # Small enough that the candidate cap cannot randomly skip the one
            # thing the test is about, and wide enough that the beam through
            # the phantom actually lands on floor beyond it - which is the only
            # evidence that can clear anything.
            pts = [_floor(rng, half=2.4, n=5_000), _rock(rng)]
            if self.window < 3:
                pts.append(phantom + rng.normal(0, 0.005, (40, 3)))
            return np.vstack(pts)

        def recent_snapshot(self, window_s=0.0, with_origins=False,
                            with_stamps=False):
            pts = self._cloud()
            inten = np.full(len(pts), 0.5, np.float32)
            out = (pts, inten)
            if with_origins:
                out += (_origins(pts, self.window, height=SENSOR_Z),)
            if with_stamps:
                out += (np.full(len(pts), float(self.window)),)
            return out if (with_origins or with_stamps) else (pts, inten)

        def current_pose(self):
            return (np.array([0.02 * self.window, 0.0, SENSOR_Z]),
                    np.array([1., 0, 0, 0]))

    engine = _Engine()
    # threshold 0 so every scored centre is a detection: this test is about the
    # clearing rule, not about what an untrained model happens to say.
    scorer = LiveScorer(str(ck), engine, device="cpu", settings=ScoreSettings(
        z_min=-2.0, z_max=2.0, range_max=0.0, max_centers=20_000,
        clear_looked_through=True, clear_separation_m=0.20, clear_free_windows=3.0))
    scorer.threshold = 0.0

    def voxel_of(p):
        return tuple(np.floor(np.asarray(p) / VOXEL).astype(int))

    for w in range(32):
        engine.window = w
        scorer._score_once()

    keys = set(scorer._map)
    floating = [k for k in keys
                if abs(k[0] * VOXEL + 1.0) < 0.15 and abs(k[1] * VOXEL - 0.5) < 0.15
                and k[2] * VOXEL > 0.30]
    assert not floating, f"the floating detection survived: {floating}"
    assert scorer._cleared > 0
    assert scorer.status_dict()["cleared"] == scorer._cleared
    # The rock is still there: some voxel of it is still remembered.
    on_rock = [k for k in keys
               if abs(k[0] * VOXEL - 1.0) < 0.25 and abs(k[1] * VOXEL) < 0.25]
    assert on_rock, "clearing took the rock with it"


def test_the_scorer_leaves_the_map_alone_when_clearing_is_off(tmp_path):
    """The default. Nothing is retracted and no evidence is even accumulated."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from rocklabel.live.scoring import LiveScorer, ScoreSettings

    scorer = LiveScorer.__new__(LiveScorer)
    scorer.settings = ScoreSettings(clear_looked_through=False)
    scorer._evidence = scorer._evidence_key = None
    assert scorer._evidence_map() is None


# --------------------------------------------------------------------------- #
# The three failures a review reproduced against the first implementation.
# Each of these passed silently before: the layer deleted things it had no
# right to, and the twelve tests above could not see it.
# --------------------------------------------------------------------------- #
#: The counterexample geometry. The query is the centre of the voxel spanning
#: x 1.00-1.05, y 0.00-0.05, z 0.40-0.45. The beam leaves the same height and
#: passes it at y = 0.085, which is 3.5 cm outside the box - but only 0.058
#: radians off it, inside the 0.072 radian angular footprint the broad search
#: uses. Angular proximity is not a crossing.
_MISS_QUERY = np.array([[1.025, 0.025, 0.425]])
_MISS_ORIGIN = np.array([0.0, 0.025, 0.425])
_MISS_END = np.array([[2.05, 0.145, 0.425]])
_HIT_END = np.array([[2.05, 0.025, 0.425]])
#: What one dead-centre crossing of a 5 cm voxel is worth at the default 3 cm
#: of pose slack: the deepest the beam gets inside, over that slack.
_CENTRE_CROSSING = 0.025 / EvidenceSettings().pose_sigma_m


def test_a_beam_that_passes_beside_a_voxel_does_not_cross_it():
    pytest.importorskip("scipy")
    from rocklabel.live.evidence import passed_through

    s = EvidenceSettings(enabled=True)
    assert passed_through(_MISS_QUERY, _MISS_ORIGIN, _MISS_END, VOXEL, s)[0] == 0.0
    # A dead-centre crossing at the default 3 cm of pose slack: most of a
    # window, not all of it - see the uncertainty test below.
    assert passed_through(_MISS_QUERY, _MISS_ORIGIN, _HIT_END, VOXEL, s)[0] > 0.8


def test_a_beam_must_come_back_from_beyond_the_voxel():
    """Stopping short of the query, or inside it, is not free space."""
    pytest.importorskip("scipy")
    from rocklabel.live.evidence import passed_through

    s = EvidenceSettings(enabled=True)
    short = np.array([[0.60, 0.025, 0.425]])
    inside = np.array([[1.03, 0.025, 0.425]])
    assert passed_through(_MISS_QUERY, _MISS_ORIGIN, short, VOXEL, s)[0] == 0.0
    assert passed_through(_MISS_QUERY, _MISS_ORIGIN, inside, VOXEL, s)[0] == 0.0


def test_more_pose_uncertainty_never_makes_clearing_easier():
    """The direction the old cone got backwards.

    Widening the acceptance angle by the pose error let a bigger assumed error
    invent stronger evidence. Uncertainty is spent on the weight instead and
    only downward: the same measurement is worth less the less sure the pose
    is, however deep the crossing, and a beam that misses is worth nothing at
    any uncertainty at all.
    """
    pytest.importorskip("scipy")
    from rocklabel.live.evidence import passed_through

    sigmas = (0.01, 0.03, 0.10)

    def weights(end):
        return [passed_through(_MISS_QUERY, _MISS_ORIGIN, end, VOXEL,
                               EvidenceSettings(enabled=True, pose_sigma_m=s))[0]
                for s in sigmas]

    # A beam clipping the top face by half a centimetre, and one dead centre.
    graze, centre = weights(np.array([[2.05, 0.065, 0.425]])), weights(_HIT_END)
    for series in (graze, centre):
        assert series == sorted(series, reverse=True)
        assert series[0] > series[-1]
    assert all(g < c for g, c in zip(graze, centre))
    assert all(w == 0.0 for w in weights(_MISS_END))


def test_reprocessing_one_snapshot_is_not_three_measurements():
    """A paused feed, a stalled sensor and a retried pass all deliver the same
    buffer again. Counting those is how one look banks three windows."""
    pytest.importorskip("scipy")
    ev = EvidenceMap(VOXEL, EvidenceSettings(enabled=True, require_detached=False))
    for _ in range(5):
        ev.observe(_MISS_ORIGIN, _HIT_END)
        assert not ev.retract(_MISS_QUERY)[0]
    assert ev.windows == 1
    assert ev.repeats == 4


def test_overlapping_windows_do_not_vote_twice_with_shared_returns():
    """Consecutive scoring windows share every scan but the newest one."""
    pytest.importorskip("scipy")
    ev = EvidenceMap(VOXEL, EvidenceSettings(enabled=True, require_detached=False))
    beams = np.repeat(_HIT_END, 4, axis=0)
    times = np.array([1.0, 2.0, 3.0, 4.0])
    for lo in range(3):                     # windows [0:2], [1:3], [2:4]
        ev.observe(_MISS_ORIGIN, beams[lo:lo + 2], times[lo:lo + 2])
        ev.retract(_MISS_QUERY)
    key = tuple(np.floor(_MISS_QUERY[0] / VOXEL).astype(int))
    # Three windows, three scans, three windows of evidence - not six.
    assert ev.windows == 3
    assert ev._evidence[key] == pytest.approx(-3 * _CENTRE_CROSSING, rel=1e-3)


def test_retract_twice_in_one_window_spends_evidence_once():
    pytest.importorskip("scipy")
    ev = EvidenceMap(VOXEL, EvidenceSettings(enabled=True, require_detached=False))
    ev.observe(_MISS_ORIGIN, _HIT_END)
    ev.retract(_MISS_QUERY)
    ev.retract(_MISS_QUERY)
    ev.retract(_MISS_QUERY)
    key = tuple(np.floor(_MISS_QUERY[0] / VOXEL).astype(int))
    assert ev._evidence[key] == pytest.approx(-_CENTRE_CROSSING, rel=1e-3)


@pytest.mark.parametrize("free_windows", [2.0, 3.0, 4.0])
def test_a_current_return_vetoes_deletion_at_every_setting(free_windows):
    """Evidence at the floor, then something lands in the voxel. It stays.

    The old rule kept a fresh hit from *adding* contradiction but not from
    being deleted by contradiction already banked, so at free_windows=2 a
    voxel at -4 was still dropped the instant a return arrived in it.
    """
    pytest.importorskip("scipy")
    s = EvidenceSettings(enabled=True, require_detached=False,
                         free_windows=free_windows)
    ev = EvidenceMap(VOXEL, s)
    for i in range(8):
        ev.observe(_MISS_ORIGIN, _HIT_END + i * 1e-6)
        ev.retract(_MISS_QUERY)
    key = tuple(np.floor(_MISS_QUERY[0] / VOXEL).astype(int))
    assert ev._evidence[key] == pytest.approx(-s.max_windows_either_way)

    # A return in the voxel: not deleted this window, and occupancy recovers.
    ev.observe(_MISS_ORIGIN, np.vstack([_HIT_END, _MISS_QUERY]))
    assert not ev.retract(_MISS_QUERY)[0]
    assert ev._evidence[key] > -s.max_windows_either_way

    # And later independent contradictions can still take it away again.
    for i in range(10):
        ev.observe(_MISS_ORIGIN, _HIT_END + (100 + i) * 1e-6)
        if ev.retract(_MISS_QUERY)[0]:
            break
    else:
        pytest.fail("a voxel with a fresh hit can never be retracted again")
