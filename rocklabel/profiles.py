"""Named ways of building a dataset.

A dataset's score depends as much on *how the frames were cut* as on the model
trained on it, and for a while that choice lived in four look-alike YAML files
at the repo root that nobody could tell apart. Two separate agents read the
wrong one and drew the wrong conclusion from it.

So the choice is a named thing now. A profile carries the handful of generator
settings that define one way of cutting frames, plus the prose that says what
it does and when to reach for it. The name goes into the dataset directory and
into the manifest, so a dataset says on its face how it was built.

Modelled on ``rocklabel/train/ablate.py``'s ``SUITES``: a plain dict of
dataclasses, torch-free, which the dashboard reads directly to build its
dropdown.

**The overrides are load-bearing history, not preferences.** ``raw-burst`` and
``full-sweep`` reproduce the exact config fingerprints of the datasets already
on disk (a81b9c29 and 3ccba26a), which is what lets a profile be attached to an
existing dataset instead of forcing it to be regenerated. Changing a number
here silently orphans every dataset built under the old one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .config import DEFAULTS, config_hash


@dataclass
class Profile:
    """One named way of turning a recording into training frames."""

    name: str
    title: str
    #: What this profile does, in plain English. Shown in the dashboard form.
    what: str
    #: When you would reach for it, and when you would not.
    when: str
    #: Config overrides, in ``apply_overrides`` dotted form.
    overrides: dict = field(default_factory=dict)
    #: Kept so old datasets can name their provenance, but not something to
    #: start a new experiment on. Hidden behind "show older profiles".
    legacy: bool = False


#: How a frame is cut, as a set of named choices. Order is the order the
#: dashboard offers them: the one to use first, then the alternatives, then
#: the ones kept only for provenance.
PROFILES: dict[str, Profile] = {
    "full-sweep": Profile(
        name="full-sweep",
        title="Full sweep — one whole sensor rotation per frame",
        what="Merges every scan inside a 0.05 second window into one frame, "
             "which on this 20 Hz sensor is exactly one complete rotation — "
             "about 1,250 points inside the crop box. Keeps every 4th frame.",
        when="The one to use. Measured worth +0.040 to +0.055 PR-AUC over raw "
             "bursts on every model and on 10 or 11 of the 11 recordings, "
             "against a noise floor of 0.013. It is also the only profile "
             "dense enough to train the whole-frame segmenter at all.",
        overrides={
            "generator.frame_window_s": 0.05,
            "generator.frame_stride": 4,
            "generator.segmentation_points": 2048,
        },
    ),
    "double-sweep": Profile(
        name="double-sweep",
        title="Double sweep — two rotations per frame",
        what="Merges a 0.1 second window, so a frame holds roughly two sensor "
             "rotations and about 2,500 points. Frame stride is halved to 2 so "
             "the dataset ends up with a similar number of frames, and the "
             "segmenter's per-frame point budget is doubled to match.",
        when="Untested. The obvious next step after full sweep, on the theory "
             "that if one rotation beat a single burst, two might beat one. "
             "The catch is that the robot moves during a longer window, so "
             "points from the start and end of it no longer describe quite the "
             "same scene — expect a smaller gain than full sweep gave, or none.",
        overrides={
            "generator.frame_window_s": 0.1,
            "generator.frame_stride": 2,
            "generator.segmentation_points": 4096,
        },
    ),
    "raw-burst": Profile(
        name="raw-burst",
        title="Raw burst — one sensor batch per frame",
        what="No merging at all: every message the sensor emits becomes its own "
             "frame. On this rig that is a raw ~4 millisecond batch holding "
             "about 110 points inside the crop box. Keeps every 5th frame.",
        when="Only to reproduce an old result. This is what the first eleven "
             "volleyball datasets were built with, and it is a bad default: so "
             "few points land inside a half-metre ball that the sample "
             "generator's 20-neighbour floor throws most candidates away, and "
             "whole frames fall below the segmenter's 512-point floor, so it "
             "produces no segmentation data whatsoever.",
        overrides={},
        legacy=True,
    ),
    "fused": Profile(
        name="fused",
        title="Fused — full sweep with levelling switched off",
        what="Same 0.05 second merge as full sweep, but the mount-tilt "
             "correction is turned off and the segmenter keeps its full 4,096 "
             "point budget.",
        when="Only for the Comforter recordings, which were captured on a level "
             "mount and labelled before the tilt correction existed. Using it "
             "on a volleyball recording would leave the arena floor sloping.",
        overrides={
            "level.mode": "off",
            "generator.frame_window_s": 0.05,
            "generator.frame_stride": 4,
        },
        legacy=True,
    ),
}

#: What ``generate`` uses when nothing says otherwise.
DEFAULT_PROFILE = "full-sweep"


class ProfileError(Exception):
    pass


def get(name: str) -> Profile:
    if name not in PROFILES:
        raise ProfileError(
            f"unknown generation profile {name!r}. Available: "
            + ", ".join(sorted(PROFILES))
        )
    return PROFILES[name]


def apply_profile(cfg: dict, name: str) -> dict:
    """Return ``cfg`` with the named profile's overrides laid on top.

    Applied *after* a ``--config`` file so a profile is the last word on the
    settings it owns; anything it does not name is left exactly as the file (or
    the built-in defaults) had it.
    """
    from .config import apply_overrides

    return apply_overrides(copy.deepcopy(cfg), dict(get(name).overrides))


def _hash_of(name: str) -> str:
    return config_hash(apply_profile(copy.deepcopy(DEFAULTS), name))


def identify(cfg_or_hash) -> str | None:
    """Which profile produced this config (or config hash), if any.

    Lets a dataset generated before profiles existed still be labelled with
    the profile it happens to match, rather than showing up as unexplained.
    Returns None for a config that no profile reproduces — an older generator
    version, or a hand-edited YAML.
    """
    want = cfg_or_hash if isinstance(cfg_or_hash, str) else config_hash(cfg_or_hash)
    for name in PROFILES:
        if _hash_of(name) == want:
            return name
    return None


def to_json() -> list[dict]:
    """Serialize the registry for the dashboard's profile dropdown."""
    from dataclasses import asdict

    return [asdict(p) | {"config_hash": _hash_of(p.name)[:12]}
            for p in PROFILES.values()]
