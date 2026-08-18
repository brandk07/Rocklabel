"""Rock label set: load/save of the versioned odom-frame JSON schema.

Schema v2 adds shaped labels. Every rock carries a bounding sphere
(center + radius) regardless of shape, so consumers that only care about
"roughly where / roughly how big" (driftcheck, preview, sample placement)
work for all shapes. Exact membership tests live in rocklabel.dataset.labeling.

Shapes:
  sphere   center [3], radius
  box      center [3], size [3] full extents, axis-aligned
  polygon  vertices [N, 2] odom xy, z_range (base, top); extruded prism

Schema v3 adds an optional ``arena``: one odom-frame xy polygon marking the
competition area. It is not a label - nothing inside it is called a rock -
it marks which ground is *eligible* to become a training sample at all. The
generator's crop is a box that follows the robot, so it cannot tell "four
metres to port, arena floor" from "four metres to port, a couch"; without an
arena the negative class ends up partly defined by whatever furniture shared
the room. Measured on the comforter runs, 29% of clear samples contained
structure over half a metre tall.

Schema v4 adds an optional ``level``: the mount rotation that was applied to
the cloud these centers were picked on (see :mod:`rocklabel.geometry.leveling`). It is
not geometry, it is a frame stamp - centers are world coordinates, so the
generator has to be able to tell that it is replaying the recording the same
way up. Absent (every v1-v3 file) means unlevelled.

Schema v5 adds an optional ``z_band``: the odom-frame (min, max) height slab
the labeler's z clip was left at. It is the vertical half of what ``arena``
does horizontally - together they bound the volume that is eligible to become
training data. Without it the only vertical bound is the generator's crop box,
which follows the sensor, so anything within crop_up_m *above* the sensor -
ceiling, lights, the person holding the rig - trains as clear ground. Absent
(every v1-v4 file) means no height restriction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from . import __version__

SCHEMA_VERSION = 5
# v1 = spheres only, v2 = shaped, v3 = + arena, v4 = + level, v5 = + z_band
_READABLE_VERSIONS = (1, 2, 3, 4, 5)


@dataclass
class Rock:
    id: int
    center: np.ndarray  # [3] odom-frame meters (bounding center for box/polygon)
    radius: float       # sphere radius; bounding radius for box/polygon
    shape: str = "sphere"
    size: np.ndarray | None = None       # box only: [sx, sy, sz] full extents
    vertices: np.ndarray | None = None   # polygon only: [N, 2] odom xy
    z_range: tuple[float, float] | None = None  # polygon only: (base, top)


def sync_bounds(rock: Rock) -> None:
    """Recompute the bounding center/radius from the shape's own fields."""
    if rock.shape == "box":
        rock.radius = float(np.linalg.norm(np.asarray(rock.size, float)) / 2.0)
    elif rock.shape == "polygon":
        v = np.asarray(rock.vertices, float)
        z0, z1 = rock.z_range
        lo = np.array([v[:, 0].min(), v[:, 1].min(), z0])
        hi = np.array([v[:, 0].max(), v[:, 1].max(), z1])
        rock.center = (lo + hi) / 2.0
        rock.radius = float(np.linalg.norm(hi - lo) / 2.0)


def translate_rock(rock: Rock, delta) -> None:
    """Move a rock rigidly; keeps shape fields and bounds consistent."""
    delta = np.asarray(delta, float)
    rock.center = rock.center + delta
    if rock.shape == "polygon":
        rock.vertices = np.asarray(rock.vertices, float) + delta[:2]
        rock.z_range = (rock.z_range[0] + delta[2], rock.z_range[1] + delta[2])


@dataclass
class LabelSet:
    mcap_file: str = ""
    run_id: str = ""
    odom_frame: str = "odom"
    intensity_available: bool = True
    accumulator_voxel_m: float = 0.03
    rocks: list[Rock] = field(default_factory=list)
    #: Optional arena footprint: [N, 2] odom xy, N >= 3. None = no boundary,
    #: which is the pre-v3 behavior (every candidate center is eligible).
    arena: np.ndarray | None = None
    #: Frame stamp: the levelling applied to the cloud these centers were
    #: picked on, as :func:`rocklabel.geometry.leveling.level_record` returns it.
    #: None = unlevelled, which is the pre-v4 behavior.
    level: dict | None = None
    #: Optional height slab, odom-frame (min, max) meters. The vertical
    #: partner of ``arena``. None = no height restriction, which is the
    #: pre-v5 behavior (the generator's crop box is the only vertical bound).
    z_band: tuple[float, float] | None = None
    _next_id: int = 1

    def set_arena(self, vertices) -> np.ndarray:
        """Replace the arena footprint. Only xy is used - the height slab is
        the separate ``z_band``, set from the labeler's z clip."""
        v = np.asarray(vertices, float).reshape(-1, 2)
        if len(v) < 3:
            raise ValueError(f"an arena polygon needs at least 3 vertices, got {len(v)}")
        self.arena = v
        return v

    def clear_arena(self) -> None:
        self.arena = None

    def set_z_band(self, z_min: float, z_max: float) -> tuple[float, float]:
        """Replace the eligible height slab. Ends are sorted, so a caller can
        hand over two slider values without caring which one is on top."""
        lo, hi = sorted((float(z_min), float(z_max)))
        self.z_band = (lo, hi)
        return self.z_band

    def clear_z_band(self) -> None:
        self.z_band = None

    def _append(self, rock: Rock) -> Rock:
        self._next_id += 1
        self.rocks.append(rock)
        return rock

    def add(self, center, radius: float) -> Rock:
        return self._append(Rock(self._next_id, np.asarray(center, float), float(radius)))

    def add_box(self, center, size) -> Rock:
        rock = Rock(self._next_id, np.asarray(center, float), 0.0,
                    shape="box", size=np.asarray(size, float))
        sync_bounds(rock)
        return self._append(rock)

    def add_polygon(self, vertices, z_min: float, z_max: float) -> Rock:
        rock = Rock(self._next_id, np.zeros(3), 0.0, shape="polygon",
                    vertices=np.asarray(vertices, float).reshape(-1, 2),
                    z_range=(float(z_min), float(z_max)))
        sync_bounds(rock)
        return self._append(rock)

    def remove(self, rock_id: int) -> None:
        self.rocks = [r for r in self.rocks if r.id != rock_id]

    def get(self, rock_id: int) -> Rock | None:
        for r in self.rocks:
            if r.id == rock_id:
                return r
        return None

    def centers_radii(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounding centers/radii for all rocks (exact for spheres)."""
        if not self.rocks:
            return np.empty((0, 3)), np.empty(0)
        return (
            np.stack([r.center for r in self.rocks]).astype(float),
            np.array([r.radius for r in self.rocks], float),
        )

    def save(self, path: str) -> None:
        data = {
            "schema_version": SCHEMA_VERSION,
            "mcap_file": self.mcap_file,
            "run_id": self.run_id,
            "odom_frame": self.odom_frame,
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tool_version": __version__,
            "intensity_available": self.intensity_available,
            "accumulator_voxel_m": self.accumulator_voxel_m,
            "rocks": [_rock_to_dict(r) for r in self.rocks],
        }
        if self.arena is not None:
            data["arena"] = {"vertices": [[_r4(x), _r4(y)]
                                          for x, y in np.asarray(self.arena, float)]}
        if self.level is not None:
            data["level"] = dict(self.level)
        if self.z_band is not None:
            data["z_band"] = [_r4(self.z_band[0]), _r4(self.z_band[1])]
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)


def _r4(x) -> float:
    return round(float(x), 4)


def _rock_to_dict(r: Rock) -> dict:
    d = {"id": r.id, "shape": r.shape,
         "center": [_r4(c) for c in r.center], "radius": _r4(r.radius)}
    if r.shape == "box":
        d["size"] = [_r4(s) for s in r.size]
    elif r.shape == "polygon":
        d["vertices"] = [[_r4(x), _r4(y)] for x, y in np.asarray(r.vertices, float)]
        d["z_range"] = [_r4(r.z_range[0]), _r4(r.z_range[1])]
    return d


def _rock_from_dict(d: dict) -> Rock:
    shape = d.get("shape", "sphere")
    rock = Rock(int(d["id"]), np.asarray(d["center"], float),
                float(d.get("radius", 0.0)), shape=shape)
    if shape == "box":
        rock.size = np.asarray(d["size"], float)
    elif shape == "polygon":
        rock.vertices = np.asarray(d["vertices"], float).reshape(-1, 2)
        z0, z1 = d["z_range"]
        rock.z_range = (float(z0), float(z1))
    if shape != "sphere":
        sync_bounds(rock)
    return rock


def load_labels(path: str) -> LabelSet:
    with open(path, "r") as f:
        data = json.load(f)
    version = data.get("schema_version")
    if version not in _READABLE_VERSIONS:
        raise ValueError(f"{path}: unsupported label schema_version {version} "
                         f"(expected one of {_READABLE_VERSIONS})")
    ls = LabelSet(
        mcap_file=data.get("mcap_file", ""),
        run_id=data.get("run_id", ""),
        odom_frame=data.get("odom_frame", "odom"),
        intensity_available=data.get("intensity_available", True),
        accumulator_voxel_m=data.get("accumulator_voxel_m", 0.03),
        level=data.get("level"),
    )
    ls.rocks = [_rock_from_dict(r) for r in data.get("rocks", [])]
    arena = data.get("arena")
    if arena:
        ls.arena = np.asarray(arena["vertices"], float).reshape(-1, 2)
    band = data.get("z_band")
    if band:
        ls.z_band = (float(band[0]), float(band[1]))
    ls._next_id = max((r.id for r in ls.rocks), default=0) + 1
    return ls
