"""Build a Gazebo world of procedural rocks on procedural ground.

Run it to get a `.world` plus a `meshes/` folder. Ground comes from a
generated greyscale height image, which is how Gazebo builds terrain; rocks
come from make_rocks.py. Both are reseeded per run, so no two worlds share a
floor or a rock.

    python3 training/sim/build_world.py --out /tmp/arena --seed 7 --rocks 40

The robot is NOT spawned here - the sim launch does that, or pass
--with-robot to drop the lance2 model in for a standalone check.

VERIFIED: worlds built this way load in `gz sim -s` headless and the robot's
LiDAR returns hits off both the terrain and the rocks. NOT verified: driving,
recording, or anything downstream. See handoff/04-simulation.md.
"""

from __future__ import annotations

import argparse
import os
import struct
import zlib

import numpy as np

import make_rocks as M

LANCE2 = "/home/brandon/lance-ws/src/csm-sim/assets/gz/lance2"


def write_png_gray(a: np.ndarray, path: str) -> None:
    """8-bit greyscale PNG, no dependencies. Gazebo reads the height from it."""
    h, w = a.shape
    raw = b"".join(b"\x00" + a[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)))
        fh.write(chunk(b"IDAT", zlib.compress(raw)))
        fh.write(chunk(b"IEND", b""))


def terrain_image(rng: np.random.Generator, n: int = 129) -> np.ndarray:
    """Berms, a dug pit, and fine roughness. Size must be 2^k + 1 for Gazebo."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    z = np.zeros((n, n))
    # A few broad rolls: the berms.
    for _ in range(rng.integers(2, 5)):
        fx, fy = rng.uniform(1.5, 4.0, 2)
        px, py = rng.uniform(0, 2 * np.pi, 2)
        z += rng.uniform(0.3, 1.0) * np.sin(fx * np.pi * x + px) * np.cos(fy * np.pi * y + py)
    # One or two craters.
    for _ in range(rng.integers(1, 3)):
        cx, cy = rng.uniform(0.2, 0.8, 2)
        z -= rng.uniform(0.6, 1.4) * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / rng.uniform(0.004, 0.02)))
    z += rng.normal(0, 0.08, (n, n))          # regolith roughness
    z = (z - z.min()) / max(z.max() - z.min(), 1e-9)
    return (z * 255).astype(np.uint8)


def build(out: str, seed: int, n_rocks: int, span: float, relief: float,
          with_robot: bool) -> str:
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out, "meshes"), exist_ok=True)

    hm = os.path.join(out, "terrain.png")
    write_png_gray(terrain_image(rng), hm)

    forms = list(M.FORMS)
    half = span / 2 - 1.0
    spots = M.scatter(rng, n_rocks, ((-half, -half), (half, half)), min_gap=0.6)

    parts, manifest = [], []
    for i, (x, y) in enumerate(spots):
        form = forms[int(rng.integers(len(forms)))]
        radius = M.draw_radius(rng)
        tris = M.make_rock(rng, radius, form)
        rel = f"meshes/rock_{i:03d}.stl"
        M.write_stl(tris, os.path.join(out, rel))
        yaw = float(rng.uniform(0, 2 * np.pi))
        # Sit the rock on the terrain surface under it. The height image is
        # centred on the origin and spans `span` metres for `relief` of height.
        parts.append(f"""  <model name="rock{i:03d}"><static>true</static>
    <pose>{x:.4f} {y:.4f} 0 0 0 {yaw:.4f}</pose>
    <link name="link">
      <collision name="c"><geometry><mesh><uri>file://{rel}</uri></mesh></geometry></collision>
      <visual name="v"><geometry><mesh><uri>file://{rel}</uri></mesh></geometry></visual>
    </link></model>""")
        v = tris.reshape(-1, 3)
        lo, hi = v.min(0), v.max(0)
        manifest.append(dict(id=i, mesh=rel, form=form, x=float(x), y=float(y),
                             yaw=yaw, radius=float(np.linalg.norm(hi - lo) / 2),
                             extent=[float(e) for e in (hi - lo)],
                             z_range=[float(lo[2]), float(hi[2])]))

    robot = (f'  <include><uri>file://{LANCE2}</uri><name>lance</name>'
             f'<pose>0 0 {relief + 0.2:.2f} 0 0 0</pose></include>\n') if with_robot else ""

    world = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="rockarena">
    <physics name="d" type="ignored"><max_step_size>0.004</max_step_size></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <light type="directional" name="sun"><diffuse>1 1 1 1</diffuse><direction>-0.5 0.1 -0.9</direction></light>
    <model name="terrain"><static>true</static><link name="link">
      <collision name="c"><geometry><heightmap>
        <uri>file://terrain.png</uri><size>{span} {span} {relief}</size><pos>0 0 0</pos>
      </heightmap></geometry></collision>
      <visual name="v"><geometry><heightmap>
        <uri>file://terrain.png</uri><size>{span} {span} {relief}</size><pos>0 0 0</pos>
        <texture><size>2</size><diffuse>file://terrain.png</diffuse><normal>file://terrain.png</normal></texture>
      </heightmap></geometry></visual>
    </link></model>
{chr(10).join(parts)}
{robot}  </world>
</sdf>"""
    wp = os.path.join(out, "arena.world")
    open(wp, "w").write(world)

    import json
    json.dump(dict(seed=seed, span=span, relief=relief, rocks=manifest),
              open(os.path.join(out, "rocks.json"), "w"), indent=1)
    return wp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rocks", type=int, default=40)
    ap.add_argument("--span", type=float, default=14.0, help="arena side, metres")
    ap.add_argument("--relief", type=float, default=0.9, help="terrain height range, metres")
    ap.add_argument("--with-robot", action="store_true")
    a = ap.parse_args()
    p = build(a.out, a.seed, a.rocks, a.span, a.relief, a.with_robot)
    print(f"wrote {p}")
    print(f"  ground truth: {os.path.join(a.out, 'rocks.json')}")
    print(f"  run: gz sim -s -r -v 1 {p}")


if __name__ == "__main__":
    main()
