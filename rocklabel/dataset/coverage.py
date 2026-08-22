"""Is every labelled rock actually visible in the dataset, and how far away?

Two failure modes have bitten this project, both silent — the dataset builds,
the manifest looks healthy, and the model simply never sees part of what was
labelled:

* **Rocks that produce no training samples at all.** A candidate ball needs
  ``min_neighbors`` returns inside ``neighborhood_radius_m`` to become a
  sample. On sparse frames a real, correctly labelled rock can fall under that
  floor in every single frame and quietly contribute nothing. Twelve of the
  sixty-three labelled volleyball rocks did exactly that on the raw-burst
  datasets.
* **An effective range far shorter than the crop box.** The same floor bites
  hardest far from the sensor, where returns thin out. The raw-burst datasets
  cropped to 6 m forward but 88% of their rock samples sat within 2 m, so the
  model was never taught what a rock looks like at range — and nothing in the
  manifest said so.

Both are measured here rather than assumed. Neither needs torch or a trained
model; it reads the generated dataset and the label file it was built from.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from .generate import MANIFEST_NAME


def _rock_centers(labels_path: str) -> tuple[np.ndarray, list[int]]:
    with open(labels_path) as f:
        labelset = json.load(f)
    rocks = labelset.get("rocks") or []
    centers = np.array([r["center"] for r in rocks], float).reshape(-1, 3)
    return centers, [int(r["id"]) for r in rocks]


def measure_run(dataset_dir: str, run_id: str, labels_path: str) -> dict:
    """Sample ranges and per-rock sample counts for one recording.

    Range is measured from the sensor's own position in the frame the sample
    came from (stored per frame as ``robot_pose``), not from the world origin —
    the rig walks, so distance from the origin would be a different question.
    """
    centers, ids = _rock_centers(labels_path)
    per_rock = np.zeros(len(centers), int)
    rock_d: list[np.ndarray] = []
    clear_d: list[np.ndarray] = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "points", run_id,
                                              "frame_*.npz"))):
        with np.load(path) as z:
            xyz = z["centers_odom"].astype(float)
            lab = z["labels"]
            origin = z["robot_pose"][:3, 3]
        d = np.linalg.norm(xyz - origin, axis=1)
        rock_d.append(d[lab == 1])
        clear_d.append(d[lab == 0])
        # Every rock-labelled sample belongs to the labelled rock it sits
        # nearest. Samples are picked inside a rock's own outline, so the
        # nearest centre is the rock it came from except where two labels
        # overlap — which is itself worth knowing and is reported separately.
        rocks_here = xyz[lab == 1]
        if len(rocks_here) and len(centers):
            near = np.linalg.norm(rocks_here[:, None] - centers[None], axis=2).argmin(1)
            np.add.at(per_rock, near, 1)

    rock = np.concatenate(rock_d) if rock_d else np.zeros(0)
    clear = np.concatenate(clear_d) if clear_d else np.zeros(0)

    # Labels closer together than their own radii: the nearest-centre rule
    # cannot tell their samples apart, so their per-rock counts are a split of
    # one pile rather than two independent measurements.
    overlapping = []
    with open(labels_path) as f:
        radii = [float(r.get("radius", 0.0)) for r in (json.load(f).get("rocks") or [])]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            gap = float(np.linalg.norm(centers[i] - centers[j]))
            if gap < radii[i] + radii[j]:
                overlapping.append({"rocks": [ids[i], ids[j]], "gap_m": round(gap, 3)})

    def pct(a, p):
        return float(np.percentile(a, p)) if len(a) else 0.0

    return {
        "run_id": run_id,
        "rocks": len(centers),
        "rocks_with_samples": int((per_rock > 0).sum()),
        "silent_rocks": [ids[i] for i in range(len(centers)) if per_rock[i] == 0],
        "samples_per_rock": {ids[i]: int(per_rock[i]) for i in range(len(centers))},
        "rock_samples": int(len(rock)),
        "rock_range_median_m": pct(rock, 50),
        "rock_range_p95_m": pct(rock, 95),
        "rock_range_max_m": float(rock.max()) if len(rock) else 0.0,
        "rock_within_2m": float((rock <= 2.0).mean()) if len(rock) else 0.0,
        "clear_range_median_m": pct(clear, 50),
        "clear_range_max_m": float(clear.max()) if len(clear) else 0.0,
        "overlapping_labels": overlapping,
    }


def measure_dataset(dataset_dir: str, labels_dir: str | None = None) -> dict:
    """Every run of one generated dataset."""
    manifest_path = os.path.join(dataset_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        raise SystemExit(f"{dataset_dir!r} has no {MANIFEST_NAME} — "
                         "point this at a folder written by 'rocklabel generate'")
    with open(manifest_path) as f:
        manifest = json.load(f)

    runs = []
    for run_id, entry in sorted(manifest["runs"].items()):
        # The manifest records where the labels came from; fall back to a
        # named folder for a dataset copied off the machine that built it.
        path = entry.get("labels_path", "")
        if not os.path.exists(path) and labels_dir:
            path = os.path.join(labels_dir, f"{run_id}.labels.json")
        if not os.path.exists(path):
            raise SystemExit(f"cannot find the label file for {run_id} "
                             f"(tried {entry.get('labels_path')!r}); pass --labels-dir")
        runs.append(measure_run(dataset_dir, run_id, path))

    total_rocks = sum(r["rocks"] for r in runs)
    silent = sum(len(r["silent_rocks"]) for r in runs)
    return {
        "dataset": dataset_dir,
        "profile": manifest.get("profile") or "",
        "runs": runs,
        "rocks": total_rocks,
        "silent_rocks": silent,
        "crop_forward_m": manifest["config"]["generator"]["crop_forward_m"],
        "min_neighbors": manifest["config"]["generator"]["min_neighbors"],
    }


def print_report(result: dict) -> None:
    print(f"\n=== rock coverage: {result['dataset']} "
          f"({result['profile'] or 'unnamed profile'}) ===\n")
    print(f"{'run':28s} {'rocks':>5s} {'seen':>5s} {'samples':>8s} "
          f"{'median m':>9s} {'p95 m':>7s} {'max m':>7s} {'<=2 m':>7s}")
    for r in result["runs"]:
        print(f"{r['run_id']:28s} {r['rocks']:5d} {r['rocks_with_samples']:5d} "
              f"{r['rock_samples']:8d} {r['rock_range_median_m']:9.2f} "
              f"{r['rock_range_p95_m']:7.2f} {r['rock_range_max_m']:7.2f} "
              f"{r['rock_within_2m']:7.1%}")

    silent = [(r["run_id"], r["silent_rocks"]) for r in result["runs"] if r["silent_rocks"]]
    print(f"\nLabelled rocks that produce no training sample at all: "
          f"{result['silent_rocks']} of {result['rocks']}")
    if silent:
        for run_id, ids in silent:
            print(f"  {run_id}: rock ids {ids}")
        print(f"  These are labelled and invisible. Each frame needs "
              f"{result['min_neighbors']} returns inside the neighbourhood "
              f"radius before a candidate becomes a sample, and these rocks "
              f"never reach it. Either the rock is too far away in every frame, "
              f"or the frames are too sparse — a denser generation profile is "
              f"the usual fix.")
    else:
        print("  none — every labelled rock is represented in the training data.")

    overlaps = [(r["run_id"], r["overlapping_labels"]) for r in result["runs"]
                if r["overlapping_labels"]]
    if overlaps:
        print("\nLabels closer together than their own radii — their samples "
              "cannot be told apart, so per-rock counts above split one pile "
              "between them:")
        for run_id, pairs in overlaps:
            joined = ", ".join(f"{p['rocks'][0]}-{p['rocks'][1]} ({p['gap_m']:.2f} m)"
                               for p in pairs)
            print(f"  {run_id}: {joined}")

    allrock = [r for r in result["runs"] if r["rock_samples"]]
    if allrock:
        far = max(r["rock_range_max_m"] for r in allrock)
        print(f"\nThe crop box reaches {result['crop_forward_m']:.1f} m forward; "
              f"the furthest rock sample in this dataset sits at {far:.2f} m. "
              "A large gap between those two numbers means the model is never "
              "taught what a rock looks like at range, whatever the crop says.")
