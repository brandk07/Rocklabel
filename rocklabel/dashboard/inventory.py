"""Scan the project folders and describe everything the pipeline has produced.

Two tiers, deliberately:

* **Cheap listings** (:func:`snapshot`) stat the filesystem only — every list in
  the dashboard renders from these, so opening a page never blocks on decoding
  an mcap.
* **Deep info** (:func:`recording_info`) actually opens a recording's index.
  It is fetched on demand when a row is expanded and memoized on
  ``(path, mtime, size)``, so a file is only ever read once per version.

Every listing here is read-only. The two writes are :func:`rename` and
:func:`delete` — a ``mv`` and an ``rm`` on the folders and files the listings
above describe. They are housekeeping, not pipeline steps worth shelling out
for, and :func:`_target` is the gate every path the browser names passes
through.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone

#: Conventional layout (README: "this repo keeps three top-level folders").
DIRS = {
    "recordings": "recordings",
    "labels": "labels",
    "datasets": "datasets",
    "training": "training",
    "runs": os.path.join("training", "runs"),
    "cache": os.path.join("training", "cache"),
    "results": os.path.join("training", "results"),
    "exported": os.path.join("training", "exported"),
}

_info_cache: dict[tuple, dict] = {}
_info_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _stat(path: str) -> dict:
    try:
        st = os.stat(path)
    except OSError:
        return {"size": 0, "mtime": 0.0}
    return {"size": st.st_size, "mtime": st.st_mtime}


def _iso(ts: float) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _dir_size(path: str) -> tuple[int, int]:
    """(total bytes, file count) — walked once, used for dataset/run cards."""
    total = files = 0
    for dirpath, _dirs, names in os.walk(path):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, n))
                files += 1
            except OSError:
                pass
    return total, files


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# listings
# --------------------------------------------------------------------------- #
def recordings(root: str) -> list[dict]:
    base = os.path.join(root, DIRS["recordings"])
    out = []
    for dirpath, _dirs, names in os.walk(base):
        for name in names:
            if not name.endswith(".mcap"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            st = _stat(full)
            out.append({
                "name": name,
                "path": rel,
                "stem": name[:-5],
                "size": st["size"],
                "mtime": st["mtime"],
                "modified": _iso(st["mtime"]),
                "labels": _labels_path_for(root, name[:-5]),
            })
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def _labels_path_for(root: str, stem: str) -> str | None:
    """Where `label` would have written this recording's labels, if it exists."""
    for cand in (os.path.join(DIRS["labels"], f"{stem}.labels.json"),
                 os.path.join(DIRS["recordings"], f"{stem}.labels.json")):
        if os.path.exists(os.path.join(root, cand)):
            return cand
    return None


def labels(root: str) -> list[dict]:
    out = []
    seen = set()
    for folder in (DIRS["labels"], DIRS["recordings"]):
        base = os.path.join(root, folder)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if not name.endswith(".labels.json") or name in seen:
                continue
            seen.add(name)
            full = os.path.join(base, name)
            data = _read_json(full) or {}
            rocks = data.get("rocks") or []
            shapes: dict[str, int] = {}
            for r in rocks:
                shapes[r.get("shape", "sphere")] = shapes.get(r.get("shape", "sphere"), 0) + 1
            st = _stat(full)
            out.append({
                "name": name,
                "path": os.path.relpath(full, root),
                # The filename's stem, which is what a rename edits — normally
                # equal to run_id, but the file on disk is the thing that moves.
                "stem": name[: -len(".labels.json")],
                "run_id": data.get("run_id", name.replace(".labels.json", "")),
                "mcap_file": data.get("mcap_file", ""),
                "rock_count": len(rocks),
                "shapes": shapes,
                "rock_ids": [r.get("id") for r in rocks][:64],
                "created": data.get("created", ""),
                "intensity": bool(data.get("intensity_available")),
                "schema_version": data.get("schema_version"),
                "size": st["size"],
                "mtime": st["mtime"],
                "modified": _iso(st["mtime"]),
            })
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def datasets(root: str) -> list[dict]:
    base = os.path.join(root, DIRS["datasets"])
    out = []
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        full = os.path.join(base, name)
        if not os.path.isdir(full):
            continue
        manifest = _read_json(os.path.join(full, "manifest.json"))
        size, files = _dir_size(full)
        entry = {
            "name": name,
            "path": os.path.relpath(full, root),
            "size": size,
            "files": files,
            "mtime": _stat(full)["mtime"],
            "has_manifest": manifest is not None,
            "runs": [],
            "samples": 0,
            "rock_samples": 0,
            "clear_samples": 0,
            "bev_frames": 0,
            "config_hash": "",
            "generated": "",
            "generator": {},
        }
        if manifest:
            entry["config_hash"] = (manifest.get("config_hash") or "")[:12]
            entry["generated"] = manifest.get("generated", "")
            entry["generator"] = (manifest.get("config") or {}).get("generator", {})
            for run_id, run in sorted((manifest.get("runs") or {}).items()):
                sl = run.get("sample_labels") or {}
                pl = run.get("point_labels") or {}
                entry["runs"].append({
                    "run_id": run_id,
                    "frames": run.get("frames_kept", 0),
                    "frames_skipped": (run.get("frames_skipped_pose", 0)
                                       + run.get("frames_skipped_empty", 0)),
                    "samples": run.get("point_samples", 0),
                    "bev_frames": run.get("bev_frames", 0),
                    "rock": sl.get("rock", 0),
                    "clear": sl.get("clear", 0),
                    "rock_points": pl.get("rock", 0),
                    "clear_points": pl.get("clear", 0),
                    "ignore_points": pl.get("ignore", 0),
                    "rock_count": run.get("rock_count", 0),
                    "intensity": bool(run.get("intensity_available")),
                    "generated": run.get("generated", ""),
                    "mcap": os.path.basename(run.get("mcap_path", "")),
                })
                entry["samples"] += run.get("point_samples", 0)
                entry["rock_samples"] += sl.get("rock", 0)
                entry["clear_samples"] += sl.get("clear", 0)
                entry["bev_frames"] += run.get("bev_frames", 0)
        out.append(entry)
    return out


#: A name we are willing to create: no path separators, no dotfiles, nothing a
#: shell or a URL would have to quote. Extensions are ours to append, never the
#: client's to supply.
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_LABELS_SUFFIX = ".labels.json"
_MCAP_SUFFIX = ".mcap"


def _inside(path: str, base: str) -> bool:
    return path == base or path.startswith(base + os.sep)


def _recording_path_for(root: str, stem: str) -> str | None:
    """The recording with this stem, wherever under ``recordings/`` it sits."""
    for rec in recordings(root):
        if rec["stem"] == stem:
            return os.path.join(root, rec["path"])
    return None


def _labels_abs_for(root: str, stem: str) -> str | None:
    rel = _labels_path_for(root, stem)
    return os.path.join(root, rel) if rel else None


def _target(root: str, rel_path: str) -> tuple[str, str, list[str]]:
    """Classify a rename/delete target: ``(kind, stem, paths that move together)``.

    ``kind`` is ``"dataset"`` or ``"run"``. A recording and the label file named
    after it are one run: :func:`rocklabel.labeler.default_labels_path` derives
    the label filename from the mcap stem, ``label`` rewrites the ``run_id`` and
    ``mcap_file`` inside it from that same stem on every session, and the
    dashboard links the two by it. Rename one alone and the pair is orphaned —
    the recording reads as unlabeled and the next `label` run starts from
    scratch — so a rename moves the whole group.

    Everything the browser can name arrives here first, so this is also the
    gate: a dataset must be a direct child of ``datasets/``, and a run file must
    be an ``.mcap`` or ``.labels.json`` under ``recordings/`` or ``labels/``.
    """
    root_n = os.path.normpath(root)
    full = os.path.normpath(os.path.join(root, rel_path))
    ds_base = os.path.join(root_n, DIRS["datasets"])
    rec_base = os.path.join(root_n, DIRS["recordings"])
    lab_base = os.path.join(root_n, DIRS["labels"])

    if os.path.dirname(full) == ds_base:
        if not os.path.isdir(full):
            raise FileNotFoundError(rel_path)
        return "dataset", os.path.basename(full), [full]

    name = os.path.basename(full)
    for suffix in (_LABELS_SUFFIX, _MCAP_SUFFIX):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            break
    else:
        raise ValueError("only a dataset folder, a recording or a label file "
                         "can be renamed or deleted")
    if not (_inside(full, rec_base) or _inside(full, lab_base)):
        raise ValueError("only files under recordings/ or labels/ can be "
                         "renamed or deleted")
    if not os.path.isfile(full):
        raise FileNotFoundError(rel_path)

    group = []
    for cand in (_recording_path_for(root, stem), _labels_abs_for(root, stem)):
        if cand and cand not in group:
            group.append(cand)
    if full not in group:  # a second label file the listings would have shadowed
        group.append(full)
    return "run", stem, group


def rename_targets(root: str, rel_path: str) -> list[str]:
    """Absolute paths that renaming ``rel_path`` would move."""
    return _target(root, rel_path)[2]


def _check_name(new_name: str) -> str:
    new_name = new_name.strip()
    if not _SAFE_NAME.fullmatch(new_name):
        raise ValueError("a name must start with a letter or digit and use only "
                         "letters, digits, dot, dash and underscore")
    return new_name


def _suffix_of(path: str) -> str:
    return _LABELS_SUFFIX if path.endswith(_LABELS_SUFFIX) else _MCAP_SUFFIX


def _moved(root: str, primary: str, moved: list[str], name: str) -> dict:
    return {"name": name, "path": os.path.relpath(primary, root),
            "renamed": [os.path.relpath(p, root) for p in moved]}


def rename(root: str, rel_path: str, new_name: str) -> dict:
    """Rename a dataset folder, or a recording and its label file together.

    Returns the new name, the new path of the item that was asked about, and
    every path that moved.

    Renaming is safe by construction in both cases, but for different reasons:

    * A dataset folder's name is not an identity — training keys on the
      ``run_id`` inside the manifest and the cache holds its own copy of the
      pooled samples, so an existing cache and every run trained from it stay
      valid. Only the ``dataset_dir`` provenance in
      ``training/cache/meta.json`` goes stale; nothing reads it.
    * A run's stem *is* its identity, so the ``run_id`` and ``mcap_file``
      recorded inside the label file are rewritten to match — exactly what the
      next ``label`` session would have written anyway.

    What a run rename does not touch is history: datasets generated before it
    keep the old ``run_id``, and their manifests keep pointing at the old
    ``mcap_path``. Regenerating into such a dataset therefore *adds* a run under
    the new id rather than replacing the old one, and pooling that dataset would
    then count the same recording twice. Delete the stale run's files (or the
    dataset) before regenerating.
    """
    kind, stem, group = _target(root, rel_path)
    new_name = _check_name(new_name)
    full = os.path.normpath(os.path.join(root, rel_path))
    if new_name == stem:
        return _moved(root, full, [], stem)

    if kind == "dataset":
        dst = os.path.join(os.path.dirname(full), new_name)
        # samefile() rather than a bare exists(): a case-only rename must still
        # work on a case-insensitive volume, where the destination "exists".
        if os.path.exists(dst) and not os.path.samefile(dst, full):
            raise ValueError(f"{os.path.relpath(dst, root)} already exists")
        os.rename(full, dst)
        return _moved(root, dst, [dst], new_name)

    dests = {src: os.path.join(os.path.dirname(src), new_name + _suffix_of(src))
             for src in group}
    # Nothing else may already answer to the new stem — including a label file in
    # the other folder `label` writes to, which the listings would shadow rather
    # than show.
    for taken in (_recording_path_for(root, new_name), _labels_abs_for(root, new_name)):
        if taken and taken not in group:
            raise ValueError(f"{os.path.relpath(taken, root)} already exists")
    for src, dst in dests.items():
        if os.path.exists(dst) and not os.path.samefile(dst, src):
            raise ValueError(f"{os.path.relpath(dst, root)} already exists")

    done: list[tuple[str, str]] = []
    try:
        for src, dst in dests.items():
            os.rename(src, dst)
            done.append((src, dst))
    except OSError:
        for src, dst in reversed(done):  # a half-renamed pair is worse than none
            os.rename(dst, src)
        raise

    mcap = next((d for _s, d in done if d.endswith(_MCAP_SUFFIX)), None)
    for _src, dst in done:
        if dst.endswith(_LABELS_SUFFIX):
            _retag_labels(dst, new_name, os.path.basename(mcap) if mcap else None)
    return _moved(root, dests[full], list(dests.values()), new_name)


def _retag_labels(path: str, stem: str, mcap_name: str | None) -> None:
    """Point a moved label file at its new identity.

    Edits the raw JSON rather than round-tripping through ``LabelSet`` so that
    fields this module does not know about survive, and writes through a temp
    file like :meth:`rocklabel.labels.LabelSet.save` does. ``mcap_file`` is left
    alone when no recording moved with it: naming an mcap that is not there
    would be a worse lie than the old name.
    """
    data = _read_json(path)
    if data is None:  # unreadable JSON: the rename still stands, the fields don't
        return
    data["run_id"] = stem
    if mcap_name:
        data["mcap_file"] = mcap_name
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def delete(root: str, rel_path: str) -> dict:
    """Delete one dataset folder, recording, or label file. No undo.

    Deliberately narrower than :func:`rename`: it removes only what was named,
    never the rest of the group. A recording and its labels are one run for
    naming purposes, but labels cost an hour of somebody's afternoon and an
    ``.mcap`` costs gigabytes — which of the two you want gone is not something
    to guess at. The caller is expected to have said out loud what will be left
    behind.

    Nothing downstream is invalidated: a built training cache holds its own copy
    of the samples, so deleting the dataset it came from leaves the cache and
    every run trained from it intact.
    """
    kind, _stem, _group = _target(root, rel_path)
    full = os.path.normpath(os.path.join(root, rel_path))
    size, files = (_dir_size(full) if kind == "dataset"
                   else (_stat(full)["size"], 1))
    if os.path.islink(full):
        os.unlink(full)  # never follow a link out of the project to rmtree it
    elif kind == "dataset":
        shutil.rmtree(full)
    else:
        os.remove(full)
    return {"path": rel_path, "kind": kind, "freed": size, "files": files}


def _history(path: str) -> list[dict]:
    """Parse history.csv into typed rows for the training-curve chart."""
    rows: list[dict] = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                typed = {}
                for k, v in row.items():
                    if k is None:
                        continue
                    try:
                        typed[k] = float(v)
                    except (TypeError, ValueError):
                        typed[k] = v
                rows.append(typed)
    except OSError:
        pass
    return rows


def runs(root: str) -> list[dict]:
    base = os.path.join(root, DIRS["runs"])
    out = []
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        full = os.path.join(base, name)
        if not os.path.isdir(full):
            continue
        cfg = _read_json(os.path.join(full, "config.json")) or {}
        metrics = _read_json(os.path.join(full, "test_metrics.json")) or {}
        best = os.path.join(full, "best.pt")
        history = _history(os.path.join(full, "history.csv"))
        out.append({
            "name": name,
            "path": os.path.relpath(full, root),
            "model": cfg.get("model") or metrics.get("model") or "",
            "test_run": cfg.get("test_run") or metrics.get("test_run") or "",
            "train_runs": cfg.get("train_runs") or [],
            "epochs_configured": cfg.get("epochs"),
            "epochs_run": len(history),
            "batch": cfg.get("batch"),
            "lr": cfg.get("lr"),
            "config": cfg,
            "metrics": metrics,
            "complete": bool(metrics),
            "checkpoint": os.path.relpath(best, root) if os.path.exists(best) else None,
            "checkpoint_size": _stat(best)["size"] if os.path.exists(best) else 0,
            "mtime": _stat(best if os.path.exists(best) else full)["mtime"],
            "history": history,
            "exported": os.path.isdir(os.path.join(root, DIRS["exported"], name)),
        })
    out.sort(key=lambda r: (r["model"], r["test_run"]))
    return out


def checkpoints(root: str) -> list[dict]:
    """Every .pt the run pickers can offer, best.pt first.

    last.pt is listed but flagged unusable: it is the resume point, carrying
    optimizer/scheduler state but none of the config, generator settings or
    threshold every consumer of this list needs, so loading one raises
    KeyError. Showing it greyed out beats hiding it — the file is on disk and
    people go looking for it.
    """
    out = []
    base = os.path.join(root, DIRS["runs"])
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        for ck in ("best.pt", "last.pt"):
            full = os.path.join(base, name, ck)
            if os.path.exists(full):
                st = _stat(full)
                out.append({
                    "name": f"{name}/{ck}",
                    "path": os.path.relpath(full, root),
                    "size": st["size"],
                    "mtime": st["mtime"],
                    "disabled": ck == "last.pt",
                    "note": "resume point, not loadable" if ck == "last.pt" else "",
                })
    out.sort(key=lambda c: (not c["name"].endswith("best.pt"), c["name"]))
    return out


def configs(root: str) -> list[dict]:
    out = []
    for name in sorted(os.listdir(root)):
        if name.endswith((".yaml", ".yml")):
            out.append({"name": name, "path": name, **_stat(os.path.join(root, name))})
    return out


def cache_runs(root: str) -> list[dict]:
    """Run ids present in training/cache — the valid --test-run values."""
    meta = _read_json(os.path.join(root, DIRS["cache"], "meta.json")) or {}
    return [{"name": r, "path": r} for r in sorted((meta.get("runs") or {}).keys())]


def results_summary(root: str) -> dict:
    return _read_json(os.path.join(root, DIRS["results"], "summary.json")) or {}


def result_figures(root: str) -> list[dict]:
    base = os.path.join(root, DIRS["results"])
    if not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        if name.endswith(".png"):
            out.append({"name": name, "path": os.path.relpath(
                os.path.join(base, name), root), **_stat(os.path.join(base, name))})
    return out


# --------------------------------------------------------------------------- #
# deep info (on demand, memoized)
# --------------------------------------------------------------------------- #
def recording_info(root: str, rel_path: str) -> dict:
    """Open a recording's index: format, duration, topics, points/frame.

    Memoized on (path, mtime, size) so expanding the same row twice is free and
    a re-recorded file is re-read.
    """
    full = os.path.normpath(os.path.join(root, rel_path))
    if not full.startswith(os.path.normpath(root) + os.sep):
        raise ValueError("path escapes the project root")
    if not os.path.exists(full):
        raise FileNotFoundError(rel_path)

    st = _stat(full)
    key = (full, st["mtime"], st["size"])
    with _info_lock:
        hit = _info_cache.get(key)
    if hit is not None:
        return hit

    info = _probe_recording(full)
    with _info_lock:
        _info_cache[key] = info
        if len(_info_cache) > 200:
            _info_cache.pop(next(iter(_info_cache)))
    return info


def _probe_recording(full: str) -> dict:
    from rocklabel.mcap_io import read_info

    out: dict = {"path": full, "ok": False, "error": "", "topics": [],
                 "format": "unknown", "duration_s": 0.0, "messages": 0}
    try:
        info = read_info(full)
    except Exception as e:  # a truncated file is a normal thing to find here
        out["error"] = f"{type(e).__name__}: {e}"
        out["hint"] = ("Unreadable index — this is what Trim salvages. Run Trim "
                       "with 'Keep every topic' to recover it up to the "
                       "corruption point.")
        return out

    out["ok"] = True
    out["format"] = "native lidarrig" if info.is_lidarrig else "ROS 2 bag"
    out["duration_s"] = round(info.duration_s, 2)
    out["start"] = _iso(info.start_time_ns / 1e9)
    out["end"] = _iso(info.end_time_ns / 1e9)
    for topic, (schema, count) in sorted(info.topics.items()):
        out["topics"].append({"topic": topic, "schema": schema, "messages": count})
        out["messages"] += count
    out["pointcloud_topics"] = info.pointcloud_topics()
    out["lidarrig_topics"] = info.lidarrig_topics()

    if info.is_lidarrig:
        out.update(_probe_lidarrig(full, info))
    return out


def _probe_lidarrig(full: str, info) -> dict:
    """Sample the first frames of a native recording for a density read-out."""
    import numpy as np

    from rocklabel import lidarrig_io

    topic = info.lidarrig_topics()[0]
    n = pts = 0
    has_inten = has_pose = False
    peak = 0.0
    try:
        for frame in lidarrig_io.iter_frames(full, topic):
            pts += len(frame.points)
            has_pose = has_pose or frame.has_pose
            if frame.intensity is not None and len(frame.intensity):
                has_inten = True
                finite = frame.intensity[np.isfinite(frame.intensity)]
                if len(finite):
                    peak = max(peak, float(finite.max()))
            n += 1
            if n >= 50:
                break
    except Exception as e:
        return {"probe_error": f"{type(e).__name__}: {e}"}
    return {
        "frames_probed": n,
        "points_per_frame": pts // max(n, 1),
        "reflectivity": has_inten,
        "reflectivity_peak": round(peak, 1),
        "poses": has_pose,
        "pose_hint": "" if has_pose else "No poses — SLAM never locked on this run.",
    }


# --------------------------------------------------------------------------- #
# aggregate snapshot
# --------------------------------------------------------------------------- #
def snapshot(root: str) -> dict:
    """Everything the dashboard needs for a full render, from stat() only."""
    recs = recordings(root)
    labs = labels(root)
    dss = datasets(root)
    rns = runs(root)
    complete = [r for r in rns if r["complete"]]
    best = max(complete, key=lambda r: r["metrics"].get("f1", 0.0), default=None)
    return {
        "recordings": recs,
        "labels": labs,
        "datasets": dss,
        "runs": rns,
        "checkpoints": checkpoints(root),
        "configs": configs(root),
        "cache_runs": cache_runs(root),
        "figures": result_figures(root),
        "summary": results_summary(root),
        "totals": {
            "recordings": len(recs),
            "recordings_bytes": sum(r["size"] for r in recs),
            "recordings_labeled": sum(1 for r in recs if r["labels"]),
            "labels": len(labs),
            "rocks": sum(r["rock_count"] for r in labs),
            "datasets": len(dss),
            "dataset_runs": sum(len(d["runs"]) for d in dss),
            "samples": sum(d["samples"] for d in dss),
            "rock_samples": sum(d["rock_samples"] for d in dss),
            "datasets_bytes": sum(d["size"] for d in dss),
            "runs": len(rns),
            "runs_complete": len(complete),
            "best_run": best["name"] if best else "",
            "best_f1": round(best["metrics"].get("f1", 0.0), 4) if best else 0.0,
            "best_model": best["model"] if best else "",
        },
    }
