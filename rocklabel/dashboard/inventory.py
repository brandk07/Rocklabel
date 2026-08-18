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

#: The project layout, one entry per folder anything here reads.
#:
#: Every derived thing sits under a root that says what produced it:
#: ``datasets/<profile>/`` says how frames were cut, ``training/caches/<profile>/``
#: says which datasets were pooled, ``training/experiments/<experiment>/<arm>/<fold>/``
#: says which run trained a checkpoint, and ``training/reports/<experiment>/``
#: holds what that experiment concluded.
DIRS = {
    "recordings": "recordings",
    "labels": "labels",
    "datasets": "datasets",
    "training": "training",
    "experiments": os.path.join("training", "experiments"),
    "caches": os.path.join("training", "caches"),
    "reports": os.path.join("training", "reports"),
    "exported": os.path.join("training", "exported"),
}

#: Experiment folders holding flat ``<model>_loro_<run>`` directories rather
#: than the ``<arm>/<fold>`` matrix a sweep writes. They are `compare` output,
#: not ablation suites, and the suite progress view must not count them.
FLAT_EXPERIMENTS = ("compare", "compare-fused")

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
                # Folder under recordings/ — "volleyball/reslam", "archive/misc".
                # This is what separates the live project from four years of
                # unrelated captures in every picker and list.
                "group": _group_of(full, os.path.join(root, DIRS["recordings"])),
                "size": st["size"],
                "mtime": st["mtime"],
                "modified": _iso(st["mtime"]),
                "labels": _labels_path_for(root, name[:-5]),
            })
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def _group_of(full: str, base: str) -> str:
    """The folder a file sits in, relative to its root ("volleyball/reslam")."""
    rel = os.path.relpath(os.path.dirname(full), base)
    return "" if rel == "." else rel.replace(os.sep, "/")


def _labels_path_for(root: str, stem: str) -> str | None:
    """This recording's label file, wherever under ``labels/`` it sits.

    Searched rather than joined: labels are foldered by project now, so the
    file for a volleyball recording is at ``labels/volleyball/<stem>.labels.json``
    and a flat join would report every recording in the project as unlabelled.
    """
    want = f"{stem}.labels.json"
    for folder in (DIRS["labels"], DIRS["recordings"]):
        base = os.path.join(root, folder)
        for dirpath, _dirs, names in os.walk(base):
            if want in names:
                return os.path.relpath(os.path.join(dirpath, want), root)
    return None


def labels(root: str) -> list[dict]:
    out = []
    seen = set()
    for folder in (DIRS["labels"], DIRS["recordings"]):
        base = os.path.join(root, folder)
        if not os.path.isdir(base):
            continue
        found = []
        for dirpath, _dirs, names in os.walk(base):
            found += [os.path.join(dirpath, n) for n in names
                      if n.endswith(".labels.json")]
        for full in sorted(found):
            name = os.path.basename(full)
            if name in seen:
                continue
            seen.add(name)
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
                "group": _group_of(full, base),
                "run_id": data.get("run_id", name.replace(".labels.json", "")),
                "mcap_file": data.get("mcap_file", ""),
                "rock_count": len(rocks),
                "shapes": shapes,
                "rock_ids": [r.get("id") for r in rocks][:64],
                "created": data.get("created", ""),
                # The two halves of "which volume is eligible to train": the
                # arena footprint bounds the floor plan, z_band bounds the
                # height. Both are optional, and a run missing them trains on
                # whatever the generator's crop box swept up.
                "arena_vertices": len((data.get("arena") or {}).get("vertices") or []),
                "z_band": data.get("z_band"),
                "intensity": bool(data.get("intensity_available")),
                "schema_version": data.get("schema_version"),
                "size": st["size"],
                "mtime": st["mtime"],
                "modified": _iso(st["mtime"]),
            })
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def _dataset_dirs(base: str) -> list[str]:
    """Every generated dataset under ``datasets/``, at any depth.

    Datasets live one folder per generation profile now
    (``datasets/full-sweep/volleyball``), so a listdir of the top level would
    return the profile names and nothing else. A manifest.json is what makes a
    folder a dataset, so that is what this looks for.
    """
    out = []
    for dirpath, dirs, names in os.walk(base):
        if "manifest.json" in names:
            out.append(dirpath)
            dirs[:] = []           # never descend into points/ bev/ seg/
        else:
            dirs[:] = [d for d in sorted(dirs) if d not in ("points", "bev", "seg")]
    return sorted(out)


def datasets(root: str) -> list[dict]:
    from ..profiles import identify as identify_profile

    base = os.path.join(root, DIRS["datasets"])
    out = []
    if not os.path.isdir(base):
        return out
    for full in _dataset_dirs(base):
        rel = os.path.relpath(full, base)
        name = rel.replace(os.sep, "/")
        manifest = _read_json(os.path.join(full, "manifest.json"))
        size, files = _dir_size(full)
        entry = {
            "name": name,
            "path": os.path.relpath(full, root),
            # How this dataset's frames were cut. Named in the manifest for
            # anything generated since profiles existed; recovered from the
            # config fingerprint for everything older, so an old dataset still
            # says on its face what it is rather than showing up blank.
            "profile": "",
            "group": rel.split(os.sep)[0] if os.sep in rel else "",
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
            entry["profile"] = (manifest.get("profile")
                                or identify_profile(manifest.get("config_hash", ""))
                                or "")
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
    after it are one run: :func:`rocklabel.gui.labeler.default_labels_path` derives
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

    # A dataset is any folder under datasets/ carrying a manifest.json. It used
    # to have to be a direct child, which stopped being true when datasets moved
    # under datasets/<profile>/ — the browser could then name a folder the
    # gate refused to touch.
    if _inside(full, ds_base) and full != ds_base:
        if not os.path.isdir(full):
            raise FileNotFoundError(rel_path)
        if not os.path.exists(os.path.join(full, "manifest.json")):
            raise ValueError("only a generated dataset folder (one holding a "
                             "manifest.json) can be renamed or deleted")
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
    """The flat ``<model>_loro_<run>`` folds that `train` and `compare` write."""
    out = []
    bases = [os.path.join(root, DIRS["experiments"], e) for e in FLAT_EXPERIMENTS]
    for base in bases:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            entry = _run_entry(root, base, name)
            if entry:
                out.append(entry)
    out.sort(key=lambda r: (r["experiment"], r["model"], r["test_run"]))
    return out


def _run_entry(root: str, base: str, name: str) -> dict | None:
    """One flat ``<model>_loro_<run>`` fold directory, or None if it is not one."""
    full = os.path.join(base, name)
    if not os.path.isdir(full):
        return None
    cfg = _read_json(os.path.join(full, "config.json")) or {}
    metrics = _read_json(os.path.join(full, "test_metrics.json")) or {}
    best = os.path.join(full, "best.pt")
    history = _history(os.path.join(full, "history.csv"))
    return {
        "name": name,
        "path": os.path.relpath(full, root),
        "experiment": os.path.basename(base),
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
    }


#: How a fold directory names the recording it held out.
_FOLD_PREFIX = "loro_"


def _fold_of(run_dir_name: str) -> str:
    """The held-out recording a fold directory is named after.

    Handles both spellings in the project: a sweep writes ``loro_<run>`` inside
    an arm folder, while `compare` writes ``<model>_loro_<run>`` flat.
    """
    if _FOLD_PREFIX in run_dir_name:
        return run_dir_name.split(_FOLD_PREFIX, 1)[1]
    return run_dir_name


def _checkpoint_dirs(base: str) -> list[str]:
    """Every directory under ``base`` holding a checkpoint, at any depth."""
    out = []
    for dirpath, dirs, names in os.walk(base):
        if "best.pt" in names or "last.pt" in names:
            out.append(dirpath)
            dirs[:] = []
        else:
            dirs[:] = sorted(dirs)
    return sorted(out)


def checkpoints(root: str) -> list[dict]:
    """Every .pt the run pickers can offer, grouped by the run that made it.

    This used to be a flat alphabetical list of every checkpoint in the project
    — 354 entries with nothing but a path to tell them apart, which is not a
    picker anyone can choose from. So each entry now carries the three things
    you actually pick on: which experiment and arm produced it, which recording
    it was tested against, and what it scored. Entries are sorted best-first
    inside their group, and the top scorer of each experiment is flagged so the
    browser can offer "just give me the good one" without scanning the list.

    ``last.pt`` is listed but flagged unusable: it is the resume point, carrying
    optimizer/scheduler state but none of the config, generator settings or
    threshold every consumer of this list needs, so loading one raises
    KeyError. Showing it greyed out beats hiding it — the file is on disk and
    people go looking for it.

    ``.superseded-*`` directories are marked ``archived`` rather than dropped:
    they are old settings kept for the record, and the browser hides them
    behind a toggle instead of the list pretending they do not exist.
    """
    base = os.path.join(root, DIRS["experiments"])
    out: list[dict] = []
    if not os.path.isdir(base):
        return out

    for run_dir in _checkpoint_dirs(base):
        parts = os.path.relpath(run_dir, base).split(os.sep)
        experiment = parts[0]
        # <experiment>/<arm>/<fold> for a sweep; <experiment>/<model>_loro_<run>
        # for a compare run, whose arm is the model named in its own config.
        if len(parts) >= 3:
            arm, fold_dir = parts[1], parts[-1]
        else:
            fold_dir = parts[-1]
            arm = fold_dir.split(_FOLD_PREFIX, 1)[0].rstrip("_") or "run"
        archived = any(".superseded-" in p for p in parts)
        metrics = _read_json(os.path.join(run_dir, "test_metrics.json")) or {}
        pr_auc = metrics.get("pr_auc")
        fold = metrics.get("test_run") or _fold_of(fold_dir)

        for ck in ("best.pt", "last.pt"):
            full = os.path.join(run_dir, ck)
            if not os.path.exists(full):
                continue
            st = _stat(full)
            score = f" · PR-AUC {pr_auc:.3f}" if pr_auc is not None else " · not yet scored"
            out.append({
                # What the dropdown shows inside its group: the recording this
                # model has never seen, and how well it did on it.
                "name": f"held out {fold}{score}",
                "path": os.path.relpath(full, root),
                "group": f"{experiment} · {arm}" + (" · archived" if archived else ""),
                "experiment": experiment,
                "arm": arm,
                "fold": fold,
                "pr_auc": pr_auc,
                "archived": archived,
                "size": st["size"],
                "mtime": st["mtime"],
                "modified": _iso(st["mtime"]),
                "best_of_experiment": False,
                "disabled": ck == "last.pt",
                "note": "resume point, not loadable" if ck == "last.pt" else "",
            })

    # Best-first inside a group; groups in name order, live ones before archives.
    out.sort(key=lambda c: (
        c["archived"], c["group"], c["disabled"],
        -(c["pr_auc"] if c["pr_auc"] is not None else -1.0), c["fold"],
    ))
    # The shortcut: for each experiment, the single loadable checkpoint that
    # scored highest. Usually the only one anybody wants.
    for experiment in {c["experiment"] for c in out}:
        pool = [c for c in out if c["experiment"] == experiment
                and not c["disabled"] and not c["archived"]
                and c["pr_auc"] is not None]
        if pool:
            max(pool, key=lambda c: c["pr_auc"])["best_of_experiment"] = True
    return out


def _dirs_at_depth(base: str, depth: int) -> list[str]:
    """Directories exactly ``depth`` levels below ``base``."""
    level = [base]
    for _ in range(depth):
        nxt = []
        for d in level:
            if os.path.isdir(d):
                nxt += [os.path.join(d, n) for n in sorted(os.listdir(d))
                        if os.path.isdir(os.path.join(d, n))]
        level = nxt
    return level


def _arm_progress(arm_dir: str) -> tuple[dict, int, list[float]]:
    """(arm metadata, folds finished, their PR-AUCs) for one arm directory."""
    meta, done, scores = {}, 0, []
    if not os.path.isdir(arm_dir):
        return meta, done, scores
    for fold in sorted(os.listdir(arm_dir)):
        fdir = os.path.join(arm_dir, fold)
        if not os.path.isdir(fdir):
            continue
        meta = meta or (_read_json(os.path.join(fdir, "arm.json")) or {})
        m = _read_json(os.path.join(fdir, "test_metrics.json"))
        if m:
            done += 1
            scores.append(m.get("pr_auc", 0.0))
    return meta, done, scores


def ablations(root: str) -> list[dict]:
    """Progress of every ablation suite under training/ablate/.

    A suite is a matrix (settings x folds), so the useful summary is how much
    of the matrix is filled in and how the settings currently rank - not a flat
    list of 121 run directories, which is what the runs table would become if
    these were folded into it.

    The denominator comes from the suite *definition* and the cache, not from
    what happens to be on disk. Counting only started arms would report the
    first hour of an overnight sweep as "2 of 3 done", which is exactly
    backwards from what someone watching it needs to know.
    """
    from ..train.ablate import SUITES  # torch-free

    base = os.path.join(root, DIRS["experiments"])
    if not os.path.isdir(base):
        return []
    n_folds = len(cache_runs(root))
    out = []
    for suite in sorted(os.listdir(base)):
        sdir = os.path.join(base, suite)
        # compare/ holds flat fold directories, not an arm x fold matrix; the
        # runs table already describes those.
        if not os.path.isdir(sdir) or suite in FLAT_EXPERIMENTS:
            continue
        declared = SUITES.get(suite)
        # Names from the definition where there is one, so arms that have not
        # started yet still appear; otherwise fall back to the directories.
        names = ([a.name for a in declared["arms"]] if declared
                 else sorted(n for n in os.listdir(sdir)
                             if os.path.isdir(os.path.join(sdir, n))))
        by_name = {a.name: a for a in declared["arms"]} if declared else {}
        arms, folds_seen = [], set()
        for name in names:
            adir = os.path.join(sdir, name)
            meta, done, scores = _arm_progress(adir)
            spec_arm = by_name.get(name)
            if os.path.isdir(adir):
                folds_seen.update(f for f in os.listdir(adir)
                                  if os.path.isdir(os.path.join(adir, f)))
            arms.append({
                "name": name,
                "label": (spec_arm.label if spec_arm else None) or meta.get("label") or name,
                "model": (spec_arm.model if spec_arm else None) or meta.get("model") or "",
                "features": list(spec_arm.features) if spec_arm else (meta.get("features") or []),
                "folds_done": done,
                "pr_auc": round(sum(scores) / len(scores), 4) if scores else None,
            })
        summary = _read_json(
            os.path.join(root, DIRS["reports"], suite, "summary.json")) or {}
        folds = n_folds or len(folds_seen)
        out.append({
            "name": suite,
            "path": os.path.relpath(sdir, root),
            "title": (declared["title"] if declared else None) or summary.get("title") or suite,
            "arms": arms,
            "folds": folds,
            "runs_done": sum(a["folds_done"] for a in arms),
            "runs_total": len(arms) * max(folds, 1),
            "reported": bool(summary),
            "noise_floor": summary.get("noise_floor_pr_auc"),
        })
    return out


def configs(root: str) -> list[dict]:
    out = []
    for name in sorted(os.listdir(root)):
        if name.endswith((".yaml", ".yml")):
            out.append({"name": name, "path": name, **_stat(os.path.join(root, name))})
    return out


def profiles(root: str) -> list[dict]:
    """The generation profiles, with how many datasets each has produced."""
    from ..profiles import to_json

    built = {}
    for d in datasets(root):
        if d["profile"]:
            built[d["profile"]] = built.get(d["profile"], 0) + len(d["runs"])
    return [p | {"name": p["name"], "path": p["name"],
                 "runs_built": built.get(p["name"], 0)} for p in to_json()]


def default_cache_dir(root: str) -> str:
    """The cache the training commands read when nothing says otherwise."""
    from ..profiles import DEFAULT_PROFILE

    return os.path.join(root, DIRS["caches"], DEFAULT_PROFILE)


def cache_runs(root: str) -> list[dict]:
    """Run ids in the default cache — the valid --test-run values."""
    meta = _read_json(os.path.join(default_cache_dir(root), "meta.json")) or {}
    return [{"name": r, "path": r} for r in sorted((meta.get("runs") or {}).keys())]


def caches(root: str) -> list[dict]:
    """Every built cache under training/, not just the default one.

    There is more than one now: a cache is tied to the dataset config it was
    pooled from, so training on whole sensor sweeps means a second cache beside
    the batch-sized one rather than a rebuild of it. A picker that only ever
    offered training/cache would silently point a run at the wrong data.
    """
    from ..profiles import identify as identify_profile

    base = os.path.join(root, DIRS["caches"])
    if not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        meta = _read_json(os.path.join(d, "meta.json"))
        if not meta or "runs" not in meta:
            continue
        runs_meta = meta.get("runs") or {}
        rel = os.path.relpath(d, root)
        out.append({
            "name": rel, "path": rel,
            # Which way the frames in this cache were cut, and which datasets
            # were pooled to make it. Training refuses to mix profiles, so this
            # is the field that says whether a cache is the right one.
            "profile": (meta.get("profile")
                        or identify_profile(meta.get("config_hash", "")) or ""),
            "datasets": [os.path.basename(x) for x in (meta.get("datasets") or [])],
            "runs": sorted(runs_meta.keys()),
            "run_count": len(runs_meta),
            "samples": sum(r.get("n", 0) for r in runs_meta.values()),
            "rock_samples": sum(r.get("rock", 0) for r in runs_meta.values()),
            # 0 for a cache whose dataset predates the segmentation format, which
            # is exactly what says "this cache cannot train a segmenter".
            "seg_frames": sum(r.get("seg_frames", 0) or 0 for r in runs_meta.values()),
            **_stat(os.path.join(d, "meta.json")),
        })
    return out


def results_summary(root: str) -> dict:
    return _read_json(os.path.join(root, DIRS["reports"], "compare",
                                   "summary.json")) or {}


def _pngs_in(base: str, root: str, group: str, blurb: str) -> list[dict]:
    if not os.path.isdir(base):
        return []
    return [{"name": n, "group": group, "blurb": blurb,
             "path": os.path.relpath(os.path.join(base, n), root),
             **_stat(os.path.join(base, n))}
            for n in sorted(os.listdir(base)) if n.endswith(".png")]


#: Prose for the report folders that are not an ablation suite.
_REPORT_BLURBS = {
    "compare": ("Model comparison", "Written by Compare models / Regenerate report."),
    "reflect": ("Reflectivity check",
                "Written by Reflectivity check — measured off the cache, no training."),
}


def result_figures(root: str) -> list[dict]:
    """Every report figure on disk, tagged with which report wrote it.

    Several reports write figures, so a flat list would mix a reflectivity
    histogram in among the confusion matrices with nothing saying which command
    produced which. The ``group`` field is what the Overview page sections on.
    """
    base = os.path.join(root, DIRS["reports"])
    out: list[dict] = []
    if not os.path.isdir(base):
        return out
    for dirpath, dirs, _names in os.walk(base):
        dirs[:] = sorted(dirs)
        rel = os.path.relpath(dirpath, base)
        if rel == ".":
            continue
        top = rel.split(os.sep)[0]
        known = _REPORT_BLURBS.get(top)
        if known:
            title, blurb = known
            if rel != top:  # e.g. compare/<channel selection>
                title = f"{title} · {rel.split(os.sep, 1)[1]}"
        else:
            title = "Experiment · " + rel.replace(os.sep, " · ")
            blurb = f"Written by Ablation sweep --suite {top}."
        out += _pngs_in(dirpath, root, title, blurb)
    return out


# --------------------------------------------------------------------------- #
# what is training right now
# --------------------------------------------------------------------------- #
#: A fold whose history.csv has been touched inside this many seconds is
#: treated as in flight. Generous on purpose: an epoch on the segmenter takes
#: a couple of minutes, and a fold that has gone quiet for longer than this is
#: better reported as stalled than as running.
LIVE_WINDOW_S = 900.0


def _fold_state(run_dir: str, now: float) -> dict:
    """How far one fold got, read off the files training already writes.

    Nothing was added to the training loop for this. ``history.csv`` gains a
    row per epoch, ``test_metrics.json`` appears when the fold finishes, and
    the modification time of the first is what says whether anything is still
    happening.
    """
    hist = _history(os.path.join(run_dir, "history.csv"))
    metrics = _read_json(os.path.join(run_dir, "test_metrics.json"))
    cfg = _read_json(os.path.join(run_dir, "config.json")) or {}
    h_stat = _stat(os.path.join(run_dir, "history.csv"))
    age = now - h_stat["mtime"] if h_stat["mtime"] else None

    if metrics:
        status = "done"
    elif not hist:
        status = "pending"
    elif age is not None and age <= LIVE_WINDOW_S:
        status = "running"
    else:
        status = "stalled"

    # Best validation score so far, and how long the fold has been going.
    val = [r.get("val_pr_auc") for r in hist if isinstance(r.get("val_pr_auc"), float)]
    started = _stat(os.path.join(run_dir, "config.json"))["mtime"]
    return {
        "fold": metrics.get("test_run") if metrics else (cfg.get("test_run")
                or _fold_of(os.path.basename(run_dir))),
        "path": run_dir,
        "status": status,
        "epochs_done": len(hist),
        "epochs_planned": cfg.get("epochs"),
        "best_val_pr_auc": max(val) if val else None,
        "test_pr_auc": (metrics or {}).get("pr_auc"),
        # The per-epoch validation curve for whatever is in flight. Trimmed to
        # the last 200 points so a long fold cannot bloat every poll.
        "curve": [{"epoch": r.get("epoch"), "train_loss": r.get("train_loss"),
                   "val_loss": r.get("val_loss"), "val_pr_auc": r.get("val_pr_auc")}
                  for r in hist[-200:]],
        "seconds_since_epoch": round(age, 1) if age is not None else None,
        "elapsed_s": round(h_stat["mtime"] - started, 1) if started and h_stat["mtime"] else None,
        "mtime": h_stat["mtime"],
    }


def training_activity(root: str) -> list[dict]:
    """Which experiments are training, how far along, and what is left.

    Works for a sweep launched from a terminal as well as one launched from the
    dashboard, because it reads the run directories rather than a job record —
    and both long sweeps this project has run were started from a terminal.

    An experiment appears here when anything under it is running or stalled, or
    when it is unfinished; a finished one drops off, which is what keeps the
    panel about the present.
    """
    import time

    base = os.path.join(root, DIRS["experiments"])
    now = time.time()
    out = []
    if not os.path.isdir(base):
        return out

    for experiment in sorted(os.listdir(base)):
        edir = os.path.join(base, experiment)
        if not os.path.isdir(edir):
            continue
        arms: dict[str, list[dict]] = {}
        for run_dir in _checkpoint_dirs(edir) + _unstarted_dirs(edir):
            if ".superseded-" in run_dir:
                continue
            parts = os.path.relpath(run_dir, edir).split(os.sep)
            arm = parts[0] if len(parts) >= 2 else (
                parts[-1].split(_FOLD_PREFIX, 1)[0].rstrip("_") or "run")
            arms.setdefault(arm, []).append(_fold_state(run_dir, now))

        folds = [f for group in arms.values() for f in group]
        if not folds:
            continue
        counts = {k: sum(1 for f in folds if f["status"] == k)
                  for k in ("running", "stalled", "done", "pending")}
        # Folds the suite declares but that have not been started at all have
        # no directory to be read, so disk alone would report the first hour of
        # an overnight sweep as finished. The denominator comes from the suite
        # definition and the cache, the same way the ablation progress table
        # gets it.
        counts["pending"] += max(_declared_total(root, experiment) - len(folds), 0)
        # A finished experiment is not news; it belongs in the reports list.
        if counts["running"] == 0 and counts["stalled"] == 0 and counts["pending"] == 0:
            continue
        live = [f for f in folds if f["status"] == "running"]
        # Rough finish estimate from how long the finished folds actually took.
        done_times = [f["elapsed_s"] for f in folds
                      if f["status"] == "done" and f["elapsed_s"]]
        per_fold = sum(done_times) / len(done_times) if done_times else None
        remaining = counts["pending"] + counts["running"] + counts["stalled"]
        out.append({
            "experiment": experiment,
            "path": os.path.relpath(edir, root),
            "arms": sorted(arms),
            "folds_total": max(len(folds), _declared_total(root, experiment)),
            "counts": counts,
            "in_flight": sorted(live, key=lambda f: -f["mtime"]),
            "seconds_per_fold": round(per_fold, 1) if per_fold else None,
            "seconds_remaining": round(per_fold * remaining, 1) if per_fold else None,
            "folds": sorted(({k: v for k, v in f.items() if k != "curve"}
                             for f in folds), key=lambda f: f["fold"]),
            "updated": _iso(max((f["mtime"] for f in folds), default=0.0)),
        })
    return out


def _declared_total(root: str, experiment: str) -> int:
    """How many (arm, fold) runs this experiment is supposed to produce, or 0."""
    from ..train.ablate import SUITES  # torch-free

    declared = SUITES.get(experiment)
    if not declared:
        return 0
    return len(declared["arms"]) * len(cache_runs(root))


def _unstarted_dirs(base: str) -> list[str]:
    """Fold directories created but not yet holding a checkpoint.

    A fold writes config.json the moment it starts and best.pt only once an
    epoch improves, so without this a sweep's very first minutes would show up
    as nothing happening at all.
    """
    out = []
    for dirpath, dirs, names in os.walk(base):
        if "config.json" in names or "arm.json" in names:
            if "best.pt" not in names and "last.pt" not in names:
                out.append(dirpath)
            dirs[:] = []
        else:
            dirs[:] = sorted(dirs)
    return sorted(out)


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
    from rocklabel.recording.mcap_io import read_info

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
    abl = ablations(root)
    complete = [r for r in rns if r["complete"]]
    best = max(complete, key=lambda r: r["metrics"].get("f1", 0.0), default=None)
    return {
        "recordings": recs,
        "labels": labs,
        "datasets": dss,
        "runs": rns,
        "checkpoints": checkpoints(root),
        "configs": configs(root),
        "profiles": profiles(root),
        "cache_runs": cache_runs(root),
        "caches": caches(root),
        "training_now": training_activity(root),
        "figures": result_figures(root),
        "summary": results_summary(root),
        "ablations": abl,
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
            "ablation_runs_done": sum(s["runs_done"] for s in abl),
            "ablation_runs_total": sum(s["runs_total"] for s in abl),
            "best_run": best["name"] if best else "",
            "best_f1": round(best["metrics"].get("f1", 0.0), 4) if best else 0.0,
            "best_model": best["model"] if best else "",
        },
    }
