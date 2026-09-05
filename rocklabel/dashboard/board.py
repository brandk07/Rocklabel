"""The training-runs board: every trained setting, ranked and annotated.

One row per *arm* — a named setting trained across every leave-one-run-out
fold of one suite. The board exists because the raw material is scattered:
per-fold numbers land in ``training/experiments/<suite>/<arm>/<fold>/``,
the aggregates in ``training/reports/<suite>/summary.json`` (written whenever
a report is rebuilt), and nothing anywhere answers "which run is best and
why". The old flat ``compare`` experiments are not part of this — they were
trained against a different population and only mislead next to these.

Reading order per fold, deliberately:

* **The fold directory on disk is the source of truth for what has finished.**
  A report can be older than the newest folds of a running sweep, and a board
  that trusted it would show finished work as missing.
* **The report is the source of truth for the rock-share-normalized score** —
  it stores each fold's rock share alongside its PR-AUC, which a bare
  ``test_metrics.json`` does not. Folds the report does not know about fall
  back to the rock share implied by the suite's own cache, and get no
  normalized score when neither has it.

Notes are the one thing this module writes: a JSON file under the project's
``.dashboard/`` folder mapping a row key to free text. Rows without a user
note fall back to the catalog — the measured conclusions from the
sweeps already run, so the board says something useful before anything has
been typed.

Scores are never compared across suites in aggregate here: different suites
trained on different caches against different fold sets, and this project has
twice been misled by reading those numbers side by side. Ranking happens
inside a suite; the caller decides what to do with two suites' numbers.
"""

from __future__ import annotations

import json
import os

from ..train import catalog
from .inventory import FLAT_EXPERIMENTS, _read_json, _stat

#: Where arm notes live. Inside ``.dashboard/`` beside the job logs: it is
#: dashboard state about the project, not project data, and nothing else
#: reads it.
NOTES_PATH = os.path.join(".dashboard", "run-notes.json")

#: Conclusions already paid for, keyed by ``<suite>/<arm>``. Shown until a
#: note is typed over them (clearing a user note simply falls back here).
#: Sourced from handoff/03-training-RESULT.md and the finished sweeps' reports.
def seed_note(key: str) -> str:
    """The measured conclusion for ``<suite>/<arm>``, or "" if there is none.

    These used to be a dict here. They now come from
    :mod:`rocklabel.train.catalog`, which the live viewer's checkpoint picker
    reads too — a conclusion written in two places is a conclusion that drifts,
    and this one had already started to (the board still called a segmenter
    "THE ARM TO EXPORT FROM" after it was measured finding 7 rocks in 54).
    """
    suite, _, arm = key.partition("/")
    return catalog.note(suite, arm or None)


def has_seed_note(key: str) -> bool:
    return bool(seed_note(key))


# --------------------------------------------------------------------------- #
# notes
# --------------------------------------------------------------------------- #
def _notes_file(root: str) -> str:
    return os.path.join(root, NOTES_PATH)


def read_notes(root: str) -> dict:
    return _read_json(_notes_file(root)) or {}


def write_note(root: str, key: str, note: str) -> dict:
    """Set one arm's note (empty string falls back to the seeded conclusion)."""
    if not isinstance(key, str) or "/" not in key:
        raise ValueError("a note key looks like <suite>/<arm>")
    path = _notes_file(root)
    notes = {}
    if os.path.exists(path):
        notes = _read_json(path) or {}
    text = (note or "").strip()
    if text:
        notes[key] = text
    else:
        # A blanked note reverts to the seed, so drop the override entirely.
        notes.pop(key, None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(notes, fh, indent=2)
    os.replace(tmp, path)
    return {"key": key, "note": text or seed_note(key),
            "source": "custom" if text
            else ("seed" if has_seed_note(key) else "")}


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    return round((sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5, 4)


def _cache_prevalences(root: str, cache_name: str) -> dict:
    """Rock share per run from a cache's meta.json: rock samples / all samples."""
    if not cache_name:
        return {}
    meta = _read_json(os.path.join(root, "training", "caches", cache_name,
                                   "meta.json")) or {}
    out = {}
    for run, r in (meta.get("runs") or {}).items():
        n = r.get("n") or 0
        rock = r.get("rock") or 0
        if n > 0:
            out[run] = rock / n
    return out


def _normalize(ap: float | None, prev: float | None) -> float | None:
    """(AP - prevalence) / (1 - prevalence): guessing scores 0, perfect 1."""
    if ap is None or prev is None or prev >= 1.0:
        return None
    return (ap - prev) / (1.0 - prev)


def _model_kind(model: str) -> str:
    return "segmenter" if "seg" in (model or "") else "classifier"


def _disk_folds(adir: str) -> list[dict]:
    """Per-fold metrics straight from the fold directories, freshest truth."""
    out = []
    if not os.path.isdir(adir):
        return out
    for name in sorted(os.listdir(adir)):
        fdir = os.path.join(adir, name)
        if not os.path.isdir(fdir):
            continue
        m = _read_json(os.path.join(fdir, "test_metrics.json"))
        if not m:
            continue
        out.append({
            "fold": m.get("test_run") or name,
            "path": fdir,
            "pr_auc": m.get("pr_auc"),
            # Newer folds carry their own normalized score and rock share,
            # computed by the training loop against exactly the population it
            # graded — always preferable to reconstructing them.
            "norm_pr_auc": m.get("norm_pr_auc"),
            "f1": m.get("f1"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "prevalence": m.get("rock_frac"),
        })
    return out


# --------------------------------------------------------------------------- #
# one row per arm
# --------------------------------------------------------------------------- #
def _finish_row(row: dict) -> dict:
    """Aggregate the per-fold list into the row's summary fields."""
    folds = row.pop("_per_fold")
    scored = [p["pr_auc"] for p in folds if p["pr_auc"] is not None]
    norms = [p["norm_pr_auc"] for p in folds if p["norm_pr_auc"] is not None]
    f1s = [p["f1"] for p in folds if p.get("f1") is not None]
    precisions = [p["precision"] for p in folds if p.get("precision") is not None]
    recalls = [p["recall"] for p in folds if p.get("recall") is not None]
    best = max((p for p in folds if p["pr_auc"] is not None),
               key=lambda p: p["pr_auc"], default=None)
    worst = min((p for p in folds if p["pr_auc"] is not None),
                key=lambda p: p["pr_auc"], default=None)
    row.update({
        "pr_auc": _mean(scored),
        "pr_auc_std": _std(scored),
        "norm_pr_auc": _mean(norms),
        "f1": _mean(f1s),
        "precision": _mean(precisions),
        "recall": _mean(recalls),
        "best_fold": best["fold"] if best else None,
        "best_fold_pr_auc": best["pr_auc"] if best else None,
        "worst_fold": worst["fold"] if worst else None,
        "worst_fold_pr_auc": worst["pr_auc"] if worst else None,
        "per_fold": sorted(folds, key=lambda p: p["fold"]),
    })
    return row


def _suite_row(suite: str, sdir: str, arm_name: str, report_arm: dict | None,
               spec_arm, suite_title: str, folds_total: int,
               cache_prev: dict) -> dict:
    """One arm's row: disk folds first, the report layered over them."""
    adir = os.path.join(sdir, arm_name)
    disk = _disk_folds(adir)
    arm_meta = (_read_json(os.path.join(adir, "arm.json"))
                if os.path.isdir(adir) else None) or {}
    # One representative fold pins the ordinary training parameters that are
    # shared by the whole arm.  The held-out/train run names vary per fold and
    # deliberately stay in the per-fold list instead.
    run_config = {}
    for disk_fold in disk:
        run_config = _read_json(os.path.join(disk_fold["path"], "config.json")) or {}
        if run_config:
            break

    # Fold order follows the report when there is one (it knows the plan),
    # then anything on disk it does not know about yet.
    order: list[str] = []
    if report_arm:
        order += [f for f in (report_arm.get("folds") or []) if f not in order]
    order += [p["fold"] for p in disk if p["fold"] not in order]

    by_fold = {p["fold"]: p for p in disk}
    rep_prev = (report_arm or {}).get("prevalence") or {}
    rep_norm = ((report_arm or {}).get("norm_pr_auc") or {}).get("per_fold") or {}
    rep_f1 = ((report_arm or {}).get("f1") or {}).get("per_fold") or {}
    rep_precision = ((report_arm or {}).get("precision") or {}).get("per_fold") or {}
    rep_recall = ((report_arm or {}).get("recall") or {}).get("per_fold") or {}

    per_fold = []
    for fold in order:
        d = by_fold.get(fold) or {"fold": fold, "pr_auc": None, "f1": None}
        # Precedence: what the training loop itself recorded (it graded the
        # exact population), then the report, then the cache's rock share.
        prev = d.get("prevalence") or rep_prev.get(fold) or cache_prev.get(fold)
        norm = d.get("norm_pr_auc") or rep_norm.get(fold)
        if norm is None:
            norm = _normalize(d.get("pr_auc"), prev)
        per_fold.append({
            "fold": fold,
            "pr_auc": d.get("pr_auc"),
            "norm_pr_auc": round(norm, 4) if isinstance(norm, float) else norm,
            "f1": d.get("f1") if d.get("f1") is not None else rep_f1.get(fold),
            "precision": (d.get("precision") if d.get("precision") is not None
                          else rep_precision.get(fold)),
            "recall": (d.get("recall") if d.get("recall") is not None
                       else rep_recall.get(fold)),
            "prevalence": prev,
            "on_disk": fold in by_fold,
        })

    model = ((spec_arm.model if spec_arm else None) or arm_meta.get("model")
             or run_config.get("model")
             or (report_arm or {}).get("model") or "")
    label = ((spec_arm.label if spec_arm else None) or arm_meta.get("label")
             or (report_arm or {}).get("label") or arm_name)
    what = (spec_arm.what if spec_arm else "")
    features = (list(spec_arm.features) if spec_arm
                else (arm_meta.get("features")
                      or run_config.get("features")
                      or (report_arm or {}).get("features") or []))

    newest = 0.0
    for p in by_fold.values():
        newest = max(newest, _stat(p["path"])["mtime"])

    seed = spec_arm.seed if spec_arm and spec_arm.seed is not None else arm_meta.get("seed")
    if seed is None:
        seed = (report_arm or {}).get("seed")
    if seed is None:
        seed = run_config.get("seed")
    training_keys = ("epochs", "batch", "lr", "weight_decay", "patience",
                     "augment", "gap_frames", "gap_seconds", "tnet", "dropout")
    training = {key: run_config[key] for key in training_keys if key in run_config}
    if run_config.get("train_runs") is not None:
        training["training_runs"] = len(run_config["train_runs"])
    total = max(folds_total or 0, len(order))

    return _finish_row({
        "key": f"{suite}/{arm_name}",
        "suite": suite,
        "suite_title": suite_title,
        "arm": arm_name,
        "label": label,
        "what": what or (report_arm or {}).get("what", ""),
        "model": model,
        "kind": _model_kind(model),
        "features": features,
        "overrides": (spec_arm.overrides if spec_arm
                      else (report_arm or {}).get("overrides") or {}),
        "seed": seed,
        "training": training,
        "folds_done": len(by_fold),
        "folds_total": total,
        "reported": bool(report_arm),
        "last_activity": newest or None,
        "completed_at": newest if newest and total and len(by_fold) >= total else None,
        "_per_fold": per_fold,
    })


def _suite_activity(rows: list[dict]) -> float | None:
    """The newest fold activity in a suite, for ordering suites newest-first."""
    stamps = [r["last_activity"] for r in rows if r.get("last_activity")]
    return max(stamps) if stamps else None


# --------------------------------------------------------------------------- #
# the board
# --------------------------------------------------------------------------- #
def board(root: str) -> dict:
    """Every trained arm as one list, plus the suites they belong to.

    Rank within a suite — the rows carry no global rank on purpose. Suites
    train on different caches against different fold sets, and a number from
    one suite next to a number from another is exactly the comparison this
    project has twice been burned by.
    """
    from ..train.ablate import SUITES  # torch-free

    base = os.path.join(root, "training", "experiments")
    reports = os.path.join(root, "training", "reports")
    rows: list[dict] = []
    suites_out = []
    if not os.path.isdir(base):
        return {"rows": [], "suites": [], "noise_floor": {}}

    noise_floors = {}
    for suite in sorted(os.listdir(base)):
        sdir = os.path.join(base, suite)
        if not os.path.isdir(sdir):
            continue
        if suite in FLAT_EXPERIMENTS:
            continue

        declared = SUITES.get(suite) or {}
        summary = _read_json(os.path.join(reports, suite, "summary.json")) or {}
        by_arm = {a["arm"]: a for a in summary.get("arms", [])}
        noise_floors[suite] = summary.get("noise_floor_pr_auc")
        cache_prev = _cache_prevalences(root, declared.get("cache", ""))
        folds_total = len(summary.get("folds") or []) or len(cache_prev)

        names = set(by_arm) | {
            n for n in os.listdir(sdir) if os.path.isdir(os.path.join(sdir, n))
        }
        # Declared-but-unstarted arms stay listed: the matrix is the plan.
        names |= {a.name for a in declared.get("arms", [])}

        suite_rows = []
        for arm_name in sorted(names):
            if ".superseded-" in arm_name:
                continue
            spec_arm = next((a for a in declared.get("arms", [])
                             if a.name == arm_name), None)
            suite_rows.append(_suite_row(
                suite, sdir, arm_name, by_arm.get(arm_name), spec_arm,
                summary.get("title", ""), folds_total, cache_prev))

        # Rank inside the suite: normalized when the folds' rock shares are
        # known, raw otherwise — never the two mixed in one ordering.
        for i, row in enumerate(sorted(
                suite_rows,
                key=lambda r: r["norm_pr_auc"] if r["norm_pr_auc"] is not None
                else r["pr_auc"] if r["pr_auc"] is not None else -1.0,
                reverse=True), 1):
            row["rank"] = i
        rows += suite_rows

        suites_out.append({
            "name": suite,
            "title": summary.get("title") or declared.get("title") or suite,
            "blurb": summary.get("blurb") or declared.get("blurb", ""),
            "cache": declared.get("cache", ""),
            "arms": len(suite_rows),
            "folds": folds_total or None,
            # When this suite last produced a fold, so the page can lead with
            # the work that is current rather than the alphabet.
            "last_activity": _suite_activity(suite_rows),
            "noise_floor": summary.get("noise_floor_pr_auc"),
            "contrasts": [
                {"baseline": c.get("baseline"), "variant": c.get("variant"),
                 "question": c.get("question", ""),
                 "n_folds": c.get("n_folds"),
                 "metric": c.get("norm_pr_auc") or c.get("pr_auc") or {},
                 "using_norm": bool(c.get("norm_pr_auc"))}
                for c in summary.get("contrasts", [])
            ],
        })

    # Notes: user-typed ones win, then the measured conclusions.
    notes = read_notes(root)
    for row in rows:
        # The catalog's plain-English name and status ride along with every
        # row, so the board can say "settled" or "dead end" beside a setting
        # instead of leaving a reader to work it out from the score.
        suite, _, arm = row["key"].partition("/")
        row["title"], row["status"], _seed = catalog.entry(suite, arm or None)
        user = notes.get(row["key"])
        if user:
            row["note"], row["note_source"] = user, "custom"
        elif has_seed_note(row["key"]):
            row["note"], row["note_source"] = seed_note(row["key"]), "seed"
        else:
            row["note"], row["note_source"] = "", ""

    return {"rows": rows, "suites": suites_out,
            "noise_floor": noise_floors}
