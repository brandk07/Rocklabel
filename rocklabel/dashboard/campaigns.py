"""Training campaigns — every sweep this project has run, in plain words.

The Training tab answers two questions and only two: **what is on the GPU right
now**, and **what has already been tried**. This module supplies the second one.

A *campaign* is one folder under ``training/experiments`` — one sweep that
occupied the GPU for hours or days. That is a deliberately coarser unit than
the Runs page, which goes one row per arm and quotes fold-level evidence. Here
a whole sweep gets a paragraph, a payoff score, and the reason it is worth
remembering.

Two rules this file exists to enforce:

* **Scores are never compared across campaigns.** Campaigns trained on
  different caches, against different recordings, at different rock shares, and
  a segmenter graded per point is not on the same scale as a classifier graded
  per candidate ball. This project has been misled by that comparison three
  times. So the 1-10 number below rates *what the campaign bought* — did it
  answer its question, is the answer trustworthy, was the GPU time worth it —
  not how high its PR-AUC printed.
* **The prose is fixed; the numbers are read from disk.** Fold counts, dates
  and GPU hours come from the run directories every time, so a card cannot
  drift away from what actually happened. Only the judgement is written here.

A campaign whose queue was abandoned also carries ``stopped_reason`` — why the
remaining folds were never run, which is a different question from what the
finished ones showed, and the one the Training screen asks when it finds folds
sitting unrun.

Sourced from handoff/03-training-RESULT.md and the reports under
``training/reports/``.
"""

from __future__ import annotations

import os

from .inventory import _FOLD_PREFIX, _read_json, _stat

#: What the payoff score means, shown above the history list.
SCORE_SCALE = (
    "1-10 rates what the sweep bought — whether it answered the question it "
    "was set, how far the answer can be trusted, and whether the GPU time was "
    "worth it. It is not the model's accuracy: campaigns trained on different "
    "data at different rock shares, and their raw scores are not on one scale."
)

#: One entry per folder under training/experiments. Order is not meaningful —
#: the page sorts by when the campaign last produced a fold.
CAMPAIGNS = {
    # ---------------------------------------------------------------- 2026-09
    "clutter": {
        "title": "The clutter recipe, repeated and taken apart",
        "score": 7,
        "verdict": ("Produced the classifier you should be running, and killed "
                    "three ideas about how to improve it."),
        "asked": (
            "One checkpoint — the stray-plus-phantom classifier — scored far "
            "better on the labelled competition cache than anything else here, "
            "while scoring mediocre on its own held-out volleyball recording. "
            "Does that recipe repeat, or was that checkpoint lucky? And can it "
            "be improved by halving the stray jitter, by stopping the phantom "
            "clumps from eating 8% of the rock examples, or by telling the "
            "model how high its candidate sits in its own neighborhood — the "
            "one coordinate its input has never carried."),
        "happened": (
            "Five settings, three seeds each, twice over: once on the winner's "
            "own split as a control, and once trained on all eleven volleyball "
            "recordings, which is what you do when the real test is a different "
            "arena. Thirty fits. Every single one beat the deployed classifier "
            "on the competition cache, by 0.05 at worst and 0.09 on average. "
            "The recipe is real and it should be deployed. Six more fits asked "
            "whether PointNet++ composes with it. It does not - that "
            "architecture already reaches the same place on its own, and "
            "adding the clutter recipe to it changes nothing. It also costs "
            "nothing extra to run, 39 ms against 40 ms for a thousand "
            "candidate balls, so the old objection to it was wrong twice."),
        "failure": (
            "Nothing improved it. All three variants landed inside the "
            "seed-to-seed spread, which on the competition cache is 0.014 to "
            "0.049 average precision — larger than any of the effects being "
            "tested. The candidate-height input was the most interesting idea "
            "and the clearest failure: the model does read it, but it learned "
            "it as a rock cue rather than a clutter cue, because on a flat "
            "court higher means rock. And the 0.7112 checkpoint did not "
            "repeat — thirty fits span 0.592 to 0.704 and only one reached it."),
        "learned": (
            "Deploy the combined recipe and stop tuning it; the knobs tried "
            "here are all inside the noise. Stop selecting on held-out "
            "volleyball recordings — across fifteen checkpoints its ranking "
            "correlates with the competition arena's at Spearman 0.28, and the "
            "best checkpoint on the arena was one of the worse ones on "
            "volleyball. Adding the twelfth recording back to training made no "
            "measurable difference either way. Graded on the map it builds "
            "rather than on a ranking, the combined recipe trained on every "
            "recording beats the deployed checkpoint at every matched "
            "operating point and takes the weakest rock from 0.29 to 0.36 "
            "coverage - the number that decides whether the arena is safe "
            "to drive. And the sharpest finding is one nobody set out to "
            "look for: the validation score that picks best.pt inside every "
            "run correlates with the arena at 0.12 out of 1, so the epoch "
            "being kept is chosen on a signal that does not track the thing "
            "the robot has to do. Scoring a run's other epochs on the arena "
            "is probably worth more than any augmentation tried here."),
    },
    # ---------------------------------------------------------------- 2026-08
    "fullsweep": {
        "title": "Full sweeps and the whole-frame segmenter",
        "score": 9,
        "verdict": "The one that paid. Produced the model you export from.",
        "asked": (
            "Everything at once, on the real volleyball data. Build each frame "
            "from a whole sensor rotation instead of a single 4-millisecond "
            "burst, then ask four questions on top of it: does the whole-frame "
            "segmenter beat the sliding-window classifier, does brightness "
            "help, does the fancier model help, and does training longer help."),
        "happened": (
            "Ten settings, eleven recordings each, 110 folds over three days. "
            "Whole rotations fixed the data problem that mattered — ten of the "
            "twelve labelled rocks that had never produced a single training "
            "sample started appearing. Training the segmenter for 80 passes "
            "over the data instead of 30 was the one change that measurably "
            "helped."),
        "failure": (
            "The win was small. Training longer gained 0.019, which is barely "
            "above the 0.017 gap you get from re-running the same setting with "
            "a different random seed — real, because 9 of 11 recordings moved "
            "the same way, but not enough to change what the model can do. And "
            "it still had not finished learning at 80 passes: five of eleven "
            "folds peaked in the last five. The cap was raised, not removed."),
        "learned": (
            "Train longer; don't bother with brightness, and don't bother with "
            "the fancier model. The best segmentation setting found anywhere is "
            "this campaign's 80-pass arm on ordinary full-sweep frames, and "
            "that is what got exported. It also established the yardstick every "
            "later claim is measured against: on this data, a difference of "
            "nothing looks like 0.017."),
    },
    "segdense": {
        "title": "Feeding the segmenter more frames",
        "score": 4,
        "verdict": "Clean negative, and it sent you down a false trail first.",
        "asked": (
            "Whether the whole-frame segmenter was simply starved. It saw about "
            "2,165 frames per fold where the classifier had 94,519 samples, so "
            "this kept every sensor rotation instead of every fourth — four "
            "times the frames — and trained for 60 passes instead of 30."),
        "happened": (
            "The headline arm finished all twelve recordings and lost: 0.008 "
            "worse than the old baseline, winning 5 folds of 11. That is a "
            "clean no. Consecutive frames are 0.05 seconds apart and overlap "
            "almost completely, so four times the frames is four times the "
            "repetition, not four times the data. It cost four times the GPU "
            "time and returned nothing."),
        "failure": (
            "Two things went wrong beyond the negative result. First, the "
            "second arm — looking at finer scales — ran 4 folds, beat the "
            "baseline on all three shared ones by a consistent 0.057, and "
            "looked like a real find. It was not: a later arm showed the stock "
            "setting was doing badly on these dense frames, rather than the "
            "finer one doing well. Second, the training score went *up* while "
            "the held-out score went *down*, because the validation slice comes "
            "from recordings the model has already seen. On this data the "
            "validation curve cannot tell you whether a change helps."),
        "stopped_reason": (
            "Stopped on purpose on 2026-08-20, not by a crash. The two arms "
            "that mattered had already answered the question and disagreed with "
            "each other, so the rest of the queue was rebuilt around the answer: "
            "the finer-scales arm was dropped after 4 of 12 folds, and the "
            "classifier partner (~11h) and the seed repeat (~9h) were never "
            "started. About twenty hours of GPU time saved. The unrun folds are "
            "not waiting their turn — nothing intends to run them."),
        "learned": (
            "The segmenter was never short of frames, so stop buying more of "
            "them. Three consistent folds are not evidence — that is the third "
            "time a tight little pattern across a handful of recordings has "
            "turned out to be nothing. Half this campaign's queue was cancelled "
            "on 2026-08-20 once the answer was in, which was the right call."),
    },
    "seedstudy": {
        "title": "How much does the random seed alone move the score?",
        "score": 6,
        "verdict": "Small, cheap, and it built the ruler everything else uses.",
        "asked": (
            "Nothing about rocks. Run the identical setting seven times over, "
            "changing only the random seed, and see how far apart the scores "
            "land. Without that number there is no way to say whether any "
            "other result is a finding or a coincidence."),
        "happened": (
            "Fourteen runs in about an hour and a half — seven seeds of "
            "shape-only and seven of shape-plus-brightness, all held out "
            "against the same recording. Shape-only landed between 0.838 and "
            "0.862, a spread of 0.024. The brightness arm spread 0.016."),
        "failure": (
            "Not a failure so much as a limit: every run held out the same "
            "single recording, so this measures seed-to-seed wobble on one "
            "fold and not the fold-to-fold scatter that a real comparison has "
            "to survive. The full sweeps compute their own noise floor across "
            "all eleven recordings, and those are the numbers to quote — 0.017 "
            "on full-sweep data, 0.008 on the older raw-burst data."),
        "learned": (
            "Any gap under about 0.02 is noise. That single sentence has since "
            "killed several results that looked convincing, and it is the "
            "cheapest thing on this page per GPU hour spent."),
    },
    "reflectivity": {
        "title": "Does brightness help at all?",
        "score": 7,
        "verdict": "A definitive no, argued properly. Stop asking.",
        "asked": (
            "Whether the brightness value the sensor reports for each point "
            "carries any signal about rock versus sand. Eleven settings: both "
            "models with and without brightness, with its randomisation on and "
            "off, plus seed repeats — and one arm fed nothing but brightness "
            "and no coordinates at all."),
        "happened": (
            "121 folds, about fifteen hours. Every comparison came back inside "
            "the noise. The decisive one is the brightness-only arm: stripped "
            "of shape it scored 0.224 against a 0.11 rock share, and its "
            "ability to tell rock from sand was 0.546 where a coin flip is "
            "0.500. A separate no-training measurement of the raw channel put "
            "it at 0.508. There is nothing in there to find."),
        "failure": (
            "It spent eleven arms and fifteen GPU hours to confirm one thing, "
            "and it ran on the raw-burst frames that the very next campaign "
            "replaced — so the numbers themselves are superseded even though "
            "the conclusion is not. The conclusion survived because the "
            "full-sweep campaign re-tested brightness on better data and agreed."),
        "learned": (
            "Brightness earns nothing, twice measured, on two different "
            "datasets. It stays switched off. The design is the part worth "
            "copying: including a deliberately crippled arm turned a pile of "
            "small negative numbers into an argument."),
    },
    # ---------------------------------------------------------------- retired
    "compare-fused": {
        "title": "First look at the whole-frame segmenter (indoor)",
        "score": 3,
        "verdict": "Retired. Where the units trap was born, unnoticed.",
        "asked": (
            "Whether labelling every point of a fused frame in one pass could "
            "work at all, next to the sliding-window classifier — on the indoor "
            "comforter recordings, before any volleyball-court data existed."),
        "happened": (
            "24 folds. The classifiers scored about 0.99 and the segmenter "
            "about 0.62, which read at the time as the segmenter being far "
            "worse. It is not a fair reading. The classifier was graded on "
            "candidate balls that were 17 to 34 percent rock; the segmenter was "
            "graded on individual points that were 1 to 3 percent rock. Those "
            "are two different measurements and the gap between them is mostly "
            "the difference in what counts as a hard question."),
        "failure": (
            "Indoors on a comforter, rocks are trivially separable, so 0.99 "
            "measured the room and not the model. None of these numbers "
            "transferred to the arena, where the same classifier setting scores "
            "about 0.78. This campaign is excluded from the Runs board on "
            "purpose — sitting next to the real numbers it only misleads."),
        "learned": (
            "Never put a segmenter's score and a classifier's score in the same "
            "column. It took two more campaigns to state that rule out loud; it "
            "was already visible here."),
    },
    "compare": {
        "title": "PointNet vs PointNet++ (indoor)",
        "score": 3,
        "verdict": "Retired. Proved the plumbing, not the model.",
        "asked": (
            "The oldest sweep here: does the hierarchical model beat the plain "
            "one, on the indoor comforter and bedroom recordings."),
        "happened": (
            "26 folds. Plain PointNet 0.957, PointNet++ 0.946 — no real "
            "difference, which is the same answer the arena data gave later at "
            "0.780 against 0.765. The question was settled correctly by "
            "accident."),
        "failure": (
            "The scores are meaningless as accuracy. A fifth to two-fifths of "
            "every sample was rock and the setting was an indoor floor, so "
            "0.95-plus was the room being easy. Anyone reading these as "
            "'the model is 95 percent right about rocks' would be badly wrong, "
            "which is exactly why it is kept off the Runs board."),
        "learned": (
            "The training and evaluation harness works end to end — that is "
            "what this campaign actually established, and it was worth having. "
            "It is also the origin of the habit of holding out a whole "
            "recording rather than a random slice, which every later sweep "
            "kept."),
    },
}

#: Campaigns retired from the Runs board — kept here as history, flagged so a
#: reader cannot mistake their scores for current ones.
RETIRED = ("compare", "compare-fused")


def _arm_dirs(edir: str) -> list[str]:
    """The distinct settings a campaign ran, ignoring logs and superseded work.

    The current suites give each setting its own folder with the folds inside
    it. The two retired campaigns instead put every (setting, fold) pair flat
    at the top as ``<arm>_loro_<fold>``, so those are folded back down to the
    setting name — otherwise a 2-arm campaign reports 26 arms.
    """
    out = []
    for name in sorted(os.listdir(edir)):
        if ".superseded-" in name or not os.path.isdir(os.path.join(edir, name)):
            continue
        arm = name.split(_FOLD_PREFIX, 1)[0].rstrip("_") if _FOLD_PREFIX in name else name
        if arm and arm not in out:
            out.append(arm)
    return out


def _walk_folds(edir: str) -> list[dict]:
    """Every finished fold under a campaign, however deeply it is nested.

    Campaigns are not all shaped the same — the current suites nest
    ``<arm>/<fold>/`` while the two retired ones put ``<arm>_loro_<fold>/``
    flat at the top — so this walks rather than assuming a depth.

    A fit that held nothing out writes ``val_metrics.json`` instead, because
    there is no unseen recording left to score it on. Those count as finished
    folds here — they occupied the graphics card exactly as long — but their
    score is a validation score and is flagged as such, so nothing downstream
    can average it in with a held-out one.
    """
    out = []
    for dirpath, dirs, names in os.walk(edir):
        if ".superseded-" in dirpath:
            dirs[:] = []
            continue
        held_out = "test_metrics.json" in names
        if not held_out and "val_metrics.json" not in names:
            continue
        metrics = _read_json(os.path.join(
            dirpath, "test_metrics.json" if held_out else "val_metrics.json")) or {}
        cfg_stat = _stat(os.path.join(dirpath, "config.json"))
        hist_stat = _stat(os.path.join(dirpath, "history.csv"))
        # Wall time for the fold: config.json is written the moment it starts,
        # history.csv gains its last row when it stops.
        elapsed = None
        if cfg_stat["mtime"] and hist_stat["mtime"]:
            delta = hist_stat["mtime"] - cfg_stat["mtime"]
            elapsed = delta if delta > 0 else None
        out.append({
            "pr_auc": metrics.get("pr_auc") if held_out else None,
            "val_pr_auc": None if held_out else metrics.get("pr_auc"),
            "held_out": held_out,
            "task": metrics.get("task"),
            "model": metrics.get("model"),
            "started": cfg_stat["mtime"] or None,
            "finished": hist_stat["mtime"] or None,
            "elapsed_s": elapsed,
        })
    return out


def campaigns(root: str) -> dict:
    """Every past campaign: the written judgement plus what disk still says.

    A campaign with prose but no folder is dropped, and a folder with no prose
    still appears — with its measured numbers and an empty write-up — so a new
    sweep is visible the day it runs rather than the day someone documents it.
    """
    base = os.path.join(root, "training", "experiments")
    out: list[dict] = []
    if not os.path.isdir(base):
        return {"campaigns": [], "scale": SCORE_SCALE}

    for name in sorted(os.listdir(base)):
        edir = os.path.join(base, name)
        if not os.path.isdir(edir):
            continue
        folds = _walk_folds(edir)
        if not folds:
            continue
        written = CAMPAIGNS.get(name, {})
        scored = [f["pr_auc"] for f in folds if isinstance(f["pr_auc"], float)]
        times = [f["elapsed_s"] for f in folds if f["elapsed_s"]]
        starts = [f["started"] for f in folds if f["started"]]
        ends = [f["finished"] for f in folds if f["finished"]]
        # Tasks are listed rather than averaged together: a per-point score and
        # a per-ball score do not belong in one mean.
        tasks = sorted({f["task"] for f in folds if f["task"]})
        out.append({
            "name": name,
            "title": written.get("title") or name,
            "score": written.get("score"),
            "verdict": written.get("verdict", ""),
            "asked": written.get("asked", ""),
            "stopped_reason": written.get("stopped_reason", ""),
            "happened": written.get("happened", ""),
            "failure": written.get("failure", ""),
            "learned": written.get("learned", ""),
            "documented": bool(written),
            "retired": name in RETIRED,
            "path": os.path.relpath(edir, root),
            "arms": _arm_dirs(edir),
            "folds_done": len(folds),
            "best_pr_auc": max(scored) if scored else None,
            "tasks": tasks,
            "gpu_hours": round(sum(times) / 3600.0, 1) if times else None,
            "started": min(starts) if starts else None,
            "finished": max(ends) if ends else None,
        })

    # Newest work first: the sweep you last ran is the one you are thinking about.
    out.sort(key=lambda c: c["finished"] or 0, reverse=True)
    return {"campaigns": out, "scale": SCORE_SCALE}
