"""What every trained setting is, what it concluded, and whether to still show it.

Three hundred and sixty checkpoints have accumulated under
``training/experiments/``, across forty-three named settings, and the pickers
were listing them by directory name with a score attached. That is not
something a person can choose from: the name says nothing about what the model
is, and the score is only comparable inside its own sweep.

This is the one place that fixes both. Each entry gives a setting a
plain-English **title** (what the pickers group by), a **status** (whether it is
worth offering at all), and a **note** (what it measured, in one paragraph).
The dashboard's run board and the live viewer's checkpoint picker both read it,
so a conclusion is written down once instead of in two places that drift.

**Statuses**, and what each one means for the pickers:

``deploy``     the model to actually run. Listed first, starred.
``active``     current work. Listed.
``reference``  kept as the thing something else is compared against. Listed.
``settled``    the question it existed to answer has been answered. Hidden
               behind the "show archived" toggle, not deleted — the numbers are
               the only record of what was concluded.
``dead-end``   measured, and it lost. Hidden, for the same reason.
``abandoned``  never finished. A few epochs, no score. Hidden.
``retired``    trained against a population nothing current matches, so its
               scores cannot be read next to anything here. Not listed at all.

On 4 September 2026 the folders behind the eleven ``reflectivity`` arms, the
two ``compare`` experiments and four abandoned stubs were deleted — 2.7 GB of
weights answering questions that are closed. Their conclusions survive in
``training/reports/``, which is small and is kept in git; the weights were not
and were rebuildable from the recordings. The two ``retired`` entries below
stay so a restored backup is hidden again rather than reappearing in a picker.

Adding a setting here is optional: anything unlisted is treated as ``active``
and named after its directory, which is what every arm did before this existed.
"""

from __future__ import annotations

#: Statuses whose checkpoints the pickers hide by default. They stay on disk
#: and stay reachable through the archive toggle.
HIDDEN = ("settled", "dead-end", "abandoned")

#: Statuses whose checkpoints are never offered, because their numbers are not
#: comparable with anything current.
NEVER_LISTED = ("retired",)

#: ``<experiment>/<arm>`` (or a bare ``<experiment>`` for a one-off directory)
#: -> (title, status, note).
CATALOG: dict[str, tuple[str, str, str]] = {

    # -- the models to actually run -----------------------------------------
    "deploy/cls-stray": (
        "Deploy · classifier, clutter-trained", "deploy",
        "THE ONE TO RUN. Finds 135-137 of 138 rocks on the competition "
        "recording with a confidence gap of +0.90 between rock and bare "
        "ground, and not one of its eleven leave-one-out siblings went flat "
        "there. Fitted on all eleven volleyball recordings with nothing held "
        "out, so it has no honest held-out score of its own - the folds under "
        "'stray' are where its numbers come from."),
    "deploy/cls-base": (
        "Deploy · classifier, no clutter training", "reference",
        "The same classifier without the clutter augmentation, kept so the two "
        "can be flipped between on one recording to see what that training "
        "did and nothing else. On the competition bag it is much weaker: 23 of "
        "54 rocks against 54 of 54."),
    "deploy/seg-base": (
        "Deploy · segmenter, best of the family", "reference",
        "The strongest whole-frame segmenter trained here, kept for "
        "comparison. Still finds only 30-63 rocks of 138 where the classifier "
        "finds 135-137. See the write-up in training/reports/stray/ before "
        "using a segmenter for anything."),
    "deploy/seg-stray": (
        "Deploy · segmenter, clutter-trained", "dead-end",
        "Clutter training did for the segmenter the opposite of what it did "
        "for the classifier: 8 of 54 rocks found at its own threshold, with a "
        "confidence gap of +0.10. Kept only as the other half of a pair."),

    # -- the clutter-augmentation sweep -------------------------------------
    "stray/cls-stray": (
        "Classifier · clutter-trained (11 folds)", "active",
        "The result this project has most confidence in. Training against "
        "stray returns is free on the volleyball data (+0.003, p=0.52, inside "
        "the 0.017 noise floor) and decisive on the competition arena: +0.093 "
        "PR-AUC, ahead on 11 of 11 folds, p=0.001, with the confidence gap "
        "going from +0.27 to +0.89 and the number of runs that go flat there "
        "dropping from 6 of 11 to none."),
    "stray/cls-base": (
        "Classifier · stock (11 folds)", "reference",
        "The control for the clutter pair, and a reproduction of the best "
        "classifier setting two earlier sweeps found - it matches "
        "fullsweep/pointnet-geom to four decimal places, which is what "
        "confirms the pair differs in one setting only. Six of its eleven runs "
        "go flat on the competition arena."),
    "stray/seg-base": (
        "Segmenter · deployment settings (6 folds)", "reference",
        "The segmenter with everything two sweeps found worth having: 80 "
        "epochs, finer levels, floor-referenced heights, frame-centred "
        "coordinates. The best of the family and still far behind the "
        "classifier on the arena."),
    "stray/seg-stray": (
        "Segmenter · clutter-trained (6 folds)", "dead-end",
        "The clutter augmentation applied to the segmenter. Mean down on both "
        "PR-AUC and confidence gap against seg-base, and the direction flips "
        "depending on which lance frames are sampled - so it is inside the "
        "noise, not a result either way."),
    "stray/seg-relief": (
        "Segmenter · trained for arena relief (4 folds)", "dead-end",
        "The targeted attempt to fix segmentation. Every competition frame is "
        "outside the training range on vertical structure (0.46-0.55 m against "
        "0.10-0.26 m) and density (3,479 points against 1,145), so this arm "
        "trained against both. Two folds better, two worse, mean down on both "
        "measures. The diagnosis was measured and real; the treatment failed, "
        "which is what closed the question."),

    # -- the full-sweep sweep ------------------------------------------------
    "fullsweep/pointnet2seg-geom-e80": (
        "Segmenter · 80 epochs (the one deployed before Sept 2026)", "dead-end",
        "Was the arm to export from, on the strength of +0.019 over the "
        "30-epoch baseline (9/11 folds, p=0.032). Measured properly on the "
        "competition recording it is the worst model in the set: PR-AUC 0.223 "
        "and a confidence of 0.638 on bare ground, which paints the whole "
        "arena warm. Kept as the before picture."),
    "fullsweep/pointnet2seg-geom": (
        "Segmenter · 30-epoch baseline", "settled",
        "The 30-epoch segmentation baseline every other fullsweep arm pairs "
        "against. Ties PointNet++ on matched spots (0.767 vs 0.764) with a "
        "much higher floor on hard recordings - a comparison that later turned "
        "out to be measured on a rock-enriched population, see the write-up."),
    "fullsweep/pointnet2seg-geom-e80-fine": (
        "Segmenter · 80 epochs and finer levels", "settled",
        "80 epochs AND finer level geometry: -0.010 vs stock at 80 epochs "
        "(3/11 wins, p=0.28). Not a gain - it converges sooner to the same "
        "place. The +0.057 its dense-cache preview showed was stock geometry "
        "doing badly THERE, not finer levels helping."),
    "fullsweep/pointnet-geom": (
        "Classifier · shape only", "settled",
        "Shape-only PointNet baseline, and the best classifier setting found "
        "before the clutter augmentation existed. PointNet vs PointNet++ "
        "showed no real difference across both sweeps (0.780 vs 0.765, "
        "p=0.10). Superseded by stray/cls-base, which reproduces it exactly."),
    "fullsweep/pointnet-refl": (
        "Classifier · shape and brightness", "settled",
        "Two sweeps agree the brightness channel earns nothing (deltas within "
        "+/-0.01, well inside noise)."),
    "fullsweep/pointnet2-refl": (
        "PointNet++ classifier · shape and brightness", "settled",
        "-0.010 vs shape-only. Reflectivity stays off."),
    "fullsweep/pointnet2-geom": (
        "PointNet++ classifier · shape only", "settled",
        "No measurable difference from plain PointNet at three times the "
        "training cost, across two sweeps."),
    "fullsweep/pointnet2-geom-s43": (
        "PointNet++ classifier · shape only, second seed", "settled",
        "A seed repeat, and the yardstick the whole sweep is read against: two "
        "runs of one setting land 0.017 apart, so nothing smaller than that "
        "counts as a difference."),
    "fullsweep/pointnet2seg-geom-s43": (
        "Segmenter · second seed", "settled",
        "The segmenter's own noise floor, 0.012 between two runs of one "
        "setting."),
    "fullsweep/pointnet2seg-refl": (
        "Segmenter · shape and brightness", "settled",
        "Brightness earns nothing for the segmenter either (-0.010, p=0.32)."),

    # -- the dense-frame sweep ----------------------------------------------
    "segdense/seg-long": (
        "Segmenter · four times the frames, 60 epochs", "settled",
        "Clean negative: 4x the frames + 60 epochs scored -0.008 vs the old "
        "baseline (5/11 wins, p=0.83). Frames 0.05 s apart overlap almost "
        "completely, so more frames is more repetition, not more data. The "
        "segmenter was never starved."),
    "segdense/seg-fine": (
        "Segmenter · dense frames, finer levels", "settled",
        "Stopped after 4 of 12 folds. Its consistent +0.057 over stock "
        "geometry on this dense cache later turned out to be stock geometry "
        "doing badly on dense frames - see the e80-fine arm of fullsweep. The "
        "four finished folds are kept for the record."),

    # -- the reflectivity sweep (question closed) ---------------------------

    # -- one-off directories -------------------------------------------------
    "segdeploy-frame-18": (
        "Segmenter · frame-centred, 18 epochs (one fold)", "dead-end",
        "Was recommended as a 4x more precise replacement for the deployed "
        "segmenter, on the strength of 82.7% precision on the competition bag. "
        "That was precision among the handful of points it dared to call rock: "
        "it finds 7 of 54. A caution about quoting precision without recall."),
    "seedstudy/geom": (
        "Seed study · shape only", "settled",
        "Repeats of one setting under different seeds, to measure how far two "
        "identical runs land apart."),
    "seedstudy/refl": (
        "Seed study · with brightness", "settled",
        "The same, with the brightness channel."),

    # -- trained against a population nothing current matches ---------------
    "compare": (
        "Retired · early per-recording comparison", "retired",
        "Trained on the myroom and comforter recordings against a different "
        "sample population. Scores here cannot be read next to anything "
        "current."),
    "compare-fused": (
        "Retired · early fused comparison", "retired",
        "As above, on the fused variant."),
}

def entry(experiment: str, arm: str | None = None) -> tuple[str, str, str]:
    """(title, status, note) for a setting; a sensible default when unlisted.

    An unlisted setting is ``active`` and titled after its directory, so a
    sweep someone starts tomorrow shows up properly without touching this file.
    """
    key = f"{experiment}/{arm}" if arm else experiment
    if key in CATALOG:
        return CATALOG[key]
    if experiment in CATALOG:
        return CATALOG[experiment]
    return (f"{experiment} · {arm}" if arm else experiment), "active", ""

def status(experiment: str, arm: str | None = None) -> str:
    return entry(experiment, arm)[1]

def title(experiment: str, arm: str | None = None) -> str:
    return entry(experiment, arm)[0]

def note(experiment: str, arm: str | None = None) -> str:
    return entry(experiment, arm)[2]
