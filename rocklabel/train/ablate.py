"""Controlled experiments: does a channel (or a model, or an augmentation)
actually change the score?

`compare` answers "how do the models do". This answers "does *this one thing*
matter", which needs a different shape of run. Two rules make the difference:

* **One arm, one run root.** ``compare`` names a run directory after its model,
  fold and channel selection — so two arms that differ only in an augmentation
  setting would land on the same directory, and the second would archive the
  first as stale. Every arm here gets ``<root>/<arm>/`` to itself, so any two
  settings can be compared, not just channel selections.
* **Paired by fold.** Arms are compared fold by fold, never as two pooled
  averages. Fold-to-fold spread on this data is far larger than any channel
  effect (PR-AUC ranges roughly 0.5-0.95 across runs), so an unpaired
  comparison drowns the thing being measured in which-run-was-held-out noise.

The seed-repeat arms exist for the same reason. A single training run is a
random draw; without knowing how far two runs of the *same* setting land apart,
a delta between two different settings cannot be called real. Repeats of one
arm under different seeds measure exactly that floor.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..dataset.neighborhoods import FEATURES, GEOMETRY
from .models_meta import BEV_CHANNELS, BEV_DENSITY
from .metrics import normalized_pr_auc

#: Where a sweep's per-fold run directories live: one folder per experiment,
#: then per arm, then per fold. Renamed from the old ``training/ablate`` (and
#: its ad-hoc twin ``training/ablate_vb``) so that every trained thing in the
#: project sits under one root that says which experiment produced it.
DEFAULT_ROOT = os.path.join("training", "experiments")

#: Where the rendered tables and figures for an experiment go.
DEFAULT_REPORT_ROOT = os.path.join("training", "reports")

#: Metrics carried into every table, primary first. PR-AUC leads because the
#: data is 5-31% rock depending on the run: accuracy is nearly free and ROC-AUC
#: is optimistic when negatives dominate.
#:
#: ``norm_pr_auc`` rides alongside it as ``(AP - prevalence) / (1 - prevalence)``:
#: raw PR-AUC starts at the fold's own rock share, so a per-fold table of raw
#: numbers partly measures how many rocks a recording has rather than how hard
#: it is. Raw stays the headline for continuity with the two finished sweeps.
METRICS = ("pr_auc", "norm_pr_auc", "roc_auc", "f1", "precision", "recall")
PRIMARY = "pr_auc"
#: The same comparison with the fold's rock share divided out.
NORMALIZED = "norm_pr_auc"


@dataclass
class Arm:
    """One training setting, trained on every fold.

    ``overrides`` goes straight into the training config, so an arm can change
    anything ``default_config`` accepts — not just channels.
    """

    name: str
    model: str
    features: list[str]
    label: str          # short name for figures
    what: str           # plain-English description, shown in the report
    overrides: dict = field(default_factory=dict)
    seed: int | None = None

    def config(self, cfg_fn, train_runs: list[str], test_run: str) -> dict:
        kw = dict(self.overrides)
        if self.seed is not None:
            kw["seed"] = self.seed
        return cfg_fn(model=self.model, features=list(self.features),
                      train_runs=train_runs, test_run=test_run, **kw)


_ALL = list(FEATURES)
_GEOM = list(GEOMETRY)
#: Turning the reflectivity jitter off. The default augmentation deliberately
#: swamps the absolute intensity level (gain +/-25%, offset +/-0.10 on a channel
#: whose entire useful range is about 0.10 wide), so a model trained with it on
#: cannot use the absolute level even if the level were informative. Any honest
#: test of "does reflectivity help" has to include an arm where it is allowed to.
_NO_JITTER = {"aug_intensity_gain": 0.0, "aug_intensity_shift": 0.0}


#: The reflectivity question, as a set of arms. Order is priority order: the
#: sweep runs top to bottom, so if it is stopped early the headline comparison
#: is the part that finished.
REFLECTIVITY_ARMS = [
    Arm("pointnet-geom", "pointnet", _GEOM, "PointNet · shape only",
        "PointNet with the reflectivity channel removed entirely. The control: "
        "whatever this scores is what pure geometry is worth."),
    Arm("pointnet-refl", "pointnet", _ALL, "PointNet · shape + reflectivity",
        "PointNet with reflectivity included, using the standard augmentation "
        "(reflectivity randomly rescaled and shifted each sample). This is the "
        "setting the existing runs used."),
    Arm("pointnet2-geom", "pointnet2", _GEOM, "PointNet++ · shape only",
        "PointNet++ with no reflectivity. The shape-only control for the "
        "hierarchical model."),
    Arm("pointnet2-refl", "pointnet2", _ALL, "PointNet++ · shape + reflectivity",
        "PointNet++ with reflectivity included and the standard augmentation."),
    Arm("pointnet-refl-raw", "pointnet", _ALL, "PointNet · reflectivity unjittered",
        "PointNet with reflectivity included and the reflectivity augmentation "
        "switched off, so the model may use the raw absolute brightness. If "
        "reflectivity helps anywhere, it helps most here — and the gap between "
        "this and the jittered arm is the size of the cue the augmentation "
        "deliberately destroys.",
        overrides=_NO_JITTER),
    Arm("pointnet2-refl-raw", "pointnet2", _ALL, "PointNet++ · reflectivity unjittered",
        "PointNet++ with reflectivity included and its augmentation off.",
        overrides=_NO_JITTER),
    Arm("pointnet-refl-only", "pointnet", ["intensity"], "PointNet · reflectivity only",
        "PointNet fed nothing but reflectivity — no coordinates at all. It "
        "cannot see shape, so its score is a direct read of how much the "
        "brightness numbers alone can separate rock from sand.",
        overrides=_NO_JITTER),
    # Seed repeats of the two headline arms: the yardstick for "is a delta real".
    Arm("pointnet-geom-s43", "pointnet", _GEOM, "PointNet · shape only (seed 43)",
        "Same setting as the shape-only arm, different random seed. Exists only "
        "to measure how far two identical settings land apart.", seed=43),
    Arm("pointnet-refl-s43", "pointnet", _ALL, "PointNet · shape + reflectivity (seed 43)",
        "Seed repeat of the shape+reflectivity arm.", seed=43),
    Arm("pointnet-geom-s44", "pointnet", _GEOM, "PointNet · shape only (seed 44)",
        "Second seed repeat of the shape-only arm.", seed=44),
    Arm("pointnet-refl-s44", "pointnet", _ALL, "PointNet · shape + reflectivity (seed 44)",
        "Second seed repeat of the shape+reflectivity arm.", seed=44),
]

#: Pairs the report tests head to head, as (baseline, variant, question).
REFLECTIVITY_CONTRASTS = [
    ("pointnet-geom", "pointnet-refl",
     "PointNet: does adding reflectivity beat shape alone?"),
    ("pointnet2-geom", "pointnet2-refl",
     "PointNet++: does adding reflectivity beat shape alone?"),
    ("pointnet-geom", "pointnet-refl-raw",
     "PointNet: does reflectivity help when its augmentation is switched off?"),
    ("pointnet2-geom", "pointnet2-refl-raw",
     "PointNet++: does reflectivity help when its augmentation is switched off?"),
    ("pointnet-refl", "pointnet-refl-raw",
     "PointNet: how much does the reflectivity augmentation cost?"),
    ("pointnet-geom", "pointnet2-geom",
     "Shape only: is PointNet++ better than PointNet?"),
    ("pointnet-refl", "pointnet2-refl",
     "Shape + reflectivity: is PointNet++ better than PointNet?"),
    ("pointnet-geom", "pointnet-geom-s43",
     "Noise floor: the same shape-only setting, two different seeds."),
    ("pointnet-refl", "pointnet-refl-s43",
     "Noise floor: the same shape+reflectivity setting, two different seeds."),
]

#: The full-sweep question, as a set of arms. Two things change at once
#: relative to the reflectivity suite, on purpose:
#:
#: * the dataset is built from whole 20 Hz sensor rotations instead of single
#:   ~4 ms sensor batches, so a frame carries ~1250 points in the crop box
#:   instead of ~110, and
#: * a third model joins in - the per-point segmenter, which could not be
#:   trained at all on the batch-sized frames (they fell below the 512-point
#:   floor, so the generator produced zero segmentation frames).
#:
#: Order is priority order: the segmenter-vs-classifier headline runs first,
#: then reflectivity, then the seed repeats that say how big a meaningless
#: difference looks. Stopping early still leaves the headline finished.
FULLSWEEP_ARMS = [
    Arm("pointnet2-geom", "pointnet2", _GEOM, "PointNet++ · shape only",
        "PointNet++ scoring one 0.5 m ball at a time, with the reflectivity "
        "channel removed. This is the arm to line up against the same-named "
        "arm of the reflectivity suite: same model, same settings, same folds - "
        "the only difference is that a frame here is a whole sensor rotation."),
    Arm("pointnet2seg-geom", "pointnet2_seg", _GEOM, "Segmentation · shape only",
        "PointNet++ labelling every point of a whole frame in one pass, shape "
        "only. The headline arm: it answers a rock question in one forward pass "
        "per frame instead of one per candidate ball."),
    Arm("pointnet-geom", "pointnet", _GEOM, "PointNet · shape only",
        "Plain PointNet on single balls, shape only - the cheapest model, kept "
        "as the floor everything else has to beat."),
    Arm("pointnet2-refl", "pointnet2", _ALL, "PointNet++ · shape + reflectivity",
        "PointNet++ on single balls with reflectivity added back, using the "
        "standard brightness jitter."),
    Arm("pointnet2seg-refl", "pointnet2_seg", _ALL, "Segmentation · shape + reflectivity",
        "The whole-frame segmenter with reflectivity added back. Denser frames "
        "give the segmenter far more brightness context than a single ball has, "
        "so this is where reflectivity has its best chance of paying off."),
    Arm("pointnet-refl", "pointnet", _ALL, "PointNet · shape + reflectivity",
        "Plain PointNet with reflectivity included."),
    Arm("pointnet2-geom-s43", "pointnet2", _GEOM, "PointNet++ · shape only (seed 43)",
        "Same setting as the shape-only PointNet++ arm, different random seed. "
        "Exists only to measure how far two identical settings land apart, so a "
        "difference between two real arms can be called real or not.", seed=43),
    Arm("pointnet2seg-geom-s43", "pointnet2_seg", _GEOM, "Segmentation · shape only (seed 43)",
        "Seed repeat of the shape-only segmentation arm - the noise floor for "
        "the segmenter.", seed=43),
    Arm("pointnet2seg-geom-e80", "pointnet2_seg", _GEOM,
        "Segmentation · shape only (80 epochs)",
        "The shape-only segmenter given 80 epochs instead of 30, with the "
        "early-stopping patience raised to match. Nothing else changes - same "
        "frames, same model, same seed - so the gap between this and the "
        "30-epoch arm is purely what stopping too early cost. Not one of the "
        "eleven 30-epoch folds early-stopped: every one ran to the cap, the "
        "median best epoch was 28 of 30, and 7 of 11 peaked in the last five "
        "epochs. That makes 0.767 a floor, not a ceiling.",
        overrides={"epochs": 80, "patience": 30}),
    Arm("pointnet2seg-geom-e80-fine", "pointnet2_seg", _GEOM,
        "Segmentation · 80 epochs, finer levels",
        "The 80-epoch segmenter looking at finer scales: the first of its three "
        "levels now keeps 1,024 of the frame's points instead of 512 and pools "
        "over a 0.10 m radius instead of 0.25 m. The labelled rocks measure "
        "21-68 cm across, so the stock 0.25 m radius is a half-metre ball that "
        "swallows a whole rock - the smallest scale the model looked at was "
        "bigger than the thing it was hunting. Everything else matches the "
        "80-epoch arm exactly, so the gap between them is the level geometry "
        "and nothing else. This is the combination of the two changes that "
        "measured as worth having, without the one that did not: more epochs "
        "helped (+0.019, p=0.032), finer levels look like they help, and "
        "keeping four times the frames did nothing at four times the cost.",
        overrides={"epochs": 80, "patience": 30,
                   "seg_npoints": [1024, 256, 64], "seg_radii": [0.1, 0.3, 0.8]}),
]

#: Head-to-head questions for the full-sweep suite.
#:
#: NOTE the first two are scored per point for the segmenter and per candidate
#: ball for the classifiers, which are different populations with very
#: different rock prevalence (~1% of points vs ~19% of balls). The paired table
#: still says which arm won each fold, but the raw PR-AUC gap between a
#: segmenter and a classifier is not a like-for-like number - `vb_compare.py`
#: re-scores both at the same candidate centers for that.
FULLSWEEP_CONTRASTS = [
    ("pointnet2-geom", "pointnet2seg-geom",
     "Shape only: does whole-frame segmentation beat the sliding-window classifier?"),
    ("pointnet2-refl", "pointnet2seg-refl",
     "Shape + reflectivity: does whole-frame segmentation beat the classifier?"),
    ("pointnet-geom", "pointnet2-geom",
     "Shape only: is PointNet++ better than plain PointNet?"),
    ("pointnet2-geom", "pointnet2-refl",
     "PointNet++: does adding reflectivity beat shape alone?"),
    ("pointnet2seg-geom", "pointnet2seg-refl",
     "Segmentation: does adding reflectivity beat shape alone?"),
    ("pointnet-geom", "pointnet-refl",
     "PointNet: does adding reflectivity beat shape alone?"),
    ("pointnet2-geom", "pointnet2-geom-s43",
     "Noise floor: the same PointNet++ setting, two different seeds."),
    ("pointnet2seg-geom", "pointnet2seg-geom-s43",
     "Noise floor: the same segmentation setting, two different seeds."),
    ("pointnet2seg-geom", "pointnet2seg-geom-e80",
     "Segmentation: does it just need longer than 30 epochs?"),
    ("pointnet2seg-geom-e80", "pointnet2seg-geom-e80-fine",
     "Segmentation at 80 epochs: do finer levels beat the stock geometry?"),
    ("pointnet2seg-geom", "pointnet2seg-geom-e80-fine",
     "Segmentation: longer training and finer levels together, against the "
     "original 30-epoch arm."),
]

#: Common to every segmentation arm of the dense suite. Two changes from the
#: fullsweep suite, both aimed at the same finding: not one of the eleven
#: 30-epoch segmentation folds ever early-stopped, so the model was still
#: improving when the run ended.
#:
#: * 60 epochs with a matching patience, on ~4x as many frames per epoch - about
#:   twenty times the gradient steps the 30-epoch runs got.
#: * A batch of 512, which the engine turns into 16 whole frames per step
#:   instead of 8. Measured 1.5x faster per epoch at the same settings, which is
#:   what pays for the extra epochs.
_SEG_LONG = {"epochs": 60, "patience": 25, "batch": 512,
             # Pinned, not inherited. These two settings changed defaults in
             # August 2026; every segmentation arm below was trained before that
             # and has to keep asking for the old behavior or a rerun would
             # quietly retrain them against a different height reference and
             # invalidate the baseline the new arm is measured against.
             "seg_height_ref": "base", "aug_ground_tilt": 0.0}

#: Finer levels for the segmenter (section 3 of the training brief). The stock
#: geometry throws three quarters of a frame away at the first level and its
#: finest ball is 0.5 m across (a 0.25 m radius) against labelled rocks that
#: measure 21-68 cm, mean 44 cm - so the smallest scale it ever looked at was
#: already bigger than the thing it was hunting. This keeps 1,024 of 1,280
#: points at the first level and starts at a 0.10 m radius, which is a patch of
#: a rock's surface rather than the whole rock.
_SEG_FINE = dict(_SEG_LONG, seg_npoints=[1024, 256, 64], seg_radii=[0.1, 0.3, 0.8])

#: The dense-frame question. Runs on the ``full-sweep-dense`` cache, which is
#: the same 0.05 s sensor rotations as ``full-sweep`` but keeps every frame
#: instead of every fourth, at a 1,280-point budget instead of 2,048.
#:
#: It also has **twelve** folds, not eleven: VolleyBallTest13 has rocks, an
#: arena and a height band, and had simply never been generated. Its scores are
#: therefore not directly comparable fold-for-fold with the fullsweep suite -
#: compare the eleven shared folds when putting the two side by side.
SEGDENSE_ARMS = [
    Arm("seg-long", "pointnet2_seg", _GEOM, "Segmentation · long, dense",
        "The whole-frame segmenter on four times as many frames, trained for 60 "
        "epochs instead of 30. The headline arm: the previous sweep's "
        "segmentation folds all ran out of epochs before they stopped improving, "
        "and the segmenter was learning from ~2,165 frames per fold where the "
        "sliding-window classifier had 94,519 samples. Both of those are fixed "
        "here, and they pull the same way.",
        overrides=_SEG_LONG),
    Arm("seg-fine", "pointnet2_seg", _GEOM, "Segmentation · long, dense, finer levels",
        "Everything the long arm does, plus a finer look: the first level now "
        "keeps 1,024 of the frame's points instead of 512 and pools over a 0.10 m "
        "radius instead of 0.25 m. The labelled rocks measure 21-68 cm across "
        "(mean 44 cm), so a 0.25 m radius is a half-metre ball - it swallows a "
        "whole rock, and the smallest scale the model looked at was already "
        "bigger than the thing it was hunting. A 0.10 m radius sees about half "
        "a rock, so the first level can pick up surface curve rather than "
        "treating a rock as one blob.",
        overrides=_SEG_FINE),
    Arm("pointnet-geom", "pointnet", _GEOM, "PointNet · shape only",
        "The sliding-window classifier on the same dense frames, so the "
        "segmenter has a partner trained on exactly the same data for the "
        "matched-population comparison. PointNet rather than PointNet++ because "
        "two sweeps found no difference between them and PointNet is three "
        "times cheaper to train."),
    Arm("seg-long-s43", "pointnet2_seg", _GEOM,
        "Segmentation · long, dense (seed 43)",
        "The long arm again with a different random seed, and nothing else "
        "changed. Its gap from the long arm is what a difference of nothing "
        "looks like at these settings - the older 0.0207 floor was measured at "
        "30 epochs on a quarter of the frames and does not transfer.",
        overrides=_SEG_LONG, seed=43),
]

SEGDENSE_CONTRASTS = [
    ("seg-long", "seg-fine",
     "Segmentation: does looking at finer scales beat the stock level geometry?"),
    ("pointnet-geom", "seg-long",
     "Does whole-frame segmentation beat the sliding-window classifier?"),
    ("pointnet-geom", "seg-fine",
     "Does the finer-level segmenter beat the sliding-window classifier?"),
    ("seg-long", "seg-long-s43",
     "Noise floor: the same long segmentation setting, two different seeds."),
]

#: The deployment segmenter, as it is actually configured for the competition
#: rig. Everything two sweeps found worth having, in one arm: 80 epochs (the
#: 30-epoch folds never once early-stopped), the finer level geometry (a 0.10 m
#: first radius against rocks that measure 21-68 cm, instead of a 0.25 m ball
#: that swallows one whole), heights measured from the frame's own floor rather
#: than the robot base, a random ground tilt, and frame-centred coordinates.
#: The last two are what a checkpoint needs to survive an arena that is not the
#: volleyball court: measured on the competition recording, this combination
#: scored 82.7% precision where the deployed base-relative segmenter scored
#: 22.5%. Batch 512 rather than 256 because it is ~1.5x faster per epoch at
#: identical settings, which is what pays for the extra epochs.
_STRAY_SEG = {"epochs": 80, "patience": 30, "batch": 512,
              "seg_npoints": [1024, 256, 64], "seg_radii": [0.1, 0.3, 0.8],
              "seg_height_ref": "floor", "seg_coord_ref": "frame",
              "aug_ground_tilt": 0.03}

#: The clutter augmentation itself, added to whichever arm is the control.
#: 5% of every training frame becomes a return sitting on no surface, pushed
#: along the line of sight it arrived on with a heavy-tailed displacement.
_STRAY_ON = {"aug_stray_frac": 0.05, "aug_stray_reach": 1.0}

#: Does training against stray returns cost anything on clean data?
#:
#: Both model families, each with and without the augmentation, on one cache -
#: so each pair is a single-variable change and the two families are trained on
#: exactly the same frames. Order is priority order, and it is deliberately
#: family-major: the classifier pair is a quarter the cost of the segmenter
#: pair, so stopping early still leaves one complete answer rather than two
#: half-finished ones.
#: The domain-gap arm: the deployment segmenter, trained against the two frame
#: level differences that were *measured* between its training frames and the
#: competition arena, rather than against clutter.
#:
#: A whole-frame segmenter reads the frame as one object, so two properties of
#: the frame decide whether it recognises what it is looking at, and both were
#: measured to be out of range on every single lance frame:
#:
#: * **Vertical structure.** Training frames hold 0.10-0.26 m of it - the
#:   volleyball court is a flat slab. Lance frames hold 0.46-0.55 m, because the
#:   arena has a berm and walls in shot. A model whose only experience is a flat
#:   floor can read "sticks up above the ground" as "rock"; in a dug bin that
#:   cue is worthless. ``aug_ground_tilt`` at 0.08 tips the floor by up to 64 cm
#:   across the 8 m crop, which puts training's vertical extent on top of the
#:   arena's instead of a third of it.
#: * **Density.** Training frames hold ~1,145 points in the crop box, lance
#:   ~3,479 in the same box - denser than *any* frame in training, which changes
#:   what every furthest-point sample and ball query returns. Points cannot be
#:   invented, so the lever is the other way: ``aug_thin_min`` at 0.25 makes the
#:   model tolerate 290-1,145 points instead of 570-1,145, widening the band it
#:   is stable across.
#:
#: The sliding-window classifier needs neither, which is the point: its input is
#: a 0.5 m ball with its own local ground subtracted and a fixed point budget,
#: so it never sees either property. This arm asks whether giving the segmenter
#: the same indifference by training closes the gap that separates them.
_SEG_RELIEF = dict(_STRAY_SEG, aug_ground_tilt=0.08, aug_thin_min=0.25)

STRAY_ARMS = [
    Arm("cls-base", "pointnet", _GEOM, "Classifier · stock",
        "Plain PointNet on 0.5 m candidate balls, shape only. The control for "
        "the classifier pair, and the best-scoring classifier setting two "
        "sweeps have found - PointNet++ measured no better and costs three "
        "times as much, and reflectivity measured no better and is the channel "
        "least likely to survive a change of arena."),
    Arm("cls-stray", "pointnet", _GEOM, "Classifier · trained with stray returns",
        "The same classifier with one thing added: 5% of every training "
        "neighborhood becomes a return that sits on no surface, slid along the "
        "line of sight it arrived on. Single-variable against cls-base.",
        overrides=_STRAY_ON),
    Arm("cls-phantom", "pointnet", _GEOM,
        "Classifier · trained against phantom clumps",
        "The classifier with 8% of every training batch replaced outright by a "
        "synthetic phantom clump labelled clear: a loose 3D scatter of sparse "
        "returns with no surface under it. Measured on the competition "
        "recording 4 September 2026, that is what the arena's bad returns "
        "actually present to a 0.5 m candidate ball - 96% of them arrive on "
        "beams pointing up from the sensor, 85% read shorter than the two "
        "neighbouring beams in their own ring, and enough land together that "
        "the generator centres a candidate on the clump. Inside the ball the "
        "clump is not obviously wrong: it measured 0.54 m of vertical extent "
        "and 0.090 m of thickness off its best-fit plane against a rock's "
        "0.46 m and 0.112 m, and the one cue that separates them - 680 returns "
        "against 4,964 - is destroyed by the fixed 256-point sample. So the "
        "gap is a missing sample, not a missing jitter: every ball in all "
        "eleven training recordings is a near-flat patch spanning under 0.12 m "
        "vertically, and nothing has ever taught the model that a tall loose "
        "scatter is not a rock. Single-variable against cls-base.",
        overrides={"aug_phantom_frac": 0.08}),
    Arm("cls-phantom-heavy", "pointnet", _GEOM,
        "Classifier · phantom clumps, heavy dose",
        "The same negative at 20% instead of 8%. Dose is the one free "
        "parameter here and the earlier stray sweep only ever tried one value, "
        "so this brackets it: if the effect is real it should move with the "
        "dose, and if 20% starts costing accuracy on clean volleyball data "
        "that is the ceiling. Single-variable against cls-phantom.",
        overrides={"aug_phantom_frac": 0.20}),
    Arm("cls-both", "pointnet", _GEOM,
        "Classifier · stray points and phantom clumps",
        "Both clutter settings at once: 5% of each ball's points slid along "
        "their line of sight, and 8% of samples replaced by a whole phantom "
        "clump. They attack different halves of the same failure. On the "
        "competition arena the stray jitter took the classifier from 39.6% to "
        "82.2% recall but dropped precision from 54.7% to 37.6% - it learned to "
        "call more things rocks. The clump negative is aimed squarely at that "
        "precision loss, because a false call there is a candidate centred on "
        "fog. If the two compose, this is the deployable one.",
        overrides={"aug_stray_frac": 0.05, "aug_stray_reach": 1.0,
                   "aug_phantom_frac": 0.08}),
    Arm("seg-base", "pointnet2_seg", _GEOM, "Segmentation · deployment settings",
        "The whole-frame segmenter as the rig is meant to run it: 80 epochs, "
        "the finer level geometry, heights measured from the frame's own floor, "
        "a random ground tilt, and frame-centred coordinates. The control for "
        "the segmenter pair.",
        overrides=_STRAY_SEG),
    Arm("seg-stray", "pointnet2_seg", _GEOM,
        "Segmentation · trained with stray returns",
        "The deployment segmenter with 5% of every training frame turned into "
        "stray returns. Real bad returns are mixed pixels and grazing-angle "
        "range errors, so they slide along their own beam and hang in mid-air "
        "rather than scattering evenly - which is why the competition cloud "
        "looks like fog while these eleven recordings do not. Nothing in the "
        "training data has ever held one, so a model has no reason not to read "
        "a clump of them as an object. Single-variable against seg-base: if it "
        "costs nothing on clean volleyball data, it is free insurance for a bin "
        "that is not clean.",
        overrides=dict(_STRAY_SEG, **_STRAY_ON)),
    Arm("seg-capped", "pointnet2_seg", _GEOM, "Segmentation \u00b7 capped loss weight",
        "The deployment segmenter with one change: a rock example counts at "
        "most ten times a clear one in the loss, instead of the raw imbalance "
        "of about 99. That weight is ~99 for any model labelling every point "
        "and ~4.3 for the sliding-window classifier, and the whole gap comes "
        "from the generator discarding 95% of clear candidates for one format "
        "and none for the other. Measured on the BEV grid model, capping it "
        "took a family from 10 of 18 checkpoints going blank on the "
        "competition arena to 0 of 12, and lifted the arena score from 0.385 "
        "to 0.593. The segmenter carries the same weight and shows the same "
        "symptom - correct ranking with confidence collapsing to near zero - "
        "so this asks whether the finding that closed segmentation was partly "
        "measuring an unfixed loss weight. Single-variable against seg-base.",
        overrides=dict(_STRAY_SEG, pos_weight_cap=10.0)),
    Arm("seg-relief", "pointnet2_seg", _GEOM,
        "Segmentation · trained for an arena with relief",
        "The deployment segmenter trained against the two frame-level "
        "differences measured between its training frames and the competition "
        "arena: a floor tipped up to 64 cm across the crop instead of 24 cm, so "
        "the vertical structure it sees matches the 0.46-0.55 m the arena "
        "actually holds rather than the 0.10-0.26 m of a flat court, and "
        "density thinning down to a quarter of a frame instead of a half. "
        "Single-variable pairs are not the point here - both settings attack "
        "the same measured gap, and the question the arm asks is whether that "
        "gap is what has been costing the segmenter its transfer.",
        overrides=_SEG_RELIEF),
]

STRAY_CONTRASTS = [
    ("seg-base", "seg-capped",
     "Does capping the rock-vs-clear loss weight stop the segmenter's "
     "confidence collapsing away from home?"),
    ("cls-base", "cls-stray",
     "Classifier: does training against stray returns cost anything on clean data?"),
    ("cls-base", "cls-phantom",
     "Classifier: does training against phantom clumps cost anything on clean data?"),
    ("cls-stray", "cls-phantom",
     "Is a whole synthetic phantom clump a better negative than nudging 5% of "
     "an ordinary ball's points?"),
    ("cls-stray", "cls-both",
     "Does adding the phantom-clump negative on top of stray jitter cost "
     "anything on clean data?"),
    ("cls-phantom", "cls-phantom-heavy",
     "Classifier: does a heavier dose of phantom clumps help or start costing?"),
    ("seg-base", "seg-stray",
     "Segmentation: does training against stray returns cost anything on clean data?"),
    ("cls-base", "seg-base",
     "Does the deployment segmenter beat the sliding-window classifier?"),
    ("cls-stray", "seg-stray",
     "The same question with both models trained against stray returns."),
    ("seg-base", "seg-relief",
     "Segmentation: does training for an arena with relief in it beat training "
     "on a flat court?"),
]

#: The BEV CNN's deployment-shaped settings. Epochs, patience and batch follow
#: the segmenter's best-measured arm so the two are separated by the model and
#: nothing else; 80 epochs is the only training change four sweeps have found
#: worth keeping (+0.0192, 9 of 11 folds, p = 0.032). Heights come from the
#: frame's own floor for the same reason the segmenter's do - the robot base
#: rode 0.87-0.97 m above the floor here and ~0.37 m above it at competition.
#: The 144-cell grid at 0.10 m holds every cached frame under any heading
#: rotation: measured over the whole cache, the furthest point from a frame's
#: centroid is 6.90 m, and a grid sized to the 8 x 8 m crop box would clip a
#: rotated corner.
_BEV = {"epochs": 80, "patience": 30, "batch": 512,
        "seg_height_ref": "floor", "aug_ground_tilt": 0.03,
        "bev_cell": 0.10, "bev_grid": 144, "bev_width": 32, "bev_depth": 3}

#: Everything except the two channels that say how many returns made a cell.
_BEV_NO_DENSITY = [c for c in BEV_CHANNELS if c not in BEV_DENSITY]

#: Six folds rather than eleven. What matters for this model family is not the
#: volleyball score - it is whether a setting survives the competition arena,
#: and that is measured by scoring every fold's checkpoint on the lance
#: recording afterwards. Transfer is a lottery (half of one classifier
#: setting's runs went flat on lance), so several checkpoints per setting is
#: what buys the answer, and six of them buys more settings than eleven does.
BEV_FOLDS = ["VolleyBallTest2.reslam", "VolleyBallTest6.reslam",
             "VolleyBallTest9.reslam", "VolleyBallTest10.reslam",
             "VolleyBallTest11.reslam", "VolleyBallTest12.reslam"]

BEV_ARMS = [
    Arm("bev-base", "bev_cnn", _GEOM, "BEV CNN \u00b7 control",
        "The frame rasterized to a 0.10 m grid and read by a small U-shaped "
        "convolutional network, one answer per point. No clutter training, raw "
        "return counts.",
        overrides=_BEV),
    Arm("bev-stray", "bev_cnn", _GEOM, "BEV CNN \u00b7 clutter training",
        "The control plus 5% of every training frame turned into returns that "
        "sit on no surface. Free-to-positive for the sliding-window classifier; "
        "measured here it took the grid from 3 of 6 checkpoints going blank on "
        "the competition arena to 6 of 6.",
        overrides=dict(_BEV, **_STRAY_ON)),
    Arm("bev-capped", "bev_cnn", _GEOM, "BEV CNN \u00b7 capped loss weight",
        "Clutter training with a rock example counting at most ten times a "
        "clear one instead of the raw 93. Tested because per-point models rank "
        "rocks correctly away from home while their confidence collapses, and "
        "an uncapped weight was the standing suspect. It did not help: both "
        "checkpoints stayed flat at +0.010 contrast.",
        overrides=dict(_BEV, **_STRAY_ON, pos_weight_cap=10.0)),
    Arm("bev-relative", "bev_cnn", _GEOM, "BEV CNN \u00b7 scale-free density",
        "Built on the control rather than on clutter training, because clutter "
        "training is what kills this family's confidence. Each cell's return "
        "count is divided by its own frame's average, so the channel says "
        "'denser or sparser than the rest of this frame' rather than an "
        "absolute number. The absolute number is not portable: a competition "
        "frame carries 3,198 points in the crop box against a training median "
        "of 1,145, and the fixed point budget turns that into a 1.8x shift in "
        "every cell of the raster.",
        overrides=dict(_BEV, bev_density_norm=True)),
    Arm("bev-capped-nostray", "bev_cnn", _GEOM, "BEV CNN \u00b7 capped, no clutter",
        "Capping the loss weight on the one footing that still produces live "
        "models. Capping was tested with clutter training and failed, but so "
        "does everything with clutter training - this asks whether the cap is "
        "worth anything once that is out of the way.",
        overrides=dict(_BEV, pos_weight_cap=10.0)),
    Arm("bev-base-s43", "bev_cnn", _GEOM, "BEV CNN \u00b7 control (seed 43)",
        "Seed repeat of the control. The control's contrast varies by +/-0.43 "
        "across its six checkpoints, and without a seed repeat there is no way "
        "to say how much of that is the setting and how much is the draw.",
        overrides=_BEV, seed=43),
    Arm("bev-local", "bev_cnn", _GEOM, "BEV CNN \u00b7 short sight",
        "The control with the network two levels deep instead of three, "
        "roughly halving how far each answer can see. The whole-frame segmenter "
        "fails away from home because it reads the frame as one object; this "
        "asks whether a shorter sight line buys the classifier's immunity.",
        overrides=dict(_BEV, bev_depth=2)),
    Arm("bev-capped-relative", "bev_cnn", _GEOM,
        "BEV CNN \u00b7 capped, scale-free density",
        "The two changes that each helped, together. Capping how much a rock "
        "outweighs a clear example took the control from 3 of 6 checkpoints "
        "going blank to 0 of 6 and lifted the arena score above the deployed "
        "classifier's; dividing each cell's count by its own frame's average "
        "also stopped the blanking but left the model firing everywhere on "
        "half its folds. Single-variable against bev-capped-nostray.",
        overrides=dict(_BEV, pos_weight_cap=10.0, bev_density_norm=True)),
]

BEV_CONTRASTS = [
    ("bev-base", "bev-local",
     "Does a shorter sight line cost anything at home?"),
    ("bev-capped-nostray", "bev-capped-relative",
     "On top of the capped loss, does a scale-free density channel add anything?"),
    ("bev-base", "bev-stray",
     "Does clutter training cost the grid anything on clean data?"),
    ("bev-stray", "bev-capped",
     "Does capping how much a rock outweighs a clear example stop the "
     "confidence collapse away from home?"),
    ("bev-base", "bev-relative",
     "Does making the density channel scale-free cost anything at home? "
     "(What it is for only shows up on the competition arena.)"),
    ("bev-base", "bev-capped-nostray",
     "Does capping the loss weight help once clutter training is out of the way?"),
    ("bev-base", "bev-base-s43",
     "Noise floor: the same control setting, two different seeds."),
]

SUITES: dict[str, dict] = {
    "bev": {
        "arms": BEV_ARMS,
        "contrasts": BEV_CONTRASTS,
        "cache": "full-sweep",
        "title": "Rocks on a grid: a convolutional network over BEV rasters",
        "blurb": "Both models trained so far read the frame as a set of points, "
                 "which means neither can see how many returns made any part of "
                 "it - the point tensor is padded by repetition and the real "
                 "count is used only to build a validity mask. A stray return "
                 "that sits on no surface is exactly a place with one return "
                 "where a surface would have many. This sweep rasterizes the "
                 "same frames onto a 0.10 m grid, runs a small U-shaped "
                 "convolutional network over it, and reads the answer back at "
                 "every original point, so it is scored on the segmenter's own "
                 "population and pairs against it fold for fold. The grid is "
                 "built inside the model rather than read from disk so that "
                 "every augmentation - above all the stray-return jitter - "
                 "still reaches it.",
    },
    "reflectivity": {
        "arms": REFLECTIVITY_ARMS,
        "contrasts": REFLECTIVITY_CONTRASTS,
        "cache": "raw-burst",
        "title": "Does reflectivity help?",
        "blurb": "PointNet and PointNet++, with and without the reflectivity "
                 "channel, plus seed repeats that show how big a meaningless "
                 "difference looks.",
    },
    "fullsweep": {
        "arms": FULLSWEEP_ARMS,
        "contrasts": FULLSWEEP_CONTRASTS,
        "cache": "full-sweep",
        "title": "Full sensor sweeps, and per-point segmentation",
        "blurb": "Trained on frames built from whole 20 Hz sensor rotations "
                 "(~1250 points in the crop box) instead of single ~4 ms sensor "
                 "batches (~110 points). Adds the whole-frame segmenter, which "
                 "the batch-sized frames were too sparse to train at all.",
    },
    "stray": {
        "arms": STRAY_ARMS,
        "contrasts": STRAY_CONTRASTS,
        "cache": "full-sweep",
        "title": "Training against stray returns",
        "blurb": "Every recording this project owns was made over flat ground "
                 "the sensor struck steeply, so almost every return in them "
                 "landed on a real surface and nothing has ever taught either "
                 "model that a return can simply be wrong. A competition arena "
                 "7 m across with the sensor 0.57 m up is seen at a grazing "
                 "angle almost everywhere, and is full of them. This sweep asks "
                 "what it costs to train against clutter that is not in the "
                 "data: both model families, each with and without 5% of every "
                 "training sample turned into returns that sit on no surface. "
                 "The segmenter arms also carry every other setting the rig is "
                 "meant to deploy with, so the control is the real deployment "
                 "candidate rather than a historical baseline.",
    },
    "segdense": {
        "arms": SEGDENSE_ARMS,
        "contrasts": SEGDENSE_CONTRASTS,
        "cache": "full-sweep-dense",
        "title": "Feeding the whole-frame segmenter properly",
        "blurb": "Every sensor rotation kept instead of every fourth (~4x the "
                 "frames), 60 epochs instead of 30, and an arm that looks at "
                 "10 cm scales instead of 25 cm. The previous sweep found the "
                 "segmenter starved of frames and stopped before it had "
                 "finished learning; this one removes both limits. Twelve "
                 "folds, not eleven - VolleyBallTest13 joins as a fold here. "
                 "Also carries the arm that fixes the segmenter's height "
                 "reference, which is what made every earlier segmenter read an "
                 "empty arena on the competition recording.",
    },
}


def cache_profile(suite: str) -> str:
    """Which generation profile's cache a suite is meant to train on.

    A suite is a set of settings compared fold by fold, and that only means
    anything if every arm saw the same frames. Two of the three suites here
    were run on different caches, and pointing one at the other's would produce
    a table of numbers that looks fine and answers nothing — so the pairing is
    written down rather than remembered.
    """
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r} (pick from {sorted(SUITES)})")
    return SUITES[suite]["cache"]


def default_cache_dir(suite: str) -> str:
    """Where that cache lives, under the standard one-cache-per-profile root."""
    return os.path.join("training", "caches", cache_profile(suite))


def check_cache_matches(suite: str, cache_dir: str) -> None:
    """Fail loudly when a suite is pointed at a cache cut a different way."""
    from ..profiles import identify as identify_profile
    from .data import load_cache_meta

    meta = load_cache_meta(cache_dir)
    got = meta.get("profile") or identify_profile(meta.get("config_hash", "")) or "unknown"
    want = cache_profile(suite)
    if got != want:
        raise SystemExit(
            f"suite {suite!r} is defined against the {want!r} cache, but "
            f"{cache_dir!r} holds {got!r} frames. Its arms would be trained on "
            "differently-cut data and the paired comparison between them would "
            f"mean nothing. Use --cache-dir {default_cache_dir(suite)}, or build "
            f"it with: rocklabel-train cache --datasets datasets/{want}/*")


def arms_of(suite: str) -> list[Arm]:
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r} (pick from {sorted(SUITES)})")
    return SUITES[suite]["arms"]


def arm_dir(root: str, suite: str, arm: Arm, test_run: str) -> str:
    return os.path.join(root, suite, arm.name, f"loro_{test_run}")


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def run_suite(suite: str, cache_dir: str, root: str, only_arms: list[str] | None,
              extra: dict, fresh: bool = False) -> None:
    """Train every arm of ``suite`` on every leave-one-run-out fold.

    Folds already carrying a test_metrics.json are skipped, so an interrupted
    sweep resumes where it stopped. ``extra`` are config overrides applied to
    every arm (epochs, device, ...); an arm's own overrides win over them, since
    the arm's overrides are the thing being tested.
    """
    from .data import load_cache_meta, loro_folds
    from .engine import default_config, train_fold

    check_cache_matches(suite, cache_dir)
    arms = arms_of(suite)
    if only_arms:
        unknown = [a for a in only_arms if a not in {x.name for x in arms}]
        if unknown:
            raise SystemExit(f"unknown arm(s) {unknown} in suite {suite!r}; "
                             f"available: {[a.name for a in arms]}")
        arms = [a for a in arms if a.name in only_arms]

    runs = sorted(load_cache_meta(cache_dir)["runs"])
    folds = loro_folds(runs)
    total = len(arms) * len(folds)
    done = 0
    print(f"suite {suite!r}: {len(arms)} arms x {len(folds)} folds = {total} runs")

    for arm in arms:
        for fold in folds:
            done += 1
            run_dir = arm_dir(root, suite, arm, fold["test"])
            tag = f"[{done}/{total}] {arm.name} / {fold['test']}"
            if os.path.exists(os.path.join(run_dir, "test_metrics.json")) and not fresh:
                print(f"{tag}: already evaluated, skipping")
                continue
            cfg = arm.config(
                lambda **kw: default_config(cache_dir=cache_dir, **{**extra, **kw}),
                fold["train"], fold["test"])
            print(f"\n=== {tag} ===")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "arm.json"), "w") as f:
                json.dump({"suite": suite, "arm": arm.name, "label": arm.label,
                           "what": arm.what, "model": arm.model,
                           "features": list(arm.features),
                           "overrides": arm.overrides, "seed": arm.seed}, f, indent=2)
            train_fold(cfg, run_dir, resume=not fresh)


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #
def wilcoxon_signed_rank(deltas: np.ndarray) -> tuple[float, float]:
    """(statistic, two-sided exact p) for the signed-rank test on paired deltas.

    Exact rather than normal-approximated because a leave-one-run-out sweep has
    ~10 pairs, where the normal approximation is not trustworthy. Zero deltas
    are dropped (the standard Wilcoxon handling); with fewer than one nonzero
    pair the p-value is 1.0 by definition — no evidence either way.

    Written out here rather than pulled from scipy because scipy is not a
    dependency of this project and one 20-line exact test is cheaper than
    making it one.
    """
    d = np.asarray([x for x in np.asarray(deltas, float) if x != 0.0])
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, float)
    ranks[order] = np.arange(1, n + 1)
    # Average the ranks of tied magnitudes, as the test requires.
    mag = np.abs(d)
    for value in np.unique(mag):
        tied = mag == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    stat = min(w_plus, w_minus)
    if n > 22:  # 2**n enumerations stop being cheap; normal approximation
        mu = n * (n + 1) / 4.0
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        from math import erfc, sqrt
        z = (stat - mu) / max(sigma, 1e-12)
        return stat, float(min(1.0, erfc(abs(z) / sqrt(2))))
    # Exact: enumerate every assignment of signs to the ranks.
    totals = np.zeros(1)
    for r in ranks:
        totals = np.concatenate([totals, totals + r])
    tail = float((totals <= stat + 1e-9).mean())
    return stat, float(min(1.0, 2.0 * tail))


def _fold_metrics(root: str, suite: str, arm_name: str) -> dict[str, dict]:
    """{test_run: test_metrics dict} for every finished fold of one arm."""
    base = os.path.join(root, suite, arm_name)
    out = {}
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name, "test_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            # Every fold trained before the normalized score existed still has
            # the two numbers it is computed from, so backfill rather than
            # leaving the old sweeps out of the new column.
            m.setdefault("baseline_pr_auc", 0.0)
            m.setdefault(NORMALIZED, normalized_pr_auc(m["pr_auc"],
                                                       m["baseline_pr_auc"]))
            out[m.get("test_run", name.replace("loro_", "", 1))] = m
    return out


def collect(root: str, suite: str) -> dict:
    """Everything the report needs, read off disk. No torch, no retraining."""
    arms = arms_of(suite)
    per_arm = {a.name: _fold_metrics(root, suite, a.name) for a in arms}
    folds = sorted({f for m in per_arm.values() for f in m})
    rows = []
    for a in arms:
        m = per_arm[a.name]
        row = {"arm": a.name, "label": a.label, "what": a.what, "model": a.model,
               "features": list(a.features), "seed": a.seed,
               "overrides": a.overrides, "folds_done": len(m), "folds": list(m),
               # What a no-skill model would score on each fold, which is that
               # fold's rock share. Carried so a reader can see how much of a
               # raw per-fold number is just prevalence.
               "prevalence": {f: float(m[f]["baseline_pr_auc"]) for f in sorted(m)}}
        for k in METRICS:
            vals = np.array([m[f][k] for f in sorted(m)], float)
            row[k] = {"mean": float(vals.mean()) if len(vals) else None,
                      "std": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                      "per_fold": {f: float(m[f][k]) for f in sorted(m)}}
        rows.append(row)

    contrasts = []
    for base_name, var_name, question in SUITES[suite]["contrasts"]:
        a, b = per_arm.get(base_name, {}), per_arm.get(var_name, {})
        shared = sorted(set(a) & set(b))
        entry = {"baseline": base_name, "variant": var_name, "question": question,
                 "n_folds": len(shared), "folds": shared}
        for k in METRICS:
            if not shared:
                entry[k] = None
                continue
            d = np.array([b[f][k] - a[f][k] for f in shared], float)
            stat, p = wilcoxon_signed_rank(d)
            entry[k] = {
                "mean_delta": float(d.mean()),
                "std_delta": float(d.std(ddof=1)) if len(d) > 1 else None,
                "median_delta": float(np.median(d)),
                "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                "ties": int((d == 0).sum()),
                "statistic": stat, "p_value": p,
                "per_fold": {f: float(x) for f, x in zip(shared, d)},
            }
        contrasts.append(entry)

    return {"suite": suite, "title": SUITES[suite]["title"],
            "blurb": SUITES[suite]["blurb"], "folds": folds,
            "arms": rows, "contrasts": contrasts}
