"""Optional training stack (`pip install -e .[train]`): PointNet / PointNet++
rock classifiers trained on the format-A neighborhood datasets.

Everything torch-dependent is imported lazily so `import rocklabel` and the
base CLI keep working without the [train] extra installed.
"""

from ..neighborhoods import FEATURES

#: Single source of truth for every training setting, kept here (torch-free)
#: rather than in engine.py so the argparse layer can quote the real defaults
#: without importing torch.
#:
#: It lives in exactly one place on purpose. When cli.py carried its own
#: argparse defaults they shadowed engine's for every overlapping key -
#: ``--patience`` silently stayed at 6 after engine's default moved to 10, so a
#: whole sweep ran with the setting it was supposed to have changed. The CLI
#: now defaults every one of these to None and lets this dict win.
TRAIN_DEFAULTS: dict = {
    "model": "pointnet",
    "features": list(FEATURES),
    "tnet": False,
    "tnet_reg": 1e-3,
    "dropout": None,
    "cache_dir": "training/cache",
    "train_runs": [],
    "test_run": "",
    "val_frac": 0.15,
    "gap_frames": 25,
    # Sized in seconds, not frames: 25 kept frames is 0.54 s on this sensor,
    # which is not a buffer between two blocks of a moving robot's data.
    "gap_seconds": 2.0,
    "epochs": 30,
    "batch": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    # Has to outlast the cosine schedule's slow tail, or every run stops while
    # the LR is still high and no fold sees the low-LR refinement phase.
    "patience": 10,
    "augment": True,
    # Wider than the ~0.045 gap between this arena's rock and clear intensity
    # levels, so the absolute reflectivity cue is denied and only the
    # within-neighborhood contrast survives.
    "aug_intensity_gain": 0.25,
    "aug_intensity_shift": 0.10,
    "aug_thin_min": 0.5,
    "seed": 42,
    "device": None,
}
