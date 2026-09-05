"""Optional training stack (`pip install -e .[train]`): PointNet / PointNet++
rock classifiers trained on the format-A neighborhood datasets.

Everything torch-dependent is imported lazily so `import rocklabel` and the
base CLI keep working without the [train] extra installed.
"""

from ..dataset.neighborhoods import FEATURES

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
    # Segmentation only: how many centroids each of the three downsampling
    # levels keeps, and how wide a ball it pools over (metres). These used to be
    # written into the model class, which made them unreachable without editing
    # it - and the built-in finest radius, 0.25 m, is about the size of a whole
    # rock, so the smallest scale the model looked at was already bigger than
    # the object it was hunting. Ignored by the two classifiers.
    "seg_npoints": [512, 128, 32],
    "seg_radii": [0.25, 0.6, 1.4],
    # Segmentation only: what "height zero" means. "floor" subtracts the frame's
    # own ground before the model sees it, so how high the robot base rides above
    # the floor stops mattering; "base" is the original behavior and is what a
    # checkpoint trained before this setting existed still loads as. The base is
    # not a stable reference - it sat 0.87-0.97 m above the floor across every
    # training recording and ~0.37 m above it in the competition bag, and a
    # shift that size takes the old segmenter's best confidence to 0.0026.
    "seg_height_ref": "floor",
    # What coordinates each PointNet++ grouping MLP receives. ``scene`` keeps
    # the historical absolute xyz alongside centroid-relative offsets;
    # ``frame`` recenters xy on each frame while retaining position within it,
    # making the model translation invariant without removing the coordinate
    # information its segmentation decoder needs. ``local`` is the stricter
    # research variant that removes absolute coordinates at every layer.
    "seg_coord_ref": "scene",
    # Segmentation only: random ground tilt, in metres of rise per metre. 0.03
    # is ~24 cm across the 8 m crop, which covers the unevenness a dug-over
    # regolith bin actually shows. Uniform height shifts are deliberately not
    # jittered - "floor" already cancels them exactly.
    "aug_ground_tilt": 0.03,
    # Fraction of each frame replaced by stray returns that are not on any
    # surface. Every training recording was made standing over flat ground the
    # sensor struck steeply, so almost every return landed where it should and
    # nothing ever taught these models that a return can be wrong. The
    # competition arena is 7 m across with the sensor 0.57 m up, so most of it
    # is hit at a grazing angle: measured against the close-range truth on the
    # lance recording, 1.0-2.4% of returns inside 1.5 m land more than 25 cm
    # off the surface, rising to 5-7% between 1.5 and 6 m and 13% beyond that.
    # 0.05 sits in the middle of that range. 0 = off, which is what every
    # checkpoint trained before this setting existed used.
    "aug_stray_frac": 0.0,
    # How far a stray is thrown, in metres. A real one is a mixed pixel or a
    # grazing-angle range error, so it slides along the beam rather than
    # appearing anywhere: the displacement is drawn heavy-tailed, median about
    # a third of this and a long tail past it, matching the 305 mm (2-4 m) and
    # 1277 mm (4-8 m) per-beam range wander measured while the robot was parked.
    "aug_stray_reach": 1.0,
    # Classifier only. Probability that a training sample is replaced outright
    # by a synthetic "phantom clump" labelled clear: a loose 3D scatter of
    # sparse returns with no surface under it. Measured on the competition
    # recording, that is the shape the arena's bad returns actually present to
    # a 0.5 m candidate ball, and no ball in any of the eleven training
    # recordings is one - referenced to its own lowest point, the median rock
    # spans 0.115 m vertically and the median clear ball 0.094 m. 0 = off.
    "aug_phantom_frac": 0.0,
    # Vertical extent (m) of a synthetic clump, drawn 0.5-1.5x this. 0.54 m is
    # what a phantom-centred ball measured on the competition arena.
    "aug_phantom_extent": 0.54,
    # BEV CNN only: the grid it rasterizes onto and the size of the network
    # over it. 0.10 m cells match the stored BEV rasters; a 144-cell grid
    # (+/- 7.2 m) holds every cached frame under any heading rotation, which
    # a grid sized to the 8 x 8 m crop box would not - rotating that box about
    # the robot base swings its corners outside. ``bev_channels`` selects the
    # per-cell inputs by name; None means all of them.
    "bev_cell": 0.10,
    "bev_grid": 144,
    "bev_width": 32,
    "bev_depth": 3,
    "bev_channels": None,
    # Divide each cell's return count by this frame's own average before the
    # network sees it. Off reproduces the first runs; on is what a frame from a
    # different arena needs, because the point budget makes the absolute count
    # mean different things in different places.
    "bev_density_norm": False,
    # Ceiling on how much one rock example outweighs one clear one in the loss.
    # None keeps the raw class imbalance, which every run before this used.
    "pos_weight_cap": None,
    "seed": 42,
    "device": None,
}
