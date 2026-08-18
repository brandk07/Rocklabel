"""Tuning knobs for the alternative SLAM.

Defaults are aimed at the volleyball-court recordings: an outdoor, nearly
flat sand surface with the useful structure (grass edge, fence, trees) sitting
10-25 m out, swept by a hand-held sensor. That scene is *geometrically
degenerate* for the stock tracker — see :mod:`rocklabel.slam.register`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AltSlamConfig:
    """Everything the offline solver can be tuned with."""

    # -- which points take part in registration ----------------------------- #
    #: Range band (m, sensor frame) of points used to align to the map. The
    #: stock tracker stops at 12 m, which throws away exactly the distant
    #: structure that pins down sliding along the sand. Reach for the far stuff.
    reg_range_min: float = 0.6
    reg_range_max: float = 30.0
    #: Edge length (m) of the map/downsample voxel. Registration only — the
    #: final point cloud is never voxelized, so this does not limit rock detail.
    voxel_size: float = 0.20
    #: Seconds of batches pooled into one registration window.
    window_sec: float = 0.10

    # -- the map ------------------------------------------------------------ #
    #: Stop letting a voxel's mean/covariance move after this many points, so a
    #: well-observed voxel is not dragged around by a late bad window.
    map_max_points: int = 60
    #: A voxel needs at least this many points before it can supply a surface
    #: normal; below it the voxel is ignored for point-to-plane matching.
    min_points_normal: int = 8
    #: Hard cap on map voxels (safety valve).
    map_capacity: int = 4_000_000

    # -- ICP ---------------------------------------------------------------- #
    #: Iterations per window. Offline we are ~20x faster than realtime, so this
    #: can be far above the stock 5.
    iterations: int = 25
    #: Correspondence radius (m): starts here, anneals down to ``corr_dist_min``.
    corr_dist_start: float = 0.60
    corr_dist_min: float = 0.06
    #: Geman-McClure scale (m). Residuals past roughly this are treated as
    #: outliers (people walking through, sand speckle) and stop pulling.
    robust_sigma: float = 0.08
    #: A window needs this many surviving correspondences to update the pose.
    min_matches: int = 150
    #: Characteristic lever arm (m) used to put the rotation and translation
    #: blocks of the normal equations into comparable units, so one degeneracy
    #: threshold is meaningful for both.
    lever_arm: float = 8.0
    #: Give every surface *direction* an equal say, so the vast flat sand
    #: surface cannot outvote the sparse fence/tree line that is the only thing
    #: telling us we have not slid sideways.
    normal_balance: bool = True
    #: Bins per axis for that direction bucketing.
    normal_balance_bins: int = 6
    #: Sand is rough and scores low on plane-likeness everywhere, so weighting
    #: purely by it would throw the court away. This is added to every score.
    planarity_floor: float = 0.15

    # -- regularization and degeneracy -------------------------------------- #
    #: Directions whose normalized constraint strength falls below this are
    #: considered unobserved: the solver refuses to move along them and leaves
    #: the IMU/velocity prediction standing. This is the setting that stops
    #: flat sand from letting the pose slide. 0 disables the guard.
    degeneracy_threshold: float = 0.03
    #: Levenberg-Marquardt damping added to every eigenvalue before solving.
    #: Keeps weakly-constrained-but-not-quite-suppressed directions from
    #: producing an enormous step.
    damping: float = 5e-3
    #: Confine ICP's rotation correction to yaw about gravity, trusting the
    #: IMU for tilt. Sounds right — the IMU's quaternion *is* gravity-referenced
    #: — and it is what the stock tracker does. On the hand-swept court
    #: recordings it is the single biggest source of error: swinging the sensor
    #: by hand accelerates it, and an accelerometer cannot tell that apart from
    #: gravity, so the "absolute" tilt is wrong exactly when the sensor is
    #: moving most. Measured on VolleyBallTest1, leaving tilt free is a 2x
    #: improvement (35 mm -> 18 mm of surface thickness), so free is the
    #: default. Turn it back on for a tripod or mast-mounted rig, where the IMU
    #: really is trustworthy and the extra freedom only adds noise.
    lock_roll_pitch: bool = False

    # -- solver passes ------------------------------------------------------ #
    #: Pass 1 is the causal forward solve. Every later pass rebuilds the map
    #: from the current poses and re-registers every window against that
    #: finished map, which is far better than the partial map pass 1 had.
    passes: int = 3
    #: Per-window safety clamps on how far one pass may move a pose.
    max_step_trans: float = 0.40
    max_step_rot_deg: float = 4.0

    # -- output ------------------------------------------------------------- #
    #: Blend each batch's pose between neighbouring window solutions instead of
    #: extrapolating on velocity. Removes the step at every window boundary.
    smooth_batch_poses: bool = True
