"""Robust point-to-plane ICP with an explicit guard against flat geometry.

Why the stock tracker slides on a volleyball court
-------------------------------------------------
The stock registration is point-to-point and reports 99% of its points matched
on these recordings — it is not losing lock, it is *confidently wrong*. A sand
court is close to a single plane. A plane tells you how far you are above it
and nothing whatsoever about where you are along it, so an aligner that only
counts matched points is perfectly happy to slide the whole scan sideways. The
match ratio stays at 99% the entire time it happens.

Two things fix that, and both are here:

**Point-to-plane.** Penalize only the part of the error along the local surface
normal. Sliding within the sand surface then costs nothing (correctly — it is
not an error), so the solver stops pretending flat ground constrains it.

**Degeneracy suppression.** Assemble the normal equations, take their
eigen-decomposition, and measure how strongly each of the six directions is
actually pinned by the geometry. Directions below a threshold are *not solved
for at all* — the IMU and velocity prediction are left standing there instead
of being overwritten by noise. This is the solution-remapping idea from Zhang
et al., "On Degeneracy of Optimization-based State Estimation Problems".

The rotation and translation halves of the problem are in different units
(radians vs metres), so the rotation columns are divided by a characteristic
lever arm first. That puts both halves on the same scale and lets one
threshold mean the same thing for each.
"""

from __future__ import annotations

import math

import numpy as np

from altslam.config import AltSlamConfig


def _normal_balance_weights(normals: np.ndarray, bins: int) -> np.ndarray:
    """Even out how much say each surface *direction* gets.

    On a volleyball court roughly half the visible surface is flat sand, all of
    it pointing the same way, while the fence and tree line — the only things
    that say anything about sliding sideways — are a thin minority far away.
    Left alone, the ground outvotes them and the solve slides.

    So bucket the correspondences by which way their surface faces and give
    each bucket the same total say, regardless of how many points it holds. A
    handful of fence points then counts for as much as an acre of sand. (This
    is normal-space sampling, from Rusinkiewicz & Levoy's survey of ICP
    variants, applied as a weight rather than as a resampling.)
    """
    # Normals are sign-ambiguous, so fold them onto one hemisphere first.
    n = normals * np.where(normals[:, 2:3] < 0.0, -1.0, 1.0)
    q = np.clip(((n + 1.0) * 0.5 * bins).astype(np.int64), 0, bins - 1)
    key = (q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv].astype(np.float64)


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Rotation matrix of an axis-angle vector (Rodrigues)."""
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


class RegistrationResult:
    """Outcome of aligning one window against the map."""

    __slots__ = ("rotation", "translation", "matched", "ratio", "rmse",
                 "ok", "eigenvalues", "suppressed")

    def __init__(self, rotation, translation, matched, ratio, rmse, ok,
                 eigenvalues, suppressed):
        self.rotation = rotation        # (3,3) correction about the sensor
        self.translation = translation  # (3,)
        self.matched = matched          # correspondences on the last iteration
        self.ratio = ratio              # matched / source points
        self.rmse = rmse                # metres, along-normal
        self.ok = ok                    # False => pose left on the prediction
        self.eigenvalues = eigenvalues  # (6,) constraint strength per direction
        self.suppressed = suppressed    # how many of the 6 were unobserved


def register(
    points: np.ndarray,
    center: np.ndarray,
    vmap,
    cfg: AltSlamConfig,
    up: np.ndarray | None = None,
) -> RegistrationResult:
    """Align ``points`` (already at the predicted pose) to ``vmap``.

    Args:
        points: ``(N, 3)`` world-frame source points for this window.
        center: ``(3,)`` sensor position — rotation is solved *about this*, not
            about the world origin, which is what keeps the problem well
            conditioned when the sensor is far from the origin.
        up: ``(3,)`` gravity up-vector in the world frame. Required when
            ``cfg.lock_roll_pitch`` is set; the rotation correction is then
            confined to yaw about it.

    Returns:
        A :class:`RegistrationResult` whose ``rotation``/``translation`` map a
        world point ``p`` to ``R (p - center) + center + t``.
    """
    R = np.eye(3)
    t = np.zeros(3)
    n_src = points.shape[0]
    empty = np.zeros(6)
    if n_src < cfg.min_matches or vmap.size == 0:
        return RegistrationResult(R, t, 0, 0.0, float("nan"), False, empty, 6)

    L = max(cfg.lever_arm, 1e-3)
    sigma2 = cfg.robust_sigma ** 2
    matched = 0
    ratio = 0.0
    rmse = float("nan")
    evals = empty
    suppressed = 6

    # Lookup radius is a property of the map, not of how converged we are: a
    # correctly aligned point can still sit most of a voxel away from that
    # voxel's centroid, so gating the *lookup* tightly would throw away good
    # correspondences. Search generously once...
    search = max(cfg.corr_dist_start, 1.5 * vmap.voxel_size)
    # ...and anneal the acceptance test on the along-normal distance instead,
    # which is the quantity point-to-plane actually cares about.
    gates = np.geomspace(cfg.corr_dist_start, cfg.corr_dist_min, max(1, cfg.iterations))

    for gate in gates:
        p = (points - center) @ R.T + center + t
        valid, cen, nrm, pla = vmap.query(p, search)
        if not np.any(valid):
            return RegistrationResult(R, t, 0, 0.0, rmse, False, evals, 6)

        pv = p[valid]
        nv = nrm[valid]
        # Along-normal residual: the only part of the error we charge for.
        r = np.einsum("ij,ij->i", nv, pv - cen[valid])

        # Accept only correspondences that are already close along the normal.
        near = np.abs(r) <= gate
        matched = int(near.sum())
        ratio = matched / n_src
        if matched < cfg.min_matches:
            return RegistrationResult(R, t, matched, ratio, rmse, False, evals, 6)
        pv, nv, r = pv[near], nv[near], r[near]
        plav, cenv = pla[valid][near], cen[valid][near]
        rmse = float(np.sqrt(np.mean(r * r)))

        # Geman-McClure: quadratic near zero, flattening out for large
        # residuals, so a person walking through cannot drag the solution.
        # Weighted further by how plane-like the target patch is — with a floor,
        # because sand is rough and scores low everywhere, and zeroing it out
        # would discard the whole court.
        w = (cfg.planarity_floor + plav) * (sigma2 / (sigma2 + r * r)) ** 2
        if cfg.normal_balance:
            w = w * _normal_balance_weights(nv, cfg.normal_balance_bins)
        wsum = float(w.sum())
        if wsum < 1e-9:
            return RegistrationResult(R, t, matched, ratio, rmse, False, evals, 6)

        local = pv - center
        # Jacobian row: d(residual)/d(rotation, translation).
        j_rot = np.cross(local, nv) / L  # scaled into translation-like units
        J = np.concatenate([j_rot, nv], axis=1)

        # Normalized normal equations, so eigenvalues mean "average constraint
        # strength per unit weight" and are comparable between recordings.
        Jw = J * w[:, None]
        A = (J.T @ Jw) / wsum
        b = -(Jw.T @ r) / wsum

        # Eigen-decompose the *undamped* matrix, so the eigenvalues still mean
        # "how strongly is this direction observed" and the threshold below
        # keeps its physical reading.
        evals, evecs = np.linalg.eigh(A)

        # Solve in the eigenbasis and simply refuse to move along any direction
        # the geometry does not actually observe.
        keep = evals > cfg.degeneracy_threshold
        suppressed = int((~keep).sum())
        g = evecs.T @ b
        y = np.where(keep, g / np.maximum(evals + cfg.damping, 1e-12), 0.0)
        x = evecs @ y

        omega = x[:3] / L
        dt = x[3:]

        # Confine rotation to yaw about gravity: the IMU quaternion already
        # carries absolute roll/pitch, so anything ICP adds there is drift.
        if cfg.lock_roll_pitch and up is not None:
            u = up / max(np.linalg.norm(up), 1e-12)
            omega = u * float(u @ omega)

        # Clamp so one bad window cannot throw the trajectory.
        max_rot = math.radians(cfg.max_step_rot_deg)
        ang = float(np.linalg.norm(omega))
        if ang > max_rot:
            omega *= max_rot / ang
        mag = float(np.linalg.norm(dt))
        if mag > cfg.max_step_trans:
            dt *= cfg.max_step_trans / mag

        dR = so3_exp(omega)
        R = dR @ R
        t = dR @ t + dt

        if ang < 1e-6 and mag < 1e-5:
            break  # converged

    return RegistrationResult(R, t, matched, ratio, rmse, True, evals, suppressed)
