"""Score a classifier checkpoint under its own neighborhood/history contract.

The visual audit, the whole-recording map evaluation and the live scorer all
used to cut 0.5 m balls out of one sweep, because every checkpoint was trained
that way. A version-2 checkpoint may want a different radius and older sweeps
as well, and scoring it the old way would measure a model on inputs it never
saw. This is the one place that decides, per checkpoint:

* version 1: exactly the historical path
  (:func:`~rocklabel.dataset.neighborhoods.build_inference_samples` with one
  ``[seed, index]`` RNG), so the deployed reference reproduces its old numbers;
* version 2: candidates from the current sweep, balls from the checkpoint's own
  history support and radius policy, with the same RNG streams as generation.

Callers own history. They hand in a :class:`~rocklabel.dataset.history.Support`
built from their own causal buffer, cropped with their own operational crop.
"""

from __future__ import annotations

import time

import numpy as np

from ..dataset.history import (HistoryPolicy, RadiusPolicy, Support, build_neighborhoods,
                               candidate_centers, candidate_rng, is_legacy, sample_rng,
                               single_sweep_support)


def history_policy(gcfg: dict) -> HistoryPolicy | None:
    """The checkpoint's history policy, or None for a single-sweep model."""
    if is_legacy(gcfg):
        return None
    h = HistoryPolicy.from_generator(gcfg)
    return h if h.uses_history else None


def _forward(model, neigh: np.ndarray, counts: np.ndarray, device, batch: int) -> np.ndarray:
    import torch

    pts = torch.from_numpy(neigh)
    cnt = torch.from_numpy(counts.astype(np.int64))
    out = []
    with torch.no_grad():
        for i in range(0, len(pts), batch):
            logits = model(pts[i:i + batch].to(device), cnt[i:i + batch].to(device))
            out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


def score_classifier(current_xyz: np.ndarray, current_intensity: np.ndarray,
                     gcfg: dict, model, device, index: int, batch: int = 512,
                     support: Support | None = None, max_centers: int | None = None,
                     arena: np.ndarray | None = None,
                     rngs: tuple[np.random.Generator, np.random.Generator] | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, dict]:
    """One scoring pass. Returns ``(centers, probabilities, diagnostics)``.

    ``index`` is the sweep's acquisition index (the RNG seed component);
    ``rngs`` overrides the ``(candidate, sampling)`` generators, which is what
    the live scorer does since it has no stable index. ``support`` defaults to
    the current sweep alone.

    ``diagnostics`` separates preprocessing from network time and counts the
    common candidate set, how much of it could be scored and the support used.
    """
    t0 = time.perf_counter()
    seed = int(gcfg["seed"])
    reads_age = bool(getattr(model, "reads_point_age", False))
    if is_legacy(gcfg):
        from ..dataset.neighborhoods import build_inference_samples

        if reads_age:
            raise ValueError("a point-age model needs a version-2 (history) contract")

        rng = rngs[0] if rngs else np.random.default_rng([seed, int(index)])
        samples = build_inference_samples(current_xyz, current_intensity, gcfg, rng,
                                          max_centers=max_centers, arena=arena)
        t1 = time.perf_counter()
        # The common opportunity set, counted the same way as version 2 so an
        # unscorable candidate never silently leaves a denominator. Timed apart
        # from preprocessing: the historical path never needed it.
        n_cand = len(candidate_centers(current_xyz, gcfg, arena))
        if max_centers is not None:
            n_cand = min(n_cand, int(max_centers))
        t1b = time.perf_counter()
        if samples is None:
            return (np.empty((0, 3), np.float32), np.empty(0, np.float32),
                    {"preprocess_s": t1 - t0, "network_s": 0.0, "candidates": n_cand,
                     "scored": 0, "unscorable": n_cand,
                     "support_points": int(len(current_xyz))})
        probs = _forward(model, samples["neighborhoods"], samples["true_counts"],
                         device, batch)
        return samples["centers_odom"], probs, {
            "preprocess_s": t1 - t0, "network_s": time.perf_counter() - t1b,
            "candidates": n_cand, "scored": int(len(probs)),
            "unscorable": n_cand - int(len(probs)),
            "support_points": int(len(current_xyz)),
            "ball_count": samples["true_counts"].astype(np.int32)}

    crng, srng = rngs if rngs else (candidate_rng(seed, index), sample_rng(seed, index))
    if support is None:
        support = single_sweep_support(current_xyz, current_intensity)
    cand = candidate_centers(current_xyz, gcfg, arena)
    capped = False
    if max_centers is not None and len(cand) > max_centers:
        cand = cand[crng.choice(len(cand), int(max_centers), replace=False)]
        capped = True
    built = build_neighborhoods(cand, support, RadiusPolicy.from_generator(gcfg), srng,
                                diagnostics=False)
    neigh = built["neighborhoods"]
    if reads_age:
        # The age channel, appended exactly as the training loader does.
        neigh = np.concatenate([neigh, built["point_age"].astype(np.float32)[..., None]],
                               axis=-1)
    t1 = time.perf_counter()
    probs = _forward(model, neigh, built["true_counts"], device, batch)
    ok = built["scorable"]
    return built["centers_odom"], probs, {
        "preprocess_s": t1 - t0, "network_s": time.perf_counter() - t1,
        "candidates": int(len(cand)), "scored": int(ok.sum()),
        "unscorable": int((~ok).sum()), "capped": capped,
        "support_points": int(len(support)),
        "slot_points": list(support.slot_points),
        "slot_ages": [float(s.age_s) for s in support.slots],
        "radius": built["radius"], "ball_count": built["ball_count"]}
