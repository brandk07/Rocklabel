# Repository guidance for model-comparison claims

## Classifier versus segmentation

Do not assume that either model family is better. Before claiming that a
segmenter and sliding-window classifier perform similarly in deployment, run
the labelled visual audit on the competition/Lance recording:

```bash
rocklabel-train visual-audit RECORDING \
  --labels LABELS.json \
  --model-a CLASSIFIER/best.pt \
  --model-b SEGMENTER/best.pt \
  --out training/reports/visual-audit
```

Read `summary.md`, `per-rock.csv`, and every automatically selected image in
`cases/` before interpreting aggregate metrics. In particular:

- Weight distinct physical rocks equally. Repeated sightings of one rock are
  not independent examples and must not dominate the conclusion.
- Judge accumulated-map completeness, persistent misses, fragmentation, and
  false detections. A two-cell contact is not equivalent to mapping a rock.
- Use the full operational floor band for the primary result. A narrow crop is
  a diagnostic ablation and cannot be presented as a deployment solution when
  it removes obstacle geometry from the map.
- Report stored-threshold behavior separately from an equal-false-positive
  threshold sweep. PR-AUC measures ranking and cannot establish that the live
  probability scale or deployed threshold works.
- Classifier candidate centers and segmenter output points are different native
  populations. Compare them only after their real inference paths, and state
  the shared spatial reduction used by the comparison.
- Volleyball held-out results and the rock-enriched matched-candidate report
  are useful in-domain measurements, not evidence of Lance deployment parity.

A future segmenter may overturn the current result. It does so by producing
comparable per-rock coverage and false-detection behavior in this audit, not by
matching an unrelated aggregate score.
