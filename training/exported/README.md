# `training/exported/` — models packaged to run outside Python

The deployable end of the project. Everything else here is working material;
these are the artifacts you would actually ship to the robot.

```
exported/<run name>/
  model.onnx        portable format — runs under almost any runtime
  model.pt          TorchScript — runs under PyTorch with no project code
  metadata.json     everything needed to feed it correctly (see below)
```

## Making one

```bash
rocklabel-train export training/experiments/<experiment>/<setting>/loro_<run>/best.pt
```

Or use the **Export checkpoint** card in the dashboard, which lets you pick the
model from a grouped list showing what each one scored.

## Why `metadata.json` matters

A model on its own is not enough to get right answers from. The metadata
records the geometry the samples were built with — the neighbourhood radius,
the number of points per sample, which input channels the model expects, and
the decision threshold chosen on validation data. Feed a model points built to
different geometry and it will return confident nonsense rather than an error.

## In git, on purpose

Unlike the caches and the trained folds, these are kept in the repository. They
are small, and they are the point of the whole exercise.
