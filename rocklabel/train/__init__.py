"""Optional training stack (`pip install -e .[train]`): PointNet / PointNet++
rock classifiers trained on the format-A neighborhood datasets.

Everything torch-dependent is imported lazily so `import rocklabel` and the
base CLI keep working without the [train] extra installed.
"""
