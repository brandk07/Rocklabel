"""The package layout rules, as tests.

`rocklabel/README.md` states four rules about where code may live. A rule that
only exists in prose is a rule that quietly rots, and two of these are load
bearing in a way that is invisible until something breaks a long way from the
edit that caused it:

* the dashboard reads the real training defaults, suites and paths straight out
  of ``rocklabel.train`` so they cannot go stale — which only works while those
  modules import without torch;
* the generator, the training stack and the dashboard all have to run on a
  machine with no display, which only works while Open3D stays inside the two
  folders that own the windows.

Both are cheap to check and expensive to discover by accident.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

import pytest

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "rocklabel")

#: Folders allowed to import Open3D at module level. Everything else must defer
#: the import inside a function, so a headless path never touches it.
WINDOW_PACKAGES = ("gui", os.path.join("live", "viz"))

#: Where each top-level module lives after the layout was foldered. Guards the
#: split itself: a module that drifts back to the top level, or into the wrong
#: folder, fails here rather than in a reviewer's memory.
EXPECTED_HOME = {
    "mcap_io": "recording", "lidarrig_io": "recording", "pose": "recording",
    "pipeline": "recording", "inspect_cmd": "recording", "trim": "recording",
    "leveling": "geometry", "accumulate": "geometry", "relief": "geometry",
    "generate": "dataset", "labeling": "dataset", "neighborhoods": "dataset",
    "bev": "dataset",
    "viewer": "gui", "camera": "gui", "labeler": "gui", "preview": "gui",
    "driftcheck": "gui",
}


def _py_files(*rel_dirs: str):
    for rel in rel_dirs:
        base = os.path.join(PKG, rel) if rel else PKG
        for dirpath, dirs, names in os.walk(base):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for n in sorted(names):
                if n.endswith(".py"):
                    yield os.path.join(dirpath, n)


def _module_level_imports(path: str) -> set[str]:
    """Top-level imported module names — not the ones deferred inside a function."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), path)
    found = set()
    for node in tree.body:                      # body only: no function bodies
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_every_module_sits_in_the_folder_the_map_says():
    for mod, home in EXPECTED_HOME.items():
        assert os.path.exists(os.path.join(PKG, home, f"{mod}.py")), \
            f"rocklabel/{home}/{mod}.py is missing — see rocklabel/README.md"
        assert not os.path.exists(os.path.join(PKG, f"{mod}.py")), \
            f"{mod}.py is back at the top level; it belongs in {home}/"


def test_every_package_folder_explains_itself():
    """A folder named after a job has to say what the job is."""
    for folder in ("recording", "geometry", "dataset", "gui", "live", "train",
                   "dashboard", "slam"):
        init = os.path.join(PKG, folder, "__init__.py")
        assert os.path.exists(init), f"rocklabel/{folder}/ has no __init__.py"
        doc = ast.get_docstring(ast.parse(open(init, encoding="utf-8").read()))
        assert doc and len(doc) > 60, \
            f"rocklabel/{folder}/__init__.py needs a real explanation of the folder"


def test_open3d_stays_inside_the_two_folders_that_own_windows():
    """Everything else has to import on a machine with no display."""
    offenders = []
    for path in _py_files(""):
        rel = os.path.relpath(path, PKG)
        if any(rel.startswith(w + os.sep) for w in WINDOW_PACKAGES):
            continue
        if "open3d" in _module_level_imports(path):
            offenders.append(rel)
    assert not offenders, (
        "these import Open3D at module level, which breaks every headless path "
        f"(generate, train, dash): {offenders}. Defer the import inside the "
        "function that opens the window."
    )


def test_torch_stays_inside_the_training_stack():
    """`train/` may use torch; nothing that the dashboard imports may."""
    offenders = []
    for path in _py_files(""):
        rel = os.path.relpath(path, PKG)
        if rel.startswith("train" + os.sep) or rel.startswith("slam" + os.sep):
            continue
        if "torch" in _module_level_imports(path):
            offenders.append(rel)
    assert not offenders, f"torch imported outside the training stack: {offenders}"


@pytest.mark.parametrize("module", [
    "rocklabel.dashboard.spec",
    "rocklabel.dashboard.inventory",
    "rocklabel.train.ablate",
    "rocklabel.train.cli",
])
def test_the_dashboard_side_imports_without_torch(module):
    """Run in a fresh interpreter — an earlier test may already have imported it.

    This is what lets the dashboard quote the real training defaults, the real
    ablation suites and the real default paths instead of copies that go stale.
    Break it and the dashboard becomes torch-dependent, which means the run
    form stops rendering on a machine that only labels.
    """
    code = f"import sys, {module}; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", f"{module} pulled in torch"


def test_the_cli_routes_rather_than_implements():
    """cli.py maps a name to a function; the work lives with its concern.

    Kept honest by size: the moment real work starts landing back in the
    router, this fails and points at the folder it should have gone to.
    """
    path = os.path.join(PKG, "cli.py")
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    top_level_defs = [n for n in tree.body
                      if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert len(top_level_defs) < 15, (
        "cli.py is growing implementation — it should only build the parser and "
        "dispatch. Move the work into recording/, geometry/, dataset/ or gui/."
    )
