"""Run the dashboard's page in node and click everything on it.

There is no browser available here (and none that can be driven on a Wayland
desktop), so `tests/frontend/` ships a minimal DOM instead. These tests boot the
real ``index.html`` + ``app.js`` + ``charts.js`` against payloads recorded from
the real Flask app, then exercise every view, tab, command drawer, row expander,
preset and help toggle.

That is a render test, not a look test — a human still has to open the page to
judge the design. What it catches cheaply is the expensive class of bug: a
renamed inventory field, an element id that no longer exists, a chart handed a
NaN by an unevaluated fold.

Skips when node is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("flask", reason="dashboard needs the [dash] extra")

FRONTEND = Path(__file__).parent / "frontend"
DASHBOARD = Path(__file__).parent.parent / "rocklabel" / "dashboard"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is needed to run the page")


def _node(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", str(FRONTEND / script), *args],
        capture_output=True, text=True, timeout=120,
    )


def test_charts_render_every_form_including_awkward_data():
    proc = _node("run_charts.mjs", str(DASHBOARD))
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "ok" in proc.stdout


@pytest.fixture
def fixtures(tmp_path):
    """Record the real API responses for a project that has one of everything."""
    from rocklabel.dashboard.server import create_app

    project_dir = tmp_path / "proj"
    _build_project(project_dir)

    app = create_app(str(project_dir))
    app.config["TESTING"] = True
    recorded = {}
    with app.test_client() as client:
        for path in ("/api/catalog", "/api/state", "/api/sensor", "/api/jobs"):
            recorded[path] = client.get(path).get_json()
        rec = recorded["/api/state"]["inventory"]["recordings"][0]["path"]
        recorded["/api/recording"] = client.get(
            "/api/recording", query_string={"path": rec}).get_json()
        recorded["/api/preview"] = client.post(
            "/api/preview", json={"command_id": "live", "values": {"source": "udp"}}
        ).get_json()

    # The page posts these; stub rather than launching real subprocesses.
    # The live job carries a panel_url, because the control-panel embed is the
    # first thing the page shows after launching one.
    job = {"id": "j0001", "command_id": "live", "title": "Live view", "gui": True,
           "command_line": "rocklabel live --source udp --web-ui --no-browser",
           "status": "running", "panel_url": "http://localhost:8770/",
           "returncode": None, "started": 0, "finished": None, "elapsed": 1.0,
           "line_count": 2}
    # …plus a finished one, so the history list has something to rerun.
    done = job | {"id": "j0002", "command_id": "inspect", "title": "Inspect",
                  "command_line": "rocklabel inspect recordings/run1.mcap",
                  "gui": False, "status": "ok", "returncode": 0,
                  "panel_url": None, "finished": 12.0, "elapsed": 12.0}
    recorded["/api/run"] = {"job": job}
    recorded["/api/jobs"] = {"jobs": [job, done]}
    recorded["/api/state"]["jobs"] = [job, done]
    recorded["/api/jobs/JOB/rerun"] = {"job": done | {"id": "j0003"}}
    recorded["/api/rename"] = {"name": "renamed", "path": "datasets/renamed",
                               "renamed": ["datasets/renamed"]}
    recorded["/api/delete"] = {"path": "datasets/d1", "kind": "dataset",
                               "freed": 1024, "files": 3}
    recorded["/api/jobs/JOB"] = job | {
        "lines": ["[dashboard] $ rocklabel live --source udp", "warming up"],
        "cursor": 2, "dropped": 0,
    }

    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(recorded))
    return path


def _build_project(root: Path) -> None:
    """A project with a recording, labels, a dataset and an evaluated run."""
    (root / "recordings").mkdir(parents=True)
    (root / "recordings" / "run1.mcap").write_bytes(b"not really an mcap")
    (root / "labels").mkdir()
    (root / "labels" / "run1.labels.json").write_text(json.dumps({
        "schema_version": 2, "run_id": "run1", "mcap_file": "run1.mcap",
        "created": "2026-01-01T00:00:00Z", "intensity_available": True,
        "rocks": [{"id": 1, "shape": "sphere"}, {"id": 2, "shape": "box"}],
    }))
    ds = root / "datasets" / "d1"
    ds.mkdir(parents=True)
    (ds / "manifest.json").write_text(json.dumps({
        "config_hash": "abc123def456789", "generated": "2026-01-02T00:00:00Z",
        "config": {"generator": {"frame_stride": 5}},
        "runs": {"run1": {"run_id": "run1", "frames_kept": 100, "point_samples": 500,
                          "bev_frames": 100, "rock_count": 2,
                          "sample_labels": {"rock": 120, "clear": 380}}},
    }))
    # Two models x two folds, one of them deliberately unevaluated so the
    # comparison chart has to cope with a NaN column.
    for model in ("pointnet", "pointnet2"):
        for fold, evaluated in (("run1", True), ("run2", model == "pointnet")):
            run = root / "training" / "runs" / f"{model}_loro_{fold}"
            run.mkdir(parents=True)
            (run / "config.json").write_text(json.dumps(
                {"model": model, "test_run": fold, "train_runs": ["other"],
                 "epochs": 30, "batch": 256, "lr": 0.001}))
            (run / "best.pt").write_bytes(b"weights")
            (run / "history.csv").write_text(
                "epoch,train_loss,val_loss,val_pr_auc\n"
                + "".join(f"{e},{0.5 / (e + 1)},{0.4 / (e + 1)},0.96\n" for e in range(8)))
            if evaluated:
                (run / "test_metrics.json").write_text(json.dumps({
                    "f1": 0.87, "precision": 0.92, "recall": 0.83, "pr_auc": 0.96,
                    "roc_auc": 0.98, "accuracy": 0.94, "threshold": 0.74,
                    "tp": 10, "fp": 2, "fn": 3, "tn": 40, "n": 55,
                    "rock_frac": 0.24, "baseline_accuracy": 0.75,
                    "model": model, "test_run": fold,
                }))
    cache = root / "training" / "cache"
    cache.mkdir(parents=True)
    (cache / "meta.json").write_text(json.dumps({"runs": {"run1": {}, "run2": {}}}))
    results = root / "training" / "results"
    results.mkdir(parents=True)
    (results / "comparison.png").write_bytes(b"\x89PNG\r\n\x1a\n")


def test_the_whole_page_renders_and_every_control_is_clickable(fixtures):
    proc = _node("run_dashboard.mjs", str(DASHBOARD), str(fixtures))
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert proc.stdout.startswith("ok"), proc.stdout


def test_the_harness_actually_fails_on_a_broken_page(fixtures, tmp_path):
    """Guard against a harness that passes because it renders nothing."""
    broken = tmp_path / "broken"
    shutil.copytree(DASHBOARD, broken)
    app_js = broken / "static" / "app.js"
    app_js.write_text(app_js.read_text().replace("const t = S.inv.totals;",
                                                 "const t = S.inv.nope;"))
    proc = _node("run_dashboard.mjs", str(broken), str(fixtures))
    assert proc.returncode == 1
    assert "FAILURES" in (proc.stderr + proc.stdout)
