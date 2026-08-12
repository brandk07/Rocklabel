"""Run the live control panel's page in node and drive every control on it.

Same arrangement as :mod:`test_dashboard_frontend`: there is no browser here (and
none that can be driven on a Wayland desktop), so ``tests/frontend/`` ships a
minimal DOM. This boots the real ``live.html`` + ``live.js`` against payloads
recorded from the real Flask app, then toggles, drags, types into and clicks
everything the schema advertises.

A render test, not a look test — a human still has to open the page to judge the
design. What it catches cheaply is the expensive class of bug: a control kind
the page cannot draw, a renamed field, and the sync bug that would make the
panel unusable (a 4 Hz poll yanking a slider out from under a drag).

Skips when node is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("flask", reason="the web UI needs the [dash] extra")

from test_live_webui import full_controller  # noqa: E402

FRONTEND = Path(__file__).parent / "frontend"
WEBUI = Path(__file__).parent.parent / "rocklabel" / "live" / "webui"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is needed to run the page")


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> str:
    """A short real recording, so the page gets a real transport to scrub."""
    import time

    from rocklabel.live.config import AppConfig
    from rocklabel.live.pipeline import IngestEngine
    from rocklabel.live.sources.simulated import SimulatedSource
    from rocklabel.live.surfaces.kalman_heightmap import KalmanHeightmap

    path = str(tmp_path_factory.mktemp("webui-fe") / "session.mcap")
    cfg = AppConfig()
    cfg.source.sim_points_per_sec = 20_000
    cfg.source.sim_batch_size = 2_000
    cfg.slam.enabled = False
    engine = IngestEngine(SimulatedSource(cfg), KalmanHeightmap(cfg), cfg)
    engine.start_recording(path)
    engine.start()
    time.sleep(0.6)
    engine.stop()
    engine.stop_recording()
    return path


@pytest.fixture
def fixtures(tmp_path, recording) -> Path:
    """Record the real API responses for a run that has one of everything.

    Deliberately the *fullest* schema — replay transport, viewer, scorer and
    levelling all present — so the harness exercises every control kind the
    page can be asked to draw.
    """
    from rocklabel.live.webui.server import create_app

    import numpy as np

    # fuse_sec: the overhead view needs a surface to look down on, so the
    # fixture has to actually run the pipeline for a moment first.
    ctl = full_controller(recording, fuse_sec=1.2)
    # Give the scorer a real two-lobed distribution and the history a few
    # samples, so the map, the histogram and the trends all have something to
    # draw — a fixture with empty views would let the page render nothing and
    # still pass.
    rng = np.random.default_rng(7)
    ctl._scorer.attach_result(
        rng.uniform(-3, 3, size=(500, 3)),
        np.concatenate([rng.uniform(0.0, 0.4, 380), rng.uniform(0.9, 1.0, 120)]),
    )
    for i in range(6):
        ctl._history._last = 0.0
        ctl._history.sample({"detections": 100 + i, "in_region": 4000 + i * 30,
                             "pass_ms": 30 + i, "rate": 60000 + i * 500})
    try:
        app = create_app(ctl)
        app.config["TESTING"] = True
        recorded = {}
        with app.test_client() as client:
            recorded["/api/schema"] = client.get("/api/schema").get_json()
            recorded["/api/state"] = client.get("/api/state").get_json()
            recorded["/api/scene"] = client.get("/api/scene").get_json()
    finally:
        ctl._engine.source.stop()

    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(recorded))
    return path


def test_page_renders_and_every_control_is_drivable(fixtures):
    proc = subprocess.run(
        ["node", str(FRONTEND / "run_webui.mjs"), str(WEBUI), str(fixtures)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "ok" in proc.stdout


def test_the_recorded_schema_is_the_full_one(fixtures):
    """Guards the fixture itself: a shrunken schema would silently shrink the
    render test's coverage without failing it."""
    schema = json.loads(fixtures.read_text())["/api/schema"]
    ids = {s["id"] for s in schema["sections"]}
    assert ids == {"status", "replay", "view", "level", "model", "region"}
    kinds = {c["kind"] for s in schema["sections"] for c in s["controls"]}
    assert kinds >= {"bool", "int", "float", "enum", "action", "readout"}
