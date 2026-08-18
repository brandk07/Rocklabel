"""Tests for `rocklabel dash`: the command catalog, the job runner, and the API.

The dashboard's contract with the browser is "every endpoint the page calls
returns the shape the page destructures", so these tests hit the real Flask app
with a real project tree rather than mocking it. Flask is an optional extra, so
the whole module skips when it is missing.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

flask = pytest.importorskip("flask", reason="dashboard needs the [dash] extra")

from rocklabel.dashboard import inventory, jobs as jobs_mod, spec  # noqa: E402
from rocklabel.dashboard.jobs import JobManager  # noqa: E402


# --------------------------------------------------------------------------- #
# catalog
# --------------------------------------------------------------------------- #
def test_every_command_has_prose_and_a_known_stage():
    stage_ids = {s["id"] for s in spec.STAGES}
    assert spec.COMMANDS, "the catalog must not be empty"
    for cmd in spec.COMMANDS:
        assert cmd.stage in stage_ids, f"{cmd.id} has an unknown stage"
        assert cmd.bin in ("rocklabel", "rocklabel-train")
        # The whole point of the drawer is the explanation, so it is required.
        assert len(cmd.what) > 40, f"{cmd.id} needs a real 'what it does'"
        assert len(cmd.why) > 30, f"{cmd.id} needs a real 'when you reach for it'"
        assert cmd.tagline and not cmd.tagline.endswith(".."), cmd.id
        for p in cmd.params:
            assert p.kind in ("path", "dir", "outpath", "outdir", "enum", "multi",
                              "int", "float", "bool", "text"), (cmd.id, p.name)
            if p.kind in ("enum", "multi"):
                assert p.choices, f"{cmd.id}.{p.name} needs choices"
            if p.kind == "enum":
                assert p.choices, f"{cmd.id}.{p.name} is an enum with no choices"
            assert p.help or p.kind == "bool" or p.arg is None or p.label, p.name


def test_catalog_matches_the_real_cli_parsers():
    """Every catalog flag must actually exist on the argparse parser.

    This is the test that catches the catalog drifting after someone renames a
    CLI flag — the failure mode would otherwise be a job that dies on
    "unrecognized arguments" only when a user clicks Run.
    """
    from rocklabel.cli import build_parser
    from rocklabel.train.cli import build_parser as build_train_parser

    parsers = {"rocklabel": build_parser(), "rocklabel-train": build_train_parser()}
    subs = {}
    for name, parser in parsers.items():
        for action in parser._actions:
            if hasattr(action, "choices") and isinstance(action.choices, dict):
                for sub_name, sub_parser in action.choices.items():
                    subs[(name, sub_name)] = sub_parser

    for cmd in spec.COMMANDS:
        sub = subs.get((cmd.bin, cmd.sub))
        assert sub is not None, f"{cmd.cli} is not a real subcommand"
        known = {opt for action in sub._actions for opt in action.option_strings}
        for p in cmd.params:
            if p.arg is not None:
                assert p.arg in known, f"{cmd.cli} has no flag {p.arg}"


def test_build_argv_orders_positionals_first_and_skips_blanks():
    cmd = spec.COMMANDS_BY_ID["generate"]
    argv = spec.build_argv(cmd, {"mcap": "recordings/a.mcap", "out": "datasets/d",
                                 "labels": "", "config": None})
    assert argv == ["rocklabel", "generate", "recordings/a.mcap", "--out", "datasets/d"]


def test_build_argv_handles_both_repeat_styles():
    # --topic is a repeatable flag; --datasets is one flag with nargs="+".
    trim = spec.build_argv(spec.COMMANDS_BY_ID["trim"],
                           {"mcap": "a.mcap", "out": "b.mcap", "topic": "/imu, /odom"})
    assert trim.count("--topic") == 2 and "/imu" in trim and "/odom" in trim

    cache = spec.build_argv(spec.COMMANDS_BY_ID["train-cache"],
                            {"datasets": "datasets/one, datasets/two"})
    assert cache == ["rocklabel-train", "cache", "--datasets",
                     "datasets/one", "datasets/two"]


def test_multi_select_emits_one_flag_with_every_ticked_choice():
    argv = spec.build_argv(spec.COMMANDS_BY_ID["train-train"],
                           {"model": "pointnet", "test_run": "r1",
                            "features": "dx, dy, dz"})
    i = argv.index("--features")
    assert argv[i + 1:i + 4] == ["dx", "dy", "dz"]
    # Leaving it empty falls through to the CLI's own default (all channels).
    plain = spec.build_argv(spec.COMMANDS_BY_ID["train-train"],
                            {"model": "pointnet", "test_run": "r1", "features": ""})
    assert "--features" not in plain


def test_build_argv_rejects_a_missing_required_value():
    with pytest.raises(ValueError, match="required"):
        spec.build_argv(spec.COMMANDS_BY_ID["generate"], {"mcap": "a.mcap"})


def test_bool_params_emit_a_bare_flag():
    argv = spec.build_argv(spec.COMMANDS_BY_ID["trim"],
                           {"mcap": "a.mcap", "out": "b.mcap", "all_topics": True})
    assert "--all-topics" in argv and argv[argv.index("--all-topics") - 1] != "--all-topics"
    off = spec.build_argv(spec.COMMANDS_BY_ID["trim"],
                          {"mcap": "a.mcap", "out": "b.mcap", "all_topics": False})
    assert "--all-topics" not in off


# --------------------------------------------------------------------------- #
# The live control panel: launched from a browser, controlled from that browser
# --------------------------------------------------------------------------- #
def test_panel_commands_get_the_web_ui_flags():
    """Launched from the dashboard, live/record serve their browser panel.

    --no-browser too: the dashboard embeds the page itself, and a child opening
    its own tab on top of that is noise.
    """
    for cid in ("live", "record"):
        cmd = spec.COMMANDS_BY_ID[cid]
        assert cmd.panel, f"{cid} should be a panel command"
        argv = spec.build_argv(cmd, {"source": "udp"})
        assert "--web-ui" in argv and "--no-browser" in argv


def test_only_the_live_rig_commands_serve_a_panel():
    """--web-ui is a live-pipeline flag; putting it on `inspect` would not parse."""
    panels = {c.id for c in spec.COMMANDS if c.panel}
    assert panels == {"live", "record"}
    assert "--web-ui" not in spec.build_argv(spec.COMMANDS_BY_ID["inspect"],
                                             {"mcap": "a.mcap"})


def test_the_default_port_is_left_implicit_but_others_are_spelled_out():
    """So the drawer's preview matches what runs, for one live job at a time."""
    cmd = spec.COMMANDS_BY_ID["live"]
    default = spec.build_argv(cmd, {}, panel_port=spec.PANEL_DEFAULT_PORT)
    assert "--web-port" not in default

    second = spec.build_argv(cmd, {}, panel_port=spec.PANEL_DEFAULT_PORT + 1)
    assert second[second.index("--web-port") + 1] == str(spec.PANEL_DEFAULT_PORT + 1)


def test_panel_port_matches_the_cli():
    """The preview hardcodes the port so it need not import the live stack."""
    pytest.importorskip("open3d")
    from rocklabel.live.webui.server import DEFAULT_PORT

    assert spec.PANEL_DEFAULT_PORT == DEFAULT_PORT


def test_concurrent_live_jobs_get_different_panel_ports():
    """Two viewers at once must not fight over one port."""
    from rocklabel.dashboard.server import _free_panel_port

    assert _free_panel_port(set()) == spec.PANEL_DEFAULT_PORT
    assert _free_panel_port({spec.PANEL_DEFAULT_PORT}) == spec.PANEL_DEFAULT_PORT + 1
    assert _free_panel_port({spec.PANEL_DEFAULT_PORT,
                             spec.PANEL_DEFAULT_PORT + 1}) == spec.PANEL_DEFAULT_PORT + 2


def test_panel_url_is_latched_from_the_childs_own_announcement(tmp_path):
    """The dashboard embeds the URL the process says it bound, not one it
    assembled — and only once the process has said it, which is the readiness
    signal that stops the frame pointing at a dead port."""
    job = jobs_mod.Job(id="j1", command_id="live", title="Live view", argv=[],
                       cwd=str(tmp_path), log_path=str(tmp_path / "j1.log"),
                       panel_port=8770)
    assert job.summary()["panel_url"] is None

    job._emit("[rocklabel] live model: pointnet (threshold 0.74; ...)")
    assert job.panel_url is None, "an unrelated line must not be read as a URL"

    job._emit("[rocklabel] control panel: http://localhost:8770/")
    assert job.summary()["panel_url"] == "http://localhost:8770/"

    job._emit("[rocklabel] control panel: http://localhost:9999/")
    assert job.panel_url == "http://localhost:8770/", "the first announcement wins"


def test_panel_url_survives_interleaved_output(tmp_path):
    """stdout and stderr are merged, so Flask's banner regularly lands *inside*
    the announcement line. Observed in a real two-job run, where the URL came
    out as 'http://localhost:8771/ * Serving Flask app ...'."""
    job = jobs_mod.Job(id="j1", command_id="live", title="Live view", argv=[],
                       cwd=str(tmp_path), log_path=str(tmp_path / "j1.log"),
                       panel_port=8771)
    job._emit("[rocklabel] control panel: http://localhost:8771/ * Serving Flask "
              "app 'rocklabel.live.webui.server'")
    assert job.panel_url == "http://localhost:8771/"


def test_a_job_with_no_panel_never_grows_a_url(tmp_path):
    job = jobs_mod.Job(id="j1", command_id="inspect", title="Inspect", argv=[],
                       cwd=str(tmp_path), log_path=str(tmp_path / "j1.log"))
    job._emit("[rocklabel] control panel: http://localhost:8770/")
    assert job.panel_url is None


def test_presets_only_reference_real_params():
    for cmd in spec.COMMANDS:
        names = {p.name for p in cmd.params}
        for preset in cmd.presets:
            assert set(preset.values) <= names, f"{cmd.id}: {preset.name}"


def test_every_command_that_replays_a_recording_offers_levelling():
    """All three must be givable the same answer.

    Rock centers are stored in world coordinates, so a form that let you level
    while labelling but not while generating would misplace every one of them.
    """
    for cid in ("label", "driftcheck", "generate"):
        names = {p.name for p in spec.COMMANDS_BY_ID[cid].params}
        assert {"level", "mount_roll", "mount_pitch"} <= names, cid


def test_levelling_is_left_to_the_config_when_the_form_is_blank():
    argv = spec.build_argv(spec.COMMANDS_BY_ID["label"], {"mcap": "a.mcap"})
    assert "--level" not in argv


def test_a_manual_mount_angle_reaches_the_cli_intact():
    argv = spec.build_argv(spec.COMMANDS_BY_ID["label"], {
        "mcap": "a.mcap", "level": "manual", "mount_roll": -0.64, "mount_pitch": 39.25,
    })
    from rocklabel.cli import build_parser

    args = build_parser().parse_args(argv[1:])
    assert (args.level, args.mount_roll, args.mount_pitch) == ("manual", -0.64, 39.25)


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #
@pytest.fixture
def project(tmp_path):
    """A miniature project tree with one of everything."""
    (tmp_path / "recordings").mkdir()
    (tmp_path / "recordings" / "run1.mcap").write_bytes(b"not really an mcap")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "run1.labels.json").write_text(json.dumps({
        "schema_version": 2, "run_id": "run1", "mcap_file": "run1.mcap",
        "created": "2026-01-01T00:00:00Z", "intensity_available": True,
        "rocks": [{"id": 1, "shape": "sphere"}, {"id": 2, "shape": "box"}],
    }))
    ds = tmp_path / "datasets" / "d1"
    ds.mkdir(parents=True)
    (ds / "manifest.json").write_text(json.dumps({
        "config_hash": "abc123def456789", "generated": "2026-01-02T00:00:00Z",
        "config": {"generator": {"frame_stride": 5}},
        "runs": {"run1": {"run_id": "run1", "frames_kept": 100, "point_samples": 500,
                          "bev_frames": 100, "rock_count": 2,
                          "sample_labels": {"rock": 120, "clear": 380}}},
    }))
    run = tmp_path / "training" / "runs" / "pointnet_loro_run1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(
        {"model": "pointnet", "test_run": "run1", "train_runs": ["run2"], "epochs": 30}))
    (run / "test_metrics.json").write_text(json.dumps(
        {"f1": 0.87, "precision": 0.92, "recall": 0.83, "pr_auc": 0.96,
         "threshold": 0.74, "tp": 1, "fp": 2, "fn": 3, "tn": 4, "n": 10}))
    (run / "best.pt").write_bytes(b"weights")
    (run / "history.csv").write_text("epoch,train_loss,val_loss\n0,0.5,0.4\n1,0.3,0.3\n")
    cache = tmp_path / "training" / "cache"
    cache.mkdir(parents=True)
    (cache / "meta.json").write_text(json.dumps({"runs": {"run1": {}, "run2": {}}}))
    return tmp_path


def test_snapshot_links_recordings_to_their_labels(project):
    snap = inventory.snapshot(str(project))
    assert snap["totals"]["recordings"] == 1
    assert snap["totals"]["recordings_labeled"] == 1
    assert snap["totals"]["rocks"] == 2
    assert snap["recordings"][0]["labels"] == os.path.join("labels", "run1.labels.json")


def test_snapshot_summarizes_datasets_runs_and_checkpoints(project):
    snap = inventory.snapshot(str(project))
    assert snap["totals"]["samples"] == 500
    assert snap["totals"]["rock_samples"] == 120
    assert snap["datasets"][0]["config_hash"] == "abc123def456"  # truncated for display
    assert snap["totals"]["runs_complete"] == 1
    assert snap["totals"]["best_f1"] == pytest.approx(0.87)
    assert [c["name"] for c in snap["checkpoints"]] == ["pointnet_loro_run1/best.pt"]
    assert [r["name"] for r in snap["cache_runs"]] == ["run1", "run2"]
    assert snap["runs"][0]["epochs_run"] == 2  # from history.csv


def test_last_pt_is_listed_but_flagged_unusable(project):
    # last.pt has no config/generator/threshold, so every consumer of the
    # checkpoint pickers would KeyError on it — the UI greys it out.
    (project / "training" / "runs" / "pointnet_loro_run1" / "last.pt").write_bytes(b"x")
    cks = {c["name"]: c for c in inventory.snapshot(str(project))["checkpoints"]}
    assert list(cks) == ["pointnet_loro_run1/best.pt", "pointnet_loro_run1/last.pt"]
    assert cks["pointnet_loro_run1/best.pt"]["disabled"] is False
    assert cks["pointnet_loro_run1/last.pt"]["disabled"] is True
    assert cks["pointnet_loro_run1/last.pt"]["note"]


def test_snapshot_survives_a_dataset_with_no_manifest(project, tmp_path):
    (project / "datasets" / "junk").mkdir()
    snap = inventory.snapshot(str(project))
    junk = next(d for d in snap["datasets"] if d["name"] == "junk")
    assert junk["has_manifest"] is False and junk["runs"] == []


DS1 = os.path.join("datasets", "d1")
MCAP1 = os.path.join("recordings", "run1.mcap")
LABELS1 = os.path.join("labels", "run1.labels.json")


def test_rename_dataset_moves_the_folder_and_leaves_its_contents_alone(project):
    entry = inventory.rename(str(project), DS1, "gravel")
    assert entry["name"] == "gravel"
    assert entry["path"] == os.path.join("datasets", "gravel")
    assert (project / "datasets" / "gravel" / "manifest.json").exists()
    assert not (project / "datasets" / "d1").exists()
    snap = inventory.snapshot(str(project))
    assert [d["name"] for d in snap["datasets"]] == ["gravel"]
    assert snap["totals"]["samples"] == 500  # the manifest is untouched


@pytest.mark.parametrize("bad", ["../escaped", "sub/dir", ".hidden", "", "no spaces"])
def test_rename_refuses_a_name_that_is_not_a_plain_filename(project, bad):
    for rel in (DS1, MCAP1, LABELS1):
        with pytest.raises(ValueError):
            inventory.rename(str(project), rel, bad)
    assert (project / "datasets" / "d1").is_dir()
    assert (project / "recordings" / "run1.mcap").is_file()


def test_rename_refuses_to_clobber_an_existing_name(project):
    (project / "datasets" / "d2").mkdir()
    (project / "recordings" / "run2.mcap").write_bytes(b"x")
    with pytest.raises(ValueError, match="already exists"):
        inventory.rename(str(project), DS1, "d2")
    with pytest.raises(ValueError, match="already exists"):
        inventory.rename(str(project), MCAP1, "run2")
    assert (project / "datasets" / "d1" / "manifest.json").exists()
    assert (project / "recordings" / "run1.mcap").is_file()


def test_rename_only_touches_the_three_project_folders(project):
    """`datasets/` itself, a training run, and a config file are not renameable."""
    (project / "config.yaml").write_text("topics: {}\n")
    for rel in ("labels", "datasets", "config.yaml",
                os.path.join("training", "runs", "pointnet_loro_run1")):
        with pytest.raises((ValueError, FileNotFoundError)):
            inventory.rename(str(project), rel, "whatever")
    assert (project / "labels").is_dir()


def test_renaming_a_recording_carries_its_labels_and_retags_them(project):
    entry = inventory.rename(str(project), MCAP1, "backroom")
    assert entry["path"] == os.path.join("recordings", "backroom.mcap")
    assert sorted(entry["renamed"]) == [os.path.join("labels", "backroom.labels.json"),
                                        os.path.join("recordings", "backroom.mcap")]
    assert not (project / "labels" / "run1.labels.json").exists()
    data = json.loads((project / "labels" / "backroom.labels.json").read_text())
    # Exactly what the next `label` session would have written for this stem.
    assert data["run_id"] == "backroom" and data["mcap_file"] == "backroom.mcap"
    assert len(data["rocks"]) == 2 and data["schema_version"] == 2  # nothing else lost
    snap = inventory.snapshot(str(project))
    assert snap["recordings"][0]["labels"] == os.path.join("labels", "backroom.labels.json")
    assert snap["totals"]["recordings_labeled"] == 1


def test_renaming_a_label_file_carries_its_recording_too(project):
    """Reached from the other tab, it is the same rename of the same run."""
    entry = inventory.rename(str(project), LABELS1, "backroom")
    assert entry["path"] == os.path.join("labels", "backroom.labels.json")
    assert (project / "recordings" / "backroom.mcap").is_file()
    assert len(entry["renamed"]) == 2


def test_renaming_a_recording_without_labels_moves_only_the_recording(project):
    (project / "labels" / "run1.labels.json").unlink()
    entry = inventory.rename(str(project), MCAP1, "backroom")
    assert entry["renamed"] == [os.path.join("recordings", "backroom.mcap")]


def test_renaming_labels_whose_recording_is_gone_keeps_the_old_mcap_file(project):
    """A `mcap_file` naming a recording that is not here would be a worse lie."""
    (project / "recordings" / "run1.mcap").unlink()
    inventory.rename(str(project), LABELS1, "backroom")
    data = json.loads((project / "labels" / "backroom.labels.json").read_text())
    assert data["run_id"] == "backroom" and data["mcap_file"] == "run1.mcap"


def test_rename_reports_the_whole_group_so_a_busy_check_can_cover_it(project):
    assert sorted(inventory.rename_targets(str(project), MCAP1)) == [
        str(project / "labels" / "run1.labels.json"),
        str(project / "recordings" / "run1.mcap"),
    ]


def test_delete_removes_a_dataset_folder_and_reports_what_it_freed(project):
    before = inventory.snapshot(str(project))["totals"]["datasets_bytes"]
    out = inventory.delete(str(project), DS1)
    assert out["kind"] == "dataset" and out["freed"] == before and out["files"] == 1
    assert not (project / "datasets" / "d1").exists()
    assert inventory.snapshot(str(project))["totals"]["datasets"] == 0


def test_delete_takes_only_what_it_was_asked_for(project):
    """Unlike rename, delete never takes the rest of the run with it."""
    inventory.delete(str(project), MCAP1)
    assert not (project / "recordings" / "run1.mcap").exists()
    assert (project / "labels" / "run1.labels.json").is_file()
    inventory.delete(str(project), LABELS1)
    assert not (project / "labels" / "run1.labels.json").exists()


def test_delete_refuses_anything_outside_the_three_project_folders(project):
    (project / "config.yaml").write_text("topics: {}\n")
    for rel in ("datasets", "config.yaml", os.path.join("training", "cache", "meta.json")):
        with pytest.raises((ValueError, FileNotFoundError)):
            inventory.delete(str(project), rel)
    assert (project / "config.yaml").is_file()
    assert (project / "training" / "cache" / "meta.json").is_file()


def test_delete_unlinks_a_symlinked_dataset_without_following_it(project, tmp_path):
    """rmtree through a link would empty a folder outside the project."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keepme").write_text("precious")
    (project / "datasets" / "linked").symlink_to(outside, target_is_directory=True)
    inventory.delete(str(project), os.path.join("datasets", "linked"))
    assert not (project / "datasets" / "linked").exists()
    assert (outside / "keepme").read_text() == "precious"


def test_recording_info_refuses_to_escape_the_project(project):
    with pytest.raises((ValueError, FileNotFoundError)):
        inventory.recording_info(str(project), "../../../etc/passwd")


def test_recording_info_reports_an_unreadable_file_instead_of_raising(project):
    info = inventory.recording_info(str(project), "recordings/run1.mcap")
    assert info["ok"] is False and info["error"]
    assert "Trim" in info["hint"]  # points at the tool that salvages it


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
def _wait(job, timeout=15.0):
    deadline = time.time() + timeout
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.05)
    return job


def test_job_captures_output_and_exit_code(tmp_path):
    jm = JobManager(str(tmp_path))
    job = _wait(jm.launch(["echo", "hello world"], command_id="t", title="echo"))
    assert job.status == "ok" and job.returncode == 0
    assert any("hello world" in line for line in job.lines)
    assert os.path.exists(job.log_path)


def test_job_merges_stderr_and_marks_failure(tmp_path):
    jm = JobManager(str(tmp_path))
    job = _wait(jm.launch(["sh", "-c", "echo oops >&2; exit 3"],
                          command_id="t", title="fail"))
    assert job.status == "failed" and job.returncode == 3
    assert any("oops" in line for line in job.lines)


def test_job_tail_is_a_cursor_not_a_full_replay(tmp_path):
    jm = JobManager(str(tmp_path))
    job = _wait(jm.launch(["sh", "-c", "echo a; echo b"], command_id="t", title="t"))
    first = job.tail(0)
    assert first["cursor"] > 0
    again = job.tail(first["cursor"])
    assert again["lines"] == [] and again["cursor"] == first["cursor"]


def test_stop_kills_a_running_job(tmp_path):
    jm = JobManager(str(tmp_path))
    job = jm.launch(["sleep", "60"], command_id="t", title="sleep")
    time.sleep(0.4)
    assert job.stop() is True
    _wait(job)
    assert job.status == "stopped"


def test_rerun_replays_the_same_command_as_a_separate_job(tmp_path):
    jm = JobManager(str(tmp_path))
    first = _wait(jm.launch(["echo", "again"], command_id="t", title="echo",
                            display="rocklabel echo again"))
    second = _wait(jm.rerun(first))
    assert second.id != first.id and second.log_path != first.log_path
    assert second.argv == first.argv
    # The short spelling survives, so the history does not sprout absolute paths.
    assert second.command_line == first.command_line == "rocklabel echo again"
    assert second.status == "ok" and any("again" in line for line in second.lines)


def test_a_missing_binary_fails_the_job_rather_than_the_server(tmp_path):
    jm = JobManager(str(tmp_path))
    job = jm.launch(["definitely-not-a-real-binary-xyz"], command_id="t", title="x")
    assert job.status == "failed"
    assert any("could not start" in line for line in job.lines)


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(project):
    from rocklabel.dashboard.server import create_app

    app = create_app(str(project))
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index_renders(client):
    res = client.get("/")
    assert res.status_code == 200 and b"rocklabel" in res.data


def test_catalog_endpoint_serves_the_shape_the_page_destructures(client):
    body = client.get("/api/catalog").get_json()
    assert {"stages", "commands", "version", "root", "sensor"} <= set(body)
    assert body["sensor"]["port"] > 0
    cmd = body["commands"][0]
    assert {"id", "title", "cli", "params", "stage", "what", "why"} <= set(cmd)


def test_state_endpoint_carries_inventory_machine_and_jobs(client):
    body = client.get("/api/state").get_json()
    assert {"inventory", "machine", "jobs"} <= set(body)
    assert body["inventory"]["totals"]["recordings"] == 1


def test_preview_returns_the_command_line_without_running_anything(client):
    body = client.post("/api/preview", json={
        "command_id": "generate",
        "values": {"mcap": "recordings/run1.mcap", "out": "datasets/new"},
    }).get_json()
    assert body["command_line"] == "rocklabel generate recordings/run1.mcap --out datasets/new"
    assert body["error"] == ""
    assert client.get("/api/jobs").get_json()["jobs"] == []


def test_preview_reports_a_missing_required_value_instead_of_500(client):
    body = client.post("/api/preview", json={"command_id": "generate", "values": {}}).get_json()
    assert body["error"] and not body["command_line"]


def test_run_launches_a_job_and_shows_the_short_command_line(client):
    body = client.post("/api/run", json={"command_id": "inspect",
                                         "values": {"mcap": "recordings/run1.mcap"}})
    job = body.get_json()["job"]
    # argv[0] is resolved to a real executable, but the display stays typeable.
    assert job["command_line"].startswith("rocklabel inspect")
    tail = client.get(f"/api/jobs/{job['id']}?since=0").get_json()
    assert "lines" in tail and "cursor" in tail


def test_rerun_endpoint_launches_the_same_command_line_again(client):
    """The Rerun button on a past job: same command, new job, new log."""
    jm = client.application.config["JOBS"]
    first = client.post("/api/run", json={"command_id": "inspect",
                                          "values": {"mcap": "recordings/run1.mcap"}}
                        ).get_json()["job"]
    _wait(jm.get(first["id"]))
    again = client.post(f"/api/jobs/{first['id']}/rerun").get_json()["job"]
    assert again["id"] != first["id"]
    assert again["command_id"] == first["command_id"]
    assert again["command_line"] == first["command_line"]
    _wait(jm.get(again["id"]))
    assert {j["id"] for j in client.get("/api/jobs").get_json()["jobs"]} == \
        {first["id"], again["id"]}


def test_rerun_refuses_while_the_job_is_still_running(client):
    """Two copies of a `live` would fight over the sensor port; stop it first."""
    jm = client.application.config["JOBS"]
    job = jm.launch([sys.executable, "-c", "import time; time.sleep(30)"],
                    command_id="live", title="Live view")
    try:
        res = client.post(f"/api/jobs/{job.id}/rerun")
        assert res.status_code == 409
        assert "still running" in res.get_json()["error"]
        assert len(jm.list()) == 1
    finally:
        job.stop()


def test_rerun_endpoint_404s_on_a_job_it_never_ran(client):
    assert client.post("/api/jobs/j9999/rerun").status_code == 404


def test_run_rejects_an_unknown_command(client):
    assert client.post("/api/run", json={"command_id": "rm-rf"}).status_code == 400


def test_rename_endpoint_renames_and_the_next_state_shows_the_new_name(client):
    body = client.post("/api/rename",
                       json={"path": "datasets/d1", "name": "gravel"}).get_json()
    assert body["path"] == os.path.join("datasets", "gravel")
    state = client.get("/api/state").get_json()
    assert [d["name"] for d in state["inventory"]["datasets"]] == ["gravel"]


def test_rename_endpoint_moves_a_recording_and_its_labels_together(client):
    body = client.post("/api/rename",
                       json={"path": "recordings/run1.mcap", "name": "backroom"}).get_json()
    assert len(body["renamed"]) == 2
    inv = client.get("/api/state").get_json()["inventory"]
    assert inv["recordings"][0]["name"] == "backroom.mcap"
    assert inv["labels"][0]["run_id"] == "backroom"
    assert inv["totals"]["recordings_labeled"] == 1  # still linked


def test_rename_endpoint_reports_a_bad_name_as_400_not_500(client):
    res = client.post("/api/rename", json={"path": "datasets/d1", "name": "a/b"})
    assert res.status_code == 400 and "name must" in res.get_json()["error"]


def test_rename_endpoint_refuses_a_path_outside_the_project(client):
    res = client.post("/api/rename", json={"path": "../../etc", "name": "pwned"})
    assert res.status_code in (403, 404)


def test_rename_endpoint_refuses_while_a_job_holds_the_dataset(client):
    """A rename under a running `generate` would fail that job halfway."""
    jm = client.application.config["JOBS"]
    job = jm.launch([sys.executable, "-c", "import time; time.sleep(30)",
                     "--out", "datasets/d1"], command_id="generate", title="Generate")
    try:
        res = client.post("/api/rename", json={"path": "datasets/d1", "name": "gravel"})
        assert res.status_code == 409
        assert "in use by a running job" in res.get_json()["error"]
    finally:
        job.stop()
    assert client.get("/api/state").get_json()["inventory"]["datasets"][0]["name"] == "d1"


def test_rename_endpoint_refuses_when_a_job_holds_the_labels_of_the_recording(client):
    """The busy check covers the whole group a rename would move, not just the
    file that was clicked."""
    jm = client.application.config["JOBS"]
    job = jm.launch([sys.executable, "-c", "import time; time.sleep(30)",
                     "--labels", "labels/run1.labels.json"],
                    command_id="generate", title="Generate")
    try:
        res = client.post("/api/rename",
                          json={"path": "recordings/run1.mcap", "name": "backroom"})
        assert res.status_code == 409
    finally:
        job.stop()
    assert (client.get("/api/state").get_json()["inventory"]["recordings"][0]["name"]
            == "run1.mcap")


def test_delete_endpoint_removes_the_file_and_reports_the_space(client, project):
    res = client.post("/api/delete", json={"path": "recordings/run1.mcap"})
    assert res.status_code == 200 and res.get_json()["freed"] > 0
    assert not (project / "recordings" / "run1.mcap").exists()
    inv = client.get("/api/state").get_json()["inventory"]
    assert inv["totals"]["recordings"] == 0
    assert inv["totals"]["labels"] == 1  # the labels were not collateral


def test_delete_endpoint_refuses_a_path_it_does_not_own(client, project):
    res = client.post("/api/delete", json={"path": "training/cache/meta.json"})
    assert res.status_code == 400
    assert (project / "training" / "cache" / "meta.json").is_file()


def test_delete_endpoint_refuses_while_a_job_holds_the_path(client, project):
    jm = client.application.config["JOBS"]
    job = jm.launch([sys.executable, "-c", "import time; time.sleep(30)",
                     "recordings/run1.mcap"], command_id="label", title="Label")
    try:
        res = client.post("/api/delete", json={"path": "recordings/run1.mcap"})
        assert res.status_code == 409
    finally:
        job.stop()
    assert (project / "recordings" / "run1.mcap").is_file()


def test_figure_endpoint_refuses_paths_outside_the_project(client):
    assert client.get("/api/figure?path=../../etc/passwd").status_code in (403, 404)


def test_figure_endpoint_refuses_non_png(client, project):
    assert client.get("/api/figure?path=config.yaml").status_code in (403, 404)


# --------------------------------------------------------------------------- #
# ablation sweeps
# --------------------------------------------------------------------------- #
def _ablate_fold(root, suite, arm, fold, pr_auc=None):
    d = root / "training" / "ablate" / suite / arm / f"loro_{fold}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "arm.json").write_text(json.dumps({"arm": arm, "label": arm, "model": "pointnet",
                                            "features": ["dx", "dy", "dz"]}))
    (d / "best.pt").write_bytes(b"weights")
    if pr_auc is not None:
        (d / "test_metrics.json").write_text(json.dumps(
            {"test_run": fold, "pr_auc": pr_auc, "f1": 0.5}))
    return d


def test_ablation_progress_counts_the_whole_declared_matrix(project):
    """Half an hour into an overnight sweep the denominator must be the full
    matrix, not the two arms that happen to have folders yet."""
    from rocklabel.train.ablate import SUITES

    _ablate_fold(project, "reflectivity", "pointnet-geom", "run1", 0.8)
    _ablate_fold(project, "reflectivity", "pointnet-geom", "run2")  # started, unfinished

    suites = inventory.ablations(str(project))
    assert len(suites) == 1
    s = suites[0]
    n_arms = len(SUITES["reflectivity"]["arms"])
    assert len(s["arms"]) == n_arms, "arms that have not started must still be listed"
    assert s["folds"] == 2                     # from the cache, not from disk
    assert s["runs_total"] == n_arms * 2
    assert s["runs_done"] == 1                 # only the evaluated fold counts
    done = {a["name"]: a["folds_done"] for a in s["arms"]}
    assert done["pointnet-geom"] == 1 and done["pointnet2-refl"] == 0


def test_ablation_arms_report_their_average_score(project):
    _ablate_fold(project, "reflectivity", "pointnet-refl", "run1", 0.60)
    _ablate_fold(project, "reflectivity", "pointnet-refl", "run2", 0.80)
    arms = {a["name"]: a for a in inventory.ablations(str(project))[0]["arms"]}
    assert arms["pointnet-refl"]["pr_auc"] == pytest.approx(0.70)
    assert arms["pointnet-geom"]["pr_auc"] is None


def test_ablation_checkpoints_join_the_picker_under_their_arm(project):
    _ablate_fold(project, "reflectivity", "pointnet-geom", "run1", 0.8)
    names = [c["name"] for c in inventory.checkpoints(str(project))]
    assert "reflectivity/pointnet-geom/loro_run1/best.pt" in names
    # the plain training/runs checkpoint is still there
    assert "pointnet_loro_run1/best.pt" in names


def test_snapshot_exposes_ablation_totals(project):
    _ablate_fold(project, "reflectivity", "pointnet-geom", "run1", 0.8)
    t = inventory.snapshot(str(project))["totals"]
    assert t["ablation_runs_done"] == 1
    assert t["ablation_runs_total"] > 1


def test_figures_are_tagged_with_the_report_that_wrote_them(project):
    for rel, name in [("training/results", "comparison.png"),
                      ("training/results_reflect", "brightness_drift.png"),
                      ("training/results_ablate/reflectivity", "paired_deltas.png")]:
        d = project / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"\x89PNG")
    groups = {f["name"]: f["group"] for f in inventory.result_figures(str(project))}
    assert groups["comparison.png"] == "Model comparison"
    assert groups["brightness_drift.png"] == "Reflectivity check"
    assert groups["paired_deltas.png"] == "Ablation · reflectivity"
    assert all(f["blurb"] for f in inventory.result_figures(str(project)))


def test_a_project_with_no_sweep_reports_no_suites(project):
    assert inventory.ablations(str(project)) == []
    assert inventory.snapshot(str(project))["totals"]["ablation_runs_done"] == 0


def test_ablate_and_reflect_are_in_the_catalog():
    for cid in ("train-ablate", "train-reflect"):
        assert cid in spec.COMMANDS_BY_ID, f"{cid} missing from the dashboard catalog"
    ablate = spec.COMMANDS_BY_ID["train-ablate"]
    assert ablate.long_running, "a 100-training sweep is not an instant command"
    suite = next(p for p in ablate.params if p.name == "suite")
    from rocklabel.train.ablate import SUITES
    assert set(suite.choices) == set(SUITES), "the suite picker drifted from the real suites"


def test_ablate_form_builds_the_argv_the_cli_expects():
    argv = spec.build_argv(spec.COMMANDS_BY_ID["train-ablate"],
                           {"suite": "reflectivity", "arms": "pointnet-geom, pointnet-refl",
                            "report_only": True})
    assert argv[:2] == ["rocklabel-train", "ablate"]
    i = argv.index("--arms")
    assert argv[i + 1:i + 3] == ["pointnet-geom", "pointnet-refl"]
    assert "--report-only" in argv
    # It must parse for real, not just look right.
    from rocklabel.train.cli import build_parser
    build_parser().parse_args(argv[1:])


def test_the_patience_default_shown_matches_the_training_default():
    """It drifted once already: the form offered 6 long after the real default
    moved to 10, so every sweep launched from the dashboard stopped early."""
    from rocklabel.train import TRAIN_DEFAULTS

    for cid in ("train-train", "train-compare", "train-ablate"):
        p = next((p for p in spec.COMMANDS_BY_ID[cid].params if p.name == "patience"), None)
        if p is not None:
            assert p.default == TRAIN_DEFAULTS["patience"], cid


# --------------------------------------------------------------------------- #
# wired-link repair suggestion
# --------------------------------------------------------------------------- #
def _netfix_cfg():
    from rocklabel.live.config import AppConfig

    return AppConfig()


def _fake_iface(**over):
    iface = {"name": "eth9", "up": True, "carrier": True, "operstate": "up",
             "nm_state": "unmanaged", "nm_managed": False,
             "addresses": ["10.11.10.5/24", "10.11.11.8/24", "10.11.10.1/24"]}
    iface.update(over)
    return [iface]


def test_netfix_script_is_the_recipe_that_actually_revives_the_link(monkeypatch):
    """Order matters: unmanage, flush, then add — an add before the flush is
    what produces "File exists" half-states."""
    from rocklabel.dashboard import netfix

    monkeypatch.setattr(netfix, "wired_interfaces",
                        lambda: _fake_iface(nm_managed=True, addresses=[]))
    d = netfix.diagnose(_netfix_cfg(), streaming=False)
    assert d["suggested"] and d["iface"] == "eth9"
    cmds = [c["cmd"] for c in d["commands"]]
    assert cmds == [
        "sudo nmcli device set eth9 managed no",
        "sudo ip addr flush dev eth9",
        "sudo ip addr add 10.11.10.5/24 dev eth9",
        "sudo ip addr add 10.11.11.8/24 dev eth9",
        "sudo ip addr add 10.11.10.1/24 dev eth9",
        "sudo ip link set dev eth9 up",
    ]
    # Every command carries its reason into the copyable script.
    assert all(c["why"] for c in d["commands"])
    assert d["script"].count("#") == len(cmds)


def test_netfix_stays_quiet_when_the_interface_is_already_correct(monkeypatch):
    from rocklabel.dashboard import netfix

    monkeypatch.setattr(netfix, "wired_interfaces", _fake_iface)
    assert netfix.diagnose(_netfix_cfg(), streaming=False)["suggested"] is False


def test_netfix_blames_the_cable_not_the_config_when_carrier_is_down(monkeypatch):
    """`ip addr add` cannot fix an unplugged port, so it must not be offered."""
    from rocklabel.dashboard import netfix

    monkeypatch.setattr(netfix, "wired_interfaces", lambda: _fake_iface(carrier=False))
    d = netfix.diagnose(_netfix_cfg(), streaming=False)
    assert d["link_down"] is True and d["suggested"] is False
    assert any("carrier" in p for p in d["problems"])


def test_netfix_never_interrupts_a_streaming_sensor(monkeypatch):
    from rocklabel.dashboard import netfix

    monkeypatch.setattr(netfix, "wired_interfaces",
                        lambda: _fake_iface(nm_managed=True, addresses=[]))
    assert netfix.diagnose(_netfix_cfg(), streaming=True)["suggested"] is False


def test_netfix_picks_the_port_that_already_carries_a_sensor_address(monkeypatch):
    from rocklabel.dashboard import netfix

    other = dict(_fake_iface()[0], name="eth0", addresses=["192.168.1.20/24"])
    mine = dict(_fake_iface()[0], name="eth9", nm_managed=True,
                addresses=["10.11.10.5/24"])
    monkeypatch.setattr(netfix, "wired_interfaces", lambda: [other, mine])
    assert netfix.diagnose(_netfix_cfg(), streaming=False)["iface"] == "eth9"


def test_sensor_endpoint_carries_the_repair_suggestion(client):
    body = client.get("/api/sensor").get_json()
    assert {"state", "ping", "listen", "interfaces", "network_fix"} <= set(body)
    fix = body["network_fix"]
    assert set(fix) >= {"supported", "suggested", "detail"}
    if fix.get("suggested"):
        assert fix["script"] and fix["commands"]


def test_the_hidden_attribute_actually_hides():
    """`el.hidden = true` is how the page closes the drawer and drops badges.

    The browser's own `[hidden] { display: none }` loses to any author rule that
    sets `display` (`.drawer` is a flex column, `.nav-badge` an inline-flex), so
    without this override the close button silently does nothing. The node DOM
    harness has no CSS engine and cannot catch it.
    """
    from pathlib import Path

    import rocklabel.dashboard as dash

    css = (Path(dash.__file__).parent / "static" / "app.css").read_text()
    assert "[hidden] { display: none !important; }" in css
