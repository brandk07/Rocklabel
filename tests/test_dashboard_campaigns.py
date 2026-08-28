"""Tests for the Training screen's campaign history.

The history is half written prose and half numbers read off disk, and the two
halves fail differently. The prose fails by going stale — a campaign renamed on
disk, a write-up missing a section, a score nobody set. The numbers fail by
being invented — a fold counted that never ran, an arm counted twice because a
campaign uses the older flat layout. There is a test here for each.
"""

from __future__ import annotations

import json

import pytest

flask = pytest.importorskip("flask", reason="dashboard needs the [dash] extra")

from rocklabel.dashboard import campaigns as camp  # noqa: E402


# --------------------------------------------------------------------------- #
# the written half
# --------------------------------------------------------------------------- #
def test_every_write_up_answers_all_four_questions():
    """A card with a blank section renders as an empty box and says nothing."""
    assert camp.CAMPAIGNS, "the history must not be empty"
    for name, c in camp.CAMPAIGNS.items():
        for field in ("title", "verdict", "asked", "happened", "failure", "learned"):
            assert c.get(field), f"{name} is missing its '{field}'"
        # Long enough to be a real explanation rather than a label.
        for field in ("asked", "happened", "failure", "learned"):
            assert len(c[field]) > 80, f"{name}'s '{field}' is too thin to be useful"
        assert not c["verdict"].endswith(".."), name


def test_every_campaign_is_scored_out_of_ten():
    for name, c in camp.CAMPAIGNS.items():
        score = c.get("score")
        assert isinstance(score, int), f"{name} has no payoff score"
        assert 1 <= score <= 10, f"{name}'s score {score} is off the scale"


def test_the_scale_is_explained_as_payoff_not_accuracy():
    """The one sentence that stops a 4/10 reading as the model's accuracy."""
    assert len(camp.SCORE_SCALE) > 80
    assert "not the model" in camp.SCORE_SCALE


def test_retired_campaigns_are_written_up_and_flagged():
    """Retired sweeps stay in the history — they are past runs and they taught
    something — but every one must be marked, because their indoor scores are
    not comparable with the volleyball court's."""
    assert camp.RETIRED
    for name in camp.RETIRED:
        assert name in camp.CAMPAIGNS, f"{name} is retired but has no write-up"


# --------------------------------------------------------------------------- #
# the measured half
# --------------------------------------------------------------------------- #
def _fold(d, *, pr_auc=0.5, evaluated=True, task="classify"):
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model": "pointnet", "epochs": 30}))
    (d / "history.csv").write_text(
        "epoch,train_loss,val_loss,val_pr_auc\n0,0.5,0.4,0.3\n1,0.4,0.3,0.4\n")
    if evaluated:
        (d / "test_metrics.json").write_text(json.dumps(
            {"pr_auc": pr_auc, "task": task, "model": "pointnet", "test_run": d.name}))


@pytest.fixture
def project(tmp_path):
    """One campaign in each layout, plus one nobody has written up."""
    base = tmp_path / "training" / "experiments"
    # Nested: <arm>/<fold>/, what every current suite writes.
    for arm in ("seg-long", "seg-fine"):
        for fold in ("run1", "run2"):
            _fold(base / "segdense" / arm / f"loro_{fold}", pr_auc=0.36, task="segment")
    # Flat: <arm>_loro_<fold>/, what the two retired campaigns wrote.
    for model in ("pointnet", "pointnet2"):
        for fold in ("run1", "run2"):
            _fold(base / "compare" / f"{model}_loro_{fold}", pr_auc=0.96)
    # An experiment with folds on disk and no entry in CAMPAIGNS.
    _fold(base / "brand-new" / "arm-a" / "loro_run1", pr_auc=0.7)
    # A fold that started and never finished: it must not be counted as run.
    _fold(base / "segdense" / "seg-long" / "loro_run3", evaluated=False)
    return tmp_path


def test_only_finished_folds_are_counted(project):
    by = {c["name"]: c for c in camp.campaigns(str(project))["campaigns"]}
    # 4 evaluated folds; the fifth wrote a history but never a test_metrics.
    assert by["segdense"]["folds_done"] == 4


def test_flat_layout_campaigns_do_not_report_every_fold_as_an_arm(project):
    """The retired layout puts (arm, fold) pairs flat at the top level, so a
    naive directory listing turns a 2-setting campaign into a 4-setting one."""
    by = {c["name"]: c for c in camp.campaigns(str(project))["campaigns"]}
    assert sorted(by["compare"]["arms"]) == ["pointnet", "pointnet2"]
    assert by["compare"]["folds_done"] == 4


def test_nested_layout_lists_its_real_arms(project):
    by = {c["name"]: c for c in camp.campaigns(str(project))["campaigns"]}
    assert sorted(by["segdense"]["arms"]) == ["seg-fine", "seg-long"]


def test_an_undocumented_sweep_still_appears(project):
    """A sweep run today is visible today, not on the day someone writes it up."""
    by = {c["name"]: c for c in camp.campaigns(str(project))["campaigns"]}
    new = by["brand-new"]
    assert new["documented"] is False
    assert new["score"] is None
    assert new["folds_done"] == 1


def test_documented_sweeps_carry_their_prose_through(project):
    by = {c["name"]: c for c in camp.campaigns(str(project))["campaigns"]}
    assert by["segdense"]["documented"] is True
    assert by["segdense"]["score"] == camp.CAMPAIGNS["segdense"]["score"]
    assert by["segdense"]["learned"] == camp.CAMPAIGNS["segdense"]["learned"]


def test_retired_campaigns_are_flagged_in_the_payload(project):
    by = {c["name"]: c for c in camp.campaigns(str(project))["campaigns"]}
    assert by["compare"]["retired"] is True
    assert by["segdense"]["retired"] is False


def test_an_abandoned_queue_explains_itself(project):
    """The Training screen asks this of any sweep with folds left unrun."""
    by = {c["name"]: c for c in camp.campaigns(str(project))["campaigns"]}
    assert "on purpose" in by["segdense"]["stopped_reason"]
    # A campaign that simply finished has nothing to explain.
    assert by["compare"]["stopped_reason"] == ""


def test_tasks_are_listed_rather_than_averaged_together(project):
    """A per-point score and a per-candidate-ball score are two different
    measurements; the payload must not blend them into one mean."""
    by = {c["name"]: c for c in camp.campaigns(str(project))["campaigns"]}
    assert by["segdense"]["tasks"] == ["segment"]
    assert by["compare"]["tasks"] == ["classify"]


def test_newest_campaign_comes_first(project):
    names = [c["name"] for c in camp.campaigns(str(project))["campaigns"]]
    finished = {c["name"]: c["finished"]
                for c in camp.campaigns(str(project))["campaigns"]}
    assert names == sorted(names, key=lambda n: finished[n] or 0, reverse=True)


def test_an_empty_project_returns_an_empty_history(tmp_path):
    out = camp.campaigns(str(tmp_path))
    assert out["campaigns"] == []
    assert out["scale"]


def test_the_endpoint_serves_the_history(project):
    from rocklabel.dashboard.server import create_app

    app = create_app(str(project))
    app.config["TESTING"] = True
    with app.test_client() as client:
        body = client.get("/api/campaigns").get_json()
    assert {c["name"] for c in body["campaigns"]} == {"segdense", "compare", "brand-new"}
    assert body["scale"]
