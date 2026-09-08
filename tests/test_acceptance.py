"""Acceptance (Part B §63, §64, §65).

Three things are checked here, and they are different in kind.

`run_levels` and `acceptance_report` are *instruments*, so the tests drive their edges — a level
whose gate fails must withhold its number, not report zero — rather than re-running the campaign,
which takes an hour and is committed in `results/m13_acceptance.json`.

`run_preflight` is a *checklist*, so the tests confirm it fails on the deployments it must refuse.

The §63 workflow is a *property*, and the only honest way to test survival is to destroy the
process: every engine disposed, every session closed, and a fresh agent asked to inherit what
earlier agents left behind.
"""

from __future__ import annotations

import uuid

import pytest

from civitas.acceptance import LEVELS, LevelResult, acceptance_report, run_levels
from civitas.config import Settings
from civitas.preflight import FAIL, PASS, WARN, run_preflight

# --------------------------------------------------------------------------- §65 structure


def test_every_level_names_its_own_negative_control():
    """A level without a control is an observation, not a claim. This is the one structural
    property of the acceptance run and it is cheap to assert."""
    assert len(LEVELS) == 6
    assert [level for level, _, _ in LEVELS] == [1, 2, 3, 4, 5, 6]
    for _level, name, control in LEVELS:
        assert name and control, name


def test_levels_one_and_two_are_different_comparisons():
    """They were reported as one number (+0.300) through M9, which conflated two claims.

    Level 1 holds the environment constant and varies sight (`collective` vs `solo_in_mature`).
    Level 2 holds the agent constant and varies the environment (`collective` vs
    `baseline_empty`). They can come apart, and a single number hides it when they do.
    """
    controls = {level: control for level, _, control in LEVELS}
    assert "solo" in controls[1] and "baseline_empty" not in controls[1]
    assert "empty environment" in controls[2]


def test_a_level_with_no_value_is_withheld_rather_than_zero():
    """ARCHITECTURE §3.1 and §3.9 together: a gated-out level reports no number at all."""
    level = LevelResult(level=1, name="n", claim="c", control="x")
    assert level.value is None
    assert level.passed is False
    assert level.as_dict()["value"] is None


def test_a_subset_of_levels_can_be_run(db, org):
    """The full campaign takes an hour. A subset must produce the same shape, so a failure can be
    re-run alone rather than by repeating everything."""
    results = run_levels(db, seeds=[0], accumulation_passes=1, include={1})
    assert [r.level for r in results] == [1]
    assert results[0].unit


def test_the_report_refuses_to_average_over_a_failed_level(db, org, monkeypatch):
    """Six levels are six separate claims. A mean over them is a number no claim supports, so one
    failure fails the run."""
    import civitas.acceptance as acceptance

    def _fake(session, **kwargs):
        return [
            LevelResult(level=1, name="a", claim="c", control="x", value=0.3, passed=True),
            LevelResult(level=2, name="b", claim="c", control="x", value=None, passed=False,
                        withheld_reason="a gate failed"),
        ]

    monkeypatch.setattr(acceptance, "run_levels", _fake)
    report = acceptance_report(db, seeds=[0])
    assert report["accepted"] is False
    assert report["failing_levels"] == [2]
    assert "average" in report["criterion"]


def test_the_report_carries_the_manifest_that_produced_it(db, org, monkeypatch):
    """§46. Six results produced by six code versions are six claims; the acceptance run is one."""
    import civitas.acceptance as acceptance

    monkeypatch.setattr(acceptance, "run_levels", lambda session, **kwargs: [])
    manifest = acceptance_report(db, seeds=[0])["manifest"]
    for key in ("code_version", "schema_version", "dependencies", "sandbox", "seeds"):
        assert manifest[key] is not None
    # A run under weaker isolation must never be mistaken for one under the real thing.
    assert "sandbox_unenforced_limits" in manifest


def test_an_empty_run_is_not_an_accepted_one(db, org, monkeypatch):
    """`all([])` is True. A report that accepted a system on the strength of running nothing is
    the most embarrassing possible pass."""
    import civitas.acceptance as acceptance

    monkeypatch.setattr(acceptance, "run_levels", lambda session, **kwargs: [])
    assert acceptance_report(db, seeds=[0])["accepted"] is False


# --------------------------------------------------------------------------- §64 preflight


def _settings(**overrides) -> Settings:
    base = {
        "database_url": "sqlite:///:memory:",
        "jwt_secret": "a" * 40,
        "auth_enabled": True,
    }
    base.update(overrides)
    return Settings(**base)


def _status(report, name):
    return next(c["status"] for c in report["checks"] if c["name"] == name)


def test_preflight_blocks_a_deployment_with_authentication_off():
    report = run_preflight(_settings(auth_enabled=False))
    assert _status(report, "auth is enabled") == FAIL
    assert report["ready"] is False


def test_preflight_blocks_a_placeholder_secret():
    for secret in ("", "CHANGE_ME", "not-a-real-key", "short"):
        report = run_preflight(_settings(jwt_secret=secret))
        assert _status(report, "jwt secret is set and not a placeholder") == FAIL, secret


def test_preflight_blocks_cors_open_to_everything():
    report = run_preflight(_settings(cors_origins=["*"]))
    assert _status(report, "cors is not open to every origin") == FAIL


def test_preflight_blocks_a_sandbox_whose_limits_do_not_hold():
    """The check that matters most. A bound the caller believes is in force but is not is worse
    than no bound, because it is relied on — measured in M4, where `RLIMIT_NPROC` was silently
    unenforced for uid 0 and 5000 processes spawned against a limit of 8."""
    from civitas.runtime.sandbox import get_sandbox

    report = run_preflight(_settings())
    unenforced = list(get_sandbox(_settings()).unenforced_limits())
    expected = FAIL if unenforced else PASS
    assert _status(report, "every requested sandbox limit is enforced") == expected


def test_preflight_fails_tracing_that_is_configured_but_off():
    """Configured-and-not-working is the failure that goes unnoticed until an incident, when the
    traces are not there."""
    from civitas.observability import configure_tracing

    settings = _settings(otel_endpoint="http://otel:4318")
    state = configure_tracing(settings)
    if state["enabled"]:
        pytest.skip("OpenTelemetry is installed here; the degraded path is not exercised")
    report = run_preflight(settings)
    assert _status(report, "tracing works if it is configured") == FAIL


def test_preflight_warns_rather_than_blocks_on_sqlite():
    """SQLite is a supported backend (§6, §33), not a defect. A checklist that blocked it would
    make the Colab runtime unshippable by its own rules."""
    report = run_preflight(_settings())
    assert _status(report, "database is postgresql") == WARN


def test_a_warning_never_blocks_and_a_failure_always_does():
    report = run_preflight(_settings(metrics_enabled=False, log_json=False))
    assert _status(report, "metrics are enabled") == WARN
    assert _status(report, "logs are structured") == WARN
    assert "metrics are enabled" not in report["blocking"]
    assert set(report["blocking"]) == {
        c["name"] for c in report["checks"] if c["status"] == FAIL
    }


def test_an_unreachable_database_is_a_failed_check_not_a_crash():
    report = run_preflight(_settings(database_url="postgresql+psycopg://nobody@127.0.0.1:1/none"))
    assert _status(report, "database is reachable") == FAIL
    assert report["ready"] is False


# --------------------------------------------------------------------------- §63 survival


def test_the_civilization_survives_the_loss_of_every_runtime_object(settings, engine, db):
    """§63's workflow, outside the notebook.

    Two agents work, then **every engine and session is destroyed** and rebuilt from the file, and
    a third fresh agent is asked to inherit what they left. Step 15 of §63 is "terminate
    application/runtime"; a test cannot kill its own process, so it does the thing that actually
    matters — drops every in-memory object and reopens from disk — and checks the result rather
    than asserting it.
    """
    from sqlalchemy.orm import sessionmaker

    from civitas.domain.enums import ExperimentArm
    from civitas.experiments.runner import run_benchmark_episode
    from civitas.experiments.tasks.hidden_rule import build_device, era_instances
    from civitas.persistence.engine import create_db_engine
    from civitas.persistence.models import Artifact, Organization, Workspace
    from civitas.persistence.session import install_guards

    org = Organization(name="acc", slug=f"acc-{uuid.uuid4().hex[:8]}")
    db.add(org)
    db.flush()
    workspace = Workspace(organization_id=org.id, name="acceptance",
                          slug=f"a-{uuid.uuid4().hex[:8]}", environment_version="acc-1")
    db.add(workspace)
    db.commit()
    workspace_id = workspace.id

    device = build_device(seed=99, era=1)
    instance = era_instances(device)[0]
    for i in range(2):
        run_benchmark_episode(
            db, workspace_id=workspace_id, device=device, instance=instance,
            arm=ExperimentArm.COLLECTIVE, record_findings=True, probe_order_seed=i,
        )
    db.commit()
    written = db.query(Artifact).filter_by(workspace_id=workspace_id).count()
    assert written > 0, "the first two agents left nothing behind"
    db.close()

    # Everything in this process goes away.
    fresh_engine = create_db_engine(settings)
    fresh_factory = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    install_guards(fresh_factory)
    fresh = fresh_factory()
    try:
        restored = fresh.query(Artifact).filter_by(workspace_id=workspace_id).count()
        assert restored == written, "the collective did not survive the reconnect"

        third = run_benchmark_episode(
            fresh, workspace_id=workspace_id, device=device, instance=instance,
            arm=ExperimentArm.COLLECTIVE, record_findings=False, probe_order_seed=2,
        )
        fresh.commit()
        assert third.artifacts_read > 0, (
            "the reconnected agent did not inherit the collective"
        )
    finally:
        fresh.close()
        fresh_engine.dispose()
