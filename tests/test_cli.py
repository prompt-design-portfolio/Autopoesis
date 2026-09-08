"""The command line (Part B §5, §46, §50).

`pyproject.toml` has declared the `civitas` entry point since M1, so the first thing worth testing
is that it resolves — a declared console script that does not import is a broken install that only
the user discovers.

The commands are tested by calling `main()` with an explicit `--database-url` rather than by
spawning a subprocess: the point is that the CLI is a thin shell over the same service functions
the API calls, and a subprocess test would pass equally well over a second implementation.
"""

from __future__ import annotations

import json

import pytest

from civitas.cli import main
from civitas.persistence.engine import dispose_engines


@pytest.fixture
def cli_db(tmp_path):
    url = f"sqlite:///{tmp_path}/cli.db"
    yield url
    dispose_engines()


def _run(capsys, *argv) -> dict | list:
    assert main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


def _bootstrapped(capsys, cli_db) -> tuple[str, str]:
    _run(capsys, "--database-url", cli_db, "init-db")
    _run(capsys, "--database-url", cli_db, "bootstrap",
         "--name", "CLI Org", "--slug", "cli-org", "--email", "admin@example.test")
    ws = _run(capsys, "--database-url", cli_db, "workspace",
              "--org", "cli-org", "--name", "W", "--slug", "cli-ws")
    return "cli-org", ws["workspace_id"]


def test_the_declared_entry_point_resolves():
    """`pyproject.toml` promises `civitas = "civitas.cli:main"`. This is that promise, checked."""
    import importlib

    module, _, attr = "civitas.cli:main".partition(":")
    assert callable(getattr(importlib.import_module(module), attr))


def test_init_db_says_which_way_it_created_the_schema(capsys, cli_db):
    """§34: a schema created by `create_all` and one created by migrations are not interchangeable,
    and a command that did not say which it used would make that indistinguishable afterwards."""
    out = _run(capsys, "--database-url", cli_db, "init-db")
    assert out["method"] == "metadata.create_all"
    assert out["dialect"] == "sqlite"
    assert out["tables"] == 44


def test_health_reports_the_journal_mode_it_actually_found(capsys, cli_db):
    _run(capsys, "--database-url", cli_db, "init-db")
    out = _run(capsys, "--database-url", cli_db, "health")
    assert out["ok"] is True
    assert out["journal_mode"] == "wal"
    assert out["foreign_keys"] is True


def test_bootstrap_prints_the_key_once_and_says_so(capsys, cli_db):
    _run(capsys, "--database-url", cli_db, "init-db")
    out = _run(capsys, "--database-url", cli_db, "bootstrap",
               "--name", "O", "--slug", "o", "--email", "a@b.test")
    assert out["api_key"] and "not recoverable" in out["note"]


def test_bootstrap_does_not_print_a_recoverable_key(capsys, cli_db):
    """§51: the plaintext is never stored. A second command must not be able to produce it."""
    from civitas.config import get_settings
    from civitas.persistence.engine import session_scope
    from civitas.persistence.models import ApiKey

    _run(capsys, "--database-url", cli_db, "init-db")
    out = _run(capsys, "--database-url", cli_db, "bootstrap",
               "--name", "O", "--slug", "o", "--email", "a@b.test")
    plaintext = out["api_key"]

    settings = get_settings().model_copy(update={"database_url": cli_db})
    with session_scope(settings) as session:
        # Every column, not a chosen few: a key stored in a column this test did not think to
        # check is stored just the same.
        rows = [
            {c.name: getattr(row, c.name) for c in ApiKey.__table__.columns}
            for row in session.query(ApiKey).all()
        ]
    assert rows
    assert all(plaintext not in json.dumps(row, default=str) for row in rows)


def test_an_unknown_organization_is_an_error_not_a_silent_no_op(capsys, cli_db):
    _run(capsys, "--database-url", cli_db, "init-db")
    assert main(["--database-url", cli_db, "workspace",
                 "--org", "nope", "--name", "W", "--slug", "w"]) == 2


def test_submit_prints_a_plan_and_not_an_answer(capsys, cli_db):
    """§5: the system responds to a request by creating work."""
    _, workspace_id = _bootstrapped(capsys, cli_db)
    plan = _run(capsys, "--database-url", cli_db, "submit", "--workspace", workspace_id,
                "investigate the intermittent data corruption in the ledger writer")

    assert [t["family"] for t in plan["tasks"]] == [
        "investigate", "hypothesise", "verify", "synthesise"
    ]
    assert [t["status"] for t in plan["tasks"]] == ["ready", "blocked", "blocked", "blocked"]
    assert plan["decomposer"] == "deterministic/1.0"


def test_status_is_the_same_across_two_separate_invocations(capsys, cli_db):
    """§32 at the process boundary: each `main()` call opens its own engine and closes it. If any
    part of a project's state lived in a process, the second call would differ from the first."""
    _, workspace_id = _bootstrapped(capsys, cli_db)
    plan = _run(capsys, "--database-url", cli_db, "submit", "--workspace", workspace_id,
                "investigate the intermittent data corruption in the ledger writer")

    first = _run(capsys, "--database-url", cli_db, "status", plan["project_id"])
    dispose_engines()
    second = _run(capsys, "--database-url", cli_db, "status", plan["project_id"])

    assert first == second
    assert first["tasks_total"] == 4 and first["tasks_blocked"] == 3
    assert first["progress"] == 0.0


def test_the_timeline_of_a_project_with_no_artifacts_is_empty(capsys, cli_db):
    _, workspace_id = _bootstrapped(capsys, cli_db)
    plan = _run(capsys, "--database-url", cli_db, "submit", "--workspace", workspace_id,
                "investigate the lock contention in the scheduler")
    assert _run(capsys, "--database-url", cli_db, "timeline", plan["project_id"]) == []


def test_domains_lists_what_this_process_can_run(capsys):
    out = _run(capsys, "domains")
    assert {d["name"] for d in out} == {"code_repair", "hidden_rule"}
    assert all(d["evaluators"] for d in out)


def test_domain_tasks_leak_checks_before_printing(capsys, monkeypatch):
    """The leak check runs in the command, not only in the test suite: a task generated on the
    command line and loaded into a workspace has bypassed the tests entirely (§47)."""
    import civitas.domains.base as base
    from civitas.domains import DomainLeak, DomainTask

    out = _run(capsys, "domain-tasks", "hidden_rule", "--seed", "3", "--count", "2")
    assert len(out) == 2 and all("evaluator_kind" in t for t in out)

    def leaky(self, *, seed, count, **kw):
        return [DomainTask(title="t", description="the answer is alpha", evaluator_spec={},
                           candidate_terms=("alpha",))]

    monkeypatch.setattr(type(base.get_domain("hidden_rule")), "generate", leaky)
    with pytest.raises(DomainLeak):
        main(["domain-tasks", "hidden_rule"])


def test_the_manifest_names_what_would_have_to_be_reproduced(capsys):
    """§46: a result is identified by the manifest that produced it, and a manifest missing the
    sandbox backend would not distinguish a run under container isolation from one without."""
    out = _run(capsys, "manifest")
    assert out["schema_version"] and out["code_version"]
    assert out["config_hash"]
    assert out["sandbox"]["backend"] and "isolation_level" in out["sandbox"]
    assert "sqlalchemy" in out["dependencies"]


def test_the_manifest_carries_no_secret(capsys, monkeypatch):
    """§52: secrets are dropped from the manifest by type, so a key added later is dropped too."""
    monkeypatch.setenv("CIVITAS_JWT_SECRET", "super-secret-value")
    from civitas.config import get_settings

    get_settings.cache_clear()
    try:
        out = _run(capsys, "manifest")
        assert "super-secret-value" not in json.dumps(out)
    finally:
        get_settings.cache_clear()


def test_an_unknown_command_exits_rather_than_guessing(capsys):
    with pytest.raises(SystemExit):
        main(["not-a-command"])
