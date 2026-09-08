"""The `civitas` command line (Part B §5, §46, §50).

`pyproject.toml` has declared `civitas = "civitas.cli:main"` since M1. This is that entry point.

Every command that changes anything prints what it did in a form that can be pasted back, and
every command that measures anything prints the manifest identity of the run alongside the number.
A benchmark figure without the configuration hash that produced it is not a result, and a CLI that
prints the figure alone is inviting exactly that (§46).

The commands are thin. Each is a few lines over a service function that the API and the notebook
call too — a CLI with its own logic would be a third implementation of the system, and the two
that already exist have to agree.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

from sqlalchemy import select

from civitas.config import Settings, get_settings


def _settings(args: argparse.Namespace) -> Settings:
    if getattr(args, "database_url", None):
        return get_settings().model_copy(update={"database_url": args.database_url})
    return get_settings()


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


# --------------------------------------------------------------------------- schema


def cmd_init_db(args: argparse.Namespace) -> int:
    """Create the schema.

    Alembic is the production path; this is the Colab and first-run path, and it says which it
    used so a schema created one way is not mistaken for one created the other (§34).
    """
    from civitas.persistence.engine import create_db_engine
    from civitas.persistence.models import Base

    settings = _settings(args)
    engine = create_db_engine(settings)
    if args.alembic:
        from alembic.config import Config

        from alembic import command

        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", settings.database_url)
        command.upgrade(cfg, "head")
        method = "alembic upgrade head"
    else:
        Base.metadata.create_all(engine)
        method = "metadata.create_all"
    _emit({"created": True, "method": method, "dialect": engine.dialect.name,
           "tables": len(Base.metadata.tables)})
    engine.dispose()
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    from civitas.persistence.engine import healthcheck

    report = healthcheck(_settings(args))
    _emit(report)
    return 0 if report.get("ok") else 1


# --------------------------------------------------------------------------- tenancy


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Create an organization, an admin and one key. The key is printed once and never again."""
    from civitas.api.security import bootstrap_organization
    from civitas.persistence.engine import session_scope

    with session_scope(_settings(args)) as session:
        organization, admin, plaintext = bootstrap_organization(
            session, name=args.name, slug=args.slug, admin_email=args.email
        )
        session.commit()
        _emit({
            "organization_id": str(organization.id), "slug": organization.slug,
            "admin": admin.email, "api_key": plaintext,
            "note": "this key is not recoverable — store it now",
        })
    return 0


def cmd_workspace(args: argparse.Namespace) -> int:
    from civitas.persistence.engine import session_scope
    from civitas.persistence.models import Organization, Workspace

    settings = _settings(args)
    with session_scope(settings) as session:
        organization = session.execute(
            select(Organization).where(Organization.slug == args.org)
        ).scalar_one_or_none()
        if organization is None:
            print(f"no organization with slug {args.org!r}", file=sys.stderr)
            return 2
        workspace = Workspace(
            organization_id=organization.id, name=args.name, slug=args.slug,
            environment_version=settings.environment_version,
        )
        session.add(workspace)
        session.commit()
        _emit({"workspace_id": str(workspace.id), "slug": workspace.slug,
               "environment_version": workspace.environment_version})
    return 0


# --------------------------------------------------------------------------- projects


def cmd_submit(args: argparse.Namespace) -> int:
    """Submit a high-level request (§5). Prints the plan, not an answer."""
    from civitas.persistence.engine import session_scope
    from civitas.services.projects import submit_request

    with session_scope(_settings(args)) as session:
        plan = submit_request(
            session, workspace_id=uuid.UUID(args.workspace), request=args.request
        )
        session.commit()
        _emit(plan.as_dict())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """A project's state, reconstructed from the database (§32).

    This is the command that demonstrates the property: it holds nothing between invocations, so
    running it after a restart gives the same answer as running it before.
    """
    from civitas.persistence.engine import session_scope
    from civitas.services.projects import project_status

    with session_scope(_settings(args)) as session:
        _emit(project_status(session, uuid.UUID(args.project)).as_dict())
    return 0


def cmd_timeline(args: argparse.Namespace) -> int:
    from civitas.persistence.engine import session_scope
    from civitas.services.projects import project_timeline

    with session_scope(_settings(args)) as session:
        _emit(project_timeline(session, uuid.UUID(args.project), limit=args.limit))
    return 0


# --------------------------------------------------------------------------- domains


def cmd_domains(args: argparse.Namespace) -> int:
    """List the domains this process can actually run (§5)."""
    from civitas.domains import all_domains

    _emit([
        {
            "name": d.name, "version": d.version,
            "stages": [s.family for s in d.stages()],
            "tools": list(d.tool_names()),
            "evaluators": [e.kind for e in d.evaluators()],
            "artifact_vocabulary": [a.value for a in d.artifact_vocabulary()],
        }
        for d in all_domains()
    ])
    return 0


def cmd_domain_tasks(args: argparse.Namespace) -> int:
    """Generate a domain's tasks and check each for a §47 leak before printing it.

    The check runs here, not only in the test suite: a task generated on the command line and
    loaded into a workspace has bypassed the tests entirely.
    """
    from civitas.domains import check_no_leak, get_domain

    domain = get_domain(args.domain)
    tasks = domain.generate(seed=args.seed, count=args.count)
    for task in tasks:
        check_no_leak(task)
    _emit([
        {"title": t.title, "family": t.task_family, "meta": t.meta,
         "evaluator_kind": t.evaluator_spec.get("kind"),
         "description_chars": len(t.description)}
        for t in tasks
    ])
    return 0


# --------------------------------------------------------------------------- measurement


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the newcomer benchmark and print the gated result (§22, §46).

    If a gate fails the metrics are absent rather than caveated — `metrics()` raises, and this
    prints the failed gates instead of a number. A gated-out run that still printed a success rate
    would be the exact failure mode the gates exist to prevent.
    """
    from civitas.experiments.benchmark import run_newcomer_benchmark
    from civitas.persistence.engine import session_scope
    from civitas.persistence.models import Organization

    settings = _settings(args)
    with session_scope(settings) as session:
        organization = session.execute(
            select(Organization).where(Organization.slug == args.org)
        ).scalar_one_or_none()
        if organization is None:
            print(f"no organization with slug {args.org!r}", file=sys.stderr)
            return 2
        result = run_newcomer_benchmark(
            session, organization_id=organization.id,
            seeds=[int(s) for s in args.seeds.split(",")],
            accumulation_passes=args.passes, settings=settings,
        )
        session.commit()

    payload: dict[str, Any] = {
        "benchmark": result.name,
        "config_hash": result.config_hash,
        "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail}
                  for g in result.gates],
    }
    try:
        payload["metrics"] = result.metrics()
    except Exception as exc:
        payload["metrics"] = None
        payload["unreadable_because"] = str(exc)
    _emit(payload)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
    return 0 if payload["metrics"] is not None else 1


def cmd_acceptance(args: argparse.Namespace) -> int:
    """Run §65's six levels in one campaign and report whether the system is accepted.

    Exits non-zero when any level fails, so this is usable as a release gate rather than as a
    report someone has to read carefully.
    """
    from civitas.acceptance import acceptance_report
    from civitas.persistence.engine import session_scope

    settings = _settings(args)
    include = (
        {int(x) for x in args.levels.split(",")} if args.levels else None
    )
    with session_scope(settings) as session:
        organization = session  # kept explicit: the report creates its own orgs per level
        del organization
        report = acceptance_report(
            session, seeds=[int(s) for s in args.seeds.split(",")],
            accumulation_passes=args.passes, settings=settings, include=include,
        )
        session.commit()

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True, default=str)

    for level in report["levels"]:
        mark = "PASS" if level["passed"] else "FAIL"
        value = level["value"]
        rendered = "withheld" if value is None else f"{value}"
        print(f"  {mark}  L{level['level']} {level['name']:34s} {rendered:>10s} "
              f"({level['unit']})")
        if level["withheld_reason"]:
            print(f"        withheld: {level['withheld_reason']}")
    print(f"\naccepted: {report['accepted']}")
    if report["failing_levels"]:
        print(f"failing levels: {report['failing_levels']}")
    return 0 if report["accepted"] else 1


def cmd_preflight(args: argparse.Namespace) -> int:
    """Check §64's production criteria against this deployment.

    Reports what is *actually* in force, not what is configured — the two differ exactly where it
    matters, and each check names the consequence rather than a rule number.
    """
    from civitas.preflight import run_preflight

    report = run_preflight(_settings(args))
    for check in report["checks"]:
        mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[check["status"]]
        print(f"  {mark}  {check['name']:36s} {check['detail']}")
    print(f"\nready for production: {report['ready']}")
    if report["blocking"]:
        print(f"blocking: {report['blocking']}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True, default=str)
    return 0 if report["ready"] else 1


def cmd_manifest(args: argparse.Namespace) -> int:
    """Print the run manifest: what would have to be reproduced to reproduce a result (§46)."""
    from civitas.experiments.manifest import (
        dependency_versions,
        environment_facts,
        git_commit,
        schema_version,
    )
    from civitas.runtime.sandbox import get_sandbox

    settings = _settings(args)
    _emit({
        "code_version": git_commit(),
        "schema_version": schema_version(),
        "environment": environment_facts(),
        "dependencies": dependency_versions(),
        # Secrets are dropped by type in `manifest_dict`, not by name — a key named
        # `provider_token` would survive a name-based filter (§52).
        "config": settings.manifest_dict(),
        "config_hash": settings.config_hash(),
        "sandbox": get_sandbox(settings).describe(),
    })
    return 0


# --------------------------------------------------------------------------- processes


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from civitas.api.app import create_app

    uvicorn.run(create_app(_settings(args)), host=args.host, port=args.port)
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    from civitas.persistence.engine import get_session_factory
    from civitas.workers.worker import WorkerConfig, run_workers

    settings = _settings(args)
    config = WorkerConfig(
        worker_id=args.worker_id,
        kinds=args.kinds.split(",") if args.kinds else None,
        poll_interval_seconds=args.poll_interval,
    )
    run_workers(get_session_factory(settings), count=args.count, config=config)
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="civitas", description=__doc__.split("\n")[0])
    parser.add_argument("--database-url", default=None,
                        help="override CIVITAS_DATABASE_URL for this command")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="create the schema")
    p.add_argument("--alembic", action="store_true",
                   help="run migrations instead of metadata.create_all")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("health", help="check the database is reachable")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("bootstrap", help="create an organization, an admin and one API key")
    p.add_argument("--name", required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--email", required=True)
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("workspace", help="create a workspace")
    p.add_argument("--org", required=True, help="organization slug")
    p.add_argument("--name", required=True)
    p.add_argument("--slug", required=True)
    p.set_defaults(func=cmd_workspace)

    p = sub.add_parser("submit", help="submit a high-level request (§5)")
    p.add_argument("--workspace", required=True)
    p.add_argument("request")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="a project's state, rebuilt from the database (§32)")
    p.add_argument("project")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("timeline", help="how a project's knowledge developed")
    p.add_argument("project")
    p.add_argument("--limit", type=int, default=200)
    p.set_defaults(func=cmd_timeline)

    p = sub.add_parser("domains", help="list registered domains")
    p.set_defaults(func=cmd_domains)

    p = sub.add_parser("domain-tasks", help="generate a domain's tasks, leak-checked")
    p.add_argument("domain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--count", type=int, default=4)
    p.set_defaults(func=cmd_domain_tasks)

    p = sub.add_parser("benchmark", help="run the newcomer benchmark (§22)")
    p.add_argument("--org", required=True, help="organization slug")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--out", default=None, help="also write the result to this path")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("acceptance", help="run §65's six scientific levels (§63, §65)")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--levels", default=None, help="comma-separated subset, e.g. 1,2,3")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_acceptance)

    p = sub.add_parser("preflight", help="check the production criteria (§64)")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("manifest", help="print the run manifest (§46)")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("serve", help="run the API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("worker", help="run job workers (§37)")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--worker-id", default="worker")
    p.add_argument("--kinds", default=None, help="comma-separated job kinds; default all")
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.set_defaults(func=cmd_worker)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
