"""M10: does the newcomer advantage survive a change of domain?

Runs §22's procedure twice — once on the hidden-rule device, once on code repair — through the
*same* `run_newcomer_procedure`, and writes both results side by side. The comparison is only
meaningful because the two runs share an implementation: two copies of §22 could differ in ways
that show up as a domain effect.

The two budgets differ (7 calls against 10 operations; 5 against 6 candidate edits) so that the
two naive ceilings match at 0.500. Matching the ceiling rather than the raw budget is what makes
the two advantages comparable.
"""
from __future__ import annotations

import json
import pathlib
import sys
import uuid

from sqlalchemy.orm import sessionmaker

from civitas.config import Settings
from civitas.experiments.benchmark import run_newcomer_benchmark, run_repair_benchmark
from civitas.experiments.manifest import (
    dependency_versions,
    environment_facts,
    git_commit,
    schema_version,
)
from civitas.persistence.engine import create_db_engine
from civitas.persistence.models import Base, Organization
from civitas.persistence.session import install_guards
from civitas.persistence.types import utcnow
from civitas.runtime.sandbox import get_sandbox

SEEDS = [0, 1, 2, 3, 4]
PASSES = 3
RETRIEVAL = {"version": "retrieval/1.0-lexical", "semantic": False}


def _session(url: str):
    settings = Settings(database_url=url, environment_version="test-env-1",
                        jwt_secret="not-a-real-key")
    engine = create_db_engine(settings)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    install_guards(factory)
    return settings, factory()


def _gated(result) -> dict:
    """Metrics if the gates passed, and the reason they are absent if not (ARCHITECTURE §3.1)."""
    try:
        return {"metrics": result.metrics(), "unreadable_because": None}
    except Exception as exc:
        return {"metrics": None, "unreadable_because": str(exc),
                "raw": result.raw_metrics()}


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/m10"
    out = {}

    settings, session = _session(f"sqlite:///{base}-device.db")
    org = Organization(name="m10 device", slug=f"m10d-{uuid.uuid4().hex[:8]}")
    session.add(org)
    session.commit()
    device = run_newcomer_benchmark(
        session, organization_id=org.id, seeds=SEEDS, accumulation_passes=PASSES,
        settings=settings, retrieval_options=RETRIEVAL,
    )
    session.commit()
    out["hidden_rule"] = _gated(device)
    session.close()

    settings, session = _session(f"sqlite:///{base}-repair.db")
    org = Organization(name="m10 repair", slug=f"m10r-{uuid.uuid4().hex[:8]}")
    session.add(org)
    session.commit()
    repair = run_repair_benchmark(
        session, organization_id=org.id, seeds=SEEDS, accumulation_passes=PASSES,
        settings=settings, retrieval_options=RETRIEVAL,
    )
    session.commit()
    out["code_repair"] = _gated(repair)
    session.close()

    def advantage(key):
        m = out[key]["metrics"]
        return None if m is None else m["derived"]["newcomer_advantage"]

    payload = {
        "generated_at": utcnow().isoformat(),
        "milestone": "M10",
        "question": (
            "Does the newcomer advantage measured on the hidden-rule device survive a change of "
            "domain — a different task shape, a different agent policy, and an evaluator that "
            "executes the submission in the sandbox rather than comparing strings?"
        ),
        "manifest": {
            "code_version": git_commit(),
            "schema_version": schema_version(),
            "dependencies": dependency_versions(),
            "environment": environment_facts(),
            "sandbox": get_sandbox(settings).describe(),
            "seeds": SEEDS,
            "accumulation_passes": PASSES,
            "retrieval_options": RETRIEVAL,
            "procedure": "civitas.experiments.benchmark.run_newcomer_procedure",
        },
        "results": out,
        "comparison": {
            "hidden_rule_advantage": advantage("hidden_rule"),
            "code_repair_advantage": advantage("code_repair"),
            "note": (
                "Both naive probe ceilings are 0.500, so the two advantages are read against the "
                "same reference. A domain effect here is a domain effect, not a difficulty one."
            ),
        },
    }
    path = pathlib.Path("results/m10_domain_transfer.json")
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(f"wrote {path}")
    print(json.dumps(payload["comparison"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
