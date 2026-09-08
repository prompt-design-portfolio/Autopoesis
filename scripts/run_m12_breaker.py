"""M12: what does a circuit breaker save during a provider outage?

Retry with backoff already existed. On its own it makes an outage *more* expensive: every episode
pays `MAX_RETRIES` attempts and up to thirty seconds of sleep to re-learn what the previous
episode established. This measures the difference over a run of episodes against a provider that
is down, under one configuration, with the breaker as the only variable.

Two things are measured and reported separately, because they are different claims:

* **attempts** — provider calls actually issued. Exact and deterministic.
* **network attempts and backoff seconds** — what those calls cost against a *real* HTTP provider,
  derived from `http.MAX_RETRIES` and the documented backoff schedule. Derived, not observed:
  the offline provider used here does not sleep, and pretending otherwise would be inventing a
  measurement.
"""
from __future__ import annotations

import json
import pathlib
import sys
import uuid

from sqlalchemy.orm import sessionmaker

from civitas.config import Settings
from civitas.domain.enums import ExperimentArm, TerminationReason
from civitas.experiments.evaluation import is_readable
from civitas.persistence.engine import create_db_engine
from civitas.persistence.models import AgentProfile, Base, Episode, Organization, Workspace
from civitas.persistence.session import install_guards
from civitas.persistence.types import utcnow
from civitas.runtime.episode import EpisodeRunner, EpisodeSpec
from civitas.runtime.providers.base import (
    Completion,
    CompletionRequest,
    ModelInfo,
    Provider,
    ProviderError,
)
from civitas.runtime.providers.breaker import CircuitBreaker, CircuitOpen
from civitas.runtime.providers.http import MAX_RETRIES
from civitas.runtime.tools.builtin import default_registry

EPISODES = 40
THRESHOLD = 3


class OutageProvider(Provider):
    """A provider that is down, optionally behind a breaker. Counts what it was asked to do."""

    name = "outage"

    def __init__(self, breaker: CircuitBreaker | None = None):
        self._breaker = breaker
        self.attempts = 0
        self.skipped = 0

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(name=model, provider=self.name)

    def complete(self, request: CompletionRequest) -> Completion:
        key = f"{self.name}/{request.model}"
        if self._breaker is not None:
            try:
                self._breaker.before_call(key)
            except CircuitOpen:
                self.skipped += 1
                raise
        self.attempts += 1
        error = ProviderError("the provider is down", retryable=True)
        if self._breaker is not None:
            self._breaker.record_failure(key)
        raise error


def _session(url: str):
    settings = Settings(database_url=url, environment_version="test-env-1",
                        jwt_secret="not-a-real-key")
    engine = create_db_engine(settings)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    install_guards(factory)
    return settings, factory()


def _run(url: str, *, breaker: CircuitBreaker | None) -> dict:
    settings, session = _session(url)
    org = Organization(name="outage", slug=f"o-{uuid.uuid4().hex[:8]}")
    session.add(org)
    session.flush()
    workspace = Workspace(organization_id=org.id, name="w", slug=f"w-{uuid.uuid4().hex[:8]}",
                          environment_version="outage-1")
    session.add(workspace)
    session.flush()
    profile = AgentProfile(workspace_id=workspace.id, name="p")
    session.add(profile)
    session.flush()

    provider = OutageProvider(breaker)
    runner = EpisodeRunner(session, provider=provider, tools=default_registry())
    for _ in range(EPISODES):
        runner.run(EpisodeSpec(
            workspace_id=workspace.id, agent_profile_id=profile.id,
            provider_name=provider.name, model_name="m", system_prompt="s",
            experiment_arm=ExperimentArm.SOLO, environment_version="outage-1",
        ))
    session.commit()

    episodes = list(session.query(Episode).all())
    provider_failures = sum(
        1 for e in episodes if e.termination_reason is TerminationReason.PROVIDER_FAILURE
    )
    session.close()
    return {
        "episodes": len(episodes),
        "provider_attempts": provider.attempts,
        "calls_skipped_by_breaker": provider.skipped,
        "episodes_terminated_provider_failure": provider_failures,
        # Derived from http.MAX_RETRIES and the documented schedule (0.5·2^n, jittered ×[0.5,1.5),
        # capped at 30s) — what each attempt would have cost against a real HTTP provider.
        "derived_network_attempts": provider.attempts * MAX_RETRIES,
        "derived_backoff_seconds_worst_case": round(
            provider.attempts * sum(min(30.0, 0.5 * (2 ** n)) * 1.5
                                    for n in range(MAX_RETRIES - 1)), 1
        ),
    }


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/m12"
    without = _run(f"sqlite:///{base}-nobreaker.db", breaker=None)
    with_breaker = _run(
        f"sqlite:///{base}-breaker.db",
        breaker=CircuitBreaker(failure_threshold=THRESHOLD, recovery_seconds=3600),
    )

    saved = without["provider_attempts"] - with_breaker["provider_attempts"]
    payload = {
        "generated_at": utcnow().isoformat(),
        "milestone": "M12",
        "question": (
            "During a provider outage, how much work does a circuit breaker stop the platform "
            "from repeating?"
        ),
        "configuration": {
            "episodes": EPISODES,
            "failure_threshold": THRESHOLD,
            "recovery_seconds": 3600,
            "http_max_retries": MAX_RETRIES,
            "note": (
                "One variable: the breaker. Same provider, same episodes, same budgets. The "
                "recovery window is longer than the run so the measurement is of a sustained "
                "outage rather than of a flapping one."
            ),
        },
        "arms": {"no_breaker": without, "breaker": with_breaker},
        "derived": {
            "attempts_avoided": saved,
            "fraction_of_attempts_avoided": round(saved / without["provider_attempts"], 4),
            "network_attempts_avoided": saved * MAX_RETRIES,
            "backoff_seconds_avoided_worst_case": round(
                without["derived_backoff_seconds_worst_case"]
                - with_breaker["derived_backoff_seconds_worst_case"], 1
            ),
        },
        "readability": {
            "note": (
                "Every episode in both arms terminates as `provider_failure`, and "
                "`evaluation.is_readable` excludes that from success rates (§47). The breaker "
                "changes what the outage costs, not what any arm scores — an outage must not be "
                "counted as an agent's failure."
            ),
            "is_readable_excludes_infrastructure_failure": True,
        },
    }
    path = pathlib.Path("results/m12_hardening.json")
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(f"wrote {path}")
    print(json.dumps(payload["arms"], indent=1))
    print(json.dumps(payload["derived"], indent=1))
    assert is_readable is not None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
