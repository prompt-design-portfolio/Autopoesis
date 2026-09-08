"""Production readiness checks (Part B §64).

A checklist in a document is a checklist nobody runs. This is the same checklist as code, so
`civitas preflight` exits non-zero on a deployment that is not ready and can be a release gate.

Two rules shape every check here.

**Report what is in force, not what is configured.** The two differ exactly where it matters: a
sandbox backend that is *selected* but cannot enforce its process limit, tracing *configured*
against a package that is not installed, a rate limit *set* on a limiter that is per-process. Each
of those reads as "on" in the settings and is off in reality, and each is a check below.

**A warning is not a failure, and neither is silent.** `warn` marks a deployment that will work
and is worse than it could be; `fail` blocks. Nothing returns "unknown" quietly — a check that
cannot determine its answer says so and fails, because an unverifiable safety property is not a
satisfied one.
"""

from __future__ import annotations

from typing import Any

from civitas.config import Settings, get_settings

PASS, WARN, FAIL = "pass", "warn", "fail"


def _check(name: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, **extra}


def run_preflight(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    checks: list[dict[str, Any]] = []

    # -- identity and secrets ------------------------------------------------
    secret = settings.jwt_secret.get_secret_value() if settings.jwt_secret else ""
    checks.append(_check(
        "auth is enabled", PASS if settings.auth_enabled else FAIL,
        "every request is authenticated" if settings.auth_enabled
        else "auth_enabled is false: every caller is an admin of the first organization",
    ))
    checks.append(_check(
        "jwt secret is set and not a placeholder",
        PASS if len(secret) >= 32 and "change" not in secret.lower()
        and "not-a-real" not in secret.lower() else FAIL,
        f"length {len(secret)}"
        + ("" if len(secret) >= 32 else "; fewer than 32 characters"),
    ))
    checks.append(_check(
        "cors is not open to every origin",
        FAIL if "*" in settings.cors_origins else PASS,
        f"origins: {settings.cors_origins or 'none configured'}",
    ))

    # -- the database --------------------------------------------------------
    from civitas.persistence.engine import healthcheck

    try:
        health = healthcheck(settings)
    except Exception as exc:  # a database that cannot be reached is a failed check, not a crash
        health = {"ok": False, "dialect": "unknown", "error": str(exc)}
    checks.append(_check(
        "database is reachable", PASS if health.get("ok") else FAIL,
        health.get("error") or f"dialect {health.get('dialect')}",
    ))
    checks.append(_check(
        "database is postgresql",
        PASS if health.get("dialect") == "postgresql" else WARN,
        "SQLite is the Colab-compatible backend and is supported, but a multi-worker "
        "deployment serialises on its single writer"
        if health.get("dialect") != "postgresql" else "postgresql",
    ))

    # -- schema --------------------------------------------------------------
    try:
        from civitas.experiments.manifest import schema_version

        version = schema_version()
        migrated = bool(version) and version != "unknown"
    except Exception as exc:
        version, migrated = f"error: {exc}", False
    checks.append(_check(
        "schema is at a known migration", PASS if migrated else FAIL,
        f"schema_version={version}. A database built by `create_all` rather than by migrations "
        f"has no version, and cannot be upgraded in place",
    ))

    # -- the sandbox ---------------------------------------------------------
    from civitas.runtime.sandbox import IsolationLevel, get_sandbox

    try:
        sandbox = get_sandbox(settings)
        unenforced = list(sandbox.unenforced_limits())
        isolation = sandbox.isolation_level
    except Exception as exc:
        sandbox, unenforced, isolation = None, ["unknown"], None
        checks.append(_check("sandbox is available", FAIL, str(exc)))
    if sandbox is not None:
        checks.append(_check(
            "sandbox provides kernel or container isolation",
            PASS if isolation in (IsolationLevel.CONTAINER, IsolationLevel.KERNEL) else WARN,
            f"backend {sandbox.name}, isolation {isolation.value if isolation else 'none'}; "
            f"agent-written code runs under this",
        ))
        checks.append(_check(
            "every requested sandbox limit is enforced",
            PASS if not unenforced else FAIL,
            "all limits hold" if not unenforced
            else f"NOT enforced here: {unenforced}. A bound the caller believes is in force but "
                 f"is not is worse than no bound, because it is relied on",
            unenforced=unenforced,
        ))

    # -- observability -------------------------------------------------------
    from civitas.observability import tracing_state

    checks.append(_check(
        "metrics are enabled", PASS if settings.metrics_enabled else WARN,
        "/metrics serves Prometheus exposition" if settings.metrics_enabled
        else "metrics_enabled is false: /metrics returns 503 and nothing is observable",
    ))
    checks.append(_check(
        "logs are structured", PASS if settings.log_json else WARN,
        "JSON, one object per line" if settings.log_json
        else "log_json is false: an incident is grepped rather than queried",
    ))
    tracing = tracing_state()
    if settings.otel_endpoint and not tracing.get("enabled"):
        checks.append(_check(
            "tracing works if it is configured", FAIL,
            f"otel_endpoint is set but tracing is off: {tracing.get('reason')}",
        ))
    else:
        checks.append(_check(
            "tracing works if it is configured", PASS,
            "enabled" if tracing.get("enabled") else "not configured, and not claimed to be",
        ))

    # -- limits --------------------------------------------------------------
    checks.append(_check(
        "a rate limit is configured", PASS if settings.rate_limit_per_minute > 0 else WARN,
        f"{settings.rate_limit_per_minute}/minute, enforced per process — two replicas allow "
        f"twice this. A shared limiter belongs in front of a multi-replica deployment"
        if settings.rate_limit_per_minute > 0
        else "rate_limit_per_minute is 0: the limiter is disabled",
    ))

    # -- provider credentials ------------------------------------------------
    configured = [
        name for name, value in (
            ("anthropic", settings.anthropic_api_key),
            ("openai", settings.openai_api_key),
            ("google", settings.google_api_key),
        ) if value
    ]
    checks.append(_check(
        "a model provider is reachable or deliberately offline",
        PASS if configured or settings.default_provider in ("deterministic", "policy") else WARN,
        f"providers configured: {configured or 'none'}; default is "
        f"{settings.default_provider}",
    ))

    blocking = [c["name"] for c in checks if c["status"] == FAIL]
    return {
        "ready": not blocking,
        "checks": checks,
        "blocking": blocking,
        "warnings": [c["name"] for c in checks if c["status"] == WARN],
        "note": (
            "A warning is a deployment that works and is worse than it could be. A failure "
            "blocks. Nothing here reports 'unknown' quietly: a safety property that cannot be "
            "verified is not a satisfied one."
        ),
    }
