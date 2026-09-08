"""Production hardening (Part B §52, §53, §54, §58).

Four settings — `log_json`, `metrics_enabled`, `otel_endpoint` and `rate_limit_per_minute` — were
declared in M1 and read by nothing until M12. Most of this file exists to make "the setting does
something" checkable, because a configuration key that is read from the environment, printed into
the manifest and then ignored tells an operator a control exists.
"""

from __future__ import annotations

import json
import logging
import uuid

import pytest
from fastapi.testclient import TestClient

from civitas.api.middleware import RateLimiter, content_security_policy, security_headers
from civitas.config import Settings
from civitas.observability import (
    JsonFormatter,
    RedactingFilter,
    Registry,
    configure_tracing,
    correlated,
    redact,
)
from civitas.quotas import Quota, QuotaExceeded, check, quota_for, remaining, set_quota, usage
from civitas.runtime.providers.base import ProviderError
from civitas.runtime.providers.breaker import CircuitBreaker, CircuitOpen, guarded

# --------------------------------------------------------------------------- §52 redaction


@pytest.mark.parametrize("secret", [
    "civ_9fT3kQx1LmPz0aWq",
    "sk-abcdefgh12345678",
    "sk-ant-api03-longsecretvalue",
])
def test_a_credential_never_survives_a_log_line(secret):
    """Secrets reach logs through exception text and echoed request bodies far more often than
    through a deliberate `log.info(key)`, so the filter runs on the rendered message (§40)."""
    assert secret not in redact(f"provider rejected the request with {secret} attached")


def test_a_labelled_secret_is_redacted_without_losing_the_label():
    """`api_key=[redacted]` still tells a reader which field was present, which is the whole
    diagnostic value of the line."""
    out = redact('{"api_key": "hunter2hunter2", "model": "policy-v1"}')
    assert "hunter2hunter2" not in out
    assert "api_key" in out and "policy-v1" in out


def test_the_filter_redacts_the_message_a_record_actually_renders(caplog):
    """A filter that inspected `record.args` would miss `log.error("failed: %s", exc)`, which is
    exactly how a key in an exception string reaches a log."""
    record = logging.LogRecord("t", logging.ERROR, __file__, 1,
                               "provider said %s", ("civ_9fT3kQx1LmPz0aWq",), None)
    assert RedactingFilter().filter(record) is True
    assert "civ_9fT3" not in record.getMessage()


def test_json_logging_is_one_object_per_line_with_the_correlation_id():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello", (), None)
    with correlated("abc123"):
        RedactingFilter().filter(record)
        payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello"
    assert payload["correlation_id"] == "abc123"
    assert payload["level"] == "INFO"


# --------------------------------------------------------------------------- §53 metrics


def test_a_counter_never_incremented_reports_none_rather_than_zero():
    """ARCHITECTURE §3.9 in the monitoring stack. "Nothing happened" and "we have no data" are
    different facts, and only the exposition format may flatten them."""
    registry = Registry()
    registry.counter("thing_total")
    assert registry.value("thing_total", kind="a") is None
    registry.inc("thing_total", kind="a")
    assert registry.value("thing_total", kind="a") == 1.0


def test_histogram_buckets_are_cumulative_and_monotonic():
    """A non-monotonic bucket series makes a histogram report latencies that never happened."""
    registry = Registry()
    registry.histogram("dur_seconds", (0.1, 1.0, 10.0))
    for value in (0.05, 0.5, 0.5, 50.0):
        registry.observe("dur_seconds", value)
    rendered = registry.render()
    counts = [
        int(line.rsplit(" ", 1)[1]) for line in rendered.splitlines()
        if line.startswith("dur_seconds_bucket")
    ]
    assert counts == sorted(counts), rendered
    assert counts[-1] == 4
    assert "dur_seconds_count{} 4" in rendered or "dur_seconds_count 4" in rendered


def test_the_exposition_escapes_label_values():
    registry = Registry()
    registry.inc("thing_total", route='/a"b\\c')
    assert '\\"' in registry.render()


def test_observing_an_undeclared_histogram_is_an_error_not_a_silent_counter():
    registry = Registry()
    with pytest.raises(ValueError):
        registry.observe("never_declared", 1.0)


def test_tracing_reports_that_it_is_configured_but_unavailable():
    """The failure this exists to surface: `otel_endpoint` set, package absent, and nobody finds
    out until an incident when the traces are not there."""
    state = configure_tracing(Settings(
        database_url="sqlite://", jwt_secret="x", otel_endpoint="http://otel:4318",
    ))
    if state["enabled"]:
        pytest.skip("OpenTelemetry is installed here; the degraded path is not exercised")
    assert "not installed" in state["reason"]


def test_a_span_is_a_no_op_when_tracing_is_off():
    from civitas.observability import span

    with span("work", kind="test") as active:
        active.set_attribute("k", "v")


# --------------------------------------------------------------------------- §52 headers


def test_the_csp_forbids_inline_script_and_every_foreign_origin_but_the_docs_cdn():
    policy = content_security_policy()
    assert "default-src 'self'" in policy
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert "frame-ancestors 'none'" in policy
    assert "connect-src 'self'" in policy
    # The one exception is explicit and scoped to scripts and styles.
    assert policy.count("https://cdn.jsdelivr.net") == 2
    assert "script-src 'self' https://cdn.jsdelivr.net" in policy


def test_hsts_is_not_sent_from_a_local_deployment():
    """Sending HSTS from a local HTTP deployment pins a browser to https for a host that does not
    serve it, and the operator cannot undo it."""
    local = security_headers(Settings(database_url="sqlite://", jwt_secret="x",
                                      api_host="127.0.0.1"))
    remote = security_headers(Settings(database_url="sqlite://", jwt_secret="x",
                                       api_host="civitas.example.com"))
    assert "strict-transport-security" not in local
    assert "strict-transport-security" in remote


def test_every_response_carries_the_security_headers(client):
    response = client.get("/healthz")
    for header in ("content-security-policy", "x-content-type-options", "referrer-policy",
                   "x-frame-options"):
        assert header in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"


def test_every_response_carries_a_correlation_id_and_echoes_the_one_it_was_given(client):
    assert client.get("/healthz").headers["x-request-id"]
    given = client.get("/healthz", headers={"x-request-id": "trace-me-42"})
    assert given.headers["x-request-id"] == "trace-me-42"


def test_an_oversized_body_is_refused_before_it_is_read(client, api_workspace):
    response = client.post(
        f"/api/v1/workspaces/{api_workspace['id']}/artifacts",
        json={"type": "observation", "title": "big", "body": "x"},
        headers={"content-length": str(64 * 1024 * 1024)},
    )
    assert response.status_code == 413


# --------------------------------------------------------------------------- §52 rate limiting


def test_the_limiter_refuses_past_the_limit_and_says_when_to_retry():
    clock = [0.0]
    limiter = RateLimiter(3, clock=lambda: clock[0])
    assert [limiter.check("k")[0] for _ in range(3)] == [True, True, True]

    allowed, remaining_calls, retry_after = limiter.check("k")
    assert allowed is False and remaining_calls == 0 and retry_after > 0


def test_the_window_slides_rather_than_resetting_on_a_boundary():
    """A fixed window lets a caller send the whole budget in the last second of one window and
    again in the first second of the next — twice the limit at the moment it matters."""
    clock = [0.0]
    limiter = RateLimiter(2, clock=lambda: clock[0])
    limiter.check("k")
    clock[0] = 59.0
    limiter.check("k")
    clock[0] = 59.5
    assert limiter.check("k")[0] is False, "a third call inside 60s was allowed"
    clock[0] = 60.5  # the first hit ages out; exactly one slot frees
    assert limiter.check("k")[0] is True
    assert limiter.check("k")[0] is False


def test_limits_are_per_principal_not_global():
    limiter = RateLimiter(1)
    assert limiter.check("key-a")[0] is True
    assert limiter.check("key-b")[0] is True
    assert limiter.check("key-a")[0] is False


def test_health_and_readiness_are_never_rate_limited(api, bootstrapped):
    """A limiter that can make a load balancer believe the process is unhealthy turns a traffic
    spike into an outage."""
    _org, _admin, key = bootstrapped
    api.state.rate_limiter.per_minute = 1
    with TestClient(api) as c:
        c.headers.update({"x-api-key": key})
        assert c.get("/api/v1/workspaces").status_code == 200
        assert c.get("/api/v1/workspaces").status_code == 429
        for _ in range(5):
            assert c.get("/healthz").status_code == 200
            assert c.get("/readyz").status_code == 200


def test_a_refused_request_is_counted_where_a_dashboard_can_see_it(api, bootstrapped):
    from civitas.observability import REGISTRY

    _org, _admin, key = bootstrapped
    api.state.rate_limiter.per_minute = 1
    before = REGISTRY.value("civitas_http_rate_limited_total", route="/api/v1/workspaces") or 0.0
    with TestClient(api) as c:
        c.headers.update({"x-api-key": key})
        c.get("/api/v1/workspaces")
        assert c.get("/api/v1/workspaces").status_code == 429
    after = REGISTRY.value("civitas_http_rate_limited_total", route="/api/v1/workspaces")
    assert after == before + 1.0


def test_the_limiter_keys_on_the_prefix_and_never_holds_a_whole_key(api, bootstrapped):
    """The limiter's state is readable in a heap dump and printed in diagnostics; a prefix
    identifies a key without being one."""
    _org, _admin, key = bootstrapped
    with TestClient(api) as c:
        c.get("/api/v1/workspaces", headers={"x-api-key": key})
    keys = list(api.state.rate_limiter._windows)
    assert keys and all(k != key for k in keys)
    assert all(len(k) <= 12 or k.startswith("addr:") for k in keys)


# --------------------------------------------------------------------------- §53 the endpoint


def test_metrics_are_served_in_prometheus_exposition_format(client):
    client.get("/healthz")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE civitas_http_requests_total counter" in response.text


def test_metrics_carry_no_tenant_identifier(client, api_workspace):
    """A metrics endpoint is the easiest place to leak a tenant list, and it is the one endpoint
    people expose to a scraper without thinking."""
    client.get(f"/api/v1/workspaces/{api_workspace['id']}/tasks")
    body = client.get("/metrics").text
    # The route *template* legitimately contains `{workspace_id}` — that is the point of using a
    # template. What must never appear is a workspace's actual id.
    assert api_workspace["id"] not in body
    assert "{workspace_id}" in body


def test_the_route_label_is_a_template_not_a_path(client, api_workspace):
    """A label whose value is an id gives the metric unbounded cardinality — which is how a
    monitoring system is taken down by the thing it monitors."""
    client.get(f"/api/v1/workspaces/{api_workspace['id']}/tasks")
    body = client.get("/metrics").text
    assert "{workspace_id}" in body


def test_metrics_can_be_turned_off_and_say_so(settings, session_factory, db, read_only_factory):
    from civitas.api.app import SAFE_METHODS, create_app, get_db

    app = create_app(settings.model_copy(update={"metrics_enabled": False}))

    def _override(request):
        factory = read_only_factory if request.method in SAFE_METHODS else session_factory
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override
    with TestClient(app) as c:
        response = c.get("/metrics")
    assert response.status_code == 503
    assert "disabled by configuration" in response.text


def test_health_reports_what_is_actually_in_force(client):
    body = client.get("/healthz").json()
    assert body["rate_limit_is_per_process"] is True
    assert body["tracing"]["enabled"] is False and body["tracing"]["reason"]
    assert isinstance(body["sandbox_unenforced_limits"], list)
    assert body["auth_enabled"] is True


# --------------------------------------------------------------------------- §54 the breaker


def _failing():
    raise ProviderError("provider is down", retryable=True)


def test_the_circuit_opens_after_consecutive_failures_and_then_skips():
    clock = [0.0]
    breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=30, clock=lambda: clock[0])

    attempted = 0
    skipped = 0
    for _ in range(10):
        try:
            guarded("flaky", "m", _failing, breaker=breaker)
        except CircuitOpen:
            skipped += 1
        except ProviderError:
            attempted += 1

    assert attempted == 3, "the breaker did not stop the attempts"
    assert skipped == 7
    assert breaker.state("flaky/m").is_open


def test_one_success_is_not_enough_to_open_and_a_success_resets_the_count():
    breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=30)
    for _ in range(2):
        with pytest.raises(ProviderError):
            guarded("p", "m", _failing, breaker=breaker)
    guarded("p", "m", lambda: "ok", breaker=breaker)
    assert breaker.state("p/m").consecutive_failures == 0
    assert not breaker.state("p/m").is_open


def test_recovery_lets_exactly_one_call_through():
    """Letting the whole backlog through at once is how a recovering provider is knocked over
    again."""
    clock = [0.0]
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=10, clock=lambda: clock[0])
    for _ in range(2):
        with pytest.raises(ProviderError):
            guarded("p", "m", _failing, breaker=breaker)

    clock[0] = 5.0
    with pytest.raises(CircuitOpen):
        guarded("p", "m", _failing, breaker=breaker)

    clock[0] = 11.0
    with pytest.raises(ProviderError) as caught:
        guarded("p", "m", _failing, breaker=breaker)
    assert not isinstance(caught.value, CircuitOpen), "the probe call was skipped"
    assert breaker.state("p/m").is_open, "the failed probe did not re-open the circuit"


def test_breakers_are_per_model_not_per_provider():
    """A provider whose large model is rate-limited while its small one answers is the common
    case; one breaker for both turns a partial outage into a total one."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=30)
    for _ in range(3):
        with pytest.raises(ProviderError):
            guarded("openai", "big", _failing, breaker=breaker)
    assert breaker.state("openai/big").is_open
    assert guarded("openai", "small", lambda: "ok", breaker=breaker) == "ok"


def test_a_local_bug_does_not_open_the_circuit():
    """A `TypeError` in this codebase is not evidence about the provider, and treating it as one
    would make a local defect look like an outage."""
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=30)

    def bug():
        raise TypeError("a defect in our own code")

    with pytest.raises(TypeError):
        guarded("p", "m", bug, breaker=breaker)
    assert not breaker.state("p/m").is_open


def test_a_skipped_call_reports_itself_as_retryable_infrastructure():
    """§47: an outage must not be counted as an agent's failure. `retryable=True` is what makes
    the episode terminate as `provider_failure`, which `is_readable` excludes from success
    rates."""
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=30)
    with pytest.raises(ProviderError):
        guarded("p", "m", _failing, breaker=breaker)
    with pytest.raises(CircuitOpen) as caught:
        guarded("p", "m", _failing, breaker=breaker)
    assert caught.value.retryable is True
    assert "not attempted" in str(caught.value)


# --------------------------------------------------------------------------- §58 quotas


def test_an_organization_with_no_quota_is_unlimited(db, org):
    assert quota_for(db, org.id).unlimited is True
    check(db, org.id)  # must not raise


def test_unlimited_headroom_is_reported_as_none_not_as_a_large_number(db, org):
    report = remaining(db, org.id)
    assert report["remaining"]["tokens"] is None
    assert report["remaining"]["cost_usd"] is None


def test_a_nonsense_limit_is_ignored_rather_than_halting_the_organization(db, org):
    """A zero or negative limit is almost always a typo, and a quota that silently stops an
    organization is worse than one that is ignored and visible in the report."""
    set_quota(db, org.id, Quota(max_tokens=0, max_cost_usd=-1.0))
    db.flush()
    assert quota_for(db, org.id).unlimited is True


def test_usage_is_computed_from_episodes_and_counts_benchmark_probes(db, org, workspace):
    """Excluding probes would let a campaign spend an organization's month while the quota report
    showed it idle."""
    from civitas.domain.enums import TerminationReason
    from civitas.persistence.models import AgentProfile, Episode

    profile = AgentProfile(workspace_id=workspace.id, name="p")
    db.add(profile)
    db.flush()
    for probe in (False, True):
        db.add(Episode(
            workspace_id=workspace.id, agent_profile_id=profile.id,
            model_provider="deterministic", model_name="d",
            termination_reason=TerminationReason.EVALUATOR_SUCCESS,
            tokens_used=500, cost_usd=0.5, is_benchmark_probe=probe,
        ))
    db.flush()

    spent = usage(db, org.id)
    assert spent.tokens == 1000 and spent.episodes == 2
    assert spent.cost_usd == pytest.approx(1.0)


def test_an_episode_is_refused_before_it_exists_when_the_quota_is_spent(db, org, workspace):
    """Checked before the row, not after: an episode created and then refused would be counted by
    the very usage query the quota is measured with."""
    from civitas.domain.enums import ExperimentArm, TerminationReason
    from civitas.persistence.models import AgentProfile, Episode
    from civitas.runtime.episode import EpisodeRunner, EpisodeSpec
    from civitas.runtime.providers.offline import DeterministicProvider
    from civitas.runtime.tools.builtin import default_registry

    profile = AgentProfile(workspace_id=workspace.id, name="p")
    db.add(profile)
    db.flush()
    db.add(Episode(
        workspace_id=workspace.id, agent_profile_id=profile.id,
        model_provider="deterministic", model_name="d",
        termination_reason=TerminationReason.EVALUATOR_SUCCESS, tokens_used=5000,
    ))
    db.flush()
    set_quota(db, org.id, Quota(max_tokens=1000))
    db.flush()

    before = db.query(Episode).count()
    runner = EpisodeRunner(db, provider=DeterministicProvider(), tools=default_registry())
    with pytest.raises(QuotaExceeded) as caught:
        runner.run(EpisodeSpec(
            workspace_id=workspace.id, agent_profile_id=profile.id,
            provider_name="deterministic", model_name="d", system_prompt="s",
            experiment_arm=ExperimentArm.SOLO,
        ))
    assert caught.value.kind == "tokens"
    assert db.query(Episode).count() == before, "a refused episode still created a row"


def test_the_quota_endpoint_is_scoped_to_the_callers_organization(client, bootstrapped):
    org, _admin, _key = bootstrapped
    assert client.get(f"/api/v1/organizations/{org.id}/quota").status_code == 200
    assert client.get(f"/api/v1/organizations/{uuid.uuid4()}/quota").status_code == 404


def test_setting_a_quota_over_the_api_is_admin_only(api, bootstrapped, db):
    from civitas.api.security import create_api_key
    from civitas.domain.enums import ActorKind, Role
    from civitas.persistence.models import ServiceIdentity

    org, _admin, _key = bootstrapped
    identity = ServiceIdentity(organization_id=org.id, name="op", kind=ActorKind.SERVICE,
                               role=Role.OPERATOR)
    db.add(identity)
    db.flush()
    _row, operator_key = create_api_key(db, organization_id=org.id, name="op",
                                        service_identity_id=identity.id)
    db.commit()

    with TestClient(api) as c:
        response = c.put(f"/api/v1/organizations/{org.id}/quota",
                         json={"max_tokens": 10}, headers={"x-api-key": operator_key})
    assert response.status_code == 403
