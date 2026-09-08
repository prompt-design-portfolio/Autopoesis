"""Structured logging, metrics and tracing (Part B §53).

Four settings have existed since M1 and none of them did anything: `log_json`, `metrics_enabled`,
`otel_endpoint` and `log_level`. A configuration key that is read from the environment, printed
into the manifest, and then ignored is worse than no key at all — it tells an operator a control
exists.

**Why the metrics registry is written here rather than imported.** `prometheus_client` is not a
dependency, and adding one to expose four counters would make an air-gapped or Colab install
(§33, §34) depend on a package it cannot fetch. The exposition format is a documented text
protocol; the parts this needs are a hundred lines. Anything that scrapes Prometheus scrapes this.

**Why tracing degrades and metrics do not.** OpenTelemetry is genuinely optional: when the package
is absent, `span()` is a context manager that does nothing, and `/metrics` still reports
everything. A metric that vanished with an optional dependency would make "we have no data" and
"nothing happened" the same observation — which is the failure ARCHITECTURE §3.9 exists to
prevent, arriving through the monitoring stack.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from civitas.config import Settings

#: The current request's correlation id. A context variable rather than a thread local: the API is
#: async and one thread serves many requests.
try:  # pragma: no cover - contextvars is stdlib on every supported version
    from contextvars import ContextVar

    _CORRELATION: ContextVar[str | None] = ContextVar("civitas_correlation_id", default=None)
except ImportError:  # pragma: no cover
    _CORRELATION = None  # type: ignore[assignment]

#: Patterns redacted from every log record. Secrets reach logs through exception text and echoed
#: request bodies far more often than through a deliberate `log.info(key)`, so the filter runs on
#: the *rendered* message rather than on the arguments (§40, §52).
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bciv_[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\b(?:sk-ant|anthropic)[A-Za-z0-9_\-]{8,}"),
    # The optional closing quote after the key name matters: a secret reaches a log as JSON far
    # more often than as `key=value`, and `"api_key": "..."` has a quote between the name and the
    # colon. Without it the most common shape is the one that slips through.
    re.compile(r"(?i)\b(authorization|x-api-key|api[_-]?key|password|secret|token)"
               r"\"?\s*[=:]\s*\"?([A-Za-z0-9_\-\.]{6,})\"?"),
)

REDACTED = "[redacted]"


def redact(text: str) -> str:
    """Remove anything that looks like a credential from a string."""
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 2:
            text = pattern.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def correlation_id() -> str | None:
    return _CORRELATION.get() if _CORRELATION is not None else None


@contextlib.contextmanager
def correlated(value: str | None = None) -> Iterator[str]:
    """Attach a correlation id to everything logged inside this block."""
    token_value = value or uuid.uuid4().hex
    token = _CORRELATION.set(token_value)
    try:
        yield token_value
    finally:
        _CORRELATION.reset(token)


class RedactingFilter(logging.Filter):
    """Redacts credentials from the rendered message and attaches the correlation id."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(record.getMessage())
            record.args = ()
        except Exception:  # pragma: no cover - a broken __str__ must not lose the record
            record.msg = "[unrenderable log record]"
            record.args = ()
        record.correlation_id = correlation_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Machine-readable is the point: a log an operator has to grep with
    a regular expression is a log that cannot be queried during an incident."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(settings: Settings | None = None) -> None:
    """Install the root handler. Idempotent, so a re-import does not double every line."""
    from civitas.config import get_settings

    settings = settings or get_settings()
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    for handler in list(root.handlers):
        if getattr(handler, "_civitas", False):
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler._civitas = True  # type: ignore[attr-defined]
    handler.setFormatter(
        JsonFormatter() if settings.log_json
        else logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(correlation_id)s] %(name)s: %(message)s"
        )
    )
    handler.addFilter(RedactingFilter())
    root.addHandler(handler)


# --------------------------------------------------------------------------- metrics


@dataclass
class _Series:
    name: str
    kind: str
    help: str
    values: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)
    #: Histogram bucket upper bounds, ascending, excluding +Inf.
    buckets: tuple[float, ...] = ()
    counts: dict[tuple[tuple[str, str], ...], list[int]] = field(default_factory=dict)
    sums: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)


def _key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, str(v)) for k, v in (labels or {}).items()))


class Registry:
    """A minimal Prometheus registry: counters, gauges and histograms.

    In-process, and that is a stated limitation rather than an oversight: with several workers the
    scrape target is one process, so a deployment that runs many must scrape each. Aggregating
    across processes needs a pushgateway or a shared store, and inventing one here would be a
    larger claim than this milestone can support.
    """

    def __init__(self) -> None:
        self._series: dict[str, _Series] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help: str = "") -> None:
        self._declare(name, "counter", help)

    def gauge(self, name: str, help: str = "") -> None:
        self._declare(name, "gauge", help)

    def histogram(self, name: str, buckets: tuple[float, ...], help: str = "") -> None:
        series = self._declare(name, "histogram", help)
        series.buckets = tuple(sorted(buckets))

    def _declare(self, name: str, kind: str, help: str) -> _Series:
        with self._lock:
            existing = self._series.get(name)
            if existing is None:
                existing = _Series(name=name, kind=kind, help=help)
                self._series[name] = existing
            return existing

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        series = self._series.get(name) or self._declare(name, "counter", "")
        with self._lock:
            key = _key(labels)
            series.values[key] = series.values.get(key, 0.0) + value

    def set(self, name: str, value: float, **labels: str) -> None:
        series = self._series.get(name) or self._declare(name, "gauge", "")
        with self._lock:
            series.values[_key(labels)] = value

    def observe(self, name: str, value: float, **labels: str) -> None:
        series = self._series.get(name)
        if series is None or series.kind != "histogram":
            raise ValueError(f"{name!r} is not a declared histogram")
        with self._lock:
            key = _key(labels)
            counts = series.counts.setdefault(key, [0] * (len(series.buckets) + 1))
            for i, bound in enumerate(series.buckets):
                if value <= bound:
                    counts[i] += 1
            counts[-1] += 1
            series.sums[key] = series.sums.get(key, 0.0) + value

    def value(self, name: str, **labels: str) -> float | None:
        """The current value, or `None` when nothing has been recorded under those labels.

        `None`, not 0.0 (ARCHITECTURE §3.9): a counter that has never been incremented and one
        that has been incremented by zero are different facts, and only the exposition format is
        allowed to flatten them.
        """
        series = self._series.get(name)
        if series is None:
            return None
        if series.kind == "histogram":
            return series.sums.get(_key(labels))
        return series.values.get(_key(labels))

    def render(self) -> str:
        """The Prometheus text exposition format, version 0.0.4."""
        lines: list[str] = []
        with self._lock:
            for name in sorted(self._series):
                series = self._series[name]
                if series.help:
                    lines.append(f"# HELP {name} {series.help}")
                lines.append(f"# TYPE {name} {series.kind}")
                if series.kind == "histogram":
                    for key in sorted(series.counts):
                        # `counts[i]` is already the number of observations <= buckets[i]:
                        # `observe` increments every bucket a value falls under. Summing them
                        # again here would double-count, and the bucket series would stop being
                        # monotonic — which is how a histogram silently reports latencies that
                        # never happened.
                        counts = series.counts[key]
                        for i, bound in enumerate(series.buckets):
                            lines.append(
                                f"{name}_bucket{_labels(key, le=_num(bound))} {counts[i]}"
                            )
                        lines.append(f"{name}_bucket{_labels(key, le='+Inf')} {counts[-1]}")
                        lines.append(f"{name}_sum{_labels(key)} {series.sums.get(key, 0.0)}")
                        lines.append(f"{name}_count{_labels(key)} {counts[-1]}")
                else:
                    for key in sorted(series.values):
                        lines.append(f"{name}{_labels(key)} {series.values[key]}")
        return "\n".join(lines) + "\n"


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _labels(key: tuple[tuple[str, str], ...], **extra: str) -> str:
    pairs = list(key) + sorted(extra.items())
    if not pairs:
        return ""
    rendered = ",".join(
        f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for k, v in pairs
    )
    return "{" + rendered + "}"


#: The process-wide registry. Declared here so every metric name in the system has one home and a
#: reader can see the whole surface at once.
REGISTRY = Registry()

REGISTRY.counter("civitas_http_requests_total", "HTTP requests by method, route and status.")
REGISTRY.histogram(
    "civitas_http_request_seconds", (0.005, 0.025, 0.1, 0.5, 1.0, 5.0),
    "HTTP request duration in seconds.",
)
REGISTRY.counter("civitas_http_rate_limited_total", "Requests refused by the rate limiter.")
REGISTRY.counter("civitas_auth_failures_total", "Authentication failures by reason.")
REGISTRY.counter("civitas_episodes_total", "Episodes completed, by arm and termination reason.")
REGISTRY.counter("civitas_episode_tokens_total", "Tokens consumed by episodes, by arm.")
REGISTRY.counter("civitas_provider_calls_total", "Provider calls, by provider and outcome.")
REGISTRY.counter("civitas_provider_calls_skipped_total",
                 "Provider calls not attempted because a circuit breaker was open.")
REGISTRY.gauge("civitas_circuit_breaker_open", "1 when a provider's breaker is open.")
REGISTRY.counter("civitas_quota_denied_total", "Actions refused for exceeding an org quota.")
REGISTRY.gauge("civitas_build_info", "Always 1; the labels carry the build identity.")


def record_build_info(settings: Settings) -> None:
    REGISTRY.set(
        "civitas_build_info", 1.0,
        version=settings.app_version, environment_version=settings.environment_version,
    )


# --------------------------------------------------------------------------- tracing


class _NoopSpan:
    """What `span()` yields when OpenTelemetry is absent. Accepts attributes and discards them,
    so instrumented code reads identically either way."""

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None


_TRACER: Any = None
_TRACING_STATE = {"enabled": False, "reason": "not configured"}


def configure_tracing(settings: Settings | None = None) -> dict[str, Any]:
    """Set up OTLP tracing when it is both configured and installed.

    Returns the state rather than logging it and moving on, so `/healthz` can report whether
    tracing is actually on. "Configured but the package is missing" is the failure that otherwise
    goes unnoticed until an incident, when the traces are not there.
    """
    global _TRACER
    from civitas.config import get_settings

    settings = settings or get_settings()
    if not settings.otel_endpoint:
        _TRACING_STATE.update(enabled=False, reason="no otel_endpoint configured")
        return dict(_TRACING_STATE)
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        _TRACING_STATE.update(
            enabled=False,
            reason=f"otel_endpoint is set but OpenTelemetry is not installed ({exc.name})",
        )
        return dict(_TRACING_STATE)

    provider = TracerProvider(resource=Resource.create({
        "service.name": "civitas",
        "service.version": settings.app_version,
        "deployment.environment": settings.environment_version,
    }))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint))
    )
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("civitas")
    _TRACING_STATE.update(enabled=True, reason="", endpoint=settings.otel_endpoint)
    return dict(_TRACING_STATE)


def tracing_state() -> dict[str, Any]:
    return dict(_TRACING_STATE)


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """A span when tracing is on, and a no-op that costs a function call when it is not."""
    if _TRACER is None:
        yield _NoopSpan()
        return
    with _TRACER.start_as_current_span(name) as active:  # pragma: no cover - needs the package
        for key, value in attributes.items():
            active.set_attribute(key, value)
        yield active


def process_info() -> dict[str, Any]:
    return {"pid": os.getpid(), "thread": threading.current_thread().name}
