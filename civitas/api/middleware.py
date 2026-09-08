"""HTTP hardening and instrumentation (Part B §52, §53, §54).

Everything here is applied to every request, which is the point: a control that each endpoint has
to remember to call is a control that a new endpoint will not have.

`rate_limit_per_minute` has been a setting since M1 and nothing read it. So had `log_json`,
`metrics_enabled` and `otel_endpoint`. This is where they start meaning something.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from civitas.config import Settings
from civitas.observability import REGISTRY, correlated

#: Requests larger than this are refused before the body is read. §52: an unbounded body is a
#: memory-exhaustion vector, and a tool's source or an artifact body is measured in kilobytes.
MAX_BODY_BYTES = 4 * 1024 * 1024

#: The header a client may set to carry its own correlation id. Echoed back on every response so a
#: caller can find its request in the logs without guessing.
REQUEST_ID_HEADER = "x-request-id"


def content_security_policy(*, allow_docs: bool = True) -> str:
    """The CSP the UI actually runs under.

    Written against `civitas/web/` rather than copied from a hardening guide: the page loads one
    script and one stylesheet from its own origin, opens an `EventSource` to its own origin, and
    fetches nothing else. So `default-src 'self'` with no `unsafe-inline` and no external origin
    is not a restriction the UI has to work around — it is a description of what the UI does, and
    a browser test asserts the page renders under it without a console error.

    `/api/v1/docs` is the one exception: Swagger UI is loaded from a CDN by FastAPI itself. It is
    allowed explicitly and only for scripts and styles, so the exception is visible here rather
    than hidden in a blanket `unsafe-inline`.
    """
    script = "'self'"
    style = "'self'"
    if allow_docs:
        script += " https://cdn.jsdelivr.net"
        style += " https://cdn.jsdelivr.net"
    return "; ".join([
        "default-src 'self'",
        f"script-src {script}",
        f"style-src {style}",
        "img-src 'self' data: https://fastapi.tiangolo.com",
        "connect-src 'self'",
        "font-src 'self'",
        "object-src 'none'",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ])


def security_headers(settings: Settings) -> dict[str, str]:
    headers = {
        "content-security-policy": content_security_policy(),
        # The browser must not sniff a JSON response into HTML — that is how a stored artifact
        # body becomes a script.
        "x-content-type-options": "nosniff",
        "referrer-policy": "no-referrer",
        "x-frame-options": "DENY",
        "permissions-policy": "geolocation=(), microphone=(), camera=()",
        "cross-origin-opener-policy": "same-origin",
    }
    if settings.api_host not in ("127.0.0.1", "localhost", "0.0.0.0"):
        # Only where TLS is plausible. Sending HSTS from a local HTTP deployment pins a browser to
        # https for a host that does not serve it, and the operator cannot undo it.
        headers["strict-transport-security"] = "max-age=31536000; includeSubDomains"
    return headers


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, settings: Settings):
        super().__init__(app)
        self._headers = security_headers(settings)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        for name, value in self._headers.items():
            response.headers.setdefault(name, value)
        return response


class BodyLimitMiddleware(BaseHTTPMiddleware):
    """Refuse an oversized body on the declared length, before it is read."""

    def __init__(self, app: Any, max_bytes: int = MAX_BODY_BYTES):
        super().__init__(app)
        self._max = max_bytes

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self._max:
            return JSONResponse(
                status_code=413,
                content={"detail": f"body exceeds {self._max} bytes"},
            )
        return await call_next(request)


@dataclass
class _Window:
    hits: deque[float]


class RateLimiter:
    """A sliding-window limiter keyed on the principal, falling back to the client address.

    Sliding window rather than a fixed one: a fixed window lets a caller send the whole budget in
    the last second of one window and again in the first second of the next, which is twice the
    limit at the moment it matters.

    **In-process, and that is a limitation with consequences.** Two API replicas allow twice the
    configured rate. Stated here, reported by `/healthz`, and the reason a real deployment puts a
    shared limiter in front — the fix is a shared store, and inventing one here would be a larger
    claim than this milestone supports.
    """

    def __init__(self, per_minute: int, *, clock: Any = time.monotonic):
        self.per_minute = per_minute
        self._clock = clock
        self._windows: dict[str, _Window] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int, float]:
        """Return `(allowed, remaining, retry_after_s)`."""
        if self.per_minute <= 0:
            return True, 0, 0.0
        now = self._clock()
        cutoff = now - 60.0
        with self._lock:
            window = self._windows.setdefault(key, _Window(hits=deque()))
            while window.hits and window.hits[0] <= cutoff:
                window.hits.popleft()
            if len(window.hits) >= self.per_minute:
                retry_after = max(0.0, 60.0 - (now - window.hits[0]))
                return False, 0, retry_after
            window.hits.append(now)
            return True, self.per_minute - len(window.hits), 0.0

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Applies the limiter, and exempts the two routes that must answer under load.

    `/healthz` and `/readyz` are exempt on purpose: a limiter that can make a load balancer believe
    the process is unhealthy turns a traffic spike into an outage.
    """

    EXEMPT = frozenset({"/healthz", "/readyz"})

    def __init__(self, app: Any, limiter: RateLimiter):
        super().__init__(app)
        self._limiter = limiter

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if request.url.path in self.EXEMPT:
            return await call_next(request)

        # Keyed on the API key's *prefix*, never the key: the limiter's state is readable in a
        # heap dump and printed in diagnostics, and the prefix identifies a key without being one.
        raw = request.headers.get("x-api-key") or ""
        identity = raw[:12] if raw else f"addr:{request.client.host if request.client else '-'}"

        allowed, remaining, retry_after = self._limiter.check(identity)
        if not allowed:
            REGISTRY.inc("civitas_http_rate_limited_total", route=request.url.path)
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded"},
                headers={
                    "retry-after": str(int(retry_after) + 1),
                    "x-ratelimit-limit": str(self._limiter.per_minute),
                    "x-ratelimit-remaining": "0",
                },
            )
        response = await call_next(request)
        response.headers.setdefault("x-ratelimit-limit", str(self._limiter.per_minute))
        response.headers.setdefault("x-ratelimit-remaining", str(remaining))
        return response


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Correlation id, request timing, and the HTTP counters (§53)."""

    def __init__(self, app: Any, enabled: bool = True):
        super().__init__(app)
        self._enabled = enabled

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        started = time.perf_counter()
        with correlated(incoming or uuid.uuid4().hex) as request_id:
            try:
                response = await call_next(request)
            except Exception:
                if self._enabled:
                    REGISTRY.inc(
                        "civitas_http_requests_total", method=request.method,
                        route=_route_of(request), status="500",
                    )
                raise
            response.headers[REQUEST_ID_HEADER] = request_id
            if self._enabled:
                elapsed = time.perf_counter() - started
                route = _route_of(request)
                REGISTRY.inc(
                    "civitas_http_requests_total", method=request.method, route=route,
                    status=str(response.status_code),
                )
                REGISTRY.observe("civitas_http_request_seconds", elapsed, route=route)
            return response


def _route_of(request: Request) -> str:
    """The route *template*, not the path.

    `/workspaces/{workspace_id}/tasks`, never `/workspaces/2f1c…/tasks`: a label whose value is an
    id gives the metric unbounded cardinality, which is how a monitoring system is taken down by
    the thing monitoring it.
    """
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"
