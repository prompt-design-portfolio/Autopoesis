"""Circuit breakers for model providers (Part B §54).

Retry with backoff already exists in `http.py`, and on its own it makes a provider outage *more*
expensive rather than less: every episode pays `MAX_RETRIES` attempts and up to thirty seconds of
sleep to learn what the previous episode already established. With a bounded token budget, a
benchmark run against a dead provider spends its whole budget on timeouts and records the result
as agent failure.

A breaker turns that into one measured fact. After `failure_threshold` consecutive failures the
circuit opens and calls fail immediately with `CircuitOpen`; after `recovery_seconds` one call is
let through, and the next success closes it.

Two decisions matter for the science:

**A skipped call is not a failure.** `CircuitOpen` is raised as a `ProviderError` with
`retryable=True` so the episode terminates with `provider_failure`, which
`evaluation.is_readable` already excludes from success rates (§47). Counting an outage as an
agent's failure would let infrastructure noise masquerade as a weaker arm.

**One breaker per (provider, model).** A provider whose large model is rate-limited while its
small one answers is the common case, and a breaker keyed on the provider alone would take out
both — turning a partial outage into a total one.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from civitas.observability import REGISTRY
from civitas.runtime.providers.base import ProviderError

#: Consecutive failures before the circuit opens. Three, not one: a single failure is often a
#: transient the existing retry already absorbed, and opening on it would make the breaker itself
#: the outage.
DEFAULT_FAILURE_THRESHOLD = 3

#: How long the circuit stays open before letting one call through.
DEFAULT_RECOVERY_SECONDS = 30.0


class CircuitOpen(ProviderError):
    """The call was not attempted: this provider is failing and the circuit is open."""

    def __init__(self, key: str, opened_for_s: float, failures: int):
        super().__init__(
            f"{key}: circuit open after {failures} consecutive failures "
            f"({opened_for_s:.1f}s ago); the call was not attempted",
            retryable=True,
        )
        self.key = key
        self.failures = failures


@dataclass
class BreakerState:
    key: str
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    recovery_seconds: float = DEFAULT_RECOVERY_SECONDS
    consecutive_failures: int = 0
    opened_at: float | None = None
    #: Counted for the record: how many calls this breaker refused to attempt. That number is the
    #: whole justification for the mechanism, so it is reported rather than inferred.
    skipped: int = 0
    half_open: bool = False
    #: The breaker's clock, so a reported age uses the same time source that opened the circuit.
    #: Reading `time.monotonic()` here instead would report nonsense whenever a caller injected a
    #: clock — which is exactly what the tests do, so the field would only ever be wrong where it
    #: was being checked.
    clock: Any = field(default=time.monotonic, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def is_open(self) -> bool:
        return self.opened_at is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "open": self.is_open,
            "half_open": self.half_open,
            "consecutive_failures": self.consecutive_failures,
            "skipped": self.skipped,
            # `is not None`, not truthiness: an injected clock starting at zero makes
            # `opened_at == 0.0`, and `if self.opened_at` reports a circuit that is open as
            # though it had never opened.
            "opened_for_s": (
                round(self.clock() - self.opened_at, 3) if self.opened_at is not None else None
            ),
        }


class CircuitBreaker:
    """A registry of breakers, one per (provider, model)."""

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_seconds: float = DEFAULT_RECOVERY_SECONDS,
        clock: Any = time.monotonic,
    ):
        self._states: dict[str, BreakerState] = {}
        self._threshold = failure_threshold
        self._recovery = recovery_seconds
        self._clock = clock
        self._lock = threading.Lock()

    def state(self, key: str) -> BreakerState:
        with self._lock:
            existing = self._states.get(key)
            if existing is None:
                existing = BreakerState(
                    key=key, failure_threshold=self._threshold,
                    recovery_seconds=self._recovery, clock=self._clock,
                )
                self._states[key] = existing
            return existing

    def states(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.as_dict() for s in self._states.values()]

    def before_call(self, key: str) -> None:
        """Raise `CircuitOpen` when the call must not be attempted."""
        state = self.state(key)
        with state.lock:
            if state.opened_at is None:
                return
            elapsed = self._clock() - state.opened_at
            if elapsed < state.recovery_seconds:
                state.skipped += 1
                REGISTRY.inc("civitas_provider_calls_skipped_total", key=key)
                raise CircuitOpen(key, elapsed, state.consecutive_failures)
            # Half-open: exactly one call is allowed through to test the provider. Letting the
            # whole backlog through at once is how a recovering provider is knocked over again.
            state.half_open = True
            state.opened_at = None

    def record_success(self, key: str) -> None:
        state = self.state(key)
        with state.lock:
            state.consecutive_failures = 0
            state.opened_at = None
            state.half_open = False
        REGISTRY.inc("civitas_provider_calls_total", key=key, outcome="success")
        REGISTRY.set("civitas_circuit_breaker_open", 0.0, key=key)

    def record_failure(self, key: str) -> None:
        state = self.state(key)
        with state.lock:
            state.consecutive_failures += 1
            state.half_open = False
            if state.consecutive_failures >= state.failure_threshold:
                state.opened_at = self._clock()
        REGISTRY.inc("civitas_provider_calls_total", key=key, outcome="failure")
        REGISTRY.set(
            "civitas_circuit_breaker_open", 1.0 if state.is_open else 0.0, key=key
        )

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._states.clear()
            else:
                self._states.pop(key, None)


#: The process-wide breaker. One per process, like the metrics registry, and with the same
#: limitation stated: several workers each learn about an outage separately.
BREAKER = CircuitBreaker()


def guarded(provider: Any, model: str, call: Any, *, breaker: CircuitBreaker | None = None) -> Any:
    """Run `call()` under the breaker for `(provider, model)`.

    A thin function rather than a decorator on `Provider.complete`: the offline and policy
    providers do not need it, and wrapping them would put a breaker between the benchmark and a
    deterministic function that cannot fail.
    """
    breaker = breaker or BREAKER
    key = f"{getattr(provider, 'name', provider)}/{model}"
    breaker.before_call(key)
    try:
        result = call()
    except ProviderError:
        breaker.record_failure(key)
        raise
    except Exception:
        # A non-provider exception is a bug in this code, not evidence about the provider. It
        # must not open the circuit, or a local defect would look like an outage.
        raise
    breaker.record_success(key)
    return result
