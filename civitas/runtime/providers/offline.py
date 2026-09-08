"""Providers that need no network (Part A §A1.6, Part B §55).

Two of them, for two different jobs:

- **`DeterministicProvider`** — output is a pure function of the request hash. It is the default
  provider, and it is what makes frozen-model mode (Part B §20) checkable: the *individual* agent
  is exactly constant, so any change in outcome across a campaign is attributable to the
  collective environment and nothing else. A stochastic model can only make that claim
  statistically; this one makes it exactly.
- **`ScriptedProvider`** — replays a fixed sequence. Tests that need a specific agent behaviour
  (create this artifact, call that tool, emit this canary) use it, so a mechanism test drives the
  real runtime rather than a re-implementation of it.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Sequence
from typing import Any

from civitas.runtime.providers.base import (
    Completion,
    CompletionRequest,
    ModelInfo,
    Provider,
    ProviderError,
    ToolCall,
    Usage,
)


class DeterministicProvider(Provider):
    """Output is a pure function of the request. Same request, same bytes, forever.

    Useful beyond testing: it is the *scientific control* for §20. With the individual agent held
    exactly fixed, a difference between arms cannot be sampling noise.
    """

    name = "deterministic"

    def __init__(
        self,
        *,
        behaviours: dict[str, Callable[[CompletionRequest], Completion]] | None = None,
    ):
        #: Optional per-model behaviours, so a benchmark can define a task-solving policy that is
        #: still deterministic. Without one, the provider emits a stable pseudo-text.
        self._behaviours = behaviours or {}

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(
            name=model,
            provider=self.name,
            context_window=32_000,
            max_output_tokens=4096,
            supports_tools=True,
            supports_structured_output=True,
            is_deterministic=True,
            version="v1",
        )

    def complete(self, request: CompletionRequest) -> Completion:
        started = time.perf_counter()
        behaviour = self._behaviours.get(request.model)
        if behaviour is not None:
            out = behaviour(request)
            out.latency_ms = (time.perf_counter() - started) * 1000
            return out

        digest = request.hash()
        rng = random.Random(int(digest[:16], 16))

        tool_calls: tuple[ToolCall, ...] = ()
        if request.tools and rng.random() < 0.5:
            spec = request.tools[rng.randrange(len(request.tools))]
            tool_calls = (
                ToolCall(id=f"call_{digest[:8]}", name=spec.name, arguments={}),
            )

        text = f"deterministic:{digest[:32]}"
        structured = None
        if request.response_schema is not None:
            structured = _schema_stub(request.response_schema, rng)
            text = json.dumps(structured, sort_keys=True)

        prompt_tokens = sum(self.count_tokens(m.content) for m in request.messages)
        return Completion(
            text=text,
            tool_calls=tool_calls,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=self.count_tokens(text)),
            stop_reason="tool_use" if tool_calls else "end_turn",
            model=request.model,
            model_version="v1",
            latency_ms=(time.perf_counter() - started) * 1000,
            structured=structured,
            structured_enforced=structured is not None,
        )


def _schema_stub(schema: dict[str, Any], rng: random.Random) -> Any:
    """A minimal value satisfying a JSON Schema, chosen deterministically."""
    kind = schema.get("type", "object")
    if kind == "object":
        props = schema.get("properties", {})
        return {k: _schema_stub(v, rng) for k, v in props.items()}
    if kind == "array":
        return [_schema_stub(schema.get("items", {"type": "string"}), rng)]
    if kind == "integer":
        return rng.randrange(0, 100)
    if kind == "number":
        return round(rng.random(), 4)
    if kind == "boolean":
        return rng.random() < 0.5
    if "enum" in schema:
        return schema["enum"][0]
    return f"s{rng.randrange(0, 1000)}"


class ScriptedProvider(Provider):
    """Replays a fixed sequence of completions, in order.

    Running past the end is an error, not a silent repeat: a test whose agent took more turns than
    the script anticipated has diverged from what it meant to check, and quietly repeating the
    last completion would hide that.
    """

    name = "scripted"

    def __init__(self, script: Sequence[Completion | str], *, model: str = "scripted-v1",
                 loop: bool = False):
        self._script = [
            c if isinstance(c, Completion) else Completion(text=c) for c in script
        ]
        self._model = model
        self._loop = loop
        self.calls: list[CompletionRequest] = []

    @property
    def index(self) -> int:
        return len(self.calls)

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(
            name=model, provider=self.name, context_window=32_000,
            supports_tools=True, supports_structured_output=True, is_deterministic=True,
        )

    def complete(self, request: CompletionRequest) -> Completion:
        if self._loop and self._script:
            i = len(self.calls) % len(self._script)
        else:
            i = len(self.calls)
        self.calls.append(request)
        if i >= len(self._script):
            raise ProviderError(
                f"scripted provider exhausted after {len(self._script)} completions; "
                "the agent took more turns than the script covers"
            )
        out = self._script[i]
        prompt_tokens = sum(self.count_tokens(m.content) for m in request.messages)
        return Completion(
            text=out.text,
            tool_calls=out.tool_calls,
            usage=out.usage if out.usage.total_tokens else Usage(
                prompt_tokens=prompt_tokens, completion_tokens=self.count_tokens(out.text)
            ),
            stop_reason=out.stop_reason,
            model=request.model or self._model,
            model_version="scripted",
            structured=out.structured,
            structured_enforced=out.structured is not None,
        )


class FailingProvider(Provider):
    """Always raises. Drives the `provider_failure` termination path (Part B §8)."""

    name = "failing"

    def __init__(self, *, retryable: bool = False, message: str = "synthetic provider failure"):
        self._retryable = retryable
        self._message = message
        self.calls = 0

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(name=model, provider=self.name)

    def complete(self, request: CompletionRequest) -> Completion:
        self.calls += 1
        raise ProviderError(self._message, retryable=self._retryable)


def digest_of(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()
