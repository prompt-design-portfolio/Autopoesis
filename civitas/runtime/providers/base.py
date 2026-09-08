"""The provider-independent model interface (Part B §19).

One interface for every backend, so the collective architecture is not coupled to a vendor. The
deterministic and scripted providers keep the whole system testable offline (Part A §A1.6), and CI
never needs a paid API (§55).

The interface carries everything §19 requires: messages, tool calls, structured output, streaming,
usage, token counts, latency, provider errors, cost and model metadata.
"""

from __future__ import annotations

import abc
import hashlib
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    #: Set on a Role.TOOL message: which call this is the result of.
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionRequest:
    messages: tuple[Message, ...]
    model: str
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    stop: tuple[str, ...] = ()
    tools: tuple[ToolSpec, ...] = ()
    #: JSON Schema for structured output (§19). Providers that cannot enforce it natively fall
    #: back to instruction plus validation, and say so in `Completion.structured_enforced`.
    response_schema: dict[str, Any] | None = None
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        """Stable identity for a request.

        Recorded on `ModelCall` so a call is identifiable and repeatable without its *content*
        being persisted — an episode's reasoning is temporary individual state (Part B §4) and
        storing it would create exactly the hidden channel §4 forbids.
        """
        blob = json.dumps(
            {
                "messages": [m.to_dict() for m in self.messages],
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stop": list(self.stop),
                "tools": [{"name": t.name, "parameters": t.parameters} for t in self.tools],
                "response_schema": self.response_schema,
                "seed": self.seed,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Completion:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"
    model: str = ""
    model_version: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    structured: Any = None
    #: False when the schema was requested but the provider could only ask for it in the prompt.
    #: A downstream parser must not treat unenforced output as guaranteed.
    structured_enforced: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A provider call failed. Carries whether a retry could plausibly succeed."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class RateLimited(ProviderError):
    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message, retryable=True, status=429)
        self.retry_after_s = retry_after_s


@dataclass(frozen=True)
class ModelInfo:
    """Model metadata (§19). `context_window` is what budget enforcement checks against."""

    name: str
    provider: str
    context_window: int = 8192
    max_output_tokens: int = 4096
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_structured_output: bool = False
    is_deterministic: bool = False
    cost_per_1k_prompt_usd: float = 0.0
    cost_per_1k_completion_usd: float = 0.0
    version: str = ""


class Provider(abc.ABC):
    """Every backend implements this and nothing else is provider-aware."""

    name: str = "base"

    @abc.abstractmethod
    def complete(self, request: CompletionRequest) -> Completion:
        """One synchronous completion. Raises `ProviderError` on failure."""

    @abc.abstractmethod
    def model_info(self, model: str) -> ModelInfo:
        ...

    def stream(self, request: CompletionRequest) -> Iterator[str]:
        """Token stream (§19).

        The default yields the whole completion as one chunk. That is a correct implementation of
        the contract for a provider without native streaming — the caller sees the same text —
        and it keeps `stream` usable everywhere instead of only where it is implemented natively.
        """
        yield self.complete(request).text

    async def astream(self, request: CompletionRequest) -> AsyncIterator[str]:  # pragma: no cover
        for chunk in self.stream(request):
            yield chunk

    def count_tokens(self, text: str) -> int:
        """Token count (§19).

        The default is a character-based estimate, deliberately *not* a tokenizer guess: budget
        enforcement must never silently under-count, so providers that can count exactly override
        this and the estimate errs high.
        """
        return max(1, (len(text) + 3) // 4)

    def count_message_tokens(self, messages: tuple[Message, ...] | list[Message]) -> int:
        """Tokens a message list will cost, **including tool-call arguments**.

        Counting only `content` under-counts systematically, and the error is largest exactly
        where it matters: an agent that writes a tool puts the entire source in a tool call's
        arguments, where a content-only count sees nothing. Measured on the tool benchmark, an
        episode that wrote a full source file and test suite was billed fewer prompt tokens than
        one that read a short listing — so a token budget (§8) would not have bounded the
        behaviour it most needs to bound.
        """
        total = 0
        for message in messages:
            total += self.count_tokens(message.content)
            for call in message.tool_calls:
                total += self.count_tokens(call.name)
                total += self.count_tokens(json.dumps(call.arguments, default=str))
        return total

    def count_request_tokens(self, request: CompletionRequest) -> int:
        """Everything the provider will be sent: messages, tool calls, and tool declarations."""
        total = self.count_message_tokens(request.messages)
        for spec in request.tools:
            total += self.count_tokens(spec.name) + self.count_tokens(spec.description)
            total += self.count_tokens(json.dumps(spec.parameters, default=str))
        return total

    def estimate_cost(self, model: str, usage: Usage) -> float:
        info = self.model_info(model)
        return (
            usage.prompt_tokens / 1000 * info.cost_per_1k_prompt_usd
            + usage.completion_tokens / 1000 * info.cost_per_1k_completion_usd
        )

    def close(self) -> None:  # noqa: B027 - intentionally optional
        """Release anything the provider holds.

        Deliberately concrete and empty rather than abstract: most providers hold nothing, and
        forcing every adapter to write an empty override adds noise without adding a guarantee.
        """
        return None
