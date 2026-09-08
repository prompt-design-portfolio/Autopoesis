"""Anthropic Messages API adapter (Part B §19).

A complete adapter, mocked in tests (Part A §A1.8). Credentials resolve from `Settings` at call
time and are never persisted, logged, or exposed to a sandbox (§40).
"""

from __future__ import annotations

import time
from typing import Any

from civitas.config import Settings, get_settings
from civitas.runtime.providers.base import (
    Completion,
    CompletionRequest,
    ModelInfo,
    Provider,
    ProviderError,
    Role,
    ToolCall,
    Usage,
)
from civitas.runtime.providers.http import HttpProviderMixin

#: Published context windows and prices. Wrong numbers here mean wrong budget enforcement and
#: wrong cost dashboards, so unknown models fall back to a conservative default rather than to
#: an optimistic guess.
MODELS: dict[str, dict[str, Any]] = {
    "claude-opus-5": {"context": 200_000, "out": 64_000, "in$": 5.0, "out$": 25.0},
    "claude-sonnet-5": {"context": 200_000, "out": 64_000, "in$": 3.0, "out$": 15.0},
    "claude-haiku-4-5-20251001": {"context": 200_000, "out": 32_000, "in$": 1.0, "out$": 5.0},
}
_DEFAULT = {"context": 200_000, "out": 8192, "in$": 3.0, "out$": 15.0}

API_VERSION = "2023-06-01"


class AnthropicProvider(HttpProviderMixin, Provider):
    name = "anthropic"

    def __init__(self, settings: Settings | None = None, base_url: str = "https://api.anthropic.com"):
        self._settings = settings or get_settings()
        self.base_url = base_url

    def _headers(self) -> dict[str, str]:
        key = self._settings.anthropic_api_key
        if key is None:
            raise ProviderError("anthropic: no API key configured", retryable=False)
        return {
            "x-api-key": key.get_secret_value(),
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def model_info(self, model: str) -> ModelInfo:
        spec = MODELS.get(model, _DEFAULT)
        return ModelInfo(
            name=model,
            provider=self.name,
            context_window=spec["context"],
            max_output_tokens=spec["out"],
            supports_tools=True,
            supports_streaming=True,
            supports_structured_output=True,
            cost_per_1k_prompt_usd=spec["in$"] / 1000,
            cost_per_1k_completion_usd=spec["out$"] / 1000,
        )

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        # Anthropic takes the system prompt as a top-level field, not as a message.
        system_parts = [m.content for m in request.messages if m.role is Role.SYSTEM]
        messages: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role is Role.SYSTEM:
                continue
            if m.role is Role.TOOL:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id or "",
                        "content": m.content,
                    }],
                })
                continue
            content: list[dict[str, Any]] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            for call in m.tool_calls:
                content.append({
                    "type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments,
                })
            messages.append(
                {"role": m.role.value, "content": content or [{"type": "text", "text": ""}]}
            )

        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": min(request.max_tokens, self.model_info(request.model).max_output_tokens),
            "messages": messages,
            "temperature": request.temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.top_p != 1.0:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = list(request.stop)
        if request.tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description,
                 "input_schema": t.parameters or {"type": "object", "properties": {}}}
                for t in request.tools
            ]
        if request.response_schema is not None:
            # Structured output is expressed as a single-tool forced call: the schema is enforced
            # by the API rather than requested in the prompt, so `structured_enforced` is true.
            payload["tools"] = [{
                "name": "emit_structured_result",
                "description": "Return the result in the required structure.",
                "input_schema": request.response_schema,
            }]
            payload["tool_choice"] = {"type": "tool", "name": "emit_structured_result"}
        return payload

    def complete(self, request: CompletionRequest) -> Completion:
        started = time.perf_counter()
        body = self._post("/v1/messages", self._payload(request))
        latency_ms = (time.perf_counter() - started) * 1000

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        structured: Any = None
        for block in body.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                is_structuring_call = block.get("name") == "emit_structured_result"
                if request.response_schema is not None and is_structuring_call:
                    structured = block.get("input")
                else:
                    tool_calls.append(
                        ToolCall(id=block.get("id", ""), name=block.get("name", ""),
                                 arguments=block.get("input") or {})
                    )

        usage_body = body.get("usage", {})
        usage = Usage(
            prompt_tokens=int(usage_body.get("input_tokens", 0)),
            completion_tokens=int(usage_body.get("output_tokens", 0)),
        )
        return Completion(
            text="".join(text_parts),
            tool_calls=tuple(tool_calls),
            usage=usage,
            stop_reason=body.get("stop_reason", "end_turn"),
            model=body.get("model", request.model),
            model_version=body.get("model", ""),
            latency_ms=latency_ms,
            cost_usd=self.estimate_cost(request.model, usage),
            structured=structured,
            structured_enforced=structured is not None,
            raw={"id": body.get("id")},
        )

    def count_tokens(self, text: str) -> int:
        """Estimate, erring high.

        The API has an exact count endpoint, but calling it on every budget check would double the
        request volume. Budget enforcement must never *under*-count, so the estimate is
        deliberately generous and the true count from `usage` replaces it after each call.
        """
        return max(1, int(len(text) / 3.5) + 1)
