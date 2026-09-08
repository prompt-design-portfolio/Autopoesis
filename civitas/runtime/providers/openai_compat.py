"""OpenAI and OpenAI-compatible endpoints (Part B §19).

One adapter covers OpenAI itself, vLLM's OpenAI server, llama.cpp's server, and anything else
speaking `/v1/chat/completions` — which is most local inference stacks. Part B §19 asks for OpenAI,
OpenAI-compatible endpoints and vLLM separately; they differ in base URL and credentials, not in
protocol, so they are configurations of this class rather than three near-identical files.
"""

from __future__ import annotations

import json
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

MODELS: dict[str, dict[str, Any]] = {
    "gpt-4o": {"context": 128_000, "out": 16_384, "in$": 2.5, "out$": 10.0},
    "gpt-4o-mini": {"context": 128_000, "out": 16_384, "in$": 0.15, "out$": 0.6},
}
_DEFAULT = {"context": 32_768, "out": 4096, "in$": 0.0, "out$": 0.0}


class OpenAICompatibleProvider(HttpProviderMixin, Provider):
    """`/v1/chat/completions`, whoever is serving it."""

    name = "openai"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        base_url: str | None = None,
        name: str | None = None,
        api_key: str | None = None,
        requires_key: bool = True,
    ):
        self._settings = settings or get_settings()
        self.base_url = base_url or self._settings.openai_base_url or "https://api.openai.com"
        self.name = name or "openai"
        self._explicit_key = api_key
        #: Local servers (vLLM, llama.cpp) usually accept any key or none. Requiring one would
        #: make the local path need a fake credential, which is how fake credentials end up in
        #: configuration files.
        self._requires_key = requires_key

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        key = self._explicit_key or (
            self._settings.openai_api_key.get_secret_value()
            if self._settings.openai_api_key
            else None
        )
        if key:
            headers["authorization"] = f"Bearer {key}"
        elif self._requires_key:
            raise ProviderError(f"{self.name}: no API key configured", retryable=False)
        return headers

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
        messages: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role is Role.TOOL:
                messages.append({
                    "role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.content,
                })
                continue
            entry: dict[str, Any] = {"role": m.role.value, "content": m.content}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                    for c in m.tool_calls
                ]
            messages.append(entry)

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.top_p != 1.0:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.tools:
            payload["tools"] = [
                {"type": "function", "function": {
                    "name": t.name, "description": t.description,
                    "parameters": t.parameters or {"type": "object", "properties": {}}}}
                for t in request.tools
            ]
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": request.response_schema,
                                "strict": True},
            }
        return payload

    def complete(self, request: CompletionRequest) -> Completion:
        started = time.perf_counter()
        body = self._post("/v1/chat/completions", self._payload(request))
        latency_ms = (time.perf_counter() - started) * 1000

        choices = body.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.name}: response contained no choices", retryable=True)
        message = choices[0].get("message", {})

        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                # A model can emit malformed JSON arguments. Surfacing the raw string is better
                # than dropping the call: the runtime rejects it at schema validation with a
                # message the agent can act on, rather than the call silently vanishing.
                args = {"__raw__": fn.get("arguments", "")}
            tool_calls.append(
                ToolCall(id=call.get("id", ""), name=fn.get("name", ""), arguments=args)
            )

        text = message.get("content") or ""
        structured = None
        if request.response_schema is not None and text:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None

        usage_body = body.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_body.get("prompt_tokens", 0)),
            completion_tokens=int(usage_body.get("completion_tokens", 0)),
        )
        return Completion(
            text=text,
            tool_calls=tuple(tool_calls),
            usage=usage,
            stop_reason=choices[0].get("finish_reason", "stop"),
            model=body.get("model", request.model),
            model_version=body.get("model", ""),
            latency_ms=latency_ms,
            cost_usd=self.estimate_cost(request.model, usage),
            structured=structured,
            structured_enforced=structured is not None,
            raw={"id": body.get("id")},
        )


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM's OpenAI-compatible server (Part B §19, §38)."""

    def __init__(self, settings: Settings | None = None, *, base_url: str = "http://127.0.0.1:8000"):
        super().__init__(settings, base_url=base_url, name="vllm", requires_key=False)


class LlamaCppProvider(OpenAICompatibleProvider):
    """llama.cpp's server (Part B §19, "where reasonable also support")."""

    def __init__(self, settings: Settings | None = None, *, base_url: str = "http://127.0.0.1:8080"):
        super().__init__(settings, base_url=base_url, name="llamacpp", requires_key=False)
