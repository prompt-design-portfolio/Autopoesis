"""Google Gemini adapter (Part B §19)."""

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

MODELS: dict[str, dict[str, Any]] = {
    "gemini-2.0-flash": {"context": 1_048_576, "out": 8192, "in$": 0.1, "out$": 0.4},
    "gemini-1.5-pro": {"context": 2_097_152, "out": 8192, "in$": 1.25, "out$": 5.0},
}
_DEFAULT = {"context": 32_768, "out": 8192, "in$": 0.0, "out$": 0.0}


class GoogleProvider(HttpProviderMixin, Provider):
    name = "google"

    def __init__(self, settings: Settings | None = None,
                 base_url: str = "https://generativelanguage.googleapis.com"):
        self._settings = settings or get_settings()
        self.base_url = base_url

    def _headers(self) -> dict[str, str]:
        key = self._settings.google_api_key
        if key is None:
            raise ProviderError("google: no API key configured", retryable=False)
        # The header form, not the `?key=` query parameter: a URL ends up in access logs and
        # exception messages, and a credential must not (Part B §40).
        return {"x-goog-api-key": key.get_secret_value(), "content-type": "application/json"}

    def model_info(self, model: str) -> ModelInfo:
        spec = MODELS.get(model, _DEFAULT)
        return ModelInfo(
            name=model, provider=self.name,
            context_window=spec["context"], max_output_tokens=spec["out"],
            supports_tools=True, supports_streaming=True, supports_structured_output=True,
            cost_per_1k_prompt_usd=spec["in$"] / 1000,
            cost_per_1k_completion_usd=spec["out$"] / 1000,
        )

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        system_parts = [m.content for m in request.messages if m.role is Role.SYSTEM]
        for m in request.messages:
            if m.role is Role.SYSTEM:
                continue
            if m.role is Role.TOOL:
                contents.append({"role": "user", "parts": [{"functionResponse": {
                    "name": m.name or "", "response": {"result": m.content}}}]})
                continue
            parts: list[dict[str, Any]] = []
            if m.content:
                parts.append({"text": m.content})
            for call in m.tool_calls:
                parts.append({"functionCall": {"name": call.name, "args": call.arguments}})
            contents.append({
                "role": "model" if m.role is Role.ASSISTANT else "user",
                "parts": parts or [{"text": ""}],
            })

        gen: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_tokens,
        }
        if request.top_p != 1.0:
            gen["topP"] = request.top_p
        if request.stop:
            gen["stopSequences"] = list(request.stop)
        if request.response_schema is not None:
            gen["responseMimeType"] = "application/json"
            gen["responseSchema"] = request.response_schema

        payload: dict[str, Any] = {"contents": contents, "generationConfig": gen}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if request.tools:
            payload["tools"] = [{"functionDeclarations": [
                {"name": t.name, "description": t.description,
                 "parameters": t.parameters or {"type": "object", "properties": {}}}
                for t in request.tools
            ]}]
        return payload

    def complete(self, request: CompletionRequest) -> Completion:
        started = time.perf_counter()
        body = self._post(f"/v1beta/models/{request.model}:generateContent", self._payload(request))
        latency_ms = (time.perf_counter() - started) * 1000

        candidates = body.get("candidates") or []
        if not candidates:
            raise ProviderError("google: response contained no candidates", retryable=True)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for i, part in enumerate(candidates[0].get("content", {}).get("parts", [])):
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(ToolCall(id=f"call_{i}", name=fc.get("name", ""),
                                           arguments=fc.get("args") or {}))

        meta = body.get("usageMetadata", {})
        usage = Usage(
            prompt_tokens=int(meta.get("promptTokenCount", 0)),
            completion_tokens=int(meta.get("candidatesTokenCount", 0)),
        )
        text = "".join(text_parts)
        structured = None
        if request.response_schema is not None and text:
            import json
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None

        return Completion(
            text=text, tool_calls=tuple(tool_calls), usage=usage,
            stop_reason=candidates[0].get("finishReason", "STOP"),
            model=request.model, latency_ms=latency_ms,
            cost_usd=self.estimate_cost(request.model, usage),
            structured=structured, structured_enforced=structured is not None,
        )
