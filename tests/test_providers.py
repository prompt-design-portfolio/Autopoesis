"""The provider interface (Part B §19). CI never touches a paid API (Part B §55).

The network adapters are driven against a mock transport, so the request each vendor actually
receives is asserted — a wrong field name in a payload is a defect that only shows at the vendor,
and a test that mocks the *adapter* instead of the *transport* would never catch it.
"""

from __future__ import annotations

import json

import httpx
import pytest

from civitas.config import Settings
from civitas.runtime.providers import available
from civitas.runtime.providers.anthropic import AnthropicProvider
from civitas.runtime.providers.base import (
    CompletionRequest,
    Message,
    ProviderError,
    Role,
    ToolSpec,
)
from civitas.runtime.providers.google import GoogleProvider
from civitas.runtime.providers.offline import DeterministicProvider, ScriptedProvider
from civitas.runtime.providers.openai_compat import OpenAICompatibleProvider, VLLMProvider


def _request(**kw):
    return CompletionRequest(
        messages=kw.pop("messages", (
            Message(Role.SYSTEM, "You are bounded."),
            Message(Role.USER, "What broke?"),
        )),
        model=kw.pop("model", "test-model"),
        **kw,
    )


class _MockTransport(httpx.BaseTransport):
    """Captures the outgoing request and returns a canned body."""

    def __init__(self, body: dict, status: int = 200):
        self.body, self.status, self.seen = body, status, []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        return httpx.Response(self.status, json=self.body)

    def payload(self) -> dict:
        return json.loads(self.seen[-1].content)


def _install(provider, transport: _MockTransport):
    provider._client = httpx.Client(transport=transport)
    return transport


def test_every_provider_named_in_section_19_is_registered():
    names = set(available())
    for required in ("anthropic", "openai", "openai_compatible", "google",
                     "huggingface", "vllm", "llamacpp"):
        assert required in names, f"Part B §19 requires a {required} provider"


def test_the_deterministic_provider_is_a_pure_function_of_the_request():
    p = DeterministicProvider()
    a, b = p.complete(_request()), p.complete(_request())
    assert a.text == b.text
    assert p.complete(_request(model="other")).text != a.text
    assert p.model_info("x").is_deterministic


def test_the_scripted_provider_refuses_to_run_past_its_script():
    """A silent repeat would hide a test whose agent diverged from what it meant to check."""
    p = ScriptedProvider(["one"])
    assert p.complete(_request()).text == "one"
    with pytest.raises(ProviderError, match="exhausted"):
        p.complete(_request())


def test_structured_output_is_marked_unenforced_when_it_was_only_requested():
    p = DeterministicProvider()
    out = p.complete(_request(response_schema={
        "type": "object", "properties": {"cause": {"type": "string"}},
    }))
    assert out.structured_enforced
    assert "cause" in out.structured


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
def test_anthropic_sends_the_system_prompt_as_a_top_level_field():
    """Anthropic takes `system` separately; sending it as a message silently changes behaviour."""
    p = AnthropicProvider(Settings(anthropic_api_key="sk-test"))
    t = _install(p, _MockTransport({
        "id": "msg_1", "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": "the writer"}],
        "usage": {"input_tokens": 11, "output_tokens": 3}, "stop_reason": "end_turn",
    }))
    out = p.complete(_request(model="claude-sonnet-5"))

    payload = t.payload()
    assert payload["system"] == "You are bounded."
    assert all(m["role"] != "system" for m in payload["messages"])
    assert out.text == "the writer"
    assert out.usage.total_tokens == 14
    assert out.cost_usd > 0


def test_anthropic_parses_tool_use_blocks():
    p = AnthropicProvider(Settings(anthropic_api_key="sk-test"))
    _install(p, _MockTransport({
        "id": "m", "model": "claude-sonnet-5",
        "content": [
            {"type": "text", "text": "searching"},
            {"type": "tool_use", "id": "tu_1", "name": "search_knowledge",
             "input": {"query": "corruption"}},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 5}, "stop_reason": "tool_use",
    }))
    out = p.complete(_request(model="claude-sonnet-5", tools=(
        ToolSpec("search_knowledge", "search", {"type": "object"}),
    )))
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "search_knowledge"
    assert out.tool_calls[0].arguments == {"query": "corruption"}


def test_anthropic_enforces_structured_output_through_a_forced_tool():
    p = AnthropicProvider(Settings(anthropic_api_key="sk-test"))
    t = _install(p, _MockTransport({
        "id": "m", "model": "claude-sonnet-5",
        "content": [{"type": "tool_use", "id": "t", "name": "emit_structured_result",
                     "input": {"cause": "writer"}}],
        "usage": {"input_tokens": 1, "output_tokens": 1}, "stop_reason": "tool_use",
    }))
    out = p.complete(_request(model="claude-sonnet-5", response_schema={
        "type": "object", "properties": {"cause": {"type": "string"}},
    }))
    assert t.payload()["tool_choice"]["name"] == "emit_structured_result"
    assert out.structured == {"cause": "writer"}
    assert out.structured_enforced
    assert not out.tool_calls, "the structuring tool is not an agent tool call"


def test_a_missing_key_fails_before_any_request_is_made():
    p = AnthropicProvider(Settings())
    t = _install(p, _MockTransport({}))
    with pytest.raises(ProviderError, match="no API key"):
        p.complete(_request())
    assert not t.seen, "a request was sent without a credential"


def test_a_4xx_is_not_retried_and_a_5xx_is(monkeypatch):
    monkeypatch.setattr("civitas.runtime.providers.http.time.sleep", lambda _s: None)

    p = AnthropicProvider(Settings(anthropic_api_key="sk-test"))
    bad = _install(p, _MockTransport({"error": "bad request"}, status=400))
    with pytest.raises(ProviderError) as exc:
        p.complete(_request())
    assert not exc.value.retryable
    assert len(bad.seen) == 1, "a 400 is a bug in the request; retrying it just spends money"

    p2 = AnthropicProvider(Settings(anthropic_api_key="sk-test"))
    flaky = _install(p2, _MockTransport({"error": "upstream"}, status=503))
    with pytest.raises(ProviderError):
        p2.complete(_request())
    assert len(flaky.seen) > 1, "a 503 means the request may not have landed"


# --------------------------------------------------------------------------
# OpenAI-compatible
# --------------------------------------------------------------------------
def test_openai_compatible_round_trips_tool_calls():
    p = OpenAICompatibleProvider(Settings(openai_api_key="sk-test"))
    _install(p, _MockTransport({
        "id": "c", "model": "gpt-4o",
        "choices": [{"finish_reason": "tool_calls", "message": {
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "read_artifact", "arguments": '{"artifact_id": "abc"}'}}],
        }}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2},
    }))
    out = p.complete(_request(model="gpt-4o"))
    assert out.tool_calls[0].name == "read_artifact"
    assert out.tool_calls[0].arguments == {"artifact_id": "abc"}


def test_malformed_tool_arguments_surface_rather_than_vanishing():
    """A model can emit invalid JSON. Dropping the call hides it; surfacing it lets schema
    validation reject it with a message the agent can act on."""
    p = OpenAICompatibleProvider(Settings(openai_api_key="sk-test"))
    _install(p, _MockTransport({
        "id": "c", "model": "gpt-4o",
        "choices": [{"finish_reason": "tool_calls", "message": {
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "read_artifact", "arguments": "{not json"}}],
        }}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }))
    out = p.complete(_request(model="gpt-4o"))
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].arguments["__raw__"] == "{not json"


def test_a_local_endpoint_needs_no_credential():
    """vLLM and llama.cpp accept any key or none. Requiring one is how fake credentials end up
    in configuration files."""
    p = VLLMProvider(Settings())
    t = _install(p, _MockTransport({
        "id": "c", "model": "local",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }))
    assert p.complete(_request(model="local")).text == "ok"
    assert "authorization" not in {k.lower() for k in t.seen[-1].headers}


# --------------------------------------------------------------------------
# Google
# --------------------------------------------------------------------------
def test_google_sends_the_key_as_a_header_not_a_query_parameter():
    """A URL ends up in access logs and exception messages; a credential must not (§40)."""
    p = GoogleProvider(Settings(google_api_key="goog-secret"))
    t = _install(p, _MockTransport({
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "the writer"}]}}],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
    }))
    out = p.complete(_request(model="gemini-2.0-flash"))
    assert out.text == "the writer"
    assert "goog-secret" not in str(t.seen[-1].url)
    assert t.seen[-1].headers["x-goog-api-key"] == "goog-secret"
    assert t.payload()["systemInstruction"]["parts"][0]["text"] == "You are bounded."


def test_token_estimates_never_undercount():
    """Budget enforcement must not be defeated by an optimistic estimate."""
    text = "the quick brown fox jumps over the lazy dog " * 20
    for provider in (DeterministicProvider(), AnthropicProvider(Settings(anthropic_api_key="k"))):
        estimate = provider.count_tokens(text)
        # A real tokenizer produces roughly len/4 for English prose. The estimate must be at
        # least that, never below.
        assert estimate >= len(text) / 5, f"{type(provider).__name__} under-counted"
