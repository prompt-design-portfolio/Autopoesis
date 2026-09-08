"""Provider construction and lookup (Part B §19).

The registry is the single place that knows which provider classes exist. Everything else asks for
one by name, so adding a vendor is one entry here and no change anywhere else — which is what
"do not tightly couple collective architecture to one model vendor" means in practice.
"""

from __future__ import annotations

from collections.abc import Callable

from civitas.config import Settings, get_settings
from civitas.runtime.providers.base import Provider

_FACTORIES: dict[str, Callable[[Settings], Provider]] = {}
_CACHE: dict[tuple[str, int], Provider] = {}


def register(name: str, factory: Callable[[Settings], Provider]) -> None:
    _FACTORIES[name] = factory


def available() -> list[str]:
    return sorted(_FACTORIES)


def get_provider(name: str, settings: Settings | None = None) -> Provider:
    """Build (and cache) a provider.

    Cached per settings object because providers hold HTTP clients and, for local inference, model
    weights — rebuilding one per episode would re-download a checkpoint.
    """
    settings = settings or get_settings()
    if name not in _FACTORIES:
        raise KeyError(f"unknown provider {name!r}; registered: {available()}")
    key = (name, id(settings))
    if key not in _CACHE:
        _CACHE[key] = _FACTORIES[name](settings)
    return _CACHE[key]


def set_provider(name: str, provider: Provider) -> None:
    """Install a provider instance directly. Tests use this to inject a scripted provider."""
    _FACTORIES[name] = lambda _settings, _p=provider: _p
    for key in [k for k in _CACHE if k[0] == name]:
        del _CACHE[key]


def clear_cache() -> None:
    for provider in _CACHE.values():
        provider.close()
    _CACHE.clear()


def _bootstrap() -> None:
    from civitas.runtime.providers.anthropic import AnthropicProvider
    from civitas.runtime.providers.google import GoogleProvider
    from civitas.runtime.providers.offline import DeterministicProvider, ScriptedProvider
    from civitas.runtime.providers.openai_compat import (
        LlamaCppProvider,
        OpenAICompatibleProvider,
        VLLMProvider,
    )

    register("deterministic", lambda _s: DeterministicProvider())
    register("scripted", lambda _s: ScriptedProvider([]))
    register("anthropic", lambda s: AnthropicProvider(s))
    register("openai", lambda s: OpenAICompatibleProvider(s))
    register("openai_compatible", lambda s: OpenAICompatibleProvider(s, name="openai_compatible"))
    register("google", lambda s: GoogleProvider(s))
    register("vllm", lambda s: VLLMProvider(s))
    register("llamacpp", lambda s: LlamaCppProvider(s))
    # Deferred: importing `local_hf` is cheap, but constructing one loads weights, so the factory
    # requires an explicit model id and is registered by the caller that knows it.
    register(
        "huggingface",
        lambda s: _huggingface_from_settings(s),
    )


def _huggingface_from_settings(settings: Settings) -> Provider:
    from civitas.runtime.providers.local_hf import HuggingFaceProvider

    return HuggingFaceProvider(settings.default_model, settings)


_bootstrap()
