"""Sandbox selection (Part B §18, §33).

The factory prefers the strongest backend that can actually run here, so the same code gives
kernel isolation on a production host and process isolation in a Colab runtime — without a
separate Colab code path (Part B §6, §33) and without silently pretending the weaker one is the
stronger one.
"""

from __future__ import annotations

import logging

from civitas.config import SandboxBackend, Settings, get_settings
from civitas.runtime.sandbox.base import (
    ENV_ALLOWLIST,
    SECRET_MARKERS,
    IsolationLevel,
    Sandbox,
    SandboxLimits,
    SandboxResult,
    SandboxUnavailable,
    clean_environment,
)
from civitas.runtime.sandbox.container import DockerSandbox, GvisorSandbox
from civitas.runtime.sandbox.subprocess_sandbox import SubprocessSandbox, parse_result

log = logging.getLogger(__name__)

#: Strongest first. `auto` walks this list and takes the first available backend.
PREFERENCE = (GvisorSandbox, DockerSandbox, SubprocessSandbox)


def get_sandbox(
    settings: Settings | None = None, *, backend: SandboxBackend | None = None
) -> Sandbox:
    """Build the sandbox for this environment.

    An explicitly requested backend that is unavailable raises rather than downgrading. A silent
    downgrade would put a result produced under process isolation into a manifest that claims
    container isolation, which is exactly the confusion §46 exists to prevent.
    """
    settings = settings or get_settings()
    requested = backend or settings.sandbox_backend

    if requested == SandboxBackend.DOCKER:
        sandbox: Sandbox = DockerSandbox()
    elif requested == SandboxBackend.GVISOR:
        sandbox = GvisorSandbox()
    elif requested == SandboxBackend.SUBPROCESS:
        sandbox = SubprocessSandbox()
    elif requested == SandboxBackend.NONE:
        raise SandboxUnavailable(
            "sandbox_backend=none: Part B §18 forbids executing agent-generated code on the host"
        )
    else:  # pragma: no cover - the enum is closed
        raise SandboxUnavailable(f"unknown sandbox backend {requested!r}")

    if not sandbox.available():
        raise SandboxUnavailable(
            f"sandbox backend {requested.value!r} is not available here. "
            "Configure an available backend explicitly rather than downgrading silently — "
            "the backend in force is recorded in the experiment manifest (Part B §46)."
        )
    return sandbox


def best_available() -> Sandbox:
    """The strongest backend that works here (Part B §33: Colab gets process isolation)."""
    for cls in PREFERENCE:
        candidate = cls()
        if candidate.available():
            return candidate
    raise SandboxUnavailable("no sandbox backend is available in this environment")


__all__ = [
    "ENV_ALLOWLIST", "SECRET_MARKERS", "DockerSandbox", "GvisorSandbox", "IsolationLevel",
    "Sandbox", "SandboxLimits", "SandboxResult", "SandboxUnavailable", "SubprocessSandbox",
    "best_available", "clean_environment", "get_sandbox", "parse_result",
]
