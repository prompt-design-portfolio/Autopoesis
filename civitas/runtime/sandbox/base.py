"""Hardened tool execution (Part B §18, §52).

Model-generated code and shell commands are untrusted input. Nothing here executes them on the
host directly; every backend enforces the limits §18 names — CPU, memory, disk, processes,
timeouts, filesystem isolation, outbound network policy and secret isolation.

The backends differ in *strength*, and that difference is recorded rather than smoothed over:
`SandboxResult.backend` and `isolation_level` reach the experiment manifest, so a result produced
under the subprocess fallback can never be mistaken for one produced under container isolation
(Part B §46, ARCHITECTURE §6.3).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IsolationLevel(str, Enum):
    """How much the backend actually guarantees.

    `PROCESS` is honest about being weaker: it constrains resources through OS rlimits and a
    scratch directory, but it shares a kernel and a network namespace with the host. A research
    result run under it is still a result; it is just not a result about untrusted code.
    """

    NONE = "none"
    PROCESS = "process"
    CONTAINER = "container"
    KERNEL = "kernel"


@dataclass(frozen=True)
class SandboxLimits:
    """The bounds of §18. Defaults are deliberately small: a tool that needs more says so, and
    the request is checked against policy rather than granted silently."""

    cpu_seconds: float = 10.0
    wall_seconds: float = 30.0
    memory_mb: int = 512
    disk_mb: int = 64
    max_processes: int = 32
    max_output_bytes: int = 1_000_000
    #: Default deny. §18 requires an outbound network policy, and the only policy that is safe by
    #: default for model-generated code is "none".
    network: bool = False
    allowed_hosts: tuple[str, ...] = ()
    read_only_paths: tuple[str, ...] = ()
    gpu: bool = False


@dataclass
class SandboxResult:
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False
    #: Which limit stopped it: cpu, memory, wall, disk, processes, output, network.
    limit_hit: str | None = None
    backend: str = ""
    isolation_level: IsolationLevel = IsolationLevel.NONE
    error: str = ""
    artifacts: dict[str, bytes] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.error


class SandboxUnavailable(RuntimeError):
    """The requested backend cannot run here."""


class Sandbox(abc.ABC):
    """Runs untrusted code under limits."""

    name: str = "base"
    isolation_level: IsolationLevel = IsolationLevel.NONE

    @abc.abstractmethod
    def run_python(
        self,
        source: str,
        *,
        args: dict[str, Any] | None = None,
        limits: SandboxLimits | None = None,
        files: dict[str, str] | None = None,
        entrypoint: str = "main",
    ) -> SandboxResult:
        """Execute `source` and call `entrypoint(**args)` inside the sandbox."""

    @abc.abstractmethod
    def available(self) -> bool:
        """Whether this backend can run in the current environment."""

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "isolation_level": self.isolation_level.value,
            "available": self.available(),
        }


#: Environment variables that must never reach a sandbox (Part B §18, §40).
#: Matched by *prefix and substring*, not by exact name: a credential added later under a new name
#: would otherwise leak, and the failure mode of over-filtering is a tool that has to be told a
#: value explicitly, which is the correct outcome anyway.
SECRET_MARKERS = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH",
    "SESSION", "COOKIE", "PRIVATE", "SIGNATURE", "CIVITAS_",
)

#: The only variables a sandboxed process inherits. An allowlist, because a denylist over a
#: process environment is a guarantee that the next unfamiliar variable gets through.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TZ", "HOME", "TMPDIR", "PYTHONHASHSEED")


def clean_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Build the environment a sandboxed process gets (Part B §40).

    Allowlist first, then a defensive sweep of the allowed values for secret markers — so a
    `PATH` that somehow carried a token is still dropped.
    """
    import os

    source = base if base is not None else dict(os.environ)
    env = {k: v for k, v in source.items() if k in ENV_ALLOWLIST}
    for key in list(env):
        upper = key.upper()
        if any(marker in upper for marker in SECRET_MARKERS):
            del env[key]
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Deterministic hashing: a tool whose output depends on dict iteration order is not
    # reproducible, and reproducibility is the point (Part B §46).
    env["PYTHONHASHSEED"] = "0"
    return env
