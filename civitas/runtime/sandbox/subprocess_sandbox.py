"""The process-isolation backend (Part B §18, §37).

This is the Colab fallback: containers are unavailable in a Colab runtime, so there has to be a
backend that works with nothing but the Python standard library. It is materially weaker than
container isolation and says so — `isolation_level` is `PROCESS`, and that value reaches the
manifest.

What it does enforce, and how:

* **CPU, memory, processes, file size, core dumps** — POSIX rlimits applied in the child via
  `preexec_fn`, so they are in force before the untrusted code's first bytecode runs.
* **Wall clock** — a timeout on `communicate`, then the whole *process group* is killed. Killing
  only the direct child leaves a fork bomb's children running.
* **Filesystem** — a fresh temporary directory as cwd, removed afterwards.
* **Network** — off by default, enforced inside the child by a socket guard installed before the
  untrusted source is executed.
* **Secrets** — the environment is rebuilt from an allowlist (`clean_environment`).

What it does **not** provide is kernel isolation: the code shares a kernel with the host and can
read world-readable paths outside its working directory. For untrusted code from an unknown source
the container or gVisor backend is required, and `SandboxFactory` prefers them when available.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from civitas.runtime.sandbox.base import (
    IsolationLevel,
    Sandbox,
    SandboxLimits,
    SandboxResult,
    clean_environment,
)

#: Installed inside the child before the tool source runs. It is source rather than a module
#: import so it works with no Civitas package on the sandbox's path.
_HARNESS = '''
import json, sys, os, builtins

_ARGS = json.loads(os.environ.get("CIVITAS_TOOL_ARGS", "{}"))
_ENTRY = os.environ.get("CIVITAS_TOOL_ENTRYPOINT", "main")
_NET = os.environ.get("CIVITAS_TOOL_NETWORK") == "1"

if not _NET:
    # Deny outbound network from inside the interpreter (Part B §18 egress policy). This is a
    # second line, not the only one: the container backend blocks at the network namespace. Here
    # it is the strongest available control, and it is installed before any tool code runs.
    import socket

    class _Denied(OSError):
        pass

    def _deny(*a, **k):
        raise _Denied("network access is denied in this sandbox")

    socket.socket = _deny
    socket.create_connection = _deny
    socket.socketpair = _deny
    if hasattr(socket, "create_server"):
        socket.create_server = _deny

_g = {"__name__": "__civitas_tool__", "__builtins__": builtins}
try:
    exec(compile(_SOURCE, "<tool>", "exec"), _g)
    fn = _g.get(_ENTRY)
    if fn is None:
        print(json.dumps({"__civitas_error__": f"entrypoint {_ENTRY!r} not defined"}))
        sys.exit(3)
    result = fn(**_ARGS) if _ARGS else fn()
    sys.stdout.write("\\n__CIVITAS_RESULT__" + json.dumps(result, default=str))
except SystemExit:
    raise
except BaseException as exc:
    import traceback
    sys.stderr.write(traceback.format_exc())
    sys.exit(1)
'''


def _apply_limits(limits: SandboxLimits, workdir: str):
    """Return a `preexec_fn` that sets rlimits and a new process group in the child."""

    def _set(which: int, soft: int, hard: int | None = None) -> None:
        """Apply one rlimit, tolerating a platform that refuses it.

        A refused limit must not abort the whole `preexec_fn`: the child would fail to start and
        the tool would report an infrastructure error rather than running under the limits that
        *can* be applied. Which limits actually took effect is not guessed at — the caller reads
        `limit_hit` from the signal the child died on.
        """
        try:
            resource.setrlimit(which, (soft, hard if hard is not None else soft))
        except (ValueError, OSError):
            pass

    def preexec() -> None:  # pragma: no cover - runs in the forked child
        # The process group comes from `start_new_session=True` on Popen, which calls setsid
        # before this runs; calling it again here raises and kills the spawn.
        cpu = int(limits.cpu_seconds) + 1
        _set(resource.RLIMIT_CPU, cpu)

        mem = limits.memory_mb * 1024 * 1024
        _set(resource.RLIMIT_AS, mem)
        _set(resource.RLIMIT_DATA, mem)
        _set(resource.RLIMIT_FSIZE, limits.disk_mb * 1024 * 1024)
        _set(resource.RLIMIT_NPROC, limits.max_processes)
        _set(resource.RLIMIT_CORE, 0)
        os.chdir(workdir)

    return preexec


class SubprocessSandbox(Sandbox):
    name = "subprocess"
    isolation_level = IsolationLevel.PROCESS

    def __init__(self, *, python: str | None = None):
        self._python = python or sys.executable

    def available(self) -> bool:
        return os.name == "posix"

    def unenforced_limits(self) -> tuple[str, ...]:
        """`max_processes` does not hold when the host process runs as root.

        `RLIMIT_NPROC` is checked against the real user id and is **not enforced for uid 0**, so a
        tool running under a root-owned worker can spawn without bound however small the limit is
        set. Measured, not assumed: a sandboxed loop performed 5000 sequential forks in under a
        second against `max_processes=8`.

        The wall-clock timeout and the process-group kill still contain such a tool, so the
        sandbox is not defeated — but the *process* bound is advisory here, and a caller relying
        on it deserves to be told rather than to find out. Running workers as an unprivileged user
        restores it; the container backend does not have the problem at all, because cgroups
        enforce `--pids-limit` regardless of uid.
        """
        return ("max_processes",) if os.geteuid() == 0 else ()

    def run_python(
        self,
        source: str,
        *,
        args: dict[str, Any] | None = None,
        limits: SandboxLimits | None = None,
        files: dict[str, str] | None = None,
        entrypoint: str = "main",
    ) -> SandboxResult:
        limits = limits or SandboxLimits()
        if not self.available():
            return SandboxResult(
                error="subprocess sandbox requires POSIX rlimits",
                backend=self.name, isolation_level=self.isolation_level,
            )

        workdir = tempfile.mkdtemp(prefix="civitas-tool-")
        started = time.perf_counter()
        try:
            for name, content in (files or {}).items():
                target = os.path.join(workdir, os.path.basename(name))
                with open(target, "w") as fh:
                    fh.write(content)

            # The source is passed as a literal inside the harness, so the tool file itself is
            # never importable by name and cannot shadow a stdlib module.
            program = f"_SOURCE = {source!r}\n{_HARNESS}"

            env = clean_environment()
            env["CIVITAS_TOOL_ARGS"] = json.dumps(args or {}, default=str)
            env["CIVITAS_TOOL_ENTRYPOINT"] = entrypoint
            env["CIVITAS_TOOL_NETWORK"] = "1" if limits.network else "0"
            env["TMPDIR"] = workdir
            env["HOME"] = workdir
            # -I: isolated mode. Ignores PYTHON* environment variables and does not put the
            # script's directory on sys.path, so the tool cannot import from the host tree.
            cmd = [self._python, "-I", "-c", program]

            proc = subprocess.Popen(
                cmd,
                cwd=workdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=_apply_limits(limits, workdir),
                start_new_session=True,
                text=True,
            )
            timed_out = False
            try:
                stdout, stderr = proc.communicate(timeout=limits.wall_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc)
                stdout, stderr = proc.communicate()

            duration_ms = (time.perf_counter() - started) * 1000
            limit_hit = None
            if timed_out:
                limit_hit = "wall"
            elif proc.returncode == -signal.SIGKILL:
                limit_hit = "memory"
            elif proc.returncode == -signal.SIGXCPU:
                limit_hit = "cpu"
            elif proc.returncode == -signal.SIGXFSZ:
                limit_hit = "disk"
            elif "MemoryError" in (stderr or ""):
                limit_hit = "memory"
            elif "BlockingIOError" in (stderr or "") or "Resource temporarily" in (stderr or ""):
                limit_hit = "processes"
            elif "network access is denied" in (stderr or ""):
                limit_hit = "network"

            truncated = False
            if len(stdout or "") > limits.max_output_bytes:
                stdout = stdout[: limits.max_output_bytes]
                truncated = True
            if len(stderr or "") > limits.max_output_bytes:
                stderr = stderr[: limits.max_output_bytes]
                truncated = True

            return SandboxResult(
                unenforced_limits=self.unenforced_limits(),
                exit_code=proc.returncode,
                stdout=stdout or "",
                stderr=stderr or "",
                duration_ms=duration_ms,
                timed_out=timed_out,
                limit_hit=limit_hit or ("output" if truncated else None),
                backend=self.name,
                isolation_level=self.isolation_level,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the whole process group.

    SIGTERM first would be politer, but untrusted code can install a handler and ignore it, and
    the timeout has already expired.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):  # pragma: no cover
        proc.kill()


def parse_result(stdout: str) -> tuple[str, Any]:
    """Split a tool's printed output from its returned value.

    The marker is what lets a tool both print diagnostics and return a value: without it, either
    the diagnostics corrupt the return value or the return value has to be smuggled through a file.
    """
    marker = "\n__CIVITAS_RESULT__"
    if marker not in stdout:
        return stdout, None
    text, _, payload = stdout.rpartition(marker)
    try:
        return text, json.loads(payload)
    except json.JSONDecodeError:
        return text, None
