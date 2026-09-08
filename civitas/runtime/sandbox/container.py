"""Container-isolated execution (Part B §18).

Docker with a hardened profile, and gVisor (`runsc`) where it is installed, which adds a
user-space kernel between the tool and the host. The two differ in one flag, so gVisor is a
subclass rather than a second implementation.

The hardening is not the default `docker run`. Every one of these is load-bearing:

* `--network none` — no egress at all unless the policy explicitly allows it (§18, §52).
* `--read-only` with a small `tmpfs` — the container filesystem cannot be modified, so a tool
  cannot persist anything outside the workspace mount.
* `--cap-drop ALL`, `--security-opt no-new-privileges` — no capability can be regained.
* `--pids-limit`, `--memory`, `--cpus` — the resource bounds §18 names, enforced by cgroups
  rather than by rlimits the code could raise.
* `--user` a non-root uid — root inside a container is still uid 0 against a mounted volume.
* No environment inheritance whatsoever: the env is built by `clean_environment` (§40).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
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
from civitas.runtime.sandbox.subprocess_sandbox import _HARNESS

DEFAULT_IMAGE = "python:3.11-slim"


class DockerSandbox(Sandbox):
    name = "docker"
    isolation_level = IsolationLevel.CONTAINER
    runtime_flag: list[str] = []

    def __init__(self, *, image: str = DEFAULT_IMAGE, docker_bin: str = "docker",
                 uid: int = 65534, gid: int = 65534):
        self.image = image
        self.docker_bin = docker_bin
        self.uid = uid
        self.gid = gid

    def available(self) -> bool:
        if shutil.which(self.docker_bin) is None:
            return False
        try:
            out = subprocess.run(
                [self.docker_bin, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, timeout=10, text=True,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if out.returncode != 0:
            return False
        return self._runtime_available()

    def _runtime_available(self) -> bool:
        return True

    def _flags(self, limits: SandboxLimits, workdir: str) -> list[str]:
        flags = [
            "run", "--rm", "-i",
            "--network", "none" if not limits.network else "bridge",
            "--read-only",
            # /tmp must be writable for the interpreter, but as a size-capped tmpfs with noexec,
            # so a tool cannot drop a binary there and run it.
            "--tmpfs", f"/tmp:rw,noexec,nosuid,size={limits.disk_mb}m",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(limits.max_processes),
            "--memory", f"{limits.memory_mb}m",
            # Without a swap limit equal to memory, a container over its memory limit swaps
            # instead of being killed, and the memory bound becomes advisory.
            "--memory-swap", f"{limits.memory_mb}m",
            "--cpus", str(max(0.1, limits.cpu_seconds / max(1.0, limits.wall_seconds))),
            "--user", f"{self.uid}:{self.gid}",
            "--workdir", "/work",
            "-v", f"{workdir}:/work:rw",
        ]
        for path in limits.read_only_paths:
            flags += ["-v", f"{path}:{path}:ro"]
        if not limits.gpu:
            pass
        else:
            flags += ["--gpus", "all"]
        flags += self.runtime_flag
        return flags

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
                error=f"{self.name} is not available in this environment",
                backend=self.name, isolation_level=self.isolation_level,
            )

        workdir = tempfile.mkdtemp(prefix="civitas-container-")
        os.chmod(workdir, 0o777)  # the container runs as a non-root uid and must be able to write
        started = time.perf_counter()
        try:
            for name, content in (files or {}).items():
                with open(os.path.join(workdir, os.path.basename(name)), "w") as fh:
                    fh.write(content)

            program = f"_SOURCE = {source!r}\n{_HARNESS}"
            env = clean_environment()
            env["CIVITAS_TOOL_ARGS"] = json.dumps(args or {}, default=str)
            env["CIVITAS_TOOL_ENTRYPOINT"] = entrypoint
            env["CIVITAS_TOOL_NETWORK"] = "1" if limits.network else "0"

            cmd = [self.docker_bin, *self._flags(limits, workdir)]
            for key, value in env.items():
                cmd += ["-e", f"{key}={value}"]
            cmd += [self.image, "python", "-I", "-c", program]

            timed_out = False
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    # A little longer than the container's own bound, so the container's limit is
                    # what fires and the reason recorded is accurate.
                    timeout=limits.wall_seconds + 15,
                )
                stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout = _decode(exc.stdout)
                stderr = _decode(exc.stderr)
                code = None

            limit_hit = None
            if timed_out:
                limit_hit = "wall"
            elif code == 137:  # 128 + SIGKILL: the cgroup OOM killer
                limit_hit = "memory"
            elif code == 152:  # 128 + SIGXCPU
                limit_hit = "cpu"
            elif "network access is denied" in (stderr or ""):
                limit_hit = "network"

            return SandboxResult(
                exit_code=code,
                stdout=(stdout or "")[: limits.max_output_bytes],
                stderr=(stderr or "")[: limits.max_output_bytes],
                duration_ms=(time.perf_counter() - started) * 1000,
                timed_out=timed_out,
                limit_hit=limit_hit,
                backend=self.name,
                isolation_level=self.isolation_level,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class GvisorSandbox(DockerSandbox):
    """Docker with the `runsc` runtime (Part B §18).

    gVisor interposes a user-space kernel, so a container escape needs a bug in gVisor rather than
    in the host kernel. `isolation_level` is `KERNEL` and the factory prefers it when present.
    """

    name = "gvisor"
    isolation_level = IsolationLevel.KERNEL
    runtime_flag = ["--runtime", "runsc"]

    def _runtime_available(self) -> bool:
        try:
            out = subprocess.run(
                [self.docker_bin, "info", "--format", "{{json .Runtimes}}"],
                capture_output=True, timeout=10, text=True,
            )
            return out.returncode == 0 and "runsc" in (out.stdout or "")
        except (OSError, subprocess.SubprocessError):
            return False


def _decode(stream) -> str:
    """`TimeoutExpired` carries bytes or str depending on how Popen was configured."""
    if stream is None:
        return ""
    return stream.decode(errors="replace") if isinstance(stream, bytes) else str(stream)
