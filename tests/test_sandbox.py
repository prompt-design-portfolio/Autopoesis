"""Sandbox isolation and secrets (Part B §18, §40, §52, §55).

These drive the real sandbox. Each limit is exercised by code that would breach it, so a limit
that silently stopped being applied fails a test rather than becoming a latent vulnerability.
"""

from __future__ import annotations

import pytest

from civitas.config import SandboxBackend, Settings
from civitas.runtime.sandbox import (
    SandboxLimits,
    SandboxUnavailable,
    SubprocessSandbox,
    best_available,
    clean_environment,
    get_sandbox,
    parse_result,
)


@pytest.fixture
def sandbox():
    sb = SubprocessSandbox()
    if not sb.available():
        pytest.skip("POSIX rlimits required")
    return sb


def test_a_tool_runs_and_returns_a_value(sandbox):
    result = sandbox.run_python(
        "def main(x):\n    print('log line')\n    return {'doubled': x * 2}\n", args={"x": 21}
    )
    assert result.succeeded
    text, value = parse_result(result.stdout)
    assert "log line" in text
    assert value == {"doubled": 42}


def test_outbound_network_is_denied_by_default(sandbox):
    """§18 requires an outbound network policy. The only safe default for model-generated code
    is none at all."""
    result = sandbox.run_python(
        "import socket\n"
        "def main():\n"
        "    return socket.create_connection(('example.com', 80))\n"
    )
    assert not result.succeeded
    assert result.limit_hit == "network"


def test_a_wall_clock_timeout_kills_the_whole_process_group(sandbox):
    """Killing only the direct child leaves its children running."""
    result = sandbox.run_python(
        "import subprocess, sys, time\n"
        "def main():\n"
        "    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "    time.sleep(60)\n",
        limits=SandboxLimits(wall_seconds=2),
    )
    assert result.timed_out
    assert result.limit_hit == "wall"
    assert result.duration_ms < 10_000


def test_the_memory_limit_binds(sandbox):
    result = sandbox.run_python(
        "def main():\n    return len(bytearray(400 * 1024 * 1024))\n",
        limits=SandboxLimits(memory_mb=64, wall_seconds=20),
    )
    assert not result.succeeded
    assert result.limit_hit == "memory"


def test_a_fork_bomb_is_contained(sandbox):
    """The process limit plus the group kill: neither alone is enough."""
    result = sandbox.run_python(
        "import os\n"
        "def main():\n"
        "    for _ in range(10000):\n"
        "        try:\n"
        "            os.fork()\n"
        "        except OSError:\n"
        "            pass\n"
        "    return 'survived'\n",
        limits=SandboxLimits(max_processes=8, wall_seconds=5, memory_mb=128),
    )
    assert result.limit_hit is not None or not result.succeeded


def test_output_is_truncated_rather_than_exhausting_memory(sandbox):
    result = sandbox.run_python(
        "def main():\n    print('x' * 5_000_000)\n    return 1\n",
        limits=SandboxLimits(max_output_bytes=10_000, wall_seconds=20, memory_mb=256),
    )
    assert len(result.stdout) <= 10_000
    assert result.limit_hit == "output"


def test_the_tool_cannot_import_from_the_host_tree(sandbox):
    """`-I` keeps the host's Python path out of the sandbox, so a tool cannot reach Civitas's own
    code — including its settings, and therefore its credentials."""
    result = sandbox.run_python(
        "def main():\n"
        "    import civitas.config\n"
        "    return 'imported civitas'\n"
    )
    assert not result.succeeded
    assert "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr


# --------------------------------------------------------------------------
# secrets (Part B §40)
# --------------------------------------------------------------------------
def test_no_credential_reaches_a_sandboxed_process(sandbox, monkeypatch):
    """§40: provider credentials are never exposed to agent sandboxes."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-LEAKED-SECRET")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-LEAKED-SECRET")
    monkeypatch.setenv("CIVITAS_JWT_SECRET", "LEAKED-SECRET")
    monkeypatch.setenv("DATABASE_PASSWORD", "LEAKED-SECRET")
    monkeypatch.setenv("MY_SESSION_COOKIE", "LEAKED-SECRET")

    result = sandbox.run_python(
        "import os\n"
        "def main():\n"
        "    return dict(os.environ)\n"
    )
    assert result.succeeded
    _text, env = parse_result(result.stdout)
    assert "LEAKED-SECRET" not in str(env), f"a credential reached the sandbox: {env}"


def test_the_environment_filter_is_an_allowlist_not_a_denylist():
    """A denylist over a process environment guarantees the next unfamiliar variable gets
    through."""
    env = clean_environment({
        "PATH": "/usr/bin",
        "ANTHROPIC_API_KEY": "sk-secret",
        "SOME_FUTURE_VENDOR_CREDENTIAL": "secret",
        "AN_ORDINARY_VARIABLE": "harmless-but-unexpected",
    })
    assert env["PATH"] == "/usr/bin"
    assert "ANTHROPIC_API_KEY" not in env
    assert "SOME_FUTURE_VENDOR_CREDENTIAL" not in env
    assert "AN_ORDINARY_VARIABLE" not in env, "unknown variables must not be inherited"


def test_a_settings_manifest_never_contains_a_secret():
    """§40, §46: manifests exclude secrets by *type*, so one added later cannot leak by name."""
    s = Settings(
        database_url="postgresql://user:HUNTER2@db.internal:5432/civitas",
        anthropic_api_key="sk-ant-SECRET",
        openai_api_key="sk-SECRET",
        jwt_secret="JWT-SECRET",
    )
    blob = str(s.manifest_dict())
    for secret in ("HUNTER2", "sk-ant-SECRET", "sk-SECRET", "JWT-SECRET"):
        assert secret not in blob, f"{secret} leaked into the manifest"
    assert s.manifest_dict()["database_url"] == "postgresql"


def test_repr_does_not_print_a_secret():
    s = Settings(anthropic_api_key="sk-ant-SECRET", jwt_secret="JWT-SECRET")
    assert "sk-ant-SECRET" not in repr(s)
    assert "JWT-SECRET" not in repr(s)


# --------------------------------------------------------------------------
# backend selection (Part B §18, §33, §46)
# --------------------------------------------------------------------------
def test_selecting_none_is_refused():
    """§18 forbids executing agent-generated code on the host."""
    s = Settings(sandbox_backend=SandboxBackend.NONE)
    with pytest.raises(SandboxUnavailable, match="§18"):
        get_sandbox(s)


def test_an_unavailable_backend_raises_rather_than_downgrading():
    """A silent downgrade would put a process-isolated result into a manifest claiming container
    isolation — the confusion §46 exists to prevent."""
    s = Settings(sandbox_backend=SandboxBackend.DOCKER)
    from civitas.runtime.sandbox.container import DockerSandbox

    if DockerSandbox().available():
        pytest.skip("docker is available here, so there is nothing to downgrade from")
    with pytest.raises(SandboxUnavailable, match="not available"):
        get_sandbox(s)


def test_the_backend_in_force_is_reported_for_the_manifest():
    sb = best_available()
    described = sb.describe()
    assert described["available"]
    assert described["isolation_level"] in ("process", "container", "kernel")
    assert described["backend"] in ("subprocess", "docker", "gvisor")
