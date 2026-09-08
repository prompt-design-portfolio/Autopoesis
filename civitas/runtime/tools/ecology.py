"""The tool ecology (Part B §17, §18) — how the civilization accumulates *technology*.

> The civilization must accumulate technology, not just text.

An agent writes a tool; the tool is validated by running its tests **in the sandbox**; later agents
discover it and run it. Three decisions carry most of the weight:

1. **A tool is untrusted code, always.** Validation runs it in the sandbox, and so does every
   later invocation. There is no path by which agent-written source executes on the host (§18).
2. **A tool that fails its own tests is kept, not discarded.** It is registered `TESTS_FAILING`
   and stays discoverable as negative knowledge (§12): "someone already tried to build this and it
   did not work" is exactly the information a later toolmaker needs, and deleting it guarantees
   the attempt is repeated.
3. **Tool quality is measured, never declared.** `ToolDefinition.downstream_utility` is written
   only by credit assignment (Part A §A2.1) and `success_rate` is a fold over real runs. An agent
   cannot claim its tool is good.
"""

from __future__ import annotations

import ast
import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from civitas.domain.enums import (
    ArtifactType,
    EventType,
    EvidenceKind,
    RelationType,
    ToolValidationState,
    ValidationState,
)
from civitas.persistence.events import emit
from civitas.persistence.models import (
    Artifact,
    ArtifactRelation,
    ToolDefinition,
    ToolRun,
    ToolVersion,
)
from civitas.persistence.types import canonical_json, utcnow
from civitas.runtime.sandbox import SandboxLimits, best_available, parse_result
from civitas.runtime.tools.base import Tool, ToolContext, ToolResult

#: Limits a tool runs under unless its version requests less. Deliberately tight: a tool that
#: needs more must say so, and the request is checked against policy rather than granted.
DEFAULT_TOOL_LIMITS = SandboxLimits(
    cpu_seconds=10.0, wall_seconds=20.0, memory_mb=256, disk_mb=32,
    max_processes=16, network=False,
)


@dataclass
class ValidationOutcome:
    state: ToolValidationState
    report: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.state is ToolValidationState.TESTS_PASSING


def _syntax_error(source: str) -> str | None:
    """Reject source that will not compile, before it reaches a sandbox.

    Cheap, and it gives the agent a precise error instead of an opaque non-zero exit — the same
    reasoning as `nbcheck` in this repository's research lineage, where a broken code cell shipped
    three times because container validity was mistaken for content validity.
    """
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return f"{exc.msg} at line {exc.lineno}"
    return None


def validate_source(
    source: str,
    *,
    test_source: str,
    entrypoint: str = "main",
    limits: SandboxLimits | None = None,
) -> ValidationOutcome:
    """Compile the tool, then run its tests inside the sandbox (§17, §18)."""
    error = _syntax_error(source)
    if error:
        return ValidationOutcome(
            ToolValidationState.TESTS_FAILING,
            {"stage": "compile", "error": error},
        )
    if not test_source.strip():
        # No tests is not the same as failing tests, and conflating them would let an untested
        # tool inherit the reputation of a tested one.
        return ValidationOutcome(ToolValidationState.UNTESTED, {"stage": "no_tests"})

    error = _syntax_error(test_source)
    if error:
        return ValidationOutcome(
            ToolValidationState.TESTS_FAILING,
            {"stage": "compile_tests", "error": error},
        )

    sandbox = best_available()
    harness = f"{source}\n\n{test_source}\n"
    result = sandbox.run_python(
        harness, args={}, limits=limits or DEFAULT_TOOL_LIMITS, entrypoint="run_tests"
    )
    _text, value = parse_result(result.stdout)

    report = {
        "stage": "tests",
        "exit_code": result.exit_code,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "limit_hit": result.limit_hit,
        "sandbox_backend": result.backend,
        "isolation_level": result.isolation_level.value,
        "unenforced_limits": list(result.unenforced_limits),
        "returned": value,
    }
    if result.succeeded:
        return ValidationOutcome(ToolValidationState.TESTS_PASSING, report)
    return ValidationOutcome(ToolValidationState.TESTS_FAILING, report)


def register_tool(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    name: str,
    description: str,
    source: str,
    parameters_schema: dict[str, Any],
    test_source: str = "",
    entrypoint: str = "main",
    episode_id: uuid.UUID | None = None,
    dependencies: list[str] | None = None,
    permissions: dict[str, Any] | None = None,
    config_hash: str = "",
) -> tuple[ToolDefinition, ToolVersion, ValidationOutcome]:
    """Create or advance a tool, validating it in the sandbox first (§17).

    An existing name adds a **version** rather than replacing the definition, which is what lets a
    tool improve across generations without every earlier reference breaking. A failing new
    version does not become current: the previous working version stays in place, so a bad edit
    degrades the collective's technology by nothing.
    """
    outcome = validate_source(source, test_source=test_source, entrypoint=entrypoint)

    definition = session.execute(
        select(ToolDefinition).where(
            ToolDefinition.workspace_id == workspace_id, ToolDefinition.name == name
        )
    ).scalar_one_or_none()
    created = definition is None
    if definition is None:
        definition = ToolDefinition(
            workspace_id=workspace_id, name=name, description=description,
            kind="python", created_by_episode_id=episode_id,
        )
        session.add(definition)
        session.flush()

    next_version = (
        session.execute(
            select(func.coalesce(func.max(ToolVersion.version), 0)).where(
                ToolVersion.tool_definition_id == definition.id
            )
        ).scalar_one()
        + 1
    )
    version = ToolVersion(
        tool_definition_id=definition.id,
        version=next_version,
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        entrypoint=entrypoint,
        parameters_schema=parameters_schema,
        dependencies=dependencies or [],
        permissions=permissions or {},
        validation_state=outcome.state,
        test_source=test_source,
        test_report=outcome.report,
        created_by_episode_id=episode_id,
    )
    session.add(version)
    session.flush()

    # A failing version never becomes current, so a bad edit cannot degrade the tool. An
    # untested one may become current only when there is nothing better to point at.
    is_first_untested = (
        definition.current_version_id is None
        and outcome.state is ToolValidationState.UNTESTED
    )
    if outcome.passed or is_first_untested:
        definition.current_version_id = version.id
    session.flush()

    # A tool is also a piece of knowledge. The artifact is what makes it retrievable alongside
    # everything else an agent might search for, and what gives credit assignment something to
    # attach to when a later episode's success depended on it.
    artifact = Artifact(
        workspace_id=workspace_id,
        type=ArtifactType.TOOL,
        title=f"Tool: {name} (v{next_version})",
        body=f"{description}\n\nEntrypoint: {entrypoint}\nValidation: {outcome.state.value}",
        creator_episode_id=episode_id,
        creator_kind="agent" if episode_id else "system",
        # A passing test suite is a tool's output verifying itself, which is stronger than an
        # assertion and weaker than an external evaluation (§13, §47).
        evidence_kind=(
            EvidenceKind.TOOL_OUTPUT if outcome.passed else EvidenceKind.MODEL_ASSERTION
        ),
        validation_state=(
            ValidationState.TOOL_VERIFIED if outcome.passed else ValidationState.SELF_REPORTED
        ),
        confidence=0.8 if outcome.passed else 0.3,
        structured={
            "tool_definition_id": str(definition.id),
            "tool_version_id": str(version.id),
            "version": next_version,
            "validation_state": outcome.state.value,
            "parameters_schema": parameters_schema,
        },
    )
    session.add(artifact)
    session.flush()
    if definition.artifact_id is None:
        definition.artifact_id = artifact.id
    elif definition.artifact_id != artifact.id:
        session.add(ArtifactRelation(
            source_id=artifact.id, target_id=definition.artifact_id,
            type=RelationType.REFINES, creator_episode_id=episode_id, creator_kind="system",
        ))

    try:
        from civitas.knowledge.embeddings import index_artifact

        index_artifact(session, artifact)
    except Exception:  # pragma: no cover - an optional index must not lose a write
        pass

    emit(
        session, workspace_id=workspace_id, type=EventType.TOOL_CREATED,
        episode_id=episode_id,
        payload={"tool": name, "version": next_version, "created": created,
                 "validation_state": outcome.state.value},
        config_hash=config_hash,
    )
    session.flush()
    return definition, version, outcome


def run_registered_tool(
    session: Session,
    *,
    definition: ToolDefinition,
    args: dict[str, Any],
    episode_id: uuid.UUID | None,
    workspace_id: uuid.UUID,
    version: ToolVersion | None = None,
    limits: SandboxLimits | None = None,
    config_hash: str = "",
) -> ToolRun:
    """Execute a registered tool in the sandbox and record the run (§17, §18, §A2.1)."""
    version = version or (
        session.get(ToolVersion, definition.current_version_id)
        if definition.current_version_id
        else None
    )
    started = utcnow()
    args_hash = hashlib.sha256(
        f"{definition.name}|{canonical_json(args)}".encode()
    ).hexdigest()

    if version is None:
        run = ToolRun(
            episode_id=episode_id, tool_definition_id=definition.id, args=args,
            args_hash=args_hash, succeeded=False,
            error="the tool has no runnable version", started_at=started, ended_at=utcnow(),
        )
        session.add(run)
        session.flush()
        return run

    requested = limits or DEFAULT_TOOL_LIMITS
    # A version may only *narrow* what it is granted. Allowing it to widen would let a tool
    # request the network by declaring it, which is the sandbox granting privileges on the word of
    # the code it is sandboxing (§18).
    permissions = version.permissions or {}
    effective = SandboxLimits(
        cpu_seconds=min(
            requested.cpu_seconds,
            float(permissions.get("cpu_seconds", requested.cpu_seconds)),
        ),
        wall_seconds=min(
            requested.wall_seconds,
            float(permissions.get("wall_seconds", requested.wall_seconds)),
        ),
        memory_mb=min(
            requested.memory_mb, int(permissions.get("memory_mb", requested.memory_mb))
        ),
        disk_mb=min(requested.disk_mb, int(permissions.get("disk_mb", requested.disk_mb))),
        max_processes=min(
            requested.max_processes,
            int(permissions.get("max_processes", requested.max_processes)),
        ),
        network=requested.network and bool(permissions.get("network", False)),
    )

    sandbox = best_available()
    result = sandbox.run_python(
        version.source, args=args, limits=effective, entrypoint=version.entrypoint
    )
    text, value = parse_result(result.stdout)

    run = ToolRun(
        episode_id=episode_id,
        tool_definition_id=definition.id,
        tool_version_id=version.id,
        args=args,
        args_hash=args_hash,
        exit_code=result.exit_code,
        succeeded=result.succeeded,
        stdout=text[-8000:],
        stderr=result.stderr[-8000:],
        error="" if result.succeeded else (result.error or result.stderr[-2000:]),
        duration_ms=result.duration_ms,
        sandbox_backend=result.backend,
        limit_hit=result.limit_hit,
        started_at=started,
        ended_at=utcnow(),
        meta={"returned": value, "isolation_level": result.isolation_level.value,
              "unenforced_limits": list(result.unenforced_limits)},
    )
    session.add(run)

    # Folds over real runs, never self-reported (§17, §29).
    definition.times_used += 1
    if result.succeeded:
        definition.times_succeeded += 1
    else:
        definition.times_failed += 1
    session.flush()

    emit(
        session, workspace_id=workspace_id, type=EventType.TOOL_COMPLETED,
        episode_id=episode_id,
        payload={"tool": definition.name, "version": version.version,
                 "succeeded": result.succeeded, "limit_hit": result.limit_hit},
        config_hash=config_hash,
    )
    return run


def discoverable_tools(
    session: Session, *, workspace_id: uuid.UUID, include_failing: bool = True
) -> list[tuple[ToolDefinition, ToolVersion | None]]:
    """Tools a later agent can find, best first (§17).

    Ranked by *measured* reliability and utility, then by usage. A failing tool is included by
    default and ranked last: "someone already tried to build this and it did not work" is
    information a toolmaker needs (§12), and hiding it guarantees the attempt is repeated.
    """
    definitions = list(
        session.execute(
            select(ToolDefinition).where(
                ToolDefinition.workspace_id == workspace_id,
                ToolDefinition.archived_at.is_(None),
            )
        ).scalars()
    )
    out: list[tuple[ToolDefinition, ToolVersion | None]] = []
    for definition in definitions:
        version = (
            session.get(ToolVersion, definition.current_version_id)
            if definition.current_version_id
            else None
        )
        if version is None:
            # No *current* version means every version failed validation — a failing version never
            # becomes current, so a bad edit cannot degrade the tool. Falling back to the latest
            # version is what makes that visible: without it a failing tool reports as "untested",
            # which is the one state it is definitely not in, and it would then outrank a working
            # tool in the ordering below.
            version = session.execute(
                select(ToolVersion)
                .where(ToolVersion.tool_definition_id == definition.id)
                .order_by(ToolVersion.version.desc())
                .limit(1)
            ).scalar_one_or_none()

        state = version.validation_state if version else ToolValidationState.UNTESTED
        if not include_failing and state is ToolValidationState.TESTS_FAILING:
            continue
        out.append((definition, version))

    #: Passing above untested above failing. Untested sits between the two on purpose: it may work
    #: and nothing has checked, whereas failing has been checked and does not.
    order = {
        ToolValidationState.VALIDATED: 4,
        ToolValidationState.TESTS_PASSING: 3,
        ToolValidationState.UNTESTED: 2,
        ToolValidationState.DEPRECATED: 1,
        ToolValidationState.TESTS_FAILING: 0,
        ToolValidationState.QUARANTINED: 0,
    }

    def rank(entry: tuple[ToolDefinition, ToolVersion | None]) -> tuple:
        definition, version = entry
        state = version.validation_state if version else ToolValidationState.UNTESTED
        # A tool that has never run has `success_rate is None`, which is not the same as 0.0 — a
        # fresh tool must not rank below a broken one.
        rate = definition.success_rate
        return (
            order.get(state, 0),
            definition.downstream_utility,
            rate if rate is not None else 0.5,
            definition.times_used,
        )

    out.sort(key=rank, reverse=True)
    return out


# --------------------------------------------------------------------------
# the agent-facing tools
# --------------------------------------------------------------------------
class CreateToolTool(Tool):
    name = "create_tool"
    description = (
        "Write a reusable Python tool that later agents can discover and run. Provide `source` "
        "defining your entrypoint function, and `test_source` defining run_tests() which raises "
        "on failure. Your tool is validated by running its tests in a sandbox before it is "
        "registered."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "source": {"type": "string"},
            "test_source": {"type": "string"},
            "entrypoint": {"type": "string"},
            "parameters_schema": {"type": "object"},
        },
        "required": ["name", "description", "source"],
    }

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        name = str(kw["name"]).strip().lower().replace(" ", "_")
        if not name.isidentifier():
            return ToolResult(ok=False, error=f"{name!r} is not a valid tool name")

        definition, version, outcome = register_tool(
            ctx.session,
            workspace_id=ctx.workspace_id,
            name=name,
            description=kw["description"],
            source=kw["source"],
            test_source=kw.get("test_source", ""),
            entrypoint=kw.get("entrypoint", "main"),
            parameters_schema=kw.get("parameters_schema") or {"type": "object", "properties": {}},
            episode_id=ctx.episode_id,
            config_hash=ctx.config_hash,
        )
        ctx.made_progress = True

        if outcome.passed:
            content = (
                f"Registered {name} v{version.version}: tests pass. "
                f"Later agents can find it with list_tools and run it with run_tool."
            )
        elif outcome.state is ToolValidationState.UNTESTED:
            content = (
                f"Registered {name} v{version.version} as UNTESTED — no tests were supplied, so "
                f"nothing verifies it. Supply test_source to have it validated."
            )
        else:
            content = (
                f"Registered {name} v{version.version} as FAILING. It is kept and remains "
                f"discoverable so nobody repeats the attempt.\n"
                f"{outcome.report.get('error') or outcome.report.get('stderr', '')[-800:]}"
            )
        return ToolResult(
            ok=True, content=content,
            data={"tool": name, "version": version.version,
                  "validation_state": outcome.state.value},
        )


class ListToolsTool(Tool):
    name = "list_tools"
    description = (
        "List the tools earlier agents have built, best first. Reuse one before writing your own."
    )
    read_only = True
    parameters = {"type": "object", "properties": {"include_failing": {"type": "boolean"}}}

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        entries = discoverable_tools(
            ctx.session, workspace_id=ctx.workspace_id,
            include_failing=bool(kw.get("include_failing", True)),
        )
        if not entries:
            return ToolResult(ok=True, content="No tools have been built yet.", data={"count": 0})

        lines = [f"{len(entries)} tool(s), best first:"]
        for definition, version in entries:
            rate = definition.success_rate
            lines.append(
                f"- {definition.name} (v{version.version if version else '-'}) "
                f"[{version.validation_state.value if version else 'untested'}] "
                f"runs={definition.times_used} "
                f"success={'n/a' if rate is None else f'{rate:.2f}'} "
                f"utility={definition.downstream_utility:+.2f}\n"
                f"    {definition.description[:200]}"
            )
        return ToolResult(ok=True, content="\n".join(lines),
                          data={"count": len(entries),
                                "names": [d.name for d, _v in entries]})


class RunToolTool(Tool):
    name = "run_tool"
    description = "Run a tool an earlier agent built, by name, with arguments."
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "args": {"type": "object"}},
        "required": ["name"],
    }

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        definition = ctx.session.execute(
            select(ToolDefinition).where(
                ToolDefinition.workspace_id == ctx.workspace_id,
                ToolDefinition.name == str(kw["name"]).strip().lower(),
            )
        ).scalar_one_or_none()
        if definition is None:
            return ToolResult(ok=False, error=f"no tool named {kw['name']!r} in this workspace")

        run = run_registered_tool(
            ctx.session, definition=definition, args=kw.get("args") or {},
            episode_id=ctx.episode_id, workspace_id=ctx.workspace_id,
            config_hash=ctx.config_hash,
        )
        ctx.made_progress = True

        if not run.succeeded:
            return ToolResult(
                ok=False,
                error=f"{definition.name} failed"
                      f"{f' ({run.limit_hit} limit)' if run.limit_hit else ''}: "
                      f"{run.error[:600]}",
            )
        returned = (run.meta or {}).get("returned")
        return ToolResult(
            ok=True,
            content=f"{definition.name} returned: {returned}\n{run.stdout[:2000]}",
            data={"returned": returned, "tool": definition.name},
        )


def toolmaker_registry():
    """The built-in tools plus the tool ecology (§17)."""
    from civitas.runtime.tools.builtin import default_registry

    registry = default_registry()
    registry.add(CreateToolTool())
    registry.add(ListToolsTool())
    registry.add(RunToolTool())
    return registry
