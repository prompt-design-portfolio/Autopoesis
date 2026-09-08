"""The device probe (Part B §17, §22).

The device is injected per episode rather than looked up from the task row, because the task row
carries the evaluator specification and an agent must never reach that (§47). The tool can answer
"does the device accept this pair" without being able to answer "what is the mapping".

Every probe emits a `ToolRun`, so probes are counted, attributable, and available to credit
assignment on the same footing as any other tool (Part A §A2.1).
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from civitas.domain.enums import EventType
from civitas.knowledge.duplicate import record_failure
from civitas.legacy.tasks.hidden_rule import DeviceSpec
from civitas.persistence.events import emit
from civitas.persistence.models import ToolDefinition, ToolRun
from civitas.persistence.types import utcnow
from civitas.runtime.tools.base import Tool, ToolContext, ToolResult

TOOL_NAME = "probe_device"


class ProbeDeviceTool(Tool):
    name = TOOL_NAME
    description = (
        "Test one (input_class, operation) pair against the device. Returns whether the device "
        "accepts it. One pair per call, and each call costs a tool call from your budget."
    )
    parameters = {
        "type": "object",
        "properties": {
            "input_class": {"type": "string"},
            "operation": {"type": "string"},
        },
        "required": ["input_class", "operation"],
        "additionalProperties": False,
    }

    def __init__(self, device: DeviceSpec, *, definition_id: uuid.UUID | None = None):
        self._device = device
        self._definition_id = definition_id

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        input_class = str(kw["input_class"]).strip().lower()
        operation = str(kw["operation"]).strip().lower()

        if input_class not in self._device.classes:
            return ToolResult(
                ok=False,
                error=f"unknown input class {input_class!r}; this device has: "
                f"{', '.join(self._device.classes)}",
            )
        if operation not in self._device.operations:
            return ToolResult(
                ok=False,
                error=f"unknown operation {operation!r}; this device has: "
                f"{', '.join(self._device.operations)}",
            )

        accepted = self._device.accepts(input_class, operation)
        started = utcnow()
        args = {"input_class": input_class, "operation": operation}

        if self._definition_id is not None:
            ctx.session.add(
                ToolRun(
                    episode_id=ctx.episode_id,
                    tool_definition_id=self._definition_id,
                    args=args,
                    args_hash=self.args_hash(args),
                    exit_code=0,
                    succeeded=accepted,
                    stdout="accepted" if accepted else "rejected",
                    started_at=started,
                    ended_at=utcnow(),
                    sandbox_backend="builtin",
                )
            )
        if not accepted:
            # A rejected probe is a documented failure of this exact (tool, arguments) pair —
            # the first of the two forms §A2.3 names. Registering it is what makes
            # `duplicate_failure_rate` a live metric rather than a structurally-zero one: without
            # it, nothing in this task family can ever match a documented failure, and the
            # benchmark would report 0.000 for a mechanism it never exercised.
            record_failure(
                ctx.session,
                workspace_id=ctx.workspace_id,
                artifact_id=None,
                episode_id=ctx.episode_id,
                environment_version=ctx.environment_version,
                summary=f"the device rejects {operation} for {input_class}",
                tool_name=TOOL_NAME,
                tool_args=args,
                ruled_out=[operation],
                reproducible=True,
            )

        emit(
            ctx.session, workspace_id=ctx.workspace_id, type=EventType.TOOL_COMPLETED,
            episode_id=ctx.episode_id, actor_kind="agent",
            payload={"tool": TOOL_NAME, **args, "accepted": accepted},
            config_hash=ctx.config_hash,
        )
        # A probe is progress whichever way it comes out: a rejection rules an operation out, and
        # under a bijection that constrains every other class too. Treating only acceptance as
        # progress would terminate a correctly-working agent for `no_progress`.
        ctx.made_progress = True
        return ToolResult(
            ok=True,
            content=(
                f"The device ACCEPTS {operation} for {input_class}."
                if accepted
                else f"The device REJECTS {operation} for {input_class}."
            ),
            data={"accepted": accepted, **args},
        )


def ensure_definition(session, *, workspace_id: uuid.UUID) -> ToolDefinition:
    """Register the probe as a built-in tool so its runs are attributable (§17)."""
    existing = session.execute(
        select(ToolDefinition).where(
            ToolDefinition.workspace_id == workspace_id, ToolDefinition.name == TOOL_NAME
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    definition = ToolDefinition(
        workspace_id=workspace_id,
        name=TOOL_NAME,
        description="Probe one (input_class, operation) pair against the device.",
        kind="builtin",
        is_builtin=True,
    )
    session.add(definition)
    session.flush()
    return definition
