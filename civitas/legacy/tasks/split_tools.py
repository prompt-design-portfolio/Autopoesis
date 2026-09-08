"""Probes for the split device (Part B §23).

Each probe is given to exactly one partition. The integrator gets **neither**, which is what makes
the benchmark a test of distributed cognition rather than of persistence: an integrator that could
probe would simply solve the task itself.
"""

from __future__ import annotations

from typing import Any

from civitas.domain.enums import EventType
from civitas.legacy.tasks.distributed import SplitDevice
from civitas.persistence.events import emit
from civitas.runtime.tools.base import Tool, ToolContext, ToolResult

#: The format partitions write their findings in, and the integrator reads them back from. An
#: unambiguous format keeps the measured effect attributable to *transmission* rather than to
#: parsing luck — the same reasoning as the hidden-rule family's finding lines.
FAMILY_FINDING = "SPLIT-FAMILY env={env} class={cls} family={fam}"
TABLE_FINDING = "SPLIT-TABLE env={env} family={fam} op={op}"


class ProbeFamilyTool(Tool):
    name = "probe_family"
    description = "Test whether an input class belongs to a material family. One pair per call."
    parameters = {
        "type": "object",
        "properties": {"input_class": {"type": "string"}, "family": {"type": "string"}},
        "required": ["input_class", "family"],
        "additionalProperties": False,
    }

    def __init__(self, device: SplitDevice):
        self._device = device

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        input_class = str(kw["input_class"]).strip().lower()
        family = str(kw["family"]).strip().lower()
        if input_class not in self._device.classes:
            return ToolResult(ok=False, error=f"unknown input class {input_class!r}")
        if family not in self._device.families:
            return ToolResult(ok=False, error=f"unknown family {family!r}")

        accepted = self._device.family_accepts(input_class, family)
        ctx.made_progress = True
        emit(ctx.session, workspace_id=ctx.workspace_id, type=EventType.TOOL_COMPLETED,
             episode_id=ctx.episode_id, actor_kind="agent",
             payload={"tool": self.name, "input_class": input_class, "family": family,
                      "accepted": accepted}, config_hash=ctx.config_hash)
        return ToolResult(
            ok=True,
            content=(
                f"{input_class} IS in family {family}." if accepted
                else f"{input_class} is NOT in family {family}."
            ),
            data={"accepted": accepted, "input_class": input_class, "family": family},
        )


class ProbeTableTool(Tool):
    name = "probe_table"
    description = "Test whether a material family accepts an operation. One pair per call."
    parameters = {
        "type": "object",
        "properties": {"family": {"type": "string"}, "operation": {"type": "string"}},
        "required": ["family", "operation"],
        "additionalProperties": False,
    }

    def __init__(self, device: SplitDevice):
        self._device = device

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        family = str(kw["family"]).strip().lower()
        operation = str(kw["operation"]).strip().lower()
        if family not in self._device.families:
            return ToolResult(ok=False, error=f"unknown family {family!r}")
        if operation not in self._device.operations:
            return ToolResult(ok=False, error=f"unknown operation {operation!r}")

        accepted = self._device.table_accepts(family, operation)
        ctx.made_progress = True
        emit(ctx.session, workspace_id=ctx.workspace_id, type=EventType.TOOL_COMPLETED,
             episode_id=ctx.episode_id, actor_kind="agent",
             payload={"tool": self.name, "family": family, "operation": operation,
                      "accepted": accepted}, config_hash=ctx.config_hash)
        return ToolResult(
            ok=True,
            content=(
                f"family {family} ACCEPTS {operation}." if accepted
                else f"family {family} REJECTS {operation}."
            ),
            data={"accepted": accepted, "family": family, "operation": operation},
        )


def registry_for(partition: str, device: SplitDevice):
    """The tools a partition gets.

    The integrator gets neither probe — that is the mechanism. An integrator able to probe would
    solve the task alone, and the benchmark would measure persistence rather than distributed
    cognition (§23).
    """
    from civitas.runtime.tools.builtin import default_registry

    registry = default_registry()
    if partition == "a":
        registry.add(ProbeFamilyTool(device))
    elif partition == "b":
        registry.add(ProbeTableTool(device))
    return registry
