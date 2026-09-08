"""The candidate-patch tool for the code-repair domain (Part B §16, §18, §22).

`try_patch` is to code repair what `probe_device` is to the hidden-rule device: one bounded,
costed test of one candidate, whose *negative* answers are as recordable as its positive one.

Two properties are load-bearing.

**It checks against the visible example, never the held-out checks.** An agent that could run the
evaluator's checks would be self-certifying, which §47 forbids. Passing the example is necessary
and not sufficient, so `try_patch` returning "accepted" is evidence, not a verdict.

**A rejection is registered as a documented failure.** That is what makes Part A §A2.3 measurable
here: a later agent that tries the same patch on the same program is repeating known-failed work,
and the duplicate index can say so.
"""

from __future__ import annotations

import uuid
from typing import Any

from civitas.domain.enums import ArtifactType, EventType
from civitas.knowledge.duplicate import record_failure
from civitas.persistence.events import emit
from civitas.persistence.models import ToolRun
from civitas.persistence.types import utcnow
from civitas.runtime.sandbox import SandboxLimits, get_sandbox
from civitas.runtime.tools.base import Tool, ToolContext, ToolResult

TOOL_NAME = "try_patch"

#: Tighter than the evaluator's: this runs one example, not a suite.
PATCH_LIMITS = SandboxLimits(cpu_seconds=5, wall_seconds=10, memory_mb=256, max_processes=1)

_HARNESS = '''
import json as _json

_ARGS = _json.loads({args!r})


def main():
    try:
        return {{"value": {entry}(*_ARGS)}}
    except Exception as exc:
        return {{"error": type(exc).__name__ + ": " + str(exc)[:200]}}
'''


class TryPatchTool(Tool):
    name = TOOL_NAME
    description = (
        "Apply one named candidate edit to the function and run it on the worked example shown "
        "in the task. Returns whether the example passes. One edit per call, and each call costs "
        "a tool call from your budget."
    )
    parameters = {
        "type": "object",
        "properties": {
            "patch": {"type": "string"},
            # Named in the call, not implied by the task, and not decoration. §A2.3 keys a
            # documented failure on (tool name, call arguments), so anything the key must
            # distinguish has to be *in* the arguments. Two functions in one workspace can be
            # offered the same surface label, and a key of `{"patch": ...}` alone would let a
            # rejection for one of them block a perfectly good attempt on the other.
            "entry": {"type": "string"},
        },
        "required": ["patch", "entry"],
        "additionalProperties": False,
    }

    def __init__(self, program: Any, *, definition_id: uuid.UUID | None = None,
                 sandbox: Any = None):
        self._program = program
        self._definition_id = definition_id
        self._sandbox = sandbox

    @property
    def sandbox(self):
        if self._sandbox is None:
            self._sandbox = get_sandbox()
        return self._sandbox

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        import json

        program = self._program
        patch = str(kw["patch"]).strip()
        entry = str(kw.get("entry", "")).strip()
        if entry != program.entry:
            return ToolResult(
                ok=False,
                error=f"this task is about {program.entry}, not {entry!r}",
            )
        if patch not in program.patch_names:
            return ToolResult(
                ok=False,
                error=f"unknown patch {patch!r}; this task offers: "
                f"{', '.join(program.patch_names)}",
            )

        source = program.apply(patch)
        args, want = program.example
        started = utcnow()
        result = self.sandbox.run_python(
            source + "\n\n" + _HARNESS.format(args=json.dumps(args), entry=program.entry),
            limits=PATCH_LIMITS, entrypoint="main",
        )
        from civitas.runtime.sandbox import parse_result

        _, payload = parse_result(result.stdout)
        ran = isinstance(payload, dict)
        accepted = bool(ran and "value" in payload and payload["value"] == want)
        why = (
            "the example passes" if accepted
            else (payload.get("error") if ran and "error" in payload
                  else "the example produces a different result"
                  if ran else f"the patched function did not run ({result.stderr[-200:]})")
        )

        session = ctx.session
        args = {"patch": patch, "entry": program.entry}
        session.add(ToolRun(
            episode_id=ctx.episode_id,
            tool_definition_id=self._definition_id,
            args=args,
            args_hash=self.args_hash(args),
            exit_code=result.exit_code if ran else 1,
            succeeded=accepted,
            stdout="accepted" if accepted else "rejected",
            started_at=started,
            ended_at=utcnow(),
            sandbox_backend=result.backend,
        ))

        if not accepted:
            # §A2.3: a rejection is a documented failure keyed by (tool, arguments), so a later
            # agent trying the same edit on the same program is repeating known-failed work.
            record_failure(
                session,
                workspace_id=ctx.workspace_id,
                artifact_id=None,
                episode_id=ctx.episode_id,
                environment_version=ctx.environment_version,
                summary=f"the patch {patch} does not repair {program.entry}: {why}",
                tool_name=TOOL_NAME,
                tool_args=args,
                ruled_out=[patch],
                reproducible=True,
            )

        emit(
            session, workspace_id=ctx.workspace_id, type=EventType.TOOL_COMPLETED,
            episode_id=ctx.episode_id,
            payload={"tool": TOOL_NAME, **args, "accepted": accepted},
            config_hash=ctx.config_hash,
        )

        # A rejection is progress. It rules a candidate out, which is exactly the information a
        # later agent inherits — and an episode that died of `no_progress` with budget unspent
        # after ruling out three edits would be recorded as an agent failure when it was a
        # bookkeeping one. M4 hit this same defect wiring §A2.3 into the device domain and it cost
        # a third of the measured advantage before it was found.
        ctx.made_progress = True
        session.flush()
        return ToolResult(
            ok=True,
            content=(
                f"PATCH-RESULT entry={program.entry} patch={patch} "
                f"verdict={'accepted' if accepted else 'rejected'} — {why}"
            ),
            data={"patch": patch, "entry": program.entry, "accepted": accepted},
        )


def suggested_artifact_type(accepted: bool) -> ArtifactType:
    """A rejection is a result. Recording it as an observation would lose that it is negative,
    and §12's negative knowledge is the half most collectives throw away."""
    return ArtifactType.OBSERVATION if accepted else ArtifactType.FAILURE
