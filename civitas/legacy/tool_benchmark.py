"""The tool-accumulation benchmark (Part B §17, §24, §48).

§17 makes a claim that is easy to assert and easy to leave unmeasured:

> Later agents should discover and reuse good tools. [...] The civilization must accumulate
> technology, not just text.

This measures it, as a matched A/B over one variable — whether discovery and reuse are available
at all:

* **`reuse`** — agents get `list_tools` and `run_tool`, so a later generation can find what an
  earlier one built.
* **`rebuild`** — the control. Identical in every other respect: the same task, the same agents,
  the same budgets, the same workspace, the same accumulated artifacts. Only discovery is removed,
  so every generation has to build the tool again.

The control is what makes this a test of *accumulation* rather than of practice. Without it a
falling cost across generations would be indistinguishable from agents simply getting the task
right more often as the artifact record grows.

The task is deliberately one where a tool is worth building: a computation an agent cannot do in
one step within its budget, but which a correct tool does in one call.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import ExperimentArm, TaskStatus
from civitas.legacy.runner import SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION, ensure_profile
from civitas.persistence.models import Episode, Task, ToolRun, ToolVersion, Workspace
from civitas.runtime.budgets import Budgets
from civitas.runtime.episode import EpisodeRunner, EpisodeSpec
from civitas.runtime.providers.base import (
    Completion,
    CompletionRequest,
    ModelInfo,
    Provider,
    Role,
    ToolCall,
    Usage,
)
from civitas.runtime.tools.ecology import toolmaker_registry

TOOL_NAME = "checksum_digits"

#: The tool the task needs. Small, correct, and genuinely useful for the task — a tool that did
#: not help would make a null result uninterpretable.
TOOL_SOURCE = """
def main(values):
    total = 0
    for i, v in enumerate(values):
        total += (i + 1) * int(v)
    return total % 97
"""
TOOL_TESTS = """
def run_tests():
    assert main([1, 2, 3]) == 14 % 97
    assert main([]) == 0
    return "ok"
"""

BUDGETS = Budgets(
    tokens=200_000, context=32_000, tool_calls=8, cost_usd=1.0, wall_clock_s=120,
    max_turns=12, max_no_progress_turns=4,
)


def expected_answer(values: list[int]) -> int:
    return sum((i + 1) * v for i, v in enumerate(values)) % 97


@dataclass
class GenerationResult:
    generation: int
    succeeded: bool
    tool_calls: int
    #: Tokens, not tool calls, is the cost that actually separates building from reusing.
    #: Measured first with tool calls, which showed no difference — and could not: writing a tool
    #: and running one are one call each. What differs is the *content* of the call, and a
    #: `create_tool` call carries the entire source and test suite while a `run_tool` call carries
    #: only its arguments.
    tokens: int
    built_tool: bool
    reused_tool: bool


@dataclass
class ToolBenchmarkResult:
    arms: dict[str, list[GenerationResult]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for arm, runs in self.arms.items():
            n = len(runs) or 1
            # Generation 0 has nothing to reuse in either arm, so it is reported separately. A
            # mean over all generations is dominated by that first episode and understates the
            # steady state, which is the regime a long-lived collective actually operates in.
            steady = runs[1:] or runs
            steady_n = len(steady) or 1
            out[arm] = {
                "episodes": len(runs),
                "success_rate": round(sum(r.succeeded for r in runs) / n, 4),
                "mean_tool_calls": round(sum(r.tool_calls for r in runs) / n, 3),
                "mean_tokens": round(sum(r.tokens for r in runs) / n, 1),
                "steady_state_mean_tokens": round(sum(r.tokens for r in steady) / steady_n, 1),
                "builds": sum(r.built_tool for r in runs),
                "reuses": sum(r.reused_tool for r in runs),
                "reuse_rate": round(sum(r.reused_tool for r in runs) / n, 4),
                "by_generation": [
                    {"generation": r.generation, "succeeded": r.succeeded,
                     "tool_calls": r.tool_calls, "tokens": r.tokens,
                     "built": r.built_tool, "reused": r.reused_tool}
                    for r in runs
                ],
            }
        return out

    def derived(self) -> dict[str, Any]:
        reuse = self.summary().get("reuse")
        rebuild = self.summary().get("rebuild")
        if not reuse or not rebuild:
            return {"cost_saved_per_episode": None,
                    "reason": "both arms are required for the comparison"}
        return {
            "tokens_saved_per_episode": round(
                rebuild["mean_tokens"] - reuse["mean_tokens"], 1
            ),
            "steady_state_tokens_saved": round(
                rebuild["steady_state_mean_tokens"] - reuse["steady_state_mean_tokens"], 1
            ),
            "steady_state_saved_fraction": (
                round(
                    1 - reuse["steady_state_mean_tokens"] / rebuild["steady_state_mean_tokens"], 4
                )
                if rebuild["steady_state_mean_tokens"] else None
            ),
            "tokens_saved_fraction": (
                round(1 - reuse["mean_tokens"] / rebuild["mean_tokens"], 4)
                if rebuild["mean_tokens"] else None
            ),
            "tool_calls_saved_per_episode": round(
                rebuild["mean_tool_calls"] - reuse["mean_tool_calls"], 3
            ),
            "reuse_rate": reuse["reuse_rate"],
            "rebuilds_avoided": rebuild["builds"] - reuse["builds"],
        }


class ToolmakerPolicy(Provider):
    """A deterministic agent that prefers reuse to rebuilding (Part B §20).

    The preference is the *whole* behaviour under test, so it is explicit: look for an existing
    tool, run it if one exists, build it if none does. An agent that always rebuilt would make the
    `reuse` arm identical to `rebuild` and the benchmark would measure nothing; an agent that
    always reused would fail in the `rebuild` arm rather than adapting. This one does what a
    competent agent does, and the arms differ only in whether looking is possible.
    """

    name = "toolmaker-policy"

    def __init__(self, *, values: list[int], can_reuse: bool, source: str | None = None):
        self._values = values
        self._can_reuse = can_reuse
        self._source = source or TOOL_SOURCE

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(name=model or "toolmaker-v1", provider=self.name,
                         context_window=32_000, supports_tools=True, is_deterministic=True,
                         version="toolmaker-v1")

    def complete(self, request: CompletionRequest) -> Completion:
        listed = False
        tool_available = False
        built = False
        answer: int | None = None
        pending: list[ToolCall] = []

        for message in request.messages:
            if message.role is Role.ASSISTANT:
                pending = list(message.tool_calls)
                continue
            if message.role is not Role.TOOL:
                continue
            call = next((c for c in pending if c.id == message.tool_call_id), None)
            name = message.name or (call.name if call else "")
            if name == "list_tools":
                listed = True
                tool_available = TOOL_NAME in message.content
            elif name == "create_tool":
                built = True
                tool_available = "tests pass" in message.content
            elif name == "run_tool" and "returned:" in message.content:
                try:
                    answer = int(message.content.split("returned:")[1].split()[0])
                except (ValueError, IndexError):
                    answer = None

        available = {t.name for t in request.tools}
        step = len(request.messages)

        if answer is not None:
            call = ToolCall(f"c{step}", "submit_result",
                            {"answer": str(answer), "confidence": 0.9})
            return self._completion(request, call, "Submitting the computed checksum.")

        if self._can_reuse and not listed and "list_tools" in available:
            return self._completion(
                request, ToolCall(f"c{step}", "list_tools", {}),
                "Checking whether an earlier agent already built this.",
            )

        if tool_available and "run_tool" in available:
            return self._completion(
                request,
                ToolCall(f"c{step}", "run_tool",
                         {"name": TOOL_NAME, "args": {"values": self._values}}),
                "Reusing the existing tool.",
            )

        if not built and "create_tool" in available:
            return self._completion(
                request,
                ToolCall(f"c{step}", "create_tool", {
                    "name": TOOL_NAME,
                    "description": "Positional checksum of a list of integers, modulo 97.",
                    "source": self._source,
                    "test_source": TOOL_TESTS,
                    "parameters_schema": {
                        "type": "object",
                        "properties": {"values": {"type": "array"}},
                        "required": ["values"],
                    },
                }),
                "No usable tool exists; building one.",
            )

        return self._completion(
            request,
            ToolCall(f"c{step}", "submit_result", {"answer": "unknown", "confidence": 0.1}),
            "Could not compute the checksum.",
        )

    def _completion(self, request: CompletionRequest, call: ToolCall, text: str) -> Completion:
        prompt_tokens = self.count_request_tokens(request)
        return Completion(
            text=text, tool_calls=(call,),
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=self.count_tokens(text)),
            stop_reason="tool_use", model=request.model, model_version="toolmaker-v1",
        )


def _task(session: Session, workspace_id: uuid.UUID, values: list[int], generation: int) -> Task:
    task = Task(
        workspace_id=workspace_id,
        title=f"Compute the positional checksum (generation {generation})",
        description=(
            "Compute the positional checksum of the values below: multiply each value by its "
            "1-based position, sum the products, and take the result modulo 97.\n\n"
            f"values = {values}\n\n"
            "Check whether an earlier agent already built a tool for this before writing one."
        ),
        task_family="tool_accumulation",
        status=TaskStatus.READY,
        evaluator_spec={"kind": "exact_match", "expected": str(expected_answer(values))},
    )
    session.add(task)
    session.flush()
    return task


def padded_source(padding_lines: int) -> str:
    """The tool, padded to a realistic size.

    Tool size is the variable the saving scales with, and it has to be swept rather than assumed:
    at fifteen lines the source is a rounding error next to the tool declarations every turn
    carries, so a single measurement at one size says nothing about whether reuse pays. The
    padding is inert comment lines — it changes what a `create_tool` call costs to transmit and
    nothing else, which is exactly the variable under test.
    """
    if padding_lines <= 0:
        return TOOL_SOURCE
    filler = "\n".join(
        f"# implementation note {i}: this line stands in for the body of a real tool, which is "
        f"typically far longer than the fifteen lines this benchmark's logic needs."
        for i in range(padding_lines)
    )
    return f"{TOOL_SOURCE}\n{filler}\n"


def run_tool_benchmark(
    session: Session,
    *,
    organization_id: uuid.UUID,
    generations: int = 6,
    seed: int = 0,
    tool_padding_lines: int = 0,
) -> ToolBenchmarkResult:
    """Both arms, identical except for whether discovery is available."""
    from civitas.legacy.evaluation import evaluate_episode

    result = ToolBenchmarkResult()

    for arm_label, can_reuse in (("reuse", True), ("rebuild", False)):
        workspace = Workspace(
            organization_id=organization_id,
            name=f"tool benchmark {arm_label}",
            slug=f"tb-{arm_label}-{uuid.uuid4().hex[:8]}",
            environment_version="tool-benchmark",
        )
        session.add(workspace)
        session.flush()
        profile = ensure_profile(session, workspace.id, "toolmaker")
        runs: list[GenerationResult] = []

        for generation in range(generations):
            values = [(seed + generation * 7 + i * 3) % 50 + 1 for i in range(6)]
            task = _task(session, workspace.id, values, generation)

            registry = toolmaker_registry()
            if not can_reuse:
                # The control removes *discovery only*. `create_tool` stays, so both arms can
                # build; what differs is whether an agent can find what already exists.
                registry = registry.filtered([
                    n for n in registry.names() if n not in ("list_tools", "run_tool_disabled")
                ])
                registry = _without(registry, ("list_tools",))

            provider = ToolmakerPolicy(
                values=values, can_reuse=can_reuse, source=padded_source(tool_padding_lines)
            )
            spec = EpisodeSpec(
                workspace_id=workspace.id,
                agent_profile_id=profile.id,
                provider_name="toolmaker-policy",
                model_name="toolmaker-v1",
                model_version="toolmaker-v1",
                system_prompt=SYSTEM_PROMPT,
                system_prompt_version=SYSTEM_PROMPT_VERSION,
                budgets=BUDGETS,
                experiment_arm=ExperimentArm.COLLECTIVE,
                task_id=task.id,
                environment_version="tool-benchmark",
                seed=seed,
            )
            outcome = EpisodeRunner(session, provider=provider, tools=registry).run(spec)
            episode = session.get(Episode, outcome.episode_id)
            evaluation = evaluate_episode(
                session, episode=episode, outcome=outcome, task=task,
            )
            session.commit()

            # "Built" means *this episode wrote a tool version*, not that it created the
            # definition. Keying on the definition counted every rebuild after the first as a
            # reuse, which made the control arm look identical to the treatment.
            built = bool(
                session.execute(
                    select(ToolVersion).where(
                        ToolVersion.created_by_episode_id == episode.id
                    ).limit(1)
                ).scalar_one_or_none()
            )
            reused = (
                session.execute(
                    select(ToolRun).where(ToolRun.episode_id == episode.id).limit(1)
                ).scalar_one_or_none()
                is not None
            ) and not built

            runs.append(GenerationResult(
                generation=generation,
                succeeded=evaluation.succeeded,
                tool_calls=episode.tool_calls_used,
                tokens=episode.tokens_used,
                built_tool=built,
                reused_tool=reused,
            ))
        result.arms[arm_label] = runs

    result.notes.append(
        "The arms differ only in whether discovery (`list_tools`) is available. Both can build."
    )
    return result


def _without(registry, names: tuple[str, ...]):
    """A registry with specific tools removed."""
    from civitas.runtime.tools.base import ToolRegistry

    return ToolRegistry([
        registry.get(n) for n in registry.names() if n not in names
    ])
