"""Cumulative culture (Part B §24) and the capability frontier (Part B §25).

## §24 — cumulative culture

> Build sequential tasks where later tasks depend on earlier collective discoveries. Task 1
> produces method A. Task 2 uses A to create tool B. Task 3 uses B to discover abstraction C.
> Task 4 depends on C. Keep the base model frozen.

The chain here is literal, and the dependency is on *which question to ask*. Generation *n*'s
input class is a deterministic function of generation *n−1*'s **answer**:

    class(n) = classes[ sha256(result(n-1)) mod |classes| ]

An agent in generation 3 that cannot find generation 2's result does not know **which class to
probe**, so it cannot start — not merely start slower. That is what makes the broken-chain control
a genuine control rather than a handicap: the agents, budgets and tasks are identical, and only the
availability of the prerequisite differs.

The control is a **broken chain**: the identical generations, the identical agents, the identical
budgets — but each generation's workspace is emptied first, so nothing carries. That is what
separates *accumulation* from *the tasks getting easier*.

## §25 — capability frontier

> Create benchmark families of increasing difficulty. Compare solo, independent, shared memory,
> collective. Hold the individual model constant. Measure maximum task difficulty successfully
> solved.

Difficulty is the number of candidate operations: a class among *k* operations needs ~k/2 probes to
resolve blind, so difficulty scales the gap between what a lone agent can afford and what the
collective already knows. The frontier is the largest *k* at which an arm still clears a stated
threshold, and it is reported as `None` — not as the smallest tested difficulty — when an arm
fails even the easiest, because "never cleared the bar" is not a frontier of zero.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import ExperimentArm, TaskStatus
from civitas.experiments.evaluation import evaluate_episode
from civitas.experiments.runner import (
    DEFAULT_BUDGETS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    ensure_profile,
    run_benchmark_episode,
)
from civitas.experiments.tasks.hidden_rule import build_device, era_instances
from civitas.experiments.tasks.probe_tool import ProbeDeviceTool, ensure_definition
from civitas.persistence.models import Artifact, Episode, Task, Workspace
from civitas.persistence.types import utcnow
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
from civitas.runtime.tools.builtin import default_registry

CHAIN_FINDING = "CHAIN env={env} generation={gen} result={result}"


# ==========================================================================
# §24 cumulative culture
# ==========================================================================
@dataclass
class GenerationOutcome:
    generation: int
    succeeded: bool
    had_prerequisite: bool
    artifacts_read: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CumulativeResult:
    arms: dict[str, list[GenerationOutcome]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arms": {
                k: [o.as_dict() for o in v] for k, v in self.arms.items()
            },
            "summary": self.summary(),
            "derived": self.derived(),
            "notes": self.notes,
        }

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for arm, outcomes in self.arms.items():
            by_generation: dict[int, list[GenerationOutcome]] = {}
            for outcome in outcomes:
                by_generation.setdefault(outcome.generation, []).append(outcome)
            out[arm] = {
                "n": len(outcomes),
                "success_rate": round(
                    sum(o.succeeded for o in outcomes) / len(outcomes), 4
                ) if outcomes else 0.0,
                "by_generation": {
                    str(gen): round(sum(o.succeeded for o in rows) / len(rows), 4)
                    for gen, rows in sorted(by_generation.items())
                },
                #: The deepest generation the arm ever reached. This is the cumulative claim:
                #: a chain that stops at generation 1 has not accumulated anything.
                "deepest_generation_reached": max(
                    (o.generation for o in outcomes if o.succeeded), default=None
                ),
            }
        return out

    def derived(self) -> dict[str, Any]:
        summary = self.summary()
        intact, broken = summary.get("chained"), summary.get("broken_chain")
        if not intact or not broken:
            return {"reason": "both arms are required for the comparison"}
        return {
            "depth_gain": (
                None
                if intact["deepest_generation_reached"] is None
                or broken["deepest_generation_reached"] is None
                else intact["deepest_generation_reached"]
                - broken["deepest_generation_reached"]
            ),
            "success_rate_gain": round(
                intact["success_rate"] - broken["success_rate"], 4
            ),
        }


class ChainPolicy(Provider):
    """A deterministic agent for one generation of the chain (§20, §24).

    Generation 1 probes for its own answer. Every later generation **cannot probe for its input**
    — it must read the previous generation's recorded result and then probe using it. That is the
    dependency: without the prior result the agent has nothing to probe *with*.
    """

    name = "chain-policy"

    def __init__(self, *, device, generation: int, input_class: str | None, env: str,
                 max_reads: int = 4):
        self._device = device
        self._generation = generation
        #: Known only for generation 1. Every later generation must derive it from the previous
        #: result, which is the dependency under test.
        self._input_class = input_class
        self._env = env
        self._max_reads = max_reads

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(name=model or "chain-v1", provider=self.name, context_window=32_000,
                         supports_tools=True, is_deterministic=True, version="chain-v1")

    def complete(self, request: CompletionRequest) -> Completion:
        import re

        from civitas.experiments.policy_agent import ID_RE

        chain_re = re.compile(
            r"CHAIN\s+env=(?P<env>\S+)\s+generation=(?P<gen>\d+)\s+result=(?P<res>\S+)"
        )
        searched = False
        ranked: list[str] = []
        read: set[str] = set()
        inherited: str | None = None
        probed: set[str] = set()
        accepted: str | None = None
        recorded = False
        pending: list[ToolCall] = []

        for message in request.messages:
            if message.role is Role.ASSISTANT:
                pending = list(message.tool_calls)
                continue
            if message.role is not Role.TOOL:
                continue
            call = next((c for c in pending if c.id == message.tool_call_id), None)
            name = message.name or (call.name if call else "")
            if name == "search_knowledge":
                searched = True
                ranked = ID_RE.findall(message.content or "")
            elif name == "read_artifact":
                if call is not None:
                    read.add(str(call.arguments.get("artifact_id", "")))
                for match in chain_re.finditer(message.content or ""):
                    if (
                        match.group("env") == self._env
                        and int(match.group("gen")) == self._generation - 1
                    ):
                        inherited = match.group("res").lower()
            elif name == "probe_device" and call is not None:
                operation = str(call.arguments.get("operation", "")).lower()
                probed.add(operation)
                if "ACCEPTS" in message.content:
                    accepted = operation
            elif name == "create_artifact":
                recorded = True

        available = {t.name for t in request.tools}
        step = len(request.messages)

        # Which class this generation is about. Generation 1 is told; every later generation must
        # derive it from the inherited result, and without that has no question to ask.
        target = self._input_class
        if self._generation > 1:
            target = (
                derived_class(self._device.classes, inherited) if inherited else None
            )

        if accepted is not None:
            if not recorded and "create_artifact" in available:
                line = CHAIN_FINDING.format(
                    env=self._env, gen=self._generation, result=accepted
                )
                return self._call(request, ToolCall(f"c{step}", "create_artifact", {
                    "type": "evidence",
                    "title": f"Generation {self._generation} result for {target}",
                    "body": line, "confidence": 0.9,
                }), "Recording this generation's result for the next.")
            return self._call(request, ToolCall(f"c{step}", "submit_result", {
                "answer": accepted, "confidence": 0.9}), "Submitting.")

        if target is None:
            if not searched and "search_knowledge" in available:
                return self._call(request, ToolCall(f"c{step}", "search_knowledge", {
                    "query": f"generation {self._generation - 1} result", "limit": 10,
                }), "Looking for the previous generation's result.")
            unread = [i for i in ranked if i not in read]
            if unread and len(read) < self._max_reads and "read_artifact" in available:
                return self._call(request, ToolCall(f"c{step}", "read_artifact", {
                    "artifact_id": unread[0]}), "Reading a candidate result.")
            # No prerequisite: this generation does not know which class to investigate, so it
            # cannot start. That is the dependency, not a handicap.
            return self._call(request, ToolCall(f"c{step}", "submit_result", {
                "answer": "unreachable", "confidence": 0.05,
                "reasoning": f"generation {self._generation - 1}'s result was not available, so "
                             f"the class under investigation is unknown",
            }), "The prerequisite is missing.")

        remaining = [o for o in self._device.operations if o not in probed]
        if remaining and "probe_device" in available:
            return self._call(request, ToolCall(f"c{step}", "probe_device", {
                "input_class": target, "operation": remaining[0],
            }), f"Probing {remaining[0]} for {target}.")

        return self._call(request, ToolCall(f"c{step}", "submit_result", {
            "answer": "unknown", "confidence": 0.05}), "Exhausted.")

    def _call(self, request: CompletionRequest, call: ToolCall, text: str) -> Completion:
        return Completion(
            text=text, tool_calls=(call,),
            usage=Usage(prompt_tokens=self.count_request_tokens(request),
                        completion_tokens=self.count_tokens(text)),
            stop_reason="tool_use", model=request.model, model_version="chain-v1",
        )


def derived_class(classes: tuple[str, ...], previous_result: str) -> str:
    """Which class generation *n* is about, given generation *n−1*'s answer.

    A stable hash, not `hash()`: Python randomises string hashing per process, and a chain whose
    links move between runs is not reproducible (§46).
    """
    import hashlib

    digest = hashlib.sha256(previous_result.encode()).hexdigest()
    return classes[int(digest[:8], 16) % len(classes)]


def run_cumulative_benchmark(
    session: Session,
    *,
    organization_id: uuid.UUID,
    generations: int = 4,
    seeds: list[int] | None = None,
    budgets: Budgets | None = None,
    config_hash: str = "cumulative-v1",
) -> CumulativeResult:
    """§24, with a broken-chain control."""
    seeds = seeds or [0, 1, 2]
    #: Enough for a full blind sweep of the operation space plus recording and submitting.
    #: Measured, not guessed: at six calls a generation-1 agent found the answer on its last probe
    #: and had nothing left to record or submit it with, so every arm read 0.000.
    budgets = budgets or Budgets(
        tokens=300_000, context=32_000, tool_calls=14, cost_usd=1.0, wall_clock_s=180,
        max_turns=20, max_no_progress_turns=5,
    )
    result = CumulativeResult(arms={"chained": [], "broken_chain": []})

    for seed in seeds:
        device = build_device(seed=seed, era=1, n_classes=max(4, generations), n_ops=10)
        instances = era_instances(device)

        for arm_label, chained in (("chained", True), ("broken_chain", False)):
            workspace = Workspace(
                organization_id=organization_id,
                name=f"cumulative {arm_label}",
                slug=f"cum-{arm_label}-{uuid.uuid4().hex[:8]}",
                environment_version=device.environment_version,
            )
            session.add(workspace)
            session.flush()
            profile = ensure_profile(session, workspace.id, "chain")
            definition = ensure_definition(session, workspace_id=workspace.id)

            previous_result: str | None = None
            for generation in range(1, generations + 1):
                if generation == 1:
                    input_class = instances[0].input_class
                else:
                    # The chain's truth, computed here from the *actual* previous answer so the
                    # evaluator and the agent agree on which question generation n was asked.
                    input_class = (
                        derived_class(device.classes, previous_result)
                        if previous_result else None
                    )
                expected = (
                    device.answer_for(input_class) if input_class else "unreachable"
                )
                task = Task(
                    workspace_id=workspace.id,
                    title=f"Generation {generation}",
                    description=(
                        "Determine the operation the device accepts for the input class under "
                        "investigation.\n\n"
                        + (
                            f"Input class: {input_class}\n"
                            if generation == 1
                            else "The class under investigation is determined by generation "
                                 f"{generation - 1}'s result, which another agent recorded. "
                                 "Search the collective knowledge base for it first — without it "
                                 "you do not know which class to investigate.\n"
                        )
                        + f"Operations: {', '.join(device.operations)}\n"
                    ),
                    task_family="cumulative", status=TaskStatus.READY,
                    evaluator_spec={"kind": "exact_match", "expected": expected},
                )
                session.add(task)
                session.flush()

                had_prerequisite = generation == 1 or chained
                registry = default_registry()
                registry.add(ProbeDeviceTool(device, definition_id=definition.id))
                spec = EpisodeSpec(
                    workspace_id=workspace.id, agent_profile_id=profile.id,
                    provider_name="chain-policy", model_name="chain-v1",
                    model_version="chain-v1", system_prompt=SYSTEM_PROMPT,
                    system_prompt_version=SYSTEM_PROMPT_VERSION, budgets=budgets,
                    experiment_arm=ExperimentArm.COLLECTIVE, task_id=task.id,
                    environment_version=device.environment_version,
                    config_hash=config_hash, seed=seed,
                )
                outcome = EpisodeRunner(
                    session,
                    provider=ChainPolicy(
                        device=device, generation=generation,
                        input_class=instances[0].input_class if generation == 1 else None,
                        env=device.environment_version,
                    ),
                    tools=registry,
                ).run(spec)
                episode = session.get(Episode, outcome.episode_id)
                evaluation = evaluate_episode(
                    session, episode=episode, outcome=outcome, task=task,
                    config_hash=config_hash,
                )
                result.arms[arm_label].append(GenerationOutcome(
                    generation=generation,
                    succeeded=evaluation.succeeded,
                    had_prerequisite=had_prerequisite,
                    artifacts_read=episode.artifacts_read,
                ))
                # The chain's next link depends on this generation's *true* answer, which is what
                # the agent recorded when it succeeded.
                if evaluation.succeeded and input_class:
                    previous_result = device.answer_for(input_class)

                if not chained:
                    # Break the chain: empty the workspace between generations, so the identical
                    # agents face the identical tasks with nothing carried. This is what separates
                    # accumulation from the tasks simply getting easier.
                    for artifact in session.execute(
                        select(Artifact).where(
                            Artifact.workspace_id == workspace.id,
                            Artifact.archived_at.is_(None),
                        )
                    ).scalars():
                        artifact.archived_at = utcnow()
                        artifact.archived_reason = "broken-chain control"
                session.commit()

    result.notes.append(
        "The broken-chain arm runs the identical generations with the identical agents and "
        "budgets, and empties the workspace between them."
    )
    return result


# ==========================================================================
# §25 capability frontier
# ==========================================================================
@dataclass
class FrontierPoint:
    difficulty: int
    n: int
    successes: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"difficulty": self.difficulty, "n": self.n, "successes": self.successes,
                "success_rate": round(self.success_rate, 4)}


@dataclass
class FrontierResult:
    threshold: float
    arms: dict[str, list[FrontierPoint]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def frontier(self, arm: str) -> int | None:
        """The largest difficulty at which the arm still clears the threshold.

        `None` when it never clears it — "never cleared the bar" is not a frontier of zero, and
        reporting the smallest tested difficulty would invent a capability the arm does not have.
        """
        points = [p for p in self.arms.get(arm, []) if p.success_rate >= self.threshold]
        return max((p.difficulty for p in points), default=None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "arms": {
                k: [p.as_dict() for p in v] for k, v in self.arms.items()
            },
            "frontier": {k: self.frontier(k) for k in self.arms},
            "notes": self.notes,
        }


def run_capability_frontier(
    session: Session,
    *,
    organization_id: uuid.UUID,
    difficulties: list[int] | None = None,
    seeds: list[int] | None = None,
    threshold: float = 0.5,
    accumulation_passes: int = 3,
    budgets: Budgets | None = None,
    config_hash: str = "frontier-v1",
) -> FrontierResult:
    """§25: hold the individual model constant, raise difficulty, compare the arms."""
    difficulties = difficulties or [6, 8, 10, 12, 14]
    seeds = seeds or [0, 1]
    budgets = budgets or DEFAULT_BUDGETS
    result = FrontierResult(threshold=threshold)

    ARMS = {
        "solo": (ExperimentArm.SOLO, False),
        "independent": (ExperimentArm.INDEPENDENT, True),
        "shared_memory": (ExperimentArm.SHARED_MEMORY, True),
        "collective": (ExperimentArm.COLLECTIVE, True),
    }
    for arm in ARMS:
        result.arms[arm] = []

    for difficulty in difficulties:
        counters = {arm: [0, 0] for arm in ARMS}
        for seed in seeds:
            device = build_device(seed=seed, era=1, n_classes=4, n_ops=difficulty)
            instances = era_instances(device)

            for arm_label, (arm, accumulate) in ARMS.items():
                workspace = Workspace(
                    organization_id=organization_id,
                    name=f"frontier {arm_label} d{difficulty}",
                    slug=f"fr-{arm_label}-{difficulty}-{uuid.uuid4().hex[:6]}",
                    environment_version=device.environment_version,
                )
                session.add(workspace)
                session.flush()

                if accumulate:
                    for pass_no in range(accumulation_passes):
                        for instance in instances:
                            run_benchmark_episode(
                                session, workspace_id=workspace.id, device=device,
                                instance=instance,
                                # `independent` accumulates episodes but transmits nothing, which
                                # is exactly §21's brute-force-repeated-sampling condition.
                                arm=(
                                    ExperimentArm.INDEPENDENT if arm is ExperimentArm.INDEPENDENT
                                    else ExperimentArm.COLLECTIVE
                                ),
                                budgets=budgets, record_findings=True,
                                config_hash=config_hash, seed=seed,
                                probe_order_seed=seed * 1000 + pass_no * 10 + instance.index,
                            )
                    session.commit()

                for instance in instances:
                    run = run_benchmark_episode(
                        session, workspace_id=workspace.id, device=device, instance=instance,
                        arm=arm, budgets=budgets, record_findings=False, is_probe=True,
                        config_hash=config_hash, seed=seed,
                        probe_order_seed=seed * 1000 + 999 + instance.index,
                    )
                    counters[arm_label][0] += 1
                    counters[arm_label][1] += int(run.succeeded)
                    for artifact in session.execute(
                        select(Artifact).where(
                            Artifact.creator_episode_id == run.episode_id,
                            Artifact.archived_at.is_(None),
                        )
                    ).scalars():
                        artifact.archived_at = utcnow()
                        artifact.archived_reason = "frontier probe output"
                session.commit()

        for arm_label, (n, successes) in counters.items():
            result.arms[arm_label].append(
                FrontierPoint(difficulty=difficulty, n=n, successes=successes)
            )

    result.notes.append(
        "Difficulty is the number of candidate operations. The model, prompts, tools and budgets "
        "are identical at every difficulty and in every arm."
    )
    return result
