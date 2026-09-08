"""The distributed-knowledge benchmark (Part B §23).

Three populations, one problem:

* **partition A** probes class→family and records what it finds;
* **partition B** probes family→operation and records what it finds;
* **the integrator** has *neither probe* and must combine what A and B wrote down.

The integrator is the measurement. Its success rate is compared across arms, and the controls are
the same ones §22 uses, because the same confound applies: an integrator that succeeded because a
store existed rather than because information passed would be indistinguishable from one that
performed distributed cognition.

| arm | what the integrator sees |
|---|---|
| `both_partitions` | everything A and B recorded — the claim |
| `partition_a_only` | only A's findings — necessary but not sufficient |
| `partition_b_only` | only B's findings — necessary but not sufficient |
| `no_partitions` | nothing — the floor |
| `scrambled` | the same quantity of artifacts, relevance destroyed |

The single-partition arms are the sharpest controls available here. They hold *quantity of relevant
knowledge* far higher than the floor while still being provably insufficient, so an integrator that
succeeds on one of them is not combining anything.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import ExperimentArm, TaskStatus
from civitas.experiments.evaluation import evaluate_episode
from civitas.experiments.runner import SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION, ensure_profile
from civitas.experiments.tasks.distributed import (
    SplitDevice,
    SplitTaskInstance,
    assert_unsolvable_alone,
    build_split_device,
    partition_instances,
)
from civitas.experiments.tasks.split_tools import FAMILY_FINDING, TABLE_FINDING, registry_for
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

FAMILY_RE = re.compile(
    r"SPLIT-FAMILY\s+env=(?P<env>\S+)\s+class=(?P<cls>\S+)\s+family=(?P<fam>\S+)"
)
TABLE_RE = re.compile(r"SPLIT-TABLE\s+env=(?P<env>\S+)\s+family=(?P<fam>\S+)\s+op=(?P<op>\S+)")

BUDGETS = Budgets(
    tokens=300_000, context=32_000, tool_calls=14, cost_usd=1.0, wall_clock_s=180,
    max_turns=20, max_no_progress_turns=5,
)


class PartitionPolicy(Provider):
    """A deterministic agent for one partition (Part B §20).

    Probes what its partition can probe, records the finding, and submits. Frozen and
    deterministic, so a difference between arms is caused by what the arms let the *integrator*
    see — not by the producers varying.
    """

    name = "partition-policy"

    def __init__(self, *, device: SplitDevice, instance: SplitTaskInstance):
        self._device = device
        self._instance = instance
        self._env = device.environment_version

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(name=model or "partition-v1", provider=self.name,
                         context_window=32_000, supports_tools=True, is_deterministic=True,
                         version="partition-v1")

    def _history(self, request: CompletionRequest) -> dict[str, Any]:
        state: dict[str, Any] = {
            "family_tested": set(), "family_found": None,
            "table_tested": set(), "table_found": {}, "recorded": False,
        }
        pending: list[ToolCall] = []
        for message in request.messages:
            if message.role is Role.ASSISTANT:
                pending = list(message.tool_calls)
                continue
            if message.role is not Role.TOOL:
                continue
            call = next((c for c in pending if c.id == message.tool_call_id), None)
            name = message.name or (call.name if call else "")
            if name == "probe_family" and call is not None:
                family = str(call.arguments.get("family", "")).lower()
                state["family_tested"].add(family)
                if " IS in family" in message.content:
                    state["family_found"] = family
            elif name == "probe_table" and call is not None:
                family = str(call.arguments.get("family", "")).lower()
                operation = str(call.arguments.get("operation", "")).lower()
                state["table_tested"].add((family, operation))
                if "ACCEPTS" in message.content:
                    state["table_found"][family] = operation
            elif name == "create_artifact":
                state["recorded"] = True
        return state

    def complete(self, request: CompletionRequest) -> Completion:
        state = self._history(request)
        available = {t.name for t in request.tools}
        step = len(request.messages)
        device, instance = self._device, self._instance

        if instance.partition == "a":
            if state["family_found"] is None:
                remaining = [f for f in device.families if f not in state["family_tested"]]
                if remaining and "probe_family" in available:
                    return self._call(request, ToolCall(f"c{step}", "probe_family", {
                        "input_class": instance.input_class, "family": remaining[0],
                    }), f"Testing family {remaining[0]}.")
                return self._call(request, ToolCall(f"c{step}", "submit_result", {
                    "answer": "unknown", "confidence": 0.1}), "Out of families to test.")
            if not state["recorded"] and "create_artifact" in available:
                line = FAMILY_FINDING.format(
                    env=self._env, cls=instance.input_class, fam=state["family_found"]
                )
                return self._call(request, ToolCall(f"c{step}", "create_artifact", {
                    "type": "evidence",
                    "title": f"Input class {instance.input_class} is in family "
                             f"{state['family_found']}",
                    "body": line, "confidence": 0.9,
                }), "Recording the family for the other populations.")
            return self._call(request, ToolCall(f"c{step}", "submit_result", {
                "answer": state["family_found"], "confidence": 0.9}), "Submitting.")

        # partition b: establish the whole family -> operation table
        missing = [f for f in device.families if f not in state["table_found"]]
        if missing and "probe_table" in available:
            family = missing[0]
            untested = [
                o for o in device.operations if (family, o) not in state["table_tested"]
            ]
            if untested:
                return self._call(request, ToolCall(f"c{step}", "probe_table", {
                    "family": family, "operation": untested[0],
                }), f"Testing {family}/{untested[0]}.")
        if not state["recorded"] and state["table_found"] and "create_artifact" in available:
            lines = [
                TABLE_FINDING.format(env=self._env, fam=f, op=o)
                for f, o in sorted(state["table_found"].items())
            ]
            return self._call(request, ToolCall(f"c{step}", "create_artifact", {
                "type": "evidence",
                "title": "Family to operation table",
                "body": "\n".join(lines), "confidence": 0.9,
            }), "Recording the table for the other populations.")
        answer = ",".join(f"{f}={o}" for f, o in sorted(state["table_found"].items()))
        return self._call(request, ToolCall(f"c{step}", "submit_result", {
            "answer": answer or "unknown", "confidence": 0.9 if answer else 0.1}), "Submitting.")

    def _call(self, request: CompletionRequest, call: ToolCall, text: str) -> Completion:
        return Completion(
            text=text, tool_calls=(call,),
            usage=Usage(prompt_tokens=self.count_request_tokens(request),
                        completion_tokens=self.count_tokens(text)),
            stop_reason="tool_use", model=request.model, model_version="partition-v1",
        )


class IntegratorPolicy(Provider):
    """The measurement. Has no probe; can only read and combine (§23).

    It searches, reads what it finds, and composes `family_of[class]` with `table[family]`. If
    either half is missing it submits an unverified guess — which is exactly what should happen,
    and is why the single-partition arms are meaningful controls rather than degenerate ones.
    """

    name = "integrator-policy"

    def __init__(self, *, device: SplitDevice, input_class: str, max_reads: int = 6):
        self._device = device
        self._input_class = input_class
        self._env = device.environment_version
        self._max_reads = max_reads

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(name=model or "integrator-v1", provider=self.name,
                         context_window=32_000, supports_tools=True, is_deterministic=True,
                         version="integrator-v1")

    def complete(self, request: CompletionRequest) -> Completion:
        from civitas.experiments.policy_agent import ID_RE

        searched = False
        ranked: list[str] = []
        read: set[str] = set()
        family: str | None = None
        table: dict[str, str] = {}
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
                for match in FAMILY_RE.finditer(message.content or ""):
                    # The environment check is the `π` discipline: a finding from another era
                    # describes a device that no longer exists.
                    if match.group("env") == self._env and \
                            match.group("cls").lower() == self._input_class:
                        family = match.group("fam").lower()
                for match in TABLE_RE.finditer(message.content or ""):
                    if match.group("env") == self._env:
                        table[match.group("fam").lower()] = match.group("op").lower()

        available = {t.name for t in request.tools}
        step = len(request.messages)

        if family is not None and family in table:
            return self._call(request, ToolCall(f"c{step}", "submit_result", {
                "answer": table[family], "confidence": 0.9,
                "reasoning": f"{self._input_class} is in family {family}, which accepts "
                             f"{table[family]}",
            }), "Combining both halves.")

        if not searched and "search_knowledge" in available:
            return self._call(request, ToolCall(f"c{step}", "search_knowledge", {
                "query": f"input class {self._input_class} family operation table",
                "limit": 12,
            }), "Looking for what the other populations recorded.")

        unread = [i for i in ranked if i not in read]
        if unread and len(read) < self._max_reads and "read_artifact" in available:
            return self._call(request, ToolCall(f"c{step}", "read_artifact", {
                "artifact_id": unread[0]}), f"Reading result {len(read) + 1}.")

        missing = []
        if family is None:
            missing.append("the class's family")
        if not table:
            missing.append("the family/operation table")
        return self._call(request, ToolCall(f"c{step}", "submit_result", {
            "answer": self._device.operations[0], "confidence": 0.05,
            "reasoning": f"could not find {' and '.join(missing) or 'both halves'}",
        }), "Cannot combine; submitting an unverified guess.")

    def _call(self, request: CompletionRequest, call: ToolCall, text: str) -> Completion:
        return Completion(
            text=text, tool_calls=(call,),
            usage=Usage(prompt_tokens=self.count_request_tokens(request),
                        completion_tokens=self.count_tokens(text)),
            stop_reason="tool_use", model=request.model, model_version="integrator-v1",
        )


@dataclass
class ArmResult:
    arm: str
    n: int = 0
    successes: int = 0
    mean_artifacts_read: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "n": self.n, "successes": self.successes,
                "success_rate": round(self.success_rate, 4),
                "mean_artifacts_read": round(self.mean_artifacts_read, 2)}


@dataclass
class DistributedResult:
    unsolvable_proof: dict[str, Any] = field(default_factory=dict)
    arms: dict[str, ArmResult] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unsolvable_proof": self.unsolvable_proof,
            "arms": {k: v.as_dict() for k, v in self.arms.items()},
            "derived": self.derived(),
            "notes": self.notes,
        }

    def derived(self) -> dict[str, Any]:
        def rate(name: str) -> float | None:
            arm = self.arms.get(name)
            return arm.success_rate if arm else None

        both, floor = rate("both_partitions"), rate("no_partitions")
        a_only, b_only = rate("partition_a_only"), rate("partition_b_only")
        best_single = (
            max(v for v in (a_only, b_only) if v is not None)
            if any(v is not None for v in (a_only, b_only))
            else None
        )
        return {
            "distributed_advantage": (
                None if both is None or floor is None else round(both - floor, 4)
            ),
            #: The number that matters. Either partition alone is provably insufficient, so an
            #: integrator that beats *both* single-partition arms is combining information rather
            #: than exploiting either half.
            "gain_over_best_single_partition": (
                None if both is None or best_single is None else round(both - best_single, 4)
            ),
            "gain_over_scrambled": (
                None if both is None or rate("scrambled") is None
                else round(both - rate("scrambled"), 4)
            ),
        }


def _run_partition_episodes(
    session: Session, *, workspace_id: uuid.UUID, device: SplitDevice,
    instances: list[SplitTaskInstance], config_hash: str,
) -> None:
    profile = ensure_profile(session, workspace_id, "partition")
    for instance in instances:
        task = Task(
            workspace_id=workspace_id, title=instance.title, description=instance.description,
            task_family="distributed", status=TaskStatus.READY,
            evaluator_spec=instance.evaluator_spec,
            information_partition=instance.partition,
        )
        session.add(task)
        session.flush()
        spec = EpisodeSpec(
            workspace_id=workspace_id, agent_profile_id=profile.id,
            provider_name="partition-policy", model_name="partition-v1",
            model_version="partition-v1", system_prompt=SYSTEM_PROMPT,
            system_prompt_version=SYSTEM_PROMPT_VERSION, budgets=BUDGETS,
            experiment_arm=ExperimentArm.COLLECTIVE, task_id=task.id,
            environment_version=device.environment_version, config_hash=config_hash,
        )
        outcome = EpisodeRunner(
            session, provider=PartitionPolicy(device=device, instance=instance),
            tools=registry_for(instance.partition, device),
        ).run(spec)
        episode = session.get(Episode, outcome.episode_id)
        evaluate_episode(session, episode=episode, outcome=outcome, task=task,
                         config_hash=config_hash)
    session.flush()


def run_distributed_benchmark(
    session: Session,
    *,
    organization_id: uuid.UUID,
    seeds: list[int] | None = None,
    config_hash: str = "distributed-v1",
) -> DistributedResult:
    """§23, with the controls that make a positive result interpretable."""
    seeds = seeds or [0, 1, 2]
    result = DistributedResult()

    ARMS = {
        "both_partitions": ("a", "b"),
        "partition_a_only": ("a",),
        "partition_b_only": ("b",),
        "no_partitions": (),
        "scrambled": ("a", "b"),
    }
    for arm in ARMS:
        result.arms[arm] = ArmResult(arm=arm)

    for seed in seeds:
        device = build_split_device(seed=seed)
        result.unsolvable_proof = assert_unsolvable_alone(device)
        instances = partition_instances(device)

        for arm_label, populations in ARMS.items():
            workspace = Workspace(
                organization_id=organization_id,
                name=f"distributed {arm_label}",
                slug=f"dist-{arm_label}-{uuid.uuid4().hex[:8]}",
                environment_version=device.environment_version,
            )
            session.add(workspace)
            session.flush()

            for population in populations:
                _run_partition_episodes(
                    session, workspace_id=workspace.id, device=device,
                    instances=instances[population], config_hash=config_hash,
                )
            session.commit()

            profile = ensure_profile(session, workspace.id, "integrator")
            reads: list[int] = []
            for instance in instances["integrator"]:
                task = Task(
                    workspace_id=workspace.id, title=instance.title,
                    description=instance.description, task_family="distributed",
                    status=TaskStatus.READY, evaluator_spec=instance.evaluator_spec,
                    information_partition="integrator",
                )
                session.add(task)
                session.flush()
                spec = EpisodeSpec(
                    workspace_id=workspace.id, agent_profile_id=profile.id,
                    provider_name="integrator-policy", model_name="integrator-v1",
                    model_version="integrator-v1", system_prompt=SYSTEM_PROMPT,
                    system_prompt_version=SYSTEM_PROMPT_VERSION, budgets=BUDGETS,
                    experiment_arm=(
                        ExperimentArm.COLLECTIVE_SCRAMBLED if arm_label == "scrambled"
                        else ExperimentArm.COLLECTIVE
                    ),
                    task_id=task.id, environment_version=device.environment_version,
                    config_hash=config_hash, is_benchmark_probe=True, seed=seed,
                )
                outcome = EpisodeRunner(
                    session,
                    provider=IntegratorPolicy(device=device, input_class=instance.input_class),
                    tools=registry_for("integrator", device),
                ).run(spec)
                episode = session.get(Episode, outcome.episode_id)
                evaluation = evaluate_episode(
                    session, episode=episode, outcome=outcome, task=task,
                    config_hash=config_hash,
                )
                arm_result = result.arms[arm_label]
                arm_result.n += 1
                arm_result.successes += int(evaluation.succeeded)
                reads.append(episode.artifacts_read)

                # The integrator's own output must not inform the next integrator.
                for artifact in session.execute(
                    select(Artifact).where(Artifact.creator_episode_id == episode.id)
                ).scalars():
                    artifact.archived_at = utcnow()
                    artifact.archived_reason = "integrator probe output"
                session.flush()
            if reads:
                arm = result.arms[arm_label]
                arm.mean_artifacts_read = (
                    (arm.mean_artifacts_read * (arm.n - len(reads)) + sum(reads)) / arm.n
                )
            session.commit()

    result.notes.append(
        "Either partition alone is provably insufficient (see unsolvable_proof), so the "
        "single-partition arms hold relevant-knowledge quantity high while remaining unable to "
        "determine the answer."
    )
    return result
