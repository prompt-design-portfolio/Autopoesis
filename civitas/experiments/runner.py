"""Running one episode of the hidden-rule benchmark, end to end (Part B §20, §22, §46, §47).

This is the seam where every M4 mechanism meets: a frozen configuration, an enforced arm, a
bounded episode, an external evaluation, and credit propagated backward over what was actually
read. Keeping it in one function means the benchmark and the Colab notebook drive *the same* path
the API does, rather than a parallel implementation that could drift from it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from civitas.config import Settings, get_settings
from civitas.domain.enums import ExperimentArm, TaskStatus
from civitas.experiments import credit as credit_module
from civitas.experiments.evaluation import evaluate_episode, is_readable
from civitas.experiments.policy_agent import PolicyAgentProvider
from civitas.experiments.tasks.hidden_rule import DeviceSpec, TaskInstance
from civitas.experiments.tasks.probe_tool import ProbeDeviceTool, ensure_definition
from civitas.persistence.models import AgentProfile, Episode, Task
from civitas.persistence.types import utcnow
from civitas.runtime.budgets import Budgets
from civitas.runtime.episode import EpisodeRunner, EpisodeSpec
from civitas.runtime.providers.base import Provider
from civitas.runtime.tools.base import ToolRegistry
from civitas.runtime.tools.builtin import default_registry

SYSTEM_PROMPT = (
    "You are a bounded research agent in a persistent collective. Your episode ends when your "
    "budget runs out or you submit a result. Nothing you think survives your episode; only what "
    "you write to the collective knowledge base does. Earlier agents may already have established "
    "part of what you need — search before you probe."
)
SYSTEM_PROMPT_VERSION = "hidden-rule/1.0"

#: Default budget — the difficulty setting, and therefore reported in the manifest rather than
#: left implicit.
#:
#: `tool_calls = 7` was **measured, not chosen**. A sweep over 5, 6, 7 and 9 (recorded in
#: docs/milestones/M04.md) shows the newcomer advantage is 0.000 at 5 — too tight for a fresh agent
#: to exploit what it reads — rises to +0.333 at 7, and falls back to +0.111 at 9 as the baseline
#: approaches ceiling. Seven is where the instrument discriminates. Reporting the sweep rather than
#: the single chosen value is the point: a benchmark tuned until it produced an advantage, with the
#: tuning unreported, would be indistinguishable from one that found one.
DEFAULT_BUDGETS = Budgets(
    tokens=200_000, context=32_000, tool_calls=7, cost_usd=1.0, wall_clock_s=120,
    max_turns=12, max_no_progress_turns=4,
)


@dataclass
class RunResult:
    episode_id: uuid.UUID
    succeeded: bool
    readable: bool
    tool_calls: int
    probes: int
    tokens: int
    artifacts_created: int
    artifacts_read: int
    duplicate_failures: int
    termination_reason: str
    submitted: str | None
    #: True when the episode believed a finding that named a different environment version.
    #: Part B §48's stale-artifact usage, and the sharpest single sign that a collective is
    #: helping by recall rather than by knowledge.
    used_stale: bool = False
    wall_time_s: float = 0.0
    cost_usd: float = 0.0


def ensure_profile(session: Session, workspace_id: uuid.UUID, name: str = "prober") -> AgentProfile:
    from sqlalchemy import select

    existing = session.execute(
        select(AgentProfile).where(
            AgentProfile.workspace_id == workspace_id, AgentProfile.name == name
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    profile = AgentProfile(workspace_id=workspace_id, name=name, role="explorer")
    session.add(profile)
    session.flush()
    return profile


def create_task(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    instance: TaskInstance,
    project_id: uuid.UUID | None = None,
) -> Task:
    task = Task(
        workspace_id=workspace_id,
        project_id=project_id,
        title=instance.title,
        description=instance.description,
        task_family="hidden_rule",
        difficulty=1.0 / max(1, len(instance.device.operations)),
        status=TaskStatus.READY,
        evaluator_spec=instance.evaluator_spec,
    )
    session.add(task)
    session.flush()
    return task


def run_benchmark_episode(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    device: DeviceSpec,
    instance: TaskInstance,
    arm: ExperimentArm,
    task: Task | None = None,
    provider: Provider | None = None,
    provider_name: str = "policy",
    model_name: str = "policy-v1",
    budgets: Budgets | None = None,
    record_findings: bool = True,
    is_probe: bool = False,
    probe_order_seed: int = 0,
    experiment_run_id: uuid.UUID | None = None,
    frozen_as_of: Any = None,
    config_hash: str = "",
    seed: int | None = None,
    settings: Settings | None = None,
    assign_credit: bool = True,
) -> RunResult:
    """One episode: run it, evaluate it externally, propagate credit."""
    settings = settings or get_settings()
    task = task or create_task(session, workspace_id=workspace_id, instance=instance)
    profile = ensure_profile(session, workspace_id)
    definition = ensure_definition(session, workspace_id=workspace_id)

    agent = provider or PolicyAgentProvider(
        input_class=instance.input_class,
        operations=list(device.operations),
        environment_version=device.environment_version,
        record_findings=record_findings,
        probe_order_seed=probe_order_seed,
    )

    tools: ToolRegistry = default_registry()
    tools.add(ProbeDeviceTool(device, definition_id=definition.id))

    spec = EpisodeSpec(
        workspace_id=workspace_id,
        agent_profile_id=profile.id,
        provider_name=provider_name,
        model_name=model_name,
        model_version="policy-v1",
        system_prompt=SYSTEM_PROMPT,
        system_prompt_version=SYSTEM_PROMPT_VERSION,
        budgets=budgets or DEFAULT_BUDGETS,
        experiment_arm=arm,
        retrieval_policy_version="retrieval/1.0-lexical",
        task_id=task.id,
        project_id=task.project_id,
        experiment_run_id=experiment_run_id,
        # The device's era. An artifact written now is applicable to this device and visibly
        # inapplicable to the next one (§15) — the mechanism `π` demands.
        environment_version=device.environment_version,
        config_hash=config_hash,
        seed=seed,
        is_benchmark_probe=is_probe,
        frozen_as_of=frozen_as_of,
    )

    outcome = EpisodeRunner(session, provider=agent, tools=tools).run(spec)
    episode = session.get(Episode, outcome.episode_id)

    evaluation = evaluate_episode(
        session, episode=episode, outcome=outcome, task=task, config_hash=config_hash
    )
    if assign_credit:
        credit_module.assign_credit(session, evaluation, config_hash=config_hash)

    task.attempts += 1
    if evaluation.succeeded:
        task.status = TaskStatus.COMPLETED
        task.completed_at = utcnow()

    probes = _count_probes(session, episode.id)

    return RunResult(
        episode_id=episode.id,
        succeeded=evaluation.succeeded,
        readable=is_readable(evaluation),
        tool_calls=episode.tool_calls_used,
        probes=probes,
        tokens=episode.tokens_used,
        artifacts_created=episode.artifacts_created,
        artifacts_read=episode.artifacts_read,
        duplicate_failures=episode.duplicate_failures,
        termination_reason=(
            episode.termination_reason.value if episode.termination_reason else "unknown"
        ),
        submitted=outcome.submitted_answer,
        used_stale=_used_stale_artifact(session, episode.id, device.environment_version),
        wall_time_s=episode.duration_s or 0.0,
        cost_usd=episode.cost_usd,
    )


def _count_probes(session: Session, episode_id: uuid.UUID) -> int:
    from sqlalchemy import func, select

    from civitas.persistence.models import ToolRun

    return int(
        session.execute(
            select(func.count(ToolRun.id)).where(ToolRun.episode_id == episode_id)
        ).scalar_one()
        or 0
    )


def _used_stale_artifact(session: Session, episode_id: uuid.UUID, env: str) -> bool:
    """Did the episode read an artifact belonging to a different era? (Part B §15, §48)

    Read from `ArtifactUsage`, not from retrieval: being *shown* a stale artifact is a retrieval
    quality question, whereas having *read* one is what can produce a confident wrong answer.
    """
    from sqlalchemy import select

    from civitas.persistence.models import Artifact, ArtifactUsage

    rows = session.execute(
        select(Artifact.environment_version)
        .join(ArtifactUsage, ArtifactUsage.artifact_id == Artifact.id)
        .where(ArtifactUsage.episode_id == episode_id)
    ).scalars()
    return any(version and version != env for version in rows)
