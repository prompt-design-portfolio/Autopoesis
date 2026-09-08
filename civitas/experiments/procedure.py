"""§22's newcomer procedure, over any domain (Part B §5, §20, §22, §46).

M4 measured a newcomer advantage of +0.300 on the hidden-rule device. Read strictly, that is a
result about *that device*. It becomes a result about the platform only when the same procedure —
the same arms, the same founder-free discipline, the same gates, the same frozen configuration —
produces a comparable number on a task of a different shape.

So the procedure is written once here and parameterised by a domain, and the hidden-rule benchmark
calls it rather than reimplementing it. That direction matters: a second copy of §22 for the
second domain would let the two drift, and the cross-domain comparison would then be between two
procedures rather than between two domains.

What a domain has to supply is small, and every piece of it is something the core genuinely cannot
know: the tasks, the tools that act on them, the policy agent, the evaluator, the era label, and
the chance level to read a rate against.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.config import Settings, get_settings
from civitas.domain.enums import ExperimentArm, TaskStatus
from civitas.domains.base import Domain, DomainTask, check_no_leak
from civitas.experiments import credit as credit_module
from civitas.experiments.evaluation import evaluate_episode, is_readable
from civitas.persistence.models import AgentProfile, Artifact, Episode, Task, ToolDefinition
from civitas.persistence.types import utcnow
from civitas.runtime.budgets import Budgets
from civitas.runtime.episode import EpisodeRunner, EpisodeSpec
from civitas.runtime.providers.base import Provider
from civitas.runtime.tools.base import ToolRegistry
from civitas.runtime.tools.builtin import default_registry

#: The budget used by every domain unless a caller narrows it. Measured, not chosen — see
#: `civitas.experiments.runner.DEFAULT_BUDGETS` for the sweep that fixed it at seven.
DEFAULT_BUDGETS = Budgets(
    tokens=200_000, context=32_000, tool_calls=7, cost_usd=1.0, wall_clock_s=120,
    max_turns=12, max_no_progress_turns=4,
)

SYSTEM_PROMPT = (
    "You are a bounded research agent in a persistent collective. Your episode ends when your "
    "budget runs out or you submit a result. Nothing you think survives your episode; only what "
    "you write to the collective knowledge base does. Earlier agents may already have established "
    "part of what you need — search before you probe."
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


def ensure_tool_definition(
    session: Session, *, workspace_id: uuid.UUID, name: str, description: str = ""
) -> ToolDefinition:
    """Register a domain tool so its runs are attributable (§17)."""
    existing = session.execute(
        select(ToolDefinition).where(
            ToolDefinition.workspace_id == workspace_id, ToolDefinition.name == name
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    definition = ToolDefinition(
        workspace_id=workspace_id, name=name,
        description=description or f"domain tool {name}", kind="builtin", is_builtin=True,
    )
    session.add(definition)
    session.flush()
    return definition


def create_domain_task(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    spec: DomainTask,
    project_id: uuid.UUID | None = None,
) -> Task:
    """Persist a generated task, leak-checked on the way in.

    The check runs here rather than only in the test suite because this is the one funnel every
    benchmark task passes through. A domain whose description gave its answer away would otherwise
    produce a perfectly clean-looking benchmark measuring nothing (§47).
    """
    check_no_leak(spec)
    task = Task(
        workspace_id=workspace_id,
        project_id=project_id,
        title=spec.title,
        description=spec.description,
        task_family=spec.task_family,
        difficulty=spec.difficulty,
        status=TaskStatus.READY,
        evaluator_spec=spec.evaluator_spec,
        meta=dict(spec.meta),
    )
    session.add(task)
    session.flush()
    return task


def run_domain_episode(
    session: Session,
    *,
    domain: Domain,
    spec: DomainTask,
    workspace_id: uuid.UUID,
    arm: ExperimentArm,
    task: Task | None = None,
    provider: Provider | None = None,
    provider_name: str = "policy",
    model_name: str = "policy-v1",
    budgets: Budgets | None = None,
    record_findings: bool = True,
    is_probe: bool = False,
    probe_order_seed: int = 0,
    retrieval_options: dict[str, Any] | None = None,
    experiment_run_id: uuid.UUID | None = None,
    frozen_as_of: Any = None,
    config_hash: str = "",
    seed: int | None = None,
    settings: Settings | None = None,
    assign_credit: bool = True,
) -> Any:
    """One episode: run it, evaluate it externally, propagate credit.

    Identical in every respect except the domain to what M4 ran. The evaluator is chosen by the
    task's `evaluator_spec["kind"]`, so a domain that judges by executing code and one that judges
    by string comparison reach it by the same path.
    """
    settings = settings or get_settings()
    task = task or create_domain_task(session, workspace_id=workspace_id, spec=spec)
    profile = ensure_profile(session, workspace_id)

    agent = provider or domain.build_agent(
        spec, record_findings=record_findings, probe_order_seed=probe_order_seed
    )

    tools: ToolRegistry = default_registry()
    for tool in domain.build_tools(spec):
        definition = ensure_tool_definition(
            session, workspace_id=workspace_id, name=tool.name, description=tool.description
        )
        # The definition id is how a tool run is attributable to a registered tool (§17). Set by
        # attribute rather than by constructor so a domain's tool does not have to know it will be
        # used inside a benchmark.
        if hasattr(tool, "_definition_id"):
            tool._definition_id = definition.id
        tools.add(tool)

    episode_spec = EpisodeSpec(
        workspace_id=workspace_id,
        agent_profile_id=profile.id,
        provider_name=provider_name,
        model_name=model_name,
        model_version="policy-v1",
        system_prompt=SYSTEM_PROMPT,
        system_prompt_version=domain.system_prompt_version(),
        budgets=budgets or DEFAULT_BUDGETS,
        experiment_arm=arm,
        retrieval_policy_version=(retrieval_options or {}).get("version", "retrieval/2.0-hybrid"),
        retrieval_options=dict(retrieval_options or {}),
        task_id=task.id,
        project_id=task.project_id,
        experiment_run_id=experiment_run_id,
        # The era. An artifact written now is applicable to this environment and visibly
        # inapplicable to the next one (§15) — the mechanism `π` demands.
        environment_version=spec.environment_version,
        config_hash=config_hash,
        seed=seed,
        is_benchmark_probe=is_probe,
        frozen_as_of=frozen_as_of,
    )

    outcome = EpisodeRunner(session, provider=agent, tools=tools).run(episode_spec)
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

    return RunResult(
        episode_id=episode.id,
        succeeded=evaluation.succeeded,
        readable=is_readable(evaluation),
        tool_calls=episode.tool_calls_used,
        probes=_count_probes(session, episode.id),
        tokens=episode.tokens_used,
        artifacts_created=episode.artifacts_created,
        artifacts_read=episode.artifacts_read,
        duplicate_failures=episode.duplicate_failures,
        termination_reason=(
            episode.termination_reason.value if episode.termination_reason else "unknown"
        ),
        submitted=outcome.submitted_answer,
        used_stale=_used_stale_artifact(session, episode.id, spec.environment_version),
        wall_time_s=episode.duration_s or 0.0,
        cost_usd=episode.cost_usd,
    )


def archive_episode_artifacts(session: Session, episode_id: uuid.UUID) -> int:
    """Remove a probe episode's own output from the collective it was measured against.

    Archived, not deleted (Part A §A1.2): the contribution stays on the record and is simply
    excluded from retrieval, so a reader can still see what each probe produced.
    """
    count = 0
    for artifact in session.execute(
        select(Artifact).where(
            Artifact.creator_episode_id == episode_id, Artifact.archived_at.is_(None)
        )
    ).scalars():
        artifact.archived_at = utcnow()
        artifact.archived_reason = "benchmark probe output, excluded from the measured environment"
        count += 1
    session.flush()
    return count


def reset_memory(session: Session, workspace_id: uuid.UUID) -> int:
    """§21 `memory_reset`: archive the collective state the arm is meant to lose.

    The duplicate-failure index is collective state too. Archiving the artifacts while leaving the
    index answering would let the reset arm keep the one thing it is meant to lose.
    """
    from civitas.knowledge.duplicate import retire_for_workspace

    count = 0
    for artifact in session.execute(
        select(Artifact).where(
            Artifact.workspace_id == workspace_id, Artifact.archived_at.is_(None)
        )
    ).scalars():
        artifact.archived_at = utcnow()
        artifact.archived_reason = "memory_reset arm"
        count += 1
    retire_for_workspace(session, workspace_id)
    session.flush()
    return count


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
