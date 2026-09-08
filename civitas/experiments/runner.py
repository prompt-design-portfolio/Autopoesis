"""Running one episode of the hidden-rule benchmark, end to end (Part B §20, §22, §46, §47).

This is the seam where every M4 mechanism meets: a frozen configuration, an enforced arm, a
bounded episode, an external evaluation, and credit propagated backward over what was actually
read. Keeping it in one function means the benchmark and the Colab notebook drive *the same* path
the API does, rather than a parallel implementation that could drift from it.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from civitas.config import Settings
from civitas.domain.enums import ExperimentArm, TaskStatus
from civitas.domains.base import DomainTask
from civitas.experiments.procedure import RunResult
from civitas.experiments.tasks.hidden_rule import DeviceSpec, TaskInstance
from civitas.persistence.models import AgentProfile, Task
from civitas.runtime.budgets import Budgets
from civitas.runtime.providers.base import Provider

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
    retrieval_options: dict[str, Any] | None = None,
    experiment_run_id: uuid.UUID | None = None,
    frozen_as_of: Any = None,
    config_hash: str = "",
    seed: int | None = None,
    settings: Settings | None = None,
    assign_credit: bool = True,
) -> RunResult:
    """One episode of the hidden-rule benchmark.

    A thin adapter over `procedure.run_domain_episode`: it turns a `(device, instance)` pair into
    the domain task the general path takes, and changes nothing else. Keeping one implementation
    is what lets the cross-domain comparison be a comparison between domains rather than between
    two copies of §22 that have drifted apart.
    """
    from civitas.domains import get_domain
    from civitas.experiments.procedure import create_domain_task, run_domain_episode

    spec = _domain_task_for(instance)
    task = task or create_domain_task(session, workspace_id=workspace_id, spec=spec)
    return run_domain_episode(
        session,
        domain=get_domain("hidden_rule"),
        spec=spec,
        workspace_id=workspace_id,
        arm=arm,
        task=task,
        provider=provider,
        provider_name=provider_name,
        model_name=model_name,
        budgets=budgets or DEFAULT_BUDGETS,
        record_findings=record_findings,
        is_probe=is_probe,
        probe_order_seed=probe_order_seed,
        retrieval_options=retrieval_options,
        experiment_run_id=experiment_run_id,
        frozen_as_of=frozen_as_of,
        config_hash=config_hash,
        seed=seed,
        settings=settings,
        assign_credit=assign_credit,
    )


def _domain_task_for(instance: TaskInstance) -> DomainTask:
    """The instance as the domain layer describes it.

    Built from the instance rather than regenerated from `(seed, era)` so that a caller which
    constructed an instance directly — the retrieval, scheduler and cumulative benchmarks all do —
    gets exactly the task it built.
    """
    from civitas.domains.base import DomainTask

    device = instance.device
    return DomainTask(
        title=instance.title,
        description=instance.description,
        evaluator_spec=instance.evaluator_spec,
        task_family="hidden_rule",
        index=instance.index,
        environment_version=device.environment_version,
        chance_level=1.0 / max(1, len(device.operations)),
        difficulty=1.0 / max(1, len(device.operations)),
        candidate_terms=tuple(device.operations),
        meta={
            "device_seed": device.seed, "era": device.era,
            "input_class": instance.input_class,
            "n_classes": len(device.classes), "n_ops": len(device.operations),
            "environment_version": device.environment_version,
        },
    )



