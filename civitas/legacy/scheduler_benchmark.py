"""Does allocation strategy matter? (Part B §27, §48)

§27 requires nine allocation methods and says the scheduler should *avoid wasting agents on
identical duplicated work unless intentionally sampling*. This measures whether that avoidance is
worth anything, by giving each strategy the **same fixed episode budget** over the same pool of
tasks and counting how many distinct tasks get solved.

The comparison is between strategies, not between a scheduler and no scheduler, because "no
scheduler" is not an option a running system has. `random` is the honest floor: it is what
allocation looks like when nothing is being reasoned about, and it is also the arm §27 explicitly
exempts from the duplication penalty.

Everything except the strategy is held fixed: same device, same seeds, same agents, same budgets,
same workspace shape. The only thing that varies is which task the next episode is spent on.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import AllocationStrategy, ExperimentArm, TaskStatus
from civitas.legacy.runner import DEFAULT_BUDGETS, create_task, run_benchmark_episode
from civitas.legacy.tasks.hidden_rule import build_device, era_instances
from civitas.persistence.models import AgentProfile, Task, Workspace
from civitas.runtime.budgets import Budgets
from civitas.scheduler import allocation


@dataclass
class StrategyResult:
    strategy: str
    episodes: int
    tasks: int
    solved: int
    #: Episodes spent on a task that was already solved. This is the waste §27 names.
    wasted: int
    coverage: float = 0.0
    waste_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class SchedulerBenchmarkResult:
    results: dict[str, StrategyResult] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": {k: v.as_dict() for k, v in self.results.items()},
            "notes": self.notes,
        }

    def derived(self) -> dict[str, Any]:
        random_arm = self.results.get(AllocationStrategy.RANDOM.value)
        if random_arm is None:
            return {"reason": "the random floor is required for the comparison"}
        out: dict[str, Any] = {}
        for name, result in self.results.items():
            if name == AllocationStrategy.RANDOM.value:
                continue
            out[name] = {
                "coverage_gain_over_random": round(
                    result.coverage - random_arm.coverage, 4
                ),
                "waste_reduction": round(random_arm.waste_rate - result.waste_rate, 4),
            }
        return out


def run_scheduler_benchmark(
    session: Session,
    *,
    organization_id: uuid.UUID,
    strategies: list[AllocationStrategy] | None = None,
    episode_budget: int = 12,
    n_classes: int = 6,
    seed: int = 0,
    budgets: Budgets | None = None,
) -> SchedulerBenchmarkResult:
    """One fixed episode budget per strategy, over an identical task pool."""
    strategies = strategies or [
        AllocationStrategy.RANDOM,
        AllocationStrategy.PRIORITY,
        AllocationStrategy.UNCERTAINTY,
        AllocationStrategy.INFORMATION_GAIN,
    ]
    budgets = budgets or DEFAULT_BUDGETS
    device = build_device(seed=seed, era=1, n_classes=n_classes, n_ops=10)
    instances = era_instances(device)
    result = SchedulerBenchmarkResult()

    for strategy in strategies:
        workspace = Workspace(
            organization_id=organization_id,
            name=f"scheduler {strategy.value}",
            slug=f"sched-{strategy.value}-{uuid.uuid4().hex[:8]}",
            environment_version=device.environment_version,
        )
        session.add(workspace)
        session.flush()
        profile = AgentProfile(workspace_id=workspace.id, name="prober")
        session.add(profile)
        session.flush()

        tasks: dict[uuid.UUID, Any] = {}
        for instance in instances:
            task = create_task(session, workspace_id=workspace.id, instance=instance)
            tasks[task.id] = instance
        session.flush()

        solved_tasks: set[uuid.UUID] = set()
        wasted = 0

        for step in range(episode_budget):
            assignments = allocation.allocate(
                session, workspace_id=workspace.id, strategy=strategy, count=1, seed=seed + step
            )
            if not assignments:
                # Every task is assigned or exhausted. Returning them to READY is what a real
                # scheduler does when an assignment is consumed; without it the pool starves and
                # the strategies become indistinguishable for the wrong reason.
                _release(session, workspace.id)
                assignments = allocation.allocate(
                    session, workspace_id=workspace.id, strategy=strategy, count=1,
                    seed=seed + step,
                )
                if not assignments:
                    break

            assignment = assignments[0]
            task = session.get(Task, assignment.task_id)
            instance = tasks[task.id]
            already_solved = task.id in solved_tasks

            run = run_benchmark_episode(
                session, workspace_id=workspace.id, device=device, instance=instance,
                task=task, arm=ExperimentArm.COLLECTIVE, budgets=budgets,
                record_findings=True, probe_order_seed=seed * 100 + step,
            )
            if already_solved:
                wasted += 1
            if run.succeeded:
                solved_tasks.add(task.id)
            else:
                # A failed attempt returns the task to the pool; otherwise one failure removes it
                # from consideration and every strategy looks equally good at covering the rest.
                task.status = TaskStatus.READY
            assignment.status = "closed"
            session.flush()
        session.commit()

        n_tasks = len(tasks)
        result.results[strategy.value] = StrategyResult(
            strategy=strategy.value,
            episodes=episode_budget,
            tasks=n_tasks,
            solved=len(solved_tasks),
            wasted=wasted,
            coverage=round(len(solved_tasks) / n_tasks, 4) if n_tasks else 0.0,
            waste_rate=round(wasted / episode_budget, 4) if episode_budget else 0.0,
        )

    result.notes.append(
        "Same device, seeds, agents and budgets in every arm; only the allocation strategy varies."
    )
    return result


def _release(session: Session, workspace_id: uuid.UUID) -> None:
    """Return assigned-but-unconsumed tasks to the pool."""
    for task in session.execute(
        select(Task).where(
            Task.workspace_id == workspace_id, Task.status == TaskStatus.ASSIGNED
        )
    ).scalars():
        task.status = TaskStatus.READY
    session.flush()
