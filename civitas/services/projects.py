"""Long-horizon projects and request decomposition (Part B §5, §32).

§32's requirement is about survival:

> Projects must survive hundreds or thousands of bounded agent episodes. Agent processes may die.
> Workers may restart. Colab sessions may terminate. Projects must survive.

Nothing in a project's state lives in a process. `project_status` reconstructs everything from the
database, so a status computed after a restart is identical to one computed before it — and
`resume` is not a special code path, it is just calling the same functions again.

§5's requirement is about shape: a user submits *"investigate the cause of intermittent data
corruption and produce a verified fix"*, and the system responds by creating investigation tasks,
hypotheses, verification work and a synthesis — not by answering it directly.

The decomposer is **deterministic by default**. That is not a limitation dressed up: a
model-written decomposition is a set of model assertions about what the work is, and if the
decomposition itself varied run to run then no campaign over it would be reproducible (§46). A
model-backed decomposer sits behind the same interface for use where variety is wanted, and the
manifest records which produced a given plan.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import (
    AgentRole,
    ArtifactType,
    EventType,
    TaskStatus,
    TerminationReason,
)
from civitas.persistence.events import emit
from civitas.persistence.models import (
    Artifact,
    Episode,
    Objective,
    Project,
    Task,
    TaskDependency,
)
from civitas.persistence.types import utcnow

#: The stages §5 describes: investigate, hypothesise, verify, then synthesise. Each is a task
#: family the scheduler and the profile-learning service can measure separately (§26).
STAGES: tuple[tuple[str, str, AgentRole, str], ...] = (
    ("investigate", "Investigate: {focus}",
     AgentRole.EXPLORER,
     "Establish what is actually happening with {focus}. Record observations, not conclusions."),
    ("hypothesise", "Propose explanations for: {focus}",
     AgentRole.RESEARCHER,
     "Read what the investigation established and propose competing explanations. Record each as "
     "a hypothesis with what would falsify it."),
    ("verify", "Test the leading explanations for: {focus}",
     AgentRole.VERIFIER,
     "Test the recorded hypotheses with tools rather than argument. Record what each test rules "
     "in and out — a falsified hypothesis is as valuable as a confirmed one."),
    ("synthesise", "Produce a verified account of: {focus}",
     AgentRole.SYNTHESIZER,
     "Combine the surviving evidence into one account, keeping the links to what it rests on. "
     "State what is still unresolved."),
)

_STOP = frozenset(
    "the a an of in on at to for with by from as that this it and or not is are was were be "
    "please can you could would should investigate find determine produce".split()
)


class Decomposer(Protocol):
    """Turns a request into a plan. Deterministic and model-backed variants share this."""

    name: str

    def focus(self, request: str) -> str: ...


@dataclass
class DeterministicDecomposer:
    """Extracts the subject of a request without a model.

    Reproducible by construction, which is what a campaign over a plan needs (§46).
    """

    name: str = "deterministic/1.0"

    def focus(self, request: str) -> str:
        words = [w for w in re.findall(r"[\w-]+", request.lower()) if w not in _STOP]
        return " ".join(words[:8]) or request[:80]


@dataclass
class ModelDecomposer:
    """A model-written decomposition, behind the same interface (§19).

    Its output is a set of model assertions about what the work is, so the plan it produces
    carries that provenance and the manifest records that a model produced it.
    """

    provider: Any
    model: str
    name: str = "model/1.0"

    def focus(self, request: str) -> str:
        from civitas.runtime.providers.base import CompletionRequest, Message, Role

        completion = self.provider.complete(CompletionRequest(
            messages=(
                Message(Role.SYSTEM, "Reply with a short noun phrase naming the subject of the "
                                     "request. No punctuation, no explanation."),
                Message(Role.USER, request),
            ),
            model=self.model, max_tokens=32, temperature=0.0,
        ))
        return completion.text.strip()[:120] or request[:80]


@dataclass
class Plan:
    project: Project
    objective: Objective
    tasks: list[Task] = field(default_factory=list)
    decomposer: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_id": str(self.project.id),
            "objective_id": str(self.objective.id),
            "decomposer": self.decomposer,
            "tasks": [
                {"id": str(t.id), "title": t.title, "family": t.task_family,
                 "status": t.status.value}
                for t in self.tasks
            ],
        }


def submit_request(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    request: str,
    decomposer: Decomposer | None = None,
    project_name: str | None = None,
    config_hash: str = "",
) -> Plan:
    """Turn a high-level request into a project, an objective and a dependency chain (§5).

    The tasks are chained: hypothesising depends on investigating, verifying on hypothesising,
    synthesising on verifying. The dependencies are what stop the scheduler spending an episode on
    synthesis before there is anything to synthesise — and §27's `ready_tasks` already refuses
    blocked work, so this needs no scheduler change.
    """
    decomposer = decomposer or DeterministicDecomposer()
    focus = decomposer.focus(request)

    project = Project(
        workspace_id=workspace_id,
        name=project_name or focus[:180],
        description=request,
        meta={"request": request, "decomposer": decomposer.name},
    )
    session.add(project)
    session.flush()

    objective = Objective(
        project_id=project.id,
        title=f"Produce a verified account of {focus}",
        description=request,
        success_criteria={
            "requires": ["an account resting on tool-verified evidence",
                         "the falsified alternatives recorded",
                         "the unresolved questions stated"],
        },
    )
    session.add(objective)
    session.flush()

    plan = Plan(project=project, objective=objective, decomposer=decomposer.name)
    previous: Task | None = None
    for family, title, role, description in STAGES:
        task = Task(
            workspace_id=workspace_id,
            project_id=project.id,
            objective_id=objective.id,
            title=title.format(focus=focus),
            description=description.format(focus=focus),
            task_family=family,
            status=TaskStatus.READY if previous is None else TaskStatus.BLOCKED,
            meta={"preferred_role": role.value},
        )
        session.add(task)
        session.flush()
        if previous is not None:
            session.add(TaskDependency(task_id=task.id, depends_on_task_id=previous.id))
        plan.tasks.append(task)
        previous = task
    session.flush()

    emit(
        session, workspace_id=workspace_id, type=EventType.TASK_CREATED,
        project_id=project.id,
        payload={"request": request[:2000], "project_id": str(project.id),
                 "decomposer": decomposer.name, "tasks": len(plan.tasks)},
        config_hash=config_hash,
    )
    return plan


@dataclass
class ProjectStatus:
    """A project's state, reconstructed entirely from the database (§32)."""

    project_id: uuid.UUID
    name: str
    tasks_total: int
    tasks_completed: int
    tasks_blocked: int
    episodes: int
    episodes_succeeded: int
    artifacts: int
    validated_artifacts: int
    open_questions: int
    tokens_used: int
    cost_usd: float
    stages: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.tasks_total > 0 and self.tasks_completed == self.tasks_total

    def as_dict(self) -> dict[str, Any]:
        return {
            **{k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in self.__dict__.items()},
            "complete": self.complete,
            "progress": (
                round(self.tasks_completed / self.tasks_total, 4) if self.tasks_total else None
            ),
        }


def project_status(session: Session, project_id: uuid.UUID) -> ProjectStatus:
    """Reconstruct a project's state from persisted rows alone (§32).

    Holds nothing in memory between calls, so a status computed after a worker restart or a Colab
    reconnect is identical to one computed before it. That is the whole of §32's requirement: the
    project is the rows, not the process.
    """
    project = session.get(Project, project_id)
    if project is None:
        raise ValueError(f"no project {project_id}")

    tasks = list(
        session.execute(select(Task).where(Task.project_id == project_id)).scalars()
    )
    episodes = list(
        session.execute(select(Episode).where(Episode.project_id == project_id)).scalars()
    )
    if not episodes and tasks:
        # Episodes carry `project_id` only when the caller set it; fall back to the task ids so a
        # project assembled by a path that did not set it is still readable.
        episodes = list(
            session.execute(
                select(Episode).where(Episode.task_id.in_([t.id for t in tasks]))
            ).scalars()
        )
    artifacts = list(
        session.execute(
            select(Artifact).where(Artifact.project_id == project_id,
                                   Artifact.archived_at.is_(None))
        ).scalars()
    )
    from civitas.domain.enums import ValidationState

    return ProjectStatus(
        project_id=project_id,
        name=project.name,
        tasks_total=len(tasks),
        tasks_completed=sum(1 for t in tasks if t.status is TaskStatus.COMPLETED),
        tasks_blocked=sum(1 for t in tasks if t.status is TaskStatus.BLOCKED),
        episodes=len(episodes),
        episodes_succeeded=sum(
            1 for e in episodes
            if e.termination_reason is TerminationReason.EVALUATOR_SUCCESS
        ),
        artifacts=len(artifacts),
        validated_artifacts=sum(
            1 for a in artifacts
            if a.validation_state in (ValidationState.EVALUATOR_CONFIRMED,
                                      ValidationState.HUMAN_CONFIRMED,
                                      ValidationState.REPRODUCED,
                                      ValidationState.TOOL_VERIFIED)
        ),
        open_questions=sum(
            1 for a in artifacts
            if a.type in (ArtifactType.QUESTION, ArtifactType.UNRESOLVED_ISSUE)
        ),
        tokens_used=sum(e.tokens_used for e in episodes),
        cost_usd=round(sum(e.cost_usd for e in episodes), 6),
        stages={t.task_family: t.status.value for t in tasks if t.task_family},
    )


def next_ready_task(session: Session, project_id: uuid.UUID) -> Task | None:
    """The next task whose dependencies are satisfied (§27, §32).

    Used by `resume`: after a restart there is no in-memory cursor to recover, so the next unit of
    work is derived from the dependency graph exactly as it would be on a first run.
    """
    from civitas.scheduler.allocation import ready_tasks

    project = session.get(Project, project_id)
    if project is None:
        return None
    candidates = [
        t for t in ready_tasks(session, project.workspace_id) if t.project_id == project_id
    ]
    return candidates[0] if candidates else None


def complete_task(session: Session, task: Task, *, succeeded: bool = True) -> list[Task]:
    """Mark a task done and unblock whatever depended on it. Returns the newly ready tasks."""
    task.status = TaskStatus.COMPLETED if succeeded else TaskStatus.FAILED
    task.completed_at = utcnow()
    session.flush()

    if not succeeded:
        return []
    dependents = list(
        session.execute(
            select(TaskDependency).where(TaskDependency.depends_on_task_id == task.id)
        ).scalars()
    )
    unblocked: list[Task] = []
    for dependency in dependents:
        dependent = session.get(Task, dependency.task_id)
        if dependent is None or dependent.status is not TaskStatus.BLOCKED:
            continue
        blockers = list(
            session.execute(
                select(TaskDependency).where(TaskDependency.task_id == dependent.id)
            ).scalars()
        )
        if all(
            session.get(Task, b.depends_on_task_id).status is TaskStatus.COMPLETED
            for b in blockers
        ):
            dependent.status = TaskStatus.READY
            unblocked.append(dependent)
    session.flush()
    return unblocked


def project_timeline(
    session: Session, project_id: uuid.UUID, *, limit: int = 200
) -> list[dict[str, Any]]:
    """How the project's knowledge developed over time (§49's collective timeline)."""
    project = session.get(Project, project_id)
    if project is None:
        return []
    artifacts = session.execute(
        select(Artifact)
        .where(Artifact.project_id == project_id)
        .order_by(Artifact.created_at)
        .limit(limit)
    ).scalars()
    return [
        {
            "at": a.created_at.isoformat(),
            "artifact_id": str(a.id),
            "type": a.type.value,
            "title": a.title,
            "validation_state": a.validation_state.value,
            "episode_id": str(a.creator_episode_id) if a.creator_episode_id else None,
        }
        for a in artifacts
    ]
