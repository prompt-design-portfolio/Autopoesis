"""Task allocation (Part B §27).

The scheduler decides where to spend the next episode. §27 requires nine strategies, and requires
allocation decisions to be **logged** — not for tidiness, but because an allocation policy cannot
be evaluated against outcomes unless the reason it fired is recoverable afterwards.

Every strategy returns a score per candidate plus the *features* that produced it, and both go onto
the `Assignment`. That is what makes a scheduler policy a versioned artifact the A2.2 gate can
approve or roll back, rather than a heuristic nobody can audit.

One rule cuts across all of them: **avoid spending agents on identical duplicated work unless
intentionally sampling** (§27). `duplication_penalty` implements it, and the `random` strategy —
which exists precisely to sample — is the one that switches it off.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from civitas.domain.enums import (
    AllocationStrategy,
    ArtifactType,
    EventType,
    TaskStatus,
    TerminationReason,
)
from civitas.persistence.events import emit
from civitas.persistence.models import (
    AgentProfile,
    Artifact,
    Assignment,
    Episode,
    Subproblem,
    Task,
    TaskDependency,
)


@dataclass
class Candidate:
    """One (task, profile) pair the scheduler could spend an episode on."""

    task: Task
    profile: AgentProfile
    subproblem: Subproblem | None = None
    score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)

    def rationale(self) -> dict[str, Any]:
        return {
            "task_id": str(self.task.id),
            "profile": self.profile.name,
            "subproblem_id": str(self.subproblem.id) if self.subproblem else None,
            "features": {k: round(v, 6) for k, v in self.features.items()},
            "score": round(self.score, 6),
        }


def ready_tasks(session: Session, workspace_id: uuid.UUID, *, limit: int = 200) -> list[Task]:
    """Tasks whose dependencies are satisfied (§27).

    A task blocked on an incomplete dependency is *not* a candidate, and returning it would let
    the scheduler spend episodes on work that cannot succeed — which looks like agent failure in
    every downstream metric.
    """
    tasks = list(
        session.execute(
            select(Task).where(
                Task.workspace_id == workspace_id,
                Task.archived_at.is_(None),
                Task.status.in_([TaskStatus.PENDING, TaskStatus.READY, TaskStatus.BLOCKED]),
            ).limit(limit)
        ).scalars()
    )
    if not tasks:
        return []

    dependencies = list(
        session.execute(
            select(TaskDependency).where(
                TaskDependency.task_id.in_([t.id for t in tasks])
            )
        ).scalars()
    )
    blocking: dict[uuid.UUID, list[uuid.UUID]] = {}
    for dependency in dependencies:
        blocking.setdefault(dependency.task_id, []).append(dependency.depends_on_task_id)

    out: list[Task] = []
    for task in tasks:
        blockers = blocking.get(task.id, [])
        if blockers:
            statuses = list(
                session.execute(
                    select(Task.status).where(Task.id.in_(blockers))
                ).scalars()
            )
            if any(s is not TaskStatus.COMPLETED for s in statuses):
                if task.status is not TaskStatus.BLOCKED:
                    task.status = TaskStatus.BLOCKED
                continue
        if task.attempts >= task.max_attempts:
            continue
        if task.status is TaskStatus.BLOCKED:
            task.status = TaskStatus.READY
        out.append(task)
    session.flush()
    return out


def _episode_stats(session: Session, workspace_id: uuid.UUID) -> dict[uuid.UUID, dict[str, int]]:
    """Attempts and successes per task, from real episodes."""
    from sqlalchemy import Integer

    rows = session.execute(
        select(
            Episode.task_id,
            func.count(Episode.id),
            func.sum(
                func.cast(
                    Episode.termination_reason == TerminationReason.EVALUATOR_SUCCESS.value,
                    Integer,
                )
            ),
            # An attempt that created an artifact or documented a failure left something a
            # successor can use, so it is continuation rather than duplication.
            func.sum(
                func.cast(
                    (Episode.artifacts_created > 0) | (Episode.duplicate_failures > 0),
                    Integer,
                )
            ),
        )
        .where(Episode.workspace_id == workspace_id, Episode.task_id.isnot(None))
        .group_by(Episode.task_id)
    ).all()
    return {
        task_id: {
            "attempts": int(attempts or 0),
            "successes": int(successes or 0),
            "productive": int(productive or 0),
        }
        for task_id, attempts, successes, productive in rows
    }


def duplication_penalty(attempts: int, productive_attempts: int = 0) -> float:
    """Discourage spending another agent on work already attempted (§27) — but only where the
    previous attempt was genuinely wasted.

    §27 says to avoid duplicated work, and a blind reading of that penalises the behaviour that
    makes a collective work. Measured directly on one task: attempts 1 and 2 failed but *recorded*
    which operations were ruled out; attempt 3 succeeded on that record, and probes fell 6 → 2 → 1
    → 0 across the sequence. A repeat that consumes what the last repeat left behind is
    **continuation**, not duplication.

    So only *unproductive* attempts — ones that left nothing for a successor — attract the
    penalty. Without this correction the reasoned strategies lost to random allocation at every
    episode budget above 9, because random re-attempted tasks and so accumulated the negative
    knowledge the reasoned strategies were spreading too thin to build.

    Saturating rather than linear: the second unproductive attempt is much less valuable than the
    first, but the tenth is not much less valuable than the ninth — by then the penalty has done
    its work and further growth would forbid persistence on a genuinely hard task.
    """
    wasted = max(0, attempts - productive_attempts)
    return -math.tanh(wasted / 2.0)


def _uncertainty(session: Session, task: Task, stats: dict[str, int]) -> float:
    """How unresolved the task is, in [0, 1].

    Highest where a task has been attempted and *not* resolved — that is where an episode buys
    the most information. A task nobody has touched sits at 0.5: genuinely unknown, but not yet
    known to be hard.
    """
    attempts = stats.get("attempts", 0)
    successes = stats.get("successes", 0)
    if attempts == 0:
        return 0.5
    if successes:
        return 0.0
    return min(1.0, 0.5 + 0.5 * math.tanh(attempts / 3.0))


def _open_subproblem_pressure(session: Session, task: Task) -> float:
    open_count = session.execute(
        select(func.count(Subproblem.id)).where(
            Subproblem.task_id == task.id, Subproblem.status == "open"
        )
    ).scalar_one()
    return math.tanh(float(open_count or 0) / 3.0)


def _evidence_gap(session: Session, task: Task) -> float:
    """How thin the evidence on this task is (§27 evidence gaps).

    Hypotheses with no supporting evidence are exactly where an experiment is worth running, so
    the gap is the share of the task's hypotheses that nothing yet supports.
    """
    hypotheses = session.execute(
        select(func.count(Artifact.id)).where(
            Artifact.task_id == task.id, Artifact.type == ArtifactType.HYPOTHESIS
        )
    ).scalar_one()
    evidence = session.execute(
        select(func.count(Artifact.id)).where(
            Artifact.task_id == task.id,
            Artifact.type.in_([ArtifactType.EVIDENCE.value,
                               ArtifactType.EXPERIMENT_RESULT.value]),
        )
    ).scalar_one()
    if not hypotheses:
        return 0.0
    return max(0.0, 1.0 - float(evidence or 0) / float(hypotheses))


def _contradiction_pressure(session: Session, task: Task) -> float:
    count = session.execute(
        select(func.count(Artifact.id)).where(
            Artifact.task_id == task.id,
            Artifact.type == ArtifactType.CONTRADICTION,
            Artifact.archived_at.is_(None),
        )
    ).scalar_one()
    return math.tanh(float(count or 0) / 2.0)


def _profile_fit(profile: AgentProfile, task: Task) -> float:
    """Measured performance of this profile on this work type (§26).

    Read from `performance_by_work_type`, which is a fold over evaluations written by the
    profile-learning service — never declared by a profile about itself. An unmeasured pairing
    returns 0.5: unknown is not the same as bad, and treating it as bad would freeze the
    allocation on whichever profile happened to be tried first.
    """
    table = profile.performance_by_work_type or {}
    entry = table.get(task.task_family)
    if not isinstance(entry, dict) or not entry.get("n"):
        return 0.5
    return float(entry.get("success_rate", 0.5))


def _cost_estimate(session: Session, task: Task, profile: AgentProfile) -> float:
    """Expected cost, normalised. Used by `cost_aware` to prefer cheap information."""
    table = profile.performance_by_work_type or {}
    entry = table.get(task.task_family) or {}
    tokens = float(entry.get("mean_tokens", 0.0))
    return math.tanh(tokens / 50_000.0) if tokens else 0.3


# --------------------------------------------------------------------------
# the strategies (§27)
# --------------------------------------------------------------------------
def score_candidates(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    tasks: list[Task],
    profiles: list[AgentProfile],
    strategy: AllocationStrategy,
    seed: int = 0,
    round_robin_cursor: int = 0,
) -> list[Candidate]:
    """Score every (task, profile) pair under one strategy."""
    stats = _episode_stats(session, workspace_id)
    candidates: list[Candidate] = []

    for task in tasks:
        task_stats = stats.get(task.id, {})
        attempts = task_stats.get("attempts", 0)
        base = {
            "priority": float(task.priority),
            "difficulty": float(task.difficulty),
            "attempts": float(attempts),
            "productive_attempts": float(task_stats.get("productive", 0)),
            "duplication_penalty": duplication_penalty(
                attempts, task_stats.get("productive", 0)
            ),
            "uncertainty": _uncertainty(session, task, task_stats),
            "open_subproblems": _open_subproblem_pressure(session, task),
            "evidence_gap": _evidence_gap(session, task),
            "contradictions": _contradiction_pressure(session, task),
        }
        for index, profile in enumerate(profiles):
            features = dict(base)
            features["profile_fit"] = _profile_fit(profile, task)
            features["cost"] = _cost_estimate(session, task, profile)
            features["score"] = 0.0

            if strategy is AllocationStrategy.RANDOM:
                # Deliberate sampling: the duplication penalty is off, because repeating work is
                # the point of a random arm and penalising it would make it not random.
                digest = hashlib.sha256(
                    f"{seed}|{task.id}|{profile.id}".encode()
                ).hexdigest()
                score = int(digest[:8], 16) / 0xFFFFFFFF
                features["random"] = score
            elif strategy is AllocationStrategy.ROUND_ROBIN:
                # Position in a stable rotation, so every task and profile gets its turn
                # regardless of score.
                position = (index + round_robin_cursor) % max(1, len(profiles))
                score = 1.0 - position / max(1, len(profiles))
                features["rotation_position"] = float(position)
            elif strategy is AllocationStrategy.PRIORITY:
                score = features["priority"] + features["duplication_penalty"]
            elif strategy is AllocationStrategy.UNCERTAINTY:
                score = features["uncertainty"] + 0.5 * features["open_subproblems"] \
                    + features["duplication_penalty"]
            elif strategy is AllocationStrategy.INFORMATION_GAIN:
                # Expected information gain: uncertainty times how much evidence is missing,
                # weighted by whether this profile can actually produce it.
                score = (
                    features["uncertainty"] * (0.5 + features["evidence_gap"])
                    * (0.5 + features["profile_fit"])
                    + features["duplication_penalty"]
                )
            elif strategy is AllocationStrategy.SPECIALIZATION:
                score = 2.0 * features["profile_fit"] + 0.3 * features["priority"] \
                    + features["duplication_penalty"]
            elif strategy is AllocationStrategy.DIVERSITY:
                # Prefer a profile that has *not* worked this family, so the collective learns
                # which profiles suit which work rather than entrenching the first guess (§26).
                table = profile.performance_by_work_type or {}
                seen = float((table.get(task.task_family) or {}).get("n", 0))
                features["profile_novelty"] = 1.0 / (1.0 + seen)
                score = features["profile_novelty"] + 0.5 * features["uncertainty"] \
                    + features["duplication_penalty"]
            elif strategy is AllocationStrategy.ADVERSARIAL:
                # Go where the collective disagrees with itself, and where claims are unchallenged.
                score = (
                    1.5 * features["contradictions"] + features["evidence_gap"]
                    + features["duplication_penalty"]
                )
            elif strategy is AllocationStrategy.COST_AWARE:
                score = (
                    features["uncertainty"] + 0.5 * features["priority"]
                    - 1.5 * features["cost"] + features["duplication_penalty"]
                )
            else:  # pragma: no cover - the enum is closed
                raise ValueError(f"unhandled allocation strategy {strategy!r}")

            features["score"] = score
            candidates.append(
                Candidate(task=task, profile=profile, score=score, features=features)
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def allocate(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    strategy: AllocationStrategy = AllocationStrategy.INFORMATION_GAIN,
    count: int = 1,
    seed: int = 0,
    scheduler_policy_id: uuid.UUID | None = None,
    config_hash: str = "",
) -> list[Assignment]:
    """Create up to `count` assignments, logging the reason for each (§27)."""
    tasks = ready_tasks(session, workspace_id)
    profiles = list(
        session.execute(
            select(AgentProfile).where(
                AgentProfile.workspace_id == workspace_id,
                AgentProfile.is_active.is_(True),
                AgentProfile.archived_at.is_(None),
            )
        ).scalars()
    )
    if not tasks or not profiles:
        return []

    candidates = score_candidates(
        session, workspace_id=workspace_id, tasks=tasks, profiles=profiles,
        strategy=strategy, seed=seed,
    )

    out: list[Assignment] = []
    used_tasks: set[uuid.UUID] = set()
    for candidate in candidates:
        if len(out) >= count:
            break
        # One assignment per task per allocation round, unless the strategy is deliberately
        # sampling. Otherwise a single high-scoring task would consume the whole round.
        if candidate.task.id in used_tasks and strategy is not AllocationStrategy.RANDOM:
            continue
        used_tasks.add(candidate.task.id)

        assignment = Assignment(
            task_id=candidate.task.id,
            agent_profile_id=candidate.profile.id,
            subproblem_id=candidate.subproblem.id if candidate.subproblem else None,
            strategy=strategy.value,
            rationale=candidate.rationale(),
            score=candidate.score,
            scheduler_policy_id=scheduler_policy_id,
        )
        session.add(assignment)
        candidate.task.status = TaskStatus.ASSIGNED
        out.append(assignment)

    session.flush()
    for assignment in out:
        emit(
            session, workspace_id=workspace_id, type=EventType.ASSIGNMENT_CREATED,
            task_id=assignment.task_id,
            payload={"assignment_id": str(assignment.id), "strategy": strategy.value,
                     "score": round(assignment.score, 6),
                     "rationale": assignment.rationale},
            config_hash=config_hash,
        )
    return out
