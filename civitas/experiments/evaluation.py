"""External evaluation (Part B §47).

Agents cannot self-certify. An evaluator reads the task's `evaluator_spec` — which no episode can
see — and writes an `Evaluation` row. Only that row may raise an artifact's validation state to
`evaluator_confirmed`, and only after credit has been propagated does the outcome reach any
utility metric.

§47 requires four scopes to be kept apart: episode performance, artifact value, agent contribution
and collective performance. They are separate rows with a `scope` discriminator rather than one
conflated score, because collapsing them is how an agent that produced a useful artifact while
failing its own task becomes invisible.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import EventType, TerminationReason, ValidationState
from civitas.persistence.events import emit
from civitas.persistence.models import Artifact, Episode, Evaluation, Task
from civitas.runtime.episode import EpisodeOutcome

EVALUATOR_VERSION = "exact_match/1.0"


def evaluate_episode(
    session: Session,
    *,
    episode: Episode,
    outcome: EpisodeOutcome,
    task: Task,
    config_hash: str = "",
) -> Evaluation:
    """Judge one episode against its task's held-out criteria."""
    spec = task.evaluator_spec or {}
    kind = spec.get("kind", "exact_match")

    if kind == "exact_match":
        expected = str(spec.get("expected", "")).strip().lower()
        given = (outcome.submitted_answer or "").strip().lower()
        succeeded = bool(expected) and given == expected
        detail: dict[str, Any] = {
            "submitted": outcome.submitted_answer,
            "matched": succeeded,
            # The expected value is deliberately absent. An evaluation row is readable through the
            # API, and putting ground truth in it would leak the benchmark to anything that can
            # read results (§47).
            "evaluator": kind,
        }
    else:  # pragma: no cover - future evaluators register here
        raise ValueError(f"no evaluator registered for kind {kind!r}")

    if outcome.termination_reason in (
        TerminationReason.PROVIDER_FAILURE,
        TerminationReason.WORKER_FAILURE,
    ):
        # An infrastructure failure is not evidence about the collective. It is recorded, and
        # excluded from success rates by `is_readable` below, because counting it as a failure
        # would let a flaky provider look like a weak arm.
        detail["infrastructure_failure"] = outcome.termination_reason.value

    evaluation = Evaluation(
        workspace_id=episode.workspace_id,
        episode_id=episode.id,
        task_id=task.id,
        scope="episode",
        target_id=episode.id,
        evaluator_kind=kind,
        evaluator_version=EVALUATOR_VERSION,
        succeeded=succeeded,
        score=1.0 if succeeded else 0.0,
        max_score=1.0,
        detail=detail,
    )
    session.add(evaluation)
    session.flush()

    # The episode's termination reason is corrected to the evaluated truth. `solved` was the
    # agent's claim; these two are the evaluator's finding (§8).
    if outcome.termination_reason is TerminationReason.SOLVED:
        episode.termination_reason = (
            TerminationReason.EVALUATOR_SUCCESS if succeeded
            else TerminationReason.EVALUATOR_FAILURE
        )

    if succeeded and outcome.submitted_artifact_id is not None:
        result = session.get(Artifact, outcome.submitted_artifact_id)
        if result is not None:
            # The only path by which an artifact reaches `evaluator_confirmed`. No tool can set it.
            result.validation_state = ValidationState.EVALUATOR_CONFIRMED
            result.last_validated_at = evaluation.created_at

    emit(
        session, workspace_id=episode.workspace_id, type=EventType.EVALUATION_COMPLETED,
        episode_id=episode.id, task_id=task.id,
        payload={"evaluation_id": str(evaluation.id), "succeeded": succeeded, "scope": "episode"},
        config_hash=config_hash,
    )
    return evaluation


def is_readable(evaluation: Evaluation) -> bool:
    """Whether this evaluation may enter a success rate.

    An infrastructure failure is excluded: it says nothing about the arm, and including it would
    let provider flakiness masquerade as a weaker condition. The exclusion is *recorded* on the
    row, so a reader can see how many were dropped rather than having to infer it from a count
    that does not add up (ARCHITECTURE §3.10).
    """
    return "infrastructure_failure" not in (evaluation.detail or {})


def record_collective_evaluation(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    label: str,
    succeeded: bool,
    score: float,
    detail: dict[str, Any] | None = None,
) -> Evaluation:
    """A judgement about the collective rather than an episode (Part B §47's fourth scope)."""
    evaluation = Evaluation(
        workspace_id=workspace_id,
        scope="collective",
        evaluator_kind=label,
        evaluator_version=EVALUATOR_VERSION,
        succeeded=succeeded,
        score=score,
        detail=detail or {},
    )
    session.add(evaluation)
    session.flush()
    return evaluation


def pending_credit(session: Session, workspace_id: uuid.UUID) -> list[Evaluation]:
    """Evaluations whose outcome has not yet been propagated (Part A §A2.1)."""
    return list(
        session.execute(
            select(Evaluation).where(
                Evaluation.workspace_id == workspace_id,
                Evaluation.credit_assigned_at.is_(None),
                Evaluation.scope == "episode",
            )
        ).scalars()
    )
