"""Outcome-linked credit assignment (Part A §A2.1).

The difference between a record that self-corrects and one that self-inflates.

Credit flows **backward along usage links**, and only along usage links. `ArtifactUsage` says the
episode *read* the artifact; retrieval returning it is not enough. That distinction is what stops
an agent earning credit for a large retrieval it never looked at.

From each used artifact, credit propagates further through `derived_from` / `depends_on` / `uses`
and the other edges in `CREDIT_EDGES`, decaying with graph distance. Not every relation carries
credit: an artifact that merely *contradicts* a successful one did not contribute to the success.

Every update is an immutable `ArtifactUtilityMetric` event, and `Artifact.downstream_utility` is
the fold of those events — never written directly, never self-reported.

**The decay constant is measured, not chosen.** The research lineage bracketed its `mark_decay`
between two constraints and reported it rather than guessing (ARCHITECTURE §6.2). The same applies
here: `DECAY_PER_HOP` is bracketed below and the M4 write-up reports what the bracket implies.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from civitas.domain.enums import CREDIT_EDGES, EventType
from civitas.persistence.events import emit
from civitas.persistence.models import (
    Artifact,
    ArtifactRelation,
    ArtifactUsage,
    ArtifactUtilityMetric,
    Evaluation,
    ToolDefinition,
    ToolRun,
)
from civitas.persistence.types import utcnow

#: Credit multiplier per graph hop.
#:
#: Bracketed by two requirements, exactly as the lineage bracketed its decay constant:
#:   * it must be **below 1**, or an artifact five hops from any success accrues as much credit as
#:     the one actually read, and utility stops discriminating;
#:   * it must be **above zero at the depths provenance actually reaches**, or nothing but the
#:     directly-read artifact is ever credited and the graph is decorative.
#: At 0.5 with MAX_DEPTH 3, an artifact three hops back still receives 12.5% of the credit — enough
#: to rank, small enough not to swamp. The M4 write-up reports the measured lineage depth this was
#: chosen against.
DECAY_PER_HOP = 0.5

#: Beyond this the contribution is below measurement noise and the traversal cost is real: a
#: densely linked workspace fans out fast, and an unbounded walk would credit most of it.
MAX_DEPTH = 3

#: Credit for a confirmed success and a confirmed failure. Asymmetric on purpose: a success has
#: many contributing causes and each deserves a share, whereas a failure indicts what was actually
#: used more sharply. Symmetric weights would make utility a slow random walk around zero.
SUCCESS_DELTA = 1.0
FAILURE_DELTA = -0.6


@dataclass
class CreditReport:
    evaluation_id: uuid.UUID
    artifacts_credited: int = 0
    tools_credited: int = 0
    total_delta: float = 0.0
    max_depth_reached: int = 0
    skipped_already_assigned: bool = False


def assign_credit(
    session: Session,
    evaluation: Evaluation,
    *,
    config_hash: str = "",
) -> CreditReport:
    """Propagate an evaluation's outcome backward over what the episode used.

    Idempotent: `Evaluation.credit_assigned_at` is set once, so a retried job or a resumed
    campaign cannot double-count an outcome (§41, §42).
    """
    report = CreditReport(evaluation_id=evaluation.id)
    if evaluation.credit_assigned_at is not None:
        report.skipped_already_assigned = True
        return report
    if evaluation.episode_id is None:
        evaluation.credit_assigned_at = utcnow()
        return report

    base = SUCCESS_DELTA if evaluation.succeeded else FAILURE_DELTA

    usages = list(
        session.execute(
            select(ArtifactUsage).where(ArtifactUsage.episode_id == evaluation.episode_id)
        ).scalars()
    )
    seen: dict[uuid.UUID, int] = {}
    frontier: deque[tuple[uuid.UUID, int]] = deque()

    for usage in usages:
        # An artifact surfaced as a duplicate-failure warning is not credited for the outcome: it
        # was injected by the runtime, not sought by the episode, and crediting it would let the
        # warning mechanism inflate the utility of whatever it happened to surface.
        if usage.surfaced_as_duplicate_warning:
            continue
        if usage.artifact_id not in seen:
            seen[usage.artifact_id] = 0
            frontier.append((usage.artifact_id, 0))

    while frontier:
        artifact_id, depth = frontier.popleft()
        report.max_depth_reached = max(report.max_depth_reached, depth)
        delta = base * (DECAY_PER_HOP**depth)

        session.add(
            ArtifactUtilityMetric(
                artifact_id=artifact_id,
                source_episode_id=evaluation.episode_id,
                evaluation_id=evaluation.id,
                delta=delta,
                graph_distance=depth,
                reason="evaluator_success" if evaluation.succeeded else "evaluator_failure",
                config_hash=config_hash,
            )
        )
        report.artifacts_credited += 1
        report.total_delta += delta

        if depth >= MAX_DEPTH:
            continue
        # Outward along the edges this artifact *rests on*: source -> target where the source is
        # the artifact we credited. An artifact derived_from another owes its value to it.
        for relation in session.execute(
            select(ArtifactRelation).where(
                ArtifactRelation.source_id == artifact_id,
                ArtifactRelation.type.in_([t.value for t in CREDIT_EDGES]),
            )
        ).scalars():
            if relation.target_id in seen:
                continue
            seen[relation.target_id] = depth + 1
            frontier.append((relation.target_id, depth + 1))

    # Tools the episode actually invoked (§17, §29 tool reliability).
    tool_ids = set(
        session.execute(
            select(ToolRun.tool_definition_id).where(ToolRun.episode_id == evaluation.episode_id)
        ).scalars()
    )
    for tool_id in tool_ids:
        definition = session.get(ToolDefinition, tool_id)
        if definition is None:
            continue
        definition.downstream_utility += base
        report.tools_credited += 1

    session.flush()
    recompute(session, list(seen))

    evaluation.credit_assigned_at = utcnow()
    emit(
        session,
        workspace_id=evaluation.workspace_id,
        type=EventType.CREDIT_ASSIGNED,
        episode_id=evaluation.episode_id,
        payload={
            "evaluation_id": str(evaluation.id),
            "succeeded": evaluation.succeeded,
            "artifacts": report.artifacts_credited,
            "tools": report.tools_credited,
            "max_depth": report.max_depth_reached,
        },
        config_hash=config_hash,
    )
    return report


def recompute(session: Session, artifact_ids: list[uuid.UUID] | None = None) -> int:
    """Rebuild `Artifact.downstream_utility` as the fold of its credit events.

    The events are the truth and this column is a cache for ranking. Rebuilding it from the events
    — rather than incrementing in place — is what makes the utility of any artifact at any time
    reconstructable from an immutable record (Part A §A1.2, §A2.1).
    """
    stmt = select(
        ArtifactUtilityMetric.artifact_id, func.sum(ArtifactUtilityMetric.delta)
    ).group_by(ArtifactUtilityMetric.artifact_id)
    if artifact_ids:
        stmt = stmt.where(ArtifactUtilityMetric.artifact_id.in_(artifact_ids))

    updated = 0
    for artifact_id, total in session.execute(stmt).all():
        artifact = session.get(Artifact, artifact_id)
        if artifact is not None:
            artifact.downstream_utility = float(total or 0.0)
            updated += 1
    session.flush()
    return updated


def calibration(session: Session, workspace_id: uuid.UUID) -> dict[str, float | int | None]:
    """Correlation between self-reported confidence and eventual validated outcome.

    Part A §A2.1 names this as the metric credit assignment must move: if the record is
    self-correcting, an artifact's stated confidence should come to track whether it was borne out.

    Returns `None` for the correlation rather than 0.0 when it cannot be computed — a workspace
    with no validated artifacts, or with no variation in either variable, has *no* calibration, and
    reporting zero would present an absence as a finding (ARCHITECTURE §3.9).
    """
    rows = session.execute(
        select(Artifact.confidence, Artifact.downstream_utility).where(
            Artifact.workspace_id == workspace_id,
            Artifact.archived_at.is_(None),
        )
    ).all()
    pairs = [(float(c), float(u)) for c, u in rows if u != 0.0]
    n = len(pairs)
    if n < 3:
        return {"n": n, "correlation": None,
                "reason": "fewer than 3 artifacts carry an outcome-derived utility"}

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return {"n": n, "correlation": None,
                "reason": "no variation in confidence or in utility"}
    return {"n": n, "correlation": sxy / (sxx**0.5 * syy**0.5), "reason": None}
