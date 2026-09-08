"""Queryable, exportable metrics (Part B §48).

Every metric §48 names that the current milestones can compute, computed from the record rather
than from anything self-reported. Metrics that require a mechanism not yet built return `None`
with a stated reason — never 0.0.

That distinction is ARCHITECTURE §3.9, and it is the difference between "we measured this and it
was zero" and "we cannot measure this yet". A dashboard that renders the second as the first
manufactures a finding.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from civitas.domain.enums import (
    NEGATIVE_TYPES,
    ArtifactStatus,
    EventType,
    TerminationReason,
    ValidationState,
)
from civitas.persistence.models import (
    Artifact,
    ArtifactRelation,
    ArtifactUsage,
    Episode,
    Evaluation,
    Event,
    RetrievalDecision,
    ToolDefinition,
    ToolRun,
)


@dataclass
class Metric:
    name: str
    value: float | int | None
    unit: str = ""
    #: Why the value is None. Present exactly when `value is None`.
    unavailable_reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "value": self.value, "unit": self.unit}
        if self.unavailable_reason:
            out["unavailable"] = self.unavailable_reason
        if self.detail:
            out["detail"] = self.detail
        return out


def _unavailable(name: str, reason: str, unit: str = "") -> Metric:
    return Metric(name=name, value=None, unit=unit, unavailable_reason=reason)


def collect(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    include_probes: bool = False,
) -> dict[str, Metric]:
    """Every §48 metric this build can compute, for one workspace.

    `include_probes` defaults to False: benchmark probe episodes are measured *against* the
    collective and must not be counted as part of it. That is the founder-free discipline of
    ARCHITECTURE §3.5, and defaulting the other way would let the measurement contaminate the
    thing measured.
    """
    metrics: dict[str, Metric] = {}

    ep_filter = [Episode.workspace_id == workspace_id]
    if not include_probes:
        ep_filter.append(Episode.is_benchmark_probe.is_(False))

    episodes = list(session.execute(select(Episode).where(*ep_filter)).scalars())
    n_ep = len(episodes)

    # --- outcome -------------------------------------------------------
    evaluated = list(
        session.execute(
            select(Evaluation).where(
                Evaluation.workspace_id == workspace_id, Evaluation.scope == "episode"
            )
        ).scalars()
    )
    readable = [e for e in evaluated if "infrastructure_failure" not in (e.detail or {})]
    excluded = len(evaluated) - len(readable)

    if readable:
        metrics["success_rate"] = Metric(
            "success_rate",
            sum(1 for e in readable if e.succeeded) / len(readable),
            "fraction",
            detail={"n": len(readable), "excluded_infrastructure_failures": excluded},
        )
    else:
        metrics["success_rate"] = _unavailable(
            "success_rate", "no episode has been externally evaluated", "fraction"
        )

    solved = [
        e for e in episodes
        if e.termination_reason is TerminationReason.EVALUATOR_SUCCESS
    ]
    for name, attr, unit in (
        ("tokens_to_solution", "tokens_used", "tokens"),
        ("cost_to_solution", "cost_usd", "usd"),
        ("tool_calls_to_solution", "tool_calls_used", "calls"),
    ):
        if solved:
            metrics[name] = Metric(
                name, sum(getattr(e, attr) for e in solved) / len(solved), unit,
                detail={"n": len(solved)},
            )
        else:
            metrics[name] = _unavailable(name, "no episode has been evaluated as a success", unit)

    durations = [e.duration_s for e in solved if e.duration_s is not None]
    metrics["time_to_solution"] = (
        Metric("time_to_solution", sum(durations) / len(durations), "seconds",
               detail={"n": len(durations)})
        if durations
        else _unavailable("time_to_solution", "no completed successful episode", "seconds")
    )

    # --- knowledge -----------------------------------------------------
    artifacts = list(
        session.execute(
            select(Artifact).where(Artifact.workspace_id == workspace_id)
        ).scalars()
    )
    active = [a for a in artifacts if a.archived_at is None]
    metrics["artifact_count"] = Metric("artifact_count", len(active), "artifacts")
    metrics["artifact_survival"] = (
        Metric("artifact_survival", len(active) / len(artifacts), "fraction",
               detail={"total": len(artifacts), "archived": len(artifacts) - len(active)})
        if artifacts
        else _unavailable("artifact_survival", "no artifacts exist", "fraction")
    )

    negative = [a for a in active if a.type in NEGATIVE_TYPES]
    metrics["negative_knowledge_share"] = (
        Metric("negative_knowledge_share", len(negative) / len(active), "fraction",
               detail={"n_negative": len(negative)})
        if active
        else _unavailable("negative_knowledge_share", "no active artifacts", "fraction")
    )

    stale = [a for a in active if a.is_stale or a.status is ArtifactStatus.STALE]
    metrics["stale_artifact_share"] = (
        Metric("stale_artifact_share", len(stale) / len(active), "fraction")
        if active
        else _unavailable("stale_artifact_share", "no active artifacts", "fraction")
    )

    # Knowledge reuse: episodes that read at least one artifact created by an earlier episode.
    # Reads, not retrievals — being shown something is not using it (Part A §A2.1).
    episode_ids = {e.id for e in episodes}
    usages = list(
        session.execute(
            select(ArtifactUsage).where(ArtifactUsage.episode_id.in_(episode_ids or {uuid.uuid4()}))
        ).scalars()
    ) if episode_ids else []
    reusing = {u.episode_id for u in usages}
    metrics["knowledge_reuse_rate"] = (
        Metric("knowledge_reuse_rate", len(reusing) / n_ep, "fraction",
               detail={"episodes_reading": len(reusing), "episodes": n_ep})
        if n_ep
        else _unavailable("knowledge_reuse_rate", "no episodes", "fraction")
    )

    # --- duplication ---------------------------------------------------
    dup_events = session.execute(
        select(func.count(Event.id)).where(
            Event.workspace_id == workspace_id, Event.type == EventType.DUPLICATE_FAILURE
        )
    ).scalar_one()
    metrics["duplicate_failure_rate"] = (
        Metric("duplicate_failure_rate", dup_events / n_ep, "per episode",
               detail={"events": int(dup_events), "episodes": n_ep})
        if n_ep
        else _unavailable("duplicate_failure_rate", "no episodes", "per episode")
    )

    # --- provenance ----------------------------------------------------
    relation_count = session.execute(
        select(func.count(ArtifactRelation.id))
        .join(Artifact, Artifact.id == ArtifactRelation.source_id)
        .where(Artifact.workspace_id == workspace_id)
    ).scalar_one()
    metrics["provenance_edges"] = Metric("provenance_edges", int(relation_count), "edges")
    metrics["mean_provenance_depth"] = (
        Metric("mean_provenance_depth", _mean_depth(session, active), "hops")
        if active
        else _unavailable("mean_provenance_depth", "no active artifacts", "hops")
    )

    validated = [
        a for a in active
        if a.validation_state in (
            ValidationState.EVALUATOR_CONFIRMED, ValidationState.HUMAN_CONFIRMED,
            ValidationState.REPRODUCED, ValidationState.TOOL_VERIFIED,
        )
    ]
    metrics["validated_share"] = (
        Metric("validated_share", len(validated) / len(active), "fraction")
        if active
        else _unavailable("validated_share", "no active artifacts", "fraction")
    )

    # --- utility and tools ---------------------------------------------
    credited = [a for a in active if a.downstream_utility != 0.0]
    metrics["artifact_utility_mean"] = (
        Metric("artifact_utility_mean",
               sum(a.downstream_utility for a in credited) / len(credited), "credit",
               detail={"n_credited": len(credited)})
        if credited
        else _unavailable(
            "artifact_utility_mean",
            "no artifact has received outcome-derived credit yet", "credit",
        )
    )

    tools = list(
        session.execute(
            select(ToolDefinition).where(ToolDefinition.workspace_id == workspace_id)
        ).scalars()
    )
    runs_by_tool = dict(
        session.execute(
            select(ToolRun.tool_definition_id, func.count(ToolRun.id)).group_by(
                ToolRun.tool_definition_id
            )
        ).all()
    )
    reused = [t for t in tools if runs_by_tool.get(t.id, 0) > 1]
    metrics["tool_reuse_rate"] = (
        Metric("tool_reuse_rate", len(reused) / len(tools), "fraction",
               detail={"tools": len(tools), "reused": len(reused)})
        if tools
        else _unavailable("tool_reuse_rate", "no tools exist in this workspace", "fraction")
    )

    tool_runs = list(
        session.execute(
            select(ToolRun).where(ToolRun.episode_id.in_(episode_ids or {uuid.uuid4()}))
        ).scalars()
    ) if episode_ids else []
    metrics["tool_success_rate"] = (
        Metric("tool_success_rate", sum(1 for r in tool_runs if r.succeeded) / len(tool_runs),
               "fraction", detail={"runs": len(tool_runs)})
        if tool_runs
        else _unavailable("tool_success_rate", "no tool runs recorded", "fraction")
    )

    # --- retrieval -----------------------------------------------------
    decisions = list(
        session.execute(
            select(RetrievalDecision).where(RetrievalDecision.workspace_id == workspace_id)
        ).scalars()
    )
    if decisions:
        returned = [d.returned_count for d in decisions]
        metrics["retrieval_mean_returned"] = Metric(
            "retrieval_mean_returned", sum(returned) / len(returned), "artifacts",
            detail={"decisions": len(decisions)},
        )
        metrics["retrieval_mean_latency_ms"] = Metric(
            "retrieval_mean_latency_ms",
            sum(d.latency_ms for d in decisions) / len(decisions), "ms",
        )
        # Precision needs a relevance judgement the platform does not yet have. Saying so is the
        # point: a made-up proxy would look like a measurement.
        metrics["retrieval_precision"] = _unavailable(
            "retrieval_precision",
            "requires relevance labels; the benchmark measures retrieval through outcome instead",
            "fraction",
        )
    else:
        for name in ("retrieval_mean_returned", "retrieval_mean_latency_ms", "retrieval_precision"):
            metrics[name] = _unavailable(name, "no retrieval has been performed")

    # --- specialization (Part B §26, §48) ------------------------------
    profiles = session.execute(
        select(func.count(distinct(Episode.agent_profile_id))).where(*ep_filter)
    ).scalar_one()
    metrics["distinct_profiles_used"] = Metric(
        "distinct_profiles_used", int(profiles), "profiles"
    )
    metrics["specialization_index"] = _unavailable(
        "specialization_index",
        "requires profile-learning from measured performance by work type (M7)",
        "index",
    )

    # --- not yet buildable ---------------------------------------------
    metrics["collective_capability_frontier"] = _unavailable(
        "collective_capability_frontier",
        "requires the graded benchmark families of M9",
        "difficulty",
    )
    metrics["cumulative_improvement"] = _unavailable(
        "cumulative_improvement",
        "requires the cumulative-culture benchmark of M9",
        "slope",
    )

    from civitas.experiments.credit import calibration

    cal = calibration(session, workspace_id)
    metrics["confidence_calibration"] = (
        Metric("confidence_calibration", cal["correlation"], "correlation",
               detail={"n": cal["n"]})
        if cal["correlation"] is not None
        else _unavailable("confidence_calibration", str(cal["reason"]), "correlation")
    )

    return metrics


def _mean_depth(session: Session, artifacts: list[Artifact], cap: int = 6) -> float:
    """Mean longest provenance chain, bounded.

    Bounded because a cycle in a user-supplied graph would otherwise not terminate, and because
    depth beyond a handful of hops does not change how a result should be read.
    """
    edges: dict[uuid.UUID, list[uuid.UUID]] = {}
    ids = {a.id for a in artifacts}
    for relation in session.execute(
        select(ArtifactRelation).where(ArtifactRelation.source_id.in_(ids or {uuid.uuid4()}))
    ).scalars() if ids else []:
        edges.setdefault(relation.source_id, []).append(relation.target_id)

    def depth(node: uuid.UUID, seen: frozenset[uuid.UUID], level: int = 0) -> int:
        if level >= cap or node not in edges:
            return level
        return max(
            (depth(child, seen | {node}, level + 1) for child in edges[node] if child not in seen),
            default=level,
        )

    if not artifacts:
        return 0.0
    return sum(depth(a.id, frozenset()) for a in artifacts) / len(artifacts)


def export(session: Session, *, workspace_id: uuid.UUID, include_probes: bool = False) -> dict:
    """Export form (§48: metrics must be queryable and exportable)."""
    collected = collect(session, workspace_id=workspace_id, include_probes=include_probes)
    return {
        "workspace_id": str(workspace_id),
        "include_probes": include_probes,
        "metrics": {name: metric.as_dict() for name, metric in collected.items()},
        "unavailable": sorted(
            name for name, metric in collected.items() if metric.value is None
        ),
    }
