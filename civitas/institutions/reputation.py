"""Multidimensional reputation (Part B §29).

> Do not collapse trust into one universal score. [...] Trust can influence retrieval and
> assignments. It must not make dissent impossible.

Both halves are enforced here.

**Multidimensional** because a scalar makes an agent that is unreliable-but-correct
indistinguishable from one that is reliable-but-wrong, and because a single number is the fastest
route to a collective that cannot disagree with its own best-reputed member. Eleven dimensions,
each computed from a different fold over outcome events, all stored as append-only observations.

**Not silencing** because reputation is bounded in how far it can move a retrieval ranking, and
because a *low*-reputation artifact is never removed — only ranked lower. §29's warning is not
decoration: a trust signal strong enough to suppress dissent is a mechanism for consensus by
authority, which §28 exists to prevent.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import (
    EVIDENCE_STRENGTH,
    ArtifactType,
    EventType,
    TerminationReason,
    ValidationState,
)
from civitas.persistence.events import emit
from civitas.persistence.models import (
    AgentProfile,
    Artifact,
    Episode,
    Evaluation,
    ReputationMetric,
    ToolDefinition,
)

#: The eleven dimensions §29 names. A closed set: an unlisted dimension is a bug, because the
#: point of the list is that trust is *not* one number and adding an unnamed twelfth silently
#: would reintroduce a hidden aggregate.
DIMENSIONS = (
    "predictive_accuracy",
    "calibration",
    "evidence_quality",
    "reproducibility",
    "downstream_usefulness",
    "false_positive_rate",
    "tool_creation",
    "tool_reliability",
    "efficiency",
    "correction_rate",
    "contradiction_resolution",
)

#: How far reputation may move a retrieval score, at most. Bounded on purpose (§29): trust
#: influences ranking, and a trust term large enough to bury a dissenting artifact would make
#: dissent impossible rather than merely less prominent.
MAX_RETRIEVAL_INFLUENCE = 0.25

#: Below this many observations a dimension is reported but marked unconfident, and it does not
#: influence ranking. Reputation earned on two episodes is an accident.
MIN_OBSERVATIONS = 5


@dataclass
class Dimension:
    name: str
    value: float | None
    n: int
    confident: bool
    unavailable_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value, "n": self.n, "confident": self.confident}
        if self.unavailable_reason:
            out["unavailable"] = self.unavailable_reason
        return out


@dataclass
class ReputationProfile:
    subject_kind: str
    subject_id: uuid.UUID
    dimensions: dict[str, Dimension] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject_kind": self.subject_kind,
            "subject_id": str(self.subject_id),
            "dimensions": {k: v.as_dict() for k, v in self.dimensions.items()},
        }

    def confident_mean(self) -> float | None:
        """Mean of the confident dimensions — **for display only**.

        Deliberately not used for ranking or assignment. §29 forbids collapsing trust into one
        universal score, and this exists so a dashboard can sort a list without any decision in
        the system depending on the collapse.
        """
        values = [d.value for d in self.dimensions.values() if d.confident and d.value is not None]
        return round(sum(values) / len(values), 4) if values else None


def _unavailable(name: str, reason: str) -> Dimension:
    return Dimension(name=name, value=None, n=0, confident=False, unavailable_reason=reason)


def compute_profile(
    session: Session, *, subject_kind: str, subject_id: uuid.UUID, workspace_id: uuid.UUID
) -> ReputationProfile:
    """Fold outcome events into the eleven dimensions (§29)."""
    profile = ReputationProfile(subject_kind=subject_kind, subject_id=subject_id)

    if subject_kind == "agent_profile":
        _agent_dimensions(session, profile, workspace_id)
    elif subject_kind == "tool":
        _tool_dimensions(session, profile)
    else:
        for name in DIMENSIONS:
            profile.dimensions[name] = _unavailable(
                name, f"no reputation model for subject kind {subject_kind!r}"
            )
    return profile


def _agent_dimensions(
    session: Session, profile: ReputationProfile, workspace_id: uuid.UUID
) -> None:
    episodes = list(
        session.execute(
            select(Episode).where(
                Episode.workspace_id == workspace_id,
                Episode.agent_profile_id == profile.subject_id,
                Episode.is_benchmark_probe.is_(False),
            )
        ).scalars()
    )
    episode_ids = {e.id for e in episodes}
    evaluations = (
        list(
            session.execute(
                select(Evaluation).where(
                    Evaluation.episode_id.in_(episode_ids), Evaluation.scope == "episode"
                )
            ).scalars()
        )
        if episode_ids
        else []
    )
    readable = [e for e in evaluations if "infrastructure_failure" not in (e.detail or {})]

    def add(name: str, value: float | None, n: int, reason: str | None = None) -> None:
        profile.dimensions[name] = (
            Dimension(name=name, value=value, n=n, confident=n >= MIN_OBSERVATIONS)
            if value is not None
            else _unavailable(name, reason or "no observations")
        )

    add(
        "predictive_accuracy",
        sum(1 for e in readable if e.succeeded) / len(readable) if readable else None,
        len(readable),
        "no externally evaluated episode",
    )

    artifacts = (
        list(
            session.execute(
                select(Artifact).where(Artifact.creator_episode_id.in_(episode_ids))
            ).scalars()
        )
        if episode_ids
        else []
    )

    # Calibration: how well stated confidence tracked what happened. |confidence - outcome|
    # averaged and inverted, so 1.0 is perfectly calibrated.
    calibrated = [
        1.0 - abs(a.confidence - (1.0 if a.validation_state in (
            ValidationState.EVALUATOR_CONFIRMED, ValidationState.HUMAN_CONFIRMED,
            ValidationState.REPRODUCED,
        ) else 0.0))
        for a in artifacts
        if a.validation_state is not ValidationState.UNVALIDATED
    ]
    add("calibration", sum(calibrated) / len(calibrated) if calibrated else None,
        len(calibrated), "no artifact has been validated either way")

    strengths = [EVIDENCE_STRENGTH.get(a.evidence_kind, 0.0) for a in artifacts]
    add("evidence_quality", sum(strengths) / len(strengths) if strengths else None,
        len(strengths), "no artifacts created")

    reproduced = [
        a for a in artifacts
        if a.validation_state in (ValidationState.REPRODUCED, ValidationState.EVALUATOR_CONFIRMED)
    ]
    add("reproducibility", len(reproduced) / len(artifacts) if artifacts else None,
        len(artifacts), "no artifacts created")

    utilities = [a.downstream_utility for a in artifacts if a.downstream_utility != 0.0]
    add("downstream_usefulness",
        math.tanh(sum(utilities) / len(utilities)) if utilities else None,
        len(utilities), "no artifact has received outcome-derived credit")

    # False positives: confident claims that were refuted. Low is good, so it is stored as a rate
    # and read as one — not inverted into a virtue, because a rate you have to remember to flip
    # is a rate someone will eventually add to the others.
    confident_claims = [a for a in artifacts if a.confidence >= 0.7]
    refuted = [a for a in confident_claims if a.validation_state is ValidationState.REFUTED]
    add("false_positive_rate",
        len(refuted) / len(confident_claims) if confident_claims else None,
        len(confident_claims), "no confident claims made")

    tools = (
        list(
            session.execute(
                select(ToolDefinition).where(
                    ToolDefinition.created_by_episode_id.in_(episode_ids)
                )
            ).scalars()
        )
        if episode_ids
        else []
    )
    add("tool_creation", math.tanh(len(tools) / 3.0) if tools else None, len(tools),
        "no tools created")
    reliabilities = [t.success_rate for t in tools if t.success_rate is not None]
    add("tool_reliability",
        sum(reliabilities) / len(reliabilities) if reliabilities else None,
        len(reliabilities), "no tool by this profile has been run")

    solved = [
        e for e in episodes
        if e.termination_reason is TerminationReason.EVALUATOR_SUCCESS and e.tokens_used
    ]
    add("efficiency",
        1.0 - math.tanh(sum(e.tokens_used for e in solved) / len(solved) / 100_000)
        if solved else None,
        len(solved), "no successful episode to measure cost against")

    corrections = [a for a in artifacts if a.type is ArtifactType.CORRECTION]
    add("correction_rate",
        len(corrections) / len(artifacts) if artifacts else None,
        len(artifacts), "no artifacts created")

    contradictions = [a for a in artifacts if a.type is ArtifactType.CONTRADICTION]
    resolved = [a for a in contradictions if a.superseded_by_id is not None]
    add("contradiction_resolution",
        len(resolved) / len(contradictions) if contradictions else None,
        len(contradictions), "this profile has raised no contradictions")


def _tool_dimensions(session: Session, profile: ReputationProfile) -> None:
    tool = session.get(ToolDefinition, profile.subject_id)
    for name in DIMENSIONS:
        profile.dimensions[name] = _unavailable(name, "not applicable to a tool")
    if tool is None:
        return
    total = tool.times_succeeded + tool.times_failed
    profile.dimensions["tool_reliability"] = (
        Dimension("tool_reliability", tool.success_rate, total, total >= MIN_OBSERVATIONS)
        if tool.success_rate is not None
        else _unavailable("tool_reliability", "this tool has never run")
    )
    profile.dimensions["downstream_usefulness"] = Dimension(
        "downstream_usefulness", math.tanh(tool.downstream_utility), tool.times_used,
        tool.times_used >= MIN_OBSERVATIONS,
    )


def record(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    subject_kind: str,
    subject_id: uuid.UUID,
    config_hash: str = "",
) -> ReputationProfile:
    """Compute the profile and append it as immutable observations (§29).

    Append-only rows rather than a running total, so a reputation is always reconstructable from
    the evaluations that produced it — the same discipline as artifact utility (§A2.1).
    """
    profile = compute_profile(
        session, subject_kind=subject_kind, subject_id=subject_id, workspace_id=workspace_id
    )
    written = 0
    for name, dimension in profile.dimensions.items():
        if dimension.value is None:
            continue
        session.add(ReputationMetric(
            workspace_id=workspace_id, subject_kind=subject_kind, subject_id=subject_id,
            dimension=name, value=dimension.value, sample_size=dimension.n,
            config_hash=config_hash,
        ))
        written += 1
    session.flush()
    if written:
        emit(
            session, workspace_id=workspace_id, type=EventType.REPUTATION_UPDATED,
            payload={"subject_kind": subject_kind, "subject_id": str(subject_id),
                     "dimensions_written": written},
            config_hash=config_hash,
        )
    return profile


def retrieval_influence(profile: ReputationProfile) -> float:
    """How much this subject's reputation may move a retrieval score (§29).

    Bounded by `MAX_RETRIEVAL_INFLUENCE` and built only from *confident* dimensions, and only
    those bearing on whether a claim is likely to hold. Deliberately small: §29 says trust may
    influence retrieval and must not make dissent impossible, and those two are only compatible
    if the influence cannot outweigh the evidence terms.
    """
    relevant = ["predictive_accuracy", "calibration", "evidence_quality", "reproducibility"]
    values = [
        profile.dimensions[name].value
        for name in relevant
        if name in profile.dimensions
        and profile.dimensions[name].confident
        and profile.dimensions[name].value is not None
    ]
    if not values:
        return 0.0
    # Centred on 0.5 so an average reputation is neutral: a subject nobody has reason to distrust
    # should not be penalised relative to one with no record at all.
    centred = sum(values) / len(values) - 0.5
    return round(max(-1.0, min(1.0, centred * 2.0)) * MAX_RETRIEVAL_INFLUENCE, 4)


def workspace_reputations(
    session: Session, workspace_id: uuid.UUID
) -> list[ReputationProfile]:
    """Every agent profile's reputation in a workspace."""
    return [
        record(session, workspace_id=workspace_id, subject_kind="agent_profile",
               subject_id=p.id)
        for p in session.execute(
            select(AgentProfile).where(AgentProfile.workspace_id == workspace_id)
        ).scalars()
    ]
