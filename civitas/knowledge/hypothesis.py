"""The hypothesis state machine and verification requirements (Part B §28).

> Do not let consensus form solely because many models copy the first answer.

Two mechanisms, both enforced here rather than requested in a prompt:

1. **A hypothesis moves between states only on evidence**, and the transitions are a closed graph.
   Nothing can jump from `proposed` to `accepted` because an agent said so.
2. **Important claims need stronger verification.** `verification_requirement` scales what a
   hypothesis must clear with what rests on it, so a load-bearing claim cannot be accepted on the
   same evidence as a trivial one.

Replication is the sharpest of these. `accepted` requires an *independent* reproduction — a
different episode — because a claim confirmed only by the episode that produced it is a claim
confirmed by its author.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import (
    ArtifactType,
    EventType,
    EvidenceKind,
    HypothesisState,
    RelationType,
    ValidationState,
)
from civitas.persistence.events import emit
from civitas.persistence.models import Artifact, ArtifactRelation

#: The closed transition graph of §28. A transition not listed here cannot happen.
TRANSITIONS: dict[HypothesisState, frozenset[HypothesisState]] = {
    HypothesisState.PROPOSED: frozenset({
        HypothesisState.TESTED, HypothesisState.CHALLENGED, HypothesisState.SUPERSEDED,
        HypothesisState.REJECTED,
    }),
    HypothesisState.TESTED: frozenset({
        HypothesisState.SUPPORTED, HypothesisState.REJECTED, HypothesisState.CHALLENGED,
        HypothesisState.SUPERSEDED,
    }),
    HypothesisState.SUPPORTED: frozenset({
        HypothesisState.REPLICATED, HypothesisState.CHALLENGED, HypothesisState.REJECTED,
        HypothesisState.SUPERSEDED,
    }),
    HypothesisState.CHALLENGED: frozenset({
        # A challenge can be answered: back to tested, and the evidence has to be redone.
        HypothesisState.TESTED, HypothesisState.SUPPORTED, HypothesisState.REJECTED,
        HypothesisState.SUPERSEDED,
    }),
    HypothesisState.REPLICATED: frozenset({
        HypothesisState.ACCEPTED, HypothesisState.CHALLENGED, HypothesisState.SUPERSEDED,
    }),
    # Accepted is not terminal. A record that could not withdraw an accepted claim would be a
    # record that cannot self-correct (§11).
    HypothesisState.ACCEPTED: frozenset({
        HypothesisState.CHALLENGED, HypothesisState.SUPERSEDED,
    }),
    HypothesisState.REJECTED: frozenset({
        HypothesisState.CHALLENGED, HypothesisState.SUPERSEDED,
    }),
    HypothesisState.SUPERSEDED: frozenset(),
}


class InvalidTransition(ValueError):
    """A state change the §28 graph does not allow."""


@dataclass
class VerificationRequirement:
    """What a hypothesis must clear before it can be accepted (§28)."""

    min_supporting: int
    min_independent_replications: int
    min_evidence_kind: EvidenceKind
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_supporting": self.min_supporting,
            "min_independent_replications": self.min_independent_replications,
            "min_evidence_kind": self.min_evidence_kind.value,
            "reason": self.reason,
        }


def importance(session: Session, hypothesis: Artifact) -> int:
    """How much rests on this hypothesis — the number of artifacts depending on it."""
    return len(
        list(
            session.execute(
                select(ArtifactRelation).where(
                    ArtifactRelation.target_id == hypothesis.id,
                    ArtifactRelation.type.in_([
                        RelationType.DERIVED_FROM.value, RelationType.DEPENDS_ON.value,
                        RelationType.SUPPORTS.value,
                    ]),
                )
            ).scalars()
        )
    )


def verification_requirement(
    session: Session, hypothesis: Artifact
) -> VerificationRequirement:
    """Stronger verification for claims more rests on (§28).

    Scaled by how much depends on the hypothesis, not by how confident its author is. Confidence
    is the one input an agent controls, and letting it set the bar would let a claim lower its own
    standard of proof.
    """
    dependents = importance(session, hypothesis)
    if dependents >= 5:
        return VerificationRequirement(
            min_supporting=3, min_independent_replications=2,
            min_evidence_kind=EvidenceKind.REPRODUCED_RESULT,
            reason=f"{dependents} artifacts depend on this claim",
        )
    if dependents >= 2:
        return VerificationRequirement(
            min_supporting=2, min_independent_replications=1,
            min_evidence_kind=EvidenceKind.TOOL_OUTPUT,
            reason=f"{dependents} artifacts depend on this claim",
        )
    return VerificationRequirement(
        min_supporting=1, min_independent_replications=1,
        min_evidence_kind=EvidenceKind.DIRECT_OBSERVATION,
        reason="nothing yet depends on this claim",
    )


@dataclass
class Assessment:
    supporting: int
    contradicting: int
    independent_replications: int
    strongest_evidence: EvidenceKind | None
    requirement: VerificationRequirement
    eligible_state: HypothesisState

    def as_dict(self) -> dict[str, Any]:
        return {
            "supporting": self.supporting,
            "contradicting": self.contradicting,
            "independent_replications": self.independent_replications,
            "strongest_evidence": (
                self.strongest_evidence.value if self.strongest_evidence else None
            ),
            "requirement": self.requirement.as_dict(),
            "eligible_state": self.eligible_state.value,
        }


def assess(session: Session, hypothesis: Artifact) -> Assessment:
    """What state the evidence currently justifies (§28).

    Independent replication counts *distinct episodes other than the hypothesis's creator*. A
    reproduction by the episode that proposed the claim is the author confirming themselves, and
    counting it would make replication a formality.
    """
    from civitas.domain.enums import EVIDENCE_STRENGTH

    incoming = list(
        session.execute(
            select(ArtifactRelation).where(ArtifactRelation.target_id == hypothesis.id)
        ).scalars()
    )
    supporting_ids: list[uuid.UUID] = []
    replication_episodes: set[uuid.UUID] = set()
    contradicting = 0

    for relation in incoming:
        if relation.type in (RelationType.SUPPORTS, RelationType.CONFIRMS):
            supporting_ids.append(relation.source_id)
        elif relation.type in (
            RelationType.CONTRADICTS, RelationType.FALSIFIES, RelationType.INVALIDATES
        ):
            contradicting += 1
        elif relation.type is RelationType.REPRODUCES:
            source = session.get(Artifact, relation.source_id)
            if source is not None and source.creator_episode_id is not None:
                if source.creator_episode_id != hypothesis.creator_episode_id:
                    replication_episodes.add(source.creator_episode_id)
                    supporting_ids.append(relation.source_id)

    strongest: EvidenceKind | None = None
    for artifact_id in supporting_ids:
        artifact = session.get(Artifact, artifact_id)
        if artifact is None:
            continue
        if strongest is None or EVIDENCE_STRENGTH.get(
            artifact.evidence_kind, 0.0
        ) > EVIDENCE_STRENGTH.get(strongest, 0.0):
            strongest = artifact.evidence_kind

    requirement = verification_requirement(session, hypothesis)
    supporting = len(set(supporting_ids))
    replications = len(replication_episodes)

    if contradicting and not supporting:
        eligible = HypothesisState.REJECTED
    elif contradicting:
        eligible = HypothesisState.CHALLENGED
    elif (
        supporting >= requirement.min_supporting
        and replications >= requirement.min_independent_replications
        and strongest is not None
        and EVIDENCE_STRENGTH.get(strongest, 0.0)
        >= EVIDENCE_STRENGTH.get(requirement.min_evidence_kind, 0.0)
    ):
        eligible = HypothesisState.ACCEPTED
    elif replications >= 1 and supporting >= requirement.min_supporting:
        eligible = HypothesisState.REPLICATED
    elif supporting >= requirement.min_supporting:
        eligible = HypothesisState.SUPPORTED
    elif supporting or replications:
        eligible = HypothesisState.TESTED
    else:
        eligible = HypothesisState.PROPOSED

    return Assessment(
        supporting=supporting, contradicting=contradicting,
        independent_replications=replications, strongest_evidence=strongest,
        requirement=requirement, eligible_state=eligible,
    )


def transition(
    session: Session,
    hypothesis: Artifact,
    to_state: HypothesisState,
    *,
    episode_id: uuid.UUID | None = None,
    force: bool = False,
    config_hash: str = "",
) -> HypothesisState:
    """Move a hypothesis, refusing a transition the graph does not allow (§28)."""
    if hypothesis.type is not ArtifactType.HYPOTHESIS:
        raise ValueError(f"{hypothesis.id} is a {hypothesis.type.value}, not a hypothesis")

    current = hypothesis.hypothesis_state or HypothesisState.PROPOSED
    if current is to_state:
        return current
    if not force and to_state not in TRANSITIONS[current]:
        raise InvalidTransition(
            f"{current.value} -> {to_state.value} is not a permitted transition (Part B §28)"
        )

    hypothesis.hypothesis_state = to_state
    if to_state is HypothesisState.ACCEPTED:
        # Accepted is a statement about *evidence*, and it is the only state that raises the
        # artifact's validation. Nothing an agent asserts can produce it.
        hypothesis.validation_state = ValidationState.REPRODUCED
    elif to_state is HypothesisState.REJECTED:
        hypothesis.validation_state = ValidationState.REFUTED
    elif to_state is HypothesisState.CHALLENGED:
        hypothesis.validation_state = ValidationState.DISPUTED
    session.flush()

    emit(
        session, workspace_id=hypothesis.workspace_id,
        type=EventType.HYPOTHESIS_CHALLENGED
        if to_state is HypothesisState.CHALLENGED
        else EventType.ARTIFACT_VERSIONED,
        episode_id=episode_id,
        payload={"artifact_id": str(hypothesis.id), "from": current.value,
                 "to": to_state.value},
        config_hash=config_hash,
    )
    return to_state


def advance(
    session: Session,
    hypothesis: Artifact,
    *,
    episode_id: uuid.UUID | None = None,
    config_hash: str = "",
) -> tuple[HypothesisState, Assessment]:
    """Move a hypothesis to whatever state the evidence justifies, one legal step at a time.

    Stepwise rather than jumping straight to the eligible state, so the record shows the path a
    claim actually took. A hypothesis that went `proposed -> accepted` in one write would be
    indistinguishable from one that had been tested, supported and replicated.
    """
    assessment = assess(session, hypothesis)
    current = hypothesis.hypothesis_state or HypothesisState.PROPOSED
    target = assessment.eligible_state

    ladder = [
        HypothesisState.PROPOSED, HypothesisState.TESTED, HypothesisState.SUPPORTED,
        HypothesisState.REPLICATED, HypothesisState.ACCEPTED,
    ]
    guard = 0
    while current is not target and guard < 8:
        guard += 1
        if target in TRANSITIONS[current]:
            current = transition(session, hypothesis, target, episode_id=episode_id,
                                 config_hash=config_hash)
            break
        if current in ladder and target in ladder:
            index = ladder.index(current)
            step = ladder[index + 1] if ladder.index(target) > index else ladder[index - 1]
            if step not in TRANSITIONS[current]:
                break
            current = transition(session, hypothesis, step, episode_id=episode_id,
                                 config_hash=config_hash)
        else:
            break
    return current, assessment
