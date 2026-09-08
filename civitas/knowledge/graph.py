"""Artifact graph traversal and provenance (Part B §10, §13).

§10 requires graph traversal APIs; §13 requires every important result to be traceable to the
evidence it rests on, with the *kind* of evidence distinguished — a model assertion is not an
observation, and no quantity of assertions sums to one.

Traversal is breadth-first with an explicit depth cap and a visited set. Both are load-bearing: an
agent-built graph can contain cycles, and a workspace can hold millions of artifacts (§59), so an
unbounded walk is a way to hang the API rather than a way to be thorough.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from civitas.domain.enums import (
    EVIDENCE_STRENGTH,
    ArtifactStatus,
    EvidenceKind,
    RelationType,
    ValidationState,
)
from civitas.persistence.models import Artifact, ArtifactRelation

#: Relations that answer "what does this rest on?" — the direction provenance runs.
SUPPORTING = frozenset({
    RelationType.DERIVED_FROM,
    RelationType.DEPENDS_ON,
    RelationType.USES,
    RelationType.SUPPORTS,
    RelationType.CITES,
    RelationType.CONFIRMS,
    RelationType.REPRODUCES,
    RelationType.TESTS,
    RelationType.ENABLED,
})

#: Relations that weaken. Traversed separately, and *reported*, because a provenance chain that
#: silently omits what contradicts it is an argument rather than a record (§11).
OPPOSING = frozenset({
    RelationType.CONTRADICTS,
    RelationType.FALSIFIES,
    RelationType.INVALIDATES,
    RelationType.SUPERSEDES,
})

MAX_DEPTH = 8


@dataclass
class Node:
    artifact: Artifact
    depth: int
    via: RelationType | None = None
    from_id: uuid.UUID | None = None


@dataclass
class ProvenanceChain:
    """A traceable path from a result to what it rests on (§13)."""

    root: Artifact
    nodes: list[Node] = field(default_factory=list)
    #: Artifacts that contradict, falsify, invalidate or supersede something in the chain.
    challenges: list[Node] = field(default_factory=list)
    truncated: bool = False

    @property
    def depth(self) -> int:
        return max((n.depth for n in self.nodes), default=0)

    def strongest_evidence(self) -> EvidenceKind | None:
        """The strongest evidence kind anywhere in the chain.

        This is what §13's distinction is *for*: a conclusion whose deepest support is a model
        assertion is a different object from one grounded in a reproduced result, and the chain
        should be able to say which without a human reading every node.
        """
        kinds = [n.artifact.evidence_kind for n in self.nodes]
        return max(kinds, key=lambda k: EVIDENCE_STRENGTH.get(k, 0.0)) if kinds else None

    def grounding_score(self) -> float:
        """How well grounded the root is, in [0, 1].

        The strongest evidence in the chain, discounted by distance — evidence five hops away
        supports the root less than the same evidence directly attached — and reduced where the
        chain is challenged. Deliberately not an average: one reproduced result grounds a claim,
        and averaging it against nine assertions would hide that.
        """
        if not self.nodes:
            return EVIDENCE_STRENGTH.get(self.root.evidence_kind, 0.0)
        best = 0.0
        for node in self.nodes:
            strength = EVIDENCE_STRENGTH.get(node.artifact.evidence_kind, 0.0)
            best = max(best, strength * (0.85**node.depth))
        if self.challenges:
            best *= 0.6
        return round(best, 4)

    def render(self, *, width: int = 72) -> str:
        """The §13 example format: a result, and what it rests on, indented by depth."""
        lines = [f"{self.root.type.value}: {self.root.title[:width]}"]
        for node in sorted(self.nodes, key=lambda n: (n.depth, n.artifact.created_at)):
            indent = "  " * node.depth
            via = f"<- {node.via.value} " if node.via else ""
            lines.append(
                f"{indent}{via}{node.artifact.type.value} "
                f"[{node.artifact.evidence_kind.value}] {node.artifact.title[:width]}"
            )
        for node in self.challenges:
            lines.append(f"  ! {node.via.value if node.via else 'challenged'}: "
                         f"{node.artifact.title[:width]}")
        if self.truncated:
            lines.append(f"  ... truncated at depth {MAX_DEPTH}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": {"id": str(self.root.id), "type": self.root.type.value,
                     "title": self.root.title},
            "depth": self.depth,
            "grounding_score": self.grounding_score(),
            "strongest_evidence": (
                self.strongest_evidence().value if self.strongest_evidence() else None
            ),
            "truncated": self.truncated,
            "nodes": [
                {"id": str(n.artifact.id), "depth": n.depth,
                 "via": n.via.value if n.via else None,
                 "type": n.artifact.type.value,
                 "evidence_kind": n.artifact.evidence_kind.value,
                 "validation_state": n.artifact.validation_state.value,
                 "title": n.artifact.title}
                for n in self.nodes
            ],
            "challenges": [
                {"id": str(n.artifact.id), "via": n.via.value if n.via else None,
                 "title": n.artifact.title}
                for n in self.challenges
            ],
        }


def neighbours(
    session: Session,
    artifact_id: uuid.UUID,
    *,
    types: frozenset[RelationType] | None = None,
    outgoing: bool = True,
) -> list[tuple[ArtifactRelation, Artifact]]:
    """One hop (§10). `outgoing` follows source→target; otherwise target→source."""
    stmt = select(ArtifactRelation).where(
        ArtifactRelation.source_id == artifact_id if outgoing
        else ArtifactRelation.target_id == artifact_id
    )
    if types:
        stmt = stmt.where(ArtifactRelation.type.in_([t.value for t in types]))

    out: list[tuple[ArtifactRelation, Artifact]] = []
    for relation in session.execute(stmt).scalars():
        other_id = relation.target_id if outgoing else relation.source_id
        other = session.get(Artifact, other_id)
        if other is not None:
            out.append((relation, other))
    return out


def traverse(
    session: Session,
    start_id: uuid.UUID,
    *,
    types: frozenset[RelationType] | None = None,
    outgoing: bool = True,
    max_depth: int = 3,
    limit: int = 200,
) -> list[Node]:
    """Breadth-first traversal with a depth cap and a visited set (§10)."""
    start = session.get(Artifact, start_id)
    if start is None:
        return []

    seen = {start_id}
    frontier: deque[tuple[uuid.UUID, int]] = deque([(start_id, 0)])
    out: list[Node] = []

    while frontier and len(out) < limit:
        current, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        for relation, other in neighbours(session, current, types=types, outgoing=outgoing):
            if other.id in seen:
                continue
            seen.add(other.id)
            out.append(Node(artifact=other, depth=depth + 1, via=relation.type,
                            from_id=current))
            frontier.append((other.id, depth + 1))
    return out


def provenance(
    session: Session,
    artifact_id: uuid.UUID,
    *,
    max_depth: int = MAX_DEPTH,
    limit: int = 200,
) -> ProvenanceChain | None:
    """Trace a result to the evidence it rests on, and to what challenges it (§13)."""
    root = session.get(Artifact, artifact_id)
    if root is None:
        return None

    nodes = traverse(session, artifact_id, types=SUPPORTING, outgoing=True,
                     max_depth=max_depth, limit=limit)
    chain = ProvenanceChain(root=root, nodes=nodes, truncated=len(nodes) >= limit)

    # Challenges point *at* the chain, so they are found on incoming edges. A record that shows
    # only what supports a claim is an argument (§11).
    in_chain = {artifact_id} | {n.artifact.id for n in nodes}
    for member in in_chain:
        for relation, other in neighbours(session, member, types=OPPOSING, outgoing=False):
            if other.id not in in_chain:
                chain.challenges.append(Node(artifact=other, depth=0, via=relation.type,
                                             from_id=member))
    return chain


def descendants(session: Session, artifact_id: uuid.UUID, *, max_depth: int = 3) -> list[Node]:
    """What rests on this artifact — the direction credit and invalidation propagate."""
    return traverse(session, artifact_id, types=SUPPORTING, outgoing=False, max_depth=max_depth)


def contradictions(
    session: Session, workspace_id: uuid.UUID
) -> list[tuple[Artifact, Artifact, ArtifactRelation]]:
    """Every open contradiction in a workspace (§11, §16).

    Open means both sides are still active: once one is superseded or retracted the disagreement
    is resolved, and listing it would make a resolved dispute look live.
    """
    out = []
    stmt = (
        select(ArtifactRelation)
        .join(Artifact, Artifact.id == ArtifactRelation.source_id)
        .where(
            Artifact.workspace_id == workspace_id,
            ArtifactRelation.type.in_([RelationType.CONTRADICTS.value,
                                       RelationType.FALSIFIES.value]),
        )
    )
    for relation in session.execute(stmt).scalars():
        source = session.get(Artifact, relation.source_id)
        target = session.get(Artifact, relation.target_id)
        if source is None or target is None:
            continue
        if source.status is ArtifactStatus.ACTIVE and target.status is ArtifactStatus.ACTIVE:
            out.append((source, target, relation))
    return out


def graph_distance(
    session: Session, a: uuid.UUID, b: uuid.UUID, *, max_depth: int = 4
) -> int | None:
    """Shortest undirected hop count, or None beyond `max_depth`.

    None rather than a large number: "further than we looked" is not the same as "far", and a
    caller ranking on proximity must be able to tell the difference.
    """
    if a == b:
        return 0
    seen = {a}
    frontier: deque[tuple[uuid.UUID, int]] = deque([(a, 0)])
    while frontier:
        current, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        stmt = select(ArtifactRelation).where(
            or_(ArtifactRelation.source_id == current, ArtifactRelation.target_id == current)
        )
        for relation in session.execute(stmt).scalars():
            other = relation.target_id if relation.source_id == current else relation.source_id
            if other == b:
                return depth + 1
            if other not in seen:
                seen.add(other)
                frontier.append((other, depth + 1))
    return None


def validation_summary(session: Session, workspace_id: uuid.UUID) -> dict[str, int]:
    """How much of a workspace has actually been checked (§11)."""
    counts: dict[str, int] = {state.value: 0 for state in ValidationState}
    for artifact in session.execute(
        select(Artifact).where(
            Artifact.workspace_id == workspace_id, Artifact.archived_at.is_(None)
        )
    ).scalars():
        counts[artifact.validation_state.value] += 1
    return counts
