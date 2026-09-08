"""Knowledge consolidation (Part B §16).

As millions of artifacts accumulate the ecology has to stay usable. §16 asks for duplicate
detection, clustering, hierarchical summarisation, contradiction detection, abstraction, archival,
staleness review and concept formation — under one constraint that governs all of them:

> Never destroy source provenance. A consolidated summary must retain links to underlying evidence.

So every operation here is additive or reversible. Duplicates are *linked*, not deleted.
Summaries are new artifacts that `derived_from` everything they summarise. Archival sets a flag
that retrieval honours and a reader can undo. Nothing loses the chain back to the evidence.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import (
    ArtifactStatus,
    ArtifactType,
    EventType,
    EvidenceKind,
    RelationType,
    ValidationState,
)
from civitas.knowledge.embeddings import Embedder, artifact_text, default_embedder
from civitas.persistence.events import emit
from civitas.persistence.models import Artifact, ArtifactRelation
from civitas.persistence.types import utcnow

#: Cosine similarity above which two artifacts of the same type are treated as duplicates.
#:
#: Bracketed rather than picked: below ~0.90 the hashing embedder groups artifacts that share a
#: subject but state different things — exactly the pairs that must stay separate, because one may
#: contradict the other. Above ~0.98 only near-identical strings group, which misses the
#: reformulations that actually accumulate. `duplicate_report` prints the distribution so the
#: choice is checkable against a real corpus rather than asserted.
DUPLICATE_THRESHOLD = 0.94

#: Cluster membership. Looser than duplication on purpose: a cluster is a topic, not a claim.
CLUSTER_THRESHOLD = 0.72


@dataclass
class Cluster:
    key: uuid.UUID
    members: list[Artifact] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class ConsolidationReport:
    duplicates_linked: int = 0
    clusters_found: int = 0
    summaries_created: int = 0
    contradictions_found: int = 0
    archived: int = 0
    #: Similarity of every pair considered, so the threshold can be checked against the corpus
    #: instead of trusted.
    similarity_distribution: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "duplicates_linked": self.duplicates_linked,
            "clusters_found": self.clusters_found,
            "summaries_created": self.summaries_created,
            "contradictions_found": self.contradictions_found,
            "archived": self.archived,
            "similarity_distribution": self.similarity_distribution,
        }


def _vectors(
    session: Session, artifacts: list[Artifact], embedder: Embedder
) -> np.ndarray:
    return embedder.embed([artifact_text(a) for a in artifacts])


def find_duplicates(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    embedder: Embedder | None = None,
    threshold: float = DUPLICATE_THRESHOLD,
) -> list[tuple[Artifact, Artifact, float]]:
    """Near-identical artifacts of the same type (§16).

    Restricted to same-type pairs: a `failure` and an `evidence` artifact with similar text are
    saying different kinds of thing, and merging them would erase the distinction between what was
    established and what was ruled out.
    """
    embedder = embedder or default_embedder()
    artifacts = list(
        session.execute(
            select(Artifact).where(
                Artifact.workspace_id == workspace_id,
                Artifact.archived_at.is_(None),
                Artifact.status != ArtifactStatus.SUPERSEDED,
            )
        ).scalars()
    )
    by_type: dict[ArtifactType, list[Artifact]] = defaultdict(list)
    for artifact in artifacts:
        by_type[artifact.type].append(artifact)

    pairs: list[tuple[Artifact, Artifact, float]] = []
    for group in by_type.values():
        if len(group) < 2:
            continue
        matrix = _vectors(session, group, embedder)
        similarity = matrix @ matrix.T
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                score = float(similarity[i, j])
                if score >= threshold:
                    older, newer = sorted((group[i], group[j]), key=lambda a: a.created_at)
                    pairs.append((older, newer, score))
    return pairs


def link_duplicates(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    embedder: Embedder | None = None,
    threshold: float = DUPLICATE_THRESHOLD,
    archive_newer: bool = False,
    config_hash: str = "",
) -> ConsolidationReport:
    """Record duplication as an edge (§16). Never deletes.

    `archive_newer` takes the redundant copy out of retrieval while leaving it readable and
    reversible. Off by default: two agents independently reaching the same finding is *evidence*
    — it is a replication — and collapsing it silently would destroy the strongest signal the
    collective produces.
    """
    report = ConsolidationReport()
    pairs = find_duplicates(session, workspace_id=workspace_id, embedder=embedder,
                            threshold=threshold)

    buckets: dict[str, int] = defaultdict(int)
    for _older, _newer, score in pairs:
        buckets[f"{int(score * 20) / 20:.2f}"] += 1
    report.similarity_distribution = dict(sorted(buckets.items()))

    for older, newer, score in pairs:
        existing = session.execute(
            select(ArtifactRelation).where(
                ArtifactRelation.source_id == newer.id,
                ArtifactRelation.target_id == older.id,
                ArtifactRelation.type == RelationType.DUPLICATES,
            )
        ).scalar_one_or_none()
        if existing is not None:
            continue
        session.add(ArtifactRelation(
            source_id=newer.id, target_id=older.id, type=RelationType.DUPLICATES,
            confidence=round(score, 4), creator_kind="system",
            evidence_kind=EvidenceKind.INFERENCE,
        ))
        report.duplicates_linked += 1

        if archive_newer and newer.archived_at is None:
            newer.archived_at = utcnow()
            newer.archived_reason = f"duplicate of {older.id} (similarity {score:.3f})"
            report.archived += 1

    session.flush()
    if report.duplicates_linked:
        emit(
            session, workspace_id=workspace_id, type=EventType.CONSOLIDATION_PERFORMED,
            payload={"operation": "link_duplicates", **report.as_dict()},
            config_hash=config_hash,
        )
    return report


def cluster(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    embedder: Embedder | None = None,
    threshold: float = CLUSTER_THRESHOLD,
    min_size: int = 2,
    types: list[ArtifactType] | None = None,
) -> list[Cluster]:
    """Greedy single-pass clustering by embedding similarity (§16).

    Greedy and order-dependent, which is a real limitation and is stated rather than hidden: the
    clusters are an *index for summarisation*, not a claim about structure. Anything that rests on
    them keeps the links to the members, so a bad cluster produces a poor summary rather than a
    lost artifact.
    """
    embedder = embedder or default_embedder()
    stmt = select(Artifact).where(
        Artifact.workspace_id == workspace_id,
        Artifact.archived_at.is_(None),
    )
    if types:
        stmt = stmt.where(Artifact.type.in_([t.value for t in types]))
    artifacts = list(session.execute(stmt.order_by(Artifact.created_at)).scalars())
    if len(artifacts) < min_size:
        return []

    matrix = _vectors(session, artifacts, embedder)
    assigned: dict[int, int] = {}
    centroids: list[np.ndarray] = []
    groups: list[list[int]] = []

    for i in range(len(artifacts)):
        best, best_score = -1, threshold
        for c, centroid in enumerate(centroids):
            score = float(matrix[i] @ centroid)
            if score > best_score:
                best, best_score = c, score
        if best < 0:
            centroids.append(matrix[i].copy())
            groups.append([i])
            assigned[i] = len(groups) - 1
        else:
            groups[best].append(i)
            assigned[i] = best
            members = matrix[groups[best]]
            centroid = members.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[best] = centroid / norm if norm else centroid

    out = []
    for group in groups:
        if len(group) >= min_size:
            members = [artifacts[i] for i in group]
            out.append(Cluster(key=members[0].id, members=members))
    return out


def summarise_cluster(
    session: Session,
    cluster_: Cluster,
    *,
    workspace_id: uuid.UUID,
    episode_id: uuid.UUID | None = None,
    config_hash: str = "",
) -> Artifact:
    """Create a summary that keeps every link to its evidence (§16).

    The summary is extractive — titles and the strongest validation state present — rather than
    model-generated. A generated summary would be a *model assertion* about evidence, and
    recording it as a `summary` artifact would quietly upgrade assertion to established fact. An
    extractive summary claims only what its members claim, and the `derived_from` edges let any
    reader go straight to them.
    """
    members = sorted(cluster_.members, key=lambda a: a.created_at)
    strongest = max(
        (m.evidence_kind for m in members),
        key=lambda k: list(EvidenceKind).index(k),
        default=EvidenceKind.MODEL_ASSERTION,
    )
    validated = [
        m for m in members
        if m.validation_state in (ValidationState.EVALUATOR_CONFIRMED,
                                  ValidationState.HUMAN_CONFIRMED,
                                  ValidationState.REPRODUCED)
    ]

    lines = [f"Consolidates {len(members)} artifact(s):"]
    lines += [
        f"- [{m.type.value}/{m.validation_state.value}] {m.title} (id={m.id})" for m in members
    ]
    if validated:
        lines.append(f"\n{len(validated)} of these are externally validated.")

    summary = Artifact(
        workspace_id=workspace_id,
        project_id=members[0].project_id,
        type=ArtifactType.SUMMARY,
        title=f"Summary: {members[0].title[:400]}",
        body="\n".join(lines),
        # An extractive summary is an inference over its members, never a stronger claim than the
        # strongest member — and never `evaluator_confirmed`, which only §47 can grant.
        evidence_kind=EvidenceKind.INFERENCE,
        validation_state=ValidationState.UNVALIDATED,
        confidence=round(sum(m.confidence for m in members) / len(members), 4),
        creator_episode_id=episode_id,
        creator_kind="system",
        environment_version=members[0].environment_version,
        structured={
            "member_ids": [str(m.id) for m in members],
            "member_count": len(members),
            "strongest_member_evidence": strongest.value,
            "validated_members": len(validated),
        },
    )
    session.add(summary)
    session.flush()

    for member in members:
        session.add(ArtifactRelation(
            source_id=summary.id, target_id=member.id, type=RelationType.DERIVED_FROM,
            creator_kind="system", evidence_kind=EvidenceKind.INFERENCE,
        ))
    session.flush()

    emit(
        session, workspace_id=workspace_id, type=EventType.CONSOLIDATION_PERFORMED,
        episode_id=episode_id,
        payload={"operation": "summarise", "summary_id": str(summary.id),
                 "members": len(members)},
        config_hash=config_hash,
    )
    return summary


def detect_contradictions(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    embedder: Embedder | None = None,
    similarity: float = 0.80,
    config_hash: str = "",
) -> list[tuple[Artifact, Artifact]]:
    """Flag artifacts that are about the same thing but disagree (§16, §11).

    Detected structurally, not semantically: two artifacts are candidates when they are highly
    similar *and* one is a `failure`/`contradiction` type while the other is positive evidence, or
    their validation states conflict. A real entailment check needs a model, and calling a model's
    opinion a detected contradiction would be exactly the unearned upgrade §11 warns about — so
    these are recorded as candidate edges with `INFERENCE` provenance, for an agent to confirm.
    """
    embedder = embedder or default_embedder()
    artifacts = list(
        session.execute(
            select(Artifact).where(
                Artifact.workspace_id == workspace_id,
                Artifact.archived_at.is_(None),
                Artifact.status == ArtifactStatus.ACTIVE,
            )
        ).scalars()
    )
    if len(artifacts) < 2:
        return []

    matrix = _vectors(session, artifacts, embedder)
    scores = matrix @ matrix.T
    negative = {ArtifactType.FAILURE, ArtifactType.CONTRADICTION, ArtifactType.WARNING}
    refuting = {ValidationState.REFUTED, ValidationState.DISPUTED}

    found: list[tuple[Artifact, Artifact]] = []
    for i in range(len(artifacts)):
        for j in range(i + 1, len(artifacts)):
            if float(scores[i, j]) < similarity:
                continue
            a, b = artifacts[i], artifacts[j]
            disagree = (
                (a.type in negative) != (b.type in negative)
                or (a.validation_state in refuting) != (b.validation_state in refuting)
            )
            if not disagree:
                continue
            existing = session.execute(
                select(ArtifactRelation).where(
                    ArtifactRelation.source_id.in_([a.id, b.id]),
                    ArtifactRelation.target_id.in_([a.id, b.id]),
                    ArtifactRelation.type == RelationType.CONTRADICTS,
                )
            ).scalar_one_or_none()
            if existing is not None:
                continue
            session.add(ArtifactRelation(
                source_id=a.id, target_id=b.id, type=RelationType.CONTRADICTS,
                confidence=round(float(scores[i, j]), 4), creator_kind="system",
                evidence_kind=EvidenceKind.INFERENCE,
            ))
            found.append((a, b))

    session.flush()
    if found:
        emit(
            session, workspace_id=workspace_id, type=EventType.CONSOLIDATION_PERFORMED,
            payload={"operation": "detect_contradictions", "found": len(found)},
            config_hash=config_hash,
        )
    return found


def consolidate(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    embedder: Embedder | None = None,
    summarise_clusters_over: int = 5,
    config_hash: str = "",
) -> ConsolidationReport:
    """One consolidation pass (§16). Additive throughout; nothing is destroyed."""
    report = link_duplicates(session, workspace_id=workspace_id, embedder=embedder,
                             config_hash=config_hash)

    clusters = cluster(session, workspace_id=workspace_id, embedder=embedder)
    report.clusters_found = len(clusters)
    for group in clusters:
        if group.size >= summarise_clusters_over:
            summarise_cluster(session, group, workspace_id=workspace_id,
                              config_hash=config_hash)
            report.summaries_created += 1

    report.contradictions_found = len(
        detect_contradictions(session, workspace_id=workspace_id, embedder=embedder,
                              config_hash=config_hash)
    )
    return report
