"""Artifact versioning and supersession (Part B §9, §11, §16).

> Artifacts must be versioned. Do not silently overwrite historical knowledge.

An edit writes the *previous* state to `ArtifactVersion` and moves the head. The version rows are
append-only (the M2 session guard enforces it), so the sequence of things that were believed
survives even when the belief changes.

Supersession is different from editing, and the distinction is load-bearing. Editing says "this
artifact now says something else"; supersession says "a *different* artifact replaces this one".
The second keeps both readable and both attributable, which is what a correction needs to be
auditable rather than a quiet rewrite.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import ArtifactStatus, EventType, RelationType, ValidationState
from civitas.persistence.events import emit
from civitas.persistence.models import Artifact, ArtifactRelation, ArtifactVersion
from civitas.persistence.types import utcnow


def revise(
    session: Session,
    artifact: Artifact,
    *,
    title: str | None = None,
    body: str | None = None,
    structured: dict | None = None,
    confidence: float | None = None,
    validation_state: ValidationState | None = None,
    episode_id: uuid.UUID | None = None,
    reason: str = "",
    config_hash: str = "",
) -> ArtifactVersion:
    """Write the current state to history, then apply the change to the head.

    The snapshot is taken *before* mutation. Taking it after would record the new state twice and
    lose the old one — which is exactly the silent overwrite §9 forbids, dressed as versioning.
    """
    snapshot = ArtifactVersion(
        artifact_id=artifact.id,
        version=artifact.version,
        title=artifact.title,
        body=artifact.body,
        structured=dict(artifact.structured or {}),
        confidence=artifact.confidence,
        validation_state=artifact.validation_state,
        created_by_episode_id=artifact.creator_episode_id,
        change_reason=reason,
    )
    session.add(snapshot)

    if title is not None:
        artifact.title = title[:500]
    if body is not None:
        artifact.body = body
    if structured is not None:
        artifact.structured = structured
    if confidence is not None:
        artifact.confidence = confidence
    if validation_state is not None:
        artifact.validation_state = validation_state
    artifact.version += 1
    session.flush()

    emit(
        session, workspace_id=artifact.workspace_id, type=EventType.ARTIFACT_VERSIONED,
        episode_id=episode_id,
        payload={"artifact_id": str(artifact.id), "version": artifact.version,
                 "reason": reason[:500]},
        config_hash=config_hash,
    )
    return snapshot


def supersede(
    session: Session,
    *,
    old: Artifact,
    new: Artifact,
    episode_id: uuid.UUID | None = None,
    reason: str = "",
    config_hash: str = "",
) -> ArtifactRelation:
    """Replace one artifact with another, keeping both readable (§9, §11).

    The superseded artifact stays retrievable and is *visibly* superseded — a reader following an
    old reference finds the claim and the fact that it was replaced, rather than a dead link or a
    silently different answer.
    """
    if old.id == new.id:
        raise ValueError("an artifact cannot supersede itself")

    old.status = ArtifactStatus.SUPERSEDED
    old.superseded_by_id = new.id
    old.meta = {
        **(old.meta or {}),
        "superseded_reason": reason,
        "superseded_at": utcnow().isoformat(),
    }

    relation = ArtifactRelation(
        source_id=new.id, target_id=old.id, type=RelationType.SUPERSEDES,
        creator_episode_id=episode_id, creator_kind="agent" if episode_id else "system",
        evidence_kind=new.evidence_kind,
    )
    session.add(relation)
    session.flush()

    emit(
        session, workspace_id=old.workspace_id, type=EventType.ARTIFACT_VERSIONED,
        episode_id=episode_id,
        payload={"artifact_id": str(old.id), "superseded_by": str(new.id),
                 "reason": reason[:500]},
        config_hash=config_hash,
    )
    return relation


def history(session: Session, artifact_id: uuid.UUID) -> list[ArtifactVersion]:
    """Every prior state, oldest first."""
    return list(
        session.execute(
            select(ArtifactVersion)
            .where(ArtifactVersion.artifact_id == artifact_id)
            .order_by(ArtifactVersion.version)
        ).scalars()
    )


def resolve_head(
    session: Session, artifact_id: uuid.UUID, *, max_hops: int = 16
) -> Artifact | None:
    """Follow `superseded_by` to the current head.

    Hop-capped: a supersession cycle is possible in an agent-built graph, and following one
    forever would hang the API. Returning the last artifact reached is more useful than raising —
    a caller gets a real artifact and can see it is superseded.
    """
    artifact = session.get(Artifact, artifact_id)
    seen: set[uuid.UUID] = set()
    for _ in range(max_hops):
        if artifact is None or artifact.superseded_by_id is None:
            return artifact
        if artifact.id in seen:
            return artifact
        seen.add(artifact.id)
        artifact = session.get(Artifact, artifact.superseded_by_id)
    return artifact
