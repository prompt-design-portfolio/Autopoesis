"""Knowledge staleness (Part B §15).

Knowledge becomes obsolete. §15's requirement is precise and easy to get wrong in one direction:

> A stale artifact should not disappear. It should be visibly stale.

So nothing here deletes, archives or hides. Staleness is a *flag* that retrieval reads and
penalises, and that a reader can see. An artifact whose era has passed is still the record of what
was believed then, and removing it would make the history unreconstructable (§A1.2).

Three independent reasons an artifact goes stale, checked separately because they mean different
things:

* **environment drift** — it was established against a device, commit or dataset that is no longer
  current. This is the `π` case: not merely unhelpful but *actively misleading*, because it names a
  specific answer that has since changed.
* **expiry** — its stated validity horizon has passed without revalidation.
* **invalidation** — something now falsifies or invalidates it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from civitas.domain.enums import ArtifactStatus, EventType, RelationType
from civitas.persistence.events import emit
from civitas.persistence.models import Artifact, ArtifactRelation
from civitas.persistence.types import utcnow


@dataclass
class StalenessReport:
    checked: int = 0
    marked_stale: int = 0
    marked_fresh: int = 0
    by_reason: dict[str, int] | None = None

    def as_dict(self) -> dict:
        return {
            "checked": self.checked,
            "marked_stale": self.marked_stale,
            "marked_fresh": self.marked_fresh,
            "by_reason": self.by_reason or {},
        }


def staleness_reason(
    artifact: Artifact, *, environment_version: str | None, now=None
) -> str | None:
    """Why this artifact is stale, or None.

    Order matters: environment drift is reported ahead of expiry because it is the stronger
    statement. An artifact from a superseded environment is wrong about a world that no longer
    exists, whereas an expired one may simply be unconfirmed.
    """
    now = now or utcnow()

    if artifact.invalidated_by_id is not None:
        return "invalidated"

    if (
        environment_version
        and artifact.environment_version
        and artifact.environment_version != environment_version
    ):
        return "environment_drift"

    if artifact.validity_horizon_s:
        anchor = artifact.last_validated_at or artifact.created_at
        if now > anchor + timedelta(seconds=artifact.validity_horizon_s):
            return "expired"

    return None


def mark_workspace(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    environment_version: str | None = None,
    emit_events: bool = True,
    config_hash: str = "",
) -> StalenessReport:
    """Re-check every active artifact's staleness against the current environment.

    Bidirectional: an artifact can become fresh again — a claim revalidated against the current
    environment, or an invalidation that was itself retracted. One-way staleness would slowly
    condemn a whole workspace with no way back.
    """
    report = StalenessReport(by_reason={})
    for artifact in session.execute(
        select(Artifact).where(
            Artifact.workspace_id == workspace_id, Artifact.archived_at.is_(None)
        )
    ).scalars():
        report.checked += 1
        reason = staleness_reason(artifact, environment_version=environment_version)

        if reason and not artifact.is_stale:
            artifact.is_stale = True
            if artifact.status is ArtifactStatus.ACTIVE:
                artifact.status = ArtifactStatus.STALE
            artifact.meta = {**(artifact.meta or {}), "stale_reason": reason}
            report.marked_stale += 1
            report.by_reason[reason] = report.by_reason.get(reason, 0) + 1
            if emit_events:
                emit(
                    session, workspace_id=workspace_id, type=EventType.STALENESS_MARKED,
                    payload={"artifact_id": str(artifact.id), "reason": reason},
                    config_hash=config_hash,
                )
        elif not reason and artifact.is_stale:
            artifact.is_stale = False
            if artifact.status is ArtifactStatus.STALE:
                artifact.status = ArtifactStatus.ACTIVE
            meta = dict(artifact.meta or {})
            meta.pop("stale_reason", None)
            artifact.meta = meta
            report.marked_fresh += 1

    session.flush()
    return report


def invalidate(
    session: Session,
    *,
    artifact: Artifact,
    by: Artifact,
    relation: RelationType = RelationType.INVALIDATES,
    cascade: bool = True,
    config_hash: str = "",
) -> list[uuid.UUID]:
    """Record that `by` invalidates `artifact`, and optionally propagate.

    Cascading marks what *depended on* the invalidated claim as stale — not as false. A conclusion
    resting on a refuted premise is not thereby refuted; it is unsupported, and it needs
    re-examination rather than deletion. Marking it stale says exactly that, which is why the
    cascade sets staleness rather than a validation state.
    """
    artifact.invalidated_by_id = by.id
    artifact.is_stale = True
    artifact.status = ArtifactStatus.STALE
    artifact.meta = {**(artifact.meta or {}), "stale_reason": "invalidated"}

    session.add(ArtifactRelation(
        source_id=by.id, target_id=artifact.id, type=relation,
        creator_episode_id=by.creator_episode_id, creator_kind=by.creator_kind,
        evidence_kind=by.evidence_kind,
    ))
    session.flush()

    emit(
        session, workspace_id=artifact.workspace_id, type=EventType.STALENESS_MARKED,
        payload={"artifact_id": str(artifact.id), "reason": "invalidated",
                 "by": str(by.id)},
        config_hash=config_hash,
    )

    affected: list[uuid.UUID] = [artifact.id]
    if cascade:
        from civitas.knowledge.graph import descendants

        for node in descendants(session, artifact.id, max_depth=3):
            if node.artifact.is_stale:
                continue
            node.artifact.is_stale = True
            if node.artifact.status is ArtifactStatus.ACTIVE:
                node.artifact.status = ArtifactStatus.STALE
            node.artifact.meta = {
                **(node.artifact.meta or {}),
                "stale_reason": "depends_on_invalidated",
                "stale_source": str(artifact.id),
            }
            affected.append(node.artifact.id)
            emit(
                session, workspace_id=artifact.workspace_id,
                type=EventType.STALENESS_MARKED,
                payload={"artifact_id": str(node.artifact.id),
                         "reason": "depends_on_invalidated",
                         "source": str(artifact.id), "distance": node.depth},
                config_hash=config_hash,
            )
        session.flush()
    return affected


def revalidate(session: Session, artifact: Artifact, *, environment_version: str) -> None:
    """Record that a claim still holds in the current environment (§15)."""
    artifact.last_validated_at = utcnow()
    artifact.environment_version = environment_version
    artifact.is_stale = False
    if artifact.status is ArtifactStatus.STALE:
        artifact.status = ArtifactStatus.ACTIVE
    meta = dict(artifact.meta or {})
    meta.pop("stale_reason", None)
    meta.pop("stale_source", None)
    artifact.meta = meta
    session.flush()


def stale_artifacts(session: Session, workspace_id: uuid.UUID) -> list[Artifact]:
    """Everything currently flagged. Visible, not hidden (§15)."""
    return list(
        session.execute(
            select(Artifact).where(
                Artifact.workspace_id == workspace_id,
                Artifact.archived_at.is_(None),
                or_(Artifact.is_stale.is_(True), Artifact.status == ArtifactStatus.STALE),
            )
        ).scalars()
    )
