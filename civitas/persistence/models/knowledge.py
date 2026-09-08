"""Artifacts, the artifact graph, retrieval decisions and usage links (Part B §9–§16, §A2.1)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from civitas.domain.enums import (
    ArtifactStatus,
    ArtifactType,
    EvidenceKind,
    HypothesisState,
    RelationType,
    ValidationState,
    Visibility,
)
from civitas.persistence.base import (
    Base,
    Immutable,
    Metadataed,
    SoftDeletable,
    Timestamped,
    UUIDPrimaryKey,
)
from civitas.persistence.types import EnumType, GUID, JSONVariant, UTCDateTime, Vector


class Artifact(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    """A unit of persistent collective knowledge (Part B §9).

    The head of a version chain: `body` is the current text and `ArtifactVersion` holds every
    prior one. Historical knowledge is never overwritten (§9), so an edit writes a version and
    moves the head — it does not mutate the past.
    """

    __tablename__ = "artifacts"
    __table_args__ = (
        Index("ix_artifacts_ws_type_status", "workspace_id", "type", "status"),
        Index("ix_artifacts_ws_created", "workspace_id", "created_at"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("projects.id", ondelete="SET NULL"), default=None, index=True
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("tasks.id", ondelete="SET NULL"), default=None, index=True
    )

    type: Mapped[ArtifactType] = mapped_column(EnumType(ArtifactType, 40), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    body: Mapped[str] = mapped_column(Text(), nullable=False, default="")

    # --- provenance (Part B §13) ----------------------------------------
    creator_episode_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("episodes.id", ondelete="SET NULL"), default=None, index=True
    )
    creator_kind: Mapped[str] = mapped_column(String(32), nullable=False, default="agent")
    creator_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    evidence_kind: Mapped[EvidenceKind] = mapped_column(
        EnumType(EvidenceKind, 40), nullable=False, default=EvidenceKind.MODEL_ASSERTION, index=True
    )

    # --- epistemic status (Part B §11) ----------------------------------
    #: The agent's own claim. Stored, but it ranks last in retrieval (Part A §A2.1): a
    #: self-reported number is the one signal an agent can inflate for free.
    confidence: Mapped[float] = mapped_column(Float(), nullable=False, default=0.5)
    validation_state: Mapped[ValidationState] = mapped_column(
        EnumType(ValidationState, 40), nullable=False, default=ValidationState.UNVALIDATED, index=True
    )
    status: Mapped[ArtifactStatus] = mapped_column(
        EnumType(ArtifactStatus, 32), nullable=False, default=ArtifactStatus.ACTIVE, index=True
    )
    hypothesis_state: Mapped[HypothesisState | None] = mapped_column(
        EnumType(HypothesisState, 32), default=None, index=True
    )
    visibility: Mapped[Visibility] = mapped_column(
        EnumType(Visibility, 32), nullable=False, default=Visibility.WORKSPACE
    )

    version: Mapped[int] = mapped_column(Integer(), nullable=False, default=1)
    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("artifacts.id", ondelete="SET NULL"), default=None, index=True
    )

    # --- environmental applicability and staleness (Part B §15) ---------
    environment_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    applicable_commit: Mapped[str | None] = mapped_column(String(100), default=None)
    dataset_version: Mapped[str | None] = mapped_column(String(100), default=None)
    tool_version: Mapped[str | None] = mapped_column(String(100), default=None)
    last_validated_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    #: Seconds after which the claim must be revalidated. NULL means "no expiry asserted", which
    #: is different from "valid forever" and is displayed as such.
    validity_horizon_s: Mapped[float | None] = mapped_column(Float(), default=None)
    is_stale: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False, index=True)
    invalidated_by_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)

    # --- retrieval statistics (Part B §9, §48) --------------------------
    times_retrieved: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    #: Retrieved and *actually read* are different numbers, and only the second one earns credit
    #: (Part A §A2.1). Keeping both is what makes retrieval quality measurable.
    times_read: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    last_retrieved_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)

    #: Denormalised fold of `ArtifactUtilityMetric` events, for ranking. The events are the truth;
    #: this is a cache and is rebuilt by `civitas.experiments.credit.recompute`.
    downstream_utility: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)

    structured: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)

    versions: Mapped[list[ArtifactVersion]] = relationship(
        back_populates="artifact", cascade="all, delete-orphan"
    )


class ArtifactVersion(Base, UUIDPrimaryKey, Timestamped, Metadataed, Immutable):
    """An immutable prior state of an artifact (Part B §9)."""

    __tablename__ = "artifact_versions"
    __table_args__ = (
        UniqueConstraint("artifact_id", "version", name="uq_artifact_version"),
    )

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer(), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    body: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    structured: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    confidence: Mapped[float] = mapped_column(Float(), nullable=False, default=0.5)
    validation_state: Mapped[ValidationState] = mapped_column(
        EnumType(ValidationState, 40), nullable=False, default=ValidationState.UNVALIDATED
    )
    created_by_episode_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    change_reason: Mapped[str] = mapped_column(Text(), nullable=False, default="")

    artifact: Mapped[Artifact] = relationship(back_populates="versions")


class ArtifactRelation(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """A typed, attributed edge (Part B §10).

    Edges carry their own creator, confidence and provenance because a claim that A supports B is
    itself a claim, and one that later evidence can weaken without touching either endpoint.
    """

    __tablename__ = "artifact_relations"
    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "type", name="uq_relation_triple"),
        Index("ix_relations_target_type", "target_id", "type"),
        Index("ix_relations_source_type", "source_id", "type"),
    )

    source_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    type: Mapped[RelationType] = mapped_column(EnumType(RelationType, 40), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float(), nullable=False, default=1.0)
    creator_episode_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None, index=True)
    creator_kind: Mapped[str] = mapped_column(String(32), nullable=False, default="agent")
    evidence_kind: Mapped[EvidenceKind] = mapped_column(
        EnumType(EvidenceKind, 40), nullable=False, default=EvidenceKind.MODEL_ASSERTION
    )


class ArtifactEmbedding(Base, UUIDPrimaryKey, Timestamped):
    """A vector for one artifact under one embedding model (Part B §14, §43).

    Keyed by model *and* dimension so a re-embedding under a new model does not silently replace
    vectors a previous experiment's retrieval was computed against.
    """

    __tablename__ = "artifact_embeddings"
    __table_args__ = (
        UniqueConstraint("artifact_id", "model", name="uq_embedding_artifact_model"),
    )

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    dim: Mapped[int] = mapped_column(Integer(), nullable=False)
    vector = mapped_column(Vector(), nullable=False)
    norm: Mapped[float] = mapped_column(Float(), nullable=False, default=1.0)


class ArtifactUsage(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """An artifact an episode **actually read** (Part A §A2.1).

    Distinct from retrieval: retrieval returned it, this says the episode consumed it. Credit
    propagates along these links and along nothing else, which is what stops an agent earning
    credit for a large retrieval it never looked at.
    """

    __tablename__ = "artifact_usages"
    __table_args__ = (
        UniqueConstraint("episode_id", "artifact_id", name="uq_usage_episode_artifact"),
        Index("ix_usage_artifact", "artifact_id"),
    )

    episode_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False
    )
    retrieval_decision_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    read_count: Mapped[int] = mapped_column(Integer(), nullable=False, default=1)
    rank: Mapped[int | None] = mapped_column(Integer(), default=None)
    #: Set when the artifact was injected because the episode was about to repeat a known failure
    #: (Part A §A2.3), so its effect can be measured separately from ordinary retrieval.
    surfaced_as_duplicate_warning: Mapped[bool] = mapped_column(
        Boolean(), nullable=False, default=False
    )


class ArtifactUtilityMetric(Base, UUIDPrimaryKey, Timestamped, Metadataed, Immutable):
    """An immutable credit-assignment event (Part A §A2.1, Part B §29).

    Never self-reported and never updated. `Artifact.downstream_utility` is the fold of these
    rows; deleting or editing one would make that fold unreproducible, so the table is append-only
    and the session guard enforces it.
    """

    __tablename__ = "artifact_utility_metrics"
    __table_args__ = (Index("ix_utility_artifact_created", "artifact_id", "created_at"),)

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: The episode whose evaluation produced the credit, not the episode that used the artifact.
    source_episode_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None, index=True)
    evaluation_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None, index=True)
    delta: Mapped[float] = mapped_column(Float(), nullable=False)
    #: Edges traversed from the used artifact. Credit decays with this (Part A §A2.1).
    graph_distance: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    reason: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")


class RetrievalDecision(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """Every retrieval, with the features that produced the ranking (Part B §14).

    §14 requires the query, policy version, candidates, scores, selection and ranking features to
    be recorded. Without the features a retrieval policy cannot be evaluated after the fact, and
    the A2.2 gate would have nothing to compare versions on.
    """

    __tablename__ = "retrieval_decisions"

    episode_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("episodes.id", ondelete="CASCADE"), default=None, index=True
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    query: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    policy_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    experiment_arm: Mapped[str] = mapped_column(String(40), nullable=False, default="", index=True)
    candidate_count: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    returned_count: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    #: [{artifact_id, rank, score, features:{...}}] for every returned candidate.
    ranking: Mapped[list] = mapped_column(JSONVariant(), default=list, nullable=False)
    #: Candidates the arm removed, with the reason. The ablation arms are only auditable if what
    #: they suppressed is recorded rather than merely absent.
    suppressed: Mapped[list] = mapped_column(JSONVariant(), default=list, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)


class FileAsset(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    """A large payload in object storage, referenced from the database (Part B §44)."""

    __tablename__ = "file_assets"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("artifacts.id", ondelete="SET NULL"), default=None, index=True
    )
    episode_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None, index=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    content_type: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    size_bytes: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    #: Content-addressed, so an identical payload written twice is one object and a corrupted
    #: read is detectable rather than merely suspected.
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    storage_key: Mapped[str] = mapped_column(String(1000), nullable=False)
    storage_backend: Mapped[str] = mapped_column(String(32), nullable=False, default="local")
