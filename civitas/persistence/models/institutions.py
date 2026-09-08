"""Procedures, policies, reputation and the experiment gate (Part B §29–§31, §60, §A2.2)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from civitas.persistence.base import (
    Base,
    Immutable,
    Metadataed,
    SoftDeletable,
    Timestamped,
    UUIDPrimaryKey,
)
from civitas.persistence.types import GUID, JSONVariant, UTCDateTime


class VersionedArtifactMixin:
    """Shared shape for everything the A2.2 gate governs (Part A §A2.2).

    Prompts, retrieval policies, scheduler policies, consolidation methods, role mixes, procedures
    and policies are all versioned artifacts, and a version is **inert until an experiment
    approves it**. The runtime's loader filters on `approved_by_experiment_run_id IS NOT NULL`, so
    uncontrolled self-modification of critical infrastructure is impossible by construction rather
    than by convention.
    """

    version: Mapped[int] = mapped_column(Integer(), nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="proposed", index=True)
    approved_by_experiment_run_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), default=None, index=True
    )
    approved_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    #: The measured improvement the approval rested on, so an adoption can be audited and a
    #: rollback can say what it is undoing.
    approval_evidence: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    rolled_back_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    rollback_reason: Mapped[str] = mapped_column(Text(), nullable=False, default="")

    @property
    def is_active(self) -> bool:
        return (
            self.status == "active"
            and self.approved_by_experiment_run_id is not None
            and self.rolled_back_at is None
        )


class Procedure(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable, VersionedArtifactMixin):
    """Institutional practice (Part B §30).

    "Always reproduce this type of result before accepting it." A procedure is a rule the runtime
    applies, not advice in a prompt — `trigger` says when it fires and `requirement` says what it
    demands — so whether an institution improves outcomes is measurable (§30).
    """

    __tablename__ = "procedures"
    __table_args__ = (
        UniqueConstraint("workspace_id", "name", "version", name="uq_procedure_ws_name_version"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    statement: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    trigger: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    requirement: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    proposed_by_episode_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None, index=True)
    times_applied: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)


class Policy(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable, VersionedArtifactMixin):
    """A versioned runtime policy (Part B §31, §60, §A2.2).

    `kind` names what it governs — retrieval, scheduler, consolidation, role_mix, verification,
    tool_use, prompt — and `body` is the parameters. This is the table the A2.2 gate protects: the
    loader will not return a version an experiment has not approved.
    """

    __tablename__ = "policies"
    __table_args__ = (
        UniqueConstraint("workspace_id", "kind", "name", "version", name="uq_policy_identity"),
        Index("ix_policy_kind_status", "kind", "status"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    body: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    proposed_by_episode_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None, index=True)


class ReputationMetric(Base, UUIDPrimaryKey, Timestamped, Metadataed, Immutable):
    """One dimension of trust, computed from outcome events (Part B §29, §A2.1).

    Multidimensional on purpose: §29 forbids collapsing trust into one universal score, because a
    single number makes an agent that is unreliable-but-correct indistinguishable from one that is
    reliable-but-wrong, and because a scalar reputation is the fastest way to make dissent
    impossible.

    Rows are append-only observations, not a running total. The current value is the fold, so a
    reputation is always reconstructable from the evaluations that produced it.
    """

    __tablename__ = "reputation_metrics"
    __table_args__ = (
        Index("ix_reputation_subject_dim", "subject_kind", "subject_id", "dimension"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: agent_profile | agent_instance | tool | artifact
    subject_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False)
    #: predictive_accuracy | calibration | evidence_quality | reproducibility |
    #: downstream_usefulness | false_positive_rate | tool_creation | tool_reliability |
    #: efficiency | correction_rate | contradiction_resolution
    dimension: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    sample_size: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    evaluation_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")


class PromptVersion(Base, UUIDPrimaryKey, Timestamped, Metadataed, VersionedArtifactMixin):
    """A system prompt under the A2.2 gate (Part A §A2.2).

    Prompts are named explicitly by A2.2 as versioned artifacts requiring experimental approval,
    because a prompt change is the cheapest and least visible way to alter the whole collective's
    behaviour between runs.
    """

    __tablename__ = "prompt_versions"
    __table_args__ = (
        UniqueConstraint("workspace_id", "name", "version", name="uq_prompt_ws_name_version"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(64), nullable=False, default="system")
    body: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    proposed_by_episode_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)


class DuplicateFailureRecord(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """A documented failure, indexed for pre-action lookup (Part A §A2.3, Part B §12).

    Two keys because §A2.3 names two forms of repetition: the same tool with the same arguments,
    and the same hypothesis under the same environment version. Both are hashed at write time so
    the pre-action check is an index lookup and can run *before* the action executes rather than
    being an after-the-fact observation.
    """

    __tablename__ = "duplicate_failure_records"
    __table_args__ = (
        Index("ix_dupfail_ws_toolkey", "workspace_id", "tool_key"),
        Index("ix_dupfail_ws_hypkey", "workspace_id", "hypothesis_key"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("artifacts.id", ondelete="CASCADE"), default=None, index=True
    )
    episode_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    tool_key: Mapped[str | None] = mapped_column(String(64), default=None)
    hypothesis_key: Mapped[str | None] = mapped_column(String(64), default=None)
    environment_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    summary: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    ruled_out: Mapped[list] = mapped_column(JSONVariant(), default=list, nullable=False)
    reproducible: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)
    times_repeated: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
