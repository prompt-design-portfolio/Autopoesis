"""Experiments, arms, runs, manifests and evaluations (Part B §20–§25, §46, §47)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from civitas.domain.enums import ExperimentArm as ArmEnum
from civitas.persistence.base import (
    Base,
    Immutable,
    Metadataed,
    SoftDeletable,
    Timestamped,
    UUIDPrimaryKey,
)
from civitas.persistence.types import GUID, EnumType, JSONVariant, UTCDateTime


class Experiment(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    """A campaign: one question, several arms (Part B §21, §22).

    `hypothesis` and `prediction` are filled *before* the runs, and `prediction` is what makes the
    result falsifiable rather than narrated — the pre-registration discipline of ARCHITECTURE §3.11.
    """

    __tablename__ = "experiments"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False, default="newcomer", index=True)
    description: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    hypothesis: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    prediction: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created", index=True)
    #: Frozen-model mode (Part B §20). When set, every arm must use this configuration and the
    #: runner refuses to start if any run's config hash disagrees.
    frozen_model_configuration_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)

    arms: Mapped[list[ExperimentArm]] = relationship(back_populates="experiment")
    runs: Mapped[list[ExperimentRun]] = relationship(back_populates="experiment")


class ExperimentArm(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """One condition (Part B §21). The arm is a *runtime policy*, not a prompt instruction: the
    retrieval layer reads `arm` and enforces it, so no prompt wording can defeat it."""

    __tablename__ = "experiment_arms"
    __table_args__ = (UniqueConstraint("experiment_id", "name", name="uq_arm_experiment_name"),)

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    arm: Mapped[ArmEnum] = mapped_column(EnumType(ArmEnum, 40), nullable=False, index=True)
    #: The arm this one is the control for. `collective_scrambled` names `collective`; without the
    #: link a reader has to guess which comparison the arm exists to support.
    control_for: Mapped[str | None] = mapped_column(String(100), default=None)
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    snapshot_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    overrides: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)

    experiment: Mapped[Experiment] = relationship(back_populates="arms")


class ExperimentRun(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """One (arm, seed) execution (Part B §41, §46).

    Resumability turns on `(experiment_arm_id, seed)` being unique: a reconnecting Colab session
    detects completed runs and skips them rather than re-running a campaign (§41).
    """

    __tablename__ = "experiment_runs"
    __table_args__ = (
        UniqueConstraint("experiment_arm_id", "seed", name="uq_run_arm_seed"),
        Index("ix_run_experiment_status", "experiment_id", "status"),
    )

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    experiment_arm_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("experiment_arms.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seed: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    manifest_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    episodes_planned: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    episodes_completed: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    #: Gate outcomes. A failed gate makes the metrics unreadable — ARCHITECTURE §3.1: the API
    #: returns the gate failure, never the number behind it.
    gates: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    error: Mapped[str] = mapped_column(Text(), nullable=False, default="")

    experiment: Mapped[Experiment] = relationship(back_populates="runs")

    @property
    def gates_passed(self) -> bool:
        return all(g.get("passed") for g in (self.gates or {}).values())


class ExperimentManifest(Base, UUIDPrimaryKey, Timestamped, Metadataed, Immutable):
    """Everything needed to reconstruct a run (Part B §46). Immutable, and secret-free by
    construction: it is built from `Settings.manifest_dict()`, which drops every `SecretStr` by
    type rather than by name (§40)."""

    __tablename__ = "experiment_manifests"

    experiment_run_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("experiment_runs.id", ondelete="CASCADE"), default=None, index=True
    )
    git_commit: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    app_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    container_digest: Mapped[str | None] = mapped_column(String(200), default=None)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    dependency_versions: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    model_name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    model_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    sampling_parameters: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    prompt_versions: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    tool_versions: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    retrieval_policy: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    experiment_arm: Mapped[str] = mapped_column(String(40), nullable=False, default="")
    budgets: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    evaluator: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    environment: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    workspace_snapshot_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    seeds: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    #: Records the sandbox actually in force (Part B §18). A result produced under the
    #: subprocess fallback must never read as one produced under isolation.
    sandbox_backend: Mapped[str] = mapped_column(String(32), nullable=False, default="")


class Evaluation(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """An external judgement (Part B §47).

    Agents cannot self-certify, so `evaluator_kind` is recorded and the four scopes §47 requires
    to be kept apart — episode, artifact, agent contribution, collective — are separate rows
    rather than one conflated score.
    """

    __tablename__ = "evaluations"
    __table_args__ = (Index("ix_eval_scope_target", "scope", "target_id"),)

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    episode_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("episodes.id", ondelete="CASCADE"), default=None, index=True
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("tasks.id", ondelete="SET NULL"), default=None, index=True
    )
    #: episode | artifact | agent_contribution | collective
    scope: Mapped[str] = mapped_column(String(32), nullable=False, default="episode", index=True)
    target_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    #: unit_test | hidden_test | static_analysis | reproduction | judge | human | performance
    evaluator_kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    evaluator_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    succeeded: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False, index=True)
    score: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    max_score: Mapped[float] = mapped_column(Float(), nullable=False, default=1.0)
    detail: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    #: Set once credit has been propagated (Part A §A2.1), so an evaluation cannot be counted
    #: twice by a retry or a resumed campaign.
    credit_assigned_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)

    episode = relationship("Episode", back_populates="evaluations", foreign_keys=[episode_id])
