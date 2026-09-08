"""The tool ecology (Part B §17, §18) — the civilization's accumulated technology."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from civitas.domain.enums import ToolValidationState
from civitas.persistence.base import Base, Metadataed, SoftDeletable, Timestamped, UUIDPrimaryKey
from civitas.persistence.types import GUID, EnumType, JSONVariant, UTCDateTime


class ToolDefinition(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    """A named, reusable capability created by an agent (Part B §17).

    The definition is the stable identity a later agent discovers; `ToolVersion` carries the
    source. Splitting them is what lets a tool *improve* across generations without every caller's
    reference breaking — which is the whole point of accumulating technology rather than text.
    """

    __tablename__ = "tool_definitions"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_tool_workspace_name"),)

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="python", index=True)
    created_by_episode_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("episodes.id", ondelete="SET NULL"), default=None, index=True
    )
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("artifacts.id", ondelete="SET NULL"), default=None, index=True
    )
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    is_builtin: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False, index=True)

    # --- usage history, folded from ToolRun (Part B §17, §29) -----------
    times_used: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    times_succeeded: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    times_failed: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    #: Fold of ArtifactUtilityMetric-equivalent credit for tools (Part A §A2.1). Computed from
    #: evaluation outcomes, never self-reported.
    downstream_utility: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)

    versions: Mapped[list[ToolVersion]] = relationship(back_populates="definition")

    @property
    def success_rate(self) -> float | None:
        """None, not 0.0, when the tool has never run. An unused tool is not a failing one, and
        collapsing the two would rank a fresh tool below a broken one."""
        total = self.times_succeeded + self.times_failed
        return self.times_succeeded / total if total else None


class ToolVersion(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """One revision of a tool's source (Part B §17).

    The source is write-once: a change is a new version, never an edit, so a run recorded against
    version 3 can always be reproduced. The row is *not* append-only, because validation state and
    experiment approval are established after the source exists and are recorded here.
    """

    __tablename__ = "tool_versions"
    __table_args__ = (
        UniqueConstraint("tool_definition_id", "version", name="uq_tool_version"),
    )

    tool_definition_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("tool_definitions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer(), nullable=False)
    source: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    entrypoint: Mapped[str] = mapped_column(String(200), nullable=False, default="main")
    #: JSON Schema for arguments. Enforced before execution, so a malformed call is a validation
    #: error rather than an arbitrary-code path (Part B §52).
    parameters_schema: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    dependencies: Mapped[list] = mapped_column(JSONVariant(), default=list, nullable=False)
    #: What the sandbox will grant. Requesting more than the policy allows fails the run rather
    #: than widening the sandbox (Part B §18).
    permissions: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    compatibility: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    validation_state: Mapped[ToolValidationState] = mapped_column(
        EnumType(ToolValidationState, 32),
        nullable=False,
        default=ToolValidationState.UNTESTED,
        index=True,
    )
    test_source: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    test_report: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    created_by_episode_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    #: Set by the A2.2 gate for tools that replace a prior version on a critical path.
    approved_by_experiment_run_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)

    definition: Mapped[ToolDefinition] = relationship(back_populates="versions")


class ToolRun(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """One sandboxed execution (Part B §17, §18, §A2.1).

    Carries `args_hash` so duplicate-failure detection can ask "has this exact call already been
    tried and documented as failing?" without re-deriving a canonical form at each call site
    (Part A §A2.3).
    """

    __tablename__ = "tool_runs"
    __table_args__ = (
        Index("ix_toolrun_def_created", "tool_definition_id", "created_at"),
        Index("ix_toolrun_argshash", "args_hash"),
    )

    episode_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("episodes.id", ondelete="CASCADE"), default=None, index=True
    )
    tool_definition_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("tool_definitions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tool_version_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("tool_versions.id", ondelete="SET NULL"), default=None
    )
    args: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    args_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    exit_code: Mapped[int | None] = mapped_column(Integer(), default=None)
    succeeded: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False, index=True)
    stdout: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    stderr: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    error: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    duration_ms: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    sandbox_backend: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    #: Which limit stopped it, when one did: cpu, memory, wall, disk, processes, network.
    limit_hit: Mapped[str | None] = mapped_column(String(32), default=None)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    output_asset_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
