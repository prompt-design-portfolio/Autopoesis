"""Tasks, assignments, agent profiles and episodes (Part B §7, §8, §26, §27)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from civitas.domain.enums import (
    AgentRole,
    EpisodeStatus,
    ExperimentArm,
    TaskStatus,
    TerminationReason,
)
from civitas.persistence.base import Base, Metadataed, SoftDeletable, Timestamped, UUIDPrimaryKey
from civitas.persistence.types import EnumType, GUID, JSONVariant, UTCDateTime


class Task(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    __tablename__ = "tasks"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("projects.id", ondelete="SET NULL"), default=None, index=True
    )
    objective_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("objectives.id", ondelete="SET NULL"), default=None, index=True
    )
    parent_task_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("tasks.id", ondelete="SET NULL"), default=None, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    task_family: Mapped[str] = mapped_column(String(100), nullable=False, default="", index=True)
    difficulty: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    status: Mapped[TaskStatus] = mapped_column(
        EnumType(TaskStatus, 32), nullable=False, default=TaskStatus.PENDING, index=True
    )
    priority: Mapped[int] = mapped_column(Integer(), nullable=False, default=0, index=True)
    #: What an evaluator needs to judge the work (Part B §47). Held on the task rather than given
    #: to the episode: an agent that can read its own success criteria can self-certify, and §47
    #: says it must not.
    evaluator_spec: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    #: Information deliberately withheld from some populations (Part B §23). The distributed
    #: -knowledge benchmark is only a real test if the partition is enforced here.
    information_partition: Mapped[str | None] = mapped_column(String(100), default=None, index=True)
    attempts: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer(), nullable=False, default=5)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)


class TaskDependency(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """`task_id` cannot start until `depends_on_task_id` completes (Part B §27)."""

    __tablename__ = "task_dependencies"
    __table_args__ = (
        UniqueConstraint("task_id", "depends_on_task_id", name="uq_task_dependency"),
    )

    task_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    depends_on_task_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="blocks")


class Subproblem(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    """A decomposition of a task (Part B §27). Kept distinct from a child task: a subproblem is a
    *question* the scheduler tracks, and only becomes a task when it is worth spending an episode
    on."""

    __tablename__ = "subproblems"

    task_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open", index=True)
    #: Drives uncertainty-directed allocation (Part B §27). High uncertainty is where an episode
    #: buys the most information.
    uncertainty: Mapped[float] = mapped_column(Float(), nullable=False, default=1.0)
    expected_information_gain: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    created_by_episode_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)


class AgentProfile(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    """A specialization (Part B §26).

    A row, not a hard-coded class, because which profile suits which work type has to be
    *discovered* from measured performance. `performance_by_work_type` is a fold over evaluations
    and is written by the profile-learning service, never by an agent.
    """

    __tablename__ = "agent_profiles"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_profile_workspace_name"),)

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[AgentRole] = mapped_column(EnumType(AgentRole, 32), nullable=False, default=AgentRole.EXPLORER)
    description: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    system_prompt_version_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    tool_policy: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    retrieval_policy_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    performance_by_work_type: Mapped[dict] = mapped_column(
        JSONVariant(), default=dict, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=True)


class AgentInstance(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """A named agent identity across episodes (Part B §7, §29).

    Carries **no state an episode can read**. It exists so reputation can accumulate against
    something, and so `tests/test_state_isolation.py` has a place to prove that accumulation does
    not leak: reputation influences *assignment and retrieval rank*, never episode input.
    """

    __tablename__ = "agent_instances"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_profile_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    episodes_run: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    generation: Mapped[int] = mapped_column(Integer(), nullable=False, default=0, index=True)


class Assignment(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """The scheduler's decision to spend an episode on a task (Part B §27).

    `rationale` and `strategy` are recorded because §27 requires allocation decisions to be
    logged, and because an allocation policy cannot be evaluated against outcomes unless the
    reason it fired is recoverable.
    """

    __tablename__ = "assignments"

    task_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_profile_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    subproblem_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("subproblems.id", ondelete="SET NULL"), default=None
    )
    strategy: Mapped[str] = mapped_column(String(32), nullable=False, default="priority")
    rationale: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    score: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open", index=True)
    scheduler_policy_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)


class Episode(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """One bounded agent lifetime (Part B §8) — the unit the whole architecture is built around.

    Every column here is either a bound the episode ran under or a fact about how it ended. There
    is deliberately **no column for episode-local reasoning state**: a scratchpad, a plan or a
    transcript would make the next episode's isolation a matter of care rather than of structure
    (Part B §4). What the episode wanted to keep, it wrote as an artifact.
    """

    __tablename__ = "episodes"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("projects.id", ondelete="SET NULL"), default=None, index=True
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("tasks.id", ondelete="SET NULL"), default=None, index=True
    )
    assignment_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("assignments.id", ondelete="SET NULL"), default=None, index=True
    )
    agent_profile_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    agent_instance_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("agent_instances.id", ondelete="SET NULL"), default=None, index=True
    )

    # --- the frozen individual variables (Part B §8, §20) ---------------
    model_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    model_parameters: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    system_prompt_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    tool_policy: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    retrieval_policy_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    experiment_arm: Mapped[ExperimentArm] = mapped_column(
        EnumType(ExperimentArm, 40), nullable=False, default=ExperimentArm.SOLO, index=True
    )
    experiment_run_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None, index=True)

    # --- budgets (Part B §8, §58) ---------------------------------------
    token_budget: Mapped[int] = mapped_column(Integer(), nullable=False, default=100_000)
    context_budget: Mapped[int] = mapped_column(Integer(), nullable=False, default=32_000)
    tool_call_budget: Mapped[int] = mapped_column(Integer(), nullable=False, default=50)
    cost_budget_usd: Mapped[float] = mapped_column(Float(), nullable=False, default=1.0)
    wall_clock_budget_s: Mapped[float] = mapped_column(Float(), nullable=False, default=600.0)

    # --- consumption ----------------------------------------------------
    tokens_used: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    tool_calls_used: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    model_calls: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    artifacts_retrieved: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    artifacts_read: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    artifacts_created: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    duplicate_failures: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)

    # --- lifecycle ------------------------------------------------------
    status: Mapped[EpisodeStatus] = mapped_column(
        EnumType(EpisodeStatus, 32), nullable=False, default=EpisodeStatus.PENDING, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    termination_reason: Mapped[TerminationReason | None] = mapped_column(
        EnumType(TerminationReason, 40), default=None, index=True
    )
    termination_detail: Mapped[str] = mapped_column(Text(), nullable=False, default="")

    # --- reproducibility (Part B §8, §46) -------------------------------
    environment_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    code_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    seed: Mapped[int | None] = mapped_column(Integer(), default=None)

    #: Newcomer-benchmark probe episodes (Part B §22). Excluded from the collective-maturity
    #: statistics they are measured against — the founder-free discipline of ARCHITECTURE §3.5.
    is_benchmark_probe: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)

    evaluations: Mapped[list] = relationship(
        "Evaluation", back_populates="episode", foreign_keys="Evaluation.episode_id"
    )

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()
