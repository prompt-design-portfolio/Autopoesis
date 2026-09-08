"""Jobs, leases and the immutable event stream (Part B §42, §45, §54)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from civitas.domain.enums import EventType, JobStatus
from civitas.persistence.base import Base, Immutable, Metadataed, Timestamped, UUIDPrimaryKey
from civitas.persistence.types import EnumType, GUID, JSONVariant, UTCDateTime


class Event(Base, UUIDPrimaryKey, Timestamped, Immutable):
    """An immutable record of something that happened (Part B §45).

    Append-only: the session guard in `civitas.persistence.session` rejects UPDATE and DELETE on
    this table, so the history a result rests on cannot be rewritten after the fact. That is what
    makes the civilization historically reconstructable rather than merely well-documented.

    `sequence` is a monotonically increasing per-workspace counter, allocated inside the same
    transaction as the insert. Timestamps alone are not an ordering — two events in one
    millisecond are indistinguishable, and clocks move backwards.
    """

    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("workspace_id", "sequence", name="uq_event_workspace_sequence"),
        Index("ix_events_ws_type_created", "workspace_id", "type", "created_at"),
        Index("ix_events_episode", "episode_id"),
        Index("ix_events_correlation", "correlation_id"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(BigInteger(), nullable=False)
    type: Mapped[EventType] = mapped_column(EnumType(EventType, 48), nullable=False, index=True)

    project_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None, index=True)
    task_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None, index=True)
    episode_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)

    actor_kind: Mapped[str] = mapped_column(String(32), nullable=False, default="system")
    actor_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)

    payload: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    correlation_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")


class EventSequence(Base):
    """Per-workspace event counter.

    A separate row rather than `MAX(sequence)+1`, because the max query races: two concurrent
    writers read the same max and both insert it, and on SQLite the unique constraint turns that
    into a lost event rather than a retry. This row is locked for update, so the allocation is
    serialised on both backends.
    """

    __tablename__ = "event_sequences"

    workspace_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True)
    next_sequence: Mapped[int] = mapped_column(BigInteger(), nullable=False, default=1)


class Job(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """A durable unit of background work (Part B §42).

    Leasing lives on the job row itself rather than in an external broker so that PostgreSQL and
    SQLite behave identically and Colab needs no Redis (§37). A worker that dies holds a lease
    that expires; the job returns to the queue rather than being lost.
    """

    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_claimable", "status", "run_after", "priority"),
        Index("ix_jobs_workspace_status", "workspace_id", "status"),
        UniqueConstraint("idempotency_key", name="uq_job_idempotency"),
    )

    workspace_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None, index=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        EnumType(JobStatus, 32), nullable=False, default=JobStatus.QUEUED, index=True
    )
    priority: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)

    #: Duplicate-execution protection (Part B §42). A resumed campaign re-enqueues the same work;
    #: the unique constraint makes that a no-op instead of a second run.
    idempotency_key: Mapped[str | None] = mapped_column(String(200), default=None)

    attempts: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer(), nullable=False, default=3)
    run_after: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, index=True)

    lease_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), default=None)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None, index=True)
    leased_by: Mapped[str | None] = mapped_column(String(200), default=None)

    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    cancelled_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    result: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    error: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    timeout_s: Mapped[float] = mapped_column(Float(), nullable=False, default=900.0)


class JobLease(Base, UUIDPrimaryKey, Timestamped, Immutable):
    """An append-only record of every lease taken (Part B §42, §54).

    The live lease is three columns on `Job`; this is the audit trail. It is what makes a
    worker-recovery test an *observation* — you can see the lease that expired and the one that
    replaced it — rather than an inference from a job that eventually succeeded.
    """

    __tablename__ = "job_leases"
    __table_args__ = (Index("ix_lease_job_created", "job_id", "created_at"),)

    job_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    worker_id: Mapped[str] = mapped_column(String(200), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    heartbeat_count: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    #: completed | expired | released | superseded — filled by whoever ends the lease.
    outcome: Mapped[str | None] = mapped_column(String(32), default=None)


class Heartbeat(Base, UUIDPrimaryKey, Timestamped):
    """Worker liveness (Part B §37, §42, §54)."""

    __tablename__ = "heartbeats"
    __table_args__ = (UniqueConstraint("worker_id", name="uq_heartbeat_worker"),)

    worker_id: Mapped[str] = mapped_column(String(200), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, index=True)
    jobs_processed: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    is_draining: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)
    info: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
