"""Durable job processing with leases (Part B §42, §54).

Leasing is on the job row rather than in a broker so PostgreSQL and SQLite behave identically and
Colab needs no Redis (Part B §37). Claiming uses `SELECT … FOR UPDATE SKIP LOCKED` on PostgreSQL
and a compare-and-set `UPDATE` on SQLite; both are exercised by the worker-recovery tests.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from civitas.domain.enums import JobStatus
from civitas.persistence.models.infra import Job, JobLease
from civitas.persistence.types import utcnow

DEFAULT_LEASE_S = 120.0


def enqueue(
    session: Session,
    *,
    kind: str,
    payload: dict[str, Any] | None = None,
    workspace_id: uuid.UUID | None = None,
    priority: int = 0,
    idempotency_key: str | None = None,
    max_attempts: int = 3,
    timeout_s: float = 900.0,
    run_after_s: float = 0.0,
) -> Job:
    """Enqueue work. An existing `idempotency_key` returns the original job unchanged.

    That is what makes a resumed campaign safe (Part B §41): re-enqueuing the same run is a
    lookup, not a second execution.
    """
    if idempotency_key:
        existing = session.execute(
            select(Job).where(Job.idempotency_key == idempotency_key)
        ).scalar_one_or_none()
        if existing is not None:
            return existing

    job = Job(
        kind=kind,
        payload=payload or {},
        workspace_id=workspace_id,
        priority=priority,
        idempotency_key=idempotency_key,
        max_attempts=max_attempts,
        timeout_s=timeout_s,
        run_after=utcnow() + timedelta(seconds=run_after_s),
    )
    session.add(job)
    session.flush()
    return job


def claim(
    session: Session,
    *,
    worker_id: str,
    kinds: list[str] | None = None,
    lease_s: float = DEFAULT_LEASE_S,
) -> Job | None:
    """Claim one job, or return None.

    A job is claimable when it is QUEUED and due, **or** when it is LEASED and its lease has
    expired — that second clause is worker-failure recovery (§42): a worker that dies mid-job
    holds a lease that lapses, and the job returns to the pool rather than being lost.
    """
    now = utcnow()
    claimable = and_(
        Job.run_after <= now,
        Job.attempts < Job.max_attempts,
        or_(
            Job.status == JobStatus.QUEUED,
            and_(Job.status == JobStatus.LEASED, Job.lease_expires_at < now),
        ),
    )
    stmt = select(Job).where(claimable)
    if kinds:
        stmt = stmt.where(Job.kind.in_(kinds))
    stmt = stmt.order_by(Job.priority.desc(), Job.run_after).limit(1)

    is_pg = session.bind is not None and session.bind.dialect.name == "postgresql"
    if is_pg:
        stmt = stmt.with_for_update(skip_locked=True)

    job = session.execute(stmt).scalar_one_or_none()
    if job is None:
        return None

    lease_id = uuid.uuid4()
    expires = now + timedelta(seconds=lease_s)
    prior_lease = job.lease_id

    # Compare-and-set on the lease we read. On PostgreSQL the row is already locked and this is
    # belt-and-braces; on SQLite it is the actual guarantee that two workers cannot both win.
    result = session.execute(
        update(Job)
        .where(
            Job.id == job.id,
            Job.lease_id.is_(None) if prior_lease is None else Job.lease_id == prior_lease,
        )
        .values(
            status=JobStatus.LEASED,
            lease_id=lease_id,
            lease_expires_at=expires,
            leased_by=worker_id,
            attempts=Job.attempts + 1,
            started_at=job.started_at or now,
        )
    )
    if result.rowcount != 1:
        return None  # another worker won the race

    if prior_lease is not None:
        session.execute(
            update(JobLease)
            .where(JobLease.job_id == job.id, JobLease.outcome.is_(None))
            .values(outcome="expired")
        )

    session.add(JobLease(job_id=job.id, worker_id=worker_id, expires_at=expires))
    session.flush()
    session.refresh(job)
    return job


def heartbeat(session: Session, job: Job, *, lease_s: float = DEFAULT_LEASE_S) -> bool:
    """Extend a lease. Returns False when the lease was already lost to another worker.

    The caller must stop work on False: continuing would mean two workers running one job, which
    is the duplicate execution §42 requires protection against.
    """
    result = session.execute(
        update(Job)
        .where(Job.id == job.id, Job.lease_id == job.lease_id, Job.status == JobStatus.LEASED)
        .values(lease_expires_at=utcnow() + timedelta(seconds=lease_s))
    )
    if result.rowcount == 1:
        session.execute(
            update(JobLease)
            .where(JobLease.job_id == job.id, JobLease.outcome.is_(None))
            .values(heartbeat_count=JobLease.heartbeat_count + 1)
        )
        session.flush()
        return True
    return False


def complete(session: Session, job: Job, result: dict[str, Any] | None = None) -> None:
    session.execute(
        update(Job)
        .where(Job.id == job.id)
        .values(
            status=JobStatus.SUCCEEDED,
            result=result or {},
            completed_at=utcnow(),
            lease_id=None,
            lease_expires_at=None,
        )
    )
    session.execute(
        update(JobLease)
        .where(JobLease.job_id == job.id, JobLease.outcome.is_(None))
        .values(outcome="completed")
    )
    session.flush()


def fail(session: Session, job: Job, error: str, *, backoff_base_s: float = 2.0) -> None:
    """Record a failure. Retries with exponential backoff until `max_attempts`, then dead-letters.

    Dead-lettering rather than retrying forever is what stops a poisoned job consuming a worker
    indefinitely (§42).
    """
    session.refresh(job)
    exhausted = job.attempts >= job.max_attempts
    delay = backoff_base_s * (2 ** max(0, job.attempts - 1))
    session.execute(
        update(Job)
        .where(Job.id == job.id)
        .values(
            status=JobStatus.DEAD_LETTER if exhausted else JobStatus.QUEUED,
            error=error[:8000],
            completed_at=utcnow() if exhausted else None,
            run_after=utcnow() + timedelta(seconds=0 if exhausted else delay),
            lease_id=None,
            lease_expires_at=None,
        )
    )
    session.execute(
        update(JobLease)
        .where(JobLease.job_id == job.id, JobLease.outcome.is_(None))
        .values(outcome="released")
    )
    session.flush()


def cancel(session: Session, job_id: uuid.UUID) -> bool:
    """Cancel a job that has not finished (Part B §42)."""
    result = session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status.in_([JobStatus.QUEUED, JobStatus.LEASED]))
        .values(
            status=JobStatus.CANCELLED,
            cancelled_at=utcnow(),
            lease_id=None,
            lease_expires_at=None,
        )
    )
    session.flush()
    return result.rowcount == 1


def reap_expired(session: Session) -> int:
    """Return jobs whose lease lapsed to the queue.

    `claim` already treats an expired lease as claimable, so this is not required for correctness.
    It exists so the *state* is honest between claims: a monitoring query for queue depth should
    not count a job as leased when no worker holds it (§53).
    """
    result = session.execute(
        update(Job)
        .where(
            Job.status == JobStatus.LEASED,
            Job.lease_expires_at < utcnow(),
            Job.attempts < Job.max_attempts,
        )
        .values(status=JobStatus.QUEUED, lease_id=None, lease_expires_at=None, leased_by=None)
    )
    session.flush()
    return int(result.rowcount or 0)
