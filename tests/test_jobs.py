"""Durable job processing (Part B §42, §54, §55 worker-recovery tests)."""

from __future__ import annotations

import threading
from datetime import timedelta

from civitas.domain.enums import JobStatus
from civitas.persistence import jobs
from civitas.persistence.models import Job, JobLease
from civitas.persistence.types import utcnow


def test_claim_returns_one_job_and_marks_it_leased(db, workspace):
    jobs.enqueue(db, kind="episode", workspace_id=workspace.id, payload={"n": 1})
    db.commit()

    job = jobs.claim(db, worker_id="w1")
    db.commit()

    assert job is not None
    assert job.status is JobStatus.LEASED
    assert job.leased_by == "w1"
    assert job.attempts == 1
    assert job.lease_expires_at > utcnow()


def test_empty_queue_returns_none(db):
    assert jobs.claim(db, worker_id="w1") is None


def test_idempotency_key_makes_re_enqueue_a_lookup(db, workspace):
    """Part B §41: a resumed campaign re-enqueues the same work. It must not run twice."""
    a = jobs.enqueue(db, kind="run", workspace_id=workspace.id, idempotency_key="arm-a-seed-0")
    db.commit()
    b = jobs.enqueue(db, kind="run", workspace_id=workspace.id, idempotency_key="arm-a-seed-0")
    db.commit()

    assert a.id == b.id
    assert db.query(Job).count() == 1


def test_a_dead_worker_s_job_is_recovered_when_its_lease_expires(db, workspace):
    """Part B §55 worker-recovery: kill a worker mid-job and recover safely.

    The worker is killed by fiat — the lease is backdated, which is exactly what a crashed process
    leaves behind — and a second worker must be able to pick the job up.
    """
    jobs.enqueue(db, kind="episode", workspace_id=workspace.id)
    db.commit()

    first = jobs.claim(db, worker_id="w1", lease_s=60)
    db.commit()
    assert first is not None

    # No second worker may take it while the lease holds.
    assert jobs.claim(db, worker_id="w2") is None

    first.lease_expires_at = utcnow() - timedelta(seconds=1)
    db.commit()

    second = jobs.claim(db, worker_id="w2")
    db.commit()

    assert second is not None
    assert second.id == first.id
    assert second.leased_by == "w2"
    assert second.attempts == 2, "the recovered attempt must be counted"

    leases = db.query(JobLease).filter(JobLease.job_id == first.id).all()
    assert len(leases) == 2, "both leases are on the record"
    assert {lease.outcome for lease in leases} == {"expired", None}


def test_heartbeat_extends_a_lease_and_fails_once_it_is_lost(session_factory, workspace):
    """A worker whose lease lapsed and was taken by another must learn that from its heartbeat.

    Driven with two sessions because that is what two workers are. A single shared session would
    have `claim` refresh the *same* ORM object, so the first worker would silently inherit the
    second's lease id and the compare-and-set would appear to succeed.
    """
    s1 = session_factory()
    jobs.enqueue(s1, kind="episode", workspace_id=workspace.id)
    s1.commit()

    job1 = jobs.claim(s1, worker_id="w1", lease_s=60)
    s1.commit()
    before = job1.lease_expires_at

    assert jobs.heartbeat(s1, job1, lease_s=300) is True
    s1.commit()
    s1.refresh(job1)
    assert job1.lease_expires_at > before

    # w1 stalls: its lease lapses and w2 legitimately takes the job.
    s1.execute(
        Job.__table__.update()
        .where(Job.id == job1.id)
        .values(lease_expires_at=utcnow() - timedelta(seconds=1))
    )
    s1.commit()

    s2 = session_factory()
    job2 = jobs.claim(s2, worker_id="w2", lease_s=60)
    s2.commit()
    assert job2 is not None and job2.id == job1.id

    # w1 wakes up holding its own stale object. The heartbeat must tell it the lease is gone.
    assert jobs.heartbeat(s1, job1) is False, "a worker whose lease was taken must stop"
    s1.close()
    s2.close()


def test_failure_retries_with_backoff_then_dead_letters(db, workspace):
    jobs.enqueue(db, kind="episode", workspace_id=workspace.id, max_attempts=2)
    db.commit()

    job = jobs.claim(db, worker_id="w1")
    db.commit()
    jobs.fail(db, job, "boom")
    db.commit()
    db.refresh(job)
    assert job.status is JobStatus.QUEUED
    assert job.run_after > utcnow(), "a retry must be delayed, not immediate"

    job.run_after = utcnow()
    db.commit()

    job = jobs.claim(db, worker_id="w1")
    db.commit()
    jobs.fail(db, job, "boom again")
    db.commit()
    db.refresh(job)

    assert job.status is JobStatus.DEAD_LETTER
    assert jobs.claim(db, worker_id="w1") is None, "a dead-lettered job is not reclaimed"


def test_completion_clears_the_lease_and_records_the_result(db, workspace):
    jobs.enqueue(db, kind="episode", workspace_id=workspace.id)
    db.commit()
    job = jobs.claim(db, worker_id="w1")
    db.commit()

    jobs.complete(db, job, {"episode_id": "x"})
    db.commit()
    db.refresh(job)

    assert job.status is JobStatus.SUCCEEDED
    assert job.lease_id is None
    assert job.result == {"episode_id": "x"}
    lease = db.query(JobLease).filter(JobLease.job_id == job.id).one()
    assert lease.outcome == "completed"


def test_cancellation_stops_a_queued_job(db, workspace):
    job = jobs.enqueue(db, kind="episode", workspace_id=workspace.id)
    db.commit()

    assert jobs.cancel(db, job.id) is True
    db.commit()
    assert jobs.claim(db, worker_id="w1") is None
    assert jobs.cancel(db, job.id) is False, "cancelling twice is not a second cancellation"


def test_reaping_makes_queue_depth_honest(db, workspace):
    """`claim` already treats an expired lease as claimable, so reaping is about *state*: a
    monitoring query must not report a job as leased when no worker holds it (Part B §53)."""
    jobs.enqueue(db, kind="episode", workspace_id=workspace.id)
    db.commit()
    job = jobs.claim(db, worker_id="w1")
    db.commit()

    job.lease_expires_at = utcnow() - timedelta(seconds=1)
    db.commit()

    assert jobs.reap_expired(db) == 1
    db.commit()
    db.refresh(job)
    assert job.status is JobStatus.QUEUED
    assert job.leased_by is None


def test_two_workers_racing_never_both_win(session_factory, workspace):
    """Duplicate-execution protection under real contention (Part B §42, §55 concurrency).

    Ten jobs, three workers, all claiming at once. Each job must be claimed exactly once.
    """
    s = session_factory()
    for i in range(10):
        jobs.enqueue(s, kind="episode", workspace_id=workspace.id, payload={"i": i})
    s.commit()
    s.close()

    claimed: list = []
    lock = threading.Lock()
    errors: list[Exception] = []

    def worker(name: str) -> None:
        try:
            while True:
                sess = session_factory()
                try:
                    job = jobs.claim(sess, worker_id=name, lease_s=300)
                    if job is None:
                        sess.commit()
                        return
                    jid = job.id
                    jobs.complete(sess, job, {"by": name})
                    sess.commit()
                    with lock:
                        claimed.append(jid)
                finally:
                    sess.close()
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"claim raced: {errors[:2]}"
    assert len(claimed) == 10
    assert len(set(claimed)) == 10, "a job was executed twice"
