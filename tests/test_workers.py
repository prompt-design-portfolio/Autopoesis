"""Worker recovery and durability (Part B §37, §42, §54, §55)."""

from __future__ import annotations

import threading
import time
from datetime import timedelta

import pytest

from civitas.domain.enums import JobStatus
from civitas.persistence import jobs
from civitas.persistence.models import Heartbeat, Job, JobLease
from civitas.persistence.types import utcnow
from civitas.workers.worker import _HANDLERS, Worker, WorkerConfig, handler


@pytest.fixture(autouse=True)
def _clean_handlers():
    saved = dict(_HANDLERS)
    _HANDLERS.clear()
    yield
    _HANDLERS.clear()
    _HANDLERS.update(saved)


def test_a_worker_claims_runs_and_completes(db, session_factory, workspace):
    done: list[str] = []

    @handler("echo")
    def _echo(session, job):
        done.append(job.payload["msg"])
        return {"echoed": job.payload["msg"]}

    jobs.enqueue(db, kind="echo", workspace_id=workspace.id, payload={"msg": "hello"})
    db.commit()

    worker = Worker(session_factory, WorkerConfig(max_jobs=1, idle_exit=True))
    assert worker.run() == 1
    assert done == ["hello"]

    db.expire_all()
    job = db.query(Job).one()
    assert job.status is JobStatus.SUCCEEDED
    assert job.result == {"echoed": "hello"}


def test_a_failing_job_is_retried_then_dead_lettered(db, session_factory, workspace):
    attempts: list[int] = []

    @handler("flaky")
    def _flaky(session, job):
        attempts.append(job.attempts)
        raise RuntimeError("always fails")

    jobs.enqueue(db, kind="flaky", workspace_id=workspace.id, max_attempts=2)
    db.commit()

    for _ in range(2):
        Worker(session_factory, WorkerConfig(max_jobs=1, idle_exit=True)).run()
        db.expire_all()
        job = db.query(Job).one()
        if job.status is JobStatus.QUEUED:
            job.run_after = utcnow()
            db.commit()

    db.expire_all()
    job = db.query(Job).one()
    assert job.status is JobStatus.DEAD_LETTER
    assert len(attempts) == 2
    assert "always fails" in job.error


def test_an_unknown_job_kind_is_not_retried(db, session_factory, workspace):
    """A missing handler is a deployment error, not a transient fault. Retrying it would occupy a
    worker for no possible benefit."""
    jobs.enqueue(db, kind="nobody_handles_this", workspace_id=workspace.id, max_attempts=5)
    db.commit()

    Worker(session_factory, WorkerConfig(max_jobs=1, idle_exit=True)).run()
    db.expire_all()
    job = db.query(Job).one()
    assert job.status is JobStatus.DEAD_LETTER
    assert "no handler registered" in job.error


def test_a_killed_worker_s_job_is_picked_up_by_another(db, session_factory, workspace):
    """Part B §55: kill a worker mid-job and recover safely."""
    ran_by: list[str] = []

    @handler("slow")
    def _slow(session, job):
        ran_by.append("second")
        return {}

    jobs.enqueue(db, kind="slow", workspace_id=workspace.id)
    db.commit()

    # Worker 1 claims and then "dies" — no completion, and its lease is backdated, which is
    # exactly what a crashed process leaves behind.
    stolen = jobs.claim(db, worker_id="w1-crashed", lease_s=300)
    db.commit()
    assert stolen is not None
    stolen.lease_expires_at = utcnow() - timedelta(seconds=1)
    db.commit()

    second = Worker(session_factory, WorkerConfig(worker_id="w2", max_jobs=1, idle_exit=True))
    assert second.run() == 1
    assert ran_by == ["second"]

    db.expire_all()
    job = db.query(Job).one()
    assert job.status is JobStatus.SUCCEEDED
    leases = (
        db.query(JobLease)
        .filter(JobLease.job_id == job.id)
        .order_by(JobLease.created_at)
        .all()
    )
    assert [lease.worker_id for lease in leases] == ["w1-crashed", "w2"]
    assert leases[0].outcome == "expired", "the crashed worker's lease is on the record"
    assert leases[1].outcome == "completed"


def test_concurrent_workers_never_execute_a_job_twice(db, session_factory, workspace):
    """Part B §42 duplicate-execution protection, under real contention."""
    executed: list[str] = []
    lock = threading.Lock()

    @handler("unit")
    def _unit(session, job):
        with lock:
            executed.append(job.payload["i"])
        time.sleep(0.005)
        return {}

    for i in range(12):
        jobs.enqueue(db, kind="unit", workspace_id=workspace.id, payload={"i": str(i)})
    db.commit()

    workers = [
        Worker(session_factory, WorkerConfig(worker_id=f"w{i}", idle_exit=True, lease_s=60))
        for i in range(4)
    ]
    threads = [threading.Thread(target=w.run) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert sorted(executed) == sorted(str(i) for i in range(12))
    assert len(executed) == len(set(executed)), "a job ran twice"
    assert sum(w.jobs_processed for w in workers) == 12


def test_graceful_shutdown_finishes_the_current_job(db, session_factory, workspace):
    """Part B §54: a stop request must not abandon work mid-flight."""
    completed: list[str] = []
    worker_ref: dict = {}

    @handler("stoppable")
    def _stoppable(session, job):
        worker_ref["w"].request_stop()  # stop arrives while this job is running
        time.sleep(0.02)
        completed.append(job.payload["i"])
        return {}

    for i in range(3):
        jobs.enqueue(db, kind="stoppable", workspace_id=workspace.id, payload={"i": str(i)})
    db.commit()

    worker = Worker(session_factory, WorkerConfig(idle_exit=True))
    worker_ref["w"] = worker
    worker.run()

    assert len(completed) == 1, "the in-flight job must finish, and no further job start"
    db.expire_all()
    statuses = [j.status for j in db.query(Job).all()]
    assert statuses.count(JobStatus.SUCCEEDED) == 1
    assert statuses.count(JobStatus.QUEUED) == 2, "the rest return to the queue, not lost"


def test_a_worker_records_a_heartbeat(db, session_factory, workspace):
    @handler("noop")
    def _noop(session, job):
        return {}

    jobs.enqueue(db, kind="noop", workspace_id=workspace.id)
    db.commit()
    Worker(session_factory, WorkerConfig(worker_id="observable", max_jobs=1, idle_exit=True)).run()

    db.expire_all()
    beat = db.query(Heartbeat).filter(Heartbeat.worker_id == "observable").one()
    assert beat.jobs_processed == 1
    assert beat.is_draining, "a stopped worker must be visible as drained, not merely stale"


def test_a_resumed_campaign_does_not_re_run_completed_work(db, session_factory, workspace):
    """Part B §41: a Colab disconnect must not destroy or duplicate a campaign."""
    runs: list[str] = []

    @handler("run")
    def _run(session, job):
        runs.append(job.payload["arm"])
        return {}

    for arm in ("collective", "solo"):
        jobs.enqueue(db, kind="run", workspace_id=workspace.id,
                     payload={"arm": arm}, idempotency_key=f"exp1-{arm}-seed0")
    db.commit()
    Worker(session_factory, WorkerConfig(idle_exit=True)).run()
    assert sorted(runs) == ["collective", "solo"]

    # The session reconnects and re-enqueues the whole campaign.
    for arm in ("collective", "solo"):
        jobs.enqueue(db, kind="run", workspace_id=workspace.id,
                     payload={"arm": arm}, idempotency_key=f"exp1-{arm}-seed0")
    db.commit()
    Worker(session_factory, WorkerConfig(idle_exit=True)).run()

    assert sorted(runs) == ["collective", "solo"], "completed work was re-run"
    assert db.query(Job).count() == 2
