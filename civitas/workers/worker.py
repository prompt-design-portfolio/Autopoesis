"""The durable worker loop (Part B §37, §42, §54).

Pure Python: no systemd, no Docker daemon, no Kubernetes, no privileged process. That is what
makes it usable in a Colab runtime (§37) while being the same code that runs in production (§6).

Reliability is in four properties, each tested:

* **Leases.** Work is claimed, not consumed. A worker that dies leaves a lease that expires and
  the job returns to the pool (§42).
* **Heartbeats.** A long job extends its lease. If the extension fails the lease was lost, and the
  worker stops immediately rather than racing the worker that took it.
* **Graceful shutdown.** SIGTERM/SIGINT set a flag; the current job finishes and its lease is
  released cleanly instead of being left to expire.
* **Idempotency.** Handlers are keyed, so a resumed campaign re-enqueuing the same work is a
  lookup rather than a second execution (§41).
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from civitas.domain.enums import EventType
from civitas.persistence import jobs
from civitas.persistence.events import emit
from civitas.persistence.models import Heartbeat, Job
from civitas.persistence.types import utcnow

log = logging.getLogger(__name__)

#: A handler takes the session and the job, and returns a JSON-serialisable result.
Handler = Callable[[Session, Job], dict[str, Any]]

_HANDLERS: dict[str, Handler] = {}


def handler(kind: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        _HANDLERS[kind] = fn
        return fn

    return register


def get_handler(kind: str) -> Handler | None:
    return _HANDLERS.get(kind)


def registered_kinds() -> list[str]:
    return sorted(_HANDLERS)


@dataclass
class WorkerConfig:
    worker_id: str = field(default_factory=lambda: f"worker-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    kinds: list[str] | None = None
    lease_s: float = 120.0
    #: How often a long job extends its lease. Comfortably shorter than the lease, so a slow tick
    #: does not lose a lease that is being held correctly.
    heartbeat_s: float = 30.0
    poll_interval_s: float = 0.5
    #: Stop after this many jobs. Bounded runs make the worker usable from a notebook cell, where
    #: an unbounded loop would hold the cell forever (§37).
    max_jobs: int | None = None
    max_runtime_s: float | None = None
    idle_exit: bool = False


class Worker:
    """One worker. Concurrency is several of these, not threads inside one (§37)."""

    def __init__(self, session_factory: sessionmaker[Session], config: WorkerConfig | None = None):
        self._factory = session_factory
        self.config = config or WorkerConfig()
        self._stopping = threading.Event()
        self.jobs_processed = 0
        self.jobs_failed = 0

    # ------------------------------------------------------------------
    def request_stop(self, *_signal_args) -> None:
        """Ask the worker to stop after the current job. Safe from a signal handler."""
        self._stopping.set()

    def install_signal_handlers(self) -> None:
        """Graceful shutdown on SIGTERM/SIGINT (Part B §54).

        Only where signals can be installed: in a notebook or a worker thread this raises, and
        the loop's bounded modes are the shutdown mechanism there instead.
        """
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):  # pragma: no cover - not the main thread
                pass

    # ------------------------------------------------------------------
    def run(self) -> int:
        started = time.monotonic()
        self._touch(draining=False)
        while not self._stopping.is_set():
            if self.config.max_jobs is not None and self.jobs_processed >= self.config.max_jobs:
                break
            if (
                self.config.max_runtime_s is not None
                and time.monotonic() - started >= self.config.max_runtime_s
            ):
                break

            did_work = self.run_once()
            if not did_work:
                if self.config.idle_exit:
                    break
                time.sleep(self.config.poll_interval_s)

        self._touch(draining=True)
        return self.jobs_processed

    def run_once(self) -> bool:
        """Claim and run one job. Returns False when the queue had nothing to do."""
        session = self._factory()
        try:
            job = jobs.claim(
                session, worker_id=self.config.worker_id,
                kinds=self.config.kinds, lease_s=self.config.lease_s,
            )
            if job is None:
                session.commit()
                return False
            emit(
                session, workspace_id=job.workspace_id, type=EventType.JOB_LEASED,
                payload={"job_id": str(job.id), "kind": job.kind,
                         "worker": self.config.worker_id, "attempt": job.attempts},
            ) if job.workspace_id else None
            session.commit()
            job_id, kind, workspace_id = job.id, job.kind, job.workspace_id
        except Exception:
            session.rollback()
            session.close()
            raise

        heartbeat_stop = threading.Event()
        beater = threading.Thread(
            target=self._heartbeat_loop, args=(job_id, heartbeat_stop), daemon=True
        )
        beater.start()
        try:
            self._execute(session, job, kind, workspace_id)
        finally:
            heartbeat_stop.set()
            beater.join(timeout=2.0)
            session.close()
        return True

    def _execute(self, session: Session, job: Job, kind: str, workspace_id) -> None:
        fn = get_handler(kind)
        if fn is None:
            # An unknown kind is a deployment error, not a transient fault. Retrying it would
            # occupy a worker until the attempts are exhausted for no possible benefit, so the
            # attempts are exhausted *first* and `fail` therefore dead-letters immediately.
            session.execute(
                Job.__table__.update().where(Job.id == job.id).values(attempts=job.max_attempts)
            )
            session.flush()
            session.refresh(job)
            jobs.fail(session, job, f"no handler registered for job kind {kind!r}")
            session.commit()
            self.jobs_failed += 1
            return

        try:
            result = fn(session, job)
            jobs.complete(session, job, result if isinstance(result, dict) else {"result": result})
            if workspace_id:
                emit(
                    session, workspace_id=workspace_id, type=EventType.JOB_COMPLETED,
                    payload={"job_id": str(job.id), "kind": kind},
                )
            session.commit()
            self.jobs_processed += 1
        except Exception as exc:
            log.exception("job %s (%s) failed", job.id, kind)
            session.rollback()
            # A fresh transaction: the failed one may have left the session unusable, and the
            # failure record is exactly what must not be lost when the work is.
            jobs.fail(session, job, f"{type(exc).__name__}: {exc}")
            session.refresh(job)
            if workspace_id and job.status.value == "dead_letter":
                emit(
                    session, workspace_id=workspace_id, type=EventType.JOB_DEAD_LETTERED,
                    payload={"job_id": str(job.id), "kind": kind, "error": str(exc)[:1000]},
                )
            session.commit()
            self.jobs_failed += 1

    # ------------------------------------------------------------------
    def _heartbeat_loop(self, job_id: uuid.UUID, stop: threading.Event) -> None:
        """Extend the lease while the job runs, in its own session.

        Its own session because the job's session is inside a transaction the handler owns; a
        heartbeat sharing it would either block behind the handler or commit the handler's
        partial work.
        """
        while not stop.wait(self.config.heartbeat_s):
            session = self._factory()
            try:
                job = session.get(Job, job_id)
                if job is None or job.lease_id is None:
                    return
                if not jobs.heartbeat(session, job, lease_s=self.config.lease_s):
                    # The lease was lost. Stop the worker rather than let two workers run one
                    # job, which is the duplicate execution §42 forbids.
                    log.warning("lost lease on job %s; stopping", job_id)
                    self.request_stop()
                    return
                session.commit()
            except Exception:  # pragma: no cover - a heartbeat must never kill the worker
                session.rollback()
            finally:
                session.close()

    def _touch(self, *, draining: bool) -> None:
        session = self._factory()
        try:
            row = (
                session.query(Heartbeat)
                .filter(Heartbeat.worker_id == self.config.worker_id)
                .one_or_none()
            )
            if row is None:
                row = Heartbeat(worker_id=self.config.worker_id, last_seen_at=utcnow())
                session.add(row)
            row.last_seen_at = utcnow()
            row.jobs_processed = self.jobs_processed
            row.is_draining = draining
            row.info = {"pid": os.getpid(), "kinds": self.config.kinds or registered_kinds()}
            session.commit()
        except Exception:  # pragma: no cover
            session.rollback()
        finally:
            session.close()


def run_workers(
    session_factory: sessionmaker[Session],
    *,
    count: int = 1,
    config: WorkerConfig | None = None,
) -> list[Worker]:
    """Run several workers as threads and wait for them (Part B §37).

    Threads rather than processes because a Colab runtime cannot reliably fork a daemon, and
    because the workers are I/O-bound on the database and the provider. Each thread gets its own
    session — the leasing, not the GIL, is what serialises the work.
    """
    workers = [
        Worker(session_factory, WorkerConfig(**{
            **(config.__dict__ if config else {}),
            "worker_id": f"{(config.worker_id if config else 'worker')}-{i}",
        }))
        for i in range(count)
    ]
    threads = [threading.Thread(target=w.run, daemon=True) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return workers
