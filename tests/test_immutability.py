"""Append-only enforcement (Part B §45, Part A §A1.2).

These drive the session guard, not the `Immutable` marker. A marker that nothing checks is a
naming convention, and the claim that a result is historically reconstructable rests on the event
stream not having been edited after the fact.
"""

from __future__ import annotations

import pytest

from civitas.domain.enums import EventType
from civitas.persistence.events import emit, read_stream
from civitas.persistence.models import ArtifactUtilityMetric, Event
from civitas.persistence.session import ImmutableViolation


def test_events_cannot_be_updated(db, workspace):
    ev = emit(db, workspace_id=workspace.id, type=EventType.ARTIFACT_CREATED, payload={"n": 1})
    db.commit()

    ev.payload = {"n": 2}
    with pytest.raises(ImmutableViolation, match="append-only"):
        db.commit()
    db.rollback()

    assert db.get(Event, ev.id).payload == {"n": 1}


def test_events_cannot_be_deleted(db, workspace):
    ev = emit(db, workspace_id=workspace.id, type=EventType.EPISODE_STARTED)
    db.commit()

    db.delete(ev)
    with pytest.raises(ImmutableViolation, match="cannot be deleted"):
        db.commit()
    db.rollback()

    assert db.get(Event, ev.id) is not None


def test_credit_events_cannot_be_rewritten(db, workspace):
    """Part A §A2.1: `downstream_utility` is a fold over these rows. An editable event would make
    that fold unreproducible, so utility could be inflated after the fact."""
    import uuid

    from civitas.domain.enums import ArtifactType
    from civitas.persistence.models import Artifact

    a = Artifact(workspace_id=workspace.id, type=ArtifactType.EVIDENCE, title="E")
    db.add(a)
    db.commit()

    m = ArtifactUtilityMetric(artifact_id=a.id, delta=0.1, graph_distance=0, reason="evaluator")
    db.add(m)
    db.commit()

    m.delta = 99.0
    with pytest.raises(ImmutableViolation):
        db.commit()
    db.rollback()
    assert db.get(ArtifactUtilityMetric, m.id).delta == pytest.approx(0.1)
    assert uuid.UUID(str(m.artifact_id)) == a.id


def test_event_sequence_is_monotonic_and_gapless(db, workspace):
    for i in range(25):
        emit(db, workspace_id=workspace.id, type=EventType.ARTIFACT_READ, payload={"i": i})
    db.commit()

    seqs = [e.sequence for e in read_stream(db, workspace.id, limit=100)]
    assert seqs == list(range(1, 26)), "sequences must be dense and ordered"


def test_sequences_are_independent_per_workspace(db, org, workspace):
    """A shared counter would make one workspace's activity visible in another's ordering."""
    import uuid

    from civitas.persistence.models import Workspace

    other = Workspace(organization_id=org.id, name="Other", slug=f"ws-{uuid.uuid4().hex[:8]}")
    db.add(other)
    db.commit()

    emit(db, workspace_id=workspace.id, type=EventType.TASK_CREATED)
    emit(db, workspace_id=other.id, type=EventType.TASK_CREATED)
    db.commit()

    assert read_stream(db, workspace.id)[0].sequence == 1
    assert read_stream(db, other.id)[0].sequence == 1


def test_stream_reads_after_a_cursor(db, workspace):
    for i in range(10):
        emit(db, workspace_id=workspace.id, type=EventType.MODEL_CALLED, payload={"i": i})
    db.commit()

    tail = read_stream(db, workspace.id, after_sequence=7)
    assert [e.sequence for e in tail] == [8, 9, 10]


def test_concurrent_writers_do_not_collide_on_sequence(session_factory, workspace):
    """Two sessions emitting at once must not both take the same sequence number.

    This is the race `MAX(sequence)+1` loses. It is driven with real concurrent sessions rather
    than asserted about the code, because the failure only appears under contention.
    """
    import threading

    errors: list[Exception] = []
    counts = 20

    def writer(n: int) -> None:
        try:
            for _ in range(counts):
                s = session_factory()
                try:
                    emit(s, workspace_id=workspace.id, type=EventType.ARTIFACT_READ,
                         payload={"w": n})
                    s.commit()
                finally:
                    s.close()
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent emit failed: {errors[:2]}"

    s = session_factory()
    try:
        seqs = sorted(e.sequence for e in read_stream(s, workspace.id, limit=1000))
    finally:
        s.close()
    assert len(seqs) == 3 * counts
    assert seqs == list(range(1, 3 * counts + 1)), "no duplicates and no gaps"
