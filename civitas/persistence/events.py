"""The immutable event stream writer (Part B §45).

Every meaningful operation emits an event. The writer is the only sanctioned way to produce one,
because sequence allocation has to be serialised and because an event written without its
correlation and configuration context is not reconstructable later.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from civitas.domain.enums import EventType
from civitas.persistence.models.infra import Event, EventSequence


def _ensure_counter(session: Session, workspace_id: uuid.UUID) -> None:
    """Create the counter row if it is missing, without racing another writer.

    A plain `SELECT … else INSERT` loses under concurrency: two transactions both see no row and
    both insert, and the second gets a unique violation. `ON CONFLICT DO NOTHING` makes the
    creation atomic — and where the row is being inserted by an uncommitted peer, the second
    writer blocks on the key until that peer commits and then correctly finds the row.

    SQLite has supported the clause since 3.24; PostgreSQL since 9.5. Both dialects therefore take
    the same code path, which is the point (Part B §6).
    """
    dialect = session.bind.dialect.name if session.bind is not None else "sqlite"
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    else:
        from sqlalchemy.dialects.sqlite import insert as dialect_insert

    session.execute(
        dialect_insert(EventSequence)
        .values(workspace_id=workspace_id, next_sequence=1)
        .on_conflict_do_nothing(index_elements=[EventSequence.workspace_id])
    )


def _next_sequence(session: Session, workspace_id: uuid.UUID) -> int:
    """Allocate the next per-workspace sequence number, serialised against concurrent writers.

    `MAX(sequence)+1` races: two writers read the same max and the unique constraint turns the
    loser into an error rather than a retry. A counter row serialises the allocation instead —
    locked `FOR UPDATE` on PostgreSQL, and on SQLite by the engine's `BEGIN IMMEDIATE`, which has
    already taken the write lock for the whole transaction.
    """
    _ensure_counter(session, workspace_id)

    stmt = select(EventSequence).where(EventSequence.workspace_id == workspace_id)
    if session.bind is not None and session.bind.dialect.name == "postgresql":
        stmt = stmt.with_for_update()

    row = session.execute(stmt).scalar_one()
    seq = row.next_sequence
    session.execute(
        update(EventSequence)
        .where(EventSequence.workspace_id == workspace_id)
        .values(next_sequence=seq + 1)
    )
    session.flush()
    return seq


def emit(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    type: EventType,
    payload: dict[str, Any] | None = None,
    project_id: uuid.UUID | None = None,
    task_id: uuid.UUID | None = None,
    episode_id: uuid.UUID | None = None,
    actor_kind: str = "system",
    actor_id: uuid.UUID | None = None,
    correlation_id: uuid.UUID | None = None,
    config_hash: str = "",
) -> Event:
    """Append one event. Never updates; there is no code path that edits an event (§45)."""
    ev = Event(
        workspace_id=workspace_id,
        sequence=_next_sequence(session, workspace_id),
        type=type,
        payload=payload or {},
        project_id=project_id,
        task_id=task_id,
        episode_id=episode_id,
        actor_kind=actor_kind,
        actor_id=actor_id,
        correlation_id=correlation_id,
        config_hash=config_hash,
    )
    session.add(ev)
    session.flush()
    return ev


def read_stream(
    session: Session,
    workspace_id: uuid.UUID,
    *,
    after_sequence: int = 0,
    types: list[EventType] | None = None,
    episode_id: uuid.UUID | None = None,
    limit: int = 500,
) -> list[Event]:
    """Read the stream in sequence order (Part B §45, §49 live updates).

    Ordered by `sequence`, never by timestamp: two events in the same millisecond are
    indistinguishable by clock, and a cursor built on timestamps silently skips or repeats rows.
    """
    stmt = select(Event).where(
        Event.workspace_id == workspace_id, Event.sequence > after_sequence
    )
    if types:
        stmt = stmt.where(Event.type.in_(types))
    if episode_id:
        stmt = stmt.where(Event.episode_id == episode_id)
    stmt = stmt.order_by(Event.sequence).limit(limit)
    return list(session.execute(stmt).scalars())
