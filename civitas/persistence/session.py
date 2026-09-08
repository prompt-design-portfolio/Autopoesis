"""Append-only enforcement (Part B §45, Part A §A1.2).

`Immutable` is a marker; this is the mechanism. Without a guard, "immutable" is a naming
convention that the first convenient `session.delete()` quietly repeals — and the entire claim
that a result is historically reconstructable rests on the event stream not having been edited.

The guard runs at flush, before SQL is emitted, so a violation is an exception rather than a
committed mutation that a test later notices.
"""

from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.orm import Session, sessionmaker

from civitas.persistence.base import Immutable


class ImmutableViolation(RuntimeError):
    """Raised when something tries to update or delete an append-only row."""


def _check(session: Session) -> None:
    for obj in session.dirty:
        if isinstance(obj, Immutable) and session.is_modified(obj, include_collections=False):
            raise ImmutableViolation(
                f"{type(obj).__name__} is append-only (Part B §45): "
                f"row {getattr(obj, 'id', '?')} was modified. "
                "Record a new row instead of editing history."
            )
    for obj in session.deleted:
        if isinstance(obj, Immutable):
            raise ImmutableViolation(
                f"{type(obj).__name__} is append-only (Part B §45): "
                f"row {getattr(obj, 'id', '?')} cannot be deleted. "
                "Nothing is ever hard-deleted (Part A §A1.2)."
            )


def install_guards(factory: sessionmaker[Session]) -> None:
    """Attach the append-only guard to every session the factory makes."""
    if getattr(factory, "_civitas_guarded", False):
        return

    @event.listens_for(factory, "before_flush")
    def _before_flush(session, _flush_context, _instances):  # pragma: no cover - via tests
        _check(session)

    factory._civitas_guarded = True  # type: ignore[attr-defined]
