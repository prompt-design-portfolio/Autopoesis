"""Declarative base and the mixins every entity shares (Part B §7)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Index, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from civitas.persistence.types import GUID, JSONVariant, UTCDateTime, new_uuid, utcnow


class Base(DeclarativeBase):
    """Registry for every mapped class. Alembic autogenerates against this metadata."""


class UUIDPrimaryKey:
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=new_uuid)


class Timestamped:
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow, nullable=False
    )


class Metadataed:
    """Free-form structured metadata.

    Named `meta` on the Python side because `metadata` is taken by SQLAlchemy's declarative
    machinery; the column is `metadata_json` so no dialect treats it as reserved.
    """

    meta: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)


class Immutable:
    """Marker for tables that are append-only (Part B §45, §A1.2).

    Enforcement is in `civitas.persistence.session`, which rejects UPDATE and DELETE on any
    `Immutable` subclass at flush time. The marker alone does nothing; the guard is what holds,
    and `tests/test_immutability.py` drives the guard rather than the marker.
    """


class SoftDeletable:
    """Nothing is ever hard-deleted (Part A §A1.2).

    Records leave circulation by being archived or superseded, never by removal, because a result
    is only reconstructable if the evidence it rested on is still there to be read.
    """

    archived_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    archived_reason: Mapped[str | None] = mapped_column(default=None)

    @property
    def is_archived(self) -> bool:
        return self.archived_at is not None


def index_for(table: str, *columns: str, unique: bool = False) -> Index:
    return Index(f"ix_{table}_{'_'.join(columns)}", *columns, unique=unique)


@event.listens_for(Base, "before_insert", propagate=True)
def _stamp_created(_mapper, _conn, target) -> None:
    if isinstance(target, Timestamped) and target.created_at is None:
        target.created_at = utcnow()
