"""Declarative base and the mixins every G table carries."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from civitas_g.persistence.types import GUID, UTCDateTime, new_uuid


class Base(DeclarativeBase):
    """The G lineage's metadata, separate from the original package's.

    Separate on purpose: this is a rebuild, and sharing metadata would mean a `create_all` here
    also created the tables of the layers B§1 makes dormant or removes.
    """


class UUIDPrimaryKey:
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=new_uuid)


class Timestamped:
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, server_default=func.now(), index=True
    )
