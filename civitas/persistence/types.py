"""Portable column types — the first of the three dialect seams (ARCHITECTURE §4.2).

PostgreSQL is the production source of truth and SQLite is the Colab-compatible implementation
(Part B §43). Part B §6 forbids separate application logic for Colab, so every dialect difference
is confined to a type decorator here and nothing above this module knows which backend it has.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
from sqlalchemy import CHAR, DateTime, String, Text, TypeDecorator
from sqlalchemy.dialects import postgresql


class GUID(TypeDecorator):
    """UUID primary keys (Part B §7). Native `uuid` on PostgreSQL, 32-char hex on SQLite.

    Values come back as `uuid.UUID` on both backends, so callers never branch on dialect.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(postgresql.UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        return value if dialect.name == "postgresql" else value.hex

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class JSONVariant(TypeDecorator):
    """Structured metadata (Part B §7, §9, §45). `JSONB` on PostgreSQL, JSON text on SQLite."""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(postgresql.JSONB())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(value, default=str, sort_keys=True)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.loads(value) if isinstance(value, str) else value


class UTCDateTime(TypeDecorator):
    """Timezone-aware timestamps on both backends.

    SQLite discards tzinfo, which silently turns an aware timestamp into a naive one and makes
    every duration and ordering comparison a landmine. Everything stored is normalised to UTC on
    the way in and re-attached to UTC on the way out, so `datetime` values are aware everywhere.
    """

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(DateTime(timezone=True))
        return dialect.type_descriptor(DateTime())

    def process_bind_param(self, value: datetime | None, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        value = value.astimezone(UTC)
        return value if dialect.name == "postgresql" else value.replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect):
        if value is None:
            return None
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class Vector(TypeDecorator):
    """Embeddings (Part B §43). `pgvector` where installed, a float32 blob otherwise.

    The blob form is exact, not quantised — a research record must not lose precision to storage
    convenience — and `VectorIndex` reads it back with numpy for brute-force search.
    """

    impl = Text
    cache_ok = True

    def __init__(self, dim: int = 0, **kw):
        self.dim = dim
        super().__init__(**kw)

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector as PGVector  # type: ignore

                return dialect.type_descriptor(PGVector(self.dim or None))
            except Exception:
                return dialect.type_descriptor(postgresql.BYTEA())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if dialect.name == "postgresql":
            try:
                import pgvector.sqlalchemy  # noqa: F401

                return arr.tolist()
            except Exception:
                return arr.tobytes()
        return arr.tobytes().hex()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, str):
            return np.frombuffer(bytes.fromhex(value), dtype=np.float32)
        if isinstance(value, bytes | memoryview):
            return np.frombuffer(bytes(value), dtype=np.float32)
        return np.asarray(value, dtype=np.float32)


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_uuid() -> uuid.UUID:
    return uuid.uuid4()


def canonical_json(value: Any) -> str:
    """Deterministic serialisation, for hashing arguments and configurations.

    Used by duplicate-failure detection (Part A §A2.3) to decide whether two tool calls are the
    same call, so key order and float formatting must not change the answer.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class EnumType(TypeDecorator):
    """A string-backed enum column that returns **enum members**, not bare strings.

    A plain `String` column round-trips an enum as `str`, and `episode.experiment_arm.value` then
    raises at a call site far from the model. That matters most for the experimental arms, which
    are machine-enforced (Part B §21): enforcement code has to be able to rely on the type it gets
    back from the database.

    Backed by `VARCHAR` rather than a native PostgreSQL `ENUM` because a native enum makes adding
    a value a migration with a lock, and the closed sets here are versioned in code.
    """

    impl = String
    cache_ok = True

    def __init__(self, enum_cls: type, length: int = 48, **kw):
        self.enum_cls = enum_cls
        super().__init__(length=length, **kw)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return value.value if isinstance(value, Enum) else str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        try:
            return self.enum_cls(value)
        except ValueError:
            # A value written by an older code version that no longer names it. Returning the raw
            # string keeps the row readable; the alternative is an unreadable database after a
            # rename, which would make old experiment records unreconstructable (Part B §46).
            return value
