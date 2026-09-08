"""Portable column types -- the dialect seam (§43, kept by B§1).

PostgreSQL is the source of truth and SQLite is the Colab-compatible implementation. §6 forbids
separate application logic for Colab, so every dialect difference lives in a type decorator and
nothing above this module knows which backend it has.

These decorators are **imported, not rewritten**. B§1 keeps §33-§46, the decorators in
`civitas.persistence.types` are the kept implementation, and they are already exercised against
both dialects. A second copy here would be a second thing to keep correct, and the failure mode of
a drifted copy is the worst kind: it would work on SQLite and be wrong on PostgreSQL, or store a
naive timestamp that every duration comparison downstream then reads as UTC.
"""

from __future__ import annotations

from civitas.persistence.types import (
    GUID,
    EnumType,
    JSONVariant,
    UTCDateTime,
    canonical_json,
    new_uuid,
    utcnow,
)

__all__ = [
    "GUID", "JSONVariant", "UTCDateTime", "EnumType",
    "canonical_json", "new_uuid", "utcnow",
]
