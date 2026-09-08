"""Organization resource limits (Part B §58).

§39 already bounds a single episode: tokens, tool calls, cost, wall clock. What it cannot bound is
*many* episodes. A scheduler that keeps finding ready work will keep spending, and every episode
is individually within budget while the organization's month is not.

So the limits here are over a rolling window and are computed from the same `Episode` rows the
cost dashboard reads. Nothing is denormalised into a running total, and that is deliberate: a
counter that drifts from the episodes it summarises is a quota that either blocks work that never
happened or allows work that did.

**Absent is not unlimited, and absent is not zero.** An organization with no configured quota is
unlimited — stated, tested, and reported by `remaining()` as `None` rather than as a large
number, so a dashboard cannot render "no limit" as a number close to running out.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from civitas.observability import REGISTRY
from civitas.persistence.models import Episode, Organization, Workspace
from civitas.persistence.types import utcnow

#: Where a quota lives. On `Organization.meta` rather than in a table of its own: quotas are
#: configuration, they change by hand, and a table would need a migration to add the third limit
#: someone asks for. The shape is validated on read.
META_KEY = "quotas"

#: The window every limit is measured over unless one says otherwise.
DEFAULT_WINDOW = timedelta(days=30)


class QuotaExceeded(RuntimeError):
    """An action was refused because it would exceed an organization's limit."""

    def __init__(self, kind: str, used: float, limit: float, window_days: float):
        super().__init__(
            f"organization quota exceeded: {kind} {used:g} of {limit:g} "
            f"in the last {window_days:g} days"
        )
        self.kind = kind
        self.used = used
        self.limit = limit


@dataclass(frozen=True)
class Quota:
    """The limits an organization runs under. `None` means unlimited for that dimension."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_episodes: int | None = None
    window_days: float = 30.0

    @classmethod
    def parse(cls, raw: Any) -> Quota:
        if not isinstance(raw, dict):
            return cls()
        def _num(key: str, cast: Any) -> Any:
            value = raw.get(key)
            if value is None:
                return None
            try:
                parsed = cast(value)
            except (TypeError, ValueError):
                return None
            # A negative or zero limit is refused rather than treated as "block everything": it is
            # almost always a typo, and a quota that silently halts an organization is worse than
            # one that is ignored and visible in the report.
            return parsed if parsed > 0 else None
        return cls(
            max_tokens=_num("max_tokens", int),
            max_cost_usd=_num("max_cost_usd", float),
            max_episodes=_num("max_episodes", int),
            window_days=float(raw.get("window_days") or 30.0),
        )

    @property
    def unlimited(self) -> bool:
        return (
            self.max_tokens is None
            and self.max_cost_usd is None
            and self.max_episodes is None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "max_cost_usd": self.max_cost_usd,
            "max_episodes": self.max_episodes,
            "window_days": self.window_days,
            "unlimited": self.unlimited,
        }


@dataclass(frozen=True)
class Usage:
    tokens: int
    cost_usd: float
    episodes: int
    window_days: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens, "cost_usd": round(self.cost_usd, 6),
            "episodes": self.episodes, "window_days": self.window_days,
        }


def quota_for(session: Session, organization_id: uuid.UUID) -> Quota:
    organization = session.get(Organization, organization_id)
    if organization is None:
        raise ValueError(f"no organization {organization_id}")
    return Quota.parse((organization.meta or {}).get(META_KEY))


def set_quota(session: Session, organization_id: uuid.UUID, quota: Quota) -> Quota:
    organization = session.get(Organization, organization_id)
    if organization is None:
        raise ValueError(f"no organization {organization_id}")
    meta = dict(organization.meta or {})
    meta[META_KEY] = {
        "max_tokens": quota.max_tokens, "max_cost_usd": quota.max_cost_usd,
        "max_episodes": quota.max_episodes, "window_days": quota.window_days,
    }
    # Reassigned rather than mutated: a JSON column mutated in place is not seen as dirty by
    # SQLAlchemy, and the write is silently lost.
    organization.meta = meta
    session.flush()
    return quota


def usage(session: Session, organization_id: uuid.UUID, *, window_days: float = 30.0) -> Usage:
    """What the organization has spent in the window, computed from episodes.

    Benchmark probes are counted. They cost the same tokens and the same money as any other
    episode, and excluding them would let a campaign spend an organization's month while the
    quota report showed it idle.
    """
    since = utcnow() - timedelta(days=window_days)
    row = session.execute(
        select(
            func.coalesce(func.sum(Episode.tokens_used), 0),
            func.coalesce(func.sum(Episode.cost_usd), 0.0),
            func.count(Episode.id),
        )
        .select_from(Episode)
        .join(Workspace, Workspace.id == Episode.workspace_id)
        .where(Workspace.organization_id == organization_id, Episode.created_at >= since)
    ).one()
    return Usage(
        tokens=int(row[0] or 0), cost_usd=float(row[1] or 0.0), episodes=int(row[2] or 0),
        window_days=window_days,
    )


def remaining(session: Session, organization_id: uuid.UUID) -> dict[str, Any]:
    """Headroom per dimension. `None` where there is no limit — never a large number."""
    quota = quota_for(session, organization_id)
    spent = usage(session, organization_id, window_days=quota.window_days)
    return {
        "quota": quota.as_dict(),
        "usage": spent.as_dict(),
        "remaining": {
            "tokens": None if quota.max_tokens is None
                      else max(0, quota.max_tokens - spent.tokens),
            "cost_usd": None if quota.max_cost_usd is None
                        else round(max(0.0, quota.max_cost_usd - spent.cost_usd), 6),
            "episodes": None if quota.max_episodes is None
                        else max(0, quota.max_episodes - spent.episodes),
        },
    }


def check(session: Session, organization_id: uuid.UUID) -> None:
    """Raise `QuotaExceeded` when the organization is already over a limit.

    Checked *before* the work, not after: an episode that starts inside the limit and ends outside
    it has already been paid for, and the only thing a post-hoc check can do is report the
    overrun. This is the same discipline `BudgetTracker` uses for a single episode (§39).
    """
    quota = quota_for(session, organization_id)
    if quota.unlimited:
        return
    spent = usage(session, organization_id, window_days=quota.window_days)

    for kind, used, limit in (
        ("tokens", spent.tokens, quota.max_tokens),
        ("cost_usd", spent.cost_usd, quota.max_cost_usd),
        ("episodes", spent.episodes, quota.max_episodes),
    ):
        if limit is not None and used >= limit:
            REGISTRY.inc("civitas_quota_denied_total", kind=kind)
            raise QuotaExceeded(kind, used, limit, quota.window_days)


def check_workspace(session: Session, workspace_id: uuid.UUID) -> None:
    """`check`, resolved from a workspace — the identifier most callers actually hold."""
    workspace = session.get(Workspace, workspace_id)
    if workspace is None:
        raise ValueError(f"no workspace {workspace_id}")
    check(session, workspace.organization_id)
