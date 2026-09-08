"""Specialization and profile learning (Part B §26).

> But do not hard-code all specialization permanently. Track performance by work type. Allow
> assignment strategies to discover which profiles work best for which tasks.

So `AgentProfile.performance_by_work_type` is a **fold over evaluations**, computed here and
written nowhere else. A profile never declares what it is good at; the record says what it has
been good at, and the `specialization` and `diversity` allocation strategies read that.

The measurement has one trap worth naming. Early in a workspace, a profile that happened to draw
two easy tasks looks better than one that drew two hard ones, and a scheduler that believed the
early numbers would entrench that accident for the rest of the run. Two things guard against it:
every rate is reported **with its sample size**, and `_profile_fit` returns 0.5 for an unmeasured
pairing rather than 0.0 — unknown is not the same as bad.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import TerminationReason
from civitas.persistence.models import AgentProfile, Episode, Evaluation, Task

#: Below this many episodes a rate is reported but should not drive allocation on its own.
#: Recorded on every entry so a reader — and the scheduler — can tell a measurement from an
#: accident.
MIN_CONFIDENT_N = 5


@dataclass
class ProfilePerformance:
    profile: str
    work_type: str
    n: int
    successes: int
    success_rate: float
    mean_tokens: float
    mean_tool_calls: float
    confident: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "mean_tokens": round(self.mean_tokens, 1),
            "mean_tool_calls": round(self.mean_tool_calls, 2),
            "confident": self.confident,
        }


@dataclass
class SpecializationReport:
    profiles_updated: int = 0
    entries: list[ProfilePerformance] = field(default_factory=list)
    #: How unevenly work types are distributed across profiles, in [0, 1]. This is the §48
    #: `specialization` metric: 0 means every profile performs identically across every work
    #: type, 1 means each work type has exactly one profile that handles it.
    specialization_index: float | None = None
    index_unavailable_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "profiles_updated": self.profiles_updated,
            "specialization_index": self.specialization_index,
            "index_unavailable_reason": self.index_unavailable_reason,
            "entries": [
                {"profile": e.profile, "work_type": e.work_type, **e.as_dict()}
                for e in self.entries
            ],
        }


def measure(session: Session, workspace_id: uuid.UUID) -> list[ProfilePerformance]:
    """Fold every evaluated episode into (profile, work type) performance.

    Benchmark probe episodes are excluded: they are measured *against* the collective and are not
    part of it (ARCHITECTURE §3.5). Including them would let the measurement contaminate the thing
    being measured.
    """
    rows = session.execute(
        select(Episode, Task.task_family, Evaluation.succeeded)
        .join(Task, Task.id == Episode.task_id)
        .join(Evaluation, Evaluation.episode_id == Episode.id)
        .where(
            Episode.workspace_id == workspace_id,
            Episode.is_benchmark_probe.is_(False),
            Evaluation.scope == "episode",
        )
    ).all()

    buckets: dict[tuple[uuid.UUID, str], dict[str, float]] = {}
    for episode, work_type, succeeded in rows:
        # An infrastructure failure says nothing about the profile, and counting it would make a
        # flaky provider look like a weak specialist.
        if episode.termination_reason in (
            TerminationReason.PROVIDER_FAILURE, TerminationReason.WORKER_FAILURE
        ):
            continue
        key = (episode.agent_profile_id, work_type or "")
        bucket = buckets.setdefault(
            key, {"n": 0.0, "successes": 0.0, "tokens": 0.0, "tool_calls": 0.0}
        )
        bucket["n"] += 1
        bucket["successes"] += 1 if succeeded else 0
        bucket["tokens"] += episode.tokens_used
        bucket["tool_calls"] += episode.tool_calls_used

    out: list[ProfilePerformance] = []
    for (profile_id, work_type), bucket in buckets.items():
        profile = session.get(AgentProfile, profile_id)
        n = int(bucket["n"])
        out.append(ProfilePerformance(
            profile=profile.name if profile else str(profile_id),
            work_type=work_type,
            n=n,
            successes=int(bucket["successes"]),
            success_rate=bucket["successes"] / n if n else 0.0,
            mean_tokens=bucket["tokens"] / n if n else 0.0,
            mean_tool_calls=bucket["tool_calls"] / n if n else 0.0,
            confident=n >= MIN_CONFIDENT_N,
        ))
    return out


def update_profiles(session: Session, workspace_id: uuid.UUID) -> SpecializationReport:
    """Write the measured performance back onto the profiles (§26)."""
    report = SpecializationReport(entries=measure(session, workspace_id))

    by_profile: dict[str, dict[str, Any]] = {}
    for entry in report.entries:
        by_profile.setdefault(entry.profile, {})[entry.work_type] = entry.as_dict()

    for profile in session.execute(
        select(AgentProfile).where(AgentProfile.workspace_id == workspace_id)
    ).scalars():
        table = by_profile.get(profile.name)
        if table is not None:
            profile.performance_by_work_type = table
            report.profiles_updated += 1

    session.flush()
    report.specialization_index, report.index_unavailable_reason = _index(report.entries)
    return report


def _index(entries: list[ProfilePerformance]) -> tuple[float | None, str | None]:
    """Division of cognitive labour, in [0, 1].

    For each work type, how concentrated success is on one profile. `None` — not 0.0 — where it
    cannot be computed: a single profile cannot specialise, and reporting zero would present
    "there is nobody to divide labour with" as "labour is not divided" (ARCHITECTURE §3.9).
    """
    work_types: dict[str, list[ProfilePerformance]] = {}
    for entry in entries:
        work_types.setdefault(entry.work_type, []).append(entry)

    usable = {
        work_type: rows for work_type, rows in work_types.items() if len(rows) >= 2
    }
    if not usable:
        return None, "fewer than two profiles have worked any single work type"

    scores: list[float] = []
    for rows in usable.values():
        rates = [r.success_rate for r in rows]
        total = sum(rates)
        if total <= 0:
            continue
        shares = [r / total for r in rates]
        # Normalised Herfindahl: 0 when every profile is equally good, 1 when one profile
        # accounts for all of the success.
        k = len(shares)
        herfindahl = sum(s * s for s in shares)
        scores.append((herfindahl - 1.0 / k) / (1.0 - 1.0 / k) if k > 1 else 0.0)

    if not scores:
        return None, "no work type has a profile that has succeeded at it"
    return round(sum(scores) / len(scores), 4), None


def best_profile_for(
    session: Session, workspace_id: uuid.UUID, work_type: str
) -> AgentProfile | None:
    """The profile with the best *confident* record on this work type (§26).

    Requires `MIN_CONFIDENT_N`: acting on a one-episode record would entrench whichever profile
    happened to draw the first easy instance. Returns None when nothing qualifies, so the caller
    falls back to a strategy that explores rather than to a guess dressed as a measurement.
    """
    best: AgentProfile | None = None
    best_rate = -1.0
    for profile in session.execute(
        select(AgentProfile).where(
            AgentProfile.workspace_id == workspace_id,
            AgentProfile.is_active.is_(True),
        )
    ).scalars():
        entry = (profile.performance_by_work_type or {}).get(work_type)
        if not isinstance(entry, dict) or not entry.get("confident"):
            continue
        rate = float(entry.get("success_rate", 0.0))
        if rate > best_rate:
            best, best_rate = profile, rate
    return best
