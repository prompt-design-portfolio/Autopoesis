"""Duplicate-failure detection (Part A §A2.3, Part B §12, §48).

Two forms of repetition, per §A2.3: the same tool with the same arguments, and the same hypothesis
under the same environment version. Both are indexed at write time so the check is a lookup and
can run **before the action executes** rather than being an after-the-fact observation.

`duplicate_failure_rate` is a primary benchmark metric (§12, §48), so the event is emitted whenever
a repeat is detected — in every arm. What differs by arm is only whether the prior failure is
*surfaced* to the agent, because surfacing is the intervention and detection is the measurement.
Conflating the two would make the metric unavailable in exactly the arms it is needed to compare.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import EventType
from civitas.persistence.events import emit
from civitas.persistence.models import Artifact, DuplicateFailureRecord
from civitas.persistence.types import canonical_json

_WORD = re.compile(r"[a-z0-9]+")
#: Words carrying no discriminative content. A hypothesis key built from these would collide
#: across unrelated hypotheses and report duplicates that are not.
_STOP = frozenset(
    "a an the is are was were be been being of in on at to for with by from as that this it "
    "and or not no if then than there here we i you they he she its our their can could may "
    "might will would should must do does did done have has had".split()
)


def tool_key(tool_name: str, args: dict) -> str:
    """Identity of (tool, arguments). Canonical, so key order cannot hide a repeat."""
    return hashlib.sha256(f"{tool_name}|{canonical_json(args)}".encode()).hexdigest()


def hypothesis_key(hypothesis: str, environment_version: str) -> str:
    """Identity of (hypothesis, environment).

    Normalised to a sorted bag of content words, so two phrasings of one hypothesis collide and a
    rewording does not defeat the check. The environment version is part of the key because a
    hypothesis that failed against one environment is not thereby false against another (§15) —
    the same claim under a new environment is new work, not a repeat.
    """
    words = sorted({w for w in _WORD.findall(hypothesis.lower()) if w not in _STOP and len(w) > 2})
    blob = f"{environment_version}|{' '.join(words)}"
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class DuplicateHit:
    record: DuplicateFailureRecord
    artifact: Artifact | None
    kind: str  # "tool" | "hypothesis"

    def warning(self) -> str:
        lines = [
            f"A previous episode already tried this and it failed ({self.kind} match).",
            f"What happened: {self.record.summary}",
        ]
        if self.record.ruled_out:
            lines.append("Already ruled out: " + "; ".join(str(x) for x in self.record.ruled_out))
        if self.record.reproducible:
            lines.append("That failure was reproducible.")
        if self.artifact is not None:
            lines.append(f"Full record: artifact {self.artifact.id}")
        lines.append(
            "Do not repeat it unless you can state why it should differ this time."
        )
        return "\n".join(lines)


def record_failure(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    artifact_id: uuid.UUID | None,
    episode_id: uuid.UUID | None,
    environment_version: str,
    summary: str,
    hypothesis: str | None = None,
    tool_name: str | None = None,
    tool_args: dict | None = None,
    ruled_out: list | None = None,
    reproducible: bool = False,
) -> DuplicateFailureRecord:
    """Index a documented failure so a later episode's repeat can be caught before it runs."""
    record = DuplicateFailureRecord(
        workspace_id=workspace_id,
        artifact_id=artifact_id,
        episode_id=episode_id,
        environment_version=environment_version,
        summary=summary,
        ruled_out=ruled_out or [],
        reproducible=reproducible,
        tool_key=tool_key(tool_name, tool_args or {}) if tool_name else None,
        hypothesis_key=hypothesis_key(hypothesis, environment_version) if hypothesis else None,
    )
    session.add(record)
    session.flush()
    return record


def check(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    environment_version: str,
    tool_name: str | None = None,
    tool_args: dict | None = None,
    hypothesis: str | None = None,
    exclude_episode_id: uuid.UUID | None = None,
) -> DuplicateHit | None:
    """Look for a documented failure matching what is about to be attempted.

    `exclude_episode_id` keeps an episode from being warned about its own failure: repeating your
    own documented failure within one episode is a within-life mistake, not a transmission
    failure, and counting it would inflate the metric with something the collective cannot fix.
    """
    stmt = select(DuplicateFailureRecord).where(
        DuplicateFailureRecord.workspace_id == workspace_id
    )
    if exclude_episode_id is not None:
        stmt = stmt.where(
            (DuplicateFailureRecord.episode_id != exclude_episode_id)
            | DuplicateFailureRecord.episode_id.is_(None)
        )

    if tool_name is not None:
        key = tool_key(tool_name, tool_args or {})
        hit = session.execute(
            stmt.where(DuplicateFailureRecord.tool_key == key).limit(1)
        ).scalar_one_or_none()
        if hit is not None:
            return DuplicateHit(hit, _artifact(session, hit), "tool")

    if hypothesis:
        key = hypothesis_key(hypothesis, environment_version)
        hit = session.execute(
            stmt.where(
                DuplicateFailureRecord.hypothesis_key == key,
                DuplicateFailureRecord.environment_version == environment_version,
            ).limit(1)
        ).scalar_one_or_none()
        if hit is not None:
            return DuplicateHit(hit, _artifact(session, hit), "hypothesis")

    return None


def report(
    session: Session,
    hit: DuplicateHit,
    *,
    workspace_id: uuid.UUID,
    episode_id: uuid.UUID | None,
    surfaced: bool,
    config_hash: str = "",
) -> None:
    """Emit the `duplicate_failure` event and count the repeat.

    Emitted in *every* arm — detection is the measurement (§48). `surfaced` records whether the
    arm also showed the prior failure to the agent, which is the intervention.
    """
    hit.record.times_repeated += 1
    emit(
        session,
        workspace_id=workspace_id,
        type=EventType.DUPLICATE_FAILURE,
        episode_id=episode_id,
        payload={
            "record_id": str(hit.record.id),
            "kind": hit.kind,
            "surfaced": surfaced,
            "artifact_id": str(hit.artifact.id) if hit.artifact else None,
            "times_repeated": hit.record.times_repeated,
        },
        config_hash=config_hash,
    )


def _artifact(session: Session, record: DuplicateFailureRecord) -> Artifact | None:
    return session.get(Artifact, record.artifact_id) if record.artifact_id else None
