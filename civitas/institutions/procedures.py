"""Institutional practice, applied at runtime (Part B §30).

> "Always reproduce this type of result before accepting it."
> "Tool X is unreliable for input class Y."
> "Before modifying subsystem Z, run test suite Q."

§30's examples are all *rules the runtime should apply*, not advice for a prompt. A procedure that
only appears in an agent's instructions is indistinguishable from a suggestion, and §30 ends by
asking whether an institution actually improves outcomes — a question that cannot be answered about
something the system does not enforce.

So a `Procedure` is a `trigger` (when it fires) and a `requirement` (what it demands), both
evaluated here. `times_applied` is the count of real firings, which is what makes the §30
measurement possible.

Only **approved** procedures apply. `active_procedures` goes through the A2.2 gate, so an agent
cannot institute a rule for the whole collective by writing one down.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import ArtifactType, EvidenceKind, ValidationState
from civitas.persistence.models import Artifact, Procedure

#: Trigger kinds a procedure may use. Closed, because a trigger the engine does not understand
#: would silently never fire — and a rule that never fires while appearing active is worse than no
#: rule, since the collective believes it is protected.
TRIGGER_KINDS = frozenset({
    "artifact_type",        # fires on artifacts of a given type
    "validation_state",     # fires on artifacts in a given validation state
    "confidence_above",     # fires on claims asserted above a confidence
    "evidence_kind",        # fires on artifacts resting on a given evidence kind
    "always",
})

#: Requirement kinds. Also closed, for the same reason.
REQUIREMENT_KINDS = frozenset({
    "min_evidence_kind",        # must rest on at least this strength of evidence
    "requires_replication",     # needs an independent reproduction
    "requires_validation_state",
    "max_confidence_without_evidence",
    "forbid_tool_for",          # "tool X is unreliable for input class Y"
})


@dataclass
class ProcedureCheck:
    procedure: Procedure
    fired: bool
    satisfied: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "procedure": self.procedure.name,
            "version": self.procedure.version,
            "fired": self.fired,
            "satisfied": self.satisfied,
            "detail": self.detail,
        }


@dataclass
class ProcedureReport:
    checks: list[ProcedureCheck] = field(default_factory=list)

    @property
    def violations(self) -> list[ProcedureCheck]:
        return [c for c in self.checks if c.fired and not c.satisfied]

    @property
    def compliant(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {
            "compliant": self.compliant,
            "checks": [c.as_dict() for c in self.checks],
            "violations": [c.as_dict() for c in self.violations],
        }


def active_procedures(session: Session, workspace_id: uuid.UUID) -> list[Procedure]:
    """Approved procedures only (§A2.2).

    An agent may *propose* a procedure freely; it governs nothing until a matched experiment says
    it improves outcomes. Otherwise any episode could institute a rule for the whole collective by
    writing it down.
    """
    return [
        p
        for p in session.execute(
            select(Procedure).where(
                Procedure.workspace_id == workspace_id,
                Procedure.status == "active",
                Procedure.approved_by_experiment_run_id.isnot(None),
                Procedure.rolled_back_at.is_(None),
                Procedure.archived_at.is_(None),
            )
        ).scalars()
    ]


def _fires(procedure: Procedure, artifact: Artifact) -> bool:
    trigger = procedure.trigger or {}
    kind = trigger.get("kind", "always")
    if kind not in TRIGGER_KINDS:
        # A trigger the engine cannot evaluate must not silently never fire. Treating it as
        # firing-and-unsatisfiable would be worse; surfacing it as a violation is the honest
        # option, and `check_artifact` does exactly that.
        return True
    if kind == "always":
        return True
    if kind == "artifact_type":
        return artifact.type.value in (trigger.get("values") or [])
    if kind == "validation_state":
        return artifact.validation_state.value in (trigger.get("values") or [])
    if kind == "confidence_above":
        return artifact.confidence > float(trigger.get("value", 1.0))
    if kind == "evidence_kind":
        return artifact.evidence_kind.value in (trigger.get("values") or [])
    return False


def _satisfied(
    session: Session, procedure: Procedure, artifact: Artifact
) -> tuple[bool, str]:
    from civitas.domain.enums import EVIDENCE_STRENGTH
    from civitas.knowledge.hypothesis import assess

    requirement = procedure.requirement or {}
    kind = requirement.get("kind")
    if kind not in REQUIREMENT_KINDS:
        return False, f"requirement kind {kind!r} is not one the engine can evaluate"

    if kind == "min_evidence_kind":
        needed = EvidenceKind(requirement["value"])
        ok = EVIDENCE_STRENGTH.get(artifact.evidence_kind, 0.0) >= EVIDENCE_STRENGTH.get(
            needed, 0.0
        )
        return ok, (
            f"rests on {artifact.evidence_kind.value}; requires at least {needed.value}"
        )

    if kind == "requires_replication":
        if artifact.type is not ArtifactType.HYPOTHESIS:
            return True, "not a hypothesis; replication requirement does not apply"
        replications = assess(session, artifact).independent_replications
        needed = int(requirement.get("value", 1))
        return replications >= needed, (
            f"{replications} independent replication(s); requires {needed}"
        )

    if kind == "requires_validation_state":
        allowed = set(requirement.get("values") or [])
        return artifact.validation_state.value in allowed, (
            f"validation is {artifact.validation_state.value}; requires one of {sorted(allowed)}"
        )

    if kind == "max_confidence_without_evidence":
        ceiling = float(requirement.get("value", 0.5))
        unvalidated = artifact.validation_state in (
            ValidationState.UNVALIDATED, ValidationState.SELF_REPORTED
        )
        ok = (not unvalidated) or artifact.confidence <= ceiling
        return ok, (
            f"unvalidated claim asserted at {artifact.confidence:.2f}; ceiling is {ceiling:.2f}"
        )

    if kind == "forbid_tool_for":
        # "Tool X is unreliable for input class Y" — evaluated against a tool run rather than an
        # artifact, so it never fires here and is checked by `check_tool_call`.
        return True, "evaluated at tool-call time"

    return False, "unreachable"


def check_artifact(
    session: Session, *, workspace_id: uuid.UUID, artifact: Artifact
) -> ProcedureReport:
    """Apply the workspace's institutional rules to an artifact (§30)."""
    report = ProcedureReport()
    for procedure in active_procedures(session, workspace_id):
        trigger_kind = (procedure.trigger or {}).get("kind", "always")
        if trigger_kind not in TRIGGER_KINDS:
            report.checks.append(ProcedureCheck(
                procedure=procedure, fired=True, satisfied=False,
                detail=f"trigger kind {trigger_kind!r} is not one the engine can evaluate",
            ))
            continue
        if not _fires(procedure, artifact):
            report.checks.append(ProcedureCheck(procedure, fired=False, satisfied=True))
            continue
        satisfied, detail = _satisfied(session, procedure, artifact)
        procedure.times_applied += 1
        report.checks.append(ProcedureCheck(procedure, fired=True, satisfied=satisfied,
                                            detail=detail))
    session.flush()
    return report


def check_tool_call(
    session: Session, *, workspace_id: uuid.UUID, tool_name: str, args: dict[str, Any]
) -> ProcedureReport:
    """Apply `forbid_tool_for` rules before a tool runs (§30's second example)."""
    report = ProcedureReport()
    for procedure in active_procedures(session, workspace_id):
        requirement = procedure.requirement or {}
        if requirement.get("kind") != "forbid_tool_for":
            continue
        if requirement.get("tool") != tool_name:
            report.checks.append(ProcedureCheck(procedure, fired=False, satisfied=True))
            continue
        field_name = requirement.get("argument")
        forbidden = set(requirement.get("values") or [])
        value = str(args.get(field_name, ""))
        fired = value in forbidden
        if fired:
            procedure.times_applied += 1
        report.checks.append(ProcedureCheck(
            procedure=procedure, fired=fired, satisfied=not fired,
            detail=f"{tool_name}({field_name}={value!r}) is documented as unreliable"
            if fired else "",
        ))
    session.flush()
    return report


def institution_effect(
    session: Session, *, workspace_id: uuid.UUID, procedure: Procedure
) -> dict[str, Any]:
    """Whether an institution actually improved outcomes (§30's closing requirement).

    Answered from the approval evidence — which is a *matched* comparison the A2.2 gate verified —
    rather than from a before/after on the live workspace. A before/after would confound the
    procedure with everything else that changed while it was active, and would credit the
    procedure for the collective simply maturing.
    """
    evidence = procedure.approval_evidence or {}
    if not evidence:
        return {
            "measured": False,
            "reason": "this procedure has no approval evidence; it was never gated",
            "times_applied": procedure.times_applied,
        }
    return {
        "measured": True,
        "metric": evidence.get("metric"),
        "treatment": evidence.get("treatment"),
        "control": evidence.get("control"),
        "improvement": (
            round(float(evidence["treatment"]) - float(evidence["control"]), 6)
            if evidence.get("treatment") is not None and evidence.get("control") is not None
            else None
        ),
        "times_applied": procedure.times_applied,
        "config_hash": evidence.get("config_hash"),
    }
