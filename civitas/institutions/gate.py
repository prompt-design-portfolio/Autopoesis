"""The controlled-experiment gate (Part A §A2.2, Part B §31, §60).

> A new version becomes **active only after a matched A/B experiment** [...] The runtime **loads
> only experiment-approved versions**. Uncontrolled self-modification of critical infrastructure
> is impossible by construction, not by convention.

"By construction" is the whole requirement, and it is what makes this module different from a
review process. There is exactly one function the runtime uses to load a versioned artifact —
`load_active` — and it filters on `approved_by_experiment_run_id IS NOT NULL`. A caller that wants
an unapproved version cannot get one by asking differently; `propose` creates versions in
`proposed` state and nothing but `approve` moves them.

The gate governs everything §A2.2 names: prompts, retrieval policies, scheduler policies,
consolidation methods, agent role mixes, procedures and policies.

**What makes a comparison matched** is checkable rather than asserted: the two arms must share a
configuration hash, differ only in the version under test, and run the same task family under the
same budgets. `approve` verifies all of that and refuses otherwise — an approval resting on an
unmatched comparison is worse than no approval, because it carries the authority of evidence.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import EventType
from civitas.persistence.events import emit
from civitas.persistence.models import (
    Artifact,
    ExperimentRun,
    Policy,
    Procedure,
    PromptVersion,
)
from civitas.persistence.types import canonical_json, utcnow

#: Everything the gate governs (Part A §A2.2). Adding a governed kind is adding an entry here,
#: which is deliberately the only place the list exists.
GOVERNED_POLICY_KINDS = frozenset({
    "retrieval", "scheduler", "consolidation", "role_mix", "verification", "tool_use",
})

Versioned = Policy | Procedure | PromptVersion


class GateViolation(RuntimeError):
    """An attempt to activate a version the gate has not approved."""


class UnmatchedComparison(GateViolation):
    """The experiment offered as evidence was not a matched A/B."""


@dataclass
class MatchCheck:
    matched: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"matched": self.matched, "reasons": self.reasons}


def body_hash(body: dict[str, Any] | str) -> str:
    blob = body if isinstance(body, str) else canonical_json(body)
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------
# the loader — the entire enforcement surface
# --------------------------------------------------------------------------
def load_active(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    kind: str | None = None,
    name: str | None = None,
    model: type[Versioned] = Policy,
) -> Versioned | None:
    """The **only** way the runtime obtains a governed version.

    Filters on approval, on `status == "active"`, and on not having been rolled back. A version
    that has never won a matched experiment is invisible here however it was created, which is
    what "impossible by construction" means: there is no argument a caller can make to this
    function that returns an unapproved version.
    """
    stmt = select(model).where(
        model.workspace_id == workspace_id,
        model.status == "active",
        model.approved_by_experiment_run_id.isnot(None),
        model.rolled_back_at.is_(None),
    )
    if name is not None:
        stmt = stmt.where(model.name == name)
    if kind is not None and model is Policy:
        stmt = stmt.where(Policy.kind == kind)
    stmt = stmt.order_by(model.version.desc())
    return session.execute(stmt).scalars().first()


def load_active_body(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    kind: str,
    name: str,
    default: dict[str, Any],
) -> dict[str, Any]:
    """An approved policy body, or the built-in default.

    The default is what the runtime ships with and is not itself a self-modification, so it needs
    no approval. Falling back to it — rather than to the most recent *proposed* version — is what
    keeps an unapproved proposal from taking effect merely by existing.
    """
    policy = load_active(session, workspace_id=workspace_id, kind=kind, name=name, model=Policy)
    return dict(policy.body) if policy is not None else dict(default)


# --------------------------------------------------------------------------
# proposal
# --------------------------------------------------------------------------
def propose_policy(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    kind: str,
    name: str,
    body: dict[str, Any],
    description: str = "",
    episode_id: uuid.UUID | None = None,
    config_hash: str = "",
) -> Policy:
    """Create a policy version in `proposed` state (§A2.2).

    Proposed is inert: `load_active` cannot see it. An agent may propose freely, which is the
    point — the gate constrains *activation*, not proposal, so the collective can still explore
    how to organise itself (§31) without any exploration taking effect unmeasured.
    """
    if kind not in GOVERNED_POLICY_KINDS:
        raise ValueError(
            f"{kind!r} is not a governed policy kind; expected one of "
            f"{sorted(GOVERNED_POLICY_KINDS)}"
        )
    next_version = _next_version(session, Policy, workspace_id, name, kind=kind)
    policy = Policy(
        workspace_id=workspace_id, kind=kind, name=name, description=description,
        body=body, body_hash=body_hash(body), version=next_version, status="proposed",
        proposed_by_episode_id=episode_id,
    )
    session.add(policy)
    session.flush()

    artifact = Artifact(
        workspace_id=workspace_id,
        type=__import__("civitas.domain.enums", fromlist=["ArtifactType"]).ArtifactType.POLICY,
        title=f"Policy proposal: {kind}/{name} v{next_version}",
        body=description or canonical_json(body),
        creator_episode_id=episode_id,
        creator_kind="agent" if episode_id else "system",
        structured={"policy_id": str(policy.id), "kind": kind, "version": next_version},
    )
    session.add(artifact)
    session.flush()
    policy.artifact_id = artifact.id

    emit(
        session, workspace_id=workspace_id, type=EventType.POLICY_CHANGED,
        episode_id=episode_id,
        payload={"policy_id": str(policy.id), "kind": kind, "name": name,
                 "version": next_version, "status": "proposed"},
        config_hash=config_hash,
    )
    return policy


def propose_procedure(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    name: str,
    statement: str,
    trigger: dict[str, Any],
    requirement: dict[str, Any],
    episode_id: uuid.UUID | None = None,
    config_hash: str = "",
) -> Procedure:
    """Propose an institutional practice (§30), inert until approved."""
    next_version = _next_version(session, Procedure, workspace_id, name)
    procedure = Procedure(
        workspace_id=workspace_id, name=name, statement=statement,
        trigger=trigger, requirement=requirement, version=next_version, status="proposed",
        proposed_by_episode_id=episode_id,
    )
    session.add(procedure)
    session.flush()
    emit(
        session, workspace_id=workspace_id, type=EventType.PROCEDURE_CHANGED,
        episode_id=episode_id,
        payload={"procedure_id": str(procedure.id), "name": name, "version": next_version,
                 "status": "proposed"},
        config_hash=config_hash,
    )
    return procedure


def _next_version(
    session: Session, model: type[Versioned], workspace_id: uuid.UUID, name: str,
    kind: str | None = None,
) -> int:
    stmt = select(model.version).where(
        model.workspace_id == workspace_id, model.name == name
    )
    if kind is not None and model is Policy:
        stmt = stmt.where(Policy.kind == kind)
    versions = list(session.execute(stmt).scalars())
    return (max(versions) + 1) if versions else 1


# --------------------------------------------------------------------------
# the matched-comparison check
# --------------------------------------------------------------------------
def check_matched(
    session: Session,
    *,
    treatment_run: ExperimentRun,
    control_run: ExperimentRun,
    metric: str,
) -> MatchCheck:
    """Is this a matched A/B, and did the treatment win? (§A2.2)

    Every clause is a way a comparison can look like evidence without being it:

    * **the same experiment** — two runs from different campaigns share no task family or budget;
    * **the same configuration hash** — otherwise the arms differ in more than the version;
    * **both gate-passing** — a run whose prerequisites failed is not read at all
      (ARCHITECTURE §3.1), so it cannot support an approval either;
    * **the metric present in both** — comparing a number against a missing one is not a
      comparison;
    * **the treatment actually better** — stated last because it is the only clause anyone
      remembers to check.
    """
    check = MatchCheck(matched=True)

    if treatment_run.experiment_id != control_run.experiment_id:
        check.reasons.append("the runs belong to different experiments")
    if treatment_run.experiment_arm_id == control_run.experiment_arm_id:
        check.reasons.append("treatment and control are the same arm")
    if treatment_run.config_hash != control_run.config_hash:
        check.reasons.append(
            f"configuration hashes differ ({treatment_run.config_hash[:12]} vs "
            f"{control_run.config_hash[:12]}): the arms are not matched"
        )
    for run, label in ((treatment_run, "treatment"), (control_run, "control")):
        if run.status != "completed":
            check.reasons.append(f"the {label} run has not completed")
        if run.gates and not run.gates_passed:
            check.reasons.append(f"the {label} run failed its gates and is not read")

    treatment_value = (treatment_run.metrics or {}).get(metric)
    control_value = (control_run.metrics or {}).get(metric)
    if treatment_value is None or control_value is None:
        check.reasons.append(f"metric {metric!r} is missing from one of the runs")
    elif not float(treatment_value) > float(control_value):
        check.reasons.append(
            f"the treatment did not improve {metric}: "
            f"{treatment_value} vs control {control_value}"
        )

    check.matched = not check.reasons
    return check


# --------------------------------------------------------------------------
# approval and rollback
# --------------------------------------------------------------------------
def approve(
    session: Session,
    version: Versioned,
    *,
    treatment_run: ExperimentRun,
    control_run: ExperimentRun,
    metric: str,
    config_hash: str = "",
) -> Versioned:
    """Activate a version, but only on a matched comparison it won (§A2.2).

    Refuses otherwise. An approval resting on an unmatched comparison is worse than no approval,
    because it carries the authority of evidence and nothing downstream re-checks it.
    """
    check = check_matched(
        session, treatment_run=treatment_run, control_run=control_run, metric=metric
    )
    if not check.matched:
        raise UnmatchedComparison(
            "cannot approve on this comparison (Part A §A2.2): " + "; ".join(check.reasons)
        )

    previous = load_active(
        session, workspace_id=version.workspace_id, name=version.name,
        kind=getattr(version, "kind", None), model=type(version),
    )
    if previous is not None and previous.id != version.id:
        previous.status = "superseded"
        previous.superseded_by_id = version.id

    version.status = "active"
    version.approved_by_experiment_run_id = treatment_run.id
    version.approved_at = utcnow()
    version.approval_evidence = {
        "metric": metric,
        "treatment": (treatment_run.metrics or {}).get(metric),
        "control": (control_run.metrics or {}).get(metric),
        "treatment_run_id": str(treatment_run.id),
        "control_run_id": str(control_run.id),
        "config_hash": treatment_run.config_hash,
        "superseded": str(previous.id) if previous and previous.id != version.id else None,
    }
    session.flush()

    emit(
        session, workspace_id=version.workspace_id, type=EventType.VERSION_APPROVED,
        payload={"id": str(version.id), "name": version.name,
                 "version": version.version, "evidence": version.approval_evidence},
        config_hash=config_hash,
    )
    return version


def roll_back(
    session: Session,
    version: Versioned,
    *,
    reason: str,
    restore_previous: bool = True,
    config_hash: str = "",
) -> Versioned | None:
    """Withdraw an active version and restore its predecessor (§60: reversible).

    Returns the restored predecessor, or `None` when there was none — in which case the runtime
    falls back to its built-in default, which is the correct outcome rather than an error: the
    default is what the system shipped with and needs no approval.
    """
    version.rolled_back_at = utcnow()
    version.rollback_reason = reason
    version.status = "rolled_back"

    restored: Versioned | None = None
    if restore_previous:
        candidates = list(
            session.execute(
                select(type(version)).where(
                    type(version).workspace_id == version.workspace_id,
                    type(version).name == version.name,
                    type(version).id != version.id,
                    type(version).approved_by_experiment_run_id.isnot(None),
                    type(version).rolled_back_at.is_(None),
                ).order_by(type(version).version.desc())
            ).scalars()
        )
        if candidates:
            restored = candidates[0]
            restored.status = "active"
            restored.superseded_by_id = None
    session.flush()

    emit(
        session, workspace_id=version.workspace_id, type=EventType.VERSION_ROLLED_BACK,
        payload={"id": str(version.id), "name": version.name, "version": version.version,
                 "reason": reason[:1000],
                 "restored": str(restored.id) if restored else None},
        config_hash=config_hash,
    )
    return restored


# --------------------------------------------------------------------------
# reporting (§A2.2 metric: adoption rate, attributed improvement, rollback count)
# --------------------------------------------------------------------------
def gate_metrics(session: Session, workspace_id: uuid.UUID) -> dict[str, Any]:
    """The three numbers §A2.2 names."""
    out: dict[str, Any] = {}
    for label, model in (("policy", Policy), ("procedure", Procedure), ("prompt", PromptVersion)):
        rows = list(
            session.execute(
                select(model).where(model.workspace_id == workspace_id)
            ).scalars()
        )
        proposed = len(rows)
        approved = sum(1 for r in rows if r.approved_by_experiment_run_id is not None)
        rolled_back = sum(1 for r in rows if r.rolled_back_at is not None)
        improvements = [
            (r.approval_evidence or {}).get("treatment", 0)
            - (r.approval_evidence or {}).get("control", 0)
            for r in rows
            if r.approval_evidence and r.approval_evidence.get("treatment") is not None
        ]
        out[label] = {
            "proposed": proposed,
            "approved": approved,
            # `None`, not 0.0, with nothing proposed: an adoption rate over an empty set is not
            # zero adoption (ARCHITECTURE §3.9).
            "adoption_rate": round(approved / proposed, 4) if proposed else None,
            "rolled_back": rolled_back,
            "mean_attributed_improvement": (
                round(sum(improvements) / len(improvements), 4) if improvements else None
            ),
        }
    return out
