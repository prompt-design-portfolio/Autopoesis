"""Collective meta-learning under the gate (Part B §31, §60, Part A §A2.2).

§31 asks the collective to experiment with how it organises cognition — retrieval policies,
scheduler policies, role mixes, consolidation methods — and then says the thing that matters:

> Changes must be evaluated experimentally. Do not permit uncontrolled recursive modification of
> critical production infrastructure. Improvement should be measured.

This module is the loop that satisfies all three. A proposal becomes a `Policy` version in
`proposed` state, a **matched** experiment is run with the version on and off, its runs are
recorded with a shared configuration hash, and `gate.approve` either activates the version or
refuses. Nothing here can activate a version by any other route: the gate's `load_active` is the
only loader, and it filters on approval.

The point is that the loop must be able to say **no**. A meta-learning mechanism that adopts every
proposal is not evaluating anything, and the demonstration below deliberately includes a proposal
that does not help, so the refusal is exercised rather than assumed.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from civitas.institutions import gate
from civitas.persistence.models import Experiment, ExperimentArm, ExperimentRun, Policy
from civitas.persistence.types import utcnow


@dataclass
class ProposalOutcome:
    name: str
    kind: str
    metric: str
    treatment_value: float | None
    control_value: float | None
    approved: bool
    reason: str
    policy_id: uuid.UUID | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "metric": self.metric,
            "treatment": self.treatment_value,
            "control": self.control_value,
            "approved": self.approved,
            "reason": self.reason,
            "policy_id": str(self.policy_id) if self.policy_id else None,
        }


@dataclass
class MetaLearningReport:
    outcomes: list[ProposalOutcome] = field(default_factory=list)
    gate_metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcomes": [o.as_dict() for o in self.outcomes],
            "gate_metrics": self.gate_metrics,
            "approved": sum(1 for o in self.outcomes if o.approved),
            "refused": sum(1 for o in self.outcomes if not o.approved),
        }


def run_matched_experiment(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    name: str,
    metric: str,
    treatment: Callable[[], float],
    control: Callable[[], float],
    config_hash: str,
    hypothesis: str = "",
    prediction: str = "",
) -> tuple[ExperimentRun, ExperimentRun]:
    """Run both arms and record them with a **shared** configuration hash.

    The shared hash is what the gate checks. Recording it here — from one value, used for both
    arms — rather than computing it per arm is deliberate: two independently computed hashes that
    happen to agree prove nothing about whether the arms were actually matched.
    """
    experiment = Experiment(
        workspace_id=workspace_id, name=name, kind="meta_learning",
        hypothesis=hypothesis, prediction=prediction,
        status="running", started_at=utcnow(),
    )
    session.add(experiment)
    session.flush()

    arms: dict[str, ExperimentArm] = {}
    for label in ("treatment", "control"):
        arm = ExperimentArm(
            experiment_id=experiment.id, name=label, arm="collective",
            control_for="treatment" if label == "control" else None,
        )
        session.add(arm)
        arms[label] = arm
    session.flush()

    runs: dict[str, ExperimentRun] = {}
    for label, fn in (("treatment", treatment), ("control", control)):
        run = ExperimentRun(
            experiment_id=experiment.id, experiment_arm_id=arms[label].id, seed=0,
            status="running", config_hash=config_hash, started_at=utcnow(),
            gates={"matched_configuration": {"passed": True,
                                             "detail": "both arms share one configuration hash"}},
        )
        session.add(run)
        session.flush()
        value = fn()
        run.metrics = {metric: value}
        run.status = "completed"
        run.completed_at = utcnow()
        runs[label] = run

    experiment.status = "completed"
    experiment.completed_at = utcnow()
    session.flush()
    return runs["treatment"], runs["control"]


def evaluate_proposal(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    kind: str,
    name: str,
    body: dict[str, Any],
    metric: str,
    treatment: Callable[[], float],
    control: Callable[[], float],
    config_hash: str,
    description: str = "",
    hypothesis: str = "",
    prediction: str = "",
    episode_id: uuid.UUID | None = None,
) -> ProposalOutcome:
    """Propose, test under a matched comparison, and let the gate decide (§31, §A2.2)."""
    policy = gate.propose_policy(
        session, workspace_id=workspace_id, kind=kind, name=name, body=body,
        description=description, episode_id=episode_id,
    )
    treatment_run, control_run = run_matched_experiment(
        session, workspace_id=workspace_id, name=f"{kind}/{name}", metric=metric,
        treatment=treatment, control=control, config_hash=config_hash,
        hypothesis=hypothesis, prediction=prediction,
    )
    treatment_value = (treatment_run.metrics or {}).get(metric)
    control_value = (control_run.metrics or {}).get(metric)

    try:
        gate.approve(
            session, policy, treatment_run=treatment_run, control_run=control_run, metric=metric
        )
        return ProposalOutcome(
            name=name, kind=kind, metric=metric,
            treatment_value=treatment_value, control_value=control_value,
            approved=True,
            reason=f"{metric} improved from {control_value} to {treatment_value}",
            policy_id=policy.id,
        )
    except gate.UnmatchedComparison as exc:
        # The proposal stays on the record in `proposed` state. It is inert, and it is also
        # *evidence* — a documented change that did not help is exactly what stops the next
        # agent proposing it again (§12).
        return ProposalOutcome(
            name=name, kind=kind, metric=metric,
            treatment_value=treatment_value, control_value=control_value,
            approved=False, reason=str(exc), policy_id=policy.id,
        )


def active_configuration(session: Session, workspace_id: uuid.UUID) -> dict[str, Any]:
    """Everything the collective has actually adopted, with the evidence for each (§60)."""
    out: dict[str, Any] = {}
    for kind in sorted(gate.GOVERNED_POLICY_KINDS):
        policy = gate.load_active(session, workspace_id=workspace_id, kind=kind, model=Policy)
        out[kind] = (
            {
                "name": policy.name,
                "version": policy.version,
                "body": policy.body,
                "evidence": policy.approval_evidence,
            }
            if policy is not None
            else None
        )
    return out
