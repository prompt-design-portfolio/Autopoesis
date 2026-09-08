"""The institutional layer (Part B §29, §30, §31, §60) and the A2.2 gate."""

from __future__ import annotations

import uuid

import pytest

from civitas.domain.enums import (
    ArtifactType,
    EvidenceKind,
    TerminationReason,
    ValidationState,
)
from civitas.institutions import gate, procedures, reputation
from civitas.persistence.models import (
    AgentProfile,
    Artifact,
    Episode,
    Evaluation,
    Experiment,
    ExperimentArm,
    ExperimentRun,
    Policy,
    Procedure,
    ReputationMetric,
)


@pytest.fixture
def experiment(db, workspace):
    exp = Experiment(workspace_id=workspace.id, name="retrieval policy A/B", kind="policy")
    db.add(exp)
    db.flush()
    arms = []
    for name in ("treatment", "control"):
        arm = ExperimentArm(experiment_id=exp.id, name=name, arm="collective")
        db.add(arm)
        arms.append(arm)
    db.flush()
    return exp, arms


def _run(db, experiment, arm, *, seed, metrics, config_hash="cfg-1", gates_pass=True,
         status="completed") -> ExperimentRun:
    run = ExperimentRun(
        experiment_id=experiment.id, experiment_arm_id=arm.id, seed=seed,
        status=status, config_hash=config_hash, metrics=metrics,
        gates={"frozen_model": {"passed": gates_pass, "detail": ""}},
    )
    db.add(run)
    db.flush()
    return run


# --------------------------------------------------------------------------
# the gate (Part A §A2.2)
# --------------------------------------------------------------------------
def test_a_proposed_version_is_invisible_to_the_runtime(db, workspace):
    """"Impossible by construction": there is no argument a caller can make to `load_active`
    that returns an unapproved version."""
    gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                        name="hybrid", body={"weights": {"vector": 9.0}})
    db.commit()

    assert gate.load_active(db, workspace_id=workspace.id, kind="retrieval",
                            name="hybrid") is None
    assert db.query(Policy).count() == 1, "the proposal exists; it is simply inert"


def test_the_runtime_falls_back_to_its_built_in_default(db, workspace):
    """Falling back to the most recent *proposed* version would let an unapproved proposal take
    effect merely by existing."""
    gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                        name="hybrid", body={"weights": {"vector": 9.0}})
    db.commit()

    body = gate.load_active_body(db, workspace_id=workspace.id, kind="retrieval",
                                 name="hybrid", default={"weights": {"vector": 0.7}})
    assert body == {"weights": {"vector": 0.7}}


def test_approval_activates_a_version_on_a_matched_comparison(db, workspace, experiment):
    exp, arms = experiment
    treatment = _run(db, exp, arms[0], seed=0, metrics={"newcomer_advantage": 0.4})
    control = _run(db, exp, arms[1], seed=1, metrics={"newcomer_advantage": 0.3})
    policy = gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                                 name="hybrid", body={"weights": {"vector": 0.9}})
    db.commit()

    gate.approve(db, policy, treatment_run=treatment, control_run=control,
                 metric="newcomer_advantage")
    db.commit()

    active = gate.load_active(db, workspace_id=workspace.id, kind="retrieval", name="hybrid")
    assert active is not None and active.id == policy.id
    assert policy.approval_evidence["treatment"] == 0.4
    assert policy.approval_evidence["control"] == 0.3


def test_an_unmatched_configuration_cannot_approve(db, workspace, experiment):
    """Different configuration hashes mean the arms differ in more than the version under test."""
    exp, arms = experiment
    treatment = _run(db, exp, arms[0], seed=0, metrics={"m": 0.9}, config_hash="cfg-A")
    control = _run(db, exp, arms[1], seed=1, metrics={"m": 0.1}, config_hash="cfg-B")
    policy = gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                                 name="hybrid", body={})
    db.commit()

    with pytest.raises(gate.UnmatchedComparison, match="configuration hashes differ"):
        gate.approve(db, policy, treatment_run=treatment, control_run=control, metric="m")
    assert gate.load_active(db, workspace_id=workspace.id, kind="retrieval",
                            name="hybrid") is None


def test_a_run_that_failed_its_gates_cannot_approve_anything(db, workspace, experiment):
    """A run whose prerequisites failed is not read at all (ARCHITECTURE §3.1), so it cannot
    support an approval either."""
    exp, arms = experiment
    treatment = _run(db, exp, arms[0], seed=0, metrics={"m": 0.9}, gates_pass=False)
    control = _run(db, exp, arms[1], seed=1, metrics={"m": 0.1})
    policy = gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                                 name="hybrid", body={})
    db.commit()

    with pytest.raises(gate.UnmatchedComparison, match="failed its gates"):
        gate.approve(db, policy, treatment_run=treatment, control_run=control, metric="m")


def test_a_version_that_did_not_improve_cannot_approve(db, workspace, experiment):
    exp, arms = experiment
    treatment = _run(db, exp, arms[0], seed=0, metrics={"m": 0.2})
    control = _run(db, exp, arms[1], seed=1, metrics={"m": 0.5})
    policy = gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                                 name="hybrid", body={})
    db.commit()

    with pytest.raises(gate.UnmatchedComparison, match="did not improve"):
        gate.approve(db, policy, treatment_run=treatment, control_run=control, metric="m")


def test_a_missing_metric_is_not_a_comparison(db, workspace, experiment):
    exp, arms = experiment
    treatment = _run(db, exp, arms[0], seed=0, metrics={"m": 0.9})
    control = _run(db, exp, arms[1], seed=1, metrics={})
    policy = gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                                 name="hybrid", body={})
    db.commit()

    with pytest.raises(gate.UnmatchedComparison, match="missing"):
        gate.approve(db, policy, treatment_run=treatment, control_run=control, metric="m")


def test_an_incomplete_run_cannot_approve(db, workspace, experiment):
    exp, arms = experiment
    treatment = _run(db, exp, arms[0], seed=0, metrics={"m": 0.9}, status="running")
    control = _run(db, exp, arms[1], seed=1, metrics={"m": 0.1})
    policy = gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                                 name="hybrid", body={})
    db.commit()

    with pytest.raises(gate.UnmatchedComparison, match="has not completed"):
        gate.approve(db, policy, treatment_run=treatment, control_run=control, metric="m")


def test_approving_a_new_version_supersedes_the_old_one(db, workspace, experiment):
    exp, arms = experiment
    first = gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                                name="hybrid", body={"v": 1})
    gate.approve(db, first,
                 treatment_run=_run(db, exp, arms[0], seed=0, metrics={"m": 0.4}),
                 control_run=_run(db, exp, arms[1], seed=1, metrics={"m": 0.3}), metric="m")
    db.commit()

    second = gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                                 name="hybrid", body={"v": 2})
    gate.approve(db, second,
                 treatment_run=_run(db, exp, arms[0], seed=2, metrics={"m": 0.6}),
                 control_run=_run(db, exp, arms[1], seed=3, metrics={"m": 0.5}), metric="m")
    db.commit()

    assert second.version == 2
    assert first.status == "superseded"
    active = gate.load_active(db, workspace_id=workspace.id, kind="retrieval", name="hybrid")
    assert active.id == second.id


def test_rollback_restores_the_previous_approved_version(db, workspace, experiment):
    """§60: every change must remain measurable *and reversible*."""
    exp, arms = experiment
    first = gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                                name="hybrid", body={"v": 1})
    gate.approve(db, first,
                 treatment_run=_run(db, exp, arms[0], seed=0, metrics={"m": 0.4}),
                 control_run=_run(db, exp, arms[1], seed=1, metrics={"m": 0.3}), metric="m")
    second = gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                                 name="hybrid", body={"v": 2})
    gate.approve(db, second,
                 treatment_run=_run(db, exp, arms[0], seed=2, metrics={"m": 0.6}),
                 control_run=_run(db, exp, arms[1], seed=3, metrics={"m": 0.5}), metric="m")
    db.commit()

    restored = gate.roll_back(db, second, reason="regressed in production")
    db.commit()

    assert restored is not None and restored.id == first.id
    active = gate.load_active(db, workspace_id=workspace.id, kind="retrieval", name="hybrid")
    assert active.id == first.id
    assert second.rolled_back_at is not None


def test_rolling_back_the_only_version_falls_back_to_the_default(db, workspace, experiment):
    """The built-in default is what the system shipped with and needs no approval, so this is
    the correct outcome rather than an error."""
    exp, arms = experiment
    only = gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                               name="hybrid", body={"v": 1})
    gate.approve(db, only,
                 treatment_run=_run(db, exp, arms[0], seed=0, metrics={"m": 0.4}),
                 control_run=_run(db, exp, arms[1], seed=1, metrics={"m": 0.3}), metric="m")
    db.commit()

    assert gate.roll_back(db, only, reason="bad") is None
    db.commit()
    body = gate.load_active_body(db, workspace_id=workspace.id, kind="retrieval",
                                 name="hybrid", default={"builtin": True})
    assert body == {"builtin": True}


def test_the_gate_governs_every_kind_part_a_names(db, workspace):
    """§A2.2 names prompts, retrieval, scheduler, consolidation, role mixes, procedures and
    policies."""
    for kind in ("retrieval", "scheduler", "consolidation", "role_mix", "verification",
                 "tool_use"):
        assert kind in gate.GOVERNED_POLICY_KINDS
    with pytest.raises(ValueError, match="not a governed policy kind"):
        gate.propose_policy(db, workspace_id=workspace.id, kind="something_else",
                            name="x", body={})


def test_gate_metrics_report_adoption_and_rollback(db, workspace, experiment):
    exp, arms = experiment
    approved = gate.propose_policy(db, workspace_id=workspace.id, kind="retrieval",
                                   name="a", body={})
    gate.propose_policy(db, workspace_id=workspace.id, kind="scheduler", name="b", body={})
    gate.approve(db, approved,
                 treatment_run=_run(db, exp, arms[0], seed=0, metrics={"m": 0.7}),
                 control_run=_run(db, exp, arms[1], seed=1, metrics={"m": 0.4}), metric="m")
    db.commit()

    metrics = gate.gate_metrics(db, workspace.id)["policy"]
    assert metrics["proposed"] == 2 and metrics["approved"] == 1
    assert metrics["adoption_rate"] == pytest.approx(0.5)
    assert metrics["mean_attributed_improvement"] == pytest.approx(0.3)


def test_adoption_rate_over_an_empty_set_is_not_zero(db, workspace):
    """ARCHITECTURE §3.9: an adoption rate over nothing proposed is not zero adoption."""
    assert gate.gate_metrics(db, workspace.id)["policy"]["adoption_rate"] is None


# --------------------------------------------------------------------------
# reputation (§29)
# --------------------------------------------------------------------------
@pytest.fixture
def profile(db, workspace):
    p = AgentProfile(workspace_id=workspace.id, name="explorer")
    db.add(p)
    db.commit()
    return p


def _episode(db, workspace, profile, *, succeeded=True, tokens=1000) -> Episode:
    episode = Episode(
        workspace_id=workspace.id, agent_profile_id=profile.id,
        model_provider="d", model_name="d", tokens_used=tokens,
        termination_reason=(
            TerminationReason.EVALUATOR_SUCCESS if succeeded
            else TerminationReason.EVALUATOR_FAILURE
        ),
    )
    db.add(episode)
    db.flush()
    db.add(Evaluation(workspace_id=workspace.id, episode_id=episode.id, scope="episode",
                      evaluator_kind="exact_match", succeeded=succeeded,
                      score=1.0 if succeeded else 0.0))
    db.flush()
    return episode


def test_reputation_has_every_dimension_the_specification_names():
    assert len(reputation.DIMENSIONS) == 11
    for required in ("predictive_accuracy", "calibration", "evidence_quality",
                     "reproducibility", "downstream_usefulness", "false_positive_rate",
                     "tool_creation", "tool_reliability", "efficiency", "correction_rate",
                     "contradiction_resolution"):
        assert required in reputation.DIMENSIONS


def test_trust_is_not_collapsed_into_one_score(db, workspace, profile):
    """§29: do not collapse trust into one universal score."""
    for succeeded in (True, True, False):
        _episode(db, workspace, profile, succeeded=succeeded)
    db.commit()

    result = reputation.compute_profile(
        db, subject_kind="agent_profile", subject_id=profile.id, workspace_id=workspace.id
    )
    assert len(result.dimensions) == 11
    assert result.dimensions["predictive_accuracy"].value == pytest.approx(2 / 3)
    # A single aggregate exists for display, and nothing in the system ranks on it.
    assert result.confident_mean() is None, "three observations is not confident"


def test_a_dimension_with_no_observations_reports_absence_not_zero(db, workspace, profile):
    result = reputation.compute_profile(
        db, subject_kind="agent_profile", subject_id=profile.id, workspace_id=workspace.id
    )
    tools = result.dimensions["tool_creation"]
    assert tools.value is None
    assert tools.unavailable_reason


def test_reputation_is_appended_as_immutable_observations(db, workspace, profile):
    """§29, and the §A2.1 discipline: a reputation must be reconstructable from what produced
    it."""
    from civitas.persistence.session import ImmutableViolation

    for _ in range(6):
        _episode(db, workspace, profile, succeeded=True)
    db.commit()
    reputation.record(db, workspace_id=workspace.id, subject_kind="agent_profile",
                      subject_id=profile.id)
    db.commit()

    rows = db.query(ReputationMetric).all()
    assert rows
    # A value distinct from whatever was computed, so SQLAlchemy actually sees a change. Setting
    # it to a value it already holds is not a modification and would make this test pass for the
    # wrong reason.
    rows[0].value = rows[0].value + 0.5
    with pytest.raises(ImmutableViolation):
        db.commit()
    db.rollback()


def test_reputation_cannot_outweigh_evidence_in_retrieval(db, workspace, profile):
    """§29: trust may influence retrieval, and must not make dissent impossible. Those are only
    compatible if the influence cannot outweigh the evidence terms."""
    from civitas.knowledge.retrieval import DEFAULT_WEIGHTS

    for _ in range(10):
        _episode(db, workspace, profile, succeeded=True)
    for i in range(10):
        db.add(Artifact(workspace_id=workspace.id, type=ArtifactType.EVIDENCE,
                        title=f"a{i}", body="b", confidence=0.9,
                        evidence_kind=EvidenceKind.REPRODUCED_RESULT,
                        validation_state=ValidationState.EVALUATOR_CONFIRMED))
    db.commit()

    result = reputation.compute_profile(
        db, subject_kind="agent_profile", subject_id=profile.id, workspace_id=workspace.id
    )
    influence = reputation.retrieval_influence(result)
    assert abs(influence) <= reputation.MAX_RETRIEVAL_INFLUENCE
    assert abs(influence) < DEFAULT_WEIGHTS["provenance"], (
        "a perfect reputation outweighs provenance; dissent could be buried"
    )
    assert abs(influence) < DEFAULT_WEIGHTS["validation"]


def test_an_unmeasured_subject_is_neutral_not_penalised(db, workspace, profile):
    """A subject nobody has reason to distrust must not rank below one with no record."""
    result = reputation.compute_profile(
        db, subject_kind="agent_profile", subject_id=profile.id, workspace_id=workspace.id
    )
    assert reputation.retrieval_influence(result) == 0.0


def test_a_thin_record_does_not_influence_ranking(db, workspace, profile):
    """Reputation earned on two episodes is an accident."""
    for _ in range(2):
        _episode(db, workspace, profile, succeeded=True)
    db.commit()
    result = reputation.compute_profile(
        db, subject_kind="agent_profile", subject_id=profile.id, workspace_id=workspace.id
    )
    assert not result.dimensions["predictive_accuracy"].confident
    assert reputation.retrieval_influence(result) == 0.0


# --------------------------------------------------------------------------
# procedures (§30)
# --------------------------------------------------------------------------
def _approved_procedure(db, workspace, experiment, **kw) -> Procedure:
    exp, arms = experiment
    procedure = gate.propose_procedure(db, workspace_id=workspace.id, **kw)
    gate.approve(db, procedure,
                 treatment_run=_run(db, exp, arms[0], seed=uuid.uuid4().int % 10_000,
                                    metrics={"m": 0.7}),
                 control_run=_run(db, exp, arms[1], seed=uuid.uuid4().int % 10_000,
                                  metrics={"m": 0.4}), metric="m")
    db.flush()
    return procedure


def test_an_unapproved_procedure_governs_nothing(db, workspace):
    """Otherwise any episode could institute a rule for the whole collective by writing it."""
    gate.propose_procedure(
        db, workspace_id=workspace.id, name="reproduce first",
        statement="Always reproduce this type of result before accepting it.",
        trigger={"kind": "artifact_type", "values": ["conclusion"]},
        requirement={"kind": "requires_replication", "value": 1},
    )
    db.commit()
    assert procedures.active_procedures(db, workspace.id) == []


def test_an_approved_procedure_is_applied_to_an_artifact(db, workspace, experiment):
    """§30's first example, as a rule the runtime evaluates rather than prompt advice."""
    _approved_procedure(
        db, workspace, experiment, name="evidence floor",
        statement="Conclusions must rest on at least a tool output.",
        trigger={"kind": "artifact_type", "values": ["conclusion"]},
        requirement={"kind": "min_evidence_kind", "value": "tool_output"},
    )
    weak = Artifact(workspace_id=workspace.id, type=ArtifactType.CONCLUSION,
                    title="asserted", body="b", evidence_kind=EvidenceKind.MODEL_ASSERTION)
    strong = Artifact(workspace_id=workspace.id, type=ArtifactType.CONCLUSION,
                      title="measured", body="b", evidence_kind=EvidenceKind.REPRODUCED_RESULT)
    db.add_all([weak, strong])
    db.commit()

    assert not procedures.check_artifact(
        db, workspace_id=workspace.id, artifact=weak
    ).compliant
    assert procedures.check_artifact(
        db, workspace_id=workspace.id, artifact=strong
    ).compliant


def test_a_procedure_that_does_not_fire_is_not_a_violation(db, workspace, experiment):
    _approved_procedure(
        db, workspace, experiment, name="conclusions only",
        statement="Applies to conclusions.",
        trigger={"kind": "artifact_type", "values": ["conclusion"]},
        requirement={"kind": "min_evidence_kind", "value": "reproduced_result"},
    )
    observation = Artifact(workspace_id=workspace.id, type=ArtifactType.OBSERVATION,
                           title="an observation", body="b")
    db.add(observation)
    db.commit()

    report = procedures.check_artifact(db, workspace_id=workspace.id, artifact=observation)
    assert report.compliant
    assert report.checks[0].fired is False


def test_a_trigger_the_engine_cannot_evaluate_is_surfaced_not_ignored(db, workspace, experiment):
    """A rule that never fires while appearing active is worse than no rule, because the
    collective believes it is protected."""
    _approved_procedure(
        db, workspace, experiment, name="mystery",
        statement="Uses a trigger the engine does not know.",
        trigger={"kind": "phase_of_the_moon"},
        requirement={"kind": "requires_replication", "value": 1},
    )
    artifact = Artifact(workspace_id=workspace.id, type=ArtifactType.RESULT, title="r", body="b")
    db.add(artifact)
    db.commit()

    report = procedures.check_artifact(db, workspace_id=workspace.id, artifact=artifact)
    assert not report.compliant
    assert "not one the engine can evaluate" in report.violations[0].detail


def test_the_unreliable_tool_rule_is_enforced_at_call_time(db, workspace, experiment):
    """§30's second example: 'Tool X is unreliable for input class Y.'"""
    _approved_procedure(
        db, workspace, experiment, name="probe unreliable for gyrix",
        statement="probe_device is unreliable for input class gyrix.",
        trigger={"kind": "always"},
        requirement={"kind": "forbid_tool_for", "tool": "probe_device",
                     "argument": "input_class", "values": ["gyrix"]},
    )
    db.commit()

    blocked = procedures.check_tool_call(
        db, workspace_id=workspace.id, tool_name="probe_device",
        args={"input_class": "gyrix", "operation": "fold"},
    )
    assert not blocked.compliant

    allowed = procedures.check_tool_call(
        db, workspace_id=workspace.id, tool_name="probe_device",
        args={"input_class": "aurex", "operation": "fold"},
    )
    assert allowed.compliant


def test_procedure_firings_are_counted(db, workspace, experiment):
    """§30 asks whether an institution improves outcomes, which needs a count of real firings."""
    procedure = _approved_procedure(
        db, workspace, experiment, name="floor", statement="s",
        trigger={"kind": "always"},
        requirement={"kind": "min_evidence_kind", "value": "tool_output"},
    )
    artifact = Artifact(workspace_id=workspace.id, type=ArtifactType.RESULT, title="r", body="b",
                        evidence_kind=EvidenceKind.TOOL_OUTPUT)
    db.add(artifact)
    db.commit()

    for _ in range(3):
        procedures.check_artifact(db, workspace_id=workspace.id, artifact=artifact)
    db.commit()
    db.refresh(procedure)
    assert procedure.times_applied == 3


def test_institution_effect_comes_from_the_matched_approval_not_a_before_after(
    db, workspace, experiment
):
    """A before/after on the live workspace would credit the procedure for the collective simply
    maturing."""
    procedure = _approved_procedure(
        db, workspace, experiment, name="floor", statement="s",
        trigger={"kind": "always"},
        requirement={"kind": "min_evidence_kind", "value": "tool_output"},
    )
    db.commit()

    effect = procedures.institution_effect(db, workspace_id=workspace.id, procedure=procedure)
    assert effect["measured"] is True
    assert effect["improvement"] == pytest.approx(0.3)
    assert effect["config_hash"]


def test_an_ungated_procedure_reports_that_it_was_never_measured(db, workspace):
    procedure = gate.propose_procedure(
        db, workspace_id=workspace.id, name="ungated", statement="s",
        trigger={"kind": "always"}, requirement={"kind": "requires_replication", "value": 1},
    )
    db.commit()
    effect = procedures.institution_effect(db, workspace_id=workspace.id, procedure=procedure)
    assert effect["measured"] is False
    assert "never gated" in effect["reason"]


# --------------------------------------------------------------------------
# collective meta-learning under the gate (§31, §60, §A2.2)
# --------------------------------------------------------------------------
def test_the_meta_learning_loop_can_say_no(db, workspace):
    """A mechanism that adopts every proposal is not evaluating anything.

    Both proposals go through the identical path under one shared configuration hash; only the
    measured outcome differs.
    """
    from civitas.legacy.meta_learning import active_configuration, evaluate_proposal

    helped = evaluate_proposal(
        db, workspace_id=workspace.id, kind="scheduler", name="better",
        body={"v": 1}, metric="coverage",
        treatment=lambda: 0.8, control=lambda: 0.5, config_hash="one-configuration",
    )
    did_not_help = evaluate_proposal(
        db, workspace_id=workspace.id, kind="retrieval", name="no_better",
        body={"v": 1}, metric="coverage",
        treatment=lambda: 0.5, control=lambda: 0.5, config_hash="one-configuration",
    )
    db.commit()

    assert helped.approved
    assert not did_not_help.approved
    assert "did not improve" in did_not_help.reason

    active = active_configuration(db, workspace.id)
    assert active["scheduler"]["name"] == "better"
    assert active["retrieval"] is None, "a refused proposal must not take effect"


def test_a_refused_proposal_is_kept_as_evidence(db, workspace):
    """A documented change that did not help is what stops the next agent proposing it (§12)."""
    from civitas.legacy.meta_learning import evaluate_proposal

    outcome = evaluate_proposal(
        db, workspace_id=workspace.id, kind="retrieval", name="no_better",
        body={"v": 1}, metric="coverage",
        treatment=lambda: 0.4, control=lambda: 0.6, config_hash="one-configuration",
    )
    db.commit()

    assert not outcome.approved
    policy = db.get(Policy, outcome.policy_id)
    assert policy is not None and policy.status == "proposed"


def test_both_arms_of_a_matched_experiment_share_one_configuration_hash(db, workspace):
    """Two independently computed hashes that happen to agree prove nothing about whether the
    arms were actually matched."""
    from civitas.legacy.meta_learning import run_matched_experiment

    treatment, control = run_matched_experiment(
        db, workspace_id=workspace.id, name="x", metric="m",
        treatment=lambda: 1.0, control=lambda: 0.0, config_hash="shared",
    )
    db.commit()
    assert treatment.config_hash == control.config_hash == "shared"
    assert treatment.experiment_id == control.experiment_id
    assert treatment.experiment_arm_id != control.experiment_arm_id
