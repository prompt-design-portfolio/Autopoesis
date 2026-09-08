"""Collective orchestration (Part B §26, §27, §28)."""

from __future__ import annotations

import pytest

from civitas.domain.enums import (
    AllocationStrategy,
    ArtifactType,
    EvidenceKind,
    ExperimentArm,
    HypothesisState,
    RelationType,
    TaskStatus,
    TerminationReason,
    ValidationState,
)
from civitas.knowledge import hypothesis as hyp
from civitas.persistence.models import (
    AgentProfile,
    Artifact,
    ArtifactRelation,
    Episode,
    Evaluation,
    Task,
    TaskDependency,
)
from civitas.scheduler import allocation, roles, specialization


def _task(db, workspace, title, family="investigate", priority=0, **kw) -> Task:
    task = Task(workspace_id=workspace.id, title=title, description=title,
                task_family=family, priority=priority, status=TaskStatus.READY, **kw)
    db.add(task)
    db.flush()
    return task


def _profile(db, workspace, name) -> AgentProfile:
    profile = AgentProfile(workspace_id=workspace.id, name=name)
    db.add(profile)
    db.flush()
    return profile


def _episode(db, workspace, profile, task, *, succeeded=True, tokens=1000, tools=3) -> Episode:
    episode = Episode(
        workspace_id=workspace.id, agent_profile_id=profile.id, task_id=task.id,
        model_provider="deterministic", model_name="d",
        termination_reason=(
            TerminationReason.EVALUATOR_SUCCESS if succeeded
            else TerminationReason.EVALUATOR_FAILURE
        ),
        tokens_used=tokens, tool_calls_used=tools,
    )
    db.add(episode)
    db.flush()
    db.add(Evaluation(workspace_id=workspace.id, episode_id=episode.id, scope="episode",
                      evaluator_kind="exact_match", succeeded=succeeded,
                      score=1.0 if succeeded else 0.0))
    db.flush()
    return episode


# --------------------------------------------------------------------------
# allocation (§27)
# --------------------------------------------------------------------------
def test_every_strategy_in_the_specification_is_implemented(db, workspace):
    _profile(db, workspace, "explorer")
    db.commit()

    assert len(list(AllocationStrategy)) == 9, "Part B §27 lists nine allocation methods"
    for strategy in AllocationStrategy:
        # A fresh task per strategy: allocating moves a task to ASSIGNED and an assigned task is
        # correctly not re-allocatable, so reusing one would test the second strategy against an
        # empty queue.
        _task(db, workspace, f"task for {strategy.value}")
        db.commit()
        assignments = allocation.allocate(
            db, workspace_id=workspace.id, strategy=strategy, count=1
        )
        db.commit()
        assert assignments, f"{strategy.value} produced no allocation"
        assert assignments[0].strategy == strategy.value


def test_every_allocation_records_why_it_fired(db, workspace):
    """§27 requires allocation decisions to be logged — a policy cannot be evaluated against
    outcomes unless the reason it fired is recoverable."""
    _task(db, workspace, "t", priority=5)
    _profile(db, workspace, "explorer")
    db.commit()

    assignment = allocation.allocate(
        db, workspace_id=workspace.id, strategy=AllocationStrategy.INFORMATION_GAIN
    )[0]
    db.commit()

    rationale = assignment.rationale
    assert rationale["features"], "no ranking features were recorded"
    for required in ("uncertainty", "evidence_gap", "duplication_penalty", "profile_fit"):
        assert required in rationale["features"]


def test_a_task_blocked_on_an_incomplete_dependency_is_not_allocated(db, workspace):
    """Spending an episode on work that cannot succeed looks like agent failure in every
    downstream metric."""
    blocker = _task(db, workspace, "must finish first")
    blocked = _task(db, workspace, "depends on the other")
    db.add(TaskDependency(task_id=blocked.id, depends_on_task_id=blocker.id))
    _profile(db, workspace, "explorer")
    db.commit()

    ready = allocation.ready_tasks(db, workspace.id)
    db.commit()
    assert blocker.id in {t.id for t in ready}
    assert blocked.id not in {t.id for t in ready}
    assert blocked.status is TaskStatus.BLOCKED

    blocker.status = TaskStatus.COMPLETED
    db.commit()
    ready = allocation.ready_tasks(db, workspace.id)
    db.commit()
    assert blocked.id in {t.id for t in ready}


def test_only_unproductive_repetition_is_penalised(db, workspace):
    """§27 says to avoid duplicated work, and a blind reading of that penalises the mechanism
    that makes a collective work.

    Measured on one task: attempts 1 and 2 failed but recorded which operations were ruled out,
    and attempt 3 succeeded on that record. A repeat that consumes what the last repeat left
    behind is *continuation*, not duplication. Before this correction the reasoned strategies lost
    to random allocation at every episode budget above 9.
    """
    assert allocation.duplication_penalty(0) == pytest.approx(0.0)
    assert allocation.duplication_penalty(1) < 0
    assert allocation.duplication_penalty(5) < allocation.duplication_penalty(1)

    # Three attempts that all left something behind attract no penalty at all.
    assert allocation.duplication_penalty(3, productive_attempts=3) == pytest.approx(0.0)
    # Three attempts of which one was productive are penalised as two wasted ones.
    assert allocation.duplication_penalty(3, productive_attempts=1) == pytest.approx(
        allocation.duplication_penalty(2)
    )


def test_a_task_whose_attempts_left_knowledge_is_not_treated_as_duplicated(db, workspace):
    """The allocation-level consequence of the correction above."""
    productive = _task(db, workspace, "attempted, and it recorded findings")
    wasteful = _task(db, workspace, "attempted, and it left nothing")
    profile = _profile(db, workspace, "explorer")

    for _ in range(2):
        episode = _episode(db, workspace, profile, productive, succeeded=False)
        episode.artifacts_created = 2
        episode.duplicate_failures = 1
        _episode(db, workspace, profile, wasteful, succeeded=False)
    db.commit()

    ranked = allocation.score_candidates(
        db, workspace_id=workspace.id, tasks=[productive, wasteful], profiles=[profile],
        strategy=AllocationStrategy.PRIORITY,
    )
    assert ranked[0].task.id == productive.id, (
        "the scheduler preferred the task whose attempts left nothing behind"
    )


def test_deliberate_sampling_ignores_the_duplication_penalty(db, workspace):
    """§27 exempts intentional sampling — penalising repetition would make it not random."""
    fresh = _task(db, workspace, "untouched")
    tried = _task(db, workspace, "attempted twice")
    profile = _profile(db, workspace, "explorer")
    for _ in range(2):
        _episode(db, workspace, profile, tried, succeeded=False)
    db.commit()

    sampled = allocation.score_candidates(
        db, workspace_id=workspace.id, tasks=[fresh, tried], profiles=[profile],
        strategy=AllocationStrategy.RANDOM,
    )
    assert all("random" in c.features for c in sampled), (
        "the sampling strategy must not apply the duplication penalty"
    )
    assert {c.task.id for c in sampled} == {fresh.id, tried.id}


def test_specialization_allocation_prefers_the_measured_profile(db, workspace):
    task = _task(db, workspace, "a debugging task", family="debug")
    good = _profile(db, workspace, "debugger")
    bad = _profile(db, workspace, "mathematician")
    good.performance_by_work_type = {"debug": {"n": 8, "success_rate": 0.9, "confident": True}}
    bad.performance_by_work_type = {"debug": {"n": 8, "success_rate": 0.1, "confident": True}}
    db.commit()

    ranked = allocation.score_candidates(
        db, workspace_id=workspace.id, tasks=[task], profiles=[good, bad],
        strategy=AllocationStrategy.SPECIALIZATION,
    )
    assert ranked[0].profile.id == good.id


def test_diversity_allocation_prefers_a_profile_that_has_not_worked_the_family(db, workspace):
    """§26: the collective must be able to discover which profiles suit which work, rather than
    entrenching whichever was tried first."""
    task = _task(db, workspace, "a debugging task", family="debug")
    experienced = _profile(db, workspace, "debugger")
    fresh = _profile(db, workspace, "explorer")
    experienced.performance_by_work_type = {
        "debug": {"n": 20, "success_rate": 0.9, "confident": True}
    }
    db.commit()

    ranked = allocation.score_candidates(
        db, workspace_id=workspace.id, tasks=[task], profiles=[experienced, fresh],
        strategy=AllocationStrategy.DIVERSITY,
    )
    assert ranked[0].profile.id == fresh.id


def test_adversarial_allocation_goes_where_the_record_disagrees_with_itself(db, workspace):
    quiet = _task(db, workspace, "no disagreement")
    disputed = _task(db, workspace, "contested")
    for _ in range(3):
        db.add(Artifact(workspace_id=workspace.id, task_id=disputed.id,
                        type=ArtifactType.CONTRADICTION, title="c", body="c"))
    profile = _profile(db, workspace, "critic")
    db.commit()

    ranked = allocation.score_candidates(
        db, workspace_id=workspace.id, tasks=[quiet, disputed], profiles=[profile],
        strategy=AllocationStrategy.ADVERSARIAL,
    )
    assert ranked[0].task.id == disputed.id


def test_information_gain_prefers_an_unresolved_task_with_an_evidence_gap(db, workspace):
    settled = _task(db, workspace, "already solved")
    open_question = _task(db, workspace, "hypotheses with no evidence")
    for i in range(3):
        db.add(Artifact(workspace_id=workspace.id, task_id=open_question.id,
                        type=ArtifactType.HYPOTHESIS, title=f"h{i}", body="h"))
    profile = _profile(db, workspace, "explorer")
    _episode(db, workspace, profile, settled, succeeded=True)
    _episode(db, workspace, profile, open_question, succeeded=False)
    db.commit()

    ranked = allocation.score_candidates(
        db, workspace_id=workspace.id, tasks=[settled, open_question], profiles=[profile],
        strategy=AllocationStrategy.INFORMATION_GAIN,
    )
    assert ranked[0].task.id == open_question.id


# --------------------------------------------------------------------------
# specialization (§26)
# --------------------------------------------------------------------------
def test_profile_performance_is_measured_not_declared(db, workspace):
    profile = _profile(db, workspace, "debugger")
    task = _task(db, workspace, "a debug task", family="debug")
    profile.performance_by_work_type = {"debug": {"n": 999, "success_rate": 1.0}}
    db.commit()

    for succeeded in (True, False, False):
        _episode(db, workspace, profile, task, succeeded=succeeded)
    db.commit()

    specialization.update_profiles(db, workspace.id)
    db.commit()
    entry = profile.performance_by_work_type["debug"]
    assert entry["n"] == 3, "the declared value was not replaced by the measured one"
    assert entry["success_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert entry["confident"] is False, "three episodes is not a confident measurement"


def test_a_thin_record_is_marked_unconfident_and_does_not_drive_allocation(db, workspace):
    profile = _profile(db, workspace, "debugger")
    task = _task(db, workspace, "t", family="debug")
    _episode(db, workspace, profile, task, succeeded=True)
    db.commit()
    specialization.update_profiles(db, workspace.id)
    db.commit()

    assert profile.performance_by_work_type["debug"]["confident"] is False
    assert specialization.best_profile_for(db, workspace.id, "debug") is None, (
        "a one-episode record must not select a specialist"
    )


def test_infrastructure_failures_do_not_count_against_a_profile(db, workspace):
    """A flaky provider must not look like a weak specialist."""
    profile = _profile(db, workspace, "explorer")
    task = _task(db, workspace, "t", family="investigate")
    _episode(db, workspace, profile, task, succeeded=True)
    broken = Episode(workspace_id=workspace.id, agent_profile_id=profile.id, task_id=task.id,
                     model_provider="d", model_name="d",
                     termination_reason=TerminationReason.PROVIDER_FAILURE)
    db.add(broken)
    db.flush()
    db.add(Evaluation(workspace_id=workspace.id, episode_id=broken.id, scope="episode",
                      evaluator_kind="exact_match", succeeded=False, score=0.0))
    db.commit()

    entries = specialization.measure(db, workspace.id)
    assert len(entries) == 1
    assert entries[0].n == 1 and entries[0].successes == 1


def test_the_specialization_index_reports_absence_rather_than_zero(db, workspace):
    """One profile cannot specialise, and reporting 0.0 would present 'nobody to divide labour
    with' as 'labour is not divided' (ARCHITECTURE §3.9)."""
    profile = _profile(db, workspace, "only")
    task = _task(db, workspace, "t", family="investigate")
    _episode(db, workspace, profile, task, succeeded=True)
    db.commit()

    report = specialization.update_profiles(db, workspace.id)
    db.commit()
    assert report.specialization_index is None
    assert "two profiles" in report.index_unavailable_reason


def test_the_specialization_index_rises_with_division_of_labour(db, workspace):
    a = _profile(db, workspace, "alpha")
    b = _profile(db, workspace, "beta")
    debug = _task(db, workspace, "debug task", family="debug")
    math = _task(db, workspace, "math task", family="math")

    # Each profile succeeds only at its own family — maximal division of labour.
    for _ in range(4):
        _episode(db, workspace, a, debug, succeeded=True)
        _episode(db, workspace, b, debug, succeeded=False)
        _episode(db, workspace, b, math, succeeded=True)
        _episode(db, workspace, a, math, succeeded=False)
    db.commit()

    report = specialization.update_profiles(db, workspace.id)
    db.commit()
    assert report.specialization_index is not None
    assert report.specialization_index > 0.9, (
        f"perfect division of labour scored {report.specialization_index}"
    )


# --------------------------------------------------------------------------
# roles (§28)
# --------------------------------------------------------------------------
def test_the_six_roles_of_section_28_exist(db, workspace):
    profiles = roles.ensure_roles(db, workspace.id)
    db.commit()
    names = {p.name for p in profiles}
    for required in ("critic", "verifier", "replicator", "adversary",
                     "alternative_hypothesis", "synthesizer"):
        assert required in names, f"Part B §28 requires a {required} role"


def test_seeding_roles_does_not_erase_measured_performance(db, workspace):
    roles.ensure_roles(db, workspace.id)
    db.commit()
    critic = db.query(AgentProfile).filter(AgentProfile.name == "critic").one()
    critic.performance_by_work_type = {"debug": {"n": 10, "success_rate": 0.7}}
    db.commit()

    roles.ensure_roles(db, workspace.id)
    db.commit()
    db.refresh(critic)
    assert critic.performance_by_work_type["debug"]["n"] == 10


def test_a_replicator_is_not_shown_the_conclusion_it_must_reproduce(db, workspace):
    """§28's sharpest requirement: an agent shown the answer it is meant to independently confirm
    is agreeing, not replicating."""
    from civitas.knowledge.embeddings import index_missing
    from civitas.knowledge.retrieval import retrieve

    conclusion = Artifact(workspace_id=workspace.id, type=ArtifactType.CONCLUSION,
                          title="the writer holds a lock across a retry", body="conclusion")
    observation = Artifact(workspace_id=workspace.id, type=ArtifactType.OBSERVATION,
                           title="the writer holds a lock across a retry", body="raw observation")
    db.add_all([conclusion, observation])
    db.commit()
    index_missing(db, workspace.id)
    db.commit()

    profiles = roles.ensure_roles(db, workspace.id)
    db.commit()
    replicator = next(p for p in profiles if p.name == "replicator")
    options = roles.retrieval_options_for(replicator)
    assert "conclusion" in options["exclude_types"]

    result = retrieve(
        db, workspace_id=workspace.id, query="writer lock retry",
        arm=ExperimentArm.COLLECTIVE, limit=10,
        exclude_types=[ArtifactType(t) for t in options["exclude_types"]],
    )
    returned = {a.id for a in result.artifacts}
    assert observation.id in returned, "the replicator must still see the raw observation"
    assert conclusion.id not in returned, "the replicator was shown the conclusion"
    assert any(e["reason"] == "role_excluded_type" for e in result.suppressed), (
        "the exclusion must be on the record, not merely applied"
    )


def test_a_critic_is_pointed_at_weak_claims_not_strong_ones(db, workspace):
    profiles = roles.ensure_roles(db, workspace.id)
    db.commit()
    critic = next(p for p in profiles if p.name == "critic")
    weights = roles.retrieval_options_for(critic)["weights"]
    assert weights["validation"] < 0 and weights["provenance"] < 0, (
        "a critic ranked toward well-supported artifacts is pointed away from what is wrong"
    )


def test_claims_needing_challenge_and_replication_are_identified(db, workspace):
    supported = Artifact(
        workspace_id=workspace.id, type=ArtifactType.HYPOTHESIS,
        title="unchallenged claim", body="h", hypothesis_state=HypothesisState.SUPPORTED,
    )
    db.add(supported)
    db.flush()
    challenged = Artifact(
        workspace_id=workspace.id, type=ArtifactType.HYPOTHESIS,
        title="already argued with", body="h", hypothesis_state=HypothesisState.SUPPORTED,
    )
    objection = Artifact(workspace_id=workspace.id, type=ArtifactType.CONTRADICTION,
                         title="objection", body="o")
    db.add_all([challenged, objection])
    db.flush()
    db.add(ArtifactRelation(source_id=objection.id, target_id=challenged.id,
                            type=RelationType.CONTRADICTS))
    db.commit()

    needs = {a.id for a in roles.claims_needing_challenge(db, workspace.id)}
    assert supported.id in needs and challenged.id not in needs

    needs_replication = {a.id for a in roles.claims_needing_replication(db, workspace.id)}
    assert supported.id in needs_replication


def test_subproblems_are_tracked_separately_from_tasks(db, workspace):
    """§27: a subproblem is a question the scheduler tracks. Making it a task would commit an
    episode to it before anyone judged it worth one."""
    task = _task(db, workspace, "investigate corruption")
    db.commit()
    created = roles.decompose(db, task_id=task.id, subproblems=[
        {"title": "is it the writer?", "uncertainty": 0.9},
        {"title": "is it the cache?", "uncertainty": 0.4},
    ])
    db.commit()

    assert len(created) == 2
    assert db.query(Task).count() == 1, "decomposition must not create tasks"
    open_ = roles.open_subproblems(db, task.id)
    assert [s.title for s in open_][0] == "is it the writer?", "most uncertain first"

    roles.resolve_subproblem(db, open_[0])
    db.commit()
    assert len(roles.open_subproblems(db, task.id)) == 1


# --------------------------------------------------------------------------
# the hypothesis state machine (§28)
# --------------------------------------------------------------------------
def _hypothesis(db, workspace, title="a claim", episode_id=None) -> Artifact:
    artifact = Artifact(
        workspace_id=workspace.id, type=ArtifactType.HYPOTHESIS, title=title, body=title,
        hypothesis_state=HypothesisState.PROPOSED, creator_episode_id=episode_id,
    )
    db.add(artifact)
    db.flush()
    return artifact


def test_the_state_machine_covers_every_state_in_the_specification():
    assert set(hyp.TRANSITIONS) == set(HypothesisState)
    assert len(HypothesisState) == 8


def test_an_illegal_transition_is_refused(db, workspace):
    """Nothing can jump from proposed to accepted because an agent said so."""
    claim = _hypothesis(db, workspace)
    db.commit()
    with pytest.raises(hyp.InvalidTransition, match="not a permitted transition"):
        hyp.transition(db, claim, HypothesisState.ACCEPTED)


def test_accepted_is_not_terminal(db, workspace):
    """A record that could not withdraw an accepted claim could not self-correct (§11)."""
    assert HypothesisState.CHALLENGED in hyp.TRANSITIONS[HypothesisState.ACCEPTED]
    assert HypothesisState.SUPERSEDED in hyp.TRANSITIONS[HypothesisState.ACCEPTED]


def test_verification_requirements_scale_with_what_depends_on_a_claim(db, workspace):
    """§28: important claims should receive stronger verification requirements."""
    minor = _hypothesis(db, workspace, "nothing rests on this")
    major = _hypothesis(db, workspace, "much rests on this")
    for i in range(6):
        dependent = Artifact(workspace_id=workspace.id, type=ArtifactType.CONCLUSION,
                             title=f"c{i}", body="c")
        db.add(dependent)
        db.flush()
        db.add(ArtifactRelation(source_id=dependent.id, target_id=major.id,
                                type=RelationType.DEPENDS_ON))
    db.commit()

    minor_req = hyp.verification_requirement(db, minor)
    major_req = hyp.verification_requirement(db, major)
    assert major_req.min_supporting > minor_req.min_supporting
    assert major_req.min_independent_replications > minor_req.min_independent_replications


def test_a_self_reproduction_does_not_count_as_replication(db, workspace):
    """A claim confirmed only by the episode that produced it is confirmed by its author."""
    profile = _profile(db, workspace, "explorer")
    task = _task(db, workspace, "t")
    episode = _episode(db, workspace, profile, task)
    claim = _hypothesis(db, workspace, episode_id=episode.id)

    self_repro = Artifact(
        workspace_id=workspace.id, type=ArtifactType.EXPERIMENT_RESULT,
        title="I reproduced my own result", body="r",
        creator_episode_id=episode.id, evidence_kind=EvidenceKind.TOOL_OUTPUT,
    )
    db.add(self_repro)
    db.flush()
    db.add(ArtifactRelation(source_id=self_repro.id, target_id=claim.id,
                            type=RelationType.REPRODUCES))
    db.commit()

    assert hyp.assess(db, claim).independent_replications == 0

    other = _episode(db, workspace, profile, task)
    independent = Artifact(
        workspace_id=workspace.id, type=ArtifactType.EXPERIMENT_RESULT,
        title="independently reproduced", body="r",
        creator_episode_id=other.id, evidence_kind=EvidenceKind.REPRODUCED_RESULT,
    )
    db.add(independent)
    db.flush()
    db.add(ArtifactRelation(source_id=independent.id, target_id=claim.id,
                            type=RelationType.REPRODUCES))
    db.commit()
    assert hyp.assess(db, claim).independent_replications == 1


def test_a_claim_advances_through_the_ladder_rather_than_jumping(db, workspace):
    """A hypothesis that went proposed -> accepted in one write would be indistinguishable from
    one that had been tested, supported and replicated."""
    profile = _profile(db, workspace, "explorer")
    task = _task(db, workspace, "t")
    author = _episode(db, workspace, profile, task)
    claim = _hypothesis(db, workspace, episode_id=author.id)

    supporting = Artifact(workspace_id=workspace.id, type=ArtifactType.EVIDENCE,
                          title="supports it", body="e",
                          evidence_kind=EvidenceKind.DIRECT_OBSERVATION,
                          creator_episode_id=author.id)
    db.add(supporting)
    db.flush()
    db.add(ArtifactRelation(source_id=supporting.id, target_id=claim.id,
                            type=RelationType.SUPPORTS))
    db.commit()

    state, _ = hyp.advance(db, claim)
    db.commit()
    assert state is HypothesisState.SUPPORTED

    replicator = _episode(db, workspace, profile, task)
    repro = Artifact(workspace_id=workspace.id, type=ArtifactType.EXPERIMENT_RESULT,
                     title="reproduced", body="r",
                     evidence_kind=EvidenceKind.REPRODUCED_RESULT,
                     creator_episode_id=replicator.id)
    db.add(repro)
    db.flush()
    db.add(ArtifactRelation(source_id=repro.id, target_id=claim.id,
                            type=RelationType.REPRODUCES))
    db.commit()

    state, assessment = hyp.advance(db, claim)
    db.commit()
    assert state is HypothesisState.ACCEPTED
    assert assessment.independent_replications == 1
    assert claim.validation_state is ValidationState.REPRODUCED


def test_a_contradiction_moves_a_claim_to_challenged(db, workspace):
    claim = _hypothesis(db, workspace)
    supporting = Artifact(workspace_id=workspace.id, type=ArtifactType.EVIDENCE,
                          title="supports", body="e",
                          evidence_kind=EvidenceKind.DIRECT_OBSERVATION)
    objection = Artifact(workspace_id=workspace.id, type=ArtifactType.CONTRADICTION,
                         title="contradicts", body="c")
    db.add_all([supporting, objection])
    db.flush()
    db.add_all([
        ArtifactRelation(source_id=supporting.id, target_id=claim.id,
                         type=RelationType.SUPPORTS),
        ArtifactRelation(source_id=objection.id, target_id=claim.id,
                         type=RelationType.CONTRADICTS),
    ])
    db.commit()

    state, _ = hyp.advance(db, claim)
    db.commit()
    assert state is HypothesisState.CHALLENGED
    assert claim.validation_state is ValidationState.DISPUTED
