"""The nine experimental arms are machine-enforced (Part B §21).

Part A §A4.2 makes this a permanent regression: every milestone after M4 must leave these passing.
An arm that silently stopped being enforced would turn every later result into an unablated one
that still looks like evidence.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from civitas.domain.enums import (
    NEGATIVE_TYPES,
    ArtifactType,
    EvidenceKind,
    ExperimentArm,
    ValidationState,
)
from civitas.knowledge.arms import all_arms, arm_policy
from civitas.knowledge.retrieval import retrieve
from civitas.persistence.models import Artifact, RetrievalDecision
from civitas.persistence.types import utcnow


@pytest.fixture
def corpus(db, workspace):
    """A workspace with positive, negative, validated and unvalidated knowledge."""
    made = {}
    V, E = ValidationState, EvidenceKind
    specs = [
        ("hit", ArtifactType.EVIDENCE, "lock retry writer corruption",
         V.EVALUATOR_CONFIRMED, E.TOOL_OUTPUT),
        ("fail", ArtifactType.FAILURE, "lock retry writer approach failed",
         V.SELF_REPORTED, E.DIRECT_OBSERVATION),
        ("warn", ArtifactType.WARNING, "lock retry writer unreliable",
         V.SELF_REPORTED, E.MODEL_ASSERTION),
        ("open", ArtifactType.UNRESOLVED_ISSUE, "lock retry writer unresolved",
         V.UNVALIDATED, E.MODEL_ASSERTION),
        ("noise1", ArtifactType.OBSERVATION, "unrelated scheduler latency",
         V.PEER_REVIEWED, E.INFERENCE),
        ("noise2", ArtifactType.OBSERVATION, "unrelated cache eviction",
         V.PEER_REVIEWED, E.INFERENCE),
    ]
    for key, kind, title, validation, evidence in specs:
        a = Artifact(
            workspace_id=workspace.id, type=kind, title=title, body=f"{title} body",
            validation_state=validation, evidence_kind=evidence, confidence=0.8,
            environment_version="test-env-1",
        )
        db.add(a)
        made[key] = a
    db.commit()
    return made


def _titles(result):
    return [a.title for a in result.artifacts]


def test_every_arm_in_the_specification_has_a_policy():
    assert len(all_arms()) == 9, "Part B §21 requires nine arms"
    for arm in all_arms():
        as_of = utcnow() if arm is ExperimentArm.COLLECTIVE_FROZEN else None
        assert arm_policy(arm, as_of=as_of).arm is arm


def test_an_unknown_arm_raises_rather_than_defaulting():
    """Falling back to `collective` would turn a mis-specified experiment into an unablated one
    whose result still looks like evidence."""
    with pytest.raises(ValueError, match="machine-enforced"):
        arm_policy("collective_with_extra_sauce")


def test_frozen_without_a_cut_raises():
    with pytest.raises(ValueError, match="snapshot cut"):
        arm_policy(ExperimentArm.COLLECTIVE_FROZEN)


@pytest.mark.parametrize("arm", [ExperimentArm.SOLO, ExperimentArm.INDEPENDENT,
                                 ExperimentArm.MEMORY_RESET])
def test_blind_arms_return_nothing(db, workspace, corpus, arm):
    result = retrieve(db, workspace_id=workspace.id, query="lock retry writer", arm=arm, limit=10)
    assert result.artifacts == []
    # The decision is still logged: a blind arm that leaves no record is indistinguishable from a
    # retrieval that happened to find nothing.
    decision = db.query(RetrievalDecision).order_by(RetrievalDecision.created_at.desc()).first()
    assert decision.experiment_arm == arm.value
    assert decision.suppressed and decision.suppressed[0]["reason"] == "arm_blind"


def test_collective_returns_negative_knowledge(db, workspace, corpus):
    result = retrieve(db, workspace_id=workspace.id, query="lock retry writer",
                      arm=ExperimentArm.COLLECTIVE, limit=10)
    kinds = {a.type for a in result.artifacts}
    assert kinds & NEGATIVE_TYPES, "the collective arm must carry failures and warnings"


def test_no_negative_arm_removes_exactly_the_negative_types(db, workspace, corpus):
    result = retrieve(db, workspace_id=workspace.id, query="lock retry writer",
                      arm=ExperimentArm.COLLECTIVE_NO_NEGATIVE, limit=10)
    assert not ({a.type for a in result.artifacts} & NEGATIVE_TYPES)
    assert corpus["hit"].title in _titles(result), "only negative knowledge is removed"


def test_shared_memory_returns_only_curated_accepted_results(db, workspace, corpus):
    """The ordinary shared-memory/RAG condition (§21): accepted results only, no failures, no
    open questions, no unvalidated work."""
    result = retrieve(db, workspace_id=workspace.id, query="lock retry writer",
                      arm=ExperimentArm.SHARED_MEMORY, limit=10)
    for artifact in result.artifacts:
        assert artifact.validation_state not in (
            ValidationState.UNVALIDATED, ValidationState.SELF_REPORTED,
            ValidationState.DISPUTED, ValidationState.REFUTED,
        )
    assert corpus["hit"].title in _titles(result)
    assert corpus["fail"].title not in _titles(result)


def test_no_provenance_arm_hides_and_does_not_rank_on_provenance(db, workspace, corpus):
    """Ranking on provenance while hiding it would give the arm provenance's benefit without its
    visibility, and it would stop being a control."""
    result = retrieve(db, workspace_id=workspace.id, query="lock retry writer",
                      arm=ExperimentArm.COLLECTIVE_NO_PROVENANCE, limit=10)
    assert result.provenance_hidden
    for features in result.features:
        assert features["provenance"] == 0.0
        assert features["validation"] == 0.0


def test_scrambled_returns_a_comparable_quantity_with_relevance_destroyed(db, workspace, corpus):
    """§21: comparable quantities of artifacts, relevance and relationships scrambled.

    Quantity held is what makes it a control for *content* rather than for the store's existence —
    the `noise record` lesson of ARCHITECTURE §3.4.
    """
    plain = retrieve(db, workspace_id=workspace.id, query="lock retry writer",
                     arm=ExperimentArm.COLLECTIVE, limit=4)
    scrambled = retrieve(db, workspace_id=workspace.id, query="lock retry writer",
                         arm=ExperimentArm.COLLECTIVE_SCRAMBLED, limit=4, seed=7)
    assert len(scrambled.artifacts) == len(plain.artifacts), "quantity must be held"
    for features in scrambled.features:
        assert features["lexical"] == 0.0, "query relevance must be destroyed"


def test_frozen_arm_cannot_see_anything_created_after_its_cut(db, workspace, corpus):
    cut = utcnow()
    later = Artifact(
        workspace_id=workspace.id, type=ArtifactType.EVIDENCE,
        title="lock retry writer later discovery", body="after the cut",
        environment_version="test-env-1",
    )
    db.add(later)
    db.commit()
    later.created_at = cut + timedelta(seconds=60)
    db.commit()

    result = retrieve(db, workspace_id=workspace.id, query="lock retry writer",
                      arm=ExperimentArm.COLLECTIVE_FROZEN, as_of=cut, limit=10)
    assert later.title not in _titles(result)
    assert corpus["hit"].title in _titles(result)


def test_suppressions_are_recorded_not_merely_applied(db, workspace, corpus):
    """An ablation whose effect can only be inferred from an absence is not auditable (§14)."""
    retrieve(db, workspace_id=workspace.id, query="lock retry writer",
             arm=ExperimentArm.SHARED_MEMORY, limit=10)
    decision = db.query(RetrievalDecision).order_by(RetrievalDecision.created_at.desc()).first()
    assert decision.suppressed, "the arm removed candidates but recorded none"
    reasons = {entry["reason"] for entry in decision.suppressed}
    # `shared_memory` removes negative types in SQL and uncurated ones in Python. Both must be on
    # the record: which layer an exclusion happened at is an implementation detail, and an
    # auditor should not have to know it to see what the arm withheld.
    assert "arm_excluded_type" in reasons
    excluded = next(e for e in decision.suppressed if e["reason"] == "arm_excluded_type")
    assert excluded["count"] >= 3
    assert "failure" in excluded["types"]


def test_every_retrieval_records_its_ranking_features(db, workspace, corpus):
    """§14: the query, policy version, candidates, scores and ranking features must be logged."""
    retrieve(db, workspace_id=workspace.id, query="lock retry writer",
             arm=ExperimentArm.COLLECTIVE, limit=3)
    decision = db.query(RetrievalDecision).order_by(RetrievalDecision.created_at.desc()).first()
    assert decision.query == "lock retry writer"
    assert decision.policy_version
    assert decision.candidate_count >= decision.returned_count
    assert decision.ranking and all(
        {"artifact_id", "rank", "score", "features"} <= set(row) for row in decision.ranking
    )


def test_self_reported_confidence_ranks_below_every_evidence_signal():
    """Part A §A2.1: self-reported confidence is stored but ranks last — it is the one signal an
    agent can inflate for free."""
    from civitas.knowledge.retrieval import DEFAULT_WEIGHTS

    assert DEFAULT_WEIGHTS["confidence"] < DEFAULT_WEIGHTS["provenance"]
    assert DEFAULT_WEIGHTS["confidence"] < DEFAULT_WEIGHTS["validation"]
    assert DEFAULT_WEIGHTS["confidence"] < DEFAULT_WEIGHTS["utility"]
    assert DEFAULT_WEIGHTS["confidence"] < DEFAULT_WEIGHTS["lexical"]


def test_duplicate_warnings_are_off_wherever_negative_knowledge_is(db):
    """The warning mechanism must not leak information an arm removed."""
    for arm in all_arms():
        as_of = utcnow() if arm is ExperimentArm.COLLECTIVE_FROZEN else None
        policy = arm_policy(arm, as_of=as_of)
        if policy.drop_negative or policy.blind or policy.scramble:
            assert not policy.duplicate_warnings, (
                f"{arm.value} removes or scrambles negative knowledge but would still surface a "
                "prior failure, leaking what the arm exists to withhold"
            )
