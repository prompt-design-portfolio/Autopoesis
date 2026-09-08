"""The knowledge system (Part B §9–§16): graph, provenance, versioning, staleness,
consolidation, and the semantic component of hybrid retrieval."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from civitas.domain.enums import (
    ArtifactStatus,
    ArtifactType,
    EvidenceKind,
    ExperimentArm,
    RelationType,
    ValidationState,
)
from civitas.knowledge import consolidation, graph, staleness, versioning
from civitas.knowledge.embeddings import (
    HashingEmbedder,
    index_artifact,
    index_missing,
    search,
)
from civitas.knowledge.retrieval import retrieve
from civitas.persistence.models import Artifact, ArtifactRelation, ArtifactVersion
from civitas.persistence.types import utcnow


def _a(db, workspace, title, body="", kind=ArtifactType.EVIDENCE,
       evidence=EvidenceKind.MODEL_ASSERTION, validation=ValidationState.UNVALIDATED,
       env="test-env-1", **kw) -> Artifact:
    artifact = Artifact(
        workspace_id=workspace.id, type=kind, title=title, body=body or title,
        evidence_kind=evidence, validation_state=validation, environment_version=env, **kw,
    )
    db.add(artifact)
    db.flush()
    return artifact


def _link(db, source, target, kind):
    relation = ArtifactRelation(source_id=source.id, target_id=target.id, type=kind)
    db.add(relation)
    db.flush()
    return relation


# --------------------------------------------------------------------------
# embeddings (§14, §43)
# --------------------------------------------------------------------------
def test_the_embedder_is_deterministic_across_instances():
    """A hosted model can be revised underneath a campaign; this one cannot (§20, §46)."""
    a, b = HashingEmbedder(), HashingEmbedder()
    assert np.allclose(a.embed(["lock held across a retry"]), b.embed(["lock held across a retry"]))


def test_the_embedder_separates_paraphrase_from_unrelated_text():
    e = HashingEmbedder()
    v = e.embed([
        "the writer holds a lock across a retry",
        "a lock is held across retries by the writer",
        "unrelated cache eviction policy",
    ])
    assert float(v[0] @ v[1]) > 0.6, "paraphrase must be close"
    assert float(v[0] @ v[2]) < 0.4, "unrelated text must be far"


def test_vector_search_finds_a_paraphrase_that_shares_no_rare_term(db, workspace):
    target = _a(db, workspace, "The writer retains its lock across retries")
    _a(db, workspace, "Scheduler latency under load")
    db.commit()
    index_missing(db, workspace.id)
    db.commit()

    hits = search(db, workspace_id=workspace.id, query="writer keeps a lock during a retry")
    assert hits and hits[0].artifact_id == target.id


def test_re_embedding_under_a_new_model_does_not_overwrite_the_old_vector(db, workspace):
    """Retrieval computed under an older model must stay reproducible (§46)."""
    artifact = _a(db, workspace, "a claim")
    db.commit()
    index_artifact(db, artifact, HashingEmbedder(dim=64))
    index_artifact(db, artifact, HashingEmbedder(dim=128))
    db.commit()

    from civitas.persistence.models import ArtifactEmbedding

    rows = db.query(ArtifactEmbedding).filter(ArtifactEmbedding.artifact_id == artifact.id).all()
    assert len({r.model for r in rows}) == 2


def test_vector_search_cannot_bypass_an_arm(db, workspace):
    """The semantic index must never be a way around an ablation (§21)."""
    _a(db, workspace, "lock retry writer corruption", kind=ArtifactType.FAILURE)
    db.commit()
    index_missing(db, workspace.id)
    db.commit()

    result = retrieve(db, workspace_id=workspace.id, query="lock retry writer",
                      arm=ExperimentArm.COLLECTIVE_NO_NEGATIVE, limit=10)
    assert not [a for a in result.artifacts if a.type is ArtifactType.FAILURE]


# --------------------------------------------------------------------------
# graph and provenance (§10, §13)
# --------------------------------------------------------------------------
def test_provenance_traces_a_result_to_the_evidence_it_rests_on(db, workspace):
    observation = _a(db, workspace, "Observation 11", evidence=EvidenceKind.DIRECT_OBSERVATION)
    hypothesis = _a(db, workspace, "Hypothesis 22", kind=ArtifactType.HYPOTHESIS)
    experiment = _a(db, workspace, "Experiment 84", kind=ArtifactType.EXPERIMENT,
                    evidence=EvidenceKind.TOOL_OUTPUT)
    result = _a(db, workspace, "Final Result", kind=ArtifactType.RESULT)

    _link(db, hypothesis, observation, RelationType.DERIVED_FROM)
    _link(db, experiment, hypothesis, RelationType.TESTS)
    _link(db, result, experiment, RelationType.SUPPORTS)
    db.commit()

    chain = graph.provenance(db, result.id)
    assert chain is not None
    titles = {n.artifact.title for n in chain.nodes}
    assert {"Experiment 84", "Hypothesis 22", "Observation 11"} <= titles
    assert chain.depth == 3
    assert "Observation 11" in chain.render()


def test_provenance_distinguishes_evidence_kinds(db, workspace):
    """§13: a model assertion is not an observation, and no amount of assertion sums to one."""
    asserted = _a(db, workspace, "asserted", evidence=EvidenceKind.MODEL_ASSERTION)
    weak = _a(db, workspace, "weak result", kind=ArtifactType.RESULT)
    _link(db, weak, asserted, RelationType.DERIVED_FROM)

    measured = _a(db, workspace, "measured", evidence=EvidenceKind.REPRODUCED_RESULT)
    strong = _a(db, workspace, "strong result", kind=ArtifactType.RESULT)
    _link(db, strong, measured, RelationType.DERIVED_FROM)
    db.commit()

    weak_chain = graph.provenance(db, weak.id)
    strong_chain = graph.provenance(db, strong.id)
    assert strong_chain.grounding_score() > weak_chain.grounding_score()
    assert strong_chain.strongest_evidence() is EvidenceKind.REPRODUCED_RESULT


def test_provenance_reports_what_challenges_the_chain(db, workspace):
    """A record that shows only what supports a claim is an argument, not a record (§11)."""
    evidence = _a(db, workspace, "supporting evidence")
    result = _a(db, workspace, "the result", kind=ArtifactType.RESULT)
    _link(db, result, evidence, RelationType.DERIVED_FROM)

    objection = _a(db, workspace, "this does not replicate", kind=ArtifactType.CONTRADICTION)
    _link(db, objection, evidence, RelationType.CONTRADICTS)
    db.commit()

    chain = graph.provenance(db, result.id)
    assert [n.artifact.title for n in chain.challenges] == ["this does not replicate"]
    assert chain.grounding_score() < graph.ProvenanceChain(
        root=result, nodes=chain.nodes
    ).grounding_score()


def test_traversal_terminates_on_a_cycle(db, workspace):
    """An agent-built graph can contain cycles; an unbounded walk is a way to hang the API."""
    a = _a(db, workspace, "a")
    b = _a(db, workspace, "b")
    _link(db, a, b, RelationType.DERIVED_FROM)
    _link(db, b, a, RelationType.DERIVED_FROM)
    db.commit()

    nodes = graph.traverse(db, a.id, max_depth=10)
    assert len(nodes) <= 2


def test_graph_distance_returns_none_beyond_the_search_depth(db, workspace):
    """"Further than we looked" is not "far", and a caller ranking on proximity must be able to
    tell them apart."""
    chain = [_a(db, workspace, f"n{i}") for i in range(7)]
    for x, y in zip(chain, chain[1:], strict=False):
        _link(db, x, y, RelationType.DERIVED_FROM)
    db.commit()

    assert graph.graph_distance(db, chain[0].id, chain[2].id, max_depth=4) == 2
    assert graph.graph_distance(db, chain[0].id, chain[6].id, max_depth=3) is None


def test_open_contradictions_exclude_resolved_ones(db, workspace):
    live_a = _a(db, workspace, "claim A")
    live_b = _a(db, workspace, "claim B")
    _link(db, live_a, live_b, RelationType.CONTRADICTS)

    settled = _a(db, workspace, "claim C", status=ArtifactStatus.SUPERSEDED)
    other = _a(db, workspace, "claim D")
    _link(db, settled, other, RelationType.CONTRADICTS)
    db.commit()

    open_pairs = graph.contradictions(db, workspace.id)
    assert len(open_pairs) == 1
    assert open_pairs[0][0].title == "claim A"


# --------------------------------------------------------------------------
# versioning (§9)
# --------------------------------------------------------------------------
def test_a_revision_keeps_the_previous_state(db, workspace):
    artifact = _a(db, workspace, "original title", "original body")
    db.commit()

    versioning.revise(db, artifact, title="revised title", body="revised body",
                      reason="corrected after reproduction")
    db.commit()

    assert artifact.version == 2
    assert artifact.title == "revised title"
    history = versioning.history(db, artifact.id)
    assert len(history) == 1
    assert history[0].title == "original title", "the prior state must survive verbatim"
    assert history[0].change_reason == "corrected after reproduction"


def test_version_rows_cannot_be_edited(db, workspace):
    from civitas.persistence.session import ImmutableViolation

    artifact = _a(db, workspace, "t", "b")
    db.commit()
    versioning.revise(db, artifact, body="new")
    db.commit()

    snapshot = db.query(ArtifactVersion).one()
    snapshot.body = "rewritten history"
    with pytest.raises(ImmutableViolation):
        db.commit()
    db.rollback()


def test_supersession_keeps_both_artifacts_readable(db, workspace):
    old = _a(db, workspace, "the earlier conclusion")
    new = _a(db, workspace, "the corrected conclusion")
    db.commit()

    versioning.supersede(db, old=old, new=new, reason="measurement contradicted it")
    db.commit()

    assert db.get(Artifact, old.id) is not None, "supersession must not remove the old artifact"
    assert old.status is ArtifactStatus.SUPERSEDED
    assert old.superseded_by_id == new.id
    assert versioning.resolve_head(db, old.id).id == new.id


def test_resolving_a_head_terminates_on_a_supersession_cycle(db, workspace):
    a = _a(db, workspace, "a")
    b = _a(db, workspace, "b")
    db.commit()
    a.superseded_by_id = b.id
    b.superseded_by_id = a.id
    db.commit()

    assert versioning.resolve_head(db, a.id) is not None


# --------------------------------------------------------------------------
# staleness (§15)
# --------------------------------------------------------------------------
def test_an_artifact_from_a_previous_environment_becomes_visibly_stale(db, workspace):
    """The `π` case: not merely unhelpful but actively misleading."""
    old_era = _a(db, workspace, "device accepts X for class A", env="device-e1-s0")
    current = _a(db, workspace, "device accepts Y for class B", env="device-e2-s0")
    db.commit()

    report = staleness.mark_workspace(db, workspace_id=workspace.id,
                                      environment_version="device-e2-s0")
    db.commit()

    assert report.marked_stale == 1
    assert old_era.is_stale and old_era.status is ArtifactStatus.STALE
    assert old_era.meta["stale_reason"] == "environment_drift"
    assert not current.is_stale


def test_a_stale_artifact_is_not_deleted_or_hidden(db, workspace):
    """§15: a stale artifact should not disappear. It should be visibly stale."""
    artifact = _a(db, workspace, "an old finding", env="old-env")
    db.commit()
    staleness.mark_workspace(db, workspace_id=workspace.id, environment_version="new-env")
    db.commit()

    assert db.get(Artifact, artifact.id) is not None
    assert artifact.archived_at is None
    assert artifact in staleness.stale_artifacts(db, workspace.id)


def test_an_expired_validity_horizon_marks_an_artifact_stale(db, workspace):
    artifact = _a(db, workspace, "time-limited claim", validity_horizon_s=60)
    db.commit()
    artifact.created_at = utcnow() - timedelta(seconds=3600)
    db.commit()

    staleness.mark_workspace(db, workspace_id=workspace.id, environment_version="test-env-1")
    db.commit()
    assert artifact.is_stale
    assert artifact.meta["stale_reason"] == "expired"


def test_staleness_is_reversible(db, workspace):
    """One-way staleness would slowly condemn a whole workspace with no way back."""
    artifact = _a(db, workspace, "a claim", env="old-env")
    db.commit()
    staleness.mark_workspace(db, workspace_id=workspace.id, environment_version="new-env")
    db.commit()
    assert artifact.is_stale

    staleness.revalidate(db, artifact, environment_version="new-env")
    db.commit()
    assert not artifact.is_stale
    assert artifact.status is ArtifactStatus.ACTIVE


def test_invalidation_cascades_as_staleness_not_as_refutation(db, workspace):
    """A conclusion resting on a refuted premise is unsupported, not thereby false. It needs
    re-examination, and marking it stale says exactly that."""
    premise = _a(db, workspace, "the premise")
    conclusion = _a(db, workspace, "the conclusion", kind=ArtifactType.CONCLUSION)
    _link(db, conclusion, premise, RelationType.DEPENDS_ON)
    refutation = _a(db, workspace, "measurement refutes the premise",
                    evidence=EvidenceKind.DIRECT_OBSERVATION)
    db.commit()

    affected = staleness.invalidate(db, artifact=premise, by=refutation)
    db.commit()

    assert premise.id in affected and conclusion.id in affected
    assert conclusion.is_stale
    assert conclusion.meta["stale_reason"] == "depends_on_invalidated"
    assert conclusion.validation_state is not ValidationState.REFUTED, (
        "a dependent conclusion must not be marked refuted by association"
    )


# --------------------------------------------------------------------------
# consolidation (§16)
# --------------------------------------------------------------------------
def test_duplicates_are_linked_never_deleted(db, workspace):
    """§16: never destroy source provenance."""
    first = _a(db, workspace, "The writer holds a lock across a retry")
    second = _a(db, workspace, "The writer holds a lock across a retry")
    db.commit()

    report = consolidation.link_duplicates(db, workspace_id=workspace.id)
    db.commit()

    assert report.duplicates_linked == 1
    assert db.get(Artifact, first.id) is not None
    assert db.get(Artifact, second.id) is not None
    assert second.archived_at is None, "independent replication is evidence, not redundancy"
    edge = db.query(ArtifactRelation).filter(
        ArtifactRelation.type == RelationType.DUPLICATES
    ).one()
    assert edge.confidence >= consolidation.DUPLICATE_THRESHOLD


def test_duplicate_detection_does_not_merge_across_types(db, workspace):
    """A failure and a piece of evidence with similar text say different kinds of thing."""
    _a(db, workspace, "probing operation X for class A", kind=ArtifactType.EVIDENCE)
    _a(db, workspace, "probing operation X for class A", kind=ArtifactType.FAILURE)
    db.commit()

    report = consolidation.link_duplicates(db, workspace_id=workspace.id)
    db.commit()
    assert report.duplicates_linked == 0


def test_the_similarity_distribution_is_reported(db, workspace):
    """The threshold must be checkable against a real corpus, not merely asserted."""
    for _ in range(4):
        _a(db, workspace, "The writer holds a lock across a retry")
    db.commit()
    report = consolidation.link_duplicates(db, workspace_id=workspace.id)
    db.commit()
    assert report.similarity_distribution


def test_a_summary_keeps_links_to_every_artifact_it_summarises(db, workspace):
    """§16: a consolidated summary must retain links to underlying evidence."""
    members = [_a(db, workspace, f"lock retry finding number {i}") for i in range(4)]
    db.commit()

    group = consolidation.Cluster(key=members[0].id, members=members)
    summary = consolidation.summarise_cluster(db, group, workspace_id=workspace.id)
    db.commit()

    linked = db.query(ArtifactRelation).filter(
        ArtifactRelation.source_id == summary.id,
        ArtifactRelation.type == RelationType.DERIVED_FROM,
    ).all()
    assert {r.target_id for r in linked} == {m.id for m in members}
    assert summary.structured["member_count"] == 4


def test_a_summary_never_claims_more_than_its_members(db, workspace):
    """A generated summary recorded as established fact would quietly upgrade assertion to
    evidence."""
    members = [
        _a(db, workspace, f"finding {i}", evidence=EvidenceKind.REPRODUCED_RESULT,
           validation=ValidationState.EVALUATOR_CONFIRMED)
        for i in range(3)
    ]
    db.commit()
    summary = consolidation.summarise_cluster(
        db, consolidation.Cluster(key=members[0].id, members=members),
        workspace_id=workspace.id,
    )
    db.commit()

    assert summary.evidence_kind is EvidenceKind.INFERENCE
    assert summary.validation_state is ValidationState.UNVALIDATED, (
        "a summary of validated artifacts is not itself validated"
    )


def test_clustering_groups_related_artifacts(db, workspace):
    for i in range(3):
        _a(db, workspace, f"the writer holds a lock across a retry variant {i}")
    for i in range(3):
        _a(db, workspace, f"scheduler latency under heavy load case {i}")
    db.commit()

    clusters = consolidation.cluster(db, workspace_id=workspace.id, min_size=2)
    assert len(clusters) >= 2
    assert all(c.size >= 2 for c in clusters)


def test_contradiction_detection_records_a_candidate_not_a_verdict(db, workspace):
    """A model's opinion recorded as a detected contradiction is exactly the unearned upgrade
    §11 warns about."""
    _a(db, workspace, "operation X works for class A", kind=ArtifactType.EVIDENCE)
    _a(db, workspace, "operation X works for class A", kind=ArtifactType.FAILURE)
    db.commit()

    found = consolidation.detect_contradictions(db, workspace_id=workspace.id)
    db.commit()
    assert found
    edge = db.query(ArtifactRelation).filter(
        ArtifactRelation.type == RelationType.CONTRADICTS
    ).first()
    assert edge.evidence_kind is EvidenceKind.INFERENCE
    assert edge.creator_kind == "system"


# --------------------------------------------------------------------------
# hybrid retrieval (§14)
# --------------------------------------------------------------------------
def test_retrieval_records_every_hybrid_feature(db, workspace):
    _a(db, workspace, "lock retry writer corruption")
    db.commit()
    index_missing(db, workspace.id)
    db.commit()

    result = retrieve(db, workspace_id=workspace.id, query="lock retry writer",
                      arm=ExperimentArm.COLLECTIVE, limit=5)
    assert result.features
    for required in ("lexical", "vector", "graph", "task_relevance", "recency",
                     "validation", "provenance", "utility", "confidence",
                     "negative_relevance"):
        assert required in result.features[0], f"§14 component {required} is not recorded"


def test_a_relevant_prior_failure_gains_relevance(db, workspace):
    """§14: if an agent is about to repeat a known failed method, prior failure artifacts should
    receive increased relevance."""
    failure = _a(db, workspace, "approach using the retry path failed",
                 kind=ArtifactType.FAILURE)
    db.commit()
    index_missing(db, workspace.id)
    db.commit()

    result = retrieve(db, workspace_id=workspace.id, query="approach using the retry path",
                      arm=ExperimentArm.COLLECTIVE, limit=5)
    index = [a.id for a in result.artifacts].index(failure.id)
    assert result.features[index]["negative_relevance"] > 0


def test_an_unrelated_failure_is_not_boosted(db, workspace):
    """A blanket boost would flood every result with unrelated failures, which is a good way to
    make agents ignore them."""
    unrelated = _a(db, workspace, "cache eviction attempt failed", kind=ArtifactType.FAILURE)
    db.commit()
    index_missing(db, workspace.id)
    db.commit()

    result = retrieve(db, workspace_id=workspace.id, query="lock retry writer",
                      arm=ExperimentArm.COLLECTIVE, limit=5)
    if unrelated.id in [a.id for a in result.artifacts]:
        i = [a.id for a in result.artifacts].index(unrelated.id)
        assert result.features[i]["negative_relevance"] == 0.0


def test_relevant_disagreement_survives_the_ranking(db, workspace):
    """§14: agents should sometimes be deliberately shown relevant disagreement.

    A pure relevance ranking removes the artifact most likely to correct the top result, because a
    contradiction of a strong match is usually a weaker match itself.
    """
    top = _a(db, workspace, "the writer holds a lock across a retry",
             validation=ValidationState.EVALUATOR_CONFIRMED,
             evidence=EvidenceKind.TOOL_OUTPUT)
    objection = _a(db, workspace, "that result did not reproduce on a second run",
                   kind=ArtifactType.CONTRADICTION)
    _link(db, objection, top, RelationType.CONTRADICTS)
    for i in range(8):
        _a(db, workspace, f"the writer holds a lock across a retry, note {i}")
    db.commit()
    index_missing(db, workspace.id)
    db.commit()

    result = retrieve(db, workspace_id=workspace.id,
                      query="the writer holds a lock across a retry",
                      arm=ExperimentArm.COLLECTIVE, limit=4)
    returned = [a.id for a in result.artifacts]
    assert top.id in returned
    assert objection.id in returned, "the contradiction was crowded out by agreeing artifacts"
    assert len(returned) <= 4, "exposing disagreement must not change how much is returned"


def test_retrieval_does_not_return_ten_near_identical_artifacts(db, workspace):
    """§14: a retrieval result should avoid ten nearly identical artifacts."""
    for _ in range(12):
        _a(db, workspace, "the writer holds a lock across a retry")
    _a(db, workspace, "the writer also drops the lock on the error path")
    db.commit()
    index_missing(db, workspace.id)
    db.commit()

    result = retrieve(db, workspace_id=workspace.id, query="writer lock retry",
                      arm=ExperimentArm.COLLECTIVE, limit=5)
    titles = [a.title for a in result.artifacts]
    assert len(set(titles)) > 1, "every returned artifact was identical"


# --------------------------------------------------------------------------
# the retrieval-quality benchmark (§14, §48)
# --------------------------------------------------------------------------
def test_the_retrieval_benchmark_separates_the_two_query_conditions(db, org):
    """The instrument that can see what the newcomer benchmark cannot.

    On exact queries both policies are perfect, which is *why* the newcomer benchmark reported a
    null for M5: BM25 already ranks the target first and there is no rank left to improve. On
    paraphrases — what an agent produces when it does not already know the answer's vocabulary —
    the hybrid policy must do better, or the semantic component is not earning its weight.
    """
    from civitas.experiments.retrieval_benchmark import run_retrieval_benchmark

    result = run_retrieval_benchmark(db, organization_id=org.id, distractors=120)
    db.commit()

    scores = result.scores
    assert scores["lexical/exact"].mrr == pytest.approx(1.0), (
        "if lexical is not already perfect on exact queries, the null needs a different "
        "explanation"
    )
    assert scores["hybrid/exact"].mrr == pytest.approx(1.0)
    assert scores["hybrid/paraphrase"].mrr > scores["lexical/paraphrase"].mrr, (
        "the semantic component adds nothing where it is supposed to be the whole point"
    )
    assert scores["hybrid/paraphrase"].recall_at_3 >= scores["lexical/paraphrase"].recall_at_3


def test_a_paraphrase_query_shares_no_content_word_with_its_target():
    """Otherwise the paraphrase condition is indistinguishable from the exact one."""
    from civitas.experiments.retrieval_benchmark import PROBES
    from civitas.knowledge.retrieval import tokenize

    for statement, _exact, paraphrase in PROBES:
        shared = set(tokenize(statement)) & set(tokenize(paraphrase))
        assert not shared, f"paraphrase leaks {shared} from its target"
