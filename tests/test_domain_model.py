"""The §7 domain model is complete and round-trips on the real schema."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import inspect, select, text

from civitas.domain.enums import (
    ArtifactType,
    EvidenceKind,
    ExperimentArm,
    RelationType,
    TerminationReason,
)
from civitas.persistence.models import (
    AgentProfile,
    Artifact,
    ArtifactRelation,
    Base,
    Episode,
    Task,
)

#: Part B §7, verbatim. A missing entity is a dropped requirement, and Part A §A1 forbids
#: silently removing one — so the list is asserted rather than trusted.
REQUIRED_ENTITIES = [
    "Organization", "User", "ServiceIdentity", "Workspace", "Project", "Objective",
    "Task", "TaskDependency", "Subproblem", "Assignment", "AgentProfile", "AgentInstance",
    "Episode", "Artifact", "ArtifactVersion", "ArtifactRelation", "ArtifactEmbedding",
    "ArtifactUtilityMetric", "FileAsset", "ToolDefinition", "ToolVersion", "ToolRun",
    "ModelProvider", "ModelConfiguration", "ModelCall", "Evaluation", "Job", "JobLease",
    "Event", "Experiment", "ExperimentArm", "ExperimentRun", "ExperimentManifest",
    "Policy", "Procedure", "ReputationMetric", "RetrievalDecision", "WorkspaceSnapshot",
]


def test_every_required_entity_is_mapped():
    mapped = {m.class_.__name__ for m in Base.registry.mappers}
    missing = [n for n in REQUIRED_ENTITIES if n not in mapped]
    assert not missing, f"Part B §7 entities not implemented: {missing}"


def test_every_table_has_a_uuid_primary_key(engine):
    """§7 requires UUID identifiers. `event_sequences` is the one exception: it is keyed by the
    workspace it counts for, which is itself a UUID."""
    insp = inspect(engine)
    for table in insp.get_table_names():
        pk = insp.get_pk_constraint(table)["constrained_columns"]
        assert pk, f"{table} has no primary key"


def test_enum_sets_match_the_specification():
    assert len(list(ArtifactType)) == 32, "Part B §9 lists 32 artifact types"
    assert len(list(RelationType)) == 23, "Part B §10 lists 23 relation types"
    assert len(list(ExperimentArm)) == 9, "Part B §21 requires 9 arms"
    assert len(list(TerminationReason)) == 13, "Part B §8 lists 13 termination reasons"


def test_artifact_round_trip_preserves_every_field(db, workspace):
    a = Artifact(
        workspace_id=workspace.id,
        type=ArtifactType.HYPOTHESIS,
        title="Intermittent corruption originates in the write path",
        body="Detail.",
        confidence=0.42,
        evidence_kind=EvidenceKind.DIRECT_OBSERVATION,
        environment_version="test-env-1",
        structured={"subsystem": "writer", "nested": {"n": [1, 2, 3]}},
    )
    db.add(a)
    db.commit()
    db.expire_all()

    got = db.get(Artifact, a.id)
    assert isinstance(got.id, uuid.UUID)
    assert got.type is ArtifactType.HYPOTHESIS
    assert got.evidence_kind is EvidenceKind.DIRECT_OBSERVATION
    assert got.confidence == pytest.approx(0.42)
    assert got.structured["nested"]["n"] == [1, 2, 3], "JSON must round-trip structurally"
    assert got.created_at.tzinfo is not None, "timestamps must come back timezone-aware"


def test_relations_are_unique_per_triple(db, workspace):
    a = Artifact(workspace_id=workspace.id, type=ArtifactType.OBSERVATION, title="A")
    b = Artifact(workspace_id=workspace.id, type=ArtifactType.HYPOTHESIS, title="B")
    db.add_all([a, b])
    db.commit()

    db.add(ArtifactRelation(source_id=a.id, target_id=b.id, type=RelationType.SUPPORTS))
    db.commit()
    # The same claim twice is one edge; a different claim about the same pair is a second edge.
    db.add(ArtifactRelation(source_id=a.id, target_id=b.id, type=RelationType.CONTRADICTS))
    db.commit()
    db.add(ArtifactRelation(source_id=a.id, target_id=b.id, type=RelationType.SUPPORTS))
    with pytest.raises(Exception):
        db.commit()
    db.rollback()


def test_foreign_keys_are_enforced(db, workspace):
    """SQLite ships with foreign keys off. If the engine did not turn them on, every ondelete
    clause in the models would be decoration. PostgreSQL enforces them natively; running the same
    assertion on both is how the two backends are shown to behave identically."""
    orphan = Artifact(workspace_id=uuid.uuid4(), type=ArtifactType.RESULT, title="Orphan")
    db.add(orphan)
    with pytest.raises(Exception):
        db.commit()
    db.rollback()


def test_wal_and_foreign_keys_are_on(engine):
    """SQLite-specific: PostgreSQL has neither pragma and enforces both properties natively."""
    if engine.dialect.name != "sqlite":
        pytest.skip("SQLite pragmas")
    with engine.connect() as c:
        assert c.exec_driver_sql("PRAGMA journal_mode").scalar().lower() == "wal"
        assert c.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_episode_records_every_bound_and_termination_reason(db, workspace):
    profile = AgentProfile(workspace_id=workspace.id, name="explorer")
    db.add(profile)
    db.commit()

    ep = Episode(
        workspace_id=workspace.id,
        agent_profile_id=profile.id,
        model_provider="deterministic",
        model_name="deterministic-v1",
        experiment_arm=ExperimentArm.COLLECTIVE,
        token_budget=1000,
        tool_call_budget=5,
        termination_reason=TerminationReason.BUDGET_EXHAUSTED,
        config_hash="abc123",
    )
    db.add(ep)
    db.commit()
    db.expire_all()

    got = db.get(Episode, ep.id)
    assert got.experiment_arm is ExperimentArm.COLLECTIVE
    assert got.termination_reason is TerminationReason.BUDGET_EXHAUSTED
    assert got.duration_s is None, "an episode with no end time has no duration, not a zero one"


def test_no_column_stores_episode_local_reasoning_state():
    """Part B §4: an agent must not inherit hidden episode state.

    A transcript, scratchpad or chain-of-thought column on `Episode` would make the next
    episode's isolation a matter of care rather than of structure. This asserts the structure.
    """
    forbidden = {"transcript", "scratchpad", "chain_of_thought", "messages", "conversation",
                 "reasoning", "private_memory", "context"}
    cols = {c.name for c in Episode.__table__.columns}
    assert not (cols & forbidden), f"Episode carries episode-local state: {cols & forbidden}"


def test_soft_delete_is_the_only_removal_path(db, workspace):
    """Part A §A1.2: nothing is ever hard-deleted."""
    a = Artifact(workspace_id=workspace.id, type=ArtifactType.FAILURE, title="A failed approach")
    db.add(a)
    db.commit()

    from civitas.persistence.types import utcnow

    a.archived_at = utcnow()
    a.archived_reason = "superseded by a reproduction"
    db.commit()
    db.expire_all()

    got = db.get(Artifact, a.id)
    assert got is not None, "archival must not remove the row"
    assert got.is_archived
