"""Outcome-linked credit assignment (Part A §A2.1) and manifests (Part B §46)."""

from __future__ import annotations

import pytest

from civitas.domain.enums import (
    ArtifactType,
    ExperimentArm,
    RelationType,
)
from civitas.experiments import credit
from civitas.experiments.manifest import build_manifest, manifest_dict
from civitas.persistence.models import (
    AgentProfile,
    Artifact,
    ArtifactRelation,
    ArtifactUsage,
    ArtifactUtilityMetric,
    Episode,
    Evaluation,
    ToolDefinition,
    ToolRun,
)


@pytest.fixture
def profile(db, workspace):
    p = AgentProfile(workspace_id=workspace.id, name="explorer")
    db.add(p)
    db.commit()
    return p


def _episode(db, workspace, profile, arm=ExperimentArm.COLLECTIVE) -> Episode:
    e = Episode(
        workspace_id=workspace.id, agent_profile_id=profile.id,
        model_provider="deterministic", model_name="deterministic-v1", experiment_arm=arm,
    )
    db.add(e)
    db.flush()
    return e


def _artifact(db, workspace, title, kind=ArtifactType.EVIDENCE) -> Artifact:
    a = Artifact(workspace_id=workspace.id, type=kind, title=title, body=title)
    db.add(a)
    db.flush()
    return a


def _evaluation(db, workspace, episode, succeeded=True) -> Evaluation:
    ev = Evaluation(
        workspace_id=workspace.id, episode_id=episode.id, scope="episode",
        evaluator_kind="exact_match", succeeded=succeeded, score=1.0 if succeeded else 0.0,
    )
    db.add(ev)
    db.flush()
    return ev


def test_credit_flows_to_what_was_read_not_to_what_was_retrieved(db, workspace, profile):
    """The distinction Part A §A2.1 is built on."""
    read = _artifact(db, workspace, "the artifact the episode read")
    merely_returned = _artifact(db, workspace, "the artifact retrieval returned")
    episode = _episode(db, workspace, profile)
    db.add(ArtifactUsage(episode_id=episode.id, artifact_id=read.id))
    db.commit()

    report = credit.assign_credit(db, _evaluation(db, workspace, episode, True))
    db.commit()

    assert report.artifacts_credited == 1
    db.refresh(read)
    db.refresh(merely_returned)
    assert read.downstream_utility > 0
    assert merely_returned.downstream_utility == 0.0


def test_credit_decays_with_graph_distance(db, workspace, profile):
    """§A2.1: credit decays with graph distance through derived_from / depends_on / uses."""
    near = _artifact(db, workspace, "read directly")
    mid = _artifact(db, workspace, "one hop back")
    far = _artifact(db, workspace, "two hops back")
    db.add_all([
        ArtifactRelation(source_id=near.id, target_id=mid.id, type=RelationType.DERIVED_FROM),
        ArtifactRelation(source_id=mid.id, target_id=far.id, type=RelationType.DEPENDS_ON),
    ])
    episode = _episode(db, workspace, profile)
    db.add(ArtifactUsage(episode_id=episode.id, artifact_id=near.id))
    db.commit()

    credit.assign_credit(db, _evaluation(db, workspace, episode, True))
    db.commit()

    for a in (near, mid, far):
        db.refresh(a)
    assert near.downstream_utility > mid.downstream_utility > far.downstream_utility > 0
    assert mid.downstream_utility == pytest.approx(near.downstream_utility * credit.DECAY_PER_HOP)


def test_credit_does_not_flow_along_a_contradiction(db, workspace, profile):
    """An artifact that merely contradicts a successful one did not contribute to the success."""
    used = _artifact(db, workspace, "used")
    opposed = _artifact(db, workspace, "contradicts it")
    db.add(ArtifactRelation(source_id=used.id, target_id=opposed.id,
                            type=RelationType.CONTRADICTS))
    episode = _episode(db, workspace, profile)
    db.add(ArtifactUsage(episode_id=episode.id, artifact_id=used.id))
    db.commit()

    credit.assign_credit(db, _evaluation(db, workspace, episode, True))
    db.commit()
    db.refresh(opposed)
    assert opposed.downstream_utility == 0.0


def test_a_failure_lowers_utility(db, workspace, profile):
    a = _artifact(db, workspace, "misleading")
    episode = _episode(db, workspace, profile)
    db.add(ArtifactUsage(episode_id=episode.id, artifact_id=a.id))
    db.commit()

    credit.assign_credit(db, _evaluation(db, workspace, episode, False))
    db.commit()
    db.refresh(a)
    assert a.downstream_utility < 0


def test_credit_is_not_double_counted_on_a_retry(db, workspace, profile):
    """A resumed campaign or a retried job must not count an outcome twice (§41, §42)."""
    a = _artifact(db, workspace, "used")
    episode = _episode(db, workspace, profile)
    db.add(ArtifactUsage(episode_id=episode.id, artifact_id=a.id))
    db.commit()
    evaluation = _evaluation(db, workspace, episode, True)

    first = credit.assign_credit(db, evaluation)
    db.commit()
    second = credit.assign_credit(db, evaluation)
    db.commit()

    assert not first.skipped_already_assigned
    assert second.skipped_already_assigned
    assert db.query(ArtifactUtilityMetric).count() == 1


def test_an_artifact_surfaced_as_a_duplicate_warning_is_not_credited(db, workspace, profile):
    """It was injected by the runtime, not sought by the episode. Crediting it would let the
    warning mechanism inflate whatever it happened to surface."""
    warned = _artifact(db, workspace, "prior failure surfaced by the runtime")
    episode = _episode(db, workspace, profile)
    db.add(ArtifactUsage(episode_id=episode.id, artifact_id=warned.id,
                         surfaced_as_duplicate_warning=True))
    db.commit()

    report = credit.assign_credit(db, _evaluation(db, workspace, episode, True))
    db.commit()
    assert report.artifacts_credited == 0
    db.refresh(warned)
    assert warned.downstream_utility == 0.0


def test_utility_is_the_fold_of_immutable_events_not_a_running_total(db, workspace, profile):
    """The events are the truth; the column is a cache. Rebuilding from the events is what makes
    an artifact's utility at any time reconstructable."""
    a = _artifact(db, workspace, "used twice")
    for succeeded in (True, True, False):
        episode = _episode(db, workspace, profile)
        db.add(ArtifactUsage(episode_id=episode.id, artifact_id=a.id))
        db.flush()
        credit.assign_credit(db, _evaluation(db, workspace, episode, succeeded))
    db.commit()

    events = db.query(ArtifactUtilityMetric).filter(
        ArtifactUtilityMetric.artifact_id == a.id
    ).all()
    assert len(events) == 3
    db.refresh(a)
    assert a.downstream_utility == pytest.approx(sum(e.delta for e in events))

    # Corrupt the cache and rebuild it: the events must fully determine the value.
    a.downstream_utility = 999.0
    db.commit()
    credit.recompute(db, [a.id])
    db.commit()
    db.refresh(a)
    assert a.downstream_utility == pytest.approx(sum(e.delta for e in events))


def test_tools_the_episode_invoked_receive_credit(db, workspace, profile):
    definition = ToolDefinition(workspace_id=workspace.id, name="probe", kind="builtin")
    db.add(definition)
    db.flush()
    episode = _episode(db, workspace, profile)
    db.add(ToolRun(episode_id=episode.id, tool_definition_id=definition.id, succeeded=True))
    db.commit()

    report = credit.assign_credit(db, _evaluation(db, workspace, episode, True))
    db.commit()
    assert report.tools_credited == 1
    db.refresh(definition)
    assert definition.downstream_utility > 0


def test_calibration_reports_absence_rather_than_zero(db, workspace, profile):
    """ARCHITECTURE §3.9: a metric that cannot be computed says so. Reporting 0.0 would present
    an absence as a finding."""
    result = credit.calibration(db, workspace.id)
    assert result["correlation"] is None
    assert "fewer than 3" in str(result["reason"])


def test_calibration_is_computed_once_outcomes_exist(db, workspace, profile):
    for i, (confidence, succeeded) in enumerate(
        [(0.9, True), (0.8, True), (0.2, False), (0.1, False), (0.85, True)]
    ):
        a = Artifact(workspace_id=workspace.id, type=ArtifactType.EVIDENCE,
                     title=f"a{i}", body="b", confidence=confidence)
        db.add(a)
        db.flush()
        episode = _episode(db, workspace, profile)
        db.add(ArtifactUsage(episode_id=episode.id, artifact_id=a.id))
        db.flush()
        credit.assign_credit(db, _evaluation(db, workspace, episode, succeeded))
    db.commit()

    result = credit.calibration(db, workspace.id)
    assert result["correlation"] is not None
    assert result["correlation"] > 0.5, "confidence should track outcome once credit is assigned"


# --------------------------------------------------------------------------
# manifests (Part B §46)
# --------------------------------------------------------------------------
def test_a_manifest_records_everything_section_46_requires(db, workspace, settings):
    manifest = build_manifest(
        db, experiment_run_id=None, arm="collective", provider="policy",
        model_name="policy-v1", model_version="policy-v1",
        sampling={"temperature": 0.0}, prompt_versions={"system": "v1"},
        tool_versions={"probe_device": "1"}, retrieval_policy={"version": "retrieval/1.0"},
        budgets={"tool_calls": 7}, evaluator={"kind": "exact_match"},
        seeds={"seed": 0}, sandbox_backend="subprocess", settings=settings,
    )
    db.commit()
    exported = manifest_dict(manifest)
    for required in (
        "git_commit", "app_version", "schema_version", "dependency_versions", "provider",
        "model_name", "model_version", "sampling_parameters", "prompt_versions", "tool_versions",
        "retrieval_policy", "experiment_arm", "budgets", "evaluator", "environment", "seeds",
        "config_hash", "sandbox_backend",
    ):
        assert required in exported, f"§46 requires {required}"
    assert exported["schema_version"] != ""


def test_a_manifest_never_contains_a_secret(db, workspace):
    from civitas.config import Settings

    leaky = Settings(
        database_url="postgresql://u:HUNTER2@host/db",
        anthropic_api_key="sk-ant-SECRET", jwt_secret="JWT-SECRET",
    )
    manifest = build_manifest(
        db, experiment_run_id=None, arm="collective", provider="anthropic",
        model_name="claude-sonnet-5", settings=leaky,
    )
    db.commit()
    blob = str(manifest_dict(manifest))
    for secret in ("HUNTER2", "sk-ant-SECRET", "JWT-SECRET"):
        assert secret not in blob


def test_a_manifest_never_contains_the_expected_answer(db, workspace, settings):
    """A manifest is readable; ground truth in one would leak the benchmark (§47)."""
    manifest = build_manifest(
        db, experiment_run_id=None, arm="collective", provider="policy", model_name="policy-v1",
        evaluator={"kind": "exact_match", "expected": "SECRET-ANSWER", "device_digest": "abc"},
        settings=settings,
    )
    db.commit()
    assert "SECRET-ANSWER" not in str(manifest_dict(manifest))
    assert manifest.evaluator["device_digest"] == "abc", "non-leaking fields are kept"


def test_two_arms_of_one_experiment_share_a_configuration_hash(db, workspace, settings):
    """The A2.2 gate needs matched comparisons to be *checkable*: the arm and the seed are the
    variables under test, so they are outside the configuration hash."""
    common = dict(
        provider="policy", model_name="policy-v1", budgets={"tool_calls": 7},
        evaluator={"kind": "exact_match"}, settings=settings, experiment_run_id=None,
    )
    a = build_manifest(db, arm="collective", seeds={"seed": 0}, **common)
    b = build_manifest(db, arm="memory_reset", seeds={"seed": 1}, **common)
    db.commit()
    assert a.config_hash == b.config_hash

    different = build_manifest(db, arm="collective", seeds={"seed": 0},
                               **{**common, "budgets": {"tool_calls": 9}})
    db.commit()
    assert different.config_hash != a.config_hash, "a changed budget is a changed configuration"


def test_a_manifest_cannot_be_edited(db, workspace, settings):
    """§46 records how a result was produced; an editable manifest records how it is now claimed
    to have been produced."""
    from civitas.persistence.session import ImmutableViolation

    manifest = build_manifest(db, experiment_run_id=None, arm="collective", provider="policy",
                              model_name="policy-v1", settings=settings)
    db.commit()
    manifest.model_name = "something-else"
    with pytest.raises(ImmutableViolation):
        db.commit()
    db.rollback()
