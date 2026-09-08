"""Long-horizon projects and request decomposition (Part B §5, §32).

The property §32 actually asks for is survival, and survival is not testable by asserting that a
function returns a number. It is testable by *destroying the process state* — disposing every
engine and session — and asking the same question again. `test_a_projects_status_survives_the_loss
_of_every_engine_and_session` is the whole of §32 in one assertion.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from civitas.domain.enums import (
    ArtifactType,
    EventType,
    TaskStatus,
    TerminationReason,
    ValidationState,
)
from civitas.persistence.engine import create_db_engine
from civitas.persistence.models import (
    AgentProfile,
    Artifact,
    Episode,
    Event,
    Project,
    Task,
    TaskDependency,
)
from civitas.persistence.session import install_guards
from civitas.services.projects import (
    STAGES,
    DeterministicDecomposer,
    ModelDecomposer,
    complete_task,
    next_ready_task,
    project_status,
    project_timeline,
    submit_request,
)

REQUEST = (
    "Investigate the cause of the intermittent data corruption in the ledger writer "
    "and produce a verified fix"
)


def _plan(db, workspace, request: str = REQUEST):
    plan = submit_request(db, workspace_id=workspace.id, request=request)
    db.commit()
    return plan


# --------------------------------------------------------------------------- §5 decomposition


def test_a_request_becomes_a_project_an_objective_and_a_chain_not_an_answer(db, workspace):
    """§5: the system responds to a high-level request by creating work, not by answering it."""
    plan = _plan(db, workspace)

    assert plan.project.description == REQUEST
    assert plan.objective.project_id == plan.project.id
    assert [t.task_family for t in plan.tasks] == [s[0] for s in STAGES]
    assert all(t.project_id == plan.project.id for t in plan.tasks)
    assert all(t.objective_id == plan.objective.id for t in plan.tasks)


def test_only_the_first_stage_is_ready_and_the_rest_are_blocked_on_their_predecessor(db, workspace):
    """Nothing about synthesis is knowable before the investigation exists, so the plan must not
    offer it. The dependency rows are what make that structural rather than advisory."""
    plan = _plan(db, workspace)

    assert plan.tasks[0].status is TaskStatus.READY
    assert [t.status for t in plan.tasks[1:]] == [TaskStatus.BLOCKED] * 3

    dependencies = {
        (d.task_id, d.depends_on_task_id)
        for d in db.execute(select(TaskDependency)).scalars()
    }
    assert dependencies == {
        (plan.tasks[i + 1].id, plan.tasks[i].id) for i in range(len(plan.tasks) - 1)
    }


def test_the_scheduler_offers_the_investigation_and_nothing_downstream_of_it(db, workspace):
    """§27's `ready_tasks` already refuses blocked work, so §5 needed no scheduler change. This
    test is what says so: the chain is enforced by the existing allocator, not by a second one."""
    plan = _plan(db, workspace)

    nxt = next_ready_task(db, plan.project.id)
    assert nxt is not None and nxt.id == plan.tasks[0].id


def test_each_stage_carries_the_role_that_stage_needs(db, workspace):
    """§26/§28: the stages are separable task families with different role preferences, which is
    what lets specialization be measured per family rather than in aggregate."""
    plan = _plan(db, workspace)
    assert [t.meta["preferred_role"] for t in plan.tasks] == [s[2].value for s in STAGES]


def test_submitting_a_request_is_recorded_as_an_event(db, workspace):
    """§45: a project that appeared without a recorded cause cannot be audited."""
    plan = _plan(db, workspace)
    events = [
        e for e in db.execute(select(Event)).scalars()
        if e.type is EventType.TASK_CREATED and e.project_id == plan.project.id
    ]
    assert len(events) == 1
    assert events[0].payload["request"].startswith("Investigate the cause")
    assert events[0].payload["decomposer"] == "deterministic/1.0"
    assert events[0].payload["tasks"] == len(STAGES)


# --------------------------------------------------------------------------- decomposers


def test_the_default_decomposition_is_reproducible(db, workspace):
    """§46: if the plan varied run to run, no campaign over that plan could support a claim about
    what changed between arms — the plan itself would be a confound."""
    d = DeterministicDecomposer()
    assert d.focus(REQUEST) == d.focus(REQUEST)
    focus = d.focus(REQUEST)
    assert "intermittent" in focus and "corruption" in focus
    assert " the " not in f" {focus} ", "stop words survived into the focus"


def test_an_empty_request_still_yields_a_focus_rather_than_an_empty_project(db, workspace):
    assert DeterministicDecomposer().focus("the and of") == "the and of"


def test_a_model_written_decomposition_is_recorded_as_model_written(db, workspace):
    """§19: a model-backed decomposer sits behind the same interface, and the provenance says so
    — the plan is a set of model assertions about what the work is, not a fact about it."""
    from civitas.runtime.providers.offline import ScriptedProvider

    provider = ScriptedProvider(["ledger writer corruption"])
    plan = submit_request(
        db, workspace_id=workspace.id, request=REQUEST,
        decomposer=ModelDecomposer(provider=provider, model="scripted"),
    )
    db.commit()

    assert plan.decomposer == "model/1.0"
    assert plan.project.meta["decomposer"] == "model/1.0"
    assert "ledger writer corruption" in plan.tasks[0].title


# --------------------------------------------------------------------------- §32 survival


def test_a_projects_status_survives_the_loss_of_every_engine_and_session(
    db, settings, engine, backend
):
    """§32, stated exactly: *"Agent processes may die. Workers may restart. Colab sessions may
    terminate. Projects must survive."*

    So the test kills the process state. Every session is closed and every engine disposed, a new
    engine is built from the same settings, and the same status is asked for again. If any part of
    a project's state lived in a process, the two dictionaries would differ.
    """
    import uuid as _uuid

    from civitas.persistence.models import Organization, Workspace

    org = Organization(name="Survivor", slug=f"org-{_uuid.uuid4().hex[:8]}")
    db.add(org)
    db.flush()
    ws = Workspace(organization_id=org.id, name="W", slug=f"ws-{_uuid.uuid4().hex[:8]}",
                   environment_version="test-env-1")
    db.add(ws)
    db.commit()

    plan = submit_request(db, workspace_id=ws.id, request=REQUEST)
    profile = AgentProfile(workspace_id=ws.id, name="a")
    db.add(profile)
    db.flush()
    db.add(Episode(workspace_id=ws.id, project_id=plan.project.id, task_id=plan.tasks[0].id,
                   agent_profile_id=profile.id, model_provider="deterministic", model_name="d",
                   termination_reason=TerminationReason.EVALUATOR_SUCCESS,
                   tokens_used=1234, cost_usd=0.5))
    db.add(Artifact(workspace_id=ws.id, project_id=plan.project.id,
                    type=ArtifactType.OBSERVATION, title="o", body="b",
                    validation_state=ValidationState.TOOL_VERIFIED))
    complete_task(db, plan.tasks[0])
    db.commit()

    before = project_status(db, plan.project.id).as_dict()
    project_id = plan.project.id

    # Everything in this process goes away.
    db.rollback()
    db.close()

    fresh_engine = create_db_engine(settings)
    fresh_factory = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    install_guards(fresh_factory)
    fresh: Session = fresh_factory()
    try:
        after = project_status(fresh, project_id).as_dict()
    finally:
        fresh.close()
        fresh_engine.dispose()

    assert after == before
    assert before["tokens_used"] == 1234
    assert before["cost_usd"] == 0.5
    assert before["episodes_succeeded"] == 1
    assert before["validated_artifacts"] == 1
    assert before["tasks_completed"] == 1
    assert before["progress"] == 0.25


def test_status_reports_stage_by_stage_and_is_incomplete_until_every_stage_is(db, workspace):
    plan = _plan(db, workspace)
    status = project_status(db, plan.project.id)
    assert status.stages == {
        "investigate": "ready", "hypothesise": "blocked",
        "verify": "blocked", "synthesise": "blocked",
    }
    assert status.complete is False
    assert status.tasks_blocked == 3

    for task in plan.tasks:
        complete_task(db, task)
    db.commit()
    assert project_status(db, plan.project.id).complete is True


def test_an_unknown_project_is_an_error_not_an_empty_status(db, workspace):
    """A status of all zeros for a project that does not exist would be a fabricated measurement
    (ARCHITECTURE §3.9: absence is not zero)."""
    import uuid as _uuid

    with pytest.raises(ValueError):
        project_status(db, _uuid.uuid4())


# --------------------------------------------------------------------------- progression


def test_completing_a_stage_unblocks_exactly_the_next_one(db, workspace):
    plan = _plan(db, workspace)

    unblocked = complete_task(db, plan.tasks[0])
    db.commit()
    assert [t.id for t in unblocked] == [plan.tasks[1].id]
    assert plan.tasks[1].status is TaskStatus.READY
    assert plan.tasks[2].status is TaskStatus.BLOCKED

    nxt = next_ready_task(db, plan.project.id)
    assert nxt is not None and nxt.id == plan.tasks[1].id


def test_a_task_with_two_blockers_waits_for_both(db, workspace):
    """The unblocking rule is *all* dependencies complete, not *any*. A parallel investigation
    that unblocked synthesis after one branch finished would let the system synthesise over
    evidence it had not yet gathered."""
    plan = _plan(db, workspace)
    second = Task(workspace_id=workspace.id, project_id=plan.project.id,
                  objective_id=plan.objective.id, title="second investigation",
                  description="d", task_family="investigate", status=TaskStatus.READY)
    db.add(second)
    db.flush()
    db.add(TaskDependency(task_id=plan.tasks[1].id, depends_on_task_id=second.id))
    db.commit()

    assert complete_task(db, plan.tasks[0]) == []
    assert plan.tasks[1].status is TaskStatus.BLOCKED

    unblocked = complete_task(db, second)
    db.commit()
    assert [t.id for t in unblocked] == [plan.tasks[1].id]


def test_a_failed_stage_unblocks_nothing(db, workspace):
    """A failure that advanced the project would make every downstream stage rest on work that
    did not happen."""
    plan = _plan(db, workspace)
    assert complete_task(db, plan.tasks[0], succeeded=False) == []
    db.commit()
    assert plan.tasks[0].status is TaskStatus.FAILED
    assert plan.tasks[1].status is TaskStatus.BLOCKED
    assert next_ready_task(db, plan.project.id) is None


def test_resuming_is_not_a_special_code_path(db, workspace):
    """§32: `resume` is `next_ready_task` again. Calling it repeatedly without completing anything
    returns the same task, so a worker that died mid-episode picks up exactly where it left off
    rather than skipping the unit of work it was holding."""
    plan = _plan(db, workspace)
    first = next_ready_task(db, plan.project.id)
    again = next_ready_task(db, plan.project.id)
    assert first is not None and again is not None and first.id == again.id


# --------------------------------------------------------------------------- timeline


def test_the_timeline_is_ordered_and_carries_the_episode_that_produced_each_artifact(db, workspace):
    """§49's collective timeline: how the project's knowledge developed, with each entry linked to
    the episode that produced it so a reader can go from the claim to its provenance."""
    plan = _plan(db, workspace)
    profile = AgentProfile(workspace_id=workspace.id, name="a")
    db.add(profile)
    db.flush()
    episode = Episode(workspace_id=workspace.id, project_id=plan.project.id,
                      agent_profile_id=profile.id, model_provider="deterministic",
                      model_name="d", termination_reason=TerminationReason.EVALUATOR_SUCCESS)
    db.add(episode)
    db.flush()
    for i, kind in enumerate((ArtifactType.OBSERVATION, ArtifactType.HYPOTHESIS,
                              ArtifactType.CONCLUSION)):
        db.add(Artifact(workspace_id=workspace.id, project_id=plan.project.id, type=kind,
                        title=f"a{i}", body="b", creator_episode_id=episode.id))
        db.flush()
    db.commit()

    timeline = project_timeline(db, plan.project.id)
    assert [e["type"] for e in timeline] == ["observation", "hypothesis", "conclusion"]
    assert [e["at"] for e in timeline] == sorted(e["at"] for e in timeline)
    assert all(e["episode_id"] == str(episode.id) for e in timeline)


def test_the_timeline_of_an_unknown_project_is_empty_not_an_error(db, workspace):
    import uuid as _uuid

    assert project_timeline(db, _uuid.uuid4()) == []


def test_a_project_reports_its_open_questions(db, workspace):
    """§32: what is still unresolved is part of the project's state, not a footnote. A project
    reporting completion with open questions outstanding is reporting something false."""
    plan = _plan(db, workspace)
    db.add(Artifact(workspace_id=workspace.id, project_id=plan.project.id,
                    type=ArtifactType.QUESTION, title="why does it only happen under load?",
                    body="b"))
    db.add(Artifact(workspace_id=workspace.id, project_id=plan.project.id,
                    type=ArtifactType.UNRESOLVED_ISSUE, title="the retry path is untested",
                    body="b"))
    db.add(Artifact(workspace_id=workspace.id, project_id=plan.project.id,
                    type=ArtifactType.CONCLUSION, title="c", body="b"))
    db.commit()

    status = project_status(db, plan.project.id)
    assert status.open_questions == 2
    assert status.artifacts == 3


def test_two_projects_in_one_workspace_do_not_read_each_others_rows(db, workspace):
    """Every count in a status is scoped by project id. A status that summed the workspace would
    report another project's progress as this one's."""
    a = _plan(db, workspace, "investigate the lock contention in the scheduler")
    b = _plan(db, workspace, "investigate the memory growth in the indexer")

    complete_task(db, a.tasks[0])
    db.commit()

    assert project_status(db, a.project.id).tasks_completed == 1
    assert project_status(db, b.project.id).tasks_completed == 0
    assert project_status(db, b.project.id).tasks_total == len(STAGES)
    assert db.get(Project, b.project.id).name != db.get(Project, a.project.id).name
