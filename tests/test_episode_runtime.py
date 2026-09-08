"""Bounded episodes (Part B §8) — every termination reason is reachable, and budgets bind."""

from __future__ import annotations

import pytest

from civitas.domain.enums import EpisodeStatus, EventType, ExperimentArm, TerminationReason
from civitas.persistence.models import AgentProfile, Artifact, ArtifactUsage, Episode, Event, Task
from civitas.runtime.budgets import BudgetExceeded, Budgets, BudgetTracker
from civitas.runtime.episode import EpisodeRunner, EpisodeSpec
from civitas.runtime.providers.base import Completion, ToolCall
from civitas.runtime.providers.offline import (
    DeterministicProvider,
    FailingProvider,
    ScriptedProvider,
)
from civitas.runtime.tools.builtin import default_registry


@pytest.fixture
def profile(db, workspace):
    p = AgentProfile(workspace_id=workspace.id, name="explorer")
    db.add(p)
    db.commit()
    return p


@pytest.fixture
def task(db, workspace):
    t = Task(workspace_id=workspace.id, title="Investigate", description="Find the cause.")
    db.add(t)
    db.commit()
    return t


def _spec(workspace, profile, task, budgets=None, **kw):
    return EpisodeSpec(
        workspace_id=workspace.id,
        agent_profile_id=profile.id,
        provider_name=kw.pop("provider_name", "scripted"),
        model_name=kw.pop("model_name", "scripted-v1"),
        system_prompt="Bounded research agent.",
        task_id=task.id,
        experiment_arm=kw.pop("arm", ExperimentArm.COLLECTIVE),
        environment_version="test-env-1",
        budgets=budgets or Budgets(tokens=50_000, tool_calls=10, max_turns=8, wall_clock_s=60),
        **kw,
    )


def test_a_solved_episode_records_everything_section_8_requires(db, workspace, profile, task):
    provider = ScriptedProvider([
        Completion(text="Recording.", tool_calls=(ToolCall("c1", "create_artifact", {
            "type": "observation", "title": "Lock held across retry", "body": "detail",
        }),)),
        Completion(text="Done.", tool_calls=(
            ToolCall("c2", "submit_result", {"answer": "the writer"}),
        )),
    ])
    out = EpisodeRunner(db, provider=provider, tools=default_registry()).run(
        _spec(workspace, profile, task)
    )
    db.commit()

    episode = db.get(Episode, out.episode_id)
    assert episode.status is EpisodeStatus.TERMINATED
    assert episode.termination_reason is TerminationReason.SOLVED
    assert episode.started_at is not None and episode.ended_at is not None
    assert episode.duration_s is not None and episode.duration_s >= 0
    assert episode.tokens_used > 0
    assert episode.tool_calls_used == 2
    assert episode.artifacts_created == 2  # the observation and the submitted result
    assert episode.model_calls == 2
    assert out.submitted_answer == "the writer"


def test_budget_exhaustion_terminates_rather_than_overrunning(db, workspace, profile, task):
    """§8: no agent may run indefinitely. The bound is checked before the action, so the
    episode stops rather than being noticed afterwards."""
    provider = ScriptedProvider(
        [Completion(text="thinking, no tool call")], loop=True
    )
    out = EpisodeRunner(db, provider=provider, tools=default_registry()).run(
        _spec(workspace, profile, task,
              Budgets(tokens=50_000, tool_calls=10, max_turns=3, max_no_progress_turns=99))
    )
    db.commit()
    assert out.termination_reason is TerminationReason.BUDGET_EXHAUSTED
    assert "max_turns" in out.detail
    assert out.turns <= 3, "the bound must not be exceeded, only reached"


def test_token_budget_refuses_a_call_that_could_exceed_it(db, workspace, profile, task):
    provider = ScriptedProvider([Completion(text="x")], loop=True)
    out = EpisodeRunner(db, provider=provider, tools=default_registry()).run(
        _spec(workspace, profile, task, Budgets(tokens=10, tool_calls=5, max_turns=20))
    )
    db.commit()
    assert out.termination_reason is TerminationReason.BUDGET_EXHAUSTED
    assert "tokens" in out.detail
    assert out.tokens_used <= 10


def test_tool_call_budget_binds(db, workspace, profile, task):
    provider = ScriptedProvider([
        Completion(text="search", tool_calls=(
            ToolCall("c", "search_knowledge", {"query": "anything"}),
        )),
    ], loop=True)
    out = EpisodeRunner(db, provider=provider, tools=default_registry()).run(
        _spec(workspace, profile, task,
              Budgets(tokens=500_000, tool_calls=3, max_turns=50, max_no_progress_turns=99))
    )
    db.commit()
    assert out.termination_reason is TerminationReason.BUDGET_EXHAUSTED
    assert out.tool_calls == 3


def test_no_progress_is_distinguished_from_budget_exhaustion(db, workspace, profile, task):
    """A stalled agent must terminate for a diagnostic reason, not by burning its budget."""
    provider = ScriptedProvider([Completion(text="I am thinking about it.")], loop=True)
    out = EpisodeRunner(db, provider=provider, tools=default_registry()).run(
        _spec(workspace, profile, task,
              Budgets(tokens=500_000, tool_calls=20, max_turns=50, max_no_progress_turns=3))
    )
    db.commit()
    assert out.termination_reason is TerminationReason.NO_PROGRESS
    assert out.turns < 50, "it must stop on the stall, not on the turn budget"


def test_a_provider_failure_is_recorded_as_such(db, workspace, profile, task):
    out = EpisodeRunner(db, provider=FailingProvider(), tools=default_registry()).run(
        _spec(workspace, profile, task, provider_name="failing", model_name="broken")
    )
    db.commit()
    episode = db.get(Episode, out.episode_id)
    assert episode.termination_reason is TerminationReason.PROVIDER_FAILURE
    assert "synthetic provider failure" in episode.termination_detail
    # The failed call is still on the record: a provider failure that leaves no ModelCall row is
    # invisible to the reliability metrics of §53.
    from civitas.persistence.models import ModelCall
    failed = db.query(ModelCall).filter(ModelCall.episode_id == episode.id).one()
    assert failed.error


def test_a_tool_raising_does_not_kill_the_episode(db, workspace, profile, task):
    """A tool is allowed to fail. The episode must survive and be told."""
    from civitas.runtime.tools.base import Tool

    class Exploding(Tool):
        name = "explode"
        description = "always raises"
        parameters = {"type": "object", "properties": {}}

        def run(self, ctx, **kw):
            raise RuntimeError("boom")

    registry = default_registry()
    registry.add(Exploding())
    provider = ScriptedProvider([
        Completion(text="try", tool_calls=(ToolCall("c1", "explode", {}),)),
        Completion(text="give up", tool_calls=(
            ToolCall("c2", "submit_result", {"answer": "could not"}),
        )),
    ])
    out = EpisodeRunner(db, provider=provider, tools=registry).run(
        _spec(workspace, profile, task)
    )
    db.commit()
    assert out.termination_reason is TerminationReason.SOLVED
    tool_message = [
        m for r in provider.calls for m in r.messages if m.role.value == "tool"
    ]
    assert any("boom" in m.content for m in tool_message)


def test_unknown_tool_and_bad_arguments_are_reported_not_raised(db, workspace, profile, task):
    provider = ScriptedProvider([
        Completion(text="a", tool_calls=(ToolCall("c1", "no_such_tool", {}),)),
        Completion(text="b", tool_calls=(
            ToolCall("c2", "create_artifact", {"type": "observation"}),
        )),
        Completion(text="c", tool_calls=(ToolCall("c3", "submit_result", {"answer": "x"}),)),
    ])
    out = EpisodeRunner(db, provider=provider, tools=default_registry()).run(
        _spec(workspace, profile, task)
    )
    db.commit()
    messages = [m.content for r in provider.calls for m in r.messages if m.role.value == "tool"]
    assert any("no tool named" in m for m in messages)
    assert any("missing required argument" in m for m in messages)
    assert out.termination_reason is TerminationReason.SOLVED


def test_the_episode_emits_the_events_section_45_requires(db, workspace, profile, task):
    provider = ScriptedProvider([
        Completion(text="search", tool_calls=(
            ToolCall("c1", "search_knowledge", {"query": "cause"}),
        )),
        Completion(text="done", tool_calls=(ToolCall("c2", "submit_result", {"answer": "x"}),)),
    ])
    out = EpisodeRunner(db, provider=provider, tools=default_registry()).run(
        _spec(workspace, profile, task)
    )
    db.commit()

    kinds = {
        e.type for e in db.query(Event).filter(Event.episode_id == out.episode_id).all()
    }
    for required in (
        EventType.EPISODE_STARTED, EventType.EPISODE_TERMINATED,
        EventType.MODEL_CALLED, EventType.TOOL_CALLED,
        EventType.RETRIEVAL_PERFORMED, EventType.ARTIFACT_CREATED,
    ):
        assert required in kinds, f"{required.value} was not emitted"


def test_reading_an_artifact_creates_a_usage_link_but_retrieval_alone_does_not(
    db, workspace, profile, task
):
    """Part A §A2.1: credit flows along what was *read*, not what was returned."""
    from civitas.domain.enums import ArtifactType

    a = Artifact(workspace_id=workspace.id, type=ArtifactType.OBSERVATION,
                 title="lock retry writer", body="detail")
    db.add(a)
    db.commit()

    searched = ScriptedProvider([
        Completion(text="s", tool_calls=(
            ToolCall("c1", "search_knowledge", {"query": "lock retry writer"}),
        )),
        Completion(text="d", tool_calls=(ToolCall("c2", "submit_result", {"answer": "x"}),)),
    ])
    out1 = EpisodeRunner(db, provider=searched, tools=default_registry()).run(
        _spec(workspace, profile, task)
    )
    db.commit()
    assert db.query(ArtifactUsage).filter(ArtifactUsage.episode_id == out1.episode_id).count() == 0
    db.refresh(a)
    assert a.times_retrieved >= 1, "retrieval must still be counted"
    assert a.times_read == 0, "a returned artifact was not read"

    read = ScriptedProvider([
        Completion(text="r", tool_calls=(
            ToolCall("c1", "read_artifact", {"artifact_id": str(a.id)}),
        )),
        Completion(text="d", tool_calls=(ToolCall("c2", "submit_result", {"answer": "x"}),)),
    ])
    out2 = EpisodeRunner(db, provider=read, tools=default_registry()).run(
        _spec(workspace, profile, task)
    )
    db.commit()
    usage = db.query(ArtifactUsage).filter(ArtifactUsage.episode_id == out2.episode_id).one()
    assert usage.artifact_id == a.id
    db.refresh(a)
    assert a.times_read == 1


def test_an_agent_cannot_certify_its_own_result(db, workspace, profile, task):
    """Part B §47. Whatever the agent claims, the artifact is self-reported until an evaluator
    says otherwise."""
    from civitas.domain.enums import ArtifactType, ValidationState

    provider = ScriptedProvider([
        Completion(text="done", tool_calls=(ToolCall("c1", "submit_result", {
            "answer": "the writer", "confidence": 1.0,
        }),)),
    ])
    out = EpisodeRunner(db, provider=provider, tools=default_registry()).run(
        _spec(workspace, profile, task)
    )
    db.commit()
    result = db.get(Artifact, out.submitted_artifact_id)
    assert result.type is ArtifactType.RESULT
    assert result.validation_state is ValidationState.SELF_REPORTED
    assert result.evidence_kind.value == "model_assertion"


def test_the_agent_is_never_shown_the_evaluator_specification(db, workspace, profile, task):
    """§47: an agent that can read its success criteria can write to them."""
    task.evaluator_spec = {"expected_answer": "SECRET-EXPECTED-ANSWER-42"}
    db.commit()

    provider = ScriptedProvider([
        Completion(text="d", tool_calls=(ToolCall("c1", "submit_result", {"answer": "guess"}),)),
    ])
    EpisodeRunner(db, provider=provider, tools=default_registry()).run(
        _spec(workspace, profile, task)
    )
    db.commit()
    shown = "\n".join(m.content for r in provider.calls for m in r.messages)
    assert "SECRET-EXPECTED-ANSWER-42" not in shown


def test_every_termination_reason_is_a_declared_value():
    """A reason not in the §8 enumeration cannot be recorded, so the set stays closed."""
    assert {r.value for r in TerminationReason} >= {
        "solved", "evaluator_success", "evaluator_failure", "budget_exhausted", "no_progress",
        "blocked", "delegated", "failed", "provider_failure", "tool_failure", "worker_failure",
        "policy_violation", "cancelled",
    }


def test_the_deterministic_provider_is_exactly_reproducible(db, workspace, profile, task):
    """Part B §20: frozen-model mode needs the individual agent held exactly constant."""
    provider = DeterministicProvider()
    spec = _spec(workspace, profile, task, provider_name="deterministic",
                 model_name="deterministic-v1",
                 budgets=Budgets(tokens=20_000, tool_calls=4, max_turns=3))
    runs = []
    for _ in range(2):
        out = EpisodeRunner(db, provider=provider, tools=default_registry()).run(spec)
        db.commit()
        runs.append((out.termination_reason, out.turns, out.tool_calls))
    assert runs[0] == runs[1], "the same configuration produced a different episode"


def test_budget_tracker_checks_before_the_action(monkeypatch):
    clock = {"t": 0.0}
    tracker = BudgetTracker(Budgets(wall_clock_s=10.0), clock=lambda: clock["t"])
    tracker.check_wall_clock()
    clock["t"] = 11.0
    with pytest.raises(BudgetExceeded) as exc:
        tracker.check_wall_clock()
    assert exc.value.kind == "wall_clock_s"
    assert exc.value.termination_reason is TerminationReason.BUDGET_EXHAUSTED


def test_cost_budget_is_rechecked_after_a_call_exceeds_its_estimate():
    """A response can cost more than predicted. The pre-check passing does not license the
    overspend to continue."""
    tracker = BudgetTracker(Budgets(cost_usd=1.0))
    with pytest.raises(BudgetExceeded) as exc:
        tracker.record_model_call(prompt_tokens=10, completion_tokens=10, cost_usd=5.0)
    assert exc.value.kind == "cost_usd"


def test_a_blocked_duplicate_counts_as_progress(db, workspace, profile, task):
    """Part A §A2.3 must not terminate the episode it is helping.

    A blocked duplicate is information the episode did not have, so it resets the no-progress
    counter. Without this, an agent successfully avoiding known-failed work accumulates
    no-progress turns and dies with budget unspent — the mechanism killing what it was built to
    assist.
    """
    from civitas.domain.enums import ArtifactType
    from civitas.knowledge.duplicate import record_failure
    from civitas.persistence.models import Artifact

    prior = Artifact(workspace_id=workspace.id, type=ArtifactType.FAILURE,
                     title="already tried", body="it failed",
                     environment_version="test-env-1")
    db.add(prior)
    db.flush()
    record_failure(
        db, workspace_id=workspace.id, artifact_id=prior.id, episode_id=None,
        environment_version="test-env-1", summary="that exact call failed",
        tool_name="create_artifact",
        tool_args={"type": "observation", "title": "repeat me", "body": "same body"},
        reproducible=True,
    )
    db.commit()

    repeat = ToolCall("c1", "create_artifact",
                      {"type": "observation", "title": "repeat me", "body": "same body"})
    provider = ScriptedProvider([
        Completion(text="try", tool_calls=(repeat,)),
        Completion(text="try again", tool_calls=(repeat,)),
        Completion(text="and again", tool_calls=(repeat,)),
        Completion(text="and again", tool_calls=(repeat,)),
        Completion(text="done", tool_calls=(ToolCall("c9", "submit_result", {"answer": "x"}),)),
    ])
    out = EpisodeRunner(db, provider=provider, tools=default_registry()).run(
        _spec(workspace, profile, task,
              Budgets(tokens=500_000, tool_calls=10, max_turns=12, max_no_progress_turns=3))
    )
    db.commit()

    assert out.termination_reason is not TerminationReason.NO_PROGRESS, (
        "blocked duplicates were counted as a stall"
    )
    assert out.duplicate_failures >= 3, "the repeats must still be detected and counted"
