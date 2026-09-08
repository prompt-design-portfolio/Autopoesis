"""Part B §4 — the foundational constraint, driven rather than asserted.

> Agents do not automatically inherit hidden episode state.

The test runs two consecutive episodes with a scripted provider. Episode 1 emits a canary into its
reasoning and its tool arguments. Episode 2 must see that canary in **no** input — under every arm,
including `collective`, where the artifact ecology is fully available.

Both halves matter. What episode 1 chose to externalise *does* reach episode 2, and the test
asserts that too: an isolation test that passed by making the collective useless would be testing
the wrong thing.
"""

from __future__ import annotations

import pytest

from civitas.domain.enums import ExperimentArm, TerminationReason
from civitas.persistence.models import AgentProfile, Episode, ModelCall, Task
from civitas.runtime.budgets import Budgets
from civitas.runtime.episode import EpisodeRunner, EpisodeSpec
from civitas.runtime.providers.base import Completion, Role, ToolCall
from civitas.runtime.providers.offline import ScriptedProvider
from civitas.runtime.tools.builtin import default_registry

CANARY = "CANARY-b3f1a9-private-scratchpad-do-not-inherit"


@pytest.fixture
def profile(db, workspace):
    p = AgentProfile(workspace_id=workspace.id, name="explorer")
    db.add(p)
    db.commit()
    return p


@pytest.fixture
def task(db, workspace):
    t = Task(
        workspace_id=workspace.id,
        title="Find the cause of intermittent corruption",
        description="Investigate and report the subsystem responsible.",
        task_family="corruption",
    )
    db.add(t)
    db.commit()
    return t


def _spec(workspace, profile, task, arm=ExperimentArm.COLLECTIVE, **kw):
    return EpisodeSpec(
        workspace_id=workspace.id,
        agent_profile_id=profile.id,
        provider_name="scripted",
        model_name="scripted-v1",
        system_prompt="You are a bounded research agent.",
        task_id=task.id,
        experiment_arm=arm,
        environment_version="test-env-1",
        budgets=Budgets(tokens=50_000, tool_calls=10, max_turns=8, wall_clock_s=60),
        **kw,
    )


def _capture_inputs(provider: ScriptedProvider) -> str:
    """Everything the *environment* put in front of the model, across every turn.

    System, user and tool-result messages only. Assistant messages are the model's own output
    echoed back within one episode — that is within-life state, which Part B §4 permits and in
    fact requires; counting it would make the test assert that an agent cannot remember its own
    previous sentence.
    """
    return "\n".join(
        m.content
        for request in provider.calls
        for m in request.messages
        if m.role in (Role.SYSTEM, Role.USER, Role.TOOL)
    )


@pytest.mark.parametrize("arm", list(ExperimentArm))
def test_no_episode_local_state_reaches_the_next_episode(db, workspace, profile, task, arm):
    """The canary must not appear in any input to episode 2, under any arm."""
    if arm is ExperimentArm.COLLECTIVE_FROZEN:
        pytest.skip("frozen needs a snapshot cut; covered by test_frozen_arm_requires_a_cut")

    # The canary lives ONLY in episode 1's reasoning — never in a tool argument. That distinction
    # is the whole point of §4: an agent that deliberately writes something into an artifact has
    # externalised it, and it is *supposed* to reach later agents. What must not survive is what
    # the agent merely thought.
    first = ScriptedProvider([
        Completion(
            text=f"My private reasoning: {CANARY}. I will record a finding.",
            tool_calls=(ToolCall("c1", "create_artifact", {
                "type": "observation",
                "title": "The write path holds a lock across a retry",
                "body": "Observed in the writer: the lock is held across the retry.",
            }),),
        ),
        Completion(
            text="Submitting.",
            tool_calls=(ToolCall("c2", "submit_result", {"answer": "the write path"}),),
        ),
    ])
    runner = EpisodeRunner(db, provider=first, tools=default_registry())
    out1 = runner.run(_spec(workspace, profile, task, arm))
    db.commit()
    assert out1.termination_reason is TerminationReason.SOLVED

    second = ScriptedProvider([
        Completion(text="Searching.", tool_calls=(
            ToolCall("s1", "search_knowledge", {"query": "write path lock retry corruption"}),
        )),
        Completion(text="Submitting.", tool_calls=(
            ToolCall("s2", "submit_result", {"answer": "the write path"}),
        )),
    ])
    EpisodeRunner(db, provider=second, tools=default_registry()).run(
        _spec(workspace, profile, task, arm)
    )
    db.commit()

    seen = _capture_inputs(second)
    assert CANARY not in seen, (
        f"episode-local state leaked into episode 2 under arm {arm.value}: "
        "Part B §4 forbids hidden state inheritance"
    )


def test_what_was_externalised_does_reach_the_next_episode(db, workspace, profile, task):
    """The other half: isolation must not be achieved by making the collective useless.

    Under `collective`, the *artifact* episode 1 wrote is retrievable by episode 2 — that is the
    entire mechanism the platform exists to provide.
    """
    first = ScriptedProvider([
        Completion(text="Recording.", tool_calls=(ToolCall("c1", "create_artifact", {
            "type": "observation",
            "title": "Distinctive marker phrase quicksilver cascade",
            "body": "The writer holds a lock across a retry.",
        }),)),
        Completion(text="Done.", tool_calls=(
            ToolCall("c2", "submit_result", {"answer": "writer"}),
        )),
    ])
    EpisodeRunner(db, provider=first, tools=default_registry()).run(
        _spec(workspace, profile, task, ExperimentArm.COLLECTIVE)
    )
    db.commit()

    second = ScriptedProvider([
        Completion(text="Searching.", tool_calls=(
            ToolCall("s1", "search_knowledge", {"query": "quicksilver cascade"}),
        )),
        Completion(text="Done.", tool_calls=(
            ToolCall("s2", "submit_result", {"answer": "writer"}),
        )),
    ])
    EpisodeRunner(db, provider=second, tools=default_registry()).run(
        _spec(workspace, profile, task, ExperimentArm.COLLECTIVE)
    )
    db.commit()

    seen = _capture_inputs(second)
    assert "quicksilver cascade" in seen, (
        "episode 2 could not retrieve what episode 1 externalised — the collective is inert"
    )


def test_a_blind_arm_sees_nothing_from_earlier_episodes(db, workspace, profile, task):
    """`solo` is the negative control for the test above (Part B §21)."""
    first = ScriptedProvider([
        Completion(text="Recording.", tool_calls=(ToolCall("c1", "create_artifact", {
            "type": "observation",
            "title": "Distinctive marker phrase quicksilver cascade",
            "body": "The writer holds a lock across a retry.",
        }),)),
        Completion(text="Done.", tool_calls=(
            ToolCall("c2", "submit_result", {"answer": "writer"}),
        )),
    ])
    EpisodeRunner(db, provider=first, tools=default_registry()).run(
        _spec(workspace, profile, task, ExperimentArm.COLLECTIVE)
    )
    db.commit()

    solo = ScriptedProvider([
        Completion(text="Searching.", tool_calls=(
            ToolCall("s1", "search_knowledge", {"query": "quicksilver cascade"}),
        )),
        Completion(text="Done.", tool_calls=(
            ToolCall("s2", "submit_result", {"answer": "?"}),
        )),
    ])
    EpisodeRunner(db, provider=solo, tools=default_registry()).run(
        _spec(workspace, profile, task, ExperimentArm.SOLO)
    )
    db.commit()

    assert "quicksilver cascade" not in _capture_inputs(solo), (
        "the solo arm returned collective knowledge; arms are machine-enforced (§21)"
    )


def test_prompts_are_not_persisted(db, workspace, profile, task):
    """No table stores an episode's reasoning.

    A persisted transcript would be a channel through which the next episode could inherit hidden
    state — by a UI that renders it, a retrieval that indexes it, or a debugging tool that
    replays it. `ModelCall` records the accounting and a request *hash*, never the content.
    """
    provider = ScriptedProvider([
        Completion(text=f"reasoning {CANARY}", tool_calls=(
            ToolCall("c1", "submit_result", {"answer": "x"}),
        )),
    ])
    EpisodeRunner(db, provider=provider, tools=default_registry()).run(
        _spec(workspace, profile, task)
    )
    db.commit()

    for call in db.query(ModelCall).all():
        blob = " ".join(str(getattr(call, c.name)) for c in ModelCall.__table__.columns)
        assert CANARY not in blob, "a model call persisted prompt content"

    for episode in db.query(Episode).all():
        blob = " ".join(str(getattr(episode, c.name)) for c in Episode.__table__.columns)
        assert CANARY not in blob, "an episode row persisted reasoning"


def test_frozen_arm_requires_a_cut(db, workspace, profile, task):
    """`collective_frozen` without a snapshot is indistinguishable from `collective`.

    It must fail loudly rather than degrade into an unablated arm that would then be reported as
    a control (Part B §21).
    """
    provider = ScriptedProvider([Completion(text="hi")])
    with pytest.raises(ValueError, match="snapshot cut"):
        EpisodeRunner(db, provider=provider, tools=default_registry()).run(
            _spec(workspace, profile, task, ExperimentArm.COLLECTIVE_FROZEN)
        )
