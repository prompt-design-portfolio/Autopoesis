"""The tool ecology (Part B §17, §18) — the civilization accumulating technology."""

from __future__ import annotations

import pytest

from civitas.domain.enums import (
    ArtifactType,
    EvidenceKind,
    ExperimentArm,
    ToolValidationState,
    ValidationState,
)
from civitas.persistence.models import AgentProfile, Artifact, ToolDefinition, ToolRun, ToolVersion
from civitas.runtime.budgets import Budgets
from civitas.runtime.episode import EpisodeRunner, EpisodeSpec
from civitas.runtime.providers.base import Completion, ToolCall
from civitas.runtime.providers.offline import ScriptedProvider
from civitas.runtime.tools.ecology import (
    discoverable_tools,
    register_tool,
    run_registered_tool,
    toolmaker_registry,
    validate_source,
)

GOOD_SOURCE = """
def main(numbers):
    return sum(int(n) for n in numbers)
"""
GOOD_TESTS = """
def run_tests():
    assert main([1, 2, 3]) == 6
    assert main([]) == 0
    return "ok"
"""
BROKEN_TESTS = """
def run_tests():
    assert main([1, 2, 3]) == 7, "deliberately wrong"
"""


@pytest.fixture
def profile(db, workspace):
    p = AgentProfile(workspace_id=workspace.id, name="toolmaker")
    db.add(p)
    db.commit()
    return p


# --------------------------------------------------------------------------
# validation runs in the sandbox (§17, §18)
# --------------------------------------------------------------------------
def test_a_tool_whose_tests_pass_is_validated():
    outcome = validate_source(GOOD_SOURCE, test_source=GOOD_TESTS)
    assert outcome.passed
    assert outcome.state is ToolValidationState.TESTS_PASSING
    assert outcome.report["isolation_level"] in ("process", "container", "kernel")


def test_a_tool_whose_tests_fail_is_marked_failing_not_validated():
    outcome = validate_source(GOOD_SOURCE, test_source=BROKEN_TESTS)
    assert not outcome.passed
    assert outcome.state is ToolValidationState.TESTS_FAILING
    assert "deliberately wrong" in outcome.report["stderr"]


def test_untested_is_distinguished_from_failing():
    """Conflating them would let an untested tool inherit a tested one's reputation."""
    assert validate_source(GOOD_SOURCE, test_source="").state is ToolValidationState.UNTESTED
    assert (
        validate_source(GOOD_SOURCE, test_source=BROKEN_TESTS).state
        is ToolValidationState.TESTS_FAILING
    )


def test_source_that_will_not_compile_is_rejected_before_the_sandbox():
    outcome = validate_source("def main(:\n  pass", test_source=GOOD_TESTS)
    assert outcome.state is ToolValidationState.TESTS_FAILING
    assert outcome.report["stage"] == "compile"


def test_tool_validation_is_sandboxed(db, workspace):
    """Agent-written source must never execute on the host (§18)."""
    escaping = """
import socket
def main():
    return socket.create_connection(("example.com", 80))
"""
    tests = """
def run_tests():
    main()
    return "reached the network"
"""
    outcome = validate_source(escaping, test_source=tests)
    assert not outcome.passed
    assert "network access is denied" in outcome.report["stderr"]


# --------------------------------------------------------------------------
# registration and versioning (§17)
# --------------------------------------------------------------------------
def test_registering_a_tool_creates_a_discoverable_artifact(db, workspace):
    definition, version, outcome = register_tool(
        db, workspace_id=workspace.id, name="sum_numbers",
        description="Sum a list of numbers.", source=GOOD_SOURCE,
        test_source=GOOD_TESTS, parameters_schema={"type": "object"},
    )
    db.commit()

    assert outcome.passed
    assert definition.current_version_id == version.id
    artifact = db.get(Artifact, definition.artifact_id)
    assert artifact.type is ArtifactType.TOOL
    assert artifact.evidence_kind is EvidenceKind.TOOL_OUTPUT
    assert artifact.validation_state is ValidationState.TOOL_VERIFIED


def test_a_failing_tool_is_kept_and_stays_discoverable(db, workspace):
    """§12: 'someone already tried this and it did not work' is what a later toolmaker needs."""
    definition, _version, outcome = register_tool(
        db, workspace_id=workspace.id, name="broken", description="Does not work.",
        source=GOOD_SOURCE, test_source=BROKEN_TESTS, parameters_schema={},
    )
    db.commit()

    assert not outcome.passed
    assert db.get(ToolDefinition, definition.id) is not None
    names = [d.name for d, _v in discoverable_tools(db, workspace_id=workspace.id)]
    assert "broken" in names

    artifact = db.get(Artifact, definition.artifact_id)
    assert artifact.evidence_kind is EvidenceKind.MODEL_ASSERTION, (
        "a tool that fails its tests must not carry tool-output provenance"
    )


def test_a_second_registration_adds_a_version_rather_than_replacing(db, workspace):
    """A tool improving across generations must not break every earlier reference."""
    definition, first, _ = register_tool(
        db, workspace_id=workspace.id, name="sum_numbers", description="v1",
        source=GOOD_SOURCE, test_source=GOOD_TESTS, parameters_schema={},
    )
    db.commit()
    _definition, second, _ = register_tool(
        db, workspace_id=workspace.id, name="sum_numbers", description="v2",
        source=GOOD_SOURCE + "\n# improved\n", test_source=GOOD_TESTS, parameters_schema={},
    )
    db.commit()

    assert second.version == 2
    versions = db.query(ToolVersion).filter(
        ToolVersion.tool_definition_id == definition.id
    ).all()
    assert len(versions) == 2
    assert db.get(ToolVersion, first.id) is not None


def test_a_failing_new_version_does_not_become_current(db, workspace):
    """A bad edit must degrade the collective's technology by nothing."""
    definition, good, _ = register_tool(
        db, workspace_id=workspace.id, name="sum_numbers", description="works",
        source=GOOD_SOURCE, test_source=GOOD_TESTS, parameters_schema={},
    )
    db.commit()
    register_tool(
        db, workspace_id=workspace.id, name="sum_numbers", description="regressed",
        source=GOOD_SOURCE, test_source=BROKEN_TESTS, parameters_schema={},
    )
    db.commit()
    db.refresh(definition)
    assert definition.current_version_id == good.id


# --------------------------------------------------------------------------
# reuse (§17)
# --------------------------------------------------------------------------
def test_a_registered_tool_can_be_run_and_the_run_is_recorded(db, workspace):
    definition, _v, _o = register_tool(
        db, workspace_id=workspace.id, name="sum_numbers", description="Sum numbers.",
        source=GOOD_SOURCE, test_source=GOOD_TESTS, parameters_schema={},
    )
    db.commit()

    run = run_registered_tool(
        db, definition=definition, args={"numbers": [4, 5, 6]},
        episode_id=None, workspace_id=workspace.id,
    )
    db.commit()

    assert run.succeeded
    assert run.meta["returned"] == 15
    db.refresh(definition)
    assert definition.times_used == 1 and definition.times_succeeded == 1
    assert definition.success_rate == pytest.approx(1.0)


def test_success_rate_is_none_for_an_unused_tool(db, workspace):
    """A fresh tool is not a failing one, and collapsing the two would rank it below a broken
    one."""
    definition, _v, _o = register_tool(
        db, workspace_id=workspace.id, name="unused", description="never run",
        source=GOOD_SOURCE, test_source=GOOD_TESTS, parameters_schema={},
    )
    db.commit()
    assert definition.success_rate is None


def test_a_tool_version_cannot_widen_its_own_sandbox(db, workspace):
    """§18: the sandbox must not grant privileges on the word of the code it is sandboxing."""
    definition, _v, _o = register_tool(
        db, workspace_id=workspace.id, name="wants_network",
        description="asks for the network",
        source="import socket\ndef main():\n    socket.create_connection(('example.com', 80))\n",
        test_source="", parameters_schema={},
        permissions={"network": True, "memory_mb": 99999},
    )
    db.commit()

    run = run_registered_tool(
        db, definition=definition, args={}, episode_id=None, workspace_id=workspace.id,
    )
    db.commit()
    assert not run.succeeded, "a tool granted itself network access"


def test_discovery_ranks_working_tools_above_failing_ones(db, workspace):
    register_tool(db, workspace_id=workspace.id, name="broken", description="b",
                  source=GOOD_SOURCE, test_source=BROKEN_TESTS, parameters_schema={})
    register_tool(db, workspace_id=workspace.id, name="works", description="w",
                  source=GOOD_SOURCE, test_source=GOOD_TESTS, parameters_schema={})
    db.commit()

    names = [d.name for d, _v in discoverable_tools(db, workspace_id=workspace.id)]
    assert names.index("works") < names.index("broken")


def test_tool_utility_is_never_self_reported(db, workspace, profile):
    """Part A §A2.1: `downstream_utility` is written only by credit assignment."""
    from civitas.experiments.credit import assign_credit
    from civitas.persistence.models import Episode, Evaluation

    definition, _v, _o = register_tool(
        db, workspace_id=workspace.id, name="useful", description="u",
        source=GOOD_SOURCE, test_source=GOOD_TESTS, parameters_schema={},
    )
    db.commit()
    assert definition.downstream_utility == 0.0

    episode = Episode(workspace_id=workspace.id, agent_profile_id=profile.id,
                      model_provider="deterministic", model_name="d")
    db.add(episode)
    db.flush()
    run_registered_tool(db, definition=definition, args={"numbers": [1]},
                        episode_id=episode.id, workspace_id=workspace.id)
    evaluation = Evaluation(workspace_id=workspace.id, episode_id=episode.id, scope="episode",
                            evaluator_kind="exact_match", succeeded=True, score=1.0)
    db.add(evaluation)
    db.flush()
    assign_credit(db, evaluation)
    db.commit()

    db.refresh(definition)
    assert definition.downstream_utility > 0


# --------------------------------------------------------------------------
# end to end: one agent builds a tool, a later agent reuses it (§17)
# --------------------------------------------------------------------------
def _spec(workspace, profile, **kw):
    return EpisodeSpec(
        workspace_id=workspace.id, agent_profile_id=profile.id,
        provider_name="scripted", model_name="scripted-v1",
        system_prompt="You are a bounded toolmaking agent.",
        experiment_arm=ExperimentArm.COLLECTIVE, environment_version="test-env-1",
        budgets=Budgets(tokens=200_000, tool_calls=8, max_turns=10, wall_clock_s=120),
        **kw,
    )


def test_one_agent_builds_a_tool_and_a_later_agent_reuses_it(db, workspace, profile):
    """The claim of §17, driven end to end: technology accumulates across episodes."""
    builder = ScriptedProvider([
        Completion(text="building", tool_calls=(ToolCall("c1", "create_tool", {
            "name": "sum_numbers",
            "description": "Sum a list of numbers.",
            "source": GOOD_SOURCE,
            "test_source": GOOD_TESTS,
            "parameters_schema": {"type": "object",
                                  "properties": {"numbers": {"type": "array"}}},
        }),)),
        Completion(text="done", tool_calls=(
            ToolCall("c2", "submit_result", {"answer": "built sum_numbers"}),
        )),
    ])
    EpisodeRunner(db, provider=builder, tools=toolmaker_registry()).run(
        _spec(workspace, profile)
    )
    db.commit()

    reuser = ScriptedProvider([
        Completion(text="looking", tool_calls=(ToolCall("c1", "list_tools", {}),)),
        Completion(text="using", tool_calls=(ToolCall("c2", "run_tool", {
            "name": "sum_numbers", "args": {"numbers": [10, 20, 12]},
        }),)),
        Completion(text="done", tool_calls=(
            ToolCall("c3", "submit_result", {"answer": "42"}),
        )),
    ])
    outcome = EpisodeRunner(db, provider=reuser, tools=toolmaker_registry()).run(
        _spec(workspace, profile)
    )
    db.commit()

    shown = "\n".join(
        m.content for r in reuser.calls for m in r.messages if m.role.value == "tool"
    )
    assert "sum_numbers" in shown, "the later agent could not discover the tool"
    assert "42" in shown, "the later agent could not run the tool it discovered"
    assert outcome.submitted_answer == "42"

    definition = db.query(ToolDefinition).filter(ToolDefinition.name == "sum_numbers").one()
    assert definition.times_used == 1 and definition.times_succeeded == 1
    runs = db.query(ToolRun).filter(ToolRun.tool_definition_id == definition.id).all()
    assert len(runs) == 1 and runs[0].episode_id == outcome.episode_id


# --------------------------------------------------------------------------
# the tool-accumulation benchmark (§17, §24, §48)
# --------------------------------------------------------------------------
def test_discovery_avoids_rebuilds_and_the_control_does_not(db, org):
    """§17's claim, as a matched A/B: both arms can build, only one can find.

    The control is what makes this a test of accumulation rather than of practice — without it a
    falling cost across generations would be indistinguishable from agents simply getting better
    as the artifact record grows.
    """
    from civitas.experiments.tool_benchmark import run_tool_benchmark

    result = run_tool_benchmark(db, organization_id=org.id, generations=4)
    db.commit()
    summary = result.summary()

    assert summary["reuse"]["builds"] == 1, "the reuse arm rebuilt a tool that already existed"
    assert summary["rebuild"]["builds"] == 4, "the control arm did not rebuild every generation"
    assert summary["reuse"]["reuses"] == 3
    assert summary["rebuild"]["reuses"] == 0
    assert summary["reuse"]["success_rate"] == pytest.approx(1.0)
    assert summary["rebuild"]["success_rate"] == pytest.approx(1.0), (
        "the control must still solve the task, or the comparison is about capability not cost"
    )


def test_the_saving_from_reuse_scales_with_what_a_tool_costs_to_write(db, org):
    """Reuse cost is flat in tool size; rebuild cost is linear in it. That is what accumulating
    technology means, and it is why a single measurement at one tool size says nothing."""
    from civitas.experiments.tool_benchmark import run_tool_benchmark

    small = run_tool_benchmark(db, organization_id=org.id, generations=4, tool_padding_lines=0)
    db.commit()
    large = run_tool_benchmark(db, organization_id=org.id, generations=4, tool_padding_lines=80)
    db.commit()

    small_reuse = small.summary()["reuse"]["steady_state_mean_tokens"]
    large_reuse = large.summary()["reuse"]["steady_state_mean_tokens"]
    small_rebuild = small.summary()["rebuild"]["steady_state_mean_tokens"]
    large_rebuild = large.summary()["rebuild"]["steady_state_mean_tokens"]

    assert large_reuse == pytest.approx(small_reuse, rel=0.05), (
        "reuse cost must not grow with tool size — you do not pay to re-transmit what you do "
        "not rewrite"
    )
    assert large_rebuild > small_rebuild * 1.5, "rebuild cost must grow with tool size"
    assert large.derived()["steady_state_saved_fraction"] > (
        small.derived()["steady_state_saved_fraction"]
    )


def test_token_accounting_includes_tool_call_arguments():
    """A content-only count under-estimates most for the agents a budget most needs to bound.

    Measured on the tool benchmark: an episode that wrote a full source file and test suite was
    billed fewer prompt tokens than one that read a short listing, so §8's token budget would not
    have bounded tool-writing at all.
    """
    from civitas.runtime.providers.base import Message, Role, ToolCall
    from civitas.runtime.providers.offline import DeterministicProvider

    provider = DeterministicProvider()
    bare = [Message(Role.ASSISTANT, "writing a tool")]
    with_source = [
        Message(Role.ASSISTANT, "writing a tool",
                tool_calls=(ToolCall("c", "create_tool", {"source": "x" * 4000}),))
    ]
    assert provider.count_message_tokens(bare) < 20
    assert provider.count_message_tokens(with_source) > 900
