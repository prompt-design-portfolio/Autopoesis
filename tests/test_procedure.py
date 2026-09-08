"""§22's procedure over any domain (Part B §5, §20, §22, §46, §47).

The refactor this file guards is the one that made M4's benchmark a *call* into a shared
procedure rather than its own implementation. Two things have to hold for that to be safe: the
shared path must still measure what the device benchmark measured, and a second domain must run
through it without the path knowing which domain it has.
"""

from __future__ import annotations

import pytest

from civitas.domain.enums import ExperimentArm, TerminationReason
from civitas.domains import get_domain
from civitas.domains.base import DomainLeak, DomainTask
from civitas.experiments.benchmark import REPAIR_BUDGETS, frozen_config
from civitas.experiments.procedure import (
    archive_episode_artifacts,
    create_domain_task,
    reset_memory,
    run_domain_episode,
)
from civitas.persistence.models import Artifact, Episode, ToolDefinition, ToolRun

REPAIR = "code_repair"
DEVICE = "hidden_rule"


def _spec(domain_name: str, *, seed: int = 0, era: int = 1) -> DomainTask:
    return get_domain(domain_name).generate(seed=seed, count=1, era=era)[0]


def _run(db, workspace, domain_name, *, seed=0, era=1, arm=ExperimentArm.COLLECTIVE,
         probe_order_seed=0, record_findings=True, is_probe=False, budgets=None, spec=None):
    domain = get_domain(domain_name)
    spec = spec or _spec(domain_name, seed=seed, era=era)
    return domain, spec, run_domain_episode(
        db, domain=domain, spec=spec, workspace_id=workspace.id, arm=arm,
        budgets=budgets or (REPAIR_BUDGETS if domain_name == REPAIR else None),
        record_findings=record_findings, is_probe=is_probe,
        probe_order_seed=probe_order_seed,
    )


# --------------------------------------------------------------------------- the shared path


@pytest.mark.parametrize("domain_name", [DEVICE, REPAIR])
def test_the_same_episode_path_carries_both_domains(db, workspace, domain_name):
    """The claim the adapter layer exists to support: one runtime, one arm enforcement, one
    evaluation path, two kinds of task."""
    _, spec, result = _run(db, workspace, domain_name)
    episode = db.get(Episode, result.episode_id)

    assert episode.experiment_arm is ExperimentArm.COLLECTIVE
    assert episode.environment_version == spec.environment_version
    assert result.termination_reason != "unknown"
    assert result.tool_calls > 0


@pytest.mark.parametrize("domain_name", [DEVICE, REPAIR])
def test_the_task_row_holds_the_evaluator_spec_and_the_episode_never_sees_it(
    db, workspace, domain_name
):
    """§47: the criteria live on the row the evaluator reads, not in what the agent reads."""
    domain, spec, result = _run(db, workspace, domain_name)
    episode = db.get(Episode, result.episode_id)
    from civitas.persistence.models import Task

    task = db.get(Task, episode.task_id)
    assert task.evaluator_spec == spec.evaluator_spec
    assert task.evaluator_spec
    for column in ("title", "description"):
        assert "evaluator_spec" not in getattr(task, column)


@pytest.mark.parametrize("domain_name", [DEVICE, REPAIR])
def test_a_domains_tool_is_registered_so_its_runs_are_attributable(db, workspace, domain_name):
    """§17: a tool run that names no registered tool cannot be audited or costed."""
    domain, spec, result = _run(db, workspace, domain_name)
    names = {t.name for t in domain.build_tools(spec)}
    registered = {
        d.name for d in db.query(ToolDefinition).filter_by(workspace_id=workspace.id).all()
    }
    assert names <= registered

    runs = db.query(ToolRun).filter_by(episode_id=result.episode_id).all()
    if runs:
        assert all(r.tool_definition_id is not None for r in runs)


def test_creating_a_task_refuses_one_that_gives_its_answer_away(db, workspace):
    """The leak check runs on the one funnel every benchmark task passes through, so a domain
    that leaked would be stopped before it produced a clean-looking measurement of nothing."""
    leaky = DomainTask(
        title="t", description="the answer is alpha", evaluator_spec={"expected": "alpha"},
        candidate_terms=("alpha", "beta"),
    )
    with pytest.raises(DomainLeak):
        create_domain_task(db, workspace_id=workspace.id, spec=leaky)


def test_the_frozen_configuration_records_the_domains_prompt_version(db, workspace):
    """§20: two domains with different prompts must not hash to the same configuration, or a
    cross-domain comparison would claim to be matched when it is not."""
    a = frozen_config(system_prompt_version=get_domain(DEVICE).system_prompt_version())
    b = frozen_config(system_prompt_version=get_domain(REPAIR).system_prompt_version())
    assert a != b
    assert a["system_prompt_version"] == "hidden-rule/1.0"
    assert b["system_prompt_version"] == "code-repair/1.0"


# --------------------------------------------------------------------------- code repair, live


def test_a_repair_agent_that_finds_the_edit_submits_source_that_the_sandbox_accepts(db, workspace):
    """End to end on the second domain: the agent writes a program, and an evaluator *executes*
    it against checks the agent never saw."""
    spec = _spec(REPAIR)
    outcomes = [
        _run(db, workspace, REPAIR, spec=spec, probe_order_seed=s)[2] for s in range(6)
    ]
    assert any(o.succeeded for o in outcomes), "no agent ever found the repair"
    winner = next(o for o in outcomes if o.succeeded)
    assert winner.submitted and "def " in winner.submitted
    episode = db.get(Episode, winner.episode_id)
    assert episode.termination_reason is TerminationReason.EVALUATOR_SUCCESS


def test_a_rejected_edit_counts_as_progress(db, workspace):
    """M4 lost a third of the measured advantage to exactly this: an episode that ruled candidates
    out but never registered progress died of `no_progress` with budget unspent, and the loss was
    recorded as an agent failure rather than as the bookkeeping bug it was."""
    spec = _spec(REPAIR)
    results = [
        _run(db, workspace, REPAIR, spec=spec, probe_order_seed=s)[2] for s in range(6)
    ]
    exhausted = [r for r in results if r.termination_reason == "budget_exhausted"]
    no_progress = [r for r in results if r.termination_reason == "no_progress"]
    assert not no_progress or all(r.tool_calls >= REPAIR_BUDGETS.tool_calls - 1
                                  for r in no_progress)
    assert all(r.tool_calls >= 4 for r in exhausted)


def test_an_edit_named_for_another_function_is_refused(db, workspace):
    """§A2.3 keys a documented failure on the call arguments, so `entry` has to be *in* them —
    two functions in one workspace can be offered the same surface label, and a key without the
    entry would let a rejection for one block a good attempt on the other."""
    from civitas.experiments.tasks.patch_tool import TryPatchTool

    domain = get_domain(REPAIR)
    spec = _spec(REPAIR)
    tool = TryPatchTool(domain._presented(spec))
    assert "entry" in tool.parameters["required"]

    _, _, result = _run(db, workspace, REPAIR, spec=spec)
    episode = db.get(Episode, result.episode_id)
    runs = db.query(ToolRun).filter_by(episode_id=episode.id).all()
    assert runs, "the agent never called try_patch"
    for run in runs:
        assert "entry" in run.args and "patch" in run.args


def test_every_candidate_edit_applies_and_exactly_one_repairs(db, workspace):
    """A candidate that cannot even be applied is not an alternative — it is a wasted tool call
    that returns an error instead of evidence. Two ambiguous anchors got this far in M10 and were
    caught only because the catalogue is checked against its own worked examples."""
    from civitas.domains.code_repair import PROGRAMS, era_program

    for seed in range(4):
        for era in (1, 2):
            for program in PROGRAMS:
                presented = era_program(program, seed=seed, era=era)
                args, want = presented.example
                passing = []
                for label in presented.patch_names:
                    source = presented.apply(label)  # must not raise
                    namespace: dict = {}
                    try:
                        exec(source, namespace)
                        got = namespace[presented.entry](*args)
                    except Exception:
                        continue
                    if got == want:
                        passing.append(label)
                assert len(passing) == 1, (
                    f"{presented.entry} s{seed}e{era}: {len(passing)} candidates pass the example"
                )
                source = presented.apply(passing[0])
                namespace = {}
                exec(source, namespace)
                for check_args, check_want in presented.checks:
                    assert namespace[presented.entry](*check_args) == check_want


def test_rotating_the_era_makes_a_memorised_label_worthless(db, workspace):
    """The `π` discipline (ARCHITECTURE §2). Both the repairing edit and the labels are redrawn
    per era: rotating only the mapping would leave the labels as a stable index a collective could
    key on, and recall would be indistinguishable from knowledge."""
    from civitas.domains.code_repair import PROGRAMS, era_program

    one = era_program(PROGRAMS[0], seed=0, era=1)
    two = era_program(PROGRAMS[0], seed=0, era=2)
    assert one.environment_version != two.environment_version
    assert set(one.patch_names) != set(two.patch_names)
    assert one.repairing_label() != two.repairing_label()


def test_a_finding_from_another_era_is_not_believed(db, workspace):
    """§15, §48: an artifact recorded under a different environment version describes a different
    catalogue, and believing it produces a confident wrong answer rather than a wasted call."""
    from civitas.domains.code_repair import PROGRAMS, era_program
    from civitas.experiments.repair_policy_agent import (
        RepairPolicyAgentProvider,
        _Belief,
        finding_line,
    )

    current = era_program(PROGRAMS[0], seed=0, era=2)
    agent = RepairPolicyAgentProvider(
        program=current, environment_version=current.environment_version
    )
    stale = finding_line("repair-e1-s0", current.entry, current.patch_names[0], True)
    belief = _Belief()
    agent._absorb_findings(belief, stale)
    assert belief.repaired_by is None

    fresh = finding_line(
        current.environment_version, current.entry, current.patch_names[0], True
    )
    belief = _Belief()
    agent._absorb_findings(belief, fresh)
    assert belief.repaired_by == current.patch_names[0]


# --------------------------------------------------------------------------- arm mechanics


@pytest.mark.parametrize("domain_name", [DEVICE, REPAIR])
def test_a_probes_own_output_is_archived_not_deleted(db, workspace, domain_name):
    """Founder-free discipline (ARCHITECTURE §3.5), and Part A §A1.2: the contribution stays on
    the record, it is only excluded from retrieval."""
    _, _, result = _run(db, workspace, domain_name, is_probe=True)
    created = db.query(Artifact).filter_by(creator_episode_id=result.episode_id).all()
    archive_episode_artifacts(db, result.episode_id)
    refreshed = db.query(Artifact).filter_by(creator_episode_id=result.episode_id).all()
    assert len(refreshed) == len(created)
    assert all(a.archived_at is not None for a in refreshed)


def test_memory_reset_archives_the_corpus_and_retires_the_duplicate_index(db, workspace):
    """Archiving the artifacts while leaving the index answering would let the reset arm keep the
    one thing it is meant to lose."""
    from civitas.persistence.models import DuplicateFailureRecord

    spec = _spec(REPAIR)
    for s in range(3):
        _run(db, workspace, REPAIR, spec=spec, probe_order_seed=s)
    db.flush()

    reset_memory(db, workspace.id)
    live = db.query(Artifact).filter_by(workspace_id=workspace.id, archived_at=None).all()
    assert live == []
    entries = db.query(DuplicateFailureRecord).filter_by(workspace_id=workspace.id).all()
    assert entries == [] or all(e.retired_at is not None for e in entries)
