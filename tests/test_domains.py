"""Domain adapters (Part B §5, §18, §47).

Two tests here are parametrized over *every registered domain* rather than over a fixed list. That
is the point of the file: a domain added later cannot be added without passing the §47 leak check
and without its declared tools existing. A test enumerating domains by name would go quiet exactly
when a new domain is introduced — which is when it is most needed.
"""

from __future__ import annotations

import json

import pytest

from civitas.domains import (
    DomainLeak,
    DomainTask,
    all_domains,
    available_domains,
    check_no_leak,
    get_domain,
    register_domain,
)
from civitas.domains.code_repair import PROGRAMS, SandboxedTestsEvaluator
from civitas.runtime.tools.builtin import default_registry
from civitas.runtime.tools.ecology import toolmaker_registry

DOMAINS = list(all_domains())
DOMAIN_IDS = [d.name for d in DOMAINS]


# --------------------------------------------------------------------------- the registry


def test_both_built_in_domains_are_registered():
    assert available_domains() == ("code_repair", "hidden_rule")


def test_registering_a_name_twice_is_refused(monkeypatch):
    """Two domains under one name would make `get_domain` return whichever imported last, and a
    manifest recording the name would identify neither (§46)."""
    with pytest.raises(ValueError, match="already registered"):
        register_domain(get_domain("hidden_rule"))


def test_an_unknown_domain_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="registered"):
        get_domain("no_such_domain")


# --------------------------------------------------------------------------- §47, over every domain


@pytest.mark.parametrize("domain", DOMAINS, ids=DOMAIN_IDS)
def test_no_generated_task_reveals_its_own_success_criteria(domain):
    """§47, checked at the boundary where domains are written rather than once per domain."""
    for seed in range(4):
        for task in domain.generate(seed=seed, count=4):
            check_no_leak(task)


@pytest.mark.parametrize("domain", DOMAINS, ids=DOMAIN_IDS)
def test_every_domain_states_an_indistinguishability_argument(domain):
    """A domain that declared neither hidden terms nor a candidate set would pass `check_no_leak`
    vacuously — the check would be running against nothing. Requiring at least one is what stops
    the guard from being satisfiable by omission."""
    for task in domain.generate(seed=0, count=2):
        assert task.hidden_terms or task.candidate_terms, (
            f"{domain.name} declares no §47 argument for {task.title!r}"
        )


@pytest.mark.parametrize("domain", DOMAINS, ids=DOMAIN_IDS)
def test_every_declared_tool_exists(domain):
    """A domain naming a tool the runtime does not have would fail at episode time, in the middle
    of a benchmark, and look like an agent failure."""
    shared = set(default_registry().names()) | set(toolmaker_registry().names())
    for task in domain.generate(seed=0, count=1):
        buildable = {t.name for t in domain.build_tools(task)}
        missing = [t for t in domain.tool_names() if t not in shared | buildable]
        assert not missing, f"{domain.name} declares tools that do not exist: {missing}"


@pytest.mark.parametrize("domain", DOMAINS, ids=DOMAIN_IDS)
def test_every_generated_task_can_be_judged(domain):
    """The evaluator a task names must be one the domain supplies. A task whose kind no evaluator
    handles fails at evaluation — after the episode has been paid for."""
    from civitas.experiments.evaluation import evaluator_for

    kinds = {e.kind for e in domain.evaluators()}
    for task in domain.generate(seed=0, count=3):
        kind = task.evaluator_spec["kind"]
        assert kind in kinds
        assert evaluator_for(kind).kind == kind


# --------------------------------------------------------------------------- the leak check itself


def test_the_leak_check_catches_a_hidden_term_in_the_description():
    with pytest.raises(DomainLeak, match="hidden term"):
        check_no_leak(DomainTask(
            title="Repair f", description="it should return [1, 3, 3, 7]",
            evaluator_spec={}, hidden_terms=("[1, 3, 3, 7]",),
        ))


def test_the_leak_check_catches_a_marker_word():
    with pytest.raises(DomainLeak, match="marker"):
        check_no_leak(DomainTask(
            title="t", description="the correct operation is one of alpha, beta",
            evaluator_spec={}, candidate_terms=("alpha", "beta"),
        ))


def test_the_leak_check_catches_a_candidate_that_was_left_out():
    """Presence by omission: listing nine of ten operations makes the tenth the answer, and no
    marker word is needed to say so."""
    with pytest.raises(DomainLeak, match="omission"):
        check_no_leak(DomainTask(
            title="t", description="options: alpha, beta",
            evaluator_spec={}, candidate_terms=("alpha", "beta", "gamma"),
        ))


def test_the_leak_check_catches_a_candidate_mentioned_more_often_than_the_others():
    """Frequency is the other way an author gives the answer away by accident — an example that
    happens to use the answer, a reminder that repeats it."""
    with pytest.raises(DomainLeak, match="frequency"):
        check_no_leak(DomainTask(
            title="t", description="options: alpha, beta. for instance alpha.",
            evaluator_spec={}, candidate_terms=("alpha", "beta"),
        ))


def test_a_task_that_hides_its_answer_properly_passes():
    check_no_leak(DomainTask(
        title="t", description="options: alpha, beta, gamma",
        evaluator_spec={"expected": "beta"}, candidate_terms=("alpha", "beta", "gamma"),
    ))


# --------------------------------------------------------------------------- hidden_rule


def test_the_hidden_rule_adapter_wraps_the_device_it_was_measured_on():
    """The adapter must be a re-description of the M4 benchmark's device, not a reimplementation
    of it — otherwise the +0.300 does not transfer to anything the adapter runs."""
    from civitas.experiments.tasks.hidden_rule import build_device, era_instances

    domain = get_domain("hidden_rule")
    tasks = domain.generate(seed=11, count=3, era=2)
    device = build_device(seed=11, era=2)
    instances = era_instances(device)[:3]

    assert [t.title for t in tasks] == [i.title for i in instances]
    assert [t.evaluator_spec for t in tasks] == [i.evaluator_spec for i in instances]
    assert tasks[0].meta["environment_version"] == device.environment_version


def test_a_task_specific_tool_is_built_per_task_and_holds_no_state_between_them():
    """§4: a domain object that carried the device would be hidden state shared across episodes —
    and, since the arms run against different devices, shared across arms."""
    domain = get_domain("hidden_rule")
    a, b = domain.generate(seed=1, count=1)[0], domain.generate(seed=2, count=1)[0]
    tool_a, tool_b = domain.build_tools(a)[0], domain.build_tools(b)[0]

    assert tool_a is not tool_b
    assert tool_a._device.mapping != tool_b._device.mapping
    assert domain.build_tools(a)[0]._device.mapping == tool_a._device.mapping


def test_the_hidden_rule_evaluator_never_returns_the_expected_value():
    """§47: an evaluation row is readable through the API."""
    from civitas.domains.hidden_rule import ExactMatchEvaluator

    _, _, detail = ExactMatchEvaluator().judge({"expected": "ferrose"}, "wrong")
    assert "ferrose" not in json.dumps(detail)


# --------------------------------------------------------------------------- code_repair


REPAIRS = {
    "running_max": (
        "def running_max(xs):\n"
        "    out, best = [], None\n"
        "    for x in xs:\n"
        "        best = x if best is None else max(best, x)\n"
        "        out.append(best)\n"
        "    return out\n"
    ),
    "rle_encode": (
        "def rle_encode(s):\n"
        "    out = []\n"
        "    if not s:\n"
        "        return out\n"
        "    current, count = s[0], 1\n"
        "    for ch in s[1:]:\n"
        "        if ch == current:\n"
        "            count += 1\n"
        "        else:\n"
        "            out.append([current, count])\n"
        "            current, count = ch, 1\n"
        "    out.append([current, count])\n"
        "    return out\n"
    ),
    "merge_intervals": (
        "def merge_intervals(intervals):\n"
        "    if not intervals:\n"
        "        return []\n"
        "    ordered = sorted(intervals)\n"
        "    out = [list(ordered[0])]\n"
        "    for start, end in ordered[1:]:\n"
        "        if start <= out[-1][1]:\n"
        "            out[-1][1] = max(out[-1][1], end)\n"
        "        else:\n"
        "            out.append([start, end])\n"
        "    return out\n"
    ),
    "first_index": (
        "def first_index(xs, target):\n"
        "    lo, hi, found = 0, len(xs) - 1, -1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if xs[mid] == target:\n"
        "            found, hi = mid, mid - 1\n"
        "        elif xs[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return found\n"
    ),
}


def _spec_for(entry: str) -> dict:
    domain = get_domain("code_repair")
    index = [p.entry for p in PROGRAMS].index(entry)
    return domain.generate(seed=index, count=1)[0].evaluator_spec


@pytest.mark.parametrize("program", PROGRAMS, ids=[p.entry for p in PROGRAMS])
def test_the_shipped_defect_really_fails_and_the_repair_really_passes(program):
    """The domain's claim is that each program is *defective*. A "broken" implementation that
    passed the checks would make the task unmeasurable while looking fine, and every arm would
    score 1.0 on it."""
    evaluator = SandboxedTestsEvaluator()
    spec = _spec_for(program.entry)

    broken_ok, broken_score, _ = evaluator.judge(spec, program.broken)
    fixed_ok, fixed_score, detail = evaluator.judge(spec, REPAIRS[program.entry])

    assert broken_ok is False and broken_score < 1.0
    assert fixed_ok is True and fixed_score == 1.0
    assert detail["passed"] == detail["total"] == len(program.checks)


def test_the_four_defects_are_of_four_different_kinds():
    """A domain whose defects are all one shape measures one skill and reports it as "repair"."""
    kinds = [p.defect for p in PROGRAMS]
    assert len(set(kinds)) == len(kinds)


def test_a_submission_that_raises_scores_zero_rather_than_crashing_the_evaluator():
    evaluator = SandboxedTestsEvaluator()
    ok, score, detail = evaluator.judge(
        _spec_for("running_max"), "def running_max(xs):\n    raise ValueError('boom')\n"
    )
    assert ok is False and score == 0.0
    assert detail["errors"] == detail["total"]


def test_an_empty_submission_is_a_failure_not_an_exception():
    ok, score, detail = SandboxedTestsEvaluator().judge(_spec_for("running_max"), "")
    assert ok is False and score == 0.0 and detail["reason"] == "no submission"


def test_a_submission_that_never_terminates_is_stopped_by_the_sandbox():
    """§18: an agent's submission is untrusted code. The evaluator's own wall-clock bound is what
    keeps a non-terminating repair from taking a worker with it."""
    ok, score, detail = SandboxedTestsEvaluator().judge(
        _spec_for("running_max"),
        "def running_max(xs):\n    while True:\n        pass\n",
    )
    assert ok is False and score == 0.0
    assert detail["reason"] in ("timeout", "submission did not run")


def test_the_evaluator_reports_the_failing_check_by_index_and_never_by_content():
    """§47: naming the failing input would hand out the check suite one attempt at a time."""
    spec = _spec_for("merge_intervals")
    _, _, detail = SandboxedTestsEvaluator().judge(
        spec, "def merge_intervals(intervals):\n    return []\n"
    )
    rendered = json.dumps(detail)
    assert isinstance(detail["first_failing_check"], int)
    for _args, want in spec["checks"]:
        if want:
            assert json.dumps(want) not in rendered


def test_the_evaluator_reports_limits_the_sandbox_could_not_enforce():
    """§46: a bound the caller believes is in force but is not must reach the record. Under uid 0
    `RLIMIT_NPROC` is silently not enforced, and a manifest that claimed it was would be wrong."""
    from civitas.runtime.sandbox import get_sandbox

    unenforced = get_sandbox().unenforced_limits()
    _, _, detail = SandboxedTestsEvaluator().judge(
        _spec_for("running_max"), REPAIRS["running_max"]
    )
    assert detail.get("unenforced_limits", []) == list(unenforced)
    assert "backend" in detail and "isolation_level" in detail
