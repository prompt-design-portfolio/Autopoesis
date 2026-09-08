"""The newcomer benchmark (Part B §22) and its gates (ARCHITECTURE §3.1)."""

from __future__ import annotations

import pytest

from civitas.experiments.benchmark import (
    BenchmarkResult,
    Gate,
    GateFailure,
    _order_seed,
    config_hash_of,
    frozen_config,
    run_newcomer_benchmark,
)
from civitas.experiments.tasks.hidden_rule import build_device, era_instances
from civitas.runtime.budgets import Budgets

#: A small budget keeps the suite fast. The advantage it produces is smaller than the committed
#: measurement's; the tests assert direction and mechanism, and `results/` carries the magnitude.
FAST = Budgets(tokens=200_000, context=32_000, tool_calls=7, cost_usd=1.0,
               wall_clock_s=60, max_turns=12, max_no_progress_turns=4)


# --------------------------------------------------------------------------
# the task family
# --------------------------------------------------------------------------
def test_a_device_is_reproducible_from_its_seed_and_era():
    a = build_device(seed=3, era=2)
    b = build_device(seed=3, era=2)
    assert a.mapping == b.mapping
    assert a.digest() == b.digest()


def test_both_the_mapping_and_the_labels_rotate_between_eras():
    """The `π` lesson (ARCHITECTURE §2). Rotating the mapping alone would leave the labels as a
    stable index a collective could key on."""
    e1 = build_device(seed=0, era=1)
    e2 = build_device(seed=0, era=2)
    assert e1.mapping != e2.mapping
    assert e1.environment_version != e2.environment_version
    assert set(e1.classes) != set(e2.classes) or set(e1.operations) != set(e2.operations)


def test_the_mapping_is_a_bijection():
    """The structure that survives a rotation, and therefore the only thing worth learning
    *about* the device rather than *from* one era of it."""
    device = build_device(seed=7, era=1, n_classes=6, n_ops=10)
    assert len(set(device.mapping.values())) == len(device.mapping)


def test_the_task_description_does_not_single_out_the_answer():
    """§47: an agent must not be able to read its own success criteria.

    The answer necessarily appears in the description — it is one of the listed operations, and
    withholding the candidate set would make the task unanswerable rather than hidden. What must
    hold is that it appears *indistinguishably*: every operation is presented identically, so a
    reader of the description can do no better than chance.
    """
    instance = era_instances(build_device(seed=0, era=1))[0]
    answer = instance.evaluator_spec["expected"]

    assert answer not in instance.title
    assert all(op in instance.description for op in instance.device.operations), (
        "every operation must be offered, or the answer is distinguishable by omission"
    )
    for marker in ("expected", "correct", "answer is", answer.upper()):
        assert marker not in instance.description, f"the description hints at the answer: {marker}"


def test_the_evaluator_specification_is_held_on_the_task_not_shown_to_the_agent():
    """The spec carries the answer, and the episode's brief is built without it (§47)."""
    instance = era_instances(build_device(seed=0, era=1))[0]
    assert instance.evaluator_spec["expected"] == instance.device.answer_for(instance.input_class)
    assert "evaluator_spec" not in instance.description


def test_the_probe_order_seed_is_stable_across_processes():
    """`hash()` is randomised per process; a benchmark whose agents differ between invocations
    cannot support a claim about what changed (§46)."""
    assert _order_seed(1, 2, 3) == _order_seed(1, 2, 3)
    assert _order_seed(1, 2, 3) == 2129087654
    assert _order_seed(1, 2, 3) != _order_seed(1, 2, 4)


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
def test_a_failed_gate_withholds_the_numbers():
    """ARCHITECTURE §3.1: a failed gate does not annotate the result, it withholds it."""
    result = BenchmarkResult(experiment="x", config_hash="h", chance_level=0.1,
                             naive_probe_ceiling=0.5)
    result.gates["frozen_model"] = Gate("frozen_model", False, "arms differed")
    with pytest.raises(GateFailure, match="not read"):
        result.metrics()
    # ...but a caller may still diagnose, and has to say so in the code to do it.
    assert result.raw_metrics()["gates"]["frozen_model"]["passed"] is False


def test_gates_passing_releases_the_numbers():
    result = BenchmarkResult(experiment="x", config_hash="h", chance_level=0.1,
                             naive_probe_ceiling=0.5)
    result.gates["frozen_model"] = Gate("frozen_model", True, "one configuration")
    assert result.metrics()["config_hash"] == "h"


def test_a_missing_arm_reports_none_not_zero():
    """A missing comparison is not a zero difference."""
    result = BenchmarkResult(experiment="x", config_hash="h", chance_level=0.1,
                             naive_probe_ceiling=0.5)
    derived = result._derived()
    assert derived["newcomer_advantage"] is None
    assert derived["ablation_removes_fraction"] is None


# --------------------------------------------------------------------------
# the benchmark end to end
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    """One benchmark run, shared by every test that reads it.

    Module-scoped and on its own database: the run is the expensive part, and repeating it per
    test made this module take three minutes for no additional coverage. Reproducibility is
    checked separately by `test_the_benchmark_is_reproducible`, which does run it twice.
    """
    from sqlalchemy.orm import sessionmaker

    from civitas.persistence.engine import create_db_engine
    from civitas.persistence.models import Base, Organization
    from civitas.persistence.session import install_guards

    path = tmp_path_factory.mktemp("bench")
    engine = create_db_engine(url=f"sqlite:///{path}/civitas.db")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    install_guards(factory)
    session = factory()
    try:
        organization = Organization(name="Bench", slug="bench")
        session.add(organization)
        session.commit()
        yield run_newcomer_benchmark(
            session, organization_id=organization.id, seeds=[0, 1],
            accumulation_passes=3, budgets=FAST,
        )
    finally:
        session.close()
        engine.dispose()


def test_the_benchmark_runs_and_passes_its_gates(bench):
    assert bench.gates_passed, f"gates failed: {bench.failed_gates()}"
    for name in ("frozen_model", "accumulation_occurred", "baseline_has_headroom", "sufficient_n"):
        assert name in bench.gates


def test_every_arm_ran_under_one_configuration(bench):
    """Part B §20 frozen-model mode. Without this the comparison is between configurations."""
    assert bench.gates["frozen_model"].passed
    assert len(bench.gates["frozen_model"].value) == 1


def test_a_fresh_agent_does_better_inside_a_mature_collective(bench):
    """§22's primary criterion, first half."""
    metrics = bench.metrics()
    baseline = metrics["arms"]["baseline_empty"]["success_rate"]
    collective = metrics["arms"]["collective"]["success_rate"]
    assert collective > baseline, (
        f"no newcomer advantage: collective {collective:.3f} vs baseline {baseline:.3f}"
    )


def test_the_advantage_falls_when_memory_is_reset(bench):
    """§22's primary criterion, second half — the ablation control."""
    metrics = bench.metrics()
    collective = metrics["arms"]["collective"]["success_rate"]
    reset = metrics["arms"]["memory_reset"]["success_rate"]
    assert reset < collective, "resetting collective memory removed none of the advantage"


def test_the_advantage_falls_when_memory_is_scrambled(bench):
    """The presence-versus-content control: `collective_scrambled` returns comparable quantities
    with relevance destroyed, so an advantage that survives it was never about content."""
    metrics = bench.metrics()
    collective = metrics["arms"]["collective"]["success_rate"]
    scrambled = metrics["arms"]["collective_scrambled"]["success_rate"]
    assert scrambled < collective, "scrambling relevance removed none of the advantage"


def test_the_collective_arm_needs_fewer_probes(bench):
    """Cost-to-solution discriminates even where success rate saturates, and is reported
    alongside it for exactly that reason."""
    metrics = bench.metrics()
    assert (
        metrics["arms"]["collective"]["mean_probes"]
        < metrics["arms"]["baseline_empty"]["mean_probes"]
    )


def test_the_baseline_is_reported_against_chance_not_against_zero(bench):
    metrics = bench.metrics()
    assert metrics["chance_level"] == pytest.approx(0.1)
    assert metrics["naive_probe_ceiling"] > 0


def test_the_benchmark_is_reproducible(db, org):
    """Two runs of the same configuration must give the same numbers."""
    kwargs = dict(seeds=[0], accumulation_passes=2, budgets=FAST)
    first = run_newcomer_benchmark(db, organization_id=org.id, **kwargs)
    second = run_newcomer_benchmark(db, organization_id=org.id, **kwargs)
    for arm in first.arms:
        assert first.arms[arm].success_rate == second.arms[arm].success_rate, (
            f"arm {arm} was not reproducible"
        )


def test_probe_episodes_do_not_contaminate_the_environment_they_are_measured_against(db, org):
    """The founder-free discipline (ARCHITECTURE §3.5). A probe's own output must not inform the
    next probe, or accumulation is smuggled into the measurement."""
    from civitas.persistence.models import Artifact, Episode

    run_newcomer_benchmark(db, organization_id=org.id, seeds=[0], accumulation_passes=2,
                           budgets=FAST)
    probe_ids = {
        e.id for e in db.query(Episode).filter(Episode.is_benchmark_probe.is_(True)).all()
    }
    assert probe_ids
    leaked = (
        db.query(Artifact)
        .filter(Artifact.creator_episode_id.in_(probe_ids), Artifact.archived_at.is_(None))
        .count()
    )
    assert leaked == 0, "a probe episode's artifacts remained visible to later probes"


def test_the_frozen_configuration_hash_is_stable():
    assert config_hash_of(frozen_config(FAST)) == config_hash_of(frozen_config(FAST))
    other = Budgets(tokens=200_000, context=32_000, tool_calls=99, cost_usd=1.0,
                    wall_clock_s=60, max_turns=12)
    assert config_hash_of(frozen_config(FAST)) != config_hash_of(frozen_config(other))
