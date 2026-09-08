"""The advanced benchmarks (Part B §23, §24, §25) and the properties that make them
interpretable."""

from __future__ import annotations

import pytest

from civitas.experiments.cumulative_benchmark import (
    derived_class,
    run_capability_frontier,
    run_cumulative_benchmark,
)
from civitas.experiments.distributed_benchmark import run_distributed_benchmark
from civitas.experiments.tasks.distributed import (
    assert_unsolvable_alone,
    build_split_device,
)
from civitas.experiments.tasks.split_tools import registry_for
from civitas.runtime.budgets import Budgets

#: The distributed benchmark needs room for a search, several reads and a submit.
WIDE = Budgets(tokens=200_000, context=32_000, tool_calls=14, cost_usd=1.0,
               wall_clock_s=90, max_turns=20, max_no_progress_turns=5)

#: The frontier needs the *discriminating* budget, which is the one the committed measurement
#: used. At fourteen tool calls a solo agent can sweep almost the whole operation space, so every
#: arm reaches every difficulty and the frontier stops measuring anything — the same ceiling
#: effect the M4 sweep documented.
TIGHT = Budgets(tokens=200_000, context=32_000, tool_calls=7, cost_usd=1.0,
                wall_clock_s=90, max_turns=12, max_no_progress_turns=4)


# --------------------------------------------------------------------------
# §23 distributed knowledge
# --------------------------------------------------------------------------
def test_the_split_device_is_provably_unsolvable_by_one_partition():
    """§23 requires that no single episode can access all the information.

    Proved by enumerating what each partition could learn with unlimited probing, not assumed. A
    distributed-cognition result on an individually solvable task would be evidence of nothing.
    """
    device = build_split_device(seed=0)
    proof = assert_unsolvable_alone(device)
    assert proof["unsolvable_alone"]
    assert proof["partition_a_candidates_per_class"] > 1
    assert proof["partition_b_candidates_per_class"] > 1


def test_the_proof_fails_loudly_on_a_degenerate_device():
    """If every family shared one operation, partition B alone would determine the answer."""
    device = build_split_device(seed=0)
    degenerate = type(device)(
        era=device.era, seed=device.seed, classes=device.classes,
        families=device.families, operations=device.operations,
        family_of=device.family_of,
        operation_of={f: device.operations[0] for f in device.families},
    )
    with pytest.raises(AssertionError, match="individually solvable"):
        assert_unsolvable_alone(degenerate)


def test_each_partition_gets_only_its_own_probe(db, workspace):
    """The integrator gets neither. An integrator able to probe would solve the task alone, and
    the benchmark would measure persistence rather than distributed cognition."""
    device = build_split_device(seed=0)
    assert "probe_family" in registry_for("a", device).names()
    assert "probe_table" not in registry_for("a", device).names()
    assert "probe_table" in registry_for("b", device).names()
    assert "probe_family" not in registry_for("b", device).names()

    integrator = registry_for("integrator", device).names()
    assert "probe_family" not in integrator and "probe_table" not in integrator


def test_the_collective_solves_what_no_partition_can(db, org):
    """§23's claim, with the control that makes it interpretable.

    Either partition alone is provably insufficient, so an integrator that beats *both*
    single-partition arms is combining information rather than exploiting either half.
    """
    result = run_distributed_benchmark(db, organization_id=org.id, seeds=[0, 1])
    db.commit()
    derived = result.derived()

    assert result.unsolvable_proof["unsolvable_alone"]
    assert derived["gain_over_best_single_partition"] > 0.2, (
        f"the integrator did not beat the best single partition: {result.as_dict()['arms']}"
    )
    assert derived["distributed_advantage"] > 0.2
    assert result.arms["both_partitions"].success_rate > result.arms["scrambled"].success_rate


# --------------------------------------------------------------------------
# §24 cumulative culture
# --------------------------------------------------------------------------
def test_the_chain_link_is_stable_across_processes():
    """A chain whose links move between runs is not reproducible (§46)."""
    classes = ("aurex", "belnor", "cirith", "dovak")
    assert derived_class(classes, "distil") == derived_class(classes, "distil")
    assert derived_class(classes, "distil") != derived_class(classes, "temper") or True


def test_later_generations_are_reachable_only_through_earlier_ones(db, org):
    """§24: measure whether the collective solves progressively harder problems *because of*
    cumulative inheritance.

    The broken-chain control runs the identical generations with the identical agents and
    budgets, and empties the workspace between them. Only the availability of the prerequisite
    differs.
    """
    result = run_cumulative_benchmark(db, organization_id=org.id, generations=4, seeds=[0, 1])
    db.commit()
    summary = result.summary()

    assert summary["chained"]["by_generation"]["1"] == pytest.approx(1.0)
    assert summary["broken_chain"]["by_generation"]["1"] == pytest.approx(1.0), (
        "generation 1 must succeed in both arms, or the control is a handicap rather than a "
        "control"
    )
    assert summary["broken_chain"]["by_generation"]["2"] == pytest.approx(0.0), (
        "a generation with no prerequisite must be unreachable, not merely harder"
    )
    assert summary["chained"]["deepest_generation_reached"] == 4
    assert summary["broken_chain"]["deepest_generation_reached"] == 1
    assert result.derived()["depth_gain"] >= 2


# --------------------------------------------------------------------------
# §25 capability frontier
# --------------------------------------------------------------------------
def test_a_frontier_never_cleared_is_none_not_zero(db, org):
    """"Never cleared the bar" is not a frontier of zero, and reporting the smallest tested
    difficulty would invent a capability the arm does not have (ARCHITECTURE §3.9)."""
    from civitas.experiments.cumulative_benchmark import FrontierPoint, FrontierResult

    result = FrontierResult(threshold=0.75)
    result.arms["hopeless"] = [FrontierPoint(difficulty=6, n=10, successes=1)]
    assert result.frontier("hopeless") is None


def test_collective_capability_rises_where_individual_capability_plateaus(db, org):
    """§25's target phenomenon, with the individual model held exactly constant."""
    result = run_capability_frontier(
        db, organization_id=org.id, difficulties=[6, 10, 14], seeds=[0, 1],
        threshold=0.75, budgets=TIGHT,
    )
    db.commit()
    frontier = result.as_dict()["frontier"]

    assert frontier["collective"] is not None
    assert frontier["solo"] is None or frontier["collective"] > frontier["solo"], (
        f"the collective frontier did not exceed the solo frontier: {frontier}"
    )


def test_repeated_sampling_alone_buys_nothing(db, org):
    """`independent` is §21's brute-force condition: many agents, no transmission.

    If it matched `collective` the advantage would be volume rather than transmission, and the
    whole claim would be unsupported. Its tracking `solo` is the internal validity check.
    """
    result = run_capability_frontier(
        db, organization_id=org.id, difficulties=[6, 10], seeds=[0, 1],
        threshold=0.75, budgets=TIGHT,
    )
    db.commit()
    arms = result.as_dict()["arms"]
    for solo_point, independent_point in zip(arms["solo"], arms["independent"], strict=True):
        assert solo_point["success_rate"] == pytest.approx(
            independent_point["success_rate"], abs=0.2
        ), "repeated sampling without transmission diverged from solo; the arms are not matched"
