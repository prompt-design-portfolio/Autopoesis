"""The arm regression (A4.2, A1.5).

    Experimental controls are load-bearing. The arms in B§4, the frozen assay, the matched nulls
    and the self-tests must keep working after every change. A commit that breaks a control is a
    broken commit.

So this module asserts, arm by arm, that what the arm removes is actually removed and what it
holds constant is actually held. It is deliberately literal: each test names one clause of B§4's
third column and checks it against the world the arm builds, so a change that quietly turns a
control into a copy of its treatment fails here rather than in a result.
"""

from __future__ import annotations

import numpy as np
import pytest

import sim_v3_13
from civitas_g.world.arms import ARMS, REFERENCE_ARM_ORDER, get_arm, runnable_arms
from civitas_g.world.spec import N_PREPS, N_TYPES


def kw(name: str) -> dict:
    return get_arm(name).config_kwargs()


# --------------------------------------------------------------------------- what each arm is

def test_memory_reset_has_no_record_and_the_read_channels_are_zero():
    """B§4: "no record; the read channels exist and are zero"."""
    assert kw("memory_reset")["record"] == "none"
    world = sim_v3_13.World(sim_v3_13.Config(**kw("memory_reset")), np.random.default_rng(0))
    # the store array still exists -- the arm removes the WRITING, not the world's shape
    assert world.marks.shape == (N_TYPES, N_PREPS, world.cfg.grid, world.cfg.grid)
    assert not world.marks.any()


def test_collective_writes_and_reads_the_record():
    assert kw("collective")["record"] == "real"
    assert kw("collective")["mode"] == "plastic"


def test_collective_scrambled_holds_density_and_sign_and_destroys_only_the_label():
    """A2.3: "a store existed" has to be separated from "information passed".

    Measured against the engine rather than asserted: write the same preparations under `real` and
    under `noise` and check that the number of marks and the signs match while the labels differ.
    """
    assert kw("collective_scrambled")["record"] == "noise"

    def write_all(record: str) -> tuple[int, list[float], list[int]]:
        cfg = sim_v3_13.Config(**dict(kw("collective"), record=record))
        world = sim_v3_13.World(cfg, np.random.default_rng(0))
        world.marks[:] = 0.0
        world.mapping, world.pi, world.chain_on = (0, 1, 2), (0, 1, 2, 3, 4), True
        for ftype in range(N_TYPES):
            for k in range(N_PREPS):
                world.write_mark(ftype, k, k == world.mapping[ftype], 5 + ftype, 5 + k, cfg)
        nz = np.abs(world.marks) > 1e-9
        signs = sorted(world.marks[nz].tolist())
        labels = sorted(int(i) for i in np.argwhere(nz)[:, 1])
        return int(nz.sum()), signs, labels

    n_real, signs_real, labels_real = write_all("real")
    n_noise, signs_noise, labels_noise = write_all("noise")
    assert n_real == n_noise, "the noise arm must write the same number of marks"
    assert signs_real == signs_noise, "the noise arm must preserve the signs"
    assert labels_real != labels_noise, "the noise arm must destroy the label association"


def test_collective_scrambled_is_not_the_modulator_scramble():
    """`scramble=True` randomises the modulator's SIGN and attacks learning. Different control."""
    assert kw("collective_scrambled").get("scramble", False) is False


def test_no_plasticity_keeps_the_record_and_removes_learning():
    """B§4: "the record, a genome that cannot learn"."""
    k = kw("no_plasticity")
    assert k["record"] == "real"
    assert k["mode"] == "fixed"


def test_solo_is_the_affordance_null_uniform_actions_in_the_same_world():
    """B§4: "the affordance null: uniform actions, same world"."""
    k = kw("solo")
    assert k["mode"] == "random"
    # same world: every other parameter is the world of record, untouched
    from civitas_g.world.spec import WORLD

    assert {key: v for key, v in k.items() if key != "mode"} == \
        {key: v for key, v in WORLD.items() if key != "mode"}


def test_the_slow_arm_differs_from_collective_only_in_the_label_clock():
    a, b = kw("collective"), kw("collective_slow_labels")
    assert {k: v for k, v in b.items() if k != "label_every"} == \
        {k: v for k, v in a.items() if k != "label_every"}
    assert a.get("label_every", 0) == 0 and b["label_every"] == 3 * a["prep_every"]


# --------------------------------------------------------------------------- the set as a whole

def test_every_runnable_arm_builds_the_world_of_record():
    """One change per arm. An arm that moved a second parameter would confound its own control."""
    from civitas_g.world.spec import WORLD

    for arm in runnable_arms():
        k = arm.config_kwargs()
        changed = {key for key in WORLD if k.get(key) != WORLD.get(key)}
        assert not changed, f"{arm.name} changes world parameters {sorted(changed)}"


def test_every_arm_keeps_the_chance_ev_of_a_preparation_at_zero():
    """If an arm could pay for chance preparations, its hit rate would stop measuring knowledge."""
    from civitas_g.world.spec import check_chance_ev_is_zero

    for arm in runnable_arms():
        assert check_chance_ev_is_zero(arm.config_kwargs()) == 0.0


def test_every_arm_presents_the_same_observation_layout():
    """No genome may see a different world shape, or a between-arm difference could be that."""
    for arm in runnable_arms():
        cfg = sim_v3_13.Config(**arm.config_kwargs())
        assert cfg.scaffold_food is False and cfg.goal_channel is False
    assert sim_v3_13.N_IN == sim_v3_13.READ + N_PREPS


def test_the_reference_arms_are_the_five_the_summary_prints():
    assert set(REFERENCE_ARM_ORDER) <= {a.name for a in ARMS}
    assert len(REFERENCE_ARM_ORDER) == 5
    assert "solo" not in REFERENCE_ARM_ORDER, \
        "v3.13 has no random arm: its baseline is v3.12's learner"


@pytest.mark.parametrize("arm", [a for a in ARMS if a.milestone == "G1" and not a.is_replay])
def test_every_g1_arm_states_what_it_holds_or_removes(arm):
    assert arm.holds_or_removes, f"{arm.name} carries no statement of what it controls"
    assert arm.research_name, f"{arm.name} has no research name, so the manifest cannot survive"


def test_the_research_name_is_the_variants_key_not_the_directives_description():
    """The reproduction is keyed by the name the summaries PRINT.

    B§4's table calls the noise arm "plastic + noise record"; `analysis_v3_13.VARIANTS` calls it
    "plastic + noise", and that is what every reference summary prints. Taking the directive's
    phrasing cost 38 fields on each side of the first reproduction -- compared against nothing,
    with every value identical underneath. The description belongs in `holds_or_removes`.
    """
    import analysis_v3_13

    for arm in ARMS:
        if arm.milestone != "G1" or arm.is_replay:
            continue
        if arm.research_name in analysis_v3_13.VARIANTS:
            continue
        assert arm.name == "solo", (
            f"{arm.name}'s research name {arm.research_name!r} is not a key of "
            f"analysis_v3_13.VARIANTS ({sorted(analysis_v3_13.VARIANTS)}), so nothing it produces "
            f"can be compared to a reference summary")
    assert get_arm("collective_scrambled").research_name == "plastic + noise"
    assert "plastic + noise record" in get_arm("collective_scrambled").holds_or_removes
