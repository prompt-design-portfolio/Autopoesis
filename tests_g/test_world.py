"""The world of record, the adapter, and the arms."""

from __future__ import annotations

import pytest

from civitas_g.world import adapter, arms, spec


def test_the_world_of_record_is_the_analysis_world_not_the_config_defaults():
    """F7. `Config()`'s defaults are v3.10 leftovers and build a different experiment."""
    import sim_v3_13

    defaults = sim_v3_13.Config()
    assert spec.WORLD["prep_value"] == 1.0 and spec.WORLD["prep_fail"] == 0.25
    assert (defaults.prep_value, defaults.prep_fail) == (1.5, 0.5)
    # the reason this matters, stated as the number it changes
    chance_ev_of_defaults = (defaults.prep_value / spec.N_PREPS
                             - defaults.prep_fail * (spec.N_PREPS - 1) / spec.N_PREPS)
    assert chance_ev_of_defaults == pytest.approx(-0.1)
    assert spec.check_chance_ev_is_zero() == 0.0


def test_a_world_that_has_moved_is_refused_rather_than_measured():
    moved = dict(spec.WORLD, prep_every=350)
    with pytest.raises(spec.WorldMismatch, match="prep_every"):
        spec.check_world(moved)


def test_the_chance_ev_invariant_is_checked_not_assumed():
    with pytest.raises(spec.WorldMismatch, match=r"prep_value"):
        spec.check_chance_ev_is_zero(dict(spec.WORLD, prep_value=1.5))


def test_the_matched_null_for_a_stale_mark_is_not_one_over_k():
    """A2.2. An agent that knows the answer never agrees with a stale mark, whatever it reads."""
    assert spec.matched_stale_null(0.0) == pytest.approx(spec.CHANCE * 1.25)
    assert spec.matched_stale_null(0.5) == pytest.approx(0.125)
    assert spec.matched_stale_null(1.0) == 0.0
    assert spec.matched_stale_null(0.5) != spec.CHANCE


def test_the_observation_layout_matches_the_engine():
    adapter.check_observation_layout()
    assert adapter.DEAD_INPUTS == (16, 17, 18, 19, 24)
    # 5 dirsum blocks + 5 here values + energy + the read channels
    assert len(adapter.OBS_LAYOUT) == 12


def test_reading_the_record_is_not_an_action():
    assert adapter.READ_IS_AN_ACTION is False
    assert len(adapter.ACTIONS) == spec.N_ACTIONS
    assert [a.index for a in adapter.ACTIONS] == list(range(spec.N_ACTIONS))
    assert sum(1 for a in adapter.ACTIONS if not a.live_in_phase_one) == spec.N_PREPS


def test_the_modulator_is_a_table_of_seven_events_four_of_them_signed():
    assert len(adapter.MODULATOR_EVENTS) == 7
    assert sorted(e.m for e in adapter.MODULATOR_EVENTS) == [-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    inedible = adapter.MODULATOR_BY_EVENT["eat_inedible"]
    assert inedible == 0.0
    assert not next(e for e in adapter.MODULATOR_EVENTS
                    if e.event == "eat_inedible").consumes_cell


@pytest.mark.slow
def test_the_modulator_does_not_track_the_economics():
    """D7. If it did, G4's first economics change would silently move the learner."""
    ok, detail = adapter.modulator_is_independent_of_economics()
    assert ok, detail


def test_no_self_echo_is_a_property_of_the_design():
    ok, detail = adapter.check_no_self_echo()
    assert ok, detail


@pytest.mark.parametrize("broken", [
    "marks_are_labels", "pi_redrawn_every_era", "read_is_observation", "read_is_here_only",
])
def test_each_clause_of_the_leak_rule_is_enforced_separately(broken):
    kw = dict(marks_are_labels=True, pi_redrawn_every_era=True,
              read_is_observation=True, read_is_here_only=True)
    kw[broken] = False
    with pytest.raises(adapter.DomainLeak):
        adapter.check_no_leak(**kw)


def test_collective_scrambled_is_the_noise_record_not_the_modulator_scramble():
    """They are different controls with confusingly similar names (A2.3)."""
    kw = arms.get_arm("collective_scrambled").config_kwargs()
    assert kw["record"] == "noise"
    assert kw.get("scramble", False) is False


def test_memory_reset_keeps_the_read_channels_and_zeroes_them():
    kw = arms.get_arm("memory_reset").config_kwargs()
    assert kw["record"] == "none"
    # the input layout is the same in every arm, so no genome sees a different world shape
    assert spec.N_IN == spec.READ + spec.N_PREPS


def test_the_frozen_assay_is_a_procedure_not_a_world_to_run():
    with pytest.raises(arms.ArmUnavailable, match="frozen replay"):
        arms.get_arm("collective_frozen").config_kwargs()


def test_the_g5_reference_arm_refuses_to_run_at_g1():
    with pytest.raises(arms.ArmUnavailable, match="G5"):
        arms.get_arm("reference").config_kwargs()


def test_a_retired_arm_is_named_as_retired_not_as_unknown():
    for name in arms.RETIRED_ARMS:
        with pytest.raises(arms.ArmUnavailable, match="retired"):
            arms.get_arm(name)


def test_an_unknown_arm_lists_the_alternatives():
    with pytest.raises(KeyError, match="unknown arm"):
        arms.get_arm("collectiv")


def test_slow_labels_outlive_the_mapping_by_three_eras():
    kw = arms.get_arm("plastic + record (slow)").config_kwargs()
    assert kw["label_every"] == 3 * spec.WORLD["prep_every"]
    clocks = spec.clocks_of(kw)
    assert clocks.pi_epochs_per_mapping == 3
