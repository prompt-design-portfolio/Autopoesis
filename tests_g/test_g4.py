"""G4 — a world that hardens, one mechanic per milestone.

Compounding is a difference of differences, so most of what can go wrong is in the bookkeeping
rather than in the simulation: the wrong baseline, a second parameter moved, a growth that lives
in the total rather than in the content. These tests hold that bookkeeping.
"""

from __future__ import annotations

import pathlib
from math import perm

import numpy as np
import pytest

from civitas_g.g3 import BArmResult, G3Result, Succession
from civitas_g.g4 import G4Result, mechanic_k
from civitas_g.world import spec
from civitas_g.world.spec import (
    N_PREPS,
    WorldMismatch,
    check_chance_ev_is_zero,
    check_hardened_world,
    world_at_k,
)

# --------------------------------------------------------------------------- the mechanic

def test_raising_k_moves_prep_value_with_it():
    """The invariant is prep_value = (K-1)*prep_fail, which holds the chance EV at zero. Raising K
    and leaving prep_value alone makes a chance preparation worth -0.250 at K=7 -- a harder world
    for a reason that has nothing to do with the mapping space."""
    w = world_at_k(7)
    assert w["n_preps"] == 7
    assert w["prep_value"] == pytest.approx(1.5)
    assert check_chance_ev_is_zero(w) == 0.0


def test_a_mechanic_declares_exactly_what_it_moves():
    m = mechanic_k(7)
    assert set(m.moved) == {"n_preps", "prep_value"}
    assert m.moved["n_preps"] == (5, 7)


def test_a_world_that_moved_a_second_parameter_is_refused():
    """A5: one change per experiment. A mechanic that moves a second parameter is two mechanics
    and its claim line cannot say which one produced the number."""
    with pytest.raises(WorldMismatch, match="no mechanic declares"):
        check_hardened_world(dict(world_at_k(7), spawn_per_patch=9.0))


def test_k_must_exceed_the_number_of_food_types():
    """A mapping sends each type to a DISTINCT preparation, so K <= T is not a harder world, it is
    an impossible one."""
    with pytest.raises(WorldMismatch, match="must exceed T"):
        world_at_k(3)


def test_the_mapping_space_actually_grows():
    assert perm(7, 3) == 210
    assert perm(N_PREPS, 3) == 60
    assert "210" in mechanic_k(7).note and "60" in mechanic_k(7).note


# --------------------------------------------------------------------------- the arithmetic

def _result(nfc, seed=0):
    arms = [BArmResult(arm=name, seed=seed, aligned=True, stale=0.2, stale_n=100, null=0.1,
                       ratio=1.5, nfc_mean=nfc[name], nfc_censored=0.0, nfc_n=50, prep_hit=0.4,
                       pop=300.0, store_sha256=None, store_density=0.3, sym_gain=0.0,
                       claim_window=700, marks_surviving_at_window_end=0.06)
            for name in ("fresh store", "inherited store", "inherited scrambled",
                         "inherited gain-zero")]
    return G3Result(succession=Succession(seed, 100, 2100, True, claim_window=700),
                    a_store_sha256="x", a_mapping=(0, 1, 2), a_pi=(0, 1, 2, 3, 4),
                    b_mapping=(0, 1, 2), arms=arms)


BASE = {"fresh store": 2.035, "inherited store": 1.841,
        "inherited scrambled": 1.972, "inherited gain-zero": 1.960}


def test_compounding_is_a_difference_of_differences():
    """Not "the effect survived a harder world" -- that would follow from the record merely
    continuing to work. The effect must get LARGER."""
    harder = {"fresh store": 2.30, "inherited store": 1.95,
              "inherited scrambled": 2.24, "inherited gain-zero": 2.22}
    r = G4Result(mechanic=mechanic_k(7), seed=0, aligned=True,
                 baseline=_result(BASE), hardened=_result(harder))
    c = r.compounding()
    assert c["content_baseline"] == pytest.approx(-0.131, abs=1e-3)
    assert c["content_hardened"] == pytest.approx(-0.29, abs=1e-3)
    assert c["content_compounding"] == pytest.approx(-0.159, abs=1e-3)
    assert r.verdict()["content_effect_grew"] is True


def test_an_effect_that_shrinks_does_not_compound():
    weaker = {"fresh store": 2.30, "inherited store": 2.20,
              "inherited scrambled": 2.25, "inherited gain-zero": 2.24}
    r = G4Result(mechanic=mechanic_k(7), seed=0, aligned=True,
                 baseline=_result(BASE), hardened=_result(weaker))
    assert r.verdict()["content_effect_grew"] is False
    assert r.verdict()["compounds"] is False


def test_growth_that_lives_in_the_total_and_not_the_content_does_not_compound():
    """G4's second falsification: a harder world making a store more valuable by EXISTING -- more
    marks, more cells covered -- without its labels mattering more. Invisible unless both the
    content and the total are reported."""
    presence_only = {"fresh store": 2.60, "inherited store": 2.05,
                     "inherited scrambled": 2.19, "inherited gain-zero": 2.17}
    r = G4Result(mechanic=mechanic_k(7), seed=0, aligned=True,
                 baseline=_result(BASE), hardened=_result(presence_only))
    c = r.compounding()
    assert abs(c["total_compounding"]) > abs(c["content_compounding"])
    assert r.verdict()["content_carries_the_growth"] is False
    assert r.verdict()["compounds"] is False


def test_controls_that_stop_agreeing_break_the_mechanic():
    """G4's third falsification. If a mechanic pulls the two information-removing arms apart, it
    has introduced something other than label information into the metric and its claim line means
    nothing until that is found."""
    split = {"fresh store": 2.30, "inherited store": 1.95,
             "inherited scrambled": 2.24, "inherited gain-zero": 1.96}
    r = G4Result(mechanic=mechanic_k(7), seed=0, aligned=True,
                 baseline=_result(BASE), hardened=_result(split))
    assert r.verdict()["controls_still_agree"] is False
    assert r.verdict()["compounds"] is False


def test_the_verdict_states_its_own_basis():
    """At one seed these are direction checks, not statistical tests, and the report says so."""
    r = G4Result(mechanic=mechanic_k(7), seed=0, aligned=True,
                 baseline=_result(BASE), hardened=_result(BASE))
    assert "not a statistical test" in r.verdict()["basis"]


def test_a_missing_side_is_nan_rather_than_a_number():
    r = G4Result(mechanic=mechanic_k(7), seed=0, aligned=True, baseline=_result(BASE))
    import math

    assert math.isnan(r.compounding()["content_hardened"])
    assert r.verdict()["compounds"] is False


# ---------------------------------------------------------------------------------------------
# K-blindness. Found by the mechanic-1 smoke test, which could not build its own record.
# ---------------------------------------------------------------------------------------------


def _marks(t, k, g=6):
    rng = np.random.default_rng(0)
    return rng.normal(size=(t, k, g, g))


def _prov():
    from civitas_g.store.record import RecordProvenance
    return RecordProvenance(run_seed=0, arm="collective", t=0, era_index=0,
                            engine_sha256="x" * 64, cfg_digest="y")


def test_a_record_may_carry_a_hardened_k():
    """The store is where G4's first mechanic lands. A `Record` that could only ever be
    K = N_PREPS would make a hardened store unrepresentable -- which is how the smoke test
    failed: `marks is (3, 7, ...) but the world is (3, 5, ...)`."""
    from civitas_g.store.record import Record
    r = Record(marks=_marks(spec.N_TYPES, 7), pi=tuple(range(7)), provenance=_prov())
    assert r.n_preps == 7
    assert sorted(r.endorsed()) == list(range(7))


def test_a_hardened_record_round_trips_and_scrambles_at_its_own_k():
    from civitas_g.store.record import Record, ScrambleMode
    r = Record(marks=_marks(spec.N_TYPES, 7), pi=tuple(range(7)), provenance=_prov())
    back = Record.from_bytes(r.to_bytes(), r.provenance)
    ok, detail = r.equals(back)
    assert ok, detail
    for mode in (ScrambleMode.PER_CELL, ScrambleMode.GLOBAL):
        s = r.scrambled(np.random.default_rng(1), mode)
        assert s.marks.shape == r.marks.shape
        assert s.sign_counts() == r.sign_counts(), mode
        assert abs(s.density() - r.density()) < 1e-12, mode


def test_a_record_still_refuses_a_world_it_is_not_from():
    """T stays pinned -- no mechanic moves the number of food types -- and K must still exceed
    it, or a mapping is not an injection and there is nothing to record."""
    from civitas_g.store.record import Record
    with pytest.raises(ValueError, match="food types"):
        Record(marks=_marks(spec.N_TYPES + 1, 7), pi=tuple(range(7)), provenance=_prov())
    with pytest.raises(ValueError, match="K must exceed T"):
        Record(marks=_marks(spec.N_TYPES, spec.N_TYPES), pi=tuple(range(spec.N_TYPES)),
               provenance=_prov())
    with pytest.raises(ValueError, match="not a permutation"):
        Record(marks=_marks(spec.N_TYPES, 7), pi=tuple(range(5)), provenance=_prov())


def test_the_matched_null_reads_k_off_the_run():
    """A2.2's null is `(1 - hit)/(K - 1)`. `civitas_g.reading.compute` had the same K-blindness
    `check_chance_ev_is_zero` and `matched_stale_null` had: at K = 5 it is the same number, so
    nothing fails until a hardened run makes it quietly wrong."""
    src = pathlib.Path("civitas_g/reading/compute.py").read_text()
    assert 'get("n_preps"' in src, "the null is computed from the module's K, not the run's"
    assert "(1.0 - hit) / (k - 1)" in src


# ---------------------------------------------------------------------------------------------
# Mechanic 1's confound is a choice, not a necessity (G4 §2.1)
# ---------------------------------------------------------------------------------------------


def test_both_hold_modes_are_ev_neutral_and_differ():
    """EV-neutrality fixes `prep_value = (K-1) * prep_fail`, which constrains the RATIO and not
    either term. So raising K can be absorbed by raising the reward or by lowering the penalty,
    and mechanic 1 took the first without there being a reason to."""
    fail_mode = spec.world_at_k(7, hold="fail")
    value_mode = spec.world_at_k(7, hold="value")
    for w in (fail_mode, value_mode):
        spec.check_chance_ev_is_zero(w)          # raises if not exactly zero
        assert w["n_preps"] == 7
    assert fail_mode["prep_value"] == 1.5 and fail_mode["prep_fail"] == 0.25
    assert value_mode["prep_value"] == 1.0
    assert value_mode["prep_fail"] == pytest.approx(1.0 / 6)
    assert fail_mode != value_mode


def test_the_default_hold_mode_is_unchanged_so_mechanic_1_stays_reproducible():
    """`hold` was added after mechanic 1 ran. If its default moved, the stored mechanic-1 results
    would silently stop matching the world their own manifest names."""
    assert spec.world_at_k(7) == spec.world_at_k(7, hold="fail")


def test_an_unknown_hold_mode_is_refused():
    with pytest.raises(spec.WorldMismatch, match="hold must be one of"):
        spec.world_at_k(7, hold="whatever")


def test_both_hold_modes_pass_the_hardened_world_check_by_declaration():
    """Not by accident. `HARDENING_PARAMETERS['K']` names both economics terms; before it did,
    the value-holding variant passed only by borrowing the economics diagnostic's allowance."""
    assert "prep_fail" in spec.HARDENING_PARAMETERS["K"]
    assert "prep_value" in spec.HARDENING_PARAMETERS["K"]
    for hold in spec.HOLD_MODES:
        moved = spec.check_hardened_world(spec.world_at_k(7, hold=hold))
        assert moved["n_preps"] == (5, 7)
