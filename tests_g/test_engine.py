"""A1.8: the compute engine is untouched, and Civitas has no per-step surface."""

from __future__ import annotations

import pytest

import sim_v3_13
from civitas_g.world import engine as E


def test_civitas_holds_no_per_step_surface():
    ok, detail = E.touches_world_only_at_era_boundaries()
    assert ok, detail


def test_the_engines_own_functions_are_unreplaced():
    assert sim_v3_13.resolve_action.__module__ == "sim_v3_13"
    assert sim_v3_13.run.__module__ == "sim_v3_13"
    assert sim_v3_13.World.__module__ == "sim_v3_13"


def test_a_run_spec_lays_the_world_of_record_over_the_config():
    spec = E.build_run_spec("collective", seed=0, phase_steps=300)
    assert spec.kwargs["prep_value"] == 1.0 and spec.kwargs["prep_fail"] == 0.25
    assert spec.kwargs["prep_every"] == 700
    assert spec.kwargs["record"] == "real"


def test_both_phases_scale_together():
    """Scaling one phase would change the ratio of the food-only phase to the chain phase, which
    is a different staging and therefore a different experiment."""
    phases = E.build_run_spec("collective", 0, 500).phases()
    assert [p["n_steps"] for p in phases] == [500, 500]
    assert [p["chain"] for p in phases] == [False, True]


def test_g3s_founder_seeding_is_refused_at_g1():
    spec = E.build_run_spec("collective", 0, 100)
    with pytest.raises(E.EngineRefusal, match="G3"):
        E.run(spec, init_genomes=[object()])


def test_an_arm_that_is_not_a_world_cannot_produce_a_run_spec():
    from civitas_g.world.arms import ArmUnavailable

    with pytest.raises(ArmUnavailable):
        E.build_run_spec("collective_frozen", 0, 100)


def test_a_run_records_the_engine_hash_it_used(short_run):
    """A1.8: its hash is in every manifest."""
    from civitas_g.manifest import engine_identity

    assert short_run.engine_sha256 == engine_identity()["sim_v3_13.py"]
    assert len(short_run.engine_sha256) == 64


def test_the_run_produced_era_rows_and_a_final_mapping(short_run):
    assert short_run.log
    assert len(short_run.final_mapping) == sim_v3_13.N_TYPES
    assert len(set(short_run.final_mapping)) == sim_v3_13.N_TYPES, \
        "a mapping sends each food type to a DISTINCT preparation"
