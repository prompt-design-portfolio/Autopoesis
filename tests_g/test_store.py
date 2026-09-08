"""The record as an artifact store (G2).

Every record here is built by driving the engine's own `write_mark`, never by synthesising a marks
array. A1.6: the simulation is never mocked; it is the mechanism. A hand-built array would test
this module against an idea of what a store looks like rather than against one.
"""

from __future__ import annotations

import numpy as np
import pytest

import sim_v3_13
from civitas_g.persistence.models import StoreArtifact
from civitas_g.provider.population import CampaignPlan, open_campaign, store_run
from civitas_g.store.persistence import (
    load_record,
    preserves_density_and_sign,
    records_for_run,
    save_control_variants,
    save_record,
)
from civitas_g.store.record import Record, ScrambleMode, record_from_world


def a_record(n_writes: int = 3000, seed: int = 0) -> Record:
    cfg = sim_v3_13.Config(seed=seed, chain=True, record="real")
    world = sim_v3_13.World(cfg, np.random.default_rng(seed))
    world.chain_on = True
    rng = np.random.default_rng(seed + 1)
    for _ in range(n_writes):
        ftype = int(rng.integers(sim_v3_13.N_TYPES))
        k = int(rng.integers(sim_v3_13.N_PREPS))
        y, x = int(rng.integers(cfg.grid)), int(rng.integers(cfg.grid))
        world.write_mark(ftype, k, k == world.mapping[ftype], y, x, cfg)
    return record_from_world(world, arm="collective", t=700, era_index=1,
                             engine_sha256="0" * 64, cfg_digest="test")


@pytest.fixture(scope="module")
def record() -> Record:
    return a_record()


# --------------------------------------------------------------------------- shape and stats

def test_a_record_lifted_from_a_world_is_a_copy_not_a_view():
    """The engine keeps mutating `world.marks` in place — decay runs every step — so a view would
    silently become a snapshot of a later moment."""
    cfg = sim_v3_13.Config(seed=0, chain=True, record="real")
    world = sim_v3_13.World(cfg, np.random.default_rng(0))
    world.chain_on = True
    world.write_mark(0, 0, True, 5, 5, cfg)
    r = record_from_world(world, arm="collective", t=1, era_index=0,
                          engine_sha256="0" * 64, cfg_digest="test")
    before = r.marks[0, world.pi[0], 5, 5]
    world.marks *= 0.5
    assert r.marks[0, world.pi[0], 5, 5] == before


def test_density_matches_the_engines_own_definition(record):
    """`mark_stats` uses `np.abs(marks) > 1e-3`. Agreeing by construction, not by coincidence."""
    live = np.abs(record.marks) > 1e-3
    assert record.density() == pytest.approx(float(live.mean()))


def test_endorsed_is_the_inverse_of_pi(record):
    for k, label in enumerate(record.pi):
        assert record.endorsed()[label] == k


def test_a_record_refuses_a_pi_that_is_not_a_permutation(record):
    with pytest.raises(ValueError, match="not a permutation"):
        Record(marks=record.marks, pi=(0, 0, 1, 2, 3), provenance=record.provenance)


def test_a_record_refuses_marks_of_the_wrong_shape(record):
    """T is pinned; K is read off the marks so a G4 hardened store can exist (see
    tests_g/test_g4.py). The shape a record refuses is therefore the wrong number of FOOD TYPES,
    and a pi that does not permute the K the marks actually carry."""
    with pytest.raises(ValueError, match="food types"):
        Record(marks=np.zeros((2, 5, 4, 4)), pi=record.pi, provenance=record.provenance)
    with pytest.raises(ValueError, match="not a permutation"):
        Record(marks=np.zeros((3, 7, 4, 4)), pi=record.pi, provenance=record.provenance)


# --------------------------------------------------------------------------- B§6: round-trip

def test_the_round_trip_is_byte_identical(record):
    loaded = Record.from_bytes(record.to_bytes(), record.provenance)
    ok, detail = loaded.equals(record)
    assert ok, detail
    assert loaded.pi == record.pi


def test_the_content_hash_is_over_the_marks_not_over_the_compressed_blob(record):
    """zlib output depends on the library version; hashing the blob would make two identical
    records hash differently on two machines."""
    import zlib

    loaded = Record.from_bytes(record.to_bytes(), record.provenance)
    assert loaded.sha256() == record.sha256()
    assert zlib.compress(b"x", 1) != zlib.compress(b"x", 9)      # the thing being avoided


def test_a_blob_from_a_future_format_is_refused_rather_than_reinterpreted(record):
    import zlib

    raw = zlib.decompress(record.to_bytes()).replace(b"\n1\n", b"\n99\n", 1)
    with pytest.raises(ValueError, match="format 99"):
        Record.from_bytes(zlib.compress(raw), record.provenance)


def test_a_blob_that_is_not_a_record_is_refused(record):
    import zlib

    with pytest.raises(ValueError, match="not a civitas-g record"):
        Record.from_bytes(zlib.compress(b"something else\n"), record.provenance)


# --------------------------------------------------------------------------- B§6: scrambled load

def test_the_scramble_preserves_density_and_signs_and_moves_the_labels(record):
    scrambled = record.scrambled(np.random.default_rng(7), ScrambleMode.PER_CELL)
    ok, detail = preserves_density_and_sign(record, scrambled)
    assert ok, detail


def test_a_scramble_that_changed_nothing_is_reported_as_not_a_control(record):
    ok, detail = preserves_density_and_sign(record, record)
    assert not ok and "not a control" in detail


def test_the_per_cell_scramble_destroys_cross_cell_consistency_and_the_global_one_does_not(record):
    """The distinction is load-bearing. A globally permuted store is isomorphic to the real one —
    it says "label sigma(j) marks what label j marked" everywhere — so a population born into it
    has exactly the learning problem it would have had with the real store. It is the assay's arm,
    never the inherited-scrambled control."""
    g = record.scrambled(np.random.default_rng(3), ScrambleMode.GLOBAL)
    p = record.scrambled(np.random.default_rng(3), ScrambleMode.PER_CELL)

    def one_permutation_explains_it(scrambled: Record) -> bool:
        """Is there a single sigma with scrambled[f, i] == record[f, sigma(i)] for every plane?

        Recovered by comparing whole (g, g) label planes rather than per-cell rank order: with a
        mostly-empty store most cells carry ties, and a rank-based recovery would read those ties
        as a permutation that is not there.
        """
        for ftype in range(sim_v3_13.N_TYPES):
            sigma = []
            for i in range(sim_v3_13.N_PREPS):
                match = [j for j in range(sim_v3_13.N_PREPS)
                         if np.array_equal(scrambled.marks[ftype, i], record.marks[ftype, j])]
                if not match:
                    return False
                sigma.append(match[0])
            if sorted(sigma) != list(range(sim_v3_13.N_PREPS)):
                return False
        return True

    assert one_permutation_explains_it(g), \
        "a global scramble must be one relabelling of the whole store"
    assert not one_permutation_explains_it(p), \
        "a per-cell scramble that reduced to one relabelling would be no control at all"


def test_the_hidden_store_is_empty_but_still_the_same_shape(record):
    """Not "no store": the read channels exist and are zero, so the input layout is unchanged."""
    hidden = record.hidden()
    assert hidden.marks.shape == record.marks.shape
    assert not hidden.marks.any()
    assert hidden.density() == 0.0
    assert hidden.pi == record.pi


# --------------------------------------------------------------------------- persistence

@pytest.fixture
def stored_run(sessions, short_run):
    plan = CampaignPlan(kind="test", arms=["collective"], seeds=[0], phase_steps=150)
    with sessions() as session:
        campaign = open_campaign(session, plan)
        run = store_run(session, campaign, short_run)
        session.commit()
        return run.id


def test_a_record_survives_the_database_byte_identically(sessions, stored_run, record):
    with sessions() as session:
        artifact = save_record(session, stored_run, record)
        session.commit()
        artifact_id = artifact.id
    with sessions() as session:
        loaded = load_record(session, artifact_id)
    ok, detail = loaded.equals(record)
    assert ok, detail


def test_the_statistics_are_measured_on_the_way_in(sessions, stored_run, record):
    with sessions() as session:
        artifact = save_record(session, stored_run, record)
        session.commit()
        assert artifact.density == pytest.approx(record.density())
        assert artifact.n_positive == record.sign_counts()["positive"]
        assert artifact.n_negative == record.sign_counts()["negative"]
        assert artifact.content_sha256 == record.sha256()


def test_an_artifact_whose_bytes_moved_is_refused(sessions, stored_run, record):
    """A1.4 keeps the row; it does not make the row trustworthy on its own."""
    with sessions() as session:
        artifact = save_record(session, stored_run, record)
        artifact.content_sha256 = "0" * 64
        session.commit()
        with pytest.raises(ValueError, match="does not match its recorded content hash"):
            load_record(session, artifact.id)


def test_controls_are_stored_beside_the_record_never_in_place_of_it(
        sessions, stored_run, record):
    with sessions() as session:
        artifact = save_record(session, stored_run, record)
        derived = save_control_variants(session, stored_run, artifact, record, seed=11)
        session.commit()
        variants = {a.variant for a in session.query(StoreArtifact).all()}
        assert variants == {"real", "hidden", "scrambled:per_cell"}
        assert all(d.derived_from == artifact.id for d in derived)
        # the parent is untouched
        assert load_record(session, artifact.id).equals(record)[0]


def test_the_scramble_seed_is_recorded_so_the_control_is_reproducible(
        sessions, stored_run, record):
    """A control drawn from an unrecorded random state cannot be reproduced — the `sr_w` lesson,
    applied to a control rather than to a parameter."""
    with sessions() as session:
        artifact = save_record(session, stored_run, record)
        derived = save_control_variants(session, stored_run, artifact, record, seed=11)
        session.commit()
        scrambled = next(d for d in derived if d.variant.startswith("scrambled"))
        assert scrambled.provenance["scramble_seed"] == 11
        assert scrambled.provenance["scramble_mode"] == "per_cell"


def test_records_for_a_run_come_back_in_era_order(sessions, stored_run, record):
    with sessions() as session:
        for era in (2, 0, 1):
            import dataclasses

            prov = dataclasses.replace(record.provenance, era_index=era, t=700 * (era + 1))
            save_record(session, stored_run, Record(record.marks, record.pi, prov))
        session.commit()
        eras = [r.provenance.era_index for r in records_for_run(session, stored_run)]
    assert eras == [0, 1, 2]


def test_provenance_carries_both_mappings_because_a_replay_gets_neither(record):
    """A replay that re-seeds the world does NOT get the mapping the genomes were selected under,
    so it travels with the artifact."""
    assert len(record.provenance.mapping) == sim_v3_13.N_TYPES
    assert record.provenance.engine_sha256
    assert "run_seed" in record.provenance.as_dict()
