"""G3 — the store outlives the run.

The claim is a *between-arm* difference, so the property everything rests on is that B's four arms
differ in the store and in nothing else. Most of these tests are about that property rather than
about any number.
"""

from __future__ import annotations

import numpy as np
import pytest

import sim_v3_13
from civitas_g.g3 import B_ARMS, Succession, b_founders_carry_no_h, store_for_arm
from civitas_g.store.record import Record, RecordProvenance, ScrambleMode


@pytest.fixture(scope="module")
def record() -> Record:
    """A record built by driving the engine's own write_mark. Cheap, and still the mechanism."""
    cfg = sim_v3_13.Config(seed=0, chain=True, record="real")
    world = sim_v3_13.World(cfg, np.random.default_rng(0))
    world.chain_on = True
    rng = np.random.default_rng(1)
    for _ in range(3000):
        ftype = int(rng.integers(sim_v3_13.N_TYPES))
        k = int(rng.integers(sim_v3_13.N_PREPS))
        world.write_mark(ftype, k, k == world.mapping[ftype],
                         int(rng.integers(cfg.grid)), int(rng.integers(cfg.grid)), cfg)
    return Record(marks=np.array(world.marks, dtype="<f8"),
                  pi=tuple(int(x) for x in world.pi),
                  provenance=RecordProvenance(
                      run_seed=0, arm="collective", t=1, era_index=0,
                      engine_sha256="0" * 64, cfg_digest="test",
                      mapping=tuple(int(x) for x in world.mapping)))


# --------------------------------------------------------------------------- B§6

def test_b_founders_carry_no_h():
    """G3's whole claim is that nothing but the externalised record crosses."""
    ok, detail = b_founders_carry_no_h()
    assert ok, detail


def test_snapshot_saves_no_learned_state():
    """`snapshot()`: "Genomes only: innate weights and genes. Nothing learned (H) is saved."."""
    assert not ({"H1", "H2", "e1", "e2"} & set(sim_v3_13.GENOME))


def test_the_engine_wrapper_refuses_to_seed_a_population_from_another():
    from civitas_g.world.engine import EngineRefusal, build_run_spec
    from civitas_g.world.engine import run as run_engine

    with pytest.raises(EngineRefusal, match="never seeds a population"):
        run_engine(build_run_spec("collective", 0, 100), init_genomes=[{}])


# --------------------------------------------------------------------------- the arms

def test_there_are_four_arms_and_fresh_store_is_the_baseline():
    assert len(B_ARMS) == 4
    assert B_ARMS[0] == "fresh store"


def test_fresh_store_is_handed_a_hidden_store_not_nothing(record):
    """A matched nuisance parameter. With `init_store=None` the world draws its own pi, so the
    baseline would differ from the treatment in the LABEL PERMUTATION as well as in the marks, and
    B's own writes would land in different channels from the inherited arms'. pi carries no
    information about the world -- it is redrawn every era, which is what makes meaning
    non-inheritable -- so holding it constant costs nothing and removes a difference that is not
    the one being measured."""
    store, extra = store_for_arm(record, "fresh store")
    assert extra == {}
    assert not store["marks"].any(), "the baseline must carry no marks"
    assert store["pi"] == record.pi, "the baseline must carry the same pi as the treatment"


def test_every_arm_is_handed_the_same_pi(record):
    """So a difference between arms is a difference in the MARKS and in nothing else."""
    pis = {arm: store_for_arm(record, arm)[0]["pi"] for arm in B_ARMS}
    assert len(set(pis.values())) == 1, pis


def test_inherited_store_is_handed_the_record_unchanged(record):
    store, extra = store_for_arm(record, "inherited store")
    assert np.array_equal(store["marks"], record.marks)
    assert store["pi"] == record.pi
    assert extra == {}


def test_gain_zero_is_handed_the_SAME_store_and_loses_only_the_reading(record):
    """A2.3. The store still changes the world by existing -- channels carry values, cells hold
    state, decay runs -- and only the reading is disabled. Removing the store instead would change
    the world, and the arm would then differ in two ways at once."""
    inherited, _ = store_for_arm(record, "inherited store")
    gain_zero, extra = store_for_arm(record, "inherited gain-zero")
    assert np.array_equal(gain_zero["marks"], inherited["marks"])
    assert gain_zero["pi"] == inherited["pi"]
    assert extra == {"sym_gain_lock": True}


def test_scrambled_preserves_density_and_signs_and_destroys_the_label(record):
    from civitas_g.store.persistence import preserves_density_and_sign

    store, _ = store_for_arm(record, "inherited scrambled", scramble_seed=3)
    scrambled = Record(marks=store["marks"], pi=store["pi"], provenance=record.provenance)
    ok, detail = preserves_density_and_sign(record, scrambled)
    assert ok, detail


def test_the_scramble_is_per_cell_not_global(record):
    """A global permutation is isomorphic to the real store: a population born into it faces
    exactly the learning problem the real store poses, so the arm would read as a null while
    information had in fact passed."""
    store, _ = store_for_arm(record, "inherited scrambled", scramble_seed=3)
    expected = record.scrambled(np.random.default_rng(3), ScrambleMode.PER_CELL)
    assert np.array_equal(store["marks"], expected.marks)
    global_ = record.scrambled(np.random.default_rng(3), ScrambleMode.GLOBAL)
    assert not np.array_equal(store["marks"], global_.marks)


def test_an_unknown_arm_is_refused(record):
    with pytest.raises(ValueError, match="unknown B arm"):
        store_for_arm(record, "inherited something")


def test_the_scramble_seed_is_a_recorded_parameter(record):
    a, _ = store_for_arm(record, "inherited scrambled", scramble_seed=11)
    b, _ = store_for_arm(record, "inherited scrambled", scramble_seed=11)
    c, _ = store_for_arm(record, "inherited scrambled", scramble_seed=12)
    assert np.array_equal(a["marks"], b["marks"])
    assert not np.array_equal(a["marks"], c["marks"])


# --------------------------------------------------------------------------- matching

def test_init_mapping_does_not_move_the_rng_stream():
    """The property the whole claim rests on: B's arms differ in the store and in nothing else.
    init_mapping overwrites a draw rather than skipping one."""
    kw = dict(seed=5, chain=True, record="real", prep_every=300, log_every=50)
    phases = [dict(n_steps=300, chain=True)]
    plain = sim_v3_13.run(sim_v3_13.Config(**kw), verbose=False, phases=phases)
    drawn = tuple(plain["log"][0]["mapping"])
    same = sim_v3_13.run(sim_v3_13.Config(**kw), verbose=False, phases=phases,
                         init_mapping=drawn)
    shared = set(plain["log"][0]) & set(same["log"][0])
    differing = [k for a, b in zip(plain["log"], same["log"], strict=True) for k in shared
                 if not np.array_equal(np.asarray(a[k], dtype=object),
                                       np.asarray(b[k], dtype=object))
                 and not (isinstance(a[k], float) and isinstance(b[k], float)
                          and a[k] != a[k] and b[k] != b[k])]
    assert not differing, f"init_mapping moved the RNG stream: {sorted(set(differing))}"


def test_sym_gain_lock_consumes_no_rng():
    """The gain-zero arm must be matched with the others step for step."""
    import inspect

    source = inspect.getsource(sim_v3_13.Agent.__init__)
    assert "sym_gain_lock" in source
    line = next(ln for ln in source.splitlines() if "sym_gain_lock" in ln)
    assert "rng" not in line, f"sym_gain_lock draws from the RNG: {line.strip()!r}"


def test_the_alignment_is_a_recorded_parameter():
    aligned = Succession(seed=0, a_phase_steps=100, b_steps=100, aligned=True)
    misaligned = Succession(seed=0, a_phase_steps=100, b_steps=100, aligned=False)
    assert aligned.alignment == "aligned"
    assert misaligned.alignment == "misaligned"


# --------------------------------------------------------------------------- the reading

def test_the_matched_null_is_not_one_over_k():
    """A2.2. An agent that knows the correct preparation never agrees with a stale mark, whatever
    it reads, so 1/K is the wrong null for the stale cell."""
    from civitas_g.world.spec import CHANCE, matched_stale_null

    assert matched_stale_null(0.6) == pytest.approx(0.1)
    assert matched_stale_null(0.6) != CHANCE


def test_claim_lines_are_stated_against_fresh_store():
    from civitas_g.g3 import BArmResult, G3Result

    def arm(name, ratio, nfc, hit):
        return BArmResult(arm=name, seed=0, aligned=True, stale=0.1, stale_n=100, null=0.1,
                          ratio=ratio, nfc_mean=nfc, nfc_censored=0.0, nfc_n=50, prep_hit=hit,
                          pop=300.0, store_sha256=None, store_density=None, sym_gain=0.0)

    r = G3Result(succession=Succession(0, 100, 100, True), a_store_sha256="x",
                 a_mapping=(0, 1, 2), a_pi=(0, 1, 2, 3, 4), b_mapping=(0, 1, 2),
                 arms=[arm("fresh store", 1.0, 2.0, 0.30),
                       arm("inherited store", 1.5, 1.6, 0.40)])
    lines = r.claim_lines()
    assert "fresh store" not in lines
    assert lines["inherited store"]["stale_ratio_vs_fresh"] == pytest.approx(0.5)
    assert lines["inherited store"]["nfc_vs_fresh"] == pytest.approx(-0.4)
    assert lines["inherited store"]["prep_hit_vs_fresh"] == pytest.approx(0.10)
