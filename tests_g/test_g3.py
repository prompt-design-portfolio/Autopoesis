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


def _fake_arm(name, ratio, nfc, hit):
    from civitas_g.g3 import BArmResult

    return BArmResult(arm=name, seed=0, aligned=True, stale=0.1, stale_n=100, null=0.1,
                      ratio=ratio, nfc_mean=nfc, nfc_censored=0.0, nfc_n=50, prep_hit=hit,
                      pop=300.0, store_sha256=None, store_density=None, sym_gain=0.0)


def _fake_result(arms):
    from civitas_g.g3 import G3Result

    return G3Result(succession=Succession(0, 100, 100, True), a_store_sha256="x",
                    a_mapping=(0, 1, 2), a_pi=(0, 1, 2, 3, 4), b_mapping=(0, 1, 2), arms=arms)


def test_the_two_claim_lines_have_different_baselines_and_that_is_deliberate():
    """B§5.2 says both go against `fresh store`, but `fresh store` has NO marks, so it has no
    stale-mark events at all -- its ratio is undefined, not small. And the pre-check showed the
    ratio's own null is not enough either: `inherited gain-zero` came out at 1.48 with sym_gain
    pinned to exactly zero, so above 1 is reachable without reading anything."""
    r = _fake_result([_fake_arm("fresh store", float("nan"), 2.0, 0.30),
                      _fake_arm("inherited store", 1.8, 1.6, 0.40),
                      _fake_arm("inherited gain-zero", 1.5, 1.9, 0.31)])
    lines = r.claim_lines()
    assert "fresh store" not in lines
    assert lines["inherited store"]["nfc_vs_fresh"] == pytest.approx(-0.4)
    assert lines["inherited store"]["prep_hit_vs_fresh"] == pytest.approx(0.10)
    # the stale ratio is against the arm that has the same store and cannot read it
    assert lines["inherited store"]["stale_ratio_vs_unreading"] == pytest.approx(0.3)
    # and the unreading arm has no such line of its own
    assert "stale_ratio_vs_unreading" not in lines["inherited gain-zero"]


def test_a_nan_baseline_never_silently_becomes_a_number():
    """`fresh store`'s ratio is nan. Subtracting it would give nan, and a reader could mistake a
    failed comparison for an absent one."""
    r = _fake_result([_fake_arm("fresh store", float("nan"), 2.0, 0.30),
                      _fake_arm("inherited store", 1.8, 1.6, 0.40),
                      _fake_arm("inherited gain-zero", 1.5, 1.9, 0.31)])
    for lines in r.claim_lines().values():
        for name, value in lines.items():
            assert not np.isnan(value), f"{name} is nan"


@pytest.mark.parametrize("gp,expect", [
    ({"eras": 1, "sd": 0.0, "z": 0.0}, "NOT AVAILABLE"),
    ({"eras": 5, "sd": 0.0, "z": 0.0}, "NOT AVAILABLE"),
    ({"eras": 5, "sd": 0.1, "z": float("nan")}, "NOT AVAILABLE"),
    ({"eras": 5, "sd": 0.1, "z": 1.2}, "PASS"),
    ({"eras": 5, "sd": 0.1, "z": 2.5}, "GATE R FIRES"),
])
def test_a_degenerate_gate_r_is_not_a_pass(gp, expect):
    """With one pi-epoch there is nothing to permute across: every permutation returns the same
    pooled value, the null equals the observation, sd is exactly zero and z comes out 0.00.
    Reporting that as PASS reports an absent measurement as a passed gate."""
    from civitas_g.g3 import _gate_r_verdict

    assert _gate_r_verdict(gp).startswith(expect)


def test_the_claim_window_and_the_surviving_fraction_travel_with_the_result():
    """A claim read over a window where almost none of A's record still stands is a claim about
    decay. The number a reader needs to see that is on the row."""
    from civitas_g.g3 import BArmResult

    row = BArmResult(arm="x", seed=0, aligned=True, stale=0.1, stale_n=1, null=0.1, ratio=1.0,
                     nfc_mean=1.0, nfc_censored=0.0, nfc_n=1, prep_hit=0.3, pop=1.0,
                     store_sha256=None, store_density=None, sym_gain=0.0,
                     claim_window=700, marks_surviving_at_window_end=0.996 ** 700)
    assert row.claim_window == 700
    assert row.marks_surviving_at_window_end == pytest.approx(0.0605, abs=1e-3)


# --------------------------------------------------------------------------- acceptance

def _result_with(nfc, stale, controls, seed=0, alignment=True, gate="PASS"):
    from civitas_g.g3 import BArmResult, G3Result

    def arm(name, ratio, nfc_mean):
        return BArmResult(arm=name, seed=seed, aligned=alignment, stale=0.2, stale_n=100,
                          null=0.1, ratio=ratio, nfc_mean=nfc_mean, nfc_censored=0.0, nfc_n=50,
                          prep_hit=0.4, pop=300.0, store_sha256=None, store_density=0.3,
                          sym_gain=0.0, claim_window=700,
                          marks_surviving_at_window_end=0.06)

    base = 2.0
    r = G3Result(succession=Succession(seed, 100, 2100, alignment, claim_window=700),
                 a_store_sha256="x", a_mapping=(0, 1, 2), a_pi=(0, 1, 2, 3, 4),
                 b_mapping=(0, 1, 2),
                 arms=[arm("fresh store", float("nan"), base),
                       arm("inherited store", 1.5 + stale, base + nfc),
                       arm("inherited scrambled", 1.5, base + controls[0]),
                       arm("inherited gain-zero", 1.5, base + controls[1])])
    r.gate_r = {"verdict": gate}
    return r


def _seed(nfc, ratio, seed=0, aligned=True, gate="PASS"):
    """A result with the four arms' nfc and stale ratios set directly."""
    from civitas_g.g3 import BArmResult, G3Result

    arms = [BArmResult(arm=name, seed=seed, aligned=aligned, stale=0.2, stale_n=100, null=0.1,
                       ratio=ratio.get(name, float("nan")), nfc_mean=nfc[name],
                       nfc_censored=0.0, nfc_n=50, prep_hit=0.4, pop=300.0, store_sha256=None,
                       store_density=0.3, sym_gain=0.0, claim_window=700,
                       marks_surviving_at_window_end=0.06)
            for name in B_ARMS]
    r = G3Result(succession=Succession(seed, 100, 2100, aligned, claim_window=700),
                 a_store_sha256="x", a_mapping=(0, 1, 2), a_pi=(0, 1, 2, 3, 4),
                 b_mapping=(0, 1, 2), arms=arms)
    r.gate_r = {"verdict": gate}
    return r


#: the numbers seed 0 actually produced, aligned
SEED0_NFC = {"fresh store": 2.035, "inherited store": 1.841,
             "inherited scrambled": 1.972, "inherited gain-zero": 1.960}
SEED0_RATIO = {"fresh store": float("nan"), "inherited store": 1.79,
               "inherited scrambled": 1.44, "inherited gain-zero": 1.48}


def test_one_seed_is_a_pre_check_not_an_acceptance():
    """B§5.2 asks for 3/3. Fewer is a pre-check and says so."""
    from civitas_g.g3 import acceptance

    a = acceptance([_seed(SEED0_NFC, SEED0_RATIO)])
    assert not a["accepted"]
    assert "pre-check" in a["why_not"]


def test_the_two_contrasts_each_hold_everything_but_one_factor():
    """content varies what the labels SAY; reading varies whether the agent can SEE them. Neither
    is the comparison against `fresh store`, which differs in the presence of marks AND their
    content at once."""
    from civitas_g.g3 import acceptance

    d = acceptance([_seed(SEED0_NFC, SEED0_RATIO)])["per_seed"][0]
    assert d["content_vs_scrambled"] == pytest.approx(-0.131, abs=1e-3)
    assert d["reading_vs_gain_zero"] == pytest.approx(-0.119, abs=1e-3)
    assert d["total_vs_fresh"] == pytest.approx(-0.194, abs=1e-3)


def test_the_two_ways_of_removing_the_information_must_agree():
    """The load-bearing check, and it is not in B§5.2's sentence. `inherited scrambled` and
    `inherited gain-zero` remove the same thing by different routes -- meaningless labels, and
    labels that cannot be seen -- so a design measuring what it thinks it is must land them
    together. If they disagree, something other than label information is moving the metric and
    neither contrast means what it says."""
    from civitas_g.g3 import acceptance

    d = acceptance([_seed(SEED0_NFC, SEED0_RATIO)])["per_seed"][0]
    assert d["controls_disagreement"] == pytest.approx(0.012, abs=1e-3)
    assert d["controls_agree"] is True

    # a run where the two controls diverge as much as the effect itself is not accepted
    split = dict(SEED0_NFC, **{"inherited gain-zero": 1.84})
    a = acceptance([_seed(split, SEED0_RATIO, seed=s) for s in (0, 1, 2)])
    assert not a["accepted"]
    assert a["per_seed"][0]["controls_agree"] is False


def test_three_seeds_of_the_measured_shape_are_accepted():
    from civitas_g.g3 import acceptance

    a = acceptance([_seed(SEED0_NFC, SEED0_RATIO, seed=s) for s in (0, 1, 2)])
    assert a["accepted"] and a["passing_seeds"] == [0, 1, 2]


def test_a_treatment_that_beats_only_one_control_is_not_accepted():
    """Both contrasts have to move: an effect that shows up against the scrambled labels but not
    against the arm that cannot read is not an effect of reading labels."""
    from civitas_g.g3 import acceptance

    only_content = dict(SEED0_NFC, **{"inherited gain-zero": 1.80})
    a = acceptance([_seed(only_content, SEED0_RATIO, seed=s) for s in (0, 1, 2)])
    assert not a["accepted"]
    assert a["per_seed"][0]["reading_moves_the_right_way"] is False


def test_a_treatment_moving_the_wrong_way_is_not_accepted():
    from civitas_g.g3 import acceptance

    worse = dict(SEED0_NFC, **{"inherited store": 2.10})
    assert not acceptance([_seed(worse, SEED0_RATIO, seed=s) for s in (0, 1, 2)])["accepted"]


def test_the_stale_line_must_move_against_both_controls_too():
    from civitas_g.g3 import acceptance

    flat = dict(SEED0_RATIO, **{"inherited store": 1.40})
    a = acceptance([_seed(SEED0_NFC, flat, seed=s) for s in (0, 1, 2)])
    assert not a["accepted"]
    assert a["per_seed"][0]["stale_moves_the_right_way"] is False


def test_the_agreement_threshold_is_reported_beside_the_verdict():
    from civitas_g.g3 import acceptance

    assert acceptance([_seed(SEED0_NFC, SEED0_RATIO)])["agree_within"] == 0.35


def test_the_surviving_fraction_travels_into_the_acceptance():
    """So a claim read where almost none of A's record still stood cannot be read as a claim
    about transmission without the reader seeing that."""
    from civitas_g.g3 import acceptance

    a = acceptance([_result_with(-0.2, 0.3, (-0.02, -0.02), seed=s) for s in (0, 1, 2)])
    assert a["per_seed"][0]["marks_surviving"] == pytest.approx(0.06)


def test_a_succession_and_its_arms_persist(sessions):
    from civitas_g.g3 import save_succession
    from civitas_g.persistence.models import SuccessionArm, SuccessionRun
    from civitas_g.provider.population import CampaignPlan, open_campaign

    plan = CampaignPlan(kind="g3", arms=["collective"], seeds=[0], phase_steps=100)
    r = _result_with(-0.2, 0.3, (-0.02, -0.02))
    with sessions() as session:
        campaign = open_campaign(session, plan)
        row = save_succession(session, campaign.id, r, engine_sha256="a" * 64)
        session.commit()
        row_id = row.id
    with sessions() as session:
        stored = session.get(SuccessionRun, row_id)
        assert stored.alignment == "aligned"
        assert stored.a_store_sha256 == "x"
        assert stored.claim_window == 700
        assert stored.marks_surviving == pytest.approx(0.06)
        arms = session.query(SuccessionArm).filter_by(succession_id=row_id).all()
        assert {a.arm for a in arms} == set(B_ARMS)
        treat = next(a for a in arms if a.arm == "inherited store")
        assert treat.claim_lines["nfc_vs_fresh"] == pytest.approx(-0.2)


def test_a_nan_statistic_is_stored_as_null_not_as_a_number(sessions):
    """`fresh store`'s ratio is nan -- an absent measurement. A float column would have turned it
    into whatever nan round-trips to."""
    from civitas_g.g3 import save_succession
    from civitas_g.persistence.models import SuccessionArm
    from civitas_g.provider.population import CampaignPlan, open_campaign

    plan = CampaignPlan(kind="g3", arms=["collective"], seeds=[0], phase_steps=100)
    with sessions() as session:
        campaign = open_campaign(session, plan)
        row = save_succession(session, campaign.id, _result_with(-0.2, 0.3, (-0.02, -0.02)))
        session.commit()
        fresh = session.query(SuccessionArm).filter_by(
            succession_id=row.id, arm="fresh store").one()
        assert fresh.ratio is None


def test_the_gate_r_verdict_is_text_so_not_available_survives(sessions):
    """A boolean column would have forced NOT AVAILABLE into a pass or a fail."""
    from civitas_g.g3 import save_succession
    from civitas_g.persistence.models import SuccessionRun
    from civitas_g.provider.population import CampaignPlan, open_campaign

    plan = CampaignPlan(kind="g3", arms=["collective"], seeds=[0], phase_steps=100)
    r = _result_with(-0.2, 0.3, (-0.02, -0.02), gate="NOT AVAILABLE: 1 pi-epoch(s).")
    with sessions() as session:
        campaign = open_campaign(session, plan)
        row = save_succession(session, campaign.id, r)
        session.commit()
        assert "NOT AVAILABLE" in session.get(SuccessionRun, row.id).gate_r_verdict


# --------------------------------------------------------------------------- the assay on B

def test_the_assay_on_b_reports_unavailable_rather_than_raising_when_b_is_too_short():
    """A B shorter than one era has no boundary to freeze. That is a fact about the run, not a
    failure of the assay, and it is reported as one."""
    from civitas_g.g3 import assay_on_b

    out = assay_on_b({"era_snaps": [], "store_snaps": [], "cfg": {"seed": 0}})
    assert out["available"] is False
    assert "shorter than one era" in out["why"]


def test_the_assay_on_b_uses_the_FIRST_boundary():
    """B inherits A's record and its own pi, and both are replaced at that first remap. After it,
    the population being frozen has lived an era under a mapping A never saw, with a record A
    never wrote -- so the first boundary is the only moment the assay asks about the inheritance
    at all."""
    import inspect

    from civitas_g import g3

    source = inspect.getsource(g3.assay_on_b)
    assert "stores[0]" in source
    assert "snap=0" in source
