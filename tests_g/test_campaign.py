"""The campaign report: fixed criteria, and the distinction it must never collapse.

The report exists because `g3.acceptance`'s coherence criterion scales with the effect
(`docs/G3_WRITEUP.md` §5.6). Its whole value is provenance — which seeds were read when the
criteria were fixed, and which criterion cannot be evaluated at all — so that is what is tested.
"""

from __future__ import annotations

import pytest

from civitas_g.campaign import (
    CRITERIA,
    SEEDS_SEEN_WHEN_WRITTEN,
    AbsolutePreparations,
    RelativeToEffect,
    SamplingNoise,
    summarise,
)


def _cell(seed, content, gap, alignment="aligned", reading=None):
    return {"seed": seed, "alignment": alignment, "content": content,
            "reading": content if reading is None else reading, "total": content,
            "gap": gap, "stale_content": 0.0, "stale_reading": 0.0, "fresh_hit": 0.5,
            "noise": None}


def test_the_gate_criterion_fails_a_seed_whose_controls_are_closest():
    """§5.6's defect, as an executable statement rather than a paragraph. Seed 1's controls sit
    0.0034 apart and seed 0's 0.0111, and it is seed 1 that the gate rejects."""
    gate = RelativeToEffect("gate", fraction=0.35, note="")
    big_effect_worse_gap = gate.agrees(gap=0.0111, content=-0.1305, reading=-0.119, noise=None)
    no_effect_better_gap = gate.agrees(gap=-0.0034, content=-0.0049, reading=-0.008, noise=None)
    assert big_effect_worse_gap is True
    assert no_effect_better_gap is False, (
        "the criterion has been changed so it no longer counts effect size twice. That is a "
        "legitimate change and it is the project owner's to make -- but it changes a gate, so it "
        "must be made deliberately and this test updated with the reason, not silently.")


def test_an_absolute_criterion_does_not_care_how_big_the_effect_is():
    crit = AbsolutePreparations("abs", limit=0.02, note="")
    assert crit.agrees(gap=0.01, content=-0.13, reading=-0.12, noise=None) is True
    assert crit.agrees(gap=0.01, content=-0.005, reading=-0.008, noise=None) is True
    assert crit.agrees(gap=0.03, content=-0.13, reading=-0.12, noise=None) is False


def test_the_noise_criterion_reports_not_computable_rather_than_agreeing():
    """The distinction that must never collapse. No stored run carries the second moment of
    `nfc_mean`, so this criterion has no answer -- and 'no answer' rendered as 'agree' would
    manufacture a passing coherence check out of a missing measurement."""
    crit = SamplingNoise("noise", multiple=2.0, note="")
    assert crit.agrees(gap=0.5, content=-0.13, reading=-0.12, noise=None) is None
    assert crit.agrees(gap=0.01, content=-0.13, reading=-0.12, noise=0.01) is True
    assert crit.agrees(gap=0.5, content=-0.13, reading=-0.12, noise=0.01) is False


def test_the_criteria_include_one_that_cannot_be_computed_yet():
    assert any(isinstance(c, SamplingNoise) for c in CRITERIA)
    assert any(isinstance(c, RelativeToEffect) for c in CRITERIA)


def test_the_in_sample_seeds_are_recorded_and_do_not_grow():
    """A criterion's provenance is the whole of its value. If this tuple is ever extended to
    cover seeds that were read first, every 'out of sample' claim in the report becomes false."""
    assert SEEDS_SEEN_WHEN_WRITTEN == (0, 1, 2)


def test_a_unanimous_sign_is_reported_with_the_smallest_p_it_could_reach():
    """Three seeds all pointing the same way is p = 0.125, and that is the FLOOR. Reporting the
    run of signs without it invites reading consistency as significance."""
    s = summarise([_cell(0, -0.13, 0.01), _cell(1, -0.005, 0.0), _cell(2, -0.05, 0.0)], "aligned")
    assert s["n"] == 3
    assert s["n_negative"] == 3
    assert s["sign_test_p_if_unanimous"] == pytest.approx(0.125)
    assert s["mean"] < 0


def test_summarise_of_nothing_is_not_a_result():
    assert summarise([], "aligned") == {"alignment": "aligned", "n": 0}


# --------------------------------------------------------------------------- the report itself

def test_the_report_renders(tmp_path):
    """The gap that let broken code through. `report` was once committed calling two functions
    that a failed patch had never added; ruff caught the NameError and these tests did not,
    because nothing here called `report`. Now something does."""
    import json

    from civitas_g.campaign import report

    def _succession(seed, aligned, nfcs):
        arms = []
        for arm, nfc in nfcs.items():
            arms.append({"arm": arm, "seed": seed, "aligned": aligned, "stale": 0.2, "stale_n": 10,
                         "null": 0.1, "ratio": 1.5, "nfc_mean": nfc, "nfc_censored": 0.1,
                         "nfc_n": 100, "prep_hit": 0.5, "pop": 200.0, "store_sha256": None,
                         "store_density": 0.3, "sym_gain": 0.05, "claim_window": 700,
                         "marks_surviving_at_window_end": 0.1})
        return {"succession": {"seed": seed, "a_phase_steps": 1500, "b_steps": 2100,
                               "aligned": aligned, "scramble_seed": 0, "claim_window": 700},
                "a_store_sha256": "x", "a_mapping": [4, 3, 1], "a_pi": [2, 4, 3, 1, 0],
                "b_mapping": [4, 3, 1], "arms": arms, "gate_r": {"verdict": "PASS"},
                "assay": {}, "notes": []}

    for seed in (0, 1, 2):
        for aligned in (True, False):
            nfcs = {"fresh store": 2.0, "inherited store": 1.8 + 0.01 * seed,
                    "inherited scrambled": 1.9, "inherited gain-zero": 1.9}
            name = f"seed{seed}_{'aligned' if aligned else 'misaligned'}.json"
            (tmp_path / name).write_text(json.dumps(_succession(seed, aligned, nfcs)))

    text = report(tmp_path)
    for expected in ("G3 CAMPAIGN", "OUT OF SAMPLE", "HOW THE ESTIMATE MOVED",
                     "CONTROLS AGREE?", "NOT COMPUTABLE", "No criterion here decides a gate"):
        assert expected in text, expected


def test_the_report_warns_when_a_seed_has_only_one_alignment(tmp_path):
    """The other thing the report is for: it once printed per-alignment means at n=6 beside a
    paired contrast at n=5, and the missing seed was the one that had moved every number."""
    import json

    from civitas_g.campaign import report

    arms = [{"arm": a, "seed": 0, "aligned": True, "stale": 0.2, "stale_n": 10, "null": 0.1,
             "ratio": 1.5, "nfc_mean": 1.9, "nfc_censored": 0.1, "nfc_n": 100, "prep_hit": 0.5,
             "pop": 200.0, "store_sha256": None, "store_density": 0.3, "sym_gain": 0.05,
             "claim_window": 700, "marks_surviving_at_window_end": 0.1}
            for a in ("fresh store", "inherited store", "inherited scrambled",
                      "inherited gain-zero")]
    for seed, aligned in ((0, True), (1, True), (1, False)):
        d = {"succession": {"seed": seed, "a_phase_steps": 1500, "b_steps": 2100,
                            "aligned": aligned, "scramble_seed": 0, "claim_window": 700},
             "a_store_sha256": "x", "a_mapping": [], "a_pi": [], "b_mapping": [],
             "arms": [dict(a, seed=seed, aligned=aligned) for a in arms],
             "gate_r": {}, "assay": {}, "notes": []}
        name = f"seed{seed}_{'aligned' if aligned else 'misaligned'}.json"
        (tmp_path / name).write_text(json.dumps(d))

    text = report(tmp_path)
    assert "have only one alignment on disk" in text
    assert "[0]" in text
