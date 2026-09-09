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
