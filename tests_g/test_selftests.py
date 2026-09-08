"""B§6's registry, and what "any failure halts" means for tests that do not exist yet."""

from __future__ import annotations

import pytest

from civitas_g import selftests as S


@pytest.fixture(scope="module")
def registry():
    return {t.name: t for t in S._registry()}


def test_every_selftest_the_directive_names_is_registered(registry):
    named = {
        "world_semantics", "record_semantics", "founder_tag", "replay_mapping",
        "frozen", "assay_selftest", "learning_rule",
        "store_round_trip", "scrambled_load", "b_founders_carry_no_h",
    }
    assert named <= set(registry)


def test_the_five_research_selftests_are_called_unchanged(registry):
    import sim_v3_13

    for name in ("world_semantics", "record_semantics", "founder_tag", "replay_mapping",
                 "learning_rule"):
        assert registry[name].source == "sim_v3_13"
        assert hasattr(sim_v3_13, f"{name}_selftest")


def test_assay_selftest_is_registered_as_a_named_gap_not_omitted(registry):
    """F6. A2.1 cites it as an existing reference; it is nowhere in the repository. Registering it
    as unavailable makes it appear in every report instead of silently not running."""
    result = registry["assay_selftest"].run()
    assert result.status == "unavailable"
    assert result.milestone == "G2"
    assert "NOT AVAILABLE" in result.detail


def test_a_g1_test_that_is_unavailable_halts_but_a_g2_one_does_not():
    g1_gap = S.SelfTestResult("x", "unavailable", "why", milestone="G1")
    g2_gap = S.SelfTestResult("y", "unavailable", "why", milestone="G2")
    failure = S.SelfTestResult("z", "fail", "why")
    assert g1_gap.halts and failure.halts
    assert not g2_gap.halts


def test_any_failure_halts_a_read():
    results = [S.SelfTestResult("ok", "pass", ""), S.SelfTestResult("bad", "fail", "reason")]
    with pytest.raises(S.SelfTestFailure, match="bad"):
        S.halt_on_failure(results)


def test_the_g2_and_g3_gaps_do_not_halt_a_g1_read():
    S.halt_on_failure([
        S.SelfTestResult("store_round_trip", "unavailable", "G2", milestone="G2"),
        S.SelfTestResult("scrambled_load", "unavailable", "G3", milestone="G3"),
    ])


def test_a_selftest_that_raises_is_a_failure_not_an_error():
    def boom():
        raise ValueError("exploded")

    original = S._registry
    S._registry = lambda: [S.SelfTest("boom", boom)]
    try:
        results = S.run_selftests(verbose=False)
    finally:
        S._registry = original
    assert results[0].status == "fail" and "exploded" in results[0].detail


@pytest.mark.slow
def test_the_world_selftest_passes():
    result = S.run_selftests(names=["world"], verbose=False)[0]
    assert result.ok, result.detail


@pytest.mark.slow
def test_the_no_self_echo_and_modulator_selftests_pass():
    results = S.run_selftests(names=["no_self_echo", "modulator"], verbose=False)
    assert all(r.ok for r in results), [r.detail for r in results]


@pytest.mark.slow
def test_the_engine_drift_selftest_confirms_heads_engine_reproduces_the_reference_blob():
    """F3 as a measurement rather than as a reading of a diff. The whole reproduction rests on
    this being true, so it is checked by running both blobs."""
    result = S.run_selftests(names=["engine_drift"], verbose=False)[0]
    assert result.ok, result.detail
    assert "identical" in result.detail
