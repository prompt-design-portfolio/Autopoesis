"""The command line. Each command answers one question a reader of a G1 number needs answered."""

from __future__ import annotations

import pytest

from civitas_g.cli import main


def test_world_prints_the_world_of_record_and_the_modulator_table(capsys):
    assert main(["world"]) == 0
    out = capsys.readouterr().out
    assert "chance EV of a preparation: +0.0000" in out
    assert "prep_value" in out and "1.0" in out
    assert "eat_inedible" in out
    assert "reading the record is NOT an action" in out
    assert "NOT AVAILABLE" in out          # standing food cover (D10)


def test_world_lists_every_arm_and_flags_the_one_outside_the_directive(capsys):
    main(["world"])
    out = capsys.readouterr().out
    assert "collective_slow_labels" in out and "not in B§4's table" in out
    assert "reference" in out and "G5" in out


def test_pins_verifies_the_g0_hashes(capsys):
    assert main(["pins"]) == 0
    out = capsys.readouterr().out
    assert "precheck_v3_13" in out and "GATED" in out
    assert "recorded, not gated" in out
    assert "NOT COMMITTED at cc8a0a6" in out
    assert "v3_12_finding.md" in out


@pytest.mark.slow
def test_selftest_reports_no_named_gaps_and_still_exits_zero(capsys):
    """This test used to assert the opposite, and the change is the point.

    It was written when `assay_selftest` was a named gap reported as `N/A` -- unavailable, and
    therefore not a failure. It is now a measurement, and it was the last one: there are no named
    gaps left at G1, G2 or G3. So the assertion inverts. A self-test suite with nothing missing
    should say so, and a test still demanding an `N/A` would quietly require the gap to come back.
    """
    code = main(["selftest", "--only", "world", "no_self_echo", "assay_selftest"])
    out = capsys.readouterr().out
    assert "assay_selftest" in out
    assert "N/A" not in out, "a self-test went unavailable again; find out which and why"
    assert code == 0


def test_store_names_the_blocker_and_the_two_scrambles(capsys):
    """G2's state has to be legible without reading the spec: the patch is not applied, and the
    two scrambles are different controls."""
    assert main(["store"]) == 0
    out = capsys.readouterr().out
    # The CURRENT engine version, read from the manifest rather than spelled out here. The engine
    # has moved twice since this test was written (G3-mapping, then G4-k), and a hard-coded
    # `G2-store` would fail on every future version while checking nothing about this one.
    from civitas_g.manifest import CURRENT_ENGINE
    assert f"engine {CURRENT_ENGINE.label}" in out
    assert "versioned, not silent" in out
    assert "checked by:" in out
    assert "per_cell" in out and "global" in out
    assert "isomorphic" in out
    assert "NOT per mark" in out
