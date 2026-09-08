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
def test_selftest_reports_the_gaps_and_still_exits_zero(capsys):
    """The G2/G3 tests are not available and must not be reported as failures."""
    code = main(["selftest", "--only", "world", "no_self_echo", "assay_selftest"])
    out = capsys.readouterr().out
    assert "N/A" in out and "assay_selftest" in out
    assert code == 0
