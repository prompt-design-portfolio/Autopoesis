"""Parsing a reference summary. The real file, not a fixture of one."""

from __future__ import annotations

import math

import pytest

from civitas_g.manifest import PRECHECK_V3_13
from civitas_g.reading.compute import TABLES
from civitas_g.reading.parse import ARM_NAMES, parse_summary


@pytest.fixture(scope="module")
def parsed():
    return parse_summary(PRECHECK_V3_13.summary.path)


def test_every_declared_table_is_found(parsed):
    """A table that stops matching must fail, not silently shrink the reproduction."""
    assert parsed.tables_missing == []
    assert set(parsed.tables_found) == set(TABLES)


def test_the_reference_yields_the_expected_number_of_fields(parsed):
    assert len(parsed) == 182


@pytest.mark.parametrize("field,expected", [
    ("gate_r/plastic + record/pooled", 1.200),
    ("gate_r/plastic + record/per_era_mean", 1.009),
    ("gate_r/plastic + record/n_writes", 38324.0),
    ("gate_r_perm/fixed + record/z", 2.08),
    ("gate_r_perm/plastic + record (slow)/epochs", 2.0),
    ("sym_gain/plastic + record (slow)/phase2", -0.3504),
    ("sym_gain_delta/fixed + record/delta", -0.3615),
    ("store_gain/plastic + record/learned_minus_innate", 0.0940),
    ("first_prep_by_mark/plastic + noise/gap", 0.241),
    ("follow/plastic + record (slow)/ratio", 2.89),
    ("follow/plastic + record/stale", 0.041),
    ("nfc/plastic + noise/mean_preps", 1.991),
    ("population/fixed + record/pop_p1", 168.0),
    ("population/fixed + record/mark_dens", 0.2420),
])
def test_cells_parse_to_what_the_file_prints(parsed, field, expected):
    assert parsed.fields[field] == pytest.approx(expected)


def test_a_cell_the_file_prints_as_nan_parses_as_nan_not_as_missing(parsed):
    """`plastic` has no record, so its follow rates are nan -- an absence it DID print."""
    assert math.isnan(parsed.fields["follow/plastic/all"])
    assert parsed.fields["follow/plastic/n_stale"] == 0.0
    # the matched null is still computable without a record, and the file prints it
    assert parsed.fields["follow/plastic/null"] == pytest.approx(0.123)


def test_a_row_with_no_numbers_yields_no_fields_rather_than_zeros(parsed):
    """`plastic  --  (no record)` in the permutation table. Zero would compare as a match."""
    assert ("gate_r_perm", "plastic") in parsed.empty_rows
    assert not any(k.startswith("gate_r_perm/plastic/") for k in parsed.fields)


def test_the_slow_arm_is_never_parsed_as_the_plain_record_arm():
    """Longest-first matching. `plastic` is a prefix of three other arm names."""
    assert ARM_NAMES[0] == "plastic + record (slow)"
    assert ARM_NAMES[-1] == "plastic"


def test_every_arm_in_the_reference_appears_in_at_least_one_table(parsed):
    seen = {k.split("/")[1] for k in parsed.fields}
    assert seen == set(ARM_NAMES)


def test_the_grid_summary_parses_nothing_it_does_not_contain():
    """The v3.11 grid is a different world with different tables. It must report them missing
    rather than producing a small, passing comparison."""
    from civitas_g.manifest import V3_11_GRID

    p = parse_summary(V3_11_GRID.summary.path)
    assert p.tables_missing, "the v3.11 grid should not carry v3.13's record tables"


def test_an_arm_with_no_record_contributes_no_permutation_row(parsed):
    """`v313_precheck` prints "--  (no record)" and moves on. There is no label axis to permute,
    so the row is an ABSENCE. Emitting five cells here would compare against a reference that
    prints none -- which cost five fields on the first reproduction."""
    assert not any(k.startswith("gate_r_perm/plastic/") for k in parsed.fields)
    # the same arm DOES have a pooled Gate R row, printed as nan
    assert math.isnan(parsed.fields["gate_r/plastic/pooled"])
