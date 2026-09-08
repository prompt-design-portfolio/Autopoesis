"""The gate's semantics. These are the rules that decide whether a number counts."""

from __future__ import annotations

import math

import pytest

from civitas_g.reading.reproduce import backends_agree, compare

KW = dict(reference_key="k", reference_path="p", reference_sha256="s", backend="sqlite",
          tolerance=5e-4)


def diff(reference, computed, **kw):
    return compare(reference, computed, **{**KW, **kw})


def test_an_empty_comparison_does_not_pass():
    """`all([])` is True. An acceptance report over zero levels reporting success is the exact
    defect this rule exists to prevent."""
    assert diff({}, {}).passed is False


def test_a_field_that_matches_at_the_printed_width_passes():
    """A reference records only what it printed: comparing a stored 0.05074321 against a printed
    0.5074 at any tolerance would test the print width, not the number."""
    r = diff({"gate_r/plastic + record/pooled": 1.200},
             {"gate_r/plastic + record/pooled": 1.2004})
    assert r.passed and r.n_matched == 1


def test_a_field_that_differs_beyond_the_printed_width_fails():
    r = diff({"gate_r/plastic + record/pooled": 1.200},
             {"gate_r/plastic + record/pooled": 1.206})
    assert not r.passed and r.n_mismatched == 1
    assert r.diffs[0].delta == pytest.approx(0.006)


def test_a_field_present_on_one_side_only_is_missing_and_fails():
    """A gate that compared the intersection would pass a reproduction that lost half its
    tables."""
    r = diff({"a/b/c": 1.0}, {})
    assert not r.passed and r.n_missing == 1
    assert r.diffs[0].status == "missing_computed"
    r2 = diff({}, {"a/b/c": 1.0})
    assert r2.diffs[0].status == "missing_reference"


def test_nan_on_both_sides_is_an_agreement_about_an_absence():
    r = diff({"follow/plastic/all": math.nan}, {"follow/plastic/all": math.nan})
    assert r.passed


def test_nan_against_a_number_is_a_mismatch_never_a_skip():
    """One side computed the cell and the other could not. That is the worst kind of
    disagreement and it must not be quietly tolerated."""
    r = diff({"follow/plastic/all": math.nan}, {"follow/plastic/all": 0.4})
    assert not r.passed and r.n_mismatched == 1
    r2 = diff({"follow/plastic/all": 0.4}, {"follow/plastic/all": math.nan})
    assert not r2.passed


def test_a_column_is_compared_at_the_width_its_own_table_prints():
    """`z` prints to two decimals and `mark_dens` to four; using three for both would be wrong
    in opposite directions."""
    tight = diff({"population/plastic/mark_dens": 0.5074},
                 {"population/plastic/mark_dens": 0.5079})
    assert not tight.passed, "four-decimal column compared too loosely"
    loose = diff({"gate_r_perm/plastic + record/z": 1.35},
                 {"gate_r_perm/plastic + record/z": 1.3549})
    assert loose.passed, "two-decimal column compared too tightly"


def test_the_report_names_the_failing_fields():
    r = diff({"a/b/c": 1.0, "d/e/f": 2.0}, {"a/b/c": 9.0, "d/e/f": 2.0})
    text = r.report()
    assert "a/b/c" in text and "FAILED" in text and "d/e/f" not in text


def test_the_stored_rows_carry_every_compared_field_not_only_the_failures():
    """A gate that recorded only failures makes a pass unfalsifiable."""
    r = diff({"a/b/c": 1.0, "d/e/f": 2.0}, {"a/b/c": 1.0, "d/e/f": 2.0})
    rows = r.as_rows()
    assert len(rows) == 2 and all(row["status"] == "match" for row in rows)


def test_non_finite_values_serialise_as_strings_for_jsonb():
    r = diff({"a/b/c": math.nan}, {"a/b/c": math.nan})
    assert r.as_rows()[0]["reference"] == "nan"


def test_backends_must_agree_at_float_noise_not_at_three_decimals():
    """Both sides recompute from the same stored numbers, so anything above noise means the
    round-trip lost precision."""
    ok, problems = backends_agree({"a/b/c": 1.0}, {"a/b/c": 1.0 + 1e-6}, 1e-9)
    assert not ok and problems
    ok2, _ = backends_agree({"a/b/c": 1.0}, {"a/b/c": 1.0}, 1e-9)
    assert ok2


def test_backends_agree_treats_two_nans_as_agreement():
    ok, _ = backends_agree({"a/b/c": math.nan}, {"a/b/c": math.nan}, 1e-9)
    assert ok


def test_a_field_on_only_one_backend_is_a_disagreement():
    ok, problems = backends_agree({"a/b/c": 1.0}, {}, 1e-9)
    assert not ok and "only one backend" in problems[0]
