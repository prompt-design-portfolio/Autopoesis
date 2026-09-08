"""The frozen assay (A2.1), through the store.

Twelve cells: `mapping ∈ {matched, shuffled} × eta_scale ∈ {0, 1} × store ∈ {visible, hidden,
label_permuted}`. Before the G2 engine change the last axis did not exist, because `frozen_replay`
builds a fresh `World` whose marks are zeroed and whose π is redrawn.

These tests are about the *instrument*, not about a result: one seed and a short window measure
nothing. What they hold is that the cells are matched, that the conditions differ only in what they
are supposed to differ in, and that the two invariants D5 names actually hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from civitas_g.assay import (
    ETA_SCALES,
    MAPPING_CONDITIONS,
    STORE_CONDITIONS,
    AssayResult,
    run_assay,
)
from civitas_g.store.record import ScrambleMode


@pytest.fixture(scope="module")
def run_and_record():
    from civitas_g.selftests import _a_run_with_a_record

    run_dict, record = _a_run_with_a_record(phase_steps=800)
    assert run_dict is not None, record
    return run_dict, record


@pytest.fixture(scope="module")
def assay(run_and_record) -> AssayResult:
    run_dict, record = run_and_record
    return run_assay(run_dict, record, steps=60)


def test_the_assay_runs_all_twelve_cells(assay):
    assert len(assay.cells) == len(STORE_CONDITIONS) * len(MAPPING_CONDITIONS) * len(ETA_SCALES)
    assert len(assay.cells) == 12
    seen = {(c.store, c.mapping, c.eta_scale) for c in assay.cells}
    assert len(seen) == 12


def test_the_shuffled_mapping_is_a_derangement_of_the_matched_one(assay):
    """No genotype sorted for the matched mapping may retain any advantage under it."""
    assert len(assay.matched_mapping) == len(assay.shuffled_mapping) == 3
    assert all(a != b for a, b in zip(assay.matched_mapping, assay.shuffled_mapping, strict=True))


def test_the_population_is_frozen_so_nothing_but_h_can_change(assay):
    """Births, deaths and injection off, energy pinned by identity. If the population moved, a
    change in hit rate could be a different sample rather than within-life learning."""
    pops = {c.pop for c in assay.cells}
    assert len(pops) == 1, f"the population is not constant across cells: {pops}"


def test_every_cell_reports_how_many_preparations_it_saw(assay):
    """A cell with a small n is not a small effect; it is a cell that could not be read."""
    assert all(c.n_preparations > 0 for c in assay.cells)


def test_the_claim_line_is_stated_on_the_shuffled_mapping(assay):
    """A2.1: a genome sorted for the matched mapping is already right. A hit rate that rises on a
    mapping no genotype was selected under, with nothing able to change but H, is within-life
    learning and can be nothing else."""
    direct = (assay.cell("visible", "shuffled", 1.0).hit
              - assay.cell("visible", "shuffled", 0.0).hit)
    assert assay.claim_line() == pytest.approx(direct)
    assert assay.claim_line() != (assay.cell("visible", "matched", 1.0).hit
                                 - assay.cell("visible", "matched", 0.0).hit)


def test_the_store_conditions_differ_only_in_what_the_population_starts_with(run_and_record):
    """All three keep record='real', so agents go on writing during the window exactly as they
    would in a live run. Setting record='none' for `hidden` would remove the writing as well as
    the content, and a gap could then be either."""
    import inspect

    from civitas_g import assay as module

    source = inspect.getsource(module._replay)
    assert 'record="none"' not in source and "record='none'" not in source


def test_hidden_starts_empty_and_permuted_starts_relabelled(run_and_record):
    from civitas_g.assay import _store_for

    _run_dict, record = run_and_record
    rng = np.random.default_rng(0)
    hidden = _store_for(record, "hidden", rng)
    visible = _store_for(record, "visible", np.random.default_rng(0))
    permuted = _store_for(record, "label_permuted", np.random.default_rng(0))

    assert not np.asarray(hidden["marks"]).any()
    assert np.array_equal(visible["marks"], record.marks)
    assert not np.array_equal(permuted["marks"], record.marks)
    # a permutation is a bijection: the density and the signs survive
    assert (np.abs(permuted["marks"]) > 1e-3).sum() == (np.abs(record.marks) > 1e-3).sum()


def test_the_permuted_arm_uses_a_global_permutation_not_the_per_cell_control(run_and_record):
    """The assay asks whether a population that already bound labels within its own life was
    reading those particular labels. B§5.2's per-cell scramble answers a different question and
    would break this arm."""
    _run_dict, record = run_and_record
    from civitas_g.assay import _store_for

    permuted = _store_for(record, "label_permuted", np.random.default_rng(3))
    expected = record.scrambled(np.random.default_rng(3), ScrambleMode.GLOBAL)
    assert np.array_equal(permuted["marks"], expected.marks)


def test_sym_gain_is_zeroed_in_the_genome_not_by_the_config_lock(run_and_record):
    """cfg.sym_gain_lock is applied in Agent.__init__ and at reproduction; restore() then writes
    every genome field back over the top, sym_gain included. A replay that set the lock would
    silently run at the snapshot's own gain."""
    import sim_v3_13

    assert "sym_gain" in sim_v3_13.GENOME

    run_dict, record = run_and_record
    muted = run_assay(run_dict, record, steps=30, sym_gain_zero=True)
    assert muted.sym_gain_zero is True
    # with no gain the read is zero whatever the labels say
    assert (muted.cell("visible", "shuffled", 1.0).hit
            == muted.cell("label_permuted", "shuffled", 1.0).hit)


def test_the_assay_refuses_a_run_with_no_era_boundary(run_and_record):
    """A run shorter than one era produces no snapshot, so there is nothing to freeze."""
    run_dict, record = run_and_record
    without = dict(run_dict, era_snaps=[])
    with pytest.raises(ValueError, match="nothing to freeze"):
        run_assay(without, record, steps=30)


def test_the_permutation_seed_is_recorded_so_the_arm_is_reproducible(run_and_record):
    """A control drawn from an unrecorded random state cannot be reproduced -- the `sr_w` lesson,
    applied to a measurement condition."""
    run_dict, record = run_and_record
    a = run_assay(run_dict, record, steps=30, permute_seed=1)
    b = run_assay(run_dict, record, steps=30, permute_seed=1)
    assert a.cell("label_permuted", "shuffled", 1.0).hit == \
        b.cell("label_permuted", "shuffled", 1.0).hit


def test_the_result_records_which_store_it_was_run_against(assay, run_and_record):
    _run_dict, record = run_and_record
    assert assay.store_sha256 == record.sha256()
    assert assay.steps == 60
    assert isinstance(assay.as_dict()["cells"], list)


def test_the_table_prints_every_cell(assay):
    table = assay.table()
    for store in STORE_CONDITIONS:
        assert store in table
    assert table.count("\n") == 12          # header plus twelve rows
