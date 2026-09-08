"""Recompute a reference summary's numbers from the persisted rows.

Every number below is produced by calling `analysis_v3_13`'s own function on a run dict rebuilt
from `g_era_rows`. That is deliberate and it is the only honest way to do it: a reimplementation
of `follow_split` or `gate_r` here would be a second definition of the measurement, and the first
time the two disagreed the reproduction gate would be testing this module rather than the
persistence it exists to test.

So the split is:

* `analysis_v3_13` decides **what a number means**. Untouched.
* this module decides **which numbers the gate compares**, and gets them from rows rather than
  from an in-memory result.

The field keys mirror `v313_precheck`'s printed tables one for one -- `table/arm/column` -- because
the gate diffs against that file and a key that did not correspond to a printed cell could not be
compared to anything.
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np

import analysis_v3_13 as A
from sim_v3_13 import N_PREPS

#: `(table, column)` pairs, in the order `v313_precheck` prints them. Declared rather than
#: discovered so that a table silently disappearing from either side is a failure, not a smaller
#: comparison that still passes.
TABLES: dict[str, tuple[str, ...]] = {
    "gate_r": ("pooled", "per_era_mean", "n_writes"),
    "gate_r_perm": ("observed", "null", "sd", "z", "epochs"),
    "sym_gain": ("phase1", "phase2", "change", "frac_pos"),
    "sym_gain_delta": ("abs_gain", "baseline", "delta"),
    "store_gain": ("learned", "innate", "learned_minus_innate"),
    "first_prep_by_mark": ("p_ok_mark", "p_ok_none", "gap", "n_mark", "n_none"),
    "follow": ("all", "endorse_ok", "stale", "null", "ratio", "n_stale"),
    "nfc": ("mean_preps", "censored", "n"),
    "population": ("pop_p1", "pop_p2", "inj_p2", "prep_hit", "mark_dens", "mark_abs"),
}


def key(table: str, arm: str, column: str) -> str:
    return f"{table}/{arm}/{column}"


@contextmanager
def _empty_slice_is_a_result():
    """An all-NaN slice is a measurement, not a warning.

    `nanmean` over an arm with no record warns "Mean of empty slice" and returns NaN. That NaN is
    the correct answer -- the arm has no record, so the cell has no value -- and it is already
    carried through to the output as NaN. The warning is suppressed only for that specific
    message, so a genuinely unexpected numpy warning still surfaces.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice",
                                category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered",
                                category=RuntimeWarning)
        yield


def _finite(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def compute_fields(results: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """`{field key: value}` for every cell `v313_precheck` prints as a number.

    `results` is `{research name: [run dict, ...]}` in seed order -- what
    `civitas_g.provider.population.load_results` returns.

    Two conventions are copied from `v313_precheck` exactly, and both matter for a three-decimal
    comparison. Arm-level aggregates are `nanmean` over seeds; the per-seed tables -- Gate R's
    permutation, and every transmission line -- read `results[name][0]`, the FIRST seed only. A
    reproduction that averaged where the reference took one seed would differ in the third decimal
    for a reason that has nothing to do with persistence.
    """
    with _empty_slice_is_a_result():
        return _compute_fields(results)


def _compute_fields(results: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    names = list(results)
    P2 = lambda r: A.phase_half(r, 1)                                   # noqa: E731
    g = lambda n, k: float(np.nanmean([A.half(P2(r), k) for r in results[n]]))  # noqa: E731

    out: dict[str, float] = {}

    # ---- Gate R, pooled ----------------------------------------------------------------
    for n in names:
        rs = [A.gate_r(r) for r in results[n]]
        out[key("gate_r", n, "pooled")] = float(np.nanmean([x["pooled"] for x in rs]))
        out[key("gate_r", n, "per_era_mean")] = float(np.nanmean(
            [np.nanmean(x["per_era"]) if x["per_era"] else np.nan for x in rs]))
        out[key("gate_r", n, "n_writes")] = float(np.nansum([x["n"] for x in rs]))

    # ---- Gate R, matched permutation null (A2.2) --------------------------------------
    # Permuted per pi-EPOCH, not per era: with slow labels several mapping eras share one label
    # epoch, and permuting within an era would destroy structure the design intends to keep.
    for n in names:
        gp = A.gate_r_permutation(results[n][0])
        if not np.isfinite(gp["z"]):
            # `v313_precheck` prints "--   (no record)" and moves on: an arm with no record has no
            # label axis to permute, so the row is an ABSENCE, not a row of zeros and NaNs. Emitting
            # fields here would compare five cells against a reference that prints none.
            continue
        out[key("gate_r_perm", n, "observed")] = _finite(gp["obs"])
        out[key("gate_r_perm", n, "null")] = _finite(gp["null"])
        out[key("gate_r_perm", n, "sd")] = _finite(gp["sd"])
        out[key("gate_r_perm", n, "z")] = _finite(gp["z"])
        out[key("gate_r_perm", n, "epochs")] = _finite(gp["eras"])

    # ---- sym_gain ----------------------------------------------------------------------
    for n in names:
        p1 = float(np.nanmean([A.half(A.phase_half(r, 0), "sym_gain") for r in results[n]]))
        p2 = g(n, "sym_gain")
        out[key("sym_gain", n, "phase1")] = p1
        out[key("sym_gain", n, "phase2")] = p2
        out[key("sym_gain", n, "change")] = p2 - p1
        out[key("sym_gain", n, "frac_pos")] = g(n, "sym_gain_pos")

    base = "plastic" if "plastic" in names else None
    if base is not None:
        baseline = abs(g(base, "sym_gain"))
        for n in names:
            if n == base:
                continue
            mag = abs(g(n, "sym_gain"))
            out[key("sym_gain_delta", n, "abs_gain")] = mag
            out[key("sym_gain_delta", n, "baseline")] = baseline
            out[key("sym_gain_delta", n, "delta")] = mag - baseline

    # ---- store_gain --------------------------------------------------------------------
    for n in names:
        learned, innate = g(n, "store_gain"), g(n, "store_gain_innate")
        out[key("store_gain", n, "learned")] = learned
        out[key("store_gain", n, "innate")] = innate
        out[key("store_gain", n, "learned_minus_innate")] = learned - innate

    # ---- (i) the density check ---------------------------------------------------------
    # NOT a transmission measure. A positive mark exists only where someone recently succeeded, so
    # it marks places where success is common; the gap is a property of where marks are, not of
    # what they say -- which the pre-check proved by finding the gap LARGEST in `noise`.
    for n in names:
        pos, none, npos, nnone = A.first_prep_by_mark(P2(results[n][0]))
        gap = (pos - none) if (np.isfinite(pos) and np.isfinite(none)) else math.nan
        out[key("first_prep_by_mark", n, "p_ok_mark")] = _finite(pos)
        out[key("first_prep_by_mark", n, "p_ok_none")] = _finite(none)
        out[key("first_prep_by_mark", n, "gap")] = _finite(gap)
        out[key("first_prep_by_mark", n, "n_mark")] = _finite(npos)
        out[key("first_prep_by_mark", n, "n_none")] = _finite(nnone)

    # ---- (ii) following the record, with A2.2's matched null ---------------------------
    for n in names:
        L = P2(results[n][0])
        f_, _n_all = A.follow_rate(L)
        endorse_ok, _n_g, stale, n_stale = A.follow_split(L)
        hit = A.prep_hit(L, True)
        null = (1.0 - hit) / (N_PREPS - 1) if np.isfinite(hit) else math.nan
        out[key("follow", n, "all")] = _finite(f_)
        out[key("follow", n, "endorse_ok")] = _finite(endorse_ok)
        out[key("follow", n, "stale")] = _finite(stale)
        out[key("follow", n, "null")] = _finite(null)
        out[key("follow", n, "ratio")] = _finite(stale / null if null else math.nan)
        out[key("follow", n, "n_stale")] = _finite(n_stale)

    # ---- preparations-to-first-correct -------------------------------------------------
    for n in names:
        mean_preps, censored, nn = A.nfc(P2(results[n][0]), True)
        out[key("nfc", n, "mean_preps")] = _finite(mean_preps)
        out[key("nfc", n, "censored")] = _finite(censored)
        out[key("nfc", n, "n")] = _finite(nn)

    # ---- population and the store ------------------------------------------------------
    for n in names:
        out[key("population", n, "pop_p1")] = float(np.nanmean(
            [A.half(A.phase_half(r, 0), "pop") for r in results[n]]))
        out[key("population", n, "pop_p2")] = g(n, "pop")
        out[key("population", n, "inj_p2")] = g(n, "injections")
        out[key("population", n, "prep_hit")] = float(np.nanmean(
            [A.prep_hit(P2(r), True) for r in results[n]]))
        out[key("population", n, "mark_dens")] = g(n, "mark_density")
        out[key("population", n, "mark_abs")] = g(n, "mark_mean_abs")

    return out


#: How many decimals `v313_precheck` prints each column to. The gate rounds both sides to this
#: before comparing, because a reference summary records only what it printed: comparing a stored
#: 0.05074321 against a printed 0.5074 at 5e-4 would fail on the print width, not on the number.
PRINTED_DECIMALS: dict[tuple[str, str], int] = {
    ("gate_r", "pooled"): 3, ("gate_r", "per_era_mean"): 3, ("gate_r", "n_writes"): 0,
    ("gate_r_perm", "observed"): 3, ("gate_r_perm", "null"): 3, ("gate_r_perm", "sd"): 3,
    ("gate_r_perm", "z"): 2, ("gate_r_perm", "epochs"): 0,
    ("sym_gain", "phase1"): 4, ("sym_gain", "phase2"): 4, ("sym_gain", "change"): 4,
    ("sym_gain", "frac_pos"): 3,
    ("sym_gain_delta", "abs_gain"): 4, ("sym_gain_delta", "baseline"): 4,
    ("sym_gain_delta", "delta"): 4,
    ("store_gain", "learned"): 4, ("store_gain", "innate"): 4,
    ("store_gain", "learned_minus_innate"): 4,
    ("first_prep_by_mark", "p_ok_mark"): 3, ("first_prep_by_mark", "p_ok_none"): 3,
    ("first_prep_by_mark", "gap"): 3, ("first_prep_by_mark", "n_mark"): 0,
    ("first_prep_by_mark", "n_none"): 0,
    ("follow", "all"): 3, ("follow", "endorse_ok"): 3, ("follow", "stale"): 3,
    ("follow", "null"): 3, ("follow", "ratio"): 2, ("follow", "n_stale"): 0,
    ("nfc", "mean_preps"): 3, ("nfc", "censored"): 3, ("nfc", "n"): 0,
    ("population", "pop_p1"): 0, ("population", "pop_p2"): 0, ("population", "inj_p2"): 2,
    ("population", "prep_hit"): 3, ("population", "mark_dens"): 4, ("population", "mark_abs"): 3,
}


def decimals_for(field: str) -> int:
    """Print width for a field key, defaulting to three -- A3's "three decimals"."""
    parts = field.split("/")
    if len(parts) != 3:
        return 3
    return PRINTED_DECIMALS.get((parts[0], parts[2]), 3)
