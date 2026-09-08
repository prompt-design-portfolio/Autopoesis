"""B§6's self-tests, plus the ones this build added.

    Run before any result is read; any failure halts.

Ten are named by B§6. Seven of them exist in the research files and are called here unchanged.
Three do not exist yet, and the reason differs in a way that matters:

* **store round-trip, scrambled-load, B-founders-carry-no-`H`** are G2 and G3 mechanisms. They do
  not exist because the thing they test does not exist. The G1 form of the first one -- does a log
  row survive the database round-trip byte-for-byte, NaNs included -- does exist and runs here.
* **`assay_selftest` does not exist and was expected to.** A2.1 cites it as an existing reference
  alongside `frozen_selftest`. It is in neither research file nor anywhere else in the repository
  (F6). It is registered here as unavailable with its specification attached, so that it appears
  in every self-test report as a named gap rather than silently not being run. D5 builds it at G2.

Four self-tests are new to this build, each attached to something the G0 audit found:

* `engine_drift_selftest` turns F3's reading of a diff into a measurement -- the claim that the
  engine drift since the reference's commit is additive recording only, and therefore
  trajectory-preserving, is checked by running both blobs.
* `modulator_selftest` checks D7: the modulator is the table, not the energy delta.
* `no_self_echo_selftest` checks that the property the transmission lines rest on holds.
* `world_selftest` checks that the world of record is the world this build was written against.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


class SelfTestFailure(RuntimeError):
    """A self-test failed, so no result is read. B§6: any failure halts."""


@dataclass
class SelfTestResult:
    name: str
    status: str            # "pass" | "fail" | "unavailable"
    detail: str
    milestone: str = "G1"

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    @property
    def halts(self) -> bool:
        """An unavailable G2/G3 test does not halt a G1 read; a failure always does."""
        return self.status == "fail" or (self.status == "unavailable" and self.milestone == "G1")


@dataclass
class SelfTest:
    name: str
    run: Callable[[], SelfTestResult]
    milestone: str = "G1"
    source: str = ""


def _quiet(fn: Callable[..., Any], *args: Any, **kw: Any) -> tuple[Any, str]:
    """Run something chatty and keep its output for the detail line."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        value = fn(*args, **kw)
    return value, buf.getvalue().strip()


def _research(name: str, fn: Callable[..., Any], **kw: Any) -> Callable[[], SelfTestResult]:
    def run() -> SelfTestResult:
        passed, output = _quiet(fn, verbose=True, **kw)
        tail = output.splitlines()[-1].strip() if output else ""
        return SelfTestResult(name, "pass" if passed else "fail", tail or "no output")
    return run


# ---------------------------------------------------------------------------------------------
# the ones this build added
# ---------------------------------------------------------------------------------------------

def _world_selftest() -> SelfTestResult:
    from civitas_g.world.adapter import check_no_leak, check_observation_layout
    from civitas_g.world.spec import check_chance_ev_is_zero, check_world

    try:
        check_world()
        ev = check_chance_ev_is_zero()
        check_observation_layout()
        check_no_leak(marks_are_labels=True, pi_redrawn_every_era=True,
                      read_is_observation=True, read_is_here_only=True)
    except AssertionError as exc:
        return SelfTestResult("world", "fail", str(exc))
    return SelfTestResult(
        "world", "pass",
        f"world of record matches EXPECTED_WORLD; chance EV of a preparation {ev:+.4f}; "
        f"31-input layout intact; leak rule holds (labels not indices, pi redrawn, reading is an "
        f"observation, read channels are here-only)")


def _modulator_selftest() -> SelfTestResult:
    from civitas_g.world.adapter import modulator_is_independent_of_economics

    ok, detail = modulator_is_independent_of_economics()
    return SelfTestResult("modulator", "pass" if ok else "fail", detail)


def _no_self_echo_selftest() -> SelfTestResult:
    from civitas_g.world.adapter import check_no_self_echo

    ok, detail = check_no_self_echo()
    return SelfTestResult("no_self_echo", "pass" if ok else "fail", detail)


def _row_round_trip_selftest() -> SelfTestResult:
    """The G1 form of B§6's store round-trip: does an engine log row survive the database?

    Run against a real row from a real short run rather than a synthetic dict, because the fields
    that break a round-trip are the ones nobody thinks to synthesise -- `mi_counts` as a (K, 2, K)
    ndarray, and the NaNs that mark a metric the arm could not compute.
    """
    from civitas_g.persistence.encoding import round_trips
    from civitas_g.world.engine import build_run_spec
    from civitas_g.world.engine import run as run_engine

    spec = build_run_spec("collective", seed=0, phase_steps=60)
    result = run_engine(spec)
    rows = result.log
    if not rows:
        return SelfTestResult("row_round_trip", "fail", "the run produced no log rows")
    bad = [(i, why) for i, row in enumerate(rows)
           for ok, why in [round_trips(row)] if not ok]
    if bad:
        return SelfTestResult("row_round_trip", "fail",
                              f"{len(bad)} of {len(rows)} rows changed; first: {bad[0][1][:200]}")
    n_nan = sum(1 for row in rows for v in row.values()
                if isinstance(v, float) and v != v)
    return SelfTestResult(
        "row_round_trip", "pass",
        f"{len(rows)} real log rows round-trip exactly, including mi_counts as (K,2,K) and "
        f"{n_nan} NaN cells preserved as NaN rather than as 0.0 or as an absent key")


def _engine_drift_selftest() -> SelfTestResult:
    """F3 as a measurement: HEAD's engine reproduces the reference blob's trajectory.

    `precheck_v3_13.txt` was produced by `sim_v3_13.py` at 3d7b228; HEAD's differs. Reading the
    diff says the difference is four added counters, two added log fields and one `else:` rewritten
    as `elif True:` -- no RNG consumed, no branch changed. That reading is what the whole
    reproduction rests on, so it is checked rather than trusted: both blobs are run at one seed and
    every field they share is compared.
    """
    import importlib.util

    from civitas_g.manifest import SIM_V3_13_DRIFT
    from civitas_g.world.engine import build_run_spec
    from civitas_g.world.engine import run as run_engine

    old_commit = SIM_V3_13_DRIFT["from"]
    blob = subprocess.run(("git", "show", f"{old_commit}:sim_v3_13.py"),
                          cwd=REPO_ROOT, capture_output=True, timeout=30)
    if blob.returncode != 0:
        return SelfTestResult("engine_drift", "fail",
                              f"cannot read sim_v3_13.py at {old_commit}: "
                              f"{blob.stderr.decode(errors='replace').strip()[:160]}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sim_v3_13_reference.py"
        path.write_bytes(blob.stdout)
        spec_ = importlib.util.spec_from_file_location("sim_v3_13_reference", path)
        old = importlib.util.module_from_spec(spec_)          # type: ignore[arg-type]
        spec_.loader.exec_module(old)                         # type: ignore[union-attr]

        # 800-step phases so that phase 2 crosses a mapping remap (prep_every 700). A shorter
        # run never enters the stale-mark branch the drift actually touches, and would pass
        # without exercising the thing it exists to check.
        run_spec = build_run_spec("collective", seed=0, phase_steps=800)
        new_result = run_engine(run_spec)
        old_raw = old.run(old.Config(seed=run_spec.seed, **run_spec.kwargs),
                          verbose=False, phases=run_spec.phases())

    new_log, old_log = new_result.log, old_raw["log"]
    if len(new_log) != len(old_log):
        return SelfTestResult("engine_drift", "fail",
                              f"{len(new_log)} log rows at HEAD vs {len(old_log)} at {old_commit}")
    shared = set(new_log[0]) & set(old_log[0])
    added = sorted(set(new_log[0]) - set(old_log[0]))
    import numpy as np

    differing = sorted({
        k for a, b in zip(old_log, new_log, strict=True) for k in shared
        if not np.array_equal(np.asarray(a[k], dtype=object),
                              np.asarray(b[k], dtype=object))
        and not (isinstance(a[k], float) and isinstance(b[k], float)
                 and a[k] != a[k] and b[k] != b[k])
    })
    if differing:
        return SelfTestResult(
            "engine_drift", "fail",
            f"{len(differing)} shared log fields differ between {old_commit} and HEAD: "
            f"{differing[:8]}. The drift is NOT trajectory-preserving, so precheck_v3_13.txt "
            f"cannot be reproduced with HEAD's engine and the reference blob must be used.")
    expected_added = {"n_follgf", "n_follgf_ok", "n_follbf", "n_follbf_ok"}
    if set(added) != expected_added:
        return SelfTestResult(
            "engine_drift", "fail",
            f"HEAD adds {added}, not the four counters the drift record claims "
            f"({sorted(expected_added)}). The manifest's SIM_V3_13_DRIFT is out of date.")
    return SelfTestResult(
        "engine_drift", "pass",
        f"{len(old_log)} log rows, {len(shared)} shared fields identical between {old_commit} and "
        f"HEAD; the only difference is the four added counters {sorted(expected_added)}. "
        f"HEAD's engine reproduces the reference blob's trajectory.")


def _assay_selftest_unavailable() -> SelfTestResult:
    return SelfTestResult(
        "assay_selftest", "unavailable",
        "NOT AVAILABLE. Named by A2.1 as an existing reference and by B§6 as a gate; it is in "
        "neither sim_v3_13.py nor analysis_v3_13.py nor anywhere else in the repository. "
        "Specified in D5 and built at G2 with the store: with record='none' the frozen assay's "
        "store-visible, store-hidden and label-permuted arms must be IDENTICAL, and with "
        "sym_gain = 0 the permuted arm must equal the visible arm.",
        milestone="G2")


def _g2_g3_unavailable(name: str, milestone: str, why: str) -> Callable[[], SelfTestResult]:
    def run() -> SelfTestResult:
        return SelfTestResult(name, "unavailable", why, milestone=milestone)
    return run


# ---------------------------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------------------------

def _registry() -> list[SelfTest]:
    import analysis_v3_13 as A
    import sim_v3_13 as S

    return [
        SelfTest("world", _world_selftest, source="civitas_g.world.spec"),
        SelfTest("world_semantics",
                 _research("world_semantics", S.world_semantics_selftest), source="sim_v3_13"),
        SelfTest("record_semantics",
                 _research("record_semantics", S.record_semantics_selftest), source="sim_v3_13"),
        SelfTest("founder_tag",
                 _research("founder_tag", S.founder_tag_selftest), source="sim_v3_13"),
        SelfTest("replay_mapping",
                 _research("replay_mapping", S.replay_mapping_selftest), source="sim_v3_13"),
        SelfTest("learning_rule",
                 _research("learning_rule", S.learning_rule_selftest), source="sim_v3_13"),
        SelfTest("frozen", _research("frozen", A.frozen_selftest), source="analysis_v3_13"),
        SelfTest("assay_selftest", _assay_selftest_unavailable, milestone="G2"),
        SelfTest("no_self_echo", _no_self_echo_selftest, source="civitas_g.world.adapter"),
        SelfTest("modulator", _modulator_selftest, source="civitas_g.world.adapter"),
        SelfTest("engine_drift", _engine_drift_selftest, source="civitas_g.manifest"),
        SelfTest("row_round_trip", _row_round_trip_selftest,
                 source="civitas_g.persistence.encoding"),
        SelfTest("store_round_trip",
                 _g2_g3_unavailable(
                     "store_round_trip", "G2",
                     "NOT AVAILABLE at G1. B§6's form -- save, load, byte-identical marks and pi "
                     "-- needs the store as an artifact store, which is G2. The row round-trip "
                     "below is the G1 half of it and does run."),
                 milestone="G2"),
        SelfTest("scrambled_load",
                 _g2_g3_unavailable(
                     "scrambled_load", "G3",
                     "NOT AVAILABLE at G1. Density and sign preserved, label destroyed on load -- "
                     "there is no load until a store outlives a run, which is G3."),
                 milestone="G3"),
        SelfTest("b_founders_carry_no_h",
                 _g2_g3_unavailable(
                     "b_founders_carry_no_h", "G3",
                     "NOT AVAILABLE at G1. There is no population B until G3; at G1 every run "
                     "starts from fresh founders and the engine refuses init_genomes."),
                 milestone="G3"),
    ]


def run_selftests(*, milestone: str = "G1", names: list[str] | None = None,
                  verbose: bool = True) -> list[SelfTestResult]:
    """Run the self-tests. Returns every result; use `halt_on_failure` to enforce B§6."""
    results: list[SelfTestResult] = []
    for test in _registry():
        if names is not None and test.name not in names:
            continue
        try:
            result = test.run()
        except Exception as exc:                        # noqa: BLE001 - a raising test is a fail
            result = SelfTestResult(test.name, "fail", f"{type(exc).__name__}: {exc}",
                                    milestone=test.milestone)
        result.milestone = test.milestone if result.status == "unavailable" else result.milestone
        results.append(result)
        if verbose:
            mark = {"pass": "PASS", "fail": "FAIL", "unavailable": "N/A "}[result.status]
            print(f"  [{mark}] {result.name:<22}{result.detail}", flush=True)
    return results


def halt_on_failure(results: list[SelfTestResult]) -> None:
    """B§6: any failure halts. Called before any number is read."""
    bad = [r for r in results if r.halts]
    if bad:
        lines = "\n".join(f"  {r.name}: {r.detail}" for r in bad)
        raise SelfTestFailure(
            f"{len(bad)} self-test(s) did not pass, so no result is read:\n{lines}")


def report(results: list[SelfTestResult]) -> str:
    counts = {s: sum(1 for r in results if r.status == s) for s in ("pass", "fail", "unavailable")}
    lines = [f"self-tests: {counts['pass']} pass, {counts['fail']} fail, "
             f"{counts['unavailable']} not available"]
    for r in results:
        mark = {"pass": "PASS", "fail": "FAIL", "unavailable": "N/A "}[r.status]
        lines.append(f"  [{mark}] {r.name:<24}({r.milestone}) {r.detail}")
    return "\n".join(lines)
