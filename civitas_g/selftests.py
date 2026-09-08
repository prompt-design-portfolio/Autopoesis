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


#: Kept for the record: the change that produced the current engine, applied at G2 under G2-D1.
#: The engine now carries it, so this file is history rather than a pending action.
PATCH_PATH = "docs/patches/g2-store-capture-and-injection.diff"


def _engine_at(commit: str, tmp: str):
    """Import `sim_v3_13.py` as it was at a named commit, without touching the working tree."""
    import importlib.util
    import subprocess

    blob = subprocess.run(("git", "show", f"{commit}:sim_v3_13.py"),
                          cwd=REPO_ROOT, capture_output=True, timeout=30)
    if blob.returncode != 0:
        raise RuntimeError(f"cannot read sim_v3_13.py at {commit}: "
                           f"{blob.stderr.decode(errors='replace').strip()[:160]}")
    path = Path(tmp) / f"sim_at_{commit[:7]}.py"
    path.write_bytes(blob.stdout)
    spec_ = importlib.util.spec_from_file_location(f"sim_at_{commit[:7]}", path)
    module = importlib.util.module_from_spec(spec_)      # type: ignore[arg-type]
    spec_.loader.exec_module(module)                     # type: ignore[union-attr]
    return module


def _logs_differ(a: list, b: list) -> list[str]:
    import numpy as np

    shared = set(a[0]) & set(b[0])
    return sorted({
        k for x, y in zip(a, b, strict=True) for k in shared
        if not np.array_equal(np.asarray(x[k], dtype=object), np.asarray(y[k], dtype=object))
        and not (isinstance(x[k], float) and isinstance(y[k], float)
                 and x[k] != x[k] and y[k] != y[k])
    })


def _engine_store_selftest() -> SelfTestResult:
    """G2-D1's safety argument, kept running now that the change is applied.

    The engine gained store capture and injection at G2. A1.8 says the engine runs byte-identical
    and its hash is in every manifest -- which is a claim about a *named* engine, not a claim that
    it can never change. So the claim that has to keep holding is that the named change is
    trajectory-neutral, and it is measured here rather than argued:

      1. with both flags off, the current engine is bit-identical to the one at G0;
      2. with capture on, the trajectory still does not move -- copying an array draws no RNG;
      3. injection works, which is the whole reason the change was made: without it
         `frozen_replay` has no store to see.

    This is the regression that stops a later edit quietly turning capture into participation.
    """
    import tempfile

    import numpy as np

    import sim_v3_13 as current
    from civitas_g.manifest import ENGINE_VERSIONS
    from civitas_g.world.engine import build_run_spec

    previous = next(v for v in ENGINE_VERSIONS if v.label == "G0")
    spec = build_run_spec("collective", seed=0, phase_steps=800)   # crosses a mapping remap

    with tempfile.TemporaryDirectory() as tmp:
        before = _engine_at(previous.at_commit, tmp)

        old_run = before.run(before.Config(seed=0, **spec.kwargs), verbose=False,
                             phases=spec.phases())
        new_run = current.run(current.Config(seed=0, **spec.kwargs), verbose=False,
                              phases=spec.phases())

    if len(old_run["log"]) != len(new_run["log"]):
        return SelfTestResult("engine_store", "fail",
                              f"{len(old_run['log'])} log rows at {previous.label} vs "
                              f"{len(new_run['log'])} now", milestone="G2")
    differing = _logs_differ(old_run["log"], new_run["log"])
    added = sorted(set(new_run["log"][0]) - set(old_run["log"][0]))
    if differing or added or old_run["final_mapping"] != new_run["final_mapping"]:
        return SelfTestResult(
            "engine_store", "fail",
            f"the store change is NOT trajectory-neutral: {len(differing)} log fields differ "
            f"{differing[:6]}, fields added {added}", milestone="G2")

    captured = current.run(current.Config(seed=0, store_snaps=True, **spec.kwargs),
                           verbose=False, phases=spec.phases())
    differing_on = _logs_differ(old_run["log"], captured["log"])
    if differing_on:
        return SelfTestResult(
            "engine_store", "fail",
            f"capture moved the trajectory in {len(differing_on)} fields {differing_on[:6]} -- "
            f"a snapshot must observe, not participate", milestone="G2")
    if not captured["store_snaps"]:
        return SelfTestResult("engine_store", "fail",
                              "capture was on and produced no store snapshots", milestone="G2")

    store = captured["store_snaps"][-1]
    probe = [dict(n_steps=int(current.Config(**spec.kwargs).log_every), chain=True)]
    injected = current.run(current.Config(seed=99, **spec.kwargs), verbose=False,
                           phases=probe, init_store=store)
    fresh = current.run(current.Config(seed=99, **spec.kwargs), verbose=False, phases=probe)
    got = float(injected["log"][-1]["mark_density"])
    none = float(fresh["log"][-1]["mark_density"])
    if not (got > none):
        return SelfTestResult(
            "engine_store", "fail",
            f"injection did not take: density {got} with a store, {none} without",
            milestone="G2")

    live = float((np.abs(store["marks"]) > 1e-3).mean())
    shared = len(set(old_run["log"][0]) & set(new_run["log"][0]))
    return SelfTestResult(
        "engine_store", "pass",
        f"{len(old_run['log'])} log rows, {shared} shared fields identical to engine "
        f"{previous.label} with the flags off and with capture on; "
        f"{len(captured['store_snaps'])} store snapshot(s) (density {live:.6f}); injection gives "
        f"a fresh world density {got:.6f} against {none:.6f} without.", milestone="G2")


def _store_decodes_selftest() -> SelfTestResult:
    """Does the pi stored with a record actually decode that record's marks?

    This is the check that would have caught the defect it now guards. `new_recipe()` redraws pi
    BEFORE the era-boundary snapshot is taken, so `world.pi` at that moment is the pi of the era
    about to start -- not the one the store's marks carry. A record stored with that pi looks
    entirely intact: the right density, the right signs, a valid permutation. It simply decodes to
    the wrong preparation for every food type, which is the worst way for a store to be wrong,
    because the arm that reads it produces a plausible null.

    A positive mark at label j says preparation `pi^-1(j)` succeeded on that type at that cell. So
    for the freshest marks -- the ones from the era just ended -- `pi^-1` of the positively marked
    labels must include the preparation that era's mapping made correct.
    """
    import numpy as np

    import sim_v3_13 as S

    res = S.run(S.Config(seed=0, chain=True, record="real", store_snaps=True,
                         prep_every=200, log_every=50, n_steps=900),
                verbose=False, phases=[dict(n_steps=900, chain=True)])
    snaps = list(res["store_snaps"]) + ([res["final_store"]] if res.get("final_store") else [])
    if not snaps:
        return SelfTestResult("store_decodes", "fail",
                              "capture was on and produced no store snapshots", milestone="G2")

    checked, bad = 0, []
    for sn in snaps:
        inverse = {label: k for k, label in enumerate(sn["pi"])}
        for ftype in range(S.N_TYPES):
            positive = np.argwhere(sn["marks"][ftype] > 1e-3)
            if not len(positive):
                continue
            endorsed = {inverse[int(label)] for label, _, _ in positive}
            checked += 1
            if sn["mapping"][ftype] not in endorsed:
                bad.append(f"t={sn['t']} type {ftype}: positively marked labels decode to "
                           f"preparations {sorted(endorsed)}, and the correct preparation for "
                           f"that era was {sn['mapping'][ftype]}")
    if bad:
        return SelfTestResult(
            "store_decodes", "fail",
            f"{len(bad)} of {checked} (snapshot, type) pairs decode wrongly. The stored pi is not "
            f"the pi the marks were written under, so any population handed this record reads it "
            f"through the wrong permutation. First: {bad[0]}", milestone="G2")
    return SelfTestResult(
        "store_decodes", "pass",
        f"{checked} (snapshot, type) pairs across {len(snaps)} captures: the positively marked "
        f"labels decode, through the stored pi, to the preparation that era's mapping made "
        f"correct.", milestone="G2")


def _assay_selftest() -> SelfTestResult:
    """D5, built at G2 as specified: the frozen assay's own two invariants.

    Both are of the same shape -- a condition under which the three store arms *cannot* differ, so
    that if they do, the assay is measuring something other than the store.

      1. **`record = "none"`.** The read channels are zero whatever the world holds, so injecting
         three different stores must give three identical results. Run against a REAL record, not
         an empty one: injecting three copies of an empty array would pass trivially and check
         nothing.
      2. **`sym_gain = 0`.** The read is `sym_gain * marks`, so it is zero whatever the labels
         say, and the label-permuted arm must equal the visible arm.

    A2.1 cites `assay_selftest` as an existing reference. It was not one -- it was absent from the
    repository, and before the G2 engine change the instrument it names could not be run at all.
    """
    from civitas_g.assay import STORE_CONDITIONS, run_assay

    run_dict, record = _a_run_with_a_record()
    if run_dict is None:
        return SelfTestResult("assay_selftest", "fail", str(record), milestone="G2")

    # 1. a world that does not read cannot be moved by what it is handed
    blind = dict(run_dict)
    blind["cfg"] = dict(run_dict["cfg"], record="none")
    deaf = run_assay(blind, record, steps=60)
    hits = {c.store: c.hit for c in deaf.cells if c.mapping == "shuffled" and c.eta_scale == 1.0}
    spread = max(hits.values()) - min(hits.values())
    if spread != 0.0:
        return SelfTestResult(
            "assay_selftest", "fail",
            f"with record='none' the three store arms differ by {spread:.6f} ({hits}). The read "
            f"channels are zero in that world, so the assay is responding to something other "
            f"than the store.", milestone="G2")

    # 2. a reader with zero gain reads nothing, whatever the labels say
    muted = run_assay(run_dict, record, steps=60, sym_gain_zero=True)
    visible = muted.cell("visible", "shuffled", 1.0).hit
    permuted = muted.cell("label_permuted", "shuffled", 1.0).hit
    if visible != permuted:
        return SelfTestResult(
            "assay_selftest", "fail",
            f"with sym_gain = 0 the permuted arm ({permuted:.6f}) does not equal the visible arm "
            f"({visible:.6f}). Note that cfg.sym_gain_lock cannot do this in a replay -- restore() "
            f"writes every genome field back over the top, sym_gain included.", milestone="G2")

    return SelfTestResult(
        "assay_selftest", "pass",
        f"with record='none' all {len(STORE_CONDITIONS)} store arms are identical against a real "
        f"record (hit {next(iter(hits.values())):.3f}); with sym_gain = 0 the permuted arm equals "
        f"the visible arm ({visible:.3f}). The assay responds to the store and to nothing else.",
        milestone="G2")


def _a_run_with_a_record(phase_steps: int = 800):
    """One real run long enough to cross an era boundary, and the record it left.

    Short by the standards of a result and not by the standards of the mechanism: an era boundary
    is what produces both a genome snapshot and a store snapshot, and the assay needs both.
    """
    import numpy as np

    from civitas_g.store.record import Record, RecordProvenance
    from civitas_g.world.engine import build_run_spec
    from civitas_g.world.engine import run as run_engine

    spec = build_run_spec("collective", seed=0, phase_steps=phase_steps,
                          overrides={"store_snaps": True})
    result = run_engine(spec)
    raw = result.raw
    if not raw.get("era_snaps") or not raw.get("store_snaps"):
        return None, ("the run produced no era-boundary snapshots, so there is nothing to "
                      "freeze; phase 2 must be at least one era long")
    snap = raw["store_snaps"][-1]
    record = Record(
        marks=np.asarray(snap["marks"], dtype="<f8"),
        pi=tuple(int(x) for x in snap["pi"]),
        provenance=RecordProvenance(
            run_seed=0, arm="collective", t=int(snap["t"]), era_index=0,
            engine_sha256=result.engine_sha256, cfg_digest="selftest",
            mapping=tuple(int(x) for x in raw["final_mapping"]),
            prev_mapping=tuple(int(x) for x in snap["mapping"])),
    )
    run_dict = {"log": raw["log"], "cfg": raw["cfg"], "phase_bounds": raw["phase_bounds"],
                "n_steps": raw["n_steps"], "chain_start": raw["chain_start"],
                "era_snaps": raw["era_snaps"], "arm": "collective", "seed": 0}
    return run_dict, record


def _a_record_from_a_real_world(n_writes: int = 4000, seed: int = 0):
    """A record built by driving the engine's own `write_mark`, not by synthesising an array.

    A1.6: the simulation is never mocked; it is the mechanism. A hand-built marks array would test
    this module against my idea of what a store looks like rather than against one.
    """
    import numpy as np

    import sim_v3_13 as S
    from civitas_g.store.record import record_from_world

    cfg = S.Config(seed=seed, chain=True, record="real")
    world = S.World(cfg, np.random.default_rng(seed))
    world.chain_on = True
    rng = np.random.default_rng(seed + 1)
    g = cfg.grid
    for _ in range(n_writes):
        ftype, k = int(rng.integers(S.N_TYPES)), int(rng.integers(S.N_PREPS))
        y, x = int(rng.integers(g)), int(rng.integers(g))
        world.write_mark(ftype, k, k == world.mapping[ftype], y, x, cfg)
    return record_from_world(world, arm="collective", t=700, era_index=1,
                             engine_sha256="0" * 64, cfg_digest="selftest")


def _store_round_trip_selftest() -> SelfTestResult:
    """B§6: save, load, byte-identical marks and pi."""
    from civitas_g.store.record import Record

    record = _a_record_from_a_real_world()
    blob = record.to_bytes()
    loaded = Record.from_bytes(blob, record.provenance)
    ok, detail = loaded.equals(record)
    if not ok:
        return SelfTestResult("store_round_trip", "fail", detail, milestone="G2")
    if loaded.sha256() != record.sha256():
        return SelfTestResult("store_round_trip", "fail",
                              "the content hashes differ across a round-trip", milestone="G2")
    return SelfTestResult(
        "store_round_trip", "pass",
        f"{detail}; {len(blob)} bytes on the wire, content hash {record.sha256()[:12]} stable",
        milestone="G2")


def _scrambled_load_selftest() -> SelfTestResult:
    """B§6: density and sign preserved, label destroyed."""
    import numpy as np

    from civitas_g.store.persistence import preserves_density_and_sign
    from civitas_g.store.record import ScrambleMode

    record = _a_record_from_a_real_world()
    scrambled = record.scrambled(np.random.default_rng(7), ScrambleMode.PER_CELL)
    ok, detail = preserves_density_and_sign(record, scrambled)
    if not ok:
        return SelfTestResult("scrambled_load", "fail", detail, milestone="G2")

    # and the part that makes it a control rather than a relabelling: a GLOBAL permutation
    # preserves the label -> preparation association everywhere, so it must NOT be what the
    # inherited-scrambled arm uses. Checked by showing the two modes differ.
    global_ = record.scrambled(np.random.default_rng(7), ScrambleMode.GLOBAL)
    if np.array_equal(global_.marks, scrambled.marks):
        return SelfTestResult(
            "scrambled_load", "fail",
            "the per-cell and global scrambles produced the same store, so the per-cell mode is "
            "not destroying cross-cell consistency", milestone="G2")
    return SelfTestResult("scrambled_load", "pass",
                          f"{detail}; per-cell and global scrambles are distinct, so the "
                          f"inherited-scrambled arm is a control and not a relabelling",
                          milestone="G2")


def _assay_preconditions_selftest() -> SelfTestResult:
    """The two clauses of D5's `assay_selftest` that do not need store injection.

    1. With `record="none"` the visible, hidden and label-permuted stores are the SAME store, so
       the three assay arms cannot differ for any reason other than a bug.
    2. With `sym_gain = 0` the read channels are zero whatever the store holds, so the permuted
       arm must equal the visible arm.

    The third clause -- that the frozen assay actually produces those equalities -- needs a replay
    that can see a store, and cannot run. See `assay_selftest`.
    """
    import numpy as np

    import sim_v3_13 as S
    from civitas_g.store.record import ScrambleMode

    empty = _a_record_from_a_real_world(n_writes=0)
    for name, derived in (("hidden", empty.hidden()),
                          ("permuted", empty.scrambled(np.random.default_rng(0),
                                                       ScrambleMode.GLOBAL))):
        ok, detail = empty.equals(derived)
        if not ok:
            return SelfTestResult("assay_preconditions", "fail",
                                  f"with no record, the {name} store differs: {detail}",
                                  milestone="G2")

    # clause 2, measured against the engine's own observation construction
    record = _a_record_from_a_real_world()
    cfg = S.Config(seed=0, chain=True, record="real")
    agent = S.Agent(cfg, np.random.default_rng(1), 0, 5, 5)
    agent.sym_gain = 0.0
    permuted = record.scrambled(np.random.default_rng(3), ScrambleMode.GLOBAL)
    for ftype in range(S.N_TYPES):
        for y, x in ((5, 5), (11, 23), (40, 2)):
            visible = agent.sym_gain * record.marks[ftype, :, y, x]
            other = agent.sym_gain * permuted.marks[ftype, :, y, x]
            if not np.array_equal(visible, other) or visible.any():
                return SelfTestResult(
                    "assay_preconditions", "fail",
                    f"with sym_gain = 0 the read channels are not zero at "
                    f"(type {ftype}, cell {y},{x})", milestone="G2")
    return SelfTestResult(
        "assay_preconditions", "pass",
        "with no record the visible, hidden and permuted stores are identical; with sym_gain = 0 "
        "the K read channels are zero whatever the store holds, so the permuted arm equals the "
        "visible arm. The third clause needs store injection -- see assay_selftest",
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
        SelfTest("assay_selftest", _assay_selftest, milestone="G2",
                 source="civitas_g.assay"),
        SelfTest("no_self_echo", _no_self_echo_selftest, source="civitas_g.world.adapter"),
        SelfTest("modulator", _modulator_selftest, source="civitas_g.world.adapter"),
        SelfTest("engine_drift", _engine_drift_selftest, source="civitas_g.manifest"),
        SelfTest("row_round_trip", _row_round_trip_selftest,
                 source="civitas_g.persistence.encoding"),
        SelfTest("store_round_trip", _store_round_trip_selftest,
                 milestone="G2", source="civitas_g.store.record"),
        SelfTest("scrambled_load", _scrambled_load_selftest,
                 milestone="G2", source="civitas_g.store.persistence"),
        SelfTest("assay_preconditions", _assay_preconditions_selftest,
                 milestone="G2", source="civitas_g.store.record"),
        SelfTest("engine_store", _engine_store_selftest,
                 milestone="G2", source="sim_v3_13 (engine G2-store-fix)"),
        SelfTest("store_decodes", _store_decodes_selftest,
                 milestone="G2", source="sim_v3_13 (engine G2-store-fix)"),
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
