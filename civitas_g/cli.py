"""`python -m civitas_g` -- the G lineage's command line.

Four commands, and between them they answer the four questions a reader of a G1 number needs
answered before the number means anything: are the instruments working (`selftest`), is this the
world and the reference the build was pinned to (`pins`, `world`), and does it reproduce
(`reproduce`).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from civitas_g import __version__


def _factories(args: argparse.Namespace) -> dict[str, Callable[[], Any]]:
    from civitas_g.persistence.engine import create_all, create_db_engine, session_factory
    from civitas_g.persistence.models import Base

    urls: dict[str, str] = {"sqlite": args.sqlite_url}
    if args.postgres_url:
        urls["postgresql"] = args.postgres_url
    out: dict[str, Callable[[], Any]] = {}
    for name, url in urls.items():
        engine = create_db_engine(url=url)
        if args.reset:
            Base.metadata.drop_all(engine)
        create_all(engine)
        out[name] = session_factory(engine=engine)
    return out


def cmd_selftest(args: argparse.Namespace) -> int:
    from civitas_g.selftests import report, run_selftests

    print("self-tests (B§6) -- run before any result is read; any failure halts\n")
    results = run_selftests(names=args.only or None, verbose=True)
    print("\n" + report(results))
    halting = [r for r in results if r.halts]
    if halting:
        print(f"\n{len(halting)} self-test(s) halt a read.")
        return 1
    return 0


def cmd_pins(args: argparse.Namespace) -> int:
    from civitas_g.manifest import (
        MISSING_ARTIFACTS,
        REFERENCES,
        SIM_V3_13_DRIFT,
        engine_identity,
        verify_pins,
    )

    print("G0 pins (A3, B§5.1) -- the reference summaries and the code they were produced by\n")
    failures = 0
    for ok, detail in verify_pins():
        print(f"  [{'OK  ' if ok else 'FAIL'}] {detail}")
        failures += 0 if ok else 1

    print("\nreferences:")
    for ref in REFERENCES:
        gate = "GATED" if ref.gated else "recorded, not gated"
        print(f"  {ref.key:<18}{gate:<22}{ref.why}")
        for label, pin in (("sim", ref.sim), ("analysis", ref.analysis)):
            if pin is None:
                print(f"    {label:<10}NOT COMMITTED at {ref.summary.commit}")
            else:
                print(f"    {label:<10}{pin.sha256[:16]}  @ {pin.commit}")

    print(f"\nengine drift {SIM_V3_13_DRIFT['from']} -> {SIM_V3_13_DRIFT['to'][:7]}:")
    print(f"  {SIM_V3_13_DRIFT['change']}")
    print(f"  claim: {SIM_V3_13_DRIFT['claim']}")
    print(f"  verified by: {SIM_V3_13_DRIFT['verified_by']}")

    print("\nengine identity now:")
    for k, v in engine_identity().items():
        print(f"  {k:<22}{v}")

    print("\nnamed by the directive and absent from the repository:")
    for name, why in MISSING_ARTIFACTS:
        print(f"  {name}\n    {why}")
    return 1 if failures else 0


def cmd_world(args: argparse.Namespace) -> int:
    from civitas_g.world.adapter import (
        ACTIONS,
        DEAD_INPUTS,
        MODULATOR_EVENTS,
        describe_observation,
    )
    from civitas_g.world.arms import ARMS
    from civitas_g.world.spec import (
        CHANCE,
        TYPE_BLIND,
        WORLD,
        check_chance_ev_is_zero,
        check_world,
        clocks_of,
        standing_density,
    )

    check_world()
    ev = check_chance_ev_is_zero()

    print("THE WORLD OF RECORD -- analysis_v3_13.WORLD over sim_v3_13.Config\n")
    for k in sorted(WORLD):
        print(f"  {k:<20}{WORLD[k]!r}")
    print(f"\n  chance EV of a preparation: {ev:+.4f}   "
          f"(prep_value == (K-1) * prep_fail, so a chance preparation is worth nothing)")
    print(f"  levels: chance {CHANCE:.3f}, type-blind {TYPE_BLIND:.3f}, full 1.000")

    c = clocks_of(WORLD)
    print(f"\nERA CLOCKS (steps)\n  flip {c.flip_every}   mapping+pi {c.prep_every}   "
          f"patch drift {c.patch_drift_every}   label epoch spans "
          f"{c.pi_epochs_per_mapping} mapping era(s)")

    d = standing_density(WORLD)
    print(f"\nSTANDING DENSITIES\n  grid {d.grid_cells} cells; patches cover "
          f"{d.patch_cover_if_disjoint:.3f} of it if disjoint")
    print(f"  {d.spawn_attempts_per_step:.0f} spawn attempts/step against rot "
          f"{d.rot_per_cell_per_step}; mark half-life {d.mark_half_life_steps:.1f} steps")
    print("  standing food cover: NOT AVAILABLE -- consumption-limited, not in the engine's log, "
          "and A1.8 forbids adding it here (D10)")

    print(f"\nACTIONS ({len(ACTIONS)})")
    for a in ACTIONS:
        when = "phase 1 and 2" if a.live_in_phase_one else "phase 2 only"
        print(f"  {a.index:>2}  {a.name:<14}{when}")
    print("  reading the record is NOT an action: it is an observation, so a null cannot be "
          "ambiguous between 'cannot read' and 'did not bother'")

    print("\nMODULATOR EVENTS -- the complete list. The modulator is this table, not the energy "
          "delta (D7)")
    print(f"  {'event':<16}{'m':>4}  {'energy':<16}{'consumes':<10}note")
    for e in MODULATOR_EVENTS:
        consumes = "yes" if e.consumes_cell else "no"
        print(f"  {e.event:<16}{e.m:>+4.0f}  {e.energy:<16}{consumes:<10}{e.note}")

    print(f"\nOBSERVATION LAYOUT ({len(DEAD_INPUTS)} of 31 inputs structurally dead)")
    print(describe_observation())

    print("\nARMS (B§4)")
    print(f"  {'manifest name':<26}{'research name':<28}{'milestone':<11}holds / removes")
    for a in ARMS:
        flag = "" if a.in_directive else "  [not in B§4's table; carried for the reproduction]"
        print(f"  {a.name:<26}{a.research_name:<28}{a.milestone:<11}{a.holds_or_removes}{flag}")
    return 0


def cmd_store(args: argparse.Namespace) -> int:
    """G2's state: what the store layer can do, and the one thing it cannot."""
    import sim_v3_13
    from civitas_g.manifest import ENGINE_VERSIONS, verify_engine
    from civitas_g.selftests import PATCH_PATH
    from civitas_g.store.record import ScrambleMode

    print("THE RECORD AS AN ARTIFACT STORE (G2)\n")
    ok, detail = verify_engine()
    print(f"  engine:            {detail}")
    if not ok:
        print("  the engine on disk is not a version this build records; nothing below is safe")
        return 1
    print("\n  engine versions (A1.8: the hash is in every manifest; a change is versioned, "
          "not silent)")
    for v in ENGINE_VERSIONS:
        print(f"    {v.label:<12}{v.sha256[:12]}  {v.change[:80]}")
        if v.equivalence:
            print(f"    {'':<12}{'':<12}  checked by: {v.equivalence[:78]}")
    print(f"\n  the change is stated readably in {PATCH_PATH}")
    print("  run() returns the store:  yes, when cfg.store_snaps")
    print("  run() accepts a store:    yes, run(..., init_store=...)")

    print(f"\n  store shape:       ({sim_v3_13.N_TYPES}, {sim_v3_13.N_PREPS}, "
          f"{sim_v3_13.Config().grid}, {sim_v3_13.Config().grid}) signed floats")
    print("\n  variants (A1.4: a control is stored BESIDE its parent, never in place of it)")
    print(f"    {'real':<22}the record as captured")
    print(f"    {'hidden':<22}same array, marks zeroed -- the presence control (A2.3)")
    print(f"    {'scrambled:':<22}", end="")
    print(f"{ScrambleMode.PER_CELL.value:<14}B§5.2's inherited-scrambled control")
    print(f"    {'scrambled:':<22}{ScrambleMode.GLOBAL.value:<14}"
          f"A2.1's label-permuted ASSAY arm -- not a control")
    print("\n  A global permutation is isomorphic to the real store, so using it as the "
          "inherited-\n  scrambled control would read as a null while information had passed. "
          "See G2-D2.")
    print("\n  provenance: at the ARTIFACT level (run, era, engine, mapping, pi). NOT per mark --")
    print("  the engine's marks carry no writer and no timestamp. See G2-D5.")
    return 0


def cmd_g3(args: argparse.Namespace) -> int:
    """A succession: A lives and dies, B is born into what it left."""
    from civitas_g.g3 import Succession, b_founders_carry_no_h, run_population_a, run_succession

    ok, detail = b_founders_carry_no_h()
    print(f"b_founders_carry_no_H: {'PASS' if ok else 'FAIL'} -- {detail}\n")
    if not ok:
        print("B§6: any failure halts. No B number is read.")
        return 1

    a = run_population_a(args.seed, args.a_phase_steps, verbose=False)
    final = a.raw.get("final_store")
    if final is None:
        print("population A produced no final store; the engine must be G2-store-fix or later")
        return 1
    import numpy as np

    print(f"population A: {a.wall_seconds:.0f}s, final era {tuple(final['mapping'])}, "
          f"pi {tuple(final['pi'])}, record density "
          f"{(np.abs(final['marks']) > 1e-3).mean():.6f}\n")

    alignments = (True, False) if args.both_alignments else (not args.misaligned,)
    failed = False
    for aligned in alignments:
        s_ = Succession(seed=args.seed, a_phase_steps=args.a_phase_steps,
                        b_steps=args.b_steps, aligned=aligned)
        r = run_succession(s_, a_result=a)
        print(f"=== population B, {s_.alignment}: A's era {r.a_mapping}, "
              f"B started at {r.b_mapping} ===")
        print(r.table())
        print("  claim lines, each arm minus `fresh store`:")
        for arm, lines in r.claim_lines().items():
            print(f"    {arm:<22}" + "  ".join(f"{k} {v:+.3f}" for k, v in lines.items()))
        if r.gate_r:
            g = r.gate_r
            print(f"  Gate R on the inherited marks under B's pi-epochs: obs {g['obs']:.3f} "
                  f"null {g['null']:.3f} z {g['z']:.2f} -> {g['verdict']}")
            failed = failed or g["verdict"] != "PASS"
        for note in r.notes:
            print(f"  NOTE: {note}")
        print()
    print("One seed is a PRE-CHECK. B§5.2 asks for 3/3 seeds with the scrambled and gain-zero")
    print("arms flat; nothing above is a claim until that runs.")
    return 1 if failed else 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    from civitas_g.g1 import run_g1

    factories = _factories(args)
    report = run_g1(factories, seed=args.seed, phase_steps=args.phase_steps,
                    verbose=True, skip_selftests=args.skip_selftests)
    print()
    print(report.report())
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(report.report())
        print(f"\nwritten to {args.out}")
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="civitas_g", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("selftest", help="B§6's self-tests; any failure halts a read")
    s.add_argument("--only", nargs="*", default=None, help="run only these by name")
    s.set_defaults(func=cmd_selftest)

    s = sub.add_parser("pins", help="verify the G0 hashes against the working tree")
    s.set_defaults(func=cmd_pins)

    s = sub.add_parser("world", help="the world of record, the arms, and the adapter")
    s.set_defaults(func=cmd_world)

    s = sub.add_parser("store", help="G2: the record as an artifact store, and its blocker")
    s.set_defaults(func=cmd_store)

    s = sub.add_parser("g3", help="a succession: B born into A's record")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--a-phase-steps", type=int, default=1500,
                   help="A's phase length; A runs two phases of this")
    s.add_argument("--b-steps", type=int, default=700,
                   help="B's single chain-on phase (G3-D1)")
    s.add_argument("--both-alignments", action="store_true", default=True)
    s.add_argument("--misaligned", action="store_true",
                   help="run only the misaligned arm (B§5.2's pre-registered control)")
    s.set_defaults(func=cmd_g3)

    s = sub.add_parser("campaign", help="a G3 campaign across seeds, under fixed criteria")
    s.add_argument("--dir", default="var/g3", help="where the per-seed results are (var/g3)")
    s.set_defaults(func=cmd_campaign)

    s = sub.add_parser("reproduce", help="the G1 gate: reproduction diff = 0")
    s.add_argument("--sqlite-url", default="sqlite:///var/civitas_g.db")
    s.add_argument("--postgres-url", default=None,
                   help="without this, D12's both-backends clause is reported UNMET")
    s.add_argument("--seed", type=int, default=None,
                   help="override the recovered seed (D4); the file records none")
    s.add_argument("--phase-steps", type=int, default=None,
                   help="override the recovered phase length; for smoke runs only")
    s.add_argument("--reset", action="store_true", help="drop and recreate the G tables first")
    s.add_argument("--skip-selftests", action="store_true",
                   help="B§6 says any failure halts; this exists for iterating on the gate "
                        "itself and must not be used to produce a reported number")
    s.add_argument("--out", default=None, help="write the report to this path")
    s.set_defaults(func=cmd_reproduce)
    return p


def cmd_campaign(args) -> int:
    """The campaign report. Reads stored results only -- it runs nothing and gates nothing."""
    from civitas_g.campaign import report
    print(report(args.dir))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
