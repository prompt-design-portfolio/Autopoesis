"""G1, end to end: self-tests, hashes, the campaign, the gate.

    No milestone starts until the previous gate is met and I have agreed the next spec. (A3)

So this module runs G1 and reports whether its gate is met. It does not decide that it is: a
`G1Report` with `passed = False` is the ordinary outcome of a first run, and the write-up A4.3 asks
for is built from the report either way.

The order is B§6's, and it is not negotiable: **self-tests first, and any failure halts**. A
reproduction computed after a failed self-test would be a number produced by an instrument known
to be broken, which is worse than no number.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from civitas_g.config import Settings, get_settings
from civitas_g.manifest import (
    MISSING_ARTIFACTS,
    PRECHECK_V3_13,
    REFERENCE_INVOCATION,
    Reference,
    verify_pins,
)
from civitas_g.persistence.models import Campaign, Reproduction
from civitas_g.provider.population import (
    CampaignPlan,
    load_results,
    run_campaign_on_backends,
)
from civitas_g.reading.gate import GateOutcome, gate_one, invocation_note
from civitas_g.selftests import SelfTestResult, halt_on_failure, run_selftests
from civitas_g.world.arms import REFERENCE_ARM_ORDER


@dataclass
class G1Report:
    selftests: list[SelfTestResult] = field(default_factory=list)
    pins: list[tuple[bool, str]] = field(default_factory=list)
    outcomes: list[GateOutcome] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def selftests_ok(self) -> bool:
        return not any(r.halts for r in self.selftests)

    @property
    def hashes_ok(self) -> bool:
        return all(ok for ok, _ in self.pins)

    @property
    def passed(self) -> bool:
        return (self.selftests_ok and self.hashes_ok and bool(self.outcomes)
                and all(o.passed for o in self.outcomes))

    def report(self) -> str:
        from civitas_g.selftests import report as selftest_report

        lines = ["=" * 78, "G1 -- population provider; reproduction diff = 0", "=" * 78, ""]
        lines.append(selftest_report(self.selftests))
        lines.append("")
        bad_pins = [d for ok, d in self.pins if not ok]
        lines.append(f"G0 hashes: {sum(1 for ok, _ in self.pins if ok)}/{len(self.pins)} verified"
                     + ("" if not bad_pins else ":\n  " + "\n  ".join(bad_pins)))
        lines.append("")
        lines.append(f"backends: {', '.join(self.backends) if self.backends else 'NONE'}")
        if len(self.backends) < 2:
            lines.append("  D12's clause is UNMET: 'both backends' means the same rows read back "
                         "on each, and only one is configured. The numeric result below stands "
                         "for the backend named; the backend-agreement clause does not.")
        lines.append("")
        lines.append(invocation_note())
        lines.append("")
        for outcome in self.outcomes:
            lines.append(outcome.report())
            lines.append("")
        if MISSING_ARTIFACTS:
            lines.append("Named by the directive and absent from the repository:")
            for name, why in MISSING_ARTIFACTS:
                lines.append(f"  {name}: {why}")
            lines.append("")
        for note in self.notes:
            lines.append(note)
        lines.append("=" * 78)
        lines.append(f"G1 gate: {'MET -- reproduction diff = 0' if self.passed else 'NOT MET'}")
        lines.append("=" * 78)
        return "\n".join(lines)


def reference_plan(reference: Reference = PRECHECK_V3_13, *, seed: int | None = None,
                   phase_steps: int | None = None) -> CampaignPlan:
    """The campaign that should reproduce a reference, at its recovered invocation (F5, D4)."""
    inv = REFERENCE_INVOCATION
    return CampaignPlan(
        kind="reproduction",
        label=f"reproduce {reference.summary.path} @ {reference.summary.sha256[:12]}",
        arms=list(REFERENCE_ARM_ORDER),
        seeds=[inv["seed"] if seed is None and inv["seed"] is not None else (seed or 0)],
        phase_steps=phase_steps or int(inv["phase_steps"]),
        notes=[
            f"reference {reference.summary.path} sha256 {reference.summary.sha256}",
            f"producing code pinned at {reference.summary.commit}: "
            f"{reference.sim.path if reference.sim else 'NOT COMMITTED'}",
            "invocation recovered by inference; the file records neither seeds nor config (F5)",
            f"seed basis: {inv['seed_basis']}",
        ],
    )


def store_outcome(session: Any, campaign_id: Any, outcome: GateOutcome,
                  settings: Settings | None = None) -> list[Reproduction]:
    """Persist the gate result. Every compared field, not only the failures."""
    settings = settings or get_settings()
    campaign = session.get(Campaign, campaign_id)
    rows: list[Reproduction] = []
    for backend, result in outcome.per_backend.items():
        row = Reproduction(
            campaign_id=campaign.id, reference_key=result.reference_key,
            reference_sha256=result.reference_sha256, backend=backend,
            tolerance=result.tolerance, n_fields=result.n_fields,
            n_matched=result.n_matched, n_missing=result.n_missing,
            passed=result.passed, withheld_reason=result.withheld_reason,
            diff=result.as_rows(), finished_at=datetime.now(UTC),
        )
        session.add(row)
        rows.append(row)
    if not outcome.per_backend and outcome.withheld_reason:
        row = Reproduction(
            campaign_id=campaign.id, reference_key=outcome.reference_key,
            reference_sha256="", backend="-", tolerance=settings.reproduction_tolerance,
            n_fields=0, n_matched=0, n_missing=0, passed=False,
            withheld_reason=outcome.withheld_reason, diff=[],
            finished_at=datetime.now(UTC),
        )
        session.add(row)
        rows.append(row)
    session.flush()
    return rows


def run_g1(session_factories: dict[str, Callable[[], Any]], *,
           reference: Reference = PRECHECK_V3_13,
           seed: int | None = None, phase_steps: int | None = None,
           settings: Settings | None = None, verbose: bool = True,
           skip_selftests: bool = False) -> G1Report:
    """Run G1 and return the report. Self-tests first; any failure halts (B§6)."""
    settings = settings or get_settings()
    report = G1Report(backends=sorted(session_factories))

    if verbose:
        print("self-tests (B§6) -- any failure halts:", flush=True)
    report.selftests = ([] if skip_selftests
                        else run_selftests(verbose=verbose))
    if not skip_selftests:
        halt_on_failure(report.selftests)

    report.pins = verify_pins()
    if not report.hashes_ok:
        report.notes.append(
            "A G0 pin no longer matches the working tree. B§5.1 requires reproducing against what "
            "was pinned at G0, so the numeric result below is not the gate it claims to be.")

    plan = reference_plan(reference, seed=seed, phase_steps=phase_steps)
    if verbose:
        print(f"\ncampaign: {len(plan.arms)} arms x {len(plan.seeds)} seed(s) at "
              f"{plan.phase_steps}-step phases on {', '.join(report.backends)}", flush=True)
    campaign_ids = run_campaign_on_backends(session_factories, plan, verbose=False)

    results_by_backend: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for backend, factory in session_factories.items():
        with factory() as session:
            results_by_backend[backend] = load_results(session, campaign_ids[backend])

    outcome = gate_one(
        reference, results_by_backend, settings=settings,
        selftests_ok=report.selftests_ok, hashes_ok=report.hashes_ok,
        seed_used=plan.seeds[0],
        seed_recovered=REFERENCE_INVOCATION["seed"] is None)
    report.outcomes.append(outcome)

    for backend, factory in session_factories.items():
        with factory() as session:
            store_outcome(session, campaign_ids[backend], outcome, settings)
            session.commit()

    if not outcome.passed and outcome.withheld_reason is None:
        report.notes.append(
            "D4: if no candidate seed reproduces the reference, the artifact is declared "
            "unreproducible and the gate falls back to v3_11_grid_summary.txt. It is NOT to be "
            "made to pass by varying the configuration -- A1.2 calls that failing.")
    return report
