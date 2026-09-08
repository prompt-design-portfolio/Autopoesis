"""The G1 gate, end to end.

Six clauses, from §3.10 of the G1 spec:

1. every numeric field matches the reference at the width the reference printed it;
2. on the reference's own seeds;
3. recomputed from the PERSISTED ROWS on both backends, and the two readings agree;
4. every self-test green before any number is read (B§6);
5. the code hashes in the manifest, naming which blob each number came from;
6. no number reported that the provider could not recompute -- absent is `None` with a reason.

Clause 3 is the one that gives "both backends" a meaning. The numbers come from numpy either way,
so running the simulation twice against two databases would prove nothing about either. What is
tested is that the round-trip through each backend lost nothing: the same run's rows are written to
each, read back, and the reading recomputed from what came out.

Clause 2 has a wrinkle the audit found. `precheck_v3_13.txt` records neither its seeds nor its
configuration (F5), so "its own seeds" has to be recovered. `recover_seed` does that by trying the
candidates in order and reporting which one matched -- and if none does, it says so and stops
rather than searching for parameters that would make the numbers agree. Tuning an invocation until
it reproduces a summary is passing by construction, which A1.2 calls failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from civitas_g.config import Settings, get_settings
from civitas_g.manifest import REFERENCE_INVOCATION, Reference
from civitas_g.reading.compute import compute_fields
from civitas_g.reading.parse import parse_summary
from civitas_g.reading.reproduce import ReproductionResult, backends_agree, compare


@dataclass
class GateOutcome:
    """One reference, gated. `passed` requires every clause, not the numeric one alone."""

    reference_key: str
    per_backend: dict[str, ReproductionResult] = field(default_factory=dict)
    backend_agreement: tuple[bool, list[str]] | None = None
    seed_used: int | None = None
    seed_recovered: bool = False
    selftests_ok: bool = False
    hashes_ok: bool = False
    withheld_reason: str | None = None

    @property
    def passed(self) -> bool:
        if self.withheld_reason is not None or not self.per_backend:
            return False
        if not (self.selftests_ok and self.hashes_ok):
            return False
        if self.backend_agreement is not None and not self.backend_agreement[0]:
            return False
        return all(r.passed for r in self.per_backend.values())

    def report(self) -> str:
        if self.withheld_reason:
            return f"{self.reference_key}: WITHHELD -- {self.withheld_reason}"
        lines = [f"{self.reference_key}: {'DIFF = 0' if self.passed else 'FAILED'}"]
        if self.seed_used is not None:
            how = "recovered by search" if self.seed_recovered else "as recorded"
            lines.append(f"  seed {self.seed_used} ({how})")
        lines.append(f"  self-tests {'green' if self.selftests_ok else 'NOT GREEN'}; "
                     f"G0 hashes {'verified' if self.hashes_ok else 'DO NOT MATCH'}")
        for _backend, result in sorted(self.per_backend.items()):
            lines.append("  " + result.report().replace("\n", "\n  "))
        if self.backend_agreement is not None:
            ok, problems = self.backend_agreement
            lines.append(f"  backends agree: {'yes' if ok else 'NO'}"
                         + ("" if ok else f" -- {len(problems)} field(s): {problems[:5]}"))
        return "\n".join(lines)


def reference_fields(reference: Reference) -> dict[str, float]:
    """The reference summary's printed numbers, keyed like the computed ones."""
    return parse_summary(reference.summary.path).fields


def gate_one(reference: Reference, results_by_backend: dict[str, dict[str, list[dict[str, Any]]]],
             *, settings: Settings | None = None, selftests_ok: bool,
             hashes_ok: bool, seed_used: int | None = None,
             seed_recovered: bool = False) -> GateOutcome:
    """Compare one reference against readings recomputed on each backend."""
    settings = settings or get_settings()
    outcome = GateOutcome(reference_key=reference.key, selftests_ok=selftests_ok,
                          hashes_ok=hashes_ok, seed_used=seed_used,
                          seed_recovered=seed_recovered)

    if not reference.gated:
        outcome.withheld_reason = reference.why
        return outcome

    ok, detail = reference.summary.verify()
    if not ok:
        outcome.withheld_reason = (
            f"the reference file no longer matches its G0 hash ({detail}). B§5.1 says to "
            f"reproduce against what was pinned at G0, not against anything that lands later.")
        return outcome

    expected = reference_fields(reference)
    computed_by_backend = {b: compute_fields(r) for b, r in results_by_backend.items()}

    for backend, computed in computed_by_backend.items():
        outcome.per_backend[backend] = compare(
            expected, computed, reference_key=reference.key,
            reference_path=reference.summary.path,
            reference_sha256=reference.summary.sha256,
            backend=backend, tolerance=settings.reproduction_tolerance)

    if len(computed_by_backend) >= 2:
        keys = sorted(computed_by_backend)
        outcome.backend_agreement = backends_agree(
            computed_by_backend[keys[0]], computed_by_backend[keys[1]],
            settings.backend_agreement_tolerance)
    return outcome


def recover_seed(reference: Reference,
                 reading_for_seed: Any,
                 candidates: tuple[int, ...] = (0, 1, 2),
                 *, settings: Settings | None = None) -> tuple[int | None, dict[int, int]]:
    """D4: find which seed `precheck_v3_13.txt` was produced on, or report that none was.

    `reading_for_seed(seed)` returns `{research name: [run dict]}` for that seed. Returns the
    matching seed and, for the record, how many fields each candidate matched -- because "seed 0
    matched 182 of 182 and seed 1 matched 31" is a different kind of evidence from "seed 0 matched
    182 and seed 1 matched 180", and a reader is entitled to see which one happened.

    If no candidate matches, the caller must report the artifact unreproducible. It must NOT go on
    to vary the configuration until something fits.
    """
    settings = settings or get_settings()
    expected = reference_fields(reference)
    scores: dict[int, int] = {}
    winner: int | None = None
    for seed in candidates:
        computed = compute_fields(reading_for_seed(seed))
        result = compare(expected, computed, reference_key=reference.key,
                         reference_path=reference.summary.path,
                         reference_sha256=reference.summary.sha256,
                         backend="sqlite", tolerance=settings.reproduction_tolerance)
        scores[seed] = result.n_matched
        if result.passed and winner is None:
            winner = seed
    return winner, scores


def invocation_note() -> str:
    """What the reference's invocation is, and which parts of it are inference (F5)."""
    inv = REFERENCE_INVOCATION
    lines = ["The reference invocation, recovered rather than recorded:"]
    for k in ("arms", "phase_steps", "prep_every", "n_seeds", "seed"):
        basis = inv.get(f"{k}_basis", "")
        value = inv.get(k)
        shown = "NOT RECOVERABLE" if value is None else repr(value)
        lines.append(f"  {k:<12}{shown}")
        if basis:
            lines.append(f"               {basis}")
    if not inv["matches_a_committed_mode"]:
        lines.append("  This configuration matches no committed MODE in make_notebook_v3_13.py "
                     f"({inv['modes']}), so the producing invocation is not committed either.")
    return "\n".join(lines)
