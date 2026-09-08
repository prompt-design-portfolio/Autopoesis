"""The G1 gate: reproduction diff = 0.

    G1 | population provider; the summaries committed in the repository at G0 ... reproduced to
    three decimals on their own seeds, both backends; the reference summaries and the code they
    were produced by recorded by hash | reproduction diff = 0

Six clauses, and the ones that are easy to get wrong are the last three.

**Rounding, not tolerance alone (D3).** A reference summary records only what it printed. Comparing
a stored `0.05074321...` against a printed `0.5074` at any tolerance tests the print width, not the
number. So both sides are rounded to the width the reference printed that column at, and then
compared. The tolerance survives underneath for the boundary case where two values straddle a
rounding edge.

**NaN equals NaN here, and NaN does not equal a number.** Both sides carrying `nan` means both
sides could not compute the cell -- an agreement about an absence, which is a real agreement. A
`nan` against a number is the worst kind of mismatch and is reported as one, never as a skip.

**Absent is not zero.** A field present on one side only is `missing`, counted separately, and it
fails the gate. A gate that quietly compared the intersection would report a pass for a
reproduction that had lost half its tables.

**Text is reported, never gated (D3).** HEAD's `analysis_v3_13` prints an extra `(ii-newborn)`
block and reworded prose against the blob that produced `precheck_v3_13.txt`. A text diff would
fail on the prose and pass on nothing, so it is produced beside the gate and is not part of it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from civitas_g.reading.compute import decimals_for


@dataclass
class FieldDiff:
    """One compared cell."""

    field: str
    reference: float | None
    computed: float | None
    decimals: int
    delta: float | None
    status: str          # "match" | "mismatch" | "missing_computed" | "missing_reference"

    @property
    def ok(self) -> bool:
        return self.status == "match"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("reference", "computed", "delta"):
            v = d[k]
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
        return d


@dataclass
class ReproductionResult:
    reference_key: str
    reference_path: str
    reference_sha256: str
    backend: str
    tolerance: float
    diffs: list[FieldDiff] = field(default_factory=list)
    withheld_reason: str | None = None

    @property
    def n_fields(self) -> int:
        return len(self.diffs)

    @property
    def n_matched(self) -> int:
        return sum(1 for d in self.diffs if d.ok)

    @property
    def n_missing(self) -> int:
        return sum(1 for d in self.diffs if d.status.startswith("missing"))

    @property
    def n_mismatched(self) -> int:
        return sum(1 for d in self.diffs if d.status == "mismatch")

    @property
    def passed(self) -> bool:
        """Diff = 0: every compared field matched, nothing was missing, and something was
        compared. `all([])` is True, which would make an empty reproduction a passing one."""
        return (self.withheld_reason is None and self.n_fields > 0
                and self.n_matched == self.n_fields)

    def failures(self) -> list[FieldDiff]:
        return [d for d in self.diffs if not d.ok]

    def report(self, limit: int = 40) -> str:
        if self.withheld_reason:
            return (f"{self.reference_key} on {self.backend}: WITHHELD -- {self.withheld_reason}\n"
                    f"  A withheld number is not a zero and not a failure to be averaged around.")
        detail = f", {self.n_mismatched} differ, {self.n_missing} missing"
        head = (f"{self.reference_key} on {self.backend}: "
                f"{self.n_matched}/{self.n_fields} fields matched"
                f"{'' if self.passed else detail}"
                f"  ->  {'DIFF = 0' if self.passed else 'FAILED'}")
        bad = self.failures()
        if not bad:
            return head
        def cell(value: float | None, decimals: int) -> str:
            if value is None:
                return "--"
            return "nan" if math.isnan(value) else f"{value:.{decimals}f}"

        lines = [head,
                 f"  {'field':<48}{'reference':>12}{'computed':>12}{'delta':>12}  status"]
        for d in bad[:limit]:
            delta = "--" if d.delta is None else (
                "nan" if math.isnan(d.delta) else f"{d.delta:+.{max(d.decimals, 4)}f}")
            lines.append(f"  {d.field:<48}{cell(d.reference, d.decimals):>12}"
                         f"{cell(d.computed, d.decimals):>12}{delta:>12}  {d.status}")
        if len(bad) > limit:
            lines.append(f"  ... {len(bad) - limit} more")
        return "\n".join(lines)

    def as_rows(self) -> list[dict[str, Any]]:
        return [d.as_dict() for d in self.diffs]


def _agrees(a: float, b: float, decimals: int, tolerance: float) -> tuple[bool, float]:
    """Rounded to the reference's print width, then compared. Returns `(ok, delta)`."""
    if math.isnan(a) and math.isnan(b):
        return True, 0.0                       # both could not compute it: a real agreement
    if math.isnan(a) or math.isnan(b):
        return False, math.nan                 # one computed it and the other did not
    delta = b - a
    if round(a, decimals) == round(b, decimals):
        return True, delta
    return abs(delta) <= tolerance, delta


def compare(reference_fields: dict[str, float], computed_fields: dict[str, float], *,
            reference_key: str, reference_path: str, reference_sha256: str,
            backend: str, tolerance: float) -> ReproductionResult:
    """Every field on either side, compared or reported missing."""
    result = ReproductionResult(
        reference_key=reference_key, reference_path=reference_path,
        reference_sha256=reference_sha256, backend=backend, tolerance=tolerance)

    for name in sorted(set(reference_fields) | set(computed_fields)):
        decimals = decimals_for(name)
        if name not in computed_fields:
            result.diffs.append(FieldDiff(name, reference_fields[name], None, decimals, None,
                                          "missing_computed"))
            continue
        if name not in reference_fields:
            result.diffs.append(FieldDiff(name, None, computed_fields[name], decimals, None,
                                          "missing_reference"))
            continue
        ref, got = float(reference_fields[name]), float(computed_fields[name])
        ok, delta = _agrees(ref, got, decimals, tolerance)
        result.diffs.append(FieldDiff(name, ref, got, decimals, delta,
                                      "match" if ok else "mismatch"))
    return result


def backends_agree(a: dict[str, float], b: dict[str, float],
                   tolerance: float) -> tuple[bool, list[str]]:
    """D12: the same rows read on two backends must give the same reading.

    This is the only sense in which "reproduced on both backends" says anything. The numbers come
    from numpy either way, so running the simulation twice against two databases would prove
    nothing about either; what is being tested is that the round-trip through each backend lost no
    precision. The tolerance is float noise, not three decimals, because both sides recompute from
    the same stored values.
    """
    problems: list[str] = []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            problems.append(f"{name}: present on only one backend")
            continue
        x, y = float(a[name]), float(b[name])
        if math.isnan(x) and math.isnan(y):
            continue
        if math.isnan(x) or math.isnan(y) or abs(x - y) > tolerance:
            problems.append(f"{name}: {x!r} vs {y!r}")
    return (not problems), problems
