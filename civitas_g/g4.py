"""G4 — a world that hardens, one mechanic per milestone.

    A record left by one population raises the competence of a population that never met it,
    **and the gain compounds.** (B§2)

G3 tested the first half. This is the second, and the spec — including what would falsify it — is
`docs/G4_SPEC.md`.

**Compounding is not "the effect survives a harder world."** That would follow from the record
merely continuing to work. It is the effect getting *larger*, because a record's value is the
trial-and-error it saves and a harder world makes trial-and-error cost more. So the number is a
difference of differences:

    compounding = content_effect(hardened) − content_effect(baseline)

with `content_effect` exactly as G3 defines it — `inherited store` minus `inherited scrambled` on
preparations-to-first-correct, everything but the labels' meaning held.

**Paired, not independent.** Both sides run at the same seed. A compounding number built from two
independent one-seed estimates would be dominated by the seed; built from a paired difference at
one seed it is at least a comparison of the same world draw against itself with one mechanic
moved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from civitas_g.g3 import G3Result, Succession, run_population_a, run_succession
from civitas_g.world.spec import check_hardened_world, world_at_k


@dataclass(frozen=True)
class Mechanic:
    """One hardening step: what it moves, and what it must not."""

    name: str
    #: The world this mechanic produces, given the baseline.
    world: dict[str, Any]
    moved: dict[str, tuple[Any, Any]]
    note: str = ""

    def describe(self) -> str:
        moves = ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in sorted(self.moved.items()))
        return f"{self.name}: {moves}"


def mechanic_k(k: int = 7) -> Mechanic:
    """Mechanic 1 — K raised.

    It hardens the world along the one axis the research lineage already established as
    load-bearing. v3.11's finding: *"At six mappings the genetic baseline is large ... and cannot
    be removed by shortening the era."* v3.12's answer was to enlarge the mapping space, and it
    worked. K = 7 takes the same lever further: P(7,3) = 210 mappings against P(5,3) = 60, so
    survival sorting over standing variation has three and a half times as much to sort through
    and within-life learning — and therefore an inherited record — is worth correspondingly more.

    It is also the mechanic with the least new machinery: no new fact, no new clock, no new
    dependency. If compounding is not visible here it is unlikely to be visible in the harder two,
    which is worth knowing before building them.
    """
    from math import perm

    world = world_at_k(k)
    moved = check_hardened_world(world)
    return Mechanic(
        name=f"K={k}",
        world=world, moved=moved,
        note=(f"mapping space P({k},3) = {perm(k, 3)} against P(5,3) = {perm(5, 3)}; "
              f"chance hit {1.0 / k:.3f} against {1.0 / 5:.3f}; prep_value moved to "
              f"{world['prep_value']} to hold the chance EV of a preparation at zero"),
    )


@dataclass
class G4Result:
    mechanic: Mechanic
    seed: int
    aligned: bool
    baseline: G3Result | None = None
    hardened: G3Result | None = None
    notes: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- the numbers

    @staticmethod
    def _content(r: G3Result | None) -> float:
        if r is None:
            return float("nan")
        by = {a.arm: a for a in r.arms}
        if "inherited store" not in by or "inherited scrambled" not in by:
            return float("nan")
        return by["inherited store"].nfc_mean - by["inherited scrambled"].nfc_mean

    @staticmethod
    def _reading(r: G3Result | None) -> float:
        if r is None:
            return float("nan")
        by = {a.arm: a for a in r.arms}
        if "inherited store" not in by or "inherited gain-zero" not in by:
            return float("nan")
        return by["inherited store"].nfc_mean - by["inherited gain-zero"].nfc_mean

    @staticmethod
    def _total(r: G3Result | None) -> float:
        if r is None:
            return float("nan")
        by = {a.arm: a for a in r.arms}
        return by["inherited store"].nfc_mean - by["fresh store"].nfc_mean

    def compounding(self) -> dict[str, float]:
        """**The G4 number**, and the two readings that say whether it means what it looks like.

        `content` is the claim. `total` is reported beside it because a compounding effect that
        shows up in the total and not in the content is the world making a store more valuable by
        *existing* — more marks, more cells covered — without its labels mattering more. G4's spec
        names that as one of three falsifications, and it is invisible unless both are reported.
        """
        return {
            "content_baseline": self._content(self.baseline),
            "content_hardened": self._content(self.hardened),
            "content_compounding": self._content(self.hardened) - self._content(self.baseline),
            "reading_baseline": self._reading(self.baseline),
            "reading_hardened": self._reading(self.hardened),
            "reading_compounding": self._reading(self.hardened) - self._reading(self.baseline),
            "total_baseline": self._total(self.baseline),
            "total_hardened": self._total(self.hardened),
            "total_compounding": self._total(self.hardened) - self._total(self.baseline),
        }

    def verdict(self) -> dict[str, Any]:
        """G4's three falsifications, checked rather than described.

        None of them is a statistical test. At one seed they are direction checks, and the report
        says so.
        """
        c = self.compounding()
        controls_agree = True
        for r in (self.baseline, self.hardened):
            if r is None:
                controls_agree = False
                continue
            by = {a.arm: a for a in r.arms}
            gap = abs(by["inherited scrambled"].nfc_mean - by["inherited gain-zero"].nfc_mean)
            scale = max(abs(self._content(r)), abs(self._reading(r)))
            controls_agree = controls_agree and scale > 0 and gap <= 0.35 * scale
        # The sign convention, because it inverts once and is easy to misread. The content
        # effect is measured on preparations-to-first-correct, where LOWER is better: G3 seed 0
        # aligned reads -0.131, a record that saves a third of a preparation. So an effect that
        # GREW under hardening is one that got MORE negative, and compounding < 0 is the claim.
        grew = bool(np.isfinite(c["content_compounding"]) and c["content_compounding"] < 0)
        content_carries_it = bool(
            np.isfinite(c["content_compounding"]) and np.isfinite(c["total_compounding"])
            and abs(c["content_compounding"]) >= 0.5 * abs(c["total_compounding"]))
        return {
            "content_effect_grew": grew,
            "content_carries_the_growth": content_carries_it,
            "controls_still_agree": controls_agree,
            "compounds": bool(grew and content_carries_it and controls_agree),
            "basis": "one seed, paired: a direction check, not a statistical test",
        }

    def table(self) -> str:
        c = self.compounding()
        rows = [f"  {self.mechanic.describe()}",
                f"  {self.mechanic.note}",
                "",
                f"  {'':<12}{'baseline':>12}{'hardened':>12}{'compounding':>14}"]
        for name in ("content", "reading", "total"):
            rows.append(f"  {name:<12}{c[f'{name}_baseline']:>12.3f}"
                        f"{c[f'{name}_hardened']:>12.3f}{c[f'{name}_compounding']:>+14.3f}")
        v = self.verdict()
        rows += ["", f"  content effect grew:        {v['content_effect_grew']}",
                 f"  content carries the growth: {v['content_carries_the_growth']}",
                 f"  controls still agree:       {v['controls_still_agree']}",
                 f"  -> compounds: {v['compounds']}   ({v['basis']})"]
        return "\n".join(rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mechanic": asdict(self.mechanic), "seed": self.seed, "aligned": self.aligned,
            "compounding": self.compounding(), "verdict": self.verdict(),
            "baseline": None if self.baseline is None else self.baseline.as_dict(),
            "hardened": None if self.hardened is None else self.hardened.as_dict(),
            "notes": self.notes,
        }


def run_mechanic(mechanic: Mechanic, *, seed: int = 0, a_phase_steps: int = 1500,
                 b_steps: int = 2100, claim_window: int = 700, aligned: bool = True,
                 baseline: G3Result | None = None, verbose: bool = False) -> G4Result:
    """Run one mechanic: the G3 succession at the baseline world and at the hardened one.

    `baseline` lets a caller pass a G3 result already in hand, so the baseline is not re-run for
    every mechanic — and so the compounding number is against the *same* baseline every time
    rather than against a fresh draw each mechanic, which would make the differences incomparable.
    """
    succession = Succession(seed=seed, a_phase_steps=a_phase_steps, b_steps=b_steps,
                            aligned=aligned, claim_window=claim_window)
    result = G4Result(mechanic=mechanic, seed=seed, aligned=aligned, baseline=baseline)

    if result.baseline is None:
        a = run_population_a(seed, a_phase_steps, verbose=verbose)
        result.baseline = run_succession(succession, a_result=a, verbose=verbose)

    a_hard = run_population_a(seed, a_phase_steps, verbose=verbose, world=mechanic.world)
    result.hardened = run_succession(succession, a_result=a_hard, verbose=verbose,
                                     world=mechanic.world)
    if result.hardened.arms and result.hardened.arms[0].store_density == 0.0:
        result.notes.append(
            "the hardened A left an empty record, so its arms are identical by construction")
    return result
