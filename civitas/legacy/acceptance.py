"""The acceptance run (Part B §63, §64, §65).

Every number in this project has been measured, but each in its own run, at its own commit, under
its own configuration. That is enough to establish a mechanism and not enough to accept a system:
six results produced by six code versions are six claims, and the thing being accepted is one
platform.

So `run_levels` runs all six of §65's scientific criteria **in one campaign, under one manifest**,
with each level's negative control beside it, and gates each one separately. A level whose gate
fails withholds its number rather than annotating it (ARCHITECTURE §3.1) — and the run as a whole
reports `accepted: false` rather than averaging around the gap.

**Levels 1 and 2 are separated here, and they were not before.** M4 and M9 reported both as
+0.300 from a single comparison, which conflated two different claims:

* **Level 1 — is external knowledge useful at all?** Hold the environment constant and vary
  whether the agent can see it: `collective` against `solo` *in the same matured workspace*. The
  control is a blinded agent, not an empty world.
* **Level 2 — does a newcomer benefit from what others built?** Vary the environment and hold the
  agent constant: `collective` in a matured workspace against `baseline_empty`.

They can come apart. An advantage at Level 2 with none at Level 1 would mean the matured workspace
helps for some reason other than its contents — which is exactly what the scramble control exists
to catch, and exactly the kind of thing a single conflated number hides.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from civitas.config import Settings, get_settings
from civitas.domain.enums import ExperimentArm
from civitas.persistence.models import Organization
from civitas.persistence.types import utcnow

#: Every level, its claim, and the control that makes it a claim rather than an observation.
LEVELS: tuple[tuple[int, str, str], ...] = (
    (1, "external knowledge usefulness",
     "an agent blinded to the same matured environment (`solo`)"),
    (2, "newcomer advantage",
     "the same agent in an empty environment (`baseline_empty`)"),
    (3, "ablation causality",
     "`memory_reset` removes the store; `collective_scrambled` keeps its size and destroys "
     "relevance"),
    (4, "distributed cognition",
     "each partition alone, proved insufficient by enumeration before the run"),
    (5, "cumulative culture",
     "the same chain with inheritance broken"),
    (6, "collective capability growth",
     "`solo` and `independent` at the same difficulty ladder"),
)


@dataclass
class LevelResult:
    """One level's outcome. `value` is `None` when a gate failed — never zero."""

    level: int
    name: str
    claim: str
    control: str
    value: float | None = None
    unit: str = ""
    control_value: float | None = None
    gates: dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    withheld_reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level, "name": self.name, "claim": self.claim,
            "control": self.control, "value": self.value, "unit": self.unit,
            "control_value": self.control_value, "gates": self.gates,
            "passed": self.passed, "withheld_reason": self.withheld_reason,
            "detail": self.detail,
        }


def _gates_of(result: Any) -> dict[str, Any]:
    return {name: gate.as_dict() for name, gate in (result.gates or {}).items()}


def _metrics_or_withheld(result: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return result.metrics(), None
    except Exception as exc:
        return None, str(exc)


def _fresh_org(session: Session, label: str) -> Organization:
    org = Organization(name=f"acceptance {label}", slug=f"acc-{uuid.uuid4().hex[:10]}")
    session.add(org)
    session.flush()
    return org


def run_levels(
    session: Session,
    *,
    seeds: list[int] | None = None,
    accumulation_passes: int = 3,
    settings: Settings | None = None,
    include: set[int] | None = None,
) -> list[LevelResult]:
    """Run §65's six levels in one campaign. Returns one `LevelResult` per level."""
    from civitas.legacy.benchmark import run_newcomer_procedure

    settings = settings or get_settings()
    seeds = seeds or [0, 1, 2, 3, 4]
    include = include or {1, 2, 3, 4, 5, 6}
    results: list[LevelResult] = []
    lookup = {level: (name, control) for level, name, control in LEVELS}

    # Levels 1, 2 and 3 come from one four-arm run plus one extra arm. Sharing the run is what
    # makes them comparable: three levels read off three different comparisons within a single
    # matched campaign, under one configuration hash.
    newcomer = None
    if include & {1, 2, 3}:
        from civitas.domains import get_domain

        org = _fresh_org(session, "newcomer")
        newcomer = run_newcomer_procedure(
            session, domain=get_domain("hidden_rule"), organization_id=org.id,
            experiment="acceptance_newcomer", seeds=seeds, count=6,
            accumulation_passes=accumulation_passes, settings=settings,
            generate_kwargs={"n_classes": 6, "n_ops": 10},
            include_arms=["baseline_empty", "collective", "memory_reset",
                          "collective_scrambled", "solo_in_mature"],
        )
        session.commit()

    if 1 in include:
        results.append(_level_one(newcomer, lookup))
    if 2 in include:
        results.append(_level_two(newcomer, lookup))
    if 3 in include:
        results.append(_level_three(newcomer, lookup))
    if 4 in include:
        results.append(_level_four(session, seeds, settings, lookup))
    if 5 in include:
        results.append(_level_five(session, seeds, settings, lookup))
    if 6 in include:
        results.append(_level_six(session, seeds, settings, lookup))
    return results


def _rate(metrics: dict[str, Any], arm: str) -> float | None:
    entry = (metrics.get("arms") or {}).get(arm)
    return None if entry is None else entry["success_rate"]


def _level_one(result: Any, lookup: dict[int, tuple[str, str]]) -> LevelResult:
    name, control = lookup[1]
    out = LevelResult(
        level=1, name=name,
        claim="an agent that can read the collective outperforms one that cannot, in the "
              "*same* environment",
        control=control, unit="success rate difference",
    )
    if result is None:
        out.withheld_reason = "level 1 was not included in this run"
        return out
    out.gates = _gates_of(result)
    metrics, withheld = _metrics_or_withheld(result)
    if metrics is None:
        out.withheld_reason = withheld
        return out
    mature = _rate(metrics, "collective")
    blind = _rate(metrics, "solo_in_mature")
    if mature is None or blind is None:
        # Absent, not zero: an arm that did not run is not an arm that scored nothing.
        out.withheld_reason = "the `solo_in_mature` control arm produced no readable episodes"
        return out
    out.value = round(mature - blind, 4)
    out.control_value = blind
    out.passed = out.value > 0
    out.detail = {"collective": mature, "solo_in_mature": blind,
                  "note": "both arms face the same matured workspace; only sight differs"}
    return out


def _level_two(result: Any, lookup: dict[int, tuple[str, str]]) -> LevelResult:
    name, control = lookup[2]
    out = LevelResult(
        level=2, name=name,
        claim="a fresh agent performs better in an environment others have worked in",
        control=control, unit="success rate difference",
    )
    if result is None:
        out.withheld_reason = "level 2 was not included in this run"
        return out
    out.gates = _gates_of(result)
    metrics, withheld = _metrics_or_withheld(result)
    if metrics is None:
        out.withheld_reason = withheld
        return out
    out.value = metrics["derived"]["newcomer_advantage"]
    out.control_value = _rate(metrics, "baseline_empty")
    out.passed = bool(out.value and out.value > 0)
    out.detail = {"collective": _rate(metrics, "collective"),
                  "baseline_empty": out.control_value,
                  "chance_level": metrics["chance_level"],
                  "naive_probe_ceiling": metrics["naive_probe_ceiling"]}
    return out


def _level_three(result: Any, lookup: dict[int, tuple[str, str]]) -> LevelResult:
    name, control = lookup[3]
    out = LevelResult(
        level=3, name=name,
        claim="the advantage is caused by what accumulated, not by the fact that something did",
        control=control, unit="fraction of the advantage removed",
    )
    if result is None:
        out.withheld_reason = "level 3 was not included in this run"
        return out
    out.gates = _gates_of(result)
    metrics, withheld = _metrics_or_withheld(result)
    if metrics is None:
        out.withheld_reason = withheld
        return out
    derived = metrics["derived"]
    reset = derived["ablation_removes_fraction"]
    scramble = derived["scramble_removes_fraction"]
    if reset is None or scramble is None:
        out.withheld_reason = (
            "there was no positive advantage to ablate, so the fraction removed is undefined"
        )
        return out
    out.value = reset
    out.control_value = scramble
    # Both ablations must remove most of it. An advantage that survived scrambling intact was
    # never about the *content* of what accumulated — that is the whole point of the control.
    out.passed = reset >= 0.5 and scramble >= 0.5
    out.detail = {"memory_reset_removes": reset, "scramble_removes": scramble,
                  "criterion": "each ablation removes at least half of the advantage"}
    return out


def _gate(name: str, passed: bool, detail: str, value: Any = None) -> dict[str, Any]:
    return {"passed": passed, "detail": detail, "value": value}


def _level_four(session: Session, seeds: list[int], settings: Settings,
                lookup: dict[int, tuple[str, str]]) -> LevelResult:
    """§23. Gates are defined here rather than on `DistributedResult`.

    Levels 1–3 come from `BenchmarkResult`, which gates itself. Levels 4–6 had no gate discipline
    at all until this milestone — their result objects report numbers unconditionally. Rather than
    reshape three result classes at the acceptance stage, the prerequisites each level needs to be
    *readable* are stated here, next to the claim they license.
    """
    from civitas.legacy.distributed_benchmark import run_distributed_benchmark

    name, control = lookup[4]
    out = LevelResult(
        level=4, name=name,
        claim="the collective solves what no single episode could, because the knowledge is "
              "split across populations",
        control=control, unit="success rate over the best single partition",
    )
    org = _fresh_org(session, "distributed")
    result = run_distributed_benchmark(session, organization_id=org.id, seeds=seeds)
    session.commit()

    payload = result.as_dict()
    derived = payload.get("derived") or {}
    arms = payload.get("arms") or {}
    singles = [arms.get("partition_a_only"), arms.get("partition_b_only")]
    smallest_n = min((a["n"] for a in arms.values() if a), default=0)

    proof = payload.get("unsolvable_proof") or {}
    out.gates = {
        # The strongest gate in the whole run, and it is cheap: the task's insufficiency is
        # *proved by enumeration* before any episode runs. A distributed-cognition result on a
        # task that turned out to be individually solvable would be evidence of nothing, and
        # discovering that afterwards costs a whole campaign.
        "task_is_provably_unsolvable_alone": _gate(
            "task_is_provably_unsolvable_alone",
            bool(proof) and not proof.get("violations"),
            f"enumeration found {len(proof.get('violations') or [])} partition(s) that could "
            f"determine an answer alone",
            proof.get("violations"),
        ),
        "both_controls_ran": _gate(
            "both_controls_ran", all(a is not None for a in singles),
            "each partition was run alone; without both, 'no single partition suffices' is "
            "an assertion rather than a measurement",
        ),
        "sufficient_n": _gate(
            "sufficient_n", smallest_n >= 3,
            f"smallest arm has n={smallest_n} readable episodes", smallest_n,
        ),
    }
    if not all(g["passed"] for g in out.gates.values()):
        out.withheld_reason = "a prerequisite gate failed; the run is not read"
        return out

    out.value = derived.get("gain_over_best_single_partition")
    out.control_value = max(
        (a["success_rate"] for a in singles if a), default=None
    )
    out.passed = bool(out.value is not None and out.value > 0)
    out.detail = {**derived, "arms": {k: v["success_rate"] for k, v in arms.items() if v}}
    return out


def _level_five(session: Session, seeds: list[int], settings: Settings,
                lookup: dict[int, tuple[str, str]]) -> LevelResult:
    """§24. The control is the same chain with inheritance broken."""
    from civitas.legacy.cumulative_benchmark import run_cumulative_benchmark

    name, control = lookup[5]
    out = LevelResult(
        level=5, name=name,
        claim="each generation builds on the last, reaching what no single generation could",
        control=control, unit="generations reached",
    )
    org = _fresh_org(session, "cumulative")
    result = run_cumulative_benchmark(session, organization_id=org.id, seeds=seeds)
    session.commit()

    payload = result.as_dict()
    summary = payload.get("summary") or result.summary()
    intact = summary.get("chained")
    broken = summary.get("broken_chain")

    out.gates = {
        "control_ran": _gate(
            "control_ran", intact is not None and broken is not None,
            "the broken-chain control is what distinguishes accumulation from four independent "
            "attempts at the same question",
        ),
        "chain_started": _gate(
            "chain_started", bool(intact and intact.get("deepest_generation_reached")),
            "generation 1 must be solvable, or the chain measures nothing",
            intact.get("deepest_generation_reached") if intact else None,
        ),
    }
    if not all(g["passed"] for g in out.gates.values()):
        out.withheld_reason = "a prerequisite gate failed; the run is not read"
        return out

    out.value = intact["deepest_generation_reached"]
    out.control_value = broken["deepest_generation_reached"]
    out.passed = bool(
        out.control_value is None or out.value > out.control_value
    )
    out.detail = {**(payload.get("derived") or {}), "summary": summary}
    return out


def _level_six(session: Session, seeds: list[int], settings: Settings,
               lookup: dict[int, tuple[str, str]]) -> LevelResult:
    """§25. The model is held constant; only the collective environment varies."""
    from civitas.legacy.cumulative_benchmark import run_capability_frontier

    name, control = lookup[6]
    out = LevelResult(
        level=6, name=name,
        claim="the collective's capability frontier advances while the model is held constant",
        control=control, unit="hardest difficulty solved",
    )
    org = _fresh_org(session, "frontier")
    # Threshold 0.75, not the function's default of 0.5. Measured, not chosen: M9 found that at
    # 0.5 every arm cleared every difficulty and the ladder reported a frontier of 14 for
    # everything. The `ladder_discriminates` gate below caught exactly that when this ran with the
    # default, which is the gate doing its job on its first outing.
    result = run_capability_frontier(
        session, organization_id=org.id, seeds=seeds[:2] or [0, 1], threshold=0.75,
    )
    session.commit()

    payload = result.as_dict()
    frontiers = payload.get("frontier") or {}
    controls = {k: v for k, v in frontiers.items() if k in ("solo", "independent")}

    out.gates = {
        "controls_ran": _gate(
            "controls_ran", len(controls) == 2,
            "`solo` and `independent` are both required: `independent` is the internal validity "
            "check that repeated sampling without transmission buys nothing",
            sorted(controls),
        ),
        "ladder_discriminates": _gate(
            "ladder_discriminates",
            len({v for v in frontiers.values()}) > 1,
            "every arm reaching the same frontier means the difficulty ladder is not "
            "discriminating, whatever the numbers say",
            frontiers,
        ),
    }
    if not all(g["passed"] for g in out.gates.values()):
        out.withheld_reason = "a prerequisite gate failed; the run is not read"
        return out

    out.value = frontiers.get("collective")
    present = [v for v in controls.values() if v is not None]
    out.control_value = max(present) if present else None
    out.passed = bool(
        out.value is not None
        and (out.control_value is None or out.value > out.control_value)
    )
    out.detail = {"frontier": frontiers, "threshold": payload.get("threshold")}
    return out


def acceptance_report(
    session: Session,
    *,
    seeds: list[int] | None = None,
    accumulation_passes: int = 3,
    settings: Settings | None = None,
    include: set[int] | None = None,
) -> dict[str, Any]:
    """The whole §65 acceptance run, stamped with one manifest."""
    from civitas.experiments.manifest import (
        dependency_versions,
        environment_facts,
        git_commit,
        schema_version,
    )
    from civitas.runtime.sandbox import get_sandbox

    settings = settings or get_settings()
    levels = run_levels(
        session, seeds=seeds, accumulation_passes=accumulation_passes, settings=settings,
        include=include,
    )
    sandbox = get_sandbox(settings)
    # `bool(levels)` first, and it is not defensive noise: `all([])` is True, so a report over
    # zero levels would accept the system on the strength of running nothing.
    accepted = bool(levels) and all(level.passed for level in levels)
    return {
        "generated_at": utcnow().isoformat(),
        "accepted": accepted,
        "criterion": (
            "Every level passes its own gates and beats its own negative control. One failing "
            "level fails the run: the levels are separate claims and an average over them would "
            "be a number no claim supports."
        ),
        "manifest": {
            "code_version": git_commit(),
            "schema_version": schema_version(),
            "dependencies": dependency_versions(),
            "environment": environment_facts(),
            "sandbox": sandbox.describe(),
            "seeds": seeds or [0, 1, 2, 3, 4],
            "accumulation_passes": accumulation_passes,
            # A run under weaker isolation must never be mistaken for one under the real thing.
            "sandbox_unenforced_limits": list(sandbox.unenforced_limits()),
        },
        "levels": [level.as_dict() for level in levels],
        "failing_levels": [level.level for level in levels if not level.passed],
    }


def arm_for_label(label: str) -> ExperimentArm:
    """The enum an acceptance arm label maps to. `solo_in_mature` is the Level 1 control: a
    genuinely blinded agent facing a workspace that others have matured."""
    return {
        "baseline_empty": ExperimentArm.COLLECTIVE,
        "collective": ExperimentArm.COLLECTIVE,
        "memory_reset": ExperimentArm.MEMORY_RESET,
        "collective_scrambled": ExperimentArm.COLLECTIVE_SCRAMBLED,
        "solo_in_mature": ExperimentArm.SOLO,
    }[label]
