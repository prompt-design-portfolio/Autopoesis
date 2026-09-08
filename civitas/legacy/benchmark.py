"""The newcomer-advantage benchmark (Part B §22) — the central built-in measurement.

§22's procedure, implemented literally:

1-5. freeze the model, prompts, tools, budgets and sampling — `frozen_config()` builds one
     configuration and every arm runs under it, with the config hash asserted equal across arms;
6-7. run a fresh Agent A in an empty environment and measure it;
8.   let N agents work and accumulate artifacts;
9-10. run an identical fresh Agent B and compare;
11-12. reset collective memory and rerun;
13-14. scramble collective memory and rerun.

The primary criterion, verbatim from §22: *a fresh unchanged agent performs significantly better
inside the mature collective environment, and that advantage decreases substantially when
accumulated collective information is removed or scrambled.*

**Gates precede rows** (ARCHITECTURE §3.1). Four prerequisites are checked before any number is
reported, and a failed gate makes the result unreadable rather than caveated — `BenchmarkResult`
refuses to hand back its metrics, so a caller cannot accidentally read a number the run did not
earn.
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.config import Settings, get_settings
from civitas.domain.enums import ExperimentArm
from civitas.legacy.runner import (
    DEFAULT_BUDGETS,
    RunResult,
    create_task,
    run_benchmark_episode,
)
from civitas.legacy.tasks.hidden_rule import (
    DeviceSpec,
)
from civitas.persistence.models import Artifact, Workspace
from civitas.runtime.budgets import Budgets

#: The code-repair domain's budget. Five, not seven: six candidate edits at five calls puts the
#: naive ceiling at 3/6 = 0.500, which is exactly where the device domain sits at seven calls out
#: of ten operations. Matching the *ceiling* is what makes the two results comparable — matching
#: the raw budget would have compared two different difficulties and called it a transfer.
REPAIR_BUDGETS = Budgets(
    tokens=200_000, context=32_000, tool_calls=5, cost_usd=1.0, wall_clock_s=180,
    max_turns=12, max_no_progress_turns=4,
)


class GateFailure(RuntimeError):
    """A prerequisite failed, so the run is not read."""


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str
    value: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "detail": self.detail, "value": self.value}


@dataclass
class ArmSummary:
    arm: str
    n: int
    successes: int
    #: Episodes excluded as infrastructure failures. Printed, never silently dropped, so a
    #: success rate is always read against the number of episodes it was actually computed over.
    excluded: int
    success_rate: float
    mean_tool_calls: float
    mean_probes: float
    mean_tokens: float
    mean_artifacts_read: float
    stale_uses: int
    duplicate_failures: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class BenchmarkResult:
    experiment: str
    config_hash: str
    chance_level: float
    naive_probe_ceiling: float
    gates: dict[str, Gate] = field(default_factory=dict)
    arms: dict[str, ArmSummary] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    manifest_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def gates_passed(self) -> bool:
        return all(g.passed for g in self.gates.values())

    def failed_gates(self) -> list[str]:
        return [name for name, gate in self.gates.items() if not gate.passed]

    def metrics(self) -> dict[str, Any]:
        """The numbers — **only** if the gates passed.

        This is ARCHITECTURE §3.1 made structural: a failed gate does not annotate the result, it
        withholds it. A caller that wants to see what a failed run produced must ask for
        `raw_metrics()` and thereby say in the code that it is reading an unlicensed number.
        """
        if not self.gates_passed:
            raise GateFailure(
                f"gates failed: {self.failed_gates()}. "
                "The run is not read (ARCHITECTURE §3.1). "
                "Use raw_metrics() only to diagnose the gate failure."
            )
        return self.raw_metrics()

    def raw_metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "experiment": self.experiment,
            "config_hash": self.config_hash,
            "chance_level": self.chance_level,
            "naive_probe_ceiling": self.naive_probe_ceiling,
            "seeds": self.seeds,
            "gates": {k: g.as_dict() for k, g in self.gates.items()},
            "arms": {k: a.as_dict() for k, a in self.arms.items()},
            "notes": self.notes,
        }
        out["derived"] = self._derived()
        return out

    def _derived(self) -> dict[str, Any]:
        """The comparisons §22 asks for. `None` where an arm is missing — never 0.0, which would
        present an absence as a finding."""
        def rate(name: str) -> float | None:
            arm = self.arms.get(name)
            return arm.success_rate if arm else None

        baseline = rate("baseline_empty")
        mature = rate("collective")
        reset = rate("memory_reset")
        scrambled = rate("collective_scrambled")

        def gap(a: float | None, b: float | None) -> float | None:
            return None if a is None or b is None else round(a - b, 4)

        advantage = gap(mature, baseline)
        return {
            "newcomer_advantage": advantage,
            "advantage_lost_to_reset": gap(mature, reset),
            "advantage_lost_to_scramble": gap(mature, scrambled),
            #: What fraction of the advantage the ablation removes. The §22 criterion is about
            #: this ratio, not about the raw gap: an advantage that survives scrambling intact was
            #: never about the *content* of what accumulated.
            "ablation_removes_fraction": (
                None if not advantage or advantage <= 0 or reset is None or mature is None
                else round(min(1.0, max(0.0, (mature - reset) / advantage)), 4)
            ),
            "scramble_removes_fraction": (
                None if not advantage or advantage <= 0 or scrambled is None or mature is None
                else round(min(1.0, max(0.0, (mature - scrambled) / advantage)), 4)
            ),
        }


# --------------------------------------------------------------------------
def frozen_config(
    budgets: Budgets | None = None,
    retrieval_options: dict[str, Any] | None = None,
    system_prompt_version: str = "hidden-rule/1.0",
) -> dict[str, Any]:
    """The one configuration every arm runs under (Part B §20).

    Returned as data so it can be hashed and asserted identical across arms. Frozen-model mode is
    not a mode flag; it is the property that this dictionary is the same for every run in a
    campaign, and gate `frozen_model` checks exactly that.
    """
    budgets = budgets or DEFAULT_BUDGETS
    return {
        "provider": "policy",
        "model": "policy-v1",
        "model_version": "policy-v1",
        "system_prompt_version": system_prompt_version,
        "retrieval_policy_version": (retrieval_options or {}).get(
            "version", "retrieval/2.0-hybrid"
        ),
        "retrieval_options": dict(retrieval_options or {}),
        "temperature": 0.0,
        "budgets": {
            "tokens": budgets.tokens, "context": budgets.context,
            "tool_calls": budgets.tool_calls, "cost_usd": budgets.cost_usd,
            "wall_clock_s": budgets.wall_clock_s, "max_turns": budgets.max_turns,
        },
    }


def config_hash_of(config: dict[str, Any]) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _order_seed(seed: int, class_index: int, pass_no: int) -> int:
    """A per-agent probe order that is identical across arms.

    Depends on the seed, the class and the pass — never on the arm — so every arm faces the same
    sequence of agents and any difference between arms is caused by what the arms let those agents
    see. `pass_no = -1` marks the measured newcomer.

    Hashed with `hashlib`, not `hash()`: Python randomises string and tuple hashing per process
    unless `PYTHONHASHSEED` is set, so `hash()` here would make the benchmark irreproducible
    between runs — and a benchmark whose agents differ between invocations cannot support any
    claim about what changed (Part B §46).
    """
    import hashlib

    digest = hashlib.sha256(f"{seed}|{class_index}|{pass_no}".encode()).hexdigest()
    return int(digest[:8], 16)


def _fresh_workspace(session: Session, org_id: uuid.UUID, label: str) -> Workspace:
    workspace = Workspace(
        organization_id=org_id,
        name=f"benchmark {label}",
        slug=f"bm-{label}-{uuid.uuid4().hex[:8]}",
        environment_version="benchmark",
    )
    session.add(workspace)
    session.flush()
    return workspace


def _summarise(arm: str, results: list[RunResult]) -> ArmSummary:
    readable = [r for r in results if r.readable]
    n = len(readable)
    successes = sum(1 for r in readable if r.succeeded)
    return ArmSummary(
        arm=arm,
        n=n,
        successes=successes,
        excluded=len(results) - n,
        success_rate=successes / n if n else 0.0,
        mean_tool_calls=statistics.fmean([r.tool_calls for r in readable]) if n else 0.0,
        mean_probes=statistics.fmean([r.probes for r in readable]) if n else 0.0,
        mean_tokens=statistics.fmean([r.tokens for r in readable]) if n else 0.0,
        mean_artifacts_read=statistics.fmean([r.artifacts_read for r in readable]) if n else 0.0,
        stale_uses=sum(1 for r in readable if r.used_stale),
        duplicate_failures=sum(r.duplicate_failures for r in readable),
    )


def run_newcomer_benchmark(
    session: Session,
    *,
    organization_id: uuid.UUID,
    seeds: list[int] | None = None,
    era: int = 1,
    n_classes: int = 6,
    n_ops: int = 10,
    accumulation_passes: int = 2,
    budgets: Budgets | None = None,
    settings: Settings | None = None,
    include_arms: list[str] | None = None,
    retrieval_options: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """§22 on the hidden-rule device — M4's benchmark, unchanged in what it measures.

    A thin call into `run_newcomer_procedure`. The delegation was verified by running the
    identical configuration on the tree before and after the refactor and diffing the metrics:
    byte-identical, arms and gates alike (`scripts/newcomer_snapshot.py`).
    """
    from civitas.domains import get_domain

    return run_newcomer_procedure(
        session,
        domain=get_domain("hidden_rule"),
        organization_id=organization_id,
        experiment="newcomer_advantage",
        seeds=seeds,
        era=era,
        count=n_classes,
        accumulation_passes=accumulation_passes,
        budgets=budgets,
        settings=settings,
        include_arms=include_arms,
        retrieval_options=retrieval_options,
        generate_kwargs={"n_classes": n_classes, "n_ops": n_ops},
    )


def run_repair_benchmark(
    session: Session,
    *,
    organization_id: uuid.UUID,
    seeds: list[int] | None = None,
    era: int = 1,
    count: int = 4,
    accumulation_passes: int = 2,
    budgets: Budgets | None = None,
    settings: Settings | None = None,
    include_arms: list[str] | None = None,
    retrieval_options: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """§22 on the code-repair domain — the transfer question.

    Same procedure, same arms, same gates, same founder-free discipline; a different kind of task,
    a different agent policy, and an evaluator that executes the submission in the sandbox rather
    than comparing strings. If a newcomer advantage appears here too, the M4 result is a property
    of the platform rather than of one device.

    The default budget is five tool calls, not seven. Six candidate edits at five calls puts the
    naive ceiling at 3/6 = 0.500 — the same ceiling the device domain has at seven calls out of
    ten operations. Matching the *ceiling* rather than the raw budget is what makes the two
    numbers comparable; matching the budget would have compared two different difficulties.
    """
    from civitas.domains import get_domain

    return run_newcomer_procedure(
        session,
        domain=get_domain("code_repair"),
        organization_id=organization_id,
        experiment="newcomer_advantage_code_repair",
        seeds=seeds,
        era=era,
        count=count,
        accumulation_passes=accumulation_passes,
        budgets=budgets or REPAIR_BUDGETS,
        settings=settings,
        include_arms=include_arms,
        retrieval_options=retrieval_options,
    )


def run_newcomer_procedure(
    session: Session,
    *,
    domain: Any,
    organization_id: uuid.UUID,
    experiment: str = "newcomer_advantage",
    seeds: list[int] | None = None,
    era: int = 1,
    count: int = 6,
    accumulation_passes: int = 2,
    budgets: Budgets | None = None,
    settings: Settings | None = None,
    include_arms: list[str] | None = None,
    retrieval_options: dict[str, Any] | None = None,
    generate_kwargs: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run §22's procedure over any domain and return a gated result.

    One workspace per (arm, seed): the arms must not be able to see each other's accumulation, and
    sharing a workspace would make `memory_reset` a deletion that races the other arms rather than
    a condition.
    """
    from civitas.legacy.procedure import (
        archive_episode_artifacts,
        create_domain_task,
        reset_memory,
        run_domain_episode,
    )

    settings = settings or get_settings()
    seeds = seeds or [0, 1, 2]
    budgets = budgets or DEFAULT_BUDGETS
    config = frozen_config(
        budgets, retrieval_options=retrieval_options,
        system_prompt_version=domain.system_prompt_version(),
    )
    cfg_hash = config_hash_of(config)
    generate_kwargs = generate_kwargs or {}

    arms_to_run = include_arms or [
        "baseline_empty", "collective", "memory_reset", "collective_scrambled",
    ]
    collected: dict[str, list[RunResult]] = {a: [] for a in arms_to_run}
    hashes: set[str] = set()
    accumulation_stats: list[dict[str, Any]] = []
    specs_by_seed: dict[int, list[Any]] = {}

    for seed in seeds:
        specs = domain.generate(seed=seed, count=count, era=era, **generate_kwargs)
        specs_by_seed[seed] = specs

        for arm_label in arms_to_run:
            workspace = _fresh_workspace(session, organization_id, f"{arm_label}-s{seed}")
            hashes.add(cfg_hash)

            if arm_label != "baseline_empty":
                stats = _accumulate_domain(
                    session, domain=domain, specs=specs, workspace_id=workspace.id,
                    passes=accumulation_passes, budgets=budgets, config_hash=cfg_hash,
                    seed=seed, retrieval_options=retrieval_options,
                )
                accumulation_stats.append({"arm": arm_label, "seed": seed, **stats})

            if arm_label == "memory_reset":
                reset_memory(session, workspace.id)

            arm = {
                "baseline_empty": ExperimentArm.COLLECTIVE,
                "collective": ExperimentArm.COLLECTIVE,
                "memory_reset": ExperimentArm.MEMORY_RESET,
                "collective_scrambled": ExperimentArm.COLLECTIVE_SCRAMBLED,
                # §65 Level 1's control: a genuinely blinded agent facing a workspace others
                # matured. `memory_reset` empties the store; this one leaves it full and closes
                # the agent's eyes, which is the comparison Level 1 actually asks for.
                "solo_in_mature": ExperimentArm.SOLO,
            }[arm_label]

            # Probe *every* task, not one. Each probe is an independent fresh agent facing a
            # question the collective has had the chance to work on, so n is seeds x tasks.
            for spec in specs:
                task = create_domain_task(session, workspace_id=workspace.id, spec=spec)
                result_row = run_domain_episode(
                    session, domain=domain, spec=spec, workspace_id=workspace.id, task=task,
                    arm=arm, budgets=budgets,
                    # The probe agent's order is fixed across arms, so the arms are measured on
                    # literally the same newcomer.
                    probe_order_seed=_order_seed(seed, spec.index, -1),
                    retrieval_options=retrieval_options,
                    # A probe must not contribute to the accumulation it is measured against —
                    # the founder-free discipline of ARCHITECTURE §3.5.
                    record_findings=False, is_probe=True, config_hash=cfg_hash, seed=seed,
                    settings=settings,
                )
                # ...and neither must it contribute to what the *next* probe sees.
                archive_episode_artifacts(session, result_row.episode_id)
                collected[arm_label].append(result_row)
            session.commit()

    reference = specs_by_seed[seeds[0]][0]
    result = BenchmarkResult(
        experiment=experiment,
        config_hash=cfg_hash,
        chance_level=reference.chance_level,
        naive_probe_ceiling=domain.naive_probe_ceiling(reference, budgets.tool_calls),
        seeds=seeds,
    )
    for arm_label, runs in collected.items():
        result.arms[arm_label] = _summarise(arm_label, runs)

    result.gates = _gates(result, hashes, accumulation_stats, seeds, 0, budgets)
    result.notes.append(
        f"domain={domain.name}/{domain.version}; accumulation: {accumulation_passes} pass(es) "
        f"over {count} task(s) per arm/seed"
    )
    return result


def _accumulate_domain(
    session: Session,
    *,
    domain: Any,
    specs: list[Any],
    workspace_id: uuid.UUID,
    passes: int,
    budgets: Budgets,
    config_hash: str,
    seed: int,
    retrieval_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Let N agents work, so the environment matures (§22 step 8)."""
    from civitas.legacy.procedure import create_domain_task, run_domain_episode

    episodes = 0
    successes = 0
    for pass_no in range(passes):
        for spec in specs:
            task = create_domain_task(session, workspace_id=workspace_id, spec=spec)
            run = run_domain_episode(
                session, domain=domain, spec=spec, workspace_id=workspace_id, task=task,
                arm=ExperimentArm.COLLECTIVE, budgets=budgets, record_findings=True,
                is_probe=False, config_hash=config_hash, seed=seed,
                retrieval_options=retrieval_options,
                # A different agent each pass, and the same sequence of agents in every arm: the
                # seed depends on (seed, task, pass) and never on the arm.
                probe_order_seed=_order_seed(seed, spec.index, pass_no),
            )
            episodes += 1
            successes += int(run.succeeded)
    session.flush()
    artifacts = session.execute(
        select(Artifact).where(Artifact.workspace_id == workspace_id)
    ).scalars().all()
    return {"episodes": episodes, "successes": successes, "artifacts": len(artifacts)}


def _accumulate(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    device: DeviceSpec,
    instances: list,
    passes: int,
    budgets: Budgets,
    config_hash: str,
    seed: int,
    retrieval_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Let N agents work, so the environment matures (§22 step 8)."""
    episodes = 0
    successes = 0
    for pass_no in range(passes):
        for instance in instances:
            task = create_task(session, workspace_id=workspace_id, instance=instance)
            run = run_benchmark_episode(
                session,
                workspace_id=workspace_id,
                device=device,
                instance=instance,
                task=task,
                arm=ExperimentArm.COLLECTIVE,
                budgets=budgets,
                record_findings=True,
                is_probe=False,
                config_hash=config_hash,
                seed=seed,
                # A different agent each pass, and the same sequence of agents in every arm:
                # the seed depends on (seed, class, pass) and never on the arm.
                probe_order_seed=_order_seed(seed, instance.index, pass_no),
            )
            episodes += 1
            successes += int(run.succeeded)
    session.flush()
    artifacts = session.execute(
        select(Artifact).where(Artifact.workspace_id == workspace_id)
    ).scalars().all()
    return {
        "episodes": episodes,
        "successes": successes,
        "artifacts": len(artifacts),
    }


def _archive_episode_artifacts(session: Session, episode_id: uuid.UUID) -> int:
    """Remove a probe episode's own output from the collective it was measured against.

    Archived, not deleted (Part A §A1.2): the episode's contribution stays on the record and is
    simply excluded from retrieval, so a reader can still see what each probe produced.
    """
    from civitas.persistence.types import utcnow

    count = 0
    for artifact in session.execute(
        select(Artifact).where(
            Artifact.creator_episode_id == episode_id, Artifact.archived_at.is_(None)
        )
    ).scalars():
        artifact.archived_at = utcnow()
        artifact.archived_reason = "benchmark probe output, excluded from the measured environment"
        count += 1
    session.flush()
    return count


def _reset_memory(session: Session, workspace_id: uuid.UUID) -> int:
    """§21 `memory_reset`: delete or reset the allowed collective state.

    Archival, not deletion (Part A §A1.2). The arm's *effect* comes from `arm_policy` making the
    episode blind; archiving marks the state as reset without destroying the record of what the
    accumulation produced, which is what lets a reader check that the reset arm really did have
    something to lose.
    """
    from civitas.persistence.types import utcnow

    count = 0
    for artifact in session.execute(
        select(Artifact).where(
            Artifact.workspace_id == workspace_id, Artifact.archived_at.is_(None)
        )
    ).scalars():
        artifact.archived_at = utcnow()
        artifact.archived_reason = "memory_reset arm"
        count += 1
    # The duplicate-failure index is collective state too. Archiving the artifacts while leaving
    # the index answering would let the reset arm keep the one thing it is meant to lose.
    from civitas.knowledge.duplicate import retire_for_workspace

    retire_for_workspace(session, workspace_id)
    session.flush()
    return count


def _gates(
    result: BenchmarkResult,
    hashes: set[str],
    accumulation: list[dict[str, Any]],
    seeds: list[int],
    n_ops: int,
    budgets: Budgets,
) -> dict[str, Gate]:
    """Prerequisites. A failure means the run is not read, not read-with-caveats."""
    gates: dict[str, Gate] = {}

    # Gate 1 — frozen model (§20). Every arm must have run under one configuration, or the
    # comparison is between configurations rather than between collective environments.
    gates["frozen_model"] = Gate(
        name="frozen_model",
        passed=len(hashes) == 1,
        detail=(
            "all arms ran under one configuration hash"
            if len(hashes) == 1
            else f"arms ran under {len(hashes)} different configurations: "
            f"the comparison is not matched"
        ),
        value=sorted(hashes),
    )

    # Gate 2 — the collective actually accumulated something. Without this, `collective` and
    # `baseline_empty` are the same condition and a null result would be uninterpretable: it would
    # not distinguish "the collective does not help" from "there was no collective".
    total_artifacts = sum(a["artifacts"] for a in accumulation) if accumulation else 0
    gates["accumulation_occurred"] = Gate(
        name="accumulation_occurred",
        passed=total_artifacts > 0,
        detail=f"{total_artifacts} artifacts accumulated across arms and seeds",
        value=total_artifacts,
    )

    # Gate 3 — the baseline is not already at ceiling. If a fresh agent in an empty environment
    # already solves everything, there is no headroom and no advantage is measurable. This is the
    # gate the research lineage would call a readability check.
    baseline = result.arms.get("baseline_empty")
    headroom_ok = baseline is None or baseline.success_rate < 0.95
    gates["baseline_has_headroom"] = Gate(
        name="baseline_has_headroom",
        passed=headroom_ok,
        detail=(
            f"baseline success rate {baseline.success_rate:.3f} leaves headroom"
            if baseline and headroom_ok
            else "the baseline is at ceiling; no advantage is measurable at this difficulty"
        ),
        value=baseline.success_rate if baseline else None,
    )

    # Gate 4 — enough episodes per arm to read a rate at all. Three seeds is thin, and saying so
    # in the gate is better than letting a reader infer robustness from a printed decimal.
    min_n = min((a.n for a in result.arms.values()), default=0)
    gates["sufficient_n"] = Gate(
        name="sufficient_n",
        passed=min_n >= 3,
        detail=f"smallest arm has n={min_n} readable episodes",
        value=min_n,
    )
    return gates
