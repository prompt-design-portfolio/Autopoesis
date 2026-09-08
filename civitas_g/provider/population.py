"""The population provider (G1).

A3's G1 row: "population provider; the summaries committed in the repository at G0 reproduced to
three decimals on their own seeds, both backends; the reference summaries and the code they were
produced by recorded by hash". This module is the first clause. `civitas_g.reading` is the second.

What a "provider" is here is narrower than §19's, and the narrowing is the whole point of A1.1.
A §19 provider supplies cognition to an episode. This one supplies a *population* -- it starts the
engine, and it writes down what the engine produced. It contains no policy, no belief state, no
LLM, and no per-step surface a learner could lean on. The only thing in the system that decides
anything is the grown network inside `sim_v3_13.py`.

**Resumability (§38).** A campaign skips any `(arm, seed)` already stored, so a dropped session
costs one run rather than the lot. This mirrors `analysis_v3_13.run_experiment`, which resumes from
a pickle, and for the same reason: these runs are minutes to hours long on a Colab CPU.

**Densities (D10).** The G0 audit found that standing food cover is recorded nowhere in the
repository, and asked the provider to fix it. It cannot, at G1: food cover is not in the engine's
log, and A1.8 says the engine runs byte-identical. So `densities()` records what the engine does
give -- mark density and mean |mark| per era -- and reports food cover and mark age as unavailable
*with the reason*, rather than deriving a plausible number. Adding them is a versioned engine
change with its own gate, which is exactly how the directive says an instrument change arrives.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas_g.manifest import build_manifest
from civitas_g.persistence.encoding import decode, encode
from civitas_g.persistence.models import Campaign, EraRow, EraSnapshot, Run
from civitas_g.world.arms import get_arm
from civitas_g.world.engine import RunResult, RunSpec, build_run_spec
from civitas_g.world.engine import run as run_engine


@dataclass
class CampaignPlan:
    """What a campaign intends to run, before any of it has run."""

    kind: str
    arms: list[str]
    seeds: list[int]
    phase_steps: int
    label: str = ""
    overrides: dict[str, Any] = None  # type: ignore[assignment]
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.overrides = dict(self.overrides or {})
        self.notes = list(self.notes or [])
        # normalise to the manifest vocabulary once, here, so nothing downstream has to guess
        self.arms = [get_arm(a).name for a in self.arms]

    def specs(self) -> list[RunSpec]:
        return [build_run_spec(arm, seed, self.phase_steps, overrides=self.overrides)
                for seed in self.seeds for arm in self.arms]


def _blob(obj: Any) -> tuple[bytes, str]:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return data, hashlib.sha256(data).hexdigest()


def _phase_of(row: dict[str, Any], phase_bounds: Sequence[Sequence[int]]) -> int:
    t = int(row["t"])
    for i, (lo, hi) in enumerate(phase_bounds):
        if lo < t <= hi:
            return i
    return len(phase_bounds) - 1 if phase_bounds else 0


def open_campaign(session: Session, plan: CampaignPlan) -> Campaign:
    """Create the campaign row, manifest and all, before any run starts.

    Written first on purpose. A campaign that crashes on its first run still leaves behind what it
    was going to be, which is the difference between an interrupted experiment and a mystery.
    """
    from civitas_g.manifest import engine_identity

    manifest = build_manifest(kind=plan.kind, arms=plan.arms, seeds=plan.seeds,
                              phase_steps=plan.phase_steps, notes=plan.notes)
    campaign = Campaign(
        kind=plan.kind, label=plan.label, manifest=encode(manifest.as_dict()),
        phase_steps=plan.phase_steps, engine_sha256=engine_identity()["sim_v3_13.py"],
    )
    session.add(campaign)
    session.flush()
    return campaign


def existing_runs(session: Session, campaign_id: Any) -> set[tuple[str, int]]:
    """The `(arm, seed)` pairs already stored for this campaign, for §38's resume."""
    rows = session.execute(
        select(Run.arm, Run.seed).where(Run.campaign_id == campaign_id,
                                        Run.status == "ok")).all()
    return {(arm, int(seed)) for arm, seed in rows}


def store_run(session: Session, campaign: Campaign, result: RunResult) -> Run:
    """Persist one run: the run row, every era row, and every era-boundary snapshot."""
    raw = result.raw
    bounds = [list(map(int, b)) for b in raw.get("phase_bounds", [])]
    run = Run(
        campaign_id=campaign.id, arm=result.arm, seed=result.seed,
        phase_steps=result.phase_steps, status="ok",
        cfg=encode(raw["cfg"]),
        final_mapping=encode(list(raw["final_mapping"])),
        chain_start=int(raw.get("chain_start", 0)),
        n_steps=int(raw.get("n_steps", 0)),
        phase_bounds=encode(bounds),
        flips=encode(list(raw.get("flips", []))),
        recipe_changes=encode(list(raw.get("recipe_changes", []))),
        final_snapshot=encode(raw.get("final")),
        wall_seconds=float(result.wall_seconds),
        engine_sha256=result.engine_sha256,
    )
    session.add(run)
    session.flush()

    session.add_all([
        EraRow(run_id=run.id, ordinal=i, t=int(row["t"]),
               phase=_phase_of(row, bounds), chain=bool(row.get("chain", False)),
               pop=int(row.get("pop", 0)), payload=encode(row))
        for i, row in enumerate(raw["log"])
    ])

    for i, snap in enumerate(raw.get("era_snaps") or []):
        data, digest = _blob(snap)
        mapping = snap.get("mapping") if isinstance(snap, dict) else None
        pi = snap.get("pi") if isinstance(snap, dict) else None
        genomes = snap.get("genomes") if isinstance(snap, dict) else None
        session.add(EraSnapshot(
            run_id=run.id, ordinal=i,
            t=int(snap["t"]) if isinstance(snap, dict) and "t" in snap else None,
            mapping=encode(list(mapping)) if mapping is not None else None,
            pi=encode(list(pi)) if pi is not None else None,
            n_agents=len(genomes) if genomes is not None else 0,
            blob=data, blob_sha256=digest,
        ))
    session.flush()
    return run


def store_failure(session: Session, campaign: Campaign, spec: RunSpec, exc: BaseException) -> Run:
    """A1.4: a run that died is evidence. It is recorded, not dropped."""
    run = Run(
        campaign_id=campaign.id, arm=spec.arm, seed=spec.seed, phase_steps=spec.phase_steps,
        status="failed", failure=f"{type(exc).__name__}: {exc}",
        cfg=encode(spec.kwargs), final_mapping=[], chain_start=0, n_steps=0,
        phase_bounds=[], flips=[], recipe_changes=[], final_snapshot=None,
        wall_seconds=0.0, engine_sha256="",
    )
    session.add(run)
    session.flush()
    return run


def run_campaign(session_factory: Callable[[], Any], plan: CampaignPlan, *,
                 campaign_id: Any = None, verbose: bool = False,
                 on_run: Callable[[Run], None] | None = None) -> Any:
    """Run every `(arm, seed)` not already stored, committing after each one.

    Committing per run rather than per campaign is §38's resumability: a session that dies mid
    campaign has lost the run in flight and nothing else.

    `session_factory` is a callable rather than a session, because a campaign outlives a single
    transaction and holding one open for hours would hold SQLite's write lock for hours.
    """
    with session_factory() as session:
        if campaign_id is None:
            campaign = open_campaign(session, plan)
            session.commit()
            campaign_id = campaign.id
        else:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                raise LookupError(f"no campaign {campaign_id!r}")

    for spec in plan.specs():
        with session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if (spec.arm, spec.seed) in existing_runs(session, campaign_id):
                continue
        try:
            result = run_engine(spec, verbose=verbose)
        except Exception as exc:                      # noqa: BLE001 - recorded, then re-raised
            with session_factory() as session:
                campaign = session.get(Campaign, campaign_id)
                store_failure(session, campaign, spec, exc)
                session.commit()
            raise
        with session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            run = store_run(session, campaign, result)
            session.commit()
            if on_run is not None:
                on_run(run)
    return campaign_id


# ---------------------------------------------------------------------------------------------
# reading back
# ---------------------------------------------------------------------------------------------

def load_run(session: Session, run: Run | Any) -> dict[str, Any]:
    """Rebuild the engine's result dict from the stored rows.

    This is what makes D12 mean anything. `analysis_v3_13`'s reading functions take exactly this
    shape -- a `log` of dicts plus `phase_bounds`, `n_steps`, `chain_start` and `cfg` -- so if the
    rows round-trip faithfully then the *entire* reading is recomputable from the database, and
    "reproduced on both backends" is a claim about persistence fidelity rather than about numpy
    having run twice.
    """
    run_obj = run if isinstance(run, Run) else session.get(Run, run)
    if run_obj is None:
        raise LookupError(f"no run {run!r}")
    rows = session.execute(
        select(EraRow).where(EraRow.run_id == run_obj.id).order_by(EraRow.ordinal)).scalars().all()
    return {
        "log": [decode(r.payload) for r in rows],
        "cfg": decode(run_obj.cfg),
        "phase_bounds": [tuple(b) for b in decode(run_obj.phase_bounds)],
        "n_steps": int(run_obj.n_steps),
        "chain_start": int(run_obj.chain_start),
        "final_mapping": tuple(decode(run_obj.final_mapping)),
        "flips": decode(run_obj.flips),
        "recipe_changes": decode(run_obj.recipe_changes),
        "final": decode(run_obj.final_snapshot),
        "arm": run_obj.arm,
        "seed": int(run_obj.seed),
    }


def load_results(session: Session, campaign_id: Any,
                 arms: Iterable[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """`{research name: [run dict, ...]}`, in seed order -- the shape `analysis_v3_13` reads.

    Keyed by the RESEARCH name, because that is what the reference summaries print and what
    `analysis_v3_13`'s readers key on. The manifest vocabulary is what the database stores; the
    translation happens here, once, and nowhere else.
    """
    from civitas_g.world.arms import BY_NAME

    stmt = select(Run).where(Run.campaign_id == campaign_id, Run.status == "ok")
    if arms is not None:
        wanted = {get_arm(a).name for a in arms}
        stmt = stmt.where(Run.arm.in_(sorted(wanted)))
    runs = session.execute(stmt.order_by(Run.arm, Run.seed)).scalars().all()

    out: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        out.setdefault(BY_NAME[r.arm].research_name, []).append(load_run(session, r))
    for name in out:
        out[name].sort(key=lambda d: d["seed"])
    return out


# ---------------------------------------------------------------------------------------------
# densities (D10)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Density:
    """One era's standing densities, and an honest account of what is not measurable here."""

    t: int
    mark_density: float | None
    mark_mean_abs: float | None
    food_cover: None
    food_cover_reason: str
    mark_mean_age: None
    mark_mean_age_reason: str


_NO_FOOD_COVER = (
    "not recorded by sim_v3_13's log, and A1.8 requires the engine to run byte-identical, so the "
    "provider cannot add it. Food cover is also not derivable from the parameters: 8 patches x "
    "Poisson(6.0) is 48 spawn attempts a step against a rot of 0.005, which would saturate the "
    "patches many times over, so what stands is set by how fast the population eats it and "
    "differs per arm and per phase. Adding it is a versioned engine change with its own gate."
)

_NO_MARK_AGE = (
    "mark_stats' docstring says 'density and age of the store', but it computes no age -- marks "
    "carry a decayed magnitude and no timestamp, so age is not recoverable from what is stored. "
    "Reported as unavailable rather than inferred from magnitude, which would assume the decay "
    "constant the measurement exists to check."
)


def densities(run_dict: dict[str, Any]) -> list[Density]:
    """Per-era densities from a stored run. Absent measurements are `None` with a reason."""
    def maybe(row: dict[str, Any], key: str) -> float | None:
        import math
        v = row.get(key)
        if v is None:
            return None
        f = float(v)
        return None if math.isnan(f) else f

    return [
        Density(t=int(row["t"]),
                mark_density=maybe(row, "mark_density"),
                mark_mean_abs=maybe(row, "mark_mean_abs"),
                food_cover=None, food_cover_reason=_NO_FOOD_COVER,
                mark_mean_age=None, mark_mean_age_reason=_NO_MARK_AGE)
        for row in run_dict["log"]
    ]


def run_campaign_on_backends(session_factories: dict[str, Callable[[], Any]],
                             plan: CampaignPlan, *, verbose: bool = False
                             ) -> dict[str, Any]:
    """Run each `(arm, seed)` ONCE and store the result on every backend.

    This is what makes D12's "both backends" a real check. Running the simulation twice, once per
    backend, would compare numpy against numpy and say nothing about persistence. Running it once
    and writing the same result to both means any later disagreement between the two readings is a
    round-trip defect, which is the only thing a backend can be wrong about here.

    Returns `{backend: campaign id}`.
    """
    campaign_ids: dict[str, Any] = {}
    for backend, factory in session_factories.items():
        with factory() as session:
            campaign = open_campaign(session, plan)
            session.commit()
            campaign_ids[backend] = campaign.id

    for spec in plan.specs():
        pending = {}
        for backend, factory in session_factories.items():
            with factory() as session:
                if (spec.arm, spec.seed) not in existing_runs(session, campaign_ids[backend]):
                    pending[backend] = factory
        if not pending:
            continue
        try:
            result = run_engine(spec, verbose=verbose)
        except Exception as exc:                      # noqa: BLE001 - recorded on every backend
            for backend, factory in pending.items():
                with factory() as session:
                    campaign = session.get(Campaign, campaign_ids[backend])
                    store_failure(session, campaign, spec, exc)
                    session.commit()
            raise
        for backend, factory in pending.items():
            with factory() as session:
                campaign = session.get(Campaign, campaign_ids[backend])
                store_run(session, campaign, result)
                session.commit()
    return campaign_ids
