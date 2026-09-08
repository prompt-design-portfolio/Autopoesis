"""G3 — the store outlives the run.

    A record left by one population raises the competence of a population that never met it, and
    the gain compounds.

This is the goal; G0-G2 were scaffolding for it. The spec, its two rulings and the list of ways
this claim could be wrong are in `docs/G3_SPEC.md`.

The mechanism in one paragraph. Population A is an ordinary `collective` run. When it ends,
`final_store` is what it left -- marks and pi as they stood mid-era, live, the record a living
population was actually using, not the era-boundary capture the frozen assay replays against.
Population B is a *new* run: fresh founders, no genomes and no `H` crossing, born into A's record
through `init_store` and started in A's era through `init_mapping`. Nothing of A survives except
what was externalised.

**Everything except the store is matched.** All four of B's arms take the same seed, the same
fresh founders and the same starting era. `sym_gain_lock` consumes no RNG; `init_mapping`
overwrites a draw rather than skipping one. So a difference between arms is a difference in the
store or it is nothing, and that is the property the whole claim rests on.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

import analysis_v3_13 as A
from civitas_g.store.record import Record, ScrambleMode
from civitas_g.world.engine import RunResult, build_run_spec
from civitas_g.world.engine import run as run_engine
from sim_v3_13 import N_PREPS

#: B's arms. The store is the only thing that varies.
B_ARMS: tuple[str, ...] = (
    "fresh store",
    "inherited store",
    "inherited scrambled",
    "inherited gain-zero",
)


@dataclass(frozen=True)
class Succession:
    """One A -> B lineage: which A, which alignment, which seed.

    `b_steps` and `claim_window` pull in opposite directions and both are needed.

    * The **claim** must be read while A's record is still live. `mark_decay` 0.004 gives a
      173-step half-life, so a window of one era (700 steps) already costs the record 94% of its
      magnitude; anything longer dilutes B's first-ever preparations with ones made after the
      inheritance has effectively gone.
    * **Gate R** needs at least two pi-epochs to have anything to permute across, and B rotates pi
      on its own clock -- so it needs a run several eras long.

    So B runs long (`b_steps`) and the claim lines are read on an early window (`claim_window`),
    with Gate R read over the whole run. Reading both on the same window would have meant choosing
    which of the two to make meaningless.
    """

    seed: int
    a_phase_steps: int
    b_steps: int
    aligned: bool
    scramble_seed: int = 0
    #: The window the claim lines are read on. None means the whole run.
    claim_window: int | None = None

    @property
    def alignment(self) -> str:
        return "aligned" if self.aligned else "misaligned"


@dataclass
class BArmResult:
    """One arm of B, with the claim-line statistics computed on first-ever preparations only."""

    arm: str
    seed: int
    aligned: bool
    #: stale-mark ratio: P(followed a mark endorsing a preparation wrong for this type NOW)
    stale: float
    stale_n: int
    #: A2.2's matched null, from this arm's own hit rate. Never 1/K.
    null: float
    ratio: float
    #: preparations-to-first-correct, founder-free
    nfc_mean: float
    nfc_censored: float
    nfc_n: int
    prep_hit: float
    pop: float
    #: what the arm was actually handed, so a reader needs nothing else
    store_sha256: str | None
    store_density: float | None
    sym_gain: float
    #: The window the claim lines were read on, and the fraction of A's marks still standing at
    #: its end. A claim read over a window where that fraction is ~0 is a claim about decay.
    claim_window: int | None = None
    marks_surviving_at_window_end: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class G3Result:
    succession: Succession
    a_store_sha256: str
    a_mapping: tuple[int, ...]
    a_pi: tuple[int, ...]
    b_mapping: tuple[int, ...]
    arms: list[BArmResult] = field(default_factory=list)
    gate_r: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def arm(self, name: str) -> BArmResult:
        for a in self.arms:
            if a.arm == name:
                return a
        raise KeyError(f"no arm {name!r}")

    # ---------------------------------------------------------------- the claim lines

    def claim_lines(self) -> dict[str, dict[str, float]]:
        """Each arm against its right baseline, on B's first-ever preparations only.

        **The two lines have different baselines, and that is not a convenience.** B§5.2 says both
        go "against `fresh store`", but `fresh store` has no marks, so it has no stale-mark events
        at all -- its ratio is not a small number, it is undefined. Subtracting it would produce
        `nan` and a reader could mistake that for a failed comparison rather than an absent one.

        So:

        * **preparations-to-first-correct** and **hit** are against `fresh store`, as specified.
        * the **stale-mark ratio** is against `inherited gain-zero` -- the arm handed the SAME
          store that cannot read it. Its own null (ratio = 1) is not enough, and the pre-check
          showed why: gain-zero came out at 1.48 with `sym_gain` pinned to exactly zero, so a
          ratio above 1 is reachable without reading anything. The population shares innate action
          biases, marks accumulate where the population prepares, and an agent that revisits such
          a cell "agrees" with a mark it never read. `(1 - hit) / (K - 1)` does not capture that.
          The unconfounded quantity is the difference between an arm that can read the store and
          one that cannot.

        The two lines also point in opposite directions on purpose: a store that helps should RAISE
        the stale-mark ratio and LOWER preparations-to-first-correct. An arm that moves one and not
        the other is reported as that, not averaged.
        """
        fresh = self.arm("fresh store")
        try:
            unreading = self.arm("inherited gain-zero")
        except KeyError:
            unreading = None
        out: dict[str, dict[str, float]] = {}
        for a in self.arms:
            if a.arm == fresh.arm:
                continue
            lines = {
                "nfc_vs_fresh": a.nfc_mean - fresh.nfc_mean,
                "prep_hit_vs_fresh": a.prep_hit - fresh.prep_hit,
            }
            if unreading is not None and a.arm != unreading.arm:
                lines["stale_ratio_vs_unreading"] = a.ratio - unreading.ratio
            out[a.arm] = lines
        return out

    def table(self) -> str:
        rows = [f"  {'arm':<22}{'stale':>8}{'null':>8}{'ratio':>8}{'n':>8}"
                f"{'nfc':>8}{'nfc n':>8}{'hit':>8}{'gain':>8}{'density':>9}"]
        for a in self.arms:
            density = "--" if a.store_density is None else f"{a.store_density:.4f}"
            rows.append(f"  {a.arm:<22}{a.stale:>8.3f}{a.null:>8.3f}{a.ratio:>8.2f}"
                        f"{a.stale_n:>8d}{a.nfc_mean:>8.3f}{a.nfc_n:>8d}{a.prep_hit:>8.3f}"
                        f"{a.sym_gain:>8.4f}{density:>9}")
        return "\n".join(rows)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["arms"] = [a.as_dict() for a in self.arms]
        return d


# ---------------------------------------------------------------------------------------------
# running a succession
# ---------------------------------------------------------------------------------------------

def run_population_a(seed: int, phase_steps: int, *, verbose: bool = False) -> RunResult:
    """An ordinary `collective` run, with store capture on. Nothing about it is special."""
    spec = build_run_spec("collective", seed=seed, phase_steps=phase_steps,
                          overrides={"store_snaps": True})
    return run_engine(spec, verbose=verbose)


def store_for_arm(record: Record, arm: str, *,
                  scramble_seed: int = 0) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """What B's arm is handed, and the extra config it needs.

    `inherited gain-zero` gets the SAME store as `inherited store` and loses the ability to read
    it. That is A2.3's presence-versus-content separation: the store still changes the world by
    existing -- channels carry values, cells hold state, decay runs -- and only the reading is
    disabled. Removing the store instead would change the world.
    """
    if arm == "fresh store":
        # A HIDDEN store, not no store, and the difference is a matched nuisance parameter.
        # With init_store=None the world draws its own pi, so the baseline would differ from the
        # treatment in the LABEL PERMUTATION as well as in the marks -- and B's own writes would
        # then land in different channels from the inherited arms'. pi carries no information
        # about the world (it is redrawn every era; that is what makes meaning non-inheritable),
        # so holding it constant costs nothing and removes a difference that is not the one being
        # measured. The marks are zero either way: World.__init__ zeroes them.
        return {"marks": np.zeros_like(record.marks), "pi": record.pi}, {}
    if arm == "inherited store":
        return {"marks": record.marks, "pi": record.pi}, {}
    if arm == "inherited scrambled":
        scrambled = record.scrambled(np.random.default_rng(scramble_seed), ScrambleMode.PER_CELL)
        return {"marks": scrambled.marks, "pi": scrambled.pi}, {}
    if arm == "inherited gain-zero":
        return {"marks": record.marks, "pi": record.pi}, {"sym_gain_lock": True}
    raise ValueError(f"unknown B arm {arm!r}")


def _b_statistics(arm: str, seed: int, aligned: bool, raw: dict[str, Any],
                  store: dict[str, Any] | None, claim_window: int | None = None) -> BArmResult:
    """The claim-line statistics for one arm of B.

    Read on an EARLY window, not the whole run: the newborn counters accumulate at the preparation
    event for agents on their first attempt, and those events go on happening long after A's record
    has decayed away. Pooling them would average a live inheritance with an absent one and call the
    result a weaker effect.
    """
    log = raw["log"] if claim_window is None else A.window(raw, 0, claim_window)
    if not log:
        log = raw["log"]
    _endorse_ok, _n_ok, stale, stale_n = A.follow_split_newborn(log)
    hit = A.prep_hit(log, True)
    null = (1.0 - hit) / (N_PREPS - 1) if np.isfinite(hit) else np.nan
    nfc_mean, nfc_censored, nfc_n = A.nfc(log, True)
    density = None
    if store is not None:
        density = float((np.abs(np.asarray(store["marks"])) > 1e-3).mean())
    decay = float(raw["cfg"].get("mark_decay", 0.0))
    window = claim_window if claim_window is not None else int(raw["n_steps"])
    return BArmResult(
        claim_window=window,
        marks_surviving_at_window_end=float((1.0 - decay) ** window),
        arm=arm, seed=seed, aligned=aligned,
        stale=float(stale), stale_n=int(stale_n),
        null=float(null), ratio=float(stale / null) if null else float("nan"),
        nfc_mean=float(nfc_mean), nfc_censored=float(nfc_censored), nfc_n=int(nfc_n),
        prep_hit=float(hit), pop=float(A.half(log, "pop")),
        store_sha256=None, store_density=density,
        sym_gain=float(A.half(log, "sym_gain")),
    )


def run_succession(succession: Succession, *, verbose: bool = False,
                   a_result: RunResult | None = None) -> G3Result:
    """A lives and dies; B is born into what it left.

    `a_result` lets a caller reuse one A across alignments -- the same record, two clocks -- which
    is the comparison G3-D2 asks for and which would be meaningless if each alignment got its own A.
    """
    from civitas_g.store.record import RecordProvenance

    a = a_result or run_population_a(succession.seed, succession.a_phase_steps, verbose=verbose)
    final = a.raw.get("final_store")
    if final is None:
        raise ValueError(
            "population A produced no final store. The run needs cfg.store_snaps, and the engine "
            "must be a version that returns final_store (G2-store-fix or later).")

    record = Record(
        marks=np.asarray(final["marks"], dtype="<f8"),
        pi=tuple(int(x) for x in final["pi"]),
        provenance=RecordProvenance(
            run_seed=succession.seed, arm="collective", t=int(final["t"]), era_index=0,
            engine_sha256=a.engine_sha256, cfg_digest="",
            mapping=tuple(int(x) for x in final["mapping"]),
            note="population A's final record -- what it left behind"),
    )
    a_mapping = tuple(int(x) for x in final["mapping"])

    result = G3Result(
        succession=succession, a_store_sha256=record.sha256(),
        a_mapping=a_mapping, a_pi=record.pi, b_mapping=(),
    )
    if record.density() == 0.0:
        result.notes.append(
            "A's final record is EMPTY. Every inherited arm is then identical to `fresh store` by "
            "construction and no claim can be read from this succession.")

    b_phases = [dict(n_steps=succession.b_steps, chain=True)]
    for arm in B_ARMS:
        store, extra = store_for_arm(record, arm, scramble_seed=succession.scramble_seed)
        spec = build_run_spec("collective", seed=succession.seed,
                              phase_steps=succession.b_steps,
                              overrides={"store_snaps": True, **extra})
        raw = run_engine(spec, verbose=verbose, phases=copy.deepcopy(b_phases),
                         init_store=store,
                         init_mapping=a_mapping if succession.aligned else None).raw
        if not result.b_mapping:
            result.b_mapping = tuple(int(x) for x in raw["log"][0]["mapping"])
        row = _b_statistics(arm, succession.seed, succession.aligned, raw, store,
                            claim_window=succession.claim_window)
        row.store_sha256 = None if store is None else record.sha256()
        result.arms.append(row)
        if arm == "inherited store":
            # Gate R on the INHERITED marks under B's own pi-epochs (B§5.2). B inherits A's pi and
            # then redraws it on B's clock, so the marks span epochs that are not A's. If the
            # binding survived that rotation, selection could have reached it -- and a gain would
            # not be evidence of transmission through a record whose meaning was not inherited.
            gp = A.gate_r_permutation({"log": raw["log"], "cfg": raw["cfg"],
                                       "n_steps": raw["n_steps"],
                                       "chain_start": raw.get("chain_start", 0),
                                       "phase_bounds": raw["phase_bounds"]})
            result.gate_r = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                             for k, v in gp.items()}
            result.gate_r["verdict"] = _gate_r_verdict(gp)
    return result


def _gate_r_verdict(gp: dict[str, Any]) -> str:
    """Gate R's verdict, refusing to call a degenerate null a pass.

    The permutation null permutes each pi-epoch's label axis INDEPENDENTLY and pools. With one
    epoch there is nothing to permute across: every permutation returns the same pooled value, the
    null equals the observation, `sd` is exactly zero and `z` comes out 0.00. Reporting that as
    PASS would be reporting an absent measurement as a passed gate, which is the failure A5 names
    and the one this project keeps having to catch.

    A B that runs a single era therefore CANNOT be gated on Gate R -- and a B that runs long enough
    to rotate pi has lost the record it inherited to decay (G3-D1's arithmetic in the other
    direction). That tension is structural and is reported, not resolved by a verdict.
    """
    epochs = int(gp.get("eras") or 0)
    if epochs < 2:
        return (f"NOT AVAILABLE: {epochs} pi-epoch(s). The permutation null needs at least two to "
                f"have anything to permute across; with one it returns the observation itself.")
    if float(gp.get("sd") or 0.0) == 0.0:
        return ("NOT AVAILABLE: the null has zero spread, so z is undefined rather than small.")
    if not np.isfinite(gp.get("z", np.nan)):
        return "NOT AVAILABLE: z is not finite."
    return "PASS" if gp["z"] <= 2.0 else "GATE R FIRES"


def b_founders_carry_no_h(seed: int = 0, steps: int = 200) -> tuple[bool, str]:
    """B§6's `B-founders-carry-no-H`, run before any B number is read.

    Three things, and the third is the one that would actually bite:

      1. `snapshot()` saves genomes only -- `H1` and `H2` are not among the fields it writes;
      2. the G-lineage engine wrapper REFUSES `init_genomes`, so a B cannot be seeded from A's
         population even by accident;
      3. a fresh population's `H` starts at zero, so nothing learned is present at step 0.
    """
    import sim_v3_13 as S
    from civitas_g.world.engine import EngineRefusal

    saved = set(S.GENOME)
    leaked = saved & {"H1", "H2", "e1", "e2"}
    if leaked:
        return False, f"snapshot() saves learned state: {sorted(leaked)}"

    spec = build_run_spec("collective", seed=seed, phase_steps=steps)
    try:
        run_engine(spec, init_genomes=[{}])
    except EngineRefusal:
        pass
    else:
        return False, "the engine wrapper accepted init_genomes; a B could be seeded from A"

    cfg = S.Config(seed=seed, chain=True, record="real")
    agent = S.Agent(cfg, np.random.default_rng(seed), 0, 5, 5)
    if float(np.abs(agent.H1).sum() + np.abs(agent.H2).sum()) != 0.0:
        return False, "a fresh agent's H is not zero"

    return True, (f"snapshot() saves {len(saved)} genome fields and no learned state; the engine "
                  f"wrapper refuses init_genomes; a fresh agent's H is exactly zero")
