"""The world of record (B§3, and §3.1-§3.8 of the G1 spec in `docs/G0_G1.md`).

The single most dangerous thing this package can do is build the wrong world, and it is easy to
do by accident. `sim_v3_13.Config`'s defaults are v3.10 leftovers: `prep_value` 1.5 against
`prep_fail` 0.5 puts the expected value of a chance preparation at **-0.100**, and `prep_every`
2000 makes the era nearly three times as long as the one every v3.13 number was measured on. The
world is `analysis_v3_13.WORLD` laid over that constructor, not the constructor alone.

So this module does three things and no more:

1. **Names the world of record** by importing `analysis_v3_13.WORLD` rather than restating it.
   A copy would drift, and a drifted copy would reproduce nothing while looking correct.
2. **Pins what it expects to find there**, in `EXPECTED_WORLD`, and checks it. Importing the
   engine's parameters without asserting them is trust; asserting them is a measurement. If the
   research lineage moves a parameter, this fails loudly at construction instead of quietly
   producing numbers against a different world.
3. **States the invariants that make the world an experiment** rather than a game --
   `prep_value == (K-1) * prep_fail`, and the modulator being a table rather than an energy
   delta -- as checks that run, not as prose.

Nothing here computes anything about a run. The engine does that, byte-identical (A1.8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import analysis_v3_13 as _research
import sim_v3_13 as _sim

# ---------------------------------------------------------------------------------------------
# the world, imported rather than restated
# ---------------------------------------------------------------------------------------------

#: The world of record. `analysis_v3_13.WORLD` is what every committed v3.13 number was produced
#: under; `sim_v3_13.Config()` is a constructor with stale defaults (F7 in `docs/G0_G1.md`).
WORLD: dict[str, Any] = dict(_research.WORLD)

#: The two phases, run back to back on ONE population. Phase 1 is exactly v3.1's two-type flip
#: world -- which is what keeps the phase-1 rows anchored to v3.1's published range -- and phase 2
#: adds the third food type, the preparations and the record.
STAGED: list[dict[str, Any]] = [dict(p) for p in _research.STAGED]

#: The engine's own vocabulary. Imported, never redefined: an independent copy of `N_PREPS` that
#: disagreed with the engine's would be undetectable until it produced a wrong null.
N_TYPES: int = _sim.N_TYPES          # 3 food types, A / B / C
N_PREPS: int = _sim.N_PREPS          # K = 5 preparations
N_ACTIONS: int = _sim.N_ACTIONS      # 10: 4 moves, eat, 5 preparations
N_IN: int = _sim.N_IN                # 31 observation inputs
N_MAPPINGS: int = _sim.N_MAPPINGS    # P(K, T) = P(5, 3) = 60
EAT: int = _sim.EAT                  # 4
PREP0: int = _sim.PREP0              # 5; preparation k is action PREP0 + k
READ: int = _sim.READ                # 26; the first of the K read channels
FOOD_CH: int = _sim.FOOD_CH
OCC_CH: int = _sim.OCC_CH
APPETITE_CH: int = _sim.APPETITE_CH
N_CH: int = _sim.N_CH                # 5 spatial channels
HERE: int = _sim.HERE                # 20; the first 'here' value
ENERGY: int = _sim.ENERGY            # 25

# ---------------------------------------------------------------------------------------------
# what we expect to find, so that a change in the research lineage is loud
# ---------------------------------------------------------------------------------------------

#: Every parameter of the world of record, pinned. A3 requires the reference files and the code
#: they were produced by to be recorded by hash; this is the same discipline one level down --
#: the *values* a reader needs, recorded beside the code that reads them (the `sr_w` lesson).
EXPECTED_WORLD: dict[str, Any] = {
    # phase 1 -- v3.1, unchanged
    "flip_every": 300,
    "eta_init": 0.2,
    "hidden": 24,
    # standing density: raised to 6.0 for D5's readability criterion once food is split three
    # ways. The ruling was to raise spawn density, NEVER to lengthen the era -- a longer era is
    # more time for the survival sorting this world exists to outrun.
    "spawn_per_patch": 6.0,
    "food_value": 0.7,
    "poison_value": 0.5,
    "repro_threshold": 3.0,
    "repro_cost": 1.5,
    "max_energy": 5.0,
    "max_pop": 1000,
    "init_pop": 300,
    # the preparation economy. prep_value = (K-1) * prep_fail = 4 * 0.25 = 1.0, so the expected
    # value of a CHANCE preparation is exactly zero and any positive return is knowledge.
    "prep_value": 1.0,
    "prep_fail": 0.25,
    # 700, after 350 was tried and reverted: a shorter era shortened the learner's payoff window
    # without touching the sorting mechanism at all.
    "prep_every": 700,
    "scaffold_food": False,
    "scaffold_chain": False,
    "goal_channel": False,
}

#: The engine's constants, pinned alongside. `N_MAPPINGS` is the one that matters most: the whole
#: point of v3.12's move from K=3 to K=5 was to enlarge the mapping space so that survival sorting
#: over standing variation stops being most of the standing number.
EXPECTED_ENGINE: dict[str, int] = {
    "N_TYPES": 3, "N_PREPS": 5, "N_ACTIONS": 10, "N_IN": 31, "N_MAPPINGS": 60,
    "EAT": 4, "PREP0": 5, "READ": 26, "HERE": 20, "ENERGY": 25, "N_CH": 5,
}

#: Phase steps of the reference pre-check, recovered by inference rather than by record -- see F5
#: and D4 in `docs/G0_G1.md`. `precheck_v3_13.txt` states neither its seeds nor its configuration;
#: every transition table in it is headed "switch at t = 3000", which is where this comes from.
REFERENCE_PHASE_STEPS: int = 3000

#: Likewise recovered. The file carries no per-seed table and prints identical phase-1 bins for
#: two arms that differ only in phase 2, which one seed explains and several do not.
REFERENCE_SEEDS: tuple[int, ...] = (0,)


#: The parameters a G4 mechanic is allowed to move, and nothing else. A mechanic that changed a
#: parameter outside this set would be two mechanics, and A5 says one change per experiment.
HARDENING_PARAMETERS: dict[str, tuple[str, ...]] = {
    # Both economics terms, because EV-neutrality fixes only their RATIO and a K mechanic may
    # absorb the change in either one -- see HOLD_MODES. Declaring only `prep_value` would have
    # meant the `hold="value"` variant passed this check by borrowing the diagnostic's allowance
    # below, which is an accident rather than a statement.
    "K": ("n_preps", "prep_value", "prep_fail"),
    # NOT a mechanic and it has no claim line. It exists because mechanic 1 returned a negative
    # result with a confound in it: raising K forces `prep_value` up to hold the chance EV at
    # zero, so K and the size of a correct preparation's payoff move together and no contrast
    # between them separates the two. This control scales the payoff WITHOUT touching K. See
    # `world_with_economics_scaled`.
    "economics-diagnostic": ("prep_value", "prep_fail"),
}


#: The two EV-neutral ways to raise K, and why there is a choice at all. The invariant is
#: `prep_value = (K-1) * prep_fail`, which fixes the RATIO and not either term -- so raising K
#: can be absorbed by raising the reward or by lowering the penalty, and the two are different
#: worlds. `docs/G4_WRITEUP.md` §2.1 found that mechanic 1 took the first without noticing there
#: was a second, and that the first is exactly the confound that spoiled it.
HOLD_MODES = ("fail", "value")


def world_at_k(k: int, world: dict[str, Any] | None = None,
               hold: str = "fail") -> dict[str, Any]:
    """G4 mechanic 1: the world of record with K raised, and the economics kept honest.

    `prep_value` moves with K because the invariant is `prep_value = (K-1) * prep_fail`, which is
    what holds the expected value of a chance preparation at exactly zero. Raising K and leaving
    `prep_value` alone would make a chance preparation worth `prep_value/K - prep_fail*(K-1)/K`,
    which at K = 7 with the K = 5 economics is **-0.250** -- a harder world for a reason that has
    nothing to do with the mapping space, with every arm getting worse together while the contrast
    between them measured something else.

    Nothing else moves. `prep_fail`, the era clocks, the metabolism, the staging and the densities
    are the world of record's.
    """
    base = dict(WORLD if world is None else world)
    if int(k) <= N_TYPES:
        raise WorldMismatch(
            f"K = {k} with T = {N_TYPES}: a mapping sends each type to a DISTINCT preparation, so "
            f"K must exceed T for the mapping space to be non-trivial.")
    if hold not in HOLD_MODES:
        raise WorldMismatch(f"hold must be one of {HOLD_MODES}; got {hold!r}")
    base["n_preps"] = int(k)
    if hold == "fail":
        # `prep_fail` fixed, so `prep_value` rises: 1.0 -> 1.5 at K = 7. This is what mechanic 1
        # ran, and G4 §2.1 is the cost: a correct preparation becomes 50% more valuable in the
        # same step the mapping space grows, which feeds the genetic channel v3.11 found dominant.
        base["prep_value"] = (int(k) - 1) * float(base["prep_fail"])
    else:
        # `prep_value` fixed, so `prep_fail` falls: 0.25 -> 0.1667 at K = 7. What being RIGHT is
        # worth no longer moves with K, which is the confound removed. It is not confound-free --
        # mistakes get cheaper, and a cheaper mistake is a cheaper search -- but that is a
        # different lever pointing the other way, and running both is how they separate.
        base["prep_fail"] = float(base["prep_value"]) / (int(k) - 1)
    return base


def world_with_economics_scaled(scale: float, world: dict[str, Any] | None = None
                                ) -> dict[str, Any]:
    """The diagnostic control for mechanic 1's negative result. **Not a mechanic.**

    Mechanic 1 moved two things at once and could not have moved only one: the invariant
    `prep_value = (K-1) * prep_fail` means raising K from 5 to 7 raises `prep_value` from 1.0 to
    1.5, so a correct preparation becomes 50% more valuable in the same step that the mapping
    space grows. The modulator is a table and does not move with the economics -- that is
    measured -- but *energy* does, and energy is survival and reproduction. A larger payoff for
    being right feeds the genetic channel, which is precisely what v3.11 found dominates:

        "At six mappings the genetic baseline is large ... and cannot be removed by shortening
        the era."

    So this scales `prep_value` and `prep_fail` together at the world of record's K, holding
    `prep_value = (K-1) * prep_fail` and therefore the chance EV at zero. If the content effect
    collapses here too, the collapse is the economics and mechanic 1 never tested K at all.

    **What it does not isolate, stated because it cannot be fixed.** Holding the chance EV at zero
    at a fixed K forces `prep_value / prep_fail = K - 1`, so scaling one scales the other: this
    control raises the reward for being right AND the penalty for being wrong. It separates *the
    scale of the preparation economics* from *K*, which is the confound in mechanic 1. It does not
    separate reward from penalty, and no EV-neutral world can.
    """
    base = dict(WORLD if world is None else world)
    if scale <= 0:
        raise WorldMismatch(f"scale must be positive; got {scale}")
    k = int(base.get("n_preps", N_PREPS))
    base["n_preps"] = k                       # stated, or it reads as "moved to None"
    base["prep_fail"] = float(base["prep_fail"]) * float(scale)
    base["prep_value"] = (k - 1) * base["prep_fail"]
    return base


def check_hardened_world(world: dict[str, Any]) -> dict[str, Any]:
    """A G4 world is the world of record with exactly one mechanic's parameters moved.

    Returns what moved, so a write-up states it rather than asserting it. Raises if anything
    outside a declared mechanic's parameter set has changed -- because a mechanic that moved a
    second parameter would be two mechanics, and its claim line could not say which one produced
    the number.
    """
    allowed = {p for params in HARDENING_PARAMETERS.values() for p in params}
    # n_preps is not in EXPECTED_WORLD -- it is a Config field the research WORLD never set -- so
    # the reference for "has it moved" carries its default explicitly. Without that, raising K
    # would not be reported as a change at all.
    reference = {**EXPECTED_WORLD, "n_preps": N_PREPS}
    moved = {k: (reference.get(k, "<absent>"), world.get(k))
             for k in set(reference) | set(world)
             if world.get(k) != reference.get(k, world.get(k))}
    illegal = {k: v for k, v in moved.items() if k not in allowed}
    if illegal:
        lines = "\n".join(f"    {k}: {a!r} -> {b!r}" for k, (a, b) in sorted(illegal.items()))
        raise WorldMismatch(
            f"a hardened world moved parameters no mechanic declares:\n{lines}\n"
            f"  A5: one change per experiment. A mechanic that moves a second parameter is two "
            f"mechanics and its claim line cannot say which produced the number.")
    check_chance_ev_is_zero(world)
    return moved


class WorldMismatch(AssertionError):
    """The research lineage's world is not the world this build was written against.

    Raised at construction, never at read time. A number produced against an unexpected world is
    not a number that failed a gate -- it is a number about a different experiment, and the only
    safe thing to do with it is to not produce it.
    """


def check_world(world: dict[str, Any] | None = None) -> None:
    """Every pinned parameter, checked against the world of record. Raises `WorldMismatch`."""
    world = WORLD if world is None else world
    wrong = {
        k: (v, world.get(k, "<absent>"))
        for k, v in EXPECTED_WORLD.items()
        if k not in world or world[k] != v
    }
    if wrong:
        lines = "\n".join(f"    {k}: expected {exp!r}, found {got!r}" for k, (exp, got) in
                          sorted(wrong.items()))
        raise WorldMismatch(
            f"analysis_v3_13.WORLD is not the world this build was written against:\n{lines}\n"
            f"  This is not a gate failure. Re-pin EXPECTED_WORLD deliberately, with a note on "
            f"what moved and why, or the numbers below are about a different experiment."
        )

    engine = {k: getattr(_sim, k) for k in EXPECTED_ENGINE}
    wrong_engine = {k: (v, engine[k]) for k, v in EXPECTED_ENGINE.items() if engine[k] != v}
    if wrong_engine:
        lines = "\n".join(f"    {k}: expected {exp!r}, found {got!r}" for k, (exp, got) in
                          sorted(wrong_engine.items()))
        raise WorldMismatch(f"sim_v3_13's constants have moved:\n{lines}")


def check_chance_ev_is_zero(world: dict[str, Any] | None = None) -> float:
    """`prep_value == (K-1) * prep_fail`, so a chance preparation is worth exactly nothing.

    This is the invariant that makes the preparation task an experiment. Without it a population
    can ride the preparation payoff upward without ever learning the mapping, and the hit rate
    stops being a measurement of knowledge. Returns the expected value, which must be 0.0.
    """
    world = WORLD if world is None else world
    value, fail = float(world["prep_value"]), float(world["prep_fail"])
    # K from the WORLD, not from the module. G4 raises it, and a guard that read the module
    # constant would pass a K=7 world carrying K=5 economics -- which is exactly the mistake this
    # guard exists to catch, committed by the guard itself.
    k = int(world.get("n_preps", N_PREPS))
    if value != (k - 1) * fail:
        raise WorldMismatch(
            f"at K = {k}, prep_value {value} != (K-1) * prep_fail = {(k - 1) * fail}. A chance "
            f"preparation is worth {value / k - fail * (k - 1) / k:+.4f}, not zero, so a positive "
            f"return is no longer evidence of knowledge."
        )
    return value / k - fail * (k - 1) / k


# ---------------------------------------------------------------------------------------------
# the three levels a hit rate is read against
# ---------------------------------------------------------------------------------------------

#: A random preparation: 1/K = 0.200. Not a null for a stale mark -- see `matched_stale_null`.
CHANCE: float = 1.0 / N_PREPS

#: "Always prepare k" for a k useful in this era: right for one food type, wrong for the others,
#: and no type knowledge at all. Note that ACROSS eras such a policy scores only 1/T, because with
#: distinct mappings k is useless in a third of them -- so a genome beats chance only by tracking
#: the era, never by holding one preparation.
TYPE_BLIND: float = 1.0 / N_TYPES

#: The conjunction: the right preparation for each type.
FULL: float = 1.0


def matched_stale_null(hit_rate: float, n_preps: int = N_PREPS) -> float:
    """A2.2: the null for following a STALE mark, derived from the arm's own structure.

    A mark that endorses a preparation which is wrong for this type now is the unconfounded cell,
    because following it is a mistake. But 1/K is the wrong null for it. An agent that simply
    KNOWS the correct preparation picks it and so never agrees with a stale mark, whatever it
    reads -- so the null is the chance of landing on the endorsed-but-wrong preparation GIVEN you
    did not pick the correct one:

        (1 - hit) / (K - 1)

    `hit_rate` is the arm's own founder-free preparation hit rate, not a constant -- and `K` is
    not one either now that G4 raises it. A null left at the module default would silently be a
    K=5 null in a K=7 world, which is the same class of mistake as a hard-coded 1/K: a number that
    looks like a matched null and is not one.
    """
    return (1.0 - float(hit_rate)) / (int(n_preps) - 1)


# ---------------------------------------------------------------------------------------------
# eras
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EraClocks:
    """Three independent clocks, all in steps. The world holds a fast fact and a slow one, and
    the whole design turns on their being different speeds."""

    #: Which of food types 0 and 1 is safe to eat raw. Faster than a generation, so genes cannot
    #: track it. Type 2 has no raw value at all, so the flip does not touch it.
    flip_every: int
    #: The type -> preparation `mapping`, AND the label permutation pi. Slow.
    prep_every: int
    #: Patch centres move; stations travel with them.
    patch_drift_every: int
    #: 0 means pi is redrawn with the mapping. A multiple of `prep_every` makes a label's meaning
    #: outlive what it names -- v3.5's tempo condition, which is the `slow` arm.
    label_every: int

    @property
    def pi_epochs_per_mapping(self) -> int:
        """How many mapping eras one label epoch spans. 1 unless labels are slow."""
        if not self.label_every:
            return 1
        return max(1, round(self.label_every / max(1, self.prep_every)))


def clocks_of(cfg_kw: dict[str, Any]) -> EraClocks:
    """The clocks a given configuration runs on, with the engine's own defaults filled in."""
    base = _sim.Config()
    return EraClocks(
        flip_every=int(cfg_kw.get("flip_every", base.flip_every)),
        prep_every=int(cfg_kw.get("prep_every", base.prep_every)),
        patch_drift_every=int(cfg_kw.get("patch_drift_every", base.patch_drift_every)),
        label_every=int(cfg_kw.get("label_every", base.label_every)),
    )


# ---------------------------------------------------------------------------------------------
# standing densities, as cover fractions
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class StandingDensity:
    """§3.6 of the G1 spec. Two of these are derivable and one is not, and saying which is which
    is the point of the record.

    Food cover is **consumption-limited, not spawn-limited**: unopposed, 8 patches x Poisson(6.0)
    is 48 spawn attempts a step against a rot of 0.005, which would saturate the patches many
    times over. What actually stands is set by how fast the population eats it -- so it differs
    per arm and per phase and cannot be derived from parameters at all. It is recorded nowhere in
    the repository, which is the `sr_w` lesson repeating, and the provider measures it (D10).
    """

    grid_cells: int
    patch_cells_if_disjoint: int
    patch_cover_if_disjoint: float
    spawn_attempts_per_step: float
    rot_per_cell_per_step: float
    mark_half_life_steps: float


def standing_density(cfg_kw: dict[str, Any] | None = None) -> StandingDensity:
    """The derivable half of §3.6. What is not derivable is not guessed."""
    import math

    base = _sim.Config()
    kw = dict(WORLD if cfg_kw is None else cfg_kw)
    grid = int(kw.get("grid", base.grid))
    n_patches = int(kw.get("n_patches", base.n_patches))
    radius = int(kw.get("patch_radius", base.patch_radius))
    spawn = float(kw.get("spawn_per_patch", base.spawn_per_patch))
    rot = float(kw.get("food_rot", base.food_rot))
    decay = float(kw.get("mark_decay", base.mark_decay))

    side = 2 * radius + 1                     # Chebyshev radius -> a square patch
    patch_cells = n_patches * side * side
    return StandingDensity(
        grid_cells=grid * grid,
        patch_cells_if_disjoint=patch_cells,
        patch_cover_if_disjoint=patch_cells / (grid * grid),
        spawn_attempts_per_step=n_patches * spawn,
        rot_per_cell_per_step=rot,
        mark_half_life_steps=(math.log(2.0) / decay) if decay else float("inf"),
    )
