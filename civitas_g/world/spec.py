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
    if value != (N_PREPS - 1) * fail:
        raise WorldMismatch(
            f"prep_value {value} != (K-1) * prep_fail = {(N_PREPS - 1) * fail}. A chance "
            f"preparation is worth {value / N_PREPS - fail * (N_PREPS - 1) / N_PREPS:+.4f}, not "
            f"zero, so a positive return is no longer evidence of knowledge."
        )
    return value / N_PREPS - fail * (N_PREPS - 1) / N_PREPS


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


def matched_stale_null(hit_rate: float) -> float:
    """A2.2: the null for following a STALE mark, derived from the arm's own structure.

    A mark that endorses a preparation which is wrong for this type now is the unconfounded cell,
    because following it is a mistake. But 1/K is the wrong null for it. An agent that simply
    KNOWS the correct preparation picks it and so never agrees with a stale mark, whatever it
    reads -- so the null is the chance of landing on the endorsed-but-wrong preparation GIVEN you
    did not pick the correct one:

        (1 - hit) / (K - 1)

    `hit_rate` is the arm's own founder-free preparation hit rate, not a constant.
    """
    return (1.0 - float(hit_rate)) / (N_PREPS - 1)


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
