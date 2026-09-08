"""The vector domain adapter (B§3), and the modulator event table (D7).

B§3 states the adapter in four clauses: observations are fixed-length numeric vectors; actions are
a discrete set; the judge is the world's energy accounting; the era rotates the mapping and pi.
This module is that statement made executable, plus the one correction the G0 audit found.

**The correction (F8/D7).** The directive says "the agent's own energy change as the modulator".
`sim_v3_13.resolve_action` says the opposite, in terms:

    The modulator is NOT the agent's own energy change in general; it is this table.

`m` is a literal +/-1 in the source, and its independence from `prep_value` and `prep_fail` is
load-bearing: it is exactly why v3.12 could halve `prep_fail` -- slowing the survival sorting that
was swamping the learner -- without touching the learning signal at all. G4 changes the economics
one mechanic at a time, so if this were recorded as "the energy delta" the first economics change
would silently move the learner and every per-mechanic claim line would be confounded from the
start. `modulator_is_independent_of_economics` is that fact as a measurement.

**The leak rule for this domain (§47, kept).** Nothing an agent reads can name an answer, because
agents read channels, not words. Concretely, and checked by `check_no_leak`:

* the store carries **labels**, never preparation indices -- a mark is written at `pi(k)`;
* pi is **redrawn every era**, so a label denotes a different preparation each time and the
  label -> preparation binding is unavailable to selection;
* reading is an **observation, not an action**, so a null cannot be ambiguous between "cannot
  read" and "did not bother";
* the read channels are **here-only**, so nothing about navigation enters the claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sim_v3_13 as _sim
from civitas_g.world.spec import (
    APPETITE_CH,
    EAT,
    ENERGY,
    HERE,
    N_ACTIONS,
    N_CH,
    N_IN,
    N_PREPS,
    N_TYPES,
    PREP0,
    READ,
)

# ---------------------------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    index: int
    name: str
    #: False for the preparations, which are masked until the chain comes on in phase 2.
    live_in_phase_one: bool


#: The discrete action set, in engine order. `MOVES` is [[-1,0],[1,0],[0,-1],[0,1]], so 0..3 are
#: -y, +y, -x, +x on the torus.
ACTIONS: tuple[Action, ...] = (
    Action(0, "move_-y", True),
    Action(1, "move_+y", True),
    Action(2, "move_-x", True),
    Action(3, "move_+x", True),
    Action(EAT, "eat", True),
    *(Action(PREP0 + k, f"prepare_{k}", False) for k in range(N_PREPS)),
)

#: Reading the record is NOT here, and that is the design. It is an observation channel, so a null
#: cannot be ambiguous between "cannot read" and "does not bother".
READ_IS_AN_ACTION = False


# ---------------------------------------------------------------------------------------------
# the modulator
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModulatorEvent:
    """One row of `resolve_action`'s table."""

    event: str
    m: float
    energy: str
    consumes_cell: bool
    note: str


#: The complete list. Seven events; four carry a modulator, three are zero. Nothing on food is
#: silent: every action on a food cell resolves immediately and two-sidedly.
MODULATOR_EVENTS: tuple[ModulatorEvent, ...] = (
    ModulatorEvent("eat_safe", +1.0, "+food_value", True,
                   "the fast fact: which of types 0/1 is safe raw, flipped every flip_every"),
    ModulatorEvent("eat_poison", -1.0, "-poison_value", True,
                   "the same fact, the other way up"),
    ModulatorEvent("prep_ok", +1.0, "+prep_value", True,
                   "the slow fact: k == mapping[ftype]. Writes a mark at label pi(k), sign +"),
    ModulatorEvent("prep_bad", -1.0, "-prep_fail", True,
                   "k != mapping[ftype]. Writes a mark at label pi(k), sign -"),
    ModulatorEvent("move", 0.0, "-move_cost", False,
                   "resolves BEFORE the food test, so a move is a move whatever is underfoot"),
    ModulatorEvent("noop", 0.0, "-noop_cost", False,
                   "eat or prepare on an empty cell"),
    ModulatorEvent("eat_inedible", 0.0, "0", False,
                   "eat on type 2. The cell is NOT consumed: if eating destroyed it a naive "
                   "agent could clear the board of exactly the food this experiment is about"),
)

MODULATOR_BY_EVENT: dict[str, float] = {e.event: e.m for e in MODULATOR_EVENTS}


def modulator_is_independent_of_economics(seed: int = 0, steps: int = 400) -> tuple[bool, str]:
    """D7 as a measurement: the modulator is the table, not the energy delta.

    Runs the same world twice at one seed with two different `(prep_value, prep_fail)` pairs and
    compares the modulator sequence. Energy differs; `m` must not. Returns `(ok, detail)` rather
    than asserting, so a caller can report the detail.

    This walks the world directly rather than through `run()`, because `run()`'s births and deaths
    are energy-driven: at 400 steps the populations would diverge for a legitimate reason and the
    comparison would be about demography rather than about the modulator.
    """
    def trace(prep_value: float, prep_fail: float) -> list[tuple[str, float]]:
        rng = np.random.default_rng(seed)
        cfg = _sim.Config(seed=seed, chain=True, prep_value=prep_value, prep_fail=prep_fail)
        world = _sim.World(cfg, rng)
        world.chain_on = True
        agent = _sim.Agent(cfg, np.random.default_rng(seed + 1), 0, 5, 5)
        out: list[tuple[str, float]] = []
        act_rng = np.random.default_rng(seed + 2)
        for t in range(steps):
            world.step(t)
            action = int(act_rng.integers(N_ACTIONS))
            m, event, _ = _sim.resolve_action(agent, action, world, cfg, rng, t)
            agent.energy = 3.0                      # pinned: the metabolism is not what is tested
            out.append((event, float(m)))
        return out

    a = trace(1.0, 0.25)
    b = trace(4.0, 1.0)                             # 4x the economics, same invariant
    if len(a) != len(b):
        return False, f"traces differ in length: {len(a)} vs {len(b)}"
    bad = [(i, x, y) for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    if bad:
        i, x, y = bad[0]
        return False, (f"{len(bad)} of {len(a)} steps differ; first at {i}: {x} vs {y}. The "
                       f"modulator is tracking the economics, which breaks G4's per-mechanic "
                       f"claim lines before the first mechanic is added.")
    return True, f"{len(a)} steps, modulator identical under a 4x change in the economics"


# ---------------------------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ObsField:
    """One slice of the observation vector, named."""

    lo: int
    hi: int                 # exclusive
    name: str
    live: bool
    note: str

    @property
    def width(self) -> int:
        return self.hi - self.lo


_CH_NAMES = [f"food_{'ABC'[i]}" for i in range(N_TYPES)] + ["occupancy", "appetite"]

#: The 31-input layout, §3.5 of the G1 spec. Five of the thirty-one are structurally dead --
#: `scaffold_food` is False, so the appetite channel is fed zeros -- and they are recorded as dead
#: rather than removed: changing the layout changes every genome and voids the reproduction (D9).
OBS_LAYOUT: tuple[ObsField, ...] = (
    *(ObsField(4 * i, 4 * (i + 1), f"{name}_dirsum", name != "appetite",
               "directional sums (-y, +y, -x, +x) over a 5x5 window, normalised by v(2v+1) = 10")
      for i, name in enumerate(_CH_NAMES)),
    *(ObsField(HERE + i, HERE + i + 1, f"{name}_here", name != "appetite",
               "the value of the channel on the agent's own cell")
      for i, name in enumerate(_CH_NAMES)),
    ObsField(ENERGY, ENERGY + 1, "energy", True, "a.energy / max_energy"),
    ObsField(READ, READ + N_PREPS, "read_channels", True,
             "sym_gain * marks[food type underfoot, :, y, x]. Here-only, type-selected, gated by "
             "one heritable scalar. Zero with no record or no food underfoot."),
)

DEAD_INPUTS: tuple[int, ...] = tuple(
    i for f in OBS_LAYOUT if not f.live for i in range(f.lo, f.hi)
)


def describe_observation() -> str:
    """The layout as a table, for a manifest or a write-up."""
    rows = [f"{'idx':>7}  {'field':<22}{'live':<6}note"]
    for f in OBS_LAYOUT:
        idx = f"{f.lo}" if f.width == 1 else f"{f.lo}-{f.hi - 1}"
        rows.append(f"{idx:>7}  {f.name:<22}{'yes' if f.live else 'DEAD':<6}{f.note}")
    return "\n".join(rows)


def check_observation_layout() -> None:
    """The layout this package describes is the layout the engine builds. Raises on a mismatch."""
    covered = sorted(i for f in OBS_LAYOUT for i in range(f.lo, f.hi))
    if covered != list(range(N_IN)):
        raise AssertionError(
            f"OBS_LAYOUT covers {len(covered)} of {N_IN} inputs, with gaps or overlaps"
        )
    if READ + N_PREPS != N_IN:
        raise AssertionError(
            f"the read channels do not end the vector: {READ} + {N_PREPS} != {N_IN}")
    if HERE != N_CH * 4:
        raise AssertionError(f"HERE {HERE} != N_CH * 4 = {N_CH * 4}")
    if APPETITE_CH != N_TYPES + 1:
        raise AssertionError("the appetite channel is not where this layout says it is")


# ---------------------------------------------------------------------------------------------
# the leak rule for the vector domain (§47)
# ---------------------------------------------------------------------------------------------

class DomainLeak(AssertionError):
    """The store, or an observation, could name an answer."""


def check_no_leak(*, marks_are_labels: bool, pi_redrawn_every_era: bool,
                  read_is_observation: bool, read_is_here_only: bool) -> None:
    """§47 for this domain, stated as the four things that must hold.

    Kept as an argument-taking check rather than a fixed assertion so a caller -- a self-test, or a
    future arm -- must state each condition explicitly about the world it built, instead of
    inheriting a pass from the world this file was written against.
    """
    if not marks_are_labels:
        raise DomainLeak(
            "the store carries preparation indices rather than labels. A mark would then name "
            "an answer directly and no permutation could hide it."
        )
    if not pi_redrawn_every_era:
        raise DomainLeak(
            "pi is not redrawn. A genome that fixed on 'label j -> preparation k' would be right "
            "forever, so the binding would be inheritable and transmission would be unnecessary."
        )
    if not read_is_observation:
        raise DomainLeak(
            "reading is an action. A null would then be ambiguous between 'cannot read' and "
            "'did not bother', and the arm would measure neither."
        )
    if not read_is_here_only:
        raise DomainLeak(
            "the read channels reach beyond the agent's own cell, so navigation enters the claim."
        )


def check_no_self_echo() -> tuple[bool, str]:
    """No self-echo is a PROPERTY of the design, not an assumption.

    A preparation consumes the food cell, so the mark it writes cannot be read for a preparation
    until food respawns there -- and the reader is then whoever is standing on it. An agent can
    never read its own mark about the food it just prepared. Verified against the engine rather
    than asserted: prepare on a cell, then check the cell holds no food.
    """
    rng = np.random.default_rng(0)
    cfg = _sim.Config(seed=0, chain=True, record="real")
    world = _sim.World(cfg, rng)
    world.chain_on = True
    agent = _sim.Agent(cfg, np.random.default_rng(1), 0, 5, 5)
    agent.energy = 3.0
    for ftype in range(N_TYPES):
        for k in range(N_PREPS):
            world.food[:, 5, 5] = False
            world.food[ftype, 5, 5] = True
            _m, event, info = _sim.resolve_action(agent, PREP0 + k, world, cfg, rng, 0)
            if event not in ("prep_ok", "prep_bad"):
                return False, f"preparation {k} on type {ftype} resolved as {event!r}"
            world.write_mark(ftype, k, info["ok"], 5, 5, cfg)
            if world.food[:, 5, 5].any():
                return False, (f"the cell still holds food after preparation {k} on type "
                               f"{ftype}: the writer could read its own mark")
    return True, f"{N_TYPES * N_PREPS} (type, preparation) pairs: the cell is consumed every time"
