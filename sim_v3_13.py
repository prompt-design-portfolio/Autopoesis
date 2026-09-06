"""
Autopoiesis v3.6 -- the grown learner on the recipe task alone.

The v2.9b tool world (items x stations -> tool -> nuts, recipe changes every
2000 steps) carrying the v3 grown learner (synapse = innate weight + learned
component H, local rule with an eligibility trace, modulator = the agent's own
energy change).  The lexicon, the shared store and the culture layer are gone.

The question: can within-life plasticity acquire a CONJUNCTIVE, DELAYED-CREDIT
fact -- (item x station) -> tool -- when the modulator m arrives at the nut,
many steps after the station attempt that earned it?

What is hand-wired and what is not
----------------------------------
Hand-wired (identical in every condition, so it cannot explain a difference):
  * approach behaviour.  An innate 'goal' channel points the agent at any item
    when empty-handed, at any station when carrying one, at nuts when carrying a
    tool; a 'food_any' channel points it at food of either type.  Navigation is
    not the question here (v2.9b made the same call).
  * the interaction verb: standing on a station while holding an item and
    choosing 'stay' IS an attempt.

Not wired, and the whole question:
  * WHICH item and WHICH station.  The observation carries the inventory
    one-hot and the per-type station/item channels; a conjunctive policy has to
    be built out of them, innately (mutation) or within life (H).
  * WHICH food type is safe.

Ports and deviations from the two parents, all deliberate:
  * hidden = 24 (v2.9b's value for this world, not v3's 16).  The nav scaffold
    saturates 10 hidden units, so 16 would leave 6 free features for H2 to read
    -- a null by construction.
  * the appetite channel is food of EITHER type (v2.9b weighted it by a
    hand-wired taste register; the v3 learner has to find the taste itself).
  * mode 'register' is gone (retired in v3.3).  The shared store is gone.
  * fail_cost defaults to 0: a wrong attempt produces NO modulator, only the
    loss of the item.  Credit for the recipe is purely delayed.
"""

import time
from dataclasses import dataclass, asdict

import numpy as np


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class Config:
    # world (v1/v3 flip world)
    grid: int = 48
    n_steps: int = 8000
    view: int = 2
    init_pop: int = 300
    min_pop: int = 40
    max_pop: int = 400
    # brain
    hidden: int = 24              # v2.9b's value for this world; see module docstring
    decay: float = 0.02
    repair_gain: float = 0.05
    repair_cost: float = 0.0005
    integrity_threshold: float = 0.5
    # metabolism
    base_cost: float = 0.006
    move_cost: float = 0.002
    noop_cost: float = 0.002      # what a no-op costs.  The spec says move_cost, and this default IS
                                  # move_cost; it is a separate field only so the pre-check can
                                  # measure what the no-op charge costs the population, since fix A
                                  # makes `interact` a guaranteed no-op for 1/6 of actions in phase 1.
    carry_cost: float = 0.0       # v3.9 amendment 3: zero.  A carry tax punishes exploration, not
                                  # the chain -- a random `interact` press picked up an item that
                                  # then taxed the agent forever, which is why `fixed` and
                                  # `scrambled` sat at the floor.  The chain's costs are
                                  # pickup_cost, the item lost on a wrong attempt, and fail_cost.
    start_energy: float = 1.5
    founder_energy: float = 3.0
    repro_threshold: float = 3.0
    repro_cost: float = 1.5
    max_energy: float = 6.0
    # food
    food_value: float = 0.7
    poison_value: float = 0.5
    n_patches: int = 8
    patch_radius: int = 6
    spawn_per_patch: float = 3.0
    food_rot: float = 0.005
    patch_drift_every: int = 400
    flip_every: int = 300         # v3 FAST: faster than a generation, genes cannot track it
    # the tool task (v2.9b)
    n_items: int = 3
    n_stations: int = 2
    stations_per_patch: int = 8       # v3.9 amendment 3: stations of EACH type per food patch.
                                      # The chain is co-located with foraging, so the recipe is an
                                      # expensive FACT rather than an expensive JOURNEY.  Nothing
                                      # spawns outside a patch; stations move with their patch.
    stations_per_type: int = 0        # retained, unused: global placement is gone
    items_per_step: float = 0.0      # retained, unused: items now spawn per patch, like food
    items_per_patch: float = 0.22    # v3.9 amendment 3: items spawn INSIDE the food patches, like
                                     # food.  Solved to ~20-25% item cover within the patches.
    nuts_per_patch: float = 0.0      # v3.9 amendment 2: no nuts in this build.  The chain is
    nuts_uniform_unused: float = 0.0 # pickup -> carry -> attempt, and the attempt pays.  The
                                     # station -> nut bridge returns in v3.11 as its own change.
    nuts_uniform: float = 0.0     # retained at 0; nuts return in v3.11 with the bridge.
    item_rot: float = 0.005
    prep_value: float = 1.5           # a CORRECT attempt pays this immediately, m = +1, and the
                                      # item is consumed.  Nothing is carried afterwards: there is
                                      # no tool state in this build.
    pickup_cost: float = 0.02
    prep_fail: float = 0.5        # a WRONG preparation costs this (poison-sized) and fires m = -1.
    frozen: bool = False          # FROZEN REPLAY: births, deaths and injection are all disabled.
                                  # Energy is still tracked and still spent -- the metabolism runs
                                  # -- it is simply not lethal.  Nothing about the POPULATION can
                                  # change, so the only thing that can move a hit rate over the
                                  # window is H.  That is what makes the replay an attribution.
    era_snap_keep: int = 2        # era-boundary snapshots retained (rolling, most recent last)
    era_snap_max: int = 300       # agents sampled per snapshot, to bound the pickle
    # --- v3.13 record ---
    record: str = "none"          # "none" | "real" | "noise".  "noise" randomises the LABEL on
                                  # write, holding mark density, channel statistics and decay
                                  # identical while destroying the label->preparation association.
    mark_decay: float = 0.004     # per-step multiplicative decay of a mark's magnitude.  A mark
                                  # must OUTLIVE the agent that wrote it or nothing passes between
                                  # agents, and must NOT outlive the era or it actively misleads.
                                  # Half-life = ln2/0.004 ~ 173 steps: ~4 half-lives inside
                                  # prep_every 700.  Measured, not assumed -- see mark_age stats.
    sym_gain_init: float = 0.05   # the heritable read gain starts NEAR ZERO: attending to the
                                  # record is evolved, not wired.  May go negative -- "do the
                                  # opposite of the mark" is a coherent policy and the right one
                                  # in `noise`.
    label_every: int = 0          # 0 = pi is redrawn with the mapping.  A MULTIPLE of prep_every
                                  # makes a label's meaning outlive what it names -- v3.5's tempo
                                  # condition, the pre-registered follow-up if v3.13 nulls on
                                  # capacity.  Not used unless that follow-up fires.
    type_spawn_w: tuple = None    # OPEN 3a: relative spawn weight per food type.  Type 2 has only
                                  # ONE consumption route (preparation) where types 0 and 1 have two
                                  # (raw eating as well), so at equal spawn it accumulates -- 49-55%
                                  # of standing food in the v3.12 pre-check at spawn_per_patch 6.0.
                                  # The spec's pre-agreed fix is to lower C's SPAWN rate, never to
                                  # let `eat` consume it: consuming would let a naive agent clear
                                  # the board of exactly the food the experiment is about.
    force_mapping: tuple = None   # DIAGNOSTIC ONLY: pin the mapping and stop all redraws, so a
                                  # replay can be run against a KNOWN mapping.  A knockout that
                                  # re-seeds the world gets that seed's FIRST mapping, which is
                                  # not the mapping the replayed genomes were selected under.
    prep_every: int = 2000        # the food -> preparation mapping is redrawn this often,
                                  # independently of the safe/poison flip.
    fail_cost: float = 0.0        # retired with the chain
    recipe_every: int = 2000      # ~8-13 generations per era
    # mutation
    mut_sigma: float = 0.15
    mut_rate: float = 0.10
    gene_sigma: float = 0.05
    action_noise: float = 0.3
    # the learner (v3.0)
    mode: str = "plastic"          # "fixed" | "plastic" | "random" (uniform over the six actions;
                                   # no plasticity, no learning -- the behavioural null for contact
                                   # rates, not an outcome arm)
    plastic_layers: str = "both"   # "both" | "W1" | "W2"
    eta_init: float = 0.2          # founders' learning-rate genes ~ U(0, eta_init)  (v3 FAST)
    eta_max: float = 0.5
    h_max: float = 2.0
    scramble: bool = False         # control: random-sign modulator, same magnitude, no information
    # v3.6b: the instinct's strength is heritable.  The pre-run positive control failed because
    # the scaffold's logit-4 push swamps anything the output layer learns; rather than pick a
    # balance by hand, evolution sets it per condition, the same move as the handoff's input-gain
    # gene.  Founder values bracket the hand-set 4.0 that the earlier passes used.
    nav_dir_init: tuple = (2.0, 6.0)    # founders' approach gene ~ U(...)
    nav_here_init: tuple = (2.0, 6.0)   # founders' interact gene ~ U(...)
    nav_sigma: float = 0.2              # per-generation mutation, ~2% of nav_max, as eta's 0.01 is of 0.5
    nav_max: float = 12.0
    scaffold_chain: bool = False   # RETIRED in v3.10 and forced False; see wire_nav.  Was: whether the instinct approaches items/stations/nuts and interacts
                                   # with them.  v3.8 sets this False: the chain has to be found
                                   # unwired, so nav_dir/nav_here stay in the genome but reach nothing.
    goal_channel: bool = False     # retired with the chain; channel 10 is dead.  Kept:  # keep the stage-machine 'goal' OBSERVATION (points at items when
                                   # empty-handed, stations when carrying, nuts when tooled).  This is
                                   # a hand-designed feature but not a valence -- it never says WHICH
                                   # item or station -- and it is identical in every condition, so it
                                   # cannot explain a between-condition difference.  Set False for a
                                   # stricter version in which the chain must be found from the
                                   # per-type channels alone.
    chain: bool = True             # is the tool chain present at all?  Set per PHASE, not per run:
                                   # v3.8 stages it off then on for one continuous population.
    scaffold_food: bool = True     # whether the instinct also approaches and eats food.  The chain
                                   # (item -> station -> nut) must be scaffolded or nothing happens
                                   # and no condition is readable.  Food need not be: it is the
                                   # POSITIVE CONTROL, and scaffolding it forces the agent to eat
                                   # whatever it stands on, so selectivity has to fight a ~5-logit
                                   # push.  With this False the food task is unscaffolded exactly as
                                   # in v3.1, while the chain keeps its instinct.
    innate_scale: float = 0.1      # multiplier on the random weights OF THE SCAFFOLD UNITS ONLY.
                                   # v2.9b applied 0.1 to the whole network; that was right there,
                                   # because its discrimination came from the hand-wired B path, but
                                   # it leaves the grown learner's H2 no basis to read -- every free
                                   # feature is crushed to ~0 while the nav units saturate.  Applying
                                   # it globally killed the v3.1 food effect; applying it to nothing
                                   # killed navigation and the population.  See the notebook.
    n_scaffold: int = 10           # hidden units the instinct occupies; the rest are the free basis
                                   # H2 reads.
    free_scale: float = 1.0        # weight scale of the FREE units, relative to the 0.3 init.  The
                                   # scaffold's nav units saturate and are type-blind, so whatever
                                   # discriminative signal the output layer can use lives entirely in
                                   # the free units; at 0.3 their response to a type channel is
                                   # tanh(0.3) ~ 0.29 against the instinct's ~1.0, which is why a
                                   # learned preference of ~0.4 logits moves behaviour almost not at
                                   # all.  Raising this scales the basis, not the instinct.
    eta_scale: float = 1.0         # v3.2 knockout hook: multiplies eta at learn time
    # the hand-wired ceiling (v2's private memory B; OFF except in that one condition)
    private_mem: bool = False      # exact per-pair credit, steering navigation and a soft veto
    pref_gain: float = 3.0
    veto_p: float = 0.8
    # logging
    trace_recency: bool = True     # log, at each phase-1 meal, the share of the eligibility trace's
                                   # L1 mass NOT attributable to steps older than `recency_k`:
                                   #     1 - ||lam2^k * e2(t-k)||_1 / ||e2(t)||_1
                                   # A trace dominated by the last few steps cannot carry credit back
                                   # to a station attempt; this measures that directly, in phase 1,
                                   # before the chain is there to confound it.
    recency_k: int = 5
    n_attempts: int = 10           # attempt-number curves run 1..n_attempts
    n_meals: int = 10
    log_every: int = 50
    seed: int = 0


SR_W = 8            # survivor-conditioned since-remap: preparations required EACH SIDE of a remap.
                    # 8, not 4.  The v3.11 grid showed the 4-window is structurally unable to see
                    # what it is for: it compares 4 preparations either side of a remap while the
                    # since-remap curve shows recovery takes ~5, so every arm read negative and the
                    # arms that had learned MOST read worst (plastic -0.600 against fixed -0.490).
                    # If n is thin at 8 the line is DROPPED, never narrowed back -- a narrower
                    # window does not measure the same thing less precisely, it measures the fall
                    # without the recovery.
SURV_EARLY = 2      # survivor curve early half: preparations 1-2 (late half is 6-10)
# v3.12: T food types and K preparations, both parameters rather than literals.  The mapping
# space is P(K,T) = K!/(K-T)!, and it has to exceed what standing variation can cover -- in the
# six-mapping world of v3.10-v3.11 a remap needed no adaptation at all, because the population
# already carried a genotype for the new mapping and survival sorting simply promoted it.
# ---------------------------------------------------------------------------
# v3.13: THE RECORD.  Writing is automatic, costless and universal -- every preparation writes
# (label, sign) to the cell for its food type.  There is no mark action, no writer gene and no
# writing policy, so there is no public good and no write-probability to collapse (finding #1).
# Labels are a permutation pi of the K preparations, REDRAWN AT EVERY REMAP, so label j denotes a
# different preparation each era and no genome can inherit what a label means (finding #2).
# Reading is an OBSERVATION, not an action, so a null cannot be ambiguous between "cannot read"
# and "does not bother".
# ---------------------------------------------------------------------------
N_TYPES = 3              # A, B, C.  C exists only once preparations are live, and is INEDIBLE RAW.
N_PREPS = 5              # prep_1 .. prep_5
EAT = 4                  # 0-3 move, 4 eat, 5..(5+K-1) prepare
PREP0 = 5
N_ACTIONS = PREP0 + N_PREPS          # 10
N_MAPPINGS = 60                      # P(5,3); asserted against the enumeration in the self-test
INTERACT = None           # retired with the chain
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
N_PAIRS = N_TYPES * N_PREPS      # the ceiling's private table: one cell per (food type, prep)


# --------------------------------------------------------------------------
# observation layout.  11 channels x (4 directional sums + 1 'here' value),
# then energy, the inventory one-hot, and 'carrying a tool'.
#   channels: 0 foodA  1 foodB  2 occupancy  3 food_any(appetite)
#             4,5,6 item types   7,8 station types   9 nuts   10 goal
# --------------------------------------------------------------------------

# v3.12 channel map.  The item / station / nut / goal / inventory / tool channels are RETIRED --
# they have been dead since v3.10 and carrying them meant every genome spent capacity on inputs
# that were structurally zero.  What remains is exactly what this world contains.
#   channels: 0 .. T-1  food types      T  occupancy      T+1  food_any (appetite)
FOOD_CH = 0
OCC_CH = N_TYPES
APPETITE_CH = N_TYPES + 1
N_CH = N_TYPES + 2                   # 5

HERE = N_CH * 4                      # 20 .. 24   'here' value of each channel
ENERGY = HERE + N_CH                 # 25
# v3.13: K READ CHANNELS for the food UNDERFOOT.  Channel j carries the age-decayed sign of the
# latest mark at label j for that food type at that cell, scaled by the agent's own sym_gain.
# They are here-only: a mark is read where you stand, never at a distance, so nothing about
# navigation enters the claim.  With no record, or no food underfoot, they are zero.
READ = ENERGY + 1                    # 26 .. 26+K-1
N_IN = READ + N_PREPS                # 31
INV = None                           # retired
TOOL = None                          # retired


def dirsum(w, v):
    return np.array([w[:v, :].sum(), w[v + 1:, :].sum(), w[:, :v].sum(), w[:, v + 1:].sum()]) / (v * (2 * v + 1))


def pair_id(item, station):
    return item * 2 + station


# --------------------------------------------------------------------------
# world
# --------------------------------------------------------------------------

class World:
    def __init__(self, cfg, rng):
        self.cfg, self.rng, g = cfg, rng, cfg.grid
        # THE STORE.  One age-decayed sign per (food type, label) per cell.  Per TYPE, because a
        # cell's food type changes over time and a mark left about type A must never be read as
        # evidence about type B.  T*K per cell: 60x60x3x5 = 54,000 floats, negligible.
        self.marks = np.zeros((N_TYPES, N_PREPS, g, g))
        self.pi = tuple(range(N_PREPS))       # label of preparation k is pi[k]; set below
        self.food = np.zeros((N_TYPES, g, g), dtype=bool)
        self.safe = 0                  # which of types 0,1 is safe to eat RAW.  Type 2 has no raw
                                       # value at all, so the flip does not touch it.
        self.patches = rng.integers(0, g, size=(cfg.n_patches, 2))
        self.items = np.full((g, g), -1, dtype=np.int8)
        self.nuts = np.zeros((g, g), dtype=bool)
        self.stations = np.full((g, g), -1, dtype=np.int8)
        # Station offsets are fixed relative to their patch, so stations travel with the patch on
        # drift.  Each patch carries cfg.stations_per_patch of EACH type, inside patch_radius.
        r = cfg.patch_radius
        self.station_offsets = []          # (patch index, dy, dx, type)
        for pi in range(cfg.n_patches):
            for st in range(cfg.n_stations):
                for _ in range(cfg.stations_per_patch):
                    dy, dx = rng.integers(-r, r + 1, size=2)
                    self.station_offsets.append((pi, int(dy), int(dx), st))
        self._place_stations()
        self.mapping = (tuple(int(x) for x in cfg.force_mapping) if cfg.force_mapping
                        else self._draw_mapping(None))   # food type -> preparation index (0..K-1)
        self.pi = self._draw_pi()
        self.flips, self.recipe_changes = [], []
        self.last_remap = 0            # step of the most recent mapping redraw
        self.chain_on = cfg.chain      # set per phase by run()
        self.chain_start = 0           # step at which the chain switched on; recipe eras are measured
                                       # from here, so phase 2 gets whole eras rather than a part-era

    def _in_patch(self):
        """Cells inside any patch, by Chebyshev radius -- the same shape spawning uses."""
        g, r = self.cfg.grid, self.cfg.patch_radius
        m = np.zeros((g, g), dtype=bool)
        idx = np.arange(g)
        for (py, px) in self.patches:
            dy = np.minimum(np.abs(idx - py), g - np.abs(idx - py))
            dx = np.minimum(np.abs(idx - px), g - np.abs(idx - px))
            m |= (dy[:, None] <= r) & (dx[None, :] <= r)
        return m

    def _place_stations(self):
        """Rebuild the station grid from the current patch positions.  Called at creation and after
        every patch drift, so a station is always the same offset from its patch."""
        g = self.cfg.grid
        self.stations[:] = -1
        for pi, dy, dx, st in self.station_offsets:
            py, px = self.patches[pi]
            self.stations[(py + dy) % g, (px + dx) % g] = st

    def _draw_mapping(self, old):
        """A mapping sends each of the T food types to a DISTINCT preparation, so the space is
        P(K, T) = K!/(K-T)!.  A redraw differs from the previous mapping in at least one type; it
        is drawn uniformly over the whole space, so it may agree with `old` on some types."""
        while True:
            m = tuple(int(x) for x in self.rng.choice(N_PREPS, N_TYPES, replace=False))
            if old is None or m != tuple(old):
                return m

    def _draw_pi(self):
        """A fresh permutation of the K preparations.  pi[k] is the LABEL written when preparation
        k is used.  Redrawn at every remap, so what a label denotes changes every era and the
        label->preparation binding is unavailable to selection: a genome that fixed on it would be
        right only until the next redraw."""
        return tuple(int(x) for x in self.rng.permutation(N_PREPS))

    def write_mark(self, ftype, k, ok, y, x, cfg):
        """Automatic, costless, universal.  Every preparation writes."""
        if cfg.record == "none":
            return
        label = self.pi[k] if cfg.record == "real" else int(self.rng.integers(N_PREPS))
        self.marks[ftype, label, y, x] = 1.0 if ok else -1.0

    def new_recipe(self, t):
        self.mapping = self._draw_mapping(self.mapping)
        # pi is redrawn WITH the mapping unless label_every says otherwise.  The tempo follow-up
        # (label_every a multiple of prep_every) is the only thing that separates them.
        if not self.cfg.label_every or (t % self.cfg.label_every == 0):
            self.pi = self._draw_pi()
        self.recipe_changes.append(t)
        self.last_remap = t

    def step(self, t):
        cfg, g, rng = self.cfg, self.cfg.grid, self.rng
        changed_recipe = False
        if t > 0 and t % cfg.patch_drift_every == 0:
            self.patches = (self.patches + rng.integers(-8, 9, size=self.patches.shape)) % g

        if t > 0 and t % cfg.flip_every == 0:
            self.safe = 1 - self.safe
            self.flips.append(t)
        if (self.chain_on and not cfg.force_mapping and t > self.chain_start
                and (t - self.chain_start) % cfg.prep_every == 0):
            self.new_recipe(t)
            changed_recipe = True
        if cfg.record != "none" and cfg.mark_decay:
            self.marks *= (1.0 - cfg.mark_decay)      # a mark fades; it does not persist forever
        # rot
        self.food &= rng.random(self.food.shape) > cfg.food_rot
        self.items[:] = -1                          # v3.10: no items, nuts or stations anywhere.
        self.nuts[:] = False                        # Their channels stay in the 60-input layout and
        self.stations[:] = -1                       # are permanently dead; nothing changes at the
                                                    # switch except which actions are available.
        # food and nuts grow in the drifting patches
        r = cfg.patch_radius
        for (py, px) in self.patches:
            k = rng.poisson(cfg.spawn_per_patch)
            for dy, dx in rng.integers(-r, r + 1, size=(k, 2)):
                y, x = (py + dy) % g, (px + dx) % g
                if not self.food[:, y, x].any():
                    # type 2 exists only once preparations are live: phase 1 is exactly v3.1's
                    # two-type flip world, which is what keeps row 1a anchored to v3.1's published
                    # range.  The third type arrives at the switch, with the preparations.
                    n_live = N_TYPES if self.chain_on else 2
                    w = cfg.type_spawn_w
                    if w is None or not self.chain_on:
                        ft = int(rng.integers(n_live))
                    else:
                        p = np.asarray(w[:n_live], dtype=float); p = p / p.sum()
                        ft = int(rng.choice(n_live, p=p))
                    self.food[ft, y, x] = True


        return changed_recipe


# --------------------------------------------------------------------------
# agent
# --------------------------------------------------------------------------

class Agent:
    """Synapse = innate weight (genome) + learned component H (starts at zero, not inherited).

        e <- lam * e + pre * post           which synapses were recently active together
        H <- H + eta * m * e                m = this agent's own energy change from consuming

    eta and lam are genes, per layer.  eta = 0 is exactly a fixed-weight agent.
    Nothing tells the agent what to learn; the only signal is its own metabolism.
    In this world m arrives when a nut is cracked -- not when the tool was made.
    """

    __slots__ = ("W1", "b1", "W2", "b2", "H1", "H2", "e1", "e2", "eta1", "eta2", "lam1", "lam2",
                 "integrity", "repair", "nav_dir", "nav_here", "energy", "y", "x", "item", "tool", "B",
                 "sym_gain",
                 "lineage", "gen", "born", "injected",
                 "attempts", "successes", "eats", "safe_eats", "cracks",
                 "since_recipe", "tool_made_t", "used_tool", "age_bins", "e2_hist", "prep_hist",
                 "sr_hist", "sr_pre")

    def __init__(self, cfg, rng, lineage, y, x, injected=False):
        h = cfg.hidden
        self.W1 = rng.normal(0, 0.3, (N_IN, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, 0.3, (h, N_ACTIONS))
        self.b2 = np.zeros(N_ACTIONS)
        scaffold_scale(self.W1, self.W2, cfg.innate_scale, cfg.n_scaffold)
        if cfg.free_scale != 1.0:
            self.W1[:, cfg.n_scaffold:] *= cfg.free_scale
            self.W2[cfg.n_scaffold:, :] *= cfg.free_scale
        self.nav_dir = float(rng.uniform(*cfg.nav_dir_init))
        self.nav_here = float(rng.uniform(*cfg.nav_here_init))
        wire_nav(self.W1, self.W2, self.nav_dir, self.nav_here, cfg.scaffold_food, cfg.scaffold_chain)
        self.H1 = np.zeros_like(self.W1); self.H2 = np.zeros_like(self.W2)
        self.e1 = np.zeros_like(self.W1); self.e2 = np.zeros_like(self.W2)
        self.eta1 = rng.uniform(0, cfg.eta_init); self.eta2 = rng.uniform(0, cfg.eta_init)
        self.lam1 = rng.uniform(0.5, 0.95); self.lam2 = rng.uniform(0.5, 0.95)
        self.integrity = np.ones(h)
        self.sym_gain = float(cfg.sym_gain_init)   # heritable read gain, starts near zero
        self.repair = rng.uniform(0.2, 0.8)
        self.energy = cfg.founder_energy
        self.y, self.x = y, x
        self.item, self.tool = -1, False
        self.B = np.zeros(N_PAIRS)          # hand-wired private memory; stays zero unless cfg.private_mem
        self.lineage, self.gen, self.born, self.injected = lineage, 0, 0, injected
        self.attempts = self.successes = 0
        self.eats = self.safe_eats = self.cracks = 0
        self.since_recipe = 0
        self.sr_hist = []        # this agent's preparation outcomes since the last remap it lived through
        self.sr_pre = None       # ok count over its last SR_W preparations BEFORE that remap
        self.tool_made_t = -1
        self.used_tool = False
        self.e2_hist = []
        self.prep_hist = []                  # outcomes of this agent's first n_attempts preparations
        self.age_bins = np.zeros((2, 2))     # [young/old, attempts/correct]

    def child(self, cfg, rng, t):
        c = Agent.__new__(Agent)
        for name in ("W1", "W2"):
            W = getattr(self, name)
            mask = rng.random(W.shape) < cfg.mut_rate
            setattr(c, name, W + mask * rng.normal(0, cfg.mut_sigma, W.shape))
        c.b1 = self.b1 + rng.normal(0, cfg.mut_sigma * 0.3, self.b1.shape)
        c.b2 = self.b2 + rng.normal(0, cfg.mut_sigma * 0.3, self.b2.shape)
        c.H1 = np.zeros_like(c.W1); c.H2 = np.zeros_like(c.W2)     # learned component is NOT inherited
        c.e1 = np.zeros_like(c.W1); c.e2 = np.zeros_like(c.W2)
        c.eta1 = float(np.clip(self.eta1 + rng.normal(0, 0.01), 0, cfg.eta_max))
        c.eta2 = float(np.clip(self.eta2 + rng.normal(0, 0.01), 0, cfg.eta_max))
        c.lam1 = float(np.clip(self.lam1 + rng.normal(0, 0.03), 0, 0.99))
        c.lam2 = float(np.clip(self.lam2 + rng.normal(0, 0.03), 0, 0.99))
        c.integrity = np.ones(cfg.hidden)
        c.repair = float(np.clip(self.repair + rng.normal(0, cfg.gene_sigma), 0, 1))
        c.nav_dir = float(np.clip(self.nav_dir + rng.normal(0, cfg.nav_sigma), 0, cfg.nav_max))
        c.nav_here = float(np.clip(self.nav_here + rng.normal(0, cfg.nav_sigma), 0, cfg.nav_max))
        c.sym_gain = float(self.sym_gain + rng.normal(0, cfg.gene_sigma))
        wire_nav(c.W1, c.W2, c.nav_dir, c.nav_here, cfg.scaffold_food, cfg.scaffold_chain)   # the instinct comes from the gene, not from mutated synapses
        c.energy = cfg.start_energy
        c.y, c.x = self.y, self.x
        c.item, c.tool = -1, False           # nothing carried is inherited
        c.B = np.zeros(N_PAIRS)              # no memory is inherited either
        # NOT c.injected: an injected agent is a fresh random genome dropped in to hold the
        # population off the floor, and its OWN events dilute every population metric.  Its
        # children are ordinary selected descendants, so the tag stops at the founder.
        c.lineage, c.gen, c.born, c.injected = self.lineage, self.gen + 1, t, False
        c.attempts = c.successes = 0
        c.eats = c.safe_eats = c.cracks = 0
        c.since_recipe = 0
        c.sr_hist, c.sr_pre = [], None
        c.tool_made_t = -1
        c.used_tool = False
        c.e2_hist = []
        c.prep_hist = []
        c.age_bins = np.zeros((2, 2))
        return c

    def act(self, obs, cfg, rng, chain_on=True):
        # All K preparations are masked while phase 2 is off, so phase 1 is EXACTLY v3.1's five
        # actions and its gate applies unchanged.  ONLY the action space changes at the switch --
        # the third food type appears then too, but it is a channel that is simply empty in phase
        # 1, not a new input.  The null is masked too, so its per-action share is 1/5 in phase 1
        # and 1/10 in phase 2, and 5/10 = 0.500 for "any preparation".
        n_av = N_ACTIONS if chain_on else PREP0        # 10 in phase 2, 5 in phase 1
        if cfg.mode == "random":
            return int(rng.integers(0, n_av))          # the behavioural null: no policy, no learning
        alive = self.integrity >= cfg.integrity_threshold
        plastic = cfg.mode == "plastic"
        W1 = self.W1 + self.H1 if plastic and cfg.plastic_layers in ("both", "W1") else self.W1
        W2 = self.W2 + self.H2 if plastic and cfg.plastic_layers in ("both", "W2") else self.W2
        h = np.tanh(obs @ W1 + self.b1) * alive
        logits = h @ W2 + self.b2 + rng.normal(0, cfg.action_noise, N_ACTIONS)
        if not chain_on:
            logits[PREP0:] = -np.inf                   # phase 1 is exactly v3.1's five actions
        a = int(np.argmax(logits))
        if plastic:
            out = np.zeros(N_ACTIONS); out[a] = 1.0
            if cfg.plastic_layers in ("both", "W1"):
                self.e1 = self.lam1 * self.e1 + np.outer(obs, h)
            if cfg.plastic_layers in ("both", "W2"):
                self.e2 = self.lam2 * self.e2 + np.outer(h, out)
        return a

    def learn(self, m, cfg):
        if cfg.mode != "plastic" or m == 0.0:
            return
        if cfg.plastic_layers in ("both", "W1"):
            self.H1 = np.clip(self.H1 + cfg.eta_scale * self.eta1 * m * self.e1, -cfg.h_max, cfg.h_max)
        if cfg.plastic_layers in ("both", "W2"):
            self.H2 = np.clip(self.H2 + cfg.eta_scale * self.eta2 * m * self.e2, -cfg.h_max, cfg.h_max)


def scaffold_scale(W1, W2, scale, n_scaffold):
    """Shrink the RANDOM weights of the instinct's own hidden units, once, at founder creation.
    v2.9b scaled the whole network; that leaves the grown learner's output layer no basis to
    read (see the notebook).  Every other hidden unit keeps full-scale random weights: they are
    that basis, and a conjunction can only be expressed by a unit responding to item AND
    station jointly.

    KNOWN DRIFT, deliberately not corrected.  This is an initialisation, not a constraint, and
    mutation regrows what it shrank: each entry gains variance mut_rate * mut_sigma^2 = 0.00225
    per generation, so the scaffold units' incidental weights go sd 0.03 -> 0.21 by generation 19
    and 0.27 by generation 32, against the free units' 0.30.  By late in a run those ten units are
    ordinary random-feature units that also carry the instinct.  It is left alone because:
      * it is the same mutation process in every condition, and every claim in the table is a
        between-condition contrast, so it cannot produce a between-condition artifact;
      * the instinct itself never degrades -- wire_nav rewrites the nav synapses from the gene at
        every birth, so only the INCIDENTAL weights of those units drift;
      * re-applying the scaling in child() would not hold the units at 0.03.  W' = (W + mut) *
        0.1 compounds to a stationary sd of 0.005 -- 6x below the initialisation it is meant to
        preserve and 63x below the free units -- which empties those units rather than keeping
        them clean, and is a larger, untested change than the one it fixes.
    The one caveat to record: conditions differ in generations reached (19-32 across the grid),
    so the drift is somewhat larger in the longer-lived ones."""
    W1[:, :n_scaffold] *= scale
    W2[:n_scaffold, :] *= scale


def wire_nav(W1, W2, nav_dir, nav_here, scaffold_food=True, scaffold_chain=True):
    """v2.9b's forager instinct: approach whatever the current stage of the chain wants ('goal')
    and approach food ('food_any'); interact when standing on either.  It says nothing about
    WHICH item, station or food type -- that is the experiment.

    nav_dir and nav_here are GENES (v3.6b).  The scaffold's synapses are written from them at
    every birth, so mutation of W1/W2 does not touch the instinct: its strength is inherited as
    a scalar and selection can raise or lower it, exactly as it does eta.  The directional
    inputs are normalised by v*(2v+1) = 10, so nav_dir buys ~10x less activation than nav_here
    at the same value; the two are separate genes for that reason."""
    if scaffold_chain:
        raise ValueError("scaffold_chain is retired in v3.10: there is no chain to scaffold. "
                         "It used to wire W2[9, INTERACT], and with INTERACT = None that "
                         "assigned the WHOLE of row 9 -- a silent landmine behind a default-True "
                         "flag.  Failing loudly instead.")
    if scaffold_food:
        for d in range(4):
            W1[APPETITE_CH * 4 + d, d] = nav_dir
            W2[d, d] = nav_here
        W1[HERE + APPETITE_CH, 8] = nav_here
        W2[8, EAT] = nav_here


GENOME = ("W1", "b1", "W2", "b2", "eta1", "eta2", "lam1", "lam2", "repair", "nav_dir",
          "nav_here", "sym_gain")


def _sv_log(agents, cfg, chain_on):
    """Standing variation, flattened for the log.  Only meaningful once preparations are live."""
    if not chain_on:
        return dict(triples=np.nan, valid=np.nan, held1=np.nan, held5=np.nan, held20=np.nan,
                    above=np.nan)
    d = standing_variation(agents, cfg)
    return dict(triples=d["n_triples"], valid=d["n_valid"], held1=d["held"][1],
                held5=d["held"][5], held20=d["held"][20], above=d["n_above"])


def snapshot(agents):
    """Genomes only: innate weights and genes.  Nothing learned (H) is saved."""
    out = []
    for a in agents:
        d = {k: (getattr(a, k).copy() if isinstance(getattr(a, k), np.ndarray) else float(getattr(a, k))) for k in GENOME}
        d["energy"] = float(a.energy)
        out.append(d)
    return out


def restore(d, cfg, rng, lineage, y, x):
    a = Agent(cfg, rng, lineage, y, x)
    for k in GENOME:
        v = d[k]
        setattr(a, k, v.copy() if isinstance(v, np.ndarray) else float(v))
    a.energy = float(d["energy"])
    wire_nav(a.W1, a.W2, a.nav_dir, a.nav_here, cfg.scaffold_food, cfg.scaffold_chain)   # cfg.scaffold_food, or a replay of a
    return a                                                          # food-unscaffolded population gets the
                                                                      # food instinct wired back in


# --------------------------------------------------------------------------
# probes (within-agent counterfactuals)
# --------------------------------------------------------------------------

def _forward(a, cfg, obs, learned):
    plastic = cfg.mode == "plastic"
    W1 = a.W1 + a.H1 if (learned and plastic and cfg.plastic_layers in ("both", "W1")) else a.W1
    W2 = a.W2 + a.H2 if (learned and plastic and cfg.plastic_layers in ("both", "W2")) else a.W2
    alive = a.integrity >= cfg.integrity_threshold
    h = np.tanh(obs @ W1 + a.b1) * alive
    return h @ W2 + a.b2


def prep_pref(a, cfg, ftype, k, learned=True):
    """'Food of type f is under me.'  Returns the logit of preparation k minus the best of the
    other actions -- how much this agent wants to apply preparation k to this food type."""
    obs = np.zeros(N_IN)
    obs[HERE + ftype] = 1.0
    obs[ENERGY] = 0.5
    logits = _forward(a, cfg, obs, learned)
    j = PREP0 + k
    return float(logits[j] - np.delete(logits, j).max())


def prep_gain(agents, cfg, mapping, learned=True, n_sample=40):
    """Preference for the CORRECT preparation minus the mean of the other two, averaged over both
    food types and over agents.  > 0 means the policy applies the right preparation to each type --
    which requires discriminating type, so this cannot be scored by a type-blind policy."""
    if len(agents) < 5:
        return np.nan
    idx = np.linspace(0, len(agents) - 1, min(n_sample, len(agents))).astype(int)
    vals = []
    for i in idx:
        a = agents[i]
        for ft in range(N_TYPES):
            p = np.array([prep_pref(a, cfg, ft, k, learned) for k in range(N_PREPS)])
            vals.append(p[mapping[ft]] - np.delete(p, mapping[ft]).mean())
    return float(np.mean(vals))


def innate_mapping(a, cfg):
    """The mapping this agent's GENOME would apply: argmax preparation per food type, from the
    innate probe (learned=False), so nothing the agent has learned enters it."""
    return tuple(int(np.argmax([prep_pref(a, cfg, ft, k, False) for k in range(N_PREPS)]))
                 for ft in range(N_TYPES))


def standing_variation(agents, cfg, n_sample=400, thresholds=(1, 5, 20)):
    """D2 -- the probe that makes the v3.12 premise CHECKABLE rather than argued.

    The premise is that P(K,T) = 60 mappings exceeds what a population can hold in standing
    variation, so a remap usually finds no matching genotype to promote and the only route to the
    new mapping is within a life.  That is an empirical claim about this population, and this
    measures it directly.

    Returns:
      n_triples    distinct innate argmax triples present at all
      n_valid      of those, how many are VALID mappings (T distinct preparations) -- a triple
                   with a repeat is a genome that has not separated the types
      held[thr]    valid mappings carried by at least `thr` living agents
      n_above      how many of the P(K,T) mappings the population would score ABOVE the
                   type-blind level 1/T on, if that mapping were the one in force.  THIS is the
                   direct statement of the premise: a handful of 60 means the space exceeds
                   standing variation; a large number means v3.12 has not achieved what it was
                   built for, and that is the finding whatever row 3b then says.
    """
    if not agents:
        return dict(n_triples=0, n_valid=0, held={t: 0 for t in thresholds}, n_above=0, n=0)
    idx = (np.arange(len(agents)) if len(agents) <= n_sample
           else np.random.default_rng(0).choice(len(agents), n_sample, replace=False))
    tri = [innate_mapping(agents[int(i)], cfg) for i in idx]
    from collections import Counter
    c = Counter(tri)
    valid = {m: k for m, k in c.items() if len(set(m)) == N_TYPES}
    held = {t: int(sum(1 for k in valid.values() if k >= t)) for t in thresholds}
    n = len(tri)
    above = 0
    for m in all_mappings():
        # expected hit if THIS were the mapping: mean over agents of the fraction of types the
        # agent's innate argmax gets right
        score = sum(sum(1 for t_ in range(N_TYPES) if tr[t_] == m[t_]) for tr in tri) / (n * N_TYPES)
        if score > 1.0 / N_TYPES:
            above += 1
    return dict(n_triples=len(c), n_valid=len(valid), held=held, n_above=above, n=n)


def _obs_with_mark(ftype, label, sign):
    """A synthetic observation: food type `ftype` underfoot, one mark present at `label` with
    `sign`, nothing else.  No behaviour, no history."""
    obs = np.zeros(N_IN)
    obs[HERE + ftype] = 1.0
    obs[ENERGY] = 0.5
    if label is not None:
        obs[READ + label] = float(sign)
    return obs


def read_pref(a, cfg, ftype, label, sign, k, learned=True):
    """Food type `ftype` underfoot and a mark of `sign` at `label`: how much does this agent want
    preparation `k`?  The READING side of the binding, within-agent."""
    obs = _obs_with_mark(ftype, label, sign)
    logits = _forward(a, cfg, obs, learned)
    j = PREP0 + k
    return float(logits[j] - np.delete(logits, j).max())


def store_gain(agents, cfg, pi, learned=True, n_sample=40, sign=1.0):
    """How much more an agent prefers the preparation a POSITIVE mark endorses than the others,
    evaluated under the era's actual pi.  A mark at label pi[k] endorses preparation k.

    learned vs innate is the binding instrument, exactly as prep_gain innate vs learned is:
    innate ~ 0 says the genome cannot read the record (which redrawing pi guarantees), learned > 0
    says this agent bound it inside its own life.  The gain is measured against the SAME agent with
    the mark absent, so it isolates what the mark adds rather than the agent's standing preference.
    """
    if not agents:
        return np.nan
    idx = (np.arange(len(agents)) if len(agents) <= n_sample
           else np.random.default_rng(0).choice(len(agents), n_sample, replace=False))
    vals = []
    for i in idx:
        a = agents[int(i)]
        for ft in range(N_TYPES):
            for k in range(N_PREPS):
                lab = pi[k]
                with_mark = read_pref(a, cfg, ft, lab, sign, k, learned)
                without = read_pref(a, cfg, ft, None, 0.0, k, learned)
                vals.append(with_mark - without)
    return float(np.mean(vals)) if vals else np.nan


def mark_stats(world, cfg):
    """Density and age of the store, so DECISION 2's decay constant is measured, not assumed."""
    if cfg.record == "none":
        return dict(mark_density=np.nan, mark_mean_abs=np.nan)
    nz = np.abs(world.marks) > 1e-3
    return dict(mark_density=float(nz.mean()),
                mark_mean_abs=float(np.abs(world.marks)[nz].mean()) if nz.any() else 0.0)


def gate_r_counts(world, cfg):
    """Gate R's raw material: the joint count of (label, correct preparation) under the CURRENT
    mapping and pi.  Pooled ACROSS eras by the reader, where it must be at or below the noise
    arm's level -- within an era it is maximal by construction, which is the design, not a fault."""
    if cfg.record == "none":
        return None
    # label j endorses preparation pi^-1(j); the correct preparation for type t is mapping[t]
    inv = {lab: k for k, lab in enumerate(world.pi)}
    return dict(pi=tuple(world.pi), mapping=tuple(world.mapping),
                endorsed=tuple(inv[j] for j in range(N_PREPS)))


def probe_advantage(agents, cfg, mapping, n_sample=40):
    """Within-agent counterfactual on the PREPARATION conjunction: prep_gain with learned synapses
    minus with innate ones.  Same genome, same world state.  Exactly zero when mode == 'fixed'."""
    if cfg.mode != "plastic" or len(agents) < 5:
        return np.nan
    return (prep_gain(agents, cfg, mapping, True, n_sample)
            - prep_gain(agents, cfg, mapping, False, n_sample))


def food_pref(a, cfg, ftype, learned=True):
    obs = np.zeros(N_IN)
    obs[HERE + ftype] = 1.0
    obs[HERE + APPETITE_CH] = 1.0
    obs[ENERGY] = 0.5
    logits = _forward(a, cfg, obs, learned)
    # v3.10: the v3.1 form was logit(eat) minus the BEST OTHER action.  With three preparations
    # live, that term moves with food type (the correct prep differs by type), so it contaminated
    # the food probe -- it read 0.04 where the safe rate said 0.60.  The moves are the only
    # food-type-independent baseline, so the probe is now logit(eat) against them.
    return float(logits[EAT] - logits[:4].mean())


def food_gain(agents, cfg, safe, learned=True, n_sample=40):
    if len(agents) < 5:
        return np.nan
    idx = np.linspace(0, len(agents) - 1, min(n_sample, len(agents))).astype(int)
    return float(np.mean([food_pref(agents[k], cfg, safe, learned) - food_pref(agents[k], cfg, 1 - safe, learned) for k in idx]))


def probe_advantage_food(agents, cfg, safe, n_sample=40):
    """The v3.1 probe, unchanged in substance: the positive control.  If this is positive and the
    recipe probe is not, the learner works in this world and the failure is specific to the
    conjunctive delayed-credit fact."""
    if cfg.mode != "plastic" or len(agents) < 5:
        return np.nan
    return food_gain(agents, cfg, safe, True, n_sample) - food_gain(agents, cfg, safe, False, n_sample)


# --------------------------------------------------------------------------
# the action x cell table -- the ONLY place an action changes the world, an agent's
# energy or the modulator.  run() calls it; world_semantics_selftest() calls it with
# constructed cells and checks every row.  One function, so the test tests the real path.
# --------------------------------------------------------------------------

def resolve_action(a, action, world, cfg, rng, t=0):
    """Apply `action` for agent `a`.  Mutates a.energy and the world.  Returns (m, event, info).

    Modulator events, complete:
        eat safe food            m = +1   energy +food_value
        eat poison               m = -1   energy -poison_value
        correct preparation      m = +1   energy +prep_value
        wrong preparation        m = -1   energy -prep_fail
        everything else          m =  0   move / no-op -move_cost, base metabolism, repair
    Nothing on food is silent: every action on a food cell resolves immediately and two-sidedly.
    The modulator is NOT the agent's own energy change in general; it is this table.

    A preparation consumes the food cell, exactly as eating does.  The prep outcome depends on the
    food TYPE and the preparation chosen, and is independent of the safe/poison flip -- the world
    holds two independent facts, a fast one (which type is safe, every 300 steps) and a slow one
    (which preparation goes with which type, every 2000).
    """
    y, x = a.y, a.x
    g = cfg.grid

    if action < 4:
        a.y, a.x = (y + MOVES[action][0]) % g, (x + MOVES[action][1]) % g
        a.energy -= cfg.move_cost
        return 0.0, "move", {}

    has_food = bool(world.food[:, y, x].any())
    if not has_food:
        a.energy -= cfg.noop_cost
        return 0.0, "noop", {}
    ftype = int(np.argmax(world.food[:, y, x]))

    if action == EAT:
        if ftype >= 2:
            # INEDIBLE RAW.  Energy change exactly 0, m = 0, and the cell is NOT consumed -- if
            # eating destroyed it, a naive agent could clear the board of precisely the food this
            # experiment is about.  The step is wasted and nothing else happens.
            return 0.0, "eat_inedible", {"ftype": ftype}
        world.food[ftype, y, x] = False
        if ftype == world.safe:
            a.energy += cfg.food_value
            return 1.0, "eat_safe", {"ftype": ftype}
        a.energy -= cfg.poison_value
        return -1.0, "eat_poison", {"ftype": ftype}

    k = action - PREP0                                  # 0 .. N_PREPS-1
    world.food[ftype, y, x] = False
    ok = (k == world.mapping[ftype])
    info = {"ftype": ftype, "prep": k, "ok": ok}
    if ok:
        a.energy += cfg.prep_value
        return 1.0, "prep_ok", info
    a.energy -= cfg.prep_fail
    return -1.0, "prep_bad", info


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def run(cfg, verbose=True, init_genomes=None, phases=None):
    """phases: a list of {"n_steps": int, "chain": bool} run back to back on ONE population.
    The agents list, their H, their eligibility traces and the world are all carried across a
    phase boundary untouched -- only cfg.chain flips.  phases=None runs a single phase of
    cfg.n_steps at cfg.chain, which is what every earlier notebook did."""
    if phases is None:
        phases = [dict(n_steps=cfg.n_steps, chain=cfg.chain)]
    schedule, bounds, acc = [], [], 0
    for p in phases:
        acc += p["n_steps"]
        schedule.append((acc, bool(p.get("chain", cfg.chain))))
        bounds.append((acc - p["n_steps"], acc))
    total_steps = acc

    rng = np.random.default_rng(cfg.seed)
    g, v = cfg.grid, cfg.view
    win = 2 * v + 1
    world = World(cfg, rng)
    world.chain_on = schedule[0][1]
    for _ in range(100):
        world.step(-1)
    if init_genomes is None:
        agents = [Agent(cfg, rng, i, *rng.integers(0, g, 2)) for i in range(cfg.init_pop)]
    else:
        agents = [restore(d, cfg, rng, i, *rng.integers(0, g, 2)) for i, d in enumerate(init_genomes)]
    next_lineage = len(agents)

    log = []
    W = dict(eats=0, safe=0, deaths=0, births=0, steps=0, inject=0, attempts=0, correct=0,
             nuts=0, pickups=0, e_food=0.0, e_nut=0.0, e_fail=0.0, bridge_sum=0.0, bridge_n=0,
             trace_w=0.0, bridge1_sum=0.0, bridge1_n=0, rec_sum=0.0, rec_n=0,
             e_bonus=0.0, noops=0, raw_meals=0, prep_meals=0,
             on_food=0, on_food_eat=0, on_food_prep=0,
             **{f"prep_n{i}": 0 for i in range(N_TYPES)},
             **{f"prep_ok{i}": 0 for i in range(N_TYPES)},
             first_n=0, first_ok=0, surv_n=0, surv_early=0, surv_late=0,
             srm_n=0, srm_pre=0, srm_post=0, first_n_late=0, first_ok_late=0)
    # FOUNDER-FREE mirror.  Injected agents are fresh random genomes; their own events dilute
    # every event-weighted metric toward chance, and the dilution is heaviest in exactly the arms
    # that need injecting -- so a non-learning arm reads as MORE random the worse it does.  WF
    # accumulates the same events with injected agents' own events excluded.  Their children are
    # not tagged, so descent back into the population is counted from the first generation.
    WF = {k: 0 for k in ("eats", "safe", "attempts", "correct", "first_n", "first_ok",
                         *[f"prep_n{i}" for i in range(N_TYPES)],
                         *[f"prep_ok{i}" for i in range(N_TYPES)],
                         "surv_n", "surv_early", "surv_late", "srm_n", "srm_pre", "srm_post",
                         "first_n_late", "first_ok_late")}
    ATT = np.zeros((cfg.n_attempts + 1, 2))     # attempt number in an agent's life -> (n, correct)
    ATT_R = np.zeros((cfg.n_attempts + 1, 2))   # attempts since the last recipe change
    ATT_F = np.zeros((cfg.n_attempts + 1, 2))   # ... both, founder-free
    ATT_R_F = np.zeros((cfg.n_attempts + 1, 2))
    MEALS = np.zeros((cfg.n_meals + 1, 2))      # meal number in an agent's life (the v3 curve)
    # frozen replay: each agent's energy is pinned to its snapshot value, keyed by identity
    # because rng.shuffle reorders the list every step.
    _frozen_e = {id(a): float(a.energy) for a in agents} if cfg.frozen else {}
    era_snaps = []                                   # rolling era-boundary genome snapshots
    prev_mapping = tuple(int(x) for x in world.mapping)
    t0 = time.time()

    track_recency = (cfg.trace_recency and cfg.mode == "plastic"
                     and cfg.plastic_layers in ("both", "W2"))
    phase_i = 0
    for t in range(total_steps):
        while t >= schedule[phase_i][0]:                 # phase boundary: nothing is rebuilt or reset
            phase_i += 1
            on = schedule[phase_i][1]
            if on and not world.chain_on:
                world.chain_start = t                    # recipe eras start when the chain does
                world.recipe_changes.append(-t)          # negative marks 'chain on', not a change
            world.chain_on = on
            for a in agents:
                a.e2_hist = []        # the ring is phase-1 only; stale entries would be nonsense
        if world.step(t):
            # ERA BOUNDARY.  Snapshot the genomes HERE, not at the end of the run: a population
            # sampled mid-era has only partly sorted to the mapping in force, which is what made
            # the run-end knockout unreadable (v3.11 `fixed` seed 0 scored 0.24 on its own
            # mapping).  At a boundary the population has just lived a whole era under
            # `prev_mapping`, so that is the mapping it is sorted for.
            if not cfg.frozen and world.chain_on:
                samp = agents if len(agents) <= cfg.era_snap_max else [
                    agents[i] for i in rng.choice(len(agents), cfg.era_snap_max, replace=False)]
                era_snaps.append(dict(t=t, mapping=prev_mapping, n_pop=len(agents),
                                      genomes=snapshot(samp)))
                del era_snaps[:-cfg.era_snap_keep]
            prev_mapping = tuple(int(x) for x in world.mapping)
            for a in agents:
                a.since_recipe = 0
                # SURVIVOR-CONDITIONED SINCE-REMAP: an agent qualifies only if it made SR_W
                # preparations BEFORE this remap and goes on to make SR_W after it.  Both halves
                # then come from the SAME agent across the SAME remap, so a change cannot be a
                # different sample -- which is what the population-level since-remap curve is open
                # to, since the agents alive at preparation 1 are not those alive at 10.
                a.sr_pre = (sum(a.sr_hist[-SR_W:]) if len(a.sr_hist) >= SR_W else None)
                a.sr_hist = []
        occ = np.zeros((g, g))
        for a in agents:
            occ[a.y, a.x] += 1
        pad = lambda arr: np.pad(np.asarray(arr, dtype=float), v, mode="wrap")
        pF = [pad(world.food[i]) for i in range(N_TYPES)]
        pO = pad(np.minimum(occ, 3) / 3.0)
        # the food_any channel exists only to drive the instinct; with no food scaffold it is dead,
        # which makes phase 1's live inputs exactly v3.1's set (4 dirsums x 3 channels + 3 here + energy)
        pFood = sum(pF) if cfg.scaffold_food else np.zeros_like(pF[0])
        rng.shuffle(agents)
        survivors, newborns = [], []

        for a in agents:
            y, x = a.y, a.x
            sl = (slice(y, y + win), slice(x, x + win))
            # the food type under the agent, needed BEFORE act() so the read channels can be built
            ft_here_pre = int(np.argmax(world.food[:, y, x])) if world.food[:, y, x].any() else -1

            chans = [p[sl] for p in pF] + [pO[sl], pFood[sl]]
            # the K read channels: the marks under this agent, for the food type under it, scaled
            # by its own heritable gain.  No food underfoot -> zeros.  No record -> zeros.
            if cfg.record != "none" and ft_here_pre >= 0:
                read = a.sym_gain * world.marks[ft_here_pre, :, y, x]
            else:
                read = np.zeros(N_PREPS)
            obs = np.concatenate([dirsum(c, v) for c in chans]
                                 + [[c[v, v] for c in chans], [a.energy / cfg.max_energy], read])
            action = a.act(obs, cfg, rng, world.chain_on)
            if track_recency and not world.chain_on:
                a.e2_hist.append(a.e2.copy())          # phase 1 only: the chain would confound it
                if len(a.e2_hist) > cfg.recency_k + 1:
                    a.e2_hist.pop(0)

            a.integrity = np.minimum(1.0, a.integrity - cfg.decay + a.repair * cfg.repair_gain)
            a.energy -= cfg.base_cost + a.repair * cfg.repair_cost * cfg.hidden

            on_food = bool(world.food[:, y, x].any())
            ft_here = int(np.argmax(world.food[:, y, x])) if on_food else -1

            # ---- the B ceiling forces its argmax preparation once it has evidence for this type.
            # There is no navigation to steer here, so a hand-wired POLICY OVERRIDE is the honest
            # analogue of v2's veto.  A reference level, not a matched comparison.
            if cfg.private_mem and on_food and world.chain_on:
                row = a.B[ft_here * N_PREPS:(ft_here + 1) * N_PREPS]
                if float(np.max(np.abs(row))) > 0.2:
                    action = PREP0 + int(np.argmax(row))

            m, event, info = resolve_action(a, action, world, cfg, rng, t)

            if on_food:
                W["on_food"] += 1
                W["on_food_eat"] += (action == EAT)
                W["on_food_prep"] += (action >= PREP0)

            fnd = not a.injected                 # this agent's own events count founder-free
            if event in ("eat_safe", "eat_poison"):
                W["eats"] += 1; a.eats += 1
                W["raw_meals"] += 1
                if fnd:
                    WF["eats"] += 1
                if event == "eat_safe":
                    W["safe"] += 1; W["e_food"] += cfg.food_value; a.safe_eats += 1
                    if fnd:
                        WF["safe"] += 1
                k = a.eats
                if k <= cfg.n_meals:
                    MEALS[k, 0] += 1; MEALS[k, 1] += (event == "eat_safe")
                if track_recency and not world.chain_on and len(a.e2_hist) == cfg.recency_k + 1:
                    tot = float(np.abs(a.e2).sum())
                    if tot > 0:
                        old_ = float(np.abs((a.lam2 ** cfg.recency_k) * a.e2_hist[0]).sum())
                        W["rec_sum"] += 1.0 - old_ / tot; W["rec_n"] += 1
            elif event in ("prep_ok", "prep_bad"):
                ok, ft = info["ok"], info["ftype"]
                a.attempts += 1; W["attempts"] += 1; W["prep_meals"] += 1
                W[f"prep_n{ft}"] += 1; W[f"prep_ok{ft}"] += ok
                if fnd:
                    WF["attempts"] += 1; WF["correct"] += ok
                    WF[f"prep_n{ft}"] += 1; WF[f"prep_ok{ft}"] += ok
                if ok:
                    W["correct"] += 1; a.successes += 1; W["e_bonus"] += cfg.prep_value
                else:
                    W["e_fail"] += cfg.prep_fail
                world.write_mark(ft, info["prep"], bool(ok), y, x, cfg)   # AUTOMATIC, costless
                if cfg.private_mem:
                    j = ft * N_PREPS + info["prep"]
                    a.B[j] = 0.7 * a.B[j] + 0.3 * (1.0 if ok else -1.0)   # exact credit, hand-wired
                old = 0 if (t - a.born) < 150 else 1
                a.age_bins[old, 0] += 1; a.age_bins[old, 1] += ok
                k = a.attempts
                if k <= cfg.n_attempts:
                    ATT[k, 0] += 1; ATT[k, 1] += ok
                    if fnd:
                        ATT_F[k, 0] += 1; ATT_F[k, 1] += ok
                if k == 1:
                    W["first_n"] += 1; W["first_ok"] += ok      # first-preparation hit
                    if fnd:
                        WF["first_n"] += 1; WF["first_ok"] += ok
                    # ... split by WHERE IN THE ERA it fell.  A first preparation made just after
                    # a remap is scored against a mapping the agent's lineage has not been
                    # selected on yet, so the raw first-prep hit conflates the genome's quality
                    # with how recently the mapping moved.  `late` is the genome reading.
                    if (t - world.last_remap) >= cfg.prep_every // 3:
                        W["first_n_late"] += 1; W["first_ok_late"] += ok
                        if fnd:
                            WF["first_n_late"] += 1; WF["first_ok_late"] += ok
                if len(a.prep_hist) < cfg.n_attempts:
                    a.prep_hist.append(bool(ok))
                    if len(a.prep_hist) == cfg.n_attempts:
                        # SURVIVOR CURVE: this agent reached 10 preparations.  Conditioning on that
                        # removes the survivorship that inflates a population-level hit rate --
                        # every agent counted here contributes both halves of its own curve.
                        # SURVIVOR CURVE halves are preparations 1-2 against 6-10: the first
                        # two are before within-life learning could have taken hold, so this is
                        # the agent's own naive rate against its own settled rate.
                        W["surv_n"] += 1
                        W["surv_early"] += sum(a.prep_hist[:SURV_EARLY])
                        W["surv_late"] += sum(a.prep_hist[5:])
                        if fnd:
                            WF["surv_n"] += 1
                            WF["surv_early"] += sum(a.prep_hist[:SURV_EARLY])
                            WF["surv_late"] += sum(a.prep_hist[5:])
                a.sr_hist.append(bool(ok))
                if a.sr_pre is not None and len(a.sr_hist) == SR_W:
                    W["srm_n"] += 1
                    W["srm_pre"] += a.sr_pre; W["srm_post"] += sum(a.sr_hist)
                    if fnd:
                        WF["srm_n"] += 1
                        WF["srm_pre"] += a.sr_pre; WF["srm_post"] += sum(a.sr_hist)
                    a.sr_pre = None                 # record once per remap per agent
                a.since_recipe += 1
                kr = a.since_recipe
                if kr <= cfg.n_attempts:
                    ATT_R[kr, 0] += 1; ATT_R[kr, 1] += ok
                    if fnd:
                        ATT_R_F[kr, 0] += 1; ATT_R_F[kr, 1] += ok
            elif event == "noop":
                W["noops"] += 1

            if cfg.scramble and m != 0.0:
                m = 1.0 if rng.random() < 0.5 else -1.0      # same magnitude, no information
            a.learn(m, cfg)

            a.energy = min(a.energy, cfg.max_energy)
            if cfg.frozen:
                # NOTHING CAN CHANGE BUT H -- and energy is an OBSERVATION CHANNEL
                # (obs[ENERGY] = energy / max_energy).  With death disabled it drifts unbounded,
                # so the policy drifts with it and a learning-OFF replay moves on its own.
                # Measured: mean energy 2.470 -> 1.303 over 300 steps while the hit fell
                # 0.481 -> 0.410, with pop and max_gen constant and no record.  Holding each
                # agent's energy at its snapshot value is what makes the claim literally true.
                # The metabolism still runs -- costs are computed -- it simply cannot leak into
                # the observation.
                a.energy = _frozen_e[id(a)]
                survivors.append(a)
                continue
            if a.energy <= 0:
                W["deaths"] += 1
                continue
            if a.energy >= cfg.repro_threshold and len(agents) + len(newborns) < cfg.max_pop:
                newborns.append(a.child(cfg, rng, t))
                a.energy -= cfg.repro_cost
                W["births"] += 1
            survivors.append(a)

        agents = survivors + newborns
        W["steps"] += len(agents)
        if not cfg.frozen:
            while len(agents) < cfg.min_pop:
                agents.append(Agent(cfg, rng, next_lineage, *rng.integers(0, g, 2), injected=True))
                next_lineage += 1
                W["inject"] += 1

        if (t + 1) % cfg.log_every == 0:
            young = np.sum([a.age_bins[0] for a in agents], axis=0)
            old = np.sum([a.age_bins[1] for a in agents], axis=0)
            income = W["e_food"] + W["e_nut"]
            log.append(dict(
                t=t + 1, phase=phase_i, chain=int(world.chain_on), pop=len(agents),
                recipe_hit=(W["correct"] / W["attempts"]) if W["attempts"] else np.nan,   # prep hit
                attempts_per_1k=1000.0 * W["attempts"] / max(W["steps"], 1),
                attempts_per_life=W["attempts"] / max(W["deaths"], 1),   # births ~ deaths in steady state
                nuts_per_1k=1000.0 * W["nuts"] / max(W["steps"], 1),
                hit_young=(young[1] / young[0]) if young[0] > 20 else np.nan,   # attempts in the first 150 steps of life
                hit_old=(old[1] / old[0]) if old[0] > 20 else np.nan,           # ... and after.  Within-life learning = old > young
                safe_rate=(W["safe"] / W["eats"]) if W["eats"] else np.nan,
                energy=float(np.mean([a.energy for a in agents])),
                eta1=float(np.mean([a.eta1 for a in agents])), eta2=float(np.mean([a.eta2 for a in agents])),
                lam1=float(np.mean([a.lam1 for a in agents])), lam2=float(np.mean([a.lam2 for a in agents])),
                h_norm=float(np.mean([np.abs(a.H1).mean() + np.abs(a.H2).mean() for a in agents])),
                repair=float(np.mean([a.repair for a in agents])),
                nav_dir=float(np.mean([a.nav_dir for a in agents])),
                nav_here=float(np.mean([a.nav_here for a in agents])),
                max_gen=max(a.gen for a in agents),
                deaths=W["deaths"], births=W["births"], injections=W["inject"],
                # raw counts, so second-half aggregates can be event-weighted rather than
                # means of per-window ratios (a mean-of-ratios artifact cost a retraction in v2)
                n_attempts_raw=W["attempts"], n_correct=W["correct"], n_eats=W["eats"],
                n_safe=W["safe"], n_nuts=W["nuts"], agent_steps=W["steps"],
                probe_adv=probe_advantage(agents, cfg, world.mapping),
                prep_gain=prep_gain(agents, cfg, world.mapping, True),
                prep_gain_innate=prep_gain(agents, cfg, world.mapping, False),
                **{f"sv_{k}": v_ for k, v_ in _sv_log(agents, cfg, world.chain_on).items()},
                probe_adv_food=probe_advantage_food(agents, cfg, world.safe),
                food_gain_innate=food_gain(agents, cfg, world.safe, False),
                crop_safe=float(world.food[world.safe].sum() / max(1, world.food[:2].sum())),
                food_share=[float(world.food[i].sum() / max(1, world.food.sum()))
                            for i in range(N_TYPES)],
                e_fail_per_1k=1000.0 * W["e_fail"] / max(W["steps"], 1),
                                                                          # attempts, so the
                                                                          # fail-cost arm's world
                                                                          # change is measured
                trace_recency=(W["rec_sum"] / W["rec_n"]) if W["rec_n"] else np.nan,
                eat_on_food=(W["on_food_eat"] / W["on_food"]) if W["on_food"] else np.nan,
                prep_on_food=(W["on_food_prep"] / W["on_food"]) if W["on_food"] else np.nan,
                prep_share=(W["prep_meals"] / (W["prep_meals"] + W["raw_meals"]))
                           if (W["prep_meals"] + W["raw_meals"]) else np.nan,
                n_prep=W["prep_meals"], n_raw=W["raw_meals"],
                **{f"n_prep{i}": W[f"prep_n{i}"] for i in range(N_TYPES)},
                **{f"n_ok{i}": W[f"prep_ok{i}"] for i in range(N_TYPES)},
                mapping=tuple(int(x) for x in world.mapping),   # T entries, not two
                prep_per_life=W["prep_meals"] / max(W["deaths"], 1),
                n_first=W["first_n"], n_first_ok=W["first_ok"],
                n_first_late=W["first_n_late"], n_first_ok_late=W["first_ok_late"],
                n_surv=W["surv_n"], n_surv_early=W["surv_early"], n_surv_late=W["surv_late"],
                n_srm=W["srm_n"], n_srm_pre=W["srm_pre"], n_srm_post=W["srm_post"],
                noops_per_1k=1000.0 * W["noops"] / max(W["steps"], 1),
                e_bonus_per_1k=1000.0 * W["e_bonus"] / max(W["steps"], 1),     # share of energy income from nuts
                att_n=ATT[1:, 0].tolist(), att_correct=ATT[1:, 1].tolist(),
                rec_n=ATT_R[1:, 0].tolist(), rec_correct=ATT_R[1:, 1].tolist(),
                # founder-free: the same events with injected agents' own events excluded
                f_n_eats=WF["eats"], f_n_safe=WF["safe"],
                f_n_attempts_raw=WF["attempts"], f_n_correct=WF["correct"],
                **{f"f_n_prep{i}": WF[f"prep_n{i}"] for i in range(N_TYPES)},
                **{f"f_n_ok{i}": WF[f"prep_ok{i}"] for i in range(N_TYPES)},
                f_n_first=WF["first_n"], f_n_first_ok=WF["first_ok"],
                f_n_first_late=WF["first_n_late"], f_n_first_ok_late=WF["first_ok_late"],
                f_n_surv=WF["surv_n"], f_n_surv_early=WF["surv_early"],
                f_n_surv_late=WF["surv_late"],
                f_n_srm=WF["srm_n"], f_n_srm_pre=WF["srm_pre"], f_n_srm_post=WF["srm_post"],
                f_att_n=ATT_F[1:, 0].tolist(), f_att_correct=ATT_F[1:, 1].tolist(),
                f_rec_n=ATT_R_F[1:, 0].tolist(), f_rec_correct=ATT_R_F[1:, 1].tolist(),
                n_founder_prep=W["attempts"] - WF["attempts"],
                n_founder_eats=W["eats"] - WF["eats"],
                meal_n=MEALS[1:, 0].tolist(), meal_safe=MEALS[1:, 1].tolist(),
            ))
            ATT[:] = 0; ATT_R[:] = 0; MEALS[:] = 0; ATT_F[:] = 0; ATT_R_F[:] = 0
            for k in W:
                W[k] = 0 if isinstance(W[k], int) else 0.0
            for k in WF:
                WF[k] = 0
            if verbose and (t + 1) % (cfg.log_every * 20) == 0:
                L = log[-1]
                print(f"t={L['t']:5d} pop={L['pop']:3d} prep_hit={L['recipe_hit']:.3f} "
                      f"prep/life={L['prep_per_life']:.1f} share={L['prep_share']:.2f} "
                      f"safe={L['safe_rate']:.2f} eta2={L['eta2']:.3f} [{time.time()-t0:.0f}s]", flush=True)
    return dict(log=log, flips=world.flips,
                recipe_changes=[c for c in world.recipe_changes if c >= 0],
                chain_start=world.chain_start, phase_bounds=bounds, n_steps=total_steps,
                # the mapping the surviving genomes were last selected under.  A replay that
                # re-seeds the world does NOT get this mapping, so a genome test has to pin it.
                final_mapping=tuple(int(x) for x in world.mapping),
                era_snaps=era_snaps,     # genomes at era boundaries, with the mapping just lived
                cfg=asdict(cfg), final=snapshot(agents))


def record_semantics_selftest(verbose=True):
    """THE v3.13 SEMANTICS TEST, built before the world it tests is trusted.

    Nine claims, each checked rather than asserted:
      1. writing is AUTOMATIC -- every preparation writes, and nothing else does;
      2. the mark lands at label pi[k] with the sign of the outcome, for the FOOD TYPE prepared;
      3. it lands on the CELL the preparation happened on, and nowhere else;
      4. `noise` writes a mark of the same sign at a RANDOM label -- density and sign preserved,
         the label->preparation association destroyed;
      5. `none` writes nothing at all;
      6. pi is REDRAWN at every remap, and a redraw generally changes it;
      7. marks DECAY, and the half-life sits inside the era and outside a lifetime;
      8. reading is an OBSERVATION -- the K read channels carry the marks for the food underfoot,
         scaled by sym_gain, and are zero with no food, no record, or zero gain;
      9. sym_gain is HERITABLE and starts near zero.
    """
    fails, n = [], 0
    g = Config().grid

    def fresh(record="real", food=0, mapping=(0, 1, 2), pi=(0, 1, 2, 3, 4)):
        cfg = Config(mode="fixed", scaffold_food=False, record=record)
        w = World(cfg, np.random.default_rng(0))
        w.food[:] = False; w.marks[:] = 0.0
        w.mapping, w.pi, w.chain_on = mapping, pi, True
        w.food[food, 5, 5] = True
        a = Agent(cfg, np.random.default_rng(1), 0, 5, 5); a.energy = 3.0
        return a, w, cfg

    # 1-3: automatic, correct label/sign/type, correct cell
    for ft in range(N_TYPES):
        for k in range(N_PREPS):
            a, w, cfg = fresh(food=ft)
            m, ev, info = resolve_action(a, PREP0 + k, w, cfg, np.random.default_rng(0))
            w.write_mark(ft, k, bool(info["ok"]), 5, 5, cfg)
            lab, want = w.pi[k], (1.0 if k == w.mapping[ft] else -1.0)
            n += 1
            if w.marks[ft, lab, 5, 5] != want:
                fails.append(f"write: type {ft} prep {k} -> label {lab} expected {want:+.0f}, "
                             f"got {w.marks[ft, lab, 5, 5]:+.2f}")
            other = np.abs(w.marks).sum() - abs(w.marks[ft, lab, 5, 5])
            n += 1
            if other > 1e-9:
                fails.append(f"write: type {ft} prep {k} wrote {other:.3f} outside its own slot")
    # eating must NOT write
    for ft in (0, 1):
        a, w, cfg = fresh(food=ft)
        resolve_action(a, EAT, w, cfg, np.random.default_rng(0))
        n += 1
        if np.abs(w.marks).sum() > 1e-9:
            fails.append(f"eat on type {ft} wrote a mark; only preparations write")
    # 4: noise preserves sign and density, destroys the label association
    a, w, cfg = fresh(record="noise", food=0)
    labs = set()
    for _ in range(200):
        w.marks[:] = 0.0
        w.write_mark(0, 0, True, 5, 5, cfg)
        hit = np.nonzero(w.marks[0, :, 5, 5])[0]
        n += 1
        if len(hit) != 1 or w.marks[0, hit[0], 5, 5] != 1.0:
            fails.append("noise: wrote no mark, or the wrong sign"); break
        labs.add(int(hit[0]))
    n += 1
    if len(labs) < N_PREPS - 1:
        fails.append(f"noise: labels not randomised -- only {sorted(labs)} in 200 writes")
    # 5: none writes nothing
    a, w, cfg = fresh(record="none", food=0)
    w.write_mark(0, 0, True, 5, 5, cfg)
    n += 1
    if np.abs(w.marks).sum() > 1e-9:
        fails.append("record='none' wrote a mark")
    # 6: pi is redrawn at each remap and generally changes
    cfg = Config(mode="fixed", record="real", prep_every=100)
    w = World(cfg, np.random.default_rng(3)); w.chain_on = True
    seen, changed = set(), 0
    prev = w.pi
    for t in range(1, 2001):
        w.step(t)
        if w.pi != prev:
            changed += 1; prev = w.pi
        seen.add(w.pi)
    n += 2
    if changed < 15:
        fails.append(f"pi changed only {changed} times in 20 remaps")
    if len(seen) < 8:
        fails.append(f"pi took only {len(seen)} distinct values over 20 remaps")
    # 7: decay -- half-life inside the era, outside a lifetime
    cfg = Config(record="real")
    hl = np.log(2) / cfg.mark_decay
    n += 2
    if not (hl < cfg.prep_every):
        fails.append(f"mark half-life {hl:.0f} >= prep_every {cfg.prep_every}: a mark outlives "
                     f"the era whose meaning it carries")
    if hl < 50:
        fails.append(f"mark half-life {hl:.0f} is too short to outlive the agent that wrote it")
    w = World(cfg, np.random.default_rng(0)); w.chain_on = True
    w.marks[:] = 0.0; w.marks[0, 0, 5, 5] = 1.0
    for t in range(1, int(hl) + 1):
        w.step(t)
    n += 1
    if not (0.4 < w.marks[0, 0, 5, 5] < 0.6):
        fails.append(f"after one half-life the mark is {w.marks[0,0,5,5]:.3f}, expected ~0.5")
    # 8: reading is an observation, gated by sym_gain
    cfgr = Config(mode="fixed", record="real", scaffold_food=False)
    ar = Agent(cfgr, np.random.default_rng(1), 0, 0, 0)
    n += 3
    o_none = _obs_with_mark(0, None, 0.0)
    o_pos = _obs_with_mark(0, 2, +1.0)
    if np.abs(o_none[READ:READ + N_PREPS]).sum() > 1e-9:
        fails.append("no mark: the read channels are not zero")
    if o_pos[READ + 2] != 1.0 or np.abs(o_pos[READ:READ + N_PREPS]).sum() != 1.0:
        fails.append("a mark at label 2 did not land on read channel 2 alone")
    if N_IN != READ + N_PREPS:
        fails.append(f"N_IN {N_IN} does not match READ {READ} + K {N_PREPS}")
    # 9: sym_gain heritable, starts near zero
    n += 3
    if abs(ar.sym_gain - cfgr.sym_gain_init) > 1e-12:
        fails.append("sym_gain does not start at sym_gain_init")
    if cfgr.sym_gain_init > 0.2:
        fails.append(f"sym_gain_init {cfgr.sym_gain_init} is not 'near zero'")
    kid = ar.child(cfgr, np.random.default_rng(5), 10)
    if "sym_gain" not in GENOME or kid.sym_gain == ar.sym_gain:
        fails.append("sym_gain is not heritable-with-mutation")
    if verbose:
        print(f"  {n} rows checked: automatic write (label, sign, type, cell) over "
              f"{N_TYPES} types x {N_PREPS} preparations, eat-writes-nothing, noise, none, "
              f"pi redraw, decay half-life {np.log(2)/Config().mark_decay:.0f} steps, "
              f"read channels, sym_gain")
        print(f"  record-semantics self-test: {'PASS' if not fails else 'FAIL'}")
        for f_ in fails[:8]:
            print("   ", f_)
    return not fails


def replay_mapping_selftest(seeds=(0, 1, 2, 3, 4, 5), verbose=True):
    """The bug this exists to catch: a knockout that RE-SEEDS the world does not get the mapping
    its replayed genomes were selected under.

    World.__init__ draws the mapping from the world rng, so a replay under a fresh seed gets that
    seed's FIRST mapping -- while the source population was last selected under the source run's
    LAST mapping, after however many redraws.  Scoring a committed genome against a mapping it has
    never seen and reporting the result as "the genome carries nothing" is exactly what the v3.10
    and v3.11 knockouts did: plastic seed 0 read 0.296 against a shuffled mapping and 0.844
    against its own.

    Three claims:
      1. force_mapping pins the mapping: the run reports it, every window carries it, no redraws;
      2. WITHOUT force_mapping, a re-seeded replay's mapping differs from the source's final
         mapping for at least one seed -- which is the condition that made the old reading wrong;
      3. the two are distinguishable, i.e. a source run whose final mapping is pinned reproduces
         it and an unpinned one need not.
    """
    src = run(Config(seed=0, mode="fixed", n_steps=2400, chain=True, scaffold_food=False,
                     prep_every=350, prep_value=1.0, prep_fail=0.5), verbose=False)
    fm = src["final_mapping"]

    pinned = run(Config(seed=99, mode="fixed", n_steps=600, chain=True, scaffold_food=False,
                        prep_every=350, force_mapping=fm), verbose=False)
    c1 = (tuple(pinned["final_mapping"]) == tuple(fm)
          and all(tuple(w["mapping"]) == tuple(fm) for w in pinned["log"])
          and not pinned["recipe_changes"])

    drawn = []
    for sd in seeds:
        w = World(Config(seed=sd, prep_every=350), np.random.default_rng(sd))
        drawn.append(tuple(int(x) for x in w.mapping))
    c2 = any(m != fm for m in drawn)

    out = [c1, c2]
    if verbose:
        print(f"  source final mapping {fm}; re-seeded worlds draw {drawn}")
        print(f"  force_mapping pins it, no redraws: {c1}")
        print(f"  a re-seeded replay can differ from it (the bug condition): {c2}")
    passed = all(out)
    print(f"  replay-mapping self-test: {'PASS' if passed else 'FAIL'}")
    return passed


def founder_tag_selftest(seed=0, verbose=True):
    """The founder tag must sit on the INJECTED agent and stop there.

    Three claims, each checked rather than asserted:
      1. an injected agent is tagged, an ordinary one is not;
      2. an injected agent's CHILD is not tagged (its descendants are selected, not dropped in);
      3. in a real run the founder-free counts never exceed the all-agents counts, and they are
         EQUAL exactly when nothing was injected -- which is the check that the gate is wired to
         the tag and not to something that merely correlates with it.
    """
    cfg = Config(mode="fixed", scaffold_food=False)
    rng = np.random.default_rng(seed)
    inj = Agent(cfg, rng, 0, 0, 0, injected=True)
    nat = Agent(cfg, rng, 1, 0, 0)
    kid = inj.child(cfg, rng, 10)
    out = [inj.injected is True, nat.injected is False, kid.injected is False]
    if verbose:
        print(f"  injected agent tagged: {out[0]}   ordinary agent untagged: {out[1]}"
              f"   injected agent's child untagged: {out[2]}")

    # min_pop far above what a non-learner can sustain, so injection fires CONSTANTLY and the
    # exclusion path is actually exercised -- a self-test that passes on a run with no injections
    # proves nothing about the gate.
    r = run(Config(seed=seed, mode="fixed", n_steps=1200, chain=True, scaffold_food=False,
                   prep_value=1.0, prep_fail=0.5, prep_every=700, max_pop=800,
                   min_pop=300, init_pop=50),
            verbose=False)
    le = sum(w["f_n_attempts_raw"] <= w["n_attempts_raw"] and w["f_n_eats"] <= w["n_eats"]
             for w in r["log"]) == len(r["log"])
    eq = all((w["f_n_attempts_raw"] == w["n_attempts_raw"]) == (w["n_founder_prep"] == 0)
             for w in r["log"])
    inj_total = sum(w["injections"] for w in r["log"])
    fs = sum(w["n_founder_prep"] for w in r["log"]) / max(1, sum(w["n_attempts_raw"] for w in r["log"]))
    fired = sum(w["n_founder_prep"] for w in r["log"]) > 0     # the path was exercised
    out += [le, eq, fired]
    if verbose:
        print(f"  founder-free counts never exceed all-agents: {le}")
        print(f"  equal exactly when no founder event occurred: {eq}")
        print(f"  founder events actually occurred, so the gate was exercised: {fired}")
        print(f"  ({inj_total} injections over 1200 steps; founder share of preparations {fs:.3f})")
    passed = all(out)
    print(f"  founder-tag self-test: {'PASS' if passed else 'FAIL'}")
    return passed


def learning_rule_selftest(seed=0, verbose=True):
    """One agent, one fixed observation, one chosen action.  m = +1 must RAISE that action's
    logit and m = -1 must LOWER it.  This drives the real act() and learn() rather than a
    re-implementation of the rule: action_noise is set to 0 so act() is deterministic, so the
    action whose logit is checked is the action the agent actually took and laid a trace on."""
    cfg = Config(mode="plastic", plastic_layers="W2", action_noise=0.0,
                 scaffold_food=False, scaffold_chain=False, trace_recency=False)
    obs = np.zeros(N_IN)
    obs[HERE + 0] = 1.0          # food type A underfoot
    obs[ENERGY] = 0.5
    out = []
    for m in (1.0, -1.0):
        rng = np.random.default_rng(seed)
        a = Agent(cfg, rng, 0, 0, 0)
        a.eta2 = 0.2
        before = _forward(a, cfg, obs, learned=True)
        action = a.act(obs, cfg, rng)          # deterministic; lays the eligibility trace
        a.learn(m, cfg)
        after = _forward(a, cfg, obs, learned=True)
        delta = float(after[action] - before[action])
        ok = (delta > 0) if m > 0 else (delta < 0)
        out.append(ok)
        if verbose:
            print(f"  m = {m:+.0f}: chosen action {action}, its logit moved {delta:+.4f}  "
                  f"{'OK' if ok else 'FAIL'}")
    passed = all(out)
    if verbose:
        print(f"  learning rule self-test: {'PASS' if passed else 'FAIL'}")
    return passed

def all_mappings():
    """Every mapping that sends the T food types to DISTINCT preparations -- the space the
    population has to cover by standing variation if it is to track remaps by sorting alone."""
    from itertools import permutations
    return [tuple(p) for p in permutations(range(N_PREPS), N_TYPES)]


def world_semantics_selftest(verbose=True):
    """Every (action, food cell, mapping) row of the action x cell table, ENUMERATED FROM T AND K
    rather than from literals: (1 eat + K preparations) x (1 empty + T food cells) x P(K,T)
    mappings, plus the poison rows, the moves, phase-1 masking, and mapping distinctness.

    Each row constructs the cell, calls one resolve_action, and asserts the energy delta, the
    modulator, the event, and whether the food cell was consumed.  This is the test that would
    catch a mapping applied to the wrong food type -- and at T = 3, K = 5 there are 1440 outcome
    rows, far past what would be caught by eye.

    THE ROW THAT IS NEW IN v3.12: food type 2 is INEDIBLE RAW.  `eat` on it must give an energy
    change of exactly 0, m = 0, and must NOT consume the cell.
    """
    cfg = Config(mode="fixed", scaffold_food=False, scaffold_chain=False)
    rng = np.random.default_rng(0)
    MAPPINGS = all_mappings()
    assert len(MAPPINGS) == N_MAPPINGS, f"expected {N_MAPPINGS} mappings, enumerated {len(MAPPINGS)}"

    def fresh(food=None, mapping=None, safe=0):
        w = World(cfg, np.random.default_rng(0))
        w.food[:] = False
        w.safe, w.mapping = safe, (mapping if mapping is not None else MAPPINGS[0])
        w.chain_on = True
        a = Agent(cfg, np.random.default_rng(1), 0, 5, 5)
        a.energy = 3.0
        if food is not None:
            w.food[food, 5, 5] = True
        return a, w

    fails, n = [], 0
    for mapping in MAPPINGS:
        for food in [None] + list(range(N_TYPES)):
            for action in [EAT] + [PREP0 + k for k in range(N_PREPS)]:
                a, w = fresh(food=food, mapping=mapping, safe=0)
                e0 = a.energy
                m, ev, _ = resolve_action(a, action, w, cfg, rng)
                d = a.energy - e0
                should_consume = True
                if food is None:
                    d_exp, m_exp, ev_exp = -cfg.noop_cost, 0.0, "noop"
                    should_consume = False
                elif action == EAT and food >= 2:
                    d_exp, m_exp, ev_exp = 0.0, 0.0, "eat_inedible"   # <-- the v3.12 row
                    should_consume = False
                elif action == EAT:
                    safe = (food == 0)
                    d_exp = cfg.food_value if safe else -cfg.poison_value
                    m_exp = 1.0 if safe else -1.0
                    ev_exp = "eat_safe" if safe else "eat_poison"
                else:
                    k = action - PREP0
                    ok = (k == mapping[food])
                    d_exp = cfg.prep_value if ok else -cfg.prep_fail
                    m_exp = 1.0 if ok else -1.0
                    ev_exp = "prep_ok" if ok else "prep_bad"
                gone = (food is None) or (not w.food[food, 5, 5])
                consumed_ok = (gone == should_consume) or food is None
                ok_row = abs(d - d_exp) < 1e-9 and m == m_exp and ev == ev_exp and consumed_ok
                n += 1
                if not ok_row:
                    fails.append(f"map {mapping} food {food} action {action}: got dE {d:+.3f} "
                                 f"m {m:+.0f} {ev} consumed={gone}, expected dE {d_exp:+.3f} "
                                 f"m {m_exp:+.0f} {ev_exp} consumed={should_consume}")
    # the flip covers both raw types both ways, and must NOT touch type 2
    for food, safe in ((0, 1), (1, 1), (0, 0), (1, 0)):
        a, w = fresh(food=food, safe=safe)
        e0 = a.energy
        m, ev, _ = resolve_action(a, EAT, w, cfg, rng)
        d_exp = cfg.food_value if food == safe else -cfg.poison_value
        m_exp = 1.0 if food == safe else -1.0
        n += 1
        if abs((a.energy - e0) - d_exp) > 1e-9 or m != m_exp:
            fails.append(f"eat food {food} safe {safe}: dE {a.energy-e0:+.3f} m {m:+.0f}")
    for safe in (0, 1):                       # type 2 is inedible under EITHER flip setting
        a, w = fresh(food=2, safe=safe)
        e0 = a.energy
        m, ev, _ = resolve_action(a, EAT, w, cfg, rng)
        n += 1
        if abs(a.energy - e0) > 1e-9 or m != 0.0 or ev != "eat_inedible" or not w.food[2, 5, 5]:
            fails.append(f"eat type 2 under safe={safe}: dE {a.energy-e0:+.4f} m {m} {ev} "
                         f"cell_present={bool(w.food[2,5,5])} -- the flip must not touch type 2")
    # moves
    for action in range(4):
        a, w = fresh()
        e0 = a.energy
        m, ev, _ = resolve_action(a, action, w, cfg, rng)
        n += 1
        if abs((a.energy - e0) + cfg.move_cost) > 1e-9 or m != 0.0 or ev != "move":
            fails.append(f"move {action}: dE {a.energy-e0:+.4f} m {m} {ev}")
    # all K preparations are masked while phase 2 is off
    cfgm = Config(mode="fixed", action_noise=0.0, scaffold_food=False, scaffold_chain=False)
    am = Agent(cfgm, np.random.default_rng(3), 0, 0, 0)
    obs = np.zeros(N_IN); obs[ENERGY] = 0.5
    chosen = {am.act(obs, cfgm, np.random.default_rng(k), chain_on=False) for k in range(60)}
    n += 1
    if not all(c < PREP0 for c in chosen):
        fails.append("a preparation was chosen in phase 1; all K must be masked")
    # type 2 must not be spawned in phase 1
    w1 = World(cfg, np.random.default_rng(11)); w1.chain_on = False
    for t in range(400):
        w1.step(t)
    n += 1
    if w1.food[2].any():
        fails.append("food type 2 appeared in phase 1; it must arrive only at the switch")
    # mappings: T distinct preparations, and a redraw changes at least one type
    w = World(cfg, np.random.default_rng(7))
    draws = [w._draw_mapping(w.mapping) for _ in range(300)]
    n += 3
    if not all(len(set(m)) == N_TYPES for m in draws):
        fails.append("a mapping gave the same preparation to two food types")
    if not all(w._draw_mapping(m) != m for m in draws):
        fails.append("a redraw returned the mapping it was meant to replace")
    if len(set(draws)) < N_MAPPINGS // 2:
        fails.append(f"only {len(set(draws))} distinct mappings in 300 draws of {N_MAPPINGS}")
    if verbose:
        print(f"  {n} rows checked: (1 eat + {N_PREPS} preps) x (1 empty + {N_TYPES} food cells)"
              f" x {len(MAPPINGS)} mappings, plus the inedible-raw rows, the flip, moves,"
              f" phase-1 masking and mapping distinctness")
        print(f"  world-semantics self-test: {'PASS' if not fails else 'FAIL'}")
        for f in fails[:8]:
            print("   ", f)
    return not fails


# v3.9 amendment 3: the chain is co-located with foraging, so the world criterion is IN-PATCH,
# not global cover.  From a random patch cell the nearest station of each type must be <= 3 steps.
PATCH_TARGETS = dict(max_station_dist=3.0, item_cover=(0.20, 0.25))


def _patch_mask(w, cfg):
    """Cells inside any food patch (Chebyshev radius, matching how spawning works)."""
    g, r = cfg.grid, cfg.patch_radius
    m = np.zeros((g, g), dtype=bool)
    yy = np.arange(g)
    for (py, px) in w.patches:
        dy = np.minimum(np.abs(yy - py), g - np.abs(yy - py))
        dx = np.minimum(np.abs(yy - px), g - np.abs(yy - px))
        m |= (dy[:, None] <= r) & (dx[None, :] <= r)
    return m


def patch_metrics(cfg=None, steps=1000, seeds=(0, 1, 2, 3), n_sample=400):
    """In-patch item cover, mean torus-Manhattan distance from a random patch cell to the nearest
    station of each type, and a check that nothing spawns outside a patch."""
    cfg = cfg or Config(chain=True)
    g = cfg.grid
    acc = {k: [] for k in ("item_cover", "d0", "d1", "outside", "n_station_cells", "patch_cells")}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        w = World(cfg, rng)
        w.chain_on = True
        for t in range(steps):
            w.step(t)
        pm = _patch_mask(w, cfg)
        items = w.items >= 0
        acc["item_cover"].append(items[pm].sum() / max(pm.sum(), 1))
        acc["outside"].append(int((items & ~pm).sum() + ((w.stations >= 0) & ~pm).sum()))
        acc["n_station_cells"].append(int((w.stations >= 0).sum()))
        acc["patch_cells"].append(int(pm.sum()))
        ys, xs = np.where(pm)
        pick = rng.choice(len(ys), size=min(n_sample, len(ys)), replace=False)
        for st, key in ((0, "d0"), (1, "d1")):
            sy, sx = np.where(w.stations == st)
            if len(sy) == 0:
                acc[key].append(np.inf); continue
            dy = np.abs(ys[pick][:, None] - sy[None, :]); dy = np.minimum(dy, g - dy)
            dx = np.abs(xs[pick][:, None] - sx[None, :]); dx = np.minimum(dx, g - dx)
            acc[key].append(float(np.mean((dy + dx).min(1))))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def assert_cover(cfg=None, steps=1000, verbose=True):
    m = patch_metrics(cfg, steps)
    fails = []
    lo, hi = PATCH_TARGETS["item_cover"]
    dmax = PATCH_TARGETS["max_station_dist"]
    checks = [
        ("in-patch item cover", f"{m['item_cover']*100:5.1f}%", lo <= m["item_cover"] <= hi,
         f"band {lo*100:.0f}-{hi*100:.0f}%"),
        ("dist to station 0", f"{m['d0']:5.2f}", m["d0"] <= dmax, f"<= {dmax:.0f} steps"),
        ("dist to station 1", f"{m['d1']:5.2f}", m["d1"] <= dmax, f"<= {dmax:.0f} steps"),
        ("items/stations outside patches", f"{m['outside']:5.0f}", m["outside"] == 0, "must be 0"),
    ]
    if verbose:
        print(f"  in-patch world criterion, no agents, {steps} steps, mean over 4 seeds:")
    for lab, val, ok, tgt in checks:
        if not ok:
            fails.append(f"{lab} {val} vs {tgt}")
        if verbose:
            print(f"    {lab:<32} {val}   {tgt:<22} {'OK' if ok else 'FAIL'}")
    if verbose:
        print(f"    {'station cells':<32} {m['n_station_cells']:5.0f}   "
              f"(some offsets coincide, so a few of the {cfg.n_patches if cfg else 8} x 2 x "
              f"{(cfg or Config()).stations_per_patch} overlap)")
        print(f"    {'patch cells':<32} {m['patch_cells']:5.0f}   of {(cfg or Config()).grid ** 2}")
        print(f"  world criterion: {'PASS' if not fails else 'FAIL -- ' + '; '.join(fails)}")
    return not fails
