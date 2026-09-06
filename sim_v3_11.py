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
N_ACTIONS = 8            # 0-3 move, 4 eat, 5-7 prep_1..3.  v3.10: the recipe chain is gone; the
EAT = 4                  # fact to be learned sits on EVERY meal, so there is no approach behaviour
PREP0 = 5                # to evolve.  Preparations are masked in phase 1, which is therefore
N_PREPS = 3              # exactly v3.1's five actions.
INTERACT = None           # retired with the chain
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
N_PAIRS = 6                                      # 3 items x 2 stations


# --------------------------------------------------------------------------
# observation layout.  11 channels x (4 directional sums + 1 'here' value),
# then energy, the inventory one-hot, and 'carrying a tool'.
#   channels: 0 foodA  1 foodB  2 occupancy  3 food_any(appetite)
#             4,5,6 item types   7,8 station types   9 nuts   10 goal
# --------------------------------------------------------------------------

N_CH = 11
APPETITE_CH = 3
ITEM_CH = 4
STATION_CH = 7
NUT_CH = 9
GOAL_CH = 10

HERE = N_CH * 4              # 44 .. 54   'here' value of each channel
ENERGY = HERE + N_CH         # 55
INV = ENERGY + 1             # 56,57,58   inventory one-hot over item types
TOOL = INV + 3               # 59
N_IN = TOOL + 1              # 60


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
        self.food = np.zeros((2, g, g), dtype=bool)
        self.safe = 0
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
        """One correct preparation per food type, and the two types map to DIFFERENT preparations,
        so exactly one preparation is useless in any era and the task cannot be solved without
        discriminating food type.  A redraw differs from the old mapping in at least one type."""
        while True:
            a = int(self.rng.integers(N_PREPS))
            b = int(self.rng.integers(N_PREPS))
            if a == b:
                continue
            m = (a, b)
            if old is None or m != old:
                return m

    def new_recipe(self, t):
        self.mapping = self._draw_mapping(self.mapping)
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
                    self.food[rng.integers(2), y, x] = True


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
        # The three preparations are masked while phase 2 is off, so phase 1 is EXACTLY v3.1's
        # five actions and its gate applies unchanged.  ONLY the action space changes at the
        # switch -- no new inputs.  The null is masked too, so its per-action share is 1/5 in
        # phase 1 and 1/8 in phase 2, and 3/8 for "any preparation".
        n_av = N_ACTIONS if chain_on else PREP0        # 8 in phase 2, 5 in phase 1
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


GENOME = ("W1", "b1", "W2", "b2", "eta1", "eta2", "lam1", "lam2", "repair", "nav_dir", "nav_here")


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
        for ft in range(2):
            p = np.array([prep_pref(a, cfg, ft, k, learned) for k in range(N_PREPS)])
            vals.append(p[mapping[ft]] - np.delete(p, mapping[ft]).mean())
    return float(np.mean(vals))


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

    has_food = bool(world.food[0, y, x] or world.food[1, y, x])
    if not has_food:
        a.energy -= cfg.noop_cost
        return 0.0, "noop", {}
    ftype = 0 if world.food[0, y, x] else 1

    if action == EAT:
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
             prep_n0=0, prep_ok0=0, prep_n1=0, prep_ok1=0,
             first_n=0, first_ok=0, surv_n=0, surv_early=0, surv_late=0,
             srm_n=0, srm_pre=0, srm_post=0, first_n_late=0, first_ok_late=0)
    # FOUNDER-FREE mirror.  Injected agents are fresh random genomes; their own events dilute
    # every event-weighted metric toward chance, and the dilution is heaviest in exactly the arms
    # that need injecting -- so a non-learning arm reads as MORE random the worse it does.  WF
    # accumulates the same events with injected agents' own events excluded.  Their children are
    # not tagged, so descent back into the population is counted from the first generation.
    WF = {k: 0 for k in ("eats", "safe", "attempts", "correct", "prep_n0", "prep_ok0",
                         "prep_n1", "prep_ok1", "first_n", "first_ok",
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
        pA, pB = pad(world.food[0]), pad(world.food[1])
        pO = pad(np.minimum(occ, 3) / 3.0)
        # the food_any channel exists only to drive the instinct; with no food scaffold it is dead,
        # which makes phase 1's live inputs exactly v3.1's set (4 dirsums x 3 channels + 3 here + energy)
        pFood = (pA + pB) if cfg.scaffold_food else np.zeros_like(pA)
        if world.chain_on:
            pI = [pad(world.items == i) for i in range(cfg.n_items)]
            pS = [pad(world.stations == s) for s in range(cfg.n_stations)]
            pN = pad(world.nuts)
        else:
            z = np.zeros_like(pA)
            pI = [z] * cfg.n_items
            pS = [z] * cfg.n_stations
            pN = z

        rng.shuffle(agents)
        survivors, newborns = [], []

        for a in agents:
            y, x = a.y, a.x
            sl = (slice(y, y + win), slice(x, x + win))

            # what, if anything, this agent believes about the six pairs.
            # Zero in every condition except the hand-wired ceiling.
            if cfg.private_mem:
                pb = a.B.reshape(cfg.n_items, cfg.n_stations)
                prefs = np.clip(np.concatenate([pb.max(1), pb.max(0)]), -1, 2)
                attract = np.clip(1.0 + cfg.pref_gain * prefs, 0.0, 1.0 + cfg.pref_gain)
            else:
                pb = None
                attract = np.ones(cfg.n_items + cfg.n_stations)

            # the stage machine: it points at a CLASS of thing, never at which one.  With
            # scaffold_chain off it is only an observation channel -- nothing is wired to it.
            if not (world.chain_on and cfg.goal_channel):
                goal = np.zeros((win, win))
            elif a.tool:
                goal = pN[sl]
            elif a.item >= 0:
                goal = sum(attract[cfg.n_items + s] * pS[s][sl] for s in range(cfg.n_stations))
            else:
                goal = sum(attract[i] * pI[i][sl] for i in range(cfg.n_items))

            chans = [pA[sl], pB[sl], pO[sl], pFood[sl]] + [p[sl] for p in pI] + [p[sl] for p in pS] + [pN[sl], goal]
            inv = np.zeros(cfg.n_items)
            if a.item >= 0:
                inv[a.item] = 1.0
            obs = np.concatenate([dirsum(c, v) for c in chans]
                                 + [[c[v, v] for c in chans], [a.energy / cfg.max_energy], inv,
                                    [1.0 if a.tool else 0.0]])
            action = a.act(obs, cfg, rng, world.chain_on)
            if track_recency and not world.chain_on:
                a.e2_hist.append(a.e2.copy())          # phase 1 only: the chain would confound it
                if len(a.e2_hist) > cfg.recency_k + 1:
                    a.e2_hist.pop(0)

            a.integrity = np.minimum(1.0, a.integrity - cfg.decay + a.repair * cfg.repair_gain)
            a.energy -= cfg.base_cost + a.repair * cfg.repair_cost * cfg.hidden
            if a.item >= 0 or a.tool:
                a.energy -= cfg.carry_cost

            on_food = bool(world.food[0, y, x] or world.food[1, y, x])
            ft_here = (0 if world.food[0, y, x] else 1) if on_food else -1

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
                probe_adv_food=probe_advantage_food(agents, cfg, world.safe),
                food_gain_innate=food_gain(agents, cfg, world.safe, False),
                crop_safe=float(world.food[world.safe].sum() / max(1, world.food.sum())),
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
                prep_hit0=(W["prep_ok0"] / W["prep_n0"]) if W["prep_n0"] else np.nan,
                prep_hit1=(W["prep_ok1"] / W["prep_n1"]) if W["prep_n1"] else np.nan,
                n_prep0=W["prep_n0"], n_ok0=W["prep_ok0"], n_prep1=W["prep_n1"], n_ok1=W["prep_ok1"],
                map_a=int(world.mapping[0]), map_b=int(world.mapping[1]),
                prep_per_life=W["prep_meals"] / max(W["deaths"], 1),
                n_first=W["first_n"], n_first_ok=W["first_ok"],
                n_first_late=W["first_n_late"], n_first_ok_late=W["first_ok_late"],
                n_surv=W["surv_n"], n_surv_early=W["surv_early"], n_surv_late=W["surv_late"],
                n_srm=W["srm_n"], n_srm_pre=W["srm_pre"], n_srm_post=W["srm_post"],
                # the window these counters were summed over.  SR_W is a module constant, so a
                # checkpoint written under one value and read under another would divide by the
                # wrong denominator -- silently, with a plausible number.  Recorded per window.
                sr_w=SR_W,
                noops_per_1k=1000.0 * W["noops"] / max(W["steps"], 1),
                e_bonus_per_1k=1000.0 * W["e_bonus"] / max(W["steps"], 1),     # share of energy income from nuts
                att_n=ATT[1:, 0].tolist(), att_correct=ATT[1:, 1].tolist(),
                rec_n=ATT_R[1:, 0].tolist(), rec_correct=ATT_R[1:, 1].tolist(),
                # founder-free: the same events with injected agents' own events excluded
                f_n_eats=WF["eats"], f_n_safe=WF["safe"],
                f_n_attempts_raw=WF["attempts"], f_n_correct=WF["correct"],
                f_n_prep0=WF["prep_n0"], f_n_ok0=WF["prep_ok0"],
                f_n_prep1=WF["prep_n1"], f_n_ok1=WF["prep_ok1"],
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
    c1 = (pinned["final_mapping"] == fm
          and all((w["map_a"], w["map_b"]) == fm for w in pinned["log"])
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

def world_semantics_selftest(verbose=True):
    """Every (action, food cell, mapping) row of the action x cell table: 4 actions
    {eat, prep_1..3} x 3 cell states {empty, food A, food B} x 6 distinct mappings = 72 rows,
    plus the moves and the poison rows.  Each constructs the cell, calls one resolve_action, and
    asserts the energy delta, the modulator, the event, and that the food cell was consumed.
    This is the test that would catch a mapping applied to the wrong food type."""
    cfg = Config(mode="fixed", scaffold_food=False, scaffold_chain=False)
    rng = np.random.default_rng(0)
    MAPPINGS = [(a, b) for a in range(N_PREPS) for b in range(N_PREPS) if a != b]

    def fresh(food=None, mapping=(0, 1), safe=0):
        w = World(cfg, np.random.default_rng(0))
        w.food[:] = False; w.items[:] = -1; w.nuts[:] = False; w.stations[:] = -1
        w.safe, w.mapping = safe, mapping
        a = Agent(cfg, np.random.default_rng(1), 0, 5, 5)
        a.energy = 3.0
        if food is not None:
            w.food[food, 5, 5] = True
        return a, w

    fails, n = [], 0
    for mapping in MAPPINGS:
        for food in (None, 0, 1):
            for action in [EAT] + [PREP0 + k for k in range(N_PREPS)]:
                a, w = fresh(food=food, mapping=mapping, safe=0)
                e0 = a.energy
                m, ev, _ = resolve_action(a, action, w, cfg, rng)
                d = a.energy - e0
                if food is None:
                    d_exp, m_exp, ev_exp = -cfg.noop_cost, 0.0, "noop"
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
                consumed = (food is None) or (not w.food[food, 5, 5])
                ok_row = abs(d - d_exp) < 1e-9 and m == m_exp and ev == ev_exp and consumed
                n += 1
                if not ok_row:
                    fails.append(f"map {mapping} food {food} action {action}: got dE {d:+.3f} m {m:+.0f} "
                                 f"{ev} consumed={consumed}, expected dE {d_exp:+.3f} m {m_exp:+.0f} {ev_exp}")
    # eat on the OTHER safe setting, so both eat outcomes are covered for both types
    for food, safe in ((0, 1), (1, 1)):
        a, w = fresh(food=food, safe=safe)
        e0 = a.energy
        m, ev, _ = resolve_action(a, EAT, w, cfg, rng)
        d_exp = cfg.food_value if food == safe else -cfg.poison_value
        m_exp = 1.0 if food == safe else -1.0
        n += 1
        if abs((a.energy - e0) - d_exp) > 1e-9 or m != m_exp:
            fails.append(f"eat food {food} safe {safe}: dE {a.energy-e0:+.3f} m {m:+.0f}")
    # moves
    for action in range(4):
        a, w = fresh()
        e0 = a.energy
        m, ev, _ = resolve_action(a, action, w, cfg, rng)
        n += 1
        if abs((a.energy - e0) + cfg.move_cost) > 1e-9 or m != 0.0 or ev != "move":
            fails.append(f"move {action}: dE {a.energy-e0:+.4f} m {m} {ev}")
    # the preparations are masked while phase 2 is off
    cfgm = Config(mode="fixed", action_noise=0.0, scaffold_food=False, scaffold_chain=False)
    am = Agent(cfgm, np.random.default_rng(3), 0, 0, 0)
    obs = np.zeros(N_IN); obs[ENERGY] = 0.5
    chosen = {am.act(obs, cfgm, np.random.default_rng(k), chain_on=False) for k in range(60)}
    masked = all(c < PREP0 for c in chosen)
    n += 1
    if not masked:
        fails.append("a preparation was chosen in phase 1; they must be masked")
    # the two types must map to DIFFERENT preparations
    w = World(cfg, np.random.default_rng(7))
    draws = [w._draw_mapping(w.mapping) for _ in range(200)]
    distinct = all(m[0] != m[1] for m in draws)
    differs = all(w._draw_mapping(m) != m for m in draws)      # a redraw changes the mapping
    n += 2
    if not distinct:
        fails.append("a mapping gave the same preparation to both food types")
    if not differs:
        fails.append("a redraw returned the mapping it was meant to replace")
    if verbose:
        print(f"  {n} rows checked: 4 actions x 3 cells x {len(MAPPINGS)} mappings, "
              f"plus poison rows, moves, masking and mapping distinctness")
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
