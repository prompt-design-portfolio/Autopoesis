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
    carry_cost: float = 0.002     # per step while holding an item or a tool (v2.9b)
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
    stations_per_type: int = 20
    items_per_step: float = 1.5
    nuts_per_patch: float = 0.3
    nuts_uniform: float = 0.0     # nuts spawned anywhere, not only in the food patches.  This sets
                                  # the length of the station -> nut bridge, i.e. the difficulty of
                                  # the delayed credit; see the tuning note in the notebook.
    item_rot: float = 0.005
    nut_value: float = 1.0
    tool_break: float = 0.25
    pickup_cost: float = 0.02
    fail_cost: float = 0.0        # 0 = a wrong attempt gives no modulator, only the lost item
    recipe_every: int = 2000      # ~8-13 generations per era
    # mutation
    mut_sigma: float = 0.15
    mut_rate: float = 0.10
    gene_sigma: float = 0.05
    action_noise: float = 0.3
    # the learner (v3.0)
    mode: str = "plastic"          # "fixed" | "plastic"
    plastic_layers: str = "both"   # "both" | "W1" | "W2"
    eta_init: float = 0.2          # founders' learning-rate genes ~ U(0, eta_init)  (v3 FAST)
    eta_max: float = 0.5
    h_max: float = 2.0
    scramble: bool = False         # control: random-sign modulator, same magnitude, no information
    eta_scale: float = 1.0         # v3.2 knockout hook: multiplies eta at learn time
    # the hand-wired ceiling (v2's private memory B; OFF except in that one condition)
    private_mem: bool = False      # exact per-pair credit, steering navigation and a soft veto
    pref_gain: float = 3.0
    veto_p: float = 0.8
    # logging
    n_attempts: int = 10           # attempt-number curves run 1..n_attempts
    n_meals: int = 10
    log_every: int = 50
    seed: int = 0


N_ACTIONS = 5                                    # up, down, left, right, stay/interact
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
        for s in range(cfg.n_stations):
            for _ in range(cfg.stations_per_type):
                y, x = rng.integers(0, g, 2)
                self.stations[y, x] = s
        self.recipe = (int(rng.integers(cfg.n_items)), int(rng.integers(cfg.n_stations)))
        self.flips, self.recipe_changes = [], []

    def new_recipe(self, t):
        cfg = self.cfg
        while True:
            r = (int(self.rng.integers(cfg.n_items)), int(self.rng.integers(cfg.n_stations)))
            if r != self.recipe:
                self.recipe = r
                self.recipe_changes.append(t)
                return

    def step(self, t):
        cfg, g, rng = self.cfg, self.cfg.grid, self.rng
        changed_recipe = False
        if t > 0 and t % cfg.patch_drift_every == 0:
            self.patches = (self.patches + rng.integers(-8, 9, size=self.patches.shape)) % g
        if t > 0 and t % cfg.flip_every == 0:
            self.safe = 1 - self.safe
            self.flips.append(t)
        if t > 0 and t % cfg.recipe_every == 0:
            self.new_recipe(t)
            changed_recipe = True
        # rot
        self.food &= rng.random(self.food.shape) > cfg.food_rot
        self.items[rng.random(self.items.shape) < cfg.item_rot] = -1
        self.nuts &= rng.random(self.nuts.shape) > cfg.item_rot
        # food and nuts grow in the drifting patches
        r = cfg.patch_radius
        for (py, px) in self.patches:
            k = rng.poisson(cfg.spawn_per_patch)
            for dy, dx in rng.integers(-r, r + 1, size=(k, 2)):
                y, x = (py + dy) % g, (px + dx) % g
                if not self.food[:, y, x].any():
                    self.food[rng.integers(2), y, x] = True
            k = rng.poisson(cfg.nuts_per_patch)
            for dy, dx in rng.integers(-r, r + 1, size=(k, 2)):
                self.nuts[(py + dy) % g, (px + dx) % g] = True
        for _ in range(rng.poisson(cfg.nuts_uniform)):
            self.nuts[tuple(rng.integers(0, g, 2))] = True
        # items anywhere
        for _ in range(rng.poisson(cfg.items_per_step)):
            y, x = rng.integers(0, g, 2)
            if self.items[y, x] < 0:
                self.items[y, x] = rng.integers(cfg.n_items)
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
                 "integrity", "repair", "energy", "y", "x", "item", "tool", "B",
                 "lineage", "gen", "born", "injected",
                 "attempts", "successes", "eats", "safe_eats", "cracks",
                 "since_recipe", "tool_made_t", "used_tool", "age_bins")

    def __init__(self, cfg, rng, lineage, y, x, injected=False):
        h = cfg.hidden
        self.W1 = rng.normal(0, 0.3, (N_IN, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, 0.3, (h, N_ACTIONS))
        self.b2 = np.zeros(N_ACTIONS)
        innate_nav(self.W1, self.W2)
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
        self.tool_made_t = -1
        self.used_tool = False
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
        c.energy = cfg.start_energy
        c.y, c.x = self.y, self.x
        c.item, c.tool = -1, False           # nothing carried is inherited
        c.B = np.zeros(N_PAIRS)              # no memory is inherited either
        c.lineage, c.gen, c.born, c.injected = self.lineage, self.gen + 1, t, self.injected
        c.attempts = c.successes = 0
        c.eats = c.safe_eats = c.cracks = 0
        c.since_recipe = 0
        c.tool_made_t = -1
        c.used_tool = False
        c.age_bins = np.zeros((2, 2))
        return c

    def act(self, obs, cfg, rng):
        alive = self.integrity >= cfg.integrity_threshold
        plastic = cfg.mode == "plastic"
        W1 = self.W1 + self.H1 if plastic and cfg.plastic_layers in ("both", "W1") else self.W1
        W2 = self.W2 + self.H2 if plastic and cfg.plastic_layers in ("both", "W2") else self.W2
        h = np.tanh(obs @ W1 + self.b1) * alive
        logits = h @ W2 + self.b2 + rng.normal(0, cfg.action_noise, N_ACTIONS)
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


def innate_nav(W1, W2):
    """v2.9b's forager instinct, ported to this observation layout.  Approach whatever the
    current stage of the chain wants ('goal') and approach food ('food_any'); interact when
    standing on either.  It says nothing about WHICH item, station or food type -- that is
    the experiment.  Identical in every condition."""
    W1 *= 0.1
    W2 *= 0.1
    for d in range(4):
        W1[APPETITE_CH * 4 + d, d] = 4.0
        W1[GOAL_CH * 4 + d, 4 + d] = 4.0
        W2[d, d] = 4.0
        W2[4 + d, d] = 4.0
    W1[HERE + APPETITE_CH, 8] = 4.0
    W1[HERE + GOAL_CH, 9] = 4.0
    W2[8, 4] = 4.0
    W2[9, 4] = 4.0


GENOME = ("W1", "b1", "W2", "b2", "eta1", "eta2", "lam1", "lam2", "repair")


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
    return a


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


def pair_pref(a, cfg, item, station, learned=True):
    """'I am holding item i and standing on a station of type s, nothing else in view.'
    Returns logit(stay/interact) - best move logit: how much this agent wants to ATTEMPT
    this pair.  goal_here is set because a carrier standing on a station really does see
    its goal underfoot -- the probe has to sit on the input distribution the policy meets."""
    obs = np.zeros(N_IN)
    obs[INV + item] = 1.0
    obs[HERE + STATION_CH + station] = 1.0
    obs[HERE + GOAL_CH] = 1.0
    obs[ENERGY] = 0.5
    logits = _forward(a, cfg, obs, learned)
    return float(logits[4] - np.delete(logits, 4).max())


def pair_gain(agents, cfg, recipe, learned=True, n_sample=40):
    """Preference for the TRUE pair minus mean preference for the other five.
    > 0 means the policy attempts the right conjunction more readily than the wrong ones."""
    if len(agents) < 5:
        return np.nan
    idx = np.linspace(0, len(agents) - 1, min(n_sample, len(agents))).astype(int)
    p_true = pair_id(*recipe)
    vals = []
    for k in idx:
        a = agents[k]
        prefs = np.array([pair_pref(a, cfg, i, s, learned) for i in range(cfg.n_items) for s in range(cfg.n_stations)])
        vals.append(prefs[p_true] - np.delete(prefs, p_true).mean())
    return float(np.mean(vals))


def probe_advantage(agents, cfg, recipe, n_sample=40):
    """Within-agent counterfactual on the RECIPE: pair_gain with learned synapses minus with
    innate ones.  Same genome, same world state: the change this individual's own plasticity
    produced.  Exactly zero by construction when mode == 'fixed'."""
    if cfg.mode != "plastic" or len(agents) < 5:
        return np.nan
    return pair_gain(agents, cfg, recipe, True, n_sample) - pair_gain(agents, cfg, recipe, False, n_sample)


def food_pref(a, cfg, ftype, learned=True):
    obs = np.zeros(N_IN)
    obs[HERE + ftype] = 1.0
    obs[HERE + APPETITE_CH] = 1.0
    obs[ENERGY] = 0.5
    logits = _forward(a, cfg, obs, learned)
    return float(logits[4] - np.delete(logits, 4).max())


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
# the run
# --------------------------------------------------------------------------

def run(cfg, verbose=True, init_genomes=None):
    rng = np.random.default_rng(cfg.seed)
    g, v = cfg.grid, cfg.view
    win = 2 * v + 1
    world = World(cfg, rng)
    for _ in range(100):
        world.step(-1)
    if init_genomes is None:
        agents = [Agent(cfg, rng, i, *rng.integers(0, g, 2)) for i in range(cfg.init_pop)]
    else:
        agents = [restore(d, cfg, rng, i, *rng.integers(0, g, 2)) for i, d in enumerate(init_genomes)]
    next_lineage = len(agents)

    log = []
    W = dict(eats=0, safe=0, deaths=0, births=0, steps=0, inject=0, attempts=0, correct=0,
             nuts=0, pickups=0, e_food=0.0, e_nut=0.0, bridge_sum=0.0, bridge_n=0, trace_w=0.0,
             bridge1_sum=0.0, bridge1_n=0)
    ATT = np.zeros((cfg.n_attempts + 1, 2))     # attempt number in an agent's life -> (n, correct)
    ATT_R = np.zeros((cfg.n_attempts + 1, 2))   # attempts since the last recipe change
    MEALS = np.zeros((cfg.n_meals + 1, 2))      # meal number in an agent's life (the v3 curve)
    t0 = time.time()

    for t in range(cfg.n_steps):
        if world.step(t):
            for a in agents:
                a.since_recipe = 0
        occ = np.zeros((g, g))
        for a in agents:
            occ[a.y, a.x] += 1
        pad = lambda arr: np.pad(np.asarray(arr, dtype=float), v, mode="wrap")
        pA, pB = pad(world.food[0]), pad(world.food[1])
        pO = pad(np.minimum(occ, 3) / 3.0)
        pFood = pA + pB                                     # approach food of either type
        pI = [pad(world.items == i) for i in range(cfg.n_items)]
        pS = [pad(world.stations == s) for s in range(cfg.n_stations)]
        pN = pad(world.nuts)

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

            # innate stage machine: it points at a class of thing, never at which one
            if a.tool:
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
            action = a.act(obs, cfg, rng)

            a.integrity = np.minimum(1.0, a.integrity - cfg.decay + a.repair * cfg.repair_gain)
            a.energy -= cfg.base_cost + a.repair * cfg.repair_cost * cfg.hidden
            if a.item >= 0 or a.tool:
                a.energy -= cfg.carry_cost

            m = 0.0
            if action < 4:
                a.y, a.x = (y + MOVES[action][0]) % g, (x + MOVES[action][1]) % g
                a.energy -= cfg.move_cost
            else:
                st = int(world.stations[y, x])
                if st >= 0 and a.item >= 0 and not a.tool and (not cfg.private_mem or a.B[pair_id(a.item, st)] > -0.3 or rng.random() > cfg.veto_p):
                    # ---- a tool attempt.  No modulator here, whatever happens: the payoff is at the nut.
                    ok = (a.item, st) == world.recipe
                    p = pair_id(a.item, st)
                    a.attempts += 1; W["attempts"] += 1
                    if ok:
                        a.tool = True
                        a.tool_made_t = t
                        a.used_tool = False
                        a.successes += 1; W["correct"] += 1
                    else:
                        a.energy -= cfg.fail_cost
                    a.item = -1
                    if cfg.private_mem:
                        a.B[p] = 0.7 * a.B[p] + 0.3 * (1.0 if ok else -1.0)   # exact pair credit, hand-wired
                    old = 0 if (t - a.born) < 150 else 1
                    a.age_bins[old, 0] += 1; a.age_bins[old, 1] += ok
                    k = a.attempts
                    if k <= cfg.n_attempts:
                        ATT[k, 0] += 1; ATT[k, 1] += ok
                    a.since_recipe += 1
                    kr = a.since_recipe
                    if kr <= cfg.n_attempts:
                        ATT_R[kr, 0] += 1; ATT_R[kr, 1] += ok
                elif world.nuts[y, x] and a.tool:
                    # ---- the payoff, arriving many steps after the attempt that earned it
                    world.nuts[y, x] = False
                    a.energy += cfg.nut_value
                    m = 1.0
                    a.cracks += 1; W["nuts"] += 1; W["e_nut"] += cfg.nut_value
                    if a.tool_made_t >= 0:
                        gap = t - a.tool_made_t
                        gap_is_first = not a.used_tool
                        a.used_tool = True
                        W["bridge_sum"] += gap; W["bridge_n"] += 1
                        W["trace_w"] += a.lam2 ** gap        # what is left of the attempt's trace now
                        if gap_is_first:
                            W["bridge1_sum"] += gap; W["bridge1_n"] += 1
                    if rng.random() < cfg.tool_break:
                        a.tool = False
                        a.tool_made_t = -1
                elif world.items[y, x] >= 0 and a.item < 0 and not a.tool and (not cfg.private_mem or pb[int(world.items[y, x])].max() > -0.3 or rng.random() > cfg.veto_p):
                    a.item = int(world.items[y, x])
                    world.items[y, x] = -1
                    a.energy -= cfg.pickup_cost
                    W["pickups"] += 1
                elif world.food[0, y, x] or world.food[1, y, x]:
                    ftype = 0 if world.food[0, y, x] else 1
                    world.food[ftype, y, x] = False
                    if ftype == world.safe:
                        a.energy += cfg.food_value; m = 1.0
                        W["safe"] += 1; W["e_food"] += cfg.food_value; a.safe_eats += 1
                    else:
                        a.energy -= cfg.poison_value; m = -1.0
                    W["eats"] += 1; a.eats += 1
                    k = a.eats
                    if k <= cfg.n_meals:
                        MEALS[k, 0] += 1; MEALS[k, 1] += (m > 0)

            if cfg.scramble and m != 0.0:
                m = 1.0 if rng.random() < 0.5 else -1.0      # same magnitude, no information
            a.learn(m, cfg)

            a.energy = min(a.energy, cfg.max_energy)
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
        while len(agents) < cfg.min_pop:
            agents.append(Agent(cfg, rng, next_lineage, *rng.integers(0, g, 2), injected=True))
            next_lineage += 1
            W["inject"] += 1

        if (t + 1) % cfg.log_every == 0:
            young = np.sum([a.age_bins[0] for a in agents], axis=0)
            old = np.sum([a.age_bins[1] for a in agents], axis=0)
            income = W["e_food"] + W["e_nut"]
            log.append(dict(
                t=t + 1, pop=len(agents),
                recipe_hit=(W["correct"] / W["attempts"]) if W["attempts"] else np.nan,
                attempts_per_1k=1000.0 * W["attempts"] / max(W["steps"], 1),
                attempts_per_life=W["attempts"] / max(W["deaths"], 1),   # births ~ deaths in steady state
                nuts_per_1k=1000.0 * W["nuts"] / max(W["steps"], 1),
                pickups_per_1k=1000.0 * W["pickups"] / max(W["steps"], 1),
                hit_young=(young[1] / young[0]) if young[0] > 20 else np.nan,   # attempts in the first 150 steps of life
                hit_old=(old[1] / old[0]) if old[0] > 20 else np.nan,           # ... and after.  Within-life learning = old > young
                safe_rate=(W["safe"] / W["eats"]) if W["eats"] else np.nan,
                holding_item=float(np.mean([a.item >= 0 for a in agents])),
                has_tool=float(np.mean([a.tool for a in agents])),
                energy=float(np.mean([a.energy for a in agents])),
                eta1=float(np.mean([a.eta1 for a in agents])), eta2=float(np.mean([a.eta2 for a in agents])),
                lam1=float(np.mean([a.lam1 for a in agents])), lam2=float(np.mean([a.lam2 for a in agents])),
                h_norm=float(np.mean([np.abs(a.H1).mean() + np.abs(a.H2).mean() for a in agents])),
                repair=float(np.mean([a.repair for a in agents])),
                max_gen=max(a.gen for a in agents),
                deaths=W["deaths"], births=W["births"], injections=W["inject"],
                # raw counts, so second-half aggregates can be event-weighted rather than
                # means of per-window ratios (a mean-of-ratios artifact cost a retraction in v2)
                n_attempts_raw=W["attempts"], n_correct=W["correct"], n_eats=W["eats"],
                n_safe=W["safe"], n_nuts=W["nuts"], agent_steps=W["steps"],
                probe_adv=probe_advantage(agents, cfg, world.recipe),
                pair_gain=pair_gain(agents, cfg, world.recipe, True),
                pair_gain_innate=pair_gain(agents, cfg, world.recipe, False),
                probe_adv_food=probe_advantage_food(agents, cfg, world.safe),
                food_gain_innate=food_gain(agents, cfg, world.safe, False),
                crop_safe=float(world.food[world.safe].sum() / max(1, world.food.sum())),
                nut_share=(W["e_nut"] / income) if income > 0 else np.nan,     # share of energy income from nuts
                bridge_steps=(W["bridge_sum"] / W["bridge_n"]) if W["bridge_n"] else np.nan,
                bridge_first=(W["bridge1_sum"] / W["bridge1_n"]) if W["bridge1_n"] else np.nan,
                trace_weight=(W["trace_w"] / W["bridge_n"]) if W["bridge_n"] else np.nan,
                att_n=ATT[1:, 0].tolist(), att_correct=ATT[1:, 1].tolist(),
                rec_n=ATT_R[1:, 0].tolist(), rec_correct=ATT_R[1:, 1].tolist(),
                meal_n=MEALS[1:, 0].tolist(), meal_safe=MEALS[1:, 1].tolist(),
                nut_cells=int(world.nuts.sum()), item_cells=int((world.items >= 0).sum()),
            ))
            ATT[:] = 0; ATT_R[:] = 0; MEALS[:] = 0
            for k in W:
                W[k] = 0 if isinstance(W[k], int) else 0.0
            if verbose and (t + 1) % (cfg.log_every * 20) == 0:
                L = log[-1]
                print(f"t={L['t']:5d} pop={L['pop']:3d} hit={L['recipe_hit']:.3f} att/1k={L['attempts_per_1k']:.2f} "
                      f"nuts/1k={L['nuts_per_1k']:.2f} safe={L['safe_rate']:.2f} eta={L['eta1']:.3f}/{L['eta2']:.3f} "
                      f"lam2={L['lam2']:.3f} bridge={L['bridge_steps']:.0f} [{time.time()-t0:.0f}s]", flush=True)
    return dict(log=log, flips=world.flips, recipe_changes=world.recipe_changes,
                cfg=asdict(cfg), final=snapshot(agents))
