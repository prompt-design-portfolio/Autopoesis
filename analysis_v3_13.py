"""
v3.13 analysis -- the record -- the preparation world, after the v3.10 acceptance run.

Two parameter changes (prep_every 2000 -> 700, prep_value 1.5 -> 1.0) and three rule fixes; see
"Changes after the v3.10 acceptance" in spec_v3_11.md.

v3.9 closed the recipe world: a chain paying only at the end is sparse, and a rare opportunity
cannot generate the selection differential that would build the approach behaviour making it less
rare.  Here the fact sits on every meal, so there is no approach behaviour to evolve.

v3.8's row 4 is void as a test of the learner: one action did attempt / crack / pickup / eat, the
chain was ambient (items on 40% of cells, nuts on 47%), declining was not a policy, and there was
no random-walk null.  This module reads a rig with those fixed (audit A, B, D, F).

Aggregates are event-weighted and computed PER PHASE.  Two window conventions, as v3.8:
  phase_half(k)  the second half of phase k
  tail()         the last quarter of the run, the same absolute window in every arm
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from sim_v3_13 import (Config, run, N_PREPS, N_TYPES, PREP0, N_MAPPINGS, all_mappings,
                 SR_W, SURV_EARLY)

SRM_MIN_N = 30      # survivor-conditioned since-remap: minimum agent-remaps per seed.  Below this
                    # the line is DROPPED rather than read on a narrower window.

# ---------------------------------------------------------------------------
# the world -- v3.1 metabolism throughout, the chain at the agreed cover targets
# ---------------------------------------------------------------------------
WORLD = dict(
    # phase 1 is v3.1, unchanged
    flip_every=300, eta_init=0.2, hidden=24,
    # 6.0 (was 3.0).  D5's readability criterion -- >= 3 preparations per TYPE per era in
    # `fixed` -- fails at 3.0 once the food is split three ways (median 2.6, min 1.2).  Raised
    # per the ruling: raise spawn density, NEVER lengthen the era, since a longer era is more
    # time for the sorting this world exists to outrun.  Judged on `fixed` only, per rule 9.
    spawn_per_patch=6.0, food_value=0.7, poison_value=0.5,
    repro_threshold=3.0, repro_cost=1.5, max_energy=5.0, max_pop=1000, init_pop=300,
    # phase 2 adds three preparations.  Nothing else changes -- no new inputs, no items,
    # stations or nuts.  The fact to be learned sits on EVERY meal.
    # v3.12: prep_value = (K-1) * prep_fail = 4 * 0.25 = 1.0 puts the CHANCE EV of a preparation
    # at exactly zero: (1/5)(+1.0) - (4/5)(0.25) = 0.  Income to a KNOWING agent is unchanged from
    # v3.11 at +1.00 a meal, so max_pop stays 800.  The halved penalty is the targeted part:
    # SORTING runs on energy -- a mismatched genotype pays prep_fail on 4 preparations in 5 -- but
    # the LEARNING signal is a sign, m = +/-1 as literals in resolve_action, independent of both
    # prep_value and prep_fail.  So this slows selection and leaves learning untouched.  Eating raw is +0.10 at chance and +0.70 knowing the flip, so
    # preparation now pays ONLY through knowledge of the mapping, and a population cannot ride the
    # preparation payoff up to the cap without it.  (v3.10 acceptance ran prep_value 1.5, where a
    # chance preparation paid +0.17 and every outcome arm sat at 675-799 against a cap of 800.)
    prep_value=1.0, prep_fail=0.25,
    # BACK TO 700.  350 was tried and reverted: it shortened the LEARNER's payoff window without
    # touching the sorting mechanism at all (pre-registered prediction failed both ways -- fixed
    # 0.642 against a predicted 0.52-0.56, plastic 0.575 against 0.65-0.70, so the learner lost to
    # the non-learner).  Genes track the mapping by SURVIVAL SORTING over standing variation, and
    # that completes well inside a third of an era, so no era length this world can support will
    # outrun it.  The answer is a bigger mapping space, not a shorter era -- see spec_v3_12.md.
    # Kept for the record, the reasoning that led to 350:
    # STANDING POLYMORPHISM: the population carries genotypes for several of the six possible
    # mappings at once, so a remap needs no mutation -- lineage selection just promotes whichever
    # genotype already matches, within the era.  350 is BELOW a generation, and deliberately not a
    # multiple of flip_every (300) so the two facts do not come into phase.  Judged against
    # `fixed` only, per rule 9.  If the gate still fires at 350, the next change is K = 4.
    prep_every=700,
    scaffold_food=False, scaffold_chain=False, goal_channel=False,
)

PHASE_STEPS = 8000
STAGED = [dict(n_steps=PHASE_STEPS, chain=False), dict(n_steps=PHASE_STEPS, chain=True)]

def _v(**kw):
    return dict(kw=dict(**dict(WORLD, **kw)), phases=STAGED)

VARIANTS = {
    # v3.13's four arms.  `plastic` is v3.12's world unchanged -- the baseline the record must
    # beat.  `noise record` is the load-bearing control: a store changes the world (channels
    # exist, cells carry state, decay runs), and an arm that improves because a store EXISTS is
    # not an arm that improved because information PASSED.  Noise holds mark density, sign and
    # decay identical and destroys only the label->preparation association.
    "plastic":              _v(mode="plastic", plastic_layers="W2", record="none"),
    "plastic + record":     _v(mode="plastic", plastic_layers="W2", record="real"),
    "plastic + noise":      _v(mode="plastic", plastic_layers="W2", record="noise"),
    "fixed + record":       _v(mode="fixed", record="real"),
    # SLOW LABELS, brought forward from the deferred follow-up.  label_every = 3 x prep_every, so
    # a label's meaning outlives what it names by three eras -- v3.5's tempo condition.  Meaning
    # is still not inheritable: a genome fixing on "label j -> preparation k" is right for three
    # eras and then wrong, which is far inside evolutionary time.
    "plastic + record (slow)": _v(mode="plastic", plastic_layers="W2", record="real",
                                  label_every=3 * WORLD["prep_every"]),
}

NULL = "plastic"        # v3.13 has no random arm: the baseline is v3.12's learner
OUTCOME_ARMS = [n for n in VARIANTS if n != NULL]

COLORS = {"plastic": "tab:grey", "plastic + record": "tab:blue",
          "plastic + noise": "black", "fixed + record": "tab:red",
          "plastic + record (slow)": "tab:green"}

# Three levels on the preparation task, and the middle one is the one to watch:
CHANCE = 1.0 / N_PREPS    # a random preparation: 1/5 = 0.200
TYPE_BLIND = 1.0 / N_TYPES  # 1/3 = 0.333. "always prep_k" for a k that is useful in this era: right for one food
                          # type, wrong for the other, and NO type knowledge at all.  Note that
                          # ACROSS eras such a policy scores only 1/3 -- with distinct mappings, k
                          # is useless in a third of them (EV -0.5/meal there) -- so a genome beats
                          # chance only by tracking the era, not by holding one preparation.
FULL = 1.0                # the conjunction: the right preparation for each type

MARGIN = 0.03
SEED_RULE = 4          # min(SEED_RULE, n_seeds): a 3-seed pass reads as 3/3


# ---------------------------------------------------------------- running

NEW_FIELDS = {
    "final_mapping":  "row 3b knockout (which mapping the genomes were selected under)",
    "n_first_late":   "first-preparation hit, late-in-era",
    "n_srm":          "survivor-conditioned since-remap",
    "n_surv_early":   "survivor halves 1-2 vs 6-10 (older runs stored the 1-5 split instead)",
}


def checkpoint_audit(results, verbose=True):
    """Which runs in this checkpoint were written by a sim that recorded the newer fields.

    Run this before reading a checkpoint that spans builds.  A grid stage appends runs from the
    CURRENT sim to seeds written by an older one, and an aggregate over that mixture would be
    reading two different quantities under one label.  None of these is recoverable from the
    stored log -- the counters are accumulated inside the sim, not derived from it.  In
    particular `final_mapping` cannot be reconstructed offline: World shares its rng with the
    agents, so the mapping draw sequence depends on every action-noise draw in the run.
    """
    stale = {}
    for name, runs in results.items():
        for r in runs:
            L = r["log"]
            miss = [k for k in ("final_mapping",) if k not in r]
            miss += [k for k in ("n_first_late", "n_srm") if k not in L[0]]
            if "n_first_late" not in L[0]:      # an older build: n_surv_early was the 1-5 split
                miss.append("n_surv_early")
            if miss:
                stale[(name, r["cfg"]["seed"])] = sorted(set(miss))
    if verbose:
        if not stale:
            print("checkpoint audit: every run carries the current fields.")
        else:
            print("CHECKPOINT AUDIT -- these runs predate fields the current reading needs.")
            print("  NOT recoverable from the stored log; the counters live inside the sim.")
            for (name, sd), miss in sorted(stale.items()):
                print(f"    {name:<22} seed {sd}   missing: {', '.join(miss)}")
            print("  Affected reads:")
            for k in sorted({m for v in stale.values() for m in v}):
                print(f"    {k:<16} -> {NEW_FIELDS[k]}")
            print("  Those lines print `nan`.  To recover them, re-run the affected (arm, seed)")
            print("  pairs with the current sim.py:  run_experiment(..., refresh=[('fixed', 0), ...])")
    return stale


def load(path):
    """Reload a checkpoint.  Returns {} if there is nothing there yet, so the run cell can be
    re-executed after a dropped session without editing it."""
    import pickle, os
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _done_seeds(results, name):
    return {r["cfg"]["seed"] for r in results.get(name, [])}


def run_experiment(seeds, phase_steps=PHASE_STEPS, variants=VARIANTS, verbose=False,
                   save_path=None, results=None, refresh=(), **overrides):
    """Sequential, single process: this has to run on a Colab CPU runtime, so no multiprocessing.

    RESUMABLE.  Pass `results` (from `load(save_path)`) and any (arm, seed) already present is
    SKIPPED, so the grid stage continues from the acceptance stage's checkpoint in the same
    session instead of re-running the six runs it shares.  Runs are pickled after every one, so a
    dropped Colab session costs one run rather than the lot.  Each arm's list is returned in seed
    order regardless of the order the runs were actually done in.

    `refresh` is a list of (arm, seed) pairs to redo even though the checkpoint already has them.
    Use it when `checkpoint_audit` reports runs written before a field the reading needs."""
    import pickle, time
    results = {k: list(vv) for k, vv in (results or {}).items()}
    for v_ in variants:
        results.setdefault(v_, [])
    stale = {tuple(x) for x in refresh}
    if stale:
        for name in list(results):
            results[name] = [r for r in results[name]
                             if (name, r["cfg"]["seed"]) not in stale]
        print(f"refreshing {len(stale)} stale run(s): {sorted(stale)}", flush=True)
    todo = [(seed, name) for seed in seeds for name in variants
            if seed not in _done_seeds(results, name)]
    skipped = len(seeds) * len(variants) - len(todo)
    if skipped:
        print(f"resuming: {skipped} run(s) already in the checkpoint, {len(todo)} to do", flush=True)
    total, done, t0 = len(todo), 0, time.time()
    for seed, name in todo:
        spec = variants[name]
        phases = [dict(p, n_steps=int(p["n_steps"] / PHASE_STEPS * phase_steps)) for p in spec["phases"]]
        kw = dict(spec["kw"]); kw.update(overrides)          # overrides win over WORLD
        t1 = time.time()
        results[name].append(run(Config(seed=seed, **kw), verbose=verbose, phases=phases))
        done += 1
        el = time.time() - t0
        print(f"[{done}/{total}] {name} seed={seed}  {time.time()-t1:.0f}s   "
              f"elapsed {el/60:.1f} min, est. stage total {el/done*total/60:.0f} min", flush=True)
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
    for name in results:                                     # seed order, not run order
        results[name].sort(key=lambda r: r["cfg"]["seed"])
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
    return {k: vv for k, vv in results.items() if vv}


# ---------------------------------------------------------------- windows

def window(run_, lo, hi):
    return [r for r in run_["log"] if lo < r["t"] <= hi]


def phase_half(run_, k):
    """Second half of phase k.  Phase 1 of a from-scratch run does not exist (the chain is on
    from step 0); it returns that run's first half, which row 1 must not be applied to."""
    bounds = run_["phase_bounds"]
    lo, hi = bounds[k] if k < len(bounds) else (0, run_["n_steps"])
    if len(bounds) == 1:                       # from-scratch: split its single phase in two
        mid = run_["n_steps"] // 2
        lo, hi = (0, mid) if k == 0 else (mid, run_["n_steps"])
    return window(run_, (lo + hi) // 2, hi)


def tail(run_, frac=0.25):
    """The last `frac` of the run -- the same absolute window in every condition."""
    n = run_["n_steps"]
    return window(run_, int(n * (1 - frac)), n)


def half(L, key):
    v = np.array([r[key] for r in L], dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def has(L, key):
    """Was this field recorded by the sim that produced these runs?  Checkpoints written before a
    field existed must read as `not recorded`, never as zero -- a missing counter summed as 0
    would make a rate look like a measured 0.000 rather than an absent measurement."""
    return bool(L) and key in L[0]


def rate(L, num, den):
    if not has(L, num) or not has(L, den):
        return np.nan
    d = float(np.sum([r[den] for r in L]))
    return float(np.sum([r[num] for r in L]) / d) if d > 0 else np.nan


def hit(L):
    return rate(L, "n_correct", "n_attempts_raw")


# FOUNDER-FREE.  Injected agents are fresh random genomes dropped in to hold the population off
# the floor.  Their own events pull every event-weighted metric toward chance, and the pull is
# heaviest in exactly the arms that need injecting -- so a non-learning arm reads as MORE random
# the worse it does.  That is founder dilution, and it is a metric artifact, not a fact about the
# arm.  Every metric below takes ff=True to exclude injected agents' OWN events; their children
# are not tagged, so a founder's descendants count from the first generation.
def _k(key, ff):
    """Founder-free log keys are the all-agents names with an `f_` prefix, verbatim."""
    return ("f_" + key) if ff else key


def safe(L, ff=False):
    return rate(L, _k("n_safe", ff), _k("n_eats", ff))


def founder_share(L, what="prep"):
    """Share of events contributed by injected agents themselves -- the size of the dilution."""
    key, den = (("n_founder_prep", "n_attempts_raw") if what == "prep"
                else ("n_founder_eats", "n_eats"))
    if not has(L, key):
        return np.nan
    d = float(np.sum([r[den] for r in L]))
    return float(np.sum([r[key] for r in L]) / d) if d > 0 else np.nan


def per_1k(L, key):
    s = float(np.sum([r["agent_steps"] for r in L]))
    return float(1000.0 * np.sum([r[key] for r in L]) / s) if s > 0 else np.nan


def curve(L, key, ff=False):
    p = "f_" + key if (ff and key in ("att", "rec")) else key      # no founder-free meal curve
    n = np.sum([r[p + "_n"] for r in L], axis=0)
    c = np.sum([r[p + ("_safe" if key == "meal" else "_correct")] for r in L], axis=0)
    return np.where(n > 0, c / np.maximum(n, 1), np.nan)


def ps(results, name, sel, f):
    """Per-seed values of f over window selector sel."""
    return np.array([f(sel(r)) for r in results[name]], dtype=float)


def smooth(y, k=5):
    y = np.asarray(y, float); out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]; w = w[~np.isnan(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


# ---------------------------------------------------------------- the transition

def transition_table(results, bin_size=500, span=2000):
    """Population, safe rate, eta2, lam2, h_norm and attempts/1k in `bin_size` bins across the
    `span` steps either side of the chain switching on.  A from-scratch run has no switch."""
    print(f"\nTRANSITION -- {bin_size}-step bins, {span} steps either side of the preparations going live")
    for name, res in results.items():
        sw = res[0].get("chain_start", 0)
        if not sw:
            print(f"\n  {name}: no transition (chain on from step 0)")
            continue
        print(f"\n  {name}   (switch at t = {sw})")
        print(f"    {'bin':>14}{'pop':>8}{'safe':>8}{'eta2':>8}{'lam2':>8}{'h_norm':>9}{'att/1k':>9}")
        for lo in range(sw - span, sw + span, bin_size):
            rows = [window(r, lo, lo + bin_size) for r in res]
            rows = [w for w in rows if w]
            if not rows:
                continue
            g = lambda f: np.nanmean([f(w) for w in rows])
            mark = "  <- switch" if lo == sw else ""
            print(f"    {lo:6d}-{lo+bin_size:<7d}{g(lambda w: half(w,'pop')):8.0f}"
                  f"{g(safe):8.3f}{g(lambda w: half(w,'eta2')):8.3f}{g(lambda w: half(w,'lam2')):8.3f}"
                  f"{g(lambda w: half(w,'h_norm')):9.3f}{g(lambda w: per_1k(w,'n_attempts_raw')):9.2f}{mark}")


def first_bin_drop(run_, bin_size=500):
    """Safe rate in the last `bin_size` steps BEFORE the chain switches on, minus the first
    `bin_size` steps after.  A basis shift shows up here, in the first bin; swamping and
    depletion take longer than one bin to build."""
    sw = run_.get("chain_start", 0)
    if not sw:
        return np.nan, np.nan, np.nan
    pre, post = window(run_, sw - bin_size, sw), window(run_, sw, sw + bin_size)
    a, b = safe(pre), safe(post)
    return a, b, a - b


# ---------------------------------------------------------------- summary

ROWS = [
    # FOUNDER-FREE first -- injected agents' own events excluded.  That is the primary reading.
    ("safe_rate  FF",       lambda L: safe(L, True)),
    ("prep hit  FF",        lambda L: prep_hit(L, True)),
    ("prep hit | A  FF",    lambda L: hit_a(L, True)),
    ("prep hit | B  FF",    lambda L: hit_b(L, True)),
    ("type-blind level",    lambda L: type_blind_level(L, True)),
    ("founder share prep",  lambda L: founder_share(L, "prep")),
    ("founder share eats",  lambda L: founder_share(L, "eat")),
    # ... and the all-agents versions alongside, for this build.
    ("safe_rate",           safe),
    ("prep hit",            lambda L: prep_hit(L)),
    ("prep hit | type A",   lambda L: rate(L, "n_ok0", "n_prep0")),
    ("prep hit | type B",   lambda L: rate(L, "n_ok1", "n_prep1")),
    ("prep/life",           lambda L: half(L, "prep_per_life")),
    ("prep share of meals", lambda L: half(L, "prep_share")),
    ("P(prep | on food)",   lambda L: half(L, "prep_on_food")),
    ("P(eat | on food)",    lambda L: half(L, "eat_on_food")),
    ("pop",                 lambda L: half(L, "pop")),
    ("injections",          lambda L: half(L, "injections")),
    ("max_gen",             lambda L: half(L, "max_gen")),
    ("probe_adv (food)",    lambda L: half(L, "probe_adv_food")),
    ("probe_adv (prep)",    lambda L: half(L, "probe_adv")),
    ("prep_gain innate",    lambda L: half(L, "prep_gain_innate")),
    ("eta2",                lambda L: half(L, "eta2")),
    ("eta1",                lambda L: half(L, "eta1")),
    ("lam2",                lambda L: half(L, "lam2")),
    ("h_norm",              lambda L: half(L, "h_norm")),
    ("nav_dir",             lambda L: half(L, "nav_dir")),
    ("nav_here",            lambda L: half(L, "nav_here")),
    ("crop_safe",           lambda L: half(L, "crop_safe")),
    ("raw meals",           lambda L: float(np.sum([r["n_raw"] for r in L]))),
    ("meal gain",           lambda L: float(curve(L, "meal")[-1] - curve(L, "meal")[0])),
    ("hit-in-life gain",    lambda L: float(curve(L, "att")[-1] - curve(L, "att")[0])),
]


def prep_hit(L, ff=False):
    """Event-weighted prep hit over both food types."""
    pre = "f_" if ff else ""
    if not has(L, f"{pre}n_prep0"):
        return np.nan
    n = float(np.sum([sum(r[f"{pre}n_prep{t}"] for t in range(N_TYPES)) for r in L]))
    k = float(np.sum([sum(r[f"{pre}n_ok{t}"] for t in range(N_TYPES)) for r in L]))
    return k / n if n else np.nan


def hit_t(L, t, ff=False):
    """Hit on food type `t`.  T is a parameter now, so nothing indexes 0/1 by name."""
    return rate(L, _k(f"n_ok{t}", ff), _k(f"n_prep{t}", ff))


def hits_by_type(L, ff=False):
    return [hit_t(L, t, ff) for t in range(N_TYPES)]


def hit_a(L, ff=False):          # kept: the frozen-replay and probe code reads two of them
    return hit_t(L, 0, ff)


def hit_b(L, ff=False):
    return hit_t(L, 1, ff)


def type_share_a(L, ff=False):
    """Share of preparations made on food type A.  The type-blind level is max(share, 1 - share),
    NOT 0.5: with skewed encounters, "always prep_k" for the commoner type beats 0.5 without any
    type knowledge at all."""
    if not has(L, _k("n_prep0", ff)):
        return np.nan
    per = [float(np.sum([r[_k(f"n_prep{t}", ff)] for r in L])) for t in range(N_TYPES)]
    tot = sum(per)
    return (per[0] / tot) if tot else np.nan


def type_shares(L, ff=False):
    per = [float(np.sum([r[_k(f"n_prep{t}", ff)] for r in L])) for t in range(N_TYPES)]
    tot = sum(per)
    return [p / tot for p in per] if tot else [np.nan] * N_TYPES


def type_blind_level(L, ff=False):
    """The level "always prep_k" reaches: the share of the commonest type.  With balanced
    encounters that is 1/T; with skew it is higher, and 1/T is then the wrong reference."""
    sh = type_shares(L, ff)
    return max(sh) if np.all(np.isfinite(sh)) else np.nan


def surv_curve(L, ff=False):
    """Hit on preparations 1-2 vs 6-10, over agents that REACHED 10 preparations.  Every agent
    counted contributes both halves of its own curve, so a rise cannot be survivorship: it is the
    same individuals, later in their own lives.  (`probe_adv` is computed over the LIVING and so
    is itself partly survivorship-selected; this line is not.)"""
    if not has(L, _k("n_surv", ff)):
        return np.nan, np.nan, np.nan, 0
    n = float(np.sum([r[_k("n_surv", ff)] for r in L]))
    if not n:
        return np.nan, np.nan, np.nan, 0
    from sim_v3_13 import SURV_EARLY
    e = float(np.sum([r[_k("n_surv_early", ff)] for r in L])) / (SURV_EARLY * n)
    l = float(np.sum([r[_k("n_surv_late", ff)] for r in L])) / (5 * n)
    return e, l, l - e, int(n)


def surv_remap(L, ff=False):
    """SURVIVOR-CONDITIONED since-remap.  Only agents that made SR_W preparations BOTH before and
    after the same remap are counted, and each contributes its own rate on either side.  The
    population-level since-remap curve is open to the objection that the agents alive at
    preparation 1 are not the ones alive at 10; this is not, because it is the same agent across
    the same remap."""
    from sim_v3_13 import SR_W
    if not has(L, _k("n_srm", ff)):
        return np.nan, np.nan, np.nan, 0
    n = float(np.sum([r[_k("n_srm", ff)] for r in L]))
    if not n:
        return np.nan, np.nan, np.nan, 0
    pre = float(np.sum([r[_k("n_srm_pre", ff)] for r in L])) / (SR_W * n)
    post = float(np.sum([r[_k("n_srm_post", ff)] for r in L])) / (SR_W * n)
    return pre, post, post - pre, int(n)


def first_prep_hit(L, ff=False):
    """Hit on an agent's FIRST preparation: it has learned nothing, so this reads the innate
    policy the population currently carries."""
    return rate(L, _k("n_first_ok", ff), _k("n_first", ff))


def shift_timing(run_, bin_size=250, span=2000):
    """When the eat -> prep shift happens, in `bin_size` bins after the switch."""
    sw = run_.get("chain_start", 0)
    out = []
    for lo in range(sw, sw + span, bin_size):
        w = window(run_, lo, lo + bin_size)
        if w:
            out.append((lo - sw, round(float(half(w, "prep_share")), 3)))
    return out


FROZEN_STEPS = 300      # the frozen-replay window.  ~5 preparations is what within-life learning
                        # needs to show (the since-remap curve recovers by preparation 5), and a
                        # 10-step window cannot contain that -- which is why the v3.11 knockout
                        # failed as an instrument rather than returning a negative result.


def frozen_replay(run_, mapping, eta, steps=FROZEN_STEPS, snap=-1, seed_offset=7000):
    """Replay one era-boundary snapshot with the POPULATION FROZEN.

    Births, deaths and injection are all off; energy is tracked and spent but is not lethal.  So
    nothing can change over the window except H.  A hit rate that moves under `eta = 1` and does
    not move under `eta = 0` is within-life learning and can be nothing else -- not sorting, not
    survivorship, not founder replacement.

    The snapshot is taken at an ERA BOUNDARY, so the population has just lived a whole era under
    `snap["mapping"]` and is sorted for it.  `mapping` is what to pin for the replay: pass that one
    for MATCHED, any other for SHUFFLED.
    """
    from sim_v3_13 import Config, run as _run
    sn = run_["era_snaps"][snap]
    cfg = dict(run_["cfg"]); cfg.pop("seed", None); cfg.pop("n_steps", None)
    cfg.update(eta_scale=float(eta), force_mapping=tuple(mapping), frozen=True,
               log_every=max(10, steps // 6))
    rr = _run(Config(seed=seed_offset + int(run_["cfg"]["seed"]), **cfg), verbose=False,
              init_genomes=sn["genomes"], phases=[dict(n_steps=steps, chain=True)])
    return rr


def frozen_curve(rr):
    """Hit per log bin across the frozen window, plus first-bin and last-bin values."""
    c = [prep_hit([w]) for w in rr["log"]]
    return c, (c[0] if c else np.nan), (c[-1] if c else np.nan)


def shuffle_mapping(m, rng=None):
    """A valid mapping that differs from `m` in every type, so no genotype sorted for `m` retains
    any advantage.  With T = 2 that is the swap; with T = 3 it is a derangement."""
    m = tuple(int(x) for x in m)
    if len(m) == 2:
        return (m[1], m[0])
    return tuple(m[(i + 1) % len(m)] for i in range(len(m)))


def frozen_knockout(results, name, steps=FROZEN_STEPS, snap=-1):
    """The attribution line: matched and shuffled mapping, learning off and on, population frozen.

    Returns one row per seed with the four cells and the within-window curves.
    """
    out = []
    for r in results[name]:
        if not r.get("era_snaps"):
            out.append(dict(seed=r["cfg"]["seed"], missing=True))
            continue
        sn = r["era_snaps"][snap]
        fm = tuple(sn["mapping"])
        row = dict(seed=r["cfg"]["seed"], mapping=fm, t=sn["t"], n_pop=sn["n_pop"],
                   n_snap=len(sn["genomes"]), missing=False)
        for tag, mp in (("matched", fm), ("shuffled", shuffle_mapping(fm))):
            for eta in (0, 1):
                rr = frozen_replay(r, mp, eta, steps=steps, snap=snap)
                curve_, first, last = frozen_curve(rr)
                row[f"{tag}_{eta}"] = dict(hit=prep_hit(rr["log"]), curve=curve_,
                                           first=first, last=last,
                                           pop=half(rr["log"], "pop"),
                                           gen=half(rr["log"], "max_gen"))
        out.append(row)
    return out


# ---------------------------------------------------------------- v3.13 readers

def _mi(counts):
    """Mutual information in bits between X = (label, sign) and Y = the correct preparation."""
    c = np.asarray(counts, dtype=float).reshape(-1, counts.shape[-1])   # (label*sign, correct)
    n = c.sum()
    if n <= 0:
        return np.nan
    p = c / n
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / (px @ py)[nz])))


def gate_r(run_, per_era=True):
    """GATE R.  Pooled ACROSS eras, the mutual information between (label, sign) and the correct
    preparation must be at or below the level the `noise record` arm produces.

    Within an era it is maximal BY CONSTRUCTION -- label = pi(k) and sign = whether k was right,
    so the mark names the preparation exactly.  That is the design, not a leak.  What must not
    survive is the association ACROSS eras: pi is redrawn at every remap, so a genome that fixed on
    "label j means preparation k" is right only until the next redraw.  A pooled MI above the noise
    arm's means pi is not rotating fast enough for selection to be excluded, and the run is not
    read.
    """
    L2 = phase_half(run_, 1)
    if not L2 or "mi_counts" not in L2[0]:
        return dict(pooled=np.nan, per_era=[], n=0)
    pooled = np.sum([np.asarray(r["mi_counts"]) for r in L2], axis=0)
    out = dict(pooled=_mi(pooled), n=float(pooled.sum()), per_era=[])
    if per_era:
        for w in era_windows(run_):
            w = [r for r in w if "mi_counts" in r]
            if not w:
                continue
            c = np.sum([np.asarray(r["mi_counts"]) for r in w], axis=0)
            if c.sum() > 0:
                out["per_era"].append(_mi(c))
    return out


def gate_r_permutation(run_, n_perm=400, seed=0):
    """GATE R, on a MATCHED null.

    The form I specified -- pooled MI compared against the `noise record` arm -- is the wrong
    instrument, and the v3.13 pre-check showed why.  Pooled MI has a floor set by the NUMBER OF
    ERAS: with E eras a label takes only E meanings, so the empirical association cannot wash out
    however well pi is doing its job.  Measured: `plastic + record` pooled 0.677 over ~3 eras and
    0.605 over 5 -- it decays with era count, not toward the noise arm.  And the noise arm is not
    a matched comparison: it has different within-era structure, so the difference confounds "pi
    rotates" with "labels are random within an era".

    The matched null permutes EACH ERA'S LABEL AXIS INDEPENDENTLY: era count, sample sizes and
    within-era structure are all preserved, and only cross-era consistency is destroyed.  That is
    exactly the question the gate asks.  z near 0 means the observed pooled association is no
    stronger than chance given the era count -- pi is doing its job and meaning is not
    inheritable.  A large positive z means it is not, and the run is not read.
    """
    rng = np.random.default_rng(seed)
    # GROUP BY PI-EPOCH, not by era.  `plastic + record (slow)` holds one pi across 3 eras BY
    # DESIGN, so permuting per era would destroy consistency that legitimately exists there and
    # inflate z -- the gate would fire on the arm's own definition.  An epoch is a run of eras
    # sharing a pi; for the fast arms an epoch IS an era.
    groups, cur, cur_pi = [], None, None
    for w in era_windows(run_):
        w = [x for x in w if "mi_counts" in x]
        if not w:
            continue
        c = np.sum([np.asarray(x["mi_counts"]) for x in w], axis=0)
        if c.sum() <= 0:
            continue
        pi_ = tuple(w[-1].get("pi", ()))
        if pi_ != cur_pi:
            if cur is not None:
                groups.append(cur)
            cur, cur_pi = c, pi_
        else:
            cur = cur + c
    if cur is not None:
        groups.append(cur)
    mats = groups
    if not mats:
        return dict(obs=np.nan, null=np.nan, sd=np.nan, z=np.nan, eras=0)
    obs = _mi(np.sum(mats, axis=0))
    null = []
    for _ in range(n_perm):
        tot = np.zeros_like(mats[0])
        for m in mats:
            tot += m[rng.permutation(m.shape[0])]
        null.append(_mi(tot))
    mu, sd = float(np.mean(null)), float(np.std(null))
    return dict(obs=obs, null=mu, sd=sd, z=(obs - mu) / sd if sd > 0 else np.nan, eras=len(mats))


def nfc(L, ff=False):
    """Newborn preparations-to-first-correct: mean over agents that got one inside the window, and
    the censoring rate (agents that did not).  Lower mean and lower censoring is better."""
    p = "f_" if ff else ""
    if not has(L, f"{p}n_nfc"):
        return np.nan, np.nan, 0
    n = float(np.sum([r[f"{p}n_nfc"] for r in L]))
    c = float(np.sum([r[f"{p}n_nfc_cens"] for r in L]))
    ssum = float(np.sum([r["f_nfc_sum" if ff else "nfc_sum"] for r in L]))
    mean = ssum / n if n else np.nan
    rate = c / (n + c) if (n + c) else np.nan
    return mean, rate, int(n + c)


def first_prep_by_mark(L):
    """(i) Among FIRST-EVER preparations, the hit rate split by whether a positive mark for this
    food type was already on the cell.  A first preparation is the agent's own genome plus
    whatever the world is telling it -- it has learned nothing and written nothing, and by the
    no-self-echo property the mark cannot be its own.  So a gap here is the record being used."""
    return (rate(L, "n_fp_pos_ok", "n_fp_pos"), rate(L, "n_fp_none_ok", "n_fp_none"),
            float(np.sum([r.get("n_fp_pos", 0) for r in L])),
            float(np.sum([r.get("n_fp_none", 0) for r in L])))


def follow_split(L):
    """Following, split by whether the mark's endorsed preparation is the CORRECT one now.

    `endorsed == correct` is CONFOUNDED: agreeing with a mark that points at the right answer is
    indistinguishable from simply being right.  `endorsed != correct` is the unconfounded cell --
    a stale mark from before the mapping moved, where following it is a MISTAKE, so an agent that
    follows it can only be reading it.  1/K is the null in both."""
    return (rate(L, "n_follg_ok", "n_follg"), float(np.sum([r.get("n_follg", 0) for r in L])),
            rate(L, "n_follb_ok", "n_follb"), float(np.sum([r.get("n_follb", 0) for r in L])))


def follow_rate(L):
    """(ii) P(chosen preparation = pi^-1(strongest positive label) | a positive mark is present),
    against 1/K.  This is FOLLOWING the record, measured directly on behaviour rather than
    inferred from an outcome -- an agent can be right for its own reasons, but it cannot agree
    with the mark this often by accident."""
    return rate(L, "n_foll_ok", "n_foll"), float(np.sum([r.get("n_foll", 0) for r in L]))


def v313_precheck(results, control=None):
    """The v3.13 pre-check reading.  Nothing here is a claim -- 1 seed, short phases."""
    names = list(results)
    P2 = lambda r: phase_half(r, 1)
    g = lambda n, key: np.nanmean([half(P2(r), key) for r in results[n]])

    print("\n" + "=" * 78)
    print("GATE R -- MI between (label, sign) and the correct preparation, bits")
    print("=" * 78)
    print("  Within an era MI is maximal BY CONSTRUCTION (label = pi(k)); that is the design.")
    print("  The gate is the POOLED value, which must be at or below the noise arm's.")
    print(f"  {'arm':<24}{'pooled':>9}{'per-era mean':>14}{'n writes':>11}")
    pooled = {}
    for n in names:
        rs = [gate_r(r) for r in results[n]]
        pl = np.nanmean([x["pooled"] for x in rs])
        pe = np.nanmean([np.nanmean(x["per_era"]) if x["per_era"] else np.nan for x in rs])
        nn = np.nansum([x["n"] for x in rs])
        pooled[n] = pl
        print(f"  {n:<24}{pl:>9.3f}{pe:>14.3f}{nn:>11.0f}")
    noise = next((k for k in names if "noise" in k), None)
    if noise and np.isfinite(pooled.get(noise, np.nan)):
        print(f"\n  against the noise arm (the form originally specified) -- NOT the gate:")
        for n in names:
            if n == noise or not np.isfinite(pooled[n]):
                continue
            print(f"    {n:<24} pooled - noise {pooled[n] - pooled[noise]:+.3f}")
        print("    That comparison is confounded.  Pooled MI has a FLOOR set by the number of")
        print("    eras -- with E eras a label takes only E meanings -- and the noise arm has")
        print("    different within-era structure, so the difference mixes 'pi rotates' with")
        print("    'labels are random within an era'.  The gate is the matched null below.")
    print("\n  GATE R, matched permutation null (each era's label axis permuted independently:")
    print("  era count, sample sizes and within-era structure preserved, only cross-era")
    print("  consistency destroyed).  z near 0 = pi is doing its job, meaning is not inheritable.")
    print(f"  {'arm':<24}{'observed':>10}{'null':>9}{'sd':>8}{'z':>8}{'epochs':>8}   verdict")
    for n in names:
        gp = gate_r_permutation(results[n][0])
        if not np.isfinite(gp["z"]):
            print(f"  {n:<24}{'--':>10}   (no record)"); continue
        v = "PASS" if gp["z"] <= 2.0 else "GATE R FIRES"
        print(f"  {n:<24}{gp['obs']:>10.3f}{gp['null']:>9.3f}{gp['sd']:>8.3f}"
              f"{gp['z']:>8.2f}{gp['eras']:>8d}   {v}")

    print("\n" + "=" * 78)
    print("sym_gain -- the heritable read gain, starting at 0.05 and free to go negative")
    print("=" * 78)
    print(f"  {'arm':<24}{'phase 1':>10}{'phase 2':>10}{'change':>9}{'frac > 0':>10}")
    for n in names:
        p1 = np.nanmean([half(phase_half(r, 0), "sym_gain") for r in results[n]])
        p2 = g(n, "sym_gain")
        print(f"  {n:<24}{p1:>10.4f}{p2:>10.4f}{p2 - p1:>+9.4f}{g(n, 'sym_gain_pos'):>10.3f}")
    base = next((k for k in names if k == "plastic"), None)
    if base:
        b = abs(g(base, "sym_gain"))
        print(f"\n  |gain| MINUS the no-record arm's -- THE LICENSING STATISTIC.")
        print(f"  |gain| rises in every arm including `{base}`, which has no record at all, so a")
        print("  rising magnitude on its own is not evidence of reading.  The sign is absorbable")
        print("  by W1, so magnitude is the statistic and the no-record arm is the baseline.")
        print(f"  {'arm':<26}{'|gain|':>9}{'baseline':>10}{'delta':>9}")
        for n in names:
            if n == base:
                continue
            print(f"  {n:<26}{abs(g(n,'sym_gain')):>9.4f}{b:>10.4f}{abs(g(n,'sym_gain')) - b:>+9.4f}")
    print("  trajectory over phase 2, 6 bins:")
    for n in names:
        r = results[n][0]
        b = [round(float(half(w, "sym_gain")), 4) for w in _bins(r, 6)]
        print(f"    {n:<24}{b}")

    print("\n" + "=" * 78)
    print("store_gain -- how much more the agent wants the preparation a POSITIVE mark endorses")
    print("=" * 78)
    print("  learned vs innate is the BINDING instrument: innate ~ 0 says the genome cannot read")
    print("  the record (which redrawing pi guarantees); learned > 0 says this agent bound it")
    print("  inside its own life.")
    print(f"  {'arm':<24}{'learned':>10}{'innate':>10}{'learned - innate':>18}")
    for n in names:
        le, i_ = g(n, "store_gain"), g(n, "store_gain_innate")
        print(f"  {n:<24}{le:>10.4f}{i_:>10.4f}{le - i_:>+18.4f}")

    print("\n" + "=" * 78)
    print("TRANSMISSION -- two conditioned lines, both on behaviour the agent could not have")
    print("learned or written itself")
    print("=" * 78)
    print("  NO SELF-ECHO is a PROPERTY of the design, not an assumption: a preparation CONSUMES")
    print("  the food cell, so the mark it writes cannot be read for a preparation until food")
    print("  respawns there -- and the reader is then whoever is standing on it.  An agent can")
    print("  never read its own mark about the food it just prepared.")
    print("\n  (i) FIRST-EVER preparations, split by whether a positive mark was already present")
    print(f"  {'arm':<26}{'P(ok|mark)':>12}{'P(ok|none)':>12}{'gap':>9}{'n mark':>9}{'n none':>9}")
    for n in names:
        pos, none, npos, nnone = first_prep_by_mark(P2(results[n][0]))
        gap = pos - none if np.isfinite(pos) and np.isfinite(none) else np.nan
        print(f"  {n:<26}{pos:>12.3f}{none:>12.3f}{gap:>+9.3f}{npos:>9.0f}{nnone:>9.0f}")
    print("\n  (ii) FOLLOWING the record: P(chosen = pi^-1(strongest positive label) | mark present)")
    print("       SPLIT by whether the mark endorses the CORRECT preparation.  Agreeing with a")
    print("       mark that points at the right answer is indistinguishable from being right;")
    print("       the STALE cell -- the mark endorses a preparation that is wrong for this type")
    print("       now -- is the unconfounded one, because following it is a mistake.")
    print("       AND 1/K is the WRONG null for the stale cell.  An agent that simply KNOWS the")
    print("       correct preparation picks it and so never agrees with a stale mark, whatever it")
    print("       reads.  The null is the chance of landing on the endorsed-but-wrong preparation")
    print("       GIVEN you did not pick the correct one: (1 - hit) / (K - 1).")
    print(f"  {'arm':<26}{'all':>8}{'endorse=ok':>12}{'STALE':>8}{'null':>8}{'ratio':>8}{'n stale':>9}")
    for n in names:
        f_, nn = follow_rate(P2(results[n][0]))
        gg, ng, bb, nb = follow_split(P2(results[n][0]))
        h = prep_hit(P2(results[n][0]), True)
        null = (1.0 - h) / (N_PREPS - 1) if np.isfinite(h) else np.nan
        print(f"  {n:<26}{f_:>8.3f}{gg:>12.3f}{bb:>8.3f}{null:>8.3f}"
              f"{(bb / null if null else np.nan):>8.2f}{nb:>9.0f}")
    print("       ratio > 1 = FOLLOWS a mark it should not; ratio < 1 = AVOIDS it.  Either way")
    print("       the label was read: you cannot avoid what you cannot see.  The noise arm is the")
    print("       reference for how far from 1 an unread channel sits.")
    print("\n  preparations-to-first-correct -- CORROBORATING ONLY, over agents that reached 5")
    print(f"  {'arm':<32}{'mean preps':>12}{'censored':>10}{'n':>8}")
    for n in names:
        m, c, nn = nfc(P2(results[n][0]), True)
        print(f"  {n:<32}{m:>12.3f}{c:>10.3f}{nn:>8d}")
    if control:
        for n, r in control.items():
            m, c, nn = nfc(P2(r), True)
            pos, none, npos, nnone = first_prep_by_mark(P2(r))
            f_, fn = follow_rate(P2(r))
            print(f"  {n + '  [sym_gain = 0]':<32}{m:>12.3f}{c:>10.3f}{nn:>8d}")
            print(f"    its (i) P(ok|mark) {pos:.3f} vs P(ok|none) {none:.3f}  gap {pos-none:+.3f}"
                  f"   its (ii) follow {f_:.3f}")

    print("\n" + "=" * 78)
    print("POPULATION and the store")
    print("=" * 78)
    print(f"  {'arm':<24}{'pop p1':>9}{'pop p2':>9}{'inj p2':>9}{'prep hit':>10}"
          f"{'mark dens':>11}{'|mark|':>9}")
    for n in names:
        print(f"  {n:<24}"
              f"{np.nanmean([half(phase_half(r,0),'pop') for r in results[n]]):>9.0f}"
              f"{g(n,'pop'):>9.0f}{g(n,'injections'):>9.2f}"
              f"{np.nanmean([prep_hit(P2(r), True) for r in results[n]]):>10.3f}"
              f"{g(n,'mark_density'):>11.4f}{g(n,'mark_mean_abs'):>9.3f}")


def _bins(run_, k):
    L = phase_half(run_, 1) or run_["log"]
    lo, hi = L[0]["t"], L[-1]["t"]
    step = max(1, (hi - lo) // k)
    return [[r for r in L if b < r["t"] <= b + step] for b in range(lo - 1, hi, step)][:k]


def frozen_selftest(seed=0, verbose=True):
    """With learning OFF, a frozen replay's hit rate must not move across the window.

    If it does, something other than H is changing -- the population, the mapping, or the arm's
    composition -- and the eta = 1 side cannot then be read as learning.  This is the test the
    v3.11 knockout never had: it used a live population, where selection moved the number and the
    reading was attributed to the genome anyway.
    """
    from sim_v3_13 import Config, run as _run
    kw = dict(WORLD); kw.update(mode="plastic", plastic_layers="W2")
    src = _run(Config(seed=seed, **kw), verbose=False,
               phases=[dict(n_steps=1500, chain=False), dict(n_steps=2100, chain=True)])
    if not src.get("era_snaps"):
        print("  frozen self-test: FAIL (no era snapshots -- the run never crossed a boundary)")
        return False
    checks, lines = [], []
    for eta in (0, 1):
        rr = frozen_replay(src, src["era_snaps"][-1]["mapping"], eta)
        curve_, first, last = frozen_curve(rr)
        pops = {w["pop"] for w in rr["log"]}
        gens = {w["max_gen"] for w in rr["log"]}
        drift = abs(last - first)
        lines.append(f"  eta {eta}: hit {first:.3f} -> {last:.3f}  (drift {drift:+.3f})"
                     f"   pop {sorted(pops)}   max_gen {sorted(gens)}")
        if eta == 0:
            checks += [drift <= 0.05, len(pops) == 1, len(gens) == 1]
    if verbose:
        for l in lines:
            print(l)
        print("  required: with eta 0 the hit does not move (drift <= 0.05), and pop and max_gen")
        print("  are single-valued across the window -- nothing but H can change.")
    passed = all(checks)
    print(f"  frozen-replay self-test: {'PASS' if passed else 'FAIL'}")
    return passed


def late_first_prep(L, ff=False):
    """First-preparation hit restricted to first preparations made at least a third of an era
    AFTER a remap.  The raw first-prep hit conflates the genome's quality with how recently the
    mapping moved: an agent whose first preparation lands just after a remap is scored against a
    mapping its lineage has not been selected on.  Measured on plastic seed 0 at prep_every 700,
    the raw number was 0.490 while the same genomes against their own mapping scored 0.817."""
    return rate(L, _k("n_first_ok_late", ff), _k("n_first_late", ff))


def _era_len(run_, era=None):
    """An era is one mapping, so it is cfg.prep_every -- not a constant.  Hardcoding 2000 here
    would silently span three eras once prep_every moved to 700."""
    return int(era if era is not None else run_["cfg"]["prep_every"])


def first_era(run_, era=None):
    """The FIRST mapping era of phase 2 -- steps 0..prep_every after the switch.  Rig check 2(a)
    is read here and only here: once the mapping is known, preparation pays on ANY food, the
    safe/poison fact matters less, and a good learner stops eating raw.  probe_adv (food) decaying
    after that is a GOOD reason, not a broken transition."""
    sw = run_.get("chain_start", 0)
    return window(run_, sw, sw + _era_len(run_, era))


def era_windows(run_, era=None):
    sw, n, e = run_.get("chain_start", 0), run_["n_steps"], _era_len(run_, era)
    return [window(run_, lo, min(lo + e, n)) for lo in range(sw, n, e)]


def summary(results):
    names = list(results)
    w = 24
    nseed = len(results[names[0]])
    print("=" * (20 + w * len(names)))
    print(f"v3.12 -- the preparation world, T=3 K=5.  {nseed} seeds.  Prep hit: chance {CHANCE:.3f}, "
          f"type-blind {TYPE_BLIND:.2f}, full {FULL:.2f}.  Chance safe rate 0.500.  Random-policy "
          f"action share 1/5 in phase 1, 1/10 in phase 2, 0.500 for any preparation.")
    print("=" * (20 + w * len(names)))

    for label, sel in [("PHASE 1 (food only) -- second half", lambda r: phase_half(r, 0)),
                       ("PHASE 2 (chain on) -- second half", lambda r: phase_half(r, 1))]:
        print(f"\n### {label}")
        if label.startswith("PHASE 1"):
            print("    food only: the three preparations are masked, so phase 1 is v3.1's five actions.")
        print(f"{'metric':<20}" + "".join(f"{n:>{w}}" for n in names))
        for lab, f in ROWS:
            print(f"{lab:<20}" + "".join(f"{np.nanmean(ps(results, n, sel, f)):>{w}.3f}" for n in names))

    print("\nper seed:")
    for lab, f, sel, tag in [("safe_rate", safe, lambda r: phase_half(r, 0), "phase 1"),
                             ("probe_adv (food)", lambda L: half(L, "probe_adv_food"), lambda r: phase_half(r, 0), "phase 1"),
                             ("prep hit  FF", lambda L: prep_hit(L, True), lambda r: phase_half(r, 1), "phase 2"),
                             ("founder share of preps", lambda L: founder_share(L, "prep"), lambda r: phase_half(r, 1), "phase 2"),
                             ("prep hit (all agents)", prep_hit, lambda r: phase_half(r, 1), "phase 2"),
                             ("prep hit | A", lambda L: rate(L, "n_ok0", "n_prep0"), lambda r: phase_half(r, 1), "phase 2"),
                             ("prep hit | B", lambda L: rate(L, "n_ok1", "n_prep1"), lambda r: phase_half(r, 1), "phase 2"),
                             ("prep/life", lambda L: half(L, "prep_per_life"), lambda r: phase_half(r, 1), "phase 2"),
                             ("probe_adv (prep)", lambda L: half(L, "probe_adv"), lambda r: phase_half(r, 1), "phase 2"),
                             ("pop", lambda L: half(L, "pop"), lambda r: phase_half(r, 1), "phase 2"),
                             ("injections", lambda L: half(L, "injections"), lambda r: phase_half(r, 1), "phase 2")]:
        print(f"  {lab} ({tag})")
        for n in names:
            print(f"    {n:<22} {np.round(ps(results, n, sel, f), 3).tolist()}")

    _decision_numbers(results)


# ---------------------------------------------------------------- decision numbers

def _decision_numbers(results):
    names = list(results)
    nseed = len(results[names[0]])
    rule = min(SEED_RULE, nseed)
    P1 = lambda r: phase_half(r, 0)
    P2 = lambda r: phase_half(r, 1)
    F, S, PL, CEIL = "fixed", "scrambled", "plastic (W2)", "fixed + B (ceiling)"
    # The acceptance stage runs three arms, the grid stage five.  Every list below is filtered to
    # what is actually in `results`, so the same decision block reads either without a KeyError.
    present = [n for n in VARIANTS if n in results]
    outcome = [n for n in OUTCOME_ARMS if n in results]
    has_null = NULL in results
    v = lambda n, sel, f: ps(results, n, sel, f)
    n_ok = lambda a, b: int(np.sum(np.asarray(a) - np.asarray(b) >= MARGIN))
    # FOUNDER-FREE is the primary reading everywhere below; the all-agents value is printed
    # beside it, in (parentheses), for this build only.
    hitA, hitB = hit_a, hit_b
    FF = lambda f: (lambda L: f(L, True))
    def pair(n, sel, f):
        """(founder-free per seed, all-agents per seed) for one arm."""
        return v(n, sel, FF(f)), v(n, sel, f)
    def show(n, sel, f, fmt="{:.3f}"):
        a, b = pair(n, sel, f)
        return (f"{np.round(a,3).tolist()}  (all {np.round(b,3).tolist()})")

    print("\n" + "-" * 78)
    print(f"DECISION NUMBERS (a difference counts when it is >= {MARGIN} in {rule}/{nseed} seeds)")
    print(f"  THREE LEVELS on prep hit:  chance {CHANCE:.3f}  |  type-blind {TYPE_BLIND:.2f}  |  full {FULL:.2f}")
    print("  Type-blind = 'always prep_k' for a k useful in this era: right for one food type, wrong")
    print("  for the other, no type knowledge.  ACROSS eras it scores only 1/3, because with distinct")
    print("  mappings k is useless in a third of them -- so a genome beats chance only by tracking the")
    print("  era.  A CONJUNCTION shows as BOTH types above 0.5, not one at 1.0 and the other at 0.")
    print("  PRE-REGISTERED PREDICTION: `fixed` sits near 0.5 and crashes in the third of eras where")
    print("  its k goes useless (EV -0.5/meal); `plastic (W2)` clears 0.5 on BOTH types and recovers")
    print("  across remaps.  Recorded before the run.")
    print("-" * 78)

    print("\nrow 0  uninterpretable?  Per phase.  `random policy` is EXEMPT.")
    print("  EXCLUDE on pop < 80 over the half.  INJECTIONS ARE REPORTED, NOT EXCLUSIONARY:")
    print("  the problem injection causes is founder DILUTION of the metrics, not population size,")
    print("  and that is fixed at the metric (every event-weighted number below is founder-free)")
    print("  rather than by discarding the arm.  A non-learning population sits at the floor in")
    print("  this world because value comes only through knowledge -- the same verdict v3.1 gave.")
    for lab, sel in [("phase 1", P1), ("phase 2", P2)]:
        for n in names:
            pop, inj = v(n, sel, lambda L: half(L, "pop")), v(n, sel, lambda L: half(L, "injections"))
            low = np.nanmin(pop) < 80
            flag = "  (null arm, exempt)" if n == NULL else ("  <-- EXCLUDE (pop < 80)" if low else "")
            print(f"  {lab}  {n:<22} pop {np.round(pop,0).tolist()}  inj {np.round(inj,1).tolist()}{flag}")

    cap = WORLD["max_pop"]
    print(f"\n  CAP CHECK -- no outcome arm above 90% of max_pop ({cap}) in phase 2's second half.")
    print("  An arm on the cap cannot express a fitness difference in its population, so the")
    print("  population column stops carrying information for every arm at once.")
    worst, capped = 0.0, []
    for n in outcome:
        pop = v(n, P2, lambda L: half(L, "pop")); frac = np.nanmax(pop) / cap
        worst = max(worst, float(frac))
        if frac > 0.90:
            capped.append(n)
        print(f"    {n:<22} pop {np.round(pop,0).tolist()}   max {frac*100:.0f}% of cap"
              f"{'   <-- CAPPED' if frac > 0.90 else ''}")
    if capped:
        print(f"    FAIL: {capped} above 90%.  Raise max_pop to 1200 and re-estimate the runtime")
        print("    before reading anything that depends on population.")
    else:
        print(f"    PASS: worst arm at {worst*100:.0f}% of the cap.")

    print("\n  FOUNDER SHARE OF EVENTS -- the size of the dilution the founder-free numbers remove.")
    print("  An injected agent is a fresh random genome; its own events are excluded below, its")
    print("  children's are not.  A high share is not a reason to discard the arm, but it does say")
    print("  how far the all-agents number has been pulled toward chance.")
    print(f"    {'arm':<22}{'phase 1 eats':>16}{'phase 2 eats':>16}{'phase 2 preps':>16}")
    for n in names:
        print(f"    {n:<22}"
              f"{np.nanmean(v(n, P1, lambda L: founder_share(L, 'eat'))):>16.3f}"
              f"{np.nanmean(v(n, P2, lambda L: founder_share(L, 'eat'))):>16.3f}"
              f"{np.nanmean(v(n, P2, lambda L: founder_share(L, 'prep'))):>16.3f}")

    print("\nrow 1a  PHASE-1 GATE = v3.1.  A STOP ROW: if it fails, nothing below is read.")
    sp, sf = v(PL, P1, FF(safe)), v(F, P1, FF(safe))          # founder-free
    pf = v(PL, P1, lambda L: half(L, "probe_adv_food"))
    fpop, finj = v(F, P1, lambda L: half(L, "pop")), v(F, P1, lambda L: half(L, "injections"))
    excluded = (fpop < 80) | (finj > 0)
    V31_HI = 0.56
    print(f"  safe_rate  plastic {np.round(sp,3).tolist()}   fixed {np.round(sf,3).tolist()}   (v3.1: 0.60-0.66 vs 0.51-0.56)")
    print(f"  all-agents  plastic {np.round(v(PL,P1,safe),3).tolist()}   fixed {np.round(v(F,P1,safe),3).tolist()}")
    if excluded.any():
        print(f"  ROW-0 FALLBACK in seed(s) {np.where(excluded)[0].tolist()}: those read against v3.1's"
              f" published range, conservative end {V31_HI}.")
    for i in range(nseed):
        d = sp[i] - (V31_HI if excluded[i] else sf[i])
        print(f"    seed {i}: {d:+.3f} ({'published' if excluded[i] else 'measured'})   {'PASS' if d >= MARGIN else 'FAIL'}")
    print(f"  probe_adv (food) plastic {np.round(pf,3).tolist()}   (target >= 1.0)")

    print("\nrow 1b  THE GENETIC BASELINE -- MEASURED, NOT A STOP.")
    print("        In a six-mapping space the genes DO track the mapping, and shortening the era")
    print("        does not stop them: sorting over standing variation completes well inside a")
    print("        third of an era.  So this row no longer halts the reading.  It reports how")
    print("        much of the standing hit rate the genome already carries, and the learner's")
    print("        contribution is read ABOVE it.  Three numbers make the baseline:")
    print("          (1) `fixed` first-preparation hit, LATE in the era -- the genome alone;")
    print("          (2) the per-era A+B sum -- above 1 means the genomes condition on type;")
    print("          (3) row 3b's matched/shuffled genome hit with learning off.")
    hf = v(F, P2, FF(prep_hit))
    tbl = v(F, P2, FF(type_blind_level))
    print(f"  fixed prep hit {np.round(hf,3).tolist()}   all-agents {np.round(v(F,P2,prep_hit),3).tolist()}")
    for t in range(N_TYPES):
        print(f"    per type {t}  {np.round(v(F, P2, FF(lambda L, ff=False, t=t: hit_t(L, t, ff))),3).tolist()}"
              f"   share {np.round(v(F, P2, FF(lambda L, ff=False, t=t: type_shares(L, ff)[t])),3).tolist()}")
    print(f"  ITS OWN TYPE-BLIND LEVEL = max over the T preparation shares {np.round(tbl,3).tolist()}"
          f"   -- NOT 0.5 when encounters are skewed.  fixed - its own level"
          f" {np.round(hf - tbl,3).tolist()}")
    fired = int(np.sum(hf > 0.55))
    print(f"  above 0.55 in {fired}/{nseed}   (recorded, not a stop -- see row 3b for how much of")
    print("  this the genome carries and how much the rule adds)")
    print("  Read the per-type split: one type high and the rest near 0 is type-blind (allowed);")
    print(f"  all {N_TYPES} above {TYPE_BLIND:.3f} in `fixed` would be genes holding the conjunction.")
    print("  READ IT PER ERA.  The phase half spans several mapping eras, so a genome that")
    print("  switches its preparation between them averages into a FALSE flat reading.")
    print("  A pure mixture of type-blind genotypes has the per-type hits summing to <= 1;")
    print("  a SUM ABOVE 1 means the genomes are conditioning on food type, whatever the")
    print("  phase-half aggregate says.")
    for n in (F, PL, S):
        if n not in results:
            continue
        for i, r in enumerate(results[n]):
            rows = [(round(rate(w, "n_ok0", "n_prep0"), 2), round(rate(w, "n_ok1", "n_prep1"), 2))
                    for w in era_windows(r) if w]
            both = sum(1 for a_, b_ in rows if a_ > TYPE_BLIND and b_ > TYPE_BLIND)
            sums = [a_ + b_ for a_, b_ in rows]
            shown = rows if len(rows) <= 8 else rows[:8]
            tail_ = "" if len(rows) <= 8 else f" ... (+{len(rows)-8} more)"
            print(f"    {n + ' seed ' + str(i):<22} both > {TYPE_BLIND} in {both}/{len(rows)} eras"
                  f"   mean A+B {np.mean(sums):.2f}   max {np.max(sums):.2f}")
            print(f"    {'':<22} (A,B) {shown}{tail_}")

    print(f"\n  THE BASELINE, reported per seed:")
    fp_f = v(F, P2, FF(late_first_prep))          # late-in-era: the genome reading
    fp_s = v(S, P2, FF(late_first_prep)) if S in results else None
    print(f"    (i)  fixed standing prep hit     {np.round(hf,3).tolist()}")
    print(f"    (ii) first-prep hit, LATE in the era -- the INNATE policy, before any learning.")
    print(f"         fixed     {np.round(fp_f,3).tolist()}"
          f"   (raw {np.round(v(F,P2,FF(first_prep_hit)),3).tolist()},"
          f" its type-blind level {np.round(tbl,3).tolist()})")
    if fp_s is not None:
        tbs = v(S, P2, FF(type_blind_level))
        print(f"         scrambled {np.round(fp_s,3).tolist()}"
              f"   (raw {np.round(v(S,P2,FF(first_prep_hit)),3).tolist()},"
              f" its type-blind level {np.round(tbs,3).tolist()})")
    print("    (ii) is the sharper of the two: the standing hit rate mixes genome with whatever")
    print("    the living have learned, while the first preparation of a life is genome alone.")
    print("    A `nan` here means the checkpoint predates the field -- run checkpoint_audit.")
    print("  PRIMARY INSTRUMENT for this row: prep_gain innate -- the GENOME's preference for the")
    print("  correct preparation on a synthetic 'type f underfoot' observation.  No gating and no")
    print("  declining enter it, so unlike the A+B sum it localises the type-conditioning to the")
    print("  CHOICE of preparation.  `random policy` is the zero; `fixed + B` sits near zero")
    print("  because its knowledge is in the private table, not in synapses -- that is the check")
    print("  that the instrument reads synapses and nothing else.")
    for n in names:
        print(f"    {n:<22} {np.round(v(n, P2, lambda L: half(L, 'prep_gain_innate')), 3).tolist()}")

    print("\nrow 1c  STANDING VARIATION -- is the premise of this world true for THIS population?")
    print(f"        The space is P({N_PREPS},{N_TYPES}) = {N_MAPPINGS} mappings.  If the population")
    print("        holds genotypes for most of them, a remap needs no adaptation and survival")
    print("        sorting promotes a matching one, exactly as in the six-mapping world.  The")
    print("        premise of v3.12 is that it CANNOT.  This measures it rather than arguing it.")
    print("        `above` is the direct statement: how many of the mappings the population would")
    print("        score above type-blind on.  A handful means the space exceeds standing")
    print("        variation.  A large number means v3.12 has NOT achieved what it was built for,")
    print("        and that is the finding whatever row 3b then says.")
    print(f"    {'arm':<22}{'triples':>9}{'valid':>7}{'held>=1':>9}{'>=5':>6}{'>=20':>7}"
          f"{'above/' + str(N_MAPPINGS):>11}")
    for n in names:
        g = lambda key: np.nanmean(v(n, P2, lambda L: half(L, key)))
        print(f"    {n:<22}{g('sv_triples'):>9.1f}{g('sv_valid'):>7.1f}{g('sv_held1'):>9.1f}"
              f"{g('sv_held5'):>6.1f}{g('sv_held20'):>7.1f}{g('sv_above'):>11.1f}")
    print("    (`nan` means the checkpoint predates the probe -- run checkpoint_audit)")

    print("\nrow 2  RIG CHECKS -- nothing below is read until these are clean.")
    print("  (a) food learning survives the switch.  READ ON WHOLE-PHASE FOUNDER-FREE SAFE RATE")
    print("      (plastic - fixed >= 0.03).  At prep_every 350 a single era is ~350 steps, far too")
    print("      few meal events to read a rate on, so the window is the whole of phase 2.  The")
    print("      corrected probe in the first 500 steps is CORROBORATING.  The v3.1 probe form was")
    print("      contaminated here -- it subtracts the best OTHER action, and with three")
    print("      preparations live that term moves with food type.")
    sfe, sffe = v(PL, P2, FF(safe)), v(F, P2, FF(safe))
    print(f"    safe (whole phase 2)  plastic {np.round(sfe,3).tolist()}  fixed {np.round(sffe,3).tolist()}"
          f"   diff {np.round(sfe - sffe,3).tolist()}   >= +{MARGIN} in {n_ok(sfe, sffe)}/{nseed}"
          f"   {'PASS' if n_ok(sfe, sffe) >= rule else 'FAIL'}")
    first500 = lambda r: window(r, r.get("chain_start", 0), r.get("chain_start", 0) + 500)
    for n in (PL, S):
        print(f"    corroborating: {n:<14} probe_adv (food), first 500 steps "
              f"{np.round(v(n, first500, lambda L: half(L,'probe_adv_food')),3).tolist()}"
              f"   first era {np.round(v(n, first_era, lambda L: half(L,'probe_adv_food')),3).tolist()}")
    print("      EAT -> PREP SHIFT TIMING (prep share in 250-step bins after the switch):")
    for n in names:
        print(f"    {n:<22} {shift_timing(results[n][0])}")
    print("      PREP SHARE BY ERA:")
    for n in names:
        print(f"    {n:<22} {[round(float(half(w,'prep_share')),3) for w in era_windows(results[n][0])]}")
    print("  (b) the opportunity exists:  prepared meals per life >= 3 in `fixed`")
    plf = v(F, P2, lambda L: half(L, "prep_per_life"))
    print(f"    fixed prep/life {np.round(plf,2).tolist()}   {'PASS' if np.all(plf >= 3.0) else 'FAIL'}")
    for n in present:
        print(f"    {n:<22} prep/life {np.round(v(n,P2,lambda L: half(L,'prep_per_life')),2).tolist()}")
    print("  (c) P(prep | on food) against the per-phase null (analytic 3/8 = 0.375 in phase 2)")
    if has_null:
        base = np.nanmean(v(NULL, P2, lambda L: half(L, "prep_on_food")))
        print(f"    null measured {base:.3f}")
    else:
        print("    null measured: `random policy` not in this stage -- analytic 0.375 stands in")
    for n in names:
        print(f"    {n:<22} {np.round(v(n,P2,lambda L: half(L,'prep_on_food')),3).tolist()}")

    print("\n" + "=" * 78)
    print("row 3  THE CONJUNCTION -- a dense, immediate, two-sided task")
    print("=" * 78)
    hp, hs = v(PL, P2, FF(prep_hit)), v(S, P2, FF(prep_hit))
    pl, fl = v(PL, P2, lambda L: half(L, "prep_per_life")), plf
    print("  ABSTENTION CHECK FIRST.  Fires only if prepared meals per life < 0.8x `fixed`")
    print("  AND (hit <= fixed OR pop <= fixed): preparing less while scoring and living BETTER is")
    print("  a learner declining bad bets, not one abstaining from the task.")
    popp, popf = v(PL, P2, lambda L: half(L, "pop")), v(F, P2, lambda L: half(L, "pop"))
    ratio = pl / np.maximum(fl, 1e-9)
    abst = (ratio < 0.80) & ((hp <= hf) | (popp <= popf))
    print(f"    prep/life plastic {np.round(pl,2).tolist()}  fixed {np.round(fl,2).tolist()}"
          f"   ratio {np.round(ratio,2).tolist()}   (< 0.80 in {int(np.sum(ratio < 0.80))}/{nseed})")
    print(f"    hit  plastic {np.round(hp,3).tolist()}  fixed {np.round(hf,3).tolist()}"
          f"      pop  plastic {np.round(popp,0).tolist()}  fixed {np.round(popf,0).tolist()}")
    print(f"    ABSTENTION FIRES in {int(np.sum(abst))}/{nseed} seed(s) {np.where(abst)[0].tolist()}")
    print("  the result lines:")
    print(f"    plastic - fixed:     {np.round(hp - hf,3).tolist()}   >= +{MARGIN} in {n_ok(hp, hf)}/{nseed}")
    print(f"    plastic - scrambled: {np.round(hp - hs,3).tolist()}   >= +{MARGIN} in {n_ok(hp, hs)}/{nseed}   <- scrambled carries it")
    print(f"    prep hit  plastic {np.round(hp,3).tolist()}  fixed {np.round(hf,3).tolist()}  scrambled {np.round(hs,3).tolist()}")
    print(f"    all-agents        {np.round(v(PL,P2,prep_hit),3).tolist()}        "
          f"{np.round(v(F,P2,prep_hit),3).tolist()}            {np.round(v(S,P2,prep_hit),3).tolist()}")
    print(f"  PER TYPE -- a conjunction is BOTH above {TYPE_BLIND:.2f}, not one at 1.0 and one at 0.")
    print("  Each arm's OWN type-blind level is printed too: with skewed encounters the level is")
    print("  max(share_A, share_B), and a hit at that level carries no type knowledge.")
    for n in names:
        per = [v(n, P2, FF(lambda L, ff=False, t=t: hit_t(L, t, ff))) for t in range(N_TYPES)]
        allabove = np.ones(nseed, dtype=bool)
        for pt in per:
            allabove &= (pt > TYPE_BLIND)
        print(f"    {n:<22} " + "  ".join(f"t{t} {np.round(per[t],3).tolist()}"
                                          for t in range(N_TYPES)))
        print(f"    {'':<22} all {N_TYPES} above {TYPE_BLIND:.3f} in {int(allabove.sum())}/{nseed}"
              f"   its type-blind level {np.round(v(n,P2,FF(type_blind_level)),3).tolist()}")
    pr = v(PL, P2, lambda L: half(L, "probe_adv"))
    ao, ay = v(PL, P2, lambda L: half(L, "hit_old")), v(PL, P2, lambda L: half(L, "hit_young"))
    ag = v(PL, P2, lambda L: float(curve(L, "att", True)[-1] - curve(L, "att", True)[0]))
    print(f"  probe_adv (prep) {np.round(pr,3).tolist()}   (> 0 in {int(np.sum(pr > 0))}/{nseed})")
    print("    NOTE: probe_adv is computed over the LIVING, so it is itself partly")
    print("    survivorship-selected.  The survivor curve below is not.")
    print("  SINCE-REMAP CURVE -- hit against preparation number since the last remap.  The")
    print("  prediction is a DIP at 1 and recovery within ~5 preparations: at a remap the learned")
    print("  H is now wrong, so a learner must pay for the change and then earn it back.  A flat")
    print("  curve means nothing is being relearned within a life; a curve that never recovers")
    print("  means the era is shorter than the learner needs.")
    print(f"    {'arm':<22}{'prep 1':>9}{'prep 5':>9}{'prep 10':>9}{'1->5':>8}{'1->10':>8}")
    for n in names:
        C = np.array([curve(P2(r), "rec", True) for r in results[n]], dtype=float)
        m = np.nanmean(C, 0)
        if len(m) >= 10:
            print(f"    {n:<22}{m[0]:>9.3f}{m[4]:>9.3f}{m[9]:>9.3f}"
                  f"{m[4]-m[0]:>+8.3f}{m[9]-m[0]:>+8.3f}")
        print(f"    {'':<22} full curve {np.round(m,3).tolist()}")

    print(f"  within-life signature -- at least one, in {rule}/{nseed}:")
    print(f"    hit-in-life curve gain {np.round(ag,3).tolist()}   (> 0 in {int(np.sum(ag > 0))}/{nseed})")
    print(f"    hit_old - hit_young    {np.round(ao - ay,3).tolist()}   (> 0 in {int(np.sum(ao > ay))}/{nseed})")

    print("\n  SURVIVORSHIP DIAGNOSTICS -- REQUIRED.  `plastic - scrambled` on hit rate is a")
    print("  SURVIVORSHIP-CONTAMINATED contrast (agents whose random H happens to help live")
    print("  longer, enriching the standing population without anything being learned).  It stays")
    print("  required, but the ATTRIBUTION is carried by the within-agent lines below.")
    print("  (i) SURVIVOR CURVE -- preps 1-5 vs 6-10, over agents that REACHED 10 preparations.")
    print("      Every agent counted contributes both halves of its own curve, so a rise is the")
    print("      same individuals later in their own lives, not a different sample.")
    print("      REQUIRED: rising in plastic, flat in scrambled.")
    print(f"    {'arm':<22}{'preps 1-5':>11}{'preps 6-10':>12}{'rise':>9}{'n agents':>10}")
    for n in names:
        rows = [surv_curve(P2(r), True) for r in results[n]]
        e = np.nanmean([x[0] for x in rows]); l = np.nanmean([x[1] for x in rows])
        d = np.array([x[2] for x in rows], dtype=float); nn = int(np.sum([x[3] for x in rows]))
        print(f"    {n:<22}{e:>11.3f}{l:>12.3f}{np.nanmean(d):>9.3f}{nn:>10d}   per seed {np.round(d,3).tolist()}")
    print(f"  (ii) SURVIVOR-CONDITIONED SINCE-REMAP -- CORROBORATING ONLY, {SR_W}-preparation window.")
    print(f"       Counted only if it made {SR_W} preparations BEFORE a remap and {SR_W} after, so a fall")
    print("       and recovery cannot be a change of sample.  The population-level since-remap")
    print("       curve below IS open to that objection; this line is not.")
    print(f"    {'arm':<22}{'8 before':>10}{'8 after':>10}{'change':>9}{'n agent-remaps':>16}")
    thin = []
    for n in names:
        rows = [surv_remap(P2(r), True) for r in results[n]]
        pre = np.nanmean([x[0] for x in rows]); post = np.nanmean([x[1] for x in rows])
        d = np.array([x[2] for x in rows], dtype=float)
        per_seed_n = [x[3] for x in rows]; nn = int(np.sum(per_seed_n))
        if min(per_seed_n) < SRM_MIN_N:
            thin.append((n, per_seed_n))
            print(f"    {n:<22}{'--':>10}{'--':>10}{'--':>9}{nn:>16d}"
                  f"   DROPPED: n per seed {per_seed_n}, below {SRM_MIN_N}")
            continue
        print(f"    {n:<22}{pre:>10.3f}{post:>10.3f}{np.nanmean(d):>9.3f}{nn:>16d}"
              f"   per seed {np.round(d,3).tolist()}")
    if thin:
        print(f"    The line is DROPPED where n < {SRM_MIN_N} in any seed, never narrowed back: a")
        print("    narrower window does not measure the same thing less precisely, it measures the")
        print("    fall across a remap WITHOUT the recovery, which is why the v3.11 4-window read")
        print("    negative in every arm and worst in the arms that had learned most.")

    print("  (iii) FIRST-PREPARATION HIT -- the agent has learned nothing, so this reads the innate")
    print("       policy the population carries (and shows the type-blind floor directly).")
    print("       READ THE `late` COLUMN.  A first preparation made just after a remap is scored")
    print("       against a mapping the agent's lineage has not been selected on, so the raw")
    print("       number mixes the genome's quality with how recently the mapping moved.  `late`")
    print("       counts only first preparations at least a third of an era after a remap.")
    print(f"    {'arm':<22}{'late':>18}{'raw':>18}{'type-blind level':>20}")
    for n in names:
        print(f"    {n:<22}{str(np.round(v(n, P2, FF(late_first_prep)),3).tolist()):>18}"
              f"{str(np.round(v(n, P2, FF(first_prep_hit)),3).tolist()):>18}"
              f"{str(np.round(v(n, P2, FF(type_blind_level)),3).tolist()):>20}")
    if CEIL in results:
        hc = v(CEIL, P2, prep_hit)
        print(f"  ceiling {np.round(hc,3).tolist()}   (reference: hand-wired exact credit, forced argmax)")

    print("\nrow 3b  THE ATTRIBUTION LINE -- FROZEN-POPULATION REPLAY.")
    print("        Genomes snapshotted at an ERA BOUNDARY, so the population has just lived a")
    print("        whole era under that mapping and is sorted for it.  Replayed for 300 steps")
    print("        with births, deaths and injection ALL DISABLED -- energy is tracked and spent")
    print("        but is not lethal -- on the matched mapping and on a shuffled one, with")
    print("        learning off (eta 0) and on (eta 1).  NOTHING CAN CHANGE BUT H.")
    print("        The SHUFFLED pair carries the claim: on a mapping no genotype was sorted for,")
    print("        eta 0 is the genetic floor and eta 1 is what the rule adds within a life.")
    print("        REQUIRED: eta1 - eta0 >= 0.10 on SHUFFLED in EVERY seed.")
    print("        Why 300 steps: within-life learning needs ~5 preparations to show (the")
    print("        since-remap curve recovers by preparation 5).  The v3.11 10-step window could")
    print("        not contain that, so it failed as an INSTRUMENT rather than returning a")
    print("        negative -- and its run-end snapshot sat mid-era, only partly sorted")
    print("        (`fixed` seed 0 scored 0.24 on its own mapping).  analysis.frozen_selftest")
    print("        holds the line: with eta 0 the hit must not move across the window.")
    gaps = []
    for n in (PL, S, F):
        if n not in results:
            continue
        try:
            rows = frozen_knockout(results, n)
        except Exception as exc:
            print(f"    {n:<20} frozen replay failed: {type(exc).__name__}: {exc}")
            continue
        for k in rows:
            if k.get("missing"):
                print(f"    {n + ' s' + str(k['seed']):<20} no era snapshots in this run -- it")
                print(f"    {'':<20} predates them.  Re-run with the current sim.py.")
                continue
            m0, m1 = k["matched_0"], k["matched_1"]
            s0, s1 = k["shuffled_0"], k["shuffled_1"]
            gap = s1["hit"] - s0["hit"]
            if n == PL:
                gaps.append(gap)
            print(f"    {n + ' s' + str(k['seed']):<20} boundary t={k['t']}  map {k['mapping']}"
                  f"  snapshot {k['n_snap']} of {k['n_pop']} agents  pop {m0['pop']:.0f} frozen")
            print(f"    {'':<20}   MATCHED   eta0 {m0['hit']:.3f}  eta1 {m1['hit']:.3f}"
                  f"   gap {m1['hit']-m0['hit']:+.3f}")
            print(f"    {'':<20}   SHUFFLED  eta0 {s0['hit']:.3f}  eta1 {s1['hit']:.3f}"
                  f"   gap {gap:+.3f}"
                  + ("   PASS" if (n == PL and gap >= 0.10) else ("   FAIL" if n == PL else "")))
            print(f"    {'':<20}   shuffled curves  eta0 {np.round(s0['curve'],3).tolist()}")
            print(f"    {'':<20}                    eta1 {np.round(s1['curve'],3).tolist()}")
    if gaps:
        print(f"    REQUIREMENT (plastic, shuffled, eta1 - eta0 >= 0.10): "
              f"{sum(g >= 0.10 for g in gaps)}/{len(gaps)}   "
              f"{'PASS' if all(g >= 0.10 for g in gaps) else 'FAIL'}")
        print(f"    THE CLAIM LINE.  Record this number: it is the v3.11 reference the v3.12")
        print(f"    result must not shrink below.   mean gap {np.mean(gaps):+.3f}")
    print(f"    reference levels: chance {CHANCE:.3f}, type-blind {TYPE_BLIND:.2f}")

    print("\nrow 4  GENE ROWS (corroborating only)")
    for g in ("eta2", "lam2", "eta1"):
        print(f"  {g}   phase 1 -> phase 2")
        for n in outcome:
            print(f"    {n:<22} {np.round(v(n,P1,lambda L: half(L,g)),3).tolist()} -> {np.round(v(n,P2,lambda L: half(L,g)),3).tolist()}")
    print("  unwired scaffold genes -- drift scale for a heritable scalar:")
    for g in ("nav_dir", "nav_here"):
        print(f"    {g:<12} " + "  ".join(f"{n}: {np.nanmean(v(n,P2,lambda L: half(L,g))):.2f}" for n in (F, PL)))

    print("\nrow 5  if row 3 is NULL with rows 1-2 clean, that is the first earned statement about")
    print("       the learner's limit: the rule does not hold a two-item conjunction even where the")
    print("       opportunity is dense, the credit immediate and two-sided, and no approach")
    print("       behaviour is required.  Spend the one rule-form change there.")
    print("-" * 78)


# ---------------------------------------------------------------- curves / plots

def curves(results, key, xlabel, show=True):
    label = {"att": "preparation number in an agent's life",
             "rec": "preparations since the last remap",
             "meal": "meal number in an agent's life"}[key]
    base = CHANCE if key in ("att", "rec") else 0.5
    sel = (lambda r: phase_half(r, 0)) if key == "meal" else (lambda r: phase_half(r, 1))
    phase = "phase 1" if key == "meal" else "phase 2"
    print(f"\n{label}, {phase} second half, event-weighted, mean over seeds (baseline {base:.3f}):")
    for name, res in results.items():
        C = np.array([curve(sel(r), key) for r in res])
        m = np.nanmean(C, 0)
        print(f"  {name:<24} {np.round(m, 3).tolist()}   gain = {m[-1] - m[0]:+.3f}")
    if plt is None or not show:
        return
    plt.figure(figsize=(8, 4.5))
    for name, res in results.items():
        C = np.array([curve(sel(r), key) for r in res])
        m = np.nanmean(C, 0)
        plt.plot(range(1, len(m) + 1), m, marker="o", color=COLORS.get(name), label=name)
        plt.fill_between(range(1, len(m) + 1), np.nanmin(C, 0), np.nanmax(C, 0), color=COLORS.get(name), alpha=0.10)
    plt.axhline(base, color="grey", lw=0.5)
    plt.xlabel(xlabel); plt.ylabel("rate"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()


def series(res, key):
    t = np.array([r["t"] for r in res[0]["log"]])
    Y = np.array([[r[key] for r in run_["log"]] for run_ in res], dtype=float)
    return t, Y


def plot_results(results):
    if plt is None:
        print("matplotlib unavailable; skipping plots")
        return
    keys = [("safe_rate", "Safe-eating rate (chance = 0.5)"),
            ("probe_adv_food", "Within-agent counterfactual on FOOD (v3.1's probe)"),
            ("pop", "Population"),
            ("prep_share", "Preparation share of meals"),
            ("recipe_hit", f"Prep hit (chance {CHANCE:.2f}, type-blind {TYPE_BLIND:.2f})"),
            ("probe_adv", "Within-agent counterfactual on the PREPARATION conjunction"),
            ("eta2", "Output-layer learning-rate gene"),
            ("lam2", "Output-layer eligibility-trace gene"),
            ("h_norm", "Mean |H|")]
    any_run = next(iter(results.values()))[0]
    sw = any_run.get("chain_start", 0)
    fig, axes = plt.subplots(5, 2, figsize=(13, 18))
    for ax, (key, title) in zip(axes.ravel(), keys):
        for name, res in results.items():
            t, Y = series(res, key)
            if not np.isfinite(Y).any():
                continue
            Ys = np.array([smooth(y) for y in Y])
            ax.plot(t, np.nanmean(Ys, 0), color=COLORS.get(name), label=name)
            ax.fill_between(t, np.nanmin(Ys, 0), np.nanmax(Ys, 0), color=COLORS.get(name), alpha=0.10)
        if sw:
            ax.axvline(sw, color="k", lw=1.5)
        for c in any_run["recipe_changes"]:
            ax.axvline(c, color="k", ls="--", lw=0.8)
        ax.set_title(title, fontsize=10); ax.set_xlabel("step")
    axes[0, 0].axhline(0.5, color="grey", lw=0.5)
    axes[2, 0].axhline(CHANCE, color="grey", lw=0.5)
    axes[0, 0].legend(fontsize=7)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    plt.suptitle("solid black = the chain switches on; dashed = recipe change", fontsize=10)
    plt.tight_layout(); plt.show()
