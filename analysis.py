"""
v3.11 analysis -- the preparation world, after the v3.10 acceptance run.

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

from sim import Config, run, N_PREPS, PREP0

# ---------------------------------------------------------------------------
# the world -- v3.1 metabolism throughout, the chain at the agreed cover targets
# ---------------------------------------------------------------------------
WORLD = dict(
    # phase 1 is v3.1, unchanged
    flip_every=300, eta_init=0.2, hidden=24,
    spawn_per_patch=3.0, food_value=0.7, poison_value=0.5,
    repro_threshold=3.0, repro_cost=1.5, max_energy=5.0, max_pop=800, init_pop=300,
    # phase 2 adds three preparations.  Nothing else changes -- no new inputs, no items,
    # stations or nuts.  The fact to be learned sits on EVERY meal.
    # prep_value 1.0 against prep_fail 0.5 puts the CHANCE EV of a preparation at exactly zero:
    # (1/3)(+1.0) + (2/3)(-0.5) = 0.  Eating raw is +0.10 at chance and +0.70 knowing the flip, so
    # preparation now pays ONLY through knowledge of the mapping, and a population cannot ride the
    # preparation payoff up to the cap without it.  (v3.10 acceptance ran prep_value 1.5, where a
    # chance preparation paid +0.17 and every outcome arm sat at 675-799 against a cap of 800.)
    prep_value=1.0, prep_fail=0.5,
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
    # uniform over the AVAILABLE actions: 5 in phase 1 (preparations masked), 8 in phase 2.
    # So the conditional null is 1/5 then 1/8 per action, and 3/8 for "any preparation".
    # Exempt from row 0: a random walker belongs at the population floor.
    "random policy":       _v(mode="random"),
    "fixed":               _v(mode="fixed"),
    # the control that carries row 3: same plasticity, same H magnitudes, random-sign m
    "scrambled":           _v(mode="plastic", plastic_layers="W2", scramble=True),
    "plastic (W2)":        _v(mode="plastic", plastic_layers="W2"),
    # hand-wired (food type, preparation) table with exact credit, forcing its argmax once it has
    # evidence.  A policy override, the honest analogue of v2's veto.  Reference, not matched.
    "fixed + B (ceiling)": _v(mode="fixed", private_mem=True),
}

NULL = "random policy"
OUTCOME_ARMS = [n for n in VARIANTS if n != NULL]

COLORS = {"random policy": "tab:grey", "fixed": "tab:red", "scrambled": "black",
          "plastic (W2)": "tab:blue", "fixed + B (ceiling)": "tab:purple"}

# Three levels on the preparation task, and the middle one is the one to watch:
CHANCE = 1.0 / 3.0        # a random preparation
TYPE_BLIND = 0.5          # "always prep_k" for a k that is useful in this era: right for one food
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
    a, b = (("f_n_prep0", "f_n_prep1"), ("f_n_ok0", "f_n_ok1")) if ff else \
           (("n_prep0", "n_prep1"), ("n_ok0", "n_ok1"))
    if not has(L, a[0]):
        return np.nan
    n = float(np.sum([r[a[0]] + r[a[1]] for r in L]))
    k = float(np.sum([r[b[0]] + r[b[1]] for r in L]))
    return k / n if n else np.nan


def hit_a(L, ff=False):
    return rate(L, _k("n_ok0", ff), _k("n_prep0", ff))


def hit_b(L, ff=False):
    return rate(L, _k("n_ok1", ff), _k("n_prep1", ff))


def type_share_a(L, ff=False):
    """Share of preparations made on food type A.  The type-blind level is max(share, 1 - share),
    NOT 0.5: with skewed encounters, "always prep_k" for the commoner type beats 0.5 without any
    type knowledge at all."""
    if not has(L, _k("n_prep0", ff)):
        return np.nan
    a = float(np.sum([r[_k("n_prep0", ff)] for r in L]))
    b = float(np.sum([r[_k("n_prep1", ff)] for r in L]))
    return a / (a + b) if (a + b) else np.nan


def type_blind_level(L, ff=False):
    sa = type_share_a(L, ff)
    return max(sa, 1.0 - sa) if np.isfinite(sa) else np.nan


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
    from sim import SURV_EARLY
    e = float(np.sum([r[_k("n_surv_early", ff)] for r in L])) / (SURV_EARLY * n)
    l = float(np.sum([r[_k("n_surv_late", ff)] for r in L])) / (5 * n)
    return e, l, l - e, int(n)


def surv_remap(L, ff=False):
    """SURVIVOR-CONDITIONED since-remap.  Only agents that made SR_W preparations BOTH before and
    after the same remap are counted, and each contributes its own rate on either side.  The
    population-level since-remap curve is open to the objection that the agents alive at
    preparation 1 are not the ones alive at 10; this is not, because it is the same agent across
    the same remap."""
    from sim import SR_W
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


KO_STEPS = 10       # the knockout replay window, in steps.  50 was still too long: max_gen
                    # reached 4.0 inside it, i.e. four generations of selection on the pinned
                    # mapping, which is sorting and not the genome.  At 10 steps a standing
                    # population of several hundred still lays down well over a thousand
                    # preparation events, so the rate is not sample-starved.


def knockout(results, name, steps=None, seed_offset=1000):
    """Replay late genomes with eta_scale = 0: same brains, no learning, MAPPING PINNED.

    THE MAPPING IS THE WHOLE POINT.  World.__init__ draws the mapping from the world rng, so the
    old form of this row -- replay under a fresh seed -- scored the genomes against a mapping they
    had never been selected on and reported it as "the genome carries nothing".  Measured on
    plastic seed 0 at prep_every 700, the same genomes score:

        0.844 (A 0.899, B 0.798)  against their OWN final mapping   -- a real conjunction
        0.296 (A 0.546, B 0.093)  against that mapping SHUFFLED     -- below chance, as a
                                                                       committed genome should be

    Both are reported, because the PAIR is the signature: a genome holding the conjunction for one
    mapping scores high on it and below chance on the swap.  A genome holding nothing scores near
    its type-blind level on both.  `replay_mapping_selftest` in sim.py guards the pinning.

    THE WINDOW IS KO_STEPS AND NOTHING MORE.  eta_scale = 0 stops learning but NOT reproduction, and this
    population turns over fast: max_gen reached 4.4 within 500 steps of a replay, which is enough
    generations for selection to re-adapt to the pinned mapping.  A 500-step read at prep_every
    350 showed a NON-PLASTIC `fixed` genome scoring 0.775 on its own mapping and 0.730 on the
    swap, with its per-type hits failing to swap -- impossible for a policy that cannot change,
    and the signature of re-evolution.  pop and max_gen are printed so that stays visible: if
    max_gen has moved, the number is not the genome.
    """
    from sim import Config, run as _run
    out = []
    for i, r in enumerate(results[name]):
        fm = tuple(r["final_mapping"])
        n_steps = int(steps if steps is not None else KO_STEPS)
        row = dict(seed=r["cfg"]["seed"], mapping=fm)
        for tag, mp in (("matched", fm), ("shuffled", (fm[1], fm[0]))):
            for eta in (0.0, 1.0):
                cfg = dict(r["cfg"]); cfg.pop("seed", None); cfg.pop("n_steps", None)
                cfg["eta_scale"], cfg["force_mapping"] = eta, mp
                cfg["log_every"] = n_steps      # exactly one bin, and it is the whole replay
                rr = _run(Config(seed=r["cfg"]["seed"], **cfg), verbose=False,
                          init_genomes=r["final"], phases=[dict(n_steps=n_steps, chain=True)])
                L = rr["log"][:1]                  # the FIRST log bin only
                row[f"{tag}_{int(eta)}"] = (prep_hit(L), hit_a(L), hit_b(L),
                                            half(L, "pop"), half(L, "max_gen"))
        out.append(row)
    return out


def knockout_window_selftest(seed=0, verbose=True):
    """The second knockout bug: the replay window was long enough for the population to RE-EVOLVE.

    eta_scale = 0 stops learning, not reproduction.  At 500 steps the replayed population reached
    max_gen 4.4 -- enough generations for selection to re-adapt to whatever mapping is pinned, so
    the "genome" number was partly a fresh adaptation.  The tell was a NON-PLASTIC `fixed` genome
    scoring 0.775 on its own mapping and 0.730 on the swap: a policy that cannot change must have
    its per-type hits SWAP when the mapping swaps.

    That is the test.  Replay a `fixed` population against its own mapping and the swap; the sign
    of (hit|A - hit|B) must flip.  It does not flip if the window lets the population re-evolve.
    """
    from sim import Config, run as _run
    kw = dict(WORLD); kw.update(mode="fixed")
    src = _run(Config(seed=seed, **kw), verbose=False,
               phases=[dict(n_steps=1500, chain=False), dict(n_steps=1500, chain=True)])
    got = knockout({"fixed": [src]}, "fixed")[0]
    # learning-off cells: a `fixed` genome has no H at all, so eta 0 is the honest comparison
    (_, ma, mb, _, mg), (_, sa, sb, _, sg) = got["matched_0"], got["shuffled_0"]
    swapped = np.sign(ma - mb) == -np.sign(sa - sb)
    if verbose:
        print(f"  mapping {got['mapping']}   matched A {ma:.3f} B {mb:.3f} (gen {mg:.1f})"
              f"   shuffled A {sa:.3f} B {sb:.3f} (gen {sg:.1f})")
        print(f"  a non-plastic genome's per-type hits swap with the mapping: {bool(swapped)}")
    print(f"  knockout-window self-test: {'PASS' if swapped else 'FAIL'}")
    return bool(swapped)


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
    print(f"v3.11 -- the preparation world.  {nseed} seeds.  Prep hit: chance {CHANCE:.3f}, "
          f"type-blind {TYPE_BLIND:.2f}, full {FULL:.2f}.  Chance safe rate 0.500.  Random-policy "
          f"action share 1/5 in phase 1, 1/8 in phase 2.")
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
    hf, hfa, hfb = v(F, P2, FF(prep_hit)), v(F, P2, FF(hitA)), v(F, P2, FF(hitB))
    tbl = v(F, P2, FF(type_blind_level))
    print(f"  fixed prep hit {np.round(hf,3).tolist()}   per type A {np.round(hfa,3).tolist()}  B {np.round(hfb,3).tolist()}")
    print(f"  all-agents     {np.round(v(F,P2,prep_hit),3).tolist()}   per type A "
          f"{np.round(v(F,P2,hitA),3).tolist()}  B {np.round(v(F,P2,hitB),3).tolist()}")
    print(f"  ITS OWN TYPE-BLIND LEVEL = max(share_A, share_B) {np.round(tbl,3).tolist()}"
          f"   -- NOT 0.5 when encounters are skewed.  fixed - its own level"
          f" {np.round(hf - tbl,3).tolist()}")
    fired = int(np.sum(hf > 0.55))
    print(f"  above 0.55 in {fired}/{nseed}   (recorded, not a stop -- see row 3b for how much of")
    print("  this the genome carries and how much the rule adds)")
    print("  Read the per-type split: one type high and the other near 0 is type-blind (allowed);")
    print("  both above 0.5 in `fixed` would be genes holding the conjunction (not allowed).")
    print("  READ IT PER ERA.  The phase half spans TWO mapping eras, so a genome that switches")
    print("  its preparation between them averages into a FALSE one-high-one-low reading.  A pure")
    print("  mixture of type-blind genotypes has P(hit|A) + P(hit|B) <= 1; a SUM ABOVE 1 means the")
    print("  genomes are conditioning on food type, whatever the phase-half aggregate says.")
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
        a_, b_ = v(n, P2, FF(hitA)), v(n, P2, FF(hitB))
        print(f"    {'':<22} its type-blind level {np.round(v(n,P2,FF(type_blind_level)),3).tolist()}")
        both = int(np.sum((a_ > TYPE_BLIND) & (b_ > TYPE_BLIND)))
        print(f"    {n:<22} A {np.round(a_,3).tolist()}  B {np.round(b_,3).tolist()}   both > {TYPE_BLIND} in {both}/{nseed}")
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
    print("  (ii) SURVIVOR-CONDITIONED SINCE-REMAP -- CORROBORATING ONLY, 4-preparation window.")
    print("       Counted only if it made 4 preparations BEFORE a remap and 4 after it, so a fall")
    print("       and recovery cannot be a change of sample.  The population-level since-remap")
    print("       curve below IS open to that objection; this line is not.")
    print(f"    {'arm':<22}{'8 before':>10}{'8 after':>10}{'change':>9}{'n agent-remaps':>16}")
    for n in names:
        rows = [surv_remap(P2(r), True) for r in results[n]]
        pre = np.nanmean([x[0] for x in rows]); post = np.nanmean([x[1] for x in rows])
        d = np.array([x[2] for x in rows], dtype=float); nn = int(np.sum([x[3] for x in rows]))
        print(f"    {n:<22}{pre:>10.3f}{post:>10.3f}{np.nanmean(d):>9.3f}{nn:>16d}"
              f"   per seed {np.round(d,3).tolist()}")

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

    print("\nrow 3b  THE ATTRIBUTION LINE -- late genomes replayed on the matched mapping and on")
    print("        a shuffled one, with learning OFF (eta 0) and ON (eta 1).  One 50-step window.")
    print("        The SHUFFLED pair is the attribution: on a mapping the genome was never sorted")
    print("        for, learning-off is the genetic floor and learning-on is what the rule adds")
    print("        within a life.  REQUIRED: eta 1 above eta 0 on SHUFFLED by >= 0.10 in EVERY")
    print("        seed.  pop and max_gen print beside every number -- eta_scale = 0 stops")
    print("        learning but NOT reproduction, so a moved max_gen means the number is not the")
    print("        genome.  sim.replay_mapping_selftest and analysis.knockout_window_selftest")
    print("        guard the pinning and the window.")
    print(f"    {'arm / seed':<18}{'map':>7}{'MATCHED eta0':>13}{'eta1':>7}{'gen':>6}"
          f"{'|':>3}{'SHUFFLED eta0':>14}{'eta1':>7}{'gen':>6}{'eta1-eta0':>11}")
    ok_all, seen = [], False
    for n in (PL, S, F):
        if n not in results:
            continue
        try:
            for k in knockout(results, n):
                m0, m1 = k["matched_0"], k["matched_1"]
                s0, s1 = k["shuffled_0"], k["shuffled_1"]
                gain = s1[0] - s0[0]
                if n == PL:
                    ok_all.append(gain >= 0.10); seen = True
                mark = "" if n != PL else ("  PASS" if gain >= 0.10 else "  FAIL")
                print(f"    {n + ' s' + str(k['seed']):<18}{str(k['mapping']):>7}"
                      f"{m0[0]:>13.3f}{m1[0]:>7.3f}{m1[4]:>6.1f}{'|':>3}"
                      f"{s0[0]:>14.3f}{s1[0]:>7.3f}{s1[4]:>6.1f}{gain:>+11.3f}{mark}")
        except Exception as exc:
            print(f"    {n:<18} knockout failed: {type(exc).__name__}: {exc}")
    if seen:
        print(f"    REQUIREMENT (plastic, shuffled, eta1 - eta0 >= 0.10): "
              f"{sum(ok_all)}/{len(ok_all)}   {'PASS' if all(ok_all) else 'FAIL'}")
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
