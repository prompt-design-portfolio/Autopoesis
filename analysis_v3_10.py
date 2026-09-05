"""
v3.10 analysis -- the preparation world.

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
    prep_value=1.5, prep_fail=0.5, prep_every=2000,
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

def run_experiment(seeds, phase_steps=PHASE_STEPS, variants=VARIANTS, verbose=False,
                   save_path=None, **overrides):
    """Sequential, single process: this has to run on a Colab CPU runtime, so no multiprocessing.
    If save_path is given the results are pickled after every run, so a dropped session loses one
    run rather than the lot."""
    import pickle, time
    results = {v: [] for v in variants}
    total, done, t0 = len(seeds) * len(variants), 0, time.time()
    for seed in seeds:
        for name, spec in variants.items():
            phases = [dict(p, n_steps=int(p["n_steps"] / PHASE_STEPS * phase_steps)) for p in spec["phases"]]
            kw = dict(spec["kw"]); kw.update(overrides)      # overrides win over WORLD
            t1 = time.time()
            results[name].append(run(Config(seed=seed, **kw), verbose=verbose, phases=phases))
            done += 1
            el = time.time() - t0
            print(f"[{done}/{total}] {name} seed={seed}  {time.time()-t1:.0f}s   "
                  f"elapsed {el/60:.1f} min, est. total {el/done*total/60:.0f} min", flush=True)
            if save_path:
                with open(save_path, "wb") as f:
                    pickle.dump(results, f)
    return results


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


def rate(L, num, den):
    d = float(np.sum([r[den] for r in L]))
    return float(np.sum([r[num] for r in L]) / d) if d > 0 else np.nan


def hit(L):
    return rate(L, "n_correct", "n_attempts_raw")


def safe(L):
    return rate(L, "n_safe", "n_eats")


def per_1k(L, key):
    s = float(np.sum([r["agent_steps"] for r in L]))
    return float(1000.0 * np.sum([r[key] for r in L]) / s) if s > 0 else np.nan


def curve(L, key):
    n = np.sum([r[key + "_n"] for r in L], axis=0)
    c = np.sum([r[key + ("_safe" if key == "meal" else "_correct")] for r in L], axis=0)
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


def prep_hit(L):
    """Event-weighted prep hit over both food types."""
    n = float(np.sum([r["n_prep0"] + r["n_prep1"] for r in L]))
    k = float(np.sum([r["n_ok0"] + r["n_ok1"] for r in L]))
    return k / n if n else np.nan


def surv_curve(L):
    """Hit on preparations 1-5 vs 6-10, over agents that REACHED 10 preparations.  Every agent
    counted contributes both halves of its own curve, so a rise cannot be survivorship: it is the
    same individuals, later in their own lives.  (`probe_adv` is computed over the LIVING and so
    is itself partly survivorship-selected; this line is not.)"""
    n = float(np.sum([r["n_surv"] for r in L]))
    if not n:
        return np.nan, np.nan, np.nan, 0
    e = float(np.sum([r["n_surv_early"] for r in L])) / (5 * n)
    l = float(np.sum([r["n_surv_late"] for r in L])) / (5 * n)
    return e, l, l - e, int(n)


def first_prep_hit(L):
    """Hit on an agent's FIRST preparation: it has learned nothing, so this reads the innate
    policy the population currently carries."""
    return rate(L, "n_first_ok", "n_first")


def shift_timing(run_, bin_size=250, span=2000):
    """When the eat -> prep shift happens, in `bin_size` bins after the switch."""
    sw = run_.get("chain_start", 0)
    out = []
    for lo in range(sw, sw + span, bin_size):
        w = window(run_, lo, lo + bin_size)
        if w:
            out.append((lo - sw, round(float(half(w, "prep_share")), 3)))
    return out


def knockout(results, name, steps=3000, seed_offset=1000, early=500):
    """Replay late genomes with eta_scale = 0 in a fresh world: same brains, no learning.

    READ THE FIRST WINDOW.  eta_scale = 0 removes learning but NOT reproduction and mutation, so
    over `steps` the replayed population RE-EVOLVES against the new mapping: measured on the v3.10
    acceptance run, max_gen went 4-8 -> 9-35 across 3000 steps and the hit rate climbed with it
    (plastic seed 0: 0.485 -> 0.657).  The second half therefore measures re-selection in the new
    world, not the genome.  Both windows and both max_gen values are returned so the contamination
    stays visible rather than being taken on trust."""
    from sim import Config, run as _run
    out = []
    for i, r in enumerate(results[name]):
        cfg = dict(r["cfg"]); cfg.pop("seed", None); cfg.pop("n_steps", None)
        cfg["eta_scale"] = 0.0
        rr = _run(Config(seed=seed_offset + i, **cfg), verbose=False,
                  init_genomes=r["final"], phases=[dict(n_steps=steps, chain=True)])
        E, L = window(rr, 0, early), rr["log"][len(rr["log"]) // 2:]
        out.append(dict(
            hit=prep_hit(E), a=rate(E, "n_ok0", "n_prep0"), b=rate(E, "n_ok1", "n_prep1"),
            pop=half(E, "pop"), gen=half(E, "max_gen"),
            late_hit=prep_hit(L), late_gen=half(L, "max_gen")))
    return out


def first_era(run_, era=2000):
    """The FIRST mapping era of phase 2 -- steps 0..era after the switch.  Rig check 2(a) is read
    here and only here: once the mapping is known, preparation pays 1.5 on ANY food, the
    safe/poison fact becomes irrelevant, and a good learner stops eating raw.  probe_adv (food)
    decaying after that is a GOOD reason, not a broken transition."""
    sw = run_.get("chain_start", 0)
    return window(run_, sw, sw + era)


def era_windows(run_, era=2000):
    sw, n = run_.get("chain_start", 0), run_["n_steps"]
    return [window(run_, lo, min(lo + era, n)) for lo in range(sw, n, era)]


def summary(results):
    names = list(results)
    w = 24
    nseed = len(results[names[0]])
    print("=" * (20 + w * len(names)))
    print(f"v3.10 -- the preparation world.  {nseed} seeds.  Prep hit: chance {CHANCE:.3f}, "
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
                             ("prep hit", prep_hit, lambda r: phase_half(r, 1), "phase 2"),
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
    v = lambda n, sel, f: ps(results, n, sel, f)
    n_ok = lambda a, b: int(np.sum(np.asarray(a) - np.asarray(b) >= MARGIN))
    hitA = lambda L: rate(L, "n_ok0", "n_prep0")
    hitB = lambda L: rate(L, "n_ok1", "n_prep1")

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
    for lab, sel in [("phase 1", P1), ("phase 2", P2)]:
        for n in names:
            pop, inj = v(n, sel, lambda L: half(L, "pop")), v(n, sel, lambda L: half(L, "injections"))
            bad = (np.nanmin(pop) < 80 or np.nanmax(inj) > 0)
            flag = "  (null arm, exempt)" if n == NULL else ("  <-- EXCLUDE" if bad else "")
            print(f"  {lab}  {n:<22} pop {np.round(pop,0).tolist()}  inj {np.round(inj,1).tolist()}{flag}")

    print("\nrow 1a  PHASE-1 GATE = v3.1.  A STOP ROW: if it fails, nothing below is read.")
    sp, sf = v(PL, P1, safe), v(F, P1, safe)
    pf = v(PL, P1, lambda L: half(L, "probe_adv_food"))
    fpop, finj = v(F, P1, lambda L: half(L, "pop")), v(F, P1, lambda L: half(L, "injections"))
    excluded = (fpop < 80) | (finj > 0)
    V31_HI = 0.56
    print(f"  safe_rate  plastic {np.round(sp,3).tolist()}   fixed {np.round(sf,3).tolist()}   (v3.1: 0.60-0.66 vs 0.51-0.56)")
    if excluded.any():
        print(f"  ROW-0 FALLBACK in seed(s) {np.where(excluded)[0].tolist()}: those read against v3.1's"
              f" published range, conservative end {V31_HI}.")
    for i in range(nseed):
        d = sp[i] - (V31_HI if excluded[i] else sf[i])
        print(f"    seed {i}: {d:+.3f} ({'published' if excluded[i] else 'measured'})   {'PASS' if d >= MARGIN else 'FAIL'}")
    print(f"  probe_adv (food) plastic {np.round(pf,3).tolist()}   (target >= 1.0)")

    print("\nrow 1b  MAPPING GATE -- genes may hold a type-blind preparation, but must not track the")
    print("        CONJUNCTION.  Gate fires at `fixed` prep hit > 0.55 in 3/3.")
    hf, hfa, hfb = v(F, P2, prep_hit), v(F, P2, hitA), v(F, P2, hitB)
    print(f"  fixed prep hit {np.round(hf,3).tolist()}   per type A {np.round(hfa,3).tolist()}  B {np.round(hfb,3).tolist()}")
    fired = int(np.sum(hf > 0.55))
    print(f"  above 0.55 in {fired}/{nseed}" + ("   <-- GATE FIRES: shorten prep_every toward the flip"
          " period, judged against `fixed` only, before reading below."
          if fired >= rule else "   gate clear."))
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
            sums = [round(a_ + b_, 2) for a_, b_ in rows]
            print(f"    {n + ' seed ' + str(i):<22} (A,B) by era {rows}")
            print(f"    {'':<22} sums {sums}   both > {TYPE_BLIND} in {both}/{len(rows)} eras")
    print("  PRIMARY INSTRUMENT for this row: prep_gain innate -- the GENOME's preference for the")
    print("  correct preparation on a synthetic 'type f underfoot' observation.  No gating and no")
    print("  declining enter it, so unlike the A+B sum it localises the type-conditioning to the")
    print("  CHOICE of preparation.  `random policy` is the zero; `fixed + B` sits near zero")
    print("  because its knowledge is in the private table, not in synapses -- that is the check")
    print("  that the instrument reads synapses and nothing else.")
    for n in names:
        print(f"    {n:<22} {np.round(v(n, P2, lambda L: half(L, 'prep_gain_innate')), 3).tolist()}")

    print("\nrow 2  RIG CHECKS -- nothing below is read until these are clean.")
    print("  (a) food learning survives the switch.  READ ON SAFE RATE in the first mapping era")
    print("      (plastic - fixed >= 0.03); the corrected probe in the first 500 steps is")
    print("      CORROBORATING.  The v3.1 probe form was contaminated here -- it subtracts the best")
    print("      OTHER action, and with three preparations live that term moves with food type.")
    sfe, sffe = v(PL, first_era, safe), v(F, first_era, safe)
    print(f"    safe (first era)  plastic {np.round(sfe,3).tolist()}  fixed {np.round(sffe,3).tolist()}"
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
    for n in OUTCOME_ARMS + [NULL]:
        print(f"    {n:<22} prep/life {np.round(v(n,P2,lambda L: half(L,'prep_per_life')),2).tolist()}")
    print("  (c) P(prep | on food) against the per-phase null (analytic 3/8 = 0.375 in phase 2)")
    base = np.nanmean(v(NULL, P2, lambda L: half(L, "prep_on_food")))
    print(f"    null measured {base:.3f}")
    for n in names:
        print(f"    {n:<22} {np.round(v(n,P2,lambda L: half(L,'prep_on_food')),3).tolist()}")

    print("\n" + "=" * 78)
    print("row 3  THE CONJUNCTION -- a dense, immediate, two-sided task")
    print("=" * 78)
    hp, hs = v(PL, P2, prep_hit), v(S, P2, prep_hit)
    pl, fl = v(PL, P2, lambda L: half(L, "prep_per_life")), plf
    print("  ABSTENTION CHECK FIRST -- prepared meals not below 0.8x `fixed`.")
    print(f"    prep/life plastic {np.round(pl,2).tolist()}  fixed {np.round(fl,2).tolist()}"
          f"   ratio {np.round(pl / np.maximum(fl, 1e-9), 2).tolist()}   (abstention if < 0.80)")
    print("  the result lines:")
    print(f"    plastic - fixed:     {np.round(hp - hf,3).tolist()}   >= +{MARGIN} in {n_ok(hp, hf)}/{nseed}")
    print(f"    plastic - scrambled: {np.round(hp - hs,3).tolist()}   >= +{MARGIN} in {n_ok(hp, hs)}/{nseed}   <- scrambled carries it")
    print(f"    prep hit  plastic {np.round(hp,3).tolist()}  fixed {np.round(hf,3).tolist()}  scrambled {np.round(hs,3).tolist()}")
    print(f"  PER TYPE -- a conjunction is BOTH above {TYPE_BLIND:.2f}, not one at 1.0 and one at 0:")
    for n in names:
        a_, b_ = v(n, P2, hitA), v(n, P2, hitB)
        both = int(np.sum((a_ > TYPE_BLIND) & (b_ > TYPE_BLIND)))
        print(f"    {n:<22} A {np.round(a_,3).tolist()}  B {np.round(b_,3).tolist()}   both > {TYPE_BLIND} in {both}/{nseed}")
    pr = v(PL, P2, lambda L: half(L, "probe_adv"))
    ao, ay = v(PL, P2, lambda L: half(L, "hit_old")), v(PL, P2, lambda L: half(L, "hit_young"))
    ag = v(PL, P2, lambda L: float(curve(L, "att")[-1] - curve(L, "att")[0]))
    print(f"  probe_adv (prep) {np.round(pr,3).tolist()}   (> 0 in {int(np.sum(pr > 0))}/{nseed})")
    print("    NOTE: probe_adv is computed over the LIVING, so it is itself partly")
    print("    survivorship-selected.  The survivor curve below is not.")
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
        rows = [surv_curve(P2(r)) for r in results[n]]
        e = np.nanmean([x[0] for x in rows]); l = np.nanmean([x[1] for x in rows])
        d = np.array([x[2] for x in rows], dtype=float); nn = int(np.sum([x[3] for x in rows]))
        print(f"    {n:<22}{e:>11.3f}{l:>12.3f}{np.nanmean(d):>9.3f}{nn:>10d}   per seed {np.round(d,3).tolist()}")
    print("  (ii) FIRST-PREPARATION HIT -- the agent has learned nothing, so this reads the innate")
    print("       policy the population carries (and shows the type-blind floor directly).")
    for n in names:
        print(f"    {n:<22} {np.round(v(n, P2, first_prep_hit),3).tolist()}")
    if CEIL in results:
        hc = v(CEIL, P2, prep_hit)
        print(f"  ceiling {np.round(hc,3).tolist()}   (reference: hand-wired exact credit, forced argmax)")

    print("\nrow 3b  KNOCKOUT -- late genomes replayed with eta_scale = 0 in a fresh world.")
    print("        Same brains, no learning.  If the standing advantage lives in H, BOTH plastic")
    print("        and scrambled fall to the type-blind floor; what separates them is then the")
    print("        survivor curve, which is learning, not luck.")
    print("        READ THE FIRST WINDOW.  eta_scale = 0 stops learning but NOT reproduction, so")
    print("        the replay RE-EVOLVES: the late columns are printed only to show that drift,")
    print("        and are NOT the genome.  A rising hit with a rising max_gen is re-selection.")
    print(f"    {'arm':<22}{'hit':>8}{'hit|A':>8}{'hit|B':>8}{'A+B':>7}{'pop':>7}{'gen':>7}"
          f"{'| late hit':>11}{'late gen':>10}")
    for n in (PL, S):
        try:
            for i, k in enumerate(knockout(results, n)):
                print(f"    {n + ' seed ' + str(i):<22}{k['hit']:>8.3f}{k['a']:>8.3f}{k['b']:>8.3f}"
                      f"{k['a'] + k['b']:>7.2f}{k['pop']:>7.0f}{k['gen']:>7.1f}"
                      f"{k['late_hit']:>11.3f}{k['late_gen']:>10.1f}")
        except Exception as exc:
            print(f"    {n:<22} knockout failed: {type(exc).__name__}: {exc}")
    print(f"    reference levels: chance {CHANCE:.3f}, type-blind {TYPE_BLIND:.2f}.  A+B <= 1 is a")
    print("    mixture of type-blind genotypes carrying NO type knowledge; A+B > 1 is conjunction")
    print("    held in the genome.")

    print("\nrow 4  GENE ROWS (corroborating only)")
    for g in ("eta2", "lam2", "eta1"):
        print(f"  {g}   phase 1 -> phase 2")
        for n in OUTCOME_ARMS:
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
