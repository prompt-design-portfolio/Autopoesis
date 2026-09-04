"""
v3.9 analysis -- the rig fixed, and the chain shortened to what v3.9 tests.

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

from sim import Config, run, PATCH_TARGETS

# ---------------------------------------------------------------------------
# the world -- v3.1 metabolism throughout, the chain at the agreed cover targets
# ---------------------------------------------------------------------------
WORLD = dict(
    # v3.1's flip world (phase 1), carried through phase 2 unchanged
    flip_every=300, eta_init=0.2, hidden=24,
    spawn_per_patch=3.0, food_value=0.7, poison_value=0.5,
    repro_threshold=3.0, repro_cost=1.5, max_energy=5.0, max_pop=400, init_pop=300,
    # the SHORTENED chain (amendment 2): pickup -> carry -> attempt, and the attempt pays.
    # No nuts, no tool state.  The station -> nut bridge returns in v3.11 as one change.
    # amendment 3: the chain is CO-LOCATED with foraging.  Items and stations spawn only inside
    # the food patches, each patch carries 8 stations of each type at fixed offsets that travel
    # with it, and items stranded by a drift are cleared.  The recipe is an expensive FACT, not
    # an expensive JOURNEY.  World criterion is in-patch: nearest station of each type <= 3 steps
    # from a random patch cell, in-patch item cover 20-25%.
    items_per_patch=0.22, stations_per_patch=8, nuts_per_patch=0.0, nuts_uniform=0.0,
    carry_cost=0.0,                 # a carry tax punishes exploration, not the chain
    tool_value=2.5, fail_cost=0.05, recipe_every=2000,
    scaffold_food=False, scaffold_chain=False,
)

PHASE_STEPS = 8000
STAGED = [dict(n_steps=PHASE_STEPS, chain=False), dict(n_steps=PHASE_STEPS, chain=True)]

def _v(**kw):
    return dict(kw=dict(**dict(WORLD, **kw)), phases=STAGED)

VARIANTS = {
    # the behavioural null.  Uniform over the AVAILABLE actions -- five in phase 1 (`interact` is
    # masked while the chain is off), six in phase 2 -- so the conditional null is 1/5 then 1/6.
    # Exempt from row 0: a random walker belongs at the population floor.
    "random policy":       _v(mode="random"),
    "fixed":               _v(mode="fixed"),
    # the control for "a modulator at stations changes behaviour": same plasticity, same H
    # magnitudes, random-sign m.
    "scrambled":           _v(mode="plastic", plastic_layers="W2", scramble=True),
    "plastic (W2)":        _v(mode="plastic", plastic_layers="W2"),
    # reference level, not a matched comparison: its steering route is only an observation
    # channel nothing is wired to, so its live route is the veto.
    "fixed + B (ceiling)": _v(mode="fixed", private_mem=True, pref_gain=8.0, veto_p=0.9),
}

NULL = "random policy"
OUTCOME_ARMS = [n for n in VARIANTS if n != NULL]

COLORS = {"random policy": "tab:grey", "fixed": "tab:red", "scrambled": "black",
          "plastic (W2)": "tab:blue", "fixed + B (ceiling)": "tab:purple"}

CHANCE = 1.0 / 6.0
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
    print(f"\nTRANSITION -- {bin_size}-step bins, {span} steps either side of the chain switching on")
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
    ("safe_rate",          safe),
    ("recipe_hit",         hit),
    ("attempts/1k",        lambda L: per_1k(L, "n_attempts_raw")),
    ("holding_item",       lambda L: half(L, "holding_item")),
    ("pop",                lambda L: half(L, "pop")),
    ("injections",         lambda L: half(L, "injections")),
    ("max_gen",            lambda L: half(L, "max_gen")),
    ("probe_adv (food)",   lambda L: half(L, "probe_adv_food")),
    ("probe_adv (recipe)", lambda L: half(L, "probe_adv")),
    ("pair_gain innate",   lambda L: half(L, "pair_gain_innate")),
    ("eta2",               lambda L: half(L, "eta2")),
    ("eta1",               lambda L: half(L, "eta1")),
    ("lam2",               lambda L: half(L, "lam2")),
    ("h_norm",             lambda L: half(L, "h_norm")),
    ("nav_dir",            lambda L: half(L, "nav_dir")),
    ("nav_here",           lambda L: half(L, "nav_here")),
    ("crop_safe",          lambda L: half(L, "crop_safe")),
    ("trace_recency",      lambda L: half(L, "trace_recency")),
    ("e_fail/1k",          lambda L: half(L, "e_fail_per_1k")),
    ("e_tool/1k",          lambda L: half(L, "e_bonus_per_1k")),
    ("pickups/1k",         lambda L: half(L, "pickups_per_1k")),
    ("declined/1k",        lambda L: half(L, "declined_per_1k")),
    ("decline_frac",       lambda L: half(L, "decline_frac")),
    ("eat on food",        lambda L: half(L, "eat_on_food")),
    ("interact on food",   lambda L: half(L, "int_on_food")),
    ("eat on item",        lambda L: half(L, "eat_on_item")),
    ("P(int | on item)",   lambda L: half(L, "int_on_item")),
    ("P(int | at station)", lambda L: half(L, "int_at_station")),
    ("noops/1k",           lambda L: half(L, "noops_per_1k")),
    ("attempts/life",      lambda L: half(L, "attempts_per_life")),
    ("meal gain",          lambda L: float(curve(L, "meal")[-1] - curve(L, "meal")[0])),
    ("attempt gain",       lambda L: float(curve(L, "att")[-1] - curve(L, "att")[0])),
]


def summary(results):
    names = list(results)
    w = 24
    nseed = len(results[names[0]])
    print("=" * (20 + w * len(names)))
    print(f"v3.9 -- the rig fixed (audit A, B, D, F).  {nseed} seeds.  Chance recipe hit "
          f"{CHANCE:.3f}, chance safe rate 0.500, random-policy action share 1/6 = {1/6:.3f}")
    print("=" * (20 + w * len(names)))

    for label, sel in [("PHASE 1 (food only) -- second half", lambda r: phase_half(r, 0)),
                       ("PHASE 2 (chain on) -- second half", lambda r: phase_half(r, 1))]:
        print(f"\n### {label}")
        if label.startswith("PHASE 1"):
            print("    food only: no items, stations or nuts, so `interact` is a permanent no-op.")
        print(f"{'metric':<20}" + "".join(f"{n:>{w}}" for n in names))
        for lab, f in ROWS:
            print(f"{lab:<20}" + "".join(f"{np.nanmean(ps(results, n, sel, f)):>{w}.3f}" for n in names))

    print("\nper seed:")
    for lab, f, sel, tag in [("safe_rate", safe, lambda r: phase_half(r, 0), "phase 1"),
                             ("probe_adv (food)", lambda L: half(L, "probe_adv_food"), lambda r: phase_half(r, 0), "phase 1"),
                             ("eta2", lambda L: half(L, "eta2"), lambda r: phase_half(r, 0), "phase 1"),
                             ("recipe_hit", hit, lambda r: phase_half(r, 1), "phase 2"),
                             ("attempts/1k", lambda L: per_1k(L, "n_attempts_raw"), lambda r: phase_half(r, 1), "phase 2"),
                             ("pop", lambda L: half(L, "pop"), lambda r: phase_half(r, 1), "phase 2"),
                             ("injections", lambda L: half(L, "injections"), lambda r: phase_half(r, 1), "phase 2"),
                             ("eta2", lambda L: half(L, "eta2"), lambda r: phase_half(r, 1), "phase 2"),
                             ("lam2", lambda L: half(L, "lam2"), lambda r: phase_half(r, 1), "phase 2")]:
        print(f"  {lab} ({tag})")
        for n in names:
            print(f"    {n:<24} {np.round(ps(results, n, sel, f), 3).tolist()}")

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
    have = lambda n: n in results

    print("\n" + "-" * 78)
    print(f"DECISION NUMBERS (a difference counts when it is >= {MARGIN} in {rule}/{nseed} seeds)")
    print("  PRE-REGISTERED PREDICTION: `plastic (W2)` clears row 3, and abstention does not occur")
    print("  under a two-sided signal.  Recorded before the run.")
    print("-" * 78)

    print("\nrow 0  uninterpretable?  Per phase.  `random policy` is EXEMPT.")
    for lab, sel in [("phase 1", P1), ("phase 2", P2)]:
        for n in names:
            pop, inj = v(n, sel, lambda L: half(L, "pop")), v(n, sel, lambda L: half(L, "injections"))
            bad = (np.nanmin(pop) < 80 or np.nanmax(inj) > 0)
            flag = "  (null arm, exempt)" if n == NULL else ("  <-- EXCLUDE" if bad else "")
            print(f"  {lab}  {n:<22} pop {np.round(pop,0).tolist()}  inj {np.round(inj,1).tolist()}{flag}")

    print("\nrow 1a  PHASE-1 GATE -- is this v3.1?  A STOP ROW: if it fails, nothing below is read.")
    sp, sf = v(PL, P1, safe), v(F, P1, safe)
    pf = v(PL, P1, lambda L: half(L, "probe_adv_food"))
    fpop, finj = v(F, P1, lambda L: half(L, "pop")), v(F, P1, lambda L: half(L, "injections"))
    excluded = (fpop < 80) | (finj > 0)
    V31_HI = 0.56
    print(f"  safe_rate  plastic {np.round(sp,3).tolist()}   fixed {np.round(sf,3).tolist()}   (v3.1: 0.60-0.66 vs 0.51-0.56)")
    if excluded.any():
        print(f"  ROW-0 FALLBACK in seed(s) {np.where(excluded)[0].tolist()}: `fixed` phase 1 excluded"
              f" (pop {np.round(fpop[excluded],0).tolist()}); those seeds read against v3.1's published"
              f" range, conservative end {V31_HI}.")
    gate_a = []
    for i in range(nseed):
        d = sp[i] - (V31_HI if excluded[i] else sf[i])
        gate_a.append(d >= MARGIN)
        print(f"    seed {i}: {d:+.3f} ({'published' if excluded[i] else 'measured'})   {'PASS' if d >= MARGIN else 'FAIL'}")
    print(f"    >= +{MARGIN} in {sum(gate_a)}/{nseed};  probe_adv (food) {np.round(pf,3).tolist()} (target >= 1.0)")

    print("\nrow 1b  RECIPE GATE -- can the genome track the recipe now that it pays?")
    hf = v(F, P2, hit)
    print(f"  fixed recipe_hit {np.round(hf,3).tolist()}   (chance {CHANCE:.3f}; gate FIRES at > 0.25 in {rule}/{nseed})")
    print(f"  fixed pair_gain innate {np.round(v(F,P2,lambda L: half(L,'pair_gain_innate')),3).tolist()}")
    fired = int(np.sum(hf > 0.25))
    print(f"  above 0.25 in {fired}/{nseed}" + ("   <-- GATE FIRES: shorten recipe_every toward the flip"
          " period, judged against `fixed` only, before reading anything below."
          if fired >= rule else "   gate clear."))

    print("\nrow 2  RIG CHECKS -- nothing below is read until these are clean.")
    print("  (a) food learning SURVIVES phase 2:  safe >= 0.58 and probe_adv (food) >= 1.0")
    for n in (PL, S, F):
        if not have(n):
            continue
        s2, p2 = v(n, P2, safe), v(n, P2, lambda L: half(L, "probe_adv_food"))
        mark = ("   PASS" if (np.all(s2 >= 0.58) and np.all(p2 >= 1.0)) else "   FAIL") if n == PL else ""
        print(f"    {n:<22} safe {np.round(s2,3).tolist()}  probe_adv (food) {np.round(p2,3).tolist()}{mark}")
    print("    the phase-1 baseline this rests on (v3.1: plastic 0.60-0.66, fixed 0.51-0.56;")
    print("    probe_adv 1.4-2.7).  No longer its own stop row, but a failure here voids phase 2:")
    for n in (PL, F):
        print(f"    {n:<22} P1 safe {np.round(v(n,P1,safe),3).tolist()}  "
              f"P1 probe_adv (food) {np.round(v(n,P1,lambda L: half(L,'probe_adv_food')),3).tolist()}  "
              f"P1 pop {np.round(v(n,P1,lambda L: half(L,'pop')),0).tolist()}")
    print("  (b) READABILITY -- judged on `fixed`, not the null.  `random policy` sits at the")
    print("      injection floor, so its attempts/life measures churn rather than the world.")
    alf = v(F, P2, lambda L: half(L, "attempts_per_life"))
    isf = v(F, P2, lambda L: half(L, "int_at_station"))
    isn = np.nanmean(v(NULL, P2, lambda L: half(L, "int_at_station"))) if have(NULL) else np.nan
    print(f"    fixed  attempts/life {np.round(alf,2).tolist()}   "
          f"{'PASS' if np.all(alf >= 3.0) else 'FAIL'}   (criterion >= 3)")
    print(f"    fixed  P(interact | at station, carrying) {np.round(isf,3).tolist()}  vs null {isn:.3f}   "
          f"{'PASS' if np.all(isf > isn) else 'FAIL'}")
    if have(NULL):
        aln = v(NULL, P2, lambda L: half(L, "attempts_per_life"))
        print(f"    {NULL} (affordance null) attempts/life {np.round(aln,2).tolist()}   "
              f"{'PASS' if np.all(aln >= 1.0) else 'FAIL'}   (criterion >= 1)")
    for n in OUTCOME_ARMS:
        print(f"    {n:<22} attempts/life {np.round(v(n,P2,lambda L: half(L,'attempts_per_life')),2).tolist()}")
    print("    STOP CONDITION: if `fixed` makes fewer than ONE attempt per life over a full")
    print("    8000-step phase 2 in the acceptance run, rig work on this world stops and the")
    print("    result is reported as a finding about sparse chains under autopoietic economics.")
    print("  (c) CONDITIONAL APPROACH -- the unconfounded test.  A per-1k rate compares action")
    print("      budgets, not approach; these condition on the opportunity.  The null is the")
    print("      `random policy` arm, whose analytic share is 1/5 in phase 1 (interact masked)")
    print("      and 1/6 in phase 2.")
    conds = [("P(interact | on item, empty-handed)", "int_on_item"),
             ("P(interact | at station, carrying)", "int_at_station"),
             ("P(eat | on food)", "eat_on_food")]
    for lab, key in conds:
        base = np.nanmean(v(NULL, P2, lambda L: half(L, key))) if have(NULL) else np.nan
        print(f"    {lab}   null (measured) {base:.3f}, analytic {1/6:.3f}")
        for n in names:
            x = v(n, P2, lambda L: half(L, key))
            mk = "" if n == NULL else ("  above null" if np.nanmean(x) >= base + MARGIN else "")
            print(f"      {n:<22} {np.round(x,3).tolist()}{mk}")

    print("\n" + "=" * 78)
    print("row 3  THE CONJUNCTION -- immediate, two-sided credit, in the world, in every arm")
    print("=" * 78)
    hp, hs = v(PL, P2, hit), v(S, P2, hit)
    ap, af = v(PL, P2, lambda L: per_1k(L, "n_attempts_raw")), v(F, P2, lambda L: per_1k(L, "n_attempts_raw"))
    print("  ABSTENTION CHECK FIRST -- pre-registered: with a two-sided signal it should NOT occur.")
    print(f"    attempts/1k  plastic {np.round(ap,2).tolist()}   fixed {np.round(af,2).tolist()}")
    print(f"    ratio {np.round(ap / np.maximum(af, 1e-9), 2).tolist()}   (abstention if < 0.80)")
    dp = v(PL, P2, lambda L: half(L, "decline_frac"))
    de = v(PL, lambda r: window(r, r["chain_start"], r["chain_start"] + (r["n_steps"] - r["chain_start"]) // 4),
           lambda L: half(L, "decline_frac"))
    print(f"    decline_frac early {np.round(de,3).tolist()} -> late {np.round(dp,3).tolist()}"
          f"   (rising in {int(np.sum(dp > de))}/{nseed})")
    print("  the result lines:")
    print(f"    plastic - fixed:     {np.round(hp - hf,3).tolist()}   >= +{MARGIN} in {n_ok(hp, hf)}/{nseed}")
    print(f"    plastic - scrambled: {np.round(hp - hs,3).tolist()}   >= +{MARGIN} in {n_ok(hp, hs)}/{nseed}   <- scrambled carries it")
    print(f"    recipe_hit  plastic {np.round(hp,3).tolist()}  fixed {np.round(hf,3).tolist()}  "
          f"scrambled {np.round(hs,3).tolist()}   (chance {CHANCE:.3f})")
    pr = v(PL, P2, lambda L: half(L, "probe_adv"))
    ao, ay = v(PL, P2, lambda L: half(L, "hit_old")), v(PL, P2, lambda L: half(L, "hit_young"))
    ag = v(PL, P2, lambda L: float(curve(L, "att")[-1] - curve(L, "att")[0]))
    print(f"  probe_adv (recipe) {np.round(pr,3).tolist()}   (> 0 in {int(np.sum(pr > 0))}/{nseed})")
    print(f"  within-life signature -- at least one, in {rule}/{nseed}:")
    print(f"    hit-in-life curve gain {np.round(ag,3).tolist()}   (> 0 in {int(np.sum(ag > 0))}/{nseed})")
    print(f"    hit_old - hit_young    {np.round(ao - ay,3).tolist()}   (> 0 in {int(np.sum(ao > ay))}/{nseed})")
    if have(CEIL):
        hc = v(CEIL, P2, hit)
        print(f"  ceiling {np.round(hc,3).tolist()}   ceiling - fixed {np.round(hc - hf,3).tolist()}   (reference, not matched)")

    print("\nrow 4  GENE ROWS (corroborating only)")
    for g in ("eta2", "lam2", "eta1"):
        print(f"  {g}   phase 1 -> phase 2")
        for n in OUTCOME_ARMS:
            print(f"    {n:<22} {np.round(v(n,P1,lambda L: half(L,g)),3).tolist()} -> {np.round(v(n,P2,lambda L: half(L,g)),3).tolist()}")
    print("  unwired scaffold genes -- drift scale for a heritable scalar (nothing is wired to them):")
    for g in ("nav_dir", "nav_here"):
        print(f"    {g:<12} " + "  ".join(f"{n}: {np.nanmean(v(n,P2,lambda L: half(L,g))):.2f}" for n in (F, PL)))
    print(f"  trace_recency (phase 1) plastic {np.round(v(PL,P1,lambda L: half(L,'trace_recency')),3).tolist()}")

    print("\nrow 5  if row 3 is NULL with rows 1-2 clean, that is the first earned statement about")
    print("       the learner's limit: the rule does not hold a conjunction even with immediate,")
    print("       two-sided credit, a separate interact action, a readable world, declining as a")
    print("       policy and a random-walk null.  Spend the one rule-form change there.")
    print("-" * 78)


# ---------------------------------------------------------------- curves / plots

def curves(results, key, xlabel, show=True):
    label = {"att": "attempt number in an agent's life",
             "rec": "attempts since the last recipe change",
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
            ("attempts_per_1k", "Tool attempts per 1k agent-steps"),
            ("recipe_hit", f"Recipe hit rate (chance = {CHANCE:.2f})"),
            ("has_tool", "Share of agents carrying a tool"),
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
