"""
v3.8 analysis -- staged growth.

Second-half aggregates are computed PER PHASE, and are event-weighted (sum of correct
attempts over sum of attempts, sum of safe meals over sum of meals) rather than means of
per-window ratios: a mean-of-ratios artifact produced the v2 "ratchet" claim that was retracted.

Two window conventions, and the rows say which they use:
  phase_half(k)  the second half of phase k.  For a staged run that is steps 4000-8000
                 (phase 1) and 12000-16000 (phase 2).
  tail()         the last quarter of the run, 12000-16000, the SAME absolute window for every
                 condition.  Row 5 needs this: the from-scratch control has no phase 1, so its
                 "second half" would be steps 8000-16000 and would not be matched to the staged
                 conditions' phase-2 second half.
For a staged run the two coincide on phase 2, by construction.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:                                  # plots are optional; the numbers are not
    plt = None

from sim import Config, run

# ---------------------------------------------------------------------------
# the world
# ---------------------------------------------------------------------------
# Phase 1 is v3.1's flip world, and the acceptance gate is whether it reproduces v3.1.  So the
# METABOLISM is v3.1's throughout (repro 3.0 / cost 1.5 / max_energy 5.0 / max_pop 400), not
# v3.6's (4.5 / 2.25 / 8.0 / 600).  v3.6 tuned those for a SCAFFOLDED recipe world; applying them
# at the phase boundary would make the transition two changes at once -- the chain switching on
# AND the cost of living changing -- and the transition is the thing being measured.  What is
# taken from v3.6's WORLD is the chain itself: items, stations, nuts, their values.
WORLD = dict(
    # v3.1's flip world (phase 1), carried through phase 2 unchanged
    flip_every=300, eta_init=0.2, hidden=24,
    spawn_per_patch=3.0, food_value=0.7, poison_value=0.5,
    repro_threshold=3.0, repro_cost=1.5, max_energy=5.0, max_pop=400, init_pop=300,
    # the chain, from v3.6's WORLD (phase 2 only -- inert while cfg.chain is False)
    items_per_step=8.0, stations_per_type=60, nuts_uniform=8.0,
    nut_value=1.3, tool_break=0.4, recipe_every=2000, fail_cost=0.0,
    # no instinct of any kind.  nav_dir / nav_here stay in the genome and reach nothing.
    scaffold_food=False, scaffold_chain=False,
)

PHASE_STEPS = 8000
STAGED = [dict(n_steps=PHASE_STEPS, chain=False), dict(n_steps=PHASE_STEPS, chain=True)]
SCRATCH = [dict(n_steps=2 * PHASE_STEPS, chain=True)]

VARIANTS = {
    # baseline and gate.  Also the drift reference for eta1/eta2/lam2 and the scaffold genes.
    "fixed (staged)":       dict(kw=dict(mode="fixed", **WORLD), phases=STAGED),
    # same plasticity, same H magnitudes, random-sign modulator, no information.
    # This control carries the claim in phase 2, not `fixed`.
    "scrambled (staged)":   dict(kw=dict(mode="plastic", plastic_layers="W2", scramble=True, **WORLD),
                                 phases=STAGED),
    # the result condition
    "plastic (W2) staged":  dict(kw=dict(mode="plastic", plastic_layers="W2", **WORLD), phases=STAGED),
    # is staging needed at all?  Same learner, full world from step 0, same total steps.
    "plastic (W2) scratch": dict(kw=dict(mode="plastic", plastic_layers="W2", **WORLD), phases=SCRATCH),
}

COLORS = {"fixed (staged)": "tab:red", "scrambled (staged)": "black",
          "plastic (W2) staged": "tab:blue", "plastic (W2) scratch": "tab:orange"}

CHANCE = 1.0 / 6.0
MARGIN = 0.03
SEED_RULE = 4          # min(SEED_RULE, n_seeds): a 3-seed pass reads as 3/3, five seeds as 4/5


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


# ---------------------------------------------------------------- summary

ROWS = [
    ("safe_rate",          safe),
    ("recipe_hit",         hit),
    ("attempts/1k",        lambda L: per_1k(L, "n_attempts_raw")),
    ("nuts/1k",            lambda L: per_1k(L, "n_nuts")),
    ("has_tool",           lambda L: half(L, "has_tool")),
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
    ("bridge_first",       lambda L: half(L, "bridge_first")),
    ("crop_safe",          lambda L: half(L, "crop_safe")),
    ("nut_share",          lambda L: half(L, "nut_share")),
    ("meal gain",          lambda L: float(curve(L, "meal")[-1] - curve(L, "meal")[0])),
    ("attempt gain",       lambda L: float(curve(L, "att")[-1] - curve(L, "att")[0])),
]


def summary(results):
    names = list(results)
    w = 24
    nseed = len(results[names[0]])
    print("=" * (20 + w * len(names)))
    print(f"v3.8 -- growing the learner with its world.  {nseed} seeds.  Chance recipe hit "
          f"{CHANCE:.3f}, chance safe rate 0.500")
    print("=" * (20 + w * len(names)))

    for label, sel in [("PHASE 1 (food only) -- second half", lambda r: phase_half(r, 0)),
                       ("PHASE 2 (chain on) -- second half", lambda r: phase_half(r, 1))]:
        print(f"\n### {label}")
        if label.startswith("PHASE 1"):
            print("    the from-scratch column is NOT a phase 1 -- its chain is on from step 0.")
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
    T = lambda r: tail(r)
    F, S, PL, SC = "fixed (staged)", "scrambled (staged)", "plastic (W2) staged", "plastic (W2) scratch"
    v = lambda n, sel, f: ps(results, n, sel, f)
    n_ok = lambda a, b: int(np.sum(np.asarray(a) - np.asarray(b) >= MARGIN))

    print("\n" + "-" * 78)
    print(f"DECISION NUMBERS (a difference counts when it is >= {MARGIN} in {rule}/{nseed} seeds)")
    print("-" * 78)

    print("\nrow 0  uninterpretable?  Checked PER PHASE.")
    for lab, sel in [("phase 1", P1), ("phase 2", P2)]:
        for n in names:
            pop, inj = v(n, sel, lambda L: half(L, "pop")), v(n, sel, lambda L: half(L, "injections"))
            flag = "  <-- EXCLUDE" if (np.nanmin(pop) < 80 or np.nanmax(inj) > 0) else ""
            print(f"  {lab}  {n:<24} pop {np.round(pop,0).tolist()}  inj {np.round(inj,1).tolist()}{flag}")

    print("\nrow 1  PHASE-1 GATE -- is this v3.1, with 60 inputs and 24 hidden?")
    sp, sf = v(PL, P1, safe), v(F, P1, safe)
    pf = v(PL, P1, lambda L: half(L, "probe_adv_food"))
    e2p, e2f = v(PL, P1, lambda L: half(L, "eta2")), v(F, P1, lambda L: half(L, "eta2"))
    print(f"  safe_rate  plastic - fixed:      {np.round(sp - sf, 3).tolist()}   >= +{MARGIN} in {n_ok(sp, sf)}/{nseed}   (v3.1: +0.08)")
    print(f"    plastic {np.round(sp,3).tolist()}   fixed {np.round(sf,3).tolist()}   (v3.1: 0.60-0.66 vs 0.51-0.56)")
    print(f"  probe_adv (food) plastic:        {np.round(pf,3).tolist()}   (target >= 1.0; v3.1: 1.4-2.7)")
    print(f"  eta2  plastic {np.round(e2p,3).tolist()}  vs fixed {np.round(e2f,3).tolist()}   above in {int(np.sum(e2p > e2f))}/{nseed}")
    print("  If this does not reproduce, the finding is about observation size and everything below is void.")

    print("\nrow 2  TRANSITION -- does the grown population survive the chain, and keep its plasticity?")
    for n in (PL, S, F):
        pop = v(n, P2, lambda L: half(L, "pop")); inj = v(n, P2, lambda L: half(L, "injections"))
        print(f"  {n:<24} phase-2 pop {np.round(pop,0).tolist()}  inj {np.round(inj,1).tolist()}")
    for g in ("eta2", "lam2"):
        a1, a2 = v(PL, P1, lambda L: half(L, g)), v(PL, P2, lambda L: half(L, g))
        f2 = v(F, P2, lambda L: half(L, g))
        print(f"  {g}  plastic phase 1 {np.round(a1,3).tolist()} -> phase 2 {np.round(a2,3).tolist()}"
              f"   fixed phase 2 {np.round(f2,3).tolist()}   above fixed in {int(np.sum(a2 > f2))}/{nseed}")
    print("  v3.6 had eta2 selected OFF and lam2 selected SHORT in the scaffolded world.  If that")
    print("  repeats here, the chain does it, not the scaffold; if it does not, the scaffold did.")

    print("\nrow 3  does approach behaviour for the chain evolve UNWIRED?")
    for n in (PL, S, F, SC):
        a_early = v(n, lambda r: window(r, r["chain_start"], r["chain_start"] + (r["n_steps"] - r["chain_start"]) // 4),
                    lambda L: per_1k(L, "n_attempts_raw"))
        a_late = v(n, P2, lambda L: per_1k(L, "n_attempts_raw"))
        tool = v(n, P2, lambda L: half(L, "has_tool"))
        print(f"  {n:<24} attempts/1k early {np.round(a_early,2).tolist()} -> late {np.round(a_late,2).tolist()}"
              f"   has_tool {np.round(tool,3).tolist()}   rising in {int(np.sum(a_late > a_early))}/{nseed}")
    print("  Above zero AND rising is the row.  Flat at zero means the chain was never found and")
    print("  row 4 cannot be read at all -- there are no attempts to compute a hit rate over.")

    print("\nrow 4  THE RECIPE (v3.6's rows 3/6/8/10b, applied to phase 2)")
    hp, hf, hs = v(PL, P2, hit), v(F, P2, hit), v(S, P2, hit)
    print(f"  plastic - fixed:      {np.round(hp - hf,3).tolist()}   >= +{MARGIN} in {n_ok(hp, hf)}/{nseed}")
    print(f"  plastic - scrambled:  {np.round(hp - hs,3).tolist()}   >= +{MARGIN} in {n_ok(hp, hs)}/{nseed}   <- scrambled carries the claim")
    print(f"  recipe_hit            plastic {np.round(hp,3).tolist()}  fixed {np.round(hf,3).tolist()}  scrambled {np.round(hs,3).tolist()}   (chance {CHANCE:.3f})")
    pr = v(PL, P2, lambda L: half(L, "probe_adv"))
    print(f"  probe_adv (recipe) plastic:      {np.round(pr,3).tolist()}   (> 0 in {int(np.sum(pr > 0))}/{nseed})")
    ao, ay = v(PL, P2, lambda L: half(L, "hit_old")), v(PL, P2, lambda L: half(L, "hit_young"))
    ag = v(PL, P2, lambda L: float(curve(L, "att")[-1] - curve(L, "att")[0]))
    print(f"  REQUIRED within-life signature -- at least one, in {rule}/{nseed}:")
    print(f"    hit_old - hit_young:           {np.round(ao - ay,3).tolist()}   (> 0 in {int(np.sum(ao > ay))}/{nseed})")
    print(f"    attempt-in-life curve gain:    {np.round(ag,3).tolist()}   (> 0 in {int(np.sum(ag > 0))}/{nseed})")
    b, tw = v(PL, P2, lambda L: half(L, "bridge_first")), v(PL, P2, lambda L: half(L, "trace_weight"))
    print(f"  bridge_first {np.round(b,1).tolist()}   lam2^gap {np.round(tw,3).tolist()}   (if ~0, a null is trace length, not the conjunction)")

    print("\nrow 5  STAGED vs FROM SCRATCH -- is staging the method, or unnecessary?")
    print("  matched window: the last quarter of the run, the same absolute steps in both.")
    for lab, f in [("pop", lambda L: half(L, "pop")), ("injections", lambda L: half(L, "injections")),
                   ("attempts/1k", lambda L: per_1k(L, "n_attempts_raw")), ("recipe_hit", hit),
                   ("safe_rate", safe), ("probe_adv (food)", lambda L: half(L, "probe_adv_food")),
                   ("eta2", lambda L: half(L, "eta2"))]:
        a, b_ = v(PL, T, f), v(SC, T, f)
        print(f"  {lab:<18} staged {np.round(a,3).tolist()}   scratch {np.round(b_,3).tolist()}")
    print("  Matching on population and attempts => staging is not needed.  Scratch collapsing")
    print("  (pop < 80 or injections > 0) while staged survives => staging is the method.")

    print("\nrow 6  the scaffold genes, unwired -- do they drift, having nothing to reach?")
    for g in ("nav_dir", "nav_here"):
        print(f"  {g}   phase 1 -> phase 2")
        for n in names:
            print(f"    {n:<24} {np.round(v(n,P1,lambda L: half(L,g)),2).tolist()} -> {np.round(v(n,P2,lambda L: half(L,g)),2).tolist()}")
    print("  Nothing is wired to these in this notebook, so they are DEAD genes here and this is")
    print("  the drift scale for a heritable scalar over a run: sigma 0.2 x sqrt(generations).")
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
