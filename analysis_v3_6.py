"""
Analysis for v3.6.  Second-half aggregates are EVENT-WEIGHTED (sum of correct
attempts over sum of attempts), not means of per-window ratios: a mean-of-ratios
artifact produced the v2 'ratchet' claim that had to be retracted.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:                                  # plots are optional; the numbers are not
    plt = None

from sim import Config, run

NO_FLIP = 10 ** 9                                  # the safe food never changes

# The world.  These are NOT v2.9b's numbers.  Nine tuning passes, every one judged against
# `fixed`, the hand-wired ceiling, or the POSITIVE CONTROL (the v3.1 food effect) -- never
# against a plastic condition's recipe hit rate:
#   items_per_step 1.5 -> 8, stations_per_type 20 -> 60   attempts/life 1.0 -> 4.4
#   nuts_uniform 0 -> 8                                   station->nut bridge 57 -> 8 steps, so
#                                                         lam2 decides rather than trace arithmetic
#   repro_threshold 3.0 -> 4.5 (cost 2.25, max_energy 8)  population off the cap
#   spawn_per_patch 1.0 -> 3.0                            nuts down to ~40% of energy income, so the
#                                                         balanced +/-1 food modulator is not swamped
#   innate_scale applied to the SCAFFOLD UNITS ONLY       v2.9b scaled the whole network by 0.1;
#     (sim.Config.n_scaffold = 10)                        that leaves H2 no basis to read and killed
#                                                         the v3.1 food effect outright.  Applying it
#                                                         to nothing killed navigation instead.
# fail_cost stays 0 (pure delayed credit).  fail_cost = 0.3, v2's STAKES value, collapses every
# population that does not already know the recipe (5 of 6 attempts fail) and leaves only the
# hand-wired ceiling standing -- it changes which agents survive, not just what they learn.
# Acceptance met at seed 0 / 5000 steps: fixed 0.172 (chance 0.167), ceiling 0.226,
# attempts/life 4.4, bridge_first ~8, pop 379-467 of 600, zero injections, probe_adv (food) 0.125.
# NOT met: eta2 is selected DOWN (0.036 vs fixed's drift 0.111) and the food advantage does not
# reach the fitness level (safe_rate +0.003).  Both are pre-registered in the decision table.
WORLD = dict(
    items_per_step=8.0, stations_per_type=60, nuts_uniform=8.0, spawn_per_patch=3.0,
    nut_value=1.3, tool_break=0.4, max_pop=600, repro_threshold=4.5, repro_cost=2.25, max_energy=8.0,
)

NO_FLIP = 10 ** 9                                  # the safe food never changes

FAIL_COST = 0.05      # small enough not to change who survives (0.3 collapsed every population
                      # that did not already know the recipe); large enough that a wrong attempt
                      # produces m = -1 at the station.  This makes the conjunction partly
                      # learnable BY ELIMINATION, which is a different and easier question than
                      # pure delayed credit -- hence a separate condition, never the default.

VARIANTS = {
    # baseline and gate: what the genome alone does with a recipe that changes every ~8 generations.
    # Also the reference for eta1/eta2/lam2 drift and for the scaffold genes, measured in this world.
    "fixed":                  dict(mode="fixed", **WORLD),
    # same plasticity machinery, same H magnitudes, same power to override the innate scaffold;
    # random-sign modulator, no information.  This control carries the claim, not `fixed`.
    "scrambled":              dict(mode="plastic", plastic_layers="W2", scramble=True, **WORLD),
    # the result condition
    "plastic (W2)":           dict(mode="plastic", plastic_layers="W2", **WORLD),
    # the ceiling: exact per-pair credit, hand-wired, steering navigation and a soft veto.
    # It acts through a DIFFERENT channel from the learner (navigation preference and a veto,
    # not the network's output), so it is a reference level, not a matched comparison.
    "fixed + B (ceiling)":    dict(mode="fixed", private_mem=True, pref_gain=8.0, veto_p=0.9, **WORLD),
    # the elimination pair.  The fixed arm is not optional: without it, any gain in
    # `plastic + fail cost` could be the 0.05 energy change rather than the -1 at the station.
    "fixed + fail cost":      dict(mode="fixed", fail_cost=FAIL_COST, **WORLD),
    "plastic (W2) + fail cost": dict(mode="plastic", plastic_layers="W2", fail_cost=FAIL_COST, **WORLD),
}

COLORS = {"fixed": "tab:red", "scrambled": "black", "plastic (W2)": "tab:blue",
          "plastic (both)": "tab:green", "fixed + B (ceiling)": "tab:purple",
          "fixed + fail cost": "tab:pink", "plastic (W2) + fail cost": "tab:cyan"}

CHANCE = 1.0 / 6.0
MARGIN = 0.03          # the project's standard margin
SEED_RULE = 4          # ... in 4 of 5 seeds; min(SEED_RULE, n_seeds), so a 3-seed pass is 3/3


# ---------------------------------------------------------------- running

def run_experiment(seeds, n_steps, variants=VARIANTS, verbose=False, save_path=None, **overrides):
    """Sequential, single-process: this has to run on a Colab CPU runtime, so no multiprocessing.
    If save_path is given the results are pickled after every run, so a dropped session loses one
    run rather than the lot -- reload with  pickle.load(open(save_path, 'rb'))."""
    import pickle, time
    results = {v: [] for v in variants}
    total = len(seeds) * len(variants)
    done, t0 = 0, time.time()
    for seed in seeds:
        for name, kw in variants.items():
            t1 = time.time()
            results[name].append(run(Config(n_steps=n_steps, seed=seed, **kw, **overrides), verbose=verbose))
            done += 1
            el = time.time() - t0
            print(f"[{done}/{total}] {name} seed={seed}  {time.time()-t1:.0f}s   "
                  f"elapsed {el/60:.1f} min, est. total {el/done*total/60:.0f} min", flush=True)
            if save_path:
                with open(save_path, "wb") as f:
                    pickle.dump(results, f)
    return results


# ---------------------------------------------------------------- aggregation

def second_half(log):
    return log[len(log) // 2:]


def half(log, key):
    v = np.array([r[key] for r in second_half(log)], dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def rate(log, num, den):
    """Event-weighted rate over the second half."""
    L = second_half(log)
    d = float(np.sum([r[den] for r in L]))
    return float(np.sum([r[num] for r in L]) / d) if d > 0 else np.nan


def hit(log):
    return rate(log, "n_correct", "n_attempts_raw")


def safe(log):
    return rate(log, "n_safe", "n_eats")


def per_1k(log, key):
    L = second_half(log)
    s = float(np.sum([r["agent_steps"] for r in L]))
    return float(1000.0 * np.sum([r[key] for r in L]) / s) if s > 0 else np.nan


def curve(log, key):
    """Event-weighted curve over the second half: hit/safe rate by event number."""
    L = second_half(log)
    n = np.sum([r[key + "_n"] for r in L], axis=0)
    c = np.sum([r[key + ("_safe" if key == "meal" else "_correct")] for r in L], axis=0)
    return np.where(n > 0, c / np.maximum(n, 1), np.nan), n


def per_seed(results, name, fn):
    return [fn(r["log"]) for r in results[name]]


def n_seeds_above(a, b, margin):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return int(np.sum(a - b >= margin))


# ---------------------------------------------------------------- recovery

def era_phase(log, changes, n_steps, window=500):
    """Hit rate in the first `window` steps of each recipe era and in the last `window`,
    event-weighted, pooled over eras.  Always defined, unlike a threshold-crossing time.
    late > early = the population is finding the new recipe within the era."""
    bounds = [0] + list(changes) + [n_steps]
    early_c = early_n = late_c = late_n = 0.0
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        for r in log:
            if s < r["t"] <= min(s + window, e):
                early_c += r["n_correct"]; early_n += r["n_attempts_raw"]
            elif max(e - window, s) < r["t"] <= e:
                late_c += r["n_correct"]; late_n += r["n_attempts_raw"]
    return (early_c / early_n if early_n else np.nan,
            late_c / late_n if late_n else np.nan)


def smooth(y, k=5):
    y = np.asarray(y, float); out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]; w = w[~np.isnan(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


def recovery_table(results, threshold=0.30):
    """Steps after each recipe change until the smoothed hit rate exceeds `threshold`.
    The search is bounded by the NEXT change (the unbounded version was a bug fixed in v2.8)."""
    any_run = next(iter(results.values()))[0]
    changes = any_run["recipe_changes"]
    names = list(results)
    print(f"Steps until smoothed recipe hit > {threshold} after each recipe change (chance = {CHANCE:.3f})")
    print(f"{'change at':>10}" + "".join(f"{n:>24}" for n in names))
    for c in changes:
        row = []
        for n in names:
            vals = []
            for run_ in results[n]:
                log = run_["log"]
                h = smooth([r["recipe_hit"] for r in log])
                ts = [r["t"] for r in log]
                nxt = next((c2 for c2 in changes if c2 > c), ts[-1] + 1)
                vals.append(next((t - c for t, hv in zip(ts, h) if c < t < nxt and hv > threshold), None))
            ok = [x for x in vals if x is not None]
            row.append(f"{np.mean(ok):.0f} ({len(ok)}/{len(vals)})" if ok else f"never (0/{len(vals)})")
        print(f"{c:>10}" + "".join(f"{v:>24}" for v in row))
    print()


# ---------------------------------------------------------------- summary

def summary(results):
    names = list(results)
    n_steps = next(iter(results.values()))[0]["cfg"]["n_steps"]
    changes = next(iter(results.values()))[0]["recipe_changes"]
    w = 24

    rows = [
        ("recipe_hit",        hit),
        ("attempts/1k",       lambda L: per_1k(L, "n_attempts_raw")),
        ("nuts/1k",           lambda L: per_1k(L, "n_nuts")),
        ("safe_rate",         safe),
        ("pop",               lambda L: half(L, "pop")),
        ("has_tool",          lambda L: half(L, "has_tool")),
        ("holding_item",      lambda L: half(L, "holding_item")),
        ("hit_young",         lambda L: half(L, "hit_young")),
        ("hit_old",           lambda L: half(L, "hit_old")),
        ("probe_adv (recipe)", lambda L: half(L, "probe_adv")),
        ("pair_gain learned", lambda L: half(L, "pair_gain")),
        ("pair_gain innate",  lambda L: half(L, "pair_gain_innate")),
        ("probe_adv (food)",  lambda L: half(L, "probe_adv_food")),
        ("eta1 (gene)",       lambda L: half(L, "eta1")),
        ("eta2 (gene)",       lambda L: half(L, "eta2")),
        ("lam2 (gene)",       lambda L: half(L, "lam2")),
        ("h_norm",            lambda L: half(L, "h_norm")),
        ("bridge_steps",      lambda L: half(L, "bridge_steps")),
        ("trace_weight",      lambda L: half(L, "trace_weight")),
        ("nut_share",         lambda L: half(L, "nut_share")),
        ("crop_safe",         lambda L: half(L, "crop_safe")),
        ("max_gen",           lambda L: half(L, "max_gen")),
        ("injections",        lambda L: half(L, "injections")),
        ("attempt gain",      lambda L: float(curve(L, "att")[0][-1] - curve(L, "att")[0][0])),
        ("recipe-era gain",   lambda L: float(curve(L, "rec")[0][-1] - curve(L, "rec")[0][0])),
        ("meal gain",         lambda L: float(curve(L, "meal")[0][-1] - curve(L, "meal")[0][0])),
    ]

    print("=" * (20 + w * len(names)))
    print(f"v3.6 -- the grown learner on the recipe task alone.  Second half of the run, "
          f"{len(results[names[0]])} seeds.  Chance recipe hit = {CHANCE:.3f}")
    print("=" * (20 + w * len(names)))
    print(f"{'metric':<20}" + "".join(f"{n:>{w}}" for n in names))
    for label, f in rows:
        print(f"{label:<20}" + "".join(f"{np.nanmean(per_seed(results, n, f)):>{w}.3f}" for n in names))

    print("\nper seed, second half:")
    for label, f in [("recipe_hit", hit), ("attempts/1k", lambda L: per_1k(L, "n_attempts_raw")),
                     ("nuts/1k", lambda L: per_1k(L, "n_nuts")), ("safe_rate", safe),
                     ("probe_adv (recipe)", lambda L: half(L, "probe_adv")),
                     ("pair_gain innate", lambda L: half(L, "pair_gain_innate")),
                     ("probe_adv (food)", lambda L: half(L, "probe_adv_food")),
                     ("eta2", lambda L: half(L, "eta2")), ("eta1", lambda L: half(L, "eta1")),
                     ("lam2", lambda L: half(L, "lam2")), ("pop", lambda L: half(L, "pop")),
                     ("injections", lambda L: half(L, "injections"))]:
        print(f"  {label}")
        for n in names:
            print(f"    {n:<24} {[round(x, 3) for x in per_seed(results, n, f)]}")

    print("\nrecipe-era phase (event-weighted hit rate, first 500 steps of an era vs last 500):")
    print(f"{'condition':<24}{'early':>10}{'late':>10}{'late-early':>12}   per-seed (late-early)")
    for n in names:
        ep = [era_phase(r["log"], changes, n_steps) for r in results[n]]
        e = np.nanmean([x[0] for x in ep]); l = np.nanmean([x[1] for x in ep])
        d = [round(x[1] - x[0], 3) for x in ep]
        print(f"{n:<24}{e:>10.3f}{l:>10.3f}{l - e:>12.3f}   {d}")

    _decision_numbers(results)


def _decision_numbers(results):
    names = list(results)
    def ps(n, f):
        return np.array(per_seed(results, n, f), dtype=float)

    H = {n: ps(n, hit) for n in names}
    S = {n: ps(n, safe) for n in names}
    A_ = {n: ps(n, lambda L: per_1k(L, "n_attempts_raw")) for n in names}
    P = {n: ps(n, lambda L: half(L, "probe_adv")) for n in names}
    PF = {n: ps(n, lambda L: half(L, "probe_adv_food")) for n in names}
    OY = {n: ps(n, lambda L: half(L, "hit_old") - half(L, "hit_young")) for n in names}
    AG = {n: ps(n, lambda L: float(curve(L, "att")[0][-1] - curve(L, "att")[0][0])) for n in names}
    nseed = len(H[names[0]])
    rule = min(SEED_RULE, nseed)          # so a QUICK 1-seed smoke run does not print "4/1"
    ok = lambda a, b: int(np.sum(np.asarray(a) - np.asarray(b) >= MARGIN))

    def diff(a, b):
        print(f"  {a + ' - ' + b:<44} {np.round(H[a] - H[b], 3).tolist()}   >= +{MARGIN} in {ok(H[a], H[b])}/{nseed}")

    print("\n" + "-" * 78)
    print(f"DECISION NUMBERS (a difference counts when it is >= {MARGIN} in {rule}/{nseed} seeds)")
    print("-" * 78)

    print("\nrow 0  is any condition uninterpretable?")
    for n in names:
        pop, inj = ps(n, lambda L: half(L, "pop")), ps(n, lambda L: half(L, "injections"))
        flag = "  <-- EXCLUDE" if (pop.min() < 80 or inj.max() > 0 or A_[n].min() < 5.0) else ""
        print(f"  {n:<26} pop {np.round(pop, 0).tolist()}  inj {np.round(inj, 1).tolist()}  att/1k {np.round(A_[n], 1).tolist()}{flag}")

    print("\nrow 1  gate -- can the genome track the recipe on its own?")
    print(f"  fixed recipe_hit:               {np.round(H['fixed'], 3).tolist()}   (chance {CHANCE:.3f}; gate fires at >= 0.25 in {rule}/{nseed})")
    print(f"  fixed pair_gain innate:         {np.round(ps('fixed', lambda L: half(L, 'pair_gain_innate')), 3).tolist()}")

    print("\nrow 2  ceiling -- does exact pair credit pay, and is there headroom to detect a learner?")
    print(f"  fixed + B (ceiling):            {np.round(H['fixed + B (ceiling)'], 3).tolist()}   (v2: 0.23-0.36)")
    diff("fixed + B (ceiling)", "fixed")
    gap = H['fixed + B (ceiling)'] - H['fixed']
    print(f"  headroom (target >= 0.10 in {rule}/{nseed}): mean {np.nanmean(gap):.3f}; "
          f"a learner must capture {MARGIN / max(np.nanmean(gap), 1e-9) * 100:.0f}% of the ceiling to clear the margin")
    print(f"  attempts/life:                  {np.round(ps('fixed', lambda L: half(L, 'attempts_per_life')), 2).tolist()} (fixed)  "
          f"{np.round(ps('fixed + B (ceiling)', lambda L: half(L, 'attempts_per_life')), 2).tolist()} (ceiling)")
    print(f"    elimination predicts a ceiling of mean_k 1/(7-k) over k = 1..attempts/life")

    print("\nrow 3  acquisition.  ALL FOUR lines must hold, plus one within-life signature.")
    diff("plastic (W2)", "fixed")
    diff("plastic (W2)", "scrambled")
    print(f"  probe_adv (recipe) plastic (W2): {np.round(P['plastic (W2)'], 3).tolist()}   (> 0 in {int(np.sum(P['plastic (W2)'] > 0))}/{nseed})")
    print(f"  probe_adv (recipe) scrambled:    {np.round(P['scrambled'], 3).tolist()}")
    print(f"  attempts/1k vs fixed (ratio):    {np.round(A_['plastic (W2)'] / np.maximum(A_['fixed'], 1e-9), 2).tolist()}   (must be >= 0.8: abstention is not knowledge)")
    print("  REQUIRED within-life signature -- at least one of these, in 4/5:")
    print(f"    hit_old - hit_young:           {np.round(OY['plastic (W2)'], 3).tolist()}   (> 0 in {int(np.sum(OY['plastic (W2)'] > 0))}/{nseed});  fixed: {np.round(OY['fixed'], 3).tolist()}")
    print(f"    attempt-in-life curve gain:    {np.round(AG['plastic (W2)'], 3).tolist()}   (> 0 in {int(np.sum(AG['plastic (W2)'] > 0))}/{nseed});  fixed: {np.round(AG['fixed'], 3).tolist()}")
    print("    (the ceiling shows what a real one looks like:"
          f" {np.round(AG['fixed + B (ceiling)'], 3).tolist()})")

    print("\nrows 4/5  plasticity without information, and abstention")
    diff("scrambled", "fixed")

    print("\nrows 6/7  positive control -- does the learner work in this world at all?")
    print(f"  safe_rate plastic (W2) - fixed: {np.round(S['plastic (W2)'] - S['fixed'], 3).tolist()}   >= +{MARGIN} in {ok(S['plastic (W2)'], S['fixed'])}/{nseed}   (v3.1: +0.08)")
    print(f"  probe_adv (food) plastic (W2):  {np.round(PF['plastic (W2)'], 3).tolist()}   (v3.1: ~2.3; acceptance >= 0.5)")
    print(f"  probe_adv (food) scrambled:     {np.round(PF['scrambled'], 3).tolist()}")

    print("\nrow 8  bridge feasibility -- is a null about trace length rather than the conjunction?")
    for n in names:
        b, w = ps(n, lambda L: half(L, "bridge_first")), ps(n, lambda L: half(L, "trace_weight"))
        print(f"  {n:<26} bridge_first {np.round(b, 1).tolist()}  lam2^gap {np.round(w, 3).tolist()}")

    print("\nrow 10b  was plasticity selected off before the question was reached?")
    print("  eta and lam are DEAD genes in the fixed conditions, so those rows are this world's drift.")
    plastics = [n for n in names if "plastic" in n or n == "scrambled"]
    for g in ("eta2", "lam2", "eta1"):
        print(f"  {g}   per seed, then (condition - fixed)")
        base = ps("fixed", lambda L: half(L, g))
        for n in names:
            v = ps(n, lambda L: half(L, g))
            tail = f"   vs fixed: {np.round(v - base, 3).tolist()}" if n in plastics else "   <- drift reference" if n == "fixed" else ""
            print(f"    {n:<26} {np.round(v, 3).tolist()}{tail}")
    print("  pre-run checks had eta2 0.033/0.049 against fixed's 0.134/0.043, and lam2 0.447/0.331")
    print("  against fixed's 0.735/0.765 -- selection SHORTENING the trace that bridging requires.")

    if "plastic (both)" in results:
        print("\nrow 11  does W1 plasticity build the conjunction?")
        diff("plastic (both)", "plastic (W2)")

    print("\nrow 12  the scaffold genes -- what balance did evolution set between instinct and override?")
    for g in ("nav_dir", "nav_here"):
        print(f"  {g}")
        for n in names:
            print(f"    {n:<26} {np.round(ps(n, lambda L: half(L, g)), 2).tolist()}")
    print("  these are NOT dead genes in any condition -- they set behaviour everywhere -- so the")
    print("  reference is `fixed`'s value, not a drift range: a plastic condition evolving a LOWER")
    print("  instinct than `fixed` is evolution buying room for the override, which is the")
    print("  mechanism row 3 needs; equal values mean the instinct/override balance did not move.")

    print("\n" + "=" * 78)
    print("row 13  ELIMINATION -- the row this pass exists to test.")
    print("  Nothing in the pre-run checks measured `plastic (W2) + fail cost`: with fail_cost = 0.05")
    print("  a wrong attempt produces m = -1 AT the station, so the conjunction becomes partly")
    print("  learnable by elimination rather than purely by delayed credit.")
    print("=" * 78)
    print("  the result line:")
    diff("plastic (W2) + fail cost", "fixed + fail cost")
    print("  the control line -- read this FIRST:")
    diff("fixed + fail cost", "fixed")
    print("    if the fixed arm moves too, the 0.05 energy cost changed the world rather than the")
    print("    signal, and the result line cannot be read as learning.")
    print("  for reference (confounds a world change with a signal change; not a decision line):")
    diff("plastic (W2) + fail cost", "plastic (W2)")
    pf, p = "plastic (W2) + fail cost", "plastic (W2)"
    print(f"  probe_adv (recipe)  {pf}: {np.round(P[pf], 3).tolist()}   {p}: {np.round(P[p], 3).tolist()}")
    print(f"  attempts/1k ratio vs its own fixed arm: {np.round(A_[pf] / np.maximum(A_['fixed + fail cost'], 1e-9), 2).tolist()}")
    print(f"  within-life signature (attempt-in-life gain) {pf}: {np.round(AG[pf], 3).tolist()}")
    print(f"  hit_old - hit_young {pf}: {np.round(OY[pf], 3).tolist()}")
    print("-" * 78)


# ---------------------------------------------------------------- curves / plots

def curves(results, key, xlabel, show=True):
    label = {"att": "attempt number in an agent's life",
             "rec": "attempts since the last recipe change",
             "meal": "meal number in an agent's life"}[key]
    base = CHANCE if key in ("att", "rec") else 0.5
    print(f"\n{label}, second half, event-weighted, mean over seeds (baseline {base:.3f}):")
    for name, res in results.items():
        C = np.array([curve(r["log"], key)[0] for r in res])
        m = np.nanmean(C, 0)
        print(f"  {name:<24} {np.round(m, 3).tolist()}   gain = {m[-1] - m[0]:+.3f}")
    if plt is None or not show:
        return
    plt.figure(figsize=(8, 4.5))
    for name, res in results.items():
        C = np.array([curve(r["log"], key)[0] for r in res])
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
    keys = [("recipe_hit", f"Recipe hit rate (chance = {CHANCE:.2f})"),
            ("nuts_per_1k", "Nuts cracked per 1k agent-steps"),
            ("attempts_per_1k", "Tool attempts per 1k agent-steps"),
            ("probe_adv", "Within-agent counterfactual on the RECIPE"),
            ("probe_adv_food", "Within-agent counterfactual on FOOD (positive control)"),
            ("safe_rate", "Safe-eating rate (chance = 0.5)"),
            ("eta2", "Output-layer learning-rate gene"),
            ("lam2", "Output-layer eligibility-trace gene"),
            ("pop", "Population")]
    any_run = next(iter(results.values()))[0]
    fig, axes = plt.subplots(5, 2, figsize=(13, 18))
    for ax, (key, title) in zip(axes.ravel(), keys):
        for name, res in results.items():
            t, Y = series(res, key)
            if not np.isfinite(Y).any():
                continue
            Ys = np.array([smooth(y) for y in Y])
            ax.plot(t, np.nanmean(Ys, 0), color=COLORS.get(name), label=name)
            ax.fill_between(t, np.nanmin(Ys, 0), np.nanmax(Ys, 0), color=COLORS.get(name), alpha=0.10)
        for c in any_run["recipe_changes"]:
            ax.axvline(c, color="k", ls="--", lw=1)
        ax.set_title(title, fontsize=10); ax.set_xlabel("step")
    axes[0, 0].axhline(CHANCE, color="grey", lw=0.5)
    axes[2, 1].axhline(0.5, color="grey", lw=0.5)
    axes[0, 0].legend(fontsize=7)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    plt.suptitle("dashed = recipe change.  Lines are running means; bands = min/max over seeds", fontsize=10)
    plt.tight_layout(); plt.show()
