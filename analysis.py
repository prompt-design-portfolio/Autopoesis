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

# The world.  These are NOT v2.9b's numbers: the first smoke test on v2.9b's settings gave
# ~1 tool attempt per agent lifetime and a hand-wired ceiling at chance -- a null by
# construction.  Five tuning passes moved four quantities, and each was checked against
# `fixed` and the hand-wired ceiling ONLY, never against a plastic condition:
#   items_per_step 1.5 -> 8, stations_per_type 20 -> 60   (attempts/life 1.0 -> 6.6)
#   nuts_uniform 0 -> 8                                   (station->nut bridge 57 -> 7.5 steps,
#                                                          so lam2 decides the outcome instead
#                                                          of trace arithmetic deciding it)
#   repro_threshold 3.0 -> 4.5 (birth cost 2.25, max_energy 8)
#                                                         (population off the cap: at a hard cap
#                                                          births are a queue, not fecundity)
# Acceptance before the run: fixed at chance, ceiling >= 0.23, attempts/life >= 5,
# bridge_first ~ 5-15, population well below max_pop, no injections.
WORLD = dict(
    items_per_step=8.0, stations_per_type=60, nuts_uniform=8.0, spawn_per_patch=1.0,
    nut_value=1.0, max_pop=600, repro_threshold=4.5, repro_cost=2.25, max_energy=8.0,
)

NO_FLIP = 10 ** 9                                  # the safe food never changes

VARIANTS = {
    # baseline: what the genome alone does with a recipe that changes every ~8 generations
    "fixed":                  dict(mode="fixed", **WORLD),
    # same plasticity machinery, same H magnitudes, random-sign modulator.  Controls for
    # "H can override the innate scaffold, and mutation cannot" independently of information.
    "scrambled":              dict(mode="plastic", plastic_layers="W2", scramble=True, **WORLD),
    # the result condition
    "plastic (W2)":           dict(mode="plastic", plastic_layers="W2", **WORLD),
    # W1 plasticity could build the conjunction features H2 reads
    "plastic (both)":         dict(mode="plastic", plastic_layers="both", **WORLD),
    # the ceiling: exact per-pair credit, hand-wired, steering navigation and a soft veto.
    # It acts through a DIFFERENT channel from the learner (navigation preference and a veto,
    # not the network's output), so it is a reference level, not a matched comparison.
    "fixed + B (ceiling)":    dict(mode="fixed", private_mem=True, pref_gain=8.0, veto_p=0.9, **WORLD),
    # interference pair: with the food reversal removed, the recipe is the only moving target.
    # The fixed arm is what makes the plastic arm attributable (a no-flip world is simply richer).
    "fixed, no flip":         dict(mode="fixed", flip_every=NO_FLIP, **WORLD),
    "plastic (W2), no flip":  dict(mode="plastic", plastic_layers="W2", flip_every=NO_FLIP, **WORLD),
}

COLORS = {"fixed": "tab:red", "scrambled": "black", "plastic (W2)": "tab:blue",
          "plastic (both)": "tab:green", "fixed + B (ceiling)": "tab:purple",
          "fixed, no flip": "tab:pink", "plastic (W2), no flip": "tab:cyan"}

CHANCE = 1.0 / 6.0
MARGIN = 0.03          # the project's standard margin
SEED_RULE = 4          # ... in 4 of 5 seeds


# ---------------------------------------------------------------- running

def run_experiment(seeds, n_steps, variants=VARIANTS, verbose=True, **overrides):
    results = {v: [] for v in variants}
    for seed in seeds:
        for name, kw in variants.items():
            print(f"--- {name}  seed={seed}", flush=True)
            results[name].append(run(Config(n_steps=n_steps, seed=seed, **kw, **overrides), verbose=verbose))
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
    H = {n: np.array(per_seed(results, n, hit)) for n in names}
    S = {n: np.array(per_seed(results, n, safe)) for n in names}
    A = {n: np.array(per_seed(results, n, lambda L: per_1k(L, "n_attempts_raw"))) for n in names}
    P = {n: np.array(per_seed(results, n, lambda L: half(L, "probe_adv"))) for n in names}
    PF = {n: np.array(per_seed(results, n, lambda L: half(L, "probe_adv_food"))) for n in names}
    nseed = len(H[names[0]])

    def line(label, a, b=None):
        d = H[a] - (H[b] if b else 0)
        tag = f"{a} - {b}" if b else a
        k = n_seeds_above(H[a], H[b] if b else np.zeros(nseed), MARGIN) if b else ""
        print(f"  {tag:<42} {np.round(d, 3).tolist()}" + (f"   >= +{MARGIN} in {k}/{nseed}" if b else ""))

    print("\n" + "-" * 78)
    print("DECISION NUMBERS (rule: a difference counts when it is >= "
          f"{MARGIN} in {SEED_RULE}/{nseed} seeds)")
    print("-" * 78)
    print("\nrow 1  gate -- can the genome track the recipe on its own?")
    print(f"  fixed, recipe_hit per seed:                {np.round(H['fixed'], 3).tolist()}   (chance {CHANCE:.3f}; gate fires at >= 0.35 in {SEED_RULE}/{nseed})")
    print(f"  fixed, pair_gain innate:                   {np.round(np.array(per_seed(results, 'fixed', lambda L: half(L, 'pair_gain_innate'))), 3).tolist()}")

    print("\nrow 2/3  is there a within-life effect, and is it the information in m?")
    line("plastic (W2)", "fixed")
    line("plastic (W2)", "scrambled")
    line("scrambled", "fixed")
    print(f"  plastic (W2) probe_adv (recipe) per seed:   {np.round(P['plastic (W2)'], 3).tolist()}   (> 0 in {int(np.sum(P['plastic (W2)'] > 0))}/{nseed})")
    print(f"  scrambled    probe_adv (recipe) per seed:   {np.round(P['scrambled'], 3).tolist()}")
    print(f"  plastic (W2) attempts/1k vs fixed:          {np.round(A['plastic (W2)'] - A['fixed'], 2).tolist()}   (a hit rate bought by attempting less is not knowledge)")

    print("\nrow 4  positive control -- is the learner working in this world at all?")
    print(f"  safe_rate  plastic (W2) - fixed:           {np.round(S['plastic (W2)'] - S['fixed'], 3).tolist()}   >= +{MARGIN} in {n_seeds_above(S['plastic (W2)'], S['fixed'], MARGIN)}/{nseed}")
    print(f"  probe_adv (food) plastic (W2):             {np.round(PF['plastic (W2)'], 3).tolist()}")

    print("\nrow 5  ceiling -- does exact pair credit pay in this world?")
    print(f"  fixed + B (ceiling) recipe_hit per seed:    {np.round(H['fixed + B (ceiling)'], 3).tolist()}   (v2: 0.23-0.36)")
    line("fixed + B (ceiling)", "fixed")

    print("\nrow 6  interference -- does the food reversal crowd the recipe out of one learner?")
    line("plastic (W2), no flip", "fixed, no flip")
    line("plastic (W2), no flip", "plastic (W2)")
    print(f"  difference of differences  (no-flip advantage) - (flip advantage):")
    dd = (H['plastic (W2), no flip'] - H['fixed, no flip']) - (H['plastic (W2)'] - H['fixed'])
    print(f"    {np.round(dd, 3).tolist()}   >= +{MARGIN} in {int(np.sum(dd >= MARGIN))}/{nseed}")

    print("\nrow 7  does W1 plasticity build the conjunction?")
    line("plastic (both)", "plastic (W2)")
    print(f"  eta1  plastic (both):                      {np.round(np.array(per_seed(results, 'plastic (both)', lambda L: half(L, 'eta1'))), 3).tolist()}")
    print(f"  eta1  fixed (dead-gene drift, this world): {np.round(np.array(per_seed(results, 'fixed', lambda L: half(L, 'eta1'))), 3).tolist()}")
    print(f"  eta2  plastic (W2):                        {np.round(np.array(per_seed(results, 'plastic (W2)', lambda L: half(L, 'eta2'))), 3).tolist()}")
    print(f"  eta2  fixed (dead-gene drift, this world): {np.round(np.array(per_seed(results, 'fixed', lambda L: half(L, 'eta2'))), 3).tolist()}")
    print(f"  lam2  plastic (W2):                        {np.round(np.array(per_seed(results, 'plastic (W2)', lambda L: half(L, 'lam2'))), 3).tolist()}")
    print(f"  lam2  fixed (dead-gene drift, this world): {np.round(np.array(per_seed(results, 'fixed', lambda L: half(L, 'lam2'))), 3).tolist()}")

    print("\nrow 9  is any condition uninterpretable?")
    for n in names:
        pop = np.array(per_seed(results, n, lambda L: half(L, "pop")))
        inj = np.array(per_seed(results, n, lambda L: half(L, "injections")))
        att = A[n]
        flag = " <-- CHECK" if (pop.min() < 60 or att.min() < 1.0) else ""
        print(f"  {n:<24} pop {np.round(pop, 0).tolist()}  injections/window {np.round(inj, 1).tolist()}  attempts/1k {np.round(att, 2).tolist()}{flag}")

    print("\nfeasibility of the bridge (station attempt -> nut, and what is left of its trace):")
    for n in names:
        b = np.array(per_seed(results, n, lambda L: half(L, "bridge_steps")))
        w = np.array(per_seed(results, n, lambda L: half(L, "trace_weight")))
        print(f"  {n:<24} bridge_steps {np.round(b, 1).tolist()}  lam2^gap {np.round(w, 4).tolist()}")
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
