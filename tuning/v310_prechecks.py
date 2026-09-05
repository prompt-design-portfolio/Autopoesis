"""v3.10 pre-checks: 1 seed, 3000-step phases, every arm."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib; matplotlib.use("Agg")
from sim import Config, run
import analysis as A

PHASE = 3000

def job(name):
    spec = A.VARIANTS[name]
    t0 = time.time()
    r = run(Config(seed=0, **spec["kw"]), verbose=False,
            phases=[dict(p, n_steps=PHASE) for p in spec["phases"]])
    return name, r, time.time() - t0

if __name__ == "__main__":
    names = list(A.VARIANTS)
    results = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for name, r, dt in ex.map(job, names):
            results[name] = [r]
            print(f"  {name:<24} {dt:5.0f}s", flush=True)

    print(f"\n{'='*120}\nPER-ARM SUMMARY  (1 seed, {PHASE}-step phases -- a pre-check, NOT a result)")
    print(f"levels: chance {A.CHANCE:.3f} | type-blind {A.TYPE_BLIND:.2f} | full {A.FULL:.2f}\n{'='*120}")
    print(f"{'arm':<22}{'P1 pop':>8}{'P1 safe':>9}{'P1 probe':>10}{'P2 pop':>8}{'P2 inj':>8}"
          f"{'prep hit':>10}{'hit|A':>8}{'hit|B':>8}{'prep/life':>11}{'P(prep)':>9}{'share':>8}{'probe':>8}")
    for n in names:
        p1, p2 = A.phase_half(results[n][0], 0), A.phase_half(results[n][0], 1)
        print(f"{n:<22}{A.half(p1,'pop'):8.0f}{A.safe(p1):9.3f}{A.half(p1,'probe_adv_food'):10.3f}"
              f"{A.half(p2,'pop'):8.0f}{A.half(p2,'injections'):8.1f}{A.prep_hit(p2):10.3f}"
              f"{A.rate(p2,'n_ok0','n_prep0'):8.3f}{A.rate(p2,'n_ok1','n_prep1'):8.3f}"
              f"{A.half(p2,'prep_per_life'):11.2f}{A.half(p2,'prep_on_food'):9.3f}"
              f"{A.half(p2,'prep_share'):8.3f}{A.half(p2,'probe_adv'):8.3f}")

    print("\nP(prep | on food) vs the per-phase null (analytic 3/8 = 0.375 in phase 2)")
    base = A.half(A.phase_half(results[A.NULL][0], 1), "prep_on_food")
    for n in names:
        print(f"  {n:<22} {A.half(A.phase_half(results[n][0],1),'prep_on_food'):.3f}"
              f"   {'above null' if A.half(A.phase_half(results[n][0],1),'prep_on_food') >= base+0.03 else ''}")

    print("\nPREP SHARE OF MEALS BY ERA (diagnostic: a learner with the mapping shifts eat -> prep)")
    for n in names:
        sh = [round(float(A.half(w, "prep_share")), 3) for w in A.era_windows(results[n][0])]
        hh = [round(float(A.prep_hit(w)), 3) for w in A.era_windows(results[n][0])]
        print(f"  {n:<22} share {sh}   hit {hh}")

    print("\nRIG CHECK 2(a): probe_adv (food) in the FIRST mapping era only")
    for n in names:
        fe = A.first_era(results[n][0])
        print(f"  {n:<22} probe_adv (food) {A.half(fe,'probe_adv_food'):6.3f}  safe {A.safe(fe):.3f}"
              f"  raw meals {float(np.sum([r['n_raw'] for r in fe])):.0f}")
    A.transition_table(results, bin_size=500, span=1500)
