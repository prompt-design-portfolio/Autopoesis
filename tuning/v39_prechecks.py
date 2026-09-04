"""v3.9 amendment-3 pre-checks: 1 seed, 3000-step phases, every arm."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib; matplotlib.use("Agg")
from sim import Config, run, assert_cover
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

    print(f"\n{'='*118}\nPER-ARM SUMMARY  (1 seed, {PHASE}-step phases -- a pre-check, NOT a result)\n{'='*118}")
    print(f"{'arm':<22}{'P1 pop':>8}{'P1 safe':>9}{'P1 probe':>10}{'P2 pop':>8}{'P2 inj':>8}"
          f"{'P2 safe':>9}{'P2 hit':>8}{'att/life':>10}{'att/1k':>8}{'pick/1k':>9}{'decl_frac':>10}")
    for n in names:
        p1, p2 = A.phase_half(results[n][0], 0), A.phase_half(results[n][0], 1)
        print(f"{n:<22}{A.half(p1,'pop'):8.0f}{A.safe(p1):9.3f}{A.half(p1,'probe_adv_food'):10.3f}"
              f"{A.half(p2,'pop'):8.0f}{A.half(p2,'injections'):8.1f}{A.safe(p2):9.3f}"
              f"{A.hit(p2):8.3f}{A.half(p2,'attempts_per_life'):10.2f}"
              f"{A.per_1k(p2,'n_attempts_raw'):8.2f}{A.half(p2,'pickups_per_1k'):9.2f}"
              f"{A.half(p2,'decline_frac'):10.3f}")

    print("\nCONDITIONAL RATES.  Null = `random policy`; analytic share 1/5 = 0.200 in phase 1")
    print("(interact masked) and 1/6 = 0.167 in phase 2.")
    for lab, key, ph, an in [("P(eat | on food)", "eat_on_food", 0, 1/5),
                             ("P(eat | on food)", "eat_on_food", 1, 1/6),
                             ("P(interact | on item, empty-handed)", "int_on_item", 1, 1/6),
                             ("P(interact | at station, carrying)", "int_at_station", 1, 1/6)]:
        base = A.half(A.phase_half(results[A.NULL][0], ph), key)
        print(f"  phase {ph+1}  {lab}   null measured {base:.3f}, analytic {an:.3f}")
        for n in names:
            x = A.half(A.phase_half(results[n][0], ph), key)
            print(f"    {n:<22} {x:.3f}   {'above null' if x >= base + 0.03 else ''}")

    print("\nREADABILITY (criterion now on `fixed`; the null is the affordance check)")
    f2 = A.phase_half(results["fixed"][0], 1)
    n2 = A.phase_half(results[A.NULL][0], 1)
    print(f"  fixed          attempts/life {A.half(f2,'attempts_per_life'):5.2f}  (>= 3)   "
          f"P(int|at station) {A.half(f2,'int_at_station'):.3f} vs null {A.half(n2,'int_at_station'):.3f}")
    print(f"  random policy  attempts/life {A.half(n2,'attempts_per_life'):5.2f}  (>= 1)")

    print("\nWORLD CRITERION")
    assert_cover(Config(chain=True, **{k: v for k, v in A.WORLD.items()
                                       if k in Config.__dataclass_fields__}))
    A.transition_table(results, bin_size=500, span=1500)
