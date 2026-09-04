"""v3.9 pre-checks (build-plan rule 4): 1 seed, 3000-step phases, EVERY arm, before any grid.
Reports the transition table, per-arm population and attempts/life.  Every defect so far was
visible at this size."""
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
    phases = [dict(p, n_steps=PHASE) for p in spec["phases"]]
    t0 = time.time()
    r = run(Config(seed=0, **spec["kw"]), verbose=False, phases=phases)
    return name, r, time.time() - t0

if __name__ == "__main__":
    names = list(A.VARIANTS)
    results = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for name, r, dt in ex.map(job, names):
            results[name] = [r]
            print(f"  {name:<24} {dt:5.0f}s", flush=True)
    print(f"\n{'='*100}\nPER-ARM SUMMARY  (1 seed, {PHASE}-step phases -- a pre-check, NOT a result)\n{'='*100}")
    hdr = f"{'arm':<24}{'P1 pop':>8}{'P1 safe':>9}{'P1 probe':>10}{'P1 eat/food':>12}" \
          f"{'P2 pop':>8}{'P2 inj':>8}{'P2 safe':>9}{'att/life':>10}{'att/1k':>8}{'pick/1k':>9}{'crack/1k':>9}{'decl_frac':>10}"
    print(hdr)
    for n in names:
        r = results[n][0]
        p1, p2 = A.phase_half(r, 0), A.phase_half(r, 1)
        print(f"{n:<24}{A.half(p1,'pop'):8.0f}{A.safe(p1):9.3f}{A.half(p1,'probe_adv_food'):10.3f}"
              f"{A.half(p1,'eat_on_food'):12.3f}{A.half(p2,'pop'):8.0f}{A.half(p2,'injections'):8.1f}"
              f"{A.safe(p2):9.3f}{A.half(p2,'attempts_per_life'):10.3f}"
              f"{A.per_1k(p2,'n_attempts_raw'):8.2f}{A.half(p2,'pickups_per_1k'):9.2f}"
              f"{A.per_1k(p2,'n_nuts'):9.2f}{A.half(p2,'decline_frac'):10.3f}")
    print("\nCHECK A (split action): in phase 1, `eat` on food cells must exceed random policy's 1/6 = 0.167")
    rnd = A.half(A.phase_half(results[A.NULL][0], 0), "eat_on_food")
    for n in names:
        e = A.half(A.phase_half(results[n][0], 0), "eat_on_food")
        print(f"  {n:<24} eat_on_food {e:.3f}   vs null {rnd:.3f}   {'OK' if e > rnd + 0.02 else 'below'}")
    A.transition_table(results, bin_size=500, span=1500)
