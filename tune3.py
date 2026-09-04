"""Third tuning pass.  Targets: bridge_first ~ 20-40 (so that lam2 in [0.65, 0.99] DECIDES the
outcome instead of the answer being trace arithmetic), population off the cap, attempts/life >= 5,
ceiling >= 0.23, fixed near chance."""
import itertools, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sim import Config, run
import analysis as A

STEPS = 2500
BASE = dict(items_per_step=8.0, stations_per_type=60)
CONFIGS = {
    "G": dict(**BASE, nuts_uniform=8.0,  nut_value=1.0, spawn_per_patch=2.0),
    "H": dict(**BASE, nuts_uniform=20.0, nut_value=1.0, spawn_per_patch=2.0),
    "I": dict(**BASE, nuts_uniform=20.0, nut_value=1.0, spawn_per_patch=1.0),
    "J": dict(**BASE, nuts_uniform=8.0,  nut_value=2.0, spawn_per_patch=1.0, max_pop=700),
}

def job(arg):
    tag, cond = arg
    kw = dict(A.VARIANTS[cond])
    if cond.startswith("fixed + B"):
        kw.update(pref_gain=8.0, veto_p=0.9)
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=0, **kw, **CONFIGS[tag]), verbose=False)
    L = r["log"]
    return (f"{tag:<3} {cond:<22} hit {A.hit(L):.3f}  att/life {A.half(L,'attempts_per_life'):5.2f}  "
            f"bridge1 {A.half(L,'bridge_first'):5.1f}  bridgeAll {A.half(L,'bridge_steps'):5.1f}  "
            f"lam2^gap {A.half(L,'trace_weight'):.3f}  nutshare {A.half(L,'nut_share'):.2f}  "
            f"pop {A.half(L,'pop'):3.0f}  nutcells {A.half(L,'nut_cells'):4.0f}  safe {A.safe(L):.3f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, ["fixed", "fixed + B (ceiling)"]))
    with ProcessPoolExecutor(max_workers=4) as ex:
        for line in ex.map(job, jobs):
            print(line, flush=True)
