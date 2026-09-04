"""Fourth tuning pass: get the population OFF the cap (at a hard cap, births are a queue
rather than differential fecundity, which blunts selection on the recipe and biases toward a
null) while keeping attempts/life >= 5, the ceiling >= 0.23, and fixed at chance."""
import itertools, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sim import Config, run
import analysis as A

STEPS = 2500
BASE = dict(items_per_step=8.0, stations_per_type=60, nut_value=1.0, max_pop=600)
CONFIGS = {
    "K": dict(**BASE, nuts_uniform=8.0, spawn_per_patch=1.5),
    "L": dict(**BASE, nuts_uniform=8.0, spawn_per_patch=1.0),
    "M": dict(**BASE, nuts_uniform=6.0, spawn_per_patch=0.75),
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
            f"bridge1 {A.half(L,'bridge_first'):5.1f}  lam2^gap {A.half(L,'trace_weight'):.3f}  "
            f"nutshare {A.half(L,'nut_share'):.2f}  pop {A.half(L,'pop'):3.0f}/{BASE['max_pop']}  "
            f"inj {A.half(L,'injections'):.1f}  safe {A.safe(L):.3f}  gen {A.half(L,'max_gen'):.0f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, ["fixed", "fixed + B (ceiling)"]))
    with ProcessPoolExecutor(max_workers=4) as ex:
        for line in ex.map(job, jobs):
            print(line, flush=True)
