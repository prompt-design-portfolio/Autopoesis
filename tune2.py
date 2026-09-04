"""Second tuning pass, in parallel.  Still only `fixed` and the hand-wired ceiling.
Targets before the question is askable:
    attempts per life  >= 5        (an agent must meet the conjunction more than once)
    bridge_steps       15..40      (hard but spannable: lam2 in [0.65, 0.99] must MATTER)
    ceiling - fixed    >= 0.05     (exact pair credit pays, as it did in v2)
    pop                well below max_pop (at the cap, births are a lottery, not selection)
    fixed              near chance (the gate)
"""
import itertools, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sim import Config, run
import analysis as A

STEPS = 2500
CONFIGS = {
    "C": dict(items_per_step=8.0, nut_value=2.0, stations_per_type=60, nuts_uniform=0.5, spawn_per_patch=2.0),
    "D": dict(items_per_step=8.0, nut_value=2.0, stations_per_type=60, nuts_uniform=2.0, spawn_per_patch=2.0),
    "E": dict(items_per_step=8.0, nut_value=1.5, stations_per_type=60, nuts_uniform=2.0, spawn_per_patch=1.5),
    "F": dict(items_per_step=8.0, nut_value=1.5, stations_per_type=120, nuts_uniform=2.0, spawn_per_patch=1.5),
}

def job(arg):
    tag, cond = arg
    kw = dict(A.VARIANTS[cond])
    if cond.startswith("fixed + B"):
        kw.update(pref_gain=8.0, veto_p=0.9)
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=0, **kw, **CONFIGS[tag]), verbose=False)
    L = r["log"]
    return (f"{tag:<3} {cond:<22} hit {A.hit(L):.3f}  att/1k {A.per_1k(L,'n_attempts_raw'):5.1f}  "
            f"att/life {A.half(L,'attempts_per_life'):5.2f}  bridge {A.half(L,'bridge_steps'):5.1f}  "
            f"lam2^gap {A.half(L,'trace_weight'):.3f}  nuts/1k {A.per_1k(L,'n_nuts'):5.2f}  "
            f"nutshare {A.half(L,'nut_share'):.2f}  pop {A.half(L,'pop'):3.0f}  safe {A.safe(L):.3f}  "
            f"[{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, ["fixed", "fixed + B (ceiling)"]))
    with ProcessPoolExecutor(max_workers=4) as ex:
        for line in ex.map(job, jobs):
            print(line, flush=True)
