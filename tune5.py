"""Fifth and last tuning pass.  Population off the cap and attempts-per-life are in tension
through crowding (a capped population lives longer per head).  The lever that buys both is a
costlier birth: longer lives (more attempts each) and a smaller standing population.
Cost: fewer generations per run, so n_steps may have to rise."""
import itertools, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sim import Config, run
import analysis as A

STEPS = 2500
L = dict(items_per_step=8.0, stations_per_type=60, nut_value=1.0, max_pop=600,
         nuts_uniform=8.0, spawn_per_patch=1.0)
CONFIGS = {
    "L":  dict(**L),
    "N":  dict(**L, repro_threshold=4.5, repro_cost=2.25, max_energy=8.0),
    "O":  dict(**L, repro_threshold=6.0, repro_cost=3.0,  max_energy=10.0),
}

def job(arg):
    tag, cond = arg
    kw = dict(A.VARIANTS[cond])
    if cond.startswith("fixed + B"):
        kw.update(pref_gain=8.0, veto_p=0.9)
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=0, **kw, **CONFIGS[tag]), verbose=False)
    Lg = r["log"]
    return (f"{tag:<3} {cond:<22} hit {A.hit(Lg):.3f}  att/life {A.half(Lg,'attempts_per_life'):5.2f}  "
            f"bridge1 {A.half(Lg,'bridge_first'):5.1f}  lam2^gap {A.half(Lg,'trace_weight'):.3f}  "
            f"nutshare {A.half(Lg,'nut_share'):.2f}  pop {A.half(Lg,'pop'):3.0f}/600  "
            f"gen {A.half(Lg,'max_gen'):.0f}  inj {A.half(Lg,'injections'):.1f}  safe {A.safe(Lg):.3f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    jobs = [j for j in itertools.product(CONFIGS, ["fixed", "fixed + B (ceiling)"]) if j[0] != "L" or True]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for line in ex.map(job, jobs):
            print(line, flush=True)
