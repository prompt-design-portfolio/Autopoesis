"""Pre-run validation of all seven conditions on the chosen world.  Checks viability and the
POSITIVE CONTROL (does the learner work at all in this world: probe_adv on food, safe_rate
above fixed).  recipe_hit is printed too and is disclosed -- the world was chosen on `fixed`
and the ceiling before this ran, not on any plastic condition."""
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sim import Config, run
import analysis as A

STEPS = 2500

def job(name):
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=0, **A.VARIANTS[name]), verbose=False)
    L = r["log"]
    return (name, f"{name:<24} hit {A.hit(L):.3f}  att/life {A.half(L,'attempts_per_life'):5.2f}  "
            f"safe {A.safe(L):.3f}  probe_food {A.half(L,'probe_adv_food'):6.3f}  "
            f"probe_recipe {A.half(L,'probe_adv'):6.3f}  pair_innate {A.half(L,'pair_gain_innate'):6.3f}  "
            f"bridge1 {A.half(L,'bridge_first'):5.1f}  lam2 {A.half(L,'lam2'):.3f}  "
            f"pop {A.half(L,'pop'):3.0f}/600  gen {A.half(L,'max_gen'):.0f}  inj {A.half(L,'injections'):.1f}  "
            f"[{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as ex:
        for _, line in ex.map(job, list(A.VARIANTS)):
            print(line, flush=True)
