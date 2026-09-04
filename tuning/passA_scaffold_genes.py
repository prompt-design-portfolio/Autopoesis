"""Pass A -- heritable scaffold strength (nav_dir, nav_here as genes).
JUDGED AGAINST: the positive control only (food safe_rate plastic - fixed >= 0.03 in 2/2 seeds,
probe_adv (food) >= 0.5).  Never against a plastic condition's recipe hit rate.
Prior state: safe_rate +0.003, probe_adv (food) 0.125, with the instinct hand-set at 4.0/4.0."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import itertools, time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

STEPS, SEEDS = 8000, [0, 1]

def job(arg):
    cond, seed = arg
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=seed, **A.VARIANTS[cond]), verbose=False)
    L = r["log"]
    return (cond, seed,
            f"{cond:<20} seed {seed}  safe {A.safe(L):.3f}  probe_food {A.half(L,'probe_adv_food'):7.3f}  "
            f"eta2 {A.half(L,'eta2'):.3f}  nav_dir {A.half(L,'nav_dir'):5.2f}  nav_here {A.half(L,'nav_here'):5.2f}  "
            f"| hit {A.hit(L):.3f}  att/life {A.half(L,'attempts_per_life'):5.2f}  "
            f"pop {A.half(L,'pop'):3.0f}  inj {A.half(L,'injections'):.1f}  gen {A.half(L,'max_gen'):3.0f}  "
            f"nutshare {A.half(L,'nut_share'):.2f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    jobs = list(itertools.product(["fixed", "plastic (W2)"], SEEDS))
    with ProcessPoolExecutor(max_workers=4) as ex:
        for _, _, line in ex.map(job, jobs):
            print(line, flush=True)
