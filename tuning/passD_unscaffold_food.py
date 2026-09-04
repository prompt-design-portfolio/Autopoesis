"""Pass D -- take FOOD out of the instinct; leave the chain's instinct intact.
JUDGED AGAINST: the positive control only.  Never against a plastic condition's recipe hit.

Passes A-C all failed on the same thing: the instinct approaches food of either type and pushes
'interact' with ~5 logits, so a learned preference of ~0.4 logits cannot move behaviour, and
plasticity never starts paying.  But food approach was not in v2.9b -- its appetite channel was
weighted by a hand-wired taste register, and I made it type-blind when that register was dropped.
The chain (item -> station -> nut) must be scaffolded or no attempts happen and nothing is
readable.  Food need not be, and it is the positive control.  Unscaffolded, the food task is
exactly v3.1's, where probe_adv reached 1.4-2.7 and safe_rate +0.08."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import itertools, time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

STEPS, SEEDS = 8000, [0, 1]
CONFIGS = {"food unscaffolded": dict(scaffold_food=False)}

def job(arg):
    tag, cond, seed = arg
    kw = dict(A.VARIANTS[cond]); kw.update(CONFIGS[tag])
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=seed, **kw), verbose=False)
    L = r["log"]
    return (f"{tag:<12} {cond:<14} seed {seed}  safe {A.safe(L):.3f}  probe_food {A.half(L,'probe_adv_food'):7.3f}  "
            f"eta2 {A.half(L,'eta2'):.3f}  nav_here {A.half(L,'nav_here'):5.2f}  "
            f"| hit {A.hit(L):.3f}  att/life {A.half(L,'attempts_per_life'):5.2f}  pop {A.half(L,'pop'):3.0f}  "
            f"inj {A.half(L,'injections'):.1f}  gen {A.half(L,'max_gen'):3.0f}  nutshare {A.half(L,'nut_share'):.2f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, ["fixed", "plastic (W2)"], SEEDS))
    with ProcessPoolExecutor(max_workers=4) as ex:
        for line in ex.map(job, jobs):
            print(line, flush=True)
