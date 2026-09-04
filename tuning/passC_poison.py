"""Pass C -- how much food discrimination is worth.
JUDGED AGAINST: the positive control only.  Never against a plastic condition's recipe hit.

Passes A and B left eta2 at or below `fixed`'s drift, which is self-reinforcing: plasticity does
not pay -> eta2 stays low -> H stays small -> the learned preference is ~0.4 logits -> plasticity
does not pay.  Breaking that needs eating badly to COST more.  At poison 0.5 / food 0.7 an
indiscriminate eater still nets +0.10 per meal, so the instinct's 'eat whatever you stand on' is
a viable strategy and selectivity is optional.  At poison >= 1.0 it nets <= -0.15 and selectivity
becomes load-bearing.  This is a food-task parameter and touches nothing in the recipe task.
v3.1 used 0.5, but v3.1 handed the agent no approach instinct: nothing forced it to eat."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import itertools, time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

STEPS, SEEDS = 8000, [0, 1]
CONFIGS = {"poison 1.0": dict(poison_value=1.0), "poison 1.4": dict(poison_value=1.4)}

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
