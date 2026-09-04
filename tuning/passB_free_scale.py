"""Pass B -- scale of the FREE hidden units (the basis the output layer reads).
JUDGED AGAINST: the positive control only (food safe_rate plastic - fixed >= 0.03 in 2/2,
probe_adv (food) >= 0.5).  Never against a plastic condition's recipe hit rate.
Pass A left probe_adv (food) at 0.39/0.23 and safe_rate at +0.005/+0.001: the learned preference
is ~0.4 logits against a ~5-logit interact push.  The nav units are saturated and TYPE-BLIND, so
all discriminative signal lives in the free units, whose response to a type channel is
tanh(0.3) ~ 0.29.  This scales that basis without touching the instinct."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import itertools, time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

STEPS, SEEDS = 8000, [0, 1]
CONFIGS = {"free 2.0": dict(free_scale=2.0), "free 3.0": dict(free_scale=3.0)}

def job(arg):
    tag, cond, seed = arg
    kw = dict(A.VARIANTS[cond]); kw.update(CONFIGS[tag])
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=seed, **kw), verbose=False)
    L = r["log"]
    return (f"{tag:<10} {cond:<14} seed {seed}  safe {A.safe(L):.3f}  probe_food {A.half(L,'probe_adv_food'):7.3f}  "
            f"eta2 {A.half(L,'eta2'):.3f}  nav_here {A.half(L,'nav_here'):5.2f}  nav_dir {A.half(L,'nav_dir'):5.2f}  "
            f"| hit {A.hit(L):.3f}  att/life {A.half(L,'attempts_per_life'):5.2f}  pop {A.half(L,'pop'):3.0f}  "
            f"inj {A.half(L,'injections'):.1f}  gen {A.half(L,'max_gen'):3.0f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, ["fixed", "plastic (W2)"], SEEDS))
    with ProcessPoolExecutor(max_workers=4) as ex:
        for line in ex.map(job, jobs):
            print(line, flush=True)
