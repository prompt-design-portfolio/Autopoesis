"""Scale only the scaffold's own hidden units; leave the rest at full random scale as the basis
H2 reads.  Judged on the positive control (the v3.1 food effect must be alive) and viability."""
import itertools, time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

STEPS = 5000
CONFIGS = {
    "V h24 world N":     dict(hidden=24),
    "W h32 world N":     dict(hidden=32),
    "X h24 richer food": dict(hidden=24, spawn_per_patch=2.0),
}

def job(arg):
    tag, cond = arg
    kw = dict(A.VARIANTS[cond]); kw.update(CONFIGS[tag])
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=0, **kw), verbose=False)
    L = r["log"]
    return (f"{tag:<20} {cond:<14} safe {A.safe(L):.3f}  probe_food {A.half(L,'probe_adv_food'):7.3f}  "
            f"eta2 {A.half(L,'eta2'):.3f}  gen {A.half(L,'max_gen'):3.0f}  "
            f"| hit {A.hit(L):.3f}  att/life {A.half(L,'attempts_per_life'):5.2f}  "
            f"bridge1 {A.half(L,'bridge_first'):5.1f}  pop {A.half(L,'pop'):3.0f}  "
            f"inj {A.half(L,'injections'):.1f}  nutshare {A.half(L,'nut_share'):.2f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, ["fixed", "plastic (W2)"]))
    with ProcessPoolExecutor(max_workers=6) as ex:
        for line in ex.map(job, jobs):
            print(line, flush=True)
