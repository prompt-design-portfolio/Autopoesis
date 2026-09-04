"""The positive control (the v3.1 food effect) is dead in this world and eta2 is selected OFF.
Diagnosis: v2.9b scaled the random innate weights by 0.1 so that its hand-wired forager instinct
would dominate.  That was right there -- v2.9b's discrimination came from the hand-wired B path,
not from the network.  Here it leaves H2 nothing to read: the free hidden units carry ~0.1x
random projections while the nav units saturate at 4.0, so the output layer can only turn
'interact' up or down in general, never tell one food type (or one pair) from another.
Fix candidates, judged ONLY on the positive control and on viability."""
import itertools, time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

STEPS = 5000
CONFIGS = {
    "P  scale 1.0 gain 4": dict(innate_scale=1.0, nav_gain=4.0),
    "Q  scale 1.0 gain 2": dict(innate_scale=1.0, nav_gain=2.0),
    "R  scale 0.4 gain 4": dict(innate_scale=0.4, nav_gain=4.0),
}

def job(arg):
    tag, cond = arg
    kw = dict(A.VARIANTS[cond]); kw.update(CONFIGS[tag])
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=0, **kw), verbose=False)
    L = r["log"]
    return (f"{tag:<22} {cond:<14} safe {A.safe(L):.3f}  probe_food {A.half(L,'probe_adv_food'):7.3f}  "
            f"eta2 {A.half(L,'eta2'):.3f}  gen {A.half(L,'max_gen'):3.0f}  "
            f"| hit {A.hit(L):.3f}  pair_innate {A.half(L,'pair_gain_innate'):6.3f}  "
            f"att/life {A.half(L,'attempts_per_life'):5.2f}  bridge1 {A.half(L,'bridge_first'):5.1f}  "
            f"pop {A.half(L,'pop'):3.0f}  inj {A.half(L,'injections'):.1f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, ["fixed", "plastic (W2)"]))
    with ProcessPoolExecutor(max_workers=6) as ex:
        for line in ex.map(job, jobs):
            print(line, flush=True)
