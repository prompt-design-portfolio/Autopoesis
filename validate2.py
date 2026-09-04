"""The positive control came out flat at 2500 steps (probe_adv food 0.009, safe_rate +0.019).
Two candidate causes, and they need separating BEFORE the run:
  (a) too few generations -- raising repro_threshold pushed generation time 120 -> 250 steps,
      so 2500 steps is only ~10 generations; v3.1 needed ~40 for probe_adv 1.4-2.7.
  (b) nuts now supply ~65% of energy income, so most modulator events are '+1 at a nut',
      which carries no food-discrimination information and swamps the food signal.
(a) is tested by running the chosen world to full length; (b) by a world with food restored.
"""
import time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

FOODIER = dict(A.WORLD); FOODIER.update(spawn_per_patch=3.0, nut_value=0.7)

JOBS = [
    ("(a) final world, 10000", "fixed",                 10000, A.WORLD),
    ("(a) final world, 10000", "plastic (W2)",          10000, A.WORLD),
    ("(a) final world, 10000", "fixed, no flip",        10000, A.WORLD),
    ("(b) foodier, 2500",      "fixed",                  2500, FOODIER),
    ("(b) foodier, 2500",      "plastic (W2)",           2500, FOODIER),
]

def job(j):
    tag, cond, steps, world = j
    kw = dict(A.VARIANTS[cond]); kw.update(world)
    t0 = time.time()
    r = run(Config(n_steps=steps, seed=0, **kw), verbose=False)
    L = r["log"]
    return (f"{tag:<26} {cond:<22} safe {A.safe(L):.3f}  probe_food {A.half(L,'probe_adv_food'):7.3f}  "
            f"eta2 {A.half(L,'eta2'):.3f}  gen {A.half(L,'max_gen'):3.0f}  nutshare {A.half(L,'nut_share'):.2f}  "
            f"| hit {A.hit(L):.3f}  att/life {A.half(L,'attempts_per_life'):5.2f}  "
            f"pop {A.half(L,'pop'):3.0f}  inj {A.half(L,'injections'):.1f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=5) as ex:
        for line in ex.map(job, JOBS):
            print(line, flush=True)
