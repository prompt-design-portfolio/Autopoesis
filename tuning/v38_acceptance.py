"""v3.8 acceptance checks, 2 seeds, `plastic (W2) staged` and `fixed (staged)`.
  1. Phase-1 second half reproduces v3.1: safe_rate plastic - fixed >= 0.03,
     probe_adv (food) >= 1.0, eta2 above fixed's.
  2. Phase 2 survives: plastic population >= 80 with no injections, in 2/2.
Nothing here is judged against a plastic condition's recipe hit, and the phase-2 world is not
tuned to make check 2 pass."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import itertools, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sim import Config, run
import analysis as A

SEEDS = [0, 1]

def job(arg):
    name, seed = arg
    spec = A.VARIANTS[name]
    t0 = time.time()
    r = run(Config(seed=seed, **spec["kw"]), verbose=False, phases=spec["phases"])
    p1, p2 = A.phase_half(r, 0), A.phase_half(r, 1)
    return (name, seed,
            f"{name:<22} seed {seed}  "
            f"P1 safe {A.safe(p1):.3f}  probe_food {A.half(p1,'probe_adv_food'):6.3f}  "
            f"eta2 {A.half(p1,'eta2'):.3f}  lam2 {A.half(p1,'lam2'):.3f}  pop {A.half(p1,'pop'):3.0f}  "
            f"gen {A.half(p1,'max_gen'):3.0f}  inj {A.half(p1,'injections'):.1f}  |  "
            f"P2 pop {A.half(p2,'pop'):3.0f}  inj {A.half(p2,'injections'):.1f}  "
            f"att/1k {A.per_1k(p2,'n_attempts_raw'):5.2f}  tool {A.half(p2,'has_tool'):.3f}  "
            f"hit {A.hit(p2):.3f}  eta2 {A.half(p2,'eta2'):.3f}  lam2 {A.half(p2,'lam2'):.3f}  "
            f"safe {A.safe(p2):.3f}  [{time.time()-t0:.0f}s]",
            r)

if __name__ == "__main__":
    jobs = list(itertools.product(["plastic (W2) staged", "fixed (staged)"], SEEDS))
    out = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for name, seed, line, r in ex.map(job, jobs):
            print(line, flush=True)
            out[(name, seed)] = r
    print("\n--- check 1: phase-1 gate (v3.1 reproduction) ---")
    for s in SEEDS:
        p = A.phase_half(out[("plastic (W2) staged", s)], 0)
        f = A.phase_half(out[("fixed (staged)", s)], 0)
        print(f"  seed {s}: safe plastic-fixed {A.safe(p)-A.safe(f):+.3f} (>=0.03)   "
              f"probe_food {A.half(p,'probe_adv_food'):.3f} (>=1.0)   "
              f"eta2 {A.half(p,'eta2'):.3f} vs fixed {A.half(f,'eta2'):.3f}")
    print("\n--- check 2: phase 2 survives ---")
    for s in SEEDS:
        p = A.phase_half(out[("plastic (W2) staged", s)], 1)
        print(f"  seed {s}: pop {A.half(p,'pop'):.0f} (>=80)   injections {A.half(p,'injections'):.2f} (=0)")
