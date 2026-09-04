"""World tuning, before the run.  Tuned ONLY on `fixed` and on the hand-wired ceiling
`fixed + B` -- never on the result condition -- against two criteria that have to hold
before the question is askable at all:
  (a) an agent must get enough tool attempts in a life for within-life learning to be
      possible in principle (the first smoke gave ~1 per life);
  (b) exact per-pair credit (the hand-wired ceiling) must beat chance, as it did in v2
      (0.23-0.36).  If it does not, no condition can be read.
The gate -- that `fixed` alone stays near chance -- is re-checked at each setting.
"""
import sys, time
import numpy as np
from sim import Config, run
import analysis as A

STEPS = 3000
GRIDS = [
    dict(name="A  items 8, nut 2.0",  items_per_step=8.0,  nut_value=2.0),
    dict(name="B  items 20, nut 2.0", items_per_step=20.0, nut_value=2.0),
]
for gspec in GRIDS:
    g = {k: v for k, v in gspec.items() if k != "name"}
    for cond in ["fixed", "fixed + B (ceiling)"]:
        kw = dict(A.VARIANTS[cond])
        if cond.startswith("fixed + B"):
            kw.update(pref_gain=8.0, veto_p=0.9)   # v2's STAKES values, the ones that gave 0.23-0.36
        t0 = time.time()
        r = run(Config(n_steps=STEPS, seed=0, **kw, **g), verbose=False)
        L = r["log"]
        print(f"{gspec['name']:<22} {cond:<22} hit {A.hit(L):.3f}  att/1k {A.per_1k(L,'n_attempts_raw'):5.1f}  "
              f"att/life {A.half(L,'attempts_per_life'):5.2f}  nuts/1k {A.per_1k(L,'n_nuts'):5.2f}  "
              f"nutshare {A.half(L,'nut_share'):.2f}  pop {A.half(L,'pop'):3.0f}  "
              f"items {A.half(L,'item_cells'):4.0f}  bridge {A.half(L,'bridge_steps'):4.1f}  "
              f"safe {A.safe(L):.3f}  [{time.time()-t0:.0f}s]", flush=True)
