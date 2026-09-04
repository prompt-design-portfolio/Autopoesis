"""Viability smoke test.  NOT the experiment: short runs, one seed, to check that the
world works at all before spending the real one.  Checks, in order:
  1. does anything happen -- attempts, tools, nuts, and a population that survives?
  2. how far is the bridge (station attempt -> nut) and what is left of the trace there?
  3. does the hand-wired ceiling (exact pair credit) beat chance?  If it does not, the
     world does not pay for the recipe and NO condition can be read.
  4. is the recipe genetically trackable at recipe_every=2000 (the gate)?
  5. how much of the energy income comes from nuts vs food?
"""
import sys, time
import numpy as np
from sim import Config, run
import analysis as A

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
conds = ["fixed", "fixed + B (ceiling)", "plastic (W2)", "scrambled"]

for name in conds:
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=0, **A.VARIANTS[name]), verbose=False)
    L = r["log"]
    print(f"\n=== {name}   ({time.time()-t0:.0f}s for {STEPS} steps)")
    print(f"  recipe_hit (2nd half, event-weighted) {A.hit(L):.3f}   chance {A.CHANCE:.3f}")
    print(f"  attempts/1k {A.per_1k(L,'n_attempts_raw'):.2f}   nuts/1k {A.per_1k(L,'n_nuts'):.2f}"
          f"   pickups/1k {A.per_1k(L,'pickups_per_1k') if False else np.nan}")
    print(f"  safe_rate {A.safe(L):.3f}   pop {A.half(L,'pop'):.0f}   max_gen {A.half(L,'max_gen'):.1f}"
          f"   injections/window {A.half(L,'injections'):.1f}")
    print(f"  has_tool {A.half(L,'has_tool'):.3f}  holding_item {A.half(L,'holding_item'):.3f}")
    print(f"  bridge_steps {A.half(L,'bridge_steps'):.1f}   lam2^gap {A.half(L,'trace_weight'):.4f}"
          f"   lam2 {A.half(L,'lam2'):.3f}")
    print(f"  nut share of energy income {A.half(L,'nut_share'):.3f}")
    print(f"  pair_gain innate {A.half(L,'pair_gain_innate'):.3f}  learned {A.half(L,'pair_gain'):.3f}"
          f"  probe_adv {A.half(L,'probe_adv'):.3f}  probe_adv_food {A.half(L,'probe_adv_food'):.3f}")
    print(f"  nut cells {A.half(L,'nut_cells'):.0f}  item cells {A.half(L,'item_cells'):.0f}")
    print(f"  attempt-in-life curve {np.round(A.curve(L,'att')[0],3).tolist()}")
    print(f"  hit trajectory {np.round(A.smooth([x['recipe_hit'] for x in L])[::10],3).tolist()}")
