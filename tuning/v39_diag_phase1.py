"""Diagnostic, not tuning: phase-1 populations fell from v3.8's 96-228 to 41-97.  Phase 1 has no
chain, so densities are irrelevant there; the only phase-1 changes are fix A's two effects --
six actions instead of five (so `eat` is 1/6 of the action space, not 1/5, and `interact` is a
GUARANTEED no-op in phase 1) and the no-op charge.  This separates them."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import itertools, time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

def job(arg):
    arm, noop = arg
    kw = dict(A.VARIANTS[arm]["kw"]); kw["noop_cost"] = noop
    r = run(Config(seed=0, **kw), verbose=False, phases=[dict(n_steps=3000, chain=False)])
    p1 = A.phase_half(r, 0)
    return (f"{arm:<22} noop_cost {noop:<6} pop {A.half(p1,'pop'):5.0f}  inj {A.half(p1,'injections'):4.1f}  "
            f"safe {A.safe(p1):.3f}  probe_food {A.half(p1,'probe_adv_food'):6.3f}  "
            f"eat_on_food {A.half(p1,'eat_on_food'):.3f}  noops/1k {A.half(p1,'noops_per_1k'):6.1f}  "
            f"gen {A.half(p1,'max_gen'):3.0f}")

if __name__ == "__main__":
    jobs = list(itertools.product(["fixed", "plastic (W2)", "random policy"], [0.002, 0.0]))
    with ProcessPoolExecutor(max_workers=4) as ex:
        for line in ex.map(job, jobs):
            print(line, flush=True)
