"""Two questions the pre-check raised.
(1) Is phase 1 damaged by fix A, or was the pre-check just short?  v3.8's 8000-step phase 1 gave
    plastic 188/228 and fixed 96/51.  Same length here, same arms.
(2) Contact rate per 1k is CONFOUNDED as a null test: an arm that learns to eat necessarily spends
    fewer actions on `interact` than a random walker, so it can fall below the null while
    approaching better.  The unconfounded measure is conditional: P(interact | on an item cell),
    whose null is exactly 1/6.  Both are printed."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import itertools, time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

def p1_job(arg):
    arm, seed = arg
    r = run(Config(seed=seed, **A.VARIANTS[arm]["kw"]), verbose=False,
            phases=[dict(n_steps=8000, chain=False)])
    p = A.phase_half(r, 0)
    return ("P1", f"{arm:<22} seed {seed}  pop {A.half(p,'pop'):5.0f}  inj {A.half(p,'injections'):4.1f}  "
                  f"safe {A.safe(p):.3f}  probe_food {A.half(p,'probe_adv_food'):6.3f}  "
                  f"eat_on_food {A.half(p,'eat_on_food'):.3f}  gen {A.half(p,'max_gen'):3.0f}   "
                  f"(v3.8 8000-step P1: plastic 188/228, fixed 96/51)")

def cond_job(arm):
    spec = A.VARIANTS[arm]
    r = run(Config(seed=0, **spec["kw"]), verbose=False,
            phases=[dict(p, n_steps=3000) for p in spec["phases"]])
    p2 = A.phase_half(r, 1)
    return ("COND", f"{arm:<22} P(interact | on item) {A.half(p2,'int_on_item'):.3f}   "
                    f"P(eat | on food) {A.half(p2,'eat_on_food'):.3f}   "
                    f"pick/1k {A.half(p2,'pickups_per_1k'):5.2f}   att/1k {A.per_1k(p2,'n_attempts_raw'):5.2f}   "
                    f"att/life {A.half(p2,'attempts_per_life'):.3f}   (null for both conditionals = 0.167)")

if __name__ == "__main__":
    jobs = [("p1", a, s) for a, s in itertools.product(["fixed", "plastic (W2)"], [0, 1])]
    out = {"P1": [], "COND": []}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for tag, line in ex.map(p1_job, [(a, s) for _, a, s in jobs]):
            out[tag].append(line)
        for tag, line in ex.map(cond_job, list(A.VARIANTS)):
            out[tag].append(line)
    print("=== (1) phase 1 at full length, 8000 steps ===")
    for l in out["P1"]: print("  " + l)
    print("\n=== (2) conditional contact, 3000-step phases, seed 0 ===")
    for l in out["COND"]: print("  " + l)
