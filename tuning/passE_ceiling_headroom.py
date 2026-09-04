"""Pass E -- ceiling headroom.
JUDGED AGAINST: `fixed` and the hand-wired ceiling only.  Target: ceiling - fixed >= 0.10 in 2/2.

Why the ceiling sits where it does.  B is exact per-pair credit, not inherited, and the veto makes
the agent avoid pairs it has already found bad.  So a life is an elimination search, and the
expected hit rate over A attempts is mean_k 1/(7-k) for k = 1..A.  At A = 4.4 that predicts 0.238
and we measured 0.226 -- the ceiling is exactly what elimination with 4.4 attempts allows, so the
lever is attempts per life, and nut_value is the wrong knob (it changes fitness, not the search).
    A = 5 -> 0.29     A = 6 -> 0.41     A = 7 -> 0.49
Two ways to buy attempts:
  tool_break up   -- each tool cracks fewer nuts, so the chain must be repeated more often
  longer lives    -- a costlier birth; tried at repro 6.0 in an earlier, poorer world and it
                     collapsed the population, worth re-testing now food is at spawn 3.0
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import itertools, time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

STEPS, SEEDS = 8000, [0, 1]
# E1 (break 0.5) hit the headroom target -- ceiling 0.302/0.289 vs fixed 0.175/0.172, gap
# 0.127/0.117 -- but halved nut income (nut share 0.16-0.23) and starved `fixed` down to pop
# 72-77 with injections, which row 0 excludes.  E2 (longer life) kept populations healthy but
# gave only 0.077/0.086.  So: keep the shorter tool life and restore the income it removed.
# nut_value does not touch the elimination search, so it is a safe lever for viability.
CONFIGS = {
    "break .5 nut 1.6": dict(tool_break=0.5, nut_value=1.6),
    "break 0.4":        dict(tool_break=0.4, nut_value=1.3),
}

def job(arg):
    tag, cond, seed = arg
    kw = dict(A.VARIANTS[cond]); kw.update(CONFIGS[tag])
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=seed, **kw), verbose=False)
    L = r["log"]
    return (f"{tag:<12} {cond:<20} seed {seed}  hit {A.hit(L):.3f}  att/life {A.half(L,'attempts_per_life'):5.2f}  "
            f"pop {A.half(L,'pop'):3.0f}  inj {A.half(L,'injections'):.1f}  gen {A.half(L,'max_gen'):3.0f}  "
            f"bridge1 {A.half(L,'bridge_first'):5.1f}  nutshare {A.half(L,'nut_share'):.2f}  "
            f"safe {A.safe(L):.3f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, ["fixed", "fixed + B (ceiling)"], SEEDS))
    with ProcessPoolExecutor(max_workers=4) as ex:
        for line in ex.map(job, jobs):
            print(line, flush=True)
