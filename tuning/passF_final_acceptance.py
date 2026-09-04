"""Final acceptance numbers at the shipped world (pass A genes + pass E `break 0.4, nut 1.3`).
`fixed` and the ceiling were measured in pass E on this world; this fills in plastic (W2)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

def job(seed):
    t0 = time.time()
    r = run(Config(n_steps=8000, seed=seed, **A.VARIANTS["plastic (W2)"]), verbose=False)
    L = r["log"]
    return (f"plastic (W2)  seed {seed}  safe {A.safe(L):.4f}  probe_food {A.half(L,'probe_adv_food'):7.3f}  "
            f"eta2 {A.half(L,'eta2'):.3f}  eta1 {A.half(L,'eta1'):.3f}  lam2 {A.half(L,'lam2'):.3f}  "
            f"nav_dir {A.half(L,'nav_dir'):5.2f}  nav_here {A.half(L,'nav_here'):5.2f}  "
            f"| hit {A.hit(L):.3f}  probe_recipe {A.half(L,'probe_adv'):7.3f}  "
            f"att/life {A.half(L,'attempts_per_life'):5.2f}  pop {A.half(L,'pop'):3.0f}  inj {A.half(L,'injections'):.1f}  "
            f"gen {A.half(L,'max_gen'):3.0f}  bridge1 {A.half(L,'bridge_first'):5.1f}  [{time.time()-t0:.0f}s]")

def jobf(arg):
    cond, seed = arg
    t0 = time.time()
    r = run(Config(n_steps=8000, seed=seed, **A.VARIANTS[cond]), verbose=False)
    L = r["log"]
    return (f"{cond:<20} seed {seed}  safe {A.safe(L):.4f}  eta2 {A.half(L,'eta2'):.3f}  "
            f"eta1 {A.half(L,'eta1'):.3f}  lam2 {A.half(L,'lam2'):.3f}  "
            f"nav_dir {A.half(L,'nav_dir'):5.2f}  nav_here {A.half(L,'nav_here'):5.2f}  "
            f"| hit {A.hit(L):.3f}  att/life {A.half(L,'attempts_per_life'):5.2f}  pop {A.half(L,'pop'):3.0f}  "
            f"inj {A.half(L,'injections'):.1f}  gen {A.half(L,'max_gen'):3.0f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as ex:
        for line in ex.map(job, [0, 1]):
            print(line, flush=True)
        for line in ex.map(jobf, [("fixed", 0), ("fixed", 1)]):
            print(line, flush=True)
