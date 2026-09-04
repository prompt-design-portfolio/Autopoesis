"""The directional inputs are normalised by 10, so one gain cannot serve both approach and
interact: at scale 1.0 the movement wiring drowns in the random weights (populations at the
floor) while the interact wiring still saturates.  Split the two gains and re-test."""
import itertools, time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

STEPS = 5000
CONFIGS = {
    "S dir20 here2 sc1.0": dict(nav_dir=20.0, nav_here=2.0, innate_scale=1.0),
    "T dir20 here4 sc1.0": dict(nav_dir=20.0, nav_here=4.0, innate_scale=1.0),
    "U dir40 here2 sc0.6": dict(nav_dir=40.0, nav_here=2.0, innate_scale=0.6),
}

def job(arg):
    tag, cond = arg
    kw = dict(A.VARIANTS[cond]); kw.update(CONFIGS[tag])
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=0, **kw), verbose=False)
    L = r["log"]
    return (f"{tag:<22} {cond:<14} safe {A.safe(L):.3f}  probe_food {A.half(L,'probe_adv_food'):7.3f}  "
            f"eta2 {A.half(L,'eta2'):.3f}  gen {A.half(L,'max_gen'):3.0f}  "
            f"| hit {A.hit(L):.3f}  att/life {A.half(L,'attempts_per_life'):5.2f}  "
            f"bridge1 {A.half(L,'bridge_first'):5.1f}  pop {A.half(L,'pop'):3.0f}  "
            f"inj {A.half(L,'injections'):.1f}  nutshare {A.half(L,'nut_share'):.2f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, ["fixed", "plastic (W2)"]))
    with ProcessPoolExecutor(max_workers=6) as ex:
        for line in ex.map(job, jobs):
            print(line, flush=True)
