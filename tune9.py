"""eta2 is selected DOWN in every plastic run (0.02-0.05 vs fixed's drift 0.10-0.14) -- the
opposite of v3.1, where selection RETAINED informative plasticity.  Candidate mechanism: with
fail_cost = 0 the recipe task supplies a POSITIVE-ONLY modulator (m = +1 at a nut, nothing at a
failed attempt), it is frequent (nuts are ~63% of energy income), and its credit lands on
whatever the agent was doing -- mostly navigation.  A frequent sign-positive modulator is
uninformative, so plasticity is a net cost and selection removes it.
Two levers, tested here so the choice can be made on numbers:
  Y: dilute it -- more food, so the balanced +/-1 food modulator is a larger share of events.
  Z: balance it -- fail_cost > 0, so a wrong attempt costs energy AT THE STATION and m = -1
     arrives immediately.  This makes avoidance learnable at once and leaves only success
     delayed; it is v2's STAKES setting, and it answers an easier question than 'purely
     delayed credit'.
"""
import itertools, time
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

STEPS = 5000
CONFIGS = {
    "Y spawn3.0 fail0.0": dict(spawn_per_patch=3.0, fail_cost=0.0),
    "Z spawn2.0 fail0.3": dict(spawn_per_patch=2.0, fail_cost=0.3),
}

def job(arg):
    tag, cond = arg
    kw = dict(A.VARIANTS[cond]); kw.update(CONFIGS[tag])
    t0 = time.time()
    r = run(Config(n_steps=STEPS, seed=0, **kw), verbose=False)
    L = r["log"]
    return (f"{tag:<20} {cond:<20} safe {A.safe(L):.3f}  probe_food {A.half(L,'probe_adv_food'):7.3f}  "
            f"eta2 {A.half(L,'eta2'):.3f}  probe_recipe {A.half(L,'probe_adv'):7.3f}  "
            f"| hit {A.hit(L):.3f}  att/life {A.half(L,'attempts_per_life'):5.2f}  "
            f"pop {A.half(L,'pop'):3.0f}  inj {A.half(L,'injections'):.1f}  "
            f"nutshare {A.half(L,'nut_share'):.2f}  gen {A.half(L,'max_gen'):3.0f}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, ["fixed", "plastic (W2)", "fixed + B (ceiling)"]))
    with ProcessPoolExecutor(max_workers=6) as ex:
        for line in ex.map(job, jobs):
            print(line, flush=True)
