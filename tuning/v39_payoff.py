"""attempts/life = lifespan x attempt rate.  Station density raises the rate but the null's
lifespan is short because a random walker eats at chance.  tool_value raises the null's lifespan
(more energy -> longer life -> more attempts) without touching density.  Judged against
`random policy` ONLY, per the agreement."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run
import analysis as A

def job(tv):
    kw = dict(A.VARIANTS["random policy"]["kw"]); kw["tool_value"] = tv
    r = run(Config(seed=0, **kw), verbose=False,
            phases=[dict(p, n_steps=3000) for p in A.VARIANTS["random policy"]["phases"]])
    p2 = A.phase_half(r, 1)
    return (f"tool_value {tv:<5} att/life {A.half(p2,'attempts_per_life'):5.2f}  "
            f"att/1k {A.per_1k(p2,'n_attempts_raw'):6.2f}  pop {A.half(p2,'pop'):4.0f}  "
            f"inj {A.half(p2,'injections'):4.1f}  hit {A.hit(p2):.3f}   (target att/life >= 3)")

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as ex:
        for line in ex.map(job, [1.5, 3.0, 5.0, 8.0]):
            print(line, flush=True)
