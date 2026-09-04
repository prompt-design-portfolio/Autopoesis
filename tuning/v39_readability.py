"""Amendment-2 readability criterion: attempts/life >= 3 in `random policy`, judged against the
null ONLY.  The agreed bands are item cover 20-25%, station cover 6-8%.  At the band centre the
null gives 0.96.  This asks what it would actually take -- attempts/life is lifespan x attempt
rate, and the attempt rate is bounded by how often a carrying agent stands on a station."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from concurrent.futures import ProcessPoolExecutor
from sim import Config, run, standing_cover
import analysis as A

GRID = [("band centre (agreed)", 3.0, 85), ("band ceiling",  3.5, 100),
        ("stations x2",          3.5, 200), ("stations x3",   3.5, 300)]

def job(spec):
    lab, items, st = spec
    kw = dict(A.VARIANTS["random policy"]["kw"]); kw.update(items_per_step=items, stations_per_type=st)
    cov, _ = standing_cover(Config(chain=True, **{k: v for k, v in kw.items()
                                                  if k in Config.__dataclass_fields__}), seeds=(0, 1))
    r = run(Config(seed=0, **kw), verbose=False,
            phases=[dict(p, n_steps=3000) for p in A.VARIANTS["random policy"]["phases"]])
    p2 = A.phase_half(r, 1)
    return (f"{lab:<22} items {cov['items']*100:5.1f}%  stations {cov['stations']*100:5.1f}%   "
            f"att/life {A.half(p2,'attempts_per_life'):5.2f}  att/1k {A.per_1k(p2,'n_attempts_raw'):6.2f}  "
            f"pick/1k {A.half(p2,'pickups_per_1k'):6.2f}  pop {A.half(p2,'pop'):4.0f}  "
            f"P(int|at station) {A.half(p2,'int_at_station'):.3f}   (target att/life >= 3)")

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as ex:
        for line in ex.map(job, GRID):
            print(line, flush=True)
