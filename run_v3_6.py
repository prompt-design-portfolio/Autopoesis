#!/usr/bin/env python3
"""
Run the v3.6 experiment and write the results to a pickle.

    python3 run_v3_6.py --seeds 0 1 2 3 4 --steps 8000 --out results_v3_6.pkl --jobs 4

Runs are independent (each is seeded from cfg.seed alone), so they are farmed out to
worker processes.  The printed summary is produced by analysis.summary and does not
depend on the order in which the runs finished.
"""

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from sim import Config, run
import analysis


def _one(job):
    name, kw, seed, steps, verbose = job
    t0 = time.time()
    r = run(Config(n_steps=steps, seed=seed, **kw), verbose=verbose)
    return name, seed, r, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default="results_v3_6.pkl")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--only", nargs="*", default=None, help="subset of condition names")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    variants = analysis.VARIANTS if not args.only else {k: analysis.VARIANTS[k] for k in args.only}
    jobs = [(name, kw, seed, args.steps, not args.quiet)
            for seed in args.seeds for name, kw in variants.items()]

    print(f"{len(jobs)} runs ({len(variants)} conditions x {len(args.seeds)} seeds), "
          f"{args.steps} steps, {args.jobs} workers", flush=True)
    results = {name: [None] * len(args.seeds) for name in variants}
    order = {s: i for i, s in enumerate(args.seeds)}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for name, seed, r, dt in ex.map(_one, jobs):
            results[name][order[seed]] = r
            done += 1
            print(f"[{done}/{len(jobs)}  {time.time()-t0:.0f}s]  done: {name} seed={seed} ({dt:.0f}s)", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s total)\n", flush=True)

    analysis.summary(results)
    print()
    analysis.recovery_table(results, threshold=0.30)
    for key, xlabel in [("att", "attempt number in an agent's life"),
                        ("rec", "attempts since the last recipe change"),
                        ("meal", "meal number in an agent's life")]:
        analysis.curves(results, key, xlabel, show=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
