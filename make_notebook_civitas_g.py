#!/usr/bin/env python3
"""Build notebooks/Civitas_G_Colab.ipynb.

A5: "Pre-checks before anything full-length: one seed, short phases, every arm, transition table.
Files come to me at the pre-check stage. I run acceptance and grids." This notebook is the vehicle
for that clause on the G lineage -- somewhere the self-tests, the G0 hashes and the reproduction
can be run without a local checkout.

Every code cell is held in a RAW literal (r'''...'''), and `write_checked` compiles all of them
before the file is written. That is not decoration: a `\n` inside a non-raw triple-quoted literal
is a real newline by the time the generator runs, which produces a notebook whose cells are
syntactically broken in a way JSON validity and notebook loading both miss. It shipped three times
before nbcheck existed.
"""
import pathlib

from nbcheck import write_checked

HERE = pathlib.Path(__file__).parent

MD_INTRO = r"""# Civitas-G — self-tests, hashes, and the G1 reproduction

A rebuild under `CIVITAS_G_MASTER_BUILD_DIRECTIVE.md`. The inversion that defines it:

> **The learner is the only thing that thinks.** Civitas provides persistence, measurement and the
> environment; it never provides cognition.

There is no agent in `civitas_g/`. No LLM is a component of one, no hand-written policy stands in
for one, and nothing reads the store on a learner's behalf. The learner is the grown network of
`sim_v3_13.py` — innate weights plus a learned component `H`, a local rule with an eligibility
trace, the agent's own modulator, and `H` is not inherited.

This notebook runs four things, in the order the directive requires:

1. **the world of record**, checked against what this build was written against;
2. **the G0 hashes** — the reference summaries and the code they were produced by;
3. **B§6's self-tests**, which run before any result is read, and any failure of which halts;
4. **the G1 reproduction** — `precheck_v3_13.txt` reproduced from rows recomputed out of a
   database.

**On Colab you get one backend.** SQLite is the Colab-compatible implementation (§43); PostgreSQL
is the source of truth (§41). D12's clause — that the same rows read back on each backend give the
same reading — needs both, and the report says so rather than quietly passing on one.
"""

MD_TIME = r"""## Wall clock

The reproduction runs five arms at one seed. At the reference's own 3000-step phases that is about
**17 minutes** on a Colab CPU (3.8 + 4.2 + 4.0 + 1.3 + 4.0 measured). `MODE = "quick"` runs the
same five arms at 300-step phases in about **90 seconds** — it exercises every code path and
reproduces nothing, which is the point of a smoke test.

Only `MODE = "reference"` can meet the gate. A shorter run is a different experiment and its diff
is meaningless; the notebook says so rather than printing a number that looks like a result.
"""

MD_READING = r"""## Reading the report

**`DIFF = 0`** means every field the reference prints was recomputed from the database and matched
at the width the reference printed it. 182 fields across nine tables and five arms.

Four things the report will tell you that are easy to miss:

* **the seed is recovered, not recorded.** `precheck_v3_13.txt` states neither its seeds nor its
  configuration. The invocation was inferred from what the file happened to print — "switch at
  t = 3000" in every transition table, five π-epochs against the slow arm's two, identical phase-1
  bins for two arms that differ only in phase 2. That configuration appears nowhere in the
  repository.
* **`assay_selftest` is NOT AVAILABLE, and that is a finding.** A2.1 cites it as an existing
  reference. It is nowhere in the repository — and at G2 the reason got worse: the instrument it
  would test cannot be run, because `run()` neither returns the store nor accepts one.
* **a G2/G3 self-test reported `N/A` does not halt a G1 read.** A milestone that has not happened
  is not an instrument that is broken. A G1 test reported `N/A` *does* halt.
* **an empty comparison does not pass.** `all([])` is `True`, and an acceptance report over zero
  fields reporting success is the defect that rule exists to prevent.

If the diff is not zero, the fix is **not** to vary the configuration until it agrees. D4: try the
candidate seeds, then declare the artifact unreproducible and fall back to
`v3_11_grid_summary.txt`. A1.2 — passing by construction is failing — applies to the reproduction
as much as to an acceptance level.
"""

CODE_SETUP = r'''# --- get the repository ------------------------------------------------------
# civitas_g is a package, not two loose files, so this clones rather than asking for uploads.
# Set BRANCH to whatever you are reading.
import os, subprocess, sys

REPO   = "https://github.com/prompt-design-portfolio/Autopoesis.git"
BRANCH = "claude/master-prompt-init-ehmsbe"
ROOT   = "/content/Autopoesis" if os.path.isdir("/content") else os.path.abspath("Autopoesis")

if not os.path.isdir(ROOT):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, ROOT], check=True)
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sqlalchemy>=2.0", "numpy"], check=True)

import numpy as np, sqlalchemy
import civitas_g

print("civitas_g", civitas_g.__version__, "| numpy", np.__version__,
      "| sqlalchemy", sqlalchemy.__version__)
print("cwd", os.getcwd())
'''

CODE_WORLD = r'''# --- 1. the world of record --------------------------------------------------
# analysis_v3_13.WORLD, not sim_v3_13.Config(). The Config defaults are v3.10 leftovers: they put
# the chance EV of a preparation at -0.100 and the era at 2000 steps. Building a bare Config()
# builds a different experiment, so the check runs before anything else does.
from civitas_g.world.spec import (WORLD, CHANCE, TYPE_BLIND, check_world,
                                  check_chance_ev_is_zero, clocks_of, standing_density)

check_world()
ev = check_chance_ev_is_zero()

for k in sorted(WORLD):
    print(f"  {k:<20}{WORLD[k]!r}")
print(f"\n  chance EV of a preparation {ev:+.4f}   "
      f"(prep_value == (K-1) * prep_fail, so a chance preparation is worth nothing)")
print(f"  levels: chance {CHANCE:.3f}   type-blind {TYPE_BLIND:.3f}   full 1.000")

c = clocks_of(WORLD)
print(f"\n  era clocks (steps): flip {c.flip_every}   mapping+pi {c.prep_every}   "
      f"patch drift {c.patch_drift_every}")

d = standing_density(WORLD)
print(f"  grid {d.grid_cells} cells; patches cover {d.patch_cover_if_disjoint:.3f} if disjoint")
print(f"  mark half-life {d.mark_half_life_steps:.1f} steps")
print("  standing food cover: NOT AVAILABLE -- consumption-limited, not in the engine's log,")
print("  and A1.8 forbids adding it in Civitas (D10)")
'''

CODE_PINS = r'''# --- 2. the G0 hashes --------------------------------------------------------
# B§5.1: record the reference files and the producing code by hash at G0 and reproduce against
# those, not against anything that lands later. Every candidate summary in this repository was
# produced by code that changed afterwards, so each reference pins the blob at ITS OWN commit.
from civitas_g.manifest import (REFERENCES, MISSING_ARTIFACTS, engine_identity, verify_pins)

failures = 0
for ok, detail in verify_pins():
    print(f"  [{'OK  ' if ok else 'FAIL'}] {detail}")
    failures += 0 if ok else 1

print("\nreferences:")
for ref in REFERENCES:
    print(f"  {ref.key:<18}{'GATED' if ref.gated else 'recorded, not gated':<22}{ref.why}")
    for label, pin in (("sim", ref.sim), ("analysis", ref.analysis)):
        if pin is None:
            print(f"    {label:<10}NOT COMMITTED at {ref.summary.commit}")
        else:
            print(f"    {label:<10}{pin.sha256[:16]}  @ {pin.commit}")

print("\nengine identity now:")
for k, v in engine_identity().items():
    print(f"  {k:<22}{v}")

print("\nnamed by the directive and absent from the repository:")
for name, why in MISSING_ARTIFACTS:
    print(f"  {name}\n    {why}")

if failures:
    raise SystemExit(f"{failures} pinned artifact(s) no longer match; nothing below is a gate")
'''

CODE_SELFTEST = r'''# --- 3. B§6's self-tests -----------------------------------------------------
# "Run before any result is read; any failure halts."
#
# FAST = True skips the three that run the engine for minutes (engine_drift, store_patch, and the
# modulator check). They are the slowest and they test the engine rather than this run, so a
# pre-check can reasonably defer them -- but a REPORTED number must not.
from civitas_g.selftests import run_selftests, report, halt_on_failure

FAST = True
SLOW = ["engine_drift", "store_patch", "modulator"]

names = None
if FAST:
    from civitas_g.selftests import _registry
    names = [t.name for t in _registry() if t.name not in SLOW]
    print(f"FAST: skipping {SLOW}. Set FAST = False before reporting a number.\n")

results = run_selftests(names=names, verbose=True)
print()
print(report(results))
halt_on_failure(results)
print("\nno self-test halts a read.")
'''

CODE_REPRODUCE = r'''# --- 4. the G1 reproduction --------------------------------------------------
#   "quick"      five arms at 300-step phases, ~90 s. Exercises every path, reproduces nothing.
#   "reference"  five arms at the reference's own 3000-step phases, ~17 min. The only mode that
#                can meet the gate.
MODE = "quick"
SEED = None          # None uses the recovered seed (0). D4: try 1 and 2 before concluding.

import os
from civitas_g.persistence.engine import create_all, create_db_engine, session_factory
from civitas_g.persistence.models import Base
from civitas_g.g1 import run_g1

phase_steps = 3000 if MODE == "reference" else 300
if MODE not in ("quick", "reference"):
    raise SystemExit(f"MODE must be quick or reference, not {MODE!r}")

os.makedirs("var", exist_ok=True)
engine = create_db_engine(url="sqlite:///var/civitas_g_colab.db")
Base.metadata.drop_all(engine)
create_all(engine)

# One backend on Colab. D12's clause needs the same rows read back on each of two, so the report
# marks it UNMET rather than passing on one.
factories = {"sqlite": session_factory(engine=engine)}

g1 = run_g1(factories, seed=SEED, phase_steps=phase_steps, verbose=True,
            skip_selftests=True)      # already run above, and halted on failure
print()
print(g1.report())

if MODE != "reference":
    print("\n" + "=" * 78)
    print("MODE was not 'reference', so the diff above is NOT the gate: a shorter run is a")
    print("different experiment. Set MODE = 'reference' for a number that means anything.")
    print("=" * 78)
'''

CODE_STORE = r'''# --- 5. where G2 stands ------------------------------------------------------
# The store layer is built; the assay is blocked on one decision about the engine.
from civitas_g.cli import main as civitas_g_main

civitas_g_main(["store"])
'''


def build():
    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def code(text):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": text.splitlines(keepends=True)}

    nb = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"colab": {"provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [
            md(MD_INTRO),
            code(CODE_SETUP),
            md("## 1. The world of record"),
            code(CODE_WORLD),
            md("## 2. The G0 hashes"),
            code(CODE_PINS),
            md("## 3. Self-tests (B§6)"),
            code(CODE_SELFTEST),
            md(MD_TIME),
            md("## 4. The G1 reproduction"),
            code(CODE_REPRODUCE),
            md(MD_READING),
            md("## 5. Where G2 stands"),
            code(CODE_STORE),
        ],
    }
    out = HERE / "notebooks" / "Civitas_G_Colab.ipynb"
    out.parent.mkdir(exist_ok=True)
    return write_checked(nb, out)


if __name__ == "__main__":
    build()
