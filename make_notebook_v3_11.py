#!/usr/bin/env python3
"""Build autopoiesis_v3_11_preparation_world.ipynb.  Colab-only: sim.py and analysis.py uploaded
alongside, no local paths, no multiprocessing, no run script."""
import json, pathlib

HERE = pathlib.Path(__file__).parent
MD_INTRO = (HERE / "spec_v3_11.md").read_text()
MD_READING = (HERE / "notebook_reading_v3_11.md").read_text()

CODE_SETUP = '''# --- Colab check -------------------------------------------------------------
# Upload sim.py and analysis.py next to this notebook (Files pane, or run:
#     from google.colab import files; files.upload()
# and pick both).  Nothing else is needed: pure numpy + matplotlib.
import os, sys

missing = [f for f in ("sim.py", "analysis.py") if not os.path.exists(f)]
if missing:
    raise SystemExit(f"missing {missing} in {os.getcwd()} -- upload them next to this notebook")
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import numpy as np
import sim, analysis as A

print("numpy", np.__version__)
print("observation size:", sim.N_IN, " actions:", sim.N_ACTIONS,
      " (eat =", sim.EAT, ", preps =", sim.PREP0, "..", sim.PREP0 + sim.N_PREPS - 1, ")",
      " hidden:", A.WORLD["hidden"], " chance recipe hit:", round(A.CHANCE, 3))
print("\\nworld (v3.1 metabolism throughout; the chain from v3.6):")
for k, v in A.WORLD.items():
    print(f"  {k:<20} {v}")
print("\\n--- world-semantics self-test (every row of the action x cell table) ---")
if not sim.world_semantics_selftest():
    raise SystemExit("the world does not match the spec; nothing below is meaningful")

print("\\n--- density ---")
print("no density to assert: the opportunity is every food cell")

print("\\n--- founder-tag self-test ---")
print("injected agents are fresh random genomes; their OWN events are excluded from every")
print("event-weighted metric, their children's are not.  The test forces injection so the")
print("exclusion path is actually exercised -- passing on a run with no injections proves nothing.")
if not sim.founder_tag_selftest():
    raise SystemExit("the founder tag is not wired correctly; every founder-free number is suspect")

print("\\n--- replay-mapping self-test ---")
print("a knockout that re-seeds the world does NOT get the mapping its genomes were selected")
print("under.  This checks that force_mapping pins it, and that a re-seeded replay can differ")
print("from the source's final mapping -- the condition that made the old knockout misread.")
if not sim.replay_mapping_selftest():
    raise SystemExit("the replay does not carry the mapping it is given; row 3b is meaningless")

print("\\n--- knockout-window self-test ---")
print("eta_scale = 0 stops learning, not reproduction.  A NON-PLASTIC genome's per-type hits")
print("must swap when the mapping swaps; they do not if the replay window lets the population")
print("re-evolve.  This is the check that row 3b reads the genome and not a fresh adaptation.")
if not A.knockout_window_selftest():
    raise SystemExit("the knockout window is too long; row 3b would read re-selection")

print("\\n--- learning-rule self-test ---")
print("one agent, one fixed observation, one chosen action; action_noise = 0 so act() is")
print("deterministic and the logit checked belongs to the action that laid the trace.")
if not sim.learning_rule_selftest():
    raise SystemExit("the learning rule is not behaving; nothing below is meaningful")

print("\\nconditions:")
for name, spec in A.VARIANTS.items():
    ph = " -> ".join(f"{p['n_steps']} steps chain={p['chain']}" for p in spec["phases"])
    print(f"  {name:<24} {ph}")
    print(f"  {'':<24} {({k: v for k, v in spec['kw'].items() if k not in A.WORLD})}")
'''

CODE_RUN = '''# MODE picks the stage.  All three write the SAME checkpoint, and each stage SKIPS any
# (arm, seed) already in it -- so run "acceptance", read it, then set MODE = "grid" and run this
# cell again in the same session: it adds only what acceptance did not do.
#
#   "quick"       smoke test.  1 seed, 1500-step phases, 3 arms, prep_every 300 so remaps
#                 actually occur.  NOT a result -- a handful of generations, so row 0 will
#                 flag conditions and row 1 will not reproduce.  Read it only to confirm
#                 every cell below produces the output it should.        ~5 min
#   "acceptance"  fixed / scrambled / plastic (W2) x seeds 0-1, full 8000-step phases.
#                 The stopping rule is read here FIRST.                  ~60-75 min
#   "grid"        continues from the acceptance checkpoint: adds `random policy` and
#                 `fixed + B (ceiling)` for seeds 0-1, and all five arms for seed 2.
#                 Six of the fifteen runs are already done, so this is nine runs. ~75-90 min
#
# Wall clock measured on a 4-core box at max_pop 800 with prep_value 1.5 (16000 steps per run):
# random 3.1 min, fixed 10.9, scrambled 11.5, plastic 14.7, ceiling 13.3.  prep_value 1.0 lowers
# the standing population, so these are UPPER bounds; Colab CPU typically runs 1.5-2x slower.
# If the cap check fails and max_pop goes to 1200, scale by roughly 1200/800.
MODE = "quick"

CKPT = "results_v3_11.pkl"
CORE = ["fixed", "scrambled", "plastic (W2)"]
ALL  = list(A.VARIANTS)

if MODE == "quick":
    SEEDS, PHASE_STEPS, ARMS, OVERRIDES = [0], 1500, CORE, dict(prep_every=300, recipe_every=500)
    CKPT = "results_v3_11_quick.pkl"
elif MODE == "acceptance":
    SEEDS, PHASE_STEPS, ARMS, OVERRIDES = [0, 1], 8000, CORE, {}
elif MODE == "grid":
    SEEDS, PHASE_STEPS, ARMS, OVERRIDES = [0, 1, 2], 8000, ALL, {}   # seeds 3-4 held in reserve
else:
    raise SystemExit(f"MODE must be quick / acceptance / grid, not {MODE!r}")

variants = {k: A.VARIANTS[k] for k in ARMS}
results = A.run_experiment(seeds=SEEDS, phase_steps=PHASE_STEPS, variants=variants,
                           results=A.load(CKPT), save_path=CKPT, **OVERRIDES)
print("\nin the checkpoint:", {k: sorted(r["cfg"]["seed"] for r in v) for k, v in results.items()})
'''

CODE_SUMMARY = 'A.summary(results)\n'
CODE_TRANSITION = 'A.transition_table(results, bin_size=500, span=2000)\n'
CODE_CURVES = ('A.curves(results, "meal", "meal number in an agent\'s life")        # phase 1\n'
               'A.curves(results, "att",  "attempt number in an agent\'s life")     # phase 2\n'
               'A.curves(results, "rec",  "attempts since the last recipe change")  # phase 2\n')
CODE_PLOTS = 'A.plot_results(results)\n'


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


nb = {
    "cells": [
        md(MD_INTRO),
        md("## 1. Setup\n\nThe sim and the analysis are imported, not inlined. This cell fails "
           "loudly if either file is missing."),
        code(CODE_SETUP),
        md("## 2. Run \u2014 staged\n\n"
           "`MODE` picks the stage. All three stages write the **same checkpoint** and each one "
           "**skips any (arm, seed) already in it**, so you can run `\"acceptance\"`, read the "
           "stopping rule, then set `MODE = \"grid\"` and re-run this cell **in the same "
           "session** \u2014 it does only the nine runs acceptance did not.\n\n"
           "| stage | arms | seeds | runs | est. wall clock |\n|---|---|---|---|---|\n"
           "| `quick` | 3 core | 0 | 3 | ~5 min |\n"
           "| `acceptance` | fixed, scrambled, plastic (W2) | 0\u20131 | 6 | **~60\u201375 min** |\n"
           "| `grid` | + random policy, ceiling | 0\u20132 | 9 more | **~75\u201390 min** |\n\n"
           "**Where the estimate comes from.** Measured on a 4-core box at `max_pop` 800 with "
           "`prep_value` 1.5, per 16000-step run: random 3.1 min, fixed 10.9, scrambled 11.5, "
           "plastic 14.7, ceiling 13.3. `prep_value` 1.0 lowers the standing population, so "
           "these are **upper bounds**. Colab CPU typically runs 1.5\u20132\u00d7 slower than "
           "that, and if the cap check fails and `max_pop` goes to 1200, scale by ~1200/800.\n\n"
           "The cell checkpoints after **every run**, so a dropped session costs one run. To "
           "resume, just re-run the cell \u2014 `A.load(CKPT)` picks up what is already there "
           "and skips it.\n\n**Seeds 3\u20134 are held in reserve.** A positive on any row gets "
           "them before it is called: set `SEEDS = [3, 4]` after the grid and re-run this cell; "
           "the checkpoint merges them and the seed criterion adapts automatically "
           "(`min(4, n_seeds)`, so 3 seeds reads 3/3 and 5 seeds reads 4/5)."),
        code(CODE_RUN),
        md("## 3. The printed summary\n\nPhase 1 and phase 2 blocks separately, then per-seed "
           "values, then the decision numbers in the order of the table above.\n\n**Read row 1 "
           "first — it is a stop condition.** If phase 1 does not reproduce v3.1, nothing in "
           "phase 2 is readable.\n\n**Reading a QUICK run:** a smoke test, not a result. Phases "
           "of 1500 steps are a handful of generations, so row 0 will flag conditions and row 1 "
           "will not reproduce. Read it only to confirm every cell produces the output it should."),
        code(CODE_SUMMARY),
        md("## 4. The transition\n\n500-step bins across 2000 steps either side of the chain "
           "switching on. This is where a collapse becomes visible, and where `eta2`/`lam2` "
           "either hold or start falling as they did in v3.6."),
        code(CODE_TRANSITION),
        md("## 5. Curves\n\n`meal` is v3.1's within-life food curve, read on **phase 1** — the "
           "positive control. `att` and `rec` are the recipe curves, read on **phase 2**; row 4 "
           "requires `att` to rise (or `hit_old` > `hit_young`)."),
        code(CODE_CURVES),
        md("## 6. Plots\n\nSolid black line = the chain switches on; dashed = recipe change."),
        code(CODE_PLOTS),
        md(MD_READING),
    ],
    "metadata": {
        "colab": {"name": "autopoiesis_v3_11_preparation_world.ipynb", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = HERE / "autopoiesis_v3_11_preparation_world.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
