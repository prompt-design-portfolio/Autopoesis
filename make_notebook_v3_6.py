#!/usr/bin/env python3
"""Build autopoiesis_v3_6_recipe_task.ipynb.  The notebook IMPORTS sim.py and analysis.py rather
than carrying the code inline (v3.5 and earlier embedded the whole simulation in the first cell);
the modules are the version-controlled source of truth.  It must run on a Colab CPU runtime with
sim.py and analysis.py uploaded alongside it: no local paths, no multiprocessing, no run script."""
import json, pathlib

HERE = pathlib.Path(__file__).parent
MD_INTRO = (HERE / "notebook_intro_v3_6.md").read_text()
MD_READING = (HERE / "notebook_reading_v3_6.md").read_text()

CODE_SETUP = '''# --- Colab check -------------------------------------------------------------
# Upload sim.py and analysis.py next to this notebook (Files pane, or run:
#     from google.colab import files; files.upload()
# and pick both).  Nothing else is needed: pure numpy + matplotlib.
import os, sys

missing = [f for f in ("sim.py", "analysis_v3_6.py") if not os.path.exists(f)]
if missing:
    raise SystemExit(f"missing {missing} in {os.getcwd()} -- upload them next to this notebook")
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import numpy as np
import sim, analysis_v3_6 as A

print("numpy", np.__version__)
print("observation size:", sim.N_IN, " pairs:", sim.N_PAIRS, " chance recipe hit:", round(A.CHANCE, 3))
print("\\nworld:")
for k, v in A.WORLD.items():
    print(f"  {k:<20} {v}")
print("\\nconditions:")
for name, kw in A.VARIANTS.items():
    print("  " + name.ljust(26) + str({k: v for k, v in kw.items() if k not in A.WORLD}))
'''

CODE_RUN = '''# QUICK = True   exercises every cell below in ~6 min: 1 seed, 1000 steps, and recipe_every
#                dropped to 400 so two recipe changes actually occur and the recovery tables and
#                the since-change curves are exercised too.  It is a smoke test, NOT a result.
# QUICK = False  the real experiment: 6 conditions x 3 seeds x 8000 steps (~2-2.5 h).
QUICK = True

if QUICK:
    SEEDS, N_STEPS, OVERRIDES = [0], 1000, dict(recipe_every=400)
else:
    SEEDS, N_STEPS, OVERRIDES = [0, 1, 2], 8000, {}      # seeds 3-4 held in reserve; see below

# Results are pickled after every run, so a dropped Colab session costs one run, not the lot.
# To resume:  import pickle; results = pickle.load(open("results_v3_6.pkl", "rb"))
results = A.run_experiment(seeds=SEEDS, n_steps=N_STEPS, save_path="results_v3_6.pkl", **OVERRIDES)
'''

CODE_SUMMARY = 'A.summary(results)\n'
CODE_RECOVERY = 'A.recovery_table(results, threshold=0.30)\nA.recovery_table(results, threshold=0.25)\n'
CODE_CURVES = ('A.curves(results, "att",  "attempt number in an agent\'s life")\n'
               'A.curves(results, "rec",  "attempts since the last recipe change")\n'
               'A.curves(results, "meal", "meal number in an agent\'s life")\n')
CODE_PLOTS = 'A.plot_results(results)\n'


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


nb = {
    "cells": [
        md(MD_INTRO),
        md("## 1. Setup\n\nThe sim and the analysis are imported, not inlined. This cell fails loudly "
           "if either file is missing."),
        code(CODE_SETUP),
        md("## 2. Run\n\n**Runtime.** ~6–8 min per 8000-step run on a Colab CPU runtime (populations "
           "of 200–420 with 24 hidden units and 11 observation channels — heavier than v3.5\'s ~2.5–5 "
           "min). This pass is 6 conditions × 3 seeds = **18 runs ≈ 2–2.5 hours**.\n\nThe cell "
           "checkpoints to `results_v3_6.pkl` after every run, so a dropped Colab session costs one "
           "run rather than the lot. To resume, reload the pickle and re-run only the missing seeds, "
           "then merge.\n\n**Seeds 3–4 are held in reserve.** A positive on any row gets them before "
           "it is called. To append them later, in a fresh session with the same three files and the "
           "same `results_v3_6.pkl` present:\n\n```python\nimport pickle\nbase = pickle.load(open(\"results_v3_6.pkl\", \"rb\"))\n"
           "more = A.run_experiment(seeds=[3, 4], n_steps=8000, save_path=\"results_seeds34.pkl\")\n"
           "for k in base:\n    base[k] += more[k]          # seed order stays [0,1,2,3,4]\n"
           "pickle.dump(base, open(\"results_v3_6_all.pkl\", \"wb\"))\nresults = base\n```\n\n"
           "The seed criterion adapts automatically: `min(4, n_seeds)`, so 3 seeds reads as 3/3 and "
           "5 seeds as 4/5.\n\nSet `QUICK = False` for the real thing."),
        code(CODE_RUN),
        md("## 3. The printed summary\n\nEvent-weighted second-half aggregates, per-seed breakdowns, "
           "then the decision numbers in the order of the table above.\n\n**Reading a QUICK run:** it "
           "is a smoke test, not a result. 1000 steps is ~4 generations, so populations have not "
           "equilibrated and row 0 will flag several conditions as uninterpretable — that is expected "
           "and says nothing about the real run. Read it only to confirm every cell produces the "
           "output it should."),
        code(CODE_SUMMARY),
        md("## 4. Recovery after a recipe change"),
        code(CODE_RECOVERY),
        md("## 5. Curves\n\n`att` is the within-life curve — **row 3 requires this to rise**, or "
           "`hit_old` > `hit_young`. `rec` is the population's recovery after a recipe change. "
           "`meal` is the v3 food curve, carried over as the positive control."),
        code(CODE_CURVES),
        md("## 6. Plots"),
        code(CODE_PLOTS),
        md(MD_READING),
    ],
    "metadata": {
        "colab": {"name": "autopoiesis_v3_6_recipe_task.ipynb", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = HERE / "autopoiesis_v3_6_recipe_task.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
