#!/usr/bin/env python3
"""Build autopoiesis_v3_6_recipe_task.ipynb.  The notebook IMPORTS sim.py and analysis.py
rather than carrying the code inline (v3.5 and earlier embedded the whole simulation in
the first cell); the modules are the version-controlled source of truth."""
import json, pathlib

MD_INTRO = open(pathlib.Path(__file__).parent / "notebook_intro.md").read()
MD_READING = open(pathlib.Path(__file__).parent / "notebook_reading.md").read()

CODE_SETUP = '''# The simulation and the analysis live in sim.py and analysis.py next to this notebook.
# On Colab, upload sim.py, analysis.py and run_v3_6.py alongside it (or clone the repo).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))

import numpy as np
from sim import Config, run, N_IN, N_PAIRS
import analysis as A

print("observation size:", N_IN, " pairs:", N_PAIRS)
print("conditions:")
for name, kw in A.VARIANTS.items():
    print(f"  {name:<24} {kw}")
'''

CODE_RUN = '''SEEDS = [0, 1, 2, 3, 4]
N_STEPS = 10000

# ~2.5 min per run single-threaded.  7 conditions x 5 seeds = 35 runs.
# Faster, out of process, using all cores:
#     python3 run_v3_6.py --seeds 0 1 2 3 4 --steps 8000 --out results_v3_6.pkl --jobs 4
results = A.run_experiment(seeds=SEEDS, n_steps=N_STEPS, verbose=False)
'''

CODE_LOAD = '''# ... or load what run_v3_6.py already produced:
# import pickle; results = pickle.load(open("results_v3_6.pkl", "rb"))
'''

CODE_SUMMARY = '''A.summary(results)
'''

CODE_RECOVERY = '''A.recovery_table(results, threshold=0.30)
A.recovery_table(results, threshold=0.25)
'''

CODE_CURVES = '''A.curves(results, "att",  "attempt number in an agent's life")
A.curves(results, "rec",  "attempts since the last recipe change")
A.curves(results, "meal", "meal number in an agent's life")
'''

CODE_PLOTS = '''A.plot_results(results)
'''


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


nb = {
    "cells": [
        md(MD_INTRO),
        md("## 1. Setup\n\nThe sim and the analysis are imported, not inlined."),
        code(CODE_SETUP),
        md("## 2. Run"),
        code(CODE_RUN),
        code(CODE_LOAD),
        md("## 3. The printed summary\n\nEvent-weighted second-half aggregates, per-seed breakdowns, "
           "then the decision numbers in the order of the table above."),
        code(CODE_SUMMARY),
        md("## 4. Recovery after a recipe change"),
        code(CODE_RECOVERY),
        md("## 5. Curves\n\n`att` is the within-life curve (does an individual get better with "
           "experience?), `rec` the population's recovery after a recipe change, `meal` the v3 food "
           "curve carried over as the positive control."),
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

out = pathlib.Path(__file__).parent / "autopoiesis_v3_6_recipe_task.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
