#!/usr/bin/env python3
"""Build autopoiesis_v3_9_rig_fixed.ipynb.  Colab-only: sim.py and analysis.py uploaded
alongside, no local paths, no multiprocessing, no run script."""
import json, pathlib

HERE = pathlib.Path(__file__).parent
MD_INTRO = (HERE / "spec_v3_9.md").read_text()
MD_READING = (HERE / "notebook_reading_v3_9.md").read_text()

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
      " (eat =", sim.EAT, ", interact =", sim.INTERACT, ")",
      " hidden:", A.WORLD["hidden"], " chance recipe hit:", round(A.CHANCE, 3))
print("\\nworld (v3.1 metabolism throughout; the chain from v3.6):")
for k, v in A.WORLD.items():
    print(f"  {k:<20} {v}")
print("\\n--- world-semantics self-test (every row of the action x cell table) ---")
if not sim.world_semantics_selftest():
    raise SystemExit("the world does not match the spec; nothing below is meaningful")

print("\\n--- standing cover (audit fix B: densities to a target, not a feel) ---")
if not sim.assert_cover(sim.Config(chain=True, **{k: v for k, v in A.WORLD.items()
                                                  if k in sim.Config.__dataclass_fields__})):
    raise SystemExit("cover targets not met; the world is saturated and contact is ambient")

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

CODE_RUN = '''# QUICK = True   exercises every cell below: 1 seed, 1500-step phases (3000 total), and
#                recipe_every dropped to 500 so recipe changes actually occur in phase 2.
#                A smoke test, NOT a result -- 1500 steps is only a handful of generations,
#                so row 0 will flag conditions as uninterpretable and row 1 will not reproduce.
# QUICK = False  the real experiment: 8 arms x 3 seeds, 8000-step phases (16000 total).
QUICK = True

if QUICK:
    SEEDS, PHASE_STEPS, OVERRIDES = [0], 1500, dict(recipe_every=500)
else:
    SEEDS, PHASE_STEPS, OVERRIDES = [0, 1, 2], 8000, {}      # seeds 3-4 held in reserve

# Pickled after every run, so a dropped Colab session costs one run rather than the lot.
# To resume:  import pickle; results = pickle.load(open("results_v3_9.pkl", "rb"))
results = A.run_experiment(seeds=SEEDS, phase_steps=PHASE_STEPS,
                           save_path="results_v3_9.pkl", **OVERRIDES)
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
        md("## 2. Run\n\n**Runtime.** A staged run is 16000 steps — twice a v3.6 run — but populations "
           "here are 170–260 rather than 300–450. Measured on a 4-core box: **3–6 min per run**, "
           "so the grid of 8 × 3 = **24 runs**; see the pre-check timings. Colab CPU may be slower.\n\nThe cell checkpoints to `results_v3_9.pkl` after every run, so a dropped "
           "session costs one run. To resume, reload the pickle and re-run only the missing "
           "seeds, then merge.\n\n**Seeds 3–4 are held in reserve.** A positive on any row gets "
           "them before it is called. To append them later, with the same three files present:"
           "\n\n```python\nimport pickle\nbase = pickle.load(open(\"results_v3_9.pkl\", \"rb\"))\n"
           "more = A.run_experiment(seeds=[3, 4], save_path=\"results_seeds34.pkl\")\n"
           "for k in base:\n    base[k] += more[k]          # seed order stays [0,1,2,3,4]\n"
           "pickle.dump(base, open(\"results_v3_8_all.pkl\", \"wb\"))\nresults = base\n```\n\n"
           "The seed criterion adapts automatically: `min(4, n_seeds)`, so 3 seeds reads 3/3 and "
           "5 seeds reads 4/5."),
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
        "colab": {"name": "autopoiesis_v3_9_rig_fixed.ipynb", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = HERE / "autopoiesis_v3_9_rig_fixed.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
