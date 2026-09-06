#!/usr/bin/env python3
"""Build autopoiesis_v3_13_record.ipynb.  Colab-only: sim.py and analysis.py uploaded
alongside, no local paths, no multiprocessing, no run script."""
import json, pathlib
from nbcheck import write_checked

HERE = pathlib.Path(__file__).parent
MD_INTRO = (HERE / "spec_v3_13.md").read_text()
MD_READING = (HERE / "notebook_reading_v3_13.md").read_text()

CODE_SETUP = r'''# --- Colab check -------------------------------------------------------------
# Upload sim.py and analysis.py next to this notebook (Files pane, or run:
#     from google.colab import files; files.upload()
# and pick both).  Nothing else is needed: pure numpy + matplotlib.
import os, sys

missing = [f for f in ("sim_v3_13.py", "analysis_v3_13.py") if not os.path.exists(f)]
if missing:
    raise SystemExit(f"missing {missing} in {os.getcwd()} -- upload them next to this notebook")
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import numpy as np
import sim_v3_13 as sim, analysis_v3_13 as A

print("numpy", np.__version__)
print("observation size:", sim.N_IN, " actions:", sim.N_ACTIONS,
      " (eat =", sim.EAT, ", preps =", sim.PREP0, "..", sim.PREP0 + sim.N_PREPS - 1, ")",
      " hidden:", A.WORLD["hidden"], " chance recipe hit:", round(A.CHANCE, 3))
print("\nworld (v3.1 metabolism throughout; the chain from v3.6):")
for k, v in A.WORLD.items():
    print(f"  {k:<20} {v}")
print("\n--- world-semantics self-test (every row of the action x cell table) ---")
if not sim.world_semantics_selftest():
    raise SystemExit("the world does not match the spec; nothing below is meaningful")

print("\n--- D5 readability: preparations per TYPE per era, judged on `fixed` only ---")
print("with T = 3 the food is split three ways, so the criterion (>= 3 per type per era) is the")
print("one most likely to fail.  It failed at spawn_per_patch 3.0 and passes at 6.0.")
print(f"  spawn_per_patch = {A.WORLD['spawn_per_patch']}   type_spawn_w = {A.WORLD.get('type_spawn_w')}")
print("\n--- economics ---")
_K, _pf, _pv = sim.N_PREPS, A.WORLD["prep_fail"], A.WORLD["prep_value"]
print(f"T = {sim.N_TYPES}, K = {_K}, mappings = {len(sim.all_mappings())}, N_ACTIONS = {sim.N_ACTIONS}")
print(f"prep_value {_pv} = (K-1) * prep_fail {(_K-1)*_pf};  chance EV "
      f"{(1/_K)*_pv - ((_K-1)/_K)*_pf:+.4f};  chance {1/_K:.3f}, type-blind {1/sim.N_TYPES:.3f}")

print("\n--- founder-tag self-test ---")
print("injected agents are fresh random genomes; their OWN events are excluded from every")
print("event-weighted metric, their children's are not.  The test forces injection so the")
print("exclusion path is actually exercised -- passing on a run with no injections proves nothing.")
if not sim.founder_tag_selftest():
    raise SystemExit("the founder tag is not wired correctly; every founder-free number is suspect")

print("\n--- replay-mapping self-test ---")
print("a knockout that re-seeds the world does NOT get the mapping its genomes were selected")
print("under.  This checks that force_mapping pins it, and that a re-seeded replay can differ")
print("from the source's final mapping -- the condition that made the old knockout misread.")
if not sim.replay_mapping_selftest():
    raise SystemExit("the replay does not carry the mapping it is given; row 3b is meaningless")

print("\n--- frozen-replay self-test ---")
print("row 3b replays an era-boundary snapshot with births, deaths and injection disabled, so")
print("nothing can change but H.  With learning OFF the hit rate must not move across the")
print("window; if it does, the eta 1 side cannot be read as learning.")
if not A.frozen_selftest():
    raise SystemExit("the frozen replay is not frozen; row 3b would not be an attribution")

print("\n--- learning-rule self-test ---")
print("one agent, one fixed observation, one chosen action; action_noise = 0 so act() is")
print("deterministic and the logit checked belongs to the action that laid the trace.")
if not sim.learning_rule_selftest():
    raise SystemExit("the learning rule is not behaving; nothing below is meaningful")

print("\nconditions:")
for name, spec in A.VARIANTS.items():
    ph = " -> ".join(f"{p['n_steps']} steps chain={p['chain']}" for p in spec["phases"])
    print(f"  {name:<24} {ph}")
    print(f"  {'':<24} {({k: v for k, v in spec['kw'].items() if k not in A.WORLD})}")
'''

CODE_RUN = r'''# MODE picks the stage.  All stages share ONE checkpoint and each SKIPS any (arm, seed)
# already in it, so "grid" continues from the acceptance checkpoint rather than redoing it.
#
#   "quick"       smoke test, 3 arms, 1 seed, 1500-step phases.            ~5 min
#   "acceptance"  fixed / scrambled / plastic (W2) x seeds 0-1, full.      ~60-75 min
#   "grid"        continues: adds `random policy` and `fixed + B (ceiling)`
#                 for seeds 0-1, and all five arms for seed 2.  Nine runs.  ~75-90 min
MODE = "acceptance"

CKPT = "results_v3_13.pkl"
CORE = ["plastic", "plastic + record", "plastic + record (slow)"]
ALL  = list(A.VARIANTS)
REFRESH = []

if MODE == "quick":
    SEEDS, PHASE_STEPS, ARMS, OVERRIDES = [0], 1500, CORE, dict(prep_every=300, recipe_every=500)
    CKPT = "results_v3_13_quick.pkl"
elif MODE == "acceptance":
    # The three arms the claim rests on: the no-record baseline, the record, and the record with
    # a label meaning that outlives what it names.  `plastic + noise` and `fixed + record` are the
    # controls and join at the grid stage.
    SEEDS, PHASE_STEPS, ARMS, OVERRIDES = [0, 1], 8000, CORE, {}
elif MODE == "grid":
    SEEDS, PHASE_STEPS, ARMS, OVERRIDES = [0, 1, 2], 8000, ALL, {}
else:
    raise SystemExit(f"MODE must be quick / acceptance / grid, not {MODE!r}")

variants = {k: A.VARIANTS[k] for k in ARMS}
existing = A.load(CKPT)
if existing:
    A.checkpoint_audit(existing)
    print()

variants = {k: A.VARIANTS[k] for k in ARMS}
results = A.run_experiment(seeds=SEEDS, phase_steps=PHASE_STEPS, variants=variants,
                           results=existing, save_path=CKPT, refresh=REFRESH, **OVERRIDES)
print("\nin the checkpoint:", {k: sorted(r["cfg"]["seed"] for r in v) for k, v in results.items()})
'''

CODE_SUMMARY = 'A.checkpoint_audit(results)\nA.v313_precheck(results)\n'
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
           "`MODE` picks the stage. All stages share **one checkpoint** and each **skips any "
           "(arm, seed) already in it**, so `\"grid\"` continues from the acceptance checkpoint "
           "rather than redoing it.\n\n"
           "| stage | arms | seeds | runs | est. wall clock |\n|---|---|---|---|---|\n"
           "| `quick` | 3 core | 0 | 3 | ~5 min |\n"
           "| `acceptance` | fixed, scrambled, plastic (W2) | 0\u20131 | 6 | ~60\u201375 min |\n"
           "| `grid` | + random policy, ceiling | 0\u20132 | 9 more | **~75\u201390 min** |\n\n"
           "### Read the checkpoint audit before you read anything else\n\n"
           "The cell prints an audit of the checkpoint it loads. Runs written **before** a field "
           "existed cannot be re-analysed for the reads that depend on it \u2014 those counters "
           "are accumulated inside the sim, not derived from the log \u2014 and `final_mapping` "
           "is not reconstructable offline at all, because `World` shares its rng with the "
           "agents, so the mapping draw sequence depends on every action-noise draw in the "
           "run.\n\n"
           "The affected reads are **row 3b (the attribution line)**, **first-preparation hit "
           "late-in-era**, **the survivor-conditioned since-remap curve**, and **the survivor "
           "halves 1\u20132 vs 6\u201310** (older runs recorded the 1\u20135 split under the "
           "same name). They print `nan` rather than a wrong number.\n\n"
           "To recover them, put the affected pairs in `REFRESH` \u2014 for the v3.11 "
           "acceptance checkpoint that is the three core arms at seeds 0\u20131, six runs, "
           "roughly an extra 60\u201375 min. Everything else (founder-free hit rates, rig check "
           "2(a) on whole-phase safe rate, the abstention rule, the per-era A+B sums, "
           "`prep_gain innate`) reads correctly off the existing checkpoint without a "
           "refresh.\n\n"
           "The cell checkpoints after **every run**, so a dropped session costs one run; just "
           "re-run it to resume. **Seeds 3\u20134 are held in reserve** \u2014 set "
           "`SEEDS = [3, 4]` after the grid and re-run; the seed criterion adapts "
           "(`min(4, n_seeds)`)."),
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
        "colab": {"name": "autopoiesis_v3_13_record.ipynb", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

# DELIVERY GATE.  Every code cell must compile before anything is written: a notebook that
# would not run is never produced.  See nbcheck.py for the defect this exists to stop.
write_checked(nb, HERE / "autopoiesis_v3_13_record.ipynb")
