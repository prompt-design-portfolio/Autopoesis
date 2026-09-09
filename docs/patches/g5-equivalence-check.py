"""G5 §7.1's equivalence check, on a patched COPY of the engine.

Neither half needs a model. Run in the scratchpad so the real tree stays clean while the seed
campaigns are in flight -- the dirty-tree guard exists because editing the engine mid-campaign
killed a run once, and this is how the check gets done anyway.
"""
import importlib.util, sys, numpy as np

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

BASE = "/home/user/Autopoesis/sim_v3_13.py"
PATCHED = "/tmp/claude-0/-home-user-Autopoesis/c92f5ce7-2509-5a58-a852-a9aaabd61779/scratchpad/patchtest/sim_v3_13.py"
base, patched = load(BASE, "sim_base"), load(PATCHED, "sim_patched")

WORLD = dict(flip_every=300, eta_init=0.2, hidden=24, spawn_per_patch=6.0, food_value=0.7,
             poison_value=0.5, repro_threshold=3.0, repro_cost=1.5, max_energy=5.0, max_pop=1000,
             init_pop=300, prep_value=1.0, prep_fail=0.25, prep_every=700, scaffold_food=False,
             scaffold_chain=False, goal_channel=False)
STEPS = 400

def run(mod, **kw):
    return mod.run(mod.Config(seed=0, **WORLD), verbose=False,
                   phases=[dict(n_steps=STEPS, chain=True)], **kw)

def compare(a, b, label):
    la, lb = a["log"], b["log"]
    if len(la) != len(lb):
        print(f"  {label}: FAIL -- {len(la)} rows vs {len(lb)}"); return False
    shared = sorted(set(la[0]) & set(lb[0]))
    bad = []
    for i, (ra, rb) in enumerate(zip(la, lb)):
        for k in shared:
            x, y = ra[k], rb[k]
            same = (np.allclose(x, y, rtol=0, atol=0) if isinstance(x, (list, tuple, np.ndarray))
                    else (x == y or (isinstance(x, float) and isinstance(y, float)
                                     and np.isnan(x) and np.isnan(y))))
            if not same:
                bad.append((i, k, x, y))
    print(f"  {label}: {len(la)} rows, {len(shared)} shared fields, {len(bad)} differing"
          f" -- {'PASS' if not bad else 'FAIL'}")
    for i, k, x, y in bad[:3]:
        print(f"      row {i} {k}: {x!r} vs {y!r}")
    return not bad

print("G5 §7.1 EQUIVALENCE CHECK (no model needed)\n")
ref = run(base)
print("1. external_policy=None reproduces the unpatched engine")
ok1 = compare(ref, run(patched), "None-policy")

print("\n2. a policy returning the network's own choice reproduces it too")
calls = {"n": 0, "overrode": 0}
def mirror(lineage, obs, chain_on, logits):
    calls["n"] += 1
    a = int(np.argmax(logits))
    if not chain_on and a >= patched.PREP0:
        return None
    calls["overrode"] += 1
    return a
ok2 = compare(ref, run(patched, external_policy=mirror), "mirror-policy")
print(f"     policy called {calls['n']} times, returned an action {calls['overrode']} times")

print("\n3. an illegal action is refused, not clipped")
try:
    run(patched, external_policy=lambda l, o, c, g: 999)
    print("  FAIL -- 999 was accepted")
    ok3 = False
except ValueError as e:
    print(f"  PASS -- {str(e)[:100]}")
    ok3 = True

print(f"\n{'ALL PASS' if (ok1 and ok2 and ok3) else 'FAILURES ABOVE'}")
