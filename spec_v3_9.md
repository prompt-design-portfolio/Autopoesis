# v3.9 world spec

*Agreed 4 September 2026, **amended 2** after the first pre-checks. The chain is shortened to what
v3.9 tests; the oracle arms are gone because the two-sided signal is now part of the world; densities
go to a readability target. This is the spec the code implements.*

> **Amendment 2's readability criterion is NOT met, and is not reachable by the levers it allows.**
> `random policy` attempts/life is 0.96 at the agreed band, 2.16 at **three times** the station band
> (23% cover), and *falls* as `tool_value` rises (0.96 → 0.85 at 8.0). It is pinned by the null arm
> living at the injection floor, so it measures the floor rather than the world's affordances —
> the same class of defect as the row 2(c) confound. Reported, not tuned past.

**What this build is.** v3.8 rerun on a rig that can show learning if it is there. Rig fixes A, B,
D, F from the audit, each with its own acceptance check. No new mechanism. `goal_channel` stays on;
it comes off in v3.10 as its own change.

---

## Cells and co-occupancy

Six cell contents, held in four independent arrays: **food A / food B** (one array pair; a cell
carries at most one food type, enforced at spawn), **item** (type 0/1/2, one per cell), **nut**
(boolean), **station** (type 0/1, placed once at world creation, permanent).

**Co-occupancy is allowed and is not prevented.** Food, item, nut and station are independent, so a
cell may carry any combination. This is deliberate: forbidding it would be a second change. Instead
the *rate* of collision is driven down by the density targets and is **measured** — the fraction of
food cells that also carry an item is printed and asserted ≤ 10%. Under independent placement that
fraction equals item cover, so the ≤ 8% item target implies it.

Agents do not block each other; occupancy is an observation channel only.

## Actions — six outputs (was five)

`0–3` move N/S/E/W · `4` **eat** · `5` **interact**

| action | food cell | item cell, empty-handed | station, carrying item | ~~nut~~ | otherwise |
|---|---|---|---|---|---|
| **eat** | eat it. safe → **+`food_value`**, **m = +1**; poison → **−`poison_value`**, **m = −1** | no-op | no-op | — | no-op |
| **interact** | no-op | pick up, **−`pickup_cost`**, **m = 0** | attempt. correct → **+`tool_value` = 1.5**, **m = +1**, item consumed; wrong → item lost, **−`fail_cost` = 0.05**, **m = −1** | — | no-op |

`interact` priority when a cell affords more than one: **attempt → pickup**. **There are no nuts
and no tool state in v3.9**: a correct attempt pays immediately and consumes the item, so nothing is
carried afterwards. The station→nut bridge returns in v3.11 as one change.
A successful action pays its own cost only; a **no-op costs `move_cost`**, so there is no free
waiting action any more — a behavioural change from v3.8, where "stay" on an empty cell was free.

In **phase 1** `interact`'s logit is forced to −∞, so phase 1 is **exactly five actions** and
v3.1's gate applies unchanged. The sixth action goes live at the switch, alongside the 39 inputs.
The null is masked too, so its per-action share is **1/5 in phase 1 and 1/6 in phase 2** — the
conditional nulls differ by phase, and the summary prints the measured null as well as the analytic.

## Modulator events — the complete list

| event | m |
|---|---|
| eat safe food | **+1** |
| eat poison | **−1** |
| correct attempt (**+`tool_value`**, item consumed) | **+1** |
| wrong attempt (**−`fail_cost`**, item lost) | **−1** |
| everything else | **0** |

"Everything else" is: pickup (`pickup_cost`), carrying (`carry_cost`/step), moving (`move_cost`),
any no-op (`move_cost`), base metabolism, repair, **and a successful attempt**. The docstring in
`sim.py` currently says "the agent's own energy change", which is not what the code does; it will
be corrected to this table (audit D, no behaviour change).

**The signal is immediate and two-sided, in the world, in every arm.** There is no oracle fixture
any more — `scrambled` is the control for "a modulator at stations changes behaviour". Delayed
credit is no longer tested in v3.9; it returns in v3.11 with the bridge.

## Observation — 60 inputs, unchanged

11 channels × (4 directional sums + 1 here) = 55, then energy (1), inventory one-hot (3), tool (1).

Channels: `0` food A · `1` food B · `2` occupancy · `3` `food_any` **(dead — it existed only to
drive the removed instinct)** · `4,5,6` item types · `7,8` station types · `9` nuts · `10` `goal`.

`goal` is **on** in v3.9: it points at *any* item when empty-handed, *any* station when carrying
one, nuts when carrying a tool. It never says which. Off in v3.10 as its own change.

Phase 1 live inputs: 16. Phase 2: 55. **39 go from zero to live at the boundary** (measured).

Hidden 24, no instinct wired (`scaffold_food=False`, `scaffold_chain=False`); `nav_dir` / `nav_here`
remain in the genome, reach nothing, and serve as the dead-gene drift scale.

## Standing densities — measured with no agents, after 1000 steps

**SUPERSEDED by amendment 2 — densities go to a READABILITY target, not a co-occupancy ceiling.**
Co-occupancy only mattered while one action did both jobs, and fix A solved that. Item cover
**20–25%** (`items_per_step` 3.0 → 22.4%), station cover **6–8%** (`stations_per_type` 85 → 7.1%),
food-with-item printed but unconstrained (21.2%). The earlier ceiling reasoning is kept below for
the record.

**Superseded:** v2.9b's stated values miss two of three targets;
the targets govern, so these solve to them. Cover is asserted in the setup cell and
food-cells-with-item is printed.

| | v3.8 (saturated) | v2.9b as written | **agreed** | target |
|---|---|---|---|---|
| `items_per_step` | 8.0 | 1.5 | **0.8** | — |
| `nuts_per_patch` / `nuts_uniform` | 0.3 / 8.0 | 0.3 / 0 | **0.12 / 0** | — |
| `stations_per_type` | 60 | 20 | **30** | — |
| item cover | 40.8% | 12.5% ✗ | **5.5–7.2%** ✓ | ≤ 8% |
| nut cover | 46.9% | 14.4% ✗ | **6.1–7.6%** ✓ | ≤ 8% |
| station cover | 5.0% | 1.7% ✓ | **2.5–2.6%** ✓ | ≤ 3% |
| food cells carrying an item | 41.9% | 13.3% ✗ | **6.2–7.3%** ✓ | ≤ 10% |
| food cover (v3.1, not a target) | 45.3% | 43.1% | 37.5–48.2% | — |

Ranges are over four seeds. Cover is printed and asserted in the setup cell.

`nuts_uniform` (my v3.6 addition) goes to **0** — nuts return to the food patches only, as in
v2.9b. Geometry check: nearest nut from a station is **3.3 steps** mean, 10 max, so the
station→nut bridge stays short without it.

**The main risk in this build, named now.** At v2.9b-scale densities the v3.6 smoke test gave
**~1 tool attempt per agent lifetime**, which is a null by construction for any within-life
mechanism. That world had an instinct, a different metabolism and 20 stations; this one has 30
stations and the network controls approach, so it is not the same measurement — but **attempts per
life is the number that decides whether row 3 can be read at all**, and the pre-checks measure it.

**RESOLVED — accepted.** Densities may rise **to** the ceilings (items 8%, nuts 8%, stations 3%)
and no further, judged against `random policy` and `fixed` only, never a plastic arm (rule 9). If
attempts/life is still ~1 at the ceilings, that is reported as a property of the world, not tuned
past. **attempts/life is printed per arm in the pre-checks.**

## Phases — one population, never rebuilt

**Phase 1:** 8000 steps, food only, safe type flips every 300. **Phase 2:** 8000 steps, items,
stations and nuts switch on for the same population. Agents, `H`, eligibility traces and the world
carry across untouched; only `chain` flips. Recipe eras are measured from the switch:
`recipe_every = 2000`, so phase 2 holds four whole eras.

## Costs and values

v3.1 metabolism throughout: `base_cost` 0.006 · `move_cost` 0.002 · `carry_cost` 0.002 ·
`start_energy` 1.5 · `founder_energy` 3.0 · `repro_threshold` 3.0 · `repro_cost` 1.5 ·
`max_energy` 5.0 · `max_pop` 400 · `init_pop` 300 · `min_pop` 40 · food +0.7 / poison −0.5 ·
`flip_every` 300 · `eta_init` 0.2 · hidden 24.

Chain: `pickup_cost` 0.02 · **`nut_value` 1.0** · **`tool_break` 0.25** · `fail_cost` 0 (0.05 in
the two fail-cost arms) · `recipe_every` 2000.

`nut_value` **1.0** and `tool_break` **0.25** — v2.9b's, accepted; v3.6's tuned 1.3 / 0.4 were for a
scaffolded, saturated world.

## Arms — five, three seeds

**Five arms:** `random policy` · `fixed` · `scrambled` · `plastic (W2)` · `fixed + B (ceiling)`

The oracle arms are removed: the two-sided signal is the world now, in every arm. `scrambled` is
the control for "a modulator at stations changes behaviour".

**`random policy`** draws uniformly over the six actions. No plasticity, no learning; a behavioural
null only. **It is expected to sit near the population floor with injections, and that must not
exclude it under row 0** — it supplies the null for contact *rates* per 1k, which stay measurable
at the floor. Row 0's exclusion applies to the six outcome arms.

**`fixed + B (ceiling)`** is weaker than v2's: its steering route (`attract` on the `goal` channel)
is now only an observation the network may ignore, since nothing is wired. Its live route is the
veto. It stays a reference level, not a matched comparison.

## Metrics added in this build

`declined` per 1k (steps at a station carrying an item where `interact` was **not** chosen — the
policy-level refusal that audit E says the hit rate needs) · `eat` vs `interact` choice rate on
food cells and on item cells · contact rates (pickups/1k, attempts/1k, cracks/1k) **against the
`random policy` arm**, not against each other · everything v3.8 logged · the v3.8 transition table.

## Acceptance checks tied to each fix

| fix | check |
|---|---|
| **A** split action | in phase 1, `eat` is chosen on food cells at a rate above `random policy`'s 1/6 by the end of the phase |
| **B** densities | cover asserted in setup: items ≤ 8%, nuts ≤ 8%, stations ≤ 3%; food cells carrying an item printed, ≤ 10% |
| **D** docstring | no behaviour change; the event table above is the docstring |
| **F** random arm | contact rates for every arm reported as a ratio to `random policy`'s |

Then, before any grid: pre-checks at 1 seed × 3000-step phases, **every arm**, reporting the
transition table and per-arm population. Then acceptance at 2 seeds — (i) phase-1 gate; (ii) food
learning survives phase 2 (safe ≥ 0.58, `probe_adv` food ≥ 1.0 **in phase 2**); (iii) oracle
survives (`plastic + fail cost` pop ≥ 80, no injections); (iv) cover targets met. **If (ii) or (iii)
fails I stop and report rather than tune past it.**

## Two additions, agreed

**Addition 1 — world-semantics unit test.** Every (action, cell content, inventory state) row of
the table above is constructed cell by cell, resolved with one call, and its energy delta, its `m`
and its event asserted against the table — including the two priority rows and the silent-by-default
attempt outcomes. It runs in the setup cell. It is the test that would have caught the shared
interact action. To make it test the *real* path rather than a re-implementation, the action × cell
table is extracted into a single function, `resolve_action`, which both `run()` and the test call.

**Addition 2 — the oracle row pre-registers abstention.** Row 3 reads `declined` and attempts/1k
against `fixed + oracle` **before** hit rate. Attempts below 0.8× with `declined` rising is
abstention, not knowledge. **Prediction, stated now: with a two-sided signal abstention should not
occur.** If it does, that is a finding about the rule under mixed signals, not a null on the
conjunction.

## Amendment proposed after the pre-checks (needs your agreement)

**Row 2(c)'s contact test as specified is confounded.** Comparing pickups/1k and attempts/1k to
`random policy` compares *action budgets*, not approach: an arm that learns to `eat` necessarily
spends fewer of its six actions on `interact` than a uniform walker, so it can fall below the null
while approaching better. The unconfounded measure is **conditional** — `P(interact | standing on
an item cell)` and `P(eat | standing on a food cell)` — whose null is exactly 1/6 by construction.
Both are now logged and both are printed; the proposal is that row 2(c) reads the conditional.

## What this spec does not change

The learning rule and its genes; the probes, except that `food_pref` now reads the `eat` logit and
`pair_pref` the `interact` logit; the staging mechanism; the metabolism; the observation layout;
`goal_channel`; event-weighted aggregation; the seed criterion.
