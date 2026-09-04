# v3.9 world spec

*Written before any code. Nothing below is implemented yet. Three items marked **DECISION** need
your agreement because they depart from the literal text of the build plan.*

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

| action | food cell | item cell, empty-handed, no tool | station, carrying item | nut cell, carrying tool | otherwise |
|---|---|---|---|---|---|
| **eat** | eat it. safe → energy **+`food_value`**, **m = +1**; poison → energy **−`poison_value`**, **m = −1** | no-op, `move_cost` | no-op, `move_cost` | no-op, `move_cost` | no-op, `move_cost` |
| **interact** | no-op, `move_cost` | pick up, **−`pickup_cost`**, **m = 0** | attempt. correct → tool, **silent, m = 0**; wrong → item lost, **−`fail_cost`**, **m = −1 iff `fail_cost` > 0** | crack, **+`nut_value`**, **m = +1**, tool breaks w.p. `tool_break` | no-op, `move_cost` |

`interact` keeps v3.8's priority when a cell affords more than one: **attempt → crack → pickup**.
A successful action pays its own cost only; a **no-op costs `move_cost`**, so there is no free
waiting action any more — a behavioural change from v3.8, where "stay" on an empty cell was free.

In **phase 1** there are no items, stations or nuts, so `interact` is a permanent no-op and the
live action set is effectively five.

## Modulator events — the complete list

| event | m |
|---|---|
| eat safe food | **+1** |
| eat poison | **−1** |
| crack a nut | **+1** |
| wrong attempt, and only if `fail_cost` > 0 | **−1** |
| everything else | **0** |

"Everything else" is: pickup (`pickup_cost`), carrying (`carry_cost`/step), moving (`move_cost`),
any no-op (`move_cost`), base metabolism, repair, **and a successful attempt**. The docstring in
`sim.py` currently says "the agent's own energy change", which is not what the code does; it will
be corrected to this table (audit D, no behaviour change).

**Consequence, stated starkly:** *there is never a positive immediate signal for a correct
attempt.* With `fail_cost = 0` a wrong attempt is silent too, so the recipe has no immediate signal
at all. With `fail_cost = 0.05` the only immediate signal is negative. **The learner-oracle arm is
therefore an *elimination* oracle, not a two-sided one** — it can learn "these five pairs hurt",
not "this one pays".

> **DECISION 1.** Build-plan rule 3 asks the oracle to give "an immediate, unambiguous signal for
> exactly the thing being tested". A one-sided negative signal is immediate and unambiguous but
> asymmetric. **My recommendation: keep it as specified** — adding a positive event for a
> successful attempt would be a new modulator event and a second change in a rig-fix build. But if
> row 3 comes back null, the asymmetry is a live explanation, and the natural follow-up is a
> two-sided oracle (`tool_value` > 0 giving m = +1 on a correct attempt) as v3.9b.

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

> **DECISION 2.** The build plan says start from v2.9b's `items_per_step 1.5`, `nuts_per_patch 0.3`,
> `stations_per_type 20`. **Measured, those miss two of the three targets** — and the plan also says
> densities go to a target, not a feel. The targets govern. Proposed values solve to them:

| | v3.8 (saturated) | v2.9b as written | **proposed** | target |
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

> **DECISION 3.** If attempts/life comes back too low, how it may be fixed: the cover figures are
> **ceilings**, so density may be raised toward them (items to 8%, nuts to 8%, stations to 3%) and
> no further, judged against `random policy` and `fixed` only — never against a plastic arm
> (rule 9). If attempts/life is still ~1 at the ceiling, that is a finding about the world and gets
> reported, not tuned past.

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

> `nut_value` and `tool_break` revert to **v2.9b's** 1.0 / 0.25 rather than v3.6's tuned 1.3 / 0.4,
> for the same reason the densities do: v3.6's tuning was for a scaffolded, saturated world. Say if
> you would rather keep the tuned pair.

## Arms — seven, three seeds

`random policy` · `fixed` · `scrambled` · `plastic (W2)` · `fixed + fail cost` ·
`plastic (W2) + fail cost` · `fixed + B (ceiling)`

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

## What this spec does not change

The learning rule and its genes; the probes, except that `food_pref` now reads the `eat` logit and
`pair_pref` the `interact` logit; the staging mechanism; the metabolism; the observation layout;
`goal_channel`; event-weighted aggregation; the seed criterion.
