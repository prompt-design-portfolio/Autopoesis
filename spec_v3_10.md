# v3.10 world spec — the preparation world

*Written before any code, per build-plan rule 1. Nothing below is implemented. Six items marked
**DECISION** need agreement; each carries my recommendation.*

**Why this world.** v3.9 closed the recipe world: a chain whose payoff arrives only at the end is
sparse, and a rare opportunity cannot generate the selection differential that would build the
approach behaviour making it less rare. The fix is not a bigger payoff or a shorter journey — both
were tried — but **a dense opportunity**. Here the fact to be learned sits on **every meal**, where
the agent already is. There is no approach behaviour to evolve, no carrying, no navigation: only
*which action to take on the food under my feet*.

---

## Cells and co-occupancy

Two contents only: **food A** and **food B**, one array pair, at most one type per cell (enforced at
spawn), spawning in drifting patches exactly as v3.1. **No items, no stations, no nuts.** Agents do
not block each other; occupancy is an observation channel only.

## Actions — eight, of which five are live in phase 1

`0–3` move N/S/E/W · `4` **eat** · `5,6,7` **prep_1 / prep_2 / prep_3**

In **phase 1** the three preparation logits are forced to −∞, so phase 1 is **exactly v3.1's five
actions** and its gate applies unchanged. **Only the action space changes at the switch** — no new
input channels, so the transition is one thing, not two.

| action | food A here | food B here | otherwise |
|---|---|---|---|
| **eat** | eat raw: safe → **+`food_value`**, **m = +1**; poison → **−`poison_value`**, **m = −1** | same | no-op, `move_cost` |
| **prep_k** | correct prep for A → **+`prep_value`**, **m = +1**; wrong → **−`prep_fail`**, **m = −1** | correct prep for B → **+`prep_value`**, **m = +1**; wrong → **−`prep_fail`**, **m = −1** | no-op, `move_cost` |

A preparation **consumes the food cell**, exactly as eating does.

> **DECISION 1 — does a correct preparation pay on currently-poisonous food?** As the design reads,
> yes: the prep outcome depends on food *type* and prep choice only, independent of the safe/poison
> flip. **Recommendation: keep it.** It makes the world hold two independent facts — a fast one
> (which type is safe, flipping every 300) and a slow one (which prep goes with which type, remapped
> every 2000) — and they interleave naturally rather than confounding: an agent that knows the safe
> type but not the mapping should **eat** (+0.7) rather than prep at chance (+0.17), and only once it
> knows the mapping does prep (+1.5) dominate. The risk to note: **late in a mapping era, a
> successful learner stops eating raw**, so raw-meal counts thin out and `safe_rate` becomes noisy
> in phase 2. The rig check should therefore lean on `probe_adv` (food), which needs no events, with
> `safe_rate` read as corroborating and its event count printed.

## Modulator events — the complete list

| event | m | energy |
|---|---|---|
| eat safe food | **+1** | +`food_value` |
| eat poison | **−1** | −`poison_value` |
| **correct preparation** | **+1** | +`prep_value` |
| **wrong preparation** | **−1** | −`prep_fail` |
| everything else | **0** | move / no-op `move_cost`, base metabolism, repair |

There is no silent event in this world: every action on food resolves immediately and two-sidedly.
The modulator is **not** the agent's own energy change in general; it is this table.

## The mapping

Each food type has exactly one correct preparation, so the mapping is a function
{A, B} → {prep_1, prep_2, prep_3}. Drawn at the switch and **redrawn every `prep_every = 2000`
steps**, independently of the safe/poison flip. Chance prep hit = **1/3**.

> **DECISION 2 — must the two food types map to *different* preparations?** If drawn independently,
> they collide one time in three, and a colliding mapping is solvable by learning a single
> preparation and never discriminating food type at all — which is not the task.
> **Recommendation: draw them distinct**, and redraw so that the new mapping differs from the old in
> at least one type (as `new_recipe` already does for the recipe). Chance stays 1/3 per prepared meal.

## Observation — unchanged, and unchanged across the switch

The v3.8/v3.9 layout, 60 inputs. Live: food A / food B / occupancy directional sums and here-values,
plus energy. Nothing tells the agent the mapping; nothing changes at the switch.

> **DECISION 3 — keep the 60-input layout with the chain channels permanently dead, or shrink to the
> ~21 live inputs?** **Recommendation: keep 60.** It costs nothing (dead inputs contribute exactly
> zero), it keeps the phase-1 gate directly comparable with v3.8's and v3.9's, and shrinking would
> change the network size at the same moment as everything else. The dead channels are named in the
> setup print so nobody mistakes them for live.

Hidden 24. No instinct wired: `scaffold_food=False`, `scaffold_chain=False`; `nav_dir` / `nav_here`
stay in the genome, reach nothing, and give the dead-gene drift scale.

## Densities

Food exactly as v3.1 (`spawn_per_patch` 3.0, 8 patches, radius 6, `food_rot` 0.005) — measured
standing cover ~43%. **There is no density to tune in this world**: the opportunity is every food
cell, so the readability criterion that failed three times in v3.9 is satisfied by construction.
Printed in setup for the record, not asserted as a band.

## Phases — one population, never rebuilt

**Phase 1:** 8000 steps, v3.1 unchanged, preparations masked. **Phase 2:** 8000 steps, preparations
live for the same population; agents, `H`, eligibility traces and the world all carry across
untouched. Mapping redrawn at the switch and every 2000 steps thereafter, so phase 2 holds four
whole mapping eras.

## Costs and values

v3.1 metabolism: `base_cost` 0.006 · `move_cost` 0.002 · `start_energy` 1.5 · `founder_energy` 3.0 ·
`repro_threshold` 3.0 · `repro_cost` 1.5 · `max_energy` 5.0 · `max_pop` 400 · `init_pop` 300 ·
`min_pop` 40 · food +0.7 / poison −0.5 · `flip_every` 300 · `eta_init` 0.2.

Preparation: **`prep_value` 1.5** · **`prep_fail` 0.5** · `prep_every` 2000 · K = 3.

Expected value of a preparation at chance: `1/3 × 1.5 − 2/3 × 0.5 = +0.167`, against `+0.1` for a
raw meal at a chance safe rate and `+0.7` for a known-safe one. Knowing the mapping is worth
**+1.5 per meal against +0.7** — and unlike v3.9, the opportunity arrives on **every meal**, so the
differential compounds over a lifetime instead of appearing half a time.

## Arms — five, three seeds

`random policy` · `fixed` · `scrambled` · `plastic (W2)` · `fixed + B (ceiling)`

`random policy` is uniform over the **available** actions — 5 in phase 1, 8 in phase 2 — so the
conditional null is **1/5** then **1/8** per action, and **3/8** for "any preparation". Exempt from
row 0: a random walker belongs at the population floor.

> **DECISION 4 — how does the `B` ceiling act?** In the recipe world B steered navigation and vetoed
> attempts. There is no navigation here; the only choice is which action to take on the food
> underfoot. **Recommendation: B is a hand-wired table over (food type, prep) updated with exact
> credit (`B[f,k] ← 0.7·B[f,k] + 0.3·outcome`), and when the agent stands on food and any entry for
> that type exceeds a threshold, the action is forced to `argmax_k B[f,k]`; otherwise the network
> chooses.** That is the honest analogue of v2's veto: a hand-wired policy override, not a hint. It
> remains a reference level, not a matched comparison, and the summary says so.

## Metrics

**prep hit rate**, event-weighted (correct preps / preps; chance **1/3**) · **P(prep | on food)**
against the null, and per-prep P(prep_k | on food) · **`probe_adv` (prep)**, within-agent, built
like the recipe probe: on a synthetic "food type f here" observation, the logit of the correct prep
minus the mean of the other two, with learned synapses minus innate · **hit-in-life curve** (prep hit
by prep number in an agent's life) and **since-remap curve** (by preps since the last remap) ·
`safe_rate`, raw-meal count and **`probe_adv` (food)** through phase 2 as the rig check · **prepared
meals per life** · `eta2` / `lam2` / `eta1` against `fixed` · unwired `nav_dir` / `nav_here` as drift
scale · transition table at 500-step bins across the switch · populations and injections per phase.

## Stopping rule

Event-weighted, per phase. Margin **0.03**, seeds **3/3**. A positive on any row gets seeds 3–4
before it is called.

| # | row | reading | attributing arm |
|---|---|---|---|
| **0** | uninterpretable per phase: pop < 80 or injections > 0 | exclude and name it. `random policy` exempt | — |
| **1a** | **phase-1 gate = v3.1** — safe rate `plastic` − `fixed` ≥ 0.03, `probe_adv` (food) ≥ 1.0 | **STOP ROW.** If phase 1 is not v3.1, nothing below is read. Row-0 fallback: where `fixed` phase 1 is excluded, that seed reads against v3.1's published range, conservative end 0.56 | `fixed`, or v3.1's published range |
| **1b** | **mapping gate** — `fixed` prep hit ≤ 0.45 in 3/3 (chance 0.333) | if genes track a 2000-step mapping, shorten `prep_every` toward the flip period, **judged against `fixed` only**, before anything below is read | `fixed` |
| **2** | **rig checks** — (a) food learning survives phase 2 (`probe_adv` (food) ≥ 1.0; `safe_rate` corroborating, with its raw-meal count printed — see DECISION 1); (b) the opportunity exists: **prepared meals per life ≥ 3 in `fixed`** | if either fails the rig is broken: stop, diagnose, no learner claim | `fixed`, `random policy` |
| **3** | **the conjunction.** `plastic` − `fixed` ≥ 0.03 **and** `plastic` − `scrambled` ≥ 0.03 in 3/3 on prep hit; `probe_adv` (prep) > 0; **one within-life signature** (hit-in-life curve rising, or `hit_old` > `hit_young`); **no abstention** — prepared meals not below 0.8× `fixed` | the learner acquires a remapping fact within life, on a dense opportunity | `scrambled` carries the claim, not `fixed`; `probe_adv` is within-agent; the abstention line is read first |
| **4** | gene rows: `eta2`, `lam2`, `eta1` against `fixed` and `scrambled`; unwired scaffold genes as drift scale | corroborating only | — |
| **5** | if row 3 is null with rows 1–2 clean | the first earned statement about the learner's limit, on a dense, immediate, two-sided task. Spend the one rule-form change there | — |

**Pre-registered prediction: `plastic (W2)` clears row 3.** Recorded before the run.

> **DECISION 5 — the within-life signature is harder here than it looks.** A mapping era is 2000
> steps and a generation is ~200, so most agents live inside a single era and never see a remap.
> `hit_old` > `hit_young` and the hit-in-life curve therefore measure learning *within* an era, which
> is what we want — but an agent born mid-era inherits nothing and must learn from its own
> preparations. **Recommendation: keep `prep_every` at 2000 and read the since-remap curve
> population-wide**, which is where a remap's cost and recovery show. If gate 1b fires and
> `prep_every` shortens, both curves become more informative, not less.

> **DECISION 6 — the semantics test needs the mapping in the loop.** The v3.9 test enumerated
> (action, cell content, inventory). Here the third dimension is the **mapping**.
> **Recommendation: enumerate every (action ∈ {eat, prep_1..3}, food ∈ {none, A, B}, mapping ∈ all
> distinct assignments) row** — 4 × 3 × 6 = 72 rows plus the four move/no-op rows — constructing the
> cell, calling one `resolve_action`, and asserting energy delta, `m`, event and that the food cell
> was consumed. It runs well under a second and it is the test that would catch a mapping applied to
> the wrong food type.

## Also carried over unchanged

The learning rule and its genes; the staging mechanism; event-weighted aggregation per phase; the
conditional-null convention (measured null printed beside the analytic one); the learning-rule unit
test in the setup cell; QUICK mode; Colab-only with `sim.py` and `analysis.py` uploaded alongside;
per-run pickling with resume instructions.
