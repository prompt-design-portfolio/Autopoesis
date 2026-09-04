# v3.6: can the grown learner acquire a conjunctive, delayed-credit fact within life?

**Where this sits.** The flip world is exhausted as a culture testbed for a structural reason
(handoff §4): the record carries one bit, and the agent's own next meal teaches the same bit for
the price of one energy loss. A record is worth interpreting only when what it carries is
expensive to discover alone. So the world moves to the delayed-credit recipe task — v2.9b's tool
world — and this notebook runs the **learner alone** on it, with no record and no culture, before
v3.7 puts a symbol record on top.

**The question.** `(item × station) → tool` is a conjunction over 6 pairs, it changes every 2000
steps, and the modulator `m` arrives **at the nut**, some steps after the station attempt that
earned it. A wrong attempt produces no modulator at all (`fail_cost = 0`) — only the loss of the
item. Can output-layer plasticity, driven by nothing but the agent's own metabolism, find it?

**What is hand-wired and what is not.** An innate forager instinct (v2.9b's `innate_nav`) points
the agent at *any* item when empty-handed, at *any* station when carrying one, at nuts when
carrying a tool, and at food of *either* type. Navigation is not the question. **Which** item,
**which** station and **which** food are not wired anywhere: the observation carries the inventory
one-hot and the per-type channels, and a conjunctive policy has to be built out of them, either by
mutation (`fixed`) or within life (`H`). The scaffold is identical in every condition.

---

## Conditions (7 × 5 seeds, 10000 steps)

| condition | what it is for |
|---|---|
| `fixed` | baseline and gate: what the genome alone does. Also the dead-gene drift reference for `eta1`, `eta2`, `lam2` *in this world*. |
| `scrambled` | **the control that carries the claim.** Same plasticity, same H magnitudes, same power to override the innate scaffold; random-sign modulator, no information. |
| `plastic (W2)` | the result condition. |
| `plastic (both)` | W1 plasticity could build the conjunction features that H2 reads. |
| `fixed + B (ceiling)` | v2's hand-wired private memory: exact per-pair credit. A reference level, **not** a matched comparison — it acts through navigation preference and a veto, not through the network's output. |
| `fixed, no flip` / `plastic (W2), no flip` | interference pair. With the food reversal removed the recipe is the only moving target. The fixed arm absorbs "a no-flip world is simply richer". |

## The world was tuned before the run, on `fixed` and the ceiling only

v2.9b's settings gave **~1 tool attempt per agent lifetime** and a hand-wired ceiling at chance —
a null by construction. Five tuning passes (seed 0, short runs) moved four quantities, each judged
against `fixed` and the hand-wired ceiling, **never against a plastic condition**:

| change | why | effect |
|---|---|---|
| `items_per_step` 1.5 → 8, `stations_per_type` 20 → 60 | an agent must meet the conjunction more than once | attempts/life 1.0 → **6.6** |
| `nuts_uniform` 0 → 8 (nuts outside the food patches) | at a 57–190 step bridge the answer is trace arithmetic (λ²^gap ≈ 0.03), not conjunctive learning | station→nut bridge → **7.5 steps**, λ²^gap ≈ 0.12, so λ2 *decides* |
| `repro_threshold` 3.0 → 4.5 (cost 2.25, max_energy 8) | at a hard population cap births are a queue, not differential fecundity | population **225–411 of 600**, no injections |

Acceptance, met at seed 0 / 2500 steps: `fixed` **0.172** (chance 0.167), ceiling **0.230**
(v2: 0.23–0.36), attempts/life 6.6, bridge_first 7.5, populations off the cap, zero injections.

**Alternative explanations this leaves open, before any result:** (a) the innate scaffold is strong
(nav weights 4.0, innate W scaled ×0.1) while H is clipped at ±2 per synapse, so plasticity can
override the scaffold where mutation moves slowly — `scrambled` is what separates that from
learning; (b) the ceiling acts through a different channel than the learner, so it bounds *what the
world pays for*, not *what this rule could reach*; (c) generation ≈ 260 steps, so 10000 steps is
~38 generations, fewer than v3's ~40.

---

## Stopping rule (agreed before the run)

Second-half aggregates, **event-weighted** (Σcorrect / Σattempts, not a mean of per-window ratios —
a mean-of-ratios artifact produced the v2 "ratchet" claim that was retracted). Margin **0.03**,
seed criterion **4/5**. Chance = 0.167.

Rows are checked in order. Every row names the condition that attributes it — the v3.4 lesson.

| # | outcome | reading | what attributes it |
|---|---|---|---|
| **0** | any condition with pop < 80, injections > 0, or attempts/1k < 5 | that condition is uninterpretable; name it and exclude it | pop, injections, attempts/1k, per seed |
| **1** | **gate:** `fixed` ≥ 0.25 in 4/5 | the genome tracks the recipe at `recipe_every=2000`. The world does not isolate within-life learning. **Stop, retune `recipe_every`, no claim.** | `fixed`'s own `pair_gain_innate` (> 0 = an innate conjunctive preference) and its era late−early |
| **2** | **ceiling:** `fixed + B` − `fixed` < 0.03 in ≥ 2/5 | exact pair credit does not pay here. **No condition can be read, a null included.** Retune, no claim either way. | attempts/life, nuts/1k, bridge_first |
| **3** | `plastic (W2)` − `fixed` ≥ +0.03 in 4/5 **and** `plastic (W2)` − `scrambled` ≥ +0.03 in 4/5 **and** `probe_adv` (recipe) > 0 in 4/5 **and** attempts/1k not < 0.8 × fixed | **the grown learner acquires a conjunctive delayed-credit fact within life.** Corroborating, not required: the attempt-in-life curve rises, `hit_old` > `hit_young`, λ2 above `fixed`'s drift, era late−early above `fixed`'s. | `scrambled` (information, not override); `probe_adv` is within-agent, so not selection or composition; attempts/1k rules out abstention |
| **4** | `plastic (W2)` > `fixed` by ≥ 0.03 in 4/5 **but** ≈ `scrambled` | the gain is having a plastic override of the innate scaffold at all, not the modulator's information. **Not learning.** | `scrambled` — the whole point of that arm |
| **5** | hit rises but attempts/1k < 0.8 × fixed **and** nuts/1k not up | bought by attempting less, not by knowing more. Abstention, not knowledge. | attempts/1k and nuts/1k together |
| **6** | no pair differs by 0.03 in 4/5, row 2 passed, **and** the positive control holds (`safe_rate` plastic − fixed ≥ 0.03 in 4/5, or `probe_adv` (food) > 0 in 4/5) | **null, and attributable:** the local rule with a metabolic modulator does not bridge station → nut, in a world where exact pair credit *does* pay and where the same learner *does* solve the 300-step food reversal. **Spends the one agreed rule-form change**, same test. Which one is indicated by λ2 and λ²^gap (see row 8). | `fixed + B` (the world pays) + the food probe (the learner works) |
| **7** | as row 6 **but** the positive control also fails | the learner is broken *in this world* — observation size, hidden count, or the nav scaffold swamping the readout. **Fix that first. No rule-form change, and no claim about delayed credit.** | `probe_adv` (food) and `safe_rate` vs `fixed` |
| **8** | on a null: λ²^gap < 0.02 **and** λ2 not above `fixed`'s drift in 4/5 | the null is about trace *length*, not about the conjunction. The rule-form change to spend is the longer trace specifically, and it must be tested where λ2 matters. | `bridge_first`, λ2 per seed vs `fixed`'s dead-gene drift |
| **9** | (`plastic no flip` − `fixed no flip`) − (`plastic` − `fixed`) ≥ +0.03 in 4/5 | the food reversal crowds the recipe out: one H2 and one trace cannot serve a 300-step reversal and a 2000-step conjunction at once. The remedy is separating the tasks, **not** changing the rule. | the `fixed, no flip` arm, which absorbs "a no-flip world is richer" |
| **10** | `plastic (W2)` < `fixed` by ≥ 0.03 in 4/5 | plasticity is a net cost here. If `scrambled` is equally below → the cost is the machinery perturbing a scaffolded policy (the v3.5 non-stationary-input mechanism again). If `plastic` < `scrambled` → the informative modulator is actively *misleading*: credit lands on navigation, not on the conjunction. That is a positive finding about the failure mode. | `scrambled` vs `plastic`, and `h_norm` |
| **11** | `plastic (both)` − `plastic (W2)` ≥ +0.03 in 4/5 **and** `eta1` above `fixed`'s drift in 4/5 | W1 plasticity builds the conjunction; the first world where `eta1` is selected up. If `both` ≤ `W2` and `eta1` sits in drift range → as in every previous world, no new information, not rerun. | `eta1` vs `fixed`'s dead-gene drift, measured in this world |

**Not claimed either way by this run:** anything about culture, records or symbols (v3.7); anything
about `both`/`eta1` beyond row 11; any comparison of absolute safe rates against v3.1's numbers
(this world hands agents an approach-food instinct that v3.1 did not have, so only the
within-notebook `plastic` − `fixed` contrast is meaningful).
