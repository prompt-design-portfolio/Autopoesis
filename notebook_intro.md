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

## The world was tuned before the run — nine passes, never on a plastic condition's hit rate

v2.9b's settings gave **~1 tool attempt per agent lifetime** and a hand-wired ceiling at chance —
a null by construction. Every tuning decision was judged against `fixed`, the hand-wired ceiling,
or the **positive control** (the v3.1 food effect), never against a plastic condition's recipe hit.

| change | why | effect |
|---|---|---|
| `items_per_step` 1.5 → 8, `stations_per_type` 20 → 60 | an agent must meet the conjunction more than once | attempts/life 1.0 → **4.4** |
| `nuts_uniform` 0 → 8 | at a 57–190 step bridge the answer is trace arithmetic (λ²^gap ≈ 0.03), not conjunctive learning | bridge → **~8 steps**, so λ2 *decides* |
| `repro_threshold` 3.0 → 4.5 (cost 2.25, max_energy 8) | at a hard population cap births are a queue, not fecundity | population **379–467 of 600**, no injections |
| `spawn_per_patch` 1.0 → 3.0 | nuts were 65% of energy income, so most modulator events were an uninformative `+1` | nut share → **0.40** |
| `innate_scale` 0.1 applied to the **scaffold's 10 units only** | v2.9b scaled the whole network by 0.1. Right there — its discrimination came from the hand-wired `B` path — but it leaves H2 **no basis to read**: free features sit at ~0.06 while the nav units saturate at 1.0. It killed the v3.1 food effect outright. Applying no scaling instead killed navigation and the population. | positive control alive |

`fail_cost` stays **0** (pure delayed credit). v2's STAKES value of 0.3 was tested and collapses
every population that does not already know the recipe — 5 of 6 attempts fail — leaving only the
hand-wired ceiling standing. It changes *which agents survive*, not just what they learn.

### What the pre-run checks did not clear — say it now, not after

Two acceptance criteria are **not** met, and both are pre-registered as table rows rather than
discovered afterwards:

- **`eta2` is selected DOWN** in every plastic run (0.036 vs `fixed`'s dead-gene drift 0.111).
  This is the *opposite* of v3.1, where selection retained informative plasticity (established
  finding #10). Candidate mechanism: the recipe task supplies a **positive-only** modulator —
  `+1` at a nut, nothing at a failed attempt — that is frequent and whose credit lands on
  whatever the agent was doing, mostly navigation. A frequent sign-positive modulator is
  uninformative, so plasticity is a net cost.
- **The food advantage does not reach the fitness level**: `probe_adv` (food) is 0.125 — positive,
  so the learner *does* change its own policy within life — but `safe_rate` is +0.003 against
  `fixed`, where v3.1 got +0.08. The innate approach instinct commits the agent to interacting
  with whatever it stands on, which caps how much a changed policy can matter.

So the positive control is alive **at the within-agent probe, not at the fitness level**, and the
table below uses the probe for it. If `eta2` goes to drift-floor in the real run, the plastic
conditions are effectively `fixed` and a recipe null is attributable to *plasticity having been
selected off*, which is a different (and reportable) finding from *the rule failing to bridge*.

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
| **10b** | `eta2` in the plastic conditions is inside `fixed`'s dead-gene drift range in ≥ 4/5 | **plasticity was selected off before the question was reached.** Then rows 3–6 are not about whether the rule can bridge station → nut; they are about a modulator stream in which plasticity does not pay. The reportable finding is the reversal of v3.1's finding #10, and the rule-form change to spend is the modulator one (timing / per-synapse coefficients), not the trace. | `eta2` per seed against `fixed`'s and `fixed + B`'s drift, measured in this world; `h_norm`; `nut_share` |
| **10c** | `probe_adv` (food) ≤ 0 in ≥ 2/5 in `plastic (W2)` | the within-agent positive control has failed too, so row 7 applies and nothing about the recipe can be read | `probe_adv` (food) per seed |
| **11** | `plastic (both)` − `plastic (W2)` ≥ +0.03 in 4/5 **and** `eta1` above `fixed`'s drift in 4/5 | W1 plasticity builds the conjunction; the first world where `eta1` is selected up. If `both` ≤ `W2` and `eta1` sits in drift range → as in every previous world, no new information, not rerun. | `eta1` vs `fixed`'s dead-gene drift, measured in this world |

**Not claimed either way by this run:** anything about culture, records or symbols (v3.7); anything
about `both`/`eta1` beyond row 11; any comparison of absolute safe rates against v3.1's numbers
(this world hands agents an approach-food instinct that v3.1 did not have, so only the
within-notebook `plastic` − `fixed` contrast is meaningful).
