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
item. A *successful* attempt is silent too: making a tool yields no energy, the nut does. Can
output-layer plasticity, driven by nothing but the agent's own metabolism, find it?

**What is hand-wired and what is not.** An innate forager instinct (v2.9b's `innate_nav`) points
the agent at *any* item when empty-handed, at *any* station when carrying one, at nuts when
carrying a tool, and at food of *either* type. Navigation is not the question. **Which** item,
**which** station and **which** food are not wired anywhere: the observation carries the inventory
one-hot and the per-type channels, and a conjunctive policy has to be built out of them, either by
mutation (`fixed`) or within life (`H`). The scaffold is identical in every condition, and since
v3.6b its strength is **heritable** (`nav_dir`, `nav_here`), so evolution rather than the author
sets the balance between instinct and override.

---

## Conditions (7 × 5 seeds, 8000 steps)

| condition | what it is for |
|---|---|
| `fixed` | baseline and gate: what the genome alone does. Also the reference for `eta1`, `eta2`, `lam2` and the scaffold genes *in this world*. |
| `scrambled` | **the control that carries the claim.** Same plasticity, same H magnitudes, same power to override the innate scaffold; random-sign modulator, no information. |
| `plastic (W2)` | the result condition. |
| `plastic (both)` | W1 plasticity could build the conjunction features that H2 reads. |
| `fixed + B (ceiling)` | v2's hand-wired private memory: exact per-pair credit. A reference level, **not** a matched comparison — it acts through navigation preference and a veto, not through the network's output. |
| `fixed + fail cost` / `plastic (W2) + fail cost` | the elimination pair (`fail_cost = 0.05`). A wrong attempt now costs energy, so `m = −1` arrives **at the station** and the conjunction becomes partly learnable by elimination — an easier question than pure delayed credit, hence a separate condition. The `fixed` arm is not optional: without it a gain in the plastic arm could be the 0.05 energy change rather than the −1 signal. |

The no-flip pair is dropped from this pass; it returns only if the main result is positive.

## Tuning log — every pass, and what it was judged against

No pass was ever judged against a plastic condition's recipe hit rate.

| pass | change | judged against | result |
|---|---|---|---|
| 1–5 | `items_per_step` 1.5→8, `stations_per_type` 20→60, `nuts_uniform` 0→8, `repro_threshold` 3.0→4.5, `spawn_per_patch`→3.0 | `fixed`, ceiling | **kept.** attempts/life 1.0→4.4; station→nut bridge 57→8 steps; population off the cap; nut share 0.65→0.40 |
| 6–9 | `innate_scale` applied to the scaffold's 10 units only (v2.9b scaled the whole network by 0.1) | positive control | **kept.** Global scaling left H2 no basis to read and killed the v3.1 food effect; no scaling killed navigation and the population |
| **A** | **scaffold strength heritable** (`nav_dir`, `nav_here` genes, founders U(2,6), σ 0.2, wired from the gene at every birth so mutation of W1/W2 cannot touch the instinct) | positive control | **kept, acceptance NOT met.** `probe_adv (food)` 0.125 → **0.393 / 0.229** (target ≥ 0.5); safe_rate +0.005 / +0.001 (target ≥ 0.03 in 2/2). `eta2` 0.036 → 0.070 / 0.061 against drift 0.083 / 0.085. The genes **drifted rather than being selected** (σ0.2 × √28 gens ≈ 1.0, and the observed spread is that size); plastic did not evolve a *weaker* instinct. |
| **B** | `free_scale` 2.0 and 3.0 — scale the free hidden units, the basis H2 reads | positive control | **rejected.** 2.0: `probe_adv (food)` 0.059 / 0.192, `eta2` 0.014 / 0.034 — worse than A. 3.0: populations 41–63 with injections. |
| **C** | `poison_value` 1.0 and 1.4 — make indiscriminate eating net-negative so selectivity is load-bearing | positive control | **rejected.** Populations collapse to the floor (41–46, injections 2–8/window, `max_gen` 3–10). Row 0 excludes them. |
| **D** | `scaffold_food = False` — take food out of the instinct, keep the chain's instinct (food approach was *my* addition; v2.9b weighted that channel by a hand-wired taste register) | positive control | **rejected.** `probe_adv (food)` 0.190 / 0.025, safe_rate +0.011 / +0.017. Populations healthy (290–371), so this is a real null, not a collapse. |
| **E** | attempts per life, for ceiling headroom: `tool_break` up (each tool cracks fewer nuts, so the chain must be repeated), and longer lives | `fixed` and ceiling | **partly kept, acceptance NOT met.** Shipped `tool_break` 0.25→0.4 with `nut_value` 1.0→1.3: gap **0.082 / 0.065** (was 0.054), attempts/life 4.3→4.9, populations 255–451. The only arm that cleared 0.10 did so by starving the baseline. |

### Pass E in full — the headroom trade-off

| arm | `fixed` | ceiling | gap (target ≥ 0.10 in 2/2) | `fixed` pop |
|---|---|---|---|---|
| `tool_break` 0.5, `nut_value` 1.0 | 0.175 / 0.172 | 0.302 / 0.289 | **0.127 / 0.117 ✓** | 77 / 72 — **row 0 excludes it** |
| longer life (`repro_threshold` 6.0) | 0.175 / 0.172 | 0.252 / 0.258 | 0.077 / 0.086 ✗ | 377 / 363 ✓ |
| `tool_break` 0.5, `nut_value` 1.6 | 0.199 / 0.176 | 0.251 / 0.249 | 0.052 / 0.073 ✗ | 323 / 332 ✓ |
| **`tool_break` 0.4, `nut_value` 1.3 (shipped)** | 0.167 / 0.187 | 0.249 / 0.252 | 0.082 / 0.065 ✗ | 255 / 319 ✓ |

The trade-off is structural, not a search failure. The ceiling is an elimination search, so its height
is set by the ceiling condition's **own** attempts per life — and B agents waste far fewer items, so
they reach a *denser* population (428–451 against `fixed`'s 255–319), which shortens their lives and
cuts their attempts back to ~3.7. Every arm that thins the world enough to lengthen those lives
starves `fixed` first. Raising nut income restores `fixed` and re-crowds the ceiling. The one arm
that broke 0.10 bought it by pushing `fixed` to the exclusion threshold, which makes the baseline
uninterpretable and the gap meaningless.

### Issue 1 (ceiling headroom) is NOT fixed either

Best viable gap is **0.082 / 0.065** against a target of ≥ 0.10 in 2/2. With a 0.03 margin the test
has more power than it did (headroom 0.054 → 0.074 mean) but still not the power you asked for: a
learner would have to capture ~40% of everything exact pair credit achieves before it cleared the
margin.

### Issue 2 (positive control) is NOT fixed — say it plainly

Four passes. Best is A, the change actually prescribed: `probe_adv (food)` **0.393 / 0.229** against
an acceptance of ≥ 0.5, and safe_rate **+0.005 / +0.001** against ≥ 0.03 in 2/2. v3.1 got
`probe_adv` ≈ 2.3 and safe_rate +0.08.

The mechanism is now well identified and self-reinforcing: the instinct's interact push is ~5
logits; a learned preference of ~0.4 logits cannot move the argmax; so plasticity does not pay; so
`eta2` sits at or below `fixed`'s drift; so `H` stays small; so the learned preference stays ~0.4.
Making the instinct heritable did not break the circle because **there is no selection gradient on
the gene** — lowering it costs meals immediately and buys discrimination only via an `H` that is
too small to help. Raising the free basis (B) and raising the stakes (C) both made it worse; taking
food out of the instinct (D) moved safe_rate a little and the probe not at all.

**Consequence for the grid, pre-registered as row 10b:** if `eta2` in `plastic (W2)` lands inside
`fixed`'s drift range, the plastic conditions are effectively `fixed`, and a recipe null is
attributable to *plasticity having been selected off*, not to *the rule failing to bridge
station → nut*. That is a finding about the modulator stream, and a different one from what v3.6
set out to test.

---

## Stopping rule (agreed before the run)

Second-half aggregates, **event-weighted** (Σcorrect / Σattempts, not a mean of per-window ratios —
a mean-of-ratios artifact produced the v2 "ratchet" claim that was retracted). Margin **0.03**,
seed criterion **4/5**. Chance = 0.167.

Rows are checked in order. Every row names the condition that attributes it — the v3.4 lesson.

| # | outcome | reading | what attributes it |
|---|---|---|---|
| **0** | any condition with pop < 80, injections > 0, or attempts/1k < 5 | that condition is uninterpretable; name it and exclude it | pop, injections, attempts/1k, per seed |
| **1** | **gate:** `fixed` ≥ 0.25 in 4/5 | the genome tracks the recipe at `recipe_every=2000`. The world does not isolate within-life learning. **Stop, retune, no claim.** | `fixed`'s `pair_gain_innate`, its era late−early |
| **2** | **ceiling:** `fixed + B` − `fixed` < 0.10 in ≥2/5 | not enough headroom for a 0.03 margin to have power. A null is then a statement about the test, not the learner. | attempts/life against the elimination prediction mean *k* 1/(7−*k*) |
| **3** | `plastic (W2)` − `fixed` ≥ +0.03 in 4/5 **and** `plastic (W2)` − `scrambled` ≥ +0.03 in 4/5 **and** `probe_adv` (recipe) > 0 in 4/5 **and** attempts/1k ≥ 0.8 × fixed **and — REQUIRED — at least one within-life signature in 4/5: `hit_old` > `hit_young`, or the attempt-in-life curve rising** | **the grown learner acquires a conjunctive delayed-credit fact within life** | `scrambled` (information, not override); `probe_adv` is within-agent, so not selection or composition; attempts/1k rules out abstention; the within-life signature is what separates acquisition *within a life* from a population-level shift. The ceiling shows what a real one looks like — its attempt-in-life curve climbs 0.13 → 0.75. |
| **4** | beats `fixed` but ≈ `scrambled` | the gain is having a plastic override of the scaffold, not the modulator's information. **Not learning.** | `scrambled` |
| **5** | hit rises, attempts/1k < 0.8 × fixed, nuts/1k flat | bought by attempting less. Abstention, not knowledge. | attempts/1k + nuts/1k |
| **6** | nothing differs by 0.03, row 2 passed, and the positive control holds | **null, attributable:** the rule doesn't bridge station→nut where exact credit *does* pay. Spends the one agreed rule-form change. | ceiling + food probe |
| **7** | as row 6 but the food control also fails | learner broken *in this world*. **Fix that first, no rule-form change.** | `probe_adv` (food), safe_rate |
| **8** | on a null: λ²^gap < 0.02 and λ2 not above `fixed`'s value | the null is about trace *length*, not the conjunction | `bridge_first`, λ2 per seed |
| **10** | `plastic` < `fixed` by ≥0.03 in 4/5 | if `scrambled` equally below → machinery cost (v3.5's mechanism). If `plastic` < `scrambled` → the modulator is actively *misleading*; credit lands on navigation. | `scrambled` vs `plastic`, `h_norm` |
| **10b** | `eta2` in the plastic conditions is inside `fixed`'s drift range in ≥4/5 | **plasticity was selected off before the question was reached.** Rows 3–6 are then about a modulator stream in which plasticity does not pay, not about bridging. Reports as a reversal of v3.1's finding #10. **Expected to fire** — see the tuning log. | `eta2` per seed vs `fixed` and `fixed + B`, whose eta genes are dead in this world; `h_norm`; `nut_share` |
| **10c** | `probe_adv` (food) ≤ 0 in ≥2/5 in `plastic (W2)` | the within-agent control failed too → row 7 | food probe per seed |
| **11** | `plastic (both)` − `plastic (W2)` ≥ +0.03 in 4/5 **and** `eta1` above `fixed`'s drift | W1 plasticity builds the conjunction; first world where `eta1` is selected up | `eta1` vs `fixed` drift |
| **12** | the scaffold genes `nav_dir`, `nav_here`, per seed, against `fixed`'s values | **these are not dead genes in any condition** — they set behaviour everywhere — so the reference is `fixed`'s value, not a drift range. A plastic condition evolving a *lower* instinct than `fixed` is evolution buying room for the override, which is the mechanism row 3 needs. Equal values mean the balance did not move, and a positive row 3 would then need another explanation. | `fixed` vs the plastic conditions; the drift scale is σ 0.2 × √(generations) ≈ 1.0 over a run |
| **13** | `plastic (W2) + fail cost` − `fixed + fail cost` ≥ +0.03 in 4/5, **and** `fixed + fail cost` ≈ `fixed` | an immediate −1 on a wrong attempt is enough: the conjunction is learnable **by elimination** even where pure delayed credit fails. If `fixed + fail cost` also moves, the 0.05 energy change altered the world rather than the signal, and the first line cannot be read as learning. | the `fixed + fail cost` arm |

**Not claimed either way by this run:** anything about culture, records or symbols (v3.7); anything
about `both`/`eta1` beyond row 11; food/recipe interference (the no-flip pair is dropped from this
pass); any comparison of absolute safe rates against v3.1's numbers (this world hands agents an
approach-food instinct that v3.1 did not have, so only the within-notebook `plastic` − `fixed`
contrast is meaningful).
