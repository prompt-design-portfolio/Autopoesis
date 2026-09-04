# v3.8: grow the learner with its world

**Where this sits.** v3.6 put the learner grown in the flip world into a scaffolded recipe world
and it did not transfer. Rows 3 and 13 were null; rows 7 and 10b fired. `eta2` was selected off,
`lam2` selected short — *more so with an informative modulator than with a scrambled one* — and the
food scaffold flattened the positive control to a safe rate of 0.50 in **every** condition,
`fixed` included. Row 7's remedy was never a rule change; it was to stop handing the learner an
instinct that overrides it.

**The one change: staging.** Per seed, **one continuous population**. Phase 1 is v3.1's flip world
— food only, safe type flips every 300 steps, **no scaffold of any kind**. Phase 2 switches the
items, stations and nuts on *for that same population*. The agents, their `H`, their eligibility
traces and the world are all carried across the boundary untouched; only `cfg.chain` flips. The
learner is not asked to solve the recipe until it is demonstrably working in the world that made
it.

**No instinct.** `scaffold_food=False` and `scaffold_chain=False`: `nav_dir` and `nav_here` stay in
the genome and reach nothing, so they are dead genes here and give the drift scale for a heritable
scalar. Approach behaviour for the chain has to evolve unwired — which is row 3, and a real risk.

**What is still hand-designed, and named so.** The `goal` observation channel remains: it points at
*any* item when empty-handed, *any* station when carrying one, nuts when tooled. It is a stage
feature, never a valence — it never says which item or station — and it is identical in every
condition, so it cannot explain a between-condition difference. `cfg.goal_channel=False` turns it
off for a stricter version.

**The deviation the gate tests.** The observation is 60 inputs and 24 hidden in both phases, where
v3.1 had 23 and 16. In phase 1 the chain channels are all zero and the `food_any` channel is dead
(it existed only to drive the instinct), so phase 1's **live** inputs are exactly v3.1's set. The
single real deviation is **24 hidden units instead of 16**, and row 1 is the gate on it.

**Metabolism is v3.1's throughout** — repro 3.0 / cost 1.5 / max_energy 5.0 / max_pop 400 — not
v3.6's tuned 4.5 / 2.25 / 8.0 / 600. Those were tuned for a *scaffolded* world, and applying them
at the phase boundary would make the transition two changes at once: the chain switching on **and**
the cost of living changing. The transition is the thing being measured. What is taken from v3.6's
`WORLD` is the chain itself — items, stations, nuts, their values, `recipe_every`, `fail_cost=0`.

---

## Conditions (5 × 3 seeds)

| condition | phases | what it is for |
|---|---|---|
| `fixed (staged)` | 8000 off, 8000 on | baseline and gate. Also the drift reference for `eta1`/`eta2`/`lam2` and the scaffold genes. |
| `scrambled (staged)` | 8000 off, 8000 on | **the control that carries the claim.** Same plasticity, same H magnitudes, random-sign modulator, no information. |
| `plastic (W2) staged` | 8000 off, 8000 on | the result condition. |
| `plastic (W2) + fail cost` | 8000 off, 8000 on | **the oracle arm.** `fail_cost = 0.05`, no scaffold, same phases: a wrong attempt costs energy, so `m = −1` arrives **at the station** while the network still has to find the chain itself. If row 4 is null, this says whether the failure is the rule or the world. |
| `plastic (W2) scratch` | 16000 on | the control for whether staging is needed at all: same learner, same total steps, full world from step 0. |

`fail_cost = 0` everywhere except the oracle arm — pure delayed credit is still the question row 4
asks.

**Runtime.** 16000 steps per run, measured at **3–6 min** (populations here are 170–260, smaller
than v3.6's 300–450). The grid of 12 runs is **~45–70 min**.

---

## Acceptance checks, run before the grid (2 seeds, `plastic (W2) staged` and `fixed (staged)`)

**Check 1 — phase 1 reproduces v3.1: two clauses of three pass cleanly.**

| | seed 0 | seed 1 | target | v3.1 |
|---|---|---|---|---|
| safe rate `plastic` | 0.627 | 0.641 | — | 0.60–0.66 |
| safe rate `fixed` | 0.517 | 0.559 | — | 0.51–0.56 |
| **`plastic` − `fixed`** | **+0.111** | **+0.081** | ≥ 0.03 ✓ 2/2 | +0.08 |
| **`probe_adv` (food)** | **1.333** | **2.104** | ≥ 1.0 ✓ 2/2 | 1.4–2.7 |
| `eta2` `plastic` vs `fixed` | 0.123 vs 0.148 ✗ | 0.136 vs 0.095 ✓ | above in 3/3 | — |

Both safe rates land **inside v3.1's published ranges**, and `probe_adv` (food) is back to 1.3–2.1
from v3.6's 0.068/0.155. Removing the instinct restored the positive control; the observation size
is not the problem. So **the gate is met on the two clauses that measure the learner**.

**The `eta2` clause fails in 1 of 2 seeds, and it is the weakest of the three instruments.**
`fixed`'s `eta2` is a *dead gene* drifting over 0.045–0.24 (the handoff's range across ten dead-gene
seeds), so "above `fixed`'s" is close to a coin flip at n=2 — both values here sit inside that
range. The better-powered version, which the summary also prints, is `plastic` against
**`scrambled`**, whose `eta2` v3.1 saw driven down to ≈0.02 when the modulator carried no
information. Row 1 should be read on the safe rate and the probe; the `eta2` line is corroborating.
Flagging rather than quietly re-specifying it.

**Check 2 — phase 2 survives: passes 2/2.** `plastic` population 198 and 172, zero injections.
Attempts happen unwired in every condition (attempts/1k ≈ 8.2–8.5, `has_tool` 0.025–0.039), which
is the first evidence for row 3.

**One result that was not asked for, and that bears on v3.6.** Safe rate in `plastic (W2) staged`
goes **0.627 in phase 1 → 0.500 in phase 2**, with *no scaffold anywhere in this notebook*. The
v3.6 reading attributed the flat positive control to the food scaffold. That attribution is at best
incomplete: **the chain switching on collapses food discrimination on its own.**

Four candidates, and **the timing separates them** — the transition table prints the first-bin drop
per seed:

| candidate | mechanism | signature |
|---|---|---|
| **(a) basis shift** | **39 of the 60 inputs go from zero to live in one step** (measured, not estimated: 16 live in phase 1, 55 in phase 2; the 5 dead in both are the `food_any` channel). `H2`'s learned readout suddenly sits on a hidden basis that has moved all at once — v3.5's non-stationary-input mechanism, at scale. | **immediate**: the drop is in the *first* 500-step bin, and `h_norm` is unchanged — `H` is intact, what it is read off has moved |
| (b) modulator swamping | nuts supply a frequent sign-positive `m` | gradual, `nut_share` rises with it |
| (c) time budget | agents spend steps on the chain rather than eating | gradual, meals/1k falls |
| (d) depletion | `crop_safe` falls as in v3.2's learner worlds | gradual |

This is n=2 from an acceptance check, not a result.

Two other n=2 observations, recorded so they are not discovered later: `lam2` **lengthens** across
the transition in both seeds (0.584→0.784, 0.682→0.720), the opposite of v3.6's shortening; and
`fixed`'s phase-1 population is low (96, 51 — the second with injections, which row 0 would exclude
in that phase). Population floors on the `fixed` arm are worth watching in the grid.

---

## Stopping rule (agreed before the run)

Event-weighted aggregates (Σcorrect/Σattempts, Σsafe/Σmeals — not means of per-window ratios; that
artifact produced the retracted v2 "ratchet"). Second-half aggregates are computed **per phase**.
Margin **0.03**, seeds **3/3** for this pass. **A positive on any row gets seeds 3–4 before it is
called.** Rows are checked in order; each names the condition that attributes it.

| # | outcome | reading | what attributes it |
|---|---|---|---|
| **0** | any condition, **in either phase**, with pop < 80 or injections > 0 | uninterpretable in that phase; name it and exclude it there | pop and injections per phase, per seed |
| **1** | **phase-1 gate.** safe rate `plastic` − `fixed` ≥ 0.03 in 3/3, `probe_adv` (food) ≥ 1.0, `eta2` above `fixed`'s | v3.1 is reproduced at 60 inputs and 24 hidden; everything below is readable. **If it fails, that is a finding about observation size — stop, and decide together.** Nothing in phase 2 can be read without it. **Row-0 fallback:** `fixed`'s phase-1 population hit 51 with injections in one acceptance seed, and row 0 would exclude it — but it is the gate's baseline. Where `fixed` phase 1 is excluded in a seed, that seed's gate reads `plastic`'s safe rate against **v3.1's published fixed range 0.51–0.56** (the conservative end, 0.56), and the summary prints which seeds used the fallback and which used the measured baseline. | `fixed (staged)` phase 1, or v3.1's published range where it is excluded; `probe_adv` 1.4–2.7 |
| **2** | **transition.** `plastic` phase-2 pop ≥ 80, no injections; `eta2`/`lam2` stay above `fixed`'s; **and the safe-rate drop across the switch is attributed** (see below) | the grown population survives the chain and keeps its plasticity. If instead `eta2` is selected off and `lam2` short as in v3.6, **the chain does that, not the scaffold** — a more general finding than v3.6's. | `fixed (staged)` phase 2 for the drift reference; the phase-1 → phase-2 change within `plastic`; the transition table's first bin |
| **7** | **oracle arm.** `plastic (W2) + fail cost` − `fixed` ≥ 0.03 in 3/3, with the energy it paid for wrong attempts small against food income | the recipe signal is immediate here, so if this arm learns and row 4 does not, **the rule is fine and delayed credit is the obstacle**. If this arm is null too, the obstacle is upstream of the credit structure — the world, or approach, or the basis. | `fixed (staged)` as the control on the energy change, with `e_fail_per_1k` measuring it. **Limitation, stated not assumed away:** without a `fixed + fail cost` arm this cannot separate the −1 *signal* from the 0.05 *energy change* as cleanly as v3.6's row 13 did; the measured energy bounds it, and the sixth arm is one line if the row fires. |
| **3** | **approach evolves unwired.** attempts/1k above zero **and** rising from the first quarter of phase 2 to its second half, `has_tool` > 0 | the chain can be found without an instinct. **Flat at zero means row 4 cannot be read at all** — there are no attempts to compute a hit rate over, and a "null" on the recipe would be a null about navigation. | attempts/1k early vs late within phase 2; `has_tool`; `fixed` and `scrambled` as the unlearned comparison |
| **4** | **the recipe** — v3.6's rows 3/6/8/10b applied to phase 2: `plastic` − `fixed` ≥ 0.03 **and** `plastic` − `scrambled` ≥ 0.03 in 3/3, `probe_adv` (recipe) > 0, **and — required — a within-life signature** (`hit_old` > `hit_young`, or the attempt-in-life curve rising) | the learner acquires the conjunction within life, in a world it grew into | `scrambled` carries the claim, not `fixed`; `probe_adv` is within-agent so it is not selection or composition; `bridge_first` and λ2^gap say whether a null is about trace length instead |
| **5** | **staged vs from-scratch**, on the matched last quarter of the run | matching on population and attempts → **staging is not needed**, and the v3.6 failure was the scaffold alone. From-scratch collapsing (pop < 80 or injections > 0) while staged survives → **staging is the method**, and that is the transferable result. | `plastic (W2) scratch` against `plastic (W2) staged`, same absolute window |
| **6** | `nav_dir`, `nav_here` per seed and per phase | nothing is wired to them here, so they are **dead genes** and this is the drift scale for a heritable scalar over a run (σ 0.2 × √generations ≈ 1.0). Any row-2 or row-4 claim that leans on a gene moving must clear this. | the genes' own spread across conditions, which have no reason to differ |

**Two instruments added on review.** A **learning-rule self-test** runs in the setup cell and halts
the notebook if it fails: one agent, one fixed observation, one chosen action, `action_noise = 0` so
`act()` is deterministic — `m = +1` must raise that action's logit and `m = −1` lower it. It drives
the real `act()` and `learn()`, not a re-implementation. And **`trace_recency`** logs, at each
phase-1 meal, the share of the eligibility trace's L1 mass *not* attributable to steps older than
five: `1 − ‖λ2⁵·e2(t−5)‖₁ / ‖e2(t)‖₁`. A trace dominated by the last few steps cannot carry credit
back to a station attempt, and this measures that in phase 1, before the chain is there to confound
it. Phase 1 only — the ring is cleared at the boundary.

**Not claimed either way by this run:** anything about culture, records or symbols; elimination
(`fail_cost` > 0) — that is a later condition, only if row 4 is positive; and any comparison of
absolute recipe hit against v3.6's, whose world had a different metabolism and an instinct.
