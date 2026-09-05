# v3.12 — the preparation world with a mapping space that exceeds standing variation

**Spec only. No code. Code starts after the v3.11 grid is read.**
All seven DECISION points are ruled; what remains open is flagged as **OPEN** and is
sub-parameter detail, not design.

## Why the world changes

v3.11 measured the mechanism that has been defeating gate 1b, and it is not one a shorter era can
beat.

In a **six-mapping** space a population of several hundred carries genotypes for several mappings
at once. A remap needs no mutation and no adaptation: **survival sorting promotes whichever
genotype already matches**, and it completes well inside a third of an era.

| measurement | value |
|---|---|
| `fixed` first-preparation hit, late in era | **0.82** |
| genome-only hit (`eta = 0`), **matched** mapping | **0.844** (A 0.899, B 0.798) |
| genome-only hit (`eta = 0`), **shuffled** mapping | **0.296** (A 0.546, B 0.093) |

Shortening the era to 350 was tried and reverted: it shortened the *learner's* payoff window
without touching the sorting, and the pre-registered prediction failed both ways — `fixed` 0.642
against a predicted 0.52–0.56, `plastic` 0.575 against 0.65–0.70, so the learner lost to the
non-learner. Sorting is not rate-limited by generations. It is rate-limited by **how fast the
mismatched fraction dies**, and that is fast.

So v3.12 attacks the mechanism on both of its terms: **the size of the space** (D1) and **the speed
at which mismatched genotypes are killed** (D4).

---

## The world

### D1 — RULED: `T = 3` food types, `K = 5` preparations

| quantity | value |
|---|---|
| distinct mappings `P(K,T) = K!/(K−T)!` | **60** (was 6) |
| chance hit `1/K` | **0.200** (was 0.333) |
| type-blind level `1/T`, before encounter skew | **0.333** (was 0.500) |
| full conjunction | 1.000 |
| `N_ACTIONS` = 4 moves + eat + K | **10** (was 8) |

A mapping sends the three food types to three **distinct** preparations. Redraws differ from the
previous mapping in at least one type, as in v3.10–v3.11.

The three levels are now cleanly separated — 0.200 / 0.333 / 1.000 — which the v3.11 world was
not: there, chance 0.333 and type-blind 0.500 were close enough that encounter skew moved the
comparison, and each arm's own `max(share)` had to be printed beside every hit rate. That stays,
but it now matters less.

**Conditional nulls change with the action count.** Phase 1 masks the preparations, so 5 actions
are available and the per-action null stays **1/5**. Phase 2 has 10 available: the per-action null
is **1/10**, and **"any preparation" is 5/10 = 0.500** (was 3/8 = 0.375). The `P(prep | on food)`
line reads against 0.500 in v3.12.

### D3 — RULED (a): phase 1 unchanged; the third type appears at the switch, inedible raw

Phase 1 stays exactly v3.1: two types, one safe and one poison, flipping every 300 steps. Row 1a
keeps its anchor in v3.1's published range, which is the only reason it has held as a stop row
across five versions.

**Type C exists only from the switch, and cannot be eaten raw: 0 energy, `m = 0`.** It is food only
through preparation. This gives the third type a reason to exist that does not disturb phase 1's
fast fact, and it means `eat` carries no information about C at all.

#### The action × cell table, complete

Cell states are `empty`, `food A`, `food B`, `food C`. Every action on a food cell resolves
immediately and two-sidedly; nothing is silent.

| action | empty | food A | food B | food C |
|---|---|---|---|---|
| **move** (0–3) | −`move_cost`, `m = 0` | −`move_cost`, `m = 0` | −`move_cost`, `m = 0` | −`move_cost`, `m = 0` |
| **eat** (4) | −`noop_cost`, `m = 0` | safe → +`food_value`, `m = +1`; poison → −`poison_value`, `m = −1`. Cell consumed | as A | **energy 0, `m = 0`, cell NOT consumed** |
| **prep_k** (5–9) | −`noop_cost`, `m = 0` | `k == mapping[A]` → +`prep_value`, `m = +1`; else −`prep_fail`, `m = −1`. Cell consumed | as A | as A |

Three consequences to hold in mind:

- **`eat` on C is a wasted step, not a loss.** It is the one cell in the table with no energy
  change at all. An agent that eats C repeatedly pays only the opportunity cost of the step.
- **C is not consumed by eating**, so `eat` cannot destroy a preparation opportunity. C's standing
  density is bounded by `food_rot` as every other type is, but it will sit higher than A and B
  because one of the two consumption routes is closed to it.
  > **OPEN 3a.** If C's density runs away in the pre-check, the fix is to lower C's spawn rate,
  > not to make `eat` consume it — consuming would let a naive agent clear the board of exactly
  > the food the experiment is about.
- **The safe/poison flip applies to A and B only.** C has no raw value to flip.

### D4 — RULED (overruled from my proposal): `max_pop` stays 800; `prep_fail = 0.25`, `prep_value = 1.0`

| | v3.11 | v3.12 |
|---|---|---|
| `prep_value` | 1.0 | **1.0** |
| `prep_fail` | 0.5 | **0.25** |
| chance EV of a preparation | `(1/3)(1.0) − (2/3)(0.5)` = **0.000** | `(1/5)(1.0) − (4/5)(0.25)` = **0.000** |
| income to a knowing agent | +1.00/meal | **+1.00/meal** |
| `max_pop` | 800 | **800** |

`prep_value = (K−1) × prep_fail` = 4 × 0.25 = 1.0 holds the chance EV at exactly zero, and income
is unchanged from v3.11, so the population economics that produced a readable world are preserved
and `max_pop` need not move.

**Why the halved penalty is the targeted intervention, and not merely a cheaper one.** Sorting and
learning run on two different quantities in this sim, and only one of them is being changed:

- **Sorting runs on energy.** A genotype mismatched to the current mapping pays `prep_fail` on
  4 preparations in 5. Halving it halves the rate at which mismatched lineages are killed, which
  is precisely the rate that sets how fast standing variation is sorted.
- **The learning signal is a sign.** `resolve_action` returns `m = +1.0` / `m = −1.0` as literals,
  independent of `prep_value` and `prep_fail`. The learning rule `H ← H + η·m·e` therefore sees
  **exactly the same signal** at `prep_fail` 0.25 as at 0.5.

So the change slows selection and leaves learning untouched. That is a cleaner instrument than
shortening the era, which slowed both.

> **The tension this creates, stated before the run.** Slowing the death of mismatched genotypes
> slows sorting — the intent — but it also lets **more** mismatched genotypes persist, which
> *raises* standing variation and works against "the space exceeds standing variation". The two
> effects pull in opposite directions and the balance is an empirical question, not an arguable
> one. **This is why D2 matters:** the standing-variation probe measures the quantity directly,
> per era, rather than leaving it to be argued from the population size.

### D5 — RULED: accepted

`prep_every` stays at **700**. The readability criterion is read on **`fixed` only**: **≥ 3
preparations per food type per era**, measured in a pre-check before anything else runs. With
`T = 3` the same food density is split three ways, so this is the criterion most likely to fail.
**If it fails, raise spawn density — never lengthen the era**, since a longer era is more time for
the sorting this world exists to outrun.

---

## D2 — RULED: the standing-variation probe, printed per era

The claim "60 mappings exceeds what the population can hold" must be **measured, not estimated**.
Two numbers per era, both built on the existing within-agent innate probe (`prep_pref` with
`learned=False`), so nothing new is wired into the agent.

### 1. Distinct innate mappings held by the living

For each living agent, for each type `t ∈ {A, B, C}`, take `argmax_k prep_pref(a, t, k,
learned=False)`. That gives a triple `(k_A, k_B, k_C)` — the mapping the agent's **genome** would
apply, before anything it has learned.

Report, per era:

- the number of **distinct triples** present at all;
- the number that are **valid mappings** (all three preparations distinct — only these are among
  the 60; a triple with a repeat is a genome that has not separated the types);
- the number held by at least **`carrier_min` living agents**, at thresholds **1 / 5 / 20**, with
  the headline at **20**.

**Why 20, and how it gets checked.** A genotype mismatched to the current mapping earns nothing
from preparation for a whole era; to still be present at the next remap it needs enough carriers
now to survive that era. 20 is an estimate of that floor, not a measurement.

> **OPEN 2a — calibrate the threshold rather than assert it.** The pre-check should measure the
> actual quantity: take the mismatched genotypes present just after a remap, and report what
> fraction of each cohort size survives to the following remap. The carrier threshold is then the
> cohort size at which survival becomes reliable, and 20 is replaced by a measured number. Until
> that is done, all three thresholds print so the headline never rests on the estimate alone.

### 2. How many of the 60 the population would score above type-blind on

For each of the 60 mappings `m`, compute the population's expected hit **if the mapping were `m`**:
the mean over sampled living agents of the fraction of types where the agent's innate argmax equals
`m[t]`. Count how many of the 60 exceed the type-blind level `1/T`.

This is the direct statement of the claim. **If the count is small — a handful of 60 — the space
exceeds standing variation and a remap usually finds no matching genotype to promote. If it is
large, v3.12 has not achieved what it was built for, and that is the finding**, whatever row 3b
then says.

Cost is one probe sweep per era: `T × K` forward passes per sampled agent, on the same sample size
the existing probes use.

---

## D6 — RULED: what counts as the record result

The claim is stated on **row 3b's shuffled pair**, not on the population hit rate:

> **On a mapping no genotype in the population was sorted for, learning-on exceeds learning-off by
> ≥ 0.10 in every seed — and the gap does not shrink from v3.11 to v3.12.**

The population hit rate cannot carry it: it mixes the genetic baseline with the learner's
contribution, and v3.11 showed the baseline can be most of it. The shuffled knockout is the only
line that isolates what the rule adds **within a life**, on a mapping selection cannot have
supplied.

**The non-shrinking requirement is the point of the pair.** v3.12 makes the task harder in two ways
at once — 5 preparations instead of 3, and a third type — so a learner that is merely coping would
show a *smaller* shuffled gap. Holding the gap while the space grows tenfold is the thing worth
recording; a bigger gap is better, and a smaller one falsifies the claim even if it stays above
0.10.

**The v3.11 reference value is whatever the grid measures**, and it is recorded before v3.12 runs.
It is not yet known: the v3.11 acceptance checkpoint predates `final_mapping`, so row 3b runs for
the first time on the refreshed grid.

The genetic baseline (gate 1b's three numbers) is **reported alongside as a measured quantity**,
never gated against — as already restated in v3.11.

---

## D7 — RULED: the semantics test is built first

The existing test enumerates 4 actions × 3 cell states × 6 mappings = 72 rows plus poison, move,
masking and distinctness rows, against literals. It must be **rewritten to enumerate from `T` and
`K`**, giving `(1 eat + 5 preps) × 4 cell states × 60 mappings = 1440` rows, plus:

- the **type-C raw row**: `eat` on C must yield energy change exactly 0, `m = 0`, and leave the
  cell in place — the one new row in the table and the one most likely to be got wrong;
- phase-1 **masking** of all 5 preparations;
- **mapping distinctness** — a redraw differs in at least one type, and all three targets distinct;
- the flip applying to A and B and **not** to C.

**This is the first thing to build.** A mapping applied to the wrong food type is exactly the bug
this world would hide, and the count of rows is now large enough that it will not be caught by eye.

---

## Carried forward from v3.11 unchanged

Founder-free metrics as the primary reading, with founder share printed per arm and phase · row 0
excluding on `pop < 80` only, injections reported · the mapping **pinned** in every replay, guarded
by `replay_mapping_selftest` · the knockout window at **`KO_STEPS = 10`**, guarded by
`knockout_window_selftest`, with `pop` and `max_gen` beside every number · each arm's own type-blind
level `max(share)` printed beside its hit rate · survivor curve halves 1–2 vs 6–10 · the
survivor-conditioned since-remap curve at a 4-preparation window, corroborating only · abstention
firing only with `prep/life < 0.8 ×` `fixed` **and** (hit ≤ `fixed` **or** pop ≤ `fixed`) · rig
check 2(a) on whole-phase founder-free safe rate · the learning-rule and founder-tag self-tests.

## Build order, when code starts

1. The semantics test (D7), enumerated from `T` and `K`, **before** the world it tests is trusted.
2. The world: 3 types, 5 preparations, the type-C raw row, the new action count and observation
   channels.
3. The standing-variation probe (D2), with all three carrier thresholds.
4. A pre-check at 1 seed: the readability criterion (D5), C's standing density (OPEN 3a), the
   carrier-threshold calibration (OPEN 2a), and the probe printing.
5. Only then the acceptance.
