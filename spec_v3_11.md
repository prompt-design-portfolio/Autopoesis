# v3.10 world spec — the preparation world

*Agreed 5 September 2026. D1 accepted with the rig check reframed; the type-blind analysis folded
in; D2–D6 accepted as recommended. **Implemented**, pre-checks run. Findings from those pre-checks
are at the end, including two that need your call.*

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

**RESOLVED — D1 accepted, and the rig check is reframed.** The consequence is stronger than "noisy":
once the mapping is known, preparation pays 1.5 on *any* food, so the safe/poison fact becomes
**irrelevant** and a good learner stops eating raw entirely. Food learning is then *unselected*, and
`probe_adv` (food) decays for a **good reason** — not because the transition broke `H`.

So **rig check 2(a) is read in the first mapping era of phase 2 only** (steps 0–2000 after the
switch), when nobody knows the mapping and `eat` is still the right action. After that, **prep share
of meals by era** is the diagnostic: a learner that has the mapping should shift from `eat` to
`prep`, and that shift is itself an efficiency signature.

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

## Three levels on prep hit — and the type-blind floor

| level | value | what it means |
|---|---|---|
| chance | **1/3** | a random preparation |
| **type-blind** | **0.5** | "always `prep_k`" for a k useful in this era: right for one food type, wrong for the other, **no type knowledge at all**, EV +0.5/meal |
| full | **1.0** | the conjunction: the right preparation for each type |

With distinct mappings, exactly one preparation is useless in any era and each of the other two is
correct for one type. A type-blind policy therefore scores **0.5 within a favourable era** — but
**only 1/3 averaged across eras**, because k is useless in a third of mappings (EV −0.5/meal there).
**A genome beats chance only by tracking the era, not by holding one preparation.**

**A conjunction shows as BOTH types above 0.5**, not one at 1.0 and the other at 0. Per-type hit is
printed for every arm.

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
`repro_threshold` 3.0 · `repro_cost` 1.5 · `max_energy` 5.0 · **`max_pop` 800** · `init_pop` 300 ·
`min_pop` 40 · food +0.7 / poison −0.5 · `flip_every` 300 · `eta_init` 0.2.

Preparation: **`prep_value` 1.0** · **`prep_fail` 0.5** · **`prep_every` 350** · K = 3.
*(v3.11: both changed after the v3.10 acceptance run. See "Changes after the v3.10 acceptance".)*

Expected value of a preparation at chance: `1/3 × 1.0 − 2/3 × 0.5 = **0.000**`, against `+0.1` for a
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
| **0** | uninterpretable per phase: pop < 80, **or injections sustained (mean ≥ 1 per window over the half)**. Plus a **cap check**: no outcome arm above **90% of `max_pop`** in phase 2's second half | exclude and name it. `random policy` exempt. **Why the change:** the v3.10 acceptance excluded phase-2 `fixed` at pop 675/789 against a cap of 800 on 0.1 injections per window in one seed — a near-cap arm is not uninterpretable, and one event in ten windows is not a propped-up population. **If the cap check fails**, raise `max_pop` to 1200 and re-estimate the runtime before reading anything population-dependent | — |
| **1a** | **phase-1 gate = v3.1** — safe rate `plastic` − `fixed` ≥ 0.03, `probe_adv` (food) ≥ 1.0 | **STOP ROW.** If phase 1 is not v3.1, nothing below is read. Row-0 fallback: where `fixed` phase 1 is excluded, that seed reads against v3.1's published range, conservative end 0.56 | `fixed`, or v3.1's published range |
| **1b** | **mapping gate** — `fixed` prep hit **≤ 0.55** in 3/3, **and** first-preparation hit **≤ ~0.55** in `fixed` and `scrambled`. Read the per-type split **PER ERA**, never on the phase half | genes **may** hold a type-blind preparation (0.5); they must not track the **conjunction**. `fixed`'s per-type hit is printed: one type high and the other near 0 is type-blind and allowed; **both above 0.5 in `fixed`** would be genes holding the conjunction. If it fires, shorten `prep_every` toward the flip period, **judged against `fixed` only** | `fixed`, per-type |
| **2** | **rig checks** — (a) food learning survives the switch, **read on safe rate in the first mapping era** (`plastic` − `fixed` ≥ 0.03); the corrected `probe_adv` (food) in the **first 500 steps** after the switch is **corroborating only**, and the **eat → prep shift timing** (prep share in 250-step bins) plus **prep share by era** are printed as diagnostics; (b) the opportunity exists: **prepared meals per life ≥ 3 in `fixed`**; (c) P(prep \| on food) against the per-phase null | if (a) or (b) fails the rig is broken: stop, diagnose, no learner claim. **Why the reframe:** the v3.1 probe form subtracts the best *other* action, and with three preparations live that term moves with food type, so it measured preparation preference, not food learning. The probe is corrected (`eat` minus the mean move logit), but a *decaying* probe after the first era is expected behaviour — once the mapping is known, preparation pays 1.5 on any food and the safe/poison fact stops mattering | `fixed`, `random policy` |
| **3** | **the conjunction.** `plastic` − `fixed` ≥ 0.03 **and** `plastic` − `scrambled` ≥ 0.03 in 3/3 on prep hit; `probe_adv` (prep) > 0; **one within-life signature**; **no abstention** (prepared meals not below 0.8× `fixed`) | **a positive means clearing the type-blind floor.** Read the per-type split: the conjunction is **both types above 0.5**. **`plastic` − `scrambled` on hit rate is a survivorship-contaminated contrast** — agents whose random `H` happens to help live longer, enriching the standing population with nothing learned. It stays required, but the **attribution** is carried by the within-agent lines: `probe_adv` (prep) and row 3a | `scrambled` carries the claim; `probe_adv` is within-agent; abstention read first |
| **3a** | **survivorship diagnostics (required).** (i) **survivor curve** — hit on preparations 1–5 vs 6–10 over agents that reached 10 preparations: **rising in `plastic`, flat in `scrambled`**. Every agent counted contributes both halves of its own curve, so a rise is the same individuals later in their own lives, not a different sample. (ii) **first-preparation hit** — the agent has learned nothing, so this reads the innate policy the standing population carries | a `plastic` − `scrambled` gap on hit rate with a **flat** survivor curve is enrichment, not learning, and row 3 is **not** called positive on it. Note `probe_adv` is computed over the **living** and so is itself partly survivorship-selected; the survivor curve is not | within-agent |
| **3b** | **knockout** — late `plastic` and `scrambled` genomes replayed with `eta_scale = 0` in a **fresh world seed**: same brains, no learning. **Read the FIRST 500 steps, not the second half.** `eta_scale = 0` stops learning but **not reproduction**: measured on the v3.10 acceptance, `max_gen` went 4–8 → 9–35 over 3000 steps and the hit rate climbed with it (`plastic` seed 0: 0.485 → 0.657), so the second half measures **re-selection in the new world**, not the genome. Both windows and both `max_gen` values print, so the contamination stays visible | if the standing advantage lives in `H`, **both** fall to the type-blind floor. What then separates them is the survivor curve, which is learning rather than luck. A `plastic` genome that keeps its advantage with learning off had it in the **genome**, not in `H` | within-genome |
| **4** | gene rows: `eta2`, `lam2`, `eta1` against `fixed` and `scrambled`; unwired scaffold genes as drift scale | corroborating only | — |
| **5** | if row 3 is null with rows 1–2 clean | the first earned statement about the learner's limit, on a dense, immediate, two-sided task. Spend the one rule-form change there | — |

**Pre-registered prediction, refined after the v3.10 acceptance and recorded before the v3.11
run:**

1. **`fixed` at 0.52–0.56** — its own type-blind level and no more. At `prep_every` = 700 gate 1b
   still fired (first-prep hit 0.67–0.72), because of **standing polymorphism**: the population
   carries genotypes for several of the six possible mappings at once, so a remap needs no
   mutation — lineage selection just promotes whichever genotype already matches. 350 is below a
   generation, so a lineage cannot be selected up within an era.
2. **`scrambled` ≈ `fixed`.** Its elevation in v3.10 was the same genetic era-tracking, not lucky
   `H`: survivor halves matched `fixed` to three decimals and first-prep hit was 0.821/0.607. It
   should track `fixed` again, and its `probe_adv` should stay at zero.
3. **`plastic (W2)` at 0.65–0.70**, and above 0.5 on **both types** — the conjunction, not a
   type-blind guess. It should lose less to the shorter era than `fixed` does, because it
   relearns within a life rather than waiting on a generation.
4. **The since-remap curve dips and recovers within ~5 preparations** in `plastic`: at a remap the
   learned `H` is wrong, so a learner pays for the change and earns it back. Flat means nothing is
   relearned within a life; never recovering means 700 steps is shorter than the learner needs,
   which would itself be the answer to whether any era length separates selection from learning.

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


---

# Findings from the pre-checks (1 seed, 3000-step phases, every arm)

**The world is readable — the thing v3.9 never achieved.** Prepared meals per life: `fixed` **16.5**,
`plastic` **19.7**, ceiling **62.2**, null **6.6**, against a criterion of ≥ 3. There is no density
to tune and no approach behaviour to evolve.

| arm | P2 pop | prep hit | hit \| A | hit \| B | prep/life | probe_adv (prep) |
|---|---|---|---|---|---|---|
| random policy | 117 | **0.334** | 0.337 | 0.330 | 6.6 | — |
| fixed | 338 | **0.500** | 0.740 | 0.256 | 16.5 | — |
| scrambled | 399 | 0.722 | 0.553 | 0.840 | 15.8 | **0.342** |
| plastic (W2) | 399 | 0.833 | 0.934 | 0.692 | 19.7 | **5.624** |
| fixed + B (ceiling) | 399 | 0.886 | 0.879 | 0.894 | 62.2 | — |

`random policy` lands on chance and **`fixed` lands on exactly the type-blind floor, 0.500, with
A 0.740 / B 0.256** — the predicted signature, and gate 1b is clear at ≤ 0.55.

### Three things that need your call

**1. `scrambled` sits well above the type-blind floor (0.722, both types above 0.5).** The control is
showing what looks like conjunction knowledge. Its **within-agent probe is 0.342 against plastic's
5.624**, a 16× gap, so the population-level hit rate carries a **survivorship** component the probe
does not: agents whose random `H` happens to favour the correct preparation live longer, so the
standing population is enriched for lucky `H` without anything being learned. Row 3 already requires
both lines, which is the right design — but the hit-rate margin against `scrambled` is partly a
survivorship comparison and should be read that way.

**2. Rig check 2(a) fails as specified, and I found the instrument at fault first.** The v3.1 food
probe is `logit(eat)` minus the **best other action** — and with three preparations live, that term
moves with food type, so the probe was measuring preparation preference. Fixed: the probe is now
`logit(eat)` against the **move** logits, which are food-type-independent. That raised the first-era
value from 0.040 to **0.196** — still far below the ≥ 1.0 criterion. The cause is that the eat→prep
shift happens **inside** the first era (prep share 0.836 in era 1), so the window the criterion
assumes is shorter than 2000 steps. Safe rate does hold there: `plastic` **0.599** against `fixed`
0.489. **Proposal:** read 2(a) on **safe rate in the first era** (`plastic` − `fixed` ≥ 0.03) with the
probe corroborating, or read the probe in the first ~500 steps after the switch. I have not measured
the 500-step version.

**3. Populations sit at the 400 cap** in three of five arms. At a hard cap births become a queue
rather than differential fecundity, which blunts selection — the same issue that cost a v3.6 tuning
pass. Raising `max_pop` is the fix; it is a spec change and yours to make.


---

## Changes after the v3.10 acceptance run

Two parameter changes, each with its own check, and three rule fixes. Nothing here was judged
against a plastic condition's outcome.

| change | from → to | why | check |
|---|---|---|---|
| `prep_every` | 2000 → **700** | at 2000 there are ~12 generations per era and selection uses them: `fixed`'s `prep_gain innate` was **0.995 / 1.611** against `random policy`'s −0.10, and its per-era split had both types above 0.5 in 2 of 4 eras | `fixed` prep hit **≤ 0.55**, and `fixed` / `scrambled` **first-prep hit ≤ ~0.55** |
| `prep_value` | 1.5 → **1.0** | at 1.5 a *chance* preparation paid **+0.167**, so a population could ride the preparation payoff without knowing anything, and every outcome arm sat at 675–799 against a cap of 800. At 1.0 the chance EV is exactly **0.000** — value comes only through knowledge of the mapping | no outcome arm above **90% of `max_pop`** in phase 2's second half; if it still caps, `max_pop` → 1200 and re-estimate the runtime |

The economy after the change: a chance preparation is worth **0.00**, eating raw at chance
**+0.10**, knowing the flip **+0.70**, knowing the mapping **+1.00**. Preparation is now strictly
worse than raw eating until the mapping is known, and strictly better once it is.

Rule fixes: **row 0** as above · **row 3b** reads its first 500 steps · **the pre-registered
prediction** refined to four numbered clauses.


---

## Changes after the v3.11 acceptance run at `prep_every` 700

Every within-life line was positive in 2/2, and gate 1b still fired.

**Why it fired: standing polymorphism.** There are six possible distinct mappings, and the
population carries genotypes for several of them at once. A remap therefore needs no mutation and
no new adaptation — **lineage selection simply promotes whichever genotype already matches**, and
it can do that inside a single era. The signature is in the per-era `(A, B)` pairs, which *flip*
between eras rather than drifting, and in a first-preparation hit of **0.67–0.72**: that is the
innate policy of the standing population, measured before the agent has learned anything.

| change | from → to | why | check |
|---|---|---|---|
| `prep_every` | 700 → **350** | below a generation, so a matching lineage cannot be selected up within an era. Deliberately **not a multiple of `flip_every` = 300**, so the fast fact and the slow fact do not come into phase | gate 1b re-read at 2 seeds. **If it still fires, the next change is K = 4 preparations** — which takes the number of distinct mappings from 6 to 12 and makes standing polymorphism across all of them much more expensive |

Rule changes, all in the reading and none in the world:

- **Rig check 2(a)** reads on **whole-phase founder-free safe rate**, `plastic` − `fixed` ≥ 0.03.
  At 350 an era is far too few meal events to read a rate on.
- **Abstention** fires only if prepared meals per life < 0.8× `fixed` **and** (hit ≤ `fixed`
  **or** pop ≤ `fixed`). Preparing less while scoring and living better is a learner declining bad
  bets, not one abstaining from the task.
- **Survivor curve** halves become preparations **1–2 against 6–10**: the first two are before
  within-life learning could have taken hold, so it is the agent's own naive rate against its own
  settled rate.
- **New: survivor-conditioned since-remap curve.** Only agents that made **8 preparations both
  before and after the same remap** are counted, each contributing its own rate either side. The
  population-level since-remap curve is open to the objection that the agents alive at preparation
  1 are not the ones alive at 10; this line is not.
