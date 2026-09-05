# v3.13 — the record: a symbol store on the preparation world

**Draft spec. No code. Written in parallel with the v3.12 build; nothing here is settled and
v3.12's result may change it. DECISION points are flagged.**

## What the record is for

v3.11 showed the grown learner acquires a two-item conjunction **within a life**, and v3.12 asks
whether it still does when the mapping space exceeds what standing variation can cover. Both are
claims about **one agent's own lifetime**. Everything an agent learns dies with it: `H` starts at
zero in every newborn and is never inherited.

The record asks the next question: **can what one agent learned reach another agent, and be used?**

That is the smallest step toward accumulation that does not smuggle in a designer. It is **not** an
LLM, not language, not a curriculum. It is a mark on the world that an agent can leave and another
can read, whose *meaning is not given* — the mapping between mark and fact has to be established by
the population, within lives, or not at all.

## The mechanism

### The symbol store

Each grid cell carries a small **symbol slot**: an integer in `0 … S−1`, or empty. Two new actions:

- **`mark`** — write the agent's current symbol choice into the cell it stands on;
- the symbol written is chosen by the policy, so `mark` is really `S` actions, or one action plus
  an `S`-way choice head.

> **DECISION 1 — action shape.** Either `S` separate `mark_s` actions (simple, matches how
> preparations are already done, but the action space grows as `4 + 1 + K + S`), or one `mark`
> action with a separate symbol head (keeps the action space small, adds a second output group and
> a second place for the learning rule to apply). I lean to **`S` separate actions**: it reuses the
> preparation machinery exactly, and the learning rule already handles a multi-way choice. With
> `K = 5` and `S = 4` the action space is 14.

Reading is **not an action**. The symbol under the agent, and the symbols in its view, arrive as
**observation channels** — a mark is part of the world, like food. This matters: if reading were an
action it would have its own cost and its own policy, and a null result would be ambiguous between
"cannot read" and "does not bother".

> **DECISION 2 — S, and persistence.** `S` must be at least `K` for a mark to be able to name a
> preparation, and small enough that a symbol is not free to disambiguate by accident. **`S = K`**
> is the clean choice. Marks should **decay** (a `mark_rot` like `food_rot`) so a stale mark from a
> previous era does not poison the next one indefinitely — but slowly enough to outlive the agent
> that wrote it, or the record cannot carry anything between agents. **Proposal: `mark_rot` tuned
> so a mark's half-life is ~2 lifetimes and well under one era.**

### The input-gain gene

A heritable scalar **`sym_gain`** multiplies the symbol observation channels on the way in. At
`sym_gain = 0` the agent is blind to marks; the population can evolve toward reading them or away.

This is the same device as `nav_dir` / `nav_here`: a gene whose value the population sets, so
"attends to the record" is an evolved property rather than a wired one. It also gives the negative
result somewhere to land — a population that drives `sym_gain` to zero has *decided* the record is
worthless, which is a finding, not a failure.

> **DECISION 3 — one gain or two.** One gain over all symbol channels, or separate gains for
> "symbol under me" and "symbols in view"? Separate gains would distinguish *using a mark where you
> stand* from *navigating toward marks*. I lean to **one gain for v3.13** — one change per
> experiment — with the split held in reserve.

## The controls

Four arms, and the two new ones are the point.

| arm | store | plasticity | what it isolates |
|---|---|---|---|
| `plastic + record` | real | yes | the claim |
| `plastic, no record` | absent | yes | v3.12's world, the baseline the record must beat |
| **`plastic + noise record`** | **symbols randomised on write** | yes | **the store's mere presence** |
| **`fixed + record`** | real | **no** | **whether a genome can use a record without learning** |

`noise record` is the essential one. A store changes the world: marks are visible, `mark` costs a
step, cells carry state. An arm that improves *because a store exists* — extra observation
dimensions, a step-wasting action that happens to slow foraging — is not an arm that improved
*because information passed between agents*. **The noise arm has every one of those properties and
none of the information.** Symbols are written and read exactly as in the real arm; only the value
written is randomised.

`fixed + record` is the genetic control: a record that a non-learning genome can exploit is a record
whose meaning is fixed and evolvable, not one established within lives.

> **DECISION 4 — should `scrambled` also get a record?** It would test whether a *useless* `H`
> plus a real store does anything. I think not: four arms is already at the readable limit, and
> `noise record` covers the "presence of a store" confound more directly. Held in reserve.

## The gate

**The record must not be readable by construction.** This is the v3.13 analogue of gate 1b, and it
is the thing most likely to go wrong.

If the symbol an agent writes is a deterministic function of what it just did — say `mark_s` is
cheapest to emit right after `prep_s` — then the mapping from mark to preparation is supplied by
the architecture, and any subsequent "communication" is an artifact. The gate:

**Gate R.** In `fixed + record`, the mutual information between the symbol in a cell and the
correct preparation for the food that was there **must be at or below the level a random writer
produces**. If it is above, the mark's meaning is coming from the wiring, and the run is not read.

> **DECISION 5 — how to measure it.** Empirical mutual information over the second half of phase 2,
> against the `noise record` arm as the null. Reported per era, since a mark's meaning can only be
> stable within an era — the mapping moves at the boundary and the marks do not.

Carried over unchanged: **row 0** at `pop < 80`; **row 1a**, the phase-1 gate against v3.1;
**rig checks 2(a)–(c)**; **founder-free** metrics with the founder share printed; each arm's own
**type-blind level** beside every hit rate.

## The probes

Two new within-agent probes, both built like the existing `prep_pref` — a synthetic observation,
the agent's own logits, no behaviour involved.

- **`mark_pref(a, ftype, s)`** — with food type `f` underfoot and no mark present, how much does
  this agent want to write symbol `s`? The **writing** side of the binding.
- **`read_pref(a, symbol, k)`** — with symbol `s` underfoot and food present, how much does this
  agent want preparation `k`? The **reading** side.

**The binding is the composition.** A population has bound symbol `s` to preparation `k` when
writers preferentially emit `s` on food whose correct preparation is `k`, **and** readers
preferentially choose `k` on cells carrying `s`. Either half alone is nothing: writing without
reading is graffiti, reading without writing is superstition.

> **DECISION 6 — the binding score.** Report the two halves separately and their composition, as an
> `S × K` matrix per era with its diagonal-dominance under the era's true mapping. Both halves must
> be present for a claim; and the probe versions (innate vs learned) separate "the genome bound it"
> from "this agent bound it within its life", exactly as `prep_gain` innate vs learned does now.

## The claim

Stated, as in v3.12, on a **frozen-population replay pair** — and here the pair is *with and without
the record*, not matched and shuffled.

> **On a mapping no genotype was sorted for, a frozen population replayed WITH the marks its
> predecessors left reaches a higher hit rate than the same frozen population replayed with the
> marks ERASED — by ≥ 0.10, in every seed — and the gap is absent in `noise record`.**

The frozen replay is what makes this readable. The population cannot change; the only difference
between the two replays is the content of the symbol store. So the gap is what the record carried,
and nothing else.

> **DECISION 7 — erased or randomised?** Erasing removes the marks; randomising keeps their
> presence and destroys their content. **Randomising is the better control** — it holds the
> observation statistics fixed — and it makes the within-run comparison the same manipulation as
> the `noise record` arm. **Proposal: the pair is real marks vs randomised marks, with erased
> reported as a third cell.**

## Order, and what would stop this

1. v3.12 must land first. If the learner does **not** clear its D6 claim in a 60-mapping space,
   v3.13 is premature — there is no within-life competence for a record to transmit.
2. The semantics test extends first, as always: the symbol slot, `mark`, decay, and the fact that
   reading is an observation and not an action.
3. Gate R before any outcome row.

**The pre-registered null, and it is a real possibility:** `sym_gain` goes to zero and stays there.
A mark costs a step to write and pays the writer nothing — the benefit, if any, lands on some later
agent, possibly not a relative. That is a public-goods problem, and this world has no mechanism that
solves one. **If the record fails, the most likely reason is that writing is altruistic and nothing
makes it pay**, and the honest next question would be what minimal change makes a mark pay its
writer — which is a different experiment, and one to specify only if v3.13 nulls.
