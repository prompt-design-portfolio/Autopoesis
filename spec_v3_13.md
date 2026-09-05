# v3.13 — the record: automatic writing with non-inheritable meaning

**Spec only. No code. Written against the v3.12 build; v3.12's result may change it. Seven
DECISION points, none settled.**

## What the previous draft got wrong

My earlier draft gave agents a `mark` action and a writer gene, and then pre-registered the null
that `sym_gain` would go to zero because writing costs a step and pays the writer nothing.

**That draft re-ran two results this project has already closed.** Write-probability collapse is
finding #1. Open semantics — a store whose meaning is unconstrained, so nothing binds — is finding
#2. And the "pre-registered null" I proposed *is v1's result*. A spec whose most likely outcome is
a known outcome is not an experiment; it is a re-derivation.

**The restructure removes writing as a decision.** Writing is automatic, costless and universal, so
there is no public-goods problem and no write-probability to collapse. Meaning is constrained by
construction, so semantics are not open. What is left is the only question that was ever open:
**can an agent use, within its life, a mark whose meaning it cannot have inherited?**

## The mechanism

### Writing is automatic

**Every preparation writes.** When an agent performs preparation `k` on food of type `t` at a cell,
the cell records `(label, sign)` for **type `t`**:

- `label = π(k)` — the *label* of the preparation used, not the preparation index;
- `sign = +1` if the preparation was correct, `−1` if not.

There is **no `mark` action, no writer gene, no writing policy**. Every agent that prepares leaves a
record, whether it wants to or not, at no cost. `mark_pref` from the previous draft is therefore
dropped: there is no writing preference to probe.

### Meaning cannot be inherited

`π` is a **permutation of the K preparations, re-drawn at every remap**, independently of the
mapping itself. So label `j` denotes preparation `π⁻¹(j)`, and *which* preparation that is changes
every era.

This is the whole design. If a mark recorded the preparation index directly, its meaning would be
fixed by the world and a genome could evolve to read it — "channel 3 means prep_3" — and the result
would be inheritance, not transmission. Re-drawing `π` each era makes **the label→preparation
binding unavailable to selection**: no genome can carry it, because it is worthless by the next era.
The population must establish it **within lives**, or not at all.

### Reading is an observation

**K new observation channels**, for the food type underfoot. Channel `j` carries the **age-decayed
sign of the latest mark at label `j`** for that type at that cell. No mark under you, or no food:
zeros.

Reading is not an action. It has no cost and no policy, so a null result cannot be ambiguous between
"cannot read" and "does not bother".

### One heritable gain

**`sym_gain`** — a single heritable scalar multiplying the whole block of K read channels,
**starting near zero**. The population can evolve to attend to the record or to ignore it, exactly
as `nav_dir` / `nav_here` work.

Note what has changed relative to the previous draft: **reading is free and writing is automatic**,
so raising `sym_gain` costs an agent nothing and benefits it directly. There is no altruism in the
loop. If `sym_gain` stays at zero here, it is because the record is *useless*, not because writing
is a public good — and that is a clean negative rather than a re-derivation of finding #1.

> **DECISION 1 — the store's shape.** "The cell for its food type" implies each cell holds a
> `T × K` array of age-decayed signs: one row per food type, one column per label. A cell's food
> type changes over time, and the mark must stay attached to the type it was about, or a mark left
> on type A would be read as evidence about type B. On a 60×60 grid that is 3600 × 3 × 5 = 54,000
> floats — negligible. **Proposal: `T × K` signs per cell.** The alternative — one `K` vector per
> cell with the type implicit — is cheaper and wrong for the reason just given.

> **DECISION 2 — the decay constant.** A mark must **outlive the agent that wrote it**, or nothing
> can pass between agents, and must **not outlive the era**, or it becomes actively misleading when
> `π` and the mapping are redrawn. Those two constraints bracket it. **Proposal: half-life ≈ 1
> lifetime, and ≥ 3 half-lives inside `prep_every` = 700.** This is a pre-check measurement, not a
> guess: report mark age at read time, and the fraction of reads whose latest mark predates the
> current era.

> **DECISION 3 — `sym_gain`'s starting value and scope.** One gain over the whole channel block,
> starting near zero (proposal: same mutation scale as the other scalar genes, initialised at
> 0.05). Should it be allowed to go **negative**? A negative gain is a coherent policy — "do the
> opposite of what the mark says" — and would be the right answer in `noise record`. **Proposal:
> allow negative, and report the sign distribution per arm**; a population that drives it negative
> in the noise arm and positive in the real arm is itself evidence the channel is being read.

## The arms

| arm | store | plasticity | what it isolates |
|---|---|---|---|
| `plastic` | **none** | yes | v3.12's world — the baseline the record must beat |
| `plastic + record` | real | yes | the claim |
| `plastic + noise record` | **labels randomised on write** | yes | the store's mere presence |
| `fixed + record` | real | **no** | whether a genome can use a record without learning |

`noise record` is the load-bearing control. A store changes the world: channels exist, cells carry
state, decay runs. An arm that improves *because a store exists* is not an arm that improved
*because information passed*. In `noise record` the sign is written at a **random label**, so mark
density, channel statistics and decay are identical and the label→preparation association is
destroyed.

`fixed + record` is the genetic control. With `π` redrawn each era it should get nothing, and if it
does get something, the meaning is leaking through the wiring.

## Gate R

**The record's meaning must not be available to selection.** This is the v3.13 stop row.

The drafted form does not transfer literally, and I should say so rather than restate it as though
it did. In the previous draft, writing was a policy, and the gate asked whether the *architecture*
supplied the symbol→preparation mapping. Here writing is deterministic by construction, so **within
an era the mutual information between `(label, sign)` and the correct preparation is maximal — that
is the point of the design, not a fault.**

The gate that carries the same intent under automatic writing is the **cross-era** one:

> **Gate R.** Pooled **across eras**, the mutual information between `(label, sign)` and the correct
> preparation must be at or below the level the `noise record` arm produces. Reported per era as
> well, where it is expected to be high.
>
> A cross-era association above noise means `π` is not doing its job — the label→preparation binding
> is stable enough for selection to capture — and the run is not read.

> **DECISION 4 — is that the right restatement?** It is my proposal, not a ruling carried over. The
> check it performs is exactly "meaning is not inheritable", which is what the mechanism claims.

Carried over unchanged: **row 0** at `pop < 80` · **row 1a** against v3.1 · **row 1b** as a measured
genetic baseline · **row 1c** standing variation · **rig checks 2(a)–(c)** · founder-free metrics
with founder share printed · each arm's own type-blind level.

## The probes

- **`read_pref(a, t, j, s, k)`** — food type `t` underfoot, channel `j` carrying sign `s`, and
  nothing else: how much does this agent want preparation `k`? The reading side, within-agent, no
  behaviour involved.
- **`store_gain`, learned vs innate, on the current `π`** — how much more the agent prefers the
  preparation the mark **endorses** than the alternatives, evaluated under the era's actual `π`.
  **The learned/innate split is the claim's instrument**, exactly as `prep_gain` innate vs learned
  is now: innate ≈ 0 says the genome cannot read the record (which `π` guarantees), and learned > 0
  says this agent bound it inside its own life.
- **`mark_pref` is dropped.** There is no writing policy.

> **DECISION 5 — the self-marking confound, and what fixes it.** An agent reads marks it wrote
> itself. That is memory, not transmission, and no probe on the standing population separates them.
> **The newborn line is what separates them**, because a newborn has written nothing — every mark it
> reads was left by someone else. I propose the newborn measure be treated as **the transmission
> claim**, and `store_gain` learned-vs-innate as the *binding* claim, and that the two be reported
> as different things rather than pooled.

## The claim

**Frozen-population replay**, as in v3.12: era-boundary snapshot, births/deaths/injection disabled,
300 steps, nothing can change but `H`. Three cells, differing only in the store:

| cell | store contents |
|---|---|
| **real** | the marks the population actually left |
| **randomised** | same marks, labels shuffled — presence held, content destroyed |
| **erased** | marks cleared |

> **Claim (binding).** Real exceeds randomised by **≥ 0.10 in every seed**, and the gap is **absent
> in `noise record`**.

Randomised is the primary comparison because it holds the observation statistics fixed; **erased is
reported as the third cell** so the contribution of the channels' mere presence is visible.

> **Claim (transmission).** **Newborn preparations-to-first-correct** — for agents in their first
> preparations, how many preparations until the first correct one — is **lower in a marked world
> than in an unmarked one**.
>
> A newborn has written nothing and learned nothing. Every mark it reads came from another agent.
> This is the line that says information *passed*, and it is not substitutable by the frozen-replay
> pair, which cannot distinguish an agent using its own marks from an agent using someone else's.

> **DECISION 6 — how the newborn measure is windowed.** Preparations 1..n of a life, in a world with
> the store live against the same world with `sym_gain` forced to zero (not the store removed — that
> changes the world; forcing the gain isolates the *reading*). **Proposal: n = 5, reported as the
> mean number of preparations to the first correct one, and as the hit on preparation 1 alone.**

## Order

1. **v3.12 must read first.** If the learner does not clear its D6 claim in a 60-mapping space,
   there is no within-life competence for a record to carry and v3.13 is premature.
2. **The semantics test extends first**, enumerated as always: the `T × K` store, automatic writing
   on every preparation, `π` redraw at each remap, decay, and the fact that reading is an
   observation and not an action.
3. **Gate R before any outcome row.**

> **DECISION 7 — the pre-registered prediction.** Stated so it can fail: `sym_gain` **rises above
> its starting value in `plastic + record` and not in `noise record`**; `store_gain` **learned > 0
> and innate ≈ 0** in every seed; newborn preparations-to-first-correct **lower with the store
> live**. The honest alternative outcome is that the within-life binding of a label that rotates
> every era is simply harder than the preparation conjunction itself — the agent must learn
> label→preparation *and* preparation→type, from the same signal, inside one era — in which case
> v3.13 nulls on a **capacity** limit, not a public-goods one. That would be a new result, and it
> is the reason this version is worth running where the previous draft was not.

## What is deferred

**Emergent writing is v3.14**, and it is specified only after v3.13 reads. If a population cannot
use a record that is handed to it for free, there is nothing to be gained by asking it to choose to
write one — that ordering is what keeps v3.14 from re-running finding #1 a third time.
