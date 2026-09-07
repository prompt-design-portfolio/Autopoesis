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
| **`plastic + record (slow)`** | real, **`label_every` = 3 × `prep_every`** | yes | **whether a label's meaning outliving what it names is what the binding needs** |

`noise record` is the load-bearing control. A store changes the world: channels exist, cells carry
state, decay runs. An arm that improves *because a store exists* is not an arm that improved
*because information passed*. In `noise record` the sign is written at a **random label**, so mark
density, channel statistics and decay are identical and the label→preparation association is
destroyed.

`fixed + record` is the genetic control. With `π` redrawn each era it should get nothing, and if it
does get something, the meaning is leaking through the wiring.

## Gate R — RULED: the matched permutation null

**The record's meaning must not be available to selection.** This is the v3.13 stop row.

**The form I specified was the wrong instrument, and the pre-check showed it.** Pooled MI against
the `noise record` arm fires on both real-record arms (+0.570 and +1.017 bits). That is the
estimator, not a leak:

- **Pooled MI has a floor set by the number of eras.** With E eras a label takes only E meanings,
  so the empirical association cannot wash out however well `π` is doing its job. Measured:
  `plastic + record` pooled **0.677 over ~3 eras and 0.605 over 5** — it decays with era count,
  not toward the noise arm.
- **The noise arm is not a matched comparison.** Its within-era structure differs, so the
  difference mixes "π rotates" with "labels are random within an era".

> **Gate R.** Permute **each era's label axis independently** and pool. Era count, sample sizes and
> within-era structure are all preserved; only cross-era consistency is destroyed — which is
> exactly what the gate asks. Grouped by **π-epoch**, a run of eras sharing a `π` — not by era. `plastic + record (slow)` holds
one `π` across three eras *by design*, so an era-wise null destroys consistency that legitimately
exists there and the gate fires on the arm's own definition (measured: z +2.21 era-wise, **+0.73**
epoch-wise). For the fast arms an epoch *is* an era, so nothing changes.
**z ≤ 2.0** means the observed pooled association is no stronger than
> chance given the era count: `π` is doing its job and meaning is not inheritable. Above that, the
> run is not read.

Pre-check, 1 seed, 5 eras: `plastic + record` z **+0.44** PASS · `plastic + noise` z **−0.10**
PASS · `fixed + record` z **+2.08** FIRES. The last is the arm where a genome could exploit a
leak, so it is the one to watch; at 1 seed and 5 eras it is not decisive.

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

### The `sym_gain` statistic — RULED

**|gain| in a record arm minus |gain| in the no-record arm, per seed.**

The magnitude, not the sign: the sign is absorbable by `W1`, so it carries no information about
whether the channel is read. And the baseline is not zero — the pre-check found **|gain| rises in
every arm including `plastic`, which has no record at all** (+0.170 plastic, +0.223 record,
**+0.332 noise**). An unused gene grows on drift, and noise grew most. So a rising magnitude on its
own is not evidence of reading, and the no-record arm is the only honest baseline.

**DROPPED as a licensing condition; REPORTED ONLY.** It licenses nothing and gates nothing. The
pre-check read it negative in every record arm (−0.257 record, −0.076 noise, −0.362 fixed, −0.079
slow), while the stale-mark ratio — the sharper instrument — pointed the other way. A statistic
that disagrees with the binding line and has no mechanism behind its sign should not decide
anything; it is kept because a large unexplained move in it would be worth knowing about.

**DECISION 5 — RULED.** An agent reads marks it wrote itself. That is memory, not transmission, and
no probe on the standing population separates them. **The newborn line separates them**, because a
newborn has written nothing — every mark it reads was left by someone else. So there are **two
claims, reported separately and NEVER pooled**:

- **binding** — `store_gain` learned vs innate;
- **transmission** — newborn preparations-to-first-correct.

A positive on binding with a null on transmission is a real and reportable outcome: the agent
learned to read a mark, but only its own.

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

> **Claim (transmission) — RULED. Two conditioned lines, replacing the newborn measure.**
>
> **(ii) THE BINDING LINE — the stale-mark ratio.** P(chosen = π⁻¹(strongest positive label) | a
> positive mark is present), **split by whether the mark endorses the correct preparation**, and
> read on the **stale** cell — where the mark endorses a preparation that is *wrong* for this type
> now, left before the mapping moved. Following a stale mark is a mistake, so an agent that
> follows one can only be reading it.
>
> **The null is `(1 − hit)/(K − 1)`, not `1/K`.** An agent that knows the answer never agrees with
> a stale mark whatever it reads, so `1/K` would score competence as illiteracy. The null is the
> chance of landing on the endorsed-but-wrong preparation *given you did not pick the correct one*.
> **Ratio above 1 = follows a mark it should not; below 1 = avoids one. Either way the label was
> read — you cannot avoid what you cannot see.**
>
> **(ii-newborn) THE TRANSMISSION LINE — the same ratio over FIRST-EVER preparations only, kept
> SEPARATE from (ii).** (ii) pools over a life, so it mixes transmission with an agent's own
> within-life binding. A first-ever preparation cannot: the agent has learned nothing, written
> nothing, and by no-self-echo the mark cannot be its own.
>
> **(i) is DEMOTED to a density check, not a transmission measure.** P(correct | mark) vs
> P(correct | none) among first-ever preparations is confounded: a positive mark exists only where
> someone recently *succeeded*, so it marks places and times where success is common. The pre-check
> proved it — the gap was **largest in `noise`** (+0.241 against `record`'s +0.120), whose labels
> carry nothing. Read it as a check that marks are present and non-uniform, and nothing more.
>
> **Preparations-to-first-correct is kept as CORROBORATING ONLY, over agents that reached 5
> preparations.** Conditioning on reaching 5 is what stops censoring being confounded by short
> lives — which is how the pre-check's version read `fixed + record` as best at 1.586 while its
> population was 261 against 717.

### No self-echo — a property of the design, not a defect

A preparation **consumes the food cell**. So the mark it writes cannot be read for a preparation
until food respawns there, and the reader is then whoever is standing on it. **An agent can never
read its own mark about the food it just prepared.**

This is what makes the self-marking confound structurally weak rather than merely unlikely, and it
is why the two lines above are transmission measures and not memory measures. It is checked in
`record_semantics_selftest` — after a preparation the cell holds no food and the mark is present —
so it is a verified property, not an argument.

**DECISION 6 — RULED.** Preparations 1..n of a life, in a world with the store live against the same
world with **`sym_gain` forced to zero**. Not the store removed: removing it changes the world —
mark density, cell state and decay all go — whereas forcing the gain leaves the world identical and
isolates the *reading*. n = 5, reported as the mean number of preparations to the first correct one
and as the hit on preparation 1 alone.

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

## Slow labels — an ARM, not a follow-up

`π` is redrawn every **3 remaps** rather than every one, so a label's meaning **outlives what it
names** by three eras. This is v3.5's tempo condition: two facts moving at different rates, with
the slower one the thing that has to be learned.

**Meaning is still not inheritable.** A genome fixing on "label *j* means preparation *k*" is right
for three eras and then wrong, far inside evolutionary time — and Gate R's permutation null tests
exactly that, on this arm as on the others.

It is an **arm now, not a follow-up conditional on a null**, so the comparison is made inside one
experiment rather than across two. Nothing else moves with it: gates, probes and claims are
identical across the record arms.

> **The pre-registered reading, recorded before the run.**
>
> | fast (`plastic + record`) | slow (`plastic + record (slow)`) | conclusion |
> |---|---|---|
> | null | **positive** | **capacity limit confirmed, and the tempo condition established.** Binding a label that rotates every era is harder than the conjunction itself; give the label three eras and it binds. |
> | null | null | **the first earned transmission null.** A record handed over free, costless to write, costless to read, its meaning stable for three eras, still carries nothing between agents. |
> | positive | positive | the record works; read the margin for whether tempo helps. |
> | positive | null | would need explaining before anything is claimed — slower meaning should not *hurt*. |

## The world, carried from v3.12 with one change

`T = 3`, `K = 5`, 60 mappings, `prep_every` 700, `prep_value` 1.0 = (K−1)·`prep_fail` 0.25 so the
chance EV of a preparation is exactly zero, `spawn_per_patch` 6.0.

**`max_pop` 800 → 1000.** Selection on `sym_gain` needs fecundity: it is one scalar among several
heritable genes, and a population held at the cap has its reproduction throttled, which is exactly
the pressure that would have to move it. The pre-check found `plastic` at 763 of 800 — 95%, which
would have failed the cap check — so the cap was binding on the arm that is the baseline for the
licensing statistic.

## What is deferred

**Emergent writing is v3.14**, and it is specified only after v3.13 reads. If a population cannot
use a record that is handed to it for free, there is nothing to be gained by asking it to choose to
write one — that ordering is what keeps v3.14 from re-running finding #1 a third time.
