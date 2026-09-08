# G5 spec — the reference arm

A3's row:

| milestone | gate | number it must move |
|---|---|---|
| G5 | reference arm: one frozen LLM against the same store | **reported, never claimed** |

B§5.4: *"One frozen LLM provider against the same store and adapter. Its numbers are reported in a
separate table and are not eligible for any claim."*

---

## 1. What this arm is for, and what it is not

It is **not** a baseline the learner is supposed to beat, and it is not evidence about the learner
either way. B§2 is explicit that the project's route "does not run through a large model", and
A1.1 that "no LLM is a component of any agent". The reference arm exists so that a reader who asks
*"how does this compare to just using a model?"* gets a number instead of an argument — and so
that the number sits in a table that says, in the table, that it cannot be combined with anything
else.

Two consequences that are easy to lose:

* **It has no claim line.** Every other arm in this project moves a number that a gate reads. This
  one moves nothing. If G5's result is surprising in either direction, that is a fact about the
  presentation in §2, not about the learner.
* **It is the arm most likely to be quoted out of context**, which is why the separation is
  structural rather than editorial: `ExperimentArm.REFERENCE` is refused by `config_kwargs()` at
  every milestone before G5, its results are stored in their own table, and no acceptance function
  reads them.

---

## 2. The problem that makes this hard: an LLM cannot perceive this world

The adapter is a **vector** domain. An observation is 31 floats — directional sums over five
channels, five 'here' values, energy, and K read channels carrying `sym_gain × marks`. An action is
one of ten (twelve at G4-K). B§3 is emphatic about why: *"Nothing an agent reads can name an
answer, because agents read channels, not words."*

A frozen LLM cannot act on floats without a **presentation**, and every possible presentation is a
design choice that can hand it the answer or hide the world from it. This is the whole difficulty
of G5 and it deserves to be stated before any number is produced.

### 2.1 The leak rule applies to the presentation, and it is where it will be broken

§47's rule for this domain (B§3): *"the store carries labels, not preparation indices, and π is
redrawn."* A presentation that renders the read channels as

> "a previous agent succeeded with **preparation 3** on this food here"

has broken it — it has decoded π on the model's behalf, which is exactly the work the learner has
to do within its own life. The honest rendering is

> "channel 3 here reads +0.42"

with no statement anywhere about what channel 3 means. `check_no_leak` is stated in terms that
apply directly: labels not indices, π redrawn, reading an observation and not an action, channels
here-only.

**Ruled (G5-D1):** the presentation is a **mechanical, lossless rendering of the 31 floats** —
each field named by its layout position and printed to a fixed precision, in a fixed order, with
no interpretation, no summary and no natural-language gloss. `civitas_g.world.adapter.OBS_LAYOUT`
already carries the names, and it names the appetite inputs `DEAD`, so those go in as zeros like
everything else rather than being helpfully removed.

Anything more helpful than that is the experimenter playing the game.

**Ruled (G5-D6): the field names stay, and the arm is therefore *generously* provisioned.**
Building the rendering surfaced an asymmetry worth stating rather than absorbing. The learner's
network receives thirty-one unlabelled floats; the rendering names them — `food_A_here`,
`energy`, `read_channel_0`. Those names are more than the learner is given. The alternative,
bare positional indices (`f20 = +0.42`), is closer to parity and was rejected: a frozen model
handed thirty-one anonymous numbers cannot do anything at all, and an arm that produces zero
because the presentation was deliberately impoverished tells a reader nothing they could not have
guessed.

So the reference arm is provisioned generously on purpose, and the consequence runs one way:
**a poor number from this arm is more informative than a good one.** A good number is confounded
with the presentation's help; a poor number is a model failing at a task it was given more
context for than the learner ever had. Neither is a claim (§3.1). The names never cross the line
that matters — `read_channel_j` is a position, not a preparation — which is the distinction
`check_presentation_no_leak` enforces and §2.1 exists to protect.

### 2.2 What the model is told about the task

The same problem one level up. A prompt that explains the preparation mapping, the era clock, or
what a positive mark means is a prompt that transmits the fact the population is supposed to
acquire.

**Ruled (G5-D2):** the model is told the **action set and the modulator table** — the seven events
and their signs, which are the world's rules and are not secret — and **nothing about the
mapping, π, the era length, or what the read channels mean.** It is told that a number arrives
after each action and that larger is better. That is exactly what an agent's own network is given:
the actions it can take, and a modulator.

### 2.3 Memory, and why the arm is a family rather than one number

An LLM with the full episode in context is not comparable to a bounded life with a learned `H`
that dies. An LLM with no context at all is not comparable either.

**Ruled (G5-D3):** report **three** presentations, in one table, never averaged:

| variant | context | what it is comparable to |
|---|---|---|
| `reference:blind` | this observation only | the innate policy — a genome with `eta = 0` |
| `reference:life` | this life's observations and modulators, truncated to a bounded window | a bounded life with `H` |
| `reference:store` | as `life`, plus the read channels from the inherited record | G3's `inherited store` arm |

`reference:store` is the only one that touches the store, and it is the one B§5.4 names. The other
two exist so the store's contribution to the reference arm can be separated the same way it is
separated for the learner — otherwise a number from `reference:store` says nothing about the store.

---

## 3. What is measured

The same statistics as every other arm, from the same code path, computed by
`civitas_g.reading.compute` off persisted rows — **not** a bespoke metric:

* founder-free preparation hit;
* preparations-to-first-correct;
* the stale-mark ratio against A2.2's matched null.

If the reference arm needed its own metric, the comparison would be meaningless.

### 3.1 What is not measured, and must not be inferred

* **No claim line.** Not against the learner, not against `fresh store`, not against anything.
* **No acceptance.** No gate reads this table and `civitas_g.g3.acceptance` does not see it.
* **No compounding.** G4's mechanics are not run against it: "does the model get better as the
  world hardens" is a question about the model, and this project has no instrument for it.

---

## 4. Cost, and the honesty it requires

A run is thousands of steps × hundreds of agents. Presenting every agent-step to a frozen model is
not affordable and never will be, so the reference arm is necessarily run on a **reduced
population over a reduced window**, and that reduction is not comparable to the learner's run.

**Ruled (G5-D4):** the reference arm runs on the **frozen-replay window** — A2.1's mechanism,
already built: an era-boundary snapshot, births and deaths disabled, energy pinned, 300 steps.
The population is sampled down to what the budget allows. The manifest records the sample size and
the per-step cost, and the table reports **N**, so nobody reads a number from twelve agents as
though it came from three hundred.

The alternative — running the model as a whole population for a whole run — would cost more than
the entire project has spent on compute so far and would produce one number with no error bars.

---

## 5. Provider

`civitas/runtime/providers/` is dormant under B§1 *"every §19 provider except the G5 reference"*.
G5 revives exactly one, and the one it revives must be **frozen**: a pinned model identifier, a
pinned prompt, temperature zero, recorded in the manifest by hash like every other instrument. A
provider that could silently change version would make the table unreproducible, which is the one
thing this project has been most careful about.

**Ruled (G5-D5):** the arm records the model identifier, the prompt hash, the sampling parameters
and the total token count in the manifest, and refuses to run if any of them is unset. An
unreproducible number is worse than no number, and this arm's whole purpose is to be quotable.

---

## 6. Order of work

1. The presentation (§2), with `check_no_leak` run over rendered observations as a test — before
   any model is called.
2. `reference:blind` on a frozen replay, tiny sample, to prove the loop.
3. The three variants at one era boundary, one seed. Files come to you.
4. The table.

**There is no step 5.** The reference arm does not get three seeds, an acceptance, or a claim,
because it is not eligible for one.
